// GPU frustum culling + order-preserving stream compaction.
//
// Visible instances are written to the output region in source order, so a GPU-culled draw
// rasterizes its instances in exactly the order the CPU path would (alpha blending and equal-
// depth surfaces are deterministic). Three passes, all sharing this bind group:
//
// 1. `cull_mark`    one workgroup per 64 instances of a job: frustum test, store the 64-bit
//                   visibility mask of the workgroup in its record (`args` tail).
// 2. `cull_scan`    one workgroup per job: exclusive prefix sum of the workgroups' visible
//                   counts (record base), total → `instance_count` of the job's indirect args.
// 3. `cull_scatter` like pass 1: each visible instance's slot = record base + visible lanes
//                   below it (popcount), then its raw words are copied to the output region.
//
// Mark/scatter dispatches cover every job that reads the same source buffer, for all views
// (up to 64 per dispatch: `frusta`); a workgroup finds its job with a binary search over the
// (frame-global) `wg_start`, and the job names its view's frustum.

struct CullParams {
    job_start: u32,
    job_count: u32,
    total_workgroups: u32,
    workgroups_x: u32,
    wg_word: u32,           // first word of the per-workgroup records in `args`
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Job {
    sphere: vec4<f32>,      // mesh-local bounding sphere
    src_word: u32,          // first word of instance 0 in `src`
    count: u32,             // instances
    stride_words: u32,
    kind: u32,              // 0: model matrix, 1: f32 translation, 2: f16 translation (+ scale)
    t_off: u32,             // matrix / translation offset in words
    s_off: u32,             // scale offset in bytes (f16 scales may start mid-word)
    s_kind: u32,            // 0: none, 1: uniform f32, 2: vec3 f32, 3: uniform f16, 4: vec3 f16
    dst_word: u32,          // first word of the output region
    draw: u32,              // index into indirect args
    wg_start: u32,          // first workgroup record of this job (frame-global)
    view: u32,              // frustum of the job's view in `frusta` (< 64)
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: CullParams;
@group(0) @binding(1) var<storage, read> jobs: array<Job>;
@group(0) @binding(2) var<storage, read> src: array<u32>;
@group(0) @binding(3) var<storage, read_write> dst: array<u32>;
// DrawIndexedIndirectArgs (5 words per draw), then 4 words per workgroup record from
// `params.wg_word`: visibility mask lo/hi, first output slot, unused.
@group(0) @binding(4) var<storage, read_write> args: array<u32>;
// Seven vectors per view (a window of 64 views; one bind group per window): the six frustum
// planes, then `Frustum::small`: a point (xyz) and a ratio (w); an instance whose radius is
// under the ratio × its distance from the point is not drawn (0: no such test).
@group(0) @binding(5) var<uniform> frusta: array<vec4<f32>, 448>;

var<workgroup> wg_mask: array<atomic<u32>, 2>;
var<workgroup> sums: array<u32, 64>;

fn load3(o: u32) -> vec3<f32> {
    return vec3<f32>(bitcast<f32>(src[o]), bitcast<f32>(src[o + 1u]), bitcast<f32>(src[o + 2u]));
}

/// IEEE binary16 bits (low 16 bits of `h`) to f32. Decoded by hand: `unpack2x16float` needs
/// `DownlevelFlags::SHADER_F16_IN_F32`, which e.g. llvmpipe lacks.
fn half_to_f32(h: u32) -> f32 {
    let e = (h >> 10u) & 0x1fu;
    let m = h & 0x3ffu;
    var v: f32;
    if (e == 0u) {
        v = f32(m) * 5.9604645e-8; // subnormal: m * 2^-24
    } else if (e == 31u) {
        v = bitcast<f32>(0x7f800000u | (m << 13u)); // inf / NaN
    } else {
        v = bitcast<f32>(((e + 112u) << 23u) | (m << 13u));
    }
    return select(v, -v, (h & 0x8000u) != 0u);
}

/// The two f16 values of a word (x = low half: little-endian memory order).
fn unpack_halves(w: u32) -> vec2<f32> {
    return vec2<f32>(half_to_f32(w & 0xffffu), half_to_f32(w >> 16u));
}

/// Three f16 values from two words.
fn load3_half(o: u32) -> vec3<f32> {
    return vec3<f32>(unpack_halves(src[o]), half_to_f32(src[o + 1u] & 0xffffu));
}

/// Instance scale for culling (`s_off` is in bytes from the instance's first word `base`).
fn instance_scale(job: Job, base: u32) -> f32 {
    let w = base + job.s_off / 4u;
    var s = 1.0;
    switch job.s_kind {
        case 1u: {
            s = bitcast<f32>(src[w]);
        }
        case 2u: {
            let v = abs(load3(w));
            s = max(v.x, max(v.y, v.z));
        }
        case 3u: {
            s = half_to_f32(select(src[w] & 0xffffu, src[w] >> 16u, (job.s_off & 2u) != 0u));
        }
        case 4u: {
            let v = abs(load3_half(w));
            s = max(v.x, max(v.y, v.z));
        }
        default: {}
    }
    return abs(s);
}

/// Index of the last job of this dispatch with `wg_start <= g`.
fn find_job(g: u32) -> u32 {
    var lo = params.job_start;
    var hi = params.job_start + params.job_count - 1u;
    loop {
        if (lo >= hi) {
            break;
        }
        let mid = (lo + hi + 1u) / 2u;
        if (jobs[mid].wg_start <= g) {
            lo = mid;
        } else {
            hi = mid - 1u;
        }
    }
    return lo;
}

fn is_visible(job: Job, base: u32) -> bool {
    var center: vec3<f32>;
    var radius: f32;
    if (job.kind == 0u) {
        let o = base + job.t_off;
        let c0 = load3(o);
        let c1 = load3(o + 4u);
        let c2 = load3(o + 8u);
        let t = load3(o + 12u);
        // Upper bound of the largest stretch (spectral norm) of the 3x3 part: Gershgorin on
        // AᵀA. Equals the longest column for rotation/scale matrices and stays conservative for
        // sheared ones (non-uniform parent scale × rotated child). Mirrors culling.rs.
        let d01 = abs(dot(c0, c1));
        let d02 = abs(dot(c0, c2));
        let d12 = abs(dot(c1, c2));
        let s2 = max(dot(c0, c0) + d01 + d02, max(dot(c1, c1) + d01 + d12, dot(c2, c2) + d02 + d12));
        center = t + c0 * job.sphere.x + c1 * job.sphere.y + c2 * job.sphere.z;
        radius = job.sphere.w * sqrt(s2);
    } else {
        let s = instance_scale(job, base);
        // Rotation is not decoded here: grow the sphere to cover any orientation.
        if (job.kind == 2u) {
            center = load3_half(base + job.t_off);
        } else {
            center = load3(base + job.t_off);
        }
        radius = (length(job.sphere.xyz) + job.sphere.w) * s;
    }
    var visible = true;
    for (var i = 0u; i < 6u; i++) {
        let p = frusta[job.view * 7u + i];
        if (dot(p.xyz, center) + p.w < -radius) {
            visible = false;
        }
    }
    let small = frusta[job.view * 7u + 6u];
    if (radius < small.w * distance(center, small.xyz)) {
        visible = false;
    }
    return visible;
}

/// Inclusive prefix sum of `v` over the 64 lanes (Hillis–Steele). Uniform control flow only.
fn workgroup_scan(lid: u32, v: u32) -> u32 {
    sums[lid] = v;
    workgroupBarrier();
    for (var off = 1u; off < 64u; off = off * 2u) {
        var x = sums[lid];
        if (lid >= off) {
            x += sums[lid - off];
        }
        workgroupBarrier();
        sums[lid] = x;
        workgroupBarrier();
    }
    return sums[lid];
}

@compute @workgroup_size(64)
fn cull_mark(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let wg = wid.x + wid.y * params.workgroups_x;
    if (wg >= params.total_workgroups) {
        return;
    }
    let g = jobs[params.job_start].wg_start + wg;
    let job = jobs[find_job(g)];
    let local = (g - job.wg_start) * 64u + lid;
    var visible = false;
    if (local < job.count) {
        visible = is_visible(job, job.src_word + local * job.stride_words);
    }
    if (lid < 2u) {
        atomicStore(&wg_mask[lid], 0u);
    }
    workgroupBarrier();
    if (visible) {
        // OR is order-independent: the mask is identical however lanes are scheduled.
        atomicOr(&wg_mask[lid / 32u], 1u << (lid % 32u));
    }
    workgroupBarrier();
    if (lid < 2u) {
        args[params.wg_word + g * 4u + lid] = atomicLoad(&wg_mask[lid]);
    }
}

fn record_count(g: u32) -> u32 {
    let r = params.wg_word + g * 4u;
    return countOneBits(args[r]) + countOneBits(args[r + 1u]);
}

@compute @workgroup_size(64)
fn cull_scan(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let j = wid.x + wid.y * params.workgroups_x;
    if (j >= params.job_count) {
        return;
    }
    let job = jobs[params.job_start + j];
    // Each lane owns a contiguous block of the job's workgroup records.
    let n = (job.count + 63u) / 64u;
    let per = (n + 63u) / 64u;
    let begin = min(lid * per, n);
    let end = min(begin + per, n);
    var sum = 0u;
    for (var i = begin; i < end; i++) {
        sum += record_count(job.wg_start + i);
    }
    let inclusive = workgroup_scan(lid, sum);
    var base = inclusive - sum;
    for (var i = begin; i < end; i++) {
        let g = job.wg_start + i;
        args[params.wg_word + g * 4u + 2u] = base;
        base += record_count(g);
    }
    if (lid == 63u) {
        args[job.draw * 5u + 1u] = inclusive;
    }
}

@compute @workgroup_size(64)
fn cull_scatter(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let wg = wid.x + wid.y * params.workgroups_x;
    if (wg >= params.total_workgroups) {
        return;
    }
    let g = jobs[params.job_start].wg_start + wg;
    let r = params.wg_word + g * 4u;
    let lo = args[r];
    let word = select(lo, args[r + 1u], lid >= 32u);
    let bit = lid % 32u;
    if (((word >> bit) & 1u) == 0u) {
        return;
    }
    var slot = args[r + 2u] + countOneBits(word & ((1u << bit) - 1u));
    if (lid >= 32u) {
        slot += countOneBits(lo);
    }
    let job = jobs[find_job(g)];
    let local = (g - job.wg_start) * 64u + lid;
    let base = job.src_word + local * job.stride_words;
    let out = job.dst_word + slot * job.stride_words;
    for (var w = 0u; w < job.stride_words; w++) {
        dst[out + w] = src[base + w];
    }
}
