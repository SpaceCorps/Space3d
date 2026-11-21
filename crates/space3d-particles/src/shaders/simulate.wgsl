// Space3d particles, GPU backend: the per-tick kernels (design-particles §7 steps 1–2), the GPU
// twins of `cpu::simulate` and `cpu::emit`. Composed after `gpu_common.wgsl`.
//
// Per replayed tick: `simulate` (a thread per pool slot below the high water mark: every
// substep of the tick, dead slots return after one read), then `emit` (a thread per new
// particle, found by binary search over the tick's spawn records). No thread reads another
// particle, so the result is a pure function of the logged ticks.

// Emitter frame of an effect instance during the tick (`gpu::tables::GpuXform`, 80 bytes).
struct S3dpXform {
    // xyz: position at the start of the tick, w: effect scale.
    prev_pos: vec4<f32>,
    prev_rot: vec4<f32>,
    pos: vec4<f32>,
    rot: vec4<f32>,
    // xyz: velocity (measured motion + `ParticleEmitter::velocity`), w: throttle
    // (`ParticleEmitter::rate_scale`).
    vel: vec4<f32>,
}

// A spawn record of the tick (`gpu::tables::GpuSpawn`, 48 bytes).
struct S3dpSpawn {
    // x: entry, y: ring slot of the record's first particle (`index mod capacity`), z: threads
    // (particles written), w: first thread.
    a: vec4<u32>,
    // x, y: 64-bit spawn index of the record's first particle, z, w: sub-tick span (1/16384).
    b: vec4<u32>,
    // x: first particle written (the ones before it are overwritten within the tick), y: the
    // record's particle count, z: extra age (f32 bits, prewarm), w: 0.
    c: vec4<u32>,
}

// One replayed tick (`gpu::tables::GpuTick`, dynamic offset).
struct S3dpTick {
    // x: dt of the tick, y: dt of a substep, z: runtime time at the tick start, w: dt / 16384.
    a: vec4<f32>,
    // x: substeps, y: first spawn record, z: spawn records, w: emit threads.
    b: vec4<u32>,
    // x: pool slots in use (high water), y: entries, z: world attractors, w: 0.
    c: vec4<u32>,
    // World attractors: (position, strength), (radius, falloff, mask bits, 0).
    attractors: array<vec4<f32>, 16>,
}

@group(0) @binding(0) var<storage, read_write> pool: array<S3dpParticle>;
@group(0) @binding(2) var<storage, read> xforms: array<S3dpXform>;
@group(0) @binding(3) var<storage, read> spawns: array<S3dpSpawn>;
@group(0) @binding(4) var<uniform> tick: S3dpTick;

// `cpu::pre_age`: closed-form motion under gravity and drag for `t` seconds.
// Below `drag * t` of S3DP_PRE_AGE_SERIES `(t - k) / drag` is a series: the difference cancels in
// f32 (see the Rust function).
const S3DP_PRE_AGE_SERIES: f32 = 0.05;
fn s3dp_pre_age(x: vec3<f32>, v: vec3<f32>, g: vec3<f32>, drag: f32, t: f32) -> mat2x3<f32> {
    let u = drag * t;
    let e = exp(-u);
    var k: f32;
    var c: f32;
    if (u < S3DP_PRE_AGE_SERIES) {
        let kp = fma(u, fma(u, fma(u, -1.0 / 24.0, 1.0 / 6.0), -0.5), 1.0);
        let cp = fma(u, fma(u, fma(u, -1.0 / 120.0, 1.0 / 24.0), -1.0 / 6.0), 0.5);
        k = t * kp;
        c = t * t * cp;
    } else {
        k = (1.0 - e) / drag;
        c = (t - k) / drag;
    }
    return mat2x3<f32>(x + v * k + g * c, v * e + g * k);
}

fn s3dp_hash3(x: i32, y: i32, z: i32, seed: u32) -> f32 {
    let k = s3dp_pcg(bitcast<u32>(x) ^ s3dp_pcg(bitcast<u32>(y) ^ s3dp_pcg(bitcast<u32>(z) ^ seed)));
    return f32(k >> 8u) * (1.0 / 16777216.0);
}

// `cpu::value_noise`.
fn s3dp_value_noise(p: vec3<f32>, seed: u32) -> f32 {
    let i = floor(p);
    let f = p - i;
    let w = f * f * (vec3<f32>(3.0) - 2.0 * f);
    let c = vec3<i32>(i);
    let h000 = s3dp_hash3(c.x, c.y, c.z, seed);
    let h100 = s3dp_hash3(c.x + 1, c.y, c.z, seed);
    let h010 = s3dp_hash3(c.x, c.y + 1, c.z, seed);
    let h110 = s3dp_hash3(c.x + 1, c.y + 1, c.z, seed);
    let h001 = s3dp_hash3(c.x, c.y, c.z + 1, seed);
    let h101 = s3dp_hash3(c.x + 1, c.y, c.z + 1, seed);
    let h011 = s3dp_hash3(c.x, c.y + 1, c.z + 1, seed);
    let h111 = s3dp_hash3(c.x + 1, c.y + 1, c.z + 1, seed);
    let a0 = h000 + (h100 - h000) * w.x;
    let a1 = h010 + (h110 - h010) * w.x;
    let b0 = h001 + (h101 - h001) * w.x;
    let b1 = h011 + (h111 - h011) * w.x;
    let x0 = a0 + (a1 - a0) * w.y;
    let x1 = b0 + (b1 - b0) * w.y;
    return x0 + (x1 - x0) * w.z;
}

// `cpu::attract`.
fn s3dp_attract(center: vec3<f32>, x: vec3<f32>, strength: f32, radius: f32, falloff: u32) -> vec3<f32> {
    let d = center - x;
    let dist = length(d);
    if (dist < 1e-5 || (radius > 0.0 && dist > radius)) {
        return vec3<f32>(0.0);
    }
    var k = 1.0;
    if (falloff == 1u) {
        if (radius > 0.0) {
            k = 1.0 - dist / radius;
        }
    } else if (falloff == 2u) {
        k = 1.0 / max(dist * dist, 0.01);
    }
    return d * (strength * k / dist);
}

// `cpu::curl_noise`: the curl of the three value noises, by central differences 0.1 apart.
fn s3dp_curl_diff(p: vec3<f32>, d: vec3<f32>, seed: u32) -> f32 {
    return s3dp_value_noise(p + d, seed) - s3dp_value_noise(p - d, seed);
}

fn s3dp_curl_noise(p: vec3<f32>) -> vec3<f32> {
    let dx = vec3<f32>(0.1, 0.0, 0.0);
    let dy = vec3<f32>(0.0, 0.1, 0.0);
    let dz = vec3<f32>(0.0, 0.0, 0.1);
    return vec3<f32>(
        s3dp_curl_diff(p, dy, 0x7feb352du) - s3dp_curl_diff(p, dz, 0x9e3779b9u),
        s3dp_curl_diff(p, dz, 0x1f123bb5u) - s3dp_curl_diff(p, dx, 0x7feb352du),
        s3dp_curl_diff(p, dx, 0x9e3779b9u) - s3dp_curl_diff(p, dy, 0x1f123bb5u),
    ) * (0.5 / 0.1);
}

// `cpu::forces`: the layer's forces and the world attractors at `x`.
fn s3dp_forces(e: u32, flags: u32, x: vec3<f32>, row: S3dpXform, time: f32) -> vec3<f32> {
    var a = vec3<f32>(0.0);
    let local = (flags & S3DP_LOCAL) != 0u;
    let n = (flags >> 16u) & 7u;
    let scale = row.prev_pos.w;
    for (var i = 0u; i < n; i = i + 1u) {
        let p = s3dp_wordf(e, 9u + 2u * i);
        let q = s3dp_word(e, 10u + 2u * i);
        if (q.w == 0u) {
            // Attractor toward an emitter-local point.
            var c = p.xyz * scale;
            if (!local) {
                c = row.pos.xyz + s3dp_rotate(row.rot, p.xyz * scale);
            }
            a += s3dp_attract(c, x, p.w, bitcast<f32>(q.x) * scale, q.y);
        } else if (q.w == 1u) {
            // Vortex around an emitter axis.
            var o = vec3<f32>(0.0);
            var ax = s3dp_normalize_or(p.xyz, vec3<f32>(0.0, 1.0, 0.0));
            if (!local) {
                o = row.pos.xyz;
                ax = s3dp_normalize_or(s3dp_rotate(row.rot, p.xyz), vec3<f32>(0.0, 1.0, 0.0));
            }
            let r = x - o;
            let rp = r - ax * dot(r, ax);
            let d = length(rp);
            if (d >= 1e-5) {
                let out = rp / d;
                a += cross(ax, out) * p.w - out * bitcast<f32>(q.x);
            }
        } else if (q.w == 2u) {
            // Turbulence: strength, frequency, scroll speed.
            let s = x * p.y + vec3<f32>(time * p.z);
            a += vec3<f32>(
                s3dp_value_noise(s, 0x1f123bb5u),
                s3dp_value_noise(s, 0x9e3779b9u),
                s3dp_value_noise(s, 0x7feb352du),
            ) * (2.0 * p.x) - vec3<f32>(p.x);
        } else {
            // Curl noise: strength, frequency, scroll speed.
            a += s3dp_curl_noise(x * p.y + vec3<f32>(time * p.z)) * p.x;
        }
    }
    let mask = s3dp_word(e, 1u).w;
    if (mask != 0u) {
        let inv = vec4<f32>(-row.rot.xyz, row.rot.w);
        for (var i = 0u; i < tick.c.z; i = i + 1u) {
            let w0 = tick.attractors[2u * i];
            let w1 = tick.attractors[2u * i + 1u];
            if ((bitcast<u32>(w1.z) & mask) == 0u) {
                continue;
            }
            var c = w0.xyz;
            if (local) {
                c = s3dp_rotate(inv, w0.xyz - row.pos.xyz);
            }
            a += s3dp_attract(c, x, w0.w, w1.x, bitcast<u32>(w1.y));
        }
    }
    return a;
}

@compute @workgroup_size(64)
fn simulate(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = s3dp_thread(gid, groups);
    if (i >= tick.c.x) {
        return;
    }
    // A dead slot costs one 16-byte read; a live one reads and writes 32 bytes (plus its
    // entry index).
    let vl = pool[i].vel_life;
    var life = vl.w;
    if (life == 0.0) {
        return;
    }
    let pa = pool[i].pos_age;
    let e = pool[i].misc.z;
    let valid = e < tick.c.y;
    var flags = 0u;
    var gravity = vec3<f32>(0.0);
    var drag = 1.0;
    var forces = false;
    var row: S3dpXform;
    if (valid) {
        let w1 = s3dp_word(e, 1u);
        flags = w1.x;
        gravity = s3dp_wordf(e, 8u).xyz;
        let k = s3dp_wordf(e, 7u).w;
        if (k > 0.0) {
            drag = exp(-k * tick.a.y);
        }
        forces = ((flags >> 16u) & 7u) != 0u || w1.w != 0u;
        if (forces) {
            row = xforms[s3dp_word(e, 0u).w];
        }
    }
    var x = pa.xyz;
    var v = vl.xyz;
    var age = pa.w;
    let sub = tick.a.y;
    for (var s = 0u; s < tick.b.x; s = s + 1u) {
        let age0 = age;
        age = age0 + sub;
        if (age0 < 0.0) {
            continue; // delayed: only ages
        }
        if (age >= life || !valid) {
            life = 0.0;
            break;
        }
        var a = gravity;
        if (forces) {
            a += s3dp_forces(e, flags, x, row, tick.a.z + f32(s) * sub);
        }
        v = (v + a * sub) * drag;
        x += v * sub;
    }
    pool[i].pos_age = vec4<f32>(x, age);
    pool[i].vel_life = vec4<f32>(v, life);
}

// `cpu::sample_shape`: emitter-local offset and "outward" direction.
fn s3dp_sample_shape(kind: u32, p: vec4<f32>, seed: u32, scale: f32) -> mat2x3<f32> {
    let u = s3dp_rnd(seed, S3DP_SHAPE_A);
    let v = s3dp_rnd(seed, S3DP_SHAPE_B);
    let w = s3dp_rnd(seed, S3DP_SHAPE_C);
    switch kind {
        case 1u: {
            // Sphere: radius, inner fraction.
            let d = s3dp_uniform_dir(u, v);
            let inner = clamp(p.y, 0.0, 1.0);
            let i3 = inner * inner * inner;
            return mat2x3<f32>(d * p.x * pow(i3 + (1.0 - i3) * w, 1.0 / 3.0) * scale, d);
        }
        case 2u: {
            // Cone around −Z: half-angle, base radius.
            let a = S3DP_TAU * u;
            let r = p.y * sqrt(v) * scale;
            let dir = s3dp_cone_dir(vec3<f32>(0.0, 0.0, -1.0), p.x, w, s3dp_rnd(seed, S3DP_DIR_B));
            return mat2x3<f32>(vec3<f32>(r * cos(a), r * sin(a), 0.0), dir);
        }
        case 3u: {
            // Ring in XZ: radius, width, arc.
            let arc = select(S3DP_TAU, p.z, p.z > 0.0);
            let a = arc * u;
            let r = (p.x + (v - 0.5) * p.y) * scale;
            let out = vec3<f32>(cos(a), 0.0, sin(a));
            return mat2x3<f32>(out * r, out);
        }
        case 4u: {
            // Box: half extents.
            let q = vec3<f32>(u * 2.0 - 1.0, v * 2.0 - 1.0, w * 2.0 - 1.0) * p.xyz * scale;
            let fallback = s3dp_uniform_dir(s3dp_rnd(seed, S3DP_DIR_A), s3dp_rnd(seed, S3DP_DIR_B));
            return mat2x3<f32>(q, s3dp_normalize_or(q, fallback));
        }
        case 5u: {
            // Line along X: length.
            return mat2x3<f32>(vec3<f32>((u - 0.5) * p.x * scale, 0.0, 0.0), vec3<f32>(0.0, 0.0, -1.0));
        }
        default: {
            return mat2x3<f32>(vec3<f32>(0.0), s3dp_uniform_dir(u, v));
        }
    }
}

@compute @workgroup_size(64)
fn emit(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let t = s3dp_thread(gid, groups);
    if (t >= tick.b.w) {
        return;
    }
    // The record of thread `t`: the last one starting at or before it.
    var lo = tick.b.y;
    var hi = tick.b.y + tick.b.z;
    while (hi - lo > 1u) {
        let mid = (lo + hi) / 2u;
        if (spawns[mid].a.w <= t) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let r = spawns[lo];
    let e = r.a.x;
    let k = r.c.x + (t - r.a.w);
    let w0 = s3dp_word(e, 0u);
    let cap = max(w0.y, 1u);
    let slot = w0.x + (r.a.y + k) % cap;
    if (slot >= tick.c.x) {
        return;
    }
    let lo_index = r.b.x + k;
    let hi_index = r.b.y + select(0u, 1u, lo_index < r.b.x);
    let seed = s3dp_particle_seed(w0.z, lo_index, hi_index);

    // Sub-tick time (`cpu::subtick`), in integers: evenly from the first particle's time to
    // the last's.
    var fq = r.b.z;
    if (r.b.w > r.b.z && r.c.y > 1u) {
        let n = r.c.y - 1u;
        fq = r.b.z + ((r.b.w - r.b.z) * min(k, n) + n / 2u) / n;
    }
    let f = f32(fq) * (1.0 / 16384.0);

    let w1 = s3dp_word(e, 1u);
    let flags = w1.x;
    let local = (flags & S3DP_LOCAL) != 0u;
    let row = xforms[w0.w];
    var origin = vec3<f32>(0.0);
    var rot = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    if (!local) {
        origin = row.prev_pos.xyz * (1.0 - f) + row.pos.xyz * f;
        rot = s3dp_qlerp(row.prev_rot, row.rot, f);
    }
    let scale = row.prev_pos.w;
    let shape = s3dp_sample_shape(w1.y, s3dp_wordf(e, 2u), seed, scale);
    let w3 = s3dp_wordf(e, 3u);
    let w4 = s3dp_wordf(e, 4u);
    let w5 = s3dp_wordf(e, 5u);
    let flatten = w4.w;
    let axis = w5.xyz;
    var offset = shape[0] + w3.xyz * scale;
    offset = offset - axis * (dot(offset, axis) * flatten);
    let outward = shape[1];
    var dir = outward;
    switch w1.z {
        case 1u: {
            dir = s3dp_normalize_or(w4.xyz, vec3<f32>(0.0, 0.0, -1.0));
        }
        case 2u: {
            dir = s3dp_uniform_dir(s3dp_rnd(seed, S3DP_DIR_A), s3dp_rnd(seed, S3DP_DIR_B));
        }
        case 3u: {
            dir = -outward;
        }
        case 4u: {
            dir = s3dp_normalize_or(cross(vec3<f32>(0.0, 1.0, 0.0), outward), vec3<f32>(1.0, 0.0, 0.0));
        }
        default: {}
    }
    if (flatten > 0.0) {
        dir = s3dp_normalize_or(dir - axis * (dot(dir, axis) * flatten), dir);
    }
    if (w3.w > 0.0) {
        dir = s3dp_cone_dir(dir, w3.w, s3dp_rnd(seed, S3DP_SPREAD_A), s3dp_rnd(seed, S3DP_SPREAD_B));
    }
    let w6 = s3dp_wordf(e, 6u);
    let w7 = s3dp_wordf(e, 7u);
    var speed_mul = 1.0;
    if (w5.w != 0.0) {
        speed_mul = pow(max(scale, 0.0), w5.w);
    }
    // `ThrottleResponse::speed` (word 21.w) of the throttle (the frame's `vel.w`) clamped to
    // `Spawn::max_rate_scale` (word 21.z): `LayerSim::speed_throttle`.
    let w21 = s3dp_wordf(e, 21u);
    if (w21.w != 0.0) {
        let throttle = clamp(row.vel.w, 0.0, max(w21.z, 0.0));
        if (w21.w == 1.0) {
            speed_mul *= throttle;
        } else {
            speed_mul *= pow(throttle, w21.w);
        }
    }
    let speed = s3dp_range(seed, S3DP_SPEED, w7.x, w7.y) * speed_mul;
    var pos = origin + s3dp_rotate(rot, offset);
    var vel = s3dp_rotate(rot, dir) * speed;
    if (!local) {
        vel += row.vel.xyz * w7.z;
    }
    let life = s3dp_range_exact(seed, S3DP_LIFETIME, w6.x, w6.y);
    let delay = s3dp_range_exact(seed, S3DP_DELAY, w6.z, w6.w);
    // One explicit rounding (`cpu::spawn_particle`: `mul_add`), which no shader compiler can
    // contract or reassociate differently.
    let age = fma(f32(16384u - fq), tick.a.w, bitcast<f32>(r.c.z) - delay);
    if (age > 0.0) {
        let aged = s3dp_pre_age(pos, vel, s3dp_wordf(e, 8u).xyz, w7.w, age);
        pos = aged[0];
        vel = aged[1];
    }
    var x_axis = 0u;
    var y_axis = 0u;
    if ((flags & S3DP_EMITTER_PLANE) != 0u) {
        x_axis = s3dp_oct_encode(s3dp_rotate(rot, vec3<f32>(1.0, 0.0, 0.0)));
        y_axis = s3dp_oct_encode(s3dp_rotate(rot, vec3<f32>(0.0, 1.0, 0.0)));
    }
    pool[slot] = S3dpParticle(
        vec4<f32>(pos, age),
        vec4<f32>(vel, max(life, 0.0)),
        vec4<u32>(x_axis, seed, e, y_axis),
    );
}
