// Space3d bloom (graphics spec §5.2 PO1, design-post §5.3; Jimenez, "Next-Generation Post
// Processing in Call of Duty: Advanced Warfare", 2014). The renderer prepends OUTPUT_WGSL and
// FULLSCREEN_WGSL. Four entry points share one layout, each drawn as a full-screen triangle
// over one level of the post group's bloom chain:
//
// * `fs_prefilter`: S (the post group's rect) → exactly half its size (level 0 at 1080p). Every
//   scene pixel is exposed (bloom works in exposed units: the composite adds it after
//   exposure), soft-thresholded and softly limited on its own (the firefly guard), then a
//   separable [1 2 1] / 4 tent over 3 × 3 pixels filters them: linear after the per-pixel
//   steps, so each pixel's weight over level 0 is exactly 1/4 wherever it sits and a small
//   bright source adds the same light at every sub-pixel position (no shimmer while it moves).
//   (M1 thresholded and Karis-weighted 2 × 2 box averages of bilinear taps: position-dependent
//   for sources of 2–4 px, whose halo varied by 4–12 % as they moved, and small sources lost
//   about 2/3 of their energy above the threshold.)
// * `fs_resample`: the prefilter's half-resolution output → level 0 when level 0 has another
//   size (the reference heights, post/bloom.rs `chain_size`): an area average, every pixel
//   weighted by its overlap with the destination pixel's box (energy kept wherever it sits).
// * `fs_down`: level i → level i + 1: 13 bilinear taps (Jimenez) with the plain weights.
// * `fs_up`: level i + 1 → level i, a 3×3 tent, blended with src × scatter + dst × (1 − scatter)
//   (the pipeline's blend constant), so level 0 ends up holding Σ (1 − s)·s^k · level_k.
//
// Only explicit-LOD sampling: no derivatives, no uniformity constraints (WebGL2-safe). Each pass
// samples one level view while it renders into another (per-level views; GL binds them with
// TEXTURE_BASE_LEVEL/MAX_LEVEL, like `generate_mips`).

// Matches `GpuBloomPass` (post/bloom.rs): one aligned uniform slot per pass.
struct BloomPass {
    // Source rect in pixels of the source texture: the post group's rect in S for the
    // prefilter, the whole source level otherwise.
    src_rect: vec4<f32>,
    // 1 / source width, 1 / source height, tap step in source pixels, exposure multiplier.
    src_texel: vec4<f32>,
    // Prefilter: the soft threshold t, t − k, 2k, 0.25 / k (k = t × soft_knee; 0 for a hard
    // cut). Resample: the destination pixel's box in source pixels (w, h, 0, 0).
    threshold: vec4<f32>,
    // x: multiply by 2^E (auto exposure). Resample: y, z = bilinear taps per axis. Prefilter:
    // w = 1 takes the cheap path (`Bloom::fast`).
    flags: vec4<u32>,
}

@group(0) @binding(0) var<uniform> bloom: BloomPass;
@group(0) @binding(1) var bloom_sampler: sampler;
@group(1) @binding(0) var bloom_src: texture_2d<f32>;
@group(1) @binding(1) var bloom_exposure: texture_2d<f32>;

// Largest value the chain stores (below the half-float and 11/10-bit float maxima): keeps an
// infinite scene value from turning the whole chain into infinities or NaNs.
const S3D_BLOOM_MAX: f32 = 65000.0;

// Bilinear tap of the scene at source pixel position `p` (texel centers at +0.5), clamped to the
// post group's rect (split screens never bleed into each other) and to 0..S3D_BLOOM_MAX
// (negative, infinite or NaN scene values never enter the chain).
fn bloom_tap_scene(p: vec2<f32>) -> vec3<f32> {
    let lo = bloom.src_rect.xy + 0.5;
    let hi = max(lo, bloom.src_rect.xy + bloom.src_rect.zw - 0.5);
    let q = clamp(p, lo, hi);
    let c = textureSampleLevel(bloom_src, bloom_sampler, q * bloom.src_texel.xy, 0.0).rgb;
    return clamp(c, vec3<f32>(0.0), vec3<f32>(S3D_BLOOM_MAX));
}

// Bilinear tap of a chain level (the whole level is the source rect: the sampler's clamp to edge
// equals clamping to its texel centers; the values are already sanitized).
fn bloom_tap(p: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(bloom_src, bloom_sampler, p * bloom.src_texel.xy, 0.0).rgb;
}

// The 13 taps around the destination pixel at `uv` (0..1 over the destination level), `step`
// source pixels apart:
//   a . b . c
//   . j . k .
//   d . e . f
//   . l . m .
//   g . h . i
struct Taps13 {
    a: vec3<f32>, b: vec3<f32>, c: vec3<f32>,
    d: vec3<f32>, e: vec3<f32>, f: vec3<f32>,
    g: vec3<f32>, h: vec3<f32>, i: vec3<f32>,
    j: vec3<f32>, k: vec3<f32>, l: vec3<f32>, m: vec3<f32>,
}

// One tap of the scene (`scene`: rect-clamped and sanitized) or of a chain level.
fn bloom_tap_of(p: vec2<f32>, scene: bool) -> vec3<f32> {
    if (scene) {
        return bloom_tap_scene(p);
    }
    return bloom_tap(p);
}

fn taps13(uv: vec2<f32>, scene: bool) -> Taps13 {
    let p = bloom.src_rect.xy + uv * bloom.src_rect.zw;
    let s = bloom.src_texel.z;
    var t: Taps13;
    t.a = bloom_tap_of(p + vec2<f32>(-2.0, -2.0) * s, scene);
    t.b = bloom_tap_of(p + vec2<f32>(0.0, -2.0) * s, scene);
    t.c = bloom_tap_of(p + vec2<f32>(2.0, -2.0) * s, scene);
    t.d = bloom_tap_of(p + vec2<f32>(-2.0, 0.0) * s, scene);
    t.e = bloom_tap_of(p, scene);
    t.f = bloom_tap_of(p + vec2<f32>(2.0, 0.0) * s, scene);
    t.g = bloom_tap_of(p + vec2<f32>(-2.0, 2.0) * s, scene);
    t.h = bloom_tap_of(p + vec2<f32>(0.0, 2.0) * s, scene);
    t.i = bloom_tap_of(p + vec2<f32>(2.0, 2.0) * s, scene);
    t.j = bloom_tap_of(p + vec2<f32>(-1.0, -1.0) * s, scene);
    t.k = bloom_tap_of(p + vec2<f32>(1.0, -1.0) * s, scene);
    t.l = bloom_tap_of(p + vec2<f32>(-1.0, 1.0) * s, scene);
    t.m = bloom_tap_of(p + vec2<f32>(1.0, 1.0) * s, scene);
    return t;
}

// Soft-knee threshold on the brightest channel (continuous at t − k and t + k).
fn bloom_threshold(c: vec3<f32>) -> vec3<f32> {
    let t = bloom.threshold;
    let br = max(c.r, max(c.g, c.b));
    var rq = clamp(br - t.y, 0.0, t.z);
    rq = t.w * rq * rq;
    return c * (max(rq, br - t.x) / max(br, 1e-4));
}

// The exposure multiplier of the prefilter (manual, or × 2^E with auto exposure).
fn bloom_exposure_multiplier() -> f32 {
    var e = bloom.src_texel.w;
    if (bloom.flags.x != 0u) {
        e = e * exp2(textureLoad(bloom_exposure, vec2<i32>(0), 0).r);
    }
    return e;
}

// Bloom contribution above which a pixel is compressed (exposed units): the firefly guard.
const S3D_BLOOM_KNEE: f32 = 32.0;

// A single pixel's bloom contribution, softly limited: unchanged up to the knee (engines,
// lasers, sparks), then compressed toward twice it, so a stray specular spike of radiance 1000
// adds at most as much glow as a pixel of 64 (the firefly guard; per pixel, so it does not
// depend on where the pixel sits).
fn bloom_limit(c: vec3<f32>) -> vec3<f32> {
    let br = max(c.r, max(c.g, c.b));
    if (br <= S3D_BLOOM_KNEE) {
        return c;
    }
    let x = br - S3D_BLOOM_KNEE;
    return c * ((S3D_BLOOM_KNEE + x / (1.0 + x / S3D_BLOOM_KNEE)) / br);
}

// The bloom contribution of the scene pixel `p` (S pixels): exposed, soft-thresholded and
// limited; 0 outside the post group's rect.
fn bloom_pixel(p: vec2<i32>, lo: vec2<i32>, hi: vec2<i32>, e: f32) -> vec3<f32> {
    if (any(p < lo) || any(p >= hi)) {
        return vec3<f32>(0.0);
    }
    let c = clamp(textureLoad(bloom_src, p, 0).rgb, vec3<f32>(0.0), vec3<f32>(S3D_BLOOM_MAX));
    return bloom_limit(bloom_threshold(c * e));
}

// Tent weight [1 2 1] / 4 of tap `k` (0..2).
fn bloom_tent(k: i32) -> f32 {
    return select(0.25, 0.5, k == 1);
}

@fragment
fn fs_prefilter(in: FullscreenOut) -> @location(0) vec4<f32> {
    // Destination pixel i, per axis: a [1 2 1] / 4 tent over the rect's pixels 2i − 1, 2i,
    // 2i + 1. Every scene pixel is exposed, thresholded and limited on its own, then filtered
    // linearly: an even pixel weighs 1/2 in one destination pixel, an odd one 1/4 in two, so
    // every pixel's total weight is exactly 1/4 wherever it sits and a small bright source adds
    // the same light at every position (no shimmer while it moves), in proportion to its energy
    // above the threshold. (9 loads: a [1 3 3 1] tent over 4 × 4 is as stable but costs twice
    // as much at 4K, a 2 × 2 box half but leaves blocky glows along thin lines.) Destination
    // pixel i is centered on scene pixel 2i, half a scene pixel before the 2 × 2 block it
    // stands for: the composite samples bloom half a scene pixel further (composite.wgsl).
    let lo = vec2<i32>(bloom.src_rect.xy);
    let hi = lo + vec2<i32>(bloom.src_rect.zw);
    let base = lo + vec2<i32>(in.pos.xy) * 2 - 1;
    let e = bloom_exposure_multiplier();
    if (bloom.flags.w != 0u) {
        // The low tiers (`Bloom::fast`): the same tent from 4 bilinear taps half a pixel
        // around scene pixel 2i (each the mean of 2 × 2 pixels, so the 3 × 3 get the weights
        // [1 2 1] / 4 per axis), then the exposure, threshold and limit once on the mean: 4
        // fetches and one threshold instead of 9 and 9. A source of 1–2 pixels is averaged
        // with its dark neighbours before the threshold, so it glows less and by an amount
        // that depends on where it sits (what the per-pixel path above removes). At the rect's
        // edge the taps repeat the edge pixels instead of reading 0.
        let c = bloom.src_rect.xy + floor(in.pos.xy) * 2.0 + 0.5;
        let mean = (bloom_tap_scene(c + vec2<f32>(-0.5, -0.5)) + bloom_tap_scene(c + vec2<f32>(0.5, -0.5))
            + bloom_tap_scene(c + vec2<f32>(-0.5, 0.5)) + bloom_tap_scene(c + vec2<f32>(0.5, 0.5))) * 0.25;
        return vec4<f32>(min(bloom_limit(bloom_threshold(mean * e)), vec3<f32>(S3D_BLOOM_MAX)), 1.0);
    }
    var sum = vec3<f32>(0.0);
    for (var y = 0; y < 3; y += 1) {
        var row = vec3<f32>(0.0);
        for (var x = 0; x < 3; x += 1) {
            row += bloom_pixel(base + vec2<i32>(x, y), lo, hi, e) * bloom_tent(x);
        }
        sum += row * bloom_tent(y);
    }
    return vec4<f32>(min(sum, vec3<f32>(S3D_BLOOM_MAX)), 1.0);
}

// The overlaps of source pixels `i` and `i + 1` (pixel `i` spans [i, i + 1]) with the box
// [lo, hi], as one bilinear tap: its position (texel centers at +0.5) and its weight.
fn bloom_area_pair(i: f32, lo: f32, hi: f32) -> vec2<f32> {
    let w0 = max(min(i + 1.0, hi) - max(i, lo), 0.0);
    let w1 = max(min(i + 2.0, hi) - max(i + 1.0, lo), 0.0);
    let w = w0 + w1;
    return vec2<f32>(i + 0.5 + w1 / max(w, 1e-6), w);
}

@fragment
fn fs_resample(in: FullscreenOut) -> @location(0) vec4<f32> {
    // The destination pixel's box in source pixels (its pixel center `in.pos` is exact),
    // clamped to the source, and `flags.yz` bilinear taps per axis (post/bloom.rs
    // `resample_taps`): uniform loop bounds. Taps outside the box weigh 0.
    let b = bloom.threshold.xy;
    let c = in.pos.xy * b;
    let lo = max(c - 0.5 * b, vec2<f32>(0.0));
    let hi = min(c + 0.5 * b, bloom.src_rect.zw);
    let first = floor(lo);
    var sum = vec3<f32>(0.0);
    var total = 0.0;
    for (var y = 0u; y < bloom.flags.z; y += 1u) {
        let ty = bloom_area_pair(first.y + 2.0 * f32(y), lo.y, hi.y);
        for (var x = 0u; x < bloom.flags.y; x += 1u) {
            let tx = bloom_area_pair(first.x + 2.0 * f32(x), lo.x, hi.x);
            let w = tx.y * ty.y;
            sum += bloom_tap(vec2<f32>(tx.x, ty.x)) * w;
            total += w;
        }
    }
    return vec4<f32>(sum / max(total, 1e-6), 1.0);
}

@fragment
fn fs_down(in: FullscreenOut) -> @location(0) vec4<f32> {
    let t = taps13(in.uv, false);
    let c = t.e * 0.125 + (t.a + t.c + t.g + t.i) * 0.03125 + (t.b + t.d + t.f + t.h) * 0.0625
        + (t.j + t.k + t.l + t.m) * 0.125;
    return vec4<f32>(c, 1.0);
}

@fragment
fn fs_up(in: FullscreenOut) -> @location(0) vec4<f32> {
    let p = bloom.src_rect.xy + in.uv * bloom.src_rect.zw;
    let s = bloom.src_texel.z;
    var c = bloom_tap(p) * 4.0;
    c += (bloom_tap(p + vec2<f32>(-s, 0.0)) + bloom_tap(p + vec2<f32>(s, 0.0))
        + bloom_tap(p + vec2<f32>(0.0, -s)) + bloom_tap(p + vec2<f32>(0.0, s))) * 2.0;
    c += bloom_tap(p + vec2<f32>(-s, -s)) + bloom_tap(p + vec2<f32>(s, -s))
        + bloom_tap(p + vec2<f32>(-s, s)) + bloom_tap(p + vec2<f32>(s, s));
    return vec4<f32>(c * (1.0 / 16.0), 1.0);
}
