// Space3d composite (graphics spec §1.9): the one pass that turns a post group's scene color S
// into the display target D (or, with FXAA, into the LDR intermediate L that the FXAA pass reads).
// The renderer prepends OUTPUT_WGSL, the hash functions and FULLSCREEN_WGSL. Every effect is a
// uniform bit (`PostParams::flags.x`); the output kind is the pipeline override `S3D_POST_OUT`.
// Only explicit-LOD sampling and loads: no derivatives, so no uniformity constraints
// (WebGL2-safe).
//
// Order (normative): the distortion offset → S at the rect [+ chromatic aberration taps] →
// un-premultiply (transparent
// groups) → exposure (manual, or 2^(E + compensation) with auto exposure) → bloom (at the same
// offset) → vignette →
// white balance × saturation, contrast → `post_hdr_hook` → tonemap → lift/gamma/gain, LUT (in
// the sRGB-encoded display space) → `post_ldr_hook` → display gamma (`PostProcess::display_gamma`,
// on the encoded value) → in the last pass: grain → dither → encode
// (before FXAA: the dither of the 8-bit intermediate) → re-premultiply. The grading hooks (`ColorGrading::shader`, spliced at the HOOKS marker at
// the end) are identity functions unless a hook shader defines them.

// 0: D has an sRGB format (hardware encodes), 1: 8-bit D without one (encode here),
// 2: float D (linear, no dither).
override S3D_POST_OUT: u32 = 0u;

const S3D_POST_BLOOM: u32 = 1u;
const S3D_POST_AUTO_EXPOSURE: u32 = 2u;
const S3D_POST_VIGNETTE: u32 = 4u;
const S3D_POST_CA: u32 = 8u;
const S3D_POST_GRAIN: u32 = 16u;
const S3D_POST_DITHER: u32 = 32u;
const S3D_POST_GRADE: u32 = 64u;
const S3D_POST_LUT: u32 = 128u;
const S3D_POST_DISTORTION: u32 = 256u;
const S3D_POST_TRANSPARENT: u32 = 512u;
const S3D_POST_SCENE_ALPHA: u32 = 1024u;
const S3D_POST_EXACT: u32 = 2048u;
const S3D_POST_GRADE_LDR: u32 = 4096u;
const S3D_POST_BLOOM_BILINEAR: u32 = 8192u;
const S3D_POST_DISPLAY_GAMMA: u32 = 16384u;

// Matches `GpuPost` (post/chain.rs): one 256-byte slot per post group.
struct PostParams {
    out_rect: vec4<f32>,        // the group's rect in D pixels (x, y, w, h), unclipped
    src_rect: vec4<f32>,        // its rect in S pixels, unclipped
    src_texel: vec4<f32>,       // 1/S.w, 1/S.h, rect aspect (w/h), D height
    exposure: vec4<f32>,        // linear multiplier (the compensation with auto exposure), 1 / display gamma, bloom intensity, bloom mode (0 add, 1 mix)
    vignette: vec4<f32>,        // intensity, smoothness, roundness, center.x
    vignette_color: vec4<f32>,  // linear multiplier at full vignette (rgb), center.y
    lens: vec4<f32>,            // chromatic aberration scale (S px⁻²), -, distortion scale (view heights per encoded unit), distortion chromatic split
    grain: vec4<f32>,           // intensity, cell size (D px), luminance response, -
    grade_matrix: mat3x3<f32>,  // white balance × saturation (linear HDR)
    lift_contrast: vec4<f32>,   // lift (offset), contrast (log2, around middle grey)
    gamma_lut: vec4<f32>,       // 1 / gamma (power), LUT contribution
    gain_lut: vec4<f32>,        // gain (slope), LUT size N
    flags: vec4<u32>,           // x effect bits, y tonemapper id, z 1 = last pass (0: FXAA follows), w 1 = the LUT texture is sRGB
    frame: vec4<u32>,           // reserved (0): the seeds are `post_seeds`
}

@group(0) @binding(0) var<uniform> post: PostParams;
@group(0) @binding(1) var post_sampler: sampler;
// This frame's pre-hashed seeds (s3d_pcg(seed)): x grain, y dither, z dither (second). A window
// of frames is uploaded at once and the frame's slot bound with a dynamic offset (post/mod.rs
// `SeedWindow`), so a still frame uploads nothing.
@group(0) @binding(2) var<uniform> post_seeds: vec4<u32>;
@group(1) @binding(0) var scene_tex: texture_2d<f32>;
@group(1) @binding(1) var bloom_tex: texture_2d<f32>;
@group(1) @binding(2) var exposure_tex: texture_2d<f32>;
@group(1) @binding(3) var lut_tex: texture_2d<f32>;
@group(1) @binding(4) var distortion_tex: texture_2d<f32>;

fn post_has(bit: u32) -> bool {
    return (post.flags.x & bit) != 0u;
}

// `s3d_hash01(p, seed)` with `hashed = s3d_pcg(seed)` computed on the CPU (bit-identical).
fn post_hash01(p: vec2<u32>, hashed: u32) -> f32 {
    return f32(s3d_pcg(p.x ^ s3d_pcg(p.y ^ hashed))) * (1.0 / 4294967295.0);
}

// The position in S (pixels) of D pixel center `px`: the same offset when S and D rects have the
// same size (render scale 1), else the rect mapped onto the rect.
fn post_scene_pos(px: vec2<f32>) -> vec2<f32> {
    if (post_has(S3D_POST_EXACT)) {
        return px - post.out_rect.xy + post.src_rect.xy;
    }
    let uv = (px - post.out_rect.xy) / post.out_rect.zw;
    return post.src_rect.xy + uv * post.src_rect.zw;
}

// The rect's part inside S (pixels): lo, hi.
fn post_scene_clip() -> vec4<f32> {
    let size = 1.0 / post.src_texel.xy;
    let lo = max(post.src_rect.xy, vec2<f32>(0.0));
    let hi = min(post.src_rect.xy + post.src_rect.zw, size);
    return vec4<f32>(lo, max(lo, hi));
}

// Bilinear scene color at S position `s`, every tap inside the rect (split screens never bleed
// into each other).
fn post_scene_bilinear(s_in: vec2<f32>) -> vec4<f32> {
    let clip = post_scene_clip();
    let s = clamp(s_in, clip.xy + 0.5, max(clip.xy + 0.5, clip.zw - 0.5));
    return textureSampleLevel(scene_tex, post_sampler, s * post.src_texel.xy, 0.0);
}

// The scene color at S position `s`: a texel load at render scale 1 (exact), else bilinear.
fn post_scene(s: vec2<f32>) -> vec4<f32> {
    if (post_has(S3D_POST_EXACT)) {
        return textureLoad(scene_tex, vec2<i32>(floor(s)), 0);
    }
    return post_scene_bilinear(s);
}

// The exposure multiplier: manual 2^stops, or with auto exposure 2^(E + compensation).
fn post_exposure() -> f32 {
    var e = post.exposure.x;
    if (post_has(S3D_POST_AUTO_EXPOSURE)) {
        e = e * exp2(textureLoad(exposure_tex, vec2<i32>(0), 0).r);
    }
    return e;
}

// White balance × saturation, then contrast in log2 space around middle grey (linear HDR).
fn post_grade_hdr(c: vec3<f32>) -> vec3<f32> {
    var g = max(post.grade_matrix * c, vec3<f32>(0.0));
    let k = post.lift_contrast.w;
    if (k != 1.0) {
        g = pow(max(g, vec3<f32>(1e-10)) * (1.0 / 0.18), vec3<f32>(k)) * 0.18;
    }
    return g;
}

// 3D LUT lookup of the sRGB-encoded color `e` (0..1) in the N²×N strip: bilinear inside the
// red/green slice, blended between the two nearest blue slices.
fn post_lut(e: vec3<f32>) -> vec3<f32> {
    let n = post.gain_lut.w;
    let b = e.b * (n - 1.0);
    let b0 = floor(b);
    let b1 = min(b0 + 1.0, n - 1.0);
    let x = (e.r * (n - 1.0) + 0.5) / n;
    let y = (e.g * (n - 1.0) + 0.5) / n;
    let s0 = textureSampleLevel(lut_tex, post_sampler, vec2<f32>((x + b0) / n, y), 0.0).rgb;
    let s1 = textureSampleLevel(lut_tex, post_sampler, vec2<f32>((x + b1) / n, y), 0.0).rgb;
    var g = mix(s0, s1, b - b0);
    if ((post.flags.w & 1u) != 0u) {
        // An sRGB-format strip is decoded when sampled: back to its encoded values.
        g = s3d_linear_to_srgb(clamp(g, vec3<f32>(0.0), vec3<f32>(1.0)));
    }
    return g;
}

// Lift/gamma/gain (ASC CDL: (e · gain + lift)^(1 / gamma)) and the LUT, on the sRGB-encoded
// display color (as grading tools and `.cube` files expect). Display-linear in and out.
fn post_grade_ldr(c: vec3<f32>) -> vec3<f32> {
    var e = s3d_linear_to_srgb(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)));
    if (post_has(S3D_POST_GRADE_LDR)) {
        e = pow(max(e * post.gain_lut.xyz + post.lift_contrast.xyz, vec3<f32>(0.0)), post.gamma_lut.xyz);
    }
    if (post_has(S3D_POST_LUT)) {
        e = clamp(e, vec3<f32>(0.0), vec3<f32>(1.0));
        e = mix(e, post_lut(e), post.gamma_lut.w);
    }
    return s3d_srgb_to_linear(clamp(e, vec3<f32>(0.0), vec3<f32>(1.0)));
}

// The display gamma (`PostProcess::display_gamma`, a player's brightness calibration): the
// sRGB-encoded display color raised to 1 / gamma, the last step of the grading. Black and white
// stay, the midtones move; nothing before the tonemapper sees it, so exposure, bloom and the
// grading are what they were. Display-linear in and out. CPU mirror: `apply_display_gamma`.
fn post_display_gamma(c: vec3<f32>) -> vec3<f32> {
    let e = s3d_linear_to_srgb(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)));
    return s3d_srgb_to_linear(pow(e, vec3<f32>(post.exposure.y)));
}

// Bloom level 0 at S position `s` (already exposed and thresholded). The chain covers the
// rect's part inside S; its prefilter centers each texel half a scene pixel before the 2 × 2
// block it stands for (bloom.wgsl `fs_prefilter`), so the lookup moves half a pixel on. Level 0
// is magnified by a cubic B-spline (4 bilinear taps, Sigg & Hadwiger 2005) rather than bilinear:
// smoother, and every texel adds the same light to the frame at any magnification (bilinear at
// 2.67×, 720p and 1440p, weighed texels 3 % apart, so a small glow pulsed as it moved). The low
// tiers (`Bloom::fast`, S3D_POST_BLOOM_BILINEAR) take the one bilinear tap: a third of this
// pass on an integrated GPU.
fn post_bloom(s: vec2<f32>) -> vec3<f32> {
    let clip = post_scene_clip();
    let uv = (s + 0.5 - clip.xy) / max(clip.zw - clip.xy, vec2<f32>(1.0));
    if (post_has(S3D_POST_BLOOM_BILINEAR)) {
        return textureSampleLevel(bloom_tex, post_sampler, uv, 0.0).rgb;
    }
    let size = vec2<f32>(textureDimensions(bloom_tex));
    let t = uv * size - 0.5;
    let i = floor(t);
    let f = t - i;
    let f2 = f * f;
    let f3 = f2 * f;
    let w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
    let w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
    let w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
    let w3 = f3 / 6.0;
    let g0 = w0 + w1;
    let g1 = w2 + w3;
    // Tap positions (texel centers at +0.5) and normalized coordinates.
    let p0 = (i - 0.5 + w1 / g0) / size;
    let p1 = (i + 1.5 + w3 / g1) / size;
    let a = textureSampleLevel(bloom_tex, post_sampler, vec2<f32>(p0.x, p0.y), 0.0).rgb;
    let b = textureSampleLevel(bloom_tex, post_sampler, vec2<f32>(p1.x, p0.y), 0.0).rgb;
    let c = textureSampleLevel(bloom_tex, post_sampler, vec2<f32>(p0.x, p1.y), 0.0).rgb;
    let d = textureSampleLevel(bloom_tex, post_sampler, vec2<f32>(p1.x, p1.y), 0.0).rgb;
    return g0.y * (g0.x * a + g1.x * b) + g1.y * (g0.x * c + g1.x * d);
}

// The distortion at S position `s` (the undistorted pixel), in S pixels: the post group's
// distortion buffer (split signs, DISTORTION_WGSL; half or full resolution over the rect,
// sampled bilinearly) holds where each pixel looks, in view heights with +y up.
fn post_distortion(s: vec2<f32>) -> vec2<f32> {
    let uv = (s - post.src_rect.xy) / post.src_rect.zw;
    let d = textureSampleLevel(distortion_tex, post_sampler, uv, 0.0);
    let o = (d.rg - d.ba) * post.lens.z;
    return vec2<f32>(o.x, -o.y) * post.src_rect.w;
}

// Red and blue apart from green, which is sampled at `s0 + ds`: chromatic aberration (radially
// from the rect's center, growing with r²: `lens.x` puts intensity × 2 % of the rect width at
// the corners) and the distortion's spectral split (R and B at (1 ∓ lens.w) × the offset `ds`).
// Up to 1.5 px from green, red and blue are single taps; further apart a spectrum of taps at
// most 1.5 pixels apart (each bilinear tap covers ±1 px, so a star's copies overlap) spans the
// split, red fading into green into blue, so a star under a strong shockwave or at a corner
// smears into a small rainbow instead of splitting into separate red, green and blue dots. At
// most 17 taps: beyond a 12-px half split (CA far above 0.5 at 1080p) they spread further
// apart, so the cost stays bounded whatever the intensity.
fn post_chromatic(s0: vec2<f32>, ds: vec2<f32>, c: vec4<f32>) -> vec4<f32> {
    var off = vec2<f32>(0.0);
    if (post_has(S3D_POST_CA)) {
        let d = s0 + ds - (post.src_rect.xy + post.src_rect.zw * 0.5);
        off = d * (dot(d, d) * post.lens.x);
    }
    let k = post.lens.w;
    // Half the red–blue split, in S pixels.
    let v = ds * k + off;
    if (dot(v, v) <= 2.25) {
        let r = post_scene_bilinear(s0 + ds - v).r;
        let b = post_scene_bilinear(s0 + ds + v).b;
        return vec4<f32>(r, c.g, b, c.a);
    }
    // Taps at t = k / m, k = −m … m: at most 1.5 pixels apart, at most 17.
    let m = i32(clamp(ceil(length(v) / 1.5), 1.0, 8.0));
    var sum = vec3<f32>(0.0);
    var weight = vec3<f32>(0.0);
    for (var k = -m; k <= m; k++) {
        let t = f32(k) / f32(m);
        // Tents: red on the −v side, green in the middle, blue on the +v side.
        let w = vec3<f32>(max(-t, 0.0), 1.0 - abs(t), max(t, 0.0));
        var x = c.rgb;
        if (k != 0) {
            x = post_scene_bilinear(s0 + ds + v * t).rgb;
        }
        sum += w * x;
        weight += w;
    }
    return vec4<f32>(sum / weight, c.a);
}

// Vignette multiplier at `uv` (0..1 over the rect in D): 1 inside, `vignette_color` × intensity
// toward the corners (distance 1 at the corners of a centered vignette; `roundness` 1 = circular,
// 0 = the rect's shape; the falloff starts `smoothness` before the corners).
fn post_vignette(uv: vec2<f32>) -> vec3<f32> {
    let v = post.vignette;
    let k = vec2<f32>(mix(1.0, post.src_texel.z, v.z), 1.0);
    let d = (uv - vec2<f32>(v.w, post.vignette_color.w)) * 2.0 * k;
    let r = length(d) / length(k);
    let f = smoothstep(1.0 - v.y, 1.0, r) * v.x;
    return mix(vec3<f32>(1.0), post.vignette_color.rgb, f);
}

// Film grain in the encoded (sRGB) space at D pixel `px`: value noise over cells of
// `grain.y` pixels (bilinear between hashed corners), mono, amplitude intensity, fading in the
// highlights with `luminance_response`.
fn post_grain(e: vec3<f32>, px: vec2<f32>) -> vec3<f32> {
    let g = max(px - post.out_rect.xy, vec2<f32>(0.0)) / post.grain.y;
    let i = vec2<u32>(floor(g));
    let f = fract(g);
    let u = f * f * (3.0 - 2.0 * f);
    let seed = post_seeds.x;
    let a = post_hash01(i, seed);
    let b = post_hash01(i + vec2<u32>(1u, 0u), seed);
    let c = post_hash01(i + vec2<u32>(0u, 1u), seed);
    let d = post_hash01(i + vec2<u32>(1u, 1u), seed);
    let n = mix(mix(a, b, u.x), mix(c, d, u.x), u.y) * 2.0 - 1.0;
    let luma = dot(e, vec3<f32>(0.2126, 0.7152, 0.0722));
    let amp = post.grain.x * mix(1.0, 1.0 - clamp(luma, 0.0, 1.0), post.grain.z);
    return e + vec3<f32>(n * amp);
}

// Decorrelates the dither of the LDR intermediate L from the last pass's (both are hashed from
// the frame's two dither seeds at the same pixel).
const S3D_POST_DITHER_L: u32 = 0xA511E9B3u;

// The dither of the 8-bit LDR intermediate L (the composite's output when FXAA follows):
// triangular, ±1 LSB in the encoded space, before the sRGB target quantizes. Without it L holds
// the bands of a smooth dark gradient and the last pass's dither only adds noise to them.
// `rgb_in` is display-linear, and so is the result (L's format encodes).
fn post_dither_intermediate(rgb_in: vec3<f32>, px_f: vec2<f32>) -> vec3<f32> {
    let px = vec2<u32>(px_f);
    let n = post_hash01(px, post_seeds.y ^ S3D_POST_DITHER_L) - post_hash01(px, post_seeds.z ^ S3D_POST_DITHER_L);
    let e = s3d_linear_to_srgb(clamp(rgb_in, vec3<f32>(0.0), vec3<f32>(1.0))) + vec3<f32>(n / 255.0);
    return s3d_srgb_to_linear(clamp(e, vec3<f32>(0.0), vec3<f32>(1.0)));
}

// Last-pass finish: grain and dither (triangular, ±1 LSB) in the encoded space, and the encoding
// for D. `rgb_in` is display-linear.
fn post_finish(rgb_in: vec3<f32>, px_f: vec2<f32>) -> vec3<f32> {
    return post_finish_with(rgb_in, px_f, true);
}

// `post_finish`, with the dither only where `may_dither` (the FXAA pass leaves it out where it
// copied an already dithered pixel of L).
fn post_finish_with(rgb_in: vec3<f32>, px_f: vec2<f32>, may_dither: bool) -> vec3<f32> {
    let grain = post_has(S3D_POST_GRAIN);
    // Float D keeps linear values and gets no dither.
    let dither = may_dither && post_has(S3D_POST_DITHER) && S3D_POST_OUT != 2u;
    if (!grain && !dither) {
        if (S3D_POST_OUT == 1u) {
            return s3d_linear_to_srgb(clamp(rgb_in, vec3<f32>(0.0), vec3<f32>(1.0)));
        }
        return rgb_in;
    }
    var e = s3d_linear_to_srgb(clamp(rgb_in, vec3<f32>(0.0), vec3<f32>(1.0)));
    if (grain) {
        e = post_grain(e, px_f);
    }
    if (dither) {
        let px = vec2<u32>(px_f);
        e = e + vec3<f32>((post_hash01(px, post_seeds.y) - post_hash01(px, post_seeds.z)) / 255.0);
    }
    e = clamp(e, vec3<f32>(0.0), vec3<f32>(1.0));
    if (S3D_POST_OUT == 1u) {
        return e;
    }
    return s3d_srgb_to_linear(e);
}

@fragment
fn fs_composite(in: FullscreenOut) -> @location(0) vec4<f32> {
    let s0 = post_scene_pos(in.pos.xy);
    var ds = vec2<f32>(0.0);
    if (post_has(S3D_POST_DISTORTION)) {
        ds = post_distortion(s0);
    }
    // Where no writer drew, exactly the undistorted pixel; elsewhere a bilinear tap.
    let distorted = any(ds != vec2<f32>(0.0));
    let s = s0 + ds;
    var c: vec4<f32>;
    if (distorted) {
        c = post_scene_bilinear(s);
    } else {
        c = post_scene(s);
    }
    if (post_has(S3D_POST_CA) || (distorted && post.lens.w > 0.0)) {
        c = post_chromatic(s0, ds, c);
    }
    let a = select(1.0, c.a, post_has(S3D_POST_SCENE_ALPHA));
    var rgb = c.rgb;
    let transparent = post_has(S3D_POST_TRANSPARENT);
    if (transparent && a > 1.0 / 255.0) {
        rgb = rgb / a;
    }
    rgb = rgb * post_exposure();
    // Bloom is in exposed units (the prefilter exposed it): added after exposure, never exposed
    // twice. Transparent groups weight it by alpha below (no haze where nothing was drawn).
    if (post_has(S3D_POST_BLOOM)) {
        let b = post_bloom(s);
        if (post.exposure.w > 0.5) {
            rgb = mix(rgb, b, post.exposure.z);
        } else {
            rgb = rgb + b * post.exposure.z;
        }
    }
    if (post_has(S3D_POST_VIGNETTE)) {
        rgb = rgb * post_vignette((in.pos.xy - post.out_rect.xy) / post.out_rect.zw);
    }
    let uv = (in.pos.xy - post.out_rect.xy) / post.out_rect.zw;
    if (post_has(S3D_POST_GRADE)) {
        rgb = post_grade_hdr(rgb);
    }
    rgb = post_hdr_hook(rgb, uv);
    rgb = s3d_tonemap(post.flags.y, rgb);
    if (post_has(S3D_POST_GRADE_LDR | S3D_POST_LUT)) {
        rgb = post_grade_ldr(rgb);
    }
    rgb = post_ldr_hook(rgb, uv);
    if (post_has(S3D_POST_DISPLAY_GAMMA)) {
        rgb = post_display_gamma(rgb);
    }
    if (post.flags.z != 0u) {
        rgb = post_finish(rgb, in.pos.xy);
    } else if (post_has(S3D_POST_DITHER)) {
        rgb = post_dither_intermediate(rgb, in.pos.xy);
    }
    if (transparent) {
        rgb = rgb * a;
    }
    return vec4<f32>(rgb, a);
}

// Grading hooks (`ColorGrading::shader`): `post_hdr_hook` (linear HDR, before the tonemapper) and
// `post_ldr_hook` (display-linear, after the LUT), `fn(c: vec3<f32>, uv: vec2<f32>) ->
// vec3<f32>` with `uv` 0..1 over the post group's rect. Identity unless a hook shader defines them.
//#HOOKS
