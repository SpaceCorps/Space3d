// Space3d forward shader: instanced meshes, directional + point lights, metallic/roughness
// Blinn-Phong with Schlick Fresnel (or GGX with `S3D_SHADING_STANDARD`), material maps
// (normal, metallic-roughness, occlusion, emissive), image-based lighting from the camera's
// `EnvironmentLight`, the shadow of one directional light (PCF with `S3D_SHADOW_TAPS`),
// linear-space lighting, sRGB output.
//
// The INSTANCE marker below is replaced by the renderer with an `InstanceInput` struct and
// accessor functions generated from the instance layout (see `pipeline/instance.rs`); the CONSTS
// marker with structural constants; the VIEW marker with the shared `View` struct
// (`shaders/common/view.wgsl`); the OUTPUT marker with the shared output tail (`OUTPUT_WGSL`:
// `s3d_output`, tonemappers); the TEXTURE marker with the material texture bindings
// (`texture_2d` for single-layer textures, `texture_2d_array` for arrays: one kind for the
// color maps, base color and emissive, one for the data maps, normal, metallic-roughness and
// occlusion) and their sampling functions (`sample_base`, `s3d_sample_*`); the EXT markers with
// the extension vertex stream (tangent, second UV set: `VertexInput` fields, `VsOut` fields and
// the `s3d_vertex_ext` / `s3d_frag_*` accessors, or constant stand-ins without the stream);
// the VARYINGS marker with extra `VsOut` fields and the HOOKS marker with `vertex_hook` /
// `fragment_hook` — identity functions, or a material's `CustomShader`.
// Single-layer textures are never bound as arrays: wgpu's GL backend binds them with the
// TEXTURE_2D target, which a `sampler2DArray` reads as black on WebGL2/GLES.
//
// A material without maps, with the default overrides, computes exactly what the shader did
// before maps existed: every map is behind a branch on its (uniform) flag, and the defaults
// bound for absent maps are never read.

//#CONSTS

// Pipeline-overridable constants (graphics spec §3.1), specialized per pipeline. Every one has
// a default, so a module never changes shape with them.
// `S3D_OUTPUT`: how `fs_main` encodes its color (`OutputEncoding::id`: 0 linear, 1 manual sRGB,
// 2 tonemapped). `MANUAL_SRGB` is set with it (`S3D_OUTPUT == 1`), for custom shaders.
override S3D_OUTPUT: u32 = 0u;
override MANUAL_SRGB: bool = false;
override S3D_ALPHA_MASK: bool = false;
override S3D_ALPHA_TO_COVERAGE: bool = false;
override S3D_SHADING_STANDARD: bool = false;
override S3D_SHADOW_TAPS: u32 = 0u;
override S3D_NORMAL_MAPS: bool = true;

const MAX_DIR_LIGHTS: u32 = 4u;
const MAX_POINT_LIGHTS: u32 = 16u;

// Material flags (`MaterialParams::flags.x`, `FLAG_*` in `material.rs`).
const S3D_MAP_BASE: u32 = 1u;
const S3D_MAP_NORMAL: u32 = 2u;
const S3D_MAP_MR: u32 = 4u;
const S3D_MAP_OCCLUSION: u32 = 8u;
const S3D_MAP_EMISSIVE: u32 = 16u;
const S3D_ORM_PACKED: u32 = 32u;
const S3D_UV_TRANSFORM: u32 = 64u;
const S3D_FIXUP_NORMAL: u32 = 256u;
const S3D_FIXUP_MR: u32 = 512u;
const S3D_FIXUP_OCCLUSION: u32 = 1024u;
const S3D_NO_SHADOW: u32 = 2048u;
const S3D_PI: f32 = 3.141592653589793;

//#VIEW

struct MaterialParams {
    base_color: vec4<f32>,
    emissive: vec4<f32>,    // w: unlit flag
    params: vec4<f32>,      // x: metallic, y: roughness, z: has texture
    maps: vec4<f32>,        // x: normal scale, y: occlusion strength, z: alpha cutoff
    flags: vec4<u32>,       // x: S3D_MAP_* | S3D_ORM_PACKED | S3D_UV_TRANSFORM | S3D_FIXUP_*,
                            // y: maps sampled with the second UV set (S3D_MAP_* bits)
    uv_row0: vec4<f32>,     // KHR_texture_transform: uv' = (dot(row0, uv1), dot(row1, uv1))
    uv_row1: vec4<f32>,
}

@group(0) @binding(0) var<uniform> view: View;
// Image-based lighting (group 0 bindings 1–4; 1×1 black cubes without an environment): the
// GGX-prefiltered specular cube (roughness × `env_params.z` = its mip), the irradiance cube
// (cosine-weighted mean radiance) and the split-sum BRDF LUT (n·v, roughness), which
// `S3D_SHADING_STANDARD` also uses for multiple-scattering compensation.
@group(0) @binding(1) var s3d_env_specular: texture_cube<f32>;
@group(0) @binding(2) var s3d_env_irradiance: texture_cube<f32>;
@group(0) @binding(3) var s3d_brdf_lut: texture_2d<f32>;
@group(0) @binding(4) var s3d_env_sampler: sampler;
// The casting directional light's shadow map (group 0 bindings 5–6; a 1×1 map without a
// shadow, which `shadow_info.x == 0` never reads) and its comparison sampler (LessEqual,
// bilinear: every tap is a 2×2 PCF).
@group(0) @binding(5) var s3d_shadow_map: texture_depth_2d;
@group(0) @binding(6) var s3d_shadow_sampler: sampler_comparison;
@group(1) @binding(0) var<uniform> material: MaterialParams;
@group(1) @binding(2) var base_sampler: sampler;
//#TEXTURE

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
//#EXT_VERTEX
}

//#INSTANCE

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) @interpolate(flat) layer: i32,
//#VARYINGS
}

//#EXT

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

//#OUTPUT

fn quat_to_mat3(q_in: vec4<f32>) -> mat3x3<f32> {
    let q = normalize(q_in);
    let x2 = q.x + q.x;
    let y2 = q.y + q.y;
    let z2 = q.z + q.z;
    let xx = q.x * x2;
    let xy = q.x * y2;
    let xz = q.x * z2;
    let yy = q.y * y2;
    let yz = q.y * z2;
    let zz = q.z * z2;
    let wx = q.w * x2;
    let wy = q.w * y2;
    let wz = q.w * z2;
    return mat3x3<f32>(
        vec3<f32>(1.0 - (yy + zz), xy + wz, xz - wy),
        vec3<f32>(xy - wz, 1.0 - (xx + zz), yz + wx),
        vec3<f32>(xz + wy, yz - wx, 1.0 - (xx + yy)),
    );
}

fn trs(t: vec3<f32>, r: mat3x3<f32>, s: vec3<f32>) -> mat4x4<f32> {
    return mat4x4<f32>(
        vec4<f32>(r[0] * s.x, 0.0),
        vec4<f32>(r[1] * s.y, 0.0),
        vec4<f32>(r[2] * s.z, 0.0),
        vec4<f32>(t, 1.0),
    );
}

// The material's UV transform (KHR_texture_transform), applied to both UV sets before the
// per-instance atlas rect. Identity transforms are skipped (exact UVs).
fn s3d_uv_transform(uv: vec2<f32>) -> vec2<f32> {
    if ((material.flags.x & S3D_UV_TRANSFORM) == 0u) {
        return uv;
    }
    let p = vec3<f32>(uv, 1.0);
    return vec2<f32>(dot(material.uv_row0.xyz, p), dot(material.uv_row1.xyz, p));
}

// The UV set a map samples: `uv1` for the maps in `flags.y` (which is UV0 on meshes without a
// second set), else `uv0`.
fn s3d_map_uv(map: u32, uv0: vec2<f32>, uv1: vec2<f32>) -> vec2<f32> {
    return select(uv0, uv1, (material.flags.y & map) != 0u);
}

//#HOOKS

@vertex
fn vs_main(v: VertexInput, inst: InstanceInput) -> VsOut {
    let model = instance_model(inst);
    let world = model * vec4<f32>(v.position, 1.0);
    // Normal matrix = cofactor of the upper 3x3 (inverse-transpose up to scale), sign-corrected
    // for mirrored transforms. Handles non-uniform scale without a CPU-side inverse.
    let a = model[0].xyz;
    let b = model[1].xyz;
    let c = model[2].xyz;
    let cof = mat3x3<f32>(cross(b, c), cross(c, a), cross(a, b));
    let det = dot(a, cof[0]);
    let n = cof * v.normal * select(1.0, -1.0, det < 0.0);

    var out: VsOut;
    out.clip = view.view_proj * world;
    out.world_pos = world.xyz;
    out.normal = n;
    let rect = instance_uv_rect(inst);
    out.uv = s3d_uv_transform(v.uv) * rect.zw + rect.xy;
    out.color = vec4<f32>(srgb_to_linear(v.color.rgb), v.color.a) * instance_color(inst);
    out.layer = instance_layer(inst);
    out = s3d_vertex_ext(v, out, mat3x3<f32>(a, b, c), det < 0.0);
    return vertex_hook(v, inst, out);
}

// Split-sum environment BRDF: (scale, bias) of f0 for n·v and perceptual roughness.
fn s3d_env_brdf(ndv: f32, roughness: f32) -> vec2<f32> {
    return textureSampleLevel(s3d_brdf_lut, s3d_env_sampler, vec2<f32>(ndv, roughness), 0.0).rg;
}

// Energy the single-scattering GGX lobe loses to multiple scattering, per color (Standard
// shading; set per fragment before the light loops, 1 otherwise).
var<private> s3d_ms_energy: vec3<f32> = vec3<f32>(1.0);

fn fresnel_schlick(f0: vec3<f32>, cos_theta: f32) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - cos_theta, 5.0);
}

fn brdf(base: vec3<f32>, metallic: f32, roughness: f32, n: vec3<f32>, v: vec3<f32>, l: vec3<f32>) -> vec3<f32> {
    let ndl = max(dot(n, l), 0.0);
    if (ndl <= 0.0) {
        return vec3<f32>(0.0);
    }
    let h = normalize(l + v);
    let ndh = max(dot(n, h), 0.0);
    let vdh = max(dot(v, h), 0.0);
    let r = clamp(roughness, 0.02, 1.0);
    let shininess = exp2(12.0 * (1.0 - r) + 1.0);
    let f0 = mix(vec3<f32>(0.04), base, metallic);
    let f = fresnel_schlick(f0, vdh);
    let spec = f * pow(ndh, shininess) * (shininess + 8.0) / 8.0;
    let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    return (kd * base + spec) * ndl;
}

// `ShadingModel::Standard`: GGX distribution, height-correlated Smith visibility, Schlick
// Fresnel, Lambert diffuse. Scaled by π like `brdf`, whose lights are "a white Lambert surface
// facing the light reflects the light's color".
fn s3d_brdf_standard(base: vec3<f32>, metallic: f32, roughness: f32, n: vec3<f32>, v: vec3<f32>, l: vec3<f32>) -> vec3<f32> {
    let ndl = max(dot(n, l), 0.0);
    if (ndl <= 0.0) {
        return vec3<f32>(0.0);
    }
    let ndv = max(dot(n, v), 1e-4);
    let h = normalize(l + v);
    let ndh = max(dot(n, h), 0.0);
    let vdh = max(dot(v, h), 0.0);
    let r = clamp(roughness, 0.045, 1.0);
    let a2 = r * r * r * r;
    let dd = ndh * ndh * (a2 - 1.0) + 1.0;
    let d = a2 / (S3D_PI * dd * dd);
    let vis = 0.5 / (ndl * sqrt(ndv * ndv * (1.0 - a2) + a2) + ndv * sqrt(ndl * ndl * (1.0 - a2) + a2));
    let f0 = mix(vec3<f32>(0.04), base, metallic);
    let f = fresnel_schlick(f0, vdh);
    let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    return (kd * base + S3D_PI * d * vis * f * s3d_ms_energy) * ndl;
}

fn s3d_brdf(base: vec3<f32>, metallic: f32, roughness: f32, n: vec3<f32>, v: vec3<f32>, l: vec3<f32>) -> vec3<f32> {
    if (S3D_SHADING_STANDARD) {
        return s3d_brdf_standard(base, metallic, roughness, n, v, l);
    }
    return brdf(base, metallic, roughness, n, v, l);
}

// Image-based lighting of a lit fragment (split sum with multiple scattering, Fdez-Agüera 2019):
// specular = prefiltered radiance × (f0·A + B) (+ the multiply scattered part, lit by the
// irradiance), diffuse = irradiance × the albedo the specular lobes leave. Energy-conserving: a
// white or metallic surface in a uniform white environment reflects exactly its radiance.
// `ao` darkens both (specular with Lagarde's specular occlusion). Rotated by `env_rotation`,
// scaled by the `EnvironmentLight` intensities.
fn s3d_ibl(base: vec3<f32>, metallic: f32, roughness: f32, n: vec3<f32>, v: vec3<f32>, ao: f32) -> vec3<f32> {
    let ndv = clamp(dot(n, v), 1e-4, 1.0);
    let r = clamp(roughness, 0.0, 1.0);
    let ab = s3d_env_brdf(ndv, r);
    let radiance = textureSampleLevel(
        s3d_env_specular,
        s3d_env_sampler,
        view.env_rotation * reflect(-v, n),
        r * view.env_params.z,
    ).rgb;
    let irradiance = textureSampleLevel(s3d_env_irradiance, s3d_env_sampler, view.env_rotation * n, 0.0).rgb;
    let f0 = mix(vec3<f32>(0.04), base, metallic);
    let fr = max(vec3<f32>(1.0 - r), f0) - f0;
    let ks = f0 + fr * pow(1.0 - ndv, 5.0);
    let fss_ess = ks * ab.x + ab.y;
    let ems = 1.0 - (ab.x + ab.y);
    let f_avg = f0 + (vec3<f32>(1.0) - f0) / 21.0;
    let fms_ems = ems * fss_ess * f_avg / (vec3<f32>(1.0) - f_avg * ems);
    let kd = base * (1.0 - metallic) * (vec3<f32>(1.0) - fss_ess - fms_ems);
    let spec_ao = clamp(pow(ndv + ao, exp2(-16.0 * r - 1.0)) - 1.0 + ao, 0.0, 1.0);
    let specular = (fss_ess * radiance + fms_ems * irradiance) * view.env_params.y * spec_ao;
    let diffuse = kd * irradiance * view.env_params.x * ao;
    return specular + diffuse;
}

// Percentage-closer filtering of the shadow map at `uv` for light depth `z` (1 = lit):
// `S3D_SHADOW_TAPS` bilinear comparisons weighted into a smooth kernel (Castaño 2013) — 1 tap:
// 2×2 texels, 4 taps: 3×3, 9 taps: 5×5, 16 taps: 7×7 — centered on `uv`. Per axis, tap `k` sits
// `o[k]` texels from `base` with weight `w[k]`; the kernels are unrolled (the override selects
// one at pipeline creation).
fn s3d_shadow_tap(base: vec2<f32>, ox: f32, oy: f32, inv: f32, z: f32) -> f32 {
    return textureSampleCompareLevel(s3d_shadow_map, s3d_shadow_sampler, base + vec2<f32>(ox, oy) * inv, z);
}

fn s3d_shadow_pcf(uv: vec2<f32>, z: f32) -> f32 {
    if (S3D_SHADOW_TAPS <= 1u) {
        return textureSampleCompareLevel(s3d_shadow_map, s3d_shadow_sampler, uv, z);
    }
    let size = f32(view.shadow_info.z);
    let inv = 1.0 / size;
    let p = uv * size;
    let b = floor(p + vec2<f32>(0.5));
    let s = p + vec2<f32>(0.5) - b;
    let base = (b - vec2<f32>(0.5)) * inv;
    if (S3D_SHADOW_TAPS <= 4u) {
        let w0 = vec2<f32>(3.0) - 2.0 * s;
        let w1 = vec2<f32>(1.0) + 2.0 * s;
        let o0 = (vec2<f32>(2.0) - s) / w0 - 1.0;
        let o1 = s / w1 + 1.0;
        var sum = w0.x * w0.y * s3d_shadow_tap(base, o0.x, o0.y, inv, z);
        sum += w1.x * w0.y * s3d_shadow_tap(base, o1.x, o0.y, inv, z);
        sum += w0.x * w1.y * s3d_shadow_tap(base, o0.x, o1.y, inv, z);
        sum += w1.x * w1.y * s3d_shadow_tap(base, o1.x, o1.y, inv, z);
        return sum * (1.0 / 16.0);
    }
    if (S3D_SHADOW_TAPS <= 9u) {
        let w0 = vec2<f32>(4.0) - 3.0 * s;
        let w2 = vec2<f32>(1.0) + 3.0 * s;
        let o0 = (vec2<f32>(3.0) - 2.0 * s) / w0 - 2.0;
        let o1 = (vec2<f32>(3.0) + s) / 7.0;
        let o2 = s / w2 + 2.0;
        var row = w0.x * s3d_shadow_tap(base, o0.x, o0.y, inv, z);
        row += 7.0 * s3d_shadow_tap(base, o1.x, o0.y, inv, z);
        row += w2.x * s3d_shadow_tap(base, o2.x, o0.y, inv, z);
        var sum = w0.y * row;
        row = w0.x * s3d_shadow_tap(base, o0.x, o1.y, inv, z);
        row += 7.0 * s3d_shadow_tap(base, o1.x, o1.y, inv, z);
        row += w2.x * s3d_shadow_tap(base, o2.x, o1.y, inv, z);
        sum += 7.0 * row;
        row = w0.x * s3d_shadow_tap(base, o0.x, o2.y, inv, z);
        row += 7.0 * s3d_shadow_tap(base, o1.x, o2.y, inv, z);
        row += w2.x * s3d_shadow_tap(base, o2.x, o2.y, inv, z);
        sum += w2.y * row;
        return sum * (1.0 / 144.0);
    }
    let w0 = 5.0 * s - 6.0;
    let w1 = 11.0 * s - 28.0;
    let w2 = -(11.0 * s + 17.0);
    let w3 = -(5.0 * s + 1.0);
    let o0 = (4.0 * s - 5.0) / w0 - 3.0;
    let o1 = (4.0 * s - 16.0) / w1 - 1.0;
    let o2 = -(7.0 * s + 5.0) / w2 + 1.0;
    let o3 = -s / w3 + 3.0;
    let ox = vec4<f32>(o0.x, o1.x, o2.x, o3.x);
    let wx = vec4<f32>(w0.x, w1.x, w2.x, w3.x);
    var sum = w0.y * s3d_shadow_row4(base, ox, wx, o0.y, inv, z);
    sum += w1.y * s3d_shadow_row4(base, ox, wx, o1.y, inv, z);
    sum += w2.y * s3d_shadow_row4(base, ox, wx, o2.y, inv, z);
    sum += w3.y * s3d_shadow_row4(base, ox, wx, o3.y, inv, z);
    return sum * (1.0 / 2704.0);
}

// One row of the 16-tap kernel: four taps at `oy`.
fn s3d_shadow_row4(base: vec2<f32>, ox: vec4<f32>, wx: vec4<f32>, oy: f32, inv: f32, z: f32) -> f32 {
    var row = wx.x * s3d_shadow_tap(base, ox.x, oy, inv, z);
    row += wx.y * s3d_shadow_tap(base, ox.y, oy, inv, z);
    row += wx.z * s3d_shadow_tap(base, ox.z, oy, inv, z);
    row += wx.w * s3d_shadow_tap(base, ox.w, oy, inv, z);
    return row;
}

// The casting light's shadow at a fragment (1 = lit, 0 = fully shadowed): the world position
// moved along the geometric normal `ng` (more at grazing angles to the light direction `l`),
// compared against the map with the depth bias, filtered, scaled by the strength and faded out
// over the outer part of the map. 1 without a shadow this frame and for materials that do not
// receive.
fn s3d_shadow_factor(world: vec3<f32>, ng: vec3<f32>, l: vec3<f32>) -> f32 {
    if (S3D_SHADOW_TAPS == 0u || view.shadow_info.x == 0u || (material.flags.x & S3D_NO_SHADOW) != 0u) {
        return 1.0;
    }
    let ndl = clamp(dot(ng, l), 0.0, 1.0);
    let p = world + ng * (view.shadow_params.y * (1.0 - ndl));
    let lp = view.shadow_view_proj * vec4<f32>(p, 1.0);
    let ndc = lp.xyz / lp.w;
    let edge = max(abs(ndc.x), abs(ndc.y));
    if (edge >= 1.0) {
        return 1.0;
    }
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
    let lit = s3d_shadow_pcf(uv, clamp(ndc.z - view.shadow_params.x, 0.0, 1.0));
    let fade = clamp((edge - view.shadow_params.w) / max(1.0 - view.shadow_params.w, 1e-4), 0.0, 1.0);
    return mix(1.0, lit, view.shadow_params.z * (1.0 - fade));
}

// Tangent frame of a normal map at a fragment with the (front-side) normal `n`: the vertex
// tangent (MikkTSpace, `w` = bitangent sign) when the mesh has one (and it is not parallel to
// the normal), else a cotangent frame from
// screen-space derivatives (Schüler 2013) oriented like glTF's: tangent toward +u, bitangent
// toward −v (up in the image).
fn s3d_tangent_frame(
    n: vec3<f32>,
    t: vec4<f32>,
    dp1: vec3<f32>,
    dp2: vec3<f32>,
    duv1: vec2<f32>,
    duv2: vec2<f32>,
) -> mat3x3<f32> {
    let ortho = t.xyz - n * dot(n, t.xyz);
    if (dot(t.xyz, t.xyz) > 0.25 && dot(ortho, ortho) > 1e-8) {
        let tt = normalize(ortho);
        let bt = cross(n, tt) * select(1.0, -1.0, t.w < 0.0);
        return mat3x3<f32>(tt, bt, n);
    }
    let dp2perp = cross(dp2, n);
    let dp1perp = cross(n, dp1);
    // The frame scales with det(dp1, dp2, n), whose sign depends on the facing and on the
    // direction of screen y: normalize it away.
    let s = select(1.0, -1.0, dot(cross(dp1, dp2), n) < 0.0);
    let tt = (dp2perp * duv1.x + dp1perp * duv2.x) * s;
    let bt = (dp2perp * duv1.y + dp1perp * duv2.y) * -s;
    let m = max(dot(tt, tt), dot(bt, bt));
    if (m <= 1e-30) {
        return mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), n);
    }
    let inv = inverseSqrt(m);
    return mat3x3<f32>(tt * inv, bt * inv, n);
}

// Tangent-space normal of a normal-map sample (glTF: linear, +Y green up): XY scaled by the
// material's normal scale, Z reconstructed (two-channel maps work too).
fn s3d_normal_ts(texel: vec4<f32>) -> vec3<f32> {
    var c = texel.rgb;
    if ((material.flags.x & S3D_FIXUP_NORMAL) != 0u) {
        c = linear_to_srgb(c);
    }
    let xy = c.xy * 2.0 - 1.0;
    let z = sqrt(clamp(1.0 - dot(xy, xy), 0.0, 1.0));
    return vec3<f32>(xy * material.maps.x, z);
}

@fragment
fn fs_main(in: VsOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let flags = material.flags.x;
    // The second UV set (UV0 on meshes without one) and the vertex tangent (zero without one).
    let ext_tangent = s3d_frag_tangent(in);
    let uv1 = select(in.uv, s3d_frag_uv1(in), abs(ext_tangent.w) > 0.5);
    // Sample unconditionally (uniform control flow); `params.z` blends it in.
    let tex = sample_base(s3d_map_uv(S3D_MAP_BASE, in.uv, uv1), in.layer);
    let texel = mix(vec4<f32>(1.0), tex, material.params.z);
    let base = material.base_color * in.color * texel;
    // Maps are sampled behind their flags: material uniforms are uniform values, so the
    // branches keep implicit-derivative sampling in uniform control flow.
    var emissive = material.emissive.rgb;
    if ((flags & S3D_MAP_EMISSIVE) != 0u) {
        emissive = emissive * s3d_sample_emissive(s3d_map_uv(S3D_MAP_EMISSIVE, in.uv, uv1), in.layer).rgb;
    }

    var rgb: vec3<f32>;
    if (material.emissive.w > 0.5) {
        rgb = base.rgb + emissive;
    } else {
        var n = normalize(in.normal);
        if (S3D_NORMAL_MAPS && (flags & S3D_MAP_NORMAL) != 0u) {
            let uvn = s3d_map_uv(S3D_MAP_NORMAL, in.uv, uv1);
            let nt = s3d_sample_normal(uvn, in.layer);
            let frame = s3d_tangent_frame(
                n,
                ext_tangent,
                dpdx(in.world_pos),
                dpdy(in.world_pos),
                dpdx(uvn),
                dpdy(uvn),
            );
            n = normalize(frame * s3d_normal_ts(nt));
        }
        // Back faces (double-sided materials) flip the whole frame (glTF).
        if (!front) {
            n = -n;
        }
        let v = normalize(view.camera_pos.xyz - in.world_pos);
        var metallic = material.params.x;
        var roughness = material.params.y;
        var occlusion = 1.0;
        if ((flags & S3D_MAP_MR) != 0u) {
            var mr = s3d_sample_mr(s3d_map_uv(S3D_MAP_MR, in.uv, uv1), in.layer).rgb;
            if ((flags & S3D_FIXUP_MR) != 0u) {
                mr = linear_to_srgb(mr);
            }
            roughness = roughness * mr.g;
            metallic = metallic * mr.b;
            occlusion = mr.r;
        }
        if ((flags & (S3D_MAP_OCCLUSION | S3D_ORM_PACKED)) == S3D_MAP_OCCLUSION) {
            var o = s3d_sample_occlusion(s3d_map_uv(S3D_MAP_OCCLUSION, in.uv, uv1), in.layer).rgb;
            if ((flags & S3D_FIXUP_OCCLUSION) != 0u) {
                o = linear_to_srgb(o);
            }
            occlusion = o.r;
        }
        var acc = view.ambient.rgb * base.rgb;
        var ao = 1.0;
        if ((flags & S3D_MAP_OCCLUSION) != 0u) {
            // glTF: occlusion darkens indirect (ambient, image-based) light only.
            ao = 1.0 + material.maps.y * (occlusion - 1.0);
            acc = acc * ao;
        }
        if (S3D_SHADING_STANDARD) {
            let ab = s3d_env_brdf(clamp(dot(n, v), 1e-4, 1.0), clamp(roughness, 0.045, 1.0));
            let f0 = mix(vec3<f32>(0.04), base.rgb, metallic);
            s3d_ms_energy = vec3<f32>(1.0) + f0 * (1.0 / max(ab.x + ab.y, 1e-4) - 1.0);
        }
        // The casting light's shadow (directional light `shadow_info.y`).
        var shadow = 1.0;
        if (S3D_SHADOW_TAPS != 0u && view.shadow_info.x != 0u) {
            let ng = normalize(in.normal) * select(-1.0, 1.0, front);
            let caster = view.dir_lights[min(view.shadow_info.y, MAX_DIR_LIGHTS - 1u)];
            shadow = s3d_shadow_factor(in.world_pos, ng, -caster.direction.xyz);
        }
        for (var i = 0u; i < min(view.counts.x, MAX_DIR_LIGHTS); i++) {
            let light = view.dir_lights[i];
            var radiance = light.color.rgb;
            if (i == view.shadow_info.y && shadow < 1.0) {
                radiance = radiance * shadow;
            }
            acc += s3d_brdf(base.rgb, metallic, roughness, n, v, -light.direction.xyz) * radiance;
        }
        for (var i = 0u; i < min(view.counts.y, MAX_POINT_LIGHTS); i++) {
            let light = view.point_lights[i];
            let to_light = light.position_range.xyz - in.world_pos;
            let d = length(to_light);
            let range = light.position_range.w;
            let falloff = clamp(1.0 - pow(d / range, 4.0), 0.0, 1.0);
            let atten = falloff * falloff / (d * d + 1.0);
            acc += s3d_brdf(base.rgb, metallic, roughness, n, v, to_light / max(d, 1e-4)) * light.color.rgb * atten;
        }
        if (view.env_params.w > 0.5) {
            acc += s3d_ibl(base.rgb, metallic, roughness, n, v, ao);
        }
        rgb = acc + emissive;
    }
    var alpha = base.a;
    if (S3D_ALPHA_TO_COVERAGE) {
        // Alpha to coverage (MSAA): alpha sharpened around the cutoff to about one pixel.
        alpha = clamp((base.a - material.maps.z) / max(fwidth(base.a), 1e-4) + 0.5, 0.0, 1.0);
    } else if (S3D_ALPHA_MASK) {
        if (base.a < material.maps.z) {
            discard;
        }
        alpha = 1.0;
    }
    let color = fragment_hook(in, vec4<f32>(rgb, alpha));
    return s3d_output(color, S3D_OUTPUT, view.exposure.x, view.output.y);
}

// SSR G-buffer-lite (`PassKind::GBuffer`, entry `fs_gbuffer`; docs/graphics/post.md, "SSR"):
// the octahedral world normal (rg), roughness (b) and F0 (a; 0 = no reflection: unlit surfaces)
// of an opaque surface, drawn depth-tested over the scene depth after the scene pass. The same
// material evaluation as `fs_main` (maps, flags, normal mapping), none of the lighting.
fn s3d_oct_encode(n: vec3<f32>) -> vec2<f32> {
    let a = n / (abs(n.x) + abs(n.y) + abs(n.z));
    var e = a.xy;
    if (a.z < 0.0) {
        e = (vec2<f32>(1.0) - abs(a.yx)) * vec2<f32>(select(-1.0, 1.0, a.x >= 0.0), select(-1.0, 1.0, a.y >= 0.0));
    }
    return e * 0.5 + vec2<f32>(0.5);
}

@fragment
fn fs_gbuffer(in: VsOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let flags = material.flags.x;
    let ext_tangent = s3d_frag_tangent(in);
    let uv1 = select(in.uv, s3d_frag_uv1(in), abs(ext_tangent.w) > 0.5);
    let tex = sample_base(s3d_map_uv(S3D_MAP_BASE, in.uv, uv1), in.layer);
    let texel = mix(vec4<f32>(1.0), tex, material.params.z);
    let base = material.base_color * in.color * texel;
    if (S3D_ALPHA_MASK && base.a < material.maps.z) {
        discard;
    }
    if (material.emissive.w > 0.5) {
        return vec4<f32>(0.0);
    }
    var n = normalize(in.normal);
    if (S3D_NORMAL_MAPS && (flags & S3D_MAP_NORMAL) != 0u) {
        let uvn = s3d_map_uv(S3D_MAP_NORMAL, in.uv, uv1);
        let nt = s3d_sample_normal(uvn, in.layer);
        let frame = s3d_tangent_frame(
            n,
            ext_tangent,
            dpdx(in.world_pos),
            dpdy(in.world_pos),
            dpdx(uvn),
            dpdy(uvn),
        );
        n = normalize(frame * s3d_normal_ts(nt));
    }
    if (!front) {
        n = -n;
    }
    var metallic = material.params.x;
    var roughness = material.params.y;
    if ((flags & S3D_MAP_MR) != 0u) {
        var mr = s3d_sample_mr(s3d_map_uv(S3D_MAP_MR, in.uv, uv1), in.layer).rgb;
        if ((flags & S3D_FIXUP_MR) != 0u) {
            mr = linear_to_srgb(mr);
        }
        roughness = roughness * mr.g;
        metallic = metallic * mr.b;
    }
    let f0 = mix(vec3<f32>(0.04), base.rgb, metallic);
    let f0_max = clamp(max(f0.r, max(f0.g, f0.b)), 0.04, 1.0);
    return vec4<f32>(s3d_oct_encode(n), clamp(roughness, 0.0, 1.0), f0_max);
}
