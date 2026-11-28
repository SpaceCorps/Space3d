// Space3d city effects (docs/graphics/post.md, "City effects"): fog, SSAO and SSR on the HDR
// scene color of a post group, before bloom and exposure. The renderer prepends VIEW_WGSL,
// the scene group (`scene_wgsl_for`, group 1: the single-sample scene depth) and
// FULLSCREEN_WGSL. Group 0 is the settings camera's view bind group (the `View` uniform, the
// environment cubemap and the BRDF table), group 2 is this file's.
//
// Passes (entry points): `fs_ao` (GTAO-style horizon search into an R8 target at half or full
// resolution), `fs_blur_h` / `fs_blur_v` (depth-aware separable blur), `fs_apply` (scene color
// → HDR color: SSR, then SSAO, then fog). All render into a texture of the scene texture's
// size (or half of it, AO) with a scissor rect = the group's rect; positions are scene-texture
// uv (`suv`). Only explicit-LOD sampling and loads (no derivatives): WebGL2-safe in form.

const FX_FOG: u32 = 1u;
const FX_AO: u32 = 2u;
const FX_SSR: u32 = 4u;

// Matches `GpuFx` (post/screenfx.rs): one 256-byte slot per post group.
struct Fx {
    fog_color: vec4<f32>,       // rgb in-scatter color, density
    fog_shape: vec4<f32>,       // height falloff, base height, start distance, max opacity
    fog_sun: vec4<f32>,         // rgb sun color * intensity, exponent
    fog_sun_dir: vec4<f32>,     // xyz travel direction, w 0 none / 1 given / 2 the view's first directional light
    fog_sky: vec4<f32>,         // sky amount, sky falloff, -, -
    ao_a: vec4<f32>,            // radius, intensity, power, bias
    ao_b: vec4<f32>,            // max distance, slices, steps, blur radius
    ao_size: vec4<f32>,         // AO target w, h, 1/w, 1/h
    ssr_a: vec4<f32>,           // max distance, steps, thickness, intensity
    ssr_b: vec4<f32>,           // max roughness, edge fade, frame (jitter), denoise radius
    ssr_c: vec4<f32>,           // roughness blur, vertical stretch, HDR boost, -
    flags: vec4<u32>,           // x effect bits
    rect: vec4<f32>,            // the group's rect in scene pixels (x, y, w, h)
}

@group(0) @binding(1) var fx_env_specular: texture_cube<f32>;
@group(0) @binding(3) var fx_brdf_lut: texture_2d<f32>;
@group(0) @binding(4) var fx_env_sampler: sampler;

@group(2) @binding(0) var<uniform> fx: Fx;
@group(2) @binding(1) var fx_scene: texture_2d<f32>;
@group(2) @binding(2) var fx_ao: texture_2d<f32>;
@group(2) @binding(3) var fx_gbuf: texture_2d<f32>;
@group(2) @binding(4) var fx_samp: sampler;
@group(2) @binding(5) var fx_refl: texture_2d<f32>;   // the SSR contribution (signed radiance)

const FX_PI: f32 = 3.14159265358979;

fn fx_has(bit: u32) -> bool {
    return (fx.flags.x & bit) != 0u;
}

fn fx_scene_size() -> vec2<f32> {
    return view.target_size.xy;
}

// Scene-texture uv → the view's own 0..1 uv (the view's viewport rect inside the texture).
fn fx_view_uv(suv: vec2<f32>) -> vec2<f32> {
    return (suv * fx_scene_size() - view.viewport.xy) / view.viewport.zw;
}

// Scene depth at `suv`, snapped to the center of the pixel it falls in (half-resolution passes
// sample between pixels, where nearest filtering would pick either one).
fn fx_snap(suv: vec2<f32>) -> vec2<f32> {
    let size = fx_scene_size();
    return (floor(suv * size) + vec2<f32>(0.5)) / size;
}

// Scene uv of the scene pixel an AO texel stands for (AO texel (i, j) at half resolution is the
// scene pixel (2i, 2j)): exactly at a pixel center, never on a pixel edge.
fn fx_ao_suv(texel: vec2<f32>) -> vec2<f32> {
    let scale = floor(fx_scene_size() * fx.ao_size.zw + vec2<f32>(0.5));
    return (floor(texel) * scale + vec2<f32>(0.5)) / fx_scene_size();
}

fn fx_depth(suv: vec2<f32>) -> f32 {
    return textureSampleLevel(s3d_scene_depth_tex, s3d_scene_sampler, fx_snap(suv), 0.0).x;
}

// View-space position (camera looks down -z) of the scene at `suv` with device depth `d`.
fn fx_view_pos(suv: vec2<f32>, d: f32) -> vec3<f32> {
    // At the center of the pixel the depth belongs to, never between pixels.
    let vuv = fx_view_uv(fx_snap(suv));
    let p = view.inv_proj * vec4<f32>(vuv.x * 2.0 - 1.0, 1.0 - vuv.y * 2.0, d, 1.0);
    return p.xyz / p.w;
}

// Positive view-space depth of device depth `d`.
fn fx_lin(d: f32) -> f32 {
    let v = view.inv_proj * vec4<f32>(0.0, 0.0, d, 1.0);
    return -v.z / v.w;
}

fn fx_view_rot() -> mat3x3<f32> {
    return mat3x3<f32>(view.view[0].xyz, view.view[1].xyz, view.view[2].xyz);
}

// Scene-texture uv of a view-space position (and 1/w).
fn fx_project(p: vec3<f32>) -> vec3<f32> {
    let c = view.proj * vec4<f32>(p, 1.0);
    let ndc = c.xy / c.w;
    let vuv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    return vec3<f32>((vuv * view.viewport.zw + view.viewport.xy) / fx_scene_size(), 1.0 / c.w);
}

// Interleaved gradient noise (Jimenez 2014), 0..1.
fn fx_ign(p: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

// View-space normal from the depth neighbours (the closer neighbour on each axis).
fn fx_normal(suv: vec2<f32>, p: vec3<f32>) -> vec3<f32> {
    let t = 1.0 / fx_scene_size();
    let pl = fx_view_pos(suv - vec2<f32>(t.x, 0.0), fx_depth(suv - vec2<f32>(t.x, 0.0)));
    let pr = fx_view_pos(suv + vec2<f32>(t.x, 0.0), fx_depth(suv + vec2<f32>(t.x, 0.0)));
    let pu = fx_view_pos(suv - vec2<f32>(0.0, t.y), fx_depth(suv - vec2<f32>(0.0, t.y)));
    let pd = fx_view_pos(suv + vec2<f32>(0.0, t.y), fx_depth(suv + vec2<f32>(0.0, t.y)));
    let dx = select(pr - p, p - pl, abs(pl.z - p.z) < abs(pr.z - p.z));
    let dy = select(pu - p, p - pd, abs(pd.z - p.z) < abs(pu.z - p.z));
    return normalize(cross(dx, dy));
}

// ------------------------------------------------------------------------------------------
// SSAO: ground-truth ambient occlusion (Jimenez et al. 2016, after XeGTAO's formulation).
// ------------------------------------------------------------------------------------------

@fragment
fn fs_ao(in: FullscreenOut) -> @location(0) vec4<f32> {
    let suv = fx_ao_suv(in.pos.xy);
    let d = fx_depth(suv);
    if (d >= 1.0) {
        return vec4<f32>(1.0);
    }
    let p = fx_view_pos(suv, d);
    let n = fx_normal(suv, p);
    let v = normalize(-p);
    let radius = fx.ao_a.x;
    let size = fx_scene_size();
    // Screen radius in scene pixels at this depth (perspective).
    let px_radius = clamp(radius * view.proj[1][1] * view.viewport.w * 0.5 / max(-p.z, 1e-3), 1.0, view.viewport.w * 0.3);
    let slices = max(u32(fx.ao_b.y), 1u);
    let steps = max(u32(fx.ao_b.z), 2u);
    let noise_a = fx_ign(in.pos.xy);
    let noise_b = fx_ign(in.pos.xy + vec2<f32>(5.588238, 5.588238));
    let falloff_range = 0.615 * radius;
    let falloff_mul = -1.0 / falloff_range;
    let falloff_add = (radius - falloff_range) / falloff_range + 1.0;
    var visibility = 0.0;
    var open = 0.0;
    for (var s = 0u; s < slices; s++) {
        let phi = (f32(s) + noise_a) * FX_PI / f32(slices);
        let dir2 = vec2<f32>(cos(phi), sin(phi));
        let dir3 = vec3<f32>(dir2, 0.0);
        let ortho = dir3 - dot(dir3, v) * v;
        let axis = normalize(cross(ortho, v));
        let proj_n = n - axis * dot(n, axis);
        let proj_len = length(proj_n);
        if (proj_len < 1e-4) {
            visibility += 1.0;
            open += 1.0;
            continue;
        }
        let sign_n = sign(dot(ortho, proj_n));
        let cos_n = clamp(dot(proj_n, v) / proj_len, -1.0, 1.0);
        let ang_n = sign_n * acos(cos_n);
        let low0 = cos(ang_n + FX_PI * 0.5);
        let low1 = cos(ang_n - FX_PI * 0.5);
        var hc0 = low0;
        var hc1 = low1;
        let screen_dir = vec2<f32>(dir2.x, -dir2.y);
        for (var k = 0u; k < steps; k++) {
            let t = (f32(k) + noise_b) / f32(steps);
            // From 4 scene pixels out: closer samples drown in depth quantization.
            let off = (4.0 + t * t * max(px_radius - 4.0, 0.0)) * screen_dir / size;
            let uv0 = suv + off;
            let uv1 = suv - off;
            let d0 = fx_depth(uv0);
            let d1 = fx_depth(uv1);
            let h0v = fx_view_pos(uv0, d0) - p;
            let h1v = fx_view_pos(uv1, d1) - p;
            let l0 = max(length(h0v), 1e-4);
            let l1 = max(length(h1v), 1e-4);
            var s0 = dot(h0v / l0, v);
            var s1 = dot(h1v / l1, v);
            // Samples on the surface's own plane (within the bias of its tangent) do not occlude:
            // depth quantization would otherwise darken flat ground and walls seen at a grazing angle.
            // The tolerance grows with distance (depth quantization) and with `bias` (a
            // fraction of the radius).
            let tol = 0.004 * (-p.z) + 0.02 + fx.ao_a.w * radius;
            let t0 = clamp((dot(h0v, n) - tol) / tol, 0.0, 1.0);
            let t1 = clamp((dot(h1v, n) - tol) / tol, 0.0, 1.0);
            let w0 = clamp(l0 * falloff_mul + falloff_add, 0.0, 1.0) * t0;
            let w1 = clamp(l1 * falloff_mul + falloff_add, 0.0, 1.0) * t1;
            // Sky (depth 1) and out-of-view samples do not occlude.
            let in0 = all(uv0 > vec2<f32>(0.0)) && all(uv0 < vec2<f32>(1.0)) && d0 < 1.0;
            let in1 = all(uv1 > vec2<f32>(0.0)) && all(uv1 < vec2<f32>(1.0)) && d1 < 1.0;
            s0 = mix(low0, s0, select(0.0, w0, in0));
            s1 = mix(low1, s1, select(0.0, w1, in1));
            hc0 = max(hc0, s0);
            hc1 = max(hc1, s1);
        }
        var h0 = -acos(clamp(hc1, -1.0, 1.0));
        var h1 = acos(clamp(hc0, -1.0, 1.0));
        h0 = ang_n + clamp(h0 - ang_n, -FX_PI * 0.5, FX_PI * 0.5);
        h1 = ang_n + clamp(h1 - ang_n, -FX_PI * 0.5, FX_PI * 0.5);
        let sin_n = sin(ang_n);
        let arc0 = (cos_n + 2.0 * h0 * sin_n - cos(2.0 * h0 - ang_n)) * 0.25;
        let arc1 = (cos_n + 2.0 * h1 * sin_n - cos(2.0 * h1 - ang_n)) * 0.25;
        visibility += proj_len * (arc0 + arc1);
        // What the same slice integrates to with nothing in the way: dividing by it makes an
        // open surface exactly 1 whatever the slice count and the noise rotation.
        open += proj_len * (cos_n + ang_n * sin_n);
    }
    var ao = clamp(visibility / max(open, 1e-4), 0.0, 1.0);
    let max_d = fx.ao_b.x;
    ao = mix(ao, 1.0, smoothstep(0.7 * max_d, max_d, -p.z));
    return vec4<f32>(ao, ao, ao, 1.0);
}

// Depth-aware separable blur of the AO texture (`fx_ao`); `axis` is the pixel step.
fn fx_blur(in: FullscreenOut, axis: vec2<f32>) -> vec4<f32> {
    let dims = vec2<i32>(textureDimensions(fx_ao));
    let px = vec2<i32>(in.pos.xy);
    let suv = fx_ao_suv(in.pos.xy);
    let dc = fx_depth(suv);
    let pc = fx_view_pos(suv, dc);
    let zc = -pc.z;
    let nc = fx_normal(suv, pc);
    let radius = i32(fx.ao_b.w);
    let center = textureLoad(fx_ao, px, 0).r;
    if (radius <= 0) {
        return vec4<f32>(center);
    }
    let sigma = max(f32(radius) * 0.6, 0.5);
    var sum = center;
    var wsum = 1.0;
    for (var i = 1; i <= radius; i++) {
        for (var side = 0; side < 2; side++) {
            let o = select(-i, i, side == 0);
            let q = clamp(px + vec2<i32>(axis) * o, vec2<i32>(0), dims - vec2<i32>(1));
            let quv = fx_ao_suv(vec2<f32>(q));
            // Distance of the tap's surface point from this pixel's tangent plane: a slanted
            // surface (ground at a grazing angle) blurs along itself, an edge does not.
            let pq = fx_view_pos(quv, fx_depth(quv));
            let wd = exp(-abs(dot(pq - pc, nc)) / (0.02 * zc + 0.05));
            let w = exp(-f32(i * i) / (2.0 * sigma * sigma)) * wd;
            sum += textureLoad(fx_ao, q, 0).r * w;
            wsum += w;
        }
    }
    let r = sum / wsum;
    return vec4<f32>(r, r, r, 1.0);
}

@fragment
fn fs_blur_h(in: FullscreenOut) -> @location(0) vec4<f32> {
    return fx_blur(in, vec2<f32>(1.0, 0.0));
}

@fragment
fn fs_blur_v(in: FullscreenOut) -> @location(0) vec4<f32> {
    return fx_blur(in, vec2<f32>(0.0, 1.0));
}

// ------------------------------------------------------------------------------------------
// Apply: SSR → SSAO → fog.
// ------------------------------------------------------------------------------------------

fn fx_oct_decode(e: vec2<f32>) -> vec3<f32> {
    let f = e * 2.0 - 1.0;
    var n = vec3<f32>(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    let t = clamp(-n.z, 0.0, 1.0);
    n = vec3<f32>(n.x + select(t, -t, n.x >= 0.0), n.y + select(t, -t, n.y >= 0.0), n.z);
    return normalize(n);
}

// The reflection: the view-space ray of `dir` from `p` marched in screen space against the
// depth; returns (hit uv xy, confidence 0..1, 0).
fn fx_trace(p: vec3<f32>, dir: vec3<f32>, jitter: f32) -> vec4<f32> {
    let max_d = fx.ssr_a.x;
    let steps = i32(fx.ssr_a.y);
    let thick = fx.ssr_a.z;
    var end = p + dir * max_d;
    let near = view.depth.x;
    if (end.z > -near) {
        // Clip at the near plane.
        let t = (-near - p.z) / dir.z;
        end = p + dir * max(t * 0.99, 0.0);
    }
    let a = fx_project(p);
    let b = fx_project(end);
    let k0 = a.z;
    let k1 = b.z;
    var lo = 0.0;
    var hit = false;
    var hit_t = 0.0;
    var prev_t = 0.0;
    for (var i = 1; i <= steps; i++) {
        let t = (f32(i) - 1.0 + jitter) / f32(steps) + 1.0 / f32(steps);
        let suv = mix(a.xy, b.xy, t);
        let vuv = fx_view_uv(suv);
        if (any(vuv < vec2<f32>(0.0)) || any(vuv > vec2<f32>(1.0))) {
            break;
        }
        let ds = fx_depth(suv);
        let ray_z = 1.0 / mix(k0, k1, t);
        if (ds < 1.0) {
            let scene_z = fx_lin(ds);
            let delta = ray_z - scene_z;
            if (delta > 0.004 * ray_z && delta < thick + 0.03 * ray_z) {
                // Surfaces facing away from the ray (the ground it leaves, the back of a wall)
                // are not reflected: their G-buffer normal points along the ray.
                let g = textureLoad(fx_gbuf, vec2<i32>(suv * fx_scene_size()), 0);
                var facing = true;
                if (g.a > 0.0) {
                    facing = dot(fx_view_rot() * fx_oct_decode(g.rg), dir) < 0.15;
                }
                if (facing) {
                    hit = true;
                    hit_t = t;
                    lo = prev_t;
                    break;
                }
            }
        }
        prev_t = t;
    }
    if (!hit) {
        return vec4<f32>(0.0);
    }
    // Binary refinement between the last miss and the hit.
    var t_lo = lo;
    var t_hi = hit_t;
    for (var r = 0; r < 5; r++) {
        let m = 0.5 * (t_lo + t_hi);
        let suv = mix(a.xy, b.xy, m);
        let ds = fx_depth(suv);
        let ray_z = 1.0 / mix(k0, k1, m);
        if (ds < 1.0 && ray_z > fx_lin(ds)) {
            t_hi = m;
        } else {
            t_lo = m;
        }
    }
    let suv = mix(a.xy, b.xy, t_hi);
    let vuv = fx_view_uv(suv);
    let edge = min(min(vuv.x, 1.0 - vuv.x), min(vuv.y, 1.0 - vuv.y));
    let edge_fade = smoothstep(0.0, max(fx.ssr_b.y, 1e-3), edge);
    let dist_fade = 1.0 - smoothstep(0.6, 1.0, t_hi);
    return vec4<f32>(suv, edge_fade * dist_fade, t_hi);
}

fn fx_fog_opacity(cam_y: f32, point_y: f32, distance: f32) -> f32 {
    let falloff = max(fx.fog_shape.x, 0.0);
    let d = max(distance - max(fx.fog_shape.z, 0.0), 0.0);
    let base = max(fx.fog_color.w, 0.0) * exp(-falloff * (cam_y - fx.fog_shape.y));
    let x = falloff * (point_y - cam_y) * d / max(distance, 1e-6);
    var avg = 1.0;
    if (abs(x) > 1e-4) {
        avg = (1.0 - exp(-x)) / x;
    }
    let tau = base * d * avg;
    return min(1.0 - exp(-tau), clamp(fx.fog_shape.w, 0.0, 1.0));
}

fn fx_fog_color(dir_w: vec3<f32>) -> vec3<f32> {
    var c = fx.fog_color.rgb;
    var sun_dir = vec3<f32>(0.0);
    var has_sun = false;
    if (fx.fog_sun_dir.w > 1.5) {
        if (view.counts.x > 0u) {
            sun_dir = view.dir_lights[0].direction.xyz;
            has_sun = true;
        }
    } else if (fx.fog_sun_dir.w > 0.5) {
        sun_dir = fx.fog_sun_dir.xyz;
        has_sun = true;
    }
    if (has_sun) {
        let lobe = pow(max(dot(dir_w, -sun_dir), 0.0), max(fx.fog_sun.w, 1e-3));
        c += fx.fog_sun.rgb * lobe;
    }
    return c;
}

// ------------------------------------------------------------------------------------------
// SSR passes: `fs_ssr` writes the reflection's contribution `weight * (hit - env_specular)`
// (signed, HDR) into its own target; a depth-aware blur (`fs_refl_blur_*`) removes the march
// noise; `fs_apply` adds it. Rough surfaces blur the hit over an ellipse stretched vertically
// (glare streaks of lights on wet ground).
// ------------------------------------------------------------------------------------------

fn fx_ssr_contribution(frag: vec2<f32>) -> vec3<f32> {
    let px = vec2<i32>(frag);
    let size = fx_scene_size();
    let suv = frag / size;
    let d = fx_depth(suv);
    if (d >= 1.0) {
        return vec3<f32>(0.0);
    }
    let g = textureLoad(fx_gbuf, px, 0);
    let f0 = g.a;
    let rough = g.b;
    let max_r = fx.ssr_b.x;
    let rough_fade = 1.0 - smoothstep(0.6 * max_r, max_r, rough);
    if (f0 <= 0.0 || rough_fade <= 0.0) {
        return vec3<f32>(0.0);
    }
    let rot = fx_view_rot();
    let p = fx_view_pos(suv, d);
    let dist = length(p);
    let dir_w = (transpose(rot) * p) / max(dist, 1e-6);
    let n_w = fx_oct_decode(g.rg);
    let n_v = rot * n_w;
    let v_v = normalize(-p);
    let r_v = reflect(-v_v, n_v);
    let noise = fx_ign(frag + vec2<f32>(fx.ssr_b.z));
    let h = fx_trace(p, r_v, noise);
    if (h.z <= 0.0) {
        return vec3<f32>(0.0);
    }
    // Hit radiance: a few taps over an ellipse that grows with roughness and ray length.
    let scale = size.y / 1080.0;
    let radius = fx.ssr_c.x * rough * (4.0 + 28.0 * h.w) * scale;
    var hit = vec3<f32>(0.0);
    if (radius < 0.75) {
        hit = textureSampleLevel(fx_scene, fx_samp, h.xy, 0.0).rgb;
    } else {
        var sum = vec3<f32>(0.0);
        // Stratified along the (stretched) vertical axis, spread sideways: a light smears into
        // a streak, the neighbours average out.
        for (var i = 0u; i < 12u; i++) {
            let s = (f32(i) + noise) / 12.0 * 2.0 - 1.0;
            let side = fx_ign(frag + vec2<f32>(f32(i) * 7.31, 3.7)) * 2.0 - 1.0;
            let o = vec2<f32>(side * radius * 0.6, s * radius * max(fx.ssr_c.y, 1.0)) / size;
            sum += min(textureSampleLevel(fx_scene, fx_samp, clamp(h.xy + o, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb, vec3<f32>(64.0));
        }
        hit = sum / 12.0;
    }
    // Lights and emissives (above display white) reflect more strongly.
    let peak = max(hit.r, max(hit.g, hit.b));
    hit = hit * (1.0 + fx.ssr_c.z * smoothstep(1.0, 4.0, peak));
    let ndv = clamp(dot(n_v, v_v), 1e-4, 1.0);
    let ab = textureSampleLevel(fx_brdf_lut, fx_env_sampler, vec2<f32>(ndv, clamp(rough, 0.0, 1.0)), 0.0).rg;
    let spec_w = f0 * ab.x + ab.y;
    var env = vec3<f32>(0.0);
    if (view.env_params.w > 0.5) {
        let r_w = reflect(dir_w, n_w);
        env = textureSampleLevel(fx_env_specular, fx_env_sampler, view.env_rotation * r_w, rough * view.env_params.z).rgb * view.env_params.y;
    }
    return h.z * rough_fade * fx.ssr_a.w * spec_w * (hit - env);
}

@fragment
fn fs_ssr(in: FullscreenOut) -> @location(0) vec4<f32> {
    return vec4<f32>(fx_ssr_contribution(in.pos.xy), 1.0);
}

// Depth-aware blur of the reflection texture (bound as `fx_ao`).
fn fx_refl_blur(in: FullscreenOut, axis: vec2<f32>) -> vec4<f32> {
    let dims = vec2<i32>(textureDimensions(fx_ao));
    let px = vec2<i32>(in.pos.xy);
    let size = fx_scene_size();
    let suv = in.pos.xy / size;
    let pc = fx_view_pos(suv, fx_depth(suv));
    let radius = i32(fx.ssr_b.w);
    let center = textureLoad(fx_ao, px, 0).rgb;
    if (radius <= 0 || fx_depth(suv) >= 1.0) {
        return vec4<f32>(center, 1.0);
    }
    var sum = center;
    var wsum = 1.0;
    for (var i = 1; i <= radius; i++) {
        for (var side = 0; side < 2; side++) {
            let o = select(-i, i, side == 0);
            let q = clamp(px + vec2<i32>(axis) * o, vec2<i32>(0), dims - vec2<i32>(1));
            let quv = (vec2<f32>(q) + 0.5) / size;
            let dq = fx_depth(quv);
            let pq = fx_view_pos(quv, dq);
            let w = exp(-f32(i * i) / (f32(radius * radius) + 1.0)) * exp(-abs(pq.z - pc.z) / (0.02 * -pc.z + 0.05)) * select(1.0, 0.0, dq >= 1.0);
            sum += textureLoad(fx_ao, q, 0).rgb * w;
            wsum += w;
        }
    }
    return vec4<f32>(sum / wsum, 1.0);
}

@fragment
fn fs_refl_blur_h(in: FullscreenOut) -> @location(0) vec4<f32> {
    return fx_refl_blur(in, vec2<f32>(1.0, 0.0));
}

@fragment
fn fs_refl_blur_v(in: FullscreenOut) -> @location(0) vec4<f32> {
    return fx_refl_blur(in, vec2<f32>(0.0, 1.0));
}


@fragment
fn fs_apply(in: FullscreenOut) -> @location(0) vec4<f32> {
    let px = vec2<i32>(in.pos.xy);
    let suv = in.pos.xy / fx_scene_size();
    var col = textureLoad(fx_scene, px, 0);
    let d = fx_depth(suv);
    let rot = fx_view_rot();
    if (d >= 1.0) {
        // Sky: fog by elevation.
        if (fx_has(FX_FOG)) {
            let vuv = fx_view_uv(suv);
            let ndc = vec2<f32>(vuv.x * 2.0 - 1.0, 1.0 - vuv.y * 2.0);
            let near = view.inv_proj_rot * vec4<f32>(ndc, 0.0, 1.0);
            let far = view.inv_proj_rot * vec4<f32>(ndc, 1.0, 1.0);
            let dir = normalize(far.xyz / far.w - near.xyz / near.w);
            let amount = clamp(fx.fog_sky.x, 0.0, 1.0) * exp(-fx.fog_sky.y * max(dir.y, 0.0)) * clamp(fx.fog_shape.w, 0.0, 1.0);
            col = vec4<f32>(mix(col.rgb, fx_fog_color(dir), amount), col.a);
        }
        return col;
    }
    let p = fx_view_pos(suv, d);
    let dist = length(p);
    let world_off = transpose(rot) * p;
    let dir_w = world_off / max(dist, 1e-6);

    if (fx_has(FX_SSR)) {
        let refl = textureSampleLevel(fx_refl, fx_samp, suv, 0.0).rgb;
        col = vec4<f32>(max(col.rgb + refl, vec3<f32>(0.0)), col.a);
    }

    if (fx_has(FX_AO)) {
        let ao = textureSampleLevel(fx_ao, fx_samp, suv, 0.0).r;
        let occ = mix(1.0, pow(max(ao, 1e-4), fx.ao_a.z), clamp(fx.ao_a.y, 0.0, 2.0));
        col = vec4<f32>(col.rgb * max(occ, 0.0), col.a);
    }

    if (fx_has(FX_FOG)) {
        let cam_y = view.camera_pos.y;
        let opacity = fx_fog_opacity(cam_y, cam_y + world_off.y, dist);
        col = vec4<f32>(mix(col.rgb, fx_fog_color(dir_w), opacity), col.a);
    }
    return col;
}
