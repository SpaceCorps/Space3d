// 3D shield (SpaceCorps 2027 proposal), port of the three.js prototype's shader.
// Prepended by the node: VIEW_WGSL (group 0), scene_wgsl (group 1), OUTPUT_WGSL, then
// `const SH_OUTPUT: u32` (the pass's output encoding).
//
// Local space: the shield's own space, centred on the ship's bounding box, in model units.
// Bubble: a unit sphere (poles on ±Y) turned to poles on ±Z and scaled by the radii.
// Hull: the baked envelope (.sbe), already recentred on the CPU.

struct Shield {
    model: mat4x4<f32>,
    color: vec4<f32>,     // rgb shield colour, w intensity
    a: vec4<f32>,         // rest, fresnel power, back-face factor, time (s)
    b: vec4<f32>,         // shield % (0..1), hex on, hex scale (1/unit), wave speed (unit/s)
    c: vec4<f32>,         // wave width (unit), wave life (s), flash, mode (0 bubble, 1 hull)
    radii: vec4<f32>,     // ellipsoid radii, w mean radius
    dis: vec4<f32>,       // break: on, front (unit), edge width, noise scale
    dis_pt: vec4<f32>,    // break origin (local), w net level (0 solid .. 1 sparse, from absorbance)
    d: vec4<f32>,         // hole radius, reveal (0..1, 2 = full), z min, z max
    e: vec4<f32>,         // burst, impact count, 0, 0
    imp: array<vec4<f32>, 16>,   // xyz local point, w start time (< 0: free)
    // x strength (< 0: flash only), y share that pierces, z kind (0 shot, 1 siphon, 2 blast,
    // 3 siphon gain: y packet count, w time between packets), w width × (siphon: drain time)
    impx: array<vec4<f32>, 16>,
}
@group(2) @binding(0) var<uniform> sh: Shield;

struct VIn {
    @location(0) pos: vec3<f32>,
    @location(1) nrm: vec3<f32>,
}
struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) local: vec3<f32>,
    @location(1) nl: vec3<f32>,
    @location(2) wpos: vec3<f32>,
    @location(3) nw: vec3<f32>,
}

fn sh_out(local: vec3<f32>, nl: vec3<f32>) -> VOut {
    var o: VOut;
    let w = sh.model * vec4<f32>(local, 1.0);
    o.clip = view.view_proj * w;
    o.local = local;
    o.nl = nl;
    o.wpos = w.xyz;
    o.nw = (sh.model * vec4<f32>(nl, 0.0)).xyz;   // uniform scale: normalised in the fragment
    return o;
}

@vertex fn vs_bubble(v: VIn) -> VOut {
    // poles on the forward axis: no pinch under the top-down camera
    let u = vec3<f32>(v.pos.x, -v.pos.z, v.pos.y);   // a rotation, not a mirror: winding is kept
    let un = vec3<f32>(v.nrm.x, -v.nrm.z, v.nrm.y);
    let r = sh.radii.xyz;
    return sh_out(u * r, un / r);
}

@vertex fn vs_hull(v: VIn) -> VOut {
    return sh_out(v.pos, v.nrm);
}

fn sh_hash(n: f32) -> f32 { return fract(sin(n) * 43758.5453); }
fn sh_h3(p: vec3<f32>) -> f32 { return fract(sin(dot(p, vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453); }
fn sh_vnoise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    var f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(sh_h3(i), sh_h3(i + vec3<f32>(1.0, 0.0, 0.0)), f.x), mix(sh_h3(i + vec3<f32>(0.0, 1.0, 0.0)), sh_h3(i + vec3<f32>(1.0, 1.0, 0.0)), f.x), f.y),
        mix(mix(sh_h3(i + vec3<f32>(0.0, 0.0, 1.0)), sh_h3(i + vec3<f32>(1.0, 0.0, 1.0)), f.x), mix(sh_h3(i + vec3<f32>(0.0, 1.0, 1.0)), sh_h3(i + vec3<f32>(1.0, 1.0, 1.0)), f.x), f.y),
        f.z);
}
fn sh_hexd(p0: vec2<f32>) -> f32 {
    var p = p0;
    p.x = p.x * 1.1547005;
    let col = floor(p.x);
    p.y = p.y + (col - 2.0 * floor(col / 2.0)) * 0.5;
    p = abs(fract(p) - 0.5);
    return abs(max(p.x * 1.5 + p.y, p.y * 2.0) - 1.0);
}
fn sh_hex_line(p: vec2<f32>) -> f32 { return 1.0 - smoothstep(0.02, 0.09, sh_hexd(p)); }

// distance along the surface: great-circle on the ellipsoid (bubble), straight line (hull)
fn sh_surf_dist(a: vec3<f32>, b: vec3<f32>) -> f32 {
    if (sh.c.w < 0.5) {
        let r = sh.radii.xyz;
        let la = length(a / r);
        let lb = length(b / r);
        if (la < 1e-6 || lb < 1e-6) { return distance(a, b); }
        let da = (a / r) / la;
        let db = (b / r) / lb;
        return acos(clamp(dot(da, db), -1.0, 1.0)) * sh.radii.w;
    }
    return distance(a, b);
}

fn sh_fs(i: VOut, front: bool) -> vec4<f32> {
    let time = sh.a.w;
    let wave_speed = sh.b.w;
    let wave_width = sh.c.x;
    let wave_life = sh.c.y;
    let zk = (i.local.z - sh.d.z) / max(sh.d.w - sh.d.z, 1e-4);
    if (zk > sh.d.y) { discard; }
    // break: the surface dissolves from the last hit, with a ragged glowing edge
    var dis_edge = 0.0;
    if (sh.dis.x > 0.5) {
        let dd = sh_surf_dist(i.local, sh.dis_pt.xyz) + (sh_vnoise(i.local * sh.dis.w) - 0.5) * sh.dis.z * 2.0;
        if (dd < sh.dis.y) { discard; }
        let xe = max(dd - sh.dis.y, 0.0) / max(wave_width, 1e-4);
        dis_edge = exp(-xe * xe * 1.2);
    }
    // NaN guards: normalize(0), 0/0 and pow(x < 0) give NaN on many GPUs (the engine's bloom
    // drops NaN pixels, but the guards keep the shield itself correct)
    let vv = view.camera_pos.xyz - i.wpos;
    let vl = length(vv);
    let V = select(vec3<f32>(0.0, 1.0, 0.0), vv / vl, vl > 1e-6);
    let nlw = length(i.nw);
    let N = select(V, i.nw / nlw, nlw > 1e-6);
    let side = select(sh.a.z, 1.0, front);
    let fr = pow(max(1.0 - clamp(abs(dot(N, V)), 0.0, 1.0), 0.0), sh.a.y);
    let nll = length(i.nl);
    let nL = select(vec3<f32>(0.0, 1.0, 0.0), abs(i.nl) / nll, nll > 1e-6);
    var w = nL * nL;
    w = w * w;
    w = w / max(w.x + w.y + w.z, 1e-6);
    let q = i.local * sh.b.z;
    let hx = sh_hex_line(q.yz) * w.x + sh_hex_line(q.xz) * w.y + sh_hex_line(q.xy) * w.z;
    // low absorbance: the shield becomes a net with broken cells (you can see it lets shots through)
    let leak = sh.dis_pt.w;
    let net_edge = hx * step(leak * 0.55, sh_vnoise(q * 0.45 + 17.0));
    let net = mix(1.0, 0.06 + 0.94 * net_edge, leak);
    let pct = clamp(sh.b.x, 0.0, 1.0);
    var rest = sh.a.x * mix(0.3, 1.0, pct) * fr * net * mix(1.0, 0.6, leak);
    if (pct < 0.25) { rest = rest * (0.45 + 0.55 * step(0.35, sh_hash(floor(time * 22.0)))); }
    var waves = 0.0;
    var flash = 0.0;
    var reveal = 0.0;
    var hole_rim = 0.0;
    var teal = 0.0;
    var drain = 0.0;
    var fill = 0.0;
    let gain_life = wave_life * 1.4;
    let n_imp = u32(sh.e.y);
    for (var k = 0u; k < 16u; k = k + 1u) {
        if (k >= n_imp) { break; }
        let t0 = sh.imp[k].w;
        if (t0 < 0.0) { continue; }
        let ix = sh.impx[k];
        let siphon = ix.z > 0.5 && ix.z < 1.5;
        let gain = ix.z > 2.5;
        let age = time - t0;
        var life = wave_life;
        if (siphon) { life = ix.w + 0.35; }
        if (gain) { life = gain_life + ix.w * max(ix.y - 1.0, 0.0); }
        if (age < 0.0 || age > life) { continue; }
        if (gain) {
            // your siphon's packets come in: from where each enters, a teal wave runs over the
            // whole shield, which glows a little brighter behind it
            let d = sh_surf_dist(i.local, sh.imp[k].xyz);
            let n = u32(clamp(ix.y, 1.0, 10.0));
            // fast enough to reach the far side before it fades (the bubble is much larger)
            let far = select(2.2, 3.1416, sh.c.w < 0.5) * sh.radii.w;
            let gspeed = max(wave_speed, far / (0.75 * gain_life));
            // thin enough that close packets still read as separate rings
            let gw = max(clamp(0.5 * gspeed * ix.w, 0.35 * wave_width, wave_width), 1e-4);
            let amp = max(ix.x, 0.0) * 1.2 * sqrt(max(wave_width, 1e-4) / gw) / sqrt(f32(n));
            let fg = d / (wave_width * 1.4);
            let entry = exp(-fg * fg);
            var ring = 0.0;
            var behind = 0.0;
            var flashes = 0.0;
            for (var j = 0u; j < 10u; j = j + 1u) {
                if (j >= n) { break; }
                let aj = age - f32(j) * ix.w;
                if (aj < 0.0 || aj > gain_life) { continue; }
                let kj = 1.0 - aj / gain_life;
                let r = gspeed * aj;
                let xg = (d - r) / gw;
                ring = ring + exp(-xg * xg) * kj * amp;
                behind = max(behind, (1.0 - smoothstep(r - gw, r + gw, d)) * kj * kj);
                // each packet that lands lights its entry point
                flashes = flashes + entry * exp(-aj * 12.0);
            }
            teal = teal + ring * (0.5 + 1.5 * sh.b.y * hx) + min(flashes, 2.0) * 0.9;
            fill = max(fill, behind * 0.45);
            continue;
        }
        let sf = abs(ix.x);
        let s = max(ix.x, 0.0);   // s < 0: flash only, no wave
        let ww = wave_width * ix.w;
        let d = sh_surf_dist(i.local, sh.imp[k].xyz);
        var kk = 1.0 - age / wave_life;
        kk = kk * kk;
        let r = wave_speed * age;
        let x = (d - r) / ww;
        // the share that pierces opens a hole at the entry point, which closes again
        if (ix.y > 0.02 && s > 0.0 && age < 0.35 && ix.z < 1.5) {
            let rh = sh.d.x * sqrt(max(ix.y, 0.0)) * (1.0 - age / 0.35);
            if (d < rh) { discard; }
            let xh = (d - rh) / (wave_width * 0.4);
            hole_rim = hole_rim + exp(-xh * xh) * (1.0 - age / 0.35);
        }
        let wv = exp(-x * x) * kk * s;
        let rv = exp(-max(d - r, 0.0) / (ww * 1.5)) * step(d, r + ww) * kk * s;
        let fd = d / (wave_width * 1.4);
        let fl = exp(-fd * fd) * exp(-age * 9.0) * sf;
        if (siphon) {
            // siphon: the shield is sucked in, a teal ring closes on the hit point, darkening behind it
            let T = max(ix.w, 0.05);
            let r0 = min(wave_speed * T, select(0.8 * sh.radii.w, 1.5708 * sh.radii.w, sh.c.w < 0.5));
            let rr = r0 * max(1.0 - age / T, 0.0);
            let xs = (d - rr) / wave_width;
            let amp = select(exp(-(age - T) * 8.0), 0.35 + 0.65 * age / T, age < T);
            let ring = exp(-xs * xs) * amp * s;
            let f_end = exp(-fd * fd) * exp(-max(age - T, 0.0) * 9.0) * step(T, age) * sf;
            teal = teal + ring * (0.5 + 1.5 * sh.b.y * hx) * 1.2 + fl * 0.6 + f_end * 2.2;
            if (age < T + 0.3) {
                let fade = select(1.0 - (age - T) / 0.3, age / T, age < T);
                drain = max(drain, step(0.0, ix.x) * step(rr, d) * (1.0 - smoothstep(r0 * 0.8, r0 + 1e-4, d)) * 0.55 * fade);
            }
            continue;
        }
        waves = waves + wv;
        reveal = reveal + rv;
        flash = flash + fl;
    }
    rest = rest * (1.0 - drain);
    teal = teal + rest * fill * 2.0;
    let hex_on = sh.b.y * hx;
    var I = rest + (waves * (0.8 + 2.2 * hex_on) + reveal * hex_on * 0.9) * net + flash * 2.0 * sh.c.z + hole_rim * 2.2;
    let ze = (sh.d.y - zk) / 0.035;   // ze*ze, not pow(ze, 2): pow of a negative base is NaN
    I = I + exp(-ze * ze) * step(sh.d.y, 1.0) * 3.0;
    I = I + dis_edge * (1.6 + 2.2 * hex_on);
    I = I * (1.0 + sh.e.x * 3.0);
    var col = sh.color.rgb * I * sh.color.w
        + vec3<f32>(flash * flash * 1.6 * sh.c.z + sh.e.x * fr * 1.5 + dis_edge * dis_edge * 1.2)
        + vec3<f32>(0.12, 1.5, 1.25) * teal * sh.color.w;
    col = col * side;
    // last guard: an invalid or infinite value becomes black for this pixel only
    if (!(col.r >= 0.0 && col.r < 1e4 && col.g >= 0.0 && col.g < 1e4 && col.b >= 0.0 && col.b < 1e4)) {
        col = vec3<f32>(0.0);
    }
    // additive light: alpha 0 with premultiplied blending adds the colour and keeps the scene
    return s3d_output(vec4<f32>(col, 0.0), SH_OUTPUT, view.exposure.x, view.output.y);
}

@fragment fn fs_shield(i: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    return sh_fs(i, front);
}

// ---------- instanced quads: fragments and sparks, bolts ----------
struct QIn {
    @builtin(vertex_index) vi: u32,
    @location(4) a: vec4<f32>,
    @location(5) b: vec4<f32>,
    @location(6) c: vec4<f32>,
}
struct QOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) col: vec3<f32>,
    @location(2) k: f32,
}

fn sh_corner(i: u32) -> vec2<f32> {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0));
    return corners[i % 6u];
}

fn sh_additive(c: vec3<f32>) -> vec4<f32> {
    var col = c;
    if (!(col.r >= 0.0 && col.r < 1e4 && col.g >= 0.0 && col.g < 1e4 && col.b >= 0.0 && col.b < 1e4)) {
        col = vec3<f32>(0.0);
    }
    return s3d_output(vec4<f32>(col, 0.0), SH_OUTPUT, view.exposure.x, view.output.y);
}

// a: start xyz + birth, b: velocity xyz + life, c: colour rgb + size
@vertex fn vs_particle(i: QIn) -> QOut {
    var o: QOut;
    let age = sh.a.w - i.a.w;
    let life = max(i.b.w, 1e-3);
    let alive = age >= 0.0 && age <= life && i.b.w > 0.0;
    let t = clamp(age / life, 0.0, 1.0);
    let drag = 2.2;
    let p = i.a.xyz + i.b.xyz * (1.0 - exp(-drag * max(age, 0.0))) / drag;
    let c = sh_corner(i.vi);
    let right = vec3<f32>(view.view[0][0], view.view[1][0], view.view[2][0]);
    let up = vec3<f32>(view.view[0][1], view.view[1][1], view.view[2][1]);
    let wp = p + (right * c.x + up * c.y) * 0.5 * i.c.w * (1.0 - 0.5 * t);   // c.w is the diameter
    o.clip = select(vec4<f32>(0.0, 0.0, -2.0, 1.0), view.view_proj * vec4<f32>(wp, 1.0), alive);
    o.uv = c;
    o.col = i.c.rgb;
    o.k = (1.0 - t) * (1.0 - t) * smoothstep(0.0, 0.05, max(age, 0.0));
    return o;
}
@fragment fn fs_particle(i: QOut) -> @location(0) vec4<f32> {
    let r = dot(i.uv, i.uv);
    if (r > 1.0) { discard; }
    let a = exp(-r * 3.0) * i.k;
    return sh_additive(i.col * sh.color.w * a * 1.4 + vec3<f32>(a * a * 0.8));
}

// a: tail xyz, b: direction xyz + length, c: colour rgb + half width
@vertex fn vs_streak(i: QIn) -> QOut {
    var o: QOut;
    let c = sh_corner(i.vi);
    let along = c.y * 0.5 + 0.5;
    let mid = i.a.xyz + i.b.xyz * along * i.b.w;
    let to_cam = view.camera_pos.xyz - mid;
    var side = cross(i.b.xyz, to_cam);
    let sl = length(side);
    side = select(vec3<f32>(1.0, 0.0, 0.0), side / sl, sl > 1e-6);
    o.clip = view.view_proj * vec4<f32>(mid + side * c.x * i.c.w, 1.0);
    o.uv = vec2<f32>(c.x, along);
    o.col = i.c.rgb;
    o.k = 1.0;
    return o;
}
@fragment fn fs_streak(i: QOut) -> @location(0) vec4<f32> {
    let p = exp(-i.uv.x * i.uv.x * 5.0);
    let ends = smoothstep(0.0, 0.12, i.uv.y) * (1.0 - smoothstep(0.82, 1.0, i.uv.y));
    let a = (p - exp(-5.0)) * ends;
    return sh_additive(i.col * max(a, 0.0) + vec3<f32>(max(a, 0.0) * a * a * 0.5));
}

// ---------- Aegis ground ring and gauge ----------
// Unit annulus in the XZ plane. `sh.color`: gauge colour; `sh.e.z`: share of the reserve left.
@vertex fn vs_ring(v: VIn) -> VOut {
    return sh_out(v.pos, vec3<f32>(0.0, 1.0, 0.0));
}
fn sh_band(r: f32, a: f32, b: f32, w: f32) -> f32 {
    return smoothstep(a - w, a + w, r) * (1.0 - smoothstep(b - w, b + w, r));
}
@fragment fn fs_ring(i: VOut) -> @location(0) vec4<f32> {
    let r = length(i.local.xz);
    let w = max(fwidth(r), 1e-5);
    // below the bloom threshold: a sharp line, not a halo over the whole view
    var col = vec3<f32>(0.95, 0.4, 0.7) * 0.75 * sh_band(r, 0.992, 1.0, w);
    let turn = fract(atan2(-i.local.x, -i.local.z) / 6.2831853 + 1.0);
    if (turn <= sh.e.z) {
        col = col + sh.color.rgb * 0.95 * sh_band(r, 1.008, 1.022, w);
    }
    return sh_additive(col);
}
