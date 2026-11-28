//! Procedural textures and the sunset sky: facades with lit windows (base color, normal, packed
//! ORM and emissive maps), shopfronts, wet asphalt with puddles, paving, roofs, the props
//! palette atlas and the HDR cubemap. Everything is generated on the compute pool.

use space3d::prelude::*;
use space3d::render::{CubemapSource, SamplerSettings, TextureSource, f32_to_f16};

pub fn hash(x: u32, y: u32, seed: u32) -> u32 {
    let mut h = x.wrapping_mul(0x8da6_b343) ^ y.wrapping_mul(0xd816_3841) ^ seed.wrapping_mul(0x9e37_79b9);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^ (h >> 16)
}

pub fn unit(h: u32) -> f32 {
    (h >> 8) as f32 / (1u32 << 24) as f32
}

/// Uniform random number in 0..1 from three integers.
pub fn rnd(x: i32, y: i32, seed: u32) -> f32 {
    unit(hash(x as u32, y as u32, seed))
}

pub fn smooth(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

pub fn mix(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Value noise on a lattice that repeats every `px` × `py` cells (tileable textures).
fn vnoise_t(x: f32, y: f32, px: i32, py: i32, seed: u32) -> f32 {
    let (xi, yi) = (x.floor(), y.floor());
    let (fx, fy) = (x - xi, y - yi);
    let (wx, wy) = (fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy));
    let c = |dx: i32, dy: i32| rnd((xi as i32 + dx).rem_euclid(px), (yi as i32 + dy).rem_euclid(py), seed);
    mix(mix(c(0, 0), c(1, 0), wx), mix(c(0, 1), c(1, 1), wx), wy)
}

/// Tileable fractal noise in 0..1: `period` cells at the first octave.
pub fn fbm_t(x: f32, y: f32, period: i32, octaves: u32, seed: u32) -> f32 {
    let (mut sum, mut amp, mut norm) = (0.0, 0.5, 0.0);
    for o in 0..octaves {
        let f = 1 << o;
        sum += amp * vnoise_t(x * f as f32, y * f as f32, period * f, period * f, seed + o);
        norm += amp;
        amp *= 0.5;
    }
    sum / norm
}

/// Non-tiling 3D value noise (the sky).
fn noise3(p: Vec3, seed: u32) -> f32 {
    let i = p.floor();
    let f = p - i;
    let w = f * f * (Vec3::splat(3.0) - 2.0 * f);
    let c = |dx: i32, dy: i32, dz: i32| {
        let (x, y, z) = (i.x as i32 + dx, i.y as i32 + dy, i.z as i32 + dz);
        unit(hash(x as u32 ^ (z as u32).rotate_left(16), y as u32, seed))
    };
    let x0 = mix(mix(c(0, 0, 0), c(1, 0, 0), w.x), mix(c(0, 1, 0), c(1, 1, 0), w.x), w.y);
    let x1 = mix(mix(c(0, 0, 1), c(1, 0, 1), w.x), mix(c(0, 1, 1), c(1, 1, 1), w.x), w.y);
    mix(x0, x1, w.z)
}

fn fbm3(p: Vec3, octaves: u32, seed: u32) -> f32 {
    let (mut sum, mut amp, mut norm, mut q) = (0.0, 0.5, 0.0, p);
    for o in 0..octaves {
        sum += amp * noise3(q, seed + o);
        norm += amp;
        amp *= 0.5;
        q *= 2.03;
    }
    sum / norm
}

pub fn to_srgb8(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let s = if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0 + 0.5) as u8
}

fn to_unorm8(c: f32) -> u8 {
    (c.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// One material's four maps.
pub struct TextureSet {
    pub base: ImageData,
    pub normal: ImageData,
    pub orm: ImageData,
    pub emissive: ImageData,
}

/// Per-texel material channels, turned into the four maps by [`Layers::finish`].
pub struct Layers {
    pub w: u32,
    pub h: u32,
    pub base: Vec<Vec3>,
    pub height: Vec<f32>,
    pub rough: Vec<f32>,
    pub metal: Vec<f32>,
    pub ao: Vec<f32>,
    pub emis: Vec<Vec3>,
}

pub struct Texel {
    pub base: Vec3,
    pub height: f32,
    pub rough: f32,
    pub metal: f32,
    pub ao: f32,
    pub emis: Vec3,
}

impl Texel {
    fn new(base: Vec3, rough: f32) -> Self {
        Self {
            base,
            height: 0.0,
            rough,
            metal: 0.0,
            ao: 1.0,
            emis: Vec3::ZERO,
        }
    }
}

impl Layers {
    pub fn build(w: u32, h: u32, mut f: impl FnMut(u32, u32) -> Texel) -> Self {
        let n = (w * h) as usize;
        let mut l = Self {
            w,
            h,
            base: Vec::with_capacity(n),
            height: Vec::with_capacity(n),
            rough: Vec::with_capacity(n),
            metal: Vec::with_capacity(n),
            ao: Vec::with_capacity(n),
            emis: Vec::with_capacity(n),
        };
        for y in 0..h {
            for x in 0..w {
                let t = f(x, y);
                l.base.push(t.base);
                l.height.push(t.height);
                l.rough.push(t.rough);
                l.metal.push(t.metal);
                l.ao.push(t.ao);
                l.emis.push(t.emis);
            }
        }
        l
    }

    /// The maps; `strength` scales the height gradient of the normal map (texels of height per
    /// texel). Wraps around at the edges (the textures tile).
    pub fn finish(self, strength: f32) -> TextureSet {
        let (w, h) = (self.w as i32, self.h as i32);
        let at = |x: i32, y: i32| self.height[(y.rem_euclid(h) * w + x.rem_euclid(w)) as usize];
        let mut base = ImageData::new(self.w, self.h);
        let mut normal = ImageData::new(self.w, self.h);
        let mut orm = ImageData::new(self.w, self.h);
        let mut emissive = ImageData::new(self.w, self.h);
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize;
                let sx = (at(x + 1, y) - at(x - 1, y)) * 0.5 * strength;
                let sy = (at(x, y - 1) - at(x, y + 1)) * 0.5 * strength;
                let n = Vec3::new(-sx, -sy, 1.0).normalize();
                let b = self.base[i];
                let e = self.emis[i];
                let (xu, yu) = (x as u32, y as u32);
                base.set(xu, yu, [to_srgb8(b.x), to_srgb8(b.y), to_srgb8(b.z), 255]);
                normal.set(
                    xu,
                    yu,
                    [
                        to_unorm8(n.x * 0.5 + 0.5),
                        to_unorm8(n.y * 0.5 + 0.5),
                        to_unorm8(n.z * 0.5 + 0.5),
                        255,
                    ],
                );
                orm.set(
                    xu,
                    yu,
                    [
                        to_unorm8(self.ao[i]),
                        to_unorm8(self.rough[i]),
                        to_unorm8(self.metal[i]),
                        255,
                    ],
                );
                emissive.set(xu, yu, [to_srgb8(e.x), to_srgb8(e.y), to_srgb8(e.z), 255]);
            }
        }
        TextureSet {
            base,
            normal,
            orm,
            emissive,
        }
    }
}

// ----------------------------------------------------------------------------------------------
// Facades
// ----------------------------------------------------------------------------------------------

/// Texels per bay and floor in the facade textures.
pub const CELL: u32 = 64;
/// Bays across and floors down one facade texture.
pub const BAYS: u32 = 8;
pub const FLOORS: u32 = 16;
/// Bay width and floor height in meters (a texel cell is one bay × one floor).
pub const BAY_M: f32 = 3.6;
pub const FLOOR_M: f32 = 3.6;
/// Height of the ground-floor shopfront band in meters.
pub const SHOP_M: f32 = 5.4;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Facade {
    Concrete,
    Brick,
    Glass,
    Stone,
}

pub const FACADES: [Facade; 4] = [Facade::Concrete, Facade::Brick, Facade::Glass, Facade::Stone];

fn inside(x: u32, y: u32, x0: u32, y0: u32, x1: u32, y1: u32) -> bool {
    x >= x0 && x < x1 && y >= y0 && y < y1
}

/// The lit-window look of one window cell: emission (linear, 0..1) at the texel.
struct Lit {
    on: bool,
    color: Vec3,
    /// Fraction of the window (from the top) hidden behind a blind or curtain.
    blind: f32,
}

fn lit_window(cx: u32, cy: u32, seed: u32, p: f32) -> Lit {
    let r = |k: u32| unit(hash(cx, cy, seed.wrapping_add(k * 977)));
    let on = r(1) < p;
    let b = 0.35 + 0.65 * r(2) * r(2);
    let t = r(3);
    let tint = if t < 0.5 {
        Vec3::new(1.0, 0.70, 0.36)
    } else if t < 0.78 {
        Vec3::new(1.0, 0.84, 0.62)
    } else if t < 0.9 {
        Vec3::new(0.62, 0.78, 1.0)
    } else if t < 0.96 {
        Vec3::new(1.0, 0.38, 0.62)
    } else {
        Vec3::new(0.5, 1.0, 0.6)
    };
    Lit {
        on,
        color: tint * b,
        blind: if r(4) > 0.55 { r(5) * 0.75 } else { 0.0 },
    }
}

/// Facade texels shared by every kind: grime streaks and joint lines.
fn grime(x: u32, y: u32, seed: u32) -> f32 {
    let n = fbm_t(x as f32 / 6.0, y as f32 / 90.0, 6, 4, seed);
    let m = fbm_t(x as f32 / 40.0, y as f32 / 40.0, 12, 3, seed + 31);
    (n * 0.7 + m * 0.5).clamp(0.0, 1.0)
}

pub fn facade_textures(kind: Facade, seed: u32) -> TextureSet {
    let (w, h) = (BAYS * CELL, FLOORS * CELL);
    let layers = Layers::build(w, h, |x, y| {
        let (cx, cy) = (x / CELL, y / CELL);
        let (lx, ly) = (x % CELL, y % CELL);
        match kind {
            Facade::Concrete => concrete_texel(x, y, cx, cy, lx, ly, seed),
            Facade::Brick => brick_texel(x, y, cx, cy, lx, ly, seed + 100),
            Facade::Glass => glass_texel(x, y, cx, cy, lx, ly, seed + 200),
            Facade::Stone => stone_texel(x, y, cx, cy, lx, ly, seed + 300),
        }
    });
    layers.finish(match kind {
        Facade::Brick => 1.4,
        _ => 1.0,
    })
}

fn window_glass(lit: &Lit, ly: u32, y0: u32, y1: u32, depth: f32) -> Texel {
    // Dark reflective glass; a lit one adds the room's light, hidden above the blind line.
    let mut t = Texel::new(Vec3::new(0.05, 0.065, 0.085), 0.05);
    t.metal = 0.55;
    t.height = -depth;
    if lit.on {
        let v = (ly - y0) as f32 / (y1 - y0).max(1) as f32;
        if v >= lit.blind {
            // The ceiling lamp lights the top of the room more than the floor.
            let shade = 1.15 - 0.55 * ((v - lit.blind) / (1.0 - lit.blind + 1e-3));
            t.emis = lit.color * shade.max(0.25);
            t.base = Vec3::new(0.08, 0.07, 0.06);
            t.metal = 0.2;
        } else {
            t.base = Vec3::new(0.10, 0.09, 0.08);
            t.emis = lit.color * 0.05;
            t.rough = 0.5;
            t.metal = 0.0;
        }
    }
    t
}

fn concrete_texel(x: u32, y: u32, cx: u32, cy: u32, lx: u32, ly: u32, seed: u32) -> Texel {
    let n = grime(x, y, seed);
    let speck = rnd(x as i32, y as i32, seed + 5);
    let tone = 0.25 + 0.10 * n + 0.03 * speck;
    let mut t = Texel::new(Vec3::new(tone * 1.02, tone, tone * 0.95), 0.62 + 0.2 * n);
    t.height = 0.25 * speck + 0.4 * n;
    // panel joints
    if lx == 0 || ly == 0 {
        t.base *= 0.62;
        t.height -= 1.6;
        t.ao = 0.7;
    }
    let (wx0, wy0, wx1, wy1) = (7, 10, 57, 50);
    if inside(lx, ly, wx0, wy0, wx1, wy1) {
        let (fx0, fy0, fx1, fy1) = (wx0 + 3, wy0 + 3, wx1 - 3, wy1 - 3);
        if inside(lx, ly, fx0, fy0, fx1, fy1) {
            // mullion in the middle
            let mull = lx.abs_diff(32) < 1;
            if mull {
                let mut f = Texel::new(Vec3::splat(0.03), 0.45);
                f.metal = 0.6;
                f.height = 0.5;
                return f;
            }
            let lit = lit_window(cx, cy, seed, 0.42);
            let mut g = window_glass(&lit, ly, fy0, fy1, 3.0);
            // the lit pane's curtain / furniture silhouette
            if g.emis != Vec3::ZERO {
                g.emis *= 0.78 + 0.22 * fbm_t(x as f32 / 5.0, y as f32 / 7.0, 16, 2, seed + 9);
            }
            return g;
        }
        // frame
        let mut f = Texel::new(Vec3::splat(0.045), 0.4);
        f.metal = 0.7;
        f.height = -0.8;
        return f;
    }
    // sill under the window, lintel above
    if inside(lx, ly, wx0 - 2, wy1, wx1 + 2, wy1 + 4) {
        t.height += 2.0;
        t.base *= 1.15;
    }
    // rain stain streaks under sills
    if inside(lx, ly, wx0, wy1 + 4, wx1, CELL) {
        let s = fbm_t(x as f32 / 2.5, y as f32 / 60.0, 10, 3, seed + 77);
        t.base *= 1.0 - 0.45 * smooth(0.5, 0.75, s);
    }
    t
}

fn brick_texel(x: u32, y: u32, cx: u32, cy: u32, lx: u32, ly: u32, seed: u32) -> Texel {
    let (bw, bh) = (10u32, 5u32);
    let row = y / bh;
    let off = (row % 2) * (bw / 2);
    let col = (x + off) / bw;
    let (bx, by) = ((x + off) % bw, y % bh);
    let mortar = bx == 0 || by == 0;
    let r = unit(hash(col, row, seed));
    let n = grime(x, y, seed);
    let mut base = Vec3::new(0.30 + 0.12 * r, 0.115 + 0.05 * r, 0.085 + 0.03 * r) * (0.8 + 0.4 * n);
    let mut t = Texel::new(base, 0.7);
    t.height = if mortar { -2.4 } else { 0.4 * r };
    if mortar {
        base = Vec3::splat(0.27) * (0.8 + 0.3 * n);
        t.base = base;
        t.rough = 0.9;
        t.ao = 0.75;
    }
    let (wx0, wy0, wx1, wy1) = (14, 12, 50, 50);
    if inside(lx, ly, wx0, wy0, wx1, wy1) {
        let (fx0, fy0, fx1, fy1) = (wx0 + 3, wy0 + 3, wx1 - 3, wy1 - 3);
        if inside(lx, ly, fx0, fy0, fx1, fy1) {
            // two sashes with a transom
            if lx.abs_diff(32) < 1 || ly.abs_diff(fy0 + (fy1 - fy0) / 3) < 1 {
                let mut f = Texel::new(Vec3::splat(0.55), 0.55);
                f.height = 0.4;
                return f;
            }
            let lit = lit_window(cx, cy, seed, 0.5);
            return window_glass(&lit, ly, fy0, fy1, 4.0);
        }
        let mut f = Texel::new(Vec3::new(0.62, 0.6, 0.55), 0.6);
        f.height = 0.6;
        return f;
    }
    // stone lintel and sill
    if inside(lx, ly, wx0 - 3, wy0 - 5, wx1 + 3, wy0) || inside(lx, ly, wx0 - 3, wy1, wx1 + 3, wy1 + 4) {
        let mut s = Texel::new(Vec3::new(0.42, 0.39, 0.34) * (0.85 + 0.3 * n), 0.65);
        s.height = 3.0;
        return s;
    }
    t
}

fn glass_texel(x: u32, y: u32, cx: u32, cy: u32, lx: u32, ly: u32, seed: u32) -> Texel {
    // One bay = two panes over a spandrel.
    let n = fbm_t(x as f32 / 30.0, y as f32 / 30.0, 12, 3, seed);
    if lx < 2 || lx >= CELL - 1 || lx.abs_diff(32) < 2 {
        let mut m = Texel::new(Vec3::splat(0.05), 0.35);
        m.metal = 0.8;
        m.height = 2.0;
        return m;
    }
    if ly >= 46 || ly < 2 {
        // spandrel: opaque dark panel
        let mut s = Texel::new(Vec3::new(0.035, 0.045, 0.05) * (0.8 + 0.4 * n), 0.25);
        s.metal = 0.6;
        s.height = -0.4;
        if ly < 2 {
            s.height = 1.5;
            s.base = Vec3::splat(0.05);
        }
        return s;
    }
    let pane = if lx < 32 { 0 } else { 1 };
    // tinted glass, a different tint per floor
    let tint = unit(hash(cy, 7, seed));
    let base = Vec3::new(0.03, 0.06, 0.075).lerp(Vec3::new(0.05, 0.075, 0.07), tint) * (0.85 + 0.3 * n);
    let mut t = Texel::new(base, 0.03);
    t.metal = 0.85;
    t.height = -0.3;
    // whole floors light up together, with gaps
    let floor_on = unit(hash(cx * 2 + pane, cy, seed + 3)) < 0.34 && unit(hash(0, cy, seed + 11)) < 0.8;
    if floor_on {
        let c = unit(hash(cx * 2 + pane, cy, seed + 4));
        let tintc = if c < 0.6 {
            Vec3::new(1.0, 0.88, 0.7)
        } else if c < 0.85 {
            Vec3::new(0.85, 0.92, 1.0)
        } else {
            Vec3::new(1.0, 0.72, 0.45)
        };
        let v = ly as f32 / 46.0;
        t.emis = tintc * (0.55 + 0.4 * (1.0 - v)) * (0.8 + 0.2 * n);
        t.metal = 0.3;
    }
    t
}

fn stone_texel(x: u32, y: u32, cx: u32, cy: u32, lx: u32, ly: u32, seed: u32) -> Texel {
    let n = grime(x, y, seed);
    let speck = rnd(x as i32, y as i32, seed + 5);
    let tone = 0.55 + 0.25 * n;
    let mut t = Texel::new(Vec3::new(0.50, 0.40, 0.29) * tone * 0.9, 0.7);
    t.height = 0.4 * speck + 0.6 * n;
    // ashlar courses
    if ly % 16 == 0 {
        t.height -= 1.6;
        t.base *= 0.78;
    }
    if (lx + (ly / 16) * 21) % 42 == 0 {
        t.height -= 1.4;
        t.base *= 0.8;
    }
    // pilasters at the cell edges
    if lx < 6 || lx >= CELL - 6 {
        t.height += 2.0;
        t.base *= 1.1;
        t.ao = 0.95;
    }
    // cornice every floor top
    if ly < 4 {
        t.height += 2.5;
        t.base *= 1.12;
    }
    let (wx0, wy0, wx1, wy1) = (16, 10, 48, 54);
    if inside(lx, ly, wx0 - 3, wy0 - 3, wx1 + 3, wy1 + 3) && !inside(lx, ly, wx0, wy0, wx1, wy1) {
        t.height += 1.5;
        t.base *= 1.15;
    }
    if inside(lx, ly, wx0, wy0, wx1, wy1) {
        if lx.abs_diff(32) < 1 || ly.abs_diff(28) < 1 {
            let mut f = Texel::new(Vec3::splat(0.04), 0.5);
            f.metal = 0.5;
            f.height = 0.8;
            return f;
        }
        let lit = lit_window(cx, cy, seed, 0.55);
        return window_glass(&lit, ly, wy0, wy1, 5.0);
    }
    t
}

/// The ground-floor shopfronts: 8 bays, each a different shop (lit windows, doors, signs,
/// shutters).
pub fn shop_textures(seed: u32) -> TextureSet {
    let (w, h) = (BAYS * CELL, 96);
    let layers = Layers::build(w, h, |x, y| {
        let (cx, lx) = (x / CELL, x % CELL);
        let kind = unit(hash(cx, 0, seed));
        let accent = unit(hash(cx, 1, seed));
        let n = fbm_t(x as f32 / 20.0, y as f32 / 20.0, 25, 3, seed);
        let mut t = Texel::new(Vec3::new(0.12, 0.10, 0.09) * (0.7 + 0.6 * n), 0.55);
        t.height = 0.3 * n;
        // sign band
        if y < 18 {
            let sign = if accent < 0.25 {
                Vec3::new(1.0, 0.12, 0.2)
            } else if accent < 0.5 {
                Vec3::new(0.1, 0.8, 1.0)
            } else if accent < 0.75 {
                Vec3::new(1.0, 0.55, 0.1)
            } else {
                Vec3::new(1.0, 0.9, 0.7)
            };
            t.base = Vec3::splat(0.03);
            t.height = 1.0;
            if inside(lx, y, 6, 4, CELL - 6, 14) {
                // letter-like bars
                let bars = unit(hash(x / 4, 3, seed + cx)) > 0.35;
                if bars && kind > 0.2 {
                    t.emis = sign * 0.9;
                    t.base = sign * 0.3;
                }
            }
            return t;
        }
        // awning stripe
        if (18..26).contains(&y) {
            let stripe = (x / 8) % 2 == 0;
            let c = if accent < 0.5 {
                Vec3::new(0.35, 0.04, 0.05)
            } else {
                Vec3::new(0.04, 0.12, 0.28)
            };
            t.base = if stripe { c } else { Vec3::splat(0.5) };
            t.rough = 0.75;
            t.height = if stripe { 0.8 } else { 0.0 };
            return t;
        }
        let shutter = kind < 0.08;
        // piers left and right
        if lx < 5 || lx >= CELL - 5 {
            t.base = Vec3::new(0.2, 0.18, 0.16) * (0.8 + 0.4 * n);
            t.height += 2.0;
            return t;
        }
        // bottom stall riser
        if y >= 86 {
            t.base = Vec3::new(0.08, 0.07, 0.07);
            t.height = 1.0;
            return t;
        }
        if shutter {
            // roller shutter: horizontal ribs, a little light leaking through
            let rib = y % 4 == 0;
            t.base = Vec3::splat(0.2) * if rib { 0.7 } else { 1.0 };
            t.metal = 0.8;
            t.rough = 0.45;
            t.height = if rib { -1.0 } else { 0.0 };
            return t;
        }
        // door in some shops (the left third of the bay)
        let door = kind > 0.55 && lx < 24;
        // lit shop window
        let warm = if accent < 0.3 {
            Vec3::new(1.0, 0.72, 0.4)
        } else if accent < 0.6 {
            Vec3::new(1.0, 0.9, 0.78)
        } else if accent < 0.8 {
            Vec3::new(0.7, 0.85, 1.0)
        } else {
            Vec3::new(1.0, 0.5, 0.7)
        };
        let v = (y - 26) as f32 / 60.0;
        let shelf = unit(hash(x / 3, y / 6, seed + cx));
        let glow = if door {
            0.55
        } else {
            (0.7 + 0.5 * shelf * (0.5 + 0.5 * (1.0 - v))).min(1.3)
        };
        t.base = Vec3::new(0.04, 0.05, 0.06);
        t.rough = 0.04;
        t.metal = 0.5;
        t.height = -2.0;
        t.emis = warm * glow * 0.8;
        // mullions
        if lx.abs_diff(34) < 1 && !door || (y == 26 || y == 85) {
            t.emis = Vec3::ZERO;
            t.base = Vec3::splat(0.04);
            t.height = 0.5;
        }
        t
    });
    layers.finish(1.0)
}

// ----------------------------------------------------------------------------------------------
// Ground
// ----------------------------------------------------------------------------------------------

/// Wet asphalt, one tile of `ASPHALT_TILE_M` meters: aggregate grain, hairline cracks, patches
/// and puddles (low roughness, flat normals).
pub const ASPHALT_TILE_M: f32 = 12.0;

pub fn asphalt_textures(seed: u32) -> TextureSet {
    let s = 1024u32;
    let layers = Layers::build(s, s, |x, y| {
        let (u, v) = (x as f32, y as f32);
        let grain = rnd(x as i32, y as i32, seed);
        let coarse = fbm_t(u / 5.0, v / 5.0, 205, 2, seed + 1);
        let large = fbm_t(u / 256.0 * 4.0, v / 256.0 * 4.0, 4, 5, seed + 2);
        let patch = fbm_t(u / 128.0, v / 128.0, 8, 4, seed + 3);
        let tone = 0.034 + 0.015 * coarse + 0.01 * grain + 0.014 * patch;
        let mut t = Texel::new(Vec3::new(tone, tone * 0.99, tone * 1.03), 0.2 + 0.22 * coarse + 0.1 * grain);
        t.height = 1.1 * grain + 1.4 * coarse;
        // cracks: thin dark ridges of a low frequency noise
        let c = fbm_t(u / 70.0, v / 70.0, 15, 4, seed + 4);
        let crack = 1.0 - smooth(0.0, 0.018, (c - 0.5).abs());
        if crack > 0.0 {
            t.base *= 1.0 - 0.6 * crack;
            t.height -= 3.0 * crack;
            t.rough = mix(t.rough, 0.7, crack);
        }
        // patch repairs: a smoother, darker rectangle-ish tone
        // puddles
        let p = fbm_t(u / 256.0 * 3.0, v / 256.0 * 3.0, 3, 5, seed + 9);
        let mask = smooth(0.585, 0.605, p * 0.8 + large * 0.2 + 0.03 * coarse);
        if mask > 0.0 {
            t.rough = mix(t.rough, 0.025, mask);
            t.base *= mix(1.0, 0.55, mask);
            t.height *= 1.0 - 0.92 * mask;
            t.metal = 0.0;
        }
        // overall: still-damp asphalt is a little glossier than dry
        t.rough = t.rough.clamp(0.02, 1.0);
        t
    });
    layers.finish(0.9)
}

/// Pavement slabs of 1.2 m, 4 × 4 per tile of 4.8 m.
pub const PAVING_TILE_M: f32 = 4.8;

pub fn paving_textures(seed: u32) -> TextureSet {
    let s = 512u32;
    let slab = s / 4;
    let layers = Layers::build(s, s, |x, y| {
        let (sx, sy) = (x / slab, y / slab);
        let (lx, ly) = (x % slab, y % slab);
        let r = unit(hash(sx, sy, seed));
        let n = fbm_t(x as f32 / 40.0, y as f32 / 40.0, 13, 4, seed + 1);
        let grain = rnd(x as i32, y as i32, seed + 2);
        let tone = (0.2 + 0.09 * r + 0.07 * n) * (0.93 + 0.07 * grain);
        let mut t = Texel::new(Vec3::new(tone, tone * 0.98, tone * 0.94), 0.5 + 0.2 * n);
        t.height = 0.5 * grain + (r - 0.5) * 1.0;
        if lx < 2 || ly < 2 {
            t.base *= 0.5;
            t.height -= 3.0;
            t.rough = 0.85;
            t.ao = 0.6;
        }
        // damp slabs
        let wet = smooth(0.5, 0.62, fbm_t(x as f32 / 70.0, y as f32 / 70.0, 7, 3, seed + 3));
        t.rough = mix(t.rough, 0.07, wet);
        t.base *= mix(1.0, 0.7, wet);
        t
    });
    layers.finish(1.0)
}

/// Rooftop tar and gravel (8 m tile).
pub const ROOF_TILE_M: f32 = 8.0;

pub fn roof_textures(seed: u32) -> TextureSet {
    let s = 256u32;
    let layers = Layers::build(s, s, |x, y| {
        let g = rnd(x as i32, y as i32, seed);
        let n = fbm_t(x as f32 / 32.0, y as f32 / 32.0, 8, 4, seed + 1);
        let tone = 0.07 + 0.05 * n + 0.04 * g;
        let mut t = Texel::new(Vec3::new(tone, tone, tone * 1.05), 0.55 + 0.25 * g);
        t.height = 2.0 * g;
        let wet = smooth(0.55, 0.62, n);
        t.rough = mix(t.rough, 0.08, wet);
        t
    });
    layers.finish(1.0)
}

// ----------------------------------------------------------------------------------------------
// The props palette: an atlas of flat material cells (nearest sampling, no mips)
// ----------------------------------------------------------------------------------------------

pub const PALETTE_CELLS: u32 = 16;

pub mod cell {
    pub const CAR_PAINT: u32 = 0;
    pub const GLASS: u32 = 1;
    pub const RUBBER: u32 = 2;
    pub const STEEL: u32 = 3;
    pub const PAINT_WHITE: u32 = 4;
    pub const PAINT_YELLOW: u32 = 5;
    pub const LAMP_WARM: u32 = 6;
    pub const RED: u32 = 7;
    pub const GREEN: u32 = 8;
    pub const AMBER: u32 = 9;
    pub const NEON_CYAN: u32 = 10;
    pub const NEON_MAGENTA: u32 = 11;
    pub const NEON_ORANGE: u32 = 12;
    pub const CONCRETE: u32 = 13;
    pub const HEADLIGHT: u32 = 14;
    pub const BUS_WINDOW: u32 = 15;
}

pub struct Palette {
    pub base: TextureSource,
    pub orm: TextureSource,
    pub emissive: TextureSource,
}

pub fn palette() -> Palette {
    // (base linear, rough, metal, emissive linear)
    let z = Vec3::ZERO;
    let cells: [(Vec3, f32, f32, Vec3); 16] = [
        (Vec3::ONE, 0.22, 0.35, z),
        (Vec3::new(0.012, 0.02, 0.028), 0.04, 0.2, z),
        (Vec3::splat(0.025), 0.75, 0.0, z),
        (Vec3::new(0.28, 0.29, 0.31), 0.42, 1.0, z),
        (Vec3::new(0.62, 0.62, 0.6), 0.45, 0.0, z),
        (Vec3::new(0.62, 0.42, 0.04), 0.45, 0.0, z),
        (Vec3::new(0.9, 0.8, 0.6), 0.3, 0.0, Vec3::new(1.0, 0.72, 0.38)),
        (Vec3::new(0.25, 0.01, 0.01), 0.3, 0.0, Vec3::new(1.0, 0.04, 0.02) * 0.6),
        (Vec3::new(0.02, 0.2, 0.08), 0.3, 0.0, Vec3::new(0.05, 1.0, 0.3) * 0.4),
        (Vec3::new(0.3, 0.15, 0.01), 0.3, 0.0, Vec3::new(1.0, 0.5, 0.04) * 0.5),
        (Vec3::new(0.0, 0.2, 0.25), 0.3, 0.0, Vec3::new(0.05, 0.85, 1.0) * 0.75),
        (Vec3::new(0.25, 0.01, 0.1), 0.3, 0.0, Vec3::new(1.0, 0.08, 0.55) * 0.75),
        (Vec3::new(0.3, 0.1, 0.01), 0.3, 0.0, Vec3::new(1.0, 0.45, 0.05) * 0.7),
        (Vec3::new(0.2, 0.2, 0.21), 0.85, 0.0, z),
        (Vec3::new(0.9, 0.9, 0.85), 0.2, 0.0, Vec3::new(1.0, 0.92, 0.75)),
        (
            Vec3::new(0.02, 0.03, 0.04),
            0.06,
            0.2,
            Vec3::new(1.0, 0.82, 0.55) * 0.18,
        ),
    ];
    let (w, h) = (PALETTE_CELLS * 4, 4);
    let mut base = ImageData::new(w, h);
    let mut orm = ImageData::new(w, h);
    let mut emi = ImageData::new(w, h);
    for (i, (b, r, m, e)) in cells.into_iter().enumerate() {
        for y in 0..h {
            for x in 0..4 {
                let px = i as u32 * 4 + x;
                base.set(px, y, [to_srgb8(b.x), to_srgb8(b.y), to_srgb8(b.z), 255]);
                orm.set(px, y, [255, to_unorm8(r), to_unorm8(m), 255]);
                emi.set(px, y, [to_srgb8(e.x), to_srgb8(e.y), to_srgb8(e.z), 255]);
            }
        }
    }
    let src = |img: ImageData, srgb: bool| {
        let mut s = if srgb {
            TextureSource::from_image(img)
        } else {
            TextureSource::from_image_linear(img)
        };
        s.generate_mipmaps = false;
        s.sampler = SamplerSettings {
            filter: space3d::render::FilterMode::Nearest,
            address: space3d::render::AddressMode::Clamp,
        };
        s
    };
    Palette {
        base: src(base, true),
        orm: src(orm, false),
        emissive: src(emi, true),
    }
}

/// UV of the center of palette cell `i`.
pub fn cell_uv(i: u32) -> [f32; 2] {
    [(i as f32 + 0.5) / PALETTE_CELLS as f32, 0.5]
}

/// A soft radial glow (premultiplied-like falloff) for light pools and halos, 128².
pub fn glow_texture() -> ImageData {
    let s = 128u32;
    let mut img = ImageData::new(s, s);
    for y in 0..s {
        for x in 0..s {
            let u = (x as f32 + 0.5) / s as f32 * 2.0 - 1.0;
            let v = (y as f32 + 0.5) / s as f32 * 2.0 - 1.0;
            let r = (u * u + v * v).sqrt();
            let f = (1.0 - r).clamp(0.0, 1.0);
            let f = f * f * f * (0.35 + 0.65 * (1.0 - r).powi(2));
            let b = to_unorm8(f);
            img.set(x, y, [b, b, b, 255]);
        }
    }
    img
}

// ----------------------------------------------------------------------------------------------
// The sky
// ----------------------------------------------------------------------------------------------

/// The direction toward the sun: low over the horizon, a little off the streets' axis.
pub fn sun_direction() -> Vec3 {
    let (el, az) = (7.5f32.to_radians(), 18.0f32.to_radians());
    Vec3::new(el.cos() * az.sin(), el.sin(), -el.cos() * az.cos()).normalize()
}

/// Linear HDR radiance of the dusk sky toward `d`.
pub fn sky(d: Vec3) -> Vec3 {
    let sun = sun_direction();
    let up = d.y;
    let cos_s = d.dot(sun).clamp(-1.0, 1.0);
    let ang = cos_s.acos();
    // Gradient: warm band at the horizon, teal-to-indigo toward the zenith.
    let h = up.clamp(0.0, 1.0);
    let zenith = Vec3::new(0.018, 0.045, 0.15);
    let mid = Vec3::new(0.09, 0.17, 0.36);
    let horizon_far = Vec3::new(0.30, 0.20, 0.30);
    let horizon_sun = Vec3::new(1.5, 0.55, 0.16);
    let toward = smooth(-0.2, 1.0, cos_s);
    let horizon = horizon_far.lerp(horizon_sun, toward);
    let t1 = h.powf(0.45);
    let t2 = h.powf(0.8);
    let mut col = horizon.lerp(mid, t1).lerp(zenith, t2 * 0.8 + t1 * 0.0);
    // Sun glow: a wide scatter lobe, a tighter bright one and the disc.
    col += Vec3::new(1.0, 0.42, 0.12) * 0.45 * (-ang * 4.0).exp();
    col += Vec3::new(1.0, 0.62, 0.3) * 2.2 * (-ang * 11.0).exp();
    col += Vec3::new(1.0, 0.85, 0.6) * 220.0 * smooth(0.0105, 0.0065, ang.min(0.0105)) * (ang < 0.0105) as u32 as f32;
    // Clouds: a layer of stratocumulus, lit from below by the low sun.
    if up > 0.01 {
        let p = Vec3::new(d.x, 0.0, d.z) / (up + 0.18) * 2.2;
        let big = fbm3(Vec3::new(p.x * 0.7, 3.7, p.z * 1.2), 5, 41);
        let fine = fbm3(Vec3::new(p.x * 3.0, 1.3, p.z * 4.0), 4, 77);
        let dens = smooth(0.46, 0.7, big * 0.75 + fine * 0.3);
        let fade = smooth(0.0, 0.14, up) * (1.0 - smooth(0.7, 1.0, up) * 0.6);
        let sun_side = smooth(-0.3, 1.0, cos_s);
        // dark violet bases, orange-pink edges toward the sun
        let base = Vec3::new(0.10, 0.09, 0.2).lerp(Vec3::new(0.85, 0.42, 0.3), sun_side * (1.0 - h * 0.6));
        let edge = (1.0 - dens).powf(1.5) * 0.9 + 0.1;
        let lit = Vec3::new(1.6, 0.7, 0.35) * sun_side * sun_side * edge;
        let cloud = base * (0.55 + 0.5 * fine) + lit;
        col = col.lerp(cloud, dens * fade * 0.92);
    }
    if up < 0.0 {
        // Below the horizon: the dark warm city ground.
        let g = Vec3::new(0.085, 0.06, 0.05);
        col = col.lerp(g, smooth(0.0, -0.08, up));
    }
    col
}

/// Six `Rgba16Float` faces of [`sky`] (face order +X, −X, +Y, −Y, +Z, −Z).
pub fn sky_cubemap(size: u32) -> CubemapSource {
    let mut data = Vec::with_capacity((size * size * 48) as usize);
    for face in 0..6 {
        for y in 0..size {
            for x in 0..size {
                let s = (x as f32 + 0.5) / size as f32 * 2.0 - 1.0;
                let t = (y as f32 + 0.5) / size as f32 * 2.0 - 1.0;
                let d = match face {
                    0 => Vec3::new(1.0, -t, -s),
                    1 => Vec3::new(-1.0, -t, s),
                    2 => Vec3::new(s, 1.0, t),
                    3 => Vec3::new(s, -1.0, -t),
                    4 => Vec3::new(s, -t, 1.0),
                    _ => Vec3::new(-s, -t, -1.0),
                }
                .normalize();
                for c in sky(d).extend(1.0).to_array() {
                    data.extend_from_slice(&f32_to_f16(c.min(60000.0)).to_le_bytes());
                }
            }
        }
    }
    CubemapSource::Faces(TextureSource {
        width: size,
        height: size,
        layers: 6,
        format: space3d::wgpu::TextureFormat::Rgba16Float,
        mip_level_count: 1,
        data,
        generate_mipmaps: true,
        sampler: SamplerSettings::default(),
    })
}
