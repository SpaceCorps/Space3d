//! The city layout and its meshes: a grid of blocks split into lots, buildings with setbacks
//! and rooftop details, pavements, road markings, street lamps, traffic signals, signs, parked
//! cars and the soft light pools under the lamps. Everything is built into a few meshes per
//! chunk of blocks (so the renderer draws a few hundred meshes, not tens of thousands) and
//! returned as plain data for the main thread to upload.

use crate::mesh::MeshBuilder;
use crate::procgen::{BAY_M, BAYS, FLOOR_M, FLOORS, Facade, PAVING_TILE_M, ROOF_TILE_M, SHOP_M, cell, hash, unit};
use crate::props;
use space3d::prelude::*;

/// Street pitch (block + pavements + road), meters.
pub const PITCH: f32 = 75.0;
/// Half the distance from a block's center to its curb.
pub const CURB: f32 = 30.0;
/// Blocks per side.
pub const GRID: i32 = 16;
/// Blocks per chunk side.
pub const CHUNK: i32 = 4;
/// Height of the pavement above the road.
pub const KERB_H: f32 = 0.16;

pub fn block_center(i: i32, j: i32) -> Vec2 {
    Vec2::new((i as f32 - 7.5) * PITCH, (j as f32 - 7.5) * PITCH)
}

/// Coordinate of street line `s` (0..=GRID).
pub fn street(s: i32) -> f32 {
    (s as f32 - 8.0) * PITCH
}

fn chunk_index(x: f32, z: f32) -> usize {
    let c = |v: f32| (((v / PITCH + 7.5).floor() as i32).clamp(0, GRID - 1) / CHUNK) as usize;
    c(z) * (GRID / CHUNK) as usize + c(x)
}

pub struct Rng(u32);

impl Rng {
    pub fn new(seed: u32) -> Self {
        Self(hash(seed, 0x51ed, 7) | 1)
    }
    pub fn f(&mut self) -> f32 {
        self.0 = hash(self.0, 0x1234, 99).wrapping_add(0x9e37_79b9);
        unit(self.0)
    }
    pub fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.f()
    }
    pub fn int(&mut self, n: u32) -> u32 {
        ((self.f() * n as f32) as u32).min(n - 1)
    }
    pub fn chance(&mut self, p: f32) -> bool {
        self.f() < p
    }
}

/// A light the scene can switch on near the camera.
#[derive(Clone, Copy)]
pub struct LightSpot {
    pub pos: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub range: f32,
}

/// What one chunk draws.
pub struct ChunkMeshes {
    pub facade: [MeshData; 4],
    pub shop: MeshData,
    pub roof: MeshData,
    pub paving: MeshData,
    pub props_hot: MeshData,
    pub props_cold: MeshData,
    pub glow: MeshData,
}

#[derive(Default)]
struct ChunkBuild {
    facade: [MeshBuilder; 4],
    shop: MeshBuilder,
    roof: MeshBuilder,
    paving: MeshBuilder,
    props_hot: MeshBuilder,
    props_cold: MeshBuilder,
    glow: MeshBuilder,
}

impl ChunkBuild {
    fn finish(self) -> ChunkMeshes {
        let [a, b, c, d] = self.facade;
        ChunkMeshes {
            facade: [a.finish(), b.finish(), c.finish(), d.finish()],
            shop: self.shop.finish(),
            roof: self.roof.finish(),
            paving: self.paving.finish(),
            props_hot: self.props_hot.finish(),
            props_cold: self.props_cold.finish(),
            glow: self.glow.finish(),
        }
    }
}

pub struct CityData {
    pub chunks: Vec<ChunkMeshes>,
    pub ground: MeshData,
    pub lights: Vec<LightSpot>,
    /// Places that steam: manholes and roof vents.
    pub steam: Vec<Vec3>,
    /// Vehicle meshes: nine sedans (the last a taxi) and a bus.
    pub vehicles: Vec<MeshData>,
    /// Intersections' centers (for traffic).
    #[allow(dead_code)]
    pub extent: f32,
}

pub fn generate() -> CityData {
    let n_chunks = (GRID / CHUNK) as usize;
    let mut chunks: Vec<ChunkBuild> = (0..n_chunks * n_chunks).map(|_| ChunkBuild::default()).collect();
    let mut lights = Vec::new();
    let mut steam = Vec::new();

    for j in 0..GRID {
        for i in 0..GRID {
            let c = block_center(i, j);
            let ch = &mut chunks[chunk_index(c.x, c.y)];
            block(ch, &mut lights, &mut steam, i, j, c);
        }
    }
    for sz in 0..=GRID {
        for sx in 0..=GRID {
            let (x, z) = (street(sx), street(sz));
            let ch = &mut chunks[chunk_index(x, z)];
            intersection(ch, &mut lights, &mut steam, sx, sz, x, z);
        }
    }
    // road markings along every street segment
    for s in 0..=GRID {
        for t in 0..GRID {
            let a = street(t) + 14.0;
            let b = street(t + 1) - 14.0;
            let mid = (a + b) * 0.5;
            // a street along Z at x = street(s)
            let ch = &mut chunks[chunk_index(street(s), mid)];
            lane_marks(&mut ch.props_cold, true, street(s), a, b);
            // a street along X at z = street(s)
            let ch = &mut chunks[chunk_index(mid, street(s))];
            lane_marks(&mut ch.props_cold, false, street(s), a, b);
        }
    }

    let mut vehicles = Vec::new();
    for (k, p) in props::PAINTS.iter().enumerate() {
        let mut b = MeshBuilder::new();
        props::add_sedan(&mut b, *p, k == props::PAINTS.len() - 1);
        vehicles.push(b.finish());
    }
    let mut b = MeshBuilder::new();
    props::add_bus(&mut b);
    vehicles.push(b.finish());

    CityData {
        chunks: chunks.into_iter().map(ChunkBuild::finish).collect(),
        ground: ground_mesh(),
        lights,
        steam,
        vehicles,
        extent: street(GRID) - street(0),
    }
}

// ----------------------------------------------------------------------------------------------
// Ground
// ----------------------------------------------------------------------------------------------

fn ground_mesh() -> MeshData {
    use crate::procgen::ASPHALT_TILE_M;
    let mut b = MeshBuilder::new();
    let half = (street(GRID) + 40.0).max(660.0);
    let step = 15.0;
    let n = (2.0 * half / step) as i32;
    let up = Vec3::Y;
    for j in 0..n {
        for i in 0..n {
            let (x0, z0) = (-half + i as f32 * step, -half + j as f32 * step);
            let (x1, z1) = (x0 + step, z0 + step);
            // slow tonal variation hides the tile repetition
            let t = |x: f32, z: f32| {
                let a = crate::procgen::fbm_t(x / 40.0 + 100.0, z / 40.0 + 100.0, 64, 3, 5);
                0.78 + 0.45 * a
            };
            let col = |x: f32, z: f32| {
                let v = t(x, z);
                Vec4::new(v, v, v * 1.02, 1.0)
            };
            let uv = |x: f32, z: f32| [x / ASPHALT_TILE_M, z / ASPHALT_TILE_M];
            let base = b.m.positions.len() as u32;
            for (x, z) in [(x0, z1), (x1, z1), (x1, z0), (x0, z0)] {
                b.m.positions.push([x, 0.0, z]);
                b.m.normals.push(up.to_array());
                b.m.uvs.push(uv(x, z));
                b.m.colors.push(col(x, z).to_array());
            }
            b.m.indices
                .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }
    // far ground out to the horizon
    let far = 6000.0;
    for (a, bb) in [
        ((-far, -far), (far, -half)),
        ((-far, half), (far, far)),
        ((-far, -half), (-half, half)),
        ((half, -half), (far, half)),
    ] {
        let ((x0, z0), (x1, z1)) = (a, bb);
        let uv = |x: f32, z: f32| [x / ASPHALT_TILE_M, z / ASPHALT_TILE_M];
        let base = b.m.positions.len() as u32;
        for (x, z) in [(x0, z1), (x1, z1), (x1, z0), (x0, z0)] {
            b.m.positions.push([x, -0.02, z]);
            b.m.normals.push(up.to_array());
            b.m.uvs.push(uv(x, z));
            b.m.colors.push([0.7, 0.7, 0.7, 1.0]);
        }
        b.m.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    b.finish()
}

// ----------------------------------------------------------------------------------------------
// Road markings
// ----------------------------------------------------------------------------------------------

fn flat_quad(b: &mut MeshBuilder, x0: f32, z0: f32, x1: f32, z1: f32, y: f32, cell_id: u32) {
    b.quad_cell(
        [
            Vec3::new(x0, y, z1),
            Vec3::new(x1, y, z1),
            Vec3::new(x1, y, z0),
            Vec3::new(x0, y, z0),
        ],
        Vec3::Y,
        cell_id,
        Vec4::ONE,
    );
}

/// Center double line and dashed lane lines of a street segment `a..b` (along Z when `along_z`).
fn lane_marks(b: &mut MeshBuilder, along_z: bool, at: f32, a: f32, c: f32) {
    let y = 0.03;
    let rect = |b: &mut MeshBuilder, w0: f32, w1: f32, l0: f32, l1: f32, cell_id: u32| {
        if along_z {
            flat_quad(b, at + w0, l0, at + w1, l1, y, cell_id);
        } else {
            flat_quad(b, l0, at + w0, l1, at + w1, y, cell_id);
        }
    };
    for w in [-0.3, 0.15] {
        rect(b, w - 0.0, w + 0.15, a, c, cell::PAINT_YELLOW);
    }
    for off in [-3.5, 3.5] {
        let mut l = a + 2.0;
        while l + 3.0 < c - 2.0 {
            rect(b, off - 0.08, off + 0.08, l, l + 3.0, cell::PAINT_WHITE);
            l += 9.0;
        }
    }
}

fn intersection(
    ch: &mut ChunkBuild,
    lights: &mut Vec<LightSpot>,
    steam: &mut Vec<Vec3>,
    sx: i32,
    sz: i32,
    x: f32,
    z: f32,
) {
    let mut rng = Rng::new((sx * 131 + sz * 7919) as u32 + 17);
    let b = &mut ch.props_cold;
    let y = 0.03;
    // crosswalks and stop bars on the four arms
    for dir in [Vec3::Z, Vec3::NEG_Z, Vec3::X, Vec3::NEG_X] {
        let (ax, az) = (dir.x, dir.z);
        let mut k = -6.5f32;
        while k <= 6.6 {
            let (c0, c1) = (k - 0.25, k + 0.25);
            if ax == 0.0 {
                let (z0, z1) = (z + az * 8.5, z + az * 11.5);
                flat_quad(b, x + c0, z0.min(z1), x + c1, z0.max(z1), y, cell::PAINT_WHITE);
            } else {
                let (x0, x1) = (x + ax * 8.5, x + ax * 11.5);
                flat_quad(b, x0.min(x1), z + c0, x0.max(x1), z + c1, y, cell::PAINT_WHITE);
            }
            k += 1.0;
        }
        // stop bar across the incoming half of the road
        let side = |s: f32| (0.5 * s, 7.0 * s);
        if ax == 0.0 {
            let zc = z + az * 12.6;
            let (w0, w1) = if az > 0.0 { side(1.0) } else { side(-1.0) };
            flat_quad(
                b,
                x + w0.min(w1),
                zc - 0.25,
                x + w0.max(w1),
                zc + 0.25,
                y,
                cell::PAINT_WHITE,
            );
        } else {
            let xc = x + ax * 12.6;
            let (w0, w1) = if ax > 0.0 { side(-1.0) } else { side(1.0) };
            flat_quad(
                b,
                xc - 0.25,
                z + w0.min(w1),
                xc + 0.25,
                z + w0.max(w1),
                y,
                cell::PAINT_WHITE,
            );
        }
    }
    // signals on the four corners: each pole shows the two approaches that pass it
    let z_green = rng.chance(0.5);
    for (cx, cz) in [(1.0f32, 1.0f32), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)] {
        let p = Vec3::new(x + cx * 8.8, 0.0, z + cz * 8.8);
        let state_z = if z_green { 2 } else { 0 };
        let state_x = if z_green { 0 } else { 2 };
        let h1 = props::add_signal(&mut ch.props_hot, p, Vec3::new(0.0, 0.0, cz), state_z);
        let h2 = props::add_signal_head(&mut ch.props_hot, p, Vec3::new(cx, 0.0, 0.0), state_x);
        for (h, s) in [(h1, state_z), (h2, state_x)] {
            let color = if s == 2 {
                Vec3::new(0.2, 1.0, 0.4)
            } else {
                Vec3::new(1.0, 0.1, 0.06)
            };
            lights.push(LightSpot {
                pos: h,
                color,
                intensity: 3.0,
                range: 11.0,
            });
        }
    }
    if rng.chance(0.3) {
        steam.push(Vec3::new(x + rng.range(2.0, 5.0), 0.03, z + rng.range(-5.0, -2.0)));
    }
}

// ----------------------------------------------------------------------------------------------
// Blocks and buildings
// ----------------------------------------------------------------------------------------------

struct Lot {
    bx0: i32,
    bz0: i32,
    bx1: i32,
    bz1: i32,
}

fn split(lot: Lot, depth: u32, rng: &mut Rng, out: &mut Vec<Lot>) {
    let (w, d) = (lot.bx1 - lot.bx0, lot.bz1 - lot.bz0);
    let p = if depth == 0 { 0.8 } else { 0.5 };
    if depth < 3 && (w >= 7 || d >= 7) && rng.chance(p) {
        if w >= d && w >= 7 {
            let at = lot.bx0 + 3 + rng.int((w - 5) as u32) as i32;
            split(Lot { bx1: at, ..lot }, depth + 1, rng, out);
            split(
                Lot {
                    bx0: at,
                    bx1: lot.bx1,
                    bz0: lot.bz0,
                    bz1: lot.bz1,
                },
                depth + 1,
                rng,
                out,
            );
            return;
        } else if d >= 7 {
            let at = lot.bz0 + 3 + rng.int((d - 5) as u32) as i32;
            split(Lot { bz1: at, ..lot }, depth + 1, rng, out);
            split(
                Lot {
                    bz0: at,
                    bz1: lot.bz1,
                    bx0: lot.bx0,
                    bx1: lot.bx1,
                },
                depth + 1,
                rng,
                out,
            );
            return;
        }
    }
    out.push(lot);
}

fn block(ch: &mut ChunkBuild, lights: &mut Vec<LightSpot>, steam: &mut Vec<Vec3>, i: i32, j: i32, c: Vec2) {
    let mut rng = Rng::new((i * 977 + j * 31337) as u32 + 3);
    paving(&mut ch.paving, c);
    street_furniture(ch, lights, &mut rng, c);

    let lots_bays = 13;
    let mut lots = Vec::new();
    // a few blocks are one big lot
    let whole = Lot {
        bx0: 0,
        bz0: 0,
        bx1: lots_bays,
        bz1: lots_bays,
    };
    split(whole, 0, &mut rng, &mut lots);
    let dist = c.length();
    let downtown = (-(dist / 300.0).powi(2)).exp();
    let origin = Vec2::new(
        c.x - lots_bays as f32 * BAY_M * 0.5,
        c.y - lots_bays as f32 * BAY_M * 0.5,
    );
    for (k, lot) in lots.iter().enumerate() {
        let (w, d) = (lot.bx1 - lot.bx0, lot.bz1 - lot.bz0);
        if w < 3 || d < 3 {
            continue;
        }
        let mut r = Rng::new(hash(i as u32 * 3 + 1, j as u32 * 5 + 2, k as u32 + 11));
        // empty lots and parking in the low-rise ring
        if r.chance(0.04 + 0.1 * (1.0 - downtown)) {
            continue;
        }
        let x0 = origin.x + lot.bx0 as f32 * BAY_M;
        let z0 = origin.y + lot.bz0 as f32 * BAY_M;
        let x1 = origin.x + lot.bx1 as f32 * BAY_M;
        let z1 = origin.y + lot.bz1 as f32 * BAY_M;
        let area = (w * d) as f32;
        let mut floors = 3.0 + downtown.powf(1.1) * 62.0 * (0.25 + 0.75 * r.f()) * (0.6 + 0.4 * (area / 90.0).min(1.5));
        floors += 9.0 * r.f() * r.f();
        if downtown > 0.5 && r.chance(0.07) {
            floors += 22.0;
        }
        let floors = floors.round().max(3.0) as u32;
        let kind = pick_facade(floors, &mut r);
        let perimeter = [
            (z0 - (c.y - 25.0)).abs() < 2.5,
            (z1 - (c.y + 25.0)).abs() < 2.5,
            (x0 - (c.x - 25.0)).abs() < 2.5,
            (x1 - (c.x + 25.0)).abs() < 2.5,
        ];
        building(ch, lights, steam, &mut r, [x0, z0, x1, z1], floors, kind, perimeter);
    }
}

fn pick_facade(floors: u32, r: &mut Rng) -> Facade {
    let p = r.f();
    if floors > 30 {
        if p < 0.55 {
            Facade::Glass
        } else if p < 0.78 {
            Facade::Stone
        } else {
            Facade::Concrete
        }
    } else if floors > 10 {
        if p < 0.3 {
            Facade::Concrete
        } else if p < 0.55 {
            Facade::Glass
        } else if p < 0.78 {
            Facade::Stone
        } else {
            Facade::Brick
        }
    } else if p < 0.5 {
        Facade::Brick
    } else if p < 0.78 {
        Facade::Stone
    } else {
        Facade::Concrete
    }
}

fn facade_slot(k: Facade) -> usize {
    match k {
        Facade::Concrete => 0,
        Facade::Brick => 1,
        Facade::Glass => 2,
        Facade::Stone => 3,
    }
}

struct Part {
    x0: f32,
    z0: f32,
    x1: f32,
    z1: f32,
    fs: u32,
    nf: u32,
}

#[allow(clippy::too_many_arguments)]
fn building(
    ch: &mut ChunkBuild,
    lights: &mut Vec<LightSpot>,
    steam: &mut Vec<Vec3>,
    rng: &mut Rng,
    rect: [f32; 4],
    floors: u32,
    kind: Facade,
    perimeter: [bool; 4],
) {
    let [x0, z0, x1, z1] = rect;
    let (wb, db) = (((x1 - x0) / BAY_M).round() as i32, ((z1 - z0) / BAY_M).round() as i32);
    let mut parts = Vec::new();
    if floors >= 14 && wb >= 8 && db >= 8 && rng.chance(0.65) {
        let n_pod = ((floors as f32 * rng.range(0.14, 0.3)).round() as u32).max(3);
        let mut ins = [0i32; 4]; // x0, x1, z0, z1 in bays
        for v in &mut ins {
            *v = rng.int(3) as i32;
        }
        if ins.iter().all(|&v| v == 0) {
            ins[rng.int(4) as usize] = 2;
        }
        // keep the tower at least 5 bays each way
        while wb - ins[0] - ins[1] < 5 {
            ins[0] = (ins[0] - 1).max(0);
            ins[1] = (ins[1] - 1).max(0);
        }
        while db - ins[2] - ins[3] < 5 {
            ins[2] = (ins[2] - 1).max(0);
            ins[3] = (ins[3] - 1).max(0);
        }
        let (tx0, tx1) = (x0 + ins[0] as f32 * BAY_M, x1 - ins[1] as f32 * BAY_M);
        let (tz0, tz1) = (z0 + ins[2] as f32 * BAY_M, z1 - ins[3] as f32 * BAY_M);
        let mut tower_floors = floors - n_pod;
        parts.push(Part {
            x0,
            z0,
            x1,
            z1,
            fs: 0,
            nf: n_pod,
        });
        let crown = floors >= 40 && rng.chance(0.55) && (tx1 - tx0) / BAY_M >= 7.0 && (tz1 - tz0) / BAY_M >= 7.0;
        if crown {
            let n_crown = ((floors as f32 * 0.12).round() as u32).max(3);
            tower_floors -= n_crown;
            parts.push(Part {
                x0: tx0,
                z0: tz0,
                x1: tx1,
                z1: tz1,
                fs: n_pod,
                nf: tower_floors,
            });
            parts.push(Part {
                x0: tx0 + BAY_M,
                z0: tz0 + BAY_M,
                x1: tx1 - BAY_M,
                z1: tz1 - BAY_M,
                fs: n_pod + tower_floors,
                nf: n_crown,
            });
        } else {
            parts.push(Part {
                x0: tx0,
                z0: tz0,
                x1: tx1,
                z1: tz1,
                fs: n_pod,
                nf: tower_floors,
            });
        }
    } else {
        parts.push(Part {
            x0,
            z0,
            x1,
            z1,
            fs: 0,
            nf: floors,
        });
    }
    let slot = facade_slot(kind);
    let tint = {
        let v = rng.range(0.82, 1.1);
        Vec4::new(v * rng.range(0.96, 1.04), v, v * rng.range(0.94, 1.04), 1.0)
    };
    let n_parts = parts.len();
    for (pi, part) in parts.iter().enumerate() {
        let y0 = SHOP_M + part.fs as f32 * FLOOR_M;
        let y1 = SHOP_M + (part.fs + part.nf) as f32 * FLOOR_M;
        let top_ref = y1;
        let voff = rng.int(FLOORS) as f32;
        let (px0, pz0, px1, pz1) = (part.x0, part.z0, part.x1, part.z1);
        // (bottom-left corner seen from outside, right axis, normal, length)
        let sides = [
            (Vec3::new(px1, 0.0, pz0), Vec3::NEG_X, Vec3::NEG_Z, px1 - px0),
            (Vec3::new(px0, 0.0, pz1), Vec3::X, Vec3::Z, px1 - px0),
            (Vec3::new(px0, 0.0, pz0), Vec3::Z, Vec3::NEG_X, pz1 - pz0),
            (Vec3::new(px1, 0.0, pz1), Vec3::NEG_Z, Vec3::X, pz1 - pz0),
        ];
        for (si, (corner, right, n, len)) in sides.into_iter().enumerate() {
            let uoff = rng.int(BAYS) as f32;
            let u0 = uoff / BAYS as f32;
            let u1 = u0 + (len / BAY_M) / BAYS as f32;
            let v = |y: f32| (voff + (top_ref - y) / FLOOR_M) / FLOORS as f32;
            let wall_y0 = if part.fs == 0 { SHOP_M } else { y0 };
            let bl = Vec3::new(corner.x, wall_y0, corner.z);
            let up = Vec3::Y * (y1 - wall_y0);
            let br = bl + right * len;
            ch.facade[slot].quad(
                [bl, br, br + up, bl + up],
                n,
                [[u0, v(wall_y0)], [u1, v(wall_y0)], [u1, v(y1)], [u0, v(y1)]],
                tint,
            );
            if part.fs == 0 {
                let bl = Vec3::new(corner.x, 0.0, corner.z);
                let br = bl + right * len;
                let up = Vec3::Y * SHOP_M;
                let su = rng.int(BAYS) as f32 / BAYS as f32;
                let su1 = su + (len / BAY_M) / BAYS as f32;
                ch.shop.quad(
                    [bl, br, br + up, bl + up],
                    n,
                    [[su, 1.0], [su1, 1.0], [su1, 0.0], [su, 0.0]],
                    Vec4::ONE,
                );
                // light spilling from the shops onto the pavement
                if pi == 0 && perimeter[si] {
                    let warm = rng.chance(0.7);
                    let col = if warm {
                        Vec4::new(1.0, 0.66, 0.32, 1.0) * 0.34
                    } else {
                        Vec4::new(0.55, 0.75, 1.0, 1.0) * 0.3
                    };
                    let mid = (bl + br) * 0.5 + Vec3::Y * (KERB_H + 0.02);
                    glow_rect(
                        &mut ch.glow,
                        mid,
                        right * (len * 0.5 + 2.5),
                        n * 4.2,
                        Vec4::new(col.x, col.y, col.z, 1.0),
                    );
                }
            }
        }
        // roof
        let ty = y1;
        let q = [
            Vec3::new(px0, ty, pz1),
            Vec3::new(px1, ty, pz1),
            Vec3::new(px1, ty, pz0),
            Vec3::new(px0, ty, pz0),
        ];
        let uv = |p: Vec3| [p.x / ROOF_TILE_M, p.z / ROOF_TILE_M];
        ch.roof
            .quad(q, Vec3::Y, [uv(q[0]), uv(q[1]), uv(q[2]), uv(q[3])], Vec4::ONE);
        if pi == n_parts - 1 {
            roof_details(ch, steam, rng, [px0, pz0, px1, pz1], ty, floors);
        }
        // a blade sign on a street side of the podium
        if pi == 0 && rng.chance(0.55) {
            let candidates: Vec<usize> = (0..4).filter(|&s| perimeter[s]).collect();
            if !candidates.is_empty() {
                let s = candidates[rng.int(candidates.len() as u32) as usize];
                let neon = [
                    cell::NEON_CYAN,
                    cell::NEON_MAGENTA,
                    cell::NEON_ORANGE,
                    cell::LAMP_WARM,
                    cell::GREEN,
                ][rng.int(5) as usize];
                blade_sign(ch, lights, rng, [px0, pz0, px1, pz1], s, neon);
            }
        }
    }
}

fn blade_sign(ch: &mut ChunkBuild, lights: &mut Vec<LightSpot>, rng: &mut Rng, r: [f32; 4], side: usize, neon: u32) {
    let [x0, z0, x1, z1] = r;
    let (along_x, normal, fixed) = match side {
        0 => (true, -1.0f32, z0),
        1 => (true, 1.0, z1),
        2 => (false, -1.0, x0),
        _ => (false, 1.0, x1),
    };
    let len = if along_x { x1 - x0 } else { z1 - z0 };
    let t = rng.range(2.5, (len - 2.5).max(2.6));
    let (y0, y1) = (3.4, 5.6);
    let depth = 1.3;
    let (min, max, cells) = if along_x {
        let x = x0 + t;
        let (za, zb) = if normal < 0.0 {
            (fixed - depth, fixed)
        } else {
            (fixed, fixed + depth)
        };
        (
            Vec3::new(x - 0.15, y0, za),
            Vec3::new(x + 0.15, y1, zb),
            [neon, neon, cell::STEEL, cell::STEEL, cell::STEEL, cell::STEEL],
        )
    } else {
        let z = z0 + t;
        let (xa, xb) = if normal < 0.0 {
            (fixed - depth, fixed)
        } else {
            (fixed, fixed + depth)
        };
        (
            Vec3::new(xa, y0, z - 0.15),
            Vec3::new(xb, y1, z + 0.15),
            [cell::STEEL, cell::STEEL, cell::STEEL, cell::STEEL, neon, neon],
        )
    };
    ch.props_hot.box_cells(min, max, cells, Vec4::ONE);
    let color = match neon {
        cell::NEON_CYAN => Vec3::new(0.1, 0.8, 1.0),
        cell::NEON_MAGENTA => Vec3::new(1.0, 0.12, 0.55),
        cell::NEON_ORANGE => Vec3::new(1.0, 0.45, 0.08),
        cell::GREEN => Vec3::new(0.1, 1.0, 0.4),
        _ => Vec3::new(1.0, 0.72, 0.4),
    };
    let c = (min + max) * 0.5;
    lights.push(LightSpot {
        pos: c + Vec3::new(0.0, 0.0, 0.0),
        color,
        intensity: 4.0,
        range: 11.0,
    });
}

fn roof_details(ch: &mut ChunkBuild, steam: &mut Vec<Vec3>, rng: &mut Rng, r: [f32; 4], y: f32, floors: u32) {
    let [x0, z0, x1, z1] = r;
    let c = Vec4::new(0.9, 0.9, 0.92, 1.0);
    let t = 0.35;
    let h = 0.9;
    let b = &mut ch.props_cold;
    // parapet
    b.box_cell(Vec3::new(x0, y, z0), Vec3::new(x1, y + h, z0 + t), cell::CONCRETE, c);
    b.box_cell(Vec3::new(x0, y, z1 - t), Vec3::new(x1, y + h, z1), cell::CONCRETE, c);
    b.box_cell(
        Vec3::new(x0, y, z0 + t),
        Vec3::new(x0 + t, y + h, z1 - t),
        cell::CONCRETE,
        c,
    );
    b.box_cell(
        Vec3::new(x1 - t, y, z0 + t),
        Vec3::new(x1, y + h, z1 - t),
        cell::CONCRETE,
        c,
    );
    let (w, d) = (x1 - x0, z1 - z0);
    if w < 6.0 || d < 6.0 {
        return;
    }
    let n = 2 + rng.int(5);
    for _ in 0..n {
        let sx = rng.range(1.2, 3.8).min(w * 0.3);
        let sz = rng.range(1.2, 3.2).min(d * 0.3);
        let sy = rng.range(0.7, 2.2);
        let px = rng.range(x0 + 1.0, x1 - 1.0 - sx);
        let pz = rng.range(z0 + 1.0, z1 - 1.0 - sz);
        b.box_cell(
            Vec3::new(px, y, pz),
            Vec3::new(px + sx, y + sy, pz + sz),
            cell::CONCRETE,
            c,
        );
        // fan grille on top of some
        if rng.chance(0.4) {
            b.box_cell(
                Vec3::new(px + 0.2, y + sy, pz + 0.2),
                Vec3::new(px + sx - 0.2, y + sy + 0.12, pz + sz - 0.2),
                cell::STEEL,
                Vec4::ONE,
            );
        }
        if rng.chance(0.0) {
            steam.push(Vec3::new(px + sx * 0.5, y + sy + 0.2, pz + sz * 0.5));
        }
    }
    if floors > 8 && rng.chance(0.3) {
        let px = rng.range(x0 + 2.5, x1 - 2.5);
        let pz = rng.range(z0 + 2.5, z1 - 2.5);
        b.cylinder(
            Vec3::new(px, y + 1.2, pz),
            Vec3::Y,
            1.3,
            2.6,
            12,
            cell::RUBBER,
            Vec4::new(0.6, 0.4, 0.3, 1.0),
        );
        for (dx, dz) in [(-0.8, -0.8), (0.8, -0.8), (-0.8, 0.8), (0.8, 0.8)] {
            b.box_cell(
                Vec3::new(px + dx - 0.08, y, pz + dz - 0.08),
                Vec3::new(px + dx + 0.08, y + 1.2, pz + dz + 0.08),
                cell::STEEL,
                Vec4::ONE,
            );
        }
    }
    if floors >= 22 && rng.chance(0.85) {
        let px = (x0 + x1) * 0.5 + rng.range(-w * 0.2, w * 0.2);
        let pz = (z0 + z1) * 0.5 + rng.range(-d * 0.2, d * 0.2);
        let hh = rng.range(10.0, 26.0);
        b.cylinder(Vec3::new(px, y, pz), Vec3::Y, 0.14, hh, 6, cell::STEEL, Vec4::ONE);
        ch.props_hot.box_cell(
            Vec3::new(px - 0.22, y + hh, pz - 0.22),
            Vec3::new(px + 0.22, y + hh + 0.5, pz + 0.22),
            cell::RED,
            Vec4::ONE,
        );
    } else if (6..20).contains(&floors) && rng.chance(0.14) {
        // a rooftop billboard
        let neon = [cell::NEON_CYAN, cell::NEON_MAGENTA, cell::NEON_ORANGE][rng.int(3) as usize];
        let wsign = (w * 0.6).min(12.0);
        let cx = (x0 + x1) * 0.5;
        let cz = (z0 + z1) * 0.5;
        ch.props_hot.box_cells(
            Vec3::new(cx - wsign * 0.5, y + 2.0, cz - 0.25),
            Vec3::new(cx + wsign * 0.5, y + 6.0, cz + 0.25),
            [cell::STEEL, cell::STEEL, cell::STEEL, cell::STEEL, neon, neon],
            Vec4::ONE,
        );
        for dx in [-wsign * 0.4, wsign * 0.4] {
            ch.props_cold.box_cell(
                Vec3::new(cx + dx - 0.15, y, cz - 0.15),
                Vec3::new(cx + dx + 0.15, y + 2.0, cz + 0.15),
                cell::STEEL,
                Vec4::ONE,
            );
        }
    }
}

// ----------------------------------------------------------------------------------------------
// Pavements, lamps, parked cars
// ----------------------------------------------------------------------------------------------

fn paving(b: &mut MeshBuilder, c: Vec2) {
    let h = CURB;
    let (x0, x1, z0, z1) = (c.x - h, c.x + h, c.y - h, c.y + h);
    let y = KERB_H;
    let uv = |x: f32, z: f32| [x / PAVING_TILE_M, z / PAVING_TILE_M];
    b.quad(
        [
            Vec3::new(x0, y, z1),
            Vec3::new(x1, y, z1),
            Vec3::new(x1, y, z0),
            Vec3::new(x0, y, z0),
        ],
        Vec3::Y,
        [uv(x0, z1), uv(x1, z1), uv(x1, z0), uv(x0, z0)],
        Vec4::ONE,
    );
    // curb faces
    let sides = [
        (Vec3::new(x1, 0.0, z0), Vec3::NEG_X, Vec3::NEG_Z, x1 - x0),
        (Vec3::new(x0, 0.0, z1), Vec3::X, Vec3::Z, x1 - x0),
        (Vec3::new(x0, 0.0, z0), Vec3::Z, Vec3::NEG_X, z1 - z0),
        (Vec3::new(x1, 0.0, z1), Vec3::NEG_Z, Vec3::X, z1 - z0),
    ];
    for (bl, right, n, len) in sides {
        let br = bl + right * len;
        let up = Vec3::Y * y;
        b.quad(
            [bl, br, br + up, bl + up],
            n,
            [
                [0.0, 0.2],
                [len / PAVING_TILE_M, 0.2],
                [len / PAVING_TILE_M, 0.0],
                [0.0, 0.0],
            ],
            Vec4::new(0.8, 0.8, 0.8, 1.0),
        );
    }
}

fn street_furniture(ch: &mut ChunkBuild, lights: &mut Vec<LightSpot>, rng: &mut Rng, c: Vec2) {
    // lamps on every side, two per side, staggered; the arms reach over the road
    let inset = CURB - 1.3;
    for (si, (dir, along)) in [
        (Vec3::NEG_Z, Vec3::X),
        (Vec3::Z, Vec3::X),
        (Vec3::NEG_X, Vec3::Z),
        (Vec3::X, Vec3::Z),
    ]
    .into_iter()
    .enumerate()
    {
        for (k, off) in [-13.0f32, 12.0].into_iter().enumerate() {
            let off = off + if (si + k) % 2 == 0 { 2.0 } else { -2.0 };
            let p = Vec3::new(c.x, 0.0, c.y) + dir * inset + along * off + Vec3::Y * KERB_H;
            let head = props::add_street_lamp(&mut ch.props_hot, p, dir);
            lights.push(LightSpot {
                pos: head,
                color: Vec3::new(1.0, 0.66, 0.34),
                intensity: 11.0,
                range: 26.0,
            });
            // a pool on the road beyond the curb and one on the pavement under the lamp
            let road_c = Vec3::new(head.x, 0.04, head.z) + dir * 0.5;
            pool(&mut ch.glow, road_c, 8.5, Vec4::new(1.0, 0.58, 0.26, 1.0) * 0.5);
            let pave_c = Vec3::new(p.x, KERB_H + 0.02, p.z);
            pool(&mut ch.glow, pave_c, 4.8, Vec4::new(1.0, 0.62, 0.3, 1.0) * 0.55);
        }
        // parked cars along this curb
        let n = rng.int(5);
        for k in 0..n {
            let t = -22.0 + k as f32 * 9.5 + rng.range(-1.0, 1.0);
            let pos = Vec3::new(c.x, 0.0, c.y) + dir * (CURB + 1.15) + along * t;
            let paint = props::PAINTS[rng.int(8) as usize];
            // cars face along the curb, a random way
            let fwd = if rng.chance(0.5) { along } else { -along };
            let rot = Quat::from_rotation_arc(Vec3::Z, fwd);
            ch.props_cold.xform = Mat4::from_rotation_translation(rot, pos);
            props::add_sedan(&mut ch.props_cold, paint, false);
            ch.props_cold.xform = Mat4::IDENTITY;
        }
    }
}

/// A soft round light pool of `radius` meters at `center` (a horizontal quad with the glow
/// texture, additive).
pub fn pool(b: &mut MeshBuilder, center: Vec3, radius: f32, color: Vec4) {
    glow_rect(b, center, Vec3::X * radius, Vec3::Z * radius, color);
}

/// A horizontal glow quad around `center` with half extents `e1` and `e2`.
pub fn glow_rect(b: &mut MeshBuilder, center: Vec3, e1: Vec3, e2: Vec3, color: Vec4) {
    let mut p = [center - e1 - e2, center + e1 - e2, center + e1 + e2, center - e1 + e2];
    let mut uv = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    if (p[1] - p[0]).cross(p[3] - p[0]).y < 0.0 {
        p.reverse();
        uv.reverse();
    }
    b.quad(p, Vec3::Y, uv, Vec4::new(color.x, color.y, color.z, 1.0));
}

#[allow(dead_code)]
const _: (u32, u32) = (BAYS, FLOORS);
