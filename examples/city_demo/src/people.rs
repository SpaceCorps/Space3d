//! Pedestrians: the CC0 "Animated Human" by Quaternius (a low-poly walking pose, baked to
//! `assets/models/quaternius_human.bin` with a colour zone per vertex), recoloured per outfit,
//! with a mirrored copy so swapping the two every step reads as a walk cycle. Umbrellas are
//! procedural.

use crate::mesh::MeshBuilder;
use crate::procgen::{cell, cell_uv};
use space3d::prelude::*;

const DATA: &[u8] = include_bytes!("../assets/models/quaternius_human.bin");

/// One path a pedestrian paces: from `origin` along `axis` for `len` meters and back.
#[derive(Clone, Copy)]
pub struct Walker {
    pub origin: Vec3,
    pub axis: Vec3,
    pub len: f32,
    pub speed: f32,
    /// Distance already walked at time zero.
    pub phase: f32,
    pub outfit: u32,
}

#[derive(Clone, Copy)]
pub struct Outfit {
    pub skin: Vec3,
    pub hair: Vec3,
    pub shirt: Vec3,
    pub pants: Vec3,
    pub umbrella: Option<Vec3>,
}

pub const OUTFITS: usize = 16;

/// Muted city clothing, a few skin tones, about every third person under an umbrella.
pub fn outfit(i: u32) -> Outfit {
    let skins = [
        Vec3::new(0.62, 0.42, 0.32),
        Vec3::new(0.45, 0.29, 0.21),
        Vec3::new(0.24, 0.14, 0.1),
        Vec3::new(0.55, 0.38, 0.28),
    ];
    let hairs = [
        Vec3::new(0.04, 0.03, 0.03),
        Vec3::new(0.16, 0.09, 0.05),
        Vec3::new(0.3, 0.2, 0.1),
        Vec3::new(0.35, 0.34, 0.33),
    ];
    let shirts = [
        Vec3::new(0.03, 0.035, 0.045),
        Vec3::new(0.3, 0.05, 0.04),
        Vec3::new(0.05, 0.1, 0.22),
        Vec3::new(0.4, 0.36, 0.3),
        Vec3::new(0.06, 0.2, 0.1),
        Vec3::new(0.5, 0.38, 0.06),
        Vec3::new(0.45, 0.45, 0.46),
        Vec3::new(0.12, 0.06, 0.18),
    ];
    let pants = [
        Vec3::new(0.02, 0.02, 0.025),
        Vec3::new(0.05, 0.08, 0.16),
        Vec3::new(0.14, 0.12, 0.09),
        Vec3::new(0.1, 0.1, 0.1),
    ];
    let umbrellas = [
        Vec3::new(0.02, 0.02, 0.03),
        Vec3::new(0.35, 0.03, 0.03),
        Vec3::new(0.03, 0.08, 0.3),
        Vec3::new(0.5, 0.4, 0.03),
    ];
    let k = i as usize;
    Outfit {
        skin: skins[(k * 3) % skins.len()],
        hair: hairs[(k * 5 + 1) % hairs.len()],
        shirt: shirts[k % shirts.len()],
        pants: pants[(k / 2) % pants.len()],
        umbrella: if k.is_multiple_of(3) {
            Some(umbrellas[(k / 3) % umbrellas.len()])
        } else {
            None
        },
    }
}

fn f32_at(b: &[u8], o: usize) -> f32 {
    f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

/// The human in outfit `o`; `mirror` flips left and right (the other half of the stride).
pub fn mesh(o: &Outfit, mirror: bool) -> MeshData {
    let (nv, ni) = (u32_at(DATA, 0) as usize, u32_at(DATA, 4) as usize);
    let uv = cell_uv(cell::CONCRETE);
    let mut b = MeshBuilder::new();
    let flip = if mirror { -1.0 } else { 1.0 };
    for i in 0..nv {
        let at = 8 + i * 28;
        let g = |k: usize| f32_at(DATA, at + k * 4);
        let col = match u32_at(DATA, at + 24) {
            0 => o.skin,
            1 => o.pants,
            2 => o.shirt,
            3 => o.hair,
            _ => Vec3::splat(0.01),
        };
        b.push_vertex(
            Vec3::new(g(0) * flip, g(1), g(2)),
            Vec3::new(g(3) * flip, g(4), g(5)),
            uv,
            col.extend(1.0),
        );
    }
    let base = 8 + nv * 28;
    for t in 0..ni / 3 {
        let ix = |k: usize| u32_at(DATA, base + (t * 3 + k) * 4);
        if mirror {
            b.m.indices.extend_from_slice(&[ix(0), ix(2), ix(1)]);
        } else {
            b.m.indices.extend_from_slice(&[ix(0), ix(1), ix(2)]);
        }
    }
    if let Some(c) = o.umbrella {
        umbrella(&mut b, c);
    }
    b.finish()
}

/// A canopy (two-sided cone) on a thin shaft, held over the right shoulder.
fn umbrella(b: &mut MeshBuilder, c: Vec3) {
    let (cx, cz) = (0.28f32, 0.05f32);
    b.cylinder(
        Vec3::new(cx, 1.0, cz),
        Vec3::Y,
        0.012,
        1.25,
        4,
        cell::STEEL,
        Vec4::new(0.3, 0.3, 0.3, 1.0),
    );
    let n = 12u32;
    let (apex, rim_y, r) = (Vec3::new(cx, 2.33, cz), 2.08f32, 0.62f32);
    let uv = cell_uv(cell::CONCRETE);
    let col = c.extend(1.0);
    for side in [1.0f32, -1.0] {
        let start = b.m.positions.len() as u32;
        b.push_vertex(apex, Vec3::Y * side, uv, col);
        for i in 0..n {
            let a = std::f32::consts::TAU * i as f32 / n as f32;
            let p = Vec3::new(cx + a.cos() * r, rim_y, cz + a.sin() * r);
            let nrm = Vec3::new(a.cos() * 0.4, 0.9, a.sin() * 0.4).normalize() * side;
            b.push_vertex(p, nrm, uv, col);
        }
        for i in 0..n {
            let (p, q) = (start + 1 + i, start + 1 + (i + 1) % n);
            if side > 0.0 {
                b.m.indices.extend_from_slice(&[start, q, p]);
            } else {
                b.m.indices.extend_from_slice(&[start, p, q]);
            }
        }
    }
}

/// Position, yaw and stride parity of a walker at `time`.
pub fn pose(w: &Walker, time: f32) -> (Vec3, f32, usize) {
    let s = (w.phase + time * w.speed).rem_euclid(2.0 * w.len);
    let (u, sign) = if s < w.len { (s, 1.0) } else { (2.0 * w.len - s, -1.0) };
    let step = 0.7;
    let bob = (std::f32::consts::PI * u / step).sin().abs() * 0.035;
    let f = w.axis * sign;
    (
        w.origin + w.axis * u + Vec3::Y * bob,
        f.x.atan2(f.z),
        ((u / step) as usize) % 2,
    )
}
