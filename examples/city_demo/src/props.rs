//! Street furniture and vehicles, built from the palette cells (see `procgen::cell`).

use crate::mesh::{MeshBuilder, sec};
use crate::procgen::cell;
use space3d::prelude::*;

const WHITE: Vec4 = Vec4::ONE;

/// Paint colors of the sedans (linear).
pub const PAINTS: [Vec3; 9] = [
    Vec3::new(0.80, 0.80, 0.78),   // white
    Vec3::new(0.025, 0.025, 0.03), // black
    Vec3::new(0.30, 0.31, 0.33),   // silver
    Vec3::new(0.01, 0.05, 0.20),   // blue
    Vec3::new(0.35, 0.02, 0.02),   // red
    Vec3::new(0.02, 0.10, 0.06),   // green
    Vec3::new(0.10, 0.10, 0.11),   // graphite
    Vec3::new(0.55, 0.30, 0.04),   // bronze
    Vec3::new(0.85, 0.55, 0.02),   // taxi yellow
];

/// A sedan facing +Z at the origin (length 4.4 m).
pub fn add_sedan(b: &mut MeshBuilder, paint: Vec3, taxi: bool) {
    let c = paint.extend(1.0);
    b.loft(
        &[
            sec(-2.23, 0.36, 0.58, 0.70, 0.64),
            sec(-2.15, 0.28, 0.80, 0.83, 0.76),
            sec(-1.98, 0.26, 0.95, 0.89, 0.82),
            sec(1.40, 0.26, 0.92, 0.89, 0.82),
            sec(2.00, 0.26, 0.84, 0.88, 0.79),
            sec(2.15, 0.28, 0.70, 0.84, 0.74),
            sec(2.23, 0.36, 0.56, 0.72, 0.64),
        ],
        &[cell::CAR_PAINT],
        cell::CAR_PAINT,
        cell::RUBBER,
        c,
    );
    b.loft(
        &[
            sec(-1.30, 0.92, 0.96, 0.78, 0.70),
            sec(-0.95, 0.92, 1.28, 0.78, 0.66),
            sec(-0.55, 0.92, 1.42, 0.77, 0.60),
            sec(0.20, 0.92, 1.42, 0.77, 0.60),
            sec(0.60, 0.92, 1.30, 0.78, 0.65),
            sec(1.00, 0.92, 0.96, 0.78, 0.70),
        ],
        &[cell::GLASS, cell::GLASS, cell::CAR_PAINT, cell::GLASS, cell::GLASS],
        cell::GLASS,
        cell::RUBBER,
        c,
    );
    for z in [-1.32, 1.38] {
        for side in [-1.0f32, 1.0] {
            let x0 = if side > 0.0 { 0.70 } else { -0.94 };
            b.cylinder(Vec3::new(x0, 0.33, z), Vec3::X, 0.33, 0.24, 12, cell::RUBBER, WHITE);
            let hub = if side > 0.0 { 0.94 } else { -0.95 };
            b.cylinder(Vec3::new(hub, 0.33, z), Vec3::X, 0.2, 0.01, 10, cell::STEEL, WHITE);
        }
    }
    for side in [-1.0f32, 1.0] {
        let (x0, x1) = (0.42 * side, 0.78 * side);
        let (lo, hi) = (x0.min(x1), x0.max(x1));
        b.box_cell(
            Vec3::new(lo, 0.54, 2.19),
            Vec3::new(hi, 0.65, 2.25),
            cell::HEADLIGHT,
            WHITE,
        );
        b.box_cell(Vec3::new(lo, 0.66, -2.25), Vec3::new(hi, 0.76, -2.19), cell::RED, WHITE);
    }
    if taxi {
        b.box_cell(
            Vec3::new(-0.25, 1.42, -0.2),
            Vec3::new(0.25, 1.58, 0.2),
            cell::LAMP_WARM,
            WHITE,
        );
    }
}

/// A city bus facing +Z at the origin (11 m).
pub fn add_bus(b: &mut MeshBuilder) {
    let c = Vec4::new(0.05, 0.14, 0.38, 1.0);
    b.box_cells(
        Vec3::new(-1.28, 0.55, -5.5),
        Vec3::new(1.28, 3.2, 5.5),
        [
            cell::CAR_PAINT,
            cell::CAR_PAINT,
            cell::CAR_PAINT,
            cell::RUBBER,
            cell::CAR_PAINT,
            cell::CAR_PAINT,
        ],
        c,
    );
    // lit window bands
    for side in [-1.0f32, 1.0] {
        let x = 1.285 * side;
        for i in 0..6 {
            let z0 = -4.9 + i as f32 * 1.65;
            let (lo, hi) = if side > 0.0 { (1.28, 1.30) } else { (-1.30, -1.28) };
            let _ = x;
            b.box_cell(
                Vec3::new(lo, 1.45, z0),
                Vec3::new(hi, 2.65, z0 + 1.4),
                cell::BUS_WINDOW,
                WHITE,
            );
        }
    }
    // windscreen and headlights
    b.box_cell(Vec3::new(-1.1, 1.6, 5.5), Vec3::new(1.1, 2.9, 5.52), cell::GLASS, WHITE);
    for side in [-1.0f32, 1.0] {
        let (lo, hi) = if side > 0.0 { (0.8, 1.15) } else { (-1.15, -0.8) };
        b.box_cell(
            Vec3::new(lo, 0.75, 5.5),
            Vec3::new(hi, 0.95, 5.53),
            cell::HEADLIGHT,
            WHITE,
        );
        b.box_cell(Vec3::new(lo, 1.0, -5.53), Vec3::new(hi, 1.2, -5.5), cell::RED, WHITE);
    }
    b.box_cell(
        Vec3::new(-0.8, 3.0, 5.5),
        Vec3::new(0.8, 3.18, 5.52),
        cell::AMBER,
        WHITE,
    );
    for z in [-3.6, 3.6] {
        for side in [-1.0f32, 1.0] {
            let x0 = if side > 0.0 { 1.0 } else { -1.28 };
            b.cylinder(Vec3::new(x0, 0.5, z), Vec3::X, 0.5, 0.28, 12, cell::RUBBER, WHITE);
        }
    }
}

/// A street lamp on a sidewalk: pole at `p` (ground), the arm reaching `out` (a unit axis
/// direction, toward the road). Returns the position of the lamp.
pub fn add_street_lamp(b: &mut MeshBuilder, p: Vec3, out: Vec3) -> Vec3 {
    let h = 8.2;
    b.cylinder(p, Vec3::Y, 0.1, h, 8, cell::STEEL, WHITE);
    b.cylinder(p, Vec3::Y, 0.16, 0.5, 8, cell::STEEL, WHITE);
    let reach = 2.6;
    let end = p + out * reach;
    let arm_min = Vec3::new(p.x.min(end.x) - 0.05, h - 0.1, p.z.min(end.z) - 0.05);
    let arm_max = Vec3::new(p.x.max(end.x) + 0.05, h + 0.02, p.z.max(end.z) + 0.05);
    b.box_cell(arm_min, arm_max, cell::STEEL, WHITE);
    // the head: a flat box with the lamp on its underside
    let hc = p + out * (reach - 0.3);
    let half = if out.x.abs() > 0.5 {
        Vec3::new(0.5, 0.08, 0.2)
    } else {
        Vec3::new(0.2, 0.08, 0.5)
    };
    let min = Vec3::new(hc.x - half.x, h - 0.2, hc.z - half.z);
    let max = Vec3::new(hc.x + half.x, h - 0.04, hc.z + half.z);
    b.box_cells(
        min,
        max,
        [
            cell::STEEL,
            cell::STEEL,
            cell::STEEL,
            cell::LAMP_WARM,
            cell::STEEL,
            cell::STEEL,
        ],
        WHITE,
    );
    Vec3::new(hc.x, h - 0.4, hc.z)
}

/// A traffic signal pole at `p` with a head facing `face` (axis direction); `state` 0 red,
/// 1 amber, 2 green. Returns the head position.
pub fn add_signal(b: &mut MeshBuilder, p: Vec3, face: Vec3, state: u32) -> Vec3 {
    b.cylinder(p, Vec3::Y, 0.09, 5.2, 8, cell::STEEL, WHITE);
    add_signal_head(b, p, face, state)
}

/// Just the head of a signal (on the pole at `p`).
pub fn add_signal_head(b: &mut MeshBuilder, p: Vec3, face: Vec3, state: u32) -> Vec3 {
    let center = p + face * 0.38 + Vec3::Y * 4.4;
    let half = if face.x.abs() > 0.5 {
        Vec3::new(0.15, 0.55, 0.19)
    } else {
        Vec3::new(0.19, 0.55, 0.15)
    };
    b.box_cell(center - half, center + half, cell::RUBBER, WHITE);
    for k in 0..3u32 {
        let y = center.y + 0.36 - 0.36 * k as f32;
        let on = k == state;
        let cell_id = if !on {
            cell::GLASS
        } else {
            [cell::RED, cell::AMBER, cell::GREEN][k as usize]
        };
        let c = Vec3::new(center.x, y, center.z) + face * (half.x.max(half.z) + 0.01);
        let r = if face.x.abs() > 0.5 {
            Vec3::new(0.012, 0.11, 0.11)
        } else {
            Vec3::new(0.11, 0.11, 0.012)
        };
        b.box_cell(c - r, c + r, cell_id, WHITE);
    }
    Vec3::new(center.x, center.y + 0.36 - 0.36 * state as f32, center.z) + face * 0.3
}
