//! A small mesh builder: quads, boxes, cylinders and lofts, with an optional transform applied
//! to everything pushed (so one car or lamp builder places itself anywhere).

use crate::procgen::cell_uv;
use space3d::prelude::*;

pub struct MeshBuilder {
    pub m: MeshData,
    /// Applied to positions (and its rotation to normals) as vertices are pushed.
    pub xform: Mat4,
}

impl Default for MeshBuilder {
    fn default() -> Self {
        Self {
            m: MeshData::default(),
            xform: Mat4::IDENTITY,
        }
    }
}

impl MeshBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.m.indices.is_empty()
    }

    fn vertex(&mut self, p: Vec3, n: Vec3, uv: [f32; 2], c: Vec4) {
        let p = self.xform.transform_point3(p);
        let n = self.xform.transform_vector3(n).normalize_or_zero();
        self.m.positions.push(p.to_array());
        self.m.normals.push(n.to_array());
        self.m.uvs.push(uv);
        self.m.colors.push(c.to_array());
    }

    /// A quad, counter-clockwise seen from the side `n` points to.
    pub fn quad(&mut self, p: [Vec3; 4], n: Vec3, uv: [[f32; 2]; 4], c: Vec4) {
        self.quad_colors(p, n, uv, [c; 4]);
    }

    /// A quad with a color per vertex (baked darkening).
    pub fn quad_colors(&mut self, p: [Vec3; 4], n: Vec3, uv: [[f32; 2]; 4], c: [Vec4; 4]) {
        let base = self.m.positions.len() as u32;
        for i in 0..4 {
            self.vertex(p[i], n, uv[i], c[i]);
        }
        self.m
            .indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    /// A quad whose four vertices sample the middle of palette cell `cell`.
    pub fn quad_cell(&mut self, p: [Vec3; 4], n: Vec3, cell: u32, c: Vec4) {
        let uv = cell_uv(cell);
        self.quad(p, n, [uv; 4], c);
    }

    /// An axis-aligned box with every face in palette cell `cell`.
    pub fn box_cell(&mut self, min: Vec3, max: Vec3, cell: u32, c: Vec4) {
        self.box_cells(min, max, [cell; 6], c);
    }

    /// An axis-aligned box; `cells` = +X, −X, +Y, −Y, +Z, −Z.
    pub fn box_cells(&mut self, min: Vec3, max: Vec3, cells: [u32; 6], c: Vec4) {
        let (a, b) = (min, max);
        let faces: [([Vec3; 4], Vec3); 6] = [
            (
                [
                    Vec3::new(b.x, a.y, b.z),
                    Vec3::new(b.x, a.y, a.z),
                    Vec3::new(b.x, b.y, a.z),
                    Vec3::new(b.x, b.y, b.z),
                ],
                Vec3::X,
            ),
            (
                [
                    Vec3::new(a.x, a.y, a.z),
                    Vec3::new(a.x, a.y, b.z),
                    Vec3::new(a.x, b.y, b.z),
                    Vec3::new(a.x, b.y, a.z),
                ],
                Vec3::NEG_X,
            ),
            (
                [
                    Vec3::new(a.x, b.y, b.z),
                    Vec3::new(b.x, b.y, b.z),
                    Vec3::new(b.x, b.y, a.z),
                    Vec3::new(a.x, b.y, a.z),
                ],
                Vec3::Y,
            ),
            (
                [
                    Vec3::new(a.x, a.y, a.z),
                    Vec3::new(b.x, a.y, a.z),
                    Vec3::new(b.x, a.y, b.z),
                    Vec3::new(a.x, a.y, b.z),
                ],
                Vec3::NEG_Y,
            ),
            (
                [
                    Vec3::new(a.x, a.y, b.z),
                    Vec3::new(b.x, a.y, b.z),
                    Vec3::new(b.x, b.y, b.z),
                    Vec3::new(a.x, b.y, b.z),
                ],
                Vec3::Z,
            ),
            (
                [
                    Vec3::new(b.x, a.y, a.z),
                    Vec3::new(a.x, a.y, a.z),
                    Vec3::new(a.x, b.y, a.z),
                    Vec3::new(b.x, b.y, a.z),
                ],
                Vec3::NEG_Z,
            ),
        ];
        for (i, (p, n)) in faces.into_iter().enumerate() {
            self.quad_cell(p, n, cells[i], c);
        }
    }

    /// A cylinder from `base` along `axis` (unit) of `length`, smooth sides and flat caps.
    #[allow(clippy::too_many_arguments)]
    pub fn cylinder(&mut self, base: Vec3, axis: Vec3, radius: f32, length: f32, segments: u32, cell: u32, c: Vec4) {
        let helper = if axis.y.abs() > 0.9 { Vec3::X } else { Vec3::Y };
        let u = axis.cross(helper).normalize();
        let v = axis.cross(u);
        let uv = cell_uv(cell);
        let top = base + axis * length;
        // sides
        let start = self.m.positions.len() as u32;
        for i in 0..=segments {
            let a = std::f32::consts::TAU * i as f32 / segments as f32;
            let n = u * a.cos() + v * a.sin();
            self.vertex(base + n * radius, n, uv, c);
            self.vertex(top + n * radius, n, uv, c);
        }
        for i in 0..segments {
            let k = start + i * 2;
            self.m
                .indices
                .extend_from_slice(&[k, k + 2, k + 1, k + 1, k + 2, k + 3]);
        }
        // caps
        for (center, n, flip) in [(top, axis, false), (base, -axis, true)] {
            let c0 = self.m.positions.len() as u32;
            self.vertex(center, n, uv, c);
            for i in 0..=segments {
                let a = std::f32::consts::TAU * i as f32 / segments as f32;
                self.vertex(center + (u * a.cos() + v * a.sin()) * radius, n, uv, c);
            }
            for i in 0..segments {
                let (p, q) = (c0 + 1 + i, c0 + 2 + i);
                if flip {
                    self.m.indices.extend_from_slice(&[c0, q, p]);
                } else {
                    self.m.indices.extend_from_slice(&[c0, p, q]);
                }
            }
        }
    }

    /// A lofted solid along Z through `sections`; end caps are closed. Side faces use `side`,
    /// the top of segment `i` uses `top[i]`, the bottom `bottom`.
    pub fn loft(&mut self, sections: &[Section], top: &[u32], side: u32, bottom: u32, c: Vec4) {
        let corner = |s: &Section, right: bool, upper: bool| {
            let x = if upper { s.half_top } else { s.half_bottom } * if right { 1.0 } else { -1.0 };
            Vec3::new(x, if upper { s.y_top } else { s.y_bottom }, s.z)
        };
        for i in 0..sections.len() - 1 {
            let (a, b) = (&sections[i], &sections[i + 1]);
            // top (normal roughly +Y; smooth enough with flat shading)
            let p = [
                corner(a, false, true),
                corner(a, true, true),
                corner(b, true, true),
                corner(b, false, true),
            ];
            // z grows toward +Z: points ordered so the normal faces up: a (back) → b (front)
            let n = (p[1] - p[0]).cross(p[3] - p[0]).normalize_or_zero();
            let (p, n) = if n.y >= 0.0 {
                (p, n)
            } else {
                ([p[3], p[2], p[1], p[0]], -n)
            };
            self.quad_cell(p, n, top[i.min(top.len() - 1)], c);
            // bottom
            let q = [
                corner(a, true, false),
                corner(a, false, false),
                corner(b, false, false),
                corner(b, true, false),
            ];
            let n = (q[1] - q[0]).cross(q[3] - q[0]).normalize_or_zero();
            let (q, n) = if n.y <= 0.0 {
                (q, n)
            } else {
                ([q[3], q[2], q[1], q[0]], -n)
            };
            self.quad_cell(q, n, bottom, c);
            // sides
            for right in [true, false] {
                let p = [
                    corner(a, right, false),
                    corner(b, right, false),
                    corner(b, right, true),
                    corner(a, right, true),
                ];
                let n = (p[1] - p[0]).cross(p[3] - p[0]).normalize_or_zero();
                let want = if right { Vec3::X } else { Vec3::NEG_X };
                let (p, n) = if n.dot(want) >= 0.0 {
                    (p, n)
                } else {
                    ([p[3], p[2], p[1], p[0]], -n)
                };
                self.quad_cell(p, n, side, c);
            }
        }
        // caps
        for (s, n) in [(&sections[0], Vec3::NEG_Z), (&sections[sections.len() - 1], Vec3::Z)] {
            let p = [
                corner(s, true, false),
                corner(s, false, false),
                corner(s, false, true),
                corner(s, true, true),
            ];
            let norm = (p[1] - p[0]).cross(p[3] - p[0]).normalize_or_zero();
            let (p, nn) = if norm.dot(n) >= 0.0 {
                (p, n)
            } else {
                ([p[3], p[2], p[1], p[0]], n)
            };
            self.quad_cell(p, nn, side, c);
        }
    }

    /// A lumpy ellipsoid (foliage, heads): smooth normals, radii jittered per vertex.
    pub fn blob(&mut self, center: Vec3, radii: Vec3, seed: u32, cell: u32, c: Vec4) {
        let (sectors, stacks) = (8u32, 5u32);
        let uv = cell_uv(cell);
        let start = self.m.positions.len() as u32;
        for i in 0..=stacks {
            let phi = std::f32::consts::PI * i as f32 / stacks as f32;
            for j in 0..=sectors {
                let theta = std::f32::consts::TAU * (j % sectors) as f32 / sectors as f32;
                let d = Vec3::new(phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin());
                let ring = if i == 0 || i == stacks { 0 } else { j % sectors };
                let jit = 0.82 + 0.3 * crate::procgen::unit(crate::procgen::hash(i, ring, seed));
                let shade = 0.75 + 0.5 * (d.y * 0.5 + 0.5);
                self.vertex(center + d * radii * jit, d, uv, c * Vec4::new(shade, shade, shade, 1.0));
            }
        }
        let row = sectors + 1;
        for i in 0..stacks {
            for j in 0..sectors {
                let a = start + i * row + j;
                let b = a + row;
                self.m.indices.extend_from_slice(&[a, a + 1, b, a + 1, b + 1, b]);
            }
        }
    }

    pub fn finish(self) -> MeshData {
        self.m
    }
}

/// A cross-section of a loft.
#[derive(Clone, Copy)]
pub struct Section {
    pub z: f32,
    pub y_bottom: f32,
    pub y_top: f32,
    pub half_bottom: f32,
    pub half_top: f32,
}

pub const fn sec(z: f32, y_bottom: f32, y_top: f32, half_bottom: f32, half_top: f32) -> Section {
    Section {
        z,
        y_bottom,
        y_top,
        half_bottom,
        half_top,
    }
}
