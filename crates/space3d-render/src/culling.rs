//! CPU frustum culling primitives (the GPU path mirrors these in `shaders/cull.wgsl`).

use glam::{Mat4, Vec3, Vec4};

/// Axis-aligned bounding box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    /// Minimum corner.
    pub min: Vec3,
    /// Maximum corner.
    pub max: Vec3,
}

impl Aabb {
    /// An "inverted" box that any `grow` call replaces.
    pub const EMPTY: Aabb = Aabb {
        min: Vec3::splat(f32::INFINITY),
        max: Vec3::splat(f32::NEG_INFINITY),
    };

    /// Box from corners.
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }
    /// True if no point was ever added.
    pub fn is_empty(&self) -> bool {
        self.min.x > self.max.x
    }
    /// Expand to include `p`.
    pub fn grow(&mut self, p: Vec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }
    /// Expand to include `other`.
    pub fn union(&mut self, other: &Aabb) {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }
    /// Center point.
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }
    /// Half extents.
    pub fn half_extents(&self) -> Vec3 {
        (self.max - self.min) * 0.5
    }
    /// Box grown by `r` on every side.
    pub fn inflated(&self, r: f32) -> Aabb {
        Aabb::new(self.min - Vec3::splat(r), self.max + Vec3::splat(r))
    }
}

/// Bounding sphere.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Sphere {
    /// Center.
    pub center: Vec3,
    /// Radius.
    pub radius: f32,
}

impl Sphere {
    /// Sphere enclosing an AABB.
    pub fn from_aabb(aabb: &Aabb) -> Self {
        if aabb.is_empty() {
            return Sphere::default();
        }
        Sphere {
            center: aabb.center(),
            radius: aabb.half_extents().length(),
        }
    }
    /// Transform by an affine matrix (radius scaled by [`max_axis_scale`], so the result
    /// encloses the transformed sphere even for sheared matrices).
    pub fn transformed(&self, m: &Mat4) -> Sphere {
        Sphere {
            center: m.transform_point3(self.center),
            radius: self.radius * max_axis_scale(m),
        }
    }
}

/// Upper bound of the largest factor by which the upper 3x3 of `m` stretches a vector (its
/// spectral norm): a conservative scale for bounding-sphere radii.
///
/// The longest column is only enough for rotation × scale matrices. Hierarchies produce shear
/// (non-uniform parent scale × rotated child), which stretches some directions more than any
/// column, so this uses the Gershgorin bound on `AᵀA`: `max_i |cᵢ|² + Σ_{j≠i} |cᵢ·cⱼ|`. It
/// equals the longest column when the columns are orthogonal. `shaders/cull.wgsl` computes the
/// same expression so CPU and GPU culling agree.
pub fn max_axis_scale(m: &Mat4) -> f32 {
    stretch_bound(m.x_axis.truncate(), m.y_axis.truncate(), m.z_axis.truncate())
}

/// [`max_axis_scale`] from the three basis columns.
pub(crate) fn stretch_bound(c0: Vec3, c1: Vec3, c2: Vec3) -> f32 {
    let d01 = c0.dot(c1).abs();
    let d02 = c0.dot(c2).abs();
    let d12 = c1.dot(c2).abs();
    (c0.length_squared() + d01 + d02)
        .max(c1.length_squared() + d01 + d12)
        .max(c2.length_squared() + d02 + d12)
        .sqrt()
}

/// Six inward-facing planes (`xyz` = normal, `w` = distance) extracted from a view-projection
/// matrix with wgpu's 0..1 clip depth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frustum {
    /// Left, right, bottom, top, near, far.
    pub planes: [Vec4; 6],
    /// Spheres too small to matter to this view: `xyz` is a point (a camera's eye) and `w` a
    /// ratio, and a sphere whose radius is under `w` × its center's distance from the point
    /// counts as outside ([`Self::intersects_sphere`], the GPU culling). `w` 0 (what
    /// [`Self::from_view_proj`] gives): no such test. Shadow views set it from
    /// `DirectionalShadow::min_caster_pixels`.
    pub small: Vec4,
}

impl Frustum {
    /// Gribb–Hartmann plane extraction.
    pub fn from_view_proj(m: &Mat4) -> Self {
        let r0 = m.row(0);
        let r1 = m.row(1);
        let r2 = m.row(2);
        let r3 = m.row(3);
        let mut planes = [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2, r3 - r2];
        for p in &mut planes {
            let len = p.truncate().length();
            if len > 0.0 {
                *p /= len;
            }
        }
        Self {
            planes,
            small: Vec4::ZERO,
        }
    }

    /// Sphere intersects or is inside the frustum, and is not too small ([`Self::small`]).
    #[inline]
    pub fn intersects_sphere(&self, s: &Sphere) -> bool {
        let c = s.center.extend(1.0);
        self.planes.iter().all(|p| p.dot(c) >= -s.radius)
            && (self.small.w <= 0.0 || s.radius >= self.small.w * s.center.distance(self.small.truncate()))
    }

    /// AABB intersects or is inside the frustum (conservative "p-vertex" test).
    pub fn intersects_aabb(&self, b: &Aabb) -> bool {
        if b.is_empty() {
            return false;
        }
        self.planes.iter().all(|p| {
            let v = Vec3::new(
                if p.x >= 0.0 { b.max.x } else { b.min.x },
                if p.y >= 0.0 { b.max.y } else { b.min.y },
                if p.z >= 0.0 { b.max.z } else { b.min.z },
            );
            p.truncate().dot(v) + p.w >= 0.0
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frustum() -> Frustum {
        let proj = glam::camera::rh::proj::directx::perspective(60f32.to_radians(), 1.0, 0.1, 100.0);
        let view = glam::camera::rh::view::look_at_mat4(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
        Frustum::from_view_proj(&(proj * view))
    }

    #[test]
    fn sphere_culling() {
        let f = frustum();
        assert!(f.intersects_sphere(&Sphere {
            center: Vec3::ZERO,
            radius: 1.0
        }));
        assert!(!f.intersects_sphere(&Sphere {
            center: Vec3::new(0.0, 0.0, 10.0),
            radius: 1.0
        }));
        assert!(!f.intersects_sphere(&Sphere {
            center: Vec3::new(100.0, 0.0, 0.0),
            radius: 1.0
        }));
        assert!(!f.intersects_sphere(&Sphere {
            center: Vec3::new(0.0, 0.0, -200.0),
            radius: 1.0
        }));
        // Straddling the left plane still counts.
        assert!(f.intersects_sphere(&Sphere {
            center: Vec3::new(-3.4, 0.0, 0.0),
            radius: 1.0
        }));
        // Too small from a point: radius under 0.1 × the distance from (0, 0, 5).
        let small = Frustum {
            small: Vec3::new(0.0, 0.0, 5.0).extend(0.1),
            ..f
        };
        let at = |z: f32, radius: f32| Sphere {
            center: Vec3::new(0.0, 0.0, z),
            radius,
        };
        assert!(small.intersects_sphere(&at(0.0, 0.6)) && !small.intersects_sphere(&at(0.0, 0.4)));
        assert!(small.intersects_sphere(&at(-5.0, 1.1)) && !small.intersects_sphere(&at(-5.0, 0.9)));
        assert!(f.intersects_sphere(&at(-5.0, 0.01)), "no test without a ratio");
    }

    #[test]
    fn aabb_culling() {
        let f = frustum();
        assert!(f.intersects_aabb(&Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0))));
        assert!(!f.intersects_aabb(&Aabb::new(Vec3::new(50.0, 0.0, 0.0), Vec3::new(51.0, 1.0, 1.0))));
        assert!(!f.intersects_aabb(&Aabb::EMPTY));
    }

    /// Largest stretch of the 3x3 part, by brute force over directions.
    fn spectral_norm(m: &Mat4) -> f32 {
        let mut best = 0.0f32;
        for i in 0..400 {
            for j in 0..200 {
                let (th, ph) = (
                    i as f32 / 400.0 * std::f32::consts::TAU,
                    j as f32 / 200.0 * std::f32::consts::PI,
                );
                let v = Vec3::new(ph.sin() * th.cos(), ph.sin() * th.sin(), ph.cos());
                best = best.max(m.transform_vector3(v).length());
            }
        }
        best
    }

    #[test]
    fn sheared_matrices_get_a_conservative_radius() {
        // Parent scale (1, 4, 1) × child rotated 45° about Z: columns are 2.92 long but the
        // matrix maps (1, 1, 0)/√2 to (0, 4, 0).
        let m = Mat4::from_scale(Vec3::new(1.0, 4.0, 1.0)) * Mat4::from_rotation_z(std::f32::consts::FRAC_PI_4);
        assert!((m.x_axis.truncate().length() - 2.9155).abs() < 1e-3);
        assert!((max_axis_scale(&m) - 4.0).abs() < 1e-4);
        // The quad from the regression scene: 3.75 below the ortho frustum's bottom plane, its
        // tip reaches 0.25 into the view.
        let proj = glam::camera::rh::proj::directx::orthographic(-5.0, 5.0, -5.0, 5.0, 0.1, 100.0);
        let view = glam::camera::rh::view::look_at_mat4(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
        let f = Frustum::from_view_proj(&(proj * view));
        let world = Mat4::from_translation(Vec3::new(0.0, -8.75, 0.0)) * m;
        let local = Sphere {
            center: Vec3::ZERO,
            radius: 1.01,
        };
        assert!(f.intersects_sphere(&local.transformed(&world)));
        // Always an upper bound, and exact for rotation × scale.
        let mut seed = 7u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1u32 << 24) as f32 * 4.0 - 2.0
        };
        for _ in 0..50 {
            let a = Mat4::from_cols(
                glam::Vec4::new(rnd(), rnd(), rnd(), 0.0),
                glam::Vec4::new(rnd(), rnd(), rnd(), 0.0),
                glam::Vec4::new(rnd(), rnd(), rnd(), 0.0),
                glam::Vec4::W,
            );
            assert!(max_axis_scale(&a) >= spectral_norm(&a) * 0.9999, "{a:?}");
            let trs = Mat4::from_scale_rotation_translation(
                Vec3::new(rnd(), rnd(), rnd()),
                glam::Quat::from_euler(glam::EulerRot::XYZ, rnd(), rnd(), rnd()),
                Vec3::ZERO,
            );
            assert!((max_axis_scale(&trs) - spectral_norm(&trs)).abs() < 1e-2 * max_axis_scale(&trs));
        }
    }

    #[test]
    fn transformed_sphere_scales() {
        let m = Mat4::from_scale_rotation_translation(Vec3::new(1.0, 3.0, 2.0), glam::Quat::IDENTITY, Vec3::X);
        let s = Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        }
        .transformed(&m);
        assert_eq!(s.center, Vec3::X);
        assert!((s.radius - 3.0).abs() < 1e-6);
    }
}
