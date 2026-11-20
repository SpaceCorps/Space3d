//! Directional shadows (graphics spec §1.12, §2.2.9; materials milestone PM5): the casting light,
//! its shadow map, the map's fit and the frame's shadow uniforms.
//!
//! * **One light casts:** among the `DirectionalLight`s with a [`DirectionalShadow`] (and a
//!   `GlobalTransform`), the one with the lowest `Entity::to_bits` (a warning is logged once when
//!   several ask).
//! * **Step 4** (`Lighting::prepare`): its `map_size`² `Depth32Float` map is created, or created
//!   again when the size changes, and dropped once no light has cast for `ENV_IDLE_FRAMES` frames.
//! * **Step 5** (`Lighting::depth_views`): the map is fitted ([`fit`]) and requested as one depth
//!   view. The frame culls and draws it like a camera (its own frustum; one more frustum of the
//!   GPU culling dispatches) in a depth-only pass before the camera views.
//! * **Step 8** (`Lighting::fill_view`): every camera view gets the light matrix, the biases, the
//!   strength and the map size, and the casting light moves to `dir_lights[0]`.
//!
//! **Fitting** (CPU, pure f32, deterministic):
//! 1. Light basis: looking along the light from the world origin, up +Y (+Z for lights within
//!    about 8° of vertical), so the map's texel grid is anchored in the world.
//! 2. Points: `ShadowFit::Camera` takes the 8 corners of the camera frustum between its near
//!    plane and `min(far, max_distance)`. With `receivers`, that slice is intersected with the
//!    box exactly: its 6 faces are clipped against the box (Sutherland–Hodgman), and the box
//!    corners inside the slice are added. `ShadowFit::Bounds` takes the box's 8 corners.
//! 3. In light space: the center of the points' bounds, and the radius around it rounded up to
//!    steps of `2^⌊log2 r⌋ / 8`, so the map's world size changes only in steps while zooming and
//!    not at all while panning.
//! 4. Texel snap: the texel is `2r / (size − 2)`; the map spans `size` texels around the center
//!    snapped to the texel grid (one texel of margin), so panning never shimmers.
//! 5. Depth: from `caster_depth` in front of the nearest point (casters outside the view toward
//!    the light) to just behind the farthest.
//!
//! The biases: `depth_bias` (world units along the light) becomes light depth; `normal_bias`
//! (texels) moves the receiver along its geometric normal, more at grazing angles; `slope_bias`
//! is the raster slope-scaled bias of the depth pass.

use super::api::{DirectionalShadow, ShadowFilter, ShadowFit};
use crate::components::DirectionalLight;
use crate::culling::Aabb;
use crate::node::{ViewInfo, ViewRole};
use crate::pipeline::{GpuDirLight, GpuView, MAX_DIR_LIGHTS};
use glam::{Mat4, Vec2, Vec3, Vec4};
use space3d_core::{Entity, GlobalTransform, World};

/// Shadows fade out over the outer 10 % of the map (`GpuView::shadow_params.w`).
pub(crate) const FADE_START: f32 = 0.9;

/// Smallest shadow map edge.
const MIN_MAP_SIZE: u32 = 16;

/// Texture lookups of a filter (`S3D_SHADOW_TAPS`; each is a bilinear comparison, 2×2 texels).
pub(crate) fn filter_taps(filter: ShadowFilter) -> u32 {
    match filter {
        ShadowFilter::Hard => 1,
        ShadowFilter::Pcf4 => 4,
        ShadowFilter::Pcf9 => 9,
        _ => 16,
    }
}

/// How far a filter reaches from the sample point, in texels (half its footprint: 2×2, 3×3,
/// 5×5, 7×7 texels). The raster slope bias scales with it: a wider kernel compares against
/// texels farther along a sloped receiver.
pub(crate) fn filter_reach(filter: ShadowFilter) -> f32 {
    match filter {
        ShadowFilter::Hard => 1.0,
        ShadowFilter::Pcf4 => 1.5,
        ShadowFilter::Pcf9 => 2.5,
        _ => 3.5,
    }
}

/// The map edge for a requested `map_size`: a power of two in [16, the device's limit].
pub(crate) fn map_size(requested: u32, max_texture: u32) -> u32 {
    // The largest power of two within the limit.
    let max = 1u32 << (31 - max_texture.max(MIN_MAP_SIZE).leading_zeros());
    requested
        .max(MIN_MAP_SIZE)
        .checked_next_power_of_two()
        .unwrap_or(max)
        .min(max)
}

/// The directional light the frame's shadow comes from.
#[derive(Clone, Copy)]
pub(crate) struct CastingLight {
    pub entity: Entity,
    /// Direction the light travels (world, unit).
    pub direction: Vec3,
    /// The light as `SceneLights` uploads it (to find it among the view's lights).
    pub gpu: GpuDirLight,
    pub shadow: DirectionalShadow,
}

/// The light that casts in `world` (the lowest entity bits), and how many asked to.
pub(crate) fn casting_light(world: &World) -> (Option<CastingLight>, u32) {
    let mut best: Option<CastingLight> = None;
    let mut n = 0;
    let mut query = world.query::<(Entity, &GlobalTransform, &DirectionalLight, &DirectionalShadow)>();
    for (entity, gt, light, shadow) in query.iter() {
        n += 1;
        if best.is_some_and(|b| b.entity.to_bits() <= entity.to_bits()) {
            continue;
        }
        // As `SceneLights::collect` computes it, bit for bit.
        let direction = (-gt.0.z_axis.truncate()).normalize_or(Vec3::NEG_Y);
        best = Some(CastingLight {
            entity,
            direction,
            gpu: GpuDirLight {
                direction: direction.extend(0.0).to_array(),
                color: (light.color * light.intensity).extend(1.0).to_array(),
            },
            shadow: *shadow,
        });
    }
    (best, n)
}

/// The filter of the light that casts in `world` (the forward pipelines' `S3D_SHADOW_TAPS`).
pub(crate) fn casting_filter(world: &World) -> Option<ShadowFilter> {
    casting_light(world).0.map(|l| l.shadow.filter)
}

/// A fitted shadow map.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Fitted {
    /// World → light clip space (orthographic, depth 0..1).
    pub view_proj: Mat4,
    /// World size of a texel.
    pub texel: f32,
    /// World length of the depth range.
    pub depth_range: f32,
}

/// The light's view: looking along `direction` from the world origin.
pub(crate) fn light_view(direction: Vec3) -> Mat4 {
    let up = if direction.dot(Vec3::Y).abs() > 0.99 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    glam::camera::rh::view::look_to_mat4(Vec3::ZERO, direction, up)
}

/// A fixed-capacity point set (no allocation).
pub(crate) struct Points {
    p: [Vec3; 96],
    n: usize,
}

impl Points {
    pub(crate) fn new() -> Self {
        Self {
            p: [Vec3::ZERO; 96],
            n: 0,
        }
    }

    fn push(&mut self, p: Vec3) {
        if self.n < self.p.len() {
            self.p[self.n] = p;
            self.n += 1;
        }
    }

    pub(crate) fn as_slice(&self) -> &[Vec3] {
        &self.p[..self.n]
    }
}

/// NDC depth of view-space distance `d` in front of `view`'s camera.
fn ndc_depth(view: &ViewInfo, d: f32) -> f32 {
    let c = view.proj * Vec4::new(0.0, 0.0, -d, 1.0);
    c.z / c.w
}

/// The 8 corners of `view`'s frustum between its near plane and `min(far, max_distance)`: near
/// (−−, +−, ++, −+ in NDC), then far.
pub(crate) fn slice_corners(view: &ViewInfo, max_distance: f32) -> Option<[Vec3; 8]> {
    let near = view.near;
    let far = view.far.min(max_distance);
    // (NaN distances fail too.)
    if far.partial_cmp(&near) != Some(std::cmp::Ordering::Greater) {
        return None;
    }
    // Offsets from the camera through the f64 inverse of the projection and of the view's
    // rotation, plus the camera's own position: the f32 inverse of `view_proj` carries the
    // position in its translation column and is off by units far from the origin.
    let inv = crate::unproject::inverse_proj_rot(view.view, view.proj);
    let (zn, zf) = (ndc_depth(view, near), ndc_depth(view, far));
    let xy = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
    let mut out = [Vec3::ZERO; 8];
    for (i, z) in [zn, zf].into_iter().enumerate() {
        for (j, &(x, y)) in xy.iter().enumerate() {
            out[i * 4 + j] = view.camera_pos + inv.project_point3(Vec3::new(x, y, z));
        }
    }
    out.iter().all(|p| p.is_finite()).then_some(out)
}

/// Clip the convex polygon `poly` to the side of the plane `p[axis] = value` where
/// `p[axis] ≥ value` (`above`) or `≤ value`.
fn clip_polygon(poly: &[Vec3], axis: usize, value: f32, above: bool, out: &mut [Vec3; 16]) -> usize {
    let d = |p: Vec3| if above { p[axis] - value } else { value - p[axis] };
    let mut n = 0;
    for i in 0..poly.len() {
        let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
        let (da, db) = (d(a), d(b));
        if da >= 0.0 && n < out.len() {
            out[n] = a;
            n += 1;
        }
        if (da >= 0.0) != (db >= 0.0) && n < out.len() {
            let t = da / (da - db);
            let mut q = a + (b - a) * t;
            // Exactly on the plane (no drift outside the box).
            q[axis] = value;
            out[n] = q;
            n += 1;
        }
    }
    n
}

/// The intersection of the frustum slice `corners` with `bounds`, as the points spanning it:
/// the slice's faces clipped to the box, and the box corners inside the slice (`inside`).
fn clip_slice(corners: &[Vec3; 8], bounds: &Aabb, inside: impl Fn(Vec3) -> bool, points: &mut Points) {
    const FACES: [[usize; 4]; 6] = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 3, 7, 4],
        [1, 5, 6, 2],
        [0, 4, 5, 1],
        [3, 2, 6, 7],
    ];
    for face in FACES {
        let mut a = [Vec3::ZERO; 16];
        let mut b = [Vec3::ZERO; 16];
        for (k, &i) in face.iter().enumerate() {
            a[k] = corners[i];
        }
        let mut n = 4;
        for axis in 0..3 {
            n = clip_polygon(&a[..n], axis, bounds.min[axis], true, &mut b);
            n = clip_polygon(&b[..n], axis, bounds.max[axis], false, &mut a);
        }
        for &p in &a[..n] {
            points.push(p);
        }
    }
    for c in box_corners(bounds) {
        // (`top_down` boxes are unbounded in x and z: those corners are never inside.)
        if c.abs().max_element() < 1e30 && inside(c) {
            points.push(c);
        }
    }
}

fn box_corners(b: &Aabb) -> [Vec3; 8] {
    std::array::from_fn(|i| {
        Vec3::new(
            if i & 1 == 0 { b.min.x } else { b.max.x },
            if i & 2 == 0 { b.min.y } else { b.max.y },
            if i & 4 == 0 { b.min.z } else { b.max.z },
        )
    })
}

/// The camera a `ShadowFit::Camera` follows: `camera`, or the first primary scene view (by
/// camera order), or the first scene view when no camera draws to the primary target.
fn fit_camera(cameras: &[ViewInfo], camera: Option<Entity>) -> Option<&ViewInfo> {
    match camera {
        Some(c) => cameras.iter().find(|v| v.camera == c && v.role == ViewRole::Scene),
        None => cameras
            .iter()
            .find(|v| v.primary && v.role == ViewRole::Scene)
            .or_else(|| cameras.iter().find(|v| v.role == ViewRole::Scene)),
    }
}

/// `Frustum::small` of the shadow view of `shadow` (`DirectionalShadow::min_caster_pixels`):
/// the eye of the camera the map follows and the radius ÷ distance that is that many pixels
/// on its screen. Zero (no test) without a size, such a camera, or with an orthographic one.
pub(crate) fn small_casters(shadow: &DirectionalShadow, cameras: &[ViewInfo]) -> Vec4 {
    let pixels = shadow.min_caster_pixels;
    if pixels.is_nan() || pixels <= 0.0 {
        return Vec4::ZERO;
    }
    let camera = match shadow.fit {
        ShadowFit::Camera { camera, .. } => camera,
        ShadowFit::Bounds(_) => None,
    };
    let Some(view) = fit_camera(cameras, camera).filter(|v| !v.orthographic) else {
        return Vec4::ZERO;
    };
    // Pixels of a sphere = radius × this ÷ distance (`lod::select_lods`).
    let scale = view.viewport[3] as f32 * 0.5 * view.proj.y_axis.y.abs();
    if scale <= 0.0 || !scale.is_finite() {
        return Vec4::ZERO;
    }
    view.camera_pos.extend(pixels / scale)
}

/// The points a shadow map must cover (graphics spec §2.2.9 `ShadowFit`); `false` when there
/// is nothing to fit (no such camera, an empty box).
pub(crate) fn fit_points(fit: &ShadowFit, cameras: &[ViewInfo], points: &mut Points) -> bool {
    match *fit {
        ShadowFit::Bounds(b) => {
            if b.is_empty() || !(b.min.is_finite() && b.max.is_finite()) {
                return false;
            }
            for c in box_corners(&b) {
                points.push(c);
            }
            true
        }
        ShadowFit::Camera {
            camera,
            max_distance,
            receivers,
        } => {
            let Some(view) = fit_camera(cameras, camera) else {
                return false;
            };
            let Some(corners) = slice_corners(view, max_distance) else {
                return false;
            };
            if let Some(b) = receivers.filter(|b| !b.is_empty()) {
                let (zn, zf) = (ndc_depth(view, view.near), ndc_depth(view, view.far.min(max_distance)));
                let inside = |p: Vec3| {
                    let c = view.view_proj * p.extend(1.0);
                    let z = c.z / c.w;
                    c.w > 0.0 && c.x.abs() <= c.w && c.y.abs() <= c.w && z >= zn.min(zf) && z <= zn.max(zf)
                };
                clip_slice(&corners, &b, inside, points);
            }
            // No receiver in view (or no box): the whole slice.
            if points.as_slice().is_empty() {
                for c in corners {
                    points.push(c);
                }
            }
            true
        }
    }
}

/// Fit a `size`² map looking along `direction` around `points` (steps 3–5 of the module docs).
pub(crate) fn fit_light(direction: Vec3, points: &[Vec3], size: u32, caster_depth: f32) -> Option<Fitted> {
    if points.is_empty() {
        return None;
    }
    let view = light_view(direction);
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for &p in points {
        let q = view.transform_point3(p);
        lo = lo.min(q);
        hi = hi.max(q);
    }
    if !(lo.is_finite() && hi.is_finite()) {
        return None;
    }
    let center = (lo.truncate() + hi.truncate()) * 0.5;
    let mut r = points
        .iter()
        .map(|&p| (view.transform_point3(p).truncate() - center).length())
        .fold(0.0f32, f32::max)
        .max(1e-3);
    let step = r.log2().floor().exp2() / 8.0;
    r = (r / step).ceil() * step;
    let size = size.max(4) as f32;
    let texel = 2.0 * r / (size - 2.0);
    let half = texel * size * 0.5;
    let center: Vec2 = (center / texel).round() * texel;
    let near = -hi.z - caster_depth.max(0.0);
    let far = -lo.z + (hi.z - lo.z).max(1.0) * 1e-3;
    let proj = glam::camera::rh::proj::directx::orthographic(
        center.x - half,
        center.x + half,
        center.y - half,
        center.y + half,
        near,
        far,
    );
    Some(Fitted {
        view_proj: proj * view,
        texel,
        depth_range: far - near,
    })
}

/// Fit `light`'s `size`² map for this frame's `cameras`.
pub(crate) fn fit(light: &CastingLight, size: u32, cameras: &[ViewInfo]) -> Option<Fitted> {
    let mut points = Points::new();
    if !fit_points(&light.shadow.fit, cameras, &mut points) {
        return None;
    }
    fit_light(light.direction, points.as_slice(), size, light.shadow.caster_depth)
}

/// The shadow map of the casting light.
pub(super) struct ShadowMap {
    pub view: wgpu::TextureView,
    pub size: u32,
    /// Resource id (view bind-group keys).
    pub id: u64,
    pub last_used: u64,
}

impl ShadowMap {
    pub(super) fn new(device: &wgpu::Device, size: u32, id: u64, frame: u64) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("space3d shadow map"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: crate::pipeline::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        Self {
            view: texture.create_view(&Default::default()),
            size,
            id,
            last_used: frame,
        }
    }
}

/// This frame's shadow, once fitted (step 5): what every camera view gets.
#[derive(Clone, Copy)]
pub(super) struct ActiveShadow {
    pub view_proj: Mat4,
    /// Depth bias (light depth), normal bias (world), strength, fade start.
    pub params: [f32; 4],
    /// Enabled, light index (0), map size, taps.
    pub info: [u32; 4],
    /// The casting light (moved to `dir_lights[0]`).
    pub light: GpuDirLight,
    /// Its map's resource id.
    pub map: u64,
}

impl ActiveShadow {
    pub(super) fn new(light: &CastingLight, fitted: &Fitted, map: &ShadowMap) -> Self {
        let s = &light.shadow;
        Self {
            view_proj: fitted.view_proj,
            params: [
                s.depth_bias.max(0.0) / fitted.depth_range.max(1e-6),
                s.normal_bias.max(0.0) * fitted.texel,
                s.strength.clamp(0.0, 1.0),
                FADE_START,
            ],
            info: [1, 0, map.size, filter_taps(s.filter)],
            light: light.gpu,
            map: map.id,
        }
    }

    /// Write the shadow fields of a camera's `GpuView` and move the casting light to
    /// `dir_lights[0]` (keeping the others' order; the last one drops out if the caster was not
    /// among the view's lights and they were full).
    pub(super) fn fill(&self, view: &mut GpuView) {
        view.shadow_view_proj = self.view_proj.to_cols_array_2d();
        view.shadow_params = self.params;
        view.shadow_info = self.info;
        let n = (view.counts[0] as usize).min(MAX_DIR_LIGHTS);
        let same = |l: &GpuDirLight| l.direction == self.light.direction && l.color == self.light.color;
        let end = match view.dir_lights[..n].iter().position(same) {
            Some(i) => i,
            None if n < MAX_DIR_LIGHTS => {
                view.counts[0] = n as u32 + 1;
                n
            }
            None => MAX_DIR_LIGHTS - 1,
        };
        view.dir_lights.copy_within(0..end, 1);
        view.dir_lights[0] = self.light;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::culling::Frustum;
    use crate::output::{OutputEncoding, Tonemapper};

    /// A camera view looking from `eye` toward `target` (up +Y, or −Z when looking straight
    /// down).
    fn camera(eye: Vec3, target: Vec3, fov_deg: f32, aspect: f32, near: f32, far: f32) -> ViewInfo {
        let dir = (target - eye).normalize();
        let up = if dir.dot(Vec3::Y).abs() > 0.99 {
            Vec3::NEG_Z
        } else {
            Vec3::Y
        };
        let view = glam::camera::rh::view::look_at_mat4(eye, target, up);
        let proj = glam::camera::rh::proj::directx::perspective(fov_deg.to_radians(), aspect, near, far);
        let view_proj = proj * view;
        ViewInfo {
            camera: Entity::DANGLING,
            order: 0,
            target: Default::default(),
            primary: true,
            mode: crate::node::FrameMode::Direct,
            role: ViewRole::Scene,
            view,
            proj,
            view_proj,
            camera_pos: eye,
            near,
            far,
            orthographic: false,
            frustum: Frustum::from_view_proj(&view_proj),
            viewport: [0, 0, 1280, 720],
            scene_size: (1280, 720),
            target_size: (1280, 720),
            render_scale: 1.0,
            samples: 1,
            color_format: wgpu::TextureFormat::Rgba8UnormSrgb,
            output: OutputEncoding::Linear,
            exposure: 1.0,
            tonemapper: Tonemapper::None,
            view_offset: 0,
        }
    }

    /// `min_caster_pixels` becomes the camera's eye and a radius ÷ distance: 10 px on a 720-high
    /// view of 60° is 10 ÷ (360 ÷ tan 30°); nothing without a size or under an orthographic
    /// camera.
    #[test]
    fn small_casters_are_sized_on_the_fitted_camera() {
        let eye = Vec3::new(0.0, 50.0, 0.0);
        let cameras = [camera(eye, Vec3::ZERO, 60.0, 16.0 / 9.0, 0.1, 500.0)];
        let shadow = DirectionalShadow::new().with_min_caster_pixels(10.0);
        let small = small_casters(&shadow, &cameras);
        assert_eq!(small.truncate(), eye);
        let ratio = 10.0 / (360.0 / 30f32.to_radians().tan());
        assert!((small.w - ratio).abs() < 1e-6, "{} vs {ratio}", small.w);
        assert_eq!(small_casters(&DirectionalShadow::new(), &cameras), Vec4::ZERO);
        assert_eq!(small_casters(&shadow, &[]), Vec4::ZERO);
        let flat = ViewInfo {
            orthographic: true,
            ..cameras[0]
        };
        assert_eq!(small_casters(&shadow, &[flat]), Vec4::ZERO);
    }

    fn light(direction: Vec3, fit: ShadowFit) -> CastingLight {
        let direction = direction.normalize();
        CastingLight {
            entity: Entity::DANGLING,
            direction,
            gpu: GpuDirLight {
                direction: direction.extend(0.0).to_array(),
                color: [1.0; 4],
            },
            shadow: DirectionalShadow::new().with_fit(fit),
        }
    }

    /// Map coordinates of a world point: x, y in texels (y down), depth 0..1.
    fn map_coords(f: &Fitted, size: u32, p: Vec3) -> Vec3 {
        let ndc = f.view_proj.project_point3(p);
        Vec3::new(
            (ndc.x * 0.5 + 0.5) * size as f32,
            (0.5 - ndc.y * 0.5) * size as f32,
            ndc.z,
        )
    }

    /// Points on the receiver plane y = 0 the camera sees (an 11 × 11 grid over its footprint).
    fn visible_ground(view: &ViewInfo) -> Vec<Vec3> {
        let inv = view.view_proj.inverse();
        let mut out = Vec::new();
        for j in 0..=10 {
            for i in 0..=10 {
                let ndc = Vec2::new(i as f32 / 5.0 - 1.0, j as f32 / 5.0 - 1.0);
                let a = inv.project_point3(ndc.extend(0.0));
                let b = inv.project_point3(ndc.extend(0.5));
                let t = a.y / (a.y - b.y);
                if t > 0.0 {
                    out.push(a + (b - a) * t);
                }
            }
        }
        out
    }

    /// The frustum slice's corners are the same offsets from the camera wherever the camera is:
    /// built from the f64 inverse projection and view rotation (not the f32 inverse of
    /// `view_proj`, off by units at 85 000).
    #[test]
    fn slice_corners_do_not_depend_on_the_camera_position() {
        let base = camera(
            Vec3::new(3.0, 40.0, 7.0),
            Vec3::new(2.0, 39.0, 6.0),
            50.0,
            16.0 / 9.0,
            0.5,
            400.0,
        );
        let rotation = glam::Quat::from_mat3(&glam::Mat3::from_mat4(base.view).transpose());
        let at = |eye: Vec3| {
            let mut v = base;
            v.view = Mat4::from_rotation_translation(rotation, eye).inverse();
            v.view_proj = v.proj * v.view;
            v.camera_pos = eye;
            slice_corners(&v, 300.0).expect("corners")
        };
        let eye0 = Vec3::new(3.0, 40.0, 7.0);
        let reference = at(eye0);
        for eye in [
            Vec3::new(85_000.0, 0.0, 85_000.0),
            Vec3::new(1_200.0, -430_000.0, 900.0),
        ] {
            let far = at(eye);
            for (a, b) in far.iter().zip(&reference) {
                // (The corners are stored in f32: 0.004 of rounding at 430 000.)
                assert!(
                    ((*a - eye) - (*b - eye0)).abs().max_element() < 0.05,
                    "{eye}: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn map_sizes_are_powers_of_two_within_the_device() {
        assert_eq!(map_size(2048, 8192), 2048);
        assert_eq!(map_size(1500, 8192), 2048);
        assert_eq!(map_size(0, 8192), 16);
        assert_eq!(map_size(16384, 8192), 8192);
        assert_eq!(map_size(u32::MAX, 16384), 16384);
        assert_eq!(map_size(4096, 3000), 2048);
        assert_eq!(filter_taps(ShadowFilter::default()), 9);
    }

    /// The fitted map contains every receiver the camera sees (inside the unfaded part), its
    /// texel grid is anchored in the world, and depths are within 0..1.
    #[test]
    fn shadow_fit_texel_snapped_and_contains_receivers() {
        let size = 2048;
        for (eye, dir) in [
            (Vec3::new(3.0, 40.0, 7.0), Vec3::new(-0.3, -1.0, -0.2)),
            (Vec3::new(-120.5, 60.0, 33.3), Vec3::new(0.05, -1.0, 0.02)),
            (Vec3::new(10.0, 25.0, 30.0), Vec3::new(0.6, -0.5, 0.1)),
        ] {
            let view = camera(eye, eye + Vec3::new(0.0, -1.0, -0.35), 50.0, 16.0 / 9.0, 0.5, 400.0);
            let l = light(dir, ShadowFit::top_down(0.0, 5.0, 2000.0));
            let f = fit(&l, size, &[view]).expect("fitted");
            for p in visible_ground(&view) {
                let c = map_coords(&f, size, p);
                let edge = (c.truncate() / size as f32 * 2.0 - Vec2::ONE).abs().max_element();
                assert!(edge < FADE_START, "{p} at {c} (edge {edge})");
                assert!(c.z > 0.0 && c.z < 1.0, "depth {c}");
            }
            // The light-space origin is on a texel corner.
            let o = map_coords(&f, size, Vec3::ZERO);
            for v in [o.x, o.y] {
                assert!((v - v.round()).abs() < 2e-3 * v.abs().max(1.0), "grid offset {v}");
            }
        }
    }

    /// Panning the camera by fractions of a texel moves the map by whole texels (or not at
    /// all): a fixed world point keeps its sub-texel position.
    #[test]
    fn shadow_fit_stable_under_subtexel_pan() {
        let size = 1024;
        let l = light(Vec3::new(-0.2, -1.0, -0.3), ShadowFit::top_down(0.0, 10.0, 2000.0));
        let at = |x: f32| {
            let eye = Vec3::new(x, 50.0, 0.0);
            let view = camera(eye, eye + Vec3::new(0.0, -1.0, -0.3), 45.0, 1.6, 0.5, 300.0);
            fit(&l, size, &[view]).unwrap()
        };
        let base = at(0.0);
        let p = Vec3::new(2.0, 1.0, -3.0);
        let c0 = map_coords(&base, size, p);
        for k in 1..40 {
            let f = at(k as f32 * 0.137 * base.texel);
            assert_eq!(f.texel, base.texel, "the map size does not change while panning");
            let c = map_coords(&f, size, p);
            for (a, b) in [(c.x, c0.x), (c.y, c0.y)] {
                let d = a - b;
                assert!((d - d.round()).abs() < 2e-3, "pan {k}: sub-texel shift {d}");
            }
        }
    }

    /// With a receiver slab, the fit covers the frustum ∩ slab (the footprint on the ground),
    /// not the whole frustum: a top-down camera 100 units up with a 400-unit far plane.
    #[test]
    fn top_down_slab_clip_is_tight() {
        let eye = Vec3::new(0.0, 100.0, 0.0);
        let view = camera(eye, Vec3::ZERO, 40.0, 1.0, 1.0, 400.0);
        let down = Vec3::new(0.0, -1.0, 0.001);
        let slab = fit(&light(down, ShadowFit::top_down(0.0, 2.0, 2000.0)), 2048, &[view]).unwrap();
        let whole = fit(
            &light(
                down,
                ShadowFit::Camera {
                    camera: None,
                    max_distance: 2000.0,
                    receivers: None,
                },
            ),
            2048,
            &[view],
        )
        .unwrap();
        // Footprint half-width at y = −2: 102 · tan 20°; the radius is its half diagonal,
        // rounded up by at most one step (1/8 of the power of two below it) plus a texel.
        let half = 102.0 * 20f32.to_radians().tan();
        let r = half * std::f32::consts::SQRT_2;
        let extent = slab.texel * 2048.0 * 0.5;
        assert!(extent >= r && extent <= r * 1.26, "{extent} vs {r}");
        let whole_extent = whole.texel * 2048.0 * 0.5;
        assert!(whole_extent > 3.0 * extent, "{whole_extent} vs {extent}");
        // Casters up to `caster_depth` above the slab are in the depth range.
        assert!(
            slab.depth_range >= 500.0 + 4.0 && slab.depth_range < 510.0,
            "{}",
            slab.depth_range
        );
        // The box's x and z are ±f32::MAX: nothing overflowed.
        assert!(slab.view_proj.is_finite());
    }

    /// A fixed `Bounds` fit covers the box; a missing camera fits nothing.
    #[test]
    fn bounds_fit_and_missing_camera() {
        let b = Aabb::new(Vec3::new(-10.0, -1.0, -5.0), Vec3::new(10.0, 3.0, 5.0));
        let l = light(Vec3::new(0.3, -1.0, 0.2), ShadowFit::Bounds(b));
        let f = fit(&l, 512, &[]).unwrap();
        for c in box_corners(&b) {
            let ndc = f.view_proj.project_point3(c);
            assert!(
                ndc.x.abs() < 1.0 && ndc.y.abs() < 1.0 && ndc.z > 0.0 && ndc.z < 1.0,
                "{ndc}"
            );
        }
        let l = light(Vec3::NEG_Y, ShadowFit::default());
        assert!(fit(&l, 512, &[]).is_none());
    }

    /// The casting light moves to `dir_lights[0]`, the others keep their order; a light missing
    /// from a full list replaces the last one.
    #[test]
    fn caster_moves_to_the_first_light() {
        let dl = |i: u32| GpuDirLight {
            direction: [i as f32, 0.0, 0.0, 0.0],
            color: [1.0; 4],
        };
        let active = |i: u32| ActiveShadow {
            view_proj: Mat4::IDENTITY,
            params: [0.0; 4],
            info: [1, 0, 16, 9],
            light: dl(i),
            map: 1,
        };
        let mut view: GpuView = bytemuck::Zeroable::zeroed();
        view.counts[0] = 3;
        for i in 0..3 {
            view.dir_lights[i] = dl(i as u32);
        }
        let order = |v: &GpuView| v.dir_lights.map(|l| l.direction[0] as u32);
        active(2).fill(&mut view);
        assert_eq!((order(&view), view.counts[0]), ([2, 0, 1, 0], 3));
        active(7).fill(&mut view);
        assert_eq!((order(&view), view.counts[0]), ([7, 2, 0, 1], 4));
        active(9).fill(&mut view);
        assert_eq!((order(&view), view.counts[0]), ([9, 7, 2, 0], 4));
        assert_eq!(view.shadow_info, [1, 0, 16, 9]);
    }

    /// The PCF kernels of `forward.wgsl` (Castaño's layouts of bilinear comparisons, emulated
    /// here per axis): for every sub-texel position the texel weights are non-negative, sum to 1
    /// and are centered on the sample point, so a filtered shadow edge sits where the hard one
    /// does.
    #[test]
    fn pcf_kernels_are_centered_and_normalized() {
        // (tap offsets, tap weights, normalization) of one axis at sub-texel position s.
        type Kernel = fn(f32) -> ([f32; 4], [f32; 4], f32);
        let kernels: [(u32, Kernel); 3] = [
            (4, |s| {
                let w = [3.0 - 2.0 * s, 1.0 + 2.0 * s, 0.0, 0.0];
                ([(2.0 - s) / w[0] - 1.0, s / w[1] + 1.0, 0.0, 0.0], w, 4.0)
            }),
            (9, |s| {
                let w = [4.0 - 3.0 * s, 7.0, 1.0 + 3.0 * s, 0.0];
                (
                    [(3.0 - 2.0 * s) / w[0] - 2.0, (3.0 + s) / w[1], s / w[2] + 2.0, 0.0],
                    w,
                    12.0,
                )
            }),
            (16, |s| {
                let w = [5.0 * s - 6.0, 11.0 * s - 28.0, -(11.0 * s + 17.0), -(5.0 * s + 1.0)];
                (
                    [
                        (4.0 * s - 5.0) / w[0] - 3.0,
                        (4.0 * s - 16.0) / w[1] - 1.0,
                        -(7.0 * s + 5.0) / w[2] + 1.0,
                        -s / w[3] + 3.0,
                    ],
                    w,
                    -52.0,
                )
            }),
        ];
        for (taps, kernel) in kernels {
            for k in 0..=20 {
                let s = k as f32 / 20.0;
                let (offsets, weights, norm) = kernel(s);
                // Texel i has its center at i + 0.5 (texel units); the base is the center of
                // texel −1 and the sample point is at base + s.
                let mut w = [0.0f32; 12];
                for (o, tw) in offsets.iter().zip(weights) {
                    if tw == 0.0 {
                        continue;
                    }
                    let x = -0.5 + o;
                    let i0 = (x - 0.5).floor();
                    let frac = x - 0.5 - i0;
                    let i0 = (i0 as i32 + 6) as usize;
                    w[i0] += tw / norm * (1.0 - frac);
                    w[i0 + 1] += tw / norm * frac;
                }
                let sum: f32 = w.iter().sum();
                assert!((sum - 1.0).abs() < 1e-5, "{taps} taps, s {s}: sum {sum}");
                assert!(w.iter().all(|&x| x >= -1e-6), "{taps} taps, s {s}: {w:?}");
                let center: f32 = w.iter().enumerate().map(|(i, x)| x * (i as f32 - 6.0 + 0.5)).sum();
                assert!(
                    (center - (-0.5 + s)).abs() < 1e-5,
                    "{taps} taps, s {s}: center {center}"
                );
            }
        }
    }
}
