//! The view uniform (`View` in `shaders/common/view.wgsl`, spliced into forward.wgsl and
//! exported as `VIEW_WGSL`) and its lights. Frozen after graphics milestone F3 (spec §3.3).

use bytemuck::{Pod, Zeroable};

pub(crate) const MAX_DIR_LIGHTS: usize = 4;

pub(crate) const MAX_POINT_LIGHTS: usize = 16;

/// The `View` struct declarations shared by the forward shader (`//#VIEW`) and `VIEW_WGSL`.
pub(crate) const VIEW_STRUCTS: &str = include_str!("../shaders/common/view.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
pub(crate) struct GpuDirLight {
    pub direction: [f32; 4],
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
pub(crate) struct GpuPointLight {
    pub position_range: [f32; 4],
    pub color: [f32; 4],
}

/// Matches `View` in `shaders/common/view.wgsl` (std140-compatible, 1392 bytes; a WGSL
/// `mat3x3<f32>` is three 16-byte columns). The first 752 bytes are the original forward view,
/// the first 1312 the layout frozen at graphics milestone F3; later fields are appended
/// (`inv_proj_rot`). It holds no per-frame-varying data: a still camera with static lights
/// uploads nothing.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuView {
    pub view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    pub ambient: [f32; 4],
    /// x: directional lights, y: point lights.
    pub counts: [u32; 4],
    pub dir_lights: [GpuDirLight; MAX_DIR_LIGHTS],
    pub point_lights: [GpuPointLight; MAX_POINT_LIGHTS],
    pub view: [[f32; 4]; 4],
    /// The projection as drawn (including the viewport fit, see `ViewportRect::fitted`).
    pub proj: [[f32; 4]; 4],
    /// The `f32` inverse of `view_proj`, kept as it was: its translation column is the camera
    /// position and it is off by pixels far from the origin. Directions use `inv_proj_rot`.
    pub inv_view_proj: [[f32; 4]; 4],
    pub inv_proj: [[f32; 4]; 4],
    /// x, y, w, h in pixels of the texture the view renders into.
    pub viewport: [f32; 4],
    /// w, h, 1/w, 1/h of that texture.
    pub target_size: [f32; 4],
    /// near, far, orthographic (0/1), 0.
    pub depth: [f32; 4],
    /// x encoding id (informative; pipelines use `S3D_OUTPUT`), y tonemapper id, z flags, w 0.
    pub output: [u32; 4],
    /// x linear exposure for `Tonemapped` (1 otherwise), y exposure in stops.
    pub exposure: [f32; 4],
    pub shadow_view_proj: [[f32; 4]; 4],
    /// Depth bias (light NDC), normal bias (world), strength, edge fade start.
    pub shadow_params: [f32; 4],
    /// Enabled, directional light index, map size, taps.
    pub shadow_info: [u32; 4],
    /// Diffuse, specular, specular max mip, enabled.
    pub env_params: [f32; 4],
    pub env_rotation: [[f32; 4]; 3],
    pub sky_rotation: [[f32; 4]; 3],
    /// Intensity, enabled, 0, 0.
    pub sky_params: [f32; 4],
    /// `(proj · view rotation)⁻¹`, clip → the world-space offset from the camera, computed in
    /// `f64` without the camera's translation (see [`crate::inverse_proj_rot`]). Unlike
    /// `inv_view_proj` (the `f32` inverse of `view_proj`, kept for compatibility) it is exact
    /// at any distance from the origin; the sky derives every pixel's direction from it.
    pub inv_proj_rot: [[f32; 4]; 4],
    /// `Skybox::ground`: linear rgb color, blend half-width (0 = off).
    pub sky_ground: [f32; 4],
}
