// Space3d view uniform (`VIEW_WGSL` appends `@group(0) @binding(0) var<uniform> view: View;`).
// Exactly the forward shader's `View` (std140-compatible, 1392 bytes; the first 1312 are the
// layout frozen at graphics milestone F3, later members are appended). Node and overlay shaders
// must not redefine `DirLight`, `PointLight`, `View` or `view`.

struct DirLight {
    direction: vec4<f32>,   // xyz: direction the light travels (normalized)
    color: vec4<f32>,       // rgb: color * intensity
}

struct PointLight {
    position_range: vec4<f32>,
    color: vec4<f32>,
}

struct View {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    ambient: vec4<f32>,
    counts: vec4<u32>,          // x: dir lights, y: point lights
    dir_lights: array<DirLight, 4>,
    point_lights: array<PointLight, 16>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    viewport: vec4<f32>,        // x, y, w, h in pixels of the texture this view renders into
    target_size: vec4<f32>,     // w, h, 1/w, 1/h of that texture
    depth: vec4<f32>,           // near, far, orthographic (0/1), 0
    output: vec4<u32>,          // x encoding id, y tonemapper id, z flags, w 0
    exposure: vec4<f32>,        // x linear exposure for Tonemapped (1 otherwise), y ev
    shadow_view_proj: mat4x4<f32>,
    shadow_params: vec4<f32>,   // depth bias (light NDC), normal bias (world), strength, edge fade start
    shadow_info: vec4<u32>,     // enabled, dir-light index, map size, taps
    env_params: vec4<f32>,      // diffuse, specular, specular max mip, enabled
    env_rotation: mat3x3<f32>,
    sky_rotation: mat3x3<f32>,
    sky_params: vec4<f32>,      // intensity, enabled, 0, 0
    // (proj * view rotation)^-1, computed in f64 on the CPU: clip -> world-space offset from the
    // camera (no camera position in it, so it is exact at any distance from the origin).
    // `unproject(ndc, 1) - unproject(ndc, 0)` (each divided by w) is the pixel's world direction.
    // `inv_view_proj` is the f32 inverse of `view_proj`: it carries the camera position and is
    // badly conditioned far from the origin; prefer this for directions.
    inv_proj_rot: mat4x4<f32>,
    sky_ground: vec4<f32>,      // Skybox ground color (linear rgb), blend half-width in view-direction Y (0 = off)
}
