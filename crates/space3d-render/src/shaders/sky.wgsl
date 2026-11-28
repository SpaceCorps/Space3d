// Space3d skybox (the built-in "space3d/skybox" node, Skybox phase; graphics spec §5.2 PM3,
// design-materials §5.4). Composed after `VIEW_WGSL` (group 0 binding 0: the camera's `View`)
// and `OUTPUT_WGSL`.
//
// One triangle covering the view at the far plane (clip z = w: depth 1.0), drawn with
// `LessEqual` and no depth write after the opaque phase, so it only fills pixels no opaque
// geometry covered. The world direction of a pixel is the difference of its unprojected far
// and near points (perspective and orthographic alike; an orthographic camera sees one
// direction), rotated into the cubemap by `view.sky_rotation` and scaled by
// `view.sky_params.x` (the `Skybox` intensity: linear radiance, above 1 feeds bloom).
//
// The points are unprojected with `view.inv_proj_rot`, the inverse of the projection times the
// view's *rotation* only, computed in f64 on the CPU: the camera position never enters the
// direction, so the sky is the same to ~0.01 px anywhere (the f32 `inv_view_proj` is badly
// conditioned far from the origin and made the sky jitter by several pixels).

// `OutputEncoding::id` of the pass (how the color is written; see `s3d_output`).
override S3D_OUTPUT: u32 = 0u;

@group(1) @binding(0) var s3d_sky_cube: texture_cube<f32>;
@group(1) @binding(1) var s3d_sky_sampler: sampler;

struct SkyOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

@vertex
fn vs_sky(@builtin(vertex_index) vertex_index: u32) -> SkyOut {
    let uv = vec2<f32>(f32((vertex_index << 1u) & 2u), f32(vertex_index & 2u));
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    var out: SkyOut;
    out.pos = vec4<f32>(ndc, 1.0, 1.0);
    out.ndc = ndc;
    return out;
}

@fragment
fn fs_sky(in: SkyOut) -> @location(0) vec4<f32> {
    let near = view.inv_proj_rot * vec4<f32>(in.ndc, 0.0, 1.0);
    let far = view.inv_proj_rot * vec4<f32>(in.ndc, 1.0, 1.0);
    let dir = far.xyz / far.w - near.xyz / near.w;
    let d = view.sky_rotation * dir;
    var c = textureSample(s3d_sky_cube, s3d_sky_sampler, d).rgb * view.sky_params.x;
    if (view.sky_ground.w > 0.0) {
        let w = 1.0 - smoothstep(-view.sky_ground.w, view.sky_ground.w, normalize(dir).y);
        c = mix(c, view.sky_ground.rgb, w);
    }
    return s3d_output(vec4<f32>(c, 1.0), S3D_OUTPUT, view.exposure.x, view.output.y);
}
