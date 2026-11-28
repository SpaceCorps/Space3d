//! Per-frame profiling counters.

/// How instanced draws are culled and issued. Chosen once at startup from adapter
/// capabilities (see [`crate::RenderCapabilities`]) and optionally downgraded via
/// [`crate::RenderSettings::draw_path`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DrawPath {
    /// CPU frustum culling + `draw_indexed` per batch. Works everywhere (WebGL2, downlevel);
    /// the only path on devices without `DownlevelFlags::BASE_VERTEX`, where its draws bind the
    /// mesh arena at each mesh instead of passing a `base_vertex`.
    #[default]
    CpuDirect,
    /// Compute-shader culling writing `DrawIndexedIndirectArgs`; one `draw_indexed_indirect`
    /// per batch (no `INDIRECT_FIRST_INSTANCE` needed: the vertex buffer is rebound per draw).
    GpuIndirect,
    /// Compute culling + `multi_draw_indexed_indirect` over runs of batches that share
    /// pipeline, material and instance buffer (needs `INDIRECT_FIRST_INSTANCE`).
    GpuMultiDraw,
    /// Compute culling + draw compaction + `multi_draw_indexed_indirect_count`
    /// (needs `MULTI_DRAW_INDIRECT_COUNT`; Vulkan/DX12). Empty batches cost nothing.
    GpuMultiDrawCount,
}

impl DrawPath {
    /// Whether this path culls on the GPU.
    pub fn gpu_culling(self) -> bool {
        self != DrawPath::CpuDirect
    }
}

/// Renderer counters for the most recent frame. Inserted as a resource by
/// [`crate::RenderPlugin`] and overwritten every frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RenderStats {
    /// Frame counter of the renderer.
    pub frame: u64,
    /// Cameras rendered.
    pub views: u32,
    /// Draw commands recorded (a multi-draw counts as one).
    pub draw_calls: u32,
    /// Of `draw_calls`, how many were indirect (single or multi).
    pub indirect_draw_calls: u32,
    /// Of `draw_calls`, how many were `multi_draw_indexed_indirect[_count]`.
    pub multi_draw_calls: u32,
    /// Distinct (pipeline, material, mesh, instance source) batches across all views.
    pub batches: u32,
    /// Instances submitted for culling (sum over views).
    pub instances: u64,
    /// Instances drawn after culling. Batches culled on the CPU count this frame's visible
    /// instances. Batches culled on the GPU count the visible instances *of an earlier frame*,
    /// read back without blocking (see [`Self::gpu_counts_frame`]; normally the previous
    /// frame, up to 3 frames back when the GPU runs behind); until the first readback arrives
    /// they count their pre-cull instances (an upper bound).
    pub instances_drawn: u64,
    /// Instances rejected by frustum culling: CPU-culled batches this frame (explicit buffers
    /// count whole culled blocks), GPU-culled batches as of [`Self::gpu_counts_frame`].
    pub culled: u64,
    /// Triangles drawn (`instances_drawn * mesh triangles`, same frames as `instances_drawn`).
    pub triangles: u64,
    /// Frame whose GPU-culling results are included in `instances_drawn`, `culled` and
    /// `triangles`. `None` when no batch was culled on the GPU this frame, or before the first
    /// readback completed (then GPU-culled batches count as fully drawn). The latency is
    /// `frame - gpu_counts_frame`.
    pub gpu_counts_frame: Option<u64>,
    /// All bytes copied CPU→GPU this frame (instances, uniforms, meshes, textures, jobs).
    pub bytes_uploaded: u64,
    /// Subset of `bytes_uploaded` that is per-instance data.
    pub instance_bytes_uploaded: u64,
    /// GPU buffers (re)allocated this frame (growth or first use).
    pub buffer_reallocations: u32,
    /// Staging chunks (4 MiB `MAP_WRITE` buffers) the upload ring created this frame. The ring
    /// grows to what the frames the GPU still holds need (their chunks cannot be reused until
    /// it is done with them), then creates none: 0 in the steady state.
    pub staging_chunks_created: u32,
    /// CPU-side scratch `Vec`s that had to grow this frame (0 in the steady state).
    pub cpu_reallocations: u32,
    /// Compute dispatches recorded (GPU culling).
    pub compute_dispatches: u32,
    /// `set_pipeline` calls in render passes (batches are sorted so that equal pipelines are
    /// adjacent: at most one per pipeline per view in the steady state).
    pub pipeline_switches: u32,
    /// Material bind-group switches in render passes (at most one per material per view).
    pub material_switches: u32,
    /// Draw path used this frame.
    pub draw_path: DrawPath,
    /// GPU culling active this frame.
    pub gpu_culling: bool,
    /// Meshes uploaded to the mesh arena this frame.
    pub meshes_uploaded: u32,
    /// Textures uploaded this frame.
    pub textures_uploaded: u32,
    /// Meshes still waiting for upload after this frame's uploads (see [`crate::UploadBudget`];
    /// 0 without a budget).
    pub pending_meshes: u32,
    /// Textures still waiting for upload after this frame's uploads.
    pub pending_textures: u32,
    /// Bytes of mesh and texture data still waiting for upload after this frame's uploads
    /// ([`crate::RenderAssets::pending_upload_bytes`]).
    pub pending_upload_bytes: u64,
    /// Render passes recorded by the graphics features (0 in Direct mode without them).
    pub render_passes: u32,
    /// Compute passes recorded by the graphics features (particles, auto exposure, …); render
    /// nodes report theirs with `PrepareContext::count_compute_passes`.
    pub compute_passes: u32,
    /// Full-screen passes (composite, bloom levels, FXAA, depth resolves, …).
    pub fullscreen_passes: u32,
    /// Target groups rendered through a scene texture (Hdr or Ldr mode).
    pub intermediate_groups: u32,
    /// Post groups (a rect with at least one post camera).
    pub post_groups: u32,
    /// Passes recorded by post (metering, bloom, composite, FXAA).
    pub post_passes: u32,
    /// Passes of the city effects (SSAO, its blurs, the apply pass; SSR's G-buffer pass is
    /// counted in `render_passes` only).
    pub screen_fx_passes: u32,
    /// SceneRead passes (depth-reading node slots).
    pub scene_read_passes: u32,
    /// MSAA depth resolves.
    pub depth_resolves: u32,
    /// Distortion passes (writers into a post group's distortion buffer).
    pub distortion_passes: u32,
    /// `RenderNode::draw` calls.
    pub node_draws: u32,
    /// Compute dispatches reported by nodes (`PrepareContext::count_dispatches`).
    pub node_dispatches: u32,
    /// Depth-only views (shadow maps).
    pub depth_views: u32,
    /// Draw calls of the depth-only views.
    pub shadow_draw_calls: u32,
    /// Instances submitted to the depth-only views.
    pub shadow_casters: u64,
    /// Transparent instances sorted back to front (`TransparentSort::PerInstance`).
    pub sorted_instances: u32,
    /// Image-based-lighting bake passes.
    pub ibl_bake_passes: u32,
    /// Frame and post textures created this frame (0 in the steady state).
    pub textures_created: u32,
    /// Bytes of the frame's and post's own render targets and of the directional shadow map
    /// (textures only; see `docs/performance.md` for the driver memory each render pass holds
    /// while its frame is in flight).
    pub target_bytes: u64,
    /// Bytes of the lighting area's other GPU textures: uploaded cubemaps (skyboxes,
    /// environment sources) and baked image-based lighting (specular and irradiance cubes),
    /// every mip.
    pub lighting_bytes: u64,
}

impl RenderStats {
    pub(crate) fn reset(&mut self, frame: u64, path: DrawPath) {
        *self = RenderStats {
            frame,
            draw_path: path,
            gpu_culling: path.gpu_culling(),
            ..Default::default()
        };
    }
}
