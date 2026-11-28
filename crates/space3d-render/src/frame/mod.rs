//! Per-frame work: extract → batch → cull (CPU or GPU) → upload → encode.
//!
//! Every scratch `Vec` here lives in [`FrameState`] and is reused, so the steady state performs
//! no heap allocation in our code (`RenderStats::cpu_reallocations` counts the exceptions).

use crate::arena::{ArenaAlloc, Vertex};
use crate::assets::RenderAssets;
use crate::components::AmbientLight;
use crate::gpu::{GrowBuffer, TrackedBuffer, Uploader};
use crate::handles::FastMap;
use crate::handles::{MeshHandle, TextureHandle};
use crate::instance::InstanceBuffers;
use crate::pipeline::{
    GpuCompactParams, GpuCullJob, GpuCullParams, GpuFrustum, GpuMaterial, GpuPointLight, GpuView, Pipelines,
};
use crate::readback::CullReadback;
use crate::renderer::{RenderCapabilities, Renderer};
use crate::stats::DrawPath;
use space3d_core::{Resources, World};
use std::ops::Range;

mod depth;
mod draws;
mod extract;
mod nodes;
mod passes;
mod plan;
mod targets;
#[cfg(test)]
mod tests;
mod views;

use draws::{DrawCmd, DrawKind, VertexSource};
use extract::{GpuInstances, KeyBuckets, MaterialTable, Source, SourceBuffer};
pub(crate) use nodes::{NodeCache, SceneInputs};
use plan::{CameraItem, CameraPlan, FramePostGroup, PrimaryDisplay, TargetGroup};
pub(crate) use plan::{GroupTarget, PlanWarnings};
use targets::{Feedback, GroupTextures, ResolvedDepth};
pub(crate) use targets::{PoolDesc, TexturePool};
use views::{Dispatch, SceneLights, ViewData, ViewTarget};

/// Runtime render options (resource).
#[derive(Clone, Debug, PartialEq)]
pub struct RenderSettings {
    /// Force a draw path; clamped to [`crate::RenderCapabilities::best_draw_path`].
    /// `None` uses the best available.
    pub draw_path: Option<DrawPath>,
    /// Frustum culling on/off (CPU and GPU).
    pub frustum_culling: bool,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            draw_path: None,
            frustum_culling: true,
        }
    }
}

const ARGS_SIZE: u64 = 20;
const WORKGROUP: u32 = 64;

/// Frames the GPU may still be executing when `render_frame` returns (like a triple-buffered
/// swapchain; see `Renderer::pace_frames`).
pub(crate) const MAX_FRAMES_IN_FLIGHT: usize = 3;

/// What to do after trying to acquire the next surface texture.
pub(crate) enum SurfaceAcquire {
    Present(wgpu::SurfaceTexture),
    /// The configuration is outdated (resize, display change): reconfigure, skip this frame.
    Reconfigure,
    /// The surface is gone: it must be recreated from the window with `Instance::create_surface`.
    /// Reconfiguring it would fail validation.
    Recreate,
    /// Timeout, occluded or a (captured) validation error: skip presenting this frame.
    Skip,
}

impl SurfaceAcquire {
    pub(crate) fn from_current(t: wgpu::CurrentSurfaceTexture) -> Self {
        use wgpu::CurrentSurfaceTexture as C;
        match t {
            C::Success(t) | C::Suboptimal(t) => Self::Present(t),
            C::Outdated => Self::Reconfigure,
            C::Lost => Self::Recreate,
            other => {
                tracing::debug!(?other, "skipping present");
                Self::Skip
            }
        }
    }
}

/// Pending mesh uploads: handle, arena placement (once allocated), converted vertices and
/// indices, and the CPU data (restored to the entry if the mesh does not fit the arena).
type MeshScratch = Vec<(
    MeshHandle,
    Option<ArenaAlloc>,
    Vec<Vertex>,
    Vec<u32>,
    space3d_core::MeshData,
)>;

/// All persistent per-frame GPU buffers, caches and reusable scratch memory.
pub(crate) struct FrameState {
    pub mesh_scratch: MeshScratch,
    ecs: TrackedBuffer,
    cpu_visible: TrackedBuffer,
    views_buf: TrackedBuffer,
    materials_buf: TrackedBuffer,
    cull_params: TrackedBuffer,
    cull_jobs: TrackedBuffer,
    templates: TrackedBuffer,
    /// `(first args index, draw count)` per multi-draw-count run (compaction input).
    run_meta: TrackedBuffer,
    compact_params: TrackedBuffer,
    cull_out: GrowBuffer,
    args: GrowBuffer,
    compact_args: GrowBuffer,
    counts: GrowBuffer,
    explicit: Vec<Option<GpuInstances>>,
    materials: MaterialTable,
    /// View bind groups (group 0; see `depth::ViewBindGroups`): one per set of lighting
    /// resources the frame's views bind.
    view_bgs: depth::ViewBindGroups,
    /// Culling bind group per (source buffer, window of frustums).
    cull_bgs: FastMap<(SourceBuffer, u32), (wgpu::BindGroup, [u32; 5])>,
    compact_bg: Option<(wgpu::BindGroup, [u32; 5])>,
    arena_buffers: Option<(wgpu::Buffer, wgpu::Buffer)>,
    /// Visible ECS entities in query order: batch key (class | material | mesh), matrix, mesh.
    ecs_keys: Vec<u64>,
    ecs_matrices: Vec<[f32; 16]>,
    ecs_meshes: Vec<MeshHandle>,
    /// Row of each extracted entity in its archetype (gather source of `ecs_matrices`).
    ecs_rows: Vec<u32>,
    /// Last frame's `ecs_keys`: while the key sequence is unchanged, so is the batch order.
    ecs_prev_keys: Vec<u64>,
    /// Batch order (indices into the `ecs_*` vectors) and `(key, start, count)` per batch.
    ecs_order: Vec<u32>,
    ecs_runs: Vec<(u64, u32, u32)>,
    ecs_buckets: KeyBuckets,
    /// Byte ranges of `ecs` drawn from directly this frame (CPU path upload filter).
    ecs_needed: Vec<Range<usize>>,
    /// The ECS matrices exceed `max_buffer_size` (this frame once batched; logged once).
    ecs_overflow: bool,
    /// On overflow: instances kept per batch of `ecs_runs`, and the kept entries of
    /// `ecs_order` (what `ecs` holds and ECS sources index instead of `ecs_order`).
    ecs_kept: Vec<u32>,
    ecs_cut: Vec<u32>,
    /// Last frame's partially visible ECS matrices did not all fit `cpu_visible` (logged once).
    visible_overflow: bool,
    /// Per-instance visibility bits of the batch being CPU-culled.
    cull_mask: Vec<u64>,
    sources: Vec<Source>,
    cameras: Vec<CameraItem>,
    /// Every point light of the frame, in query order (see `views::select_point_lights`).
    point_lights: Vec<GpuPointLight>,
    /// Scratch of the per-view point-light selection: (squared distance, light index).
    light_order: Vec<(f32, u32)>,
    /// The frame plan (graphics spec §1.3): target groups, post groups and every camera's
    /// group and role (parallel to `cameras`). Kept until the next frame (capture replays).
    groups: Vec<TargetGroup>,
    post_groups: Vec<FramePostGroup>,
    /// The single-sample scene depth of each post group's settings camera, for the groups
    /// running the city effects (parallel to `post_groups`, filled by the scene pass) with the
    /// view's bind group index and the dynamic offset of its `View` uniform.
    post_depths: Vec<Option<(wgpu::TextureView, usize, u32)>>,
    camera_plans: Vec<CameraPlan>,
    /// Scene textures of intermediate groups, by target (freed 3 frames after last use).
    group_textures: FastMap<GroupTarget, GroupTextures>,
    texture_generation: u64,
    /// Per-group encode state of `encode_views` (reused scratch).
    group_encode: Vec<passes::GroupEncode>,
    /// The one texture pool of the frame (spec §2.6), for post chains.
    pub(crate) pool: TexturePool,
    /// Deterministic facts of the current frame (graphics spec §4.3).
    pub(crate) info: crate::node::FrameInfo,
    views: Vec<ViewData>,
    /// Public description of every view (parallel to `views`), for nodes and overlays.
    view_infos: Vec<crate::node::ViewInfo>,
    /// The camera views' `GpuView`s (parallel to `views`) until the lighting filled them in
    /// (`Renderer::write_view_uniforms`).
    gpu_views: Vec<GpuView>,
    /// Depth-only views of this frame (shadow maps, graphics spec §1.12), drawn before the
    /// camera views; their `GpuView` slots follow the camera views'.
    depth_views: Vec<depth::DepthView>,
    /// Indirect args of the camera views (the first ones; depth views' follow).
    camera_args: u32,
    /// Depth pipeline kind of every caster, instead of `extract::caster_pass` (set through
    /// `crate::testing`: renderer tests of the `DepthMask` path).
    pub(crate) force_caster_pass: Option<crate::pipeline::PassKind>,
    /// Slot masks reported by the nodes' `prepare_view`: `views × nodes`, row-major.
    node_masks: Vec<u32>,
    /// Scratch: the pass target of each slot of one node in one view.
    slot_targets: Vec<crate::node::PassTarget>,
    /// Single-sample depths multisampled groups resolve into (graphics spec §1.6).
    resolved_depth: FastMap<GroupTarget, ResolvedDepth>,
    /// The last primary scene view: HdrScene overlays draw after it and see its depth.
    hdr_scene_view: Option<usize>,
    draws: Vec<DrawCmd>,
    jobs: Vec<(SourceBuffer, GpuCullJob)>,
    dispatches: Vec<Dispatch>,
    /// Grid of the draw-compaction dispatch (`GpuMultiDrawCount`: one workgroup per run).
    compact: Option<(u32, u32)>,
    /// Params offset and grid of the per-job prefix-sum dispatch (`cull_scan`).
    scan: Option<(u32, (u32, u32))>,
    /// Frustum planes of every view with GPU-culled sources, in view order (written to the
    /// start of `cull_params`, before the dispatch params).
    cull_planes: Vec<GpuFrustum>,
    feedback: FastMap<TextureHandle, Feedback>,
    args_count: u32,
    run_count: u32,
    /// Instances and (pre-cull) triangles of this frame's GPU-culled batches.
    gpu_submitted: u64,
    gpu_triangles: u64,
    /// Readback of GPU-culled visible counts for `RenderStats` (1–3 frames of latency).
    readback: CullReadback,
    /// Workgroup records of all culling jobs this frame.
    wg_total: u32,
    /// Primary target size the current `views` were prepared for (windowed capture).
    last_primary_size: (u32, u32),
    path: DrawPath,
    storage: bool,
    uniform_align: usize,
    /// Largest storage binding, a multiple of 4 and at most `max_buffer_size`.
    max_storage_binding: u64,
    max_buffer_size: u64,
    /// The device's `max_texture_dimension_2d` (bounds valid viewports).
    max_texture_2d: u32,
    /// Largest `base_vertex` passed to a direct draw since the last `prepare`.
    #[cfg(test)]
    pub(crate) max_direct_base_vertex: i32,
    /// Test stand-in for a window whose surface is lost at the next frame's acquire: the
    /// renderer leaves its offscreen target alone like a windowed one, and the next frame
    /// takes the lost-surface path after preparing its views.
    #[cfg(test)]
    pub(crate) simulate_surface_loss: bool,
    /// Captures take the windowed replay path (`encode_capture_replay`) although headless.
    #[cfg(test)]
    pub(crate) force_capture_replay: bool,
    /// Test stand-in for an occluded or minimized window: frames get no surface texture, so
    /// nothing is drawn into the primary target (`Renderer::presents_to_window`).
    #[cfg(test)]
    pub(crate) simulate_occluded: bool,
    /// Times the ECS batch order was rebuilt.
    #[cfg(test)]
    pub(crate) ecs_reorders: u64,
    /// Every view renders with this output encoding, linear exposure and tonemapper.
    #[cfg(test)]
    pub(crate) force_output: Option<(crate::output::OutputEncoding, f32, crate::output::Tonemapper)>,
    /// Depth-view passes recorded so far (tests: capture replays record none).
    #[cfg(test)]
    pub(crate) depth_passes_encoded: u64,
}

fn strided(size: usize, align: usize) -> usize {
    size.next_multiple_of(align)
}

fn binding(buffer: &wgpu::Buffer, max: u64) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer,
        offset: 0,
        size: wgpu::BufferSize::new(buffer.size().min(max)),
    })
}

fn uniform_binding(buffer: &wgpu::Buffer, size: usize) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer,
        offset: 0,
        size: wgpu::BufferSize::new(size as u64),
    })
}

impl FrameState {
    pub fn new(caps: &RenderCapabilities, device: &wgpu::Device) -> Self {
        use wgpu::BufferUsages as U;
        let storage = caps.compute && caps.indirect;
        let s = if storage { U::STORAGE } else { U::empty() };
        Self {
            mesh_scratch: Vec::new(),
            ecs: TrackedBuffer::new("space3d ecs instances", U::VERTEX | s),
            cpu_visible: TrackedBuffer::new("space3d visible instances", U::VERTEX),
            views_buf: TrackedBuffer::new("space3d views", U::UNIFORM),
            materials_buf: TrackedBuffer::new("space3d materials", U::UNIFORM),
            cull_params: TrackedBuffer::new("space3d cull params", U::UNIFORM),
            cull_jobs: TrackedBuffer::new("space3d cull jobs", s),
            templates: TrackedBuffer::new("space3d indirect templates", U::COPY_SRC),
            run_meta: TrackedBuffer::new("space3d run meta", s),
            compact_params: TrackedBuffer::new("space3d compact params", U::UNIFORM),
            cull_out: GrowBuffer::new("space3d culled instances", U::VERTEX | s),
            // COPY_SRC: the culling-stats readback copies the visible counts out.
            args: GrowBuffer::new("space3d indirect args", U::INDIRECT | U::COPY_DST | U::COPY_SRC | s),
            compact_args: GrowBuffer::new("space3d compacted args", U::INDIRECT | s),
            counts: GrowBuffer::new("space3d draw counts", U::INDIRECT | U::COPY_DST | s),
            explicit: Vec::new(),
            materials: MaterialTable::default(),
            view_bgs: Default::default(),
            cull_bgs: FastMap::default(),
            compact_bg: None,
            arena_buffers: None,
            ecs_keys: Vec::new(),
            ecs_matrices: Vec::new(),
            ecs_meshes: Vec::new(),
            ecs_rows: Vec::new(),
            ecs_prev_keys: Vec::new(),
            ecs_order: Vec::new(),
            ecs_runs: Vec::new(),
            ecs_buckets: KeyBuckets::default(),
            ecs_needed: Vec::new(),
            ecs_overflow: false,
            ecs_kept: Vec::new(),
            ecs_cut: Vec::new(),
            visible_overflow: false,
            cull_mask: Vec::new(),
            sources: Vec::new(),
            cameras: Vec::new(),
            point_lights: Vec::new(),
            light_order: Vec::new(),
            groups: Vec::new(),
            post_groups: Vec::new(),
            post_depths: Vec::new(),
            camera_plans: Vec::new(),
            group_textures: FastMap::default(),
            texture_generation: 0,
            group_encode: Vec::new(),
            pool: TexturePool::default(),
            info: crate::node::FrameInfo {
                frame: 0,
                tick: 0,
                alpha: 0.0,
                time: 0.0,
                delta: 0.0,
                kind: crate::node::FrameKind::Frame,
                renderer: 0,
                device_lost: false,
            },
            views: Vec::new(),
            view_infos: Vec::new(),
            gpu_views: Vec::new(),
            depth_views: Vec::new(),
            camera_args: 0,
            force_caster_pass: None,
            node_masks: Vec::new(),
            slot_targets: Vec::new(),
            resolved_depth: FastMap::default(),
            hdr_scene_view: None,
            draws: Vec::new(),
            jobs: Vec::new(),
            dispatches: Vec::new(),
            compact: None,
            scan: None,
            cull_planes: Vec::new(),
            feedback: FastMap::default(),
            args_count: 0,
            run_count: 0,
            gpu_submitted: 0,
            gpu_triangles: 0,
            readback: CullReadback::default(),
            wg_total: 0,
            last_primary_size: (0, 0),
            path: DrawPath::CpuDirect,
            storage,
            uniform_align: device.limits().min_uniform_buffer_offset_alignment as usize,
            max_storage_binding: caps.max_storage_binding,
            max_buffer_size: caps.max_buffer_size,
            max_texture_2d: device.limits().max_texture_dimension_2d,
            #[cfg(test)]
            max_direct_base_vertex: 0,
            #[cfg(test)]
            simulate_surface_loss: false,
            #[cfg(test)]
            force_capture_replay: false,
            #[cfg(test)]
            simulate_occluded: false,
            #[cfg(test)]
            ecs_reorders: 0,
            #[cfg(test)]
            force_output: None,
            #[cfg(test)]
            depth_passes_encoded: 0,
        }
    }

    fn view_stride(&self) -> usize {
        strided(std::mem::size_of::<GpuView>(), self.uniform_align)
    }
    fn material_stride(&self) -> usize {
        strided(std::mem::size_of::<GpuMaterial>(), self.uniform_align)
    }
    fn params_stride(&self) -> usize {
        strided(std::mem::size_of::<GpuCullParams>(), self.uniform_align)
    }

    /// Whether any prepared view of the last frame draws to the primary target.
    pub(crate) fn has_primary_view(&self) -> bool {
        self.views.iter().any(|v| matches!(v.target, ViewTarget::Primary))
    }

    /// Capacity snapshot of scratch vectors (to count CPU reallocations).
    fn capacities(&self) -> [usize; 32] {
        [
            self.ecs_keys.capacity(),
            self.ecs_rows.capacity(),
            self.ecs_prev_keys.capacity(),
            self.ecs_matrices.capacity(),
            self.ecs_meshes.capacity(),
            self.ecs_buckets.of.capacity(),
            self.ecs_buckets.sorted.capacity(),
            self.ecs_order.capacity(),
            self.ecs_runs.capacity(),
            self.ecs_needed.capacity(),
            self.ecs_kept.capacity(),
            self.ecs_cut.capacity(),
            self.cull_mask.capacity(),
            self.sources.capacity(),
            self.cameras.capacity(),
            self.views.capacity(),
            self.draws.capacity(),
            self.jobs.capacity(),
            self.dispatches.capacity(),
            self.cull_planes.capacity(),
            self.point_lights.capacity(),
            self.light_order.capacity(),
            self.groups.capacity(),
            self.post_groups.capacity(),
            self.camera_plans.capacity(),
            self.group_encode.capacity(),
            self.view_infos.capacity(),
            self.node_masks.capacity(),
            self.slot_targets.capacity(),
            self.gpu_views.capacity(),
            self.depth_views.capacity(),
            self.view_bgs.capacity(),
        ]
    }
}

impl FrameState {
    /// Flush this frame's CPU-side buffers (instances, views, materials, culling inputs) to
    /// the GPU through the staging ring, and size the GPU-written ones.
    pub(super) fn upload(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        up: &mut Uploader,
        out_cursor: u64,
        wg_word: u32,
        reallocs: &mut u32,
    ) {
        let f = self;
        let mut cpu_reallocs = 0u32;
        let gpu = f.path.gpu_culling();
        // Bindings cover one full element even when empty.
        f.views_buf
            .ensure_len(std::mem::size_of::<GpuView>(), &mut cpu_reallocs);
        f.materials_buf
            .ensure_len(std::mem::size_of::<GpuMaterial>(), &mut cpu_reallocs);
        f.cull_params
            .ensure_len(std::mem::size_of::<GpuCullParams>(), &mut cpu_reallocs);
        f.compact_params
            .ensure_len(std::mem::size_of::<GpuCompactParams>(), &mut cpu_reallocs);
        let inst_before = up.bytes;
        if gpu {
            f.ecs.flush(device, encoder, up);
        } else {
            // Only batches drawn straight from `ecs` read it on the CPU path: culled and
            // partially visible ones (drawn from `cpu_visible`) keep their changes pending
            // until they are drawn from it.
            f.ecs_needed.clear();
            for d in &f.draws {
                if let (VertexSource::Ecs, DrawKind::Direct { instances }) = (d.vb, d.kind) {
                    let start = d.vb_offset as usize;
                    f.ecs_needed.push(start..start + instances as usize * 64);
                }
            }
            f.ecs_needed.sort_unstable_by_key(|r| r.start);
            f.ecs_needed.dedup_by(|next, kept| {
                let overlaps = next.start <= kept.end;
                if overlaps {
                    kept.end = kept.end.max(next.end);
                }
                overlaps
            });
            f.ecs.flush_needed(device, encoder, up, &f.ecs_needed);
        }
        f.cpu_visible.flush(device, encoder, up);
        up.instance_bytes += up.bytes - inst_before;
        f.views_buf.flush(device, encoder, up);
        f.materials_buf.flush(device, encoder, up);
        if gpu {
            f.cull_params.flush(device, encoder, up);
            f.cull_jobs.flush(device, encoder, up);
            f.templates.flush(device, encoder, up);
            f.run_meta.ensure_len(8, &mut cpu_reallocs);
            f.run_meta.flush(device, encoder, up);
            f.compact_params.flush(device, encoder, up);
            f.cull_out.ensure(device, out_cursor, false, up);
            let args_bytes = f.args_count as u64 * ARGS_SIZE;
            let records = wg_word as u64 * 4 + f.wg_total as u64 * 16;
            debug_assert!(
                records <= f.max_storage_binding,
                "cull records exceed the storage binding"
            );
            f.args.ensure(device, records, false, up);
            f.compact_args.ensure(device, args_bytes, false, up);
            f.counts.ensure(device, f.run_count as u64 * 4, false, up);
        }
        *reallocs += cpu_reallocs;
    }
}

impl Renderer {
    /// [`Renderer::prewarm`] (graphics spec §3.5): for every view of the last frame (and the
    /// primary target before the first frame), and for each requested extra format on primary
    /// views, the forward pipelines of the requested materials × instance layouts, every node's
    /// `prewarm` for the targets its slots draw with, and the composite pipelines of the
    /// intermediate groups' targets.
    pub(crate) fn prewarm_pipelines(&mut self, request: &crate::renderer::PrewarmRequest) {
        let primary = self.primary_display();
        let Renderer {
            device,
            frame: f,
            pipelines,
            nodes,
            post,
            frame_graphics,
            ..
        } = self;
        let ecs = <crate::instance::InstanceTransform as crate::instance::InstanceData>::LAYOUT;
        let layouts: Vec<u16> = std::iter::once(ecs)
            .chain(request.layouts.iter().copied())
            .map(|l| pipelines.layout_id(l))
            .collect();
        let forward = |f: &FrameState, pipelines: &mut Pipelines, format, samples, output| {
            for m in &request.materials {
                let shader = match m.shader {
                    Some(h) => match pipelines.custom_version(h) {
                        Some(v) => Some((h, v)),
                        // Custom shaders are prewarmed once drawn (their source is synced then).
                        None => continue,
                    },
                    None => None,
                };
                for &layout in &layouts {
                    // The materials area's key (class, texture kinds, globals) and caster.
                    f.prewarm_material(device, pipelines, m, shader, layout, (format, samples, output));
                }
            }
        };
        let primary_output = if primary.manual_srgb {
            crate::output::OutputEncoding::ManualSrgb
        } else {
            crate::output::OutputEncoding::Linear
        };
        forward(f, pipelines, primary.format, primary.samples, primary_output);
        for v in 0..f.views.len() {
            let is_primary = matches!(f.views[v].target, ViewTarget::Primary);
            let extra = request
                .formats
                .iter()
                .filter(|_| is_primary)
                .map(|&format| Some(format));
            for format in std::iter::once(None).chain(extra) {
                let t = f.view_targets(v, frame_graphics, format);
                forward(f, pipelines, t.scene.color, t.scene.samples, t.scene.output);
                for i in 0..nodes.len() {
                    let c = &nodes.nodes[i];
                    f.slot_targets.clear();
                    f.slot_targets.extend(c.slots.iter().map(|slot| t.slot(slot)));
                    let valid = t.valid(c);
                    nodes.prewarm(device, i, valid, &f.slot_targets);
                }
            }
        }
        for g in f.groups.iter().filter(|g| g.intermediate()) {
            post.prewarm(device, g.display_format, g.manual_srgb);
            if g.target == GroupTarget::Primary {
                for &format in &request.formats {
                    post.prewarm(device, format, false);
                }
            }
        }
    }

    /// Extract the world, cull, upload and draw one frame. Normally called by the
    /// `Stage::Render` system of [`crate::RenderPlugin`].
    pub fn render_frame(&mut self, world: &World, resources: &mut Resources) {
        use crate::timing::part;
        self.frame_index += 1;
        self.ensure_offscreen();
        // GPU-culling counts of earlier frames whose copies the GPU has finished.
        self.frame.readback.collect();
        let settings = resources.get::<RenderSettings>().cloned().unwrap_or_default();
        let ambient = resources.get::<AmbientLight>().copied().unwrap_or_default();
        let best = self.caps.best_draw_path;
        let path = settings.draw_path.map_or(best, |p| p.min(best));
        self.stats.reset(self.frame_index, path);
        self.uploader.reset_counters();
        self.frame.path = path;
        let mut frame_settings = resources
            .get::<crate::settings::FrameSettings>()
            .cloned()
            .unwrap_or_default();
        // Dynamic resolution scales the scene on top of the app's own setting.
        if let Some(d) = resources.get::<crate::DynamicResolution>() {
            frame_settings.render_scale *= d.scale();
        }
        self.frame_graphics = self.graphics.masked(frame_settings.downlevel);
        // GPU timings (spec §2.2.6): publish the newest finished frame's, then time this one.
        let changed = self.timer.collect(&mut self.gpu_timings);
        if let Some(t) = resources.get_mut::<crate::timing::GpuTimings>()
            && (changed || t.supported != self.gpu_timings.supported)
        {
            t.clone_from(&self.gpu_timings);
        }
        self.timer
            .begin_frame(&self.device, frame_settings.gpu_timing, self.frame_index);
        self.update_frame_info(resources.get::<space3d_core::Time>());
        self.post.begin_frame(&self.frame.info, &frame_settings);
        self.frame.pool.begin_frame(self.frame_index);
        self.scene_inputs.begin_frame(self.frame_index);
        if let Some(fx) = &mut self.screen_fx {
            fx.begin_frame(self.frame_index);
        }
        // Snapshots of the registered nodes and overlays (re-read only when they changed); the
        // overlays' stages are read every frame.
        self.nodes.sync(resources.get::<crate::node::RenderNodes>());
        self.overlays.sync(resources.get::<crate::overlay::RenderOverlays>());

        // Move the asset/instance registries out without re-boxing resources.
        let mut assets = resources
            .get_mut::<RenderAssets>()
            .map(std::mem::take)
            .unwrap_or_default();
        let mut buffers = resources
            .get_mut::<InstanceBuffers>()
            .map(std::mem::take)
            .unwrap_or_default();

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("space3d frame"),
        });
        // A headless renderer uploads everything at once unless the budget asks otherwise: its
        // frames are captures, which must not show a scene half loaded.
        let headless = self.surface.is_none();
        let upload_budget = resources
            .get::<crate::settings::UploadBudget>()
            .filter(|b| b.headless || !headless)
            .and_then(|b| b.bytes);
        self.upload_assets_into(&mut encoder, &mut assets, upload_budget);
        // Cubemap conversion and image-based-lighting bakes (spec §1.4 step 4).
        let lighting_settings = resources
            .get::<crate::lighting::api::LightingSettings>()
            .cloned()
            .unwrap_or_default();
        self.lighting.prepare(&mut crate::lighting::LightingPrepare {
            device: &self.device,
            queue: &self.queue,
            encoder: &mut encoder,
            world,
            assets: &mut assets,
            settings: &lighting_settings,
            graphics: &self.frame_graphics,
            headless,
            stats: &mut self.stats,
            timing: &self.timer,
        });
        for h in buffers.removed.drain(..) {
            if let Some(slot) = self.frame.explicit.get_mut(h.index() as usize) {
                *slot = None;
            }
            let gone = SourceBuffer::Explicit(h.index());
            self.frame.cull_bgs.retain(|(b, _), _| *b != gone);
        }
        {
            let _part = part("render.prepare");
            self.prepare(
                world,
                resources,
                &assets,
                &mut buffers,
                ambient,
                &settings,
                &frame_settings,
                &lighting_settings,
                &mut encoder,
            );
        }
        {
            let _part = part("render.compute");
            self.encode_compute(&mut encoder);
            self.apply_gpu_cull_counts();
        }
        // Depth views (shadow maps) before every camera view (spec §1.4 step 11).
        {
            let _part = part("render.depth_views");
            self.encode_depth_views(&mut encoder);
        }

        // Acquire the surface only when something presents.
        let mut surface_tex = None;
        let mut lost = false;
        if let Some(s) = &mut self.surface {
            match SurfaceAcquire::from_current(s.surface.get_current_texture()) {
                SurfaceAcquire::Present(t) => {
                    s.presented = true;
                    surface_tex = Some(t);
                }
                SurfaceAcquire::Reconfigure => s.surface.configure(&self.device, &s.config),
                SurfaceAcquire::Recreate
                    if crate::renderer::retry_without_view_formats(
                        cfg!(target_arch = "wasm32"),
                        s.presented,
                        &s.config,
                    ) =>
                {
                    tracing::warn!("the canvas rejected its sRGB view format; encoding sRGB in the shader instead");
                    s.drop_srgb_view(&self.device);
                }
                SurfaceAcquire::Recreate => lost = true,
                SurfaceAcquire::Skip => {}
            }
        }
        #[cfg(test)]
        {
            lost |= std::mem::take(&mut self.frame.simulate_surface_loss);
        }
        if lost {
            self.surface_lost();
        }
        let surface_view = surface_tex.as_ref().map(|t| {
            t.texture.create_view(&wgpu::TextureViewDescriptor {
                format: self.surface.as_ref().map(|s| s.view_format),
                ..Default::default()
            })
        });
        {
            let _part = part("render.views");
            self.encode_views(&mut encoder, surface_view.as_ref(), false);
        }
        {
            let _part = part("render.overlays");
            self.encode_frame_overlays(&mut encoder, surface_view.as_ref());
        }
        self.timer.end_frame(&mut encoder);

        let submit = part("render.submit");
        let span = part("render.submit.uploads");
        self.uploader.finish();
        drop(span);
        let span = part("render.submit.encoder");
        let commands = encoder.finish();
        drop(span);
        let span = part("render.submit.queue");
        let submission = self.queue.submit([commands]);
        self.timer.submitted(&self.queue);
        self.uploader.recall();
        drop(span);
        if let Some(t) = surface_tex {
            let _span = part("render.submit.present");
            self.queue.present(t);
        }
        drop(submit);
        {
            // (with the GPU behind, this is where the frame waits for it)
            let _part = part("render.pace");
            if !self.pace_frames(submission) {
                // Non-blocking: fires staging-chunk recycling and capture callbacks.
                let _ = self.device.poll(wgpu::PollType::Poll);
            }
        }

        self.post.end_frame(&mut self.stats);
        self.scene_inputs.evict();
        self.frame.pool.release_all();
        self.frame.pool.evict();
        self.stats.textures_created += self.frame.pool.take_created();
        self.stats.bytes_uploaded = self.uploader.bytes;
        self.stats.instance_bytes_uploaded = self.uploader.instance_bytes;
        self.stats.buffer_reallocations = self.uploader.reallocations;
        self.stats.staging_chunks_created = self.uploader.chunks_created;

        if let Some(r) = resources.get_mut::<RenderAssets>() {
            *r = assets;
        } else {
            resources.insert(assets);
        }
        if let Some(r) = resources.get_mut::<InstanceBuffers>() {
            *r = buffers;
        } else {
            resources.insert(buffers);
        }
        match resources.get_mut::<crate::RenderStats>() {
            Some(s) => s.clone_from(&self.stats),
            None => {
                resources.insert(self.stats.clone());
            }
        }
    }

    /// Keep at most [`MAX_FRAMES_IN_FLIGHT`] frames queued on the GPU: wait for the frame
    /// submitted that many frames ago (native; the browser paces its frames and cannot block).
    /// A window's presentation already paces frames and the wait returns at once; without one
    /// (headless renderers, an unpaced `App::update` / `step_ticks` loop) recording ran ahead of
    /// the GPU without bound — Metal queues up to 4096 command buffers — and every queued frame
    /// kept its staging chunks and command memory (gigabytes after a few thousand frames).
    /// Returns whether it waited (the wait fired the map callbacks a poll would).
    fn pace_frames(&mut self, submission: wgpu::SubmissionIndex) -> bool {
        let slot = (self.frame_index % MAX_FRAMES_IN_FLIGHT as u64) as usize;
        let oldest = self.in_flight[slot].replace(submission);
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(oldest) = oldest {
            // An error is a lost device (reported through the device-lost flag); a timeout is
            // a GPU that stalled for seconds: the frame goes on either way.
            let _ = self.device.poll(wgpu::PollType::Wait {
                submission_index: Some(oldest),
                timeout: Some(std::time::Duration::from_secs(2)),
            });
            return true;
        }
        #[cfg(target_arch = "wasm32")]
        let _ = oldest;
        false
    }

    /// The presentation surface is gone (`SurfaceAcquire::Recreate`) after this frame's views
    /// were prepared for it. `has_surface()` turns false: the platform layer attaches a new
    /// surface with `set_surface`. Until then frames render offscreen, this one included — into
    /// an offscreen target of the size its primary views were prepared for. A windowed
    /// renderer sizes that target only for captures, so it may still have an older window
    /// size, and a viewport's scissor rect outside it would fail validation, losing the
    /// frame's commands and with them its uploads.
    fn surface_lost(&mut self) {
        tracing::warn!("presentation surface lost; dropping it until a new one is attached");
        self.surface = None;
        self.size_offscreen(self.frame.last_primary_size);
    }

    /// The primary target of this frame: the surface (windowed) or the offscreen target.
    fn primary_display(&self) -> PrimaryDisplay {
        match &self.surface {
            Some(s) => PrimaryDisplay {
                size: (s.config.width, s.config.height),
                format: s.view_format,
                samples: s.attachments.samples,
                manual_srgb: s.manual_srgb,
            },
            None => PrimaryDisplay {
                size: (self.config.width, self.config.height),
                format: crate::pipeline::OFFSCREEN_FORMAT,
                samples: self.config.msaa_samples,
                manual_srgb: false,
            },
        }
    }

    /// Frame facts (graphics spec §4.3): `Time::frame` (the renderer's frame count without
    /// `Time`), the fixed tick and alpha, render time `(tick + alpha) × fixed_delta`, and
    /// `delta` only when `Time::frame` advanced since this renderer's previous frame.
    fn update_frame_info(&mut self, time: Option<&space3d_core::Time>) {
        let previous = self.frame.info;
        let info = &mut self.frame.info;
        info.kind = crate::node::FrameKind::Frame;
        info.renderer = self.id;
        info.device_lost = self.device_lost.load(std::sync::atomic::Ordering::Acquire);
        match time {
            Some(t) => {
                let advanced = self.frame_index == 1 || t.frame != previous.frame;
                info.frame = t.frame;
                info.tick = t.tick;
                info.alpha = t.alpha;
                info.time = (t.tick as f64 + t.alpha as f64) * t.fixed_delta.as_secs_f64();
                info.delta = if advanced { t.delta_secs() } else { 0.0 };
            }
            None => {
                info.frame = self.frame_index;
                info.tick = 0;
                info.alpha = 0.0;
                info.time = 0.0;
                info.delta = 0.0;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare(
        &mut self,
        world: &World,
        resources: &mut Resources,
        assets: &RenderAssets,
        buffers: &mut InstanceBuffers,
        ambient: AmbientLight,
        settings: &RenderSettings,
        frame_settings: &crate::settings::FrameSettings,
        lighting_settings: &crate::lighting::api::LightingSettings,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let primary = self.primary_display();
        self.frame.last_primary_size = primary.size;
        let frame_index = self.frame_index;
        let max_msaa = self.caps.max_msaa;
        let msaa_depth_binding = self.graphics.msaa_depth_resolve;
        let Renderer {
            device,
            queue,
            caps,
            frame: f,
            pipelines,
            uploader: up,
            stats,
            white,
            white_sampler,
            post,
            plan_warnings,
            frame_graphics,
            layouts,
            nodes,
            overlays,
            lighting,
            injected_depth_views,
            timer,
            ..
        } = self;
        let caps_before = f.capacities();
        let mut cpu_reallocs = 0u32;

        f.arena_buffers = match (&assets.arena.vertex, &assets.arena.index) {
            (Some(v), Some(i)) => Some((v.clone(), i.clone())),
            _ => None,
        };

        let lights = SceneLights::collect(world, &mut f.point_lights);
        f.collect_cameras(world);
        // The frame plan (spec §1.4 step 3): groups, modes, post groups, roles, group textures.
        f.plan_frame(assets, primary, frame_settings, frame_graphics, max_msaa, plan_warnings);
        f.prepare_group_textures(device, frame_index, msaa_depth_binding, stats);
        for g in &f.groups {
            if g.intermediate() {
                stats.intermediate_groups += 1;
                stats.post_groups += f.post_groups[g.post_groups.clone()]
                    .iter()
                    .filter(|p| !p.passthrough)
                    .count() as u32;
            }
        }
        let span = crate::timing::part("render.prepare.extract");
        f.extract(
            world,
            assets,
            buffers,
            device,
            encoder,
            up,
            pipelines,
            lighting_settings,
            frame_index,
            stats,
            &mut cpu_reallocs,
        );
        drop(span);
        let span = crate::timing::part("render.prepare.views");
        let mut out_cursor = 0;
        let mut visible_overflow = false;
        f.prepare_views(
            assets,
            buffers,
            &lights,
            ambient,
            settings.frustum_culling,
            stats,
            &mut out_cursor,
            &mut visible_overflow,
            &mut cpu_reallocs,
        );
        // Depth views (spec §1.4 step 5; shadow maps), then the camera views' uniforms with the
        // lighting's fields (step 8).
        let requests = lighting.depth_views(world, &f.view_infos, lighting_settings);
        f.prepare_depth_views(
            requests.iter().chain(injected_depth_views.iter()),
            buffers,
            settings.frustum_culling,
            stats,
            &mut out_cursor,
            &mut visible_overflow,
            &mut cpu_reallocs,
        );
        f.finish_views(visible_overflow);
        f.write_view_uniforms(world, lighting, &mut cpu_reallocs);
        drop(span);
        // Render nodes (spec §1.4 steps 7–8): `prepare`, then `prepare_view` per camera view.
        let span = crate::timing::part("render.prepare.nodes");
        f.prepare_nodes(nodes::NodePrepare {
            device,
            queue,
            encoder,
            world,
            resources,
            assets,
            caps,
            graphics: frame_graphics,
            layouts,
            uploader: up,
            stats,
            nodes,
            overlay_depth: overlays.hdr_scene_depth(),
            frame_index,
            timing: timer,
        });
        drop(span);
        let span = crate::timing::part("render.prepare.upload");
        // Post groups (spec §1.4 step 6): their composite parameters, in plan order, and the
        // distortion buffers of the groups whose Distortion phase runs (created here only).
        for pg in &f.post_groups {
            let g = &f.groups[pg.group];
            let rect = |r: plan::ViewportRect| [r.x, r.y, r.w as i32, r.h as i32];
            let plan = crate::post::PostGroupPlan {
                camera: f.cameras[pg.camera].id,
                settings: pg.settings,
                mode: g.mode,
                scene_rect: rect(pg.scene_rect),
                output_rect: rect(pg.display_rect),
                scene_size: g.scene_size,
                output_size: g.display_size,
                scene_alpha: g.scene_format != wgpu::TextureFormat::Rg11b10Ufloat,
                transparent: g.transparent,
                passthrough: pg.passthrough,
                distortion: pg.distortion,
            };
            post.plan_group(world, assets, &plan, &mut cpu_reallocs);
            if pg.distortion {
                // The group's rect in `S` (spec §1.8 as amended): the buffer follows the render
                // scale, and scene depth maps onto it at a fixed ratio.
                let rect = pg.scene_rect.scissor(g.scene_size);
                post.distortion_target(device, f.cameras[pg.camera].id, rect, pg.settings.distortion.resolution);
            }
        }
        let wg_word = f.prepare_dispatches(&mut cpu_reallocs);
        f.upload(device, encoder, up, out_cursor, wg_word, &mut cpu_reallocs);
        post.flush(device, encoder, up);
        drop(span);
        let _span = crate::timing::part("render.prepare.bind_groups");

        // Custom-shader pipelines whose (asynchronous, browser) validation failed since the
        // last frame fall back to the built-in shader before anything is drawn with them.
        pipelines.poll_pending(device);
        f.prepare_view_bind_groups(device, pipelines, lighting, frame_index);
        f.prepare_feedback(device, assets, frame_index);
        f.prepare_material_bind_groups(device, assets, pipelines, white, white_sampler);
        f.prepare_cull_bind_groups(device, pipelines, frame_index);

        let caps_after = f.capacities();
        cpu_reallocs += caps_before
            .iter()
            .zip(caps_after.iter())
            .filter(|(a, b)| a != b)
            .count() as u32;
        stats.cpu_reallocations += cpu_reallocs;
    }
}
