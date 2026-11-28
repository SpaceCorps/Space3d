//! Render nodes: the extension API for drawing into frame-owned passes (graphics spec
//! §2.2.4). A [`RenderNode`] registered in [`RenderNodes`] declares [`PhaseSlot`]s — the scene
//! phases it draws in — and gets `prepare` (once per frame, compute allowed), `prepare_view`
//! (once per camera view, reports which slots have work) and `draw` (per slot and view) calls.
//!
//! Particles, the skybox and distortion meshes are nodes; so can be any user draw that needs
//! the scene's depth, its HDR color space or its view uniform.
//!
//! ```no_run
//! use space3d_render::*;
//! struct DepthTint;
//! static SLOTS: [PhaseSlot; 1] = [PhaseSlot::new(RenderPhase::Transparent, 100).reading(SceneReads::DEPTH)];
//! impl RenderNode for DepthTint {
//!     fn slots(&self) -> &[PhaseSlot] { &SLOTS }
//!     fn draw(&mut self, pass: &mut wgpu::RenderPass<'static>, ctx: &DrawContext<'_>) {
//!         // pipeline built for `ctx.target` from VIEW_WGSL + scene_wgsl(&ctx.target) + OUTPUT_WGSL
//!         pass.set_bind_group(0, ctx.view_bind_group, &[ctx.view.view_offset]);
//!         pass.set_bind_group(1, ctx.scene_bind_group, &[]);
//!         pass.draw(0..3, 0..1);
//!     }
//! }
//! let mut app = space3d_core::App::new();
//! app.add_plugin(RenderPlugin::default());
//! app.resources.expect_mut::<RenderNodes>().insert("depth-tint", DepthTint);
//! ```
//!
//! Scheduling (graphics spec §1.4, §1.6, §1.7):
//! * `prepare` runs once per rendered frame, `prepare_view` once per camera view, both before any
//!   view pass; capture replays call neither (only `draw`, with [`FrameKind::CaptureReplay`]).
//! * Within a phase, slots draw by `(order, registration, slot index)`; built-in work sits at
//!   order 0 and draws before nodes of the same order (so negative orders draw before it).
//! * `Transparent` slots reading [`SceneReads::DEPTH`] draw in a separate single-sample
//!   *SceneRead* pass after the view's scene pass, with the scene depth bound in group 1 and
//!   either attached read-only ([`DepthAccess::ReadOnlyAttached`]) or not attached at all
//!   ([`DepthAccess::SampledOnly`], WebGL2/GLES). Multisampled groups without a depth resolve
//!   (GL, or several views sharing the multisampled target) draw them inside the scene pass
//!   instead, with the far dummy bound ([`DepthAccess::None`]): no soft fade there.
//! * `Distortion` slots draw into the distortion buffer of a post group, only in its settings
//!   camera's view and only when that camera's `PostProcess::distortion` is enabled.

use crate::assets::RenderAssets;
use crate::caps::GraphicsCaps;
use crate::components::RenderTarget;
use crate::culling::{Frustum, Sphere};
use crate::gpu::Uploader;
use crate::handles::{MeshHandle, TextureHandle};
use crate::output::{OutputEncoding, Tonemapper};
use crate::renderer::RenderCapabilities;
use crate::stats::RenderStats;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

/// Scene phases a node can draw in: per camera view, in this order; `Distortion` per post
/// group, into its distortion buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum RenderPhase {
    /// With the opaque and alpha-masked meshes (depth written).
    Opaque,
    /// After opaque work, at the far plane (the built-in skybox).
    Skybox,
    /// Blended work, after the built-in blended meshes (depth read, not written).
    Transparent,
    /// Distortion writers (post cameras only; see `DISTORTION_WGSL`).
    Distortion,
}

/// Scene inputs a slot samples (a bit set).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SceneReads(u8);

impl SceneReads {
    /// Nothing.
    pub const NONE: Self = Self(0);
    /// Scene depth of the view. On a `Transparent` slot this moves the whole slot to the
    /// SceneRead pass (graphics spec §1.6). Implied for `Distortion` slots when depth exists.
    pub const DEPTH: Self = Self(1);

    /// Both sets.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
    /// Every read of `other` is in `self`.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

impl std::ops::BitOr for SceneReads {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        self.union(rhs)
    }
}

/// One phase a node draws in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PhaseSlot {
    /// The phase.
    pub phase: RenderPhase,
    /// Order within the phase (ascending; built-in work sits at 0 and draws before nodes at
    /// the same order).
    pub order: i32,
    /// Scene inputs sampled by this slot's draws.
    pub reads: SceneReads,
}

impl PhaseSlot {
    /// A slot reading nothing. Built-in work sits at order 0 of its phase and draws before nodes
    /// at the same order.
    pub const fn new(phase: RenderPhase, order: i32) -> Self {
        Self {
            phase,
            order,
            reads: SceneReads::NONE,
        }
    }
    /// With these scene inputs.
    pub const fn reading(self, reads: SceneReads) -> Self {
        Self { reads, ..self }
    }
}

/// Per-view work report: bit `i` = `slots()[i]` has draws in this view (at most 32 slots).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SlotMask(pub u32);

impl SlotMask {
    /// No slot has work.
    pub const NONE: Self = Self(0);
    /// Every slot has work.
    pub const ALL: Self = Self(u32::MAX);
    /// Only `slot` (nothing for `slot ≥ 32`).
    pub const fn only(slot: usize) -> Self {
        Self::NONE.with(slot)
    }
    /// Also `slot` (unchanged for `slot ≥ 32`).
    pub const fn with(self, slot: usize) -> Self {
        if slot < 32 { Self(self.0 | 1 << slot) } else { self }
    }
    /// `slot` has work.
    pub const fn contains(self, slot: usize) -> bool {
        slot < 32 && self.0 >> slot & 1 == 1
    }
}

/// Draws into frame-owned passes. Registered in [`RenderNodes`].
///
/// Everything `draw` needs must be captured in `prepare`/`prepare_view`: windowed captures call
/// `draw` again without a `World` ([`FrameKind::CaptureReplay`]).
pub trait RenderNode: Send + Sync + 'static {
    /// Slots, read at registration and after [`RenderNodes::invalidate`] (never per frame).
    /// At most 32.
    fn slots(&self) -> &[PhaseSlot];

    /// Once per rendered frame (never for replays), after uploads, extraction and view planning
    /// and before GPU culling and every view pass: extract, upload ([`PrepareContext::upload`]),
    /// record compute (particle simulation).
    fn prepare(&mut self, ctx: &mut PrepareContext<'_>) {
        let _ = ctx;
    }

    /// Once per camera view per frame, after every node's `prepare`, in camera order. Returns
    /// the slots with work in this view: slots without work cost nothing (no SceneRead pass, no
    /// depth store, no distortion buffer write).
    fn prepare_view(&mut self, ctx: &mut ViewPrepareContext<'_>) -> SlotMask {
        let _ = ctx;
        SlotMask::ALL
    }

    /// Build pipelines for the pass signatures this node will be drawn with; called before the
    /// first draw with a new `(slot, PassTarget)` (registration, mode change, MSAA change) and
    /// by [`crate::Renderer::prewarm`]. Build pipelines here (WebGL2 links programs at creation).
    fn prewarm(&mut self, device: &wgpu::Device, targets: &[(usize, PassTarget)]) {
        let _ = (device, targets);
    }

    /// Record draws for `ctx.slot` of `ctx.view`. Pipelines must match `ctx.target`. Viewport and
    /// scissor are set to the view's rect (Distortion: the distortion buffer's rect); bind groups
    /// and pipeline are unset on entry.
    fn draw(&mut self, pass: &mut wgpu::RenderPass<'static>, ctx: &DrawContext<'_>);
}

/// A registered node, shared between its owner and the renderer.
pub type SharedNode = Arc<Mutex<dyn RenderNode>>;

/// Lock a node, recovering from a poisoned mutex (a panicking node never takes the renderer
/// down).
pub fn lock_node(node: &SharedNode) -> MutexGuard<'_, dyn RenderNode> {
    node.lock().unwrap_or_else(|e| e.into_inner())
}

/// At most this many slots per node.
pub(crate) const MAX_SLOTS: usize = 32;

struct NodeEntry {
    name: String,
    seq: u64,
    node: SharedNode,
}

/// Globally unique generations, so the renderer notices any change (even when the resource is
/// replaced wholesale).
fn next_generation() -> u64 {
    static GENERATION: AtomicU64 = AtomicU64::new(1);
    GENERATION.fetch_add(1, Ordering::Relaxed)
}

/// Registered nodes (resource, inserted by [`crate::RenderPlugin`] with the built-ins
/// `"space3d/skybox"` and `"space3d/distortion-meshes"`). The renderer re-reads the list and
/// the nodes' slots only when it changed ([`Self::invalidate`] forces it).
pub struct RenderNodes {
    entries: Vec<NodeEntry>,
    next_seq: u64,
    generation: u64,
}

impl Default for RenderNodes {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            next_seq: 0,
            generation: next_generation(),
        }
    }
}

impl std::fmt::Debug for RenderNodes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenderNodes")
            .field("nodes", &self.names().collect::<Vec<_>>())
            .finish()
    }
}

impl RenderNodes {
    /// Register `node` under `name` (replacing any node with that name) and return a typed
    /// handle the owner can lock to feed it data.
    pub fn insert<T: RenderNode>(&mut self, name: impl Into<String>, node: T) -> Arc<Mutex<T>> {
        let shared = Arc::new(Mutex::new(node));
        self.insert_shared(name, shared.clone());
        shared
    }

    /// Register an already shared node. Returns the node previously registered under `name`.
    pub fn insert_shared(&mut self, name: impl Into<String>, node: SharedNode) -> Option<SharedNode> {
        let name = name.into();
        let slots = lock_node(&node).slots().len();
        if slots > MAX_SLOTS {
            tracing::warn!(node = %name, slots, "a render node has more than {MAX_SLOTS} slots; the rest never draw");
        }
        let old = self.remove(&name);
        let seq = self.next_seq;
        self.next_seq += 1;
        self.entries.push(NodeEntry { name, seq, node });
        self.generation = next_generation();
        old
    }

    /// Unregister a node; returns it if it existed.
    pub fn remove(&mut self, name: &str) -> Option<SharedNode> {
        let i = self.entries.iter().position(|e| e.name == name)?;
        self.generation = next_generation();
        Some(self.entries.remove(i).node)
    }

    /// Node registered under `name`.
    pub fn get(&self, name: &str) -> Option<&SharedNode> {
        self.entries.iter().find(|e| e.name == name).map(|e| &e.node)
    }

    /// Whether a node with this name is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.entries.iter().any(|e| e.name == name)
    }

    /// Re-read every node's `slots()` next frame.
    pub fn invalidate(&mut self) {
        self.generation = next_generation();
    }

    /// `(node name, slot index, phase, order)` in execution order: by phase, then order, then
    /// registration, then slot index. Reads the nodes' current `slots()`.
    pub fn schedule(&self) -> impl Iterator<Item = (&str, usize, RenderPhase, i32)> {
        let mut out = Vec::new();
        for e in &self.entries {
            let node = lock_node(&e.node);
            for (i, s) in node.slots().iter().take(MAX_SLOTS).enumerate() {
                out.push((s.phase, s.order, e.seq, i, e.name.as_str()));
            }
        }
        out.sort_unstable_by_key(|&(phase, order, seq, i, _)| (phase, order, seq, i));
        out.into_iter()
            .map(|(phase, order, _, i, name)| (name, i, phase, order))
    }

    /// Node names in registration order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|e| e.name.as_str())
    }

    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// No nodes registered.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Changes whenever the list or its slots may have changed.
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    /// The registered nodes in registration order.
    pub(crate) fn nodes(&self) -> impl Iterator<Item = &SharedNode> {
        self.entries.iter().map(|e| &e.node)
    }
}

/// Deterministic per-frame facts.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct FrameInfo {
    /// `Time::frame` (App frames), or the renderer's frame count when no `Time` exists. Seeds
    /// all noise.
    pub frame: u64,
    /// `Time::tick`.
    pub tick: u64,
    /// `Time::alpha` (fraction of the next fixed tick).
    pub alpha: f32,
    /// Render time in seconds: `(tick + alpha) × fixed_delta` (never wall clock).
    pub time: f64,
    /// `Time::delta` if `Time::frame` advanced since this renderer's previous frame, else 0
    /// (and 0 in replays).
    pub delta: f32,
    /// Frame or capture replay.
    pub kind: FrameKind,
    /// Unique id of this `Renderer` instance (a new renderer or device = a new id: rebuild GPU
    /// state).
    pub renderer: u64,
    /// The device was lost (resources created now are invalid; skip work).
    pub device_lost: bool,
}

/// Why a node is called.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FrameKind {
    /// A rendered frame.
    Frame,
    /// A windowed capture re-recording the last frame (no `prepare`, no `World`).
    CaptureReplay,
}

/// How a target group renders this frame (graphics spec §1.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FrameMode {
    /// Straight into the target (no post camera, render scale 1): the default.
    Direct,
    /// Into an HDR scene texture, then post.
    Hdr,
    /// Into an LDR scene texture (the fallback, or render scale < 1 without post), then the
    /// composite.
    Ldr,
}

/// A camera view's role in its target group (graphics spec §1.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ViewRole {
    /// Renders the scene (into the scene texture in intermediate modes).
    Scene,
    /// Drawn LDR, Direct-style, over the post-processed image (gizmos, minimaps without post).
    Display,
}

/// One camera view of this frame.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct ViewInfo {
    /// The camera entity.
    pub camera: space3d_core::Entity,
    /// `Camera3d::order`.
    pub order: i32,
    /// Where the camera renders.
    pub target: RenderTarget,
    /// The target is the primary target.
    pub primary: bool,
    /// Frame mode of the view's target group.
    pub mode: FrameMode,
    /// Scene or display view.
    pub role: ViewRole,
    /// World → view.
    pub view: glam::Mat4,
    /// View → clip.
    pub proj: glam::Mat4,
    /// World → clip.
    pub view_proj: glam::Mat4,
    /// Camera position in world space.
    pub camera_pos: glam::Vec3,
    /// Near plane distance.
    pub near: f32,
    /// Far plane distance.
    pub far: f32,
    /// Orthographic projection.
    pub orthographic: bool,
    /// Culling frustum.
    pub frustum: Frustum,
    /// Rect `[x, y, w, h]` in pixels of the texture the view renders into (the scene texture
    /// for scene views in intermediate modes, the target otherwise).
    pub viewport: [u32; 4],
    /// Size of the scene texture (the target's size in Direct mode).
    pub scene_size: (u32, u32),
    /// Size of the target.
    pub target_size: (u32, u32),
    /// Render scale of the view's target group.
    pub render_scale: f32,
    /// MSAA samples of the view's scene pass.
    pub samples: u32,
    /// Color format of the view's scene pass.
    pub color_format: wgpu::TextureFormat,
    /// Encoding of the view's scene pass.
    pub output: OutputEncoding,
    /// Linear exposure for [`OutputEncoding::Tonemapped`] (1.0 otherwise).
    pub exposure: f32,
    /// Operator for [`OutputEncoding::Tonemapped`] (`None` otherwise).
    pub tonemapper: Tonemapper,
    /// Dynamic offset of this view's `View` uniform (bind group 0).
    pub view_offset: u32,
}

/// How scene depth is available to a pass (graphics spec §1.6).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DepthAccess {
    /// Far dummy bound in group 1 (fades must treat it as "no occluder").
    None,
    /// Real depth in group 1 and attached read-only: the hardware depth test works.
    ReadOnlyAttached,
    /// Real depth in group 1, no depth attachment: test manually (`s3d_scene_occluded`).
    SampledOnly,
}

/// Exact attachment signature of the pass a draw is recorded into.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct PassTarget {
    /// Color attachment format.
    pub color: wgpu::TextureFormat,
    /// Sample count.
    pub samples: u32,
    /// `Some(Depth32Float)` in scene and read-only-depth SceneRead passes; `None` in GL
    /// SceneRead and Distortion passes.
    pub depth: Option<wgpu::TextureFormat>,
    /// The depth attachment is read-only.
    pub depth_read_only: bool,
    /// Scene depth in group 1.
    pub depth_access: DepthAccess,
    /// Encoding fragment shaders must apply.
    pub output: OutputEncoding,
}

impl PassTarget {
    /// Color target state for this pass with `blend`.
    pub fn color_target(&self, blend: Option<wgpu::BlendState>) -> wgpu::ColorTargetState {
        wgpu::ColorTargetState {
            format: self.color,
            blend,
            write_mask: wgpu::ColorWrites::ALL,
        }
    }

    /// Depth state for this pass: `None` when the pass has no depth attachment. `depth_write` is
    /// forced off when the attachment is read-only. Nodes outside the Opaque phase must pass
    /// `false` (the scene pass depth is writable, but only opaque work may write it).
    pub fn depth_stencil(&self, compare: wgpu::CompareFunction, depth_write: bool) -> Option<wgpu::DepthStencilState> {
        self.depth.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: Some(depth_write && !self.depth_read_only),
            depth_compare: Some(compare),
            stencil: Default::default(),
            bias: Default::default(),
        })
    }

    /// Multisample state for this pass.
    pub fn multisample(&self) -> wgpu::MultisampleState {
        wgpu::MultisampleState {
            count: self.samples,
            ..Default::default()
        }
    }
}

/// A mesh of [`RenderAssets`] for node draws (cheap clones; valid until the mesh is removed or
/// the mesh arena grows — fetch it again every frame).
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct MeshRef {
    /// The mesh arena's vertex buffer (layout: [`MESH_VERTEX_LAYOUT`]).
    pub vertex_buffer: wgpu::Buffer,
    /// The mesh arena's `u32` index buffer.
    pub index_buffer: wgpu::Buffer,
    /// First index of the mesh.
    pub first_index: u32,
    /// Index count of the mesh.
    pub index_count: u32,
    /// Base vertex of the mesh. Without `RenderCapabilities::base_vertex` (WebGL2) bind
    /// `vertex_buffer` at `base_vertex × stride` and draw with base 0.
    pub base_vertex: i32,
    /// Mesh-local bounding sphere.
    pub bounds: Sphere,
}

/// Vertex layout of the mesh arena (position, normal, uv, color at locations 0..=3).
pub const MESH_VERTEX_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: crate::arena::VERTEX_SIZE,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &crate::arena::VERTEX_ATTRIBUTES,
};

/// A texture of [`RenderAssets`] for node draws (flipbooks, sprites, LUTs).
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct TextureRef {
    /// Sampled view (`D2`, or `D2Array` when `layered`).
    pub view: wgpu::TextureView,
    /// The texture's sampler.
    pub sampler: wgpu::Sampler,
    /// Texture format.
    pub format: wgpu::TextureFormat,
    /// Size of mip 0.
    pub size: (u32, u32),
    /// Array layers.
    pub layers: u32,
    /// Bound as `texture_2d_array` iff true (more than one layer).
    pub layered: bool,
    /// Bumped whenever the GPU texture is replaced: rebuild bind groups on change.
    pub version: u32,
}

/// Raw scene depth for custom bind groups.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct SceneDepthRef<'a> {
    /// `Depth32Float`; bind as an unfilterable float `texture_2d`.
    pub view: &'a wgpu::TextureView,
    /// Size in pixels.
    pub size: (u32, u32),
    /// Changes whenever the texture is recreated (cache bind groups by it).
    pub generation: u64,
}

/// A mesh of `assets` as a [`MeshRef`].
pub(crate) fn mesh_ref(assets: &RenderAssets, handle: MeshHandle) -> Option<MeshRef> {
    let (gpu, _) = assets.gpu_mesh(handle)?;
    Some(MeshRef {
        vertex_buffer: assets.arena.vertex.clone()?,
        index_buffer: assets.arena.index.clone()?,
        first_index: gpu.alloc.first_index,
        index_count: gpu.alloc.index_count,
        base_vertex: gpu.alloc.base_vertex as i32,
        bounds: gpu.sphere,
    })
}

/// A texture of `assets` as a [`TextureRef`].
pub(crate) fn texture_ref(assets: &RenderAssets, handle: TextureHandle) -> Option<TextureRef> {
    let e = assets.textures.get(handle.raw())?;
    let g = e.gpu.as_ref()?;
    Some(TextureRef {
        view: g.view.clone(),
        sampler: g.sampler.clone(),
        format: g.format,
        size: (e.width, e.height),
        layers: g.layers,
        layered: g.layered,
        version: e.version,
    })
}

/// Context of [`RenderNode::prepare`].
#[non_exhaustive]
pub struct PrepareContext<'a> {
    /// The device.
    pub device: &'a wgpu::Device,
    /// The queue.
    pub queue: &'a wgpu::Queue,
    /// The frame's command encoder (record compute passes here).
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// The world being rendered.
    pub world: &'a space3d_core::World,
    /// Resources. `RenderAssets` and `InstanceBuffers` are moved out during the frame: use
    /// [`Self::assets`].
    pub resources: &'a mut space3d_core::Resources,
    /// The render assets.
    pub assets: &'a RenderAssets,
    /// Frame facts.
    pub frame: FrameInfo,
    /// Camera views of this frame (depth views excluded).
    pub views: &'a [ViewInfo],
    /// Device capabilities and draw path.
    pub caps: &'a RenderCapabilities,
    /// Graphics capabilities (after downlevel masking).
    pub graphics: &'a GraphicsCaps,
    /// Shared bind group layouts and the full-screen module.
    pub layouts: &'a FrameLayouts,
    pub(crate) uploader: &'a mut Uploader,
    pub(crate) stats: &'a mut RenderStats,
    pub(crate) timing: &'a crate::timing::GpuTimer,
}

impl<'a> PrepareContext<'a> {
    /// Copy `data` into `buffer` at `offset` through the renderer's staging ring (recorded in the
    /// frame encoder; no per-call allocation; counted in `RenderStats::bytes_uploaded`).
    /// `buffer` needs `COPY_DST`.
    pub fn upload(&mut self, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
        self.uploader.write(self.encoder, buffer, offset, data);
    }

    /// A mesh for node draws (`None` until uploaded).
    pub fn mesh(&self, handle: MeshHandle) -> Option<MeshRef> {
        mesh_ref(self.assets, handle)
    }

    /// A texture for node draws (`None` until uploaded).
    pub fn texture(&self, handle: TextureHandle) -> Option<TextureRef> {
        texture_ref(self.assets, handle)
    }

    /// Timestamp writes for a compute pass the caller records into [`Self::encoder`], reported
    /// as `label` in [`crate::GpuTimings`] (`None` unless `FrameSettings::gpu_timing` is on and
    /// the device offers `TIMESTAMP_QUERY`). The writes borrow the frame's query set, not this
    /// context: pass them to `self.encoder.begin_compute_pass`. Call it once per pass, right
    /// before recording it.
    pub fn compute_timestamps(&mut self, label: &'static str) -> Option<wgpu::ComputePassTimestampWrites<'a>> {
        self.timing.compute(label, None)
    }

    /// Timestamp writes for a render pass the caller records (see [`Self::compute_timestamps`]).
    pub fn render_timestamps(&mut self, label: &'static str) -> Option<wgpu::RenderPassTimestampWrites<'a>> {
        self.timing.render(label, None)
    }

    /// Add `n` compute dispatches to `RenderStats::node_dispatches`.
    pub fn count_dispatches(&mut self, n: u32) {
        self.stats.node_dispatches += n;
    }

    /// Add `n` compute passes the node recorded in [`Self::encoder`] to
    /// `RenderStats::compute_passes`.
    pub fn count_compute_passes(&mut self, n: u32) {
        self.stats.compute_passes += n;
    }
}

/// Context of [`RenderNode::prepare_view`].
#[non_exhaustive]
pub struct ViewPrepareContext<'a> {
    /// The device.
    pub device: &'a wgpu::Device,
    /// The queue.
    pub queue: &'a wgpu::Queue,
    /// The frame's command encoder.
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// The world being rendered.
    pub world: &'a space3d_core::World,
    /// The view.
    pub view: &'a ViewInfo,
    /// Frame facts.
    pub frame: FrameInfo,
    /// Pass target each slot will be drawn with in this view (index = slot). Slots that cannot
    /// draw in this view (a Distortion slot on a camera that is not a post group's settings
    /// camera) get targets with [`DepthAccess::None`] and their mask bits are ignored.
    pub targets: &'a [PassTarget],
    /// Graphics capabilities (after downlevel masking).
    pub graphics: &'a GraphicsCaps,
    pub(crate) uploader: &'a mut Uploader,
    pub(crate) timing: &'a crate::timing::GpuTimer,
}

impl<'a> ViewPrepareContext<'a> {
    /// Copy through the staging ring (see [`PrepareContext::upload`]).
    pub fn upload(&mut self, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
        self.uploader.write(self.encoder, buffer, offset, data);
    }

    /// Timestamp writes for a compute pass the caller records, reported as `label` for this
    /// view's camera (see [`PrepareContext::compute_timestamps`]).
    pub fn compute_timestamps(&mut self, label: &'static str) -> Option<wgpu::ComputePassTimestampWrites<'a>> {
        self.timing.compute(label, Some(self.view.camera))
    }
}

/// Context of [`RenderNode::draw`].
#[non_exhaustive]
pub struct DrawContext<'a> {
    /// The device.
    pub device: &'a wgpu::Device,
    /// Index into `slots()` being drawn.
    pub slot: usize,
    /// Its phase.
    pub phase: RenderPhase,
    /// The view.
    pub view: &'a ViewInfo,
    /// The pass's attachment signature (pipelines must match it).
    pub target: PassTarget,
    /// Size of the attachment (Distortion: the distortion buffer's size).
    pub target_size: (u32, u32),
    /// Group 0 ([`FrameLayouts::view`]):
    /// `pass.set_bind_group(0, ctx.view_bind_group, &[ctx.view.view_offset])`.
    pub view_bind_group: &'a wgpu::BindGroup,
    /// Group 1 ([`FrameLayouts::scene`]): scene depth per `target.depth_access` (far dummy
    /// otherwise) and the view's post exposure (0 EV dummy without auto exposure; last frame's
    /// value during scene passes).
    pub scene_bind_group: &'a wgpu::BindGroup,
    /// Raw scene depth, when real depth is bound.
    pub scene_depth: Option<SceneDepthRef<'a>>,
    /// Frame facts.
    pub frame: FrameInfo,
}

/// Shared layouts (renderer lifetime, never recreated).
#[non_exhaustive]
pub struct FrameLayouts {
    /// Group 0: binding 0 the `View` uniform (dynamic offset, VERTEX | FRAGMENT); bindings 1–6
    /// the view's lighting, FRAGMENT: environment specular cube and irradiance cube
    /// (`texture_cube<f32>`), BRDF LUT (`texture_2d<f32>`), their filtering sampler, shadow map
    /// (`texture_depth_2d`) and its comparison sampler (graphics spec §3.4; neutral 1×1 dummies
    /// without an environment or a shadow). Identical to the forward pipelines' group 0: a
    /// shader declares only the bindings it uses. Compute passes of nodes use their own layouts.
    pub view: wgpu::BindGroupLayout,
    /// Group 1: binding 0 scene depth `texture_2d<f32>` (unfilterable), binding 1 exposure
    /// `texture_2d<f32>` (unfilterable, 1×1, EV in `.r`), binding 2 a non-filtering sampler
    /// (nearest, clamped) to sample the depth with: WebGL2 depth textures read as 0 through
    /// `textureLoad`, which uses the texture's own (linear) filtering there. FRAGMENT.
    pub scene: wgpu::BindGroupLayout,
    /// Full-screen triangle (`vs_fullscreen`, [`FULLSCREEN_WGSL`]).
    pub fullscreen: wgpu::ShaderModule,
}

impl FrameLayouts {
    /// The layouts, with group 0 = the forward pipelines' view layout.
    pub(crate) fn new(device: &wgpu::Device, view: &wgpu::BindGroupLayout) -> Self {
        let texture = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let scene = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("space3d scene inputs"),
            entries: &[
                texture(0),
                texture(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });
        let fullscreen = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("space3d fullscreen"),
            source: wgpu::ShaderSource::Wgsl(FULLSCREEN_WGSL.into()),
        });
        Self {
            view: view.clone(),
            scene,
            fullscreen,
        }
    }
}

/// `struct DirLight`, `struct PointLight`, `struct View` (the exact forward layout, graphics
/// spec §3.3) and `@group(0) @binding(0) var<uniform> view: View;`. Node shaders must not
/// redefine those names.
pub const VIEW_WGSL: &str = concat!(
    include_str!("shaders/common/view.wgsl"),
    "\n@group(0) @binding(0) var<uniform> view: View;\n"
);

/// Group 1 with real depth.
const SCENE_DEPTH_WGSL: &str = concat!(
    "const S3D_SCENE_DEPTH: bool = true;\n",
    include_str!("shaders/common/scene.wgsl")
);
/// Group 1 with the far dummy.
const SCENE_NO_DEPTH_WGSL: &str = concat!(
    "const S3D_SCENE_DEPTH: bool = false;\n",
    include_str!("shaders/common/scene.wgsl")
);

/// Group 1 declarations and helpers matching `target.depth_access` (requires [`VIEW_WGSL`]
/// first): `s3d_scene_uv(frag_xy)`, `s3d_scene_depth(px) -> f32` (device depth; 1.0 without
/// depth), `s3d_view_depth_at(px) -> f32`, `s3d_view_depth_of(world) -> f32`,
/// `s3d_depth_valid() -> bool`, `s3d_soft_fade(px, frag_view_depth, softness) -> f32` (1
/// without depth), `s3d_scene_occluded(px, frag_device_depth) -> bool` (false without depth),
/// `s3d_exposure_ev() -> f32`.
pub fn scene_wgsl(target: &PassTarget) -> &'static str {
    scene_wgsl_for(target.depth_access)
}

/// [`scene_wgsl`] for a depth access instead of a pass target: for shaders drawn outside
/// node passes, such as `HdrScene` overlays ([`crate::OverlayScene::scene_wgsl`]).
pub fn scene_wgsl_for(access: DepthAccess) -> &'static str {
    match access {
        DepthAccess::None => SCENE_NO_DEPTH_WGSL,
        DepthAccess::ReadOnlyAttached | DepthAccess::SampledOnly => SCENE_DEPTH_WGSL,
    }
}

/// `struct FullscreenOut { @builtin(position) pos, @location(0) uv }` and
/// `vs_fullscreen(vertex_index)` (one triangle; `uv` (0, 0) at the top left).
pub const FULLSCREEN_WGSL: &str = include_str!("shaders/common/fullscreen.wgsl");

#[cfg(test)]
mod tests {
    use super::*;

    struct Slots(Vec<PhaseSlot>);
    impl RenderNode for Slots {
        fn slots(&self) -> &[PhaseSlot] {
            &self.0
        }
        fn draw(&mut self, _: &mut wgpu::RenderPass<'static>, _: &DrawContext<'_>) {}
    }

    #[test]
    fn bit_sets() {
        assert!(SceneReads::DEPTH.contains(SceneReads::NONE));
        assert!((SceneReads::NONE | SceneReads::DEPTH).contains(SceneReads::DEPTH));
        assert!(!SceneReads::NONE.contains(SceneReads::DEPTH));
        let m = SlotMask::only(3).with(0).with(40);
        assert!(m.contains(0) && m.contains(3) && !m.contains(1) && !m.contains(40));
        assert_eq!(SlotMask::only(32), SlotMask::NONE);
        assert!(SlotMask::ALL.contains(31) && !SlotMask::NONE.contains(0));
        let s = PhaseSlot::new(RenderPhase::Transparent, 5).reading(SceneReads::DEPTH);
        assert_eq!(
            (s.phase, s.order, s.reads),
            (RenderPhase::Transparent, 5, SceneReads::DEPTH)
        );
    }

    #[test]
    fn registry_orders_by_phase_order_and_registration() {
        let mut nodes = RenderNodes::default();
        let g0 = nodes.generation();
        nodes.insert(
            "b",
            Slots(vec![
                PhaseSlot::new(RenderPhase::Transparent, 0),
                PhaseSlot::new(RenderPhase::Opaque, 10),
            ]),
        );
        nodes.insert("a", Slots(vec![PhaseSlot::new(RenderPhase::Transparent, 0)]));
        nodes.insert("c", Slots(vec![PhaseSlot::new(RenderPhase::Transparent, -1)]));
        assert_ne!(nodes.generation(), g0);
        let order: Vec<_> = nodes.schedule().collect();
        assert_eq!(
            order,
            [
                ("b", 1, RenderPhase::Opaque, 10),
                ("c", 0, RenderPhase::Transparent, -1),
                ("b", 0, RenderPhase::Transparent, 0),
                ("a", 0, RenderPhase::Transparent, 0),
            ]
        );
        assert_eq!(nodes.names().collect::<Vec<_>>(), ["b", "a", "c"]);
        // Replacing keeps the name, moves it to the end of the registration order.
        assert!(nodes.insert_shared("b", Arc::new(Mutex::new(Slots(vec![])))).is_some());
        assert_eq!(nodes.names().collect::<Vec<_>>(), ["a", "c", "b"]);
        let g = nodes.generation();
        nodes.invalidate();
        assert_ne!(nodes.generation(), g);
        assert!(nodes.remove("a").is_some() && nodes.remove("a").is_none());
        assert_eq!(
            (nodes.len(), nodes.contains("c"), nodes.get("zz").is_none()),
            (2, true, true)
        );
    }

    #[test]
    fn poisoned_nodes_still_lock() {
        let node: SharedNode = Arc::new(Mutex::new(Slots(vec![])));
        let n2 = node.clone();
        let _ = std::thread::spawn(move || {
            let _g = n2.lock().unwrap();
            panic!("node panicked");
        })
        .join();
        assert!(node.is_poisoned());
        assert!(lock_node(&node).slots().is_empty());
    }

    #[test]
    fn pass_targets() {
        let t = PassTarget {
            color: wgpu::TextureFormat::Rgba16Float,
            samples: 4,
            depth: Some(wgpu::TextureFormat::Depth32Float),
            depth_read_only: true,
            depth_access: DepthAccess::ReadOnlyAttached,
            output: OutputEncoding::Linear,
        };
        let d = t.depth_stencil(wgpu::CompareFunction::LessEqual, true).unwrap();
        assert_eq!(d.depth_write_enabled, Some(false), "read-only attachment");
        assert_eq!(t.multisample().count, 4);
        assert_eq!(t.color_target(None).format, wgpu::TextureFormat::Rgba16Float);
        let gl = PassTarget {
            depth: None,
            depth_read_only: false,
            depth_access: DepthAccess::SampledOnly,
            ..t
        };
        assert!(gl.depth_stencil(wgpu::CompareFunction::Always, false).is_none());
        assert_eq!(scene_wgsl(&gl), SCENE_DEPTH_WGSL);
        assert_eq!(
            scene_wgsl(&PassTarget {
                depth_access: DepthAccess::None,
                ..gl
            }),
            SCENE_NO_DEPTH_WGSL
        );
    }

    /// The shared WGSL modules validate on their own and composed the way nodes use them.
    #[test]
    fn wgsl_modules_validate() {
        use crate::pipeline::validate_wgsl;
        let fs = "@fragment fn fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
            let px = vec2<i32>(p.xy);
            let fade = s3d_soft_fade(px, 3.0, 0.5) * f32(!s3d_scene_occluded(px, p.z));
            let d = s3d_view_depth_at(px) + s3d_view_depth_of(vec3<f32>(0.0)) + s3d_exposure_ev();
            return s3d_output(vec4<f32>(s3d_scene_uv(p.xy), d, fade), view.output.x, view.exposure.x, view.output.y);
        }";
        for depth in [SCENE_DEPTH_WGSL, SCENE_NO_DEPTH_WGSL] {
            let src = format!("{VIEW_WGSL}\n{depth}\n{}\n{fs}", crate::output::OUTPUT_WGSL);
            validate_wgsl(&src).unwrap_or_else(|e| panic!("{e}\n{src}"));
            let distortion = format!(
                "{VIEW_WGSL}\n{depth}\n{}\n@fragment fn fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {{
                    return distortion_encode(vec2<f32>(0.01, -0.02), distortion_depth_fade(p, vec2<f32>(64.0), 0.5));
                }}",
                crate::DISTORTION_WGSL
            );
            validate_wgsl(&distortion).unwrap_or_else(|e| panic!("{e}\n{distortion}"));
        }
        validate_wgsl(&format!("{FULLSCREEN_WGSL}\n@fragment fn fs(i: FullscreenOut) -> @location(0) vec4<f32> {{ return vec4<f32>(i.uv, 0.0, 1.0); }}"))
            .unwrap_or_else(|e| panic!("{e}"));
    }

    /// `View` in `VIEW_WGSL` is the 1392-byte layout of graphics spec §3.3 (its first 752 bytes
    /// are today's forward `View`, the first 1312 the layout frozen at F3; `inv_proj_rot` is
    /// appended after them).
    #[test]
    fn view_wgsl_layout() {
        let module = wgpu::naga::front::wgsl::parse_str(VIEW_WGSL).unwrap();
        let (_, view) = module
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some("View"))
            .unwrap();
        let wgpu::naga::TypeInner::Struct { members, span } = &view.inner else {
            panic!("View is a struct")
        };
        assert_eq!(*span, 1392);
        let offset = |name: &str| members.iter().find(|m| m.name.as_deref() == Some(name)).unwrap().offset;
        assert_eq!(offset("view"), 752, "the forward prefix is unchanged");
        assert_eq!(offset("viewport"), 1008);
        assert_eq!(offset("shadow_view_proj"), 1088);
        assert_eq!(offset("sky_params"), 1296);
        assert_eq!(offset("inv_proj_rot"), 1312, "appended after the frozen layout");
    }
}
