//! Render nodes in the frame (graphics spec §1.4 steps 7–8, §1.6–§1.8): the renderer's snapshot
//! of [`RenderNodes`] with per-phase sorted slot lists, the pass targets slots draw with, how a
//! view reads scene depth, recording node slots into passes, and the scene-input bind groups
//! (group 1).

use crate::handles::FastMap;
use crate::node::{
    DepthAccess, DrawContext, FrameInfo, FrameLayouts, MAX_SLOTS, PassTarget, PhaseSlot, RenderNodes, RenderPhase,
    SceneDepthRef, SceneReads, SharedNode, ViewInfo, lock_node,
};
use crate::output::OutputEncoding;
use crate::pipeline::DEPTH_FORMAT;
use crate::post::api::DISTORTION_FORMAT;
use std::sync::Arc;

/// The scheduling list a slot belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SlotList {
    Opaque = 0,
    Skybox = 1,
    /// `Transparent` slots that do not read scene depth (inside the scene pass).
    Transparent = 2,
    /// `Transparent` slots reading [`SceneReads::DEPTH`] (the SceneRead pass, or in-pass).
    Reading = 3,
    Distortion = 4,
}

const LISTS: usize = 5;

/// Which orders of a list to draw: built-in work sits at order 0 and draws before nodes of the
/// same order, so negative orders draw before it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Orders {
    BeforeBuiltins,
    AfterBuiltins,
    All,
}

impl Orders {
    fn contains(self, order: i32) -> bool {
        match self {
            Self::BeforeBuiltins => order < 0,
            Self::AfterBuiltins => order >= 0,
            Self::All => true,
        }
    }
}

/// A scheduled slot: node (index in [`NodeCache::nodes`]), slot index and order.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SlotRef {
    pub node: u32,
    pub slot: u32,
    pub order: i32,
}

/// A registered node as the renderer sees it: its slots as of the last snapshot.
pub(crate) struct CachedNode {
    pub node: SharedNode,
    pub slots: Vec<PhaseSlot>,
    /// `(slot, target)` pairs `prewarm` was called with.
    pub prewarmed: Vec<(usize, PassTarget)>,
    /// Slot bits per [`SlotList`].
    pub bits: [u32; LISTS],
    /// Bits of the slots with a negative order.
    pub negative: u32,
}

/// The renderer's snapshot of [`RenderNodes`] (re-read only when its generation changes, like
/// the overlay snapshot), kept for capture replays, and the per-phase schedule: slots sorted by
/// `(order, registration, slot index)`.
#[derive(Default)]
pub(crate) struct NodeCache {
    generation: u64,
    pub nodes: Vec<CachedNode>,
    lists: [Vec<SlotRef>; LISTS],
}

/// Work bits of a view (`ViewData::work`): which node lists have a slot with work.
pub(crate) mod work {
    /// Opaque slots with a negative order.
    pub const OPAQUE_PRE: u8 = 1;
    /// Opaque slots at order ≥ 0.
    pub const OPAQUE_POST: u8 = 2;
    pub const SKYBOX: u8 = 4;
    /// Non-reading Transparent slots with a negative order.
    pub const TRANSPARENT_PRE: u8 = 8;
    /// Non-reading Transparent slots at order ≥ 0.
    pub const TRANSPARENT_POST: u8 = 16;
    /// Depth-reading Transparent slots.
    pub const READING: u8 = 32;
    /// Distortion slots (only in a post group's settings camera with distortion enabled).
    pub const DISTORTION: u8 = 64;
    /// The city effects of the view's post group need its single-sample scene depth (set by
    /// the frame, not by nodes).
    pub const SCREEN_FX: u8 = 128;
    /// Node work between the built-in opaque and blended batches.
    pub const BETWEEN: u8 = OPAQUE_POST | SKYBOX | TRANSPARENT_PRE;
}

impl NodeCache {
    /// Re-read the registered nodes and their slots if `nodes` changed since the last call.
    /// Nodes still registered with the same slots keep their prewarm records.
    pub fn sync(&mut self, nodes: Option<&RenderNodes>) {
        let generation = nodes.map_or(0, RenderNodes::generation);
        if generation == self.generation {
            return;
        }
        self.generation = generation;
        let old = std::mem::take(&mut self.nodes);
        for list in &mut self.lists {
            list.clear();
        }
        let Some(nodes) = nodes else { return };
        for (n, shared) in nodes.nodes().enumerate() {
            let slots: Vec<PhaseSlot> = lock_node(shared).slots().iter().take(MAX_SLOTS).copied().collect();
            let prewarmed = old
                .iter()
                .find(|o| Arc::ptr_eq(&o.node, shared) && o.slots == slots)
                .map(|o| o.prewarmed.clone())
                .unwrap_or_default();
            let (mut bits, mut negative) = ([0u32; LISTS], 0u32);
            for (s, slot) in slots.iter().enumerate() {
                let list = match slot.phase {
                    RenderPhase::Opaque => SlotList::Opaque,
                    RenderPhase::Skybox => SlotList::Skybox,
                    RenderPhase::Transparent if slot.reads.contains(SceneReads::DEPTH) => SlotList::Reading,
                    RenderPhase::Transparent => SlotList::Transparent,
                    RenderPhase::Distortion => SlotList::Distortion,
                };
                bits[list as usize] |= 1 << s;
                negative |= u32::from(slot.order < 0) << s;
                self.lists[list as usize].push(SlotRef {
                    node: n as u32,
                    slot: s as u32,
                    order: slot.order,
                });
            }
            self.nodes.push(CachedNode {
                node: shared.clone(),
                slots,
                prewarmed,
                bits,
                negative,
            });
        }
        for list in &mut self.lists {
            list.sort_unstable_by_key(|r| (r.order, r.node, r.slot));
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Work bits ([`work`]) of a view whose nodes reported `masks` (one per node).
    pub fn work(&self, masks: &[u32]) -> u8 {
        let mut w = 0u8;
        for (n, &mask) in self.nodes.iter().zip(masks) {
            let any = |list: SlotList| mask & n.bits[list as usize];
            let (neg, pos) = (mask & n.negative, mask & !n.negative);
            let opaque = any(SlotList::Opaque);
            let transparent = any(SlotList::Transparent);
            if opaque & neg != 0 {
                w |= work::OPAQUE_PRE;
            }
            if opaque & pos != 0 {
                w |= work::OPAQUE_POST;
            }
            if any(SlotList::Skybox) != 0 {
                w |= work::SKYBOX;
            }
            if transparent & neg != 0 {
                w |= work::TRANSPARENT_PRE;
            }
            if transparent & pos != 0 {
                w |= work::TRANSPARENT_POST;
            }
            if any(SlotList::Reading) != 0 {
                w |= work::READING;
            }
            if any(SlotList::Distortion) != 0 {
                w |= work::DISTORTION;
            }
        }
        w
    }

    /// Call `prewarm` for the slots of `mask` whose target in `targets` (index = slot) is new.
    pub fn prewarm(&mut self, device: &wgpu::Device, node: usize, mask: u32, targets: &[PassTarget]) {
        let CachedNode {
            node: shared,
            prewarmed,
            ..
        } = &mut self.nodes[node];
        for (slot, target) in targets.iter().enumerate() {
            if mask >> slot & 1 == 1 && !prewarmed.contains(&(slot, *target)) {
                lock_node(shared).prewarm(device, &[(slot, *target)]);
                prewarmed.push((slot, *target));
            }
        }
    }

    /// Record the slots of `list` (orders `orders`) with work in `pass.masks` into `rpass`, in
    /// schedule order. Viewport and scissor are set before every draw. Returns the draws.
    pub fn draw(
        &mut self,
        rpass: &mut wgpu::RenderPass<'static>,
        list: SlotList,
        orders: Orders,
        pass: &NodePass<'_>,
    ) -> u32 {
        let mut draws = 0;
        for k in 0..self.lists[list as usize].len() {
            let r = self.lists[list as usize][k];
            if !orders.contains(r.order) || pass.masks.get(r.node as usize).is_none_or(|m| m >> r.slot & 1 == 0) {
                continue;
            }
            let slot = r.slot as usize;
            let CachedNode {
                node: shared,
                slots,
                prewarmed,
                ..
            } = &mut self.nodes[r.node as usize];
            let mut node = lock_node(shared);
            // Pipelines are built before the first draw with a new pass signature (normally in
            // `prepare_view`; capture replays may draw into another format).
            if !prewarmed.contains(&(slot, pass.target)) {
                node.prewarm(pass.device, &[(slot, pass.target)]);
                prewarmed.push((slot, pass.target));
            }
            let [x, y, w, h] = pass.viewport;
            rpass.set_viewport(x, y, w, h, 0.0, 1.0);
            let [x, y, w, h] = pass.scissor;
            rpass.set_scissor_rect(x, y, w, h);
            let ctx = DrawContext {
                device: pass.device,
                slot,
                phase: slots[slot].phase,
                view: pass.view,
                target: pass.target,
                target_size: pass.target_size,
                view_bind_group: pass.view_bind_group,
                scene_bind_group: pass.scene_bind_group,
                scene_depth: pass.scene_depth,
                frame: pass.frame,
            };
            node.draw(rpass, &ctx);
            draws += 1;
        }
        draws
    }
}

/// What node draws of one view in one pass need.
pub(crate) struct NodePass<'a> {
    pub device: &'a wgpu::Device,
    pub view: &'a ViewInfo,
    /// The view's slot masks (one per node).
    pub masks: &'a [u32],
    pub target: PassTarget,
    pub target_size: (u32, u32),
    pub view_bind_group: &'a wgpu::BindGroup,
    pub scene_bind_group: &'a wgpu::BindGroup,
    pub scene_depth: Option<SceneDepthRef<'a>>,
    pub frame: FrameInfo,
    /// Set before every draw.
    pub viewport: [f32; 4],
    pub scissor: [u32; 4],
}

/// Signature of a view's scene pass (depth attached and writable; nothing bound in group 1).
pub(crate) fn scene_target(color: wgpu::TextureFormat, samples: u32, output: OutputEncoding) -> PassTarget {
    PassTarget {
        color,
        samples,
        depth: Some(DEPTH_FORMAT),
        depth_read_only: false,
        depth_access: DepthAccess::None,
        output,
    }
}

/// Signature of a single-sample SceneRead pass: depth attached read-only, or not attached.
pub(crate) fn read_target(color: wgpu::TextureFormat, output: OutputEncoding, attached: bool) -> PassTarget {
    PassTarget {
        color,
        samples: 1,
        depth: attached.then_some(DEPTH_FORMAT),
        depth_read_only: attached,
        depth_access: if attached {
            DepthAccess::ReadOnlyAttached
        } else {
            DepthAccess::SampledOnly
        },
        output,
    }
}

/// Signature of a Distortion pass (into the post group's distortion buffer; scene depth
/// sampled when there is a single-sample one).
pub(crate) fn distortion_target(depth: bool) -> PassTarget {
    PassTarget {
        color: DISTORTION_FORMAT,
        samples: 1,
        depth: None,
        depth_read_only: false,
        depth_access: if depth {
            DepthAccess::SampledOnly
        } else {
            DepthAccess::None
        },
        output: OutputEncoding::Linear,
    }
}

/// How a view's depth-reading Transparent slots get scene depth (graphics spec §1.6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReadMode {
    /// A single-sample SceneRead pass with the view's depth attached read-only and sampled.
    Attached,
    /// A single-sample SceneRead pass without a depth attachment (GL): depth sampled only.
    Sampled,
    /// Resolve the multisampled depth (minimum of the samples) into a single-sample depth,
    /// then [`Self::Attached`] on the resolved color.
    Resolved,
    /// Inside the multisampled scene pass, far dummy bound: no soft fade.
    InPass,
}

/// The read mode of a view drawn with `samples` samples. `rods`: read-only depth attachments;
/// `resolve`: multisampled depth can be resolved (and a resolve target exists); `single`: no
/// other view of the frame draws into the same multisampled attachments (a later view would
/// load multisampled content without the single-sample pass's draws).
pub(crate) fn read_mode(samples: u32, rods: bool, resolve: bool, single: bool) -> ReadMode {
    match samples {
        0 | 1 if rods => ReadMode::Attached,
        0 | 1 => ReadMode::Sampled,
        _ if rods && resolve && single => ReadMode::Resolved,
        _ => ReadMode::InPass,
    }
}

/// Everything the node steps of `prepare` (graphics spec §1.4 steps 7–8) borrow from the
/// renderer.
pub(crate) struct NodePrepare<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub encoder: &'a mut wgpu::CommandEncoder,
    pub world: &'a space3d_core::World,
    pub resources: &'a mut space3d_core::Resources,
    pub assets: &'a crate::assets::RenderAssets,
    pub caps: &'a crate::renderer::RenderCapabilities,
    pub graphics: &'a crate::caps::GraphicsCaps,
    pub layouts: &'a FrameLayouts,
    pub uploader: &'a mut crate::gpu::Uploader,
    pub stats: &'a mut crate::stats::RenderStats,
    pub nodes: &'a mut NodeCache,
    /// Some HdrScene overlay reads scene depth.
    pub overlay_depth: bool,
    pub frame_index: u64,
    pub timing: &'a crate::timing::GpuTimer,
}

impl super::FrameState {
    /// The pass signature a view's scene pass has this frame: color format, samples and the
    /// encoding its fragments apply.
    pub(super) fn view_pass(&self, v: usize) -> (wgpu::TextureFormat, u32, OutputEncoding) {
        let (info, view) = (&self.view_infos[v], &self.views[v]);
        let group = &self.groups[view.group];
        let manual = !(group.intermediate() && view.role == crate::node::ViewRole::Scene) && group.manual_srgb;
        (
            info.color_format,
            info.samples,
            super::passes::pass_output(view.output, manual),
        )
    }

    /// Steps 7–8 (graphics spec §1.4) for render nodes: every node's `prepare` once, then per
    /// camera view every node's `prepare_view` with the targets its slots draw with there (and
    /// `prewarm` for new ones); the views' work bits, the post groups' Distortion triggers,
    /// the last primary scene view and the resolved depths this frame needs.
    pub(super) fn prepare_nodes(&mut self, cx: NodePrepare<'_>) {
        let f = self;
        let NodePrepare {
            device,
            queue,
            encoder,
            world,
            resources,
            assets,
            caps,
            graphics,
            layouts,
            uploader,
            stats,
            nodes,
            overlay_depth,
            frame_index,
            timing,
        } = cx;
        let n = nodes.len();
        f.node_masks.clear();
        f.node_masks.resize(f.views.len() * n, 0);
        for pg in &mut f.post_groups {
            let camera = pg.camera;
            pg.view = f.views.iter().position(|v| v.camera == camera);
            pg.distortion = false;
        }
        // The HdrScene overlays' view: the last primary scene view.
        f.hdr_scene_view = f.views.iter().rposition(|v| {
            matches!(v.target, super::views::ViewTarget::Primary) && v.role == crate::node::ViewRole::Scene
        });
        let info = f.info;
        if n > 0 {
            for c in &nodes.nodes {
                lock_node(&c.node).prepare(&mut crate::node::PrepareContext {
                    device,
                    queue,
                    encoder,
                    world,
                    resources,
                    assets,
                    frame: info,
                    views: &f.view_infos,
                    caps,
                    graphics,
                    layouts,
                    uploader,
                    stats,
                    timing,
                });
            }
        }
        let resolve = graphics.msaa_depth_resolve;
        for v in 0..f.views.len() {
            let t = f.view_targets(v, graphics, None);
            let (mode, distortion_group, samples) = (t.mode, t.distortion_group, t.scene.samples);
            for i in 0..n {
                let c = &nodes.nodes[i];
                f.slot_targets.clear();
                f.slot_targets.extend(c.slots.iter().map(|slot| t.slot(slot)));
                let valid = t.valid(c);
                let mask = lock_node(&c.node).prepare_view(&mut crate::node::ViewPrepareContext {
                    device,
                    queue,
                    encoder,
                    world,
                    view: &f.view_infos[v],
                    frame: info,
                    targets: &f.slot_targets,
                    graphics,
                    uploader,
                    timing,
                });
                let mask = mask.0 & valid;
                f.node_masks[v * n + i] = mask;
                if mask != 0 {
                    nodes.prewarm(device, i, mask, &f.slot_targets);
                }
            }
            let work = nodes.work(&f.node_masks[v * n..(v + 1) * n]);
            let fx = f.post_groups.iter().any(|p| p.view == Some(v) && p.fx.any());
            let work = if fx { work | work::SCREEN_FX } else { work };
            f.views[v].work = work;
            if let Some(p) = distortion_group
                && work & work::DISTORTION != 0
            {
                f.post_groups[p].distortion = true;
            }
            // Resolved depth: for a SceneRead pass after a multisampled one, and for the
            // Distortion writers and HdrScene overlays reading this view's depth.
            let wants_1x = (mode == ReadMode::Resolved && work & work::READING != 0)
                || work & (work::DISTORTION | work::SCREEN_FX) != 0
                || (overlay_depth && f.hdr_scene_view == Some(v));
            if samples > 1 && resolve && wants_1x {
                let target = group_target(&f.views[v].target);
                let size = f.views[v].size;
                f.ensure_resolved_depth(device, target, size, frame_index, stats);
            }
        }
        f.evict_resolved_depth(frame_index);
    }
}

/// The pass targets a view's node slots draw with (graphics spec §1.6, §1.8).
#[derive(Clone, Copy, Debug)]
pub(super) struct ViewTargets {
    pub scene: PassTarget,
    /// Depth-reading Transparent slots: the SceneRead pass, or the scene pass (in-pass).
    pub read: PassTarget,
    pub mode: ReadMode,
    /// Distortion slots (a `DepthAccess::None` target where they cannot draw).
    pub distortion: PassTarget,
    /// The post group whose settings camera this view is, when its Distortion phase may run.
    pub distortion_group: Option<usize>,
}

impl ViewTargets {
    /// The target of `slot`.
    pub fn slot(&self, slot: &PhaseSlot) -> PassTarget {
        match slot.phase {
            RenderPhase::Transparent if slot.reads.contains(SceneReads::DEPTH) => self.read,
            RenderPhase::Distortion => self.distortion,
            _ => self.scene,
        }
    }

    /// The slots of `node` that can draw in this view (Distortion slots only in a settings
    /// camera with distortion enabled).
    pub fn valid(&self, node: &CachedNode) -> u32 {
        let all = if node.slots.len() >= 32 {
            u32::MAX
        } else {
            (1u32 << node.slots.len()) - 1
        };
        if self.distortion_group.is_some() {
            all
        } else {
            all & !node.bits[SlotList::Distortion as usize]
        }
    }
}

impl super::FrameState {
    /// The targets of view `v`'s node slots this frame (its scene pass in `format` when given:
    /// a capture's format).
    pub(super) fn view_targets(
        &self,
        v: usize,
        graphics: &crate::caps::GraphicsCaps,
        format: Option<wgpu::TextureFormat>,
    ) -> ViewTargets {
        let (pass_format, samples, output) = self.view_pass(v);
        let format = format.unwrap_or(pass_format);
        let view = &self.views[v];
        let single = self.groups[view.group].scene_views <= 1 || view.role == crate::node::ViewRole::Display;
        let (rods, resolve) = (graphics.read_only_depth, graphics.msaa_depth_resolve);
        let mode = read_mode(samples, rods, resolve, single);
        let scene = scene_target(format, samples, output);
        let read = match mode {
            ReadMode::Attached | ReadMode::Resolved => read_target(format, output, true),
            ReadMode::Sampled => read_target(format, output, false),
            ReadMode::InPass => scene,
        };
        // Distortion slots draw only in the settings camera of a post group with distortion
        // enabled (never in passthrough groups), with depth when a single-sample one exists.
        let distortion_group = self.post_groups.iter().position(|pg| {
            pg.view == Some(v) && !pg.passthrough && pg.settings.enabled && pg.settings.distortion.enabled
        });
        let depth_1x = samples <= 1 || resolve;
        ViewTargets {
            scene,
            read,
            mode,
            distortion: distortion_target(distortion_group.is_some() && depth_1x),
            distortion_group,
        }
    }
}

/// The target group key of a view's target.
pub(super) fn group_target(target: &super::views::ViewTarget) -> super::plan::GroupTarget {
    match target {
        super::views::ViewTarget::Primary => super::plan::GroupTarget::Primary,
        super::views::ViewTarget::Texture { handle, .. } => super::plan::GroupTarget::Texture(*handle),
    }
}

/// A group-1 bind group with a depth texture bound.
struct DepthEntry {
    bind_group: wgpu::BindGroup,
    generation: u64,
    last_used: u64,
}

/// Group 1 bind groups ([`FrameLayouts::scene`]): the far dummy (`DepthAccess::None`) and one
/// per depth texture read this frame (created on first use, dropped 3 frames after the last);
/// and the depth resolve's bind groups. The exposure binding is the 0 EV dummy until auto
/// exposure exists.
pub(crate) struct SceneInputs {
    exposure: wgpu::TextureView,
    /// Nearest, clamped: how scene depth is sampled (binding 2).
    sampler: wgpu::Sampler,
    /// Group 1 with the far dummy.
    pub none: wgpu::BindGroup,
    by_depth: FastMap<wgpu::TextureView, DepthEntry>,
    resolves: FastMap<wgpu::TextureView, (wgpu::BindGroup, u64)>,
    generation: u64,
    frame: u64,
}

/// A 1×1 `R32Float` texture holding `value`.
fn constant_texture(device: &wgpu::Device, queue: &wgpu::Queue, label: &str, value: f32) -> wgpu::TextureView {
    use wgpu::util::DeviceExt;
    device
        .create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &value.to_le_bytes(),
        )
        .create_view(&Default::default())
}

fn scene_bind_group(
    device: &wgpu::Device,
    layouts: &FrameLayouts,
    depth: &wgpu::TextureView,
    exposure: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("space3d scene inputs"),
        layout: &layouts.scene,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(depth),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(exposure),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

impl SceneInputs {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, layouts: &FrameLayouts) -> Self {
        let far_depth = constant_texture(device, queue, "space3d far depth", 1.0);
        let exposure = constant_texture(device, queue, "space3d exposure (0 EV)", 0.0);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("space3d scene depth"),
            ..Default::default()
        });
        let none = scene_bind_group(device, layouts, &far_depth, &exposure, &sampler);
        Self {
            exposure,
            sampler,
            none,
            by_depth: FastMap::default(),
            resolves: FastMap::default(),
            generation: 0,
            frame: 0,
        }
    }

    pub fn begin_frame(&mut self, frame: u64) {
        self.frame = frame;
    }

    /// Group 1 with `depth` bound (a single-sample `Depth32Float` view) and its generation.
    pub fn with_depth(
        &mut self,
        device: &wgpu::Device,
        layouts: &FrameLayouts,
        depth: &wgpu::TextureView,
    ) -> (&wgpu::BindGroup, u64) {
        let frame = self.frame;
        if !self.by_depth.contains_key(depth) {
            self.generation += 1;
            let bind_group = scene_bind_group(device, layouts, depth, &self.exposure, &self.sampler);
            self.by_depth.insert(
                depth.clone(),
                DepthEntry {
                    bind_group,
                    generation: self.generation,
                    last_used: frame,
                },
            );
        }
        let e = self.by_depth.get_mut(depth).expect("inserted");
        e.last_used = frame;
        (&e.bind_group, e.generation)
    }

    /// The depth resolve's bind group reading the multisampled `depth`.
    pub fn resolve(
        &mut self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        depth: &wgpu::TextureView,
    ) -> &wgpu::BindGroup {
        let frame = self.frame;
        if !self.resolves.contains_key(depth) {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("space3d depth resolve"),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(depth),
                }],
            });
            self.resolves.insert(depth.clone(), (bg, frame));
        }
        let e = self.resolves.get_mut(depth).expect("inserted");
        e.1 = frame;
        &e.0
    }

    /// Drop the bind groups of depth textures not read for 3 frames (they keep their
    /// textures alive).
    pub fn evict(&mut self) {
        let frame = self.frame;
        self.by_depth.retain(|_, e| e.last_used + 3 > frame);
        self.resolves.retain(|_, e| e.1 + 3 > frame);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{RenderNode, SlotMask};

    struct Slots(Vec<PhaseSlot>);
    impl RenderNode for Slots {
        fn slots(&self) -> &[PhaseSlot] {
            &self.0
        }
        fn draw(&mut self, _: &mut wgpu::RenderPass<'static>, _: &DrawContext<'_>) {}
    }

    #[test]
    fn read_modes_follow_the_depth_matrix() {
        use ReadMode::*;
        assert_eq!(read_mode(1, true, true, false), Attached);
        assert_eq!(read_mode(1, false, false, false), Sampled);
        assert_eq!(read_mode(4, true, true, true), Resolved);
        assert_eq!(read_mode(4, true, true, false), InPass, "multi-view MSAA");
        assert_eq!(read_mode(4, false, false, true), InPass, "GL + MSAA");
        assert_eq!(read_mode(4, true, false, true), InPass, "no resolve target");
    }

    #[test]
    fn schedule_classifies_and_sorts_slots() {
        let mut nodes = RenderNodes::default();
        nodes.insert(
            "a",
            Slots(vec![
                PhaseSlot::new(RenderPhase::Transparent, 5).reading(SceneReads::DEPTH),
                PhaseSlot::new(RenderPhase::Opaque, -1),
                PhaseSlot::new(RenderPhase::Transparent, -2),
            ]),
        );
        nodes.insert(
            "b",
            Slots(vec![
                PhaseSlot::new(RenderPhase::Opaque, -1),
                PhaseSlot::new(RenderPhase::Distortion, 0),
                PhaseSlot::new(RenderPhase::Skybox, 0),
            ]),
        );
        let mut cache = NodeCache::default();
        cache.sync(Some(&nodes));
        assert_eq!(cache.len(), 2);
        let order = |l: SlotList| {
            cache.lists[l as usize]
                .iter()
                .map(|r| (r.node, r.slot))
                .collect::<Vec<_>>()
        };
        assert_eq!(order(SlotList::Opaque), [(0, 1), (1, 0)], "registration breaks ties");
        assert_eq!(order(SlotList::Reading), [(0, 0)]);
        assert_eq!(order(SlotList::Transparent), [(0, 2)]);
        assert_eq!(order(SlotList::Distortion), [(1, 1)]);
        assert_eq!(order(SlotList::Skybox), [(1, 2)]);
        assert_eq!(cache.work(&[0, 0]), 0);
        assert_eq!(
            cache.work(&[SlotMask::ALL.0, 0]),
            work::OPAQUE_PRE | work::TRANSPARENT_PRE | work::READING
        );
        assert_eq!(cache.work(&[0, SlotMask::only(2).0]), work::SKYBOX);
        assert_eq!(cache.work(&[0, SlotMask::only(1).0]), work::DISTORTION);
        // Unchanged generation: nothing re-read; a new one re-reads and keeps prewarm records.
        cache.nodes[0]
            .prewarmed
            .push((1, scene_target(DEPTH_FORMAT, 1, OutputEncoding::Linear)));
        cache.sync(Some(&nodes));
        nodes.invalidate();
        cache.sync(Some(&nodes));
        assert_eq!(cache.nodes[0].prewarmed.len(), 1);
        cache.sync(None);
        assert_eq!(cache.len(), 0);
    }
}
