//! Draw commands of the built-in batches and their encoding into a view's pass (material bind
//! groups, pipelines, vertex buffers, direct / indirect / multi-draw calls).

use super::views::{ViewData, ViewTarget};
use super::{ARGS_SIZE, FrameState, uniform_binding};
use crate::arena::{EXT_VERTEX_SIZE, VERTEX_SIZE};
use crate::assets::RenderAssets;
use crate::handles::{ShaderHandle, TextureHandle};
use crate::material::CLASS_EXT;
use crate::node::ViewRole;
use crate::output::OutputEncoding;
use crate::pipeline::{GpuMaterial, KIND_COLOR_LAYERED, KIND_DATA_LAYERED, PassKind, PipelineKey, Pipelines};
use crate::renderer::Renderer;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VertexSource {
    Ecs,
    Explicit(u32),
    CpuVisible,
    CullOut,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum DrawKind {
    Direct { instances: u32 },
    Indirect { args: u32 },
    Multi { args_first: u32, count: u32 },
    MultiCount { args_first: u32, count: u32, run: u32 },
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DrawCmd {
    pub(super) layout: u16,
    pub(super) class: u8,
    pub(super) material: u32,
    /// Custom shader of the material and its source version (resolved in `prepare`).
    pub(super) shader: Option<(ShaderHandle, u32)>,
    pub(super) first_index: u32,
    pub(super) index_count: u32,
    pub(super) base_vertex: i32,
    pub(super) vb: VertexSource,
    pub(super) vb_offset: u64,
    pub(super) kind: DrawKind,
}

/// What selects a draw's forward pipeline besides the pass's signature (format, samples,
/// output encoding): (instance layout, class, texture kinds, custom shader and its version).
pub(super) type PipelineState = (u16, u8, u8, Option<(ShaderHandle, u32)>);

/// Where vertex slot 0 starts in the mesh arena (in vertices) and the `base_vertex` to draw
/// with. Normally the whole arena is bound and draws select their mesh with `base_vertex`.
/// Without `DownlevelFlags::BASE_VERTEX` (WebGL2, GLES < 3.2) a non-zero base is not possible
/// in direct draws — wgpu does not reject it, and the GL backend then calls a missing entry
/// point (a panic on the web) — so the arena is bound from the mesh's first vertex instead
/// (`36 * n` bytes: always a valid 4-byte-aligned offset) and the draw uses base 0. Indirect
/// draws carry their base in the args; GPU paths are only chosen with `BASE_VERTEX`.
pub(super) fn direct_vertex_base(base_vertex: i32, direct: bool, supported: bool) -> (u32, i32) {
    if direct && !supported && base_vertex > 0 {
        (base_vertex as u32, 0)
    } else {
        (0, base_vertex)
    }
}

/// Material bind group entries of a texture set: the materials uniform and five maps (bound
/// texture or the default of the slot's kind) with their samplers. `replace` swaps the render
/// target a view draws into for its copy (or the default).
#[allow(clippy::too_many_arguments)]
fn set_bind_group(
    device: &wgpu::Device,
    pipelines: &Pipelines,
    assets_views: &[Option<(&wgpu::TextureView, &wgpu::Sampler)>; 5],
    kinds: u8,
    materials: &wgpu::Buffer,
    white: &wgpu::TextureView,
    white_array: &wgpu::TextureView,
    white_sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    let color = kinds & KIND_COLOR_LAYERED != 0;
    let data = kinds & KIND_DATA_LAYERED != 0;
    let slot = |i: usize| {
        let layered = if (1..=3).contains(&i) { data } else { color };
        assets_views[i].unwrap_or((if layered { white_array } else { white }, white_sampler))
    };
    let slots: [(&wgpu::TextureView, &wgpu::Sampler); 5] = std::array::from_fn(slot);
    let entry = |binding: u32, i: usize, sampler: bool| wgpu::BindGroupEntry {
        binding,
        resource: if sampler {
            wgpu::BindingResource::Sampler(slots[i].1)
        } else {
            wgpu::BindingResource::TextureView(slots[i].0)
        },
    };
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("space3d material"),
        layout: pipelines.material_bgl(kinds),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_binding(materials, std::mem::size_of::<GpuMaterial>()),
            },
            entry(1, 0, false),
            entry(2, 0, true),
            entry(3, 1, false),
            entry(4, 1, true),
            entry(5, 2, false),
            entry(6, 2, true),
            entry(7, 3, false),
            entry(8, 3, true),
            entry(9, 4, false),
            entry(10, 4, true),
        ],
    })
}

/// The bound textures of a resolved texture set (`None` for slots bound to the default).
fn set_views<'a>(
    set: &super::extract::TextureSetEntry,
    assets: &'a RenderAssets,
) -> [Option<(&'a wgpu::TextureView, &'a wgpu::Sampler)>; 5] {
    std::array::from_fn(|i| {
        let h = set.maps[i].filter(|_| set.available & (1 << i) != 0)?;
        let g = assets.textures.get(h.raw())?.gpu.as_ref()?;
        Some((&g.view, &g.sampler))
    })
}

impl FrameState {
    /// Resolve the custom shaders of this frame's draws and (re)build the bind group of every
    /// texture set they draw with (after the frame's uploads: a new materials buffer or texture
    /// version rebinds).
    pub(super) fn prepare_material_bind_groups(
        &mut self,
        device: &wgpu::Device,
        assets: &RenderAssets,
        pipelines: &mut Pipelines,
        white: &wgpu::TextureView,
        white_sampler: &wgpu::Sampler,
    ) {
        let f = self;
        let mgen = f.materials_buf.gpu.generation;
        for d in 0..f.draws.len() {
            // Custom shaders: resolve the current source version (stale handles: built-in).
            if let Some(h) = f.materials.shader(f.draws[d].material) {
                f.draws[d].shader = assets.shaders.get(h.raw()).map(|e| {
                    pipelines.sync_custom(h, e.version, &e.shader);
                    (h, e.version)
                });
            }
            let id = f.materials.set_of(f.draws[d].material) as usize;
            let Some(white_array) = f.materials.white_array.as_ref() else {
                continue;
            };
            let Some(set) = f.materials.sets.get(id) else {
                continue;
            };
            let valid = set
                .bind_group
                .as_ref()
                .is_some_and(|(_, g, v)| *g == mgen && *v == set.versions);
            if !valid {
                let views = set_views(set, assets);
                let bg = set_bind_group(
                    device,
                    pipelines,
                    &views,
                    set.kinds,
                    f.materials_buf.buffer(),
                    white,
                    white_array,
                    white_sampler,
                );
                let versions = set.versions;
                f.materials.sets[id].bind_group = Some((bg, mgen, versions));
            }
        }
        // Views drawing into a render target that their draws sample: those sets bind the
        // target's copy (views with `feedback`: the base color samples the target) or the
        // default instead — a texture cannot be sampled while it is the pass's attachment.
        let Some(white_array) = f.materials.white_array.clone() else {
            return;
        };
        for v in 0..f.views.len() {
            let Some(into) = drawn_target(f, &f.views[v]) else {
                continue;
            };
            let copy = if f.views[v].feedback {
                f.feedback.get(&into).map(|fb| fb.view.clone())
            } else {
                None
            };
            for d in f.views[v].draws.clone() {
                let id = f.materials.set_of(f.draws[d].material) as usize;
                let Some(set) = f.materials.sets.get(id) else {
                    continue;
                };
                if !binds_target(set, into) {
                    continue;
                }
                let current = set.own_target.as_ref().is_some_and(|(t, c, g, versions, _)| {
                    *t == into && *c == copy && *g == mgen && *versions == set.versions
                });
                if current {
                    continue;
                }
                let mut views = set_views(set, assets);
                for (i, view) in views.iter_mut().enumerate() {
                    if set.maps[i] == Some(into) {
                        *view = match (&copy, *view) {
                            (Some(c), Some((_, sampler))) => Some((c, sampler)),
                            _ => None,
                        };
                    }
                }
                let bg = set_bind_group(
                    device,
                    pipelines,
                    &views,
                    set.kinds,
                    f.materials_buf.buffer(),
                    white,
                    &white_array,
                    white_sampler,
                );
                let versions = set.versions;
                f.materials.sets[id].own_target = Some((into, copy.clone(), mgen, versions, bg));
            }
        }
    }
}

/// The render target a view's pass draws into (`None` for the primary target, and for scene
/// views of intermediate groups, which draw into the group's scene texture).
fn drawn_target(f: &FrameState, view: &ViewData) -> Option<TextureHandle> {
    match &view.target {
        ViewTarget::Texture { handle, .. }
            if !(view.role == ViewRole::Scene && f.groups[view.group].intermediate()) =>
        {
            Some(*handle)
        }
        _ => None,
    }
}

/// Whether a texture set binds `target` (the set's resolution bound it).
fn binds_target(set: &super::extract::TextureSetEntry, target: TextureHandle) -> bool {
    (0..5).any(|i| set.maps[i] == Some(target) && set.available & (1 << i) != 0)
}

impl FrameState {
    /// The material bind group `material`'s draws bind in group 1 and its texture kinds
    /// (`PipelineKey::kinds`), once `prepare_material_bind_groups` built it (depth views'
    /// alpha-masked casters).
    pub(super) fn material_bind_group(&self, material: u32) -> Option<(&wgpu::BindGroup, u8)> {
        let set = self.materials.sets.get(self.materials.set_of(material) as usize)?;
        set.bind_group.as_ref().map(|(bg, _, _)| (bg, set.kinds))
    }
}

impl Renderer {
    /// Record the draws `range` (within `view.draws`) of a view into its pass. Sets every
    /// state it uses (bind groups, buffers, pipelines), so node draws may precede it.
    pub(super) fn encode_draws(
        &mut self,
        pass: &mut wgpu::RenderPass<'_>,
        view: &ViewData,
        range: std::ops::Range<usize>,
        format: wgpu::TextureFormat,
        samples: u32,
        output: OutputEncoding,
    ) {
        self.encode_draws_as(pass, view, range, format, samples, output, PassKind::Forward);
    }

    /// [`Self::encode_draws`] with the pipelines of another pass kind (the SSR G-buffer pass).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_draws_as(
        &mut self,
        pass: &mut wgpu::RenderPass<'_>,
        view: &ViewData,
        range: std::ops::Range<usize>,
        format: wgpu::TextureFormat,
        samples: u32,
        output: OutputEncoding,
        pass_kind: PassKind,
    ) {
        let f = &self.frame;
        let (Some((vb, ib)), Some(view_bg)) = (&f.arena_buffers, f.view_bind_group(view.bind_group)) else {
            return;
        };
        if range.is_empty() {
            return;
        }
        pass.set_bind_group(0, view_bg, &[view.view_offset]);
        pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
        let material_stride = f.material_stride() as u32;
        let base_vertex_supported = self.caps.base_vertex;
        // The render target this pass draws into must not be sampled by it (see
        // `prepare_material_bind_groups`).
        let own_target = drawn_target(f, view);
        let globals = f.materials.globals;
        let ext_buffer = f.materials.ext_buffer.as_ref();
        let mut last_pipeline: Option<PipelineState> = None;
        let mut last_material = u32::MAX;
        let mut kinds = 0u8;
        let mut last_vb: Option<(VertexSource, u64)> = None;
        // First arena vertex bound to slot 0 (see `direct_vertex_base`).
        let mut slot0: Option<u32> = None;
        let mut slot2: Option<u32> = None;
        #[cfg(test)]
        let mut max_direct_base_vertex = self.frame.max_direct_base_vertex;
        for cmd in &f.draws[range] {
            // (Indirect draws take their counts from the args buffer.)
            let direct = matches!(cmd.kind, DrawKind::Direct { .. });
            if direct && cmd.index_count == 0 {
                continue;
            }
            if cmd.material != last_material {
                let Some(set) = f.materials.sets.get(f.materials.set_of(cmd.material) as usize) else {
                    continue;
                };
                let bg = match own_target {
                    Some(t) if binds_target(set, t) => set.own_target.as_ref().filter(|o| o.0 == t).map(|o| &o.4),
                    _ => set.bind_group.as_ref().map(|(bg, _, _)| bg),
                };
                let Some(bg) = bg else {
                    continue;
                };
                pass.set_bind_group(1, bg, &[cmd.material * material_stride]);
                self.stats.material_switches += 1;
                last_material = cmd.material;
                kinds = set.kinds;
            }
            // The extension stream (slot 2) of pipelines reading it.
            let ext = cmd.class & CLASS_EXT != 0;
            if ext && ext_buffer.is_none() {
                continue;
            }
            if last_pipeline != Some((cmd.layout, cmd.class, kinds, cmd.shader)) {
                let key = PipelineKey {
                    layout: cmd.layout,
                    class: cmd.class,
                    format,
                    samples,
                    output,
                    kinds,
                    shader: cmd.shader,
                    pass: pass_kind,
                    globals,
                };
                pass.set_pipeline(self.pipelines.forward(&self.device, key));
                self.stats.pipeline_switches += 1;
                last_pipeline = Some((cmd.layout, cmd.class, kinds, cmd.shader));
            }
            let (first_vertex, base_vertex) = direct_vertex_base(cmd.base_vertex, direct, base_vertex_supported);
            if slot0 != Some(first_vertex) {
                pass.set_vertex_buffer(0, vb.slice(first_vertex as u64 * VERTEX_SIZE..));
                slot0 = Some(first_vertex);
            }
            // (Rebased with slot 0 without `BASE_VERTEX`: the stream is in lockstep with it.)
            if let Some(b) = ext_buffer.filter(|_| ext && slot2 != Some(first_vertex)) {
                pass.set_vertex_buffer(2, b.slice(first_vertex as u64 * EXT_VERTEX_SIZE..));
                slot2 = Some(first_vertex);
            }
            if last_vb != Some((cmd.vb, cmd.vb_offset)) {
                let buffer = match cmd.vb {
                    VertexSource::Ecs => f.ecs.buffer(),
                    VertexSource::CpuVisible => f.cpu_visible.buffer(),
                    VertexSource::CullOut => f.cull_out.get(),
                    VertexSource::Explicit(slot) => match f.explicit.get(slot as usize) {
                        Some(Some(g)) => g.buf.get(),
                        _ => continue,
                    },
                };
                pass.set_vertex_buffer(1, buffer.slice(cmd.vb_offset..));
                last_vb = Some((cmd.vb, cmd.vb_offset));
            }
            let indices = cmd.first_index..cmd.first_index + cmd.index_count;
            match cmd.kind {
                DrawKind::Direct { instances } => {
                    #[cfg(test)]
                    {
                        max_direct_base_vertex = max_direct_base_vertex.max(base_vertex);
                    }
                    pass.draw_indexed(indices, base_vertex, 0..instances);
                }
                DrawKind::Indirect { args } => {
                    pass.draw_indexed_indirect(f.args.get(), args as u64 * ARGS_SIZE);
                    self.stats.indirect_draw_calls += 1;
                }
                DrawKind::Multi { args_first, count } => {
                    pass.multi_draw_indexed_indirect(f.args.get(), args_first as u64 * ARGS_SIZE, count);
                    self.stats.indirect_draw_calls += 1;
                    self.stats.multi_draw_calls += 1;
                }
                DrawKind::MultiCount { args_first, count, run } => {
                    pass.multi_draw_indexed_indirect_count(
                        f.compact_args.get(),
                        args_first as u64 * ARGS_SIZE,
                        f.counts.get(),
                        run as u64 * 4,
                        count,
                    );
                    self.stats.indirect_draw_calls += 1;
                    self.stats.multi_draw_calls += 1;
                }
            }
            self.stats.draw_calls += 1;
        }
        #[cfg(test)]
        {
            self.frame.max_direct_base_vertex = max_direct_base_vertex;
        }
    }
}
