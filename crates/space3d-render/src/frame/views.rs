//! Per-view preparation: lights, `GpuView` writes, draw commands, CPU culling and the GPU
//! culling jobs and dispatches (graphics spec §1.4 step 8–9).

use super::draws::{DrawCmd, DrawKind, VertexSource};
use super::extract::SourceBuffer;
use super::plan::{ViewportRect, viewport_rect};
use super::{ARGS_SIZE, FrameState, WORKGROUP, binding, uniform_binding};
use crate::assets::RenderAssets;
use crate::components::{AmbientLight, DirectionalLight, PointLight, Projection, RenderTarget};
use crate::culling::{Aabb, Frustum, Sphere};
use crate::gpu::align_up;
use crate::handles::TextureHandle;
use crate::instance::{CULL_BLOCK, CullBounds, CullFloat, CullSource, InstanceBuffers, ScaleKind};
use crate::node::{ViewInfo, ViewRole};
use crate::output::{OutputEncoding, Tonemapper};
use crate::pipeline::{
    CULL_VIEWS, GpuCompactParams, GpuCullJob, GpuCullParams, GpuDirLight, GpuFrustum, GpuPointLight, GpuView,
    MAX_DIR_LIGHTS, MAX_POINT_LIGHTS, Pipelines,
};
use crate::renderer::Renderer;
use crate::stats::{DrawPath, RenderStats};
use crate::unproject::inverse_proj_rot;
use bytemuck::Zeroable;
use glam::{Mat4, Vec3, Vec4};
use space3d_core::{GlobalTransform, World};
use std::ops::Range;

/// The frame's directional lights, as written into every view's `GpuView` (the first
/// `MAX_DIR_LIGHTS` in query order). Point lights are collected into
/// `FrameState::point_lights` and chosen per view ([`select_point_lights`]).
#[derive(Clone, Copy)]
pub(super) struct SceneLights {
    pub(super) dir: [GpuDirLight; MAX_DIR_LIGHTS],
    pub(super) nd: usize,
}

impl SceneLights {
    /// The directional lights; every point light (in query order) goes into `points`.
    pub(super) fn collect(world: &World, points: &mut Vec<GpuPointLight>) -> Self {
        let mut dir = [GpuDirLight::zeroed(); MAX_DIR_LIGHTS];
        let mut nd = 0;
        for (gt, l) in world.query::<(&GlobalTransform, &DirectionalLight)>().iter() {
            if nd < MAX_DIR_LIGHTS {
                let d = (-gt.0.z_axis.truncate()).normalize_or(Vec3::NEG_Y);
                dir[nd] = GpuDirLight {
                    direction: d.extend(0.0).to_array(),
                    color: (l.color * l.intensity).extend(1.0).to_array(),
                };
                nd += 1;
            }
        }
        points.clear();
        for (gt, l) in world.query::<(&GlobalTransform, &PointLight)>().iter() {
            points.push(GpuPointLight {
                position_range: gt.0.w_axis.truncate().extend(l.range.max(1e-3)).to_array(),
                color: (l.color * l.intensity).extend(1.0).to_array(),
            });
        }
        Self { dir, nd }
    }
}

/// The point lights a view shades with (graphics spec D19): all of them, in query order, when
/// there are at most `MAX_POINT_LIGHTS` (exactly the lights of the original renderer); else the
/// `MAX_POINT_LIGHTS` nearest to the view's camera, still in query order (ties: query order).
/// `scratch` is reused. Returns the number written into `out`.
pub(super) fn select_point_lights(
    all: &[GpuPointLight],
    camera: Vec3,
    scratch: &mut Vec<(f32, u32)>,
    out: &mut [GpuPointLight; MAX_POINT_LIGHTS],
) -> usize {
    if all.len() <= MAX_POINT_LIGHTS {
        out[..all.len()].copy_from_slice(all);
        return all.len();
    }
    scratch.clear();
    scratch.extend(all.iter().enumerate().map(|(i, l)| {
        let p = Vec3::from_slice(&l.position_range[..3]);
        (p.distance_squared(camera), i as u32)
    }));
    let nearer = |a: &(f32, u32), b: &(f32, u32)| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1));
    scratch.select_nth_unstable_by(MAX_POINT_LIGHTS - 1, nearer);
    let chosen = &mut scratch[..MAX_POINT_LIGHTS];
    chosen.sort_unstable_by_key(|&(_, i)| i);
    for (slot, &(_, i)) in out.iter_mut().zip(chosen.iter()) {
        *slot = all[i as usize];
    }
    MAX_POINT_LIGHTS
}

#[derive(Clone)]
#[allow(
    clippy::large_enum_variant,
    reason = "kept in the reused `FrameState::views`: boxing would allocate every frame"
)]
pub(crate) enum ViewTarget {
    Primary,
    Texture {
        handle: TextureHandle,
        color: wgpu::TextureView,
        depth: wgpu::TextureView,
        /// Multisampled color resolved into `color` (`RenderTargetDesc::samples > 1`).
        msaa: Option<wgpu::TextureView>,
        format: wgpu::TextureFormat,
        samples: u32,
    },
}

#[derive(Clone)]
pub(crate) struct ViewData {
    pub(super) target: ViewTarget,
    /// The view's camera (index in `FrameState::cameras`).
    pub(super) camera: usize,
    /// The view's target group (`FrameState::groups`) and its role there.
    pub(super) group: usize,
    pub(super) role: ViewRole,
    /// How the view's fragments encode their color (the pipelines' `S3D_OUTPUT`). A capture
    /// replay into the offscreen target draws `ManualSrgb` views as `Linear`.
    pub(super) output: OutputEncoding,
    pub(super) clear: Option<Vec4>,
    /// The camera's viewport in whole pixels of the texture the view renders into (`S` for
    /// scene views of intermediate groups, else `D`), as drawn (see [`ViewportRect::fitted`]);
    /// `None` = the whole texture.
    pub(super) rect: Option<ViewportRect>,
    /// Size of that texture.
    pub(super) size: (u32, u32),
    pub(super) view_offset: u32,
    pub(super) draws: Range<usize>,
    /// First draw of the built-in blended batches (`draws.end` without any): node slots of the
    /// Opaque (order ≥ 0), Skybox and Transparent (order < 0) phases draw between.
    pub(super) opaque_end: usize,
    /// Which node lists have work in this view (`nodes::work` bits; set after `prepare_view`).
    pub(super) work: u8,
    /// Some draw of this view samples the view's own render target. A texture cannot be a
    /// color attachment and a sampled resource in the same pass, so such draws sample a copy
    /// of the target taken just before the pass (its previous content) instead.
    pub(super) feedback: bool,
    /// Its group-0 bind group (`FrameState::view_bind_group`; set once the views buffer is
    /// flushed).
    pub(super) bind_group: usize,
}

/// One mark and one scatter dispatch: the culling jobs of one source buffer, for the views of
/// one window of [`CULL_VIEWS`] frustums.
pub(super) struct Dispatch {
    pub(super) buffer: SourceBuffer,
    pub(super) params: GpuCullParams,
    pub(super) params_offset: u32,
    /// Window of frustums the dispatch's jobs read (bound at a fixed offset of
    /// `cull_params`: one bind group per source buffer and window).
    pub(super) window: u32,
    pub(super) wg: (u32, u32),
}

/// Bytes of one window of frustums (the `frusta` binding of cull.wgsl).
pub(super) const FRUSTA_WINDOW: usize = CULL_VIEWS * std::mem::size_of::<GpuFrustum>();

/// Workgroup grid for `n` workgroups within the 65535-per-dimension limit.
pub(super) fn dispatch_dims(n: u32) -> (u32, u32) {
    const MAX_DIM: u32 = 65535;
    if n <= MAX_DIM {
        (n, 1)
    } else {
        (MAX_DIM, n.div_ceil(MAX_DIM))
    }
}

/// ECS batches at or above this many instances are frustum-tested on all rayon workers.
#[cfg(not(target_arch = "wasm32"))]
const PARALLEL_CULL: usize = 16_384;

/// Frustum-test one ECS batch (`order` indexes `matrices`), one bit per instance in `mask`;
/// returns the number visible.
pub(super) fn cull_mask(
    matrices: &[[f32; 16]],
    order: &[u32],
    sphere: Sphere,
    frustum: &Frustum,
    mask: &mut Vec<u64>,
) -> usize {
    mask.clear();
    mask.resize(order.len().div_ceil(64), 0);
    let word = |w: &mut u64, items: &[u32]| {
        let mut bits = 0u64;
        for (k, &i) in items.iter().enumerate() {
            let m = Mat4::from_cols_array(&matrices[i as usize]);
            bits |= (frustum.intersects_sphere(&sphere.transformed(&m)) as u64) << k;
        }
        *w = bits;
        bits.count_ones() as usize
    };
    #[cfg(not(target_arch = "wasm32"))]
    if order.len() >= PARALLEL_CULL {
        use rayon::prelude::*;
        return mask
            .par_iter_mut()
            .zip(order.par_chunks(64))
            .with_min_len(16)
            .map(|(w, items)| word(w, items))
            .sum();
    }
    mask.iter_mut()
        .zip(order.chunks(64))
        .map(|(w, items)| word(w, items))
        .sum()
}

/// Transform a mesh-local sphere for culling instances of a given kind (mirror of cull.wgsl).
#[cfg(test)]
pub(super) fn instance_sphere(mesh: &Sphere, cull: CullSource, bytes: &[u8]) -> Option<Sphere> {
    let f = |o: u32| -> f32 {
        let o = o as usize;
        f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]])
    };
    let h = |o: u32| -> f32 {
        let o = o as usize;
        crate::instance::f16_to_f32(u16::from_le_bytes([bytes[o], bytes[o + 1]]))
    };
    let v = |o: u32, format: CullFloat| match format {
        CullFloat::F32 => Vec3::new(f(o), f(o + 4), f(o + 8)),
        CullFloat::F16 => Vec3::new(h(o), h(o + 2), h(o + 4)),
    };
    match cull {
        CullSource::Matrix { offset } => {
            let m = Mat4::from_cols_array(&std::array::from_fn(|i| f(offset + i as u32 * 4)));
            Some(mesh.transformed(&m))
        }
        CullSource::TranslationScale {
            translation,
            format,
            scale,
        } => {
            let s = match scale {
                None => 1.0,
                Some((o, ScaleKind::Uniform, CullFloat::F32)) => f(o).abs(),
                Some((o, ScaleKind::Uniform, CullFloat::F16)) => h(o).abs(),
                Some((o, ScaleKind::Vector, sf)) => v(o, sf).abs().max_element(),
            };
            Some(Sphere {
                center: v(translation, format),
                radius: (mesh.center.length() + mesh.radius) * s,
            })
        }
        CullSource::None => None,
    }
}

/// `(kind, t_off, s_off, s_kind)` of a [`GpuCullJob`] (see `Job` in cull.wgsl): translation
/// offset in words, scale offset in bytes (f16 scales may sit in the upper half of a word).
pub(super) fn cull_job_encoding(cull: CullSource) -> (u32, u32, u32, u32) {
    match cull {
        CullSource::Matrix { offset } => (0, offset / 4, 0, 0),
        CullSource::TranslationScale {
            translation,
            format,
            scale,
        } => {
            let kind = match format {
                CullFloat::F32 => 1,
                CullFloat::F16 => 2,
            };
            let (s_off, s_kind) = match scale {
                None => (0, 0),
                Some((o, ScaleKind::Uniform, CullFloat::F32)) => (o, 1),
                Some((o, ScaleKind::Vector, CullFloat::F32)) => (o, 2),
                Some((o, ScaleKind::Uniform, CullFloat::F16)) => (o, 3),
                Some((o, ScaleKind::Vector, CullFloat::F16)) => (o, 4),
            };
            (kind, translation / 4, s_off, s_kind)
        }
        CullSource::None => unreachable!("not GPU-culled"),
    }
}

/// The frame-owned fields of a view's `GpuView` (graphics spec §3.3): matrices, viewport,
/// target size, depth range and output encoding. `proj` is the projection as drawn (with the
/// viewport fit); `rect` the drawn viewport (`None`: the whole `size` target).
pub(super) fn write_view_info(
    gv: &mut GpuView,
    projection: &Projection,
    view: Mat4,
    proj: Mat4,
    rect: Option<ViewportRect>,
    size: (u32, u32),
    output: OutputEncoding,
) {
    let (near, far, ortho) = match *projection {
        Projection::Perspective { near, far, .. } => (near, far, 0.0),
        Projection::Orthographic { near, far, .. } => (near, far, 1.0),
    };
    let viewport = rect.map_or([0.0, 0.0, size.0 as f32, size.1 as f32], |r| {
        [r.x as f32, r.y as f32, r.w as f32, r.h as f32]
    });
    let (w, h) = (size.0.max(1) as f32, size.1.max(1) as f32);
    gv.view = view.to_cols_array_2d();
    gv.proj = proj.to_cols_array_2d();
    // Kept for compatibility: the f32 inverse carries the camera position and is badly
    // conditioned far from the origin. Directions (the sky) use `inv_proj_rot` instead.
    gv.inv_view_proj = (proj * view).inverse().to_cols_array_2d();
    gv.inv_proj = proj.inverse().to_cols_array_2d();
    gv.inv_proj_rot = inverse_proj_rot(view, proj).to_cols_array_2d();
    gv.viewport = viewport;
    gv.target_size = [w, h, 1.0 / w, 1.0 / h];
    gv.depth = [near, far, ortho, 0.0];
    gv.output = [output.id(), Tonemapper::None.id(), 0, 0];
    gv.exposure = [1.0, 0.0, 0.0, 0.0];
}

/// Instance counts of one view's sources (see [`FrameState::record_sources`]).
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct SourceCounts {
    /// Instances submitted for culling.
    pub instances: u64,
    /// Instances drawn: CPU-culled visible ones, and every GPU-culled one (an upper bound until
    /// the readback).
    pub drawn: u64,
    /// Instances rejected by CPU culling.
    pub culled: u64,
    pub triangles: u64,
    /// Of `instances` and `triangles`, the GPU-culled part.
    pub gpu_submitted: u64,
    pub gpu_triangles: u64,
}

impl FrameState {
    /// Record one view's draws of the frame's sources into `draws`, culled against `frustum`:
    /// on GPU paths a culling job (and indirect draw) per source, whose frustum slot this view
    /// takes, so the view shares the dispatches of every other view reading the same source
    /// buffers; on `CpuDirect` (and for sources the GPU culler cannot take) CPU culling and
    /// direct draws. `depth`: a depth view — only opaque sources whose material casts
    /// ([`super::extract::source_casts`]) are recorded. Returns the index of the view's first
    /// blended draw (`draws.len()` without any). `camera_view`: the camera's view matrix, for
    /// batches sorted back to front per instance (`TransparentSort::PerInstance`; `None` in
    /// depth views, which draw no blended sources).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_sources(
        &mut self,
        frustum: &Frustum,
        culling: bool,
        depth: bool,
        camera_view: Option<glam::Mat4>,
        buffers: &mut InstanceBuffers,
        out_cursor: &mut u64,
        visible_overflow: &mut bool,
        counts: &mut SourceCounts,
        cpu_reallocs: &mut u32,
    ) -> usize {
        let f = self;
        let path = f.path;
        let gpu = path.gpu_culling();
        let multi = matches!(path, DrawPath::GpuMultiDraw | DrawPath::GpuMultiDrawCount);
        let mut opaque_end = None;
        let jobs_start = f.jobs.len();
        // Current multi-draw run: (layout, class, material, run_base, draw cmd index, first args).
        let mut run: Option<(u16, u8, u32, u64, usize, u32)> = None;

        for si in 0..f.sources.len() {
            let s = f.sources[si];
            if s.count == 0 {
                continue;
            }
            if depth && (s.blend || !super::extract::source_casts(&f.materials, s.material)) {
                continue;
            }
            counts.instances += s.count as u64;
            // Sources are sorted opaque first: the view's blended draws start here.
            if s.blend && opaque_end.is_none() {
                opaque_end = Some(f.draws.len());
            }
            let tris = s.triangles as u64;
            let alloc = s.mesh.alloc;
            let base_cmd = DrawCmd {
                layout: s.layout,
                class: s.class,
                material: s.material,
                shader: None,
                first_index: alloc.first_index,
                index_count: alloc.index_count,
                base_vertex: alloc.base_vertex as i32,
                vb: VertexSource::Ecs,
                vb_offset: 0,
                kind: DrawKind::Direct { instances: 0 },
            };

            // GPU culling writes the visible instances into this view's region of
            // `cull_out`, which is bound as one storage binding: a region that would end
            // past it falls back to CPU culling for this view.
            let run_key = (s.layout, s.class, s.material);
            let same_run = multi && matches!(run, Some((l, c, m, ..)) if (l, c, m) == run_key);
            let region_start = if multi && !same_run {
                align_up(*out_cursor, 256)
            } else {
                *out_cursor
            };
            let fits = region_start + s.count as u64 * s.stride as u64 <= f.max_storage_binding;
            // Instances sorted back to front per view (`TransparentSort::PerInstance`) are
            // culled and ordered on the CPU on every path.
            let sort = camera_view.filter(|_| f.materials.sorts(&s));
            if gpu && s.cull != CullSource::None && fits && sort.is_none() {
                // GPU culling: reserve an output region, emit a job and an indirect draw.
                if multi && !same_run {
                    *out_cursor = region_start;
                    let cmd_index = f.draws.len();
                    let kind = if path == DrawPath::GpuMultiDrawCount {
                        DrawKind::MultiCount {
                            args_first: f.args_count,
                            count: 0,
                            run: f.run_count,
                        }
                    } else {
                        DrawKind::Multi {
                            args_first: f.args_count,
                            count: 0,
                        }
                    };
                    f.draws.push(DrawCmd {
                        vb: VertexSource::CullOut,
                        vb_offset: *out_cursor,
                        kind,
                        ..base_cmd
                    });
                    run = Some((s.layout, s.class, s.material, *out_cursor, cmd_index, f.args_count));
                    f.run_count += 1;
                }
                let region = *out_cursor;
                *out_cursor += s.count as u64 * s.stride as u64;
                let draw = f.args_count;
                f.args_count += 1;
                let first_instance = match run {
                    Some((.., base, _, _)) if multi => ((region - base) / s.stride as u64) as u32,
                    _ => 0,
                };
                let template = wgpu::util::DrawIndexedIndirectArgs {
                    index_count: alloc.index_count,
                    instance_count: 0,
                    first_index: alloc.first_index,
                    base_vertex: alloc.base_vertex as i32,
                    first_instance,
                };
                f.templates.push(template.as_bytes(), cpu_reallocs);
                if let Some((.., cmd, _)) = run
                    && multi
                {
                    match &mut f.draws[cmd].kind {
                        DrawKind::Multi { count, .. } | DrawKind::MultiCount { count, .. } => *count += 1,
                        _ => {}
                    }
                }
                if !multi {
                    f.draws.push(DrawCmd {
                        vb: VertexSource::CullOut,
                        vb_offset: region,
                        kind: DrawKind::Indirect { args: draw },
                        ..base_cmd
                    });
                }
                let (kind, t_off, s_off, s_kind) = cull_job_encoding(s.cull);
                let sp = s.mesh.sphere;
                f.jobs.push((
                    s.buffer,
                    GpuCullJob {
                        sphere: [sp.center.x, sp.center.y, sp.center.z, sp.radius],
                        src_word: s.first * s.stride / 4,
                        count: s.count,
                        stride_words: s.stride / 4,
                        kind,
                        t_off,
                        s_off,
                        s_kind,
                        dst_word: (region / 4) as u32,
                        draw,
                        wg_start: 0,
                        // Set once the view's frustum slot is known (below).
                        view: 0,
                        pad: 0,
                    },
                ));
                // Upper bound until the readback of the visible count (see `render_frame`).
                counts.drawn += s.count as u64;
                counts.triangles += s.count as u64 * tris;
                counts.gpu_submitted += s.count as u64;
                counts.gpu_triangles += s.count as u64 * tris;
                continue;
            }

            // CPU path (or buffers the GPU culler cannot decode).
            run = None;
            let mut visible = 0u32;
            match s.buffer {
                SourceBuffer::Ecs => {
                    let range = s.first as usize..(s.first + s.count) as usize;
                    let order = if f.ecs_overflow {
                        &f.ecs_cut[range]
                    } else {
                        &f.ecs_order[range]
                    };
                    visible = if culling {
                        cull_mask(&f.ecs_matrices, order, s.mesh.sphere, frustum, &mut f.cull_mask) as u32
                    } else {
                        s.count
                    };
                    // `cpu_visible` holds every view's partially visible batches and
                    // cannot exceed `max_buffer_size`: a batch that no longer fits is
                    // drawn whole from `ecs` (unculled, so the image is the same).
                    if visible < s.count
                        && visible > 0
                        && f.cpu_visible.len() as u64 + visible as u64 * 64 > f.max_buffer_size
                    {
                        *visible_overflow = true;
                        visible = s.count;
                    }
                    let sorted_fits = f.cpu_visible.len() as u64 + visible as u64 * 64 <= f.max_buffer_size;
                    if let Some(camera_view) = sort.filter(|_| visible > 0 && sorted_fits) {
                        // Back to front: the visible instances in view-depth order, copied
                        // (diffed) into `cpu_visible`.
                        let mask = culling.then_some(&f.cull_mask[..]);
                        let sorted =
                            f.materials
                                .sort_back_to_front(order, mask, &f.ecs_matrices, camera_view, cpu_reallocs);
                        let at = f.cpu_visible.len();
                        let matrices = &f.ecs_matrices;
                        f.cpu_visible.write_all(
                            at,
                            sorted.len(),
                            sorted.iter().map(|&i| &matrices[i as usize]),
                            cpu_reallocs,
                        );
                        f.draws.push(DrawCmd {
                            vb: VertexSource::CpuVisible,
                            vb_offset: at as u64,
                            kind: DrawKind::Direct { instances: visible },
                            ..base_cmd
                        });
                    } else if visible == s.count {
                        f.draws.push(DrawCmd {
                            vb: VertexSource::Ecs,
                            vb_offset: s.first as u64 * 64,
                            kind: DrawKind::Direct { instances: visible },
                            ..base_cmd
                        });
                    } else if visible > 0 {
                        // Partially visible: copy the visible matrices (diffed) into
                        // `cpu_visible`, in batch order.
                        let at = f.cpu_visible.len();
                        let (mask, matrices) = (&f.cull_mask, &f.ecs_matrices);
                        f.cpu_visible.write_all(
                            at,
                            visible as usize,
                            order
                                .iter()
                                .enumerate()
                                .filter(|(k, _)| mask[k / 64] >> (k % 64) & 1 != 0)
                                .map(|(_, &i)| &matrices[i as usize]),
                            cpu_reallocs,
                        );
                        f.draws.push(DrawCmd {
                            vb: VertexSource::CpuVisible,
                            vb_offset: at as u64,
                            kind: DrawKind::Direct { instances: visible },
                            ..base_cmd
                        });
                    }
                }
                SourceBuffer::Explicit(slot) => {
                    // Visible runs of culling blocks become one draw each, offset into
                    // the instance buffer (no `first_instance` needed).
                    let mut draw = |f: &mut FrameState, start: usize, end: usize| {
                        let end = end.min(s.count as usize);
                        f.draws.push(DrawCmd {
                            vb: VertexSource::Explicit(slot),
                            vb_offset: (start * s.stride as usize) as u64,
                            kind: DrawKind::Direct {
                                instances: (end - start) as u32,
                            },
                            ..base_cmd
                        });
                        visible += (end - start) as u32;
                    };
                    // Bounds are brought up to date here, lazily: only CPU culling reads
                    // them, and only blocks edited since the last refresh are recomputed.
                    let bounds = if culling {
                        let generation = f.explicit[slot as usize].as_ref().map_or(0, |g| g.generation);
                        buffers
                            .buffers
                            .get_mut(crate::handles::RawHandle {
                                index: slot,
                                generation,
                            })
                            .map(|ib| ib.cull_bounds())
                    } else {
                        None
                    };
                    let r = s.mesh.sphere.center.length() + s.mesh.sphere.radius;
                    // Empty boxes: no decodable positions, never culled.
                    let seen =
                        |aabb: &Aabb, scale: f32| aabb.is_empty() || frustum.intersects_aabb(&aabb.inflated(r * scale));
                    match bounds {
                        Some(CullBounds::User(aabb)) => {
                            if seen(&aabb, 1.0) {
                                draw(f, 0, s.count as usize);
                            }
                        }
                        Some(CullBounds::Blocks { total, blocks }) => {
                            if seen(&total.0, total.1) {
                                let mut start = None;
                                for (b, (aabb, scale)) in blocks.iter().enumerate() {
                                    if seen(aabb, *scale) {
                                        start.get_or_insert(b);
                                    } else if let Some(a) = start.take() {
                                        draw(f, a * CULL_BLOCK, b * CULL_BLOCK);
                                    }
                                }
                                if let Some(a) = start {
                                    draw(f, a * CULL_BLOCK, s.count as usize);
                                }
                            }
                        }
                        None => draw(f, 0, s.count as usize),
                    }
                }
            }
            counts.culled += (s.count - visible) as u64;
            counts.drawn += visible as u64;
            counts.triangles += visible as u64 * tris;
        }

        // This view's frustum, read by its culling jobs. The jobs of all views are grouped
        // into dispatches once every view is recorded (`prepare_dispatches`).
        if f.jobs.len() > jobs_start {
            // Without culling: planes everything is inside of, and no size test.
            let mut planes: GpuFrustum = [[0.0, 0.0, 0.0, 1.0]; 7];
            planes[6] = [0.0; 4];
            if culling {
                for (to, plane) in planes.iter_mut().zip(frustum.planes) {
                    *to = plane.to_array();
                }
                planes[6] = frustum.small.to_array();
            }
            let slot = f.cull_planes.len() as u32;
            f.cull_planes.push(planes);
            for (_, job) in &mut f.jobs[jobs_start..] {
                job.view = slot;
            }
        }
        opaque_end.unwrap_or(f.draws.len())
    }

    /// One `ViewData` per camera with a live target: its `GpuView` (kept in `gpu_views` until
    /// the lighting filled it), its draws and, on GPU paths, its culling jobs (frustum slot,
    /// output region). `out_cursor` gets the bytes of culled instances written by the jobs
    /// (`cull_out`), `visible_overflow` whether `cpu_visible` overflowed (see
    /// [`Self::finish_views`]).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare_views(
        &mut self,
        assets: &RenderAssets,
        buffers: &mut InstanceBuffers,
        lights: &SceneLights,
        ambient: AmbientLight,
        culling: bool,
        stats: &mut RenderStats,
        out_cursor: &mut u64,
        visible_overflow: &mut bool,
        reallocs: &mut u32,
    ) {
        let f = self;
        let mut cpu_reallocs = 0u32;
        let view_stride = f.view_stride();
        let SceneLights { dir, nd } = *lights;
        // ---- views -------------------------------------------------------------------
        f.views.clear();
        f.view_infos.clear();
        f.gpu_views.clear();
        f.draws.clear();
        #[cfg(test)]
        {
            f.max_direct_base_vertex = 0;
        }
        f.jobs.clear();
        f.dispatches.clear();
        f.compact = None;
        f.views_buf.begin();
        f.cpu_visible.begin();
        f.cull_params.begin();
        f.cull_jobs.begin();
        f.cull_planes.clear();
        f.templates.begin();
        f.run_meta.begin();
        f.compact_params.begin();
        f.args_count = 0;
        f.run_count = 0;
        f.gpu_submitted = 0;
        f.gpu_triangles = 0;
        f.wg_total = 0;
        f.scan = None;
        *out_cursor = 0;
        *visible_overflow = false;

        for ci in 0..f.cameras.len() {
            let cam = f.cameras[ci];
            let plan = f.camera_plans[ci];
            let Some(gi) = plan.group else {
                continue;
            };
            let target = match cam.camera.target {
                RenderTarget::Primary => ViewTarget::Primary,
                RenderTarget::Texture(h) => {
                    let Some(gpu) = assets.textures.get(h.raw()).and_then(|e| e.gpu.as_ref()) else {
                        continue;
                    };
                    let (Some(color), Some(depth)) = (&gpu.target_view, &gpu.depth_view) else {
                        continue;
                    };
                    ViewTarget::Texture {
                        handle: h,
                        color: color.clone(),
                        depth: depth.clone(),
                        msaa: gpu.msaa_view.clone(),
                        format: gpu.format,
                        samples: gpu.samples,
                    }
                }
            };
            let group = &f.groups[gi];
            let display_size = group.display_size;
            // Scene views of intermediate groups render into the group's scene texture `S`;
            // Direct views and display views into the display target `D`.
            let scene = group.intermediate() && plan.role == ViewRole::Scene;
            let size = if scene { group.scene_size } else { display_size };
            // The aspect of the view's rect in `D` (`S` is stretched back onto it).
            let display_rect = cam.camera.viewport.map(|v| viewport_rect(v, display_size));
            let (vw, vh) = display_rect.map_or((display_size.0 as f32, display_size.1 as f32), |r| {
                (r.w as f32, r.h as f32)
            });
            let aspect = (vw / vh.max(1.0)).max(1e-4);
            let rect = if scene {
                cam.camera.viewport.map(|v| viewport_rect(v, size))
            } else {
                display_rect
            };
            // A viewport past wgpu's viewport limits is drawn through its part within the target,
            // with a projection showing the same pixels there.
            let (rect, fit) = match rect {
                Some(r) => {
                    let (r, fit) = r.fitted(size, f.max_texture_2d);
                    (Some(r), fit)
                }
                None => (None, Mat4::IDENTITY),
            };
            let camera_view = cam.transform.inverse();
            let proj = fit * cam.camera.projection.matrix(aspect);
            let view_proj = proj * camera_view;
            let frustum = Frustum::from_view_proj(&view_proj);
            let view_offset = f.views.len() * view_stride;
            // How fragments encode their color, and the clear color in that encoding.
            let pg = plan.post_group.filter(|_| scene).map(|p| f.post_groups[p]);
            let (output, exposure, tonemapper) = match pg {
                Some(pg) => (pg.scene_output, pg.exposure, pg.tonemapper),
                None if group.manual_srgb => (OutputEncoding::ManualSrgb, 1.0, Tonemapper::None),
                None => (OutputEncoding::Linear, 1.0, Tonemapper::None),
            };
            let clear = match cam.camera.clear_color {
                // The Ldr fallback tonemaps per fragment: so do clears, on the CPU.
                Some(c) if output == OutputEncoding::Tonemapped => {
                    Some(tonemapper.apply(c.truncate() * exposure).extend(c.w))
                }
                c => c,
            };
            #[cfg(test)]
            let (output, exposure, tonemapper) = match f.force_output {
                Some(forced) => forced,
                None => (output, exposure, tonemapper),
            };
            let mut gv = GpuView::zeroed();
            gv.view_proj = view_proj.to_cols_array_2d();
            gv.camera_pos = cam.transform.w_axis.to_array();
            gv.ambient = (ambient.color * ambient.intensity).extend(1.0).to_array();
            gv.dir_lights = dir;
            let np = select_point_lights(
                &f.point_lights,
                cam.transform.w_axis.truncate(),
                &mut f.light_order,
                &mut gv.point_lights,
            );
            gv.counts = [nd as u32, np as u32, 0, 0];
            write_view_info(&mut gv, &cam.camera.projection, camera_view, proj, rect, size, output);
            if output == OutputEncoding::Tonemapped {
                gv.exposure = [exposure, exposure.log2(), 0.0, 0.0];
                gv.output[1] = tonemapper.id();
            }
            // Written once the lighting filled its fields (`Renderer::write_view_uniforms`).
            f.gpu_views.push(gv);

            let draws_start = f.draws.len();
            let mut counts = SourceCounts::default();
            let opaque_end = f.record_sources(
                &frustum,
                culling,
                false,
                Some(camera_view),
                buffers,
                out_cursor,
                visible_overflow,
                &mut counts,
                &mut cpu_reallocs,
            );
            stats.instances += counts.instances;
            stats.instances_drawn += counts.drawn;
            stats.culled += counts.culled;
            stats.triangles += counts.triangles;
            f.gpu_submitted += counts.gpu_submitted;
            f.gpu_triangles += counts.gpu_triangles;

            // (Scene views write `S`: they sample their render target directly.)
            let feedback = match &target {
                ViewTarget::Texture { handle, .. } if !scene => f.draws[draws_start..]
                    .iter()
                    .any(|d| f.materials.texture(d.material) == Some(*handle)),
                _ => false,
            };
            if scene {
                f.groups[gi].last_scene_view = Some(f.views.len());
            }
            // Views drawing into the group's scene-pass attachments (all views in Direct mode).
            if scene || !f.groups[gi].intermediate() {
                f.groups[gi].scene_views += 1;
            }
            let group = &f.groups[gi];
            let (color_format, samples) = if scene {
                (group.scene_format, group.samples)
            } else if group.intermediate() {
                (group.display_format, 1)
            } else {
                (group.display_format, group.display_samples)
            };
            let (near, far, orthographic) = match cam.camera.projection {
                Projection::Perspective { near, far, .. } => (near, far, false),
                Projection::Orthographic { near, far, .. } => (near, far, true),
            };
            f.view_infos.push(ViewInfo {
                camera: cam.id,
                order: cam.order,
                target: cam.camera.target,
                primary: matches!(cam.camera.target, RenderTarget::Primary),
                mode: group.mode,
                role: plan.role,
                view: camera_view,
                proj,
                view_proj,
                camera_pos: cam.transform.w_axis.truncate(),
                near,
                far,
                orthographic,
                frustum,
                viewport: rect.map_or([0, 0, size.0, size.1], |r| r.scissor(size)),
                scene_size: group.scene_size,
                target_size: display_size,
                render_scale: group.scale,
                samples,
                color_format,
                output,
                exposure,
                tonemapper,
                view_offset: view_offset as u32,
            });
            f.views.push(ViewData {
                target,
                camera: ci,
                group: gi,
                role: plan.role,
                output,
                clear,
                rect,
                size,
                view_offset: view_offset as u32,
                draws: draws_start..f.draws.len(),
                opaque_end,
                work: 0,
                feedback,
                bind_group: 0,
            });
        }
        stats.views = f.views.len() as u32;
        // The camera views' indirect args come first: only they feed the stats readback.
        f.camera_args = f.args_count;
        *reallocs += cpu_reallocs;
    }

    /// After every view (camera and depth views) is recorded: log a `cpu_visible` overflow
    /// once.
    pub(super) fn finish_views(&mut self, visible_overflow: bool) {
        if visible_overflow && !self.visible_overflow {
            tracing::warn!(
                max_buffer_size = self.max_buffer_size,
                "the partially visible ECS instances of all views exceed the device's \
                 max_buffer_size; the batches past it are drawn unculled"
            );
        }
        self.visible_overflow = visible_overflow;
    }

    /// Group the culling jobs of all views into dispatches (one mark and one scatter per
    /// source buffer and window of frustums), the compaction and the scan, and write their
    /// parameters. Returns the first word of the per-workgroup records in `args`.
    pub(super) fn prepare_dispatches(&mut self, reallocs: &mut u32) -> u32 {
        let f = self;
        let mut cpu_reallocs = 0u32;
        let path = f.path;
        let params_stride = f.params_stride();
        // Culling dispatches: every job reading the same source buffer, over all views (in
        // windows of CULL_VIEWS frustums), is one mark and one scatter dispatch — the dispatch
        // count depends on the number of source buffers, not on buffers × views. Jobs keep
        // their output regions and draws, so their order here does not change the image.
        f.jobs
            .sort_unstable_by_key(|(b, j)| (j.view as usize / CULL_VIEWS, *b, j.draw));
        let mut j = 0;
        while j < f.jobs.len() {
            let buffer = f.jobs[j].0;
            let window = f.jobs[j].1.view as usize / CULL_VIEWS;
            let job_start = (f.cull_jobs.len() / std::mem::size_of::<GpuCullJob>()) as u32;
            let mut wg = 0u32;
            let mut n = 0u32;
            while j < f.jobs.len() && f.jobs[j].0 == buffer && f.jobs[j].1.view as usize / CULL_VIEWS == window {
                let job = &mut f.jobs[j].1;
                // Frame-global record index: jobs of all dispatches share the records.
                job.wg_start = f.wg_total + wg;
                wg += job.count.div_ceil(WORKGROUP);
                let gpu_job = GpuCullJob {
                    view: job.view % CULL_VIEWS as u32,
                    ..*job
                };
                f.cull_jobs.push(bytemuck::bytes_of(&gpu_job), &mut cpu_reallocs);
                n += 1;
                j += 1;
            }
            f.wg_total += wg;
            let wgs = dispatch_dims(wg);
            f.dispatches.push(Dispatch {
                buffer,
                // `wg_word` is set below, once the record offset is known.
                params: GpuCullParams {
                    job_start,
                    job_count: n,
                    total_workgroups: wg,
                    workgroups_x: wgs.0,
                    wg_word: 0,
                    pad: [0; 3],
                },
                params_offset: 0,
                window: window as u32,
                wg: wgs,
            });
        }
        if path == DrawPath::GpuMultiDrawCount && f.run_count > 0 {
            // Draw compaction of every view's runs: one workgroup per run.
            for d in &f.draws {
                if let DrawKind::MultiCount { args_first, count, run } = d.kind {
                    f.run_meta.write_at(
                        run as usize * 8,
                        bytemuck::cast_slice(&[args_first, count]),
                        &mut cpu_reallocs,
                    );
                }
            }
            let wg = dispatch_dims(f.run_count);
            let p = GpuCompactParams {
                run_start: 0,
                run_count: f.run_count,
                workgroups_x: wg.0,
                pad: 0,
            };
            f.compact_params.write_at(0, bytemuck::bytes_of(&p), &mut cpu_reallocs);
            f.compact = Some(wg);
        }

        // Cull params: the frustums first, in whole windows of CULL_VIEWS (bound at fixed
        // offsets, in the same uniform buffer so culling binds no extra buffer), then one
        // params block per dispatch and the scan's (dynamic offsets; all offsets are
        // multiples of FRUSTA_WINDOW or `params_stride`, hence aligned). The per-workgroup
        // compaction records follow the indirect args.
        let wg_word = align_up(f.args_count as u64 * 5, 4) as u32;
        f.cull_params
            .write_at(0, bytemuck::cast_slice(&f.cull_planes), &mut cpu_reallocs);
        let params_base = f.cull_planes.len().div_ceil(CULL_VIEWS) * FRUSTA_WINDOW;
        f.cull_params.ensure_len(params_base, &mut cpu_reallocs);
        for (i, d) in f.dispatches.iter_mut().enumerate() {
            d.params.wg_word = wg_word;
            d.params_offset = (params_base + i * params_stride) as u32;
            f.cull_params.write_at(
                d.params_offset as usize,
                bytemuck::bytes_of(&d.params),
                &mut cpu_reallocs,
            );
        }
        let job_total = (f.cull_jobs.len() / std::mem::size_of::<GpuCullJob>()) as u32;
        if job_total > 0 {
            // One prefix-sum workgroup per job, over the jobs of every view.
            let wg = dispatch_dims(job_total);
            let p = GpuCullParams {
                job_start: 0,
                job_count: job_total,
                total_workgroups: job_total,
                workgroups_x: wg.0,
                wg_word,
                pad: [0; 3],
            };
            let off = params_base + f.dispatches.len() * params_stride;
            f.cull_params.write_at(off, bytemuck::bytes_of(&p), &mut cpu_reallocs);
            f.scan = Some((off as u32, wg));
        }
        *reallocs += cpu_reallocs;
        wg_word
    }

    /// Culling and compaction bind groups (rebuilt only when a buffer they bind changed).
    pub(super) fn prepare_cull_bind_groups(&mut self, device: &wgpu::Device, pipelines: &Pipelines, frame_index: u64) {
        let f = self;
        let gpu = f.path.gpu_culling();
        if gpu && let Some(cull) = &pipelines.cull {
            if frame_index.is_multiple_of(256) {
                // Bind groups of windows no longer dispatched would keep reallocated buffers
                // alive: drop the outdated ones.
                let (params, jobs, out, args) = (
                    f.cull_params.gpu.generation,
                    f.cull_jobs.gpu.generation,
                    f.cull_out.generation,
                    f.args.generation,
                );
                let (ecs, explicit) = (f.ecs.gpu.generation, &f.explicit);
                f.cull_bgs.retain(|(b, _), (_, g)| {
                    let src = match b {
                        SourceBuffer::Ecs => Some(ecs),
                        SourceBuffer::Explicit(slot) => explicit
                            .get(*slot as usize)
                            .and_then(Option::as_ref)
                            .map(|e| e.buf.generation),
                    };
                    src.is_some_and(|src| *g == [params, jobs, src, out, args])
                });
            }
            for d in 0..f.dispatches.len() {
                let buffer = f.dispatches[d].buffer;
                let src = match buffer {
                    SourceBuffer::Ecs => &f.ecs.gpu,
                    SourceBuffer::Explicit(slot) => &f.explicit[slot as usize].as_ref().expect("synced").buf,
                };
                let gens = [
                    f.cull_params.gpu.generation,
                    f.cull_jobs.gpu.generation,
                    src.generation,
                    f.cull_out.generation,
                    f.args.generation,
                ];
                let window = f.dispatches[d].window;
                if f.cull_bgs.get(&(buffer, window)).is_some_and(|(_, g)| *g == gens) {
                    continue;
                }
                let max = f.max_storage_binding;
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("space3d cull"),
                    layout: &cull.bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform_binding(f.cull_params.buffer(), std::mem::size_of::<GpuCullParams>()),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: binding(f.cull_jobs.buffer(), max),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: binding(src.get(), max),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: binding(f.cull_out.get(), max),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: binding(f.args.get(), max),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: f.cull_params.buffer(),
                                offset: window as u64 * FRUSTA_WINDOW as u64,
                                size: wgpu::BufferSize::new(FRUSTA_WINDOW as u64),
                            }),
                        },
                    ],
                });
                f.cull_bgs.insert((buffer, window), (bg, gens));
            }
            if f.compact.is_some() {
                let gens = [
                    f.compact_params.gpu.generation,
                    f.run_meta.gpu.generation,
                    f.args.generation,
                    f.counts.generation,
                    f.compact_args.generation,
                ];
                if f.compact_bg.as_ref().is_none_or(|(_, g)| *g != gens) {
                    let max = f.max_storage_binding;
                    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("space3d compact"),
                        layout: &cull.compact_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: uniform_binding(
                                    f.compact_params.buffer(),
                                    std::mem::size_of::<GpuCompactParams>(),
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: binding(f.run_meta.buffer(), max),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: binding(f.args.get(), max),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: binding(f.counts.get(), max),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: binding(f.compact_args.get(), max),
                            },
                        ],
                    });
                    f.compact_bg = Some((bg, gens));
                }
            }
        }
    }
}

impl Renderer {
    /// Reset indirect args from templates, then run culling (mark → scan → scatter, see
    /// `shaders/cull.wgsl`) and, for `GpuMultiDrawCount`, draw compaction.
    pub(super) fn encode_compute(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let f = &mut self.frame;
        if !f.path.gpu_culling() || f.dispatches.is_empty() {
            return;
        }
        let Some(cull) = &self.pipelines.cull else {
            return;
        };
        let args_bytes = f.args_count as u64 * ARGS_SIZE;
        encoder.copy_buffer_to_buffer(f.templates.buffer(), 0, f.args.get(), 0, args_bytes);
        if f.compact.is_some() {
            encoder.clear_buffer(f.counts.get(), 0, Some(align_up(f.run_count as u64 * 4, 4)));
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("space3d cull"),
            timestamp_writes: self.timer.compute("cull", None),
        });
        // Each dispatch is its own usage scope, so wgpu orders the passes' storage accesses.
        pass.set_pipeline(&cull.mark);
        for d in &f.dispatches {
            let (bg, _) = &f.cull_bgs[&(d.buffer, d.window)];
            pass.set_bind_group(0, bg, &[d.params_offset]);
            pass.dispatch_workgroups(d.wg.0, d.wg.1, 1);
            self.stats.compute_dispatches += 1;
        }
        if let Some((offset, wg)) = f.scan {
            // The scan only touches jobs and args, which every cull bind group shares.
            let d = &f.dispatches[0];
            let (bg, _) = &f.cull_bgs[&(d.buffer, d.window)];
            pass.set_pipeline(&cull.scan);
            pass.set_bind_group(0, bg, &[offset]);
            pass.dispatch_workgroups(wg.0, wg.1, 1);
            self.stats.compute_dispatches += 1;
        }
        pass.set_pipeline(&cull.scatter);
        for d in &f.dispatches {
            let (bg, _) = &f.cull_bgs[&(d.buffer, d.window)];
            pass.set_bind_group(0, bg, &[d.params_offset]);
            pass.dispatch_workgroups(d.wg.0, d.wg.1, 1);
            self.stats.compute_dispatches += 1;
        }
        if let Some((bg, _)) = &f.compact_bg
            && let Some(wg) = f.compact
        {
            pass.set_pipeline(&cull.compact);
            pass.set_bind_group(0, bg, &[0]);
            pass.dispatch_workgroups(wg.0, wg.1, 1);
            self.stats.compute_dispatches += 1;
        }
        drop(pass);
        // Visible counts for `RenderStats`, read back without waiting (see `readback.rs`).
        // (The camera views' args only: depth views' draws are not in `instances_drawn`.)
        self.uploader.reallocations += f.readback.record(
            &self.device,
            encoder,
            f.args.get(),
            f.camera_args,
            f.gpu_submitted,
            self.frame_index,
        );
    }

    /// Replace the upper bounds of GPU-culled batches in the stats with the newest read-back
    /// visible counts (of an earlier frame).
    pub(super) fn apply_gpu_cull_counts(&mut self) {
        let f = &mut self.frame;
        if f.gpu_submitted == 0 {
            f.readback.clear_latest();
            return;
        }
        if let Some(c) = f.readback.latest() {
            let s = &mut self.stats;
            s.instances_drawn = s.instances_drawn - f.gpu_submitted + c.visible;
            s.culled += c.submitted.saturating_sub(c.visible);
            s.triangles = s.triangles - f.gpu_triangles + c.triangles;
            s.gpu_counts_frame = Some(c.frame);
        }
    }
}
