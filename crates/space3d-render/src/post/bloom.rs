//! Bloom (graphics spec §5.2 PO1; design-post §5.3): chain sizing, the per-pass uniforms, the
//! pipelines and the CPU mirror of the soft threshold. `shaders/post/bloom.wgsl` holds the
//! passes; [`super::PostState`] plans the chain per post group and records it before the
//! composite, which adds level 0 after exposure (graphics spec §1.9).
//!
//! A post group's chain is one pooled texture (`TexturePool`, per-level views) of
//! `GraphicsCaps::bloom_format`: level 0 has the reference chain's heights (half the rect at
//! 1080p, capped at `Bloom::max_mip_height`; [`chain_size`]); every further level halves it.
//! Passes: one prefilter (S → exactly half the rect: exposure, soft threshold and the firefly
//! limit per pixel, then a 3 × 3 tent),
//! an area resample of its output into level 0 when level 0 is smaller (720p, 1440p, 4K, …;
//! never larger: 900p or High's 720 cap at 1080p use the half rect as level 0), `mips − 1`
//! downsamples and `mips − 1` upsamples blended by `scatter` — 11 render passes for the default
//! 6 levels at 1080p, 10 (with the resample) for the 5 of 720p. A rect shorter than
//! `Bloom::min_rect_height` gets no chain.

use super::PostEncode;
use super::api::Bloom;
use crate::frame::PoolDesc;
use crate::gpu::{TrackedBuffer, Uploader};
use crate::handles::FastMap;
use crate::node::FULLSCREEN_WGSL;
use crate::output::OUTPUT_WGSL;
use bytemuck::{Pod, Zeroable};
#[cfg(test)]
use glam::Vec3;
use std::borrow::Cow;

/// Source of the bloom passes (`shaders/post/bloom.wgsl`, after the output helpers and the
/// full-screen triangle).
pub(crate) const BLOOM_WGSL: &str = include_str!("../shaders/post/bloom.wgsl");

pub(crate) fn bloom_source() -> String {
    format!("{OUTPUT_WGSL}\n{FULLSCREEN_WGSL}\n{BLOOM_WGSL}")
}

/// Most levels a chain has (`Bloom::mip_count` is clamped to 1..=8).
pub(crate) const MAX_MIPS: u32 = 8;

/// Render passes of a chain with `mips` levels: the prefilter, the resample (when `resample`),
/// the downsamples and the upsamples.
pub(crate) const fn pass_count(mips: u32, resample: bool) -> u32 {
    2 * mips - 1 + resample as u32
}

/// Matches `BloomPass` in bloom.wgsl: one aligned uniform slot per pass.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct GpuBloomPass {
    /// Source rect in pixels of the source texture (prefilter: the post group's rect in `S`;
    /// chain passes: the whole source level).
    pub src_rect: [f32; 4],
    /// 1 / source width, 1 / source height, tap step in source pixels, exposure multiplier.
    pub src_texel: [f32; 4],
    /// Prefilter: the soft threshold t, t − k, 2k, 0.25 / k (k = t × soft_knee; all 0 but t
    /// for a hard cut). Resample: the destination pixel's box in source pixels (w, h, 0, 0).
    pub threshold: [f32; 4],
    /// x: multiply by 2^E (auto exposure). Resample: y, z = bilinear taps per axis.
    pub flags: [u32; 4],
}

/// A post group's bloom chain: level 0's size, the number of levels, how many of the reference
/// chain's finest levels level 0 stands for and the prefilter's own target ([`chain_size`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ChainSize {
    pub size: (u32, u32),
    pub mips: u32,
    /// Finest levels of the reference chain merged into level 0 (their weights too).
    pub merged: u32,
    /// The prefilter's target, exactly half the rect (rounded up), when level 0 has another
    /// size: an area resample turns it into level 0.
    pub half: Option<(u32, u32)>,
}

/// The chain of a post group whose rect in the scene texture is `rect` (pixels).
///
/// The reference chain has `mip_count` levels (1..=8) of heights `max_mip_height / 2^k`: what a
/// rect of height `2 × max_mip_height` or more gets (half its size, scaled down to the cap, so
/// the cost stays constant at 1440p and 4K). Level `k`'s blur spans the same fraction of the
/// rect at any resolution only if its height does not depend on the rect, so a smaller rect
/// keeps the reference heights too: it drops the `n` finest levels, where `2^n` is the power of
/// two nearest to `max_mip_height / (rect height / 2)`, and level 0 stands for them (resampled
/// to `max_mip_height / 2^n`, within √2 of half the rect, but never above half the rect: the
/// prefilter's resolution, so level 0 is never resampled up; their weights merge into it,
/// [`upsample_blend`]). The halo's size and energy relative to
/// the frame are then the same at 360p, 720p and 1080p (only the tight glow of the merged
/// levels widens), and smaller rects cost less. Levels are then dropped from the coarse end
/// while the smallest one has fewer than 2 pixels on its short side. Level `k` is
/// `max(1, level0 >> k)` (the texture's mip chain).
///
/// The prefilter's tent (3 × 3 pixels per destination pixel) weighs every source pixel alike
/// only at exactly 2× reduction: at 2.67× (720p, 1440p, 360p) or 4× (4K) a fixed footprint
/// leaves gaps, and a 1–2 px bright source gets a full, a double or no halo depending on its
/// position (moving, it blinks). So the prefilter always halves the rect exactly (`half`,
/// rounded up), and when level 0 has another size an area resample makes level 0 from it: each
/// level-0 pixel averages the pixels under its box, fractional ones by their overlap, so every
/// pixel keeps its energy wherever it sits. (Resampling S to twice level 0 before the prefilter
/// instead would keep its footprint at 1080p's fraction of the frame at 1440p and 4K, but a
/// small source split across resampled pixels then thresholds differently at every position:
/// its halo varied by half.)
pub(crate) fn chain_size(rect: (u32, u32), bloom: &Bloom) -> ChainSize {
    let (rw, rh) = (rect.0.max(1), rect.1.max(1));
    let cap = bloom.max_mip_height.max(1);
    let levels = bloom.mip_count.clamp(1, MAX_MIPS);
    let half = rh as f64 / 2.0;
    let merged = if half >= cap as f64 {
        0
    } else {
        ((cap as f64 / half).log2().round().max(0.0) as u32).min(levels - 1)
    };
    let height = ((cap as f64 / (1u64 << merged) as f64).round() as u32).max(1);
    let half_rect = (rw.div_ceil(2), rh.div_ceil(2));
    // Never above the prefilter's half-rect resolution: a bigger level 0 would be resampled up
    // from it, costing a larger target and a pass for no detail (High's 720 cap at 1080p). The
    // levels then stand within √2 of the reference heights (`merged` rounds to the nearest
    // power of two).
    let size = if height >= half_rect.1 {
        half_rect
    } else {
        (((rw as f64 * height as f64 / rh as f64).round() as u32).max(1), height)
    };
    let mut mips = levels - merged;
    while mips > 1 && (size.0 >> (mips - 1)).min(size.1 >> (mips - 1)) < 2 {
        mips -= 1;
    }
    ChainSize {
        size,
        mips,
        merged,
        half: (half_rect != size).then_some(half_rect),
    }
}

/// Bilinear taps per axis that average a box of `b` source pixels: it touches at most
/// `ceil(b) + 1` pixels (an aligned box `ceil(b)`, but rounding can shift it by a hair), two
/// per tap.
pub(crate) fn resample_taps(b: f32) -> u32 {
    ((b.ceil() + 1.0) / 2.0).ceil() as u32
}

/// Blend constant of the upsample into `target` (`dst = src × c + dst × (1 − c)`): `scatter`,
/// and `scatter^(merged + 1)` into level 0, so level 0 keeps the weights of the reference
/// chain's finest levels it stands for (Σ_{k ≤ merged} (1 − s)·s^k = 1 − s^(merged + 1)) and
/// every coarser level keeps its own, (1 − s)·s^k.
pub(crate) fn upsample_blend(scatter: f32, merged: u32, target: u32) -> f32 {
    let s = scatter.clamp(0.0, 1.0);
    if target == 0 { s.powi(merged as i32 + 1) } else { s }
}

/// Size of level `level` of a chain whose level 0 is `size`.
pub(crate) fn level_size(size: (u32, u32), level: u32) -> (u32, u32) {
    ((size.0 >> level).max(1), (size.1 >> level).max(1))
}

/// The soft threshold's uniform constants (t, t − k, 2k, 0.25 / k).
pub(crate) fn threshold_constants(threshold: f32, soft_knee: f32) -> [f32; 4] {
    let t = threshold.max(0.0);
    let k = t * soft_knee.clamp(0.0, 1.0);
    if k > 0.0 {
        [t, t - k, 2.0 * k, 0.25 / k]
    } else {
        [t, t, 0.0, 0.0]
    }
}

/// CPU mirror of `bloom_threshold` in bloom.wgsl: the part of the exposed color `c` that
/// blooms. Below `threshold − knee` nothing, above `threshold + knee` everything above the
/// threshold, a quadratic knee in between (continuous at both ends); scales the whole color by
/// the ratio on its brightest channel.
#[cfg(test)]
pub(crate) fn soft_threshold(c: Vec3, threshold: f32, soft_knee: f32) -> Vec3 {
    let [t, lo, knee2, quad] = threshold_constants(threshold, soft_knee);
    let br = c.max_element();
    let rq = (br - lo).clamp(0.0, knee2);
    let rq = quad * rq * rq;
    c * (rq.max(br - t) / br.max(1e-4))
}

/// Uniforms of every pass of a chain, in recording order: the prefilter, the resample into
/// level 0 (with `chain.half`), the downsamples (into level 1, 2, …) and the upsamples (into
/// level `mips − 2`, …, 0).
///
/// `rect` is the post group's rect in the scene texture of `scene_size`; `chain` the chain
/// ([`chain_size`]); `(exposure, auto)` the linear manual exposure multiplier (the compensation
/// with auto exposure) and whether the prefilter also multiplies by 2^E.
pub(crate) fn passes(
    scene_size: (u32, u32),
    rect: [u32; 4],
    chain: &ChainSize,
    bloom: &Bloom,
    (exposure, auto_exposure): (f32, bool),
    mut emit: impl FnMut(GpuBloomPass),
) {
    let (size, mips) = (chain.size, chain.mips);
    let texel = |(w, h): (u32, u32)| [1.0 / w.max(1) as f32, 1.0 / h.max(1) as f32];
    let [sx, sy] = texel(scene_size);
    // Half a destination texel, in source pixels (the 13-tap downsamples' step; the prefilter
    // always halves and loads its pixels).
    let step = |src_h: f32, dst_h: u32| 0.5 * src_h / dst_h.max(1) as f32;
    let prefiltered = chain.half.unwrap_or(size);
    emit(GpuBloomPass {
        src_rect: [rect[0] as f32, rect[1] as f32, rect[2] as f32, rect[3] as f32],
        src_texel: [sx, sy, step(rect[3] as f32, prefiltered.1), exposure],
        threshold: threshold_constants(bloom.threshold, bloom.soft_knee),
        flags: [u32::from(auto_exposure), 0, 0, u32::from(bloom.fast)],
    });
    if let Some((w, h)) = chain.half {
        // Each level-0 pixel's box in half-resolution pixels.
        let (bx, by) = (w as f32 / size.0.max(1) as f32, h as f32 / size.1.max(1) as f32);
        let [tx, ty] = texel((w, h));
        emit(GpuBloomPass {
            src_rect: [0.0, 0.0, w as f32, h as f32],
            src_texel: [tx, ty, 0.0, 1.0],
            threshold: [bx, by, 0.0, 0.0],
            flags: [0, resample_taps(bx), resample_taps(by), 0],
        });
    }
    let level = |l: u32| {
        let (w, h) = level_size(size, l);
        let [tx, ty] = texel((w, h));
        ([0.0, 0.0, w as f32, h as f32], tx, ty, h)
    };
    for l in 1..mips {
        let (src_rect, tx, ty, src_h) = level(l - 1);
        emit(GpuBloomPass {
            src_rect,
            src_texel: [tx, ty, step(src_h as f32, level_size(size, l).1), 1.0],
            ..GpuBloomPass::zeroed()
        });
    }
    for l in (0..mips - 1).rev() {
        let (src_rect, tx, ty, _) = level(l + 1);
        emit(GpuBloomPass {
            src_rect,
            src_texel: [tx, ty, 1.0, 1.0],
            ..GpuBloomPass::zeroed()
        });
    }
}

/// Which pass a bloom pipeline draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum BloomPassKind {
    Prefilter,
    Resample,
    Down,
    Up,
}

/// The four bloom pipelines of one chain format.
pub(crate) struct BloomPipelines {
    pub prefilter: wgpu::RenderPipeline,
    pub resample: wgpu::RenderPipeline,
    pub down: wgpu::RenderPipeline,
    pub up: wgpu::RenderPipeline,
}

impl BloomPipelines {
    pub(crate) fn new(
        device: &wgpu::Device,
        module: &wgpu::ShaderModule,
        layout: &wgpu::PipelineLayout,
        format: wgpu::TextureFormat,
    ) -> Self {
        let pipeline = |kind: BloomPassKind| {
            let (entry, blend) = match kind {
                BloomPassKind::Prefilter => ("fs_prefilter", None),
                BloomPassKind::Resample => ("fs_resample", None),
                BloomPassKind::Down => ("fs_down", None),
                // dst = src × scatter + dst × (1 − scatter) (scatter = the blend constant).
                BloomPassKind::Up => (
                    "fs_up",
                    Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::Constant,
                            dst_factor: wgpu::BlendFactor::OneMinusConstant,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                ),
            };
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("space3d bloom"),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some("vs_fullscreen"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        Self {
            prefilter: pipeline(BloomPassKind::Prefilter),
            resample: pipeline(BloomPassKind::Resample),
            down: pipeline(BloomPassKind::Down),
            up: pipeline(BloomPassKind::Up),
        }
    }

    pub(crate) fn get(&self, kind: BloomPassKind) -> &wgpu::RenderPipeline {
        match kind {
            BloomPassKind::Prefilter => &self.prefilter,
            BloomPassKind::Resample => &self.resample,
            BloomPassKind::Down => &self.down,
            BloomPassKind::Up => &self.up,
        }
    }
}

/// The bloom shader module.
pub(crate) fn module(device: &wgpu::Device) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("space3d bloom"),
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(bloom_source())),
    })
}

/// A post group's bloom chain this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct BloomPlan {
    pub format: wgpu::TextureFormat,
    /// Level 0 size and level count.
    pub size: (u32, u32),
    pub mips: u32,
    /// Reference levels merged into level 0 ([`ChainSize::merged`]).
    pub merged: u32,
    /// The prefilter's half-resolution target when level 0 is resampled from it
    /// ([`ChainSize::half`]).
    pub half: Option<(u32, u32)>,
    /// Uniform slot of the first pass (the others follow in recording order).
    pub first_slot: u32,
    pub scatter: f32,
}

impl BloomPlan {
    /// Bytes of the chain (every level) and of the prefilter's half-resolution target.
    pub fn bytes(&self) -> u64 {
        let texel = self.format.block_copy_size(None).unwrap_or(8) as u64;
        let (w, h) = self.half.unwrap_or((0, 0));
        texel * w as u64 * h as u64
            + (0..self.mips)
                .map(|l| {
                    let (w, h) = level_size(self.size, l);
                    texel * w as u64 * h as u64
                })
                .sum::<u64>()
    }

    /// Render passes of the chain ([`pass_count`]).
    pub fn passes(&self) -> u32 {
        pass_count(self.mips, self.half.is_some())
    }
}

/// GPU timestamps of a chain: query `begin` at the start of its first pass, `end` at the end of
/// its last one.
pub(crate) struct ChainTimestamps<'a> {
    pub query_set: &'a wgpu::QuerySet,
    pub begin: Option<u32>,
    pub end: Option<u32>,
}

/// Bloom resources shared by every post group.
pub(super) struct BloomState {
    /// `GpuBloomPass` of every pass planned this frame, one aligned slot each.
    pub params: TrackedBuffer,
    pub slot: usize,
    /// Slots written this frame.
    pub slots: u32,
    /// Group 0: pass uniforms (dynamic offset) + linear-clamp sampler; group 1: the source
    /// level (or `S`) and the exposure texture.
    params_bgl: wgpu::BindGroupLayout,
    source_bgl: wgpu::BindGroupLayout,
    layout: wgpu::PipelineLayout,
    module: wgpu::ShaderModule,
    pipelines: FastMap<wgpu::TextureFormat, BloomPipelines>,
    params_bg: Option<(wgpu::BindGroup, u32)>,
    /// Source bind groups by (0 = scene texture / 1 = pooled chain / 2 = the prefilter's
    /// half-resolution target, its generation, the exposure texture's generation for the scene
    /// and the level for the chain), with the frame they were last used.
    pub sources: FastMap<(u8, u64, u64), (wgpu::BindGroup, u64)>,
}

impl BloomState {
    pub fn new(device: &wgpu::Device, align: usize) -> Self {
        let params_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("space3d bloom params"),
            entries: &[
                crate::pipeline::uniform_entry(
                    0,
                    wgpu::ShaderStages::FRAGMENT,
                    true,
                    std::mem::size_of::<GpuBloomPass>() as u64,
                ),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let texture = |binding, filterable| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let source_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("space3d bloom source"),
            entries: &[texture(0, true), texture(1, false)],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("space3d bloom"),
            bind_group_layouts: &[Some(&params_bgl), Some(&source_bgl)],
            immediate_size: 0,
        });
        Self {
            params: TrackedBuffer::new("space3d bloom params", wgpu::BufferUsages::UNIFORM),
            slot: std::mem::size_of::<GpuBloomPass>().next_multiple_of(align),
            slots: 0,
            params_bgl,
            source_bgl,
            layout,
            module: module(device),
            pipelines: FastMap::default(),
            params_bg: None,
            sources: FastMap::default(),
        }
    }

    /// The pipelines of a chain in `format` (created on first use).
    pub fn pipelines(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat) -> &BloomPipelines {
        let (module, layout) = (&self.module, &self.layout);
        self.pipelines
            .entry(format)
            .or_insert_with(|| BloomPipelines::new(device, module, layout, format))
    }

    /// Upload this frame's pass uniforms (after every plan, before any encode).
    pub fn flush(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        up: &mut Uploader,
        sampler: &wgpu::Sampler,
    ) {
        if self.slots == 0 && self.params_bg.is_some() {
            return;
        }
        let mut reallocs = 0;
        self.params
            .ensure_len(std::mem::size_of::<GpuBloomPass>(), &mut reallocs);
        self.params.flush(device, encoder, up);
        let generation = self.params.gpu.generation;
        if self.params_bg.as_ref().is_none_or(|(_, g)| *g != generation) {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("space3d bloom params"),
                layout: &self.params_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: self.params.buffer(),
                            offset: 0,
                            size: wgpu::BufferSize::new(std::mem::size_of::<GpuBloomPass>() as u64),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            });
            self.params_bg = Some((bg, generation));
        }
    }

    /// Record a post group's chain (prefilter, resample, downsamples, upsamples) from `scene`
    /// (the scene texture, bind groups cached by `scene_generation`) into a pooled texture.
    /// Returns level 0's view and the texture's generation for the composite. `exposure`: the
    /// camera's exposure texture E (auto exposure; any 1×1 texture otherwise) and its generation.
    /// `timestamps`: GPU timestamps written at the start of the first pass and the end of the
    /// last one.
    pub fn encode(
        &mut self,
        cx: &mut PostEncode<'_>,
        (scene, scene_generation): (&wgpu::TextureView, u64),
        plan: &BloomPlan,
        (exposure, exposure_generation): (&wgpu::TextureView, u64),
        frame: u64,
        timestamps: Option<ChainTimestamps<'_>>,
    ) -> Option<(wgpu::TextureView, u64)> {
        use wgpu::TextureUsages as U;
        let (params_bg, _) = self.params_bg.as_ref()?;
        let pipelines = {
            let (module, layout) = (&self.module, &self.layout);
            self.pipelines
                .entry(plan.format)
                .or_insert_with(|| BloomPipelines::new(cx.device, module, layout, plan.format))
        };
        // Unit tests read level 0 back.
        let copy = if cfg!(test) { U::COPY_SRC } else { U::empty() };
        let desc = |size, mips| PoolDesc {
            size,
            format: plan.format,
            mips,
            usage: U::RENDER_ATTACHMENT | U::TEXTURE_BINDING | copy,
        };
        let index = cx.pool.acquire(cx.device, desc(plan.size, plan.mips));
        let half = plan.half.map(|size| cx.pool.acquire(cx.device, desc(size, 1)));
        let chain = cx.pool.get(index);
        let half = half.map(|i| cx.pool.get(i));
        let (sources, source_bgl) = (&mut self.sources, &self.source_bgl);
        let last = plan.passes() - 1;
        let resample = u32::from(half.is_some());
        for j in 0..=last {
            // Prefilter into level 0 (or the half-resolution texture, then resample that into
            // level 0), downsample into 1..mips, upsample into mips-2..=0.
            let (kind, source, target) = if j == 0 {
                let target = half.map_or(Target::Level(0), |_| Target::Half);
                (BloomPassKind::Prefilter, Target::Scene, target)
            } else if j <= resample {
                (BloomPassKind::Resample, Target::Half, Target::Level(0))
            } else if j - resample < plan.mips {
                let l = j - resample;
                (BloomPassKind::Down, Target::Level(l - 1), Target::Level(l))
            } else {
                let target = 2 * plan.mips - 2 - (j - resample);
                (BloomPassKind::Up, Target::Level(target + 1), Target::Level(target))
            };
            let view_of = |t: Target| match t {
                Target::Scene => scene,
                Target::Half => half.map_or(&chain.mips[0], |h| &h.mips[0]),
                Target::Level(l) => &chain.mips[l as usize],
            };
            let key = match source {
                Target::Scene => (0, scene_generation, exposure_generation),
                Target::Half => (2, half.map_or(0, |h| h.generation), 0),
                Target::Level(level) => (1, chain.generation, level as u64),
            };
            let view = view_of(source);
            let bg = sources.entry(key).or_insert_with(|| {
                let bg = cx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("space3d bloom source"),
                    layout: source_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(exposure),
                        },
                    ],
                });
                (bg, frame)
            });
            bg.1 = frame;
            let load = match kind {
                BloomPassKind::Up => wgpu::LoadOp::Load,
                _ => wgpu::LoadOp::Clear(wgpu::Color::BLACK),
            };
            let mut pass = cx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("space3d bloom"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: view_of(target),
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                timestamp_writes: timestamps.as_ref().and_then(|t| {
                    let begin = t.begin.filter(|_| j == 0);
                    let end = t.end.filter(|_| j == last);
                    (begin.is_some() || end.is_some()).then_some(wgpu::RenderPassTimestampWrites {
                        query_set: t.query_set,
                        beginning_of_pass_write_index: begin,
                        end_of_pass_write_index: end,
                    })
                }),
                ..Default::default()
            });
            pass.set_pipeline(pipelines.get(kind));
            pass.set_bind_group(0, params_bg, &[((plan.first_slot + j) as usize * self.slot) as u32]);
            pass.set_bind_group(1, &bg.0, &[]);
            if let (BloomPassKind::Up, Target::Level(target)) = (kind, target) {
                let s = upsample_blend(plan.scatter, plan.merged, target) as f64;
                pass.set_blend_constant(wgpu::Color { r: s, g: s, b: s, a: s });
            }
            pass.draw(0..3, 0..1);
            drop(pass);
            cx.stats.render_passes += 1;
            cx.stats.fullscreen_passes += 1;
            cx.stats.post_passes += 1;
        }
        Some((chain.mips[0].clone(), chain.generation))
    }
}

/// A texture a bloom pass reads or writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Target {
    /// The scene texture (the prefilter's source).
    Scene,
    /// The prefilter's half-resolution target ([`ChainSize::half`]).
    Half,
    /// A level of the chain.
    Level(u32),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_sizes() {
        let glow = Bloom::glow();
        let c = |rect| {
            let ChainSize { size, mips, merged, .. } = chain_size(rect, &glow);
            (size, mips, merged)
        };
        let ChainSize {
            size,
            mips,
            merged,
            half,
        } = chain_size((1920, 1080), &glow);
        assert_eq!((size, mips, merged, half), ((960, 540), 6, 0, None));
        let levels: Vec<_> = (0..mips).map(|l| level_size(size, l)).collect();
        assert_eq!(
            levels,
            [(960, 540), (480, 270), (240, 135), (120, 67), (60, 33), (30, 16)]
        );
        // 4K is capped: the same chain as 1080p.
        assert_eq!(c((3840, 2160)), ((960, 540), 6, 0));
        // 1440p: capped to 540 high.
        assert_eq!(c((2560, 1440)), ((960, 540), 6, 0));
        // Below 1080p the levels keep the reference heights (the halo keeps its size relative to
        // the frame): 720p drops the 540 level (270 is the power of two nearest to its 360),
        // 360p and the 180p goldens drop 2 and 3.
        assert_eq!(c((1280, 720)), ((480, 270), 5, 1));
        // 900p: 540 would be resampled up from 450: level 0 is the half rect itself.
        assert_eq!(c((1600, 900)), ((800, 450), 6, 0));
        assert_eq!(c((640, 360)), ((240, 135), 4, 2));
        assert_eq!(c((320, 180)), ((121, 68), 3, 3));
        for rect in [(1920, 1080), (1280, 720), (640, 360), (320, 180)] {
            let ChainSize { size, mips, .. } = chain_size(rect, &glow);
            let last = level_size(size, mips - 1).1;
            assert!((16..=17).contains(&last), "{rect:?}: smallest level {last} px high");
        }
        // The prefilter halves the rect exactly (rounded up); level 0 is resampled from that
        // unless it is that size (1080p; 2160p with a 1080 cap).
        let half = |rect, bloom: &Bloom| chain_size(rect, bloom).half;
        assert_eq!(half((1920, 1080), &glow), None);
        assert_eq!(half((960, 1080), &glow), None);
        assert_eq!(half((1280, 720), &glow), Some((640, 360)));
        assert_eq!(half((1600, 900), &glow), None);
        assert_eq!(half((2560, 1440), &glow), Some((1280, 720)));
        assert_eq!(half((3840, 2160), &glow), Some((1920, 1080)));
        assert_eq!(half((320, 180), &glow), Some((160, 90)));
        assert_eq!(half((321, 181), &glow), Some((161, 91)));
        assert_eq!(half((1920, 1079), &glow), None);
        let tall = Bloom {
            max_mip_height: 1080,
            ..glow
        };
        assert_eq!(half((3840, 2160), &tall), None);
        // Tiny and odd rects never produce 0-sized levels, and the last level keeps ≥ 2 px.
        for rect in [(1, 1), (3, 5), (17, 9), (64, 32), (101, 7), (4000, 3), (3, 4000)] {
            let ChainSize {
                size,
                mips,
                merged,
                half,
            } = chain_size(rect, &glow);
            assert!(size.0 >= 1 && size.1 >= 1 && (1..=6).contains(&mips), "{rect:?}");
            assert!(merged + mips <= 6, "{rect:?}");
            let last = level_size(size, mips - 1);
            assert!(mips == 1 || last.0.min(last.1) >= 2, "{rect:?}: {size:?} × {mips}");
            assert!(half.is_none_or(|(w, h)| w >= 1 && h >= 1), "{rect:?}");
        }
        assert_eq!(c((64, 32)), ((32, 16), 1, 5));
        assert_eq!(c((1, 1)), ((1, 1), 1, 5));
        assert_eq!(half((1, 1), &glow), None);
        // mip_count 0 and 20 clamp to 1 and 8.
        let b = |mip_count| Bloom { mip_count, ..glow };
        assert_eq!(chain_size((4096, 4096), &b(0)).mips, 1);
        assert_eq!(chain_size((4096, 4096), &b(20)).mips, 8);
        // Low quality: 4 levels below 270.
        let low = Bloom {
            mip_count: 4,
            max_mip_height: 270,
            ..glow
        };
        let ChainSize { size, mips, merged, .. } = chain_size((1440, 810), &low);
        assert_eq!((size, mips, merged), ((480, 270), 4, 0));
        // The quality tiers at 1080p: High (7 / 720) and Ultra (8 / 720) run at the half rect
        // (never a 1280 × 720 level 0 resampled up from 960 × 540), Ultra one level wider.
        for (mip_count, mips) in [(7, 7), (8, 8)] {
            let tier = Bloom {
                mip_count,
                max_mip_height: 720,
                ..glow
            };
            let ChainSize {
                size,
                mips: m,
                merged,
                half,
            } = chain_size((1920, 1080), &tier);
            assert_eq!(
                (size, m, merged, half),
                ((960, 540), mips, 0, None),
                "{mip_count} levels"
            );
        }
    }

    /// CPU mirror of `fs_resample` in bloom.wgsl along one axis: the bilinear taps (position,
    /// weight) of the destination pixel `j` of `dst` pixels over `src` source pixels.
    fn area_taps(j: u32, src: u32, dst: u32) -> Vec<(f32, f32)> {
        let b = src as f32 / dst as f32;
        let c = (j as f32 + 0.5) * b;
        let (lo, hi) = ((c - 0.5 * b).max(0.0), (c + 0.5 * b).min(src as f32));
        let first = lo.floor();
        (0..resample_taps(b))
            .map(|k| {
                let i = first + 2.0 * k as f32;
                let w0 = ((i + 1.0).min(hi) - i.max(lo)).max(0.0);
                let w1 = ((i + 2.0).min(hi) - (i + 1.0).max(lo)).max(0.0);
                let w = w0 + w1;
                (i + 0.5 + w1 / w.max(1e-6), w)
            })
            .collect()
    }

    /// The resample is an area average: summed over the destination pixels, every source
    /// pixel's weight (its share of each pixel it lands in) is exactly the destination/source
    /// area ratio, wherever it sits: no gaps, no double counting (a lone bright pixel keeps its
    /// energy at every position). Bilinear taps land between the two pixels they average.
    #[test]
    fn resample_keeps_every_pixel_energy() {
        for (src, dst) in [
            (720u32, 540u32),
            (1440, 1080),
            (2160, 1080),
            (360, 270),
            (180, 136),
            (1081, 1080),
            (4320, 1080),
            (3, 2),
            (450, 540),
        ] {
            let mut received = vec![0.0f64; src as usize];
            for j in 0..dst {
                let taps = area_taps(j, src, dst);
                let total: f32 = taps.iter().map(|t| t.1).sum();
                assert!(
                    (total - src as f32 / dst as f32).abs() < 1e-3,
                    "{src}->{dst} px {j}: {total}"
                );
                for (p, w) in taps {
                    if w == 0.0 {
                        continue;
                    }
                    // Bilinear: texel centers at +0.5.
                    let i = (p - 0.5).floor();
                    let f = p - 0.5 - i;
                    assert!((0.0..=1.0).contains(&f));
                    for (k, share) in [(i, 1.0 - f), (i + 1.0, f)] {
                        if share > 1e-6 {
                            assert!((0.0..src as f32).contains(&k), "{src}->{dst} px {j} reads {k}");
                            received[k as usize] += (w * share) as f64 / total as f64;
                        }
                    }
                }
            }
            let want = dst as f64 / src as f64;
            for (i, r) in received.iter().enumerate() {
                assert!(
                    (r - want).abs() < 2e-3,
                    "{src}->{dst}: pixel {i} weighs {r}, want {want}"
                );
            }
        }
        // Taps per axis: 1 up to a box of 1 pixel, 2 up to 3 pixels.
        assert_eq!(
            [0.833, 1.0, 1.333, 2.0, 2.5, 3.0, 4.0].map(resample_taps),
            [1, 1, 2, 2, 2, 2, 3]
        );
    }

    /// Level 0 of a chain that merged `n` reference levels carries their weights: the final
    /// weights of the levels equal the reference chain's (1 − s)·s^k, summed over the merged
    /// ones, and add up to 1.
    #[test]
    fn merged_levels_keep_their_weights() {
        let s = 0.7f32;
        let reference: Vec<f32> = (0..6)
            .map(|k| if k < 5 { (1.0 - s) * s.powi(k) } else { s.powi(5) })
            .collect();
        for merged in 0..6u32 {
            let mips = 6 - merged;
            // Level j's final weight: (1 − c_j) · Π_{i<j} c_i (the last level: Π c_i).
            let mut weights = Vec::new();
            let mut carried = 1.0;
            for j in 0..mips {
                if j + 1 == mips {
                    weights.push(carried);
                } else {
                    let c = upsample_blend(s, merged, j);
                    weights.push(carried * (1.0 - c));
                    carried *= c;
                }
            }
            let want0: f32 = reference[..=merged as usize].iter().sum();
            assert!((weights[0] - want0).abs() < 1e-6, "merged {merged}: {weights:?}");
            for j in 1..mips as usize {
                assert!(
                    (weights[j] - reference[j + merged as usize]).abs() < 1e-6,
                    "merged {merged}: {weights:?}"
                );
            }
            assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn soft_threshold_is_continuous_and_cuts() {
        let (t, knee) = (1.0, 0.5);
        let at = |v: f32| soft_threshold(Vec3::splat(v), t, knee).x;
        assert_eq!(at(0.4), 0.0, "below the knee");
        assert!(at(0.6) > 0.0 && at(0.6) < 0.01, "in the knee: {}", at(0.6));
        // Continuous at both ends of the knee, and ≥ c − t above it.
        let eps = 1e-3;
        assert!((at(0.5 - eps) - at(0.5 + eps)).abs() < 1e-4);
        assert!((at(1.5 - eps) - at(1.5 + eps)).abs() < 1e-2);
        assert!((at(8.0) - 7.0).abs() < 1e-5);
        // Monotonic.
        let mut last = 0.0;
        for i in 0..400 {
            let v = at(i as f32 * 0.01);
            assert!(v >= last - 1e-6, "not monotonic at {}", i as f32 * 0.01);
            last = v;
        }
        // Hard cut: nothing up to t, c − t above.
        assert_eq!(soft_threshold(Vec3::splat(0.99), 1.0, 0.0), Vec3::ZERO);
        assert!((soft_threshold(Vec3::splat(3.0), 1.0, 0.0).x - 2.0).abs() < 1e-6);
        // Threshold 0: everything blooms unchanged.
        let c = Vec3::new(0.2, 0.5, 0.1);
        assert!((soft_threshold(c, 0.0, 0.5) - c).abs().max_element() < 1e-6);
        // The ratio applies to the whole color (hue kept).
        let c = soft_threshold(Vec3::new(4.0, 2.0, 0.0), 1.0, 0.5);
        assert!((c.y / c.x - 0.5).abs() < 1e-6 && c.z == 0.0);
    }

    #[test]
    fn pass_uniforms() {
        let glow = Bloom::glow();
        let chain = chain_size((1920, 1080), &glow);
        let mips = chain.mips;
        let mut all = Vec::new();
        passes((1920, 1080), [0, 0, 1920, 1080], &chain, &glow, (2.0, false), |p| {
            all.push(p)
        });
        assert_eq!(all.len() as u32, pass_count(mips, false));
        assert_eq!(all.len(), 11);
        let pre = all[0];
        assert_eq!(pre.src_rect, [0.0, 0.0, 1920.0, 1080.0]);
        assert_eq!(pre.src_texel, [1.0 / 1920.0, 1.0 / 1080.0, 1.0, 2.0]);
        assert_eq!(pre.threshold, [1.0, 0.5, 1.0, 0.5]);
        assert_eq!(pre.flags, [0; 4]);
        // Downsample into level 1 reads level 0.
        assert_eq!(all[1].src_rect, [0.0, 0.0, 960.0, 540.0]);
        assert_eq!(all[1].src_texel[2], 1.0);
        // The first upsample reads the last level.
        assert_eq!(all[mips as usize].src_rect, [0.0, 0.0, 30.0, 16.0]);
        // The last upsample reads level 1.
        assert_eq!(all[10].src_rect, [0.0, 0.0, 480.0, 270.0]);

        // 720p (a viewport at y = 360 of a 1280 × 1080 scene texture): the prefilter halves
        // the rect exactly (step 1), the resample boxes 4/3 of its pixels into each level-0 one.
        let run = |scene, rect: [u32; 4]| {
            let chain = chain_size((rect[2], rect[3]), &glow);
            let mut all = Vec::new();
            passes(scene, rect, &chain, &glow, (2.0, false), |p| all.push(p));
            assert_eq!(all.len() as u32, pass_count(chain.mips, chain.half.is_some()));
            all
        };
        let all = run((1280, 1080), [0, 360, 1280, 720]);
        assert_eq!(all.len(), 10);
        let (pre, resample) = (all[0], all[1]);
        assert_eq!(pre.src_rect, [0.0, 360.0, 1280.0, 720.0]);
        assert_eq!(pre.src_texel, [1.0 / 1280.0, 1.0 / 1080.0, 1.0, 2.0]);
        assert_eq!(pre.threshold, [1.0, 0.5, 1.0, 0.5]);
        assert_eq!(resample.src_rect, [0.0, 0.0, 640.0, 360.0]);
        assert_eq!(resample.src_texel, [1.0 / 640.0, 1.0 / 360.0, 0.0, 1.0]);
        assert_eq!(resample.threshold, [4.0 / 3.0, 4.0 / 3.0, 0.0, 0.0]);
        assert_eq!(resample.flags, [0, 2, 2, 0]);
        // Then the chain as before: the first downsample reads level 0.
        assert_eq!(all[2].src_rect, [0.0, 0.0, 480.0, 270.0]);
        // 4K: 2 × 2 half-resolution pixels per level-0 pixel (2 taps per axis: robust to
        // rounding). 900p: level 0 is the half rect, no resample (it would be one up).
        let all = run((3840, 2160), [0, 0, 3840, 2160]);
        assert_eq!(all.len(), 12);
        assert_eq!(all[1].threshold, [2.0, 2.0, 0.0, 0.0]);
        assert_eq!(all[1].flags, [0, 2, 2, 0]);
        let all = run((1600, 900), [0, 0, 1600, 900]);
        assert_eq!(all.len(), 11, "prefilter, 5 downsamples, 5 upsamples");
        assert_eq!(all[1].src_rect, [0.0, 0.0, 800.0, 450.0]);
    }

    /// GPU milliseconds of one bloom chain over a 1920 × 1080 scene texture in `format` with the
    /// GPU loaded: after a warm-up, `rounds` command buffers of `chains` chains are submitted back
    /// to back (no idle between them); each round is timed from the start of its first pass to
    /// the end of its last, and the best round counts (other processes' GPU work only ever adds).
    /// `None` without an adapter or `TIMESTAMP_QUERY`.
    fn gpu_ms_per_chain(bloom: &Bloom, format: wgpu::TextureFormat, chains: u32, rounds: u32) -> Option<f64> {
        gpu_ms_per_chain_at((1920, 1080), bloom, format, chains, rounds)
    }

    fn gpu_ms_per_chain_at(
        scene_size: (u32, u32),
        bloom: &Bloom,
        format: wgpu::TextureFormat,
        chains: u32,
        rounds: u32,
    ) -> Option<f64> {
        use wgpu::TextureUsages as U;
        let r = pollster::block_on(crate::Renderer::new_headless(64, 64)).ok()?;
        if !r.graphics_caps().timestamps {
            return None;
        }
        let (device, queue) = (r.device(), r.queue());
        let texture = |size: (u32, u32), format, usage| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("bloom gate"),
                    size: wgpu::Extent3d {
                        width: size.0,
                        height: size.1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let scene = texture(
            scene_size,
            wgpu::TextureFormat::Rgba16Float,
            U::RENDER_ATTACHMENT | U::TEXTURE_BINDING,
        );
        let exposure = texture((1, 1), wgpu::TextureFormat::Rgba8Unorm, U::TEXTURE_BINDING);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let align = device.limits().min_uniform_buffer_offset_alignment as usize;
        let mut state = BloomState::new(device, align);
        let chain = chain_size(scene_size, bloom);
        let ChainSize {
            size,
            mips,
            merged,
            half,
        } = chain;
        let mut reallocs = 0;
        let mut n = 0;
        let slot = state.slot;
        let params = &mut state.params;
        let rect = [0, 0, scene_size.0, scene_size.1];
        passes(scene_size, rect, &chain, bloom, (1.0, false), |p| {
            params.write_at(n * slot, bytemuck::bytes_of(&p), &mut reallocs);
            n += 1;
        });
        state.slots = n as u32;
        let plan = BloomPlan {
            format,
            size,
            mips,
            merged,
            half,
            first_slot: 0,
            scatter: bloom.scatter,
        };
        let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("bloom gate"),
            ty: wgpu::QueryType::Timestamp,
            count: 2 * rounds,
        });
        let bytes = 16 * rounds as u64;
        let buffer = |usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("bloom gate"),
                size: bytes,
                usage,
                mapped_at_creation: false,
            })
        };
        let resolve = buffer(wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC);
        let read = buffer(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        let mut pool = crate::frame::TexturePool::default();
        let mut stats = crate::RenderStats::default();
        let mut up = Uploader::new(device);
        // Scene content (the passes have no data-dependent branches) and uniforms.
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &scene,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 4.0,
                        g: 2.0,
                        b: 1.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        state.flush(device, &mut encoder, &mut up, &sampler);
        up.finish();
        queue.submit([encoder.finish()]);
        let timer = crate::timing::GpuTimer::new(false, 1.0);
        let mut record = |encoder: &mut wgpu::CommandEncoder, timed: Option<u32>| {
            for i in 0..chains {
                pool.begin_frame(0);
                let mut cx = PostEncode {
                    device,
                    queue,
                    encoder,
                    stats: &mut stats,
                    pool: &mut pool,
                    timing: &timer,
                };
                let timestamps = timed.map(|round| ChainTimestamps {
                    query_set: &queries,
                    begin: (i == 0).then_some(2 * round),
                    end: (i + 1 == chains).then_some(2 * round + 1),
                });
                state
                    .encode(&mut cx, (&scene, 1), &plan, (&exposure, 0), 0, timestamps)
                    .expect("params");
                pool.release_all();
            }
        };
        // Warm-up, so the GPU runs at its loaded clocks.
        for _ in 0..30 {
            let mut encoder = device.create_command_encoder(&Default::default());
            record(&mut encoder, None);
            queue.submit([encoder.finish()]);
        }
        for round in 0..rounds {
            let mut encoder = device.create_command_encoder(&Default::default());
            record(&mut encoder, Some(round));
            queue.submit([encoder.finish()]);
        }
        // Resolve once every pass finished (Metal writes end-of-pass samples late).
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.resolve_query_set(&queries, 0..2 * rounds, &resolve, 0);
        encoder.copy_buffer_to_buffer(&resolve, 0, &read, 0, bytes);
        queue.submit([encoder.finish()]);
        read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let t: Vec<u64> = bytemuck::pod_collect_to_vec(&read.slice(..).get_mapped_range().expect("mapped"));
        read.unmap();
        let period = queue.get_timestamp_period() as f64 * 1e-6;
        let best = t
            .chunks(2)
            .map(|p| p[1].saturating_sub(p[0]) as f64 * period / chains as f64)
            .filter(|ms| *ms > 0.0)
            .fold(f64::MAX, f64::min);
        assert_eq!(r.gpu_error_count(), 0);
        Some(best)
    }

    /// Measurement for docs/graphics/post.md ("Cost"): the chain's GPU time (loaded, best of 3
    /// runs of 40 rounds) by rect size, with the default glow and the Low-quality chain. Run
    /// alone: `cargo test -p space3d-render --lib bloom_gpu_time_by_size -- --ignored --nocapture`.
    #[test]
    #[ignore = "perf measurement: run alone with --ignored"]
    fn bloom_gpu_time_by_size() {
        let glow = Bloom::glow();
        let low = Bloom {
            mip_count: 4,
            max_mip_height: 270,
            ..glow
        };
        let cases = [
            ("720p", (1280, 720), glow),
            ("1080p", (1920, 1080), glow),
            ("1440p", (2560, 1440), glow),
            ("4K", (3840, 2160), glow),
            ("Low 1440x810", (1440, 810), low),
        ];
        for (name, size, bloom) in cases {
            let Some(format) = pollster::block_on(crate::Renderer::new_headless(4, 4))
                .ok()
                .and_then(|r| r.graphics_caps().bloom_format)
            else {
                eprintln!("skipping: no adapter or no float targets");
                return;
            };
            let best = (0..3)
                .filter_map(|_| gpu_ms_per_chain_at(size, &bloom, format, 5, 40))
                .fold(f64::MAX, f64::min);
            eprintln!("{name}: {best:.3} ms GPU per chain ({format:?}, loaded)");
        }
    }

    /// Gate (graphics spec §4.4): bloom with 6 levels at 1080p ≤ 0.20 ms of GPU time on the dev
    /// machine (Apple M4 Max), loaded (measured 0.14–0.17 ms). Skips without `TIMESTAMP_QUERY`;
    /// `SPACE3D_PERF_BUDGET_SCALE` relaxes it on slower machines. Each run is the best of 40
    /// rounds; a run disturbed by other processes' GPU work (this machine is shared: 30–70 %
    /// device utilization from other apps reads 0.19–0.30 ms) is retried, best of 3 runs, like
    /// `gpu_100k_within_budget`. Ignored by default: next to the other GPU tests of this binary
    /// (run in parallel) the timing is not the chain's alone, and hosted CI runners cannot
    /// measure Dev-machine budgets. Run it alone on an idle Dev machine: `cargo test -p
    /// space3d-render --lib bloom_gate -- --ignored --nocapture`.
    #[test]
    #[ignore = "perf gate: run alone with --ignored"]
    fn bloom_gate_1080p() {
        const GATE_MS: f64 = 0.20;
        let glow = Bloom::glow();
        assert_eq!(chain_size((1920, 1080), &glow).mips, 6);
        let scale = std::env::var("SPACE3D_PERF_BUDGET_SCALE")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(1.0);
        let caps = pollster::block_on(crate::Renderer::new_headless(4, 4)).map(|r| r.graphics_caps().clone());
        let Some(format) = caps.ok().and_then(|c| c.bloom_format) else {
            eprintln!("skipping GPU test: no adapter or no float targets");
            return;
        };
        let mut best = f64::MAX;
        for run in 0..3 {
            for f in [format, wgpu::TextureFormat::Rgba16Float] {
                let Some(ms) = gpu_ms_per_chain(&glow, f, 5, 40) else {
                    eprintln!("skipping GPU timing: no TIMESTAMP_QUERY");
                    return;
                };
                eprintln!("run {run}: bloom 6 mips @1080p, {f:?} chain: {ms:.3} ms GPU (loaded)");
                if f == format {
                    best = best.min(ms);
                }
            }
            if best <= GATE_MS * scale {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        assert!(
            best <= GATE_MS * scale,
            "bloom {best:.3} ms > {:.3} ms",
            GATE_MS * scale
        );
    }

    /// Level 0 of the chain (after the upsamples) for a `size` scene texture holding `radiance`
    /// on the pixels of `lit`, black elsewhere, exposure 1: `(level 0 size, texels)`, read back
    /// in `Rgba16Float`. `None` without an adapter.
    fn level0_of(
        r: &crate::Renderer,
        size: (u32, u32),
        lit: &[(u32, u32)],
        radiance: f32,
        bloom: &Bloom,
    ) -> Option<((u32, u32), Vec<f32>)> {
        use wgpu::TextureUsages as U;
        let (device, queue) = (r.device(), r.queue());
        let scene_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("bloom motion scene"),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: U::TEXTURE_BINDING | U::COPY_DST,
            view_formats: &[],
        });
        let mut data = vec![0u16; (size.0 * size.1 * 4) as usize];
        let one = crate::instance::f32_to_f16(1.0);
        for i in 0..(size.0 * size.1) as usize {
            data[i * 4 + 3] = one;
        }
        let v = crate::instance::f32_to_f16(radiance);
        for &(x, y) in lit {
            let i = ((y * size.0 + x) * 4) as usize;
            data[i..i + 3].copy_from_slice(&[v; 3]);
        }
        queue.write_texture(
            scene_texture.as_image_copy(),
            bytemuck::cast_slice(&data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size.0 * 8),
                rows_per_image: None,
            },
            scene_texture.size(),
        );
        let scene = scene_texture.create_view(&Default::default());
        let exposure = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("bloom motion exposure"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: U::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let align = device.limits().min_uniform_buffer_offset_alignment as usize;
        let mut state = BloomState::new(device, align);
        let chain = chain_size(size, bloom);
        let (mut reallocs, mut n) = (0, 0);
        let slot = state.slot;
        let params = &mut state.params;
        passes(size, [0, 0, size.0, size.1], &chain, bloom, (1.0, false), |p| {
            params.write_at(n * slot, bytemuck::bytes_of(&p), &mut reallocs);
            n += 1;
        });
        state.slots = n as u32;
        let plan = BloomPlan {
            format: wgpu::TextureFormat::Rgba16Float,
            size: chain.size,
            mips: chain.mips,
            merged: chain.merged,
            half: chain.half,
            first_slot: 0,
            scatter: bloom.scatter,
        };
        let mut pool = crate::frame::TexturePool::default();
        let mut stats = crate::RenderStats::default();
        let mut up = Uploader::new(device);
        let mut encoder = device.create_command_encoder(&Default::default());
        state.flush(device, &mut encoder, &mut up, &sampler);
        let timer = crate::timing::GpuTimer::new(false, 1.0);
        let (level0, _) = {
            let mut cx = PostEncode {
                device,
                queue,
                encoder: &mut encoder,
                stats: &mut stats,
                pool: &mut pool,
                timing: &timer,
            };
            state.encode(&mut cx, (&scene, 1), &plan, (&exposure, 0), 0, None)?
        };
        let texture = level0.texture();
        let (w, h) = chain.size;
        let row = (w * 8).next_multiple_of(256);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bloom motion readback"),
            size: (row * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        up.finish();
        queue.submit([encoder.finish()]);
        buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let bytes = buffer.slice(..).get_mapped_range().expect("mapped");
        let mut out = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                let at = (y * row + x * 8) as usize;
                out.push(crate::instance::f16_to_f32(u16::from_le_bytes([
                    bytes[at],
                    bytes[at + 1],
                ])));
            }
        }
        assert_eq!(r.gpu_error_count(), 0);
        Some((chain.size, out))
    }

    /// The light bloom adds to a `size` frame: the composite's lookup of level 0 at every pixel
    /// center (composite.wgsl `post_bloom`: half a scene pixel on, cubic B-spline), summed, red
    /// channel.
    fn added_light(size: (u32, u32), level: (u32, u32), texels: &[f32]) -> f64 {
        let (lw, lh) = (level.0 as f32, level.1 as f32);
        let at = |x: i64, y: i64| {
            let x = x.clamp(0, level.0 as i64 - 1) as usize;
            let y = y.clamp(0, level.1 as i64 - 1) as usize;
            texels[y * level.0 as usize + x] as f64
        };
        let bspline = |t: f64| {
            let (t2, t3) = (t * t, t * t * t);
            [
                (1.0 - t).powi(3) / 6.0,
                (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
                (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0,
                t3 / 6.0,
            ]
        };
        let mut sum = 0.0;
        for py in 0..size.1 {
            for px in 0..size.0 {
                let u = (px as f32 + 1.0) / size.0 as f32 * lw - 0.5;
                let v = (py as f32 + 1.0) / size.1 as f32 * lh - 0.5;
                let (x0, y0) = (u.floor(), v.floor());
                let (wx, wy) = (bspline((u - x0) as f64), bspline((v - y0) as f64));
                for (j, wy) in wy.iter().enumerate() {
                    for (i, wx) in wx.iter().enumerate() {
                        sum += at(x0 as i64 - 1 + i as i64, y0 as i64 - 1 + j as i64) * wx * wy;
                    }
                }
            }
        }
        sum
    }

    /// Sub-pixel motion (the M1 review's shimmer finding): a `px` × `px` source of radiance
    /// `radiance`, rasterized (whole pixels), moved one pixel at a time over 8 positions of each
    /// axis (every phase of the prefilter's 2 × 2 blocks and of level 0's texels); returns the
    /// spread `(max − min) / mean` of the light bloom adds.
    fn motion_spread(r: &crate::Renderer, size: (u32, u32), px: u32, radiance: f32, bloom: &Bloom) -> Option<f32> {
        let mut added = Vec::new();
        for step in 0..8u32 {
            let (x0, y0) = (size.0 / 2 + step, size.1 / 2 + step / 2);
            let lit: Vec<(u32, u32)> = (0..px)
                .flat_map(|dy| (0..px).map(move |dx| (x0 + dx, y0 + dy)))
                .collect();
            let (level, texels) = level0_of(r, size, &lit, radiance, bloom)?;
            added.push(added_light(size, level, &texels));
        }
        let mean = added.iter().sum::<f64>() / added.len() as f64;
        let (lo, hi) = added
            .iter()
            .fold((f64::MAX, f64::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
        Some(((hi - lo) / mean) as f32)
    }

    /// The M1 review's shimmer finding: a small bright source moving over the pixel grid (every
    /// phase of the prefilter's 2 × 2 blocks and of level 0's texels) gets the same added light
    /// within 0.3 % (0.02–0.1 % measured; M1: 2 px sources varied by 11 %, 4 px ones by 4 %, at
    /// 720p, 1080p and 1440p). The per-pixel prefilter makes level 0's energy exact; the
    /// composite's B-spline magnification keeps it exact at 2.67× (bilinear left 3 %).
    #[test]
    fn bloom_is_stable_under_motion() {
        let Ok(r) = pollster::block_on(crate::Renderer::new_headless(8, 8)) else {
            eprintln!("skipping GPU test: no adapter available");
            return;
        };
        let glow = Bloom::glow();
        for size in [(1280, 720), (1920, 1080), (2560, 1440)] {
            let limit = 0.003;
            for (px, radiance) in [(1, 30.0), (2, 10.0), (3, 6.0), (4, 4.0)] {
                let spread = motion_spread(&r, size, px, radiance, &glow).expect("chain");
                eprintln!("{size:?}: {px} px of {radiance}: spread {:.2} %", spread * 100.0);
                assert!(spread <= limit, "{size:?}, {px} px: {spread}");
            }
        }
    }

    /// The glow is proportional to the energy above the threshold whatever the source's size
    /// (M1: 1–2 px sources got a third of it, 8 px ones 70 %), and a hot pixel is limited: a
    /// pixel of 500 glows like one of about 63.
    #[test]
    fn bloom_energy_is_proportional_and_limited() {
        let Ok(r) = pollster::block_on(crate::Renderer::new_headless(8, 8)) else {
            eprintln!("skipping GPU test: no adapter available");
            return;
        };
        let glow = Bloom::glow();
        let size = (1920, 1080);
        let added = |px: u32, radiance: f32| {
            let lit: Vec<(u32, u32)> = (0..px)
                .flat_map(|dy| (0..px).map(move |dx| (960 + dx, 540 + dy)))
                .collect();
            let (level, texels) = level0_of(&r, size, &lit, radiance, &glow).expect("chain");
            added_light(size, level, &texels)
        };
        for (px, radiance) in [(1u32, 30.0f32), (2, 10.0), (4, 4.0), (8, 4.0), (32, 4.0)] {
            let above = (px * px) as f64 * (radiance as f64 - 1.0);
            let ratio = added(px, radiance) / above;
            assert!((ratio - 1.0).abs() < 0.02, "{px} px of {radiance}: {ratio}");
        }
        let spike = added(1, 500.0);
        let limited = limit(Vec3::splat(499.0)).x as f64;
        assert!((spike / limited - 1.0).abs() < 0.02, "{spike} vs {limited}");
        assert!(limited < 64.0);
    }

    /// `Bloom::fast` (the low tiers' prefilter: 4 bilinear taps, one threshold on their mean): a
    /// large source keeps its glow (its inside is the same; only its rim, averaged with the
    /// dark around it, loses), a small one glows less but never more, and the pass uniforms
    /// carry the switch.
    #[test]
    fn the_fast_prefilter_keeps_large_sources_and_dims_small_ones() {
        let fast = Bloom::glow().with_fast(true);
        let chain = chain_size((1920, 1080), &fast);
        let mut all = Vec::new();
        passes((1920, 1080), [0, 0, 1920, 1080], &chain, &fast, (1.0, false), |p| {
            all.push(p)
        });
        assert_eq!(all[0].flags, [0, 0, 0, 1]);
        assert!(all[1..].iter().all(|p| p.flags[3] == 0));
        let Ok(r) = pollster::block_on(crate::Renderer::new_headless(8, 8)) else {
            eprintln!("skipping GPU test: no adapter available");
            return;
        };
        let size = (1920, 1080);
        let added = |px: u32, radiance: f32| {
            let lit: Vec<(u32, u32)> = (0..px)
                .flat_map(|dy| (0..px).map(move |dx| (960 + dx, 540 + dy)))
                .collect();
            let (level, texels) = level0_of(&r, size, &lit, radiance, &fast).expect("chain");
            added_light(size, level, &texels) / ((px * px) as f64 * (radiance as f64 - 1.0))
        };
        let (large, medium, small) = (added(32, 4.0), added(8, 4.0), added(2, 10.0));
        eprintln!("fast prefilter, glow kept: 32 px {large:.3}, 8 px {medium:.3}, 2 px {small:.3}");
        assert!((0.9..=1.02).contains(&large), "{large}");
        assert!((0.6..=1.02).contains(&medium), "{medium}");
        assert!((0.2..=1.02).contains(&small), "{small}");
    }

    /// CPU mirror of `bloom_limit` in bloom.wgsl.
    fn limit(c: Vec3) -> Vec3 {
        const KNEE: f32 = 32.0;
        let br = c.max_element();
        if br <= KNEE {
            return c;
        }
        let x = br - KNEE;
        c * ((KNEE + x / (1.0 + x / KNEE)) / br)
    }

    #[test]
    fn firefly_limit_is_continuous_and_bounded() {
        assert_eq!(limit(Vec3::new(20.0, 5.0, 1.0)), Vec3::new(20.0, 5.0, 1.0));
        let at = |v: f32| limit(Vec3::splat(v)).x;
        assert!((at(32.0 + 1e-3) - 32.0).abs() < 1e-2, "continuous at the knee");
        // Slope 1 at the knee, falling; never above 64.
        assert!((at(33.0) - at(32.0) - 1.0).abs() < 0.05);
        let mut last = 0.0;
        for v in [1.0, 10.0, 32.0, 40.0, 64.0, 100.0, 1e3, 1e5] {
            assert!(at(v) >= last && at(v) < 64.0, "{v}: {}", at(v));
            last = at(v);
        }
        // The hue is kept.
        let c = limit(Vec3::new(200.0, 100.0, 50.0));
        assert!((c.y / c.x - 0.5).abs() < 1e-6 && (c.z / c.x - 0.25).abs() < 1e-6);
    }

    #[test]
    fn gpu_bloom_pass_matches_wgsl() {
        let src = bloom_source();
        let module = naga::front::wgsl::parse_str(&src).unwrap();
        let (_, pass) = module
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some("BloomPass"))
            .unwrap();
        let naga::TypeInner::Struct { members, span } = &pass.inner else {
            panic!("BloomPass is a struct")
        };
        assert_eq!(*span as usize, std::mem::size_of::<GpuBloomPass>());
        let offset = |name: &str| members.iter().find(|m| m.name.as_deref() == Some(name)).unwrap().offset as usize;
        assert_eq!(offset("src_texel"), std::mem::offset_of!(GpuBloomPass, src_texel));
        assert_eq!(offset("threshold"), std::mem::offset_of!(GpuBloomPass, threshold));
        assert_eq!(offset("flags"), std::mem::offset_of!(GpuBloomPass, flags));
        crate::pipeline::validate_wgsl(&src).unwrap_or_else(|e| panic!("{e}"));
    }
}
