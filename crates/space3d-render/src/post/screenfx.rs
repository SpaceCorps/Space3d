//! City effects: fog, SSAO and SSR on the HDR scene color of a post group, before bloom and the
//! composite (`shaders/post/screenfx.wgsl`; docs/graphics/post.md, "City effects").
//!
//! The frame decides per post group whether the effects run ([`plan`]): at least one of them
//! enabled, `FrameMode::Hdr`, not the passthrough group, the settings camera's scene depth
//! available at one sample (multisampled depth is resolved for it, `work::SCREEN_FX`), and for SSR
//! a device with read-only depth plus the camera's G-buffer-lite target, which the frame fills
//! after the scene pass ([`ScreenFx::gbuffer_target`], pass kind `PassKind::GBuffer`).
//!
//! Passes of one group, recorded by [`ScreenFx::encode`] in `finish_group` before the post
//! chain: the AO pass and two blur passes (R8, half resolution by default, pooled), then one
//! apply pass `S → S'` (SSR, SSAO, fog; the pool's `S'` has the scene's format and size, the
//! group's rect is written, the rest cleared). The post chain then reads `S'` instead of `S`.
//! A camera with all three off records none of it and allocates nothing.

use super::api::{Fog, PostProcess, Ssao, Ssr};
use crate::frame::{PoolDesc, TexturePool};
use crate::handles::FastMap;
use crate::node::{FULLSCREEN_WGSL, FrameLayouts, FrameMode, VIEW_WGSL, scene_wgsl_for};
use crate::stats::RenderStats;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use space3d_core::Entity;

/// Source of the effects (after VIEW_WGSL, the scene group and FULLSCREEN_WGSL).
const SCREENFX_WGSL: &str = include_str!("../shaders/post/screenfx.wgsl");

/// Format of the AO targets.
pub(crate) const AO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;
/// Format of the SSR G-buffer-lite: octahedral world normal (rg), roughness (b), F0 (a, 0 = no
/// reflection).
pub(crate) const GBUFFER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Tag of the generation of the effects' output (see [`ScreenFx::encode`]).
const FX_GENERATION: u64 = 1 << 62;

const SLOT: u64 = 256;
const SLOTS: u64 = 64;

/// Effect bits of `Fx::flags.x` (`FX_*` in screenfx.wgsl).
pub(crate) mod bits {
    pub const FOG: u32 = 1;
    pub const AO: u32 = 2;
    pub const SSR: u32 = 4;
}

/// Matches `Fx` in screenfx.wgsl (192 bytes, one 256-byte slot per group).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct GpuFx {
    pub fog_color: [f32; 4],
    pub fog_shape: [f32; 4],
    pub fog_sun: [f32; 4],
    pub fog_sun_dir: [f32; 4],
    pub fog_sky: [f32; 4],
    pub ao_a: [f32; 4],
    pub ao_b: [f32; 4],
    pub ao_size: [f32; 4],
    pub ssr_a: [f32; 4],
    pub ssr_b: [f32; 4],
    pub flags: [u32; 4],
    pub rect: [f32; 4],
}

/// Which effects run for a post group this frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct FxPlan {
    pub fog: bool,
    pub ao: bool,
    pub ssr: bool,
}

impl FxPlan {
    pub fn any(self) -> bool {
        self.fog || self.ao || self.ssr
    }
}

/// What the effects of `s` can do for a post group in `mode`. `depth`: the settings camera's
/// single-sample scene depth can exist this frame; `ssr_caps`: the device draws the G-buffer
/// pass (read-only depth attachments). `passthrough`: the implicit Ldr group (never).
pub(crate) fn plan(s: &PostProcess, mode: FrameMode, passthrough: bool, depth: bool, ssr_caps: bool) -> FxPlan {
    if passthrough || !s.enabled || mode != FrameMode::Hdr || !depth {
        return FxPlan::default();
    }
    FxPlan {
        fog: s.fog.enabled,
        ao: s.ssao.enabled && s.ssao.intensity > 0.0,
        ssr: s.ssr.enabled && ssr_caps && s.ssr.intensity > 0.0,
    }
}

fn finite(v: f32, default: f32) -> f32 {
    if v.is_finite() { v } else { default }
}

/// The AO target size for a scene of `scene` pixels.
pub(crate) fn ao_size(ssao: &Ssao, scene: (u32, u32)) -> (u32, u32) {
    if ssao.half_res {
        (scene.0.div_ceil(2).max(1), scene.1.div_ceil(2).max(1))
    } else {
        scene
    }
}

impl GpuFx {
    /// Parameters of a group: `rect` is its (clipped) rect in scene pixels, `ao_size` the AO
    /// target size, `frame` the jitter seed of SSR.
    pub fn new(s: &PostProcess, plan: FxPlan, rect: [u32; 4], ao_size: (u32, u32), frame: u64) -> Self {
        let mut p = Self::zeroed();
        let mut flags = 0;
        if plan.fog {
            flags |= bits::FOG;
            let f: &Fog = &s.fog;
            p.fog_color = [f.color.x, f.color.y, f.color.z, finite(f.density, 0.0).max(0.0)];
            p.fog_shape = [
                finite(f.height_falloff, 0.0).max(0.0),
                finite(f.base_height, 0.0),
                finite(f.start_distance, 0.0).max(0.0),
                finite(f.max_opacity, 1.0).clamp(0.0, 1.0),
            ];
            let sun = f.sun_color * finite(f.sun_intensity, 0.0).max(0.0);
            p.fog_sun = [sun.x, sun.y, sun.z, finite(f.sun_exponent, 8.0).max(0.01)];
            let (dir, mode) = match f.sun_direction {
                Some(d) if d.is_finite() && d.length_squared() > 1e-12 => (d.normalize(), 1.0),
                Some(_) => (Vec3::ZERO, 0.0),
                None => (Vec3::ZERO, 2.0),
            };
            let mode = if f.sun_intensity > 0.0 { mode } else { 0.0 };
            p.fog_sun_dir = [dir.x, dir.y, dir.z, mode];
            p.fog_sky = [
                finite(f.sky_amount, 1.0).clamp(0.0, 1.0),
                finite(f.sky_falloff, 3.0).max(0.0),
                0.0,
                0.0,
            ];
        }
        if plan.ao {
            flags |= bits::AO;
            let a = &s.ssao;
            p.ao_a = [
                finite(a.radius, 1.0).max(0.01),
                finite(a.intensity, 1.0).clamp(0.0, 2.0),
                finite(a.power, 1.5).clamp(0.1, 8.0),
                finite(a.bias, 0.05).clamp(0.0, 0.9),
            ];
            p.ao_b = [
                finite(a.max_distance, 150.0).max(1.0),
                a.slices.clamp(1, 4) as f32,
                a.steps.clamp(2, 12) as f32,
                a.blur_radius.min(8) as f32,
            ];
            p.ao_size = [
                ao_size.0 as f32,
                ao_size.1 as f32,
                1.0 / ao_size.0 as f32,
                1.0 / ao_size.1 as f32,
            ];
        }
        if plan.ssr {
            flags |= bits::SSR;
            let r: &Ssr = &s.ssr;
            p.ssr_a = [
                finite(r.max_distance, 80.0).max(0.1),
                r.steps.clamp(8, 128) as f32,
                finite(r.thickness, 0.4).max(0.001),
                finite(r.intensity, 1.0).clamp(0.0, 2.0),
            ];
            p.ssr_b = [
                finite(r.max_roughness, 0.55).clamp(0.01, 1.0),
                finite(r.edge_fade, 0.1).clamp(0.0, 0.5),
                (frame % 8) as f32 * 3.7,
                0.0,
            ];
        }
        p.flags = [flags, 0, 0, 0];
        p.rect = [rect[0] as f32, rect[1] as f32, rect[2] as f32, rect[3] as f32];
        p
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Kind {
    Ao,
    BlurH,
    BlurV,
    Apply,
}

impl Kind {
    fn entry(self) -> &'static str {
        match self {
            Self::Ao => "fs_ao",
            Self::BlurH => "fs_blur_h",
            Self::BlurV => "fs_blur_v",
            Self::Apply => "fs_apply",
        }
    }
}

/// A camera's G-buffer-lite target.
struct GBuffer {
    size: (u32, u32),
    view: wgpu::TextureView,
    last_used: u64,
}

/// Encoder and device access of [`ScreenFx::encode`].
pub(crate) struct FxEncode<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub encoder: &'a mut wgpu::CommandEncoder,
    pub stats: &'a mut RenderStats,
    pub pool: &'a mut TexturePool,
    pub timing: &'a crate::timing::GpuTimer,
}

/// Inputs of one post group's effects.
pub(crate) struct FxInput<'a> {
    pub camera: Entity,
    pub settings: PostProcess,
    pub plan: FxPlan,
    pub scene: &'a wgpu::TextureView,
    pub scene_format: wgpu::TextureFormat,
    pub scene_size: (u32, u32),
    /// The group's rect in scene pixels, clipped.
    pub rect: [u32; 4],
    /// The settings camera's view bind group (group 0) and the dynamic offset of its `View`.
    pub view_bind_group: &'a wgpu::BindGroup,
    pub view_offset: u32,
    /// Group 1 with the single-sample scene depth bound.
    pub depth_bind_group: &'a wgpu::BindGroup,
    pub frame: u64,
}

/// Pipelines, layouts and the camera G-buffers of the city effects.
pub(crate) struct ScreenFx {
    bgl: wgpu::BindGroupLayout,
    layout: wgpu::PipelineLayout,
    module: wgpu::ShaderModule,
    sampler: wgpu::Sampler,
    params: wgpu::Buffer,
    pipelines: FastMap<(Kind, wgpu::TextureFormat), wgpu::RenderPipeline>,
    gbuffers: FastMap<Entity, GBuffer>,
    /// 1×1 stand-ins: AO = 1 (white), G-buffer = 0 (no reflection).
    white: wgpu::TextureView,
    black: wgpu::TextureView,
    frame: u64,
    /// Parameter slots written this frame.
    slots: u64,
    /// Pooled textures counted this frame (generation → bytes), for `RenderStats::target_bytes`.
    counted: Vec<u64>,
}

fn texel_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
    texel: [u8; 4],
) -> wgpu::TextureView {
    use wgpu::util::DeviceExt;
    let bytes = if format == AO_FORMAT { &texel[..1] } else { &texel[..] };
    device
        .create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("space3d screen fx stand-in"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            bytes,
        )
        .create_view(&Default::default())
}

impl ScreenFx {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, layouts: &FrameLayouts, frame: u64) -> Self {
        let tex = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("space3d screen fx"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<GpuFx>() as u64),
                    },
                    count: None,
                },
                tex(1),
                tex(2),
                tex(3),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("space3d screen fx"),
            bind_group_layouts: &[Some(&layouts.view), Some(&layouts.scene), Some(&bgl)],
            immediate_size: 0,
        });
        let source = format!(
            "{VIEW_WGSL}\n{}\n{FULLSCREEN_WGSL}\n{SCREENFX_WGSL}",
            scene_wgsl_for(crate::node::DepthAccess::ReadOnlyAttached)
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("space3d screen fx"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("space3d screen fx"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("space3d screen fx params"),
            size: SLOT * SLOTS,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            bgl,
            layout,
            module,
            sampler,
            params,
            pipelines: FastMap::default(),
            gbuffers: FastMap::default(),
            white: texel_texture(device, queue, AO_FORMAT, [255; 4]),
            black: texel_texture(device, queue, GBUFFER_FORMAT, [0; 4]),
            frame,
            slots: 0,
            counted: Vec::new(),
        }
    }

    /// Start of a renderer frame: forget the frame's parameter slots, evict G-buffers of
    /// cameras not drawn for a while.
    pub fn begin_frame(&mut self, frame: u64) {
        self.frame = frame;
        self.slots = 0;
        self.counted.clear();
        self.gbuffers.retain(|_, g| g.last_used + 3 > frame);
    }

    /// The G-buffer-lite target of `camera` at `size` (created or resized on demand).
    pub fn gbuffer_target(&mut self, device: &wgpu::Device, camera: Entity, size: (u32, u32)) -> wgpu::TextureView {
        let frame = self.frame;
        let g = self.gbuffers.get(&camera).filter(|g| g.size == size);
        if g.is_none() {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("space3d ssr g-buffer"),
                size: wgpu::Extent3d {
                    width: size.0.max(1),
                    height: size.1.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: GBUFFER_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.gbuffers.insert(
                camera,
                GBuffer {
                    size,
                    view: texture.create_view(&Default::default()),
                    last_used: frame,
                },
            );
        }
        let g = self.gbuffers.get_mut(&camera).expect("inserted");
        g.last_used = frame;
        g.view.clone()
    }

    /// The camera's G-buffer written this frame, if any.
    pub fn gbuffer(&self, camera: Entity) -> Option<&wgpu::TextureView> {
        self.gbuffers
            .get(&camera)
            .filter(|g| g.last_used == self.frame)
            .map(|g| &g.view)
    }

    fn pipeline(&mut self, device: &wgpu::Device, kind: Kind, format: wgpu::TextureFormat) {
        let (module, layout) = (&self.module, &self.layout);
        self.pipelines.entry((kind, format)).or_insert_with(|| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(kind.entry()),
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
                    entry_point: Some(kind.entry()),
                    compilation_options: Default::default(),
                    targets: &[Some(format.into())],
                }),
                multiview_mask: None,
                cache: None,
            })
        });
    }

    fn inputs(
        &self,
        device: &wgpu::Device,
        scene: &wgpu::TextureView,
        ao: &wgpu::TextureView,
        gbuf: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("space3d screen fx inputs"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.params,
                        offset: 0,
                        size: wgpu::BufferSize::new(std::mem::size_of::<GpuFx>() as u64),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(scene),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(ao),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(gbuf),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }

    /// Record the effects of one post group; returns the HDR color the post chain reads instead
    /// of `S` (pooled: valid until the pool is released after the group) and its generation.
    pub fn encode(&mut self, cx: &mut FxEncode<'_>, input: &FxInput<'_>) -> Option<(wgpu::TextureView, u64)> {
        let plan = input.plan;
        if !plan.any() || self.slots >= SLOTS {
            return None;
        }
        let ao_dims = ao_size(&input.settings.ssao, input.scene_size);
        let params = GpuFx::new(&input.settings, plan, input.rect, ao_dims, input.frame);
        let offset = self.slots * SLOT;
        self.slots += 1;
        cx.queue.write_buffer(&self.params, offset, bytemuck::bytes_of(&params));
        let offsets = [input.view_offset];
        let fx_offset = [offset as u32];
        let camera = Some(input.camera);

        // SSAO: raw AO into A, blurred H into B, blurred V back into A.
        let mut ao_view: Option<wgpu::TextureView> = None;
        if plan.ao {
            let desc = PoolDesc {
                size: ao_dims,
                format: AO_FORMAT,
                mips: 1,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            };
            let a = cx.pool.acquire(cx.device, desc);
            let b = cx.pool.acquire(cx.device, desc);
            let (a_view, a_gen) = {
                let t = cx.pool.get(a);
                (t.view.clone(), t.generation)
            };
            let (b_view, b_gen) = {
                let t = cx.pool.get(b);
                (t.view.clone(), t.generation)
            };
            self.count(cx.stats, a_gen, ao_bytes(ao_dims));
            self.count(cx.stats, b_gen, ao_bytes(ao_dims));
            let scale = |r: [u32; 4]| -> [u32; 4] {
                if ao_dims == input.scene_size {
                    return r;
                }
                let x = r[0] / 2;
                let y = r[1] / 2;
                let x1 = (r[0] + r[2]).div_ceil(2).min(ao_dims.0);
                let y1 = (r[1] + r[3]).div_ceil(2).min(ao_dims.1);
                [x, y, x1.saturating_sub(x).max(1), y1.saturating_sub(y).max(1)]
            };
            let rect = scale(input.rect);
            let run = |this: &mut Self,
                       cx: &mut FxEncode<'_>,
                       kind: Kind,
                       label: &'static str,
                       scope: &'static str,
                       src: &wgpu::TextureView,
                       dst: &wgpu::TextureView| {
                this.pipeline(cx.device, kind, AO_FORMAT);
                let bg = this.inputs(cx.device, input.scene, src, &this.black);
                this.draw(
                    cx,
                    (label, scope, camera),
                    &this.pipelines[&(kind, AO_FORMAT)],
                    dst,
                    (ao_dims, rect, wgpu::LoadOp::Clear(wgpu::Color::WHITE)),
                    (input.view_bind_group, input.depth_bind_group, &bg),
                    (offsets, fx_offset),
                );
            };
            let white = self.white.clone();
            run(self, cx, Kind::Ao, "space3d ssao", "post.ssao", &white, &a_view);
            if params.ao_b[3] > 0.0 {
                run(
                    self,
                    cx,
                    Kind::BlurH,
                    "space3d ssao blur h",
                    "post.ssao_blur",
                    &a_view,
                    &b_view,
                );
                run(
                    self,
                    cx,
                    Kind::BlurV,
                    "space3d ssao blur v",
                    "post.ssao_blur",
                    &b_view,
                    &a_view,
                );
            }
            ao_view = Some(a_view);
        }

        // Apply: S -> S'.
        let desc = PoolDesc {
            size: input.scene_size,
            format: input.scene_format,
            mips: 1,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        };
        let out = cx.pool.acquire(cx.device, desc);
        let (out_view, out_gen) = {
            let t = cx.pool.get(out);
            (t.view.clone(), t.generation)
        };
        let texel = input.scene_format.block_copy_size(None).unwrap_or(8) as u64;
        self.count(
            cx.stats,
            out_gen,
            texel * input.scene_size.0 as u64 * input.scene_size.1 as u64,
        );
        self.pipeline(cx.device, Kind::Apply, input.scene_format);
        let gbuf = if plan.ssr {
            self.gbuffer(input.camera).unwrap_or(&self.black)
        } else {
            &self.black
        };
        let bg = self.inputs(cx.device, input.scene, ao_view.as_ref().unwrap_or(&self.white), gbuf);
        self.draw(
            cx,
            ("space3d screen fx apply", "post.screenfx", camera),
            &self.pipelines[&(Kind::Apply, input.scene_format)],
            &out_view,
            (
                input.scene_size,
                input.rect,
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            ),
            (input.view_bind_group, input.depth_bind_group, &bg),
            (offsets, fx_offset),
        );
        // The pool and the scene textures count generations separately: tag this one so a
        // composite bind group cached for a scene texture is never taken for it.
        Some((out_view, out_gen | FX_GENERATION))
    }

    fn count(&mut self, stats: &mut RenderStats, generation: u64, bytes: u64) {
        if !self.counted.contains(&generation) {
            self.counted.push(generation);
            stats.target_bytes += bytes;
        }
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn draw(
        &self,
        cx: &mut FxEncode<'_>,
        (label, scope, camera): (&'static str, &'static str, Option<Entity>),
        pipeline: &wgpu::RenderPipeline,
        target: &wgpu::TextureView,
        (size, rect, load): ((u32, u32), [u32; 4], wgpu::LoadOp<wgpu::Color>),
        (view_bg, depth_bg, inputs): (&wgpu::BindGroup, &wgpu::BindGroup, &wgpu::BindGroup),
        (view_offsets, fx_offsets): ([u32; 1], [u32; 1]),
    ) {
        let mut pass = cx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            timestamp_writes: cx.timing.render(scope, camera),
            ..Default::default()
        });
        pass.set_viewport(0.0, 0.0, size.0 as f32, size.1 as f32, 0.0, 1.0);
        let [x, y, w, h] = rect;
        pass.set_scissor_rect(x, y, w, h);
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, view_bg, &view_offsets);
        pass.set_bind_group(1, depth_bg, &[]);
        pass.set_bind_group(2, inputs, &fx_offsets);
        pass.draw(0..3, 0..1);
        drop(pass);
        cx.stats.render_passes += 1;
        cx.stats.fullscreen_passes += 1;
        cx.stats.post_passes += 1;
        cx.stats.screen_fx_passes += 1;
    }
}

fn ao_bytes(size: (u32, u32)) -> u64 {
    size.0 as u64 * size.1 as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn on() -> PostProcess {
        PostProcess::default()
            .with_fog(Fog::city_haze())
            .with_ssao(Ssao::on())
            .with_ssr(Ssr::on())
    }

    #[test]
    fn gpu_fx_is_192_bytes_and_fits_a_slot() {
        assert_eq!(std::mem::size_of::<GpuFx>(), 192);
        assert!(std::mem::size_of::<GpuFx>() as u64 <= SLOT);
    }

    #[test]
    fn plan_needs_hdr_depth_and_a_real_group() {
        let s = on();
        let all = FxPlan {
            fog: true,
            ao: true,
            ssr: true,
        };
        assert_eq!(plan(&s, FrameMode::Hdr, false, true, true), all);
        assert_eq!(
            plan(&s, FrameMode::Hdr, false, true, false),
            FxPlan { ssr: false, ..all }
        );
        assert!(!plan(&s, FrameMode::Hdr, false, false, true).any(), "no depth");
        assert!(!plan(&s, FrameMode::Hdr, true, true, true).any(), "passthrough");
        assert!(!plan(&s, FrameMode::Ldr, false, true, true).any(), "ldr fallback");
        assert!(!plan(&s, FrameMode::Direct, false, true, true).any(), "direct");
        let off = PostProcess::default();
        assert!(!plan(&off, FrameMode::Hdr, false, true, true).any());
        let disabled = PostProcess { enabled: false, ..on() };
        assert!(!plan(&disabled, FrameMode::Hdr, false, true, true).any());
    }

    #[test]
    fn params_pack_the_settings() {
        let s = on().with_fog(Fog {
            sun_direction: Some(Vec3::new(0.0, -2.0, 0.0)),
            ..Fog::city_haze()
        });
        let p = GpuFx::new(
            &s,
            plan(&s, FrameMode::Hdr, false, true, true),
            [4, 5, 100, 50],
            (50, 25),
            3,
        );
        assert_eq!(p.flags[0], bits::FOG | bits::AO | bits::SSR);
        assert_eq!(p.fog_sun_dir, [0.0, -1.0, 0.0, 1.0], "explicit direction normalized");
        assert_eq!(p.ao_b[1], 2.0);
        assert_eq!(p.ao_size, [50.0, 25.0, 0.02, 0.04]);
        assert_eq!(p.rect, [4.0, 5.0, 100.0, 50.0]);
        let none = GpuFx::new(&s, FxPlan::default(), [0, 0, 1, 1], (1, 1), 0);
        assert_eq!(none.flags[0], 0);
        assert_eq!(none.fog_color, [0.0; 4]);
        // The view's own sun when none is given; none at all without intensity.
        let own = PostProcess::default().with_fog(Fog::city_haze());
        let p = GpuFx::new(
            &own,
            plan(&own, FrameMode::Hdr, false, true, true),
            [0, 0, 1, 1],
            (1, 1),
            0,
        );
        assert_eq!(p.fog_sun_dir[3], 2.0);
        let dark = PostProcess::default().with_fog(Fog::city_haze().with_sun(Vec3::ONE, 0.0, 8.0));
        let p = GpuFx::new(
            &dark,
            plan(&dark, FrameMode::Hdr, false, true, true),
            [0, 0, 1, 1],
            (1, 1),
            0,
        );
        assert_eq!(p.fog_sun_dir[3], 0.0);
    }

    #[test]
    fn params_sanitize_non_finite_values() {
        let s = PostProcess::default().with_fog(Fog {
            density: f32::NAN,
            max_opacity: f32::INFINITY,
            ..Fog::city_haze()
        });
        let p = GpuFx::new(&s, plan(&s, FrameMode::Hdr, false, true, true), [0, 0, 1, 1], (1, 1), 0);
        assert!(p.fog_color.iter().chain(p.fog_shape.iter()).all(|v| v.is_finite()));
    }

    #[test]
    fn ao_size_halves_on_request() {
        let mut a = Ssao::on();
        assert_eq!(ao_size(&a, (101, 51)), (51, 26));
        a.half_res = false;
        assert_eq!(ao_size(&a, (101, 51)), (101, 51));
    }
}
