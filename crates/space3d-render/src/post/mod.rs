//! Post-processing.
//!
//! [`api`] is the public hub (re-exported from the crate root). The rest is the crate-internal
//! contract between the frame and the post area (graphics spec §2.6): the frame plans post
//! groups, hands each one's scene color, depth and distortion buffer to
//! [`PostState::encode_group`] and owns the pass slot (targets, formats, capture replay); post
//! owns the composite and its effects. The frame milestone F4 wrote the composite-only chain
//! (exposure, tonemapping, dither, alpha, rect mapping, upscale; `chain.rs`,
//! `shaders/post/composite.wgsl`); the post milestones add the effects: bloom (PO1,
//! `bloom.rs`), FXAA (`fxaa.rs`) and the vignette, chromatic aberration and film grain in the
//! composite (PO3), auto exposure (PO2, `exposure.rs`, read back by `readback.rs`) and color
//! grading with 3D LUTs and WGSL hooks in the composite (PO4, `grading.rs`, `lut.rs`), and
//! distortion (PO5): the frame's Distortion phase (F5) clears each post group's persistent
//! distortion buffer and draws the writers into it — `DistortionMaterial` meshes through the
//! built-in node (`distortion_mesh.rs`), particles and user nodes through `DISTORTION_WGSL` —
//! and the composite samples the scene, its chromatic split and the bloom where the buffer
//! points.

pub(crate) mod api;
mod bloom;
mod chain;
mod city;
mod distortion_mesh;
mod exposure;
mod fxaa;
mod grading;
mod lut;
mod readback;
mod screenfx;

use crate::assets::RenderAssets;
use crate::caps::GraphicsCaps;
use crate::frame::PoolDesc;
use crate::gpu::{TrackedBuffer, Uploader};
use crate::handles::{FastMap, ShaderHandle, TextureHandle};
use crate::node::{FULLSCREEN_WGSL, FrameInfo, FrameMode, RenderNodes};
use crate::output::{HASH_WGSL, OUTPUT_WGSL, OutputEncoding};
use crate::settings::FrameSettings;
use crate::shader::CustomShader;
use crate::stats::RenderStats;
use api::{AntiAliasing, BloomComposite, ColorLut, DISTORTION_FORMAT, DistortionResolution, FxaaQuality, PostProcess};
use bloom::{BloomPlan, BloomState};
use chain::{CompositePlan, GpuPost, SeedKey};
use exposure::{MeterPath, MeterPlan, Meters};
use fxaa::{FxaaModules, LDR_FORMAT};
pub(crate) use screenfx::{FxEncode, FxInput, FxPlan, GBUFFER_FORMAT, ScreenFx};

/// The city effects a post group runs (see `screenfx::plan`).
pub(crate) fn screen_fx_plan(s: &PostProcess, mode: FrameMode, depth: bool, read_only_depth: bool) -> FxPlan {
    screenfx::plan(s, mode, false, depth, read_only_depth)
}
use readback::ExposureReadback;
use space3d_core::{Entity, World};
use std::borrow::Cow;
use std::sync::Arc;

/// Source of the composite pass (`shaders/post/composite.wgsl` after the output tail, the hash
/// functions and the full-screen triangle; its `//#HOOKS` marker takes the grading hooks).
pub(crate) const COMPOSITE_WGSL: &str = include_str!("../shaders/post/composite.wgsl");

/// The composite source with identity grading hooks.
pub(crate) fn composite_source() -> String {
    composite_source_with(None)
}

/// The composite source with the grading hooks of `hook` (identity for the ones it lacks).
pub(crate) fn composite_source_with(hook: Option<&CustomShader>) -> String {
    let composite = COMPOSITE_WGSL.replace("//#HOOKS", &grading::hooks_wgsl(hook));
    format!("{OUTPUT_WGSL}\n{HASH_WGSL}\n{FULLSCREEN_WGSL}\n{composite}")
}

/// A grading hook's (shader, version).
type HookKey = (ShaderHandle, u32);

/// Generations of a composite input bind group's textures: scene, distortion buffer, bloom
/// chain, exposure, LUT (0: the dummy).
type InputsKey = (u64, u64, u64, u64, u64);

/// One post group of the frame plan (graphics spec §1.3): the settings camera, its rect and the
/// storage of its target group.
pub(crate) struct PostGroupPlan {
    /// The settings camera (highest-order post camera of the rect).
    pub camera: Entity,
    pub settings: PostProcess,
    pub mode: FrameMode,
    /// The rect in the scene texture and in the output target, `[x, y, w, h]` pixels,
    /// unclipped (a camera viewport may reach past its target).
    pub scene_rect: [i32; 4],
    pub output_rect: [i32; 4],
    /// Size of the scene texture `S` and of the output target `D`.
    pub scene_size: (u32, u32),
    pub output_size: (u32, u32),
    /// `S` has an alpha channel (not `Rg11b10Ufloat`).
    pub scene_alpha: bool,
    /// Render-target group whose lead clear alpha is below 1: alpha is preserved.
    pub transparent: bool,
    /// The implicit group of an Ldr passthrough target (every effect off).
    pub passthrough: bool,
    /// The group's Distortion phase runs this frame (distortion enabled and a writer has work
    /// in the settings camera's view): its distortion buffer is written before the composite.
    pub distortion: bool,
}

/// What a post group needs this frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[allow(
    dead_code,
    reason = "graphics contract (spec §2.6): wired by the F4–F8 frame milestones"
)]
pub(crate) struct PostGroupNeeds {
    pub bloom: bool,
    pub auto_exposure: bool,
    pub fxaa: bool,
    pub distortion: bool,
    /// Scene depth (distortion occlusion).
    pub depth: bool,
}

/// Encoder and device access of [`PostState::encode_group`].
#[allow(
    dead_code,
    reason = "graphics contract (spec §2.6): wired by the F4–F8 frame milestones"
)]
pub(crate) struct PostEncode<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub encoder: &'a mut wgpu::CommandEncoder,
    pub stats: &'a mut RenderStats,
    /// The frame's texture pool (bloom chain, LDR intermediate, log-luminance chain): textures
    /// acquired here are released after the group.
    #[allow(dead_code, reason = "the post area's effects (PO1–PO3) acquire pooled textures")]
    pub pool: &'a mut crate::frame::TexturePool,
    /// GPU timings (`FrameSettings::gpu_timing`): labels `post.exposure`, `post.bloom`,
    /// `post.composite`, `post.fxaa` (graphics spec §4.4).
    pub timing: &'a crate::timing::GpuTimer,
}

/// Inputs of one post group's passes.
#[allow(
    dead_code,
    reason = "graphics contract (spec §2.6): wired by the F4–F8 frame milestones"
)]
pub(crate) struct PostGroupInput<'a> {
    pub camera: Entity,
    pub settings: PostProcess,
    pub mode: FrameMode,
    pub scene_output: OutputEncoding,
    pub scene: &'a wgpu::TextureView,
    pub scene_format: wgpu::TextureFormat,
    pub scene_size: (u32, u32),
    pub scene_rect: [u32; 4],
    /// The scene texture has alpha (not `Rg11b10Ufloat`).
    pub scene_alpha: bool,
    /// Render-target group whose lead clear alpha is below 1: alpha is preserved.
    pub transparent: bool,
    pub depth: Option<&'a wgpu::TextureView>,
    pub distortion: Option<&'a wgpu::TextureView>,
    pub output: &'a wgpu::TextureView,
    pub output_format: wgpu::TextureFormat,
    pub output_size: (u32, u32),
    pub output_rect: [u32; 4],
    pub output_manual_srgb: bool,
    /// First write into the output this frame: clear it (to black) outside the rect.
    pub output_clear: bool,
    /// The implicit group of an Ldr passthrough target (every effect off).
    pub passthrough: bool,
    /// Windowed-capture replay: no metering, no adaptation.
    pub replay: bool,
    pub frame: FrameInfo,
    /// Index of the group among this frame's [`PostState::plan_group`] calls.
    pub index: usize,
    /// Identity of `scene` (bind groups are cached by it).
    pub scene_generation: u64,
    /// The rect drawn in `output` (within the pass limits, see `ViewportRect::fitted`):
    /// viewport `[x, y, w, h]`.
    pub output_viewport: [f32; 4],
}

/// A post group's persistent distortion buffer.
struct DistortionTarget {
    size: (u32, u32),
    view: wgpu::TextureView,
    last_used: u64,
    /// Unique per creation (bind groups sampling it are cached by it).
    generation: u64,
}

/// Composite pipelines' output kind (`S3D_POST_OUT`): how the last pass writes `D`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum OutputKind {
    /// sRGB-format target: hardware encodes.
    Srgb,
    /// 8-bit target without an sRGB format or view: the shader encodes.
    ManualSrgb,
    /// Float target: linear, no dither.
    Linear,
}

impl OutputKind {
    fn of(format: wgpu::TextureFormat, manual_srgb: bool) -> Self {
        if manual_srgb {
            Self::ManualSrgb
        } else if format.is_srgb() {
            Self::Srgb
        } else if matches!(
            format,
            wgpu::TextureFormat::Rgba16Float | wgpu::TextureFormat::Rgba32Float | wgpu::TextureFormat::Rg11b10Ufloat
        ) {
            Self::Linear
        } else {
            Self::ManualSrgb
        }
    }
    fn id(self) -> f64 {
        match self {
            Self::Srgb => 0.0,
            Self::ManualSrgb => 1.0,
            Self::Linear => 2.0,
        }
    }
}

/// Post state: pool users, per-camera history (exposure, distortion buffers), pipelines,
/// readback.
pub(crate) struct PostState {
    frame: u64,
    distortion: FastMap<Entity, DistortionTarget>,
    /// `GpuPost` of every post group planned this frame, one aligned slot each.
    params: TrackedBuffer,
    slot: usize,
    planned: usize,
    /// Group 0: params (dynamic offset) + linear-clamp sampler; group 1: `S`, bloom level 0,
    /// exposure, LUT, distortion buffer, LDR intermediate.
    params_bgl: wgpu::BindGroupLayout,
    inputs_bgl: wgpu::BindGroupLayout,
    layout: wgpu::PipelineLayout,
    module: wgpu::ShaderModule,
    sampler: wgpu::Sampler,
    /// 1×1 black stand-in for inputs a group does not have (bloom, exposure = 0 EV, LUT, …).
    dummy: wgpu::TextureView,
    /// The params bind group, with the params and seed buffers' generations it was made for.
    params_bg: Option<(wgpu::BindGroup, u32, u64)>,
    /// The post groups' per-frame seeds, a window of frames at a time.
    seeds: SeedWindow,
    /// Input bind groups by (scene-texture generation, distortion-buffer generation or 0,
    /// bloom-chain generation or 0, exposure-texture generation or 0, LUT generation or 0), with
    /// the frame they were last used.
    inputs_bgs: FastMap<InputsKey, (wgpu::BindGroup, u64)>,
    /// Distortion buffers created (generation counter) and created since `end_frame`.
    distortion_generation: u64,
    created: u32,
    /// Composite pipelines by output format, kind and grading hook.
    composite: FastMap<(wgpu::TextureFormat, OutputKind, Option<HookKey>), wgpu::RenderPipeline>,
    /// Grading hooks: the current source of each hook shader used, their composite modules, the
    /// first build error of a version, and pipelines whose validation is still pending
    /// (browser WebGPU answers error scopes asynchronously).
    hooks: FastMap<ShaderHandle, (u32, Arc<CustomShader>)>,
    hook_modules: FastMap<HookKey, wgpu::ShaderModule>,
    hook_errors: FastMap<ShaderHandle, (u32, String)>,
    hook_pending: Vec<(HookKey, space3d_tasks::Task<Option<String>>)>,
    /// LUT strips in use: texture version, view, generation (bind groups), sRGB, and the renderer
    /// frame last used; LUTs warned about (bad size).
    luts: FastMap<TextureHandle, LutEntry>,
    lut_generation: u64,
    warned_luts: Vec<TextureHandle>,
    /// Auto exposure: metering state, and the exposure readback.
    meters: Meters,
    readback: ExposureReadback,
    /// `FrameSettings::read_exposure` of this frame.
    read_exposure: bool,
    /// Auto-exposure groups metered this frame.
    metered: usize,
    /// `FrameInfo::delta` of this frame (auto-exposure adaptation).
    delta: f32,
    /// Renderer frames (`begin_frame` calls): eviction clock, never feeds pixels.
    tick: u64,
    /// Warned (once) that auto exposure is off in the Ldr fallback / without a float exposure
    /// format.
    warned_ldr_exposure: bool,
    warned_exposure_format: bool,
    fxaa_modules: FxaaModules,
    fxaa: FastMap<(FxaaQuality, wgpu::TextureFormat, OutputKind), wgpu::RenderPipeline>,
    /// FXAA input bind groups by LDR-intermediate generation, with the frame last used.
    fxaa_bgs: FastMap<u64, (wgpu::BindGroup, u64)>,
    /// Device capabilities, and this frame's (masked by `FrameSettings::downlevel`).
    caps: GraphicsCaps,
    frame_caps: GraphicsCaps,
    /// What each post group planned this frame needs (index = plan order = `PostGroupInput::index`).
    groups: Vec<GroupPost>,
    bloom: BloomState,
    /// Pooled textures whose memory `RenderStats::target_bytes` counts this frame.
    counted: Vec<u64>,
    /// Warned (once) that bloom is off in the Ldr fallback.
    warned_ldr_bloom: bool,
}

/// What a planned post group records this frame besides the composite.
#[derive(Clone, Copy, Debug, Default)]
struct GroupPost {
    bloom: Option<BloomPlan>,
    /// FXAA after the composite (which then writes the LDR intermediate).
    fxaa: Option<FxaaQuality>,
    /// Auto-exposure metering before the bloom chain.
    meter: Option<MeterPlan>,
    /// The group's rect in `S`, clipped.
    scene_rect: [u32; 4],
    /// The LUT strip bound (its handle).
    lut: Option<TextureHandle>,
    /// The grading hook compiled into the composite.
    hook: Option<HookKey>,
    /// Exposure compensation (stops; readback).
    compensation: f32,
}

/// A LUT strip in use (see `PostState::luts`).
struct LutEntry {
    version: u32,
    view: wgpu::TextureView,
    generation: u64,
    srgb: bool,
    size: u32,
    last_used: u64,
}

/// The part of `rect` (pixels, unclipped) inside a `size` texture, at least 1×1: `[x, y, w, h]`
/// (the frame's `ViewportRect::scissor`).
fn clip_rect(rect: [i32; 4], size: (u32, u32)) -> [u32; 4] {
    let axis = |start: i32, len: i32, extent: u32| {
        let extent = extent.max(1) as i64;
        let a = (start as i64).clamp(0, extent - 1);
        let b = (start as i64 + len.max(0) as i64).clamp(a + 1, extent);
        (a as u32, (b - a) as u32)
    };
    let (x, w) = axis(rect[0], rect[2], size.0);
    let (y, h) = axis(rect[1], rect[3], size.1);
    [x, y, w, h]
}

/// Frames of post seeds uploaded at once.
const SEED_WINDOW: u64 = 64;
/// Bytes of one post group's seeds (`post_seeds` in composite.wgsl).
const SEED_BYTES: u64 = 16;

/// The per-frame seeds (graphics spec §4.3: animated grain, dither) of every post group for a
/// window of [`SEED_WINDOW`] frames: slot `group × SEED_WINDOW + frame % SEED_WINDOW`, one
/// aligned uniform each, bound with a dynamic offset. Written when a window starts, the frame
/// leaves it, or the groups' [`SeedKey`]s change; a still scene uploads nothing in between. (It
/// uploaded each group's 16 bytes every frame: a staging copy costing about 4 µs of CPU and
/// 20–40 µs of GPU throughput in an otherwise idle frame.)
struct SeedWindow {
    align: u64,
    buffer: Option<wgpu::Buffer>,
    generation: u64,
    /// First frame of the uploaded window.
    start: Option<u64>,
    /// The keys the window was written for, and this frame's (plan order).
    keys: Vec<SeedKey>,
    current: Vec<SeedKey>,
    bytes: Vec<u8>,
    /// Collections that grew (`RenderStats::cpu_reallocations` at `end_frame`).
    reallocations: u32,
}

impl SeedWindow {
    fn new(align: u64) -> Self {
        Self {
            align: align.max(SEED_BYTES),
            buffer: None,
            generation: 0,
            start: None,
            keys: Vec::new(),
            current: Vec::new(),
            bytes: Vec::new(),
            reallocations: 0,
        }
    }

    /// Upload the window starting at `frame` unless the uploaded one covers it for the same
    /// keys.
    fn flush(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, up: &mut Uploader, frame: u64) {
        if self.current.is_empty() {
            return;
        }
        let need = self.current.len() as u64 * SEED_WINDOW * self.align;
        if self.buffer.as_ref().is_none_or(|b| b.size() < need) {
            self.buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("space3d post seeds"),
                size: need,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.generation += 1;
            self.start = None;
            up.reallocations += 1;
        }
        let covered = self.start.is_some_and(|s| (s..s + SEED_WINDOW).contains(&frame));
        if covered && self.keys == self.current {
            return;
        }
        let capacities = (self.bytes.capacity(), self.keys.capacity());
        self.bytes.clear();
        self.bytes.resize(need as usize, 0);
        for (g, key) in self.current.iter().enumerate() {
            for f in frame..frame + SEED_WINDOW {
                let at = self.offset(g, f) as usize;
                self.bytes[at..at + SEED_BYTES as usize].copy_from_slice(bytemuck::cast_slice(&key.seeds(f)));
            }
        }
        if let Some(buffer) = &self.buffer {
            up.write(encoder, buffer, 0, &self.bytes);
        }
        self.keys.clone_from(&self.current);
        self.start = Some(frame);
        self.reallocations += u32::from((self.bytes.capacity(), self.keys.capacity()) != capacities);
    }

    /// Dynamic offset of post group `group`'s seeds at `frame`.
    fn offset(&self, group: usize, frame: u64) -> u32 {
        ((group as u64 * SEED_WINDOW + frame % SEED_WINDOW) * self.align) as u32
    }
}

impl PostState {
    pub fn new(device: &wgpu::Device, caps: &GraphicsCaps) -> Self {
        let params_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("space3d post params"),
            entries: &[
                crate::pipeline::uniform_entry(
                    0,
                    wgpu::ShaderStages::FRAGMENT,
                    true,
                    std::mem::size_of::<GpuPost>() as u64,
                ),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                crate::pipeline::uniform_entry(2, wgpu::ShaderStages::FRAGMENT, true, SEED_BYTES),
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
        let inputs_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("space3d post inputs"),
            entries: &[
                texture(0, true),
                texture(1, true),
                texture(2, false),
                texture(3, true),
                texture(4, true),
                texture(5, true),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("space3d composite"),
            bind_group_layouts: &[Some(&params_bgl), Some(&inputs_bgl)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("space3d composite"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(composite_source())),
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("space3d post linear clamp"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let dummy = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("space3d post dummy"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
        let align = device.limits().min_uniform_buffer_offset_alignment as usize;
        let bloom = BloomState::new(device, align);
        Self {
            frame: 0,
            distortion: FastMap::default(),
            params: TrackedBuffer::new("space3d post params", wgpu::BufferUsages::UNIFORM),
            slot: std::mem::size_of::<GpuPost>().next_multiple_of(align),
            planned: 0,
            params_bgl,
            inputs_bgl,
            layout,
            module,
            sampler,
            dummy,
            params_bg: None,
            seeds: SeedWindow::new(align as u64),
            inputs_bgs: FastMap::default(),
            composite: FastMap::default(),
            hooks: FastMap::default(),
            hook_modules: FastMap::default(),
            hook_errors: FastMap::default(),
            hook_pending: Vec::new(),
            luts: FastMap::default(),
            lut_generation: 0,
            warned_luts: Vec::new(),
            meters: Meters::new(align),
            readback: ExposureReadback::default(),
            read_exposure: false,
            metered: 0,
            delta: 0.0,
            tick: 0,
            warned_ldr_exposure: false,
            warned_exposure_format: false,
            fxaa_modules: FxaaModules::default(),
            fxaa: FastMap::default(),
            fxaa_bgs: FastMap::default(),
            distortion_generation: 0,
            created: 0,
            caps: caps.clone(),
            frame_caps: caps.clone(),
            groups: Vec::new(),
            bloom,
            counted: Vec::new(),
            warned_ldr_bloom: false,
        }
    }

    pub fn begin_frame(&mut self, frame: &FrameInfo, settings: &FrameSettings) {
        self.frame_caps = self.caps.masked(settings.downlevel);
        self.frame = frame.frame;
        self.delta = frame.delta;
        self.tick += 1;
        self.read_exposure = settings.read_exposure;
        self.metered = 0;
        self.meters.begin_frame();
        self.readback.begin_frame(self.tick);
        self.poll_hooks();
        self.planned = 0;
        self.params.begin();
        self.seeds.current.clear();
        self.groups.clear();
        self.counted.clear();
        self.bloom.params.begin();
        self.bloom.slots = 0;
    }

    /// Step 6: what a post group needs (bloom, auto exposure, FXAA, distortion, depth). Writes
    /// the group's composite parameters (slot = call order this frame); `assets`: the LUT
    /// textures and grading hook shaders.
    pub fn plan_group(
        &mut self,
        world: &World,
        assets: &RenderAssets,
        group: &PostGroupPlan,
        reallocs: &mut u32,
    ) -> PostGroupNeeds {
        // The Distortion phase's writers are nodes: the frame finds them (`group.distortion`).
        let _ = world;
        let s = &group.settings;
        let effects = !group.passthrough && s.enabled;
        let scene_rect = clip_rect(group.scene_rect, group.scene_size);
        let meter = self.plan_exposure(group, scene_rect, reallocs);
        let bloom = self.plan_bloom(group, meter.is_some(), reallocs);
        // FXAA runs in every mode (display space), never in the passthrough group.
        let fxaa = match s.anti_aliasing {
            AntiAliasing::Fxaa(q) if effects => Some(q),
            _ => None,
        };
        let g = &s.color_grading;
        let lut = g
            .lut
            .filter(|_| effects)
            .and_then(|l| self.plan_lut(assets, &l, reallocs));
        let hook = g
            .shader
            .filter(|_| effects)
            .and_then(|h| self.plan_hook(assets, h, reallocs));
        let capacity = self.groups.capacity();
        self.groups.push(GroupPost {
            bloom,
            fxaa,
            meter,
            scene_rect,
            lut: lut.map(|(handle, _, _)| handle),
            hook,
            compensation: s.exposure,
        });
        *reallocs += u32::from(self.groups.capacity() != capacity);
        let params = GpuPost::new(
            group,
            CompositePlan {
                bloom: bloom.is_some(),
                fxaa: fxaa.is_some(),
                auto_exposure: meter.is_some(),
                lut: lut.map(|(_, n, srgb)| (n, srgb)),
            },
        );
        // Nothing in it changes per frame: a still scene uploads nothing.
        self.params
            .write_at(self.planned * self.slot, bytemuck::bytes_of(&params), reallocs);
        let capacity = self.seeds.current.capacity();
        self.seeds
            .current
            .push(SeedKey::of(params.flags[0], &group.settings.film_grain));
        *reallocs += u32::from(self.seeds.current.capacity() != capacity);
        self.planned += 1;
        PostGroupNeeds {
            bloom: bloom.is_some(),
            auto_exposure: meter.is_some(),
            fxaa: fxaa.is_some(),
            distortion: !group.passthrough && group.settings.enabled && group.settings.distortion.enabled,
            ..Default::default()
        }
    }

    /// The auto-exposure metering of a post group this frame (`rect`: its rect in `S`,
    /// clipped): auto exposure enabled, in Hdr mode (off in the Ldr fallback, warned once; never
    /// in the passthrough group) where `GraphicsCaps::exposure_format` exists. Cameras without
    /// it report their manual exposure to `camera_exposure` (with `read_exposure`).
    fn plan_exposure(&mut self, group: &PostGroupPlan, rect: [u32; 4], reallocs: &mut u32) -> Option<MeterPlan> {
        let s = &group.settings;
        let wanted = !group.passthrough && s.enabled && s.auto_exposure.enabled;
        let path = if !wanted {
            None
        } else if group.mode != FrameMode::Hdr {
            if !self.warned_ldr_exposure {
                self.warned_ldr_exposure = true;
                tracing::warn!("auto exposure is off in the Ldr fallback (no float render targets or HdrMode::Ldr)");
            }
            None
        } else {
            let path = MeterPath::of(&self.frame_caps);
            if path.is_none() && !self.warned_exposure_format {
                self.warned_exposure_format = true;
                tracing::warn!("auto exposure is off: the device renders no float exposure format");
            }
            path
        };
        let Some(path) = path else {
            self.meters.disable(group.camera);
            if self.read_exposure && !group.passthrough {
                self.readback.set(group.camera, s.exposure);
            }
            return None;
        };
        self.metered += 1;
        Some(
            self.meters
                .plan(group.camera, rect, &s.auto_exposure, path, self.delta, reallocs),
        )
    }

    /// A post group's LUT this frame: `(texture, N, sRGB)` for a single-layer N²×N strip that is
    /// uploaded (others are ignored, warned once) with a positive contribution.
    fn plan_lut(
        &mut self,
        assets: &RenderAssets,
        lut: &ColorLut,
        reallocs: &mut u32,
    ) -> Option<(TextureHandle, u32, bool)> {
        if lut.contribution.is_nan() || lut.contribution <= 0.0 {
            return None;
        }
        let t = crate::node::texture_ref(assets, lut.texture)?;
        let n = lut.size;
        if t.layered || !(2..=lut::MAX_SIZE).contains(&n) || t.size != (n * n, n) {
            if !self.warned_luts.contains(&lut.texture) {
                tracing::warn!(
                    size = n,
                    texture = ?t.size,
                    "color grading LUT ignored: the texture must be one N²×N strip of `ColorLut::size` N (2..=64)"
                );
                let capacity = self.warned_luts.capacity();
                self.warned_luts.push(lut.texture);
                *reallocs += u32::from(self.warned_luts.capacity() != capacity);
            }
            return None;
        }
        let tick = self.tick;
        let srgb = t.format.is_srgb();
        match self.luts.get_mut(&lut.texture) {
            Some(e) if e.version == t.version => e.last_used = tick,
            _ => {
                self.lut_generation += 1;
                let capacity = self.luts.capacity();
                self.luts.insert(
                    lut.texture,
                    LutEntry {
                        version: t.version,
                        view: t.view,
                        generation: self.lut_generation,
                        srgb,
                        size: n,
                        last_used: tick,
                    },
                );
                *reallocs += u32::from(self.luts.capacity() != capacity);
            }
        }
        let e = &self.luts[&lut.texture];
        Some((lut.texture, e.size, e.srgb))
    }

    /// A post group's grading hook this frame: the shader's current version, unless that version
    /// failed to build (identity then).
    fn plan_hook(&mut self, assets: &RenderAssets, handle: ShaderHandle, reallocs: &mut u32) -> Option<HookKey> {
        let e = assets.shaders.get(handle.raw())?;
        match self.hooks.get(&handle) {
            Some((v, _)) if *v == e.version => {}
            _ => {
                let capacity = self.hooks.capacity();
                self.hooks.insert(handle, (e.version, e.shader.clone()));
                *reallocs += u32::from(self.hooks.capacity() != capacity);
                // Older versions' pipelines and modules.
                let version = e.version;
                self.composite
                    .retain(|(_, _, h), _| h.is_none_or(|(hh, v)| hh != handle || v == version));
                self.hook_modules.retain(|(hh, v), _| *hh != handle || *v == version);
            }
        }
        if self.hook_errors.get(&handle).is_some_and(|(v, _)| *v == e.version) {
            return None;
        }
        Some((handle, e.version))
    }

    /// Resolve grading-hook pipelines whose (asynchronous) validation answered: a failed one is
    /// reported and replaced by the plain composite from the next plan on.
    fn poll_hooks(&mut self) {
        let mut i = 0;
        while i < self.hook_pending.len() {
            let (key, task) = &mut self.hook_pending[i];
            let result = match task.poll() {
                Some(r) => r,
                None if task.state() == space3d_tasks::TaskState::Cancelled => None,
                None => {
                    i += 1;
                    continue;
                }
            };
            let key = *key;
            self.hook_pending.swap_remove(i);
            if let Some(e) = result {
                self.hook_failed(key, e);
            }
        }
    }

    /// Record (and log once) that the grading hook `key` failed to build; drop its pipelines.
    fn hook_failed(&mut self, (handle, version): HookKey, error: String) {
        let name = self.hooks.get(&handle).map_or("?", |(_, s)| s.name.as_str());
        tracing::error!(
            shader = name,
            "color grading hook failed to build; grading without it:\n{error}"
        );
        self.hook_errors.insert(handle, (version, error));
        self.composite.retain(|(_, _, h), _| *h != Some((handle, version)));
        self.hook_modules.remove(&(handle, version));
    }

    /// Build error of the current version of a grading hook shader (see
    /// `Renderer::shader_error`).
    pub fn shader_error(&self, shader: ShaderHandle) -> Option<&str> {
        let (version, e) = self.hook_errors.get(&shader)?;
        (self.hooks.get(&shader).map(|(v, _)| *v) == Some(*version)).then_some(e.as_str())
    }

    /// The bloom chain of a post group this frame, its pass uniforms written: bloom enabled
    /// with a non-zero intensity, on a rect at least `min_rect_height` tall, in Hdr mode (off in the Ldr fallback, warned once; never in
    /// the passthrough group) where `GraphicsCaps::bloom_format` exists. `auto_exposure`: the
    /// prefilter also multiplies by 2^E.
    fn plan_bloom(&mut self, group: &PostGroupPlan, auto_exposure: bool, reallocs: &mut u32) -> Option<BloomPlan> {
        let s = &group.settings;
        let b = &s.bloom;
        if group.passthrough || !s.enabled || !b.enabled || b.intensity <= 0.0 {
            return None;
        }
        if group.mode != FrameMode::Hdr {
            if !self.warned_ldr_bloom {
                self.warned_ldr_bloom = true;
                tracing::warn!("bloom is off in the Ldr fallback (no float render targets or HdrMode::Ldr)");
            }
            return None;
        }
        // The energy-conserving composite promises the scene's mean: its chain is
        // `Rgba16Float`. `Rg11b10Ufloat` (5–6 mantissa bits) converts by truncation on some GPUs
        // (Apple), which over the chain's ~9 writes loses about 5 % of the energy (and more of
        // blue) — invisible in an additive glow, whose chain keeps the compact format.
        let format = match b.composite {
            BloomComposite::EnergyConserving => self.frame_caps.hdr_format,
            _ => self.frame_caps.bloom_format,
        }?;
        let rect = clip_rect(group.scene_rect, group.scene_size);
        if rect[3] < b.min_rect_height {
            return None;
        }
        let chain = bloom::chain_size((rect[2], rect[3]), b);
        let first_slot = self.bloom.slots;
        let (params, slot) = (&mut self.bloom.params, self.bloom.slot);
        let mut n = first_slot as usize;
        let exposure = (s.exposure.exp2(), auto_exposure);
        bloom::passes(group.scene_size, rect, &chain, b, exposure, |p| {
            params.write_at(n * slot, bytemuck::bytes_of(&p), reallocs);
            n += 1;
        });
        self.bloom.slots = n as u32;
        Some(BloomPlan {
            format,
            size: chain.size,
            mips: chain.mips,
            merged: chain.merged,
            half: chain.half,
            first_slot,
            scatter: b.scatter.clamp(0.0, 1.0),
        })
    }

    /// Upload this frame's parameters (after every [`Self::plan_group`], before any
    /// [`Self::encode_group`]).
    pub fn flush(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, up: &mut Uploader) {
        if self.planned == 0 {
            return;
        }
        let mut reallocs = 0;
        self.params.ensure_len(std::mem::size_of::<GpuPost>(), &mut reallocs);
        self.params.flush(device, encoder, up);
        self.seeds.flush(device, encoder, up, self.frame);
        let generation = self.params.gpu.generation;
        let seeds_generation = self.seeds.generation;
        let Some(seeds) = &self.seeds.buffer else {
            return;
        };
        if self
            .params_bg
            .as_ref()
            .is_none_or(|(_, g, sg)| *g != generation || *sg != seeds_generation)
        {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("space3d post params"),
                layout: &self.params_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: self.params.buffer(),
                            offset: 0,
                            size: wgpu::BufferSize::new(std::mem::size_of::<GpuPost>() as u64),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: seeds,
                            offset: 0,
                            size: wgpu::BufferSize::new(SEED_BYTES),
                        }),
                    },
                ],
            });
            self.params_bg = Some((bg, generation, seeds_generation));
        }
        self.bloom.flush(device, encoder, up, &self.sampler);
        self.meters.flush(device, encoder, up);
        if self.read_exposure {
            self.readback.begin(device, encoder, self.metered);
        }
    }

    /// The persistent distortion buffer of a post group (created on demand; freed 60 frames
    /// after its last use), sized from `rect` and `res`. The caller's pass clears it.
    pub fn distortion_target(
        &mut self,
        device: &wgpu::Device,
        camera: Entity,
        rect: [u32; 4],
        res: DistortionResolution,
    ) -> (&wgpu::TextureView, (u32, u32)) {
        let size = match res {
            DistortionResolution::Half => (rect[2].div_ceil(2).max(1), rect[3].div_ceil(2).max(1)),
            _ => (rect[2].max(1), rect[3].max(1)),
        };
        let frame = self.frame;
        let fresh = self.distortion.get(&camera).is_none_or(|t| t.size != size);
        if fresh {
            self.distortion_generation += 1;
            self.created += 1;
            let t = create_distortion_target(device, size, frame, self.distortion_generation);
            self.distortion.insert(camera, t);
        }
        let t = self.distortion.get_mut(&camera).expect("inserted");
        t.last_used = frame;
        (&t.view, t.size)
    }

    /// A post group's existing distortion buffer: its view (a cheap clone), size and
    /// generation.
    pub fn distortion_view(&self, camera: Entity) -> Option<(wgpu::TextureView, (u32, u32), u64)> {
        self.distortion
            .get(&camera)
            .map(|t| (t.view.clone(), t.size, t.generation))
    }

    /// Build the composite pipeline for an output target of `format` (`manual_srgb`: the shader
    /// encodes sRGB) ahead of use (`Renderer::prewarm`), with FXAA (Medium), the bloom chains and
    /// the auto-exposure metering of this device's path.
    pub fn prewarm(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat, manual_srgb: bool) {
        let kind = OutputKind::of(format, manual_srgb);
        self.composite_pipeline(device, format, kind, None);
        // FXAA (Medium, the default quality) and the composite into its LDR intermediate.
        self.composite_pipeline(device, LDR_FORMAT, OutputKind::Srgb, None);
        if let Some((path, format)) = MeterPath::of(&self.frame_caps) {
            self.meters.prewarm(device, path, format);
        }
        self.fxaa_pipeline(device, FxaaQuality::Medium, format, kind);
        // The additive chain's format and the energy-conserving one's (see `plan_bloom`).
        for format in [self.caps.bloom_format, self.caps.hdr_format].into_iter().flatten() {
            self.bloom.pipelines(device, format);
        }
    }

    /// The composite pipeline for `format`, `kind` and the grading hook `hook` (created on first
    /// use). A hook composite is validated with naga and built inside error scopes; if it fails
    /// (now, or later when the scopes answer asynchronously) the hook is reported and the plain
    /// composite is used. Returns the key of the pipeline to draw with.
    fn composite_pipeline(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        kind: OutputKind,
        hook: Option<HookKey>,
    ) -> (wgpu::TextureFormat, OutputKind, Option<HookKey>) {
        let key = (format, kind, hook);
        if self.composite.contains_key(&key) {
            return key;
        }
        let Some((handle, version)) = hook else {
            let (module, layout) = (&self.module, &self.layout);
            let p = fullscreen_pipeline(device, layout, module, "fs_composite", format, kind);
            self.composite.insert(key, p);
            return key;
        };
        let Some((_, source)) = self.hooks.get(&handle).filter(|(v, _)| *v == version) else {
            return self.composite_pipeline(device, format, kind, None);
        };
        let module = match self.hook_modules.get(&(handle, version)) {
            Some(m) => Ok(m.clone()),
            None => {
                let src = composite_source_with(Some(source));
                grading::validate_hook_module(&src).map(|()| {
                    device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some(source.name.as_str()),
                        source: wgpu::ShaderSource::Wgsl(Cow::Owned(src)),
                    })
                })
            }
        };
        let module = match module {
            Ok(m) => m,
            Err(e) => {
                self.hook_failed((handle, version), e);
                return self.composite_pipeline(device, format, kind, None);
            }
        };
        // Backend translation failures are internal errors, not validation errors.
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let p = fullscreen_pipeline(device, &self.layout, &module, "fs_composite", format, kind);
        let (internal, validation) = (internal.pop(), validation.pop());
        let mut error = Box::pin(async move {
            match internal.await {
                Some(e) => Some(e.to_string()),
                None => validation.await.map(|e| e.to_string()),
            }
        });
        match futures::FutureExt::now_or_never(&mut error) {
            Some(Some(e)) => {
                self.hook_failed((handle, version), e);
                return self.composite_pipeline(device, format, kind, None);
            }
            Some(None) => {}
            None => self.hook_pending.push(((handle, version), space3d_tasks::spawn(error))),
        }
        self.hook_modules.insert((handle, version), module);
        self.composite.insert(key, p);
        key
    }

    /// The FXAA pipeline for a quality, `format` and `kind` (created on first use).
    fn fxaa_pipeline(
        &mut self,
        device: &wgpu::Device,
        q: FxaaQuality,
        format: wgpu::TextureFormat,
        kind: OutputKind,
    ) -> &wgpu::RenderPipeline {
        let (modules, layout) = (&mut self.fxaa_modules, &self.layout);
        self.fxaa.entry((q, format, kind)).or_insert_with(|| {
            let module = modules.get(device, q);
            fullscreen_pipeline(device, layout, module, "fs_fxaa", format, kind)
        })
    }

    /// Record meter → bloom → composite (→ FXAA) for one post group into the frame encoder: auto
    /// exposure (PO2), bloom (PO1), the composite with the lens effects (PO3), grading (PO4)
    /// and the distortion apply (PO5), FXAA (PO3).
    pub fn encode_group(&mut self, cx: &mut PostEncode<'_>, input: &PostGroupInput<'_>) {
        if self.params_bg.is_none() {
            return;
        }
        let group = self.groups.get(input.index).copied().unwrap_or_default();
        // Auto exposure first: the bloom prefilter and the composite read E.
        let exposure = group.meter.and_then(|plan| {
            let scene = (input.scene, input.scene_generation);
            let e = self
                .meters
                .encode(cx, input.camera, plan, scene, group.scene_rect, input.replay)?;
            for i in 0..self.meters.pooled().len() {
                let (generation, bytes) = self.meters.pooled()[i];
                self.count_target(cx.stats, input.replay, generation, bytes);
            }
            if self.read_exposure
                && !input.replay
                && let Some((texture, format)) = self.meters.texture(input.camera)
            {
                self.readback
                    .record(cx.encoder, input.camera, texture, format, group.compensation);
            }
            Some(e)
        });
        let bloom = group.bloom.and_then(|plan| {
            let scene = (input.scene, input.scene_generation);
            let e = exposure.as_ref().map_or((&self.dummy, 0), |(view, g)| (view, *g));
            let timing = cx.timing;
            let timestamps = timing
                .span("post.bloom", Some(input.camera))
                .map(|(query_set, begin, end)| bloom::ChainTimestamps {
                    query_set,
                    begin: Some(begin),
                    end: Some(end),
                });
            let (view, generation) = self.bloom.encode(cx, scene, &plan, e, self.frame, timestamps)?;
            self.count_target(cx.stats, input.replay, generation, plan.bytes());
            Some((view, generation))
        });
        // With FXAA the composite writes the LDR intermediate L (pooled, the output's size, the
        // same rect) and the FXAA pass writes the output.
        let ldr = group.fxaa.map(|_| {
            let desc = PoolDesc {
                size: input.output_size,
                format: LDR_FORMAT,
                mips: 1,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            };
            let index = cx.pool.acquire(cx.device, desc);
            let texture = cx.pool.get(index);
            let bytes = 4 * input.output_size.0 as u64 * input.output_size.1 as u64;
            let (view, generation) = (texture.view.clone(), texture.generation);
            self.count_target(cx.stats, input.replay, generation, bytes);
            (view, generation)
        });
        let kind = OutputKind::of(input.output_format, input.output_manual_srgb);
        let (composite_format, composite_kind) = match ldr {
            Some(_) => (LDR_FORMAT, OutputKind::Srgb),
            None => (input.output_format, kind),
        };
        let composite_key = self.composite_pipeline(cx.device, composite_format, composite_kind, group.hook);
        let lut = group
            .lut
            .and_then(|h| self.luts.get(&h))
            .map(|l| (l.view.clone(), l.generation));
        if let Some(q) = group.fxaa {
            self.fxaa_pipeline(cx.device, q, input.output_format, kind);
        }
        let Some((params_bg, _, _)) = &self.params_bg else {
            return;
        };
        let frame = self.frame;
        let (inputs_bgl, dummy) = (&self.inputs_bgl, &self.dummy);
        let distortion_target = input.distortion.and_then(|_| self.distortion.get(&input.camera));
        let distortion = distortion_target.map_or(0, |t| t.generation);
        if let Some(t) = distortion_target
            && !input.replay
        {
            // Rgba8Unorm: 4 bytes per texel.
            cx.stats.target_bytes += 4 * t.size.0 as u64 * t.size.1 as u64;
        }
        let key = (
            input.scene_generation,
            distortion,
            bloom.as_ref().map_or(0, |(_, g)| *g),
            exposure.as_ref().map_or(0, |(_, g)| *g),
            lut.as_ref().map_or(0, |(_, g)| *g),
        );
        let entry = |binding, view| wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::TextureView(view),
        };
        let inputs = &mut self.inputs_bgs.entry(key).or_insert_with(|| {
            let bg = cx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("space3d post inputs"),
                layout: inputs_bgl,
                entries: &[
                    entry(0, input.scene),
                    entry(1, bloom.as_ref().map_or(dummy, |(view, _)| view)),
                    entry(2, exposure.as_ref().map_or(dummy, |(view, _)| view)),
                    entry(3, lut.as_ref().map_or(dummy, |(view, _)| view)),
                    entry(4, input.distortion.unwrap_or(dummy)),
                    // Never L: the composite renders into it.
                    entry(5, dummy),
                ],
            });
            (bg, frame)
        });
        inputs.1 = frame;
        let output_load = if input.output_clear {
            wgpu::LoadOp::Clear(wgpu::Color::BLACK)
        } else {
            wgpu::LoadOp::Load
        };
        let offsets = [(input.index * self.slot) as u32, self.seeds.offset(input.index, frame)];
        let (composite_target, composite_load) = match &ldr {
            Some((view, _)) => (view, wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)),
            None => (input.output, output_load),
        };
        let pipeline = &self.composite[&composite_key];
        let pass = FullscreenPass {
            label: "space3d composite",
            scope: ("post.composite", input.camera),
            target: composite_target,
            load: composite_load,
            viewport: input.output_viewport,
            scissor: input.output_rect,
        };
        pass.draw(cx, pipeline, params_bg, offsets, &inputs.0);
        let (Some(q), Some((view, generation))) = (group.fxaa, &ldr) else {
            return;
        };
        let fxaa_bg = &mut self.fxaa_bgs.entry(*generation).or_insert_with(|| {
            let bg = cx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("space3d fxaa inputs"),
                layout: inputs_bgl,
                entries: &[
                    entry(0, dummy),
                    entry(1, dummy),
                    entry(2, dummy),
                    entry(3, dummy),
                    entry(4, dummy),
                    entry(5, view),
                ],
            });
            (bg, frame)
        });
        fxaa_bg.1 = frame;
        let pipeline = &self.fxaa[&(q, input.output_format, kind)];
        let pass = FullscreenPass {
            label: "space3d fxaa",
            scope: ("post.fxaa", input.camera),
            target: input.output,
            load: output_load,
            viewport: input.output_viewport,
            scissor: input.output_rect,
        };
        pass.draw(cx, pipeline, params_bg, offsets, &fxaa_bg.0);
    }

    /// Count a pooled texture's memory in `RenderStats::target_bytes` once per frame (post
    /// groups of one size share it; replays count nothing).
    fn count_target(&mut self, stats: &mut RenderStats, replay: bool, generation: u64, bytes: u64) {
        if replay || self.counted.contains(&generation) {
            return;
        }
        let capacity = self.counted.capacity();
        self.counted.push(generation);
        stats.cpu_reallocations += u32::from(self.counted.capacity() != capacity);
        stats.target_bytes += bytes;
    }

    pub fn end_frame(&mut self, stats: &mut RenderStats) {
        stats.textures_created += std::mem::take(&mut self.created) + self.meters.take_created();
        stats.cpu_reallocations += std::mem::take(&mut self.seeds.reallocations)
            + std::mem::take(&mut self.meters.reallocations)
            + std::mem::take(&mut self.readback.reallocations);
        self.meters.end_frame();
        self.readback.end_frame();
        let tick = self.tick;
        self.luts.retain(|_, l| l.last_used + 60 > tick);
        let frame = self.frame;
        self.distortion.retain(|_, t| t.last_used + 60 > frame);
        // Bind groups of scene textures no longer composited (replaced or freed).
        self.inputs_bgs.retain(|_, (_, used)| *used + 3 > frame);
        self.bloom.sources.retain(|_, (_, used)| *used + 3 > frame);
        self.fxaa_bgs.retain(|_, (_, used)| *used + 3 > frame);
    }

    /// Register the built-in post nodes (`"space3d/distortion-meshes"`).
    pub fn register_nodes(nodes: &mut RenderNodes) {
        distortion_mesh::register(nodes);
    }

    /// Last read-back exposure of a post camera (stops, compensation included; with
    /// `FrameSettings::read_exposure`): metered cameras 1–3 frames late, others their manual
    /// exposure.
    pub fn camera_exposure(&self, camera: Entity) -> Option<f32> {
        self.readback.get(camera)
    }

    /// Forget a camera's temporal history: auto exposure snaps on its next frame.
    pub fn reset_history(&mut self, camera: Entity) {
        self.distortion.remove(&camera);
        self.meters.reset(camera);
    }
}

/// A full-screen post pass: its target, load op and the post group's rect.
struct FullscreenPass<'a> {
    label: &'static str,
    /// GPU timing label and camera.
    scope: (&'static str, Entity),
    target: &'a wgpu::TextureView,
    load: wgpu::LoadOp<wgpu::Color>,
    viewport: [f32; 4],
    scissor: [u32; 4],
}

impl FullscreenPass<'_> {
    /// Record the pass: one full-screen triangle with the group's parameters and seeds
    /// (`offsets`) and `inputs`.
    fn draw(
        &self,
        cx: &mut PostEncode<'_>,
        pipeline: &wgpu::RenderPipeline,
        params: &wgpu::BindGroup,
        offsets: [u32; 2],
        inputs: &wgpu::BindGroup,
    ) {
        let mut pass = cx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(self.label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: self.target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: self.load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            timestamp_writes: cx.timing.render(self.scope.0, Some(self.scope.1)),
            ..Default::default()
        });
        let [x, y, w, h] = self.viewport;
        pass.set_viewport(x, y, w, h, 0.0, 1.0);
        let [sx, sy, sw, sh] = self.scissor;
        pass.set_scissor_rect(sx, sy, sw, sh);
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, params, &offsets);
        pass.set_bind_group(1, inputs, &[]);
        pass.draw(0..3, 0..1);
        drop(pass);
        cx.stats.render_passes += 1;
        cx.stats.fullscreen_passes += 1;
        cx.stats.post_passes += 1;
    }
}

/// A post pipeline: the full-screen triangle and fragment `entry` of `module`, writing `format`
/// with the output kind `kind` (`S3D_POST_OUT`).
fn fullscreen_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    entry: &str,
    format: wgpu::TextureFormat,
    kind: OutputKind,
) -> wgpu::RenderPipeline {
    let constants = [("S3D_POST_OUT", kind.id())];
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(entry),
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
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                ..Default::default()
            },
            targets: &[Some(format.into())],
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn create_distortion_target(
    device: &wgpu::Device,
    (w, h): (u32, u32),
    frame: u64,
    generation: u64,
) -> DistortionTarget {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("space3d distortion buffer"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DISTORTION_FORMAT,
        // COPY_SRC: tests read the buffer back.
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    DistortionTarget {
        size: (w, h),
        view: texture.create_view(&Default::default()),
        last_used: frame,
        generation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distortion_targets_persist_per_camera() {
        let Ok(r) = pollster::block_on(crate::Renderer::new_headless(4, 4)) else {
            eprintln!("skipping GPU test: no adapter available");
            return;
        };
        let mut post = PostState::new(r.device(), r.graphics_caps());
        let mut world = World::new();
        let (a, b) = (world.spawn(()), world.spawn(()));
        let (_, size) = post.distortion_target(r.device(), a, [0, 0, 101, 60], DistortionResolution::Half);
        assert_eq!(size, (51, 30));
        let (_, size) = post.distortion_target(r.device(), b, [0, 0, 101, 60], DistortionResolution::Full);
        assert_eq!(size, (101, 60));
        assert_eq!(post.distortion.len(), 2);
        let plan = PostGroupPlan {
            camera: a,
            settings: PostProcess::default(),
            mode: FrameMode::Hdr,
            scene_rect: [0, 0, 4, 4],
            output_rect: [0, 0, 4, 4],
            scene_size: (4, 4),
            output_size: (4, 4),
            scene_alpha: true,
            transparent: false,
            passthrough: false,
            distortion: false,
        };
        let mut reallocs = 0;
        assert!(
            post.plan_group(&world, &RenderAssets::default(), &plan, &mut reallocs)
                .distortion
        );
        let passthrough = PostGroupPlan {
            passthrough: true,
            ..plan
        };
        assert!(
            !post
                .plan_group(&world, &RenderAssets::default(), &passthrough, &mut reallocs)
                .distortion
        );
        post.reset_history(a);
        let mut stats = RenderStats::default();
        post.end_frame(&mut stats);
        assert_eq!(post.distortion.len(), 1);
        assert_eq!(post.camera_exposure(b), None);
        let mut nodes = RenderNodes::default();
        PostState::register_nodes(&mut nodes);
        assert!(nodes.contains("space3d/distortion-meshes"));
        assert_eq!(r.gpu_error_count(), 0);
    }
}
