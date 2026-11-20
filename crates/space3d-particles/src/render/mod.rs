//! The particles render node, `"space3d/particles"` in `RenderNodes` (graphics spec §2.5).
//!
//! * Slots: `[Transparent @100 (reading DEPTH iff ParticleSettings::soft_particles),
//!   Distortion @100]`. Soft particles therefore draw in the view's SceneRead pass (§1.6), after
//!   the built-in blended meshes, as one slot, so their own back-to-front order is never split.
//! * `prepare` (once per frame, inside `render_frame`): replay the pending ticks of the tick log
//!   — with compute kernels recorded into the frame's encoder (GPU backend, `crate::gpu`) or in
//!   Rust (CPU backend, `crate::cpu`) — build the render instances (GPU compaction or CPU pack),
//!   lay the render-time bridges of emitters moved since the last tick (world-space emission
//!   from the tick's pose to the render pose, built on the CPU and drawn directly on both
//!   backends: `Runtime::bridge`), cull and sort the draws of every camera view, and upload what
//!   changed through the renderer's staging ring. A frame without new ticks, local layers,
//!   moved emitters or extrapolation uploads nothing.
//! * Ribbons and beams (`strips`) are built on the CPU every frame there are any, on both
//!   backends: the ribbon tracks the tick recorded plus a head at each emitter's render pose,
//!   and beams from the render poses of their ends. Textures (`textures`: the procedural
//!   built-ins, and assets by handle) get their bind groups before the draws are planned.
//! * `draw`: bind groups 0 (view) and 1 (scene depth) of the frame, group 2 (LUT, per-view
//!   flags, per-layer draw parameters), group 3 for textured layers, one instanced strip per
//!   draw item (an indirect draw on the GPU backend, its instance count written by the
//!   compaction; a strip segment per instance for ribbons and beams).
//! * Pipelines per `PassTarget`, built in `prewarm` (WebGL2 links programs at creation). The
//!   shader reads scene depth according to `PassTarget::depth_access`: the hardware test plus
//!   `s3d_soft_fade` with a read-only attachment, a manual `s3d_scene_occluded` test plus the
//!   fade without one (GL), and the hardware test alone in the MSAA in-pass fallback.
//! * Colors go through `s3d_output` with the pass encoding, exposure and tonemapper, so particles
//!   match meshes in every frame mode; in `FrameMode::Direct` (no tonemapper) they soft-clip
//!   with `1 − e^−x` (`ParticleSettings::direct_soft_clip`).

mod layers;
pub(crate) mod strips;
pub(crate) mod textures;

pub(crate) use layers::RenderLayers;

use crate::cpu::{
    BridgeInput, CpuSim, DIRECT, INSTANCE_SIZE, PackInput, PackItem, Particle, ParticleInstance, pack, pack_bridges,
};
use crate::gpu::sort::{GpuSort, MAX_SORT, SortJob, sort_size};
use crate::gpu::{GpuInput, GpuSim, PARTICLE_SIZE};
use crate::runtime::{ParticleRuntime, Runtime};
use crate::{ParticleBackend, ParticleEffects, ParticleSettings, ParticleStats, resolve_backend};
use layers::{GpuDrawParams, LayerKind, TexKey};
use space3d_core::glam::{Quat, Vec3};
use space3d_core::{Entity, GlobalTransform, Parent, Time, Transform};
use space3d_render::wgpu;
use space3d_render::{
    Aabb, DISTORTION_BLEND, DISTORTION_WGSL, DepthAccess, DrawContext, FrameMode, OUTPUT_WGSL, OutputEncoding,
    PassTarget, PhaseSlot, PrepareContext, RenderNode, RenderNodes, RenderPhase, SceneReads, SlotMask, Tonemapper,
    VIEW_WGSL, ViewInfo, ViewPrepareContext, scene_wgsl_for,
};
use strips::{POINT_SIZE, StripInput, StripPoint, StripScratch};

/// Name of the node in `RenderNodes`.
pub(crate) const NODE_NAME: &str = "space3d/particles";
/// Slot of color sprites.
pub(crate) const SLOT_COLOR: usize = 0;
/// Slot of distortion sprites (the post group's distortion buffer).
pub(crate) const SLOT_DISTORTION: usize = 1;

const COMMON_WGSL: &str = include_str!("../shaders/particles.wgsl");
const SPRITE_WGSL: &str = include_str!("../shaders/sprite.wgsl");
const STRIP_WGSL: &str = include_str!("../shaders/strip.wgsl");
const DISTORTION_SPRITE_WGSL: &str = include_str!("../shaders/distortion.wgsl");

static SLOTS_SOFT: [PhaseSlot; 2] = [
    PhaseSlot::new(RenderPhase::Transparent, 100).reading(SceneReads::DEPTH),
    PhaseSlot::new(RenderPhase::Distortion, 100),
];
static SLOTS_HARD: [PhaseSlot; 2] = [
    PhaseSlot::new(RenderPhase::Transparent, 100),
    PhaseSlot::new(RenderPhase::Distortion, 100),
];

/// Premultiplied "over" for every particle blend (additive = alpha 0).
pub(crate) const PARTICLE_BLEND: wgpu::BlendState = wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING;

/// Per-view flags (`S3dpView`, 32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParticleView {
    flags: [u32; 4],
    params: [f32; 4],
}

/// The draws of one camera view: color `view_items[start..end]`, then distortion
/// `view_items[end..dist_end]`; its per-particle sorted draws `sorted_draws[sorted.0..sorted.1]`;
/// uniform slot `uniform`.
#[derive(Clone, Copy, Debug)]
struct ViewList {
    camera: Entity,
    start: u32,
    end: u32,
    dist_end: u32,
    sorted: (u32, u32),
    uniform: u32,
}

/// A draw item of a `SortMode::ByDistance` layer as one view draws it: its instances sorted
/// back to front from `first` of the sorted instance buffer — the CPU-sorted one (direct draws:
/// the CPU backend, render-time bridges) or the GPU sort's (`gpu`, indirect draws).
#[derive(Clone, Copy, Debug, PartialEq)]
struct SortedDraw {
    item: u32,
    first: u32,
    gpu: bool,
}

/// The order-preserving u32 of a float (`order_key` in `sort.wgsl`).
fn order_key(f: f32) -> u32 {
    let bits = f.to_bits();
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

impl ViewList {
    /// The color draws (`distortion` false) or the distortion draws.
    fn range(&self, distortion: bool) -> std::ops::Range<usize> {
        if distortion {
            self.end as usize..self.dist_end as usize
        } else {
            self.start as usize..self.end as usize
        }
    }
}

/// The per-instance vertex layout of `ParticleInstance`.
const INSTANCE_ATTRIBUTES: [wgpu::VertexAttribute; 4] =
    wgpu::vertex_attr_array![0 => Float32x4, 1 => Float16x4, 2 => Float16x4, 3 => Uint32x2];
/// The two per-instance layouts of a strip segment: its first point, and the next one (the same
/// buffer, bound one point further).
const STRIP_A_ATTRIBUTES: [wgpu::VertexAttribute; 4] =
    wgpu::vertex_attr_array![0 => Float32x4, 1 => Float16x4, 2 => Float16x4, 3 => Uint32x2];
const STRIP_B_ATTRIBUTES: [wgpu::VertexAttribute; 4] =
    wgpu::vertex_attr_array![4 => Float32x4, 5 => Float16x4, 6 => Float16x4, 7 => Uint32x2];

/// What a pipeline draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum DrawKind {
    /// Sprite billboards (`sprite.wgsl`).
    Sprite,
    /// Ribbons and beams (`strip.wgsl`).
    Strip,
    /// Distortion sprites (`sprite.wgsl`'s vertex stage, `distortion.wgsl`) into the post
    /// distortion buffer.
    Distortion,
}

/// The texture binding of a pipeline (group 3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum TexVariant {
    /// No texture (three bind groups).
    None,
    /// A single-layer texture (`texture_2d`).
    D2,
    /// A texture array of 2 or more layers (`texture_2d_array`; flipbook frames).
    Array,
}

/// The texture declarations of a variant: `s3dp_tex(uv, layer, ddx, ddy)`.
fn texture_wgsl(tex: TexVariant) -> &'static str {
    match tex {
        TexVariant::None => {
            "const S3DP_TEXTURED: bool = false;\n\
             fn s3dp_tex(uv: vec2<f32>, layer: f32, gx: vec2<f32>, gy: vec2<f32>) -> vec4<f32> {\n    return vec4<f32>(1.0);\n}\n"
        }
        TexVariant::D2 => {
            "const S3DP_TEXTURED: bool = true;\n@group(3) @binding(0) var s3dp_tex_t: texture_2d<f32>;\n@group(3) @binding(1) var s3dp_tex_s: sampler;\n\
             fn s3dp_tex(uv: vec2<f32>, layer: f32, gx: vec2<f32>, gy: vec2<f32>) -> vec4<f32> {\n    \
             return textureSampleGrad(s3dp_tex_t, s3dp_tex_s, uv, gx, gy);\n}\n"
        }
        TexVariant::Array => {
            "const S3DP_TEXTURED: bool = true;\n@group(3) @binding(0) var s3dp_tex_t: texture_2d_array<f32>;\n@group(3) @binding(1) var s3dp_tex_s: sampler;\n\
             fn s3dp_tex(uv: vec2<f32>, layer: f32, gx: vec2<f32>, gy: vec2<f32>) -> vec4<f32> {\n    \
             let l = clamp(i32(layer), 0, i32(textureNumLayers(s3dp_tex_t)) - 1);\n    \
             return textureSampleGrad(s3dp_tex_t, s3dp_tex_s, uv, l, gx, gy);\n}\n"
        }
    }
}

/// The particle shader for a pass signature, a draw kind and a texture variant.
pub(crate) fn shader_for(output: OutputEncoding, access: DepthAccess, kind: DrawKind, tex: TexVariant) -> String {
    let body = match kind {
        DrawKind::Sprite => SPRITE_WGSL.to_string(),
        DrawKind::Strip => STRIP_WGSL.to_string(),
        DrawKind::Distortion => format!("{DISTORTION_WGSL}\n{SPRITE_WGSL}\n{DISTORTION_SPRITE_WGSL}"),
    };
    format!(
        "const S3DP_OUTPUT: u32 = {}u;\nconst S3DP_MANUAL_DEPTH: bool = {};\n{VIEW_WGSL}\n{}\n{OUTPUT_WGSL}\n{}\n{COMMON_WGSL}\n{body}",
        output.id(),
        access == DepthAccess::SampledOnly,
        scene_wgsl_for(access),
        texture_wgsl(tex),
    )
}

/// A pipeline's key: pass signature, draw kind, texture variant.
type PipelineKey = (PassTarget, DrawKind, TexVariant);

/// The bind group of a texture a layer samples (group 3).
struct TexBind {
    key: TexKey,
    /// `TextureRef::version` of an asset texture (0 for built-ins).
    version: u32,
    layered: bool,
    group: wgpu::BindGroup,
}

/// Bytes between consecutive `size`-byte uniforms bound at dynamic offsets aligned to `align`.
fn uniform_stride(size: usize, align: u64) -> u64 {
    (size as u64).next_multiple_of(align.max(1))
}

/// GPU state of one renderer.
struct Gpu {
    renderer: u64,
    /// Bytes between two layers' draw parameters and between two views' flags: their size
    /// rounded up to the dynamic offset alignment. The alignment alone is not enough: a
    /// WebGL2 device may report 32 bytes (Chrome's ANGLE on Metal), less than the 112-byte
    /// draw parameters, whose slots then overlapped and overran their buffer.
    draw_stride: u64,
    view_stride: u64,
    bind_layout: wgpu::BindGroupLayout,
    /// Pipeline layouts: untextured, with a `texture_2d`, with a `texture_2d_array` (group 3).
    pipeline_layouts: [wgpu::PipelineLayout; 3],
    /// Group 3 layouts: `texture_2d`, `texture_2d_array`.
    tex_layouts: [wgpu::BindGroupLayout; 2],
    /// Sampler of the built-in textures (trilinear; repeats along `u` for strips).
    tex_sampler: wgpu::Sampler,
    builtins: Vec<textures::GpuBuiltin>,
    tex_binds: Vec<TexBind>,
    strips: Option<(wgpu::Buffer, u64)>,
    sampler: wgpu::Sampler,
    lut: Option<(wgpu::Texture, wgpu::TextureView, u32)>,
    lut_version: Option<u64>,
    draws: Option<(wgpu::Buffer, u64)>,
    draws_version: Option<u64>,
    views: Option<(wgpu::Buffer, u64)>,
    view_mirror: Vec<GpuParticleView>,
    instances: Option<(wgpu::Buffer, u64)>,
    /// The CPU-sorted instances of `ByDistance` layers (per view).
    sorted: Option<(wgpu::Buffer, u64)>,
    bind_group: Option<wgpu::BindGroup>,
    pipelines: Vec<(PipelineKey, wgpu::RenderPipeline)>,
    /// Scratch for the draw-parameter upload.
    bytes: Vec<u8>,
}

impl Gpu {
    fn new(device: &wgpu::Device, renderer: u64, view: &wgpu::BindGroupLayout, scene: &wgpu::BindGroupLayout) -> Self {
        let align = device.limits().min_uniform_buffer_offset_alignment.max(16) as u64;
        let uniform = |binding, size: u64| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: wgpu::BufferSize::new(size),
            },
            count: None,
        };
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("space3d particles"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                uniform(2, std::mem::size_of::<GpuParticleView>() as u64),
                uniform(3, std::mem::size_of::<GpuDrawParams>() as u64),
            ],
        });
        let tex_layout = |dim, label| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: dim,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            })
        };
        let tex_layouts = [
            tex_layout(wgpu::TextureViewDimension::D2, "space3d particles texture"),
            tex_layout(wgpu::TextureViewDimension::D2Array, "space3d particles texture array"),
        ];
        let layout = |tex: Option<&wgpu::BindGroupLayout>| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("space3d particles"),
                bind_group_layouts: &[Some(view), Some(scene), Some(&bind_layout), tex],
                immediate_size: 0,
            })
        };
        let pipeline_layouts = [
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("space3d particles"),
                bind_group_layouts: &[Some(view), Some(scene), Some(&bind_layout)],
                immediate_size: 0,
            }),
            layout(Some(&tex_layouts[0])),
            layout(Some(&tex_layouts[1])),
        ];
        let tex_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("space3d particles textures"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("space3d particles LUT"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            renderer,
            draw_stride: uniform_stride(std::mem::size_of::<GpuDrawParams>(), align),
            view_stride: uniform_stride(std::mem::size_of::<GpuParticleView>(), align),
            bind_layout,
            pipeline_layouts,
            tex_layouts,
            tex_sampler,
            builtins: Vec::new(),
            tex_binds: Vec::new(),
            strips: None,
            sampler,
            lut: None,
            lut_version: None,
            draws: None,
            draws_version: None,
            views: None,
            view_mirror: Vec::new(),
            instances: None,
            sorted: None,
            bind_group: None,
            pipelines: Vec::new(),
            bytes: Vec::new(),
        }
    }

    /// The pipeline for `key` (built on first use).
    fn pipeline(&mut self, device: &wgpu::Device, key: PipelineKey) -> usize {
        if let Some(i) = self.pipelines.iter().position(|(k, _)| *k == key) {
            return i;
        }
        let (target, kind, tex) = key;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("space3d particles"),
            source: wgpu::ShaderSource::Wgsl(shader_for(target.output, target.depth_access, kind, tex).into()),
        });
        let layout = &self.pipeline_layouts[match tex {
            TexVariant::None => 0,
            TexVariant::D2 => 1,
            TexVariant::Array => 2,
        }];
        let sprite = [Some(wgpu::VertexBufferLayout {
            array_stride: INSTANCE_SIZE,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &INSTANCE_ATTRIBUTES,
        })];
        let strip = [
            Some(wgpu::VertexBufferLayout {
                array_stride: POINT_SIZE,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &STRIP_A_ATTRIBUTES,
            }),
            Some(wgpu::VertexBufferLayout {
                array_stride: POINT_SIZE,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &STRIP_B_ATTRIBUTES,
            }),
        ];
        let (vs, fs, buffers): (_, _, &[Option<wgpu::VertexBufferLayout<'_>>]) = match kind {
            DrawKind::Sprite => ("vs_sprite", "fs_sprite", &sprite),
            DrawKind::Strip => ("vs_strip", "fs_strip", &strip),
            DrawKind::Distortion => ("vs_distortion", "fs_distortion", &sprite),
        };
        // Distortion sprites add their offsets (One/One) into the distortion buffer.
        let blend = match kind {
            DrawKind::Distortion => DISTORTION_BLEND,
            _ => PARTICLE_BLEND,
        };
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(match kind {
                DrawKind::Sprite => "space3d particles sprite",
                DrawKind::Strip => "space3d particles strip",
                DrawKind::Distortion => "space3d particles distortion",
            }),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some(vs),
                compilation_options: Default::default(),
                buffers,
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            // Blended: test against the scene depth, never write it.
            depth_stencil: target.depth_stencil(wgpu::CompareFunction::LessEqual, false),
            multisample: target.multisample(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some(fs),
                compilation_options: Default::default(),
                targets: &[Some(target.color_target(Some(blend)))],
            }),
            multiview_mask: None,
            cache: None,
        });
        self.pipelines.push((key, pipeline));
        self.pipelines.len() - 1
    }

    /// The bind group of texture `key` (and whether it is an array), if ready.
    fn tex_bind(&self, key: TexKey) -> Option<&TexBind> {
        self.tex_binds.iter().find(|b| b.key == key)
    }
}

/// The pipeline kind of a render layer.
fn draw_kind(kind: LayerKind) -> DrawKind {
    match kind {
        LayerKind::Sprite => DrawKind::Sprite,
        LayerKind::Ribbon | LayerKind::Beam => DrawKind::Strip,
    }
}

/// Grow-only GPU buffer: `Some(new buffer)` when `current` is missing or too small.
fn ensure_buffer(
    device: &wgpu::Device,
    current: &mut Option<(wgpu::Buffer, u64)>,
    size: u64,
    usage: wgpu::BufferUsages,
    label: &'static str,
) -> bool {
    if current.as_ref().is_some_and(|(_, s)| *s >= size) {
        return false;
    }
    let size = size.max(256).next_power_of_two();
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    *current = Some((buffer, size));
    true
}

/// Render pose (position, rotation) of every effect instance: its entity's `GlobalTransform`
/// while attached, else its last emitter pose; and whether the emitter moved since the last
/// tick (its `Transform` chain, read the way the tick reads it, gives another pose: gameplay
/// moved it after the tick, e.g. in `Stage::Update`), which gives its world-space layers a
/// render-time bridge ([`crate::runtime::Runtime::bridge`]).
fn update_poses(poses: &mut Vec<(Vec3, Quat)>, moved: &mut Vec<bool>, rt: &Runtime, world: &space3d_core::World) {
    poses.clear();
    moved.clear();
    for inst in &rt.instances {
        let attached = inst.live && inst.attached && !inst.orphaned;
        let pose = attached
            .then(|| world.get::<&GlobalTransform>(inst.entity).ok())
            .flatten()
            .map(|g| {
                let (_, r, t) = g.0.to_scale_rotation_translation();
                (t, r)
            });
        poses.push(pose.unwrap_or((inst.pos, inst.rot)));
        let now = attached
            .then(|| {
                let t = world.get::<&Transform>(inst.entity).ok()?;
                let parent = world.get::<&Parent>(inst.entity).ok();
                let (p, r, broken) = crate::runtime::world_pose(world, &t, parent.as_deref());
                (!broken).then_some((p, r))
            })
            .flatten();
        moved.push(now.is_some_and(|(p, r)| p != inst.pos || r != inst.rot));
    }
}

/// The particles render node (registered by `ParticlesPlugin`).
pub(crate) struct ParticlesNode {
    soft: bool,
    enabled: bool,
    sim: CpuSim,
    /// The GPU backend's state (per renderer).
    gpu_sim: Option<GpuSim>,
    /// Backend of the last frame.
    backend: Option<ParticleBackend>,
    layers: RenderLayers,
    instances: Vec<ParticleInstance>,
    items: Vec<PackItem>,
    keys: Vec<(u32, u64, u32)>,
    poses: Vec<(Vec3, Quat)>,
    /// Per instance: the emitter moved since the last tick (`update_poses`).
    moved: Vec<bool>,
    /// Instances and items of the last pack of the simulated particles (CPU backend); the
    /// render-time bridges follow them.
    packed: (usize, usize),
    /// Bridge particles of the last frame.
    bridged: u32,
    /// The frame's ribbon and beam points (`render::strips`), their scratch and segments.
    strip_points: Vec<StripPoint>,
    strip_scratch: StripScratch,
    segments: u32,
    /// The strip points changed since they were last uploaded.
    strips_dirty: bool,
    view_items: Vec<u32>,
    views: Vec<ViewList>,
    view_uniforms: Vec<GpuParticleView>,
    /// Per-particle sorted draws of every view (`ViewList::sorted`).
    sorted_draws: Vec<SortedDraw>,
    /// The CPU-sorted instances, what the GPU buffer holds, and the sort's scratch keys.
    sorted_instances: Vec<ParticleInstance>,
    sorted_uploaded: Vec<ParticleInstance>,
    sort_keys: Vec<(u32, u32)>,
    /// What the CPU-sorted instances were built from: the instance stream's generation, the
    /// sorting views' depth rows and the sorted draws.
    sort_source: (u64, Vec<[f32; 4]>, Vec<SortedDraw>),
    sort_rows: Vec<[f32; 4]>,
    /// The GPU sort's jobs this frame, and its state (per renderer).
    sort_jobs: Vec<SortJob>,
    /// `instances.capacity()` when the CPU sort's per-particle buffers were last sized for it.
    sort_reserved: usize,
    gpu_sort: Option<GpuSort>,
    warned_sort: bool,
    /// Bumped whenever the CPU instance stream is rewritten (pack, bridges).
    pack_gen: u64,
    /// The packed instances changed since they were last uploaded.
    dirty: bool,
    packed_once: bool,
    last_extrapolation: f32,
    gpu: Option<Gpu>,
    warned_gpu_backend: bool,
    /// The frame bind groups of the last color draw (perf tests draw with them).
    #[cfg(test)]
    pub(crate) test_capture: Option<(wgpu::BindGroup, wgpu::BindGroup, u32, PassTarget)>,
}

/// What `simulate` reports to the stats.
struct Simulated {
    settings: ParticleSettings,
    reallocs: u32,
    ticks: u32,
    alive: u32,
    alive_frame: Option<u64>,
    dispatches: u32,
    bytes: u64,
}

impl ParticlesNode {
    pub(crate) fn new(settings: &ParticleSettings) -> Self {
        Self {
            soft: settings.soft_particles,
            enabled: settings.enabled,
            sim: CpuSim::default(),
            gpu_sim: None,
            backend: None,
            layers: RenderLayers::default(),
            instances: Vec::new(),
            items: Vec::new(),
            keys: Vec::new(),
            poses: Vec::new(),
            moved: Vec::new(),
            packed: (0, 0),
            bridged: 0,
            strip_points: Vec::new(),
            strip_scratch: StripScratch::default(),
            segments: 0,
            strips_dirty: false,
            view_items: Vec::new(),
            views: Vec::new(),
            view_uniforms: Vec::new(),
            sorted_draws: Vec::new(),
            sorted_instances: Vec::new(),
            sorted_uploaded: Vec::new(),
            sort_keys: Vec::new(),
            sort_source: (u64::MAX, Vec::new(), Vec::new()),
            sort_rows: Vec::new(),
            sort_jobs: Vec::new(),
            sort_reserved: 0,
            gpu_sort: None,
            warned_sort: false,
            pack_gen: 0,
            dirty: false,
            packed_once: false,
            last_extrapolation: 0.0,
            gpu: None,
            warned_gpu_backend: false,
            #[cfg(test)]
            test_capture: None,
        }
    }

    fn capacities(&self) -> [usize; 10] {
        [
            self.sorted_draws.capacity()
                + self.sorted_instances.capacity()
                + self.sorted_uploaded.capacity()
                + self.sort_keys.capacity()
                + self.sort_source.1.capacity()
                + self.sort_source.2.capacity()
                + self.sort_rows.capacity()
                + self.sort_jobs.capacity()
                + self.gpu_sort.as_ref().map_or(0, GpuSort::capacities),
            self.instances.capacity(),
            self.items.capacity(),
            self.keys.capacity(),
            self.poses.capacity() + self.moved.capacity(),
            self.view_items.capacity(),
            self.views.capacity(),
            self.view_uniforms.capacity(),
            self.gpu_sim.as_ref().map_or(0, GpuSim::capacities),
            self.strip_points.capacity() + self.strip_scratch.capacity(),
        ]
    }

    /// Replay the log on the frame's backend and build the draw items; publishes nothing.
    fn simulate(&mut self, ctx: &mut PrepareContext<'_>, rebuilt: bool) -> Option<Simulated> {
        let settings = ctx.resources.get::<ParticleSettings>().cloned().unwrap_or_default();
        if !ctx.resources.contains::<ParticleRuntime>() {
            return None;
        }
        if settings.soft_particles != self.soft {
            self.soft = settings.soft_particles;
            if let Some(nodes) = ctx.resources.get_mut::<RenderNodes>() {
                nodes.invalidate();
            }
        }
        // Device lost: the CPU backend carries on until the renderer is recreated.
        let compute = ctx.graphics.compute && !ctx.frame.device_lost;
        let backend = resolve_backend(settings.backend, Some(compute));
        if settings.backend == ParticleBackend::Gpu && backend == ParticleBackend::Cpu && !self.warned_gpu_backend {
            self.warned_gpu_backend = true;
            tracing::warn!("ParticleBackend::Gpu needs compute shaders and indirect draws; simulating on the CPU");
        }
        let switched = self.backend.is_some_and(|b| b != backend);
        if switched && backend == ParticleBackend::Cpu {
            // The GPU pool cannot be read back without waiting: the live particles are dropped.
            self.sim.pool.fill(Particle::default());
        }
        self.backend = Some(backend);
        let dt = ctx
            .resources
            .get::<Time>()
            .map_or(0.0, |t| t.fixed_delta_secs() * settings.time_scale.max(0.0));
        let extrapolate = if settings.extrapolate {
            ctx.frame.alpha * dt
        } else {
            0.0
        };
        // The runtime moves out for the frame (no allocation), so the context stays free for
        // uploads and compute passes.
        let mut rt = std::mem::take(&mut ctx.resources.get_mut::<ParticleRuntime>()?.rt);
        let mut out = Simulated {
            settings,
            reallocs: 0,
            ticks: 0,
            alive: 0,
            alive_frame: None,
            dispatches: 0,
            bytes: 0,
        };
        let layers = &self.layers;
        // Textured layers whose texture is not ready yet wait (distortion layers are drawn
        // like the others, in the Distortion slot).
        let ready = |l: &layers::LayerInfo| {
            l.texture
                .is_none_or(|k| self.gpu.as_ref().is_some_and(|g| g.tex_bind(k).is_some()))
        };
        let layer_of = |e: &crate::runtime::Entry| layers.layer(e.effect, e.layer).filter(|l| ready(l)).map(|l| l.pack);
        match backend {
            ParticleBackend::Gpu => {
                if self.gpu_sim.as_ref().is_none_or(|g| g.renderer != ctx.frame.renderer) {
                    self.gpu_sim = Some(GpuSim::new(ctx.device, ctx.frame.renderer));
                }
                update_poses(&mut self.poses, &mut self.moved, &rt, ctx.world);
                let gpu = self.gpu_sim.as_mut().expect("created above");
                let carry = switched.then_some(&self.sim.pool[..]);
                gpu.prepare(
                    ctx,
                    &mut rt,
                    &layer_of,
                    &GpuInput {
                        poses: &self.poses,
                        extrapolate,
                        dt,
                        max_catchup: out.settings.max_catchup_ticks,
                        carry,
                    },
                    &mut self.items,
                );
                let s = gpu.stats;
                (out.ticks, out.dispatches, out.bytes, out.reallocs) = (s.ticks, s.dispatches, s.bytes, s.created);
                if let Some((frame, alive)) = gpu.readback.latest {
                    (out.alive, out.alive_frame) = (alive, Some(frame));
                }
                // The simulated particles draw from the GPU pool; the bridges are direct draws.
                self.instances.clear();
            }
            _ => {
                out.reallocs += self.sim.replay(&mut rt, out.settings.max_catchup_ticks);
                out.ticks = self.sim.ticks;
                update_poses(&mut self.poses, &mut self.moved, &rt, ctx.world);
                let has_local = rt.entries.iter().any(|e| e.live && e.sim.local);
                if switched
                    || !self.packed_once
                    || self.sim.ticks > 0
                    || rebuilt
                    || has_local
                    || extrapolate != self.last_extrapolation
                {
                    self.packed_once = true;
                    self.last_extrapolation = extrapolate;
                    pack(
                        &PackInput {
                            rt: &rt,
                            pool: &self.sim.pool,
                            poses: &self.poses,
                            extrapolate,
                        },
                        &layer_of,
                        &mut self.keys,
                        &mut self.instances,
                        &mut self.items,
                    );
                    self.packed = (self.instances.len(), self.items.len());
                    self.dirty = true;
                    self.pack_gen += 1;
                } else {
                    // Last frame's bridges go; this frame's follow the same pack.
                    self.instances.truncate(self.packed.0);
                    self.items.truncate(self.packed.1);
                }
                out.alive = self.packed.0 as u32;
                out.alive_frame = Some(ctx.frame.frame);
            }
        }
        // Render-time bridges: world-space emission from each moved emitter's last tick pose
        // to its render pose (both backends; CPU-generated, direct draws).
        let bridged = pack_bridges(
            &BridgeInput {
                rt: &rt,
                poses: &self.poses,
                moved: &self.moved,
                dt,
                age: extrapolate,
            },
            &layer_of,
            &mut self.keys,
            &mut self.instances,
            &mut self.items,
        );
        if bridged > 0 || self.bridged > 0 {
            self.dirty = true;
            self.pack_gen += 1;
        }
        self.bridged = bridged;
        // Ribbons and beams: rebuilt every frame there are any (they follow the render poses).
        let had = !self.strip_points.is_empty();
        self.segments = 0;
        if let Some(effects) = ctx.resources.get::<ParticleEffects>()
            && (self.layers.has_strips() || had)
        {
            let start = self.items.len();
            self.segments = strips::build(
                &StripInput {
                    rt: &rt,
                    effects,
                    layers: &self.layers,
                    poses: &self.poses,
                    world: ctx.world,
                    extrapolate,
                    budget: out.settings.max_ribbon_segments,
                },
                &mut self.strip_scratch,
                &mut self.strip_points,
                &mut self.items,
            );
            // Strips whose texture is not ready yet wait.
            let (gpu, info) = (self.gpu.as_ref(), &self.layers.info);
            let mut k = start;
            while k < self.items.len() {
                let tex = info[self.items[k].layer as usize].texture;
                if tex.is_some_and(|t| gpu.is_none_or(|g| g.tex_bind(t).is_none())) {
                    self.items.remove(k);
                } else {
                    k += 1;
                }
            }
            self.strips_dirty |= had || !self.strip_points.is_empty();
        }
        out.reallocs += rt.reallocations;
        rt.reallocations = 0;
        if let Some(r) = ctx.resources.get_mut::<ParticleRuntime>() {
            r.rt = rt;
        }
        Some(out)
    }

    /// The backend of the last frame and a copy of its pool (tests; waits for the GPU).
    pub(crate) fn read_pool(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Option<(ParticleBackend, Vec<Particle>)> {
        match self.backend? {
            ParticleBackend::Gpu => {
                let pool = self.gpu_sim.as_ref()?.pool_buffer()?;
                let size = pool.size();
                let staging = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("space3d particles pool readback (tests)"),
                    size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let mut encoder = device.create_command_encoder(&Default::default());
                encoder.copy_buffer_to_buffer(pool, 0, &staging, 0, size);
                queue.submit([encoder.finish()]);
                let slice = staging.slice(..);
                slice.map_async(wgpu::MapMode::Read, |_| {});
                device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
                let data = slice.get_mapped_range().ok()?;
                let pool: Vec<Particle> =
                    bytemuck::cast_slice(&data[..(size / PARTICLE_SIZE * PARTICLE_SIZE) as usize]).to_vec();
                Some((ParticleBackend::Gpu, pool))
            }
            b => Some((b, self.sim.pool.clone())),
        }
    }

    /// Cull and sort the draw items of every camera view; per-view flags.
    fn plan_views(&mut self, ctx: &PrepareContext<'_>, settings: &ParticleSettings) {
        self.view_items.clear();
        self.views.clear();
        self.view_uniforms.clear();
        self.sorted_draws.clear();
        self.sort_jobs.clear();
        self.sort_rows.clear();
        let (mut cpu_sorted, mut gpu_sorted) = (0u32, 0u32);
        for (vi, v) in ctx.views.iter().enumerate() {
            let start = self.view_items.len();
            let mut end = start;
            let sorted_start = self.sorted_draws.len() as u32;
            if settings.enabled {
                // Color draws first, then distortion draws (culled alike).
                for distortion in [false, true] {
                    for (i, it) in self.items.iter().enumerate() {
                        if self.layers.info[it.layer as usize].distortion != distortion {
                            continue;
                        }
                        // Unbounded items (world attractors on the GPU backend) are never culled.
                        let unbounded = !(it.min.is_finite() && it.max.is_finite());
                        if unbounded
                            || v.frustum.intersects_aabb(&Aabb {
                                min: it.min,
                                max: it.max,
                            })
                        {
                            self.view_items.push(i as u32);
                        }
                    }
                    if !distortion {
                        end = self.view_items.len();
                    }
                }
                let (items, layers, view) = (&self.items, &self.layers.info, v.view);
                // Order, then alpha/premultiplied before additive, then back to front for sorted
                // layers; ties by layer and allocation serial, then the simulated particles
                // before their render-time bridge (deterministic).
                let key = |i: u32| {
                    let it = &items[i as usize];
                    let l = &layers[it.layer as usize];
                    let depth = if l.pack.batched {
                        0.0
                    } else {
                        view.transform_point3(it.center).z
                    };
                    (l.order, l.additive, depth, it.layer, it.serial)
                };
                self.view_items[start..end].sort_unstable_by(|&a, &b| {
                    let (ka, kb) = (key(a), key(b));
                    ka.0.cmp(&kb.0)
                        .then(ka.1.cmp(&kb.1))
                        .then(ka.2.total_cmp(&kb.2))
                        .then(ka.3.cmp(&kb.3))
                        .then(ka.4.cmp(&kb.4))
                        .then(a.cmp(&b))
                });
                // Distortion adds up (order-independent): layer and serial, for determinism.
                self.view_items[end..].sort_unstable_by_key(|&i| {
                    let it = &items[i as usize];
                    (it.layer, it.serial, i)
                });
                // `ByDistance` layers: their particles back to front in this view, sorted on the
                // CPU (direct draws) or by the GPU sort (indirect draws).
                let row = v.view.row(2).to_array();
                let mut rows_pushed = false;
                for &i in &self.view_items[start..end] {
                    let it = &self.items[i as usize];
                    if !self.layers.info[it.layer as usize].sorted {
                        continue;
                    }
                    if it.args == DIRECT {
                        self.sorted_draws.push(SortedDraw {
                            item: i,
                            first: cpu_sorted,
                            gpu: false,
                        });
                        cpu_sorted += it.count;
                        if !rows_pushed {
                            self.sort_rows.push(row);
                            rows_pushed = true;
                        }
                    } else if let Some(sim) = self.gpu_sim.as_ref() {
                        if sort_size(it.count) > MAX_SORT {
                            if !self.warned_sort {
                                self.warned_sort = true;
                                tracing::warn!(
                                    particles = it.count,
                                    "a SortMode::ByDistance layer holds more than 65 536 particles: drawn unsorted"
                                );
                            }
                            continue;
                        }
                        self.sort_jobs.push(SortJob {
                            src: it.first,
                            capacity: it.count,
                            count_word: (sim.args_offset(it.args) / 4) as u32 + 1,
                            row,
                        });
                        self.sorted_draws.push(SortedDraw {
                            item: i,
                            first: gpu_sorted,
                            gpu: true,
                        });
                        gpu_sorted += it.count;
                    }
                }
            }
            self.views.push(ViewList {
                camera: v.camera,
                start: start as u32,
                end: end as u32,
                dist_end: self.view_items.len() as u32,
                sorted: (sorted_start, self.sorted_draws.len() as u32),
                uniform: vi as u32,
            });
            let soft_clip = settings.direct_soft_clip && soft_clips(v);
            self.view_uniforms.push(GpuParticleView {
                flags: [soft_clip as u32, settings.soft_particles as u32, 0, 0],
                params: [settings.max_screen_size.max(0.0), 0.0, 0.0, 0.0],
            });
        }
        self.sort_on_cpu(ctx.views);
    }

    /// Record the GPU sort of this frame's `ByDistance` draw groups (GPU backend); returns the
    /// buffers created and the bytes uploaded.
    fn sort_on_gpu(&mut self, ctx: &mut PrepareContext<'_>) -> (u32, u64) {
        let Some(sim) = self.gpu_sim.as_ref() else {
            return (0, 0);
        };
        if self.sort_jobs.is_empty() && self.gpu_sort.is_none() {
            return (0, 0);
        }
        let Some((work, instances)) = sim.draw_buffers() else {
            return (0, 0);
        };
        let compacted = sim.stats.compacted;
        let (work, instances) = (work.clone(), instances.clone());
        let sort = self.gpu_sort.get_or_insert_with(|| GpuSort::new(ctx.device));
        sort.prepare(ctx, (&work, &instances), &self.sort_jobs, compacted)
    }

    /// Sort the direct draws of `ByDistance` layers per view (`sorted_draws` without `gpu`)
    /// into `sorted_instances`: farthest first by view depth, ties by instance order (the GPU
    /// sort's keys). Skipped while the instance stream and the sorting views are unchanged.
    fn sort_on_cpu(&mut self, views: &[space3d_render::ViewInfo]) {
        if self.sort_rows.is_empty() {
            self.sorted_instances.clear();
            return;
        }
        let same_draws = self.sort_source.2.len() == self.sorted_draws.len()
            && self.sort_source.2.iter().zip(&self.sorted_draws).all(|(a, b)| a == b);
        if self.sort_source.0 == self.pack_gen && self.sort_source.1 == self.sort_rows && same_draws {
            return;
        }
        self.sort_source.0 = self.pack_gen;
        self.sort_source.1.clear();
        self.sort_source.1.extend_from_slice(&self.sort_rows);
        self.sort_source.2.clear();
        self.sort_source.2.extend_from_slice(&self.sorted_draws);
        self.sorted_instances.clear();
        for list in &self.views {
            let Some(v) = views.get(list.uniform as usize) else {
                continue;
            };
            let row = v.view.row(2);
            for d in &self.sorted_draws[list.sorted.0 as usize..list.sorted.1 as usize] {
                if d.gpu {
                    continue;
                }
                let it = &self.items[d.item as usize];
                let instances = &self.instances[it.first as usize..(it.first + it.count) as usize];
                self.sort_keys.clear();
                for (j, p) in instances.iter().enumerate() {
                    let z = row.dot(space3d_core::glam::Vec4::new(
                        p.pos_size[0],
                        p.pos_size[1],
                        p.pos_size[2],
                        1.0,
                    ));
                    self.sort_keys.push((order_key(z), j as u32));
                }
                self.sort_keys.sort_unstable();
                debug_assert_eq!(self.sorted_instances.len(), d.first as usize);
                self.sorted_instances
                    .extend(self.sort_keys.iter().map(|&(_, j)| instances[j as usize]));
            }
        }
    }

    /// Create or refresh the GPU resources and upload what changed; returns the bytes uploaded
    /// and the buffers or textures (re)created.
    /// Create the GPU state for this renderer, the built-in textures the layers use and the
    /// bind groups of every texture they sample (before the draws are planned, so a texture
    /// drawn on the first frame is ready then). Returns the textures and bind groups created.
    fn prepare_textures(&mut self, ctx: &mut PrepareContext<'_>) -> u32 {
        let device = ctx.device;
        if self.gpu.as_ref().is_none_or(|g| g.renderer != ctx.frame.renderer) {
            self.gpu = Some(Gpu::new(
                device,
                ctx.frame.renderer,
                &ctx.layouts.view,
                &ctx.layouts.scene,
            ));
            self.dirty = true;
            self.strips_dirty = true;
        }
        let Some(gpu) = self.gpu.as_mut() else {
            return 0;
        };
        let mut created = 0;
        for info in &self.layers.info {
            let Some(key) = info.texture else {
                continue;
            };
            let (view, sampler, layered, version) = match key {
                TexKey::Builtin(kind) => {
                    if gpu.tex_bind(key).is_some() {
                        continue;
                    }
                    if !gpu.builtins.iter().any(|b| b.kind == kind) {
                        gpu.builtins.push(textures::create_builtin(device, ctx.queue, kind));
                        created += 1;
                    }
                    let b = gpu.builtins.iter().find(|b| b.kind == kind).expect("created above");
                    (b.view.clone(), gpu.tex_sampler.clone(), b.layered, 0)
                }
                TexKey::Handle(h) => {
                    let Some(t) = ctx.texture(h) else {
                        continue;
                    };
                    if gpu
                        .tex_bind(key)
                        .is_some_and(|b| b.version == t.version && b.layered == t.layered)
                    {
                        continue;
                    }
                    (t.view, t.sampler, t.layered, t.version)
                }
            };
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("space3d particles texture"),
                layout: &gpu.tex_layouts[layered as usize],
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            });
            created += 1;
            let bind = TexBind {
                key,
                version,
                layered,
                group,
            };
            match gpu.tex_binds.iter_mut().find(|b| b.key == key) {
                Some(b) => *b = bind,
                None => gpu.tex_binds.push(bind),
            }
        }
        created
    }

    fn upload(&mut self, ctx: &mut PrepareContext<'_>) -> (u64, u32) {
        let device = ctx.device;
        let Some(gpu) = self.gpu.as_mut() else {
            return (0, 0);
        };
        let (mut bytes, mut created) = (0u64, 0u32);
        let mut rebind = gpu.bind_group.is_none();

        // The curve LUT (only when the effects changed).
        if gpu.lut_version != self.layers.version {
            let rows = self.layers.rows.max(1);
            if gpu.lut.as_ref().is_none_or(|l| l.2 != rows) {
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("space3d particles LUT"),
                    size: wgpu::Extent3d {
                        width: crate::effect::LUT_WIDTH,
                        height: rows,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                let view = texture.create_view(&Default::default());
                gpu.lut = Some((texture, view, rows));
                created += 1;
                rebind = true;
            }
            if let Some((texture, _, rows)) = &gpu.lut {
                let data: &[u8] = bytemuck::cast_slice(&self.layers.lut);
                ctx.queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    data,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(crate::effect::LUT_WIDTH * 8),
                        rows_per_image: Some(*rows),
                    },
                    wgpu::Extent3d {
                        width: crate::effect::LUT_WIDTH,
                        height: *rows,
                        depth_or_array_layers: 1,
                    },
                );
                bytes += data.len() as u64;
            }
            gpu.lut_version = self.layers.version;
        }

        // Draw parameters, one aligned slot per render layer.
        let (draw_stride, view_stride) = (gpu.draw_stride, gpu.view_stride);
        if gpu.draws_version != self.layers.version {
            let n = self.layers.params.len().max(1) as u64;
            if ensure_buffer(
                device,
                &mut gpu.draws,
                n * draw_stride,
                wgpu::BufferUsages::UNIFORM,
                "space3d particles draws",
            ) {
                created += 1;
                rebind = true;
            }
            gpu.bytes.clear();
            for p in &self.layers.params {
                gpu.bytes.extend_from_slice(bytemuck::bytes_of(p));
                gpu.bytes
                    .resize(gpu.bytes.len().next_multiple_of(draw_stride as usize), 0);
            }
            if let (Some((buffer, _)), false) = (&gpu.draws, gpu.bytes.is_empty()) {
                ctx.upload(buffer, 0, &gpu.bytes);
                bytes += gpu.bytes.len() as u64;
            }
            gpu.draws_version = self.layers.version;
        }

        // Per-view flags (only the views whose flags changed).
        let n = self.view_uniforms.len().max(1) as u64;
        if ensure_buffer(
            device,
            &mut gpu.views,
            n * view_stride,
            wgpu::BufferUsages::UNIFORM,
            "space3d particles views",
        ) {
            created += 1;
            rebind = true;
            gpu.view_mirror.clear();
        }
        if let Some((buffer, _)) = &gpu.views {
            for (i, u) in self.view_uniforms.iter().enumerate() {
                if gpu.view_mirror.get(i) != Some(u) {
                    ctx.upload(buffer, i as u64 * view_stride, bytemuck::bytes_of(u));
                    bytes += std::mem::size_of::<GpuParticleView>() as u64;
                    if i < gpu.view_mirror.len() {
                        gpu.view_mirror[i] = *u;
                    } else {
                        gpu.view_mirror.push(*u);
                    }
                }
            }
        }

        // Instances (only after a repack).
        if self.dirty {
            let size = self.instances.len() as u64 * INSTANCE_SIZE;
            if ensure_buffer(
                device,
                &mut gpu.instances,
                size,
                wgpu::BufferUsages::VERTEX,
                "space3d particles instances",
            ) {
                created += 1;
            }
            if let (Some((buffer, _)), true) = (&gpu.instances, size > 0) {
                ctx.upload(buffer, 0, bytemuck::cast_slice(&self.instances));
                bytes += size;
            }
            self.dirty = false;
        }

        // CPU-sorted instances of `ByDistance` layers (only when they changed).
        let sorted: &[u8] = bytemuck::cast_slice(&self.sorted_instances);
        if sorted != bytemuck::cast_slice::<_, u8>(&self.sorted_uploaded) {
            let size = sorted.len() as u64;
            if ensure_buffer(
                device,
                &mut gpu.sorted,
                size,
                wgpu::BufferUsages::VERTEX,
                "space3d particles sorted instances",
            ) {
                created += 1;
            }
            if let (Some((buffer, _)), true) = (&gpu.sorted, size > 0) {
                ctx.upload(buffer, 0, sorted);
                bytes += size;
            }
            self.sorted_uploaded.clear();
            self.sorted_uploaded.extend_from_slice(&self.sorted_instances);
        }

        // Ribbon and beam points (rebuilt every frame there are any).
        if self.strips_dirty {
            let size = self.strip_points.len() as u64 * POINT_SIZE;
            if ensure_buffer(
                device,
                &mut gpu.strips,
                size,
                wgpu::BufferUsages::VERTEX,
                "space3d particles strips",
            ) {
                created += 1;
            }
            if let (Some((buffer, _)), true) = (&gpu.strips, size > 0) {
                ctx.upload(buffer, 0, bytemuck::cast_slice(&self.strip_points));
                bytes += size;
            }
            self.strips_dirty = false;
        }

        if rebind && let (Some((_, lut, _)), Some((draws, _)), Some((views, _))) = (&gpu.lut, &gpu.draws, &gpu.views) {
            gpu.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("space3d particles"),
                layout: &gpu.bind_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(lut),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&gpu.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: views,
                            offset: 0,
                            size: wgpu::BufferSize::new(std::mem::size_of::<GpuParticleView>() as u64),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: draws,
                            offset: 0,
                            size: wgpu::BufferSize::new(std::mem::size_of::<GpuDrawParams>() as u64),
                        }),
                    },
                ],
            }));
        }
        (bytes, created)
    }
}

/// Whether `direct_soft_clip` applies in view `v`: its colors reach the display without a
/// tonemapper — Direct mode, and the Ldr passthrough of a render scale below 1 without post
/// (game#3: the same look at every render scale). Not in Hdr mode, nor in the Ldr fallback of
/// a post camera, whose tonemapper runs per fragment (`OutputEncoding::Tonemapped`).
fn soft_clips(v: &ViewInfo) -> bool {
    match v.mode {
        FrameMode::Direct => true,
        FrameMode::Hdr => false,
        _ => v.output != OutputEncoding::Tonemapped || v.tonemapper == Tonemapper::None,
    }
}

impl RenderNode for ParticlesNode {
    fn slots(&self) -> &[PhaseSlot] {
        if self.soft { &SLOTS_SOFT } else { &SLOTS_HARD }
    }

    fn prepare(&mut self, ctx: &mut PrepareContext<'_>) {
        // (a span of the kind the schedule opens round every system: the profiler lists it under the renderer's `render.prepare.nodes`)
        let _span = tracing::trace_span!("system", name = "render.prepare.nodes.particles").entered();
        let before = self.capacities();
        let rebuilt = ctx
            .resources
            .get::<ParticleEffects>()
            .is_some_and(|fx| self.layers.sync(fx));
        let textures = if ctx.frame.device_lost {
            0
        } else {
            self.prepare_textures(ctx)
        };
        let Some(sim) = self.simulate(ctx, rebuilt) else {
            return;
        };
        // The CPU-sorted particles of `ByDistance` layers are a subset of the frame's
        // instances: their buffers grow with the instance stream, not at every new peak of
        // their own (which the steady state would count as allocations).
        if self.instances.capacity() != self.sort_reserved {
            self.sort_reserved = self.instances.capacity();
            let n = self.sort_reserved;
            crate::runtime::reserve_to(&mut self.sorted_instances, n);
            crate::runtime::reserve_to(&mut self.sorted_uploaded, n);
            crate::runtime::reserve_to(&mut self.sort_keys, n);
        }
        self.enabled = sim.settings.enabled;
        self.plan_views(ctx, &sim.settings);
        let (bytes, created) = if ctx.frame.device_lost {
            (0, 0)
        } else {
            let (b, c) = self.upload(ctx);
            let (sort_created, sort_bytes) = self.sort_on_gpu(ctx);
            (b + sort_bytes, c + textures + sort_created)
        };
        let after = self.capacities();
        let node_reallocs = before.iter().zip(&after).filter(|(a, b)| a != b).count() as u32;
        let draw_calls = self.view_items.len() as u32;
        let backend = self.backend;
        let resets = self.sim.resets + self.gpu_sim.as_ref().map_or(0, |g| g.resets);
        if let Some(runtime) = ctx.resources.get_mut::<ParticleRuntime>() {
            let rt = &mut runtime.rt;
            let (effects, layer_instances) = rt.counts();
            let stats = ParticleStats {
                backend,
                effects,
                layer_instances,
                pool_capacity: rt.pool_capacity(),
                pool_reserved: rt.pool.reserved(),
                pool_allocated: match &self.gpu_sim {
                    Some(g) if backend == Some(ParticleBackend::Gpu) => g.pool_slots(),
                    _ => self.sim.pool.len() as u32,
                },
                pool_high_water: rt.pool.high_water(),
                alive: sim.alive,
                alive_frame: sim.alive_frame,
                spawned: rt.spawned,
                refused: rt.refused,
                shrunk: rt.shrunk,
                ticks_simulated: sim.ticks,
                catchup_resets: resets,
                dispatches: sim.dispatches + self.gpu_sort.as_ref().map_or(0, |g| g.recorded),
                draw_calls,
                ribbon_segments: self.segments,
                bytes_uploaded: bytes + sim.bytes,
                cpu_reallocations: sim.reallocs + node_reallocs + created,
            };
            rt.refused = 0;
            rt.shrunk = 0;
            if let Some(s) = ctx.resources.get_mut::<ParticleStats>() {
                *s = stats;
            }
        }
    }

    fn prepare_view(&mut self, ctx: &mut ViewPrepareContext<'_>) -> SlotMask {
        if !self.enabled || self.gpu.as_ref().is_none_or(|g| g.bind_group.is_none()) {
            return SlotMask::NONE;
        }
        let Some(l) = self.views.iter().find(|l| l.camera == ctx.view.camera) else {
            return SlotMask::NONE;
        };
        let mut mask = SlotMask::NONE;
        if l.end > l.start {
            mask = mask.with(SLOT_COLOR);
        }
        // Only the settings camera of a post group with distortion draws the Distortion slot
        // (the frame ignores the bit elsewhere).
        if l.dist_end > l.end {
            mask = mask.with(SLOT_DISTORTION);
        }
        mask
    }

    fn prewarm(&mut self, device: &wgpu::Device, targets: &[(usize, PassTarget)]) {
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };
        for &(slot, target) in targets {
            let distortion = match slot {
                SLOT_COLOR => false,
                SLOT_DISTORTION => true,
                _ => continue,
            };
            // Color: untextured sprites always; strips and the textured variants the layers
            // use. Distortion: the variants of the distortion layers.
            if !distortion {
                gpu.pipeline(device, (target, DrawKind::Sprite, TexVariant::None));
            }
            for info in &self.layers.info {
                if info.distortion != distortion {
                    continue;
                }
                let tex = match info.texture {
                    None => TexVariant::None,
                    Some(k) => match gpu.tex_bind(k) {
                        Some(b) if b.layered => TexVariant::Array,
                        Some(_) => TexVariant::D2,
                        None => continue,
                    },
                };
                let kind = if distortion {
                    DrawKind::Distortion
                } else {
                    draw_kind(info.kind)
                };
                gpu.pipeline(device, (target, kind, tex));
            }
        }
    }

    fn draw(&mut self, pass: &mut wgpu::RenderPass<'static>, ctx: &DrawContext<'_>) {
        let distortion = match ctx.slot {
            SLOT_COLOR => false,
            SLOT_DISTORTION => true,
            _ => return,
        };
        let Some(list) = self.views.iter().find(|l| l.camera == ctx.view.camera).copied() else {
            return;
        };
        #[cfg(test)]
        if !distortion {
            self.test_capture = Some((
                ctx.view_bind_group.clone(),
                ctx.scene_bind_group.clone(),
                ctx.view.view_offset,
                ctx.target,
            ));
        }
        self.record(
            pass,
            ctx.device,
            ctx.target,
            (ctx.view_bind_group, ctx.scene_bind_group, ctx.view.view_offset),
            list,
            distortion,
        );
    }
}

impl ParticlesNode {
    /// Record the color (or `distortion`) draws of `list` with the frame's view and scene bind
    /// groups.
    fn record(
        &mut self,
        pass: &mut wgpu::RenderPass<'static>,
        device: &wgpu::Device,
        target: PassTarget,
        (view_bind_group, scene_bind_group, view_offset): (&wgpu::BindGroup, &wgpu::BindGroup, u32),
        list: ViewList,
        distortion: bool,
    ) {
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };
        let range = list.range(distortion);
        if range.is_empty() || gpu.bind_group.is_none() {
            return;
        }
        let indirect = self.gpu_sim.as_ref().and_then(GpuSim::draw_buffers);
        pass.set_bind_group(0, view_bind_group, &[view_offset]);
        pass.set_bind_group(1, scene_bind_group, &[]);
        let view_uniform = (list.uniform as u64 * gpu.view_stride) as u32;
        let mut current: Option<usize> = None;
        let mut bound_tex: Option<TexKey> = None;
        for &i in &self.view_items[range] {
            let item = self.items[i as usize];
            let info = self.layers.info[item.layer as usize];
            let kind = if distortion {
                DrawKind::Distortion
            } else {
                draw_kind(info.kind)
            };
            // The texture's bind group (group 3) and variant; a texture not ready is skipped.
            let tex = match info.texture {
                None => TexVariant::None,
                Some(k) => match gpu.tex_bind(k) {
                    Some(b) if b.layered => TexVariant::Array,
                    Some(_) => TexVariant::D2,
                    None => continue,
                },
            };
            let p = gpu.pipeline(device, (target, kind, tex));
            if current != Some(p) {
                pass.set_pipeline(&gpu.pipelines[p].1);
                current = Some(p);
                bound_tex = None;
            }
            if let Some(k) = info.texture
                && bound_tex != Some(k)
                && let Some(b) = gpu.tex_bind(k)
            {
                pass.set_bind_group(3, &b.group, &[]);
                bound_tex = Some(k);
            }
            let Some(bind_group) = &gpu.bind_group else {
                return;
            };
            pass.set_bind_group(
                2,
                bind_group,
                &[view_uniform, (item.layer as u64 * gpu.draw_stride) as u32],
            );
            match kind {
                DrawKind::Strip => {
                    // Segment k: point k from slot 0, point k + 1 from slot 1.
                    let Some((strips, _)) = &gpu.strips else { continue };
                    let from = item.first as u64 * POINT_SIZE;
                    let len = item.count as u64 * POINT_SIZE;
                    pass.set_vertex_buffer(0, strips.slice(from..from + len));
                    pass.set_vertex_buffer(1, strips.slice(from + POINT_SIZE..from + POINT_SIZE + len));
                    pass.draw(0..4, 0..item.count);
                }
                DrawKind::Sprite | DrawKind::Distortion => {
                    // `ByDistance` layers draw their instances sorted for this view.
                    let sorted = if info.sorted && !distortion {
                        self.sorted_draws[list.sorted.0 as usize..list.sorted.1 as usize]
                            .iter()
                            .find(|d| d.item == i)
                            .copied()
                    } else {
                        None
                    };
                    let bytes = item.count as u64 * INSTANCE_SIZE;
                    if let Some(d) = sorted {
                        let from = d.first as u64 * INSTANCE_SIZE;
                        if d.gpu {
                            let (Some(buffer), Some((work, _)), Some(sim)) = (
                                self.gpu_sort.as_ref().and_then(GpuSort::sorted_buffer),
                                indirect,
                                self.gpu_sim.as_ref(),
                            ) else {
                                continue;
                            };
                            pass.set_vertex_buffer(0, buffer.slice(from..from + bytes));
                            pass.draw_indirect(work, sim.args_offset(item.args));
                        } else {
                            let Some((buffer, _)) = &gpu.sorted else { continue };
                            pass.set_vertex_buffer(0, buffer.slice(from..from + bytes));
                            pass.draw(0..4, 0..item.count);
                        }
                        continue;
                    }
                    let from = item.first as u64 * INSTANCE_SIZE;
                    let to = from + item.count as u64 * INSTANCE_SIZE;
                    if item.args == DIRECT {
                        let Some((instances, _)) = &gpu.instances else { continue };
                        pass.set_vertex_buffer(0, instances.slice(from..to));
                        pass.draw(0..4, 0..item.count);
                    } else {
                        let (Some((work, instances)), Some(sim)) = (indirect, self.gpu_sim.as_ref()) else {
                            continue;
                        };
                        pass.set_vertex_buffer(0, instances.slice(from..to));
                        pass.draw_indirect(work, sim.args_offset(item.args));
                    }
                }
            }
        }
    }

    /// The GPU backend's state (tests).
    pub(crate) fn gpu_sim(&mut self) -> Option<&mut GpuSim> {
        self.gpu_sim.as_mut()
    }

    /// Record the first view's draws again, into a fresh `size` target of the last color draw's
    /// pass signature (depth cleared to far), inside a render pass writing timestamps `index`
    /// and `index + 1` (the perf tests time the particle draws alone). Returns the draw calls.
    #[cfg(test)]
    pub(crate) fn record_timed_draws(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        queries: &wgpu::QuerySet,
        index: u32,
        size: (u32, u32),
    ) -> Option<u32> {
        let (view_bg, scene_bg, view_offset, target) = self.test_capture.clone()?;
        let list = *self.views.first()?;
        let texture = |format, usage| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("space3d particles timed draws"),
                    size: wgpu::Extent3d {
                        width: size.0,
                        height: size.1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: target.samples,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let color = texture(target.color, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let depth = target.depth.map(|f| texture(f, wgpu::TextureUsages::RENDER_ATTACHMENT));
        let mut pass = encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("particles.draw (perf test)"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: depth.as_ref().map(|view| wgpu::RenderPassDepthStencilAttachment {
                    view,
                    depth_ops: (!target.depth_read_only).then_some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: Some(wgpu::RenderPassTimestampWrites {
                    query_set: queries,
                    beginning_of_pass_write_index: Some(index),
                    end_of_pass_write_index: Some(index + 1),
                }),
                occlusion_query_set: None,
                multiview_mask: None,
            })
            .forget_lifetime();
        self.record(
            &mut pass,
            device,
            target,
            (&view_bg, &scene_bg, view_offset),
            list,
            false,
        );
        Some(list.end - list.start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use naga::back::glsl;

    /// The sprite and strip shaders validate for every depth access, output encoding and
    /// texture variant, and translate to GLSL ES 3.00 (WebGL2) for both stages, the way wgpu's GL
    /// backend does.
    #[test]
    fn shaders_validate_and_translate_to_webgl2() {
        let options = glsl::Options {
            version: glsl::Version::Embedded {
                version: 300,
                is_webgl: true,
            },
            writer_flags: glsl::WriterFlags::ADJUST_COORDINATE_SPACE | glsl::WriterFlags::FORCE_POINT_SIZE,
            binding_map: Default::default(),
            zero_initialize_workgroup_memory: true,
        };
        for access in [
            DepthAccess::None,
            DepthAccess::ReadOnlyAttached,
            DepthAccess::SampledOnly,
        ] {
            for (output, kind, tex) in [
                (OutputEncoding::Linear, DrawKind::Sprite, TexVariant::None),
                (OutputEncoding::ManualSrgb, DrawKind::Sprite, TexVariant::D2),
                (OutputEncoding::Tonemapped, DrawKind::Sprite, TexVariant::Array),
                (OutputEncoding::Linear, DrawKind::Strip, TexVariant::Array),
                (OutputEncoding::ManualSrgb, DrawKind::Strip, TexVariant::None),
                (OutputEncoding::Tonemapped, DrawKind::Strip, TexVariant::D2),
                (OutputEncoding::Linear, DrawKind::Distortion, TexVariant::None),
                (OutputEncoding::Linear, DrawKind::Distortion, TexVariant::D2),
                (OutputEncoding::Linear, DrawKind::Distortion, TexVariant::Array),
            ] {
                let src = shader_for(output, access, kind, tex);
                let module =
                    naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&src)));
                let info = naga::valid::Validator::new(
                    naga::valid::ValidationFlags::all(),
                    naga::valid::Capabilities::empty(),
                )
                .validate(&module)
                .unwrap_or_else(|e| panic!("{access:?} {output:?} {kind:?} {tex:?}: {e:?}"));
                let entries = match kind {
                    DrawKind::Sprite => ["vs_sprite", "fs_sprite"],
                    DrawKind::Strip => ["vs_strip", "fs_strip"],
                    DrawKind::Distortion => ["vs_distortion", "fs_distortion"],
                };
                for (stage, entry) in [
                    (naga::ShaderStage::Vertex, entries[0]),
                    (naga::ShaderStage::Fragment, entries[1]),
                ] {
                    let pipeline = glsl::PipelineOptions {
                        shader_stage: stage,
                        entry_point: entry.to_string(),
                        multiview: None,
                    };
                    let mut out = String::new();
                    glsl::Writer::new(&mut out, &module, &info, &options, &pipeline, Default::default())
                        .and_then(|mut w| w.write().map(drop))
                        .unwrap_or_else(|e| panic!("{access:?} {output:?} {stage:?}: {e}"));
                    // No storage buffers in any stage (WebGL2), and the vertex shader fetches
                    // curves through `textureLod`.
                    assert!(!out.contains("buffer "), "{stage:?} uses a storage buffer");
                    if stage == naga::ShaderStage::Vertex {
                        assert!(out.contains("textureLod"), "LUT fetch");
                    }
                }
            }
        }
    }

    /// The instance layout matches the vertex attributes: 40 bytes, WebGL2-safe formats.
    #[test]
    fn instance_layout() {
        assert_eq!(INSTANCE_SIZE, 40);
        let offsets: Vec<u64> = INSTANCE_ATTRIBUTES.iter().map(|a| a.offset).collect();
        assert_eq!(offsets, [0, 16, 24, 32]);
        assert_eq!(std::mem::size_of::<GpuParticleView>(), 32);
        assert_eq!(std::mem::size_of::<GpuDrawParams>(), 112);
        assert_eq!(POINT_SIZE, 40);
        let offsets: Vec<u64> = STRIP_B_ATTRIBUTES.iter().map(|a| a.offset).collect();
        assert_eq!(offsets, [0, 16, 24, 32]);
        assert!(SLOTS_SOFT[0].reads.contains(SceneReads::DEPTH));
        assert!(!SLOTS_HARD[0].reads.contains(SceneReads::DEPTH));
        assert_eq!(SLOTS_SOFT[1].phase, RenderPhase::Distortion);
    }

    /// Per-layer draw parameters and per-view flags sit at dynamic offsets one stride apart:
    /// their size rounded up to the device's uniform offset alignment. Chrome's WebGL2 on
    /// ANGLE / Metal reports 32 bytes, less than the 112-byte draw parameters: with the
    /// alignment as the stride the slots overlapped (every layer but the first drew with its
    /// neighbours' parameters) and the upload overran the buffer (a validation error on the
    /// graphics showcase's first frame: 9344 bytes into 4096).
    #[test]
    fn uniform_strides_hold_a_whole_slot() {
        for align in [16u64, 32, 64, 128, 256] {
            for size in [
                std::mem::size_of::<GpuDrawParams>(),
                std::mem::size_of::<GpuParticleView>(),
            ] {
                let stride = uniform_stride(size, align);
                assert!(
                    stride >= size as u64 && stride.is_multiple_of(align),
                    "{size} B at {align}: {stride}"
                );
                assert!(stride < size as u64 + align, "tight: {size} B at {align}: {stride}");
            }
        }
        assert_eq!(uniform_stride(std::mem::size_of::<GpuDrawParams>(), 32), 128);
        assert_eq!(uniform_stride(std::mem::size_of::<GpuDrawParams>(), 256), 256);
    }
}
