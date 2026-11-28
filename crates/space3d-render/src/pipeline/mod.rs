//! Shader generation, GPU uniform layouts and the pipeline cache.

use crate::handles::{FastMap, ShaderHandle};
use crate::instance::InstanceLayout;
use crate::output::OutputEncoding;
use crate::shader::CustomShader;
use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use std::sync::Arc;

mod depth;
mod forward;
mod instance;
mod view;

pub(crate) use forward::{GpuMaterial, lighting_globals};
pub(crate) use instance::ext_fits;
pub(crate) use view::{GpuDirLight, GpuPointLight, GpuView, MAX_DIR_LIGHTS, MAX_POINT_LIGHTS, VIEW_STRUCTS};

/// Depth buffer format used by every pass.
pub(crate) const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Format of the offscreen/capture target and render-target textures.
pub(crate) const OFFSCREEN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// Views whose frustums one culling dispatch can read: the `frusta` uniform of cull.wgsl
/// (`CULL_VIEWS * 112` bytes = 7 KiB, a multiple of every allowed uniform offset alignment).
pub(crate) const CULL_VIEWS: usize = 64;

/// Frustum of one view (`frusta` in cull.wgsl: 7 × `vec4<f32>` per view): the six planes, then
/// `Frustum::small` (a point and the ratio under which a sphere is too small to draw).
pub(crate) type GpuFrustum = [[f32; 4]; 7];

/// Matches `CullParams` in cull.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
pub(crate) struct GpuCullParams {
    pub job_start: u32,
    pub job_count: u32,
    pub total_workgroups: u32,
    pub workgroups_x: u32,
    /// First word of the per-workgroup compaction records in the args buffer.
    pub wg_word: u32,
    pub pad: [u32; 3],
}

/// Matches `Job` in cull.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
pub(crate) struct GpuCullJob {
    pub sphere: [f32; 4],
    pub src_word: u32,
    pub count: u32,
    pub stride_words: u32,
    pub kind: u32,
    pub t_off: u32,
    pub s_off: u32,
    pub s_kind: u32,
    pub dst_word: u32,
    pub draw: u32,
    pub wg_start: u32,
    /// Frustum of the job's view in the dispatch's `frusta` window (`< CULL_VIEWS`).
    pub view: u32,
    pub pad: u32,
}

/// Matches `CompactParams` in compact.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
pub(crate) struct GpuCompactParams {
    pub run_start: u32,
    pub run_count: u32,
    pub workgroups_x: u32,
    pub pad: u32,
}

/// Pipeline "class" bits that come from the material.
pub(crate) const CLASS_DOUBLE_SIDED: u8 = 1;
pub(crate) const CLASS_BLEND: u8 = 2;

/// Kind of pass a mesh pipeline draws in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) enum PassKind {
    /// Color pass (the forward shader).
    #[default]
    Forward,
    /// Depth-only pass (shadow maps), vertex stage only.
    Depth,
    /// Depth-only pass with alpha-masked discards.
    DepthMask,
    /// The SSR G-buffer-lite pass (`fs_gbuffer`): normal, roughness and F0 of opaque surfaces
    /// into an `Rgba8Unorm` target, depth-tested (no depth writes) over the scene depth.
    GBuffer,
}

/// `PipelineKey::kinds` bit: the color maps (base color, emissive) are multi-layer arrays
/// (`texture_2d_array`; see [`Pipelines::material_bgl`]).
pub(crate) const KIND_COLOR_LAYERED: u8 = 1;
/// `PipelineKey::kinds` bit: the data maps (normal, metallic-roughness, occlusion) are arrays.
pub(crate) const KIND_DATA_LAYERED: u8 = 2;

/// Everything that selects a forward (or depth) pipeline. Frozen after graphics milestone F3
/// (spec §3.5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PipelineKey {
    pub layout: u16,
    /// Material class bits: [`CLASS_DOUBLE_SIDED`], [`CLASS_BLEND`]; the materials area defines
    /// bits 2..=5.
    pub class: u8,
    pub format: wgpu::TextureFormat,
    pub samples: u32,
    /// How the fragment encodes its color: the `S3D_OUTPUT` (and `MANUAL_SRGB`) overrides.
    pub output: OutputEncoding,
    /// Texture kinds: [`KIND_COLOR_LAYERED`], [`KIND_DATA_LAYERED`].
    pub kinds: u8,
    /// Custom shader and its source version (see [`Pipelines::sync_custom`]).
    pub shader: Option<(ShaderHandle, u32)>,
    pub pass: PassKind,
    /// Frame-wide override set (shading model, normal maps, shadow taps); 0 = every override
    /// at its default.
    pub globals: u8,
}

impl PipelineKey {
    /// The shader module this pipeline is built from (overrides are not part of it).
    pub fn module_key(&self) -> ModuleKey {
        // The extension vertex stream (tangents, second UV set): a class bit of the materials
        // area.
        let ext = forward::class_ext(self.class);
        (self.layout, self.kinds, ext, self.pass, self.shader)
    }

    /// Pipeline-overridable constants of this key: only those differing from their defaults
    /// (output encoding, alpha mask, frame-wide `globals`; see `forward::pipeline_constants`).
    pub fn constants(&self) -> ([(&'static str, f64); 8], usize) {
        forward::pipeline_constants(self)
    }
}

/// The forward overrides selecting an output encoding (`S3D_OUTPUT`, and `MANUAL_SRGB` for
/// custom shaders), those differing from their defaults: `(constants, count)`.
pub(crate) fn output_constants(output: OutputEncoding) -> ([(&'static str, f64); 2], usize) {
    let mut c = [("", 0.0); 2];
    let mut n = 0;
    if output != OutputEncoding::Linear {
        c[n] = ("S3D_OUTPUT", output.id() as f64);
        n += 1;
    }
    if output == OutputEncoding::ManualSrgb {
        c[n] = ("MANUAL_SRGB", 1.0);
        n += 1;
    }
    (c, n)
}

/// Shader module cache key: (layout, texture kinds, extension vertex stream, pass kind,
/// custom shader). The output encoding and the other overrides are specialized per pipeline.
pub(crate) type ModuleKey = (u16, u8, bool, PassKind, Option<(ShaderHandle, u32)>);

const CULL_SHADER: &str = include_str!("../shaders/cull.wgsl");
const COMPACT_SHADER: &str = include_str!("../shaders/compact.wgsl");
const BLIT_TEMPLATE: &str = include_str!("../shaders/blit.wgsl");
const CLEAR_SHADER: &str = include_str!("../shaders/clear.wgsl");
const DEPTH_RESOLVE_SHADER: &str = include_str!("../shaders/frame/depth_resolve.wgsl");

/// Source of the depth resolve pass (`shaders/frame/depth_resolve.wgsl` after the full-screen
/// triangle).
pub(crate) fn depth_resolve_source() -> String {
    format!("{}\n{DEPTH_RESOLVE_SHADER}", crate::node::FULLSCREEN_WGSL)
}

/// The depth resolve (graphics spec §1.6): layout (one multisampled depth texture), module and
/// a pipeline per sample count. Created on first use: only read-only-depth backends with
/// multisampled groups resolve depth (GLES cannot sample multisampled textures).
pub(crate) struct DepthResolve {
    pub bgl: wgpu::BindGroupLayout,
    layout: wgpu::PipelineLayout,
    module: wgpu::ShaderModule,
    pipelines: FastMap<u32, wgpu::RenderPipeline>,
}

/// Mip-generation blit shader sampling a `texture_2d` or a `texture_2d_array` source.
pub(crate) fn blit_shader(layered: bool) -> String {
    let source = if layered {
        "@group(0) @binding(0) var src_tex: texture_2d_array<f32>;\n\
         @group(0) @binding(1) var src_sampler: sampler;\n\n\
         @fragment\n\
         fn fs_main(in: VsOut) -> @location(0) vec4<f32> {\n    \
         return textureSample(src_tex, src_sampler, in.uv, in.layer);\n}\n"
    } else {
        "@group(0) @binding(0) var src_tex: texture_2d<f32>;\n\
         @group(0) @binding(1) var src_sampler: sampler;\n\n\
         @fragment\n\
         fn fs_main(in: VsOut) -> @location(0) vec4<f32> {\n    \
         return textureSample(src_tex, src_sampler, in.uv);\n}\n"
    };
    BLIT_TEMPLATE.replace("//#SOURCE", source)
}

/// Parse and validate WGSL with naga (the portable subset: no optional capabilities), with a
/// readable error message.
#[cfg(test)]
pub(crate) fn validate_wgsl(src: &str) -> Result<(), String> {
    parse_and_validate(src).map(drop)
}

pub(super) fn parse_and_validate(src: &str) -> Result<(naga::Module, naga::valid::ModuleInfo), String> {
    let module = naga::front::wgsl::parse_str(src).map_err(|e| e.emit_to_string(src))?;
    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty())
        .validate(&module)
        .map_err(|e| e.emit_to_string(src))?;
    Ok((module, info))
}

pub(super) fn uniform_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    dynamic: bool,
    size: u64,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: dynamic,
            min_binding_size: wgpu::BufferSize::new(size),
        },
        count: None,
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Compute pipelines for GPU culling (only built when compute is available). The three cull
/// passes share `bgl`; see `shaders/cull.wgsl`.
pub(crate) struct CullPipelines {
    pub bgl: wgpu::BindGroupLayout,
    pub mark: wgpu::ComputePipeline,
    pub scan: wgpu::ComputePipeline,
    pub scatter: wgpu::ComputePipeline,
    pub compact_bgl: wgpu::BindGroupLayout,
    pub compact: wgpu::ComputePipeline,
}

/// A custom-shader forward pipeline in use while its validation result is still unknown.
struct PendingPipeline {
    key: PipelineKey,
    module_key: ModuleKey,
    /// Error message, if the pipeline (or its module) turned out to be invalid.
    error: space3d_tasks::Task<Option<String>>,
}

/// Mip-generation blit for one source view dimension (`[single-layer, array]`).
struct Blit {
    bgl: wgpu::BindGroupLayout,
    module: wgpu::ShaderModule,
    layout: wgpu::PipelineLayout,
}

/// Bind group layouts, shaders and cached pipelines.
pub(crate) struct Pipelines {
    pub view_bgl: wgpu::BindGroupLayout,
    /// Material layouts by texture kinds (`PipelineKey::kinds`: color maps and data maps
    /// `texture_2d` or `texture_2d_array`).
    material_bgls: [wgpu::BindGroupLayout; 4],
    layouts_by_texture: [wgpu::PipelineLayout; 4],
    shaders: FastMap<ModuleKey, wgpu::ShaderModule>,
    forward: FastMap<PipelineKey, wgpu::RenderPipeline>,
    /// Sources of the custom shaders drawn so far: (version, source).
    custom: FastMap<ShaderHandle, (u32, Arc<CustomShader>)>,
    /// First build error of a custom shader version.
    custom_errors: FastMap<ShaderHandle, (u32, String)>,
    /// First portability warning of a custom shader version (it builds here, but not on
    /// GLSL ES 3.00 devices).
    custom_warnings: FastMap<ShaderHandle, (u32, String)>,
    /// The device compiles shaders as GLSL ES 3.00 (WebGL2, GLES 3.0): custom shaders using
    /// what that lacks are rejected instead of warned about.
    glsl_es_300: bool,
    /// Custom-shader pipelines whose error scope has not answered yet (browser WebGPU; native
    /// backends answer immediately). See [`Self::poll_pending`].
    pending: Vec<PendingPipeline>,
    /// Registered instance layouts; index = layout id.
    pub layouts: Vec<&'static InstanceLayout>,
    pub cull: Option<CullPipelines>,
    blits: [Blit; 2],
    blit: FastMap<(wgpu::TextureFormat, bool), wgpu::RenderPipeline>,
    pub blit_sampler: wgpu::Sampler,
    /// Viewport clears (see [`Self::viewport_clear`]): shader, bind-group-less layout and the
    /// pipelines per (target format, sample count).
    clear_module: wgpu::ShaderModule,
    clear_layout: wgpu::PipelineLayout,
    clear: FastMap<(wgpu::TextureFormat, u32), wgpu::RenderPipeline>,
    /// The depth resolve, once a frame needed it.
    depth_resolve: Option<DepthResolve>,
    /// Layout of `PassKind::Depth` pipelines: group 0 (the view) only.
    depth_layout: wgpu::PipelineLayout,
    /// Depth-only pipelines of depth views (see [`Self::depth`]): by key and depth bias
    /// (constant, slope scale bits, clamp bits).
    depth: FastMap<(PipelineKey, [u32; 3]), wgpu::RenderPipeline>,
    /// The device clamps depth bias (`DownlevelFlags::DEPTH_BIAS_CLAMP`); without it a
    /// requested clamp is dropped (a non-zero clamp would fail validation).
    pub depth_bias_clamp: bool,
}

/// Group 0 of forward, depth and node pipelines (graphics spec §3.4): binding 0 the `View`
/// uniform (dynamic offset, vertex and fragment); bindings 1–6 the lighting inputs (fragment):
/// environment specular cube, irradiance cube, BRDF LUT, environment sampler, shadow map
/// (`texture_depth_2d`) and its comparison sampler. Shaders declare only what they use.
fn view_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let fragment = wgpu::ShaderStages::FRAGMENT;
    let texture = |binding, sample_type, view_dimension| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: fragment,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension,
            multisampled: false,
        },
        count: None,
    };
    let sampler = |binding, ty| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: fragment,
        ty: wgpu::BindingType::Sampler(ty),
        count: None,
    };
    let float = wgpu::TextureSampleType::Float { filterable: true };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("space3d view"),
        entries: &[
            uniform_entry(
                0,
                wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                true,
                std::mem::size_of::<GpuView>() as u64,
            ),
            texture(1, float, wgpu::TextureViewDimension::Cube),
            texture(2, float, wgpu::TextureViewDimension::Cube),
            texture(3, float, wgpu::TextureViewDimension::D2),
            sampler(4, wgpu::SamplerBindingType::Filtering),
            texture(5, wgpu::TextureSampleType::Depth, wgpu::TextureViewDimension::D2),
            sampler(6, wgpu::SamplerBindingType::Comparison),
        ],
    })
}

/// Bind group layout entry of a filterable float texture of the given view dimension.
pub(super) fn texture_entry(binding: u32, view_dimension: wgpu::TextureViewDimension) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension,
            multisampled: false,
        },
        count: None,
    }
}

/// View dimension of a sampled 2D texture: single-layer textures are `D2`, arrays `D2Array`.
/// GL binds a texture with a target chosen from its layer count alone, so the view (and the
/// shader's texture type) must match that count.
pub(crate) fn sampled_view_dimension(layered: bool) -> wgpu::TextureViewDimension {
    if layered {
        wgpu::TextureViewDimension::D2Array
    } else {
        wgpu::TextureViewDimension::D2
    }
}

impl Pipelines {
    /// Layouts and the built-in modules; culling pipelines when `compute`. `glsl_es_300`: the
    /// device compiles shaders as GLSL ES 3.00 (WebGL2, GLES 3.0; see
    /// [`forward::validate_custom_wgsl`]).
    pub fn new(device: &wgpu::Device, compute: bool, glsl_es_300: bool) -> Self {
        let view_bgl = view_layout(device);
        let material_bgls = forward::material_bgls(device);
        let layouts_by_texture = [0, 1, 2, 3].map(|i| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("space3d forward"),
                bind_group_layouts: &[Some(&view_bgl), Some(&material_bgls[i])],
                immediate_size: 0,
            })
        });
        let depth_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("space3d depth"),
            bind_group_layouts: &[Some(&view_bgl)],
            immediate_size: 0,
        });

        let cull = compute.then(|| {
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("space3d cull"),
                entries: &[
                    uniform_entry(
                        0,
                        wgpu::ShaderStages::COMPUTE,
                        true,
                        std::mem::size_of::<GpuCullParams>() as u64,
                    ),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, false),
                    storage_entry(4, false),
                    // Frustums of up to CULL_VIEWS views (a uniform: storage bindings stay at
                    // four, the minimum some GLES devices offer). A static offset: one more
                    // dynamic offset per bind costs wgpu an allocation per frame.
                    uniform_entry(
                        5,
                        wgpu::ShaderStages::COMPUTE,
                        false,
                        (CULL_VIEWS * std::mem::size_of::<GpuFrustum>()) as u64,
                    ),
                ],
            });
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("space3d cull"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(CULL_SHADER)),
            });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("space3d cull"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
            let entry = |name: &'static str| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(name),
                    layout: Some(&pl),
                    module: &module,
                    entry_point: Some(name),
                    compilation_options: Default::default(),
                    cache: None,
                })
            };
            let (mark, scan, scatter) = (entry("cull_mark"), entry("cull_scan"), entry("cull_scatter"));
            let compact_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("space3d compact"),
                entries: &[
                    uniform_entry(
                        0,
                        wgpu::ShaderStages::COMPUTE,
                        true,
                        std::mem::size_of::<GpuCompactParams>() as u64,
                    ),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, false),
                    storage_entry(4, false),
                ],
            });
            let cmodule = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("space3d compact"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(COMPACT_SHADER)),
            });
            let cpl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("space3d compact"),
                bind_group_layouts: &[Some(&compact_bgl)],
                immediate_size: 0,
            });
            let compact = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("space3d compact"),
                layout: Some(&cpl),
                module: &cmodule,
                entry_point: Some("compact"),
                compilation_options: Default::default(),
                cache: None,
            });
            CullPipelines {
                bgl,
                mark,
                scan,
                scatter,
                compact_bgl,
                compact,
            }
        });

        let blits = [false, true].map(|layered| {
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("space3d blit"),
                entries: &[
                    texture_entry(0, sampled_view_dimension(layered)),
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("space3d blit"),
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(blit_shader(layered))),
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("space3d blit"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
            Blit { bgl, module, layout }
        });
        let blit_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("space3d blit"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let clear_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("space3d viewport clear"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(CLEAR_SHADER)),
        });
        let clear_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("space3d viewport clear"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });
        Self {
            view_bgl,
            material_bgls,
            layouts_by_texture,
            shaders: FastMap::default(),
            forward: FastMap::default(),
            custom: FastMap::default(),
            custom_errors: FastMap::default(),
            custom_warnings: FastMap::default(),
            glsl_es_300,
            pending: Vec::new(),
            layouts: vec![<crate::instance::InstanceTransform as crate::instance::InstanceData>::LAYOUT],
            cull,
            blits,
            blit: FastMap::default(),
            blit_sampler,
            clear_module,
            clear_layout,
            clear: FastMap::default(),
            depth_resolve: None,
            depth_layout,
            depth: FastMap::default(),
            depth_bias_clamp: false,
        }
    }

    /// Material bind group layout for texture kinds (`PipelineKey::kinds`).
    pub fn material_bgl(&self, kinds: u8) -> &wgpu::BindGroupLayout {
        &self.material_bgls[(kinds & (KIND_COLOR_LAYERED | KIND_DATA_LAYERED)) as usize]
    }

    /// Stable id for an instance layout (registers it on first use).
    pub fn layout_id(&mut self, layout: &'static InstanceLayout) -> u16 {
        if let Some(i) = self
            .layouts
            .iter()
            .position(|l| std::ptr::eq(*l, layout) || **l == *layout)
        {
            return i as u16;
        }
        if layout.cull_source() == crate::instance::CullSource::None {
            tracing::warn!(
                layout = layout.name,
                "instance positions are not in a float format the culler decodes (f32/f16): \
                 these instances are never culled on the GPU, and on the CPU only per block if \
                 `InstanceData::bounds` is overridden or `InstanceBuffer::set_bounds` is used"
            );
        }
        self.layouts.push(layout);
        (self.layouts.len() - 1) as u16
    }

    /// Make `source` (version `version`) the current code of custom shader `handle`. A new
    /// version drops the pipelines, modules and error of the old one.
    pub fn sync_custom(&mut self, handle: ShaderHandle, version: u32, source: &Arc<CustomShader>) {
        match self.custom.get(&handle) {
            Some((v, _)) if *v == version => return,
            Some(_) => {
                self.forward.retain(|k, _| k.shader.is_none_or(|(h, _)| h != handle));
                self.shaders.retain(|k, _| k.4.is_none_or(|(h, _)| h != handle));
                self.custom_errors.remove(&handle);
                self.pending.retain(|p| p.key.shader.is_none_or(|(h, _)| h != handle));
            }
            None => {}
        }
        self.custom.insert(handle, (version, source.clone()));
    }

    /// The source version of a custom shader drawn (or prewarmed) before.
    pub fn custom_version(&self, handle: ShaderHandle) -> Option<u32> {
        self.custom.get(&handle).map(|(v, _)| *v)
    }

    /// Forget custom shaders for which `live` is false (removed from `RenderAssets`): their
    /// sources, errors, modules and pipelines.
    pub fn retain_custom(&mut self, live: impl Fn(ShaderHandle) -> bool) {
        if self.custom.keys().all(|&h| live(h)) {
            return;
        }
        self.custom.retain(|&h, _| live(h));
        self.custom_errors.retain(|&h, _| live(h));
        self.custom_warnings.retain(|&h, _| live(h));
        self.pending.retain(|p| p.key.shader.is_none_or(|(h, _)| live(h)));
        self.forward.retain(|k, _| k.shader.is_none_or(|(h, _)| live(h)));
        self.shaders.retain(|k, _| k.4.is_none_or(|(h, _)| live(h)));
    }

    /// Custom shaders with cached state, and forward pipelines built for custom shaders.
    #[cfg(test)]
    pub fn custom_counts(&self) -> (usize, usize) {
        (
            self.custom.len(),
            self.forward.keys().filter(|k| k.shader.is_some()).count(),
        )
    }

    /// Error of the current version of a custom shader, if it failed to build.
    pub fn custom_error(&self, handle: ShaderHandle) -> Option<&str> {
        let (version, e) = self.custom_errors.get(&handle)?;
        (self.custom.get(&handle).map(|(v, _)| *v) == Some(*version)).then_some(e.as_str())
    }

    /// Portability warning of the current version of a custom shader: it builds on this device
    /// but not on WebGL2 / GLES 3.0 (see [`forward::validate_custom_wgsl`]).
    pub fn custom_warning(&self, handle: ShaderHandle) -> Option<&str> {
        let (version, w) = self.custom_warnings.get(&handle)?;
        (self.custom.get(&handle).map(|(v, _)| *v) == Some(*version)).then_some(w.as_str())
    }

    /// Record (and log once) a portability warning of the custom shader of `key`.
    fn custom_warned(&mut self, key: PipelineKey, warning: String) {
        let Some((handle, version)) = key.shader else { return };
        if self.custom_warnings.get(&handle).is_some_and(|(v, _)| *v == version) {
            return;
        }
        let name = self.custom.get(&handle).map_or("?", |(_, s)| s.name.as_str());
        tracing::warn!(shader = name, "{warning}");
        self.custom_warnings.insert(handle, (version, warning));
    }

    /// Get or build the forward pipeline for `key`. A custom shader that fails to build (or is
    /// unknown) is reported once and replaced by the built-in pipeline for that key.
    pub fn forward(&mut self, device: &wgpu::Device, key: PipelineKey) -> &wgpu::RenderPipeline {
        if !self.forward.contains_key(&key) {
            let pipeline = match key.shader {
                None => self.build_forward(device, key, None).expect("built-in shader builds").0,
                Some((handle, version)) => {
                    let source = self
                        .custom
                        .get(&handle)
                        .filter(|(v, _)| *v == version)
                        .map(|(_, s)| s.clone());
                    let built = match &source {
                        Some(s) => self.build_forward(device, key, Some(s)),
                        None => Err("shader not registered".to_string()),
                    };
                    match built {
                        Ok((p, pending)) => {
                            if let Some((module_key, error)) = pending {
                                self.pending.push(PendingPipeline { key, module_key, error });
                            }
                            p
                        }
                        Err(e) => self.custom_failed(device, key, e),
                    }
                }
            };
            self.forward.insert(key, pipeline);
        }
        &self.forward[&key]
    }

    /// Report a custom-shader pipeline that failed (`key.shader` is set) and return the
    /// built-in pipeline drawn in its place.
    fn custom_failed(&mut self, device: &wgpu::Device, key: PipelineKey, error: String) -> wgpu::RenderPipeline {
        let layout = self.layouts[key.layout as usize].name;
        if let Some((handle, version)) = key.shader {
            let name = self.custom.get(&handle).map_or("?", |(_, s)| s.name.as_str());
            tracing::error!(
                shader = name,
                layout,
                "custom shader failed to build; drawing with the built-in shader:\n{error}"
            );
            self.custom_errors.entry(handle).or_insert((version, error));
        }
        self.forward(device, PipelineKey { shader: None, ..key }).clone()
    }

    /// Resolve custom-shader pipelines whose error scope answered since the last frame (only
    /// browser WebGPU answers asynchronously). An invalid one is reported like a synchronous
    /// failure and replaced by the built-in pipeline, so from then on its draws no longer
    /// invalidate the frame. Called once per frame; nothing to do on native backends.
    pub fn poll_pending(&mut self, device: &wgpu::Device) {
        let mut i = 0;
        while i < self.pending.len() {
            let p = &mut self.pending[i];
            let result = match p.error.poll() {
                Some(r) => r,
                None if p.error.state() == space3d_tasks::TaskState::Cancelled => None,
                None => {
                    i += 1;
                    continue;
                }
            };
            let PendingPipeline { key, module_key, .. } = self.pending.swap_remove(i);
            let current = key
                .shader
                .is_some_and(|(h, v)| self.custom.get(&h).is_some_and(|(cv, _)| *cv == v));
            // Stale versions were dropped by `sync_custom` / `retain_custom` already.
            if let Some(e) = result
                && current
            {
                self.shaders.remove(&module_key);
                let fallback = self.custom_failed(device, key, e);
                self.forward.insert(key, fallback);
            }
        }
    }

    /// Resolve `key`'s pending validation with `error` (tests: native backends never pend).
    #[cfg(test)]
    pub fn defer_validation(&mut self, key: PipelineKey, error: Option<String>) {
        let module_key = key.module_key();
        let error = space3d_tasks::spawn(async move { error });
        self.pending.push(PendingPipeline { key, module_key, error });
    }

    /// Keys of the cached forward pipelines of custom shaders (tests).
    #[cfg(test)]
    pub fn custom_keys(&self) -> Vec<PipelineKey> {
        self.forward.keys().filter(|k| k.shader.is_some()).copied().collect()
    }

    /// Validation of pipelines still pending (tests).
    #[cfg(test)]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Mip-generation pipeline for a color format, sampling a single-layer (`D2`) or an array
    /// (`D2Array`, layer chosen by the draw's vertex range) source view.
    pub fn blit(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        layered: bool,
    ) -> (&wgpu::RenderPipeline, &wgpu::BindGroupLayout) {
        let blit = &self.blits[layered as usize];
        let (module, layout) = (&blit.module, &blit.layout);
        let p = self.blit.entry((format, layered)).or_insert_with(|| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("space3d mipgen"),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(format.into())],
                }),
                multiview_mask: None,
                cache: None,
            })
        });
        (p, &blit.bgl)
    }

    /// Pipeline clearing a view's viewport inside a forward pass (see `shaders/clear.wgsl`):
    /// draw 3 vertices with the clear color as the pass's blend constant. It matches the
    /// forward pass — target format, sample count and the depth attachment, which it neither
    /// tests nor writes. Every target a view renders into is blendable (alpha-blended
    /// materials need that too), so the color needs no buffer or bind group: nothing to update
    /// per frame, whatever the colors. (HDR colors above 1 need blend constants that are not
    /// clamped for float targets: tested on Metal and ANGLE.)
    pub fn viewport_clear(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        samples: u32,
    ) -> &wgpu::RenderPipeline {
        let (module, layout) = (&self.clear_module, &self.clear_layout);
        self.clear.entry((format, samples)).or_insert_with(|| {
            let constant = wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Constant,
                dst_factor: wgpu::BlendFactor::Zero,
                operation: wgpu::BlendOperation::Add,
            };
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("space3d viewport clear"),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: Default::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: samples,
                    ..Default::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState {
                            color: constant,
                            alpha: constant,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        })
    }

    /// The depth resolve pipeline for `samples` and its bind group layout (group 0: the
    /// multisampled depth texture). Draw 3 vertices into a `DEPTH_FORMAT` attachment.
    pub fn depth_resolve(
        &mut self,
        device: &wgpu::Device,
        samples: u32,
    ) -> (&wgpu::RenderPipeline, &wgpu::BindGroupLayout) {
        let r = self.depth_resolve.get_or_insert_with(|| {
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("space3d depth resolve"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: true,
                    },
                    count: None,
                }],
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("space3d depth resolve"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("space3d depth resolve"),
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(depth_resolve_source())),
            });
            DepthResolve {
                bgl,
                layout,
                module,
                pipelines: FastMap::default(),
            }
        });
        let (module, layout) = (&r.module, &r.layout);
        let p = r.pipelines.entry(samples).or_insert_with(|| {
            let constants = [("S3D_SAMPLES", samples as f64)];
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("space3d depth resolve"),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some("vs_fullscreen"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: Default::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some("fs_resolve"),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &constants,
                        ..Default::default()
                    },
                    targets: &[],
                }),
                multiview_mask: None,
                cache: None,
            })
        });
        (p, &r.bgl)
    }
}

#[cfg(test)]
mod tests {
    use super::forward::*;
    use super::instance::*;
    use super::*;
    use crate::instance::*;
    use wgpu::VertexFormat as F;

    fn validate(src: &str) {
        validate_wgsl(src).unwrap_or_else(|e| panic!("{e}\n{src}"));
    }

    /// Atlas + texture layer + custom data: every semantic the built-in layouts leave out.
    const ATLAS: &InstanceLayout = &InstanceLayout {
        name: "Atlas",
        stride: 64,
        attributes: &[
            InstanceAttribute::new(InstanceSemantic::Translation, F::Float32x3, 0),
            InstanceAttribute::new(InstanceSemantic::UniformScale, F::Float32, 12),
            InstanceAttribute::new(InstanceSemantic::UvRect, F::Float32x4, 16),
            InstanceAttribute::new(InstanceSemantic::Custom(0), F::Float32x4, 32),
            InstanceAttribute::new(InstanceSemantic::TextureLayer, F::Uint32, 48),
            InstanceAttribute::new(InstanceSemantic::Custom(7), F::Uint32x2, 52),
            InstanceAttribute::new(InstanceSemantic::Scale, F::Float16x2, 60),
        ],
    };

    /// f16 position with a separate f16 uniform scale in the upper half of a word.
    const HALF_SPLIT: &InstanceLayout = &InstanceLayout {
        name: "HalfSplit",
        stride: 16,
        attributes: &[
            InstanceAttribute::new(InstanceSemantic::Translation, F::Float16x4, 0),
            InstanceAttribute::new(InstanceSemantic::UniformScale, F::Float16, 6),
            InstanceAttribute::new(InstanceSemantic::Rotation, F::Snorm16x4, 8),
        ],
    };

    #[test]
    fn generated_shaders_validate() {
        let custom = CustomShader::new(
            "tint",
            "fn vertex_hook(v: VertexInput, inst: InstanceInput, out: VsOut) -> VsOut {\n\
             \x20   var o = out;\n    o.custom = instance_custom0(inst) + instance_custom3(inst);\n    return o;\n}\n\
             fn fragment_hook(in: VsOut, color: vec4<f32>) -> vec4<f32> {\n    return in.custom * color;\n}\n",
        );
        let only_fragment = CustomShader::new(
            "f",
            "fn fragment_hook(in: VsOut, c: vec4<f32>) -> vec4<f32> { return c.bgra; }",
        );
        for l in [
            InstanceTransform::LAYOUT,
            InstanceTransformColor::LAYOUT,
            InstanceCompact::LAYOUT,
            InstancePacked::LAYOUT,
            InstanceHalf::LAYOUT,
            ATLAS,
            HALF_SPLIT,
        ] {
            l.validate().unwrap();
            for kinds in 0..4 {
                for c in [None, Some(&custom), Some(&only_fragment)] {
                    validate(&forward_shader(l, kinds, false, c));
                }
            }
        }
        validate(CULL_SHADER);
        validate(COMPACT_SHADER);
        validate(&blit_shader(false));
        validate(&blit_shader(true));
        validate(CLEAR_SHADER);
        // A broken custom shader is rejected with a message, not a panic.
        let broken = CustomShader::new(
            "broken",
            "fn fragment_hook(in: VsOut) -> f32 { return undefined_thing; }",
        );
        let e = validate_wgsl(&forward_shader(ATLAS, 0, false, Some(&broken))).unwrap_err();
        assert!(e.contains("undefined_thing") || e.contains("fragment_hook"), "{e}");
    }

    /// Regression: naga accepts a custom shader that declares its own resources, but the
    /// forward pipeline layout lacks them, so the pipeline is invalid. Native backends report
    /// that through the error scope at once; browser WebGPU only later, and meanwhile every
    /// frame drawing with it was lost. Such shaders are now rejected before any GPU work.
    #[test]
    fn custom_shaders_cannot_declare_resources() {
        let hook = "fn fragment_hook(in: VsOut, c: vec4<f32>) -> vec4<f32> { return c * extra; }";
        for decl in [
            "@group(2) @binding(0) var<uniform> extra: vec4<f32>;",
            "@group(1) @binding(11) var<uniform> extra: vec4<f32>;",
            "@group(0) @binding(1) var<storage, read> extra_buf: array<vec4<f32>>;
             fn extra_at() -> vec4<f32> { return extra_buf[0]; }
             const extra = vec4<f32>(1.0);",
        ] {
            let custom = CustomShader::new(
                "own",
                format!(
                    "{decl}
{hook}"
                ),
            );
            let src = forward_shader(InstanceTransform::LAYOUT, 0, false, Some(&custom));
            validate(&src); // naga alone accepts it
            for glsl_es_300 in [false, true] {
                let e = validate_custom_wgsl(&src, glsl_es_300).unwrap_err();
                assert!(e.contains("extra") && e.contains("cannot declare resources"), "{e}");
            }
        }
        // Private variables and the template's own resources are fine.
        let ok = CustomShader::new(
            "ok",
            "var<private> k: vec4<f32> = vec4<f32>(0.5);
             fn fragment_hook(in: VsOut, c: vec4<f32>) -> vec4<f32> {
                 return c * k * material.base_color * sample_base(in.uv, 0) + view.ambient;
}
",
        );
        for layered in [false, true] {
            let src = forward_shader(InstanceTransform::LAYOUT, u8::from(layered), false, Some(&ok));
            for glsl_es_300 in [false, true] {
                assert_eq!(validate_custom_wgsl(&src, glsl_es_300), Ok(None));
            }
        }
    }

    /// Regression: `var<workgroup>` passed validation. WGSL allows workgroup memory in compute
    /// shaders only, but naga accepts it in any stage, and the hooks run in the vertex and
    /// fragment stages, where backends disagree: Metal ran such a shader, wgpu's GL backend
    /// failed to translate it ("doesn't support COMPUTE_SHADER" on ANGLE / GLES 3.0), browsers
    /// reject it (asynchronously: frames drawn with it are lost until then). It is now
    /// rejected up front on every backend, used or not. So are pipeline constants (`override`)
    /// without a default: the forward pipeline sets none, so creating it would fail.
    #[test]
    fn custom_shaders_cannot_use_workgroup_memory_or_unset_overrides() {
        let fragment = |body: &str| format!("fn fragment_hook(in: VsOut, c: vec4<f32>) -> vec4<f32> {{ {body} }}");
        for (wgsl, expected) in [
            (
                format!(
                    "var<workgroup> tile: array<vec4<f32>, 4>;\n{}",
                    fragment("tile[0] = c; return tile[0];")
                ),
                "`var<workgroup> tile`",
            ),
            (
                "var<workgroup> clip: vec4<f32>;\n\
                 fn vertex_hook(v: VertexInput, inst: InstanceInput, out: VsOut) -> VsOut {\n\
                 \x20   var o = out;\n    clip = o.clip;\n    o.clip = clip * 0.5;\n    return o;\n}\n"
                    .to_string(),
                "`var<workgroup> clip`",
            ),
            (
                format!("var<workgroup> unused: u32;\n{}", fragment("return c;")),
                "`var<workgroup> unused`",
            ),
            (
                format!("override strength: f32;\n{}", fragment("return c * strength;")),
                "`override strength`",
            ),
        ] {
            let custom = CustomShader::new("portability", wgsl);
            for layered in [false, true] {
                let src = forward_shader(InstanceTransform::LAYOUT, u8::from(layered), false, Some(&custom));
                validate(&src); // naga alone accepts it
                for glsl_es_300 in [false, true] {
                    let e = validate_custom_wgsl(&src, glsl_es_300).unwrap_err();
                    assert!(e.contains(expected), "{e}");
                }
            }
        }
        // Private variables and overrides with a default are portable.
        let ok = CustomShader::new(
            "ok",
            format!(
                "override strength: f32 = 0.5;\nvar<private> k: vec4<f32>;\n{}",
                fragment("k = c; return k * strength;")
            ),
        );
        let src = forward_shader(InstanceTransform::LAYOUT, 0, false, Some(&ok));
        for glsl_es_300 in [false, true] {
            assert_eq!(validate_custom_wgsl(&src, glsl_es_300), Ok(None));
        }
    }

    /// Custom shaders calling WGSL built-ins GLSL ES 3.00 lacks (or anything naga cannot
    /// translate to it) do not compile on WebGL2 / GLES 3.0 devices: there they are rejected up
    /// front — used or not — with a message naming the construct. Everywhere else they run as
    /// they did before the graphics upgrade (rejecting them there changed the pixels of shaders
    /// that worked), and the same finding comes back as a portability warning. The built-ins naga polyfills for
    /// GLSL ES 3.00 raise neither.
    #[test]
    fn custom_shaders_using_what_webgl2_lacks_are_rejected_there_and_warned_elsewhere() {
        let fragment = |body: &str| format!("fn fragment_hook(in: VsOut, c: vec4<f32>) -> vec4<f32> {{ {body} }}");
        let cases = [
            (
                fragment("return c * f32(countOneBits(u32(c.r * 255.0)));"),
                "`countOneBits`",
            ),
            (fragment("return c * f32(reverseBits(u32(c.g)));"), "`reverseBits`"),
            (
                fragment("return c * f32(extractBits(u32(c.g), 1u, 2u));"),
                "`extractBits`",
            ),
            (
                fragment("return c * f32(insertBits(u32(c.g), 1u, 1u, 2u));"),
                "`insertBits`",
            ),
            (
                fragment("return c * f32(firstLeadingBit(u32(c.g)));"),
                "`firstLeadingBit`",
            ),
            (
                fragment("return c * f32(firstTrailingBit(u32(c.g)));"),
                "`firstTrailingBit`",
            ),
            (
                fragment("return c * f32(countTrailingZeros(u32(c.g)));"),
                "`countTrailingZeros`",
            ),
            (fragment("return c * ldexp(c.r, i32(c.g));"), "`ldexp`"),
            (fragment("return c * vec4<f32>(unpack4xU8(u32(c.g)));"), "`unpack4xU8`"),
            (
                fragment("return c * textureGather(0, base_tex, base_sampler, in.uv);"),
                "`textureGather`",
            ),
            (
                format!(
                    "fn my_unused(x: u32) -> u32 {{ return countOneBits(x); }}\n{}",
                    fragment("return c;")
                ),
                "`countOneBits`",
            ),
            (fragment("return c * f32(textureNumLevels(base_tex));"), "GLSL ES 3.00"),
            (fragment("return unpack4x8unorm(pack4x8unorm(c));"), "GLSL ES 3.00"),
        ];
        for (wgsl, expected) in cases {
            let custom = CustomShader::new("gles3", wgsl);
            let src = forward_shader(InstanceTransform::LAYOUT, 0, false, Some(&custom));
            validate(&src); // naga alone accepts it
            let e = validate_custom_wgsl(&src, true).expect_err(expected);
            assert!(e.contains(expected) && e.contains("GLES 3.0"), "{expected}: {e}");
            let w = validate_custom_wgsl(&src, false)
                .unwrap_or_else(|e| panic!("{expected}: rejected off GLSL ES 3.00: {e}"))
                .unwrap_or_else(|| panic!("{expected}: no warning"));
            assert!(
                w.contains(expected) && w.contains("WebGL2") && w.contains("builds on this device"),
                "{expected}: {w}"
            );
        }
        // Calls on constants are folded to literals by naga: nothing left for GLSL to lack.
        let folded = CustomShader::new(
            "folded",
            fragment("return c * f32(reverseBits(1u) + countOneBits(7u));"),
        );
        let src = forward_shader(InstanceTransform::LAYOUT, 0, false, Some(&folded));
        for glsl_es_300 in [false, true] {
            assert_eq!(validate_custom_wgsl(&src, glsl_es_300), Ok(None));
        }
        // Built-ins naga polyfills for GLSL ES 3.00 (or that GLSL ES 3.00 has) are fine.
        let portable = CustomShader::new(
            "portable",
            fragment(
                "let f = frexp(c.r); let m = modf(c.g);\n\
                 let bits = f32(countLeadingZeros(4u)) + unpack4x8unorm(7u).x + unpack4x8snorm(9u).y;\n\
                 let h = unpack2x16unorm(pack2x16unorm(c.xy)) + unpack2x16snorm(pack2x16snorm(c.zw));\n\
                 let d = dpdxFine(in.uv.x) + fwidthCoarse(in.uv.y) + dpdy(in.uv.x);\n\
                 return vec4<f32>(fma(c.r, 2.0, 1.0), f.fract + m.fract, bits + d, h.x) + \
                 textureSampleLevel(base_tex, base_sampler, in.uv, 0.0) * f32(textureDimensions(base_tex).x);",
            ),
        );
        for layered in [false, true] {
            let custom = if layered {
                CustomShader::new("portable", portable.wgsl.replace("in.uv, 0.0)", "in.uv, 0, 0.0)"))
            } else {
                portable.clone()
            };
            let src = forward_shader(InstanceTransform::LAYOUT, u8::from(layered), false, Some(&custom));
            for glsl_es_300 in [false, true] {
                assert_eq!(validate_custom_wgsl(&src, glsl_es_300), Ok(None), "layered {layered}");
            }
        }
        // The built-in shader itself, for every layout and texture kind.
        for l in [InstanceTransform::LAYOUT, InstanceHalf::LAYOUT, ATLAS] {
            for layered in [false, true] {
                let (module, info) = parse_and_validate(&forward_shader(l, u8::from(layered), false, None)).unwrap();
                assert_eq!(gles3_translation_error(&module, &info), None, "{}", l.name);
                assert_eq!(gles3_missing_builtin(&module), None);
            }
        }
    }

    /// The generated accessors read the attributes the layout declares.
    #[test]
    fn generated_accessors_use_the_declared_attributes() {
        let src = instance_wgsl(ATLAS);
        for line in [
            "@location(4) a0: vec3<f32>",
            "@location(5) a1: f32",
            "@location(6) a2: vec4<f32>",
            "@location(7) custom0: vec4<f32>",
            "@location(8) a4: u32",
            "@location(9) custom7: vec2<u32>",
            "@location(10) a6: vec2<f32>",
            "return inst.a2;",
            "return i32(inst.a4);",
            "fn instance_custom0(inst: InstanceInput) -> vec4<f32> {\n    return inst.custom0;",
            "return vec4<f32>(vec2<f32>(inst.custom7), 0.0, 0.0);",
            "fn instance_custom3(inst: InstanceInput) -> vec4<f32> {\n    return vec4<f32>(0.0);",
            "trs(inst.a0, mat3x3<f32>",
            "vec3<f32>(inst.a6, 0.0))",
        ] {
            assert!(src.contains(line), "missing `{line}` in\n{src}");
        }
        let half = instance_wgsl(InstanceHalf::LAYOUT);
        assert!(
            half.contains("return trs(inst.a0.xyz, quat_to_mat3(inst.a1), vec3<f32>(inst.a0.w));"),
            "{half}"
        );
    }

    #[test]
    fn uniform_sizes_match_wgsl() {
        assert_eq!(
            std::mem::size_of::<GpuView>(),
            64 + 16 * 3 + 32 * (MAX_DIR_LIGHTS + MAX_POINT_LIGHTS) + 560 + 64
        );
        assert_eq!(std::mem::size_of::<GpuView>(), 1392);
        assert_eq!(std::mem::size_of::<GpuMaterial>(), 112);
        assert_eq!(std::mem::size_of::<GpuCullParams>(), 32);
        assert_eq!(std::mem::size_of::<GpuFrustum>(), 112);
        // Dynamic offsets of the `frusta` window must be aligned (256 at most in WebGPU).
        assert!((CULL_VIEWS * std::mem::size_of::<GpuFrustum>()).is_multiple_of(256));
        assert_eq!(std::mem::size_of::<GpuCompactParams>(), 16);
        assert_eq!(std::mem::size_of::<GpuCullJob>(), 64);
    }

    /// Every member of the forward template's `View` sits where `GpuView` writes it (graphics
    /// spec §3.3): the template is parsed with naga and each member's offset compared with the
    /// Rust field's, by name.
    #[test]
    fn gpu_view_matches_wgsl() {
        use std::mem::offset_of;
        let src = forward_shader(InstanceTransform::LAYOUT, 0, false, None);
        let module = naga::front::wgsl::parse_str(&src).unwrap();
        let (_, view) = module
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some("View"))
            .expect("View struct");
        let naga::TypeInner::Struct { members, span } = &view.inner else {
            panic!("View is a struct")
        };
        assert_eq!(*span as usize, std::mem::size_of::<GpuView>());
        let rust = [
            ("view_proj", offset_of!(GpuView, view_proj)),
            ("camera_pos", offset_of!(GpuView, camera_pos)),
            ("ambient", offset_of!(GpuView, ambient)),
            ("counts", offset_of!(GpuView, counts)),
            ("dir_lights", offset_of!(GpuView, dir_lights)),
            ("point_lights", offset_of!(GpuView, point_lights)),
            ("view", offset_of!(GpuView, view)),
            ("proj", offset_of!(GpuView, proj)),
            ("inv_view_proj", offset_of!(GpuView, inv_view_proj)),
            ("inv_proj", offset_of!(GpuView, inv_proj)),
            ("viewport", offset_of!(GpuView, viewport)),
            ("target_size", offset_of!(GpuView, target_size)),
            ("depth", offset_of!(GpuView, depth)),
            ("output", offset_of!(GpuView, output)),
            ("exposure", offset_of!(GpuView, exposure)),
            ("shadow_view_proj", offset_of!(GpuView, shadow_view_proj)),
            ("shadow_params", offset_of!(GpuView, shadow_params)),
            ("shadow_info", offset_of!(GpuView, shadow_info)),
            ("env_params", offset_of!(GpuView, env_params)),
            ("env_rotation", offset_of!(GpuView, env_rotation)),
            ("sky_rotation", offset_of!(GpuView, sky_rotation)),
            ("sky_params", offset_of!(GpuView, sky_params)),
            ("inv_proj_rot", offset_of!(GpuView, inv_proj_rot)),
            ("sky_ground", offset_of!(GpuView, sky_ground)),
        ];
        assert_eq!(members.len(), rust.len(), "every WGSL member has a Rust field");
        for (member, (name, offset)) in members.iter().zip(rust) {
            assert_eq!(member.name.as_deref(), Some(name), "member order");
            assert_eq!(member.offset as usize, offset, "offset of `{name}`");
        }
        // The lights too.
        for (name, fields) in [
            (
                "DirLight",
                [offset_of!(GpuDirLight, direction), offset_of!(GpuDirLight, color)],
            ),
            (
                "PointLight",
                [
                    offset_of!(GpuPointLight, position_range),
                    offset_of!(GpuPointLight, color),
                ],
            ),
        ] {
            let (_, t) = module
                .types
                .iter()
                .find(|(_, t)| t.name.as_deref() == Some(name))
                .unwrap();
            let naga::TypeInner::Struct { members, span } = &t.inner else {
                panic!()
            };
            assert_eq!(*span, 32);
            let offsets: Vec<usize> = members.iter().map(|m| m.offset as usize).collect();
            assert_eq!(offsets, fields, "{name}");
        }
        // `VIEW_WGSL` (render nodes, overlays) declares the same struct.
        assert!(crate::VIEW_WGSL.starts_with(VIEW_STRUCTS));
    }

    /// Custom shaders the tests use (their fixtures), plus ones reading the pipeline overrides
    /// and the view fields added for the graphics upgrade.
    const PERMUTATION_FIXTURES: [&str; 6] = [
        // tests/custom_shader.rs `TINT`
        "fn vertex_hook(v: VertexInput, inst: InstanceInput, out: VsOut) -> VsOut {
            var o = out;
            o.custom = instance_custom0(inst);
            o.color = vec4<f32>(o.color.rgb + instance_custom3(inst).rgb, o.color.a);
            return o;
        }
        fn fragment_hook(in: VsOut, color: vec4<f32>) -> vec4<f32> {
            return vec4<f32>(color.rgb * in.custom.rgb, color.a);
        }",
        "fn fragment_hook(in: VsOut, c: vec4<f32>) -> vec4<f32> { return c.bgra; }",
        "var<private> k: vec4<f32> = vec4<f32>(0.5);
         fn fragment_hook(in: VsOut, c: vec4<f32>) -> vec4<f32> {
             return c * k * material.base_color * sample_base(in.uv, in.layer) + view.ambient;
         }",
        "override strength: f32 = 0.5;
         fn vertex_hook(v: VertexInput, inst: InstanceInput, out: VsOut) -> VsOut {
             var o = out;
             o.clip = view.proj * view.view * vec4<f32>(o.world_pos, 1.0);
             return o;
         }
         fn fragment_hook(in: VsOut, c: vec4<f32>) -> vec4<f32> { return c * strength; }",
        // The legacy `MANUAL_SRGB` flag (an override now) and the output helpers.
        "fn fragment_hook(in: VsOut, c: vec4<f32>) -> vec4<f32> {
             if (MANUAL_SRGB || S3D_OUTPUT == S3D_OUTPUT_TONEMAPPED) { return c * 0.5; }
             return vec4<f32>(s3d_tonemap(view.output.y, c.rgb * view.exposure.x), c.a);
         }",
        "fn my_luma(c: vec3<f32>) -> f32 { return s3d_luma(c) + f32(view.counts.y); }
         fn fragment_hook(in: VsOut, c: vec4<f32>) -> vec4<f32> {
             let uv = (in.clip.xy - view.viewport.xy) * view.target_size.zw;
             return vec4<f32>(c.rgb * my_luma(c.rgb) + vec3<f32>(uv, view.depth.x), c.a);
         }",
    ];

    /// Every custom-shader fixture builds against every forward permutation — instance
    /// layouts, texture kinds and every value of the output override — on the device (inside
    /// the custom-shader error scopes) and through the GLSL ES 3.00 translation (graphics spec
    /// §3.2).
    #[test]
    fn custom_shaders_compile_against_every_permutation() {
        let Ok(mut r) = pollster::block_on(crate::Renderer::new_headless(4, 4)) else {
            eprintln!("skipping GPU test: no adapter available");
            return;
        };
        let mut assets = crate::RenderAssets::default();
        let fixtures: Vec<(ShaderHandle, Arc<CustomShader>)> = PERMUTATION_FIXTURES
            .iter()
            .enumerate()
            .map(|(i, wgsl)| {
                let shader = CustomShader::new(format!("fixture {i}"), *wgsl);
                (assets.add_shader(shader.clone()), Arc::new(shader))
            })
            .collect();
        let layouts: Vec<u16> = [
            InstanceTransform::LAYOUT,
            InstanceTransformColor::LAYOUT,
            InstanceCompact::LAYOUT,
            InstancePacked::LAYOUT,
            InstanceHalf::LAYOUT,
            ATLAS,
            HALF_SPLIT,
        ]
        .into_iter()
        .map(|l| r.pipelines.layout_id(l))
        .collect();
        let mut built = 0;
        for (handle, shader) in &fixtures {
            r.pipelines.sync_custom(*handle, 0, shader);
            for &layout in &layouts {
                for kinds in [0, KIND_COLOR_LAYERED] {
                    let (module, info) = parse_and_validate(&forward_shader(
                        r.pipelines.layouts[layout as usize],
                        kinds,
                        false,
                        Some(shader),
                    ))
                    .unwrap();
                    assert_eq!(
                        forward::gles3_translation_error(&module, &info),
                        None,
                        "{}",
                        shader.name
                    );
                    for output in [
                        OutputEncoding::Linear,
                        OutputEncoding::ManualSrgb,
                        OutputEncoding::Tonemapped,
                    ] {
                        for (format, samples) in [
                            (wgpu::TextureFormat::Rgba8UnormSrgb, 1),
                            (wgpu::TextureFormat::Rgba16Float, 4),
                        ] {
                            let key = PipelineKey {
                                layout,
                                class: CLASS_BLEND,
                                format,
                                samples,
                                output,
                                kinds,
                                shader: Some((*handle, 0)),
                                pass: PassKind::Forward,
                                globals: 0,
                            };
                            r.pipelines.forward(&r.device, key);
                            assert_eq!(r.pipelines.custom_error(*handle), None, "{} × {key:?}", shader.name);
                            built += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(built, fixtures.len() * layouts.len() * 2 * 3 * 2);
        assert_eq!(r.gpu_error_count(), 0);
    }

    /// Regression: `Unorm8x4Bgra` reached wgpu's GL backend, which cannot express it (WebGL2 /
    /// GLES: INVALID_VALUE and garbage colors; ANGLE: a crash). It is now fetched as `Unorm8x4`
    /// and swizzled in the generated shader — for `Custom(n)` attributes too (their
    /// `instance_customN` accessor used to return the fetched BGRA bytes with red and blue
    /// swapped).
    #[test]
    fn bgra_instance_colors_are_emulated() {
        const LAYOUT: &InstanceLayout = &InstanceLayout {
            name: "Bgra",
            stride: 20,
            attributes: &[
                InstanceAttribute::new(InstanceSemantic::Translation, F::Float32x3, 0),
                InstanceAttribute::new(InstanceSemantic::Color, F::Unorm8x4Bgra, 12),
                InstanceAttribute::new(InstanceSemantic::Custom(1), F::Unorm8x4Bgra, 16),
            ],
        };
        LAYOUT.validate().expect("valid layout");
        let attrs = instance_attributes(LAYOUT);
        assert!(attrs.iter().all(|a| a.format != F::Unorm8x4Bgra));
        assert_eq!(attrs[1].format, F::Unorm8x4);
        assert_eq!(attrs[2].format, F::Unorm8x4);
        let src = instance_wgsl(LAYOUT);
        assert!(src.contains("return inst.a1.zyxw;"), "{src}");
        assert!(src.contains("@location(6) custom1: vec4<f32>"), "{src}");
        assert!(
            src.contains("fn instance_custom1(inst: InstanceInput) -> vec4<f32> {\n    return inst.custom1.zyxw;"),
            "{src}"
        );
        let custom = CustomShader::new(
            "c1",
            "fn fragment_hook(in: VsOut, c: vec4<f32>) -> vec4<f32> { return c; }\n\
             fn vertex_hook(v: VertexInput, inst: InstanceInput, out: VsOut) -> VsOut {\n\
             \x20   var o = out;\n    o.custom = instance_custom1(inst) + inst.custom1;\n    return o;\n}\n",
        );
        validate(&forward_shader(LAYOUT, 0, false, None));
        validate(&forward_shader(LAYOUT, 0, false, Some(&custom)));
    }

    /// Exact `(format, offset, location)` of every built-in layout: a wrong offset or format
    /// (e.g. a tint read from the matrix) would still validate and render something.
    #[test]
    fn instance_attributes_match_the_structs() {
        let tuples = |l: &InstanceLayout| {
            instance_attributes(l)
                .iter()
                .map(|a| (a.format, a.offset, a.shader_location))
                .collect::<Vec<_>>()
        };
        let matrix = |extra: &[(F, u64, u32)]| {
            let mut v: Vec<_> = (0..4).map(|c| (F::Float32x4, c * 16, 4 + c as u32)).collect();
            v.extend_from_slice(extra);
            v
        };
        assert_eq!(tuples(InstanceTransform::LAYOUT), matrix(&[]));
        assert_eq!(tuples(InstanceTransformColor::LAYOUT), matrix(&[(F::Float32x4, 64, 8)]));
        assert_eq!(
            tuples(InstanceCompact::LAYOUT),
            [(F::Float32x3, 0, 4), (F::Float32, 12, 5), (F::Float32x4, 16, 6)]
        );
        assert_eq!(
            tuples(InstancePacked::LAYOUT),
            [
                (F::Float32x3, 0, 4),
                (F::Float32, 12, 5),
                (F::Snorm16x4, 16, 6),
                (F::Unorm8x4, 24, 7),
                (F::Uint32, 28, 8)
            ]
        );
        assert_eq!(
            tuples(InstanceHalf::LAYOUT),
            [(F::Float16x4, 0, 4), (F::Snorm16x4, 8, 5)]
        );
        // Custom attributes get locations too (custom shaders read them).
        assert_eq!(tuples(ATLAS)[3], (F::Float32x4, 32, 7));
        assert_eq!(tuples(ATLAS).len(), 7);
        // Byte offsets inside the Rust structs.
        assert_eq!(std::mem::offset_of!(InstanceTransformColor, color), 64);
        assert_eq!(std::mem::offset_of!(InstanceCompact, rotation), 16);
        assert_eq!(std::mem::offset_of!(InstancePacked, rotation), 16);
        assert_eq!(std::mem::offset_of!(InstancePacked, color), 24);
        assert_eq!(std::mem::offset_of!(InstancePacked, layer), 28);
        assert_eq!(std::mem::offset_of!(InstanceHalf, rotation), 8);
    }
}
