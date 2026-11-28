//! Forward shader composition (`forward.wgsl` + instance layout + texture kind + custom
//! hooks), the material uniform and bind group layouts, forward pipelines, and custom-shader
//! validation.

use super::instance::{instance_attributes, instance_locations, instance_wgsl};
use super::{
    CLASS_BLEND, CLASS_DOUBLE_SIDED, DEPTH_FORMAT, KIND_COLOR_LAYERED, KIND_DATA_LAYERED, ModuleKey, PipelineKey,
    Pipelines, VIEW_STRUCTS, output_constants, parse_and_validate, sampled_view_dimension, texture_entry,
    uniform_entry,
};
use crate::arena::{EXT_VERTEX_SIZE, VERTEX_ATTRIBUTES, VERTEX_SIZE, ext_attributes};
use crate::instance::InstanceLayout;
use crate::material::{BlendMode, CLASS_EXT};
use crate::output::{OUTPUT_WGSL, OutputEncoding};
use crate::shader::{CustomShader, hooks_wgsl};
use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use std::sync::Arc;

/// Matches `MaterialParams` in forward.wgsl. The first three members are the legacy layout
/// (custom shaders read them).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub(crate) struct GpuMaterial {
    pub base_color: [f32; 4],
    /// rgb: emissive factor; w: unlit flag.
    pub emissive: [f32; 4],
    /// x: metallic, y: roughness, z: has a base color texture.
    pub params: [f32; 4],
    /// x: normal scale, y: occlusion strength, z: alpha cutoff.
    pub maps: [f32; 4],
    /// x: `FLAG_*` bits, y: maps sampled with the second UV set (`MapMask` bits).
    pub flags: [u32; 4],
    /// Rows of the UV transform (`TextureTransform::matrix`).
    pub uv_row0: [f32; 4],
    pub uv_row1: [f32; 4],
}

pub(super) const FORWARD_TEMPLATE: &str = include_str!("../shaders/forward.wgsl");

/// Material texture bindings of a texture-kind combination and their sampling functions:
/// `sample_base` (legacy name) and `s3d_sample_{normal,mr,occlusion,emissive}`. The color maps
/// (base color, emissive) are arrays with `color_layered`, the data maps (normal,
/// metallic-roughness, occlusion) with `data_layered`; every map of an array kind samples the
/// per-instance layer.
pub(super) fn texture_wgsl(color_layered: bool, data_layered: bool) -> String {
    let decl = |binding: u32, name: &str, layered: bool| {
        let ty = if layered {
            "texture_2d_array<f32>"
        } else {
            "texture_2d<f32>"
        };
        format!("@group(1) @binding({binding}) var {name}: {ty};\n")
    };
    // Single-layer textures: the instance layer is irrelevant (an array lookup would clamp it to
    // layer 0 as well).
    let sample = |f: &str, tex: &str, sampler: &str, layered: bool| {
        let args = if layered { "uv, layer" } else { "uv" };
        format!(
            "fn {f}(uv: vec2<f32>, layer: i32) -> vec4<f32> {{\n    \
             return textureSample({tex}, {sampler}, {args});\n}}\n"
        )
    };
    let mut s = decl(1, "base_tex", color_layered);
    for (binding, name, layered) in [
        (3, "s3d_normal", data_layered),
        (5, "s3d_mr", data_layered),
        (7, "s3d_occlusion", data_layered),
        (9, "s3d_emissive", color_layered),
    ] {
        s += &decl(binding, &format!("{name}_tex"), layered);
        s += &format!("@group(1) @binding({}) var {name}_sampler: sampler;\n", binding + 1);
    }
    s += "\n";
    s += &sample("sample_base", "base_tex", "base_sampler", color_layered);
    for (f, name, layered) in [
        ("s3d_sample_normal", "s3d_normal", data_layered),
        ("s3d_sample_mr", "s3d_mr", data_layered),
        ("s3d_sample_occlusion", "s3d_occlusion", data_layered),
        ("s3d_sample_emissive", "s3d_emissive", color_layered),
    ] {
        s += &sample(f, &format!("{name}_tex"), &format!("{name}_sampler"), layered);
    }
    s
}

/// First shader location of the extension stream after `layout`'s instance attributes.
pub(crate) fn ext_location(layout: &InstanceLayout) -> u32 {
    4 + instance_locations(layout)
}

/// The extension-stream parts of the forward shader: `VertexInput` fields, `VsOut` fields and
/// the accessors (`s3d_vertex_ext`, `s3d_frag_tangent`, `s3d_frag_uv1`, and for custom shaders
/// `s3d_vertex_tangent` / `s3d_vertex_uv1`). Without the stream the accessors return the "no
/// extension data" values.
fn ext_wgsl(layout: &InstanceLayout, ext: bool) -> (String, &'static str, &'static str) {
    if !ext {
        let accessors = "fn s3d_vertex_tangent(v: VertexInput) -> vec4<f32> {\n    return vec4<f32>(0.0);\n}\n\n\
             fn s3d_vertex_uv1(v: VertexInput) -> vec2<f32> {\n    return v.uv;\n}\n\n\
             fn s3d_vertex_ext(v: VertexInput, out: VsOut, model: mat3x3<f32>, mirrored: bool) -> VsOut {\n    \
             return out;\n}\n\n\
             fn s3d_frag_tangent(in: VsOut) -> vec4<f32> {\n    return vec4<f32>(0.0);\n}\n\n\
             fn s3d_frag_uv1(in: VsOut) -> vec2<f32> {\n    return in.uv;\n}\n";
        return (String::new(), "", accessors);
    }
    let at = ext_location(layout);
    let vertex = format!(
        "    @location({at}) s3d_tangent: vec4<f32>,\n    @location({}) s3d_uv1: vec2<f32>,",
        at + 1
    );
    let varyings = "    @location(6) s3d_tangent: vec4<f32>,\n    @location(7) s3d_uv1: vec2<f32>,";
    let accessors = "fn s3d_vertex_tangent(v: VertexInput) -> vec4<f32> {\n    return v.s3d_tangent;\n}\n\n\
         fn s3d_vertex_uv1(v: VertexInput) -> vec2<f32> {\n    \
         return select(v.uv, v.s3d_uv1, abs(v.s3d_tangent.w) > 0.5);\n}\n\n\
         // Tangent to world space (model 3x3; mirrored transforms flip the bitangent sign), and the\n\
         // transformed second UV set.\n\
         fn s3d_vertex_ext(v: VertexInput, out: VsOut, model: mat3x3<f32>, mirrored: bool) -> VsOut {\n    \
         var o = out;\n    \
         let t = model * v.s3d_tangent.xyz;\n    \
         let len2 = dot(t, t);\n    \
         let world = select(vec3<f32>(0.0), t * inverseSqrt(max(len2, 1e-30)), len2 > 1e-30);\n    \
         o.s3d_tangent = vec4<f32>(world, v.s3d_tangent.w * select(1.0, -1.0, mirrored));\n    \
         o.s3d_uv1 = s3d_uv_transform(v.s3d_uv1);\n    \
         return o;\n}\n\n\
         fn s3d_frag_tangent(in: VsOut) -> vec4<f32> {\n    return in.s3d_tangent;\n}\n\n\
         fn s3d_frag_uv1(in: VsOut) -> vec2<f32> {\n    return in.s3d_uv1;\n}\n";
    (vertex, varyings, accessors)
}

/// Full forward shader source for an instance layout, texture kinds (`PipelineKey::kinds`), the
/// extension vertex stream and a custom shader. The output encoding and the other pipeline
/// overrides are not part of the source (see `PipelineKey::constants`).
pub(crate) fn forward_shader(layout: &InstanceLayout, kinds: u8, ext: bool, custom: Option<&CustomShader>) -> String {
    let mut varyings = String::new();
    if custom.is_some() {
        varyings += "    @location(5) custom: vec4<f32>,";
    }
    let (ext_vertex, ext_varyings, ext_accessors) = ext_wgsl(layout, ext);
    if !ext_varyings.is_empty() {
        if !varyings.is_empty() {
            varyings += "\n";
        }
        varyings += ext_varyings;
    }
    let color_layered = kinds & KIND_COLOR_LAYERED != 0;
    let data_layered = kinds & KIND_DATA_LAYERED != 0;
    let consts = format!(
        "const S3D_COLOR_LAYERED: bool = {color_layered};\n\
         const S3D_DATA_LAYERED: bool = {data_layered};\n\
         const S3D_EXT: bool = {ext};"
    );
    // The user's code is spliced in last, so markers inside it are left alone. `//#EXT_VERTEX`
    // goes before `//#EXT` (its prefix).
    FORWARD_TEMPLATE
        .replace("//#CONSTS", &consts)
        .replace("//#VIEW", VIEW_STRUCTS)
        .replace("//#OUTPUT", OUTPUT_WGSL)
        .replace("//#TEXTURE", &texture_wgsl(color_layered, data_layered))
        .replace("//#EXT_VERTEX", &ext_vertex)
        .replace("//#EXT", ext_accessors)
        .replace("//#INSTANCE", &instance_wgsl(layout))
        .replace("//#VARYINGS", &varyings)
        .replace("//#HOOKS", &hooks_wgsl(custom))
}

/// The overrides of a forward pipeline that differ from their defaults: the output encoding
/// (`S3D_OUTPUT`, `MANUAL_SRGB`), alpha masking (`S3D_ALPHA_MASK`, and
/// `S3D_ALPHA_TO_COVERAGE` with MSAA) and the frame-wide globals (`S3D_SHADING_STANDARD`,
/// `S3D_NORMAL_MAPS`, `S3D_SHADOW_TAPS`; see [`GLOBAL_SHADING_STANDARD`]).
pub(crate) fn pipeline_constants(key: &PipelineKey) -> ([(&'static str, f64); 8], usize) {
    let (output, n_out) = output_constants(key.output);
    let mut c = [("", 0.0); 8];
    c[..n_out].copy_from_slice(&output[..n_out]);
    let mut n = n_out;
    let mut push = |name: &'static str, value: f64| {
        c[n] = (name, value);
        n += 1;
    };
    if BlendMode::from_class(key.class) == BlendMode::AlphaMask {
        push("S3D_ALPHA_MASK", 1.0);
        if key.samples > 1 {
            push("S3D_ALPHA_TO_COVERAGE", 1.0);
        }
    }
    if key.globals & GLOBAL_SHADING_STANDARD != 0 {
        push("S3D_SHADING_STANDARD", 1.0);
    }
    if key.globals & GLOBAL_NO_NORMAL_MAPS != 0 {
        push("S3D_NORMAL_MAPS", 0.0);
    }
    let taps = shadow_taps(key.globals);
    if taps != 0 {
        push("S3D_SHADOW_TAPS", taps as f64);
    }
    (c, n)
}

/// `PipelineKey::globals` bit: `ShadingModel::Standard`.
pub(crate) const GLOBAL_SHADING_STANDARD: u8 = 1;
/// `PipelineKey::globals` bit: `LightingSettings::normal_maps` is off.
pub(crate) const GLOBAL_NO_NORMAL_MAPS: u8 = 2;
/// `PipelineKey::globals` bits 2..=4: the casting light's filter (0 without a casting light,
/// then `Hard`, `Pcf4`, `Pcf9`, `Pcf16`), the `S3D_SHADOW_TAPS` override.
pub(crate) const GLOBAL_SHADOW_SHIFT: u8 = 2;
const GLOBAL_SHADOW_MASK: u8 = 0b111 << GLOBAL_SHADOW_SHIFT;

/// The frame-wide override bits (`PipelineKey::globals`) of the lighting settings and the
/// casting light's filter (`None`: no light casts).
pub(crate) fn lighting_globals(
    s: &crate::lighting::api::LightingSettings,
    shadow: Option<crate::lighting::api::ShadowFilter>,
) -> u8 {
    use crate::lighting::api::ShadowFilter;
    let mut g = 0;
    if s.shading == crate::lighting::api::ShadingModel::Standard {
        g |= GLOBAL_SHADING_STANDARD;
    }
    if !s.normal_maps {
        g |= GLOBAL_NO_NORMAL_MAPS;
    }
    let filter: u8 = match shadow {
        None => 0,
        Some(ShadowFilter::Hard) => 1,
        Some(ShadowFilter::Pcf4) => 2,
        Some(ShadowFilter::Pcf9) => 3,
        Some(_) => 4,
    };
    g | filter << GLOBAL_SHADOW_SHIFT
}

/// `S3D_SHADOW_TAPS` of a globals set (0: shadows compiled out).
pub(crate) fn shadow_taps(globals: u8) -> u32 {
    match (globals & GLOBAL_SHADOW_MASK) >> GLOBAL_SHADOW_SHIFT {
        0 => 0,
        1 => 1,
        2 => 4,
        3 => 9,
        _ => 16,
    }
}

/// Whether a pipeline class reads the extension vertex stream (`PipelineKey::module_key`).
pub(crate) fn class_ext(class: u8) -> bool {
    class & CLASS_EXT != 0
}

/// WGSL built-in functions GLSL ES 3.00 — WebGL2, and GLES 3.0 devices through wgpu's GL
/// backend — lacks: GLSL ES 3.10 brought them. naga's GLSL backend emits them regardless of the
/// version (it polyfills `frexp`, `countLeadingZeros`, `fma` and the fine/coarse derivatives,
/// and refuses `pack4x8snorm`/`pack4x8unorm` itself), so the GL driver fails to compile the
/// shader — only on those devices, and in the browser only once the pipeline is created.
pub(super) const GLES3_MISSING: [(naga::MathFunction, &str); 12] = {
    use naga::MathFunction as M;
    [
        (M::CountOneBits, "countOneBits"),
        (M::ReverseBits, "reverseBits"),
        (M::ExtractBits, "extractBits"),
        (M::InsertBits, "insertBits"),
        (M::FirstTrailingBit, "firstTrailingBit"),
        (M::FirstLeadingBit, "firstLeadingBit"),
        (M::CountTrailingZeros, "countTrailingZeros"),
        (M::Ldexp, "ldexp"),
        (M::Dot4I8Packed, "dot4I8Packed"),
        (M::Dot4U8Packed, "dot4U8Packed"),
        (M::Unpack4xI8, "unpack4xI8"),
        (M::Unpack4xU8, "unpack4xU8"),
    ]
};

/// The first built-in function of [`GLES3_MISSING`] (or `textureGather*`, also GLSL ES 3.10)
/// any function of `module` calls, used or not.
pub(super) fn gles3_missing_builtin(module: &naga::Module) -> Option<&'static str> {
    let functions = module
        .functions
        .iter()
        .map(|(_, f)| f)
        .chain(module.entry_points.iter().map(|e| &e.function));
    for f in functions {
        for (_, e) in f.expressions.iter() {
            match *e {
                naga::Expression::Math { fun, .. } => {
                    if let Some((_, name)) = GLES3_MISSING.iter().find(|(m, _)| *m == fun) {
                        return Some(name);
                    }
                }
                naga::Expression::ImageSample { gather: Some(_), .. } => return Some("textureGather"),
                _ => {}
            }
        }
    }
    None
}

/// Translate the forward entry points (`vs_main`, `fs_main`) to GLSL ES 3.00 the way wgpu's GL
/// backend does on WebGL2 (overrides at their defaults); the error, if naga refuses (image
/// queries GLSL ES lacks such as `textureNumLevels`, `pack4x8snorm`, …).
pub(super) fn gles3_translation_error(module: &naga::Module, info: &naga::valid::ModuleInfo) -> Option<String> {
    use naga::back::glsl;
    let options = glsl::Options {
        version: glsl::Version::Embedded {
            version: 300,
            is_webgl: true,
        },
        writer_flags: glsl::WriterFlags::ADJUST_COORDINATE_SPACE | glsl::WriterFlags::FORCE_POINT_SIZE,
        binding_map: Default::default(),
        zero_initialize_workgroup_memory: true,
    };
    // Every output encoding: the fragment tail differs per `S3D_OUTPUT` value.
    let outputs = [
        OutputEncoding::Linear,
        OutputEncoding::ManualSrgb,
        OutputEncoding::Tonemapped,
    ];
    let stages = outputs.into_iter().flat_map(|output| {
        [
            (naga::ShaderStage::Vertex, "vs_main", output),
            (naga::ShaderStage::Fragment, "fs_main", output),
        ]
    });
    for (stage, entry_point, output) in stages {
        let key_constants = output_constants(output);
        let constants: naga::back::PipelineConstants = key_constants.0[..key_constants.1]
            .iter()
            .map(|(k, v)| (k.to_string(), *v))
            .collect();
        let translated =
            naga::back::pipeline_constants::process_overrides(module, info, Some((stage, entry_point)), &constants)
                .map_err(|e| e.to_string())
                .and_then(|(module, info)| {
                    let pipeline = glsl::PipelineOptions {
                        shader_stage: stage,
                        entry_point: entry_point.to_string(),
                        multiview: None,
                    };
                    let mut out = String::new();
                    let policies = naga::proc::BoundsCheckPolicies::default();
                    glsl::Writer::new(&mut out, &module, &info, &options, &pipeline, policies)
                        .and_then(|mut w| w.write().map(drop))
                        .map_err(|e| e.to_string())
                });
        if let Err(e) = translated {
            return Some(format!("{stage:?} stage ({output:?} output): {e}"));
        }
    }
    None
}

/// Resources the forward template declares, `(group, binding, name)`: the only ones in the
/// forward pipeline layout, so the only ones a custom shader can use.
pub(super) const TEMPLATE_RESOURCES: [(u32, u32, &str); 18] = [
    (0, 0, "view"),
    (0, 1, "s3d_env_specular"),
    (0, 2, "s3d_env_irradiance"),
    (0, 3, "s3d_brdf_lut"),
    (0, 4, "s3d_env_sampler"),
    (0, 5, "s3d_shadow_map"),
    (0, 6, "s3d_shadow_sampler"),
    (1, 0, "material"),
    (1, 1, "base_tex"),
    (1, 2, "base_sampler"),
    (1, 3, "s3d_normal_tex"),
    (1, 4, "s3d_normal_sampler"),
    (1, 5, "s3d_mr_tex"),
    (1, 6, "s3d_mr_sampler"),
    (1, 7, "s3d_occlusion_tex"),
    (1, 8, "s3d_occlusion_sampler"),
    (1, 9, "s3d_emissive_tex"),
    (1, 10, "s3d_emissive_sampler"),
];

/// naga validation ([`parse_and_validate`]) of a forward shader with a custom shader spliced in,
/// plus what naga accepts but the forward pipeline cannot run the same way on every backend.
///
/// Rejected on every backend (`Err`):
///
/// * resources of its own: the pipeline layout has no bindings for them, so the pipeline
///   fails validation;
/// * workgroup memory (`var<workgroup>`, used or not): WGSL allows it in compute shaders only,
///   naga in any stage, and backends disagree about it in the vertex and fragment stages the
///   hooks run in (Metal accepts it, GL and browsers reject it);
/// * any other address space but `function` and `private` (immediates, mesh and ray-tracing
///   payloads: optional capabilities the forward pipeline does not use);
/// * pipeline constants (`override`) without a default: the forward pipeline sets none, so
///   creating it fails.
///
/// What GLSL ES 3.00 (WebGL2, GLES 3.0) cannot run — the built-in functions of
/// [`GLES3_MISSING`] and `textureGather*`, and anything else naga cannot translate to GLSL
/// ES 3.00 (`textureNumLevels`, `textureNumSamples`, `pack4x8snorm`, `pack4x8unorm`, …) — is
/// rejected only when `glsl_es_300` (the device is such a GL device: the driver would fail to
/// compile it, in the browser only once the pipeline exists). Elsewhere Metal, Vulkan, DX12,
/// WebGPU and GLES 3.1+ run it, so the shader builds and draws exactly as before; the
/// portability problem comes back as `Ok(Some(warning))` (reported through
/// [`crate::Renderer::shader_warning`]), so it still shows up on the machine the shader is
/// written on.
///
/// Detected here synchronously, because on browser WebGPU the error scope around pipeline
/// creation only answers later (see [`Pipelines::poll_pending`]): until then every frame
/// drawing with the invalid pipeline is lost.
pub(crate) fn validate_custom_wgsl(src: &str, glsl_es_300: bool) -> Result<Option<String>, String> {
    use naga::AddressSpace as A;
    let (module, info) = parse_and_validate(src)?;
    for (_, g) in module.global_variables.iter() {
        let name = g.name.as_deref().unwrap_or("<unnamed>");
        match g.space {
            A::Function | A::Private => {}
            A::WorkGroup => {
                return Err(format!(
                    "custom shaders cannot use workgroup memory: `var<workgroup> {name}` exists \
                     only in compute shaders, and the hooks run in the vertex and fragment stages \
                     (use `var<private>` for per-invocation state)"
                ));
            }
            A::Uniform | A::Storage { .. } | A::Handle => {
                let template = g
                    .binding
                    .as_ref()
                    .is_some_and(|b| TEMPLATE_RESOURCES.contains(&(b.group, b.binding, name)));
                if !template {
                    let at = g.binding.as_ref().map_or(String::new(), |b| {
                        format!(" at @group({}) @binding({})", b.group, b.binding)
                    });
                    return Err(format!(
                        "custom shaders cannot declare resources: `{name}` ({:?}){at} is not bound \
                         by the forward pipeline (only `view`, the `s3d_env_*` / `s3d_brdf_lut` / \
                         `s3d_shadow_*` lighting inputs, `material`, `base_tex`, `base_sampler` and the \
                         `s3d_*_tex` / `s3d_*_sampler` material maps are)",
                        g.space
                    ));
                }
            }
            space => {
                return Err(format!(
                    "custom shaders cannot declare `{name}` in the {space:?} address space: \
                     only `var<private>` globals (and the template's resources) are portable to \
                     every backend"
                ));
            }
        }
    }
    for (_, o) in module.overrides.iter() {
        if o.init.is_none() {
            let name = o.name.as_deref().unwrap_or("<unnamed>");
            return Err(format!(
                "custom shaders cannot declare pipeline constants without a default: \
                 `override {name}` needs one (`override {name}: T = value;`), since the forward \
                 pipeline sets no constants"
            ));
        }
    }
    let gles3 = if let Some(name) = gles3_missing_builtin(&module) {
        format!("it uses `{name}`, which GLSL ES 3.00 lacks")
    } else if let Some(e) = gles3_translation_error(&module, &info) {
        format!("it does not translate to GLSL ES 3.00: {e}")
    } else {
        return Ok(None);
    };
    if glsl_es_300 {
        Err(format!(
            "custom shader cannot build on this device, which compiles GLSL ES 3.00 like WebGL2 \
             and GLES 3.0: {gles3}"
        ))
    } else {
        Ok(Some(format!(
            "custom shader builds on this device but not on WebGL2 / GLES 3.0 (GLSL ES 3.00), \
             where its draws use the built-in shader: {gles3}"
        )))
    }
}

/// Material bind group layouts by texture kinds (`PipelineKey::kinds`: color maps layered,
/// data maps layered): the material uniform (binding 0, vertex and fragment: the vertex stage
/// applies the UV transform) and five maps with their samplers (bindings 1–10: base color,
/// normal, metallic-roughness, occlusion, emissive).
pub(super) fn material_bgls(device: &wgpu::Device) -> [wgpu::BindGroupLayout; 4] {
    [0u8, 1, 2, 3].map(|kinds| {
        let color = sampled_view_dimension(kinds & KIND_COLOR_LAYERED != 0);
        let data = sampled_view_dimension(kinds & KIND_DATA_LAYERED != 0);
        let sampler = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(match kinds {
                0 => "space3d material",
                1 => "space3d material (color arrays)",
                2 => "space3d material (data arrays)",
                _ => "space3d material (arrays)",
            }),
            entries: &[
                uniform_entry(
                    0,
                    wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    true,
                    std::mem::size_of::<GpuMaterial>() as u64,
                ),
                texture_entry(1, color),
                sampler(2),
                texture_entry(3, data),
                sampler(4),
                texture_entry(5, data),
                sampler(6),
                texture_entry(7, data),
                sampler(8),
                texture_entry(9, color),
                sampler(10),
            ],
        })
    })
}

impl Pipelines {
    /// Forward and depth pipelines built so far (tests).
    #[cfg(test)]
    pub fn pipeline_counts(&self) -> (usize, usize) {
        (self.forward.len(), self.depth.len())
    }
    /// Build a forward pipeline. Custom shaders are validated with naga first (including that
    /// they declare no resources the layout lacks; see [`validate_custom_wgsl`] for what is
    /// rejected and what only warned about) and built inside validation and internal
    /// error scopes, so their errors never reach the uncaptured-error handler. Where the
    /// scopes do not answer immediately (browser WebGPU), the pipeline is returned with its
    /// pending result (for [`Self::poll_pending`]).
    #[allow(clippy::type_complexity)]
    pub(super) fn build_forward(
        &mut self,
        device: &wgpu::Device,
        key: PipelineKey,
        custom: Option<&Arc<CustomShader>>,
    ) -> Result<
        (
            wgpu::RenderPipeline,
            Option<(ModuleKey, space3d_tasks::Task<Option<String>>)>,
        ),
        String,
    > {
        let layout = self.layouts[key.layout as usize];
        let module_key = key.module_key();
        let cached = self.shaders.get(&module_key).cloned();
        let source = match cached {
            Some(_) => None,
            None => {
                let (_, kinds, ext, _, _) = module_key;
                let source = forward_shader(layout, kinds, ext, custom.map(|c| &**c));
                if custom.is_some()
                    && let Some(warning) = validate_custom_wgsl(&source, self.glsl_es_300)?
                {
                    self.custom_warned(key, warning);
                }
                Some(source)
            }
        };
        // Backend shader translation failures are internal errors, not validation errors.
        // (Pushed after every early return: the scopes must be popped in reverse order.)
        let scopes = custom.map(|_| {
            (
                device.push_error_scope(wgpu::ErrorFilter::Validation),
                device.push_error_scope(wgpu::ErrorFilter::Internal),
            )
        });
        let module = match cached {
            Some(m) => m,
            None => {
                let source = source.expect("generated when not cached");
                let label = custom.map_or(layout.name, |c| c.name.as_str());
                let m = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(label),
                    source: wgpu::ShaderSource::Wgsl(Cow::Owned(source)),
                });
                self.shaders.insert(module_key, m.clone());
                m
            }
        };
        let inst_attrs = instance_attributes(layout);
        let (_, kinds, ext, _, _) = module_key;
        let ext_attrs = ext_attributes(ext_location(layout));
        let mode = BlendMode::from_class(key.class);
        let gbuffer = key.pass == super::PassKind::GBuffer;
        // (Transparent classes without mode bits are the legacy alpha-blended class.)
        let blend = if key.class & CLASS_BLEND != 0 {
            mode.blend_state().or(Some(wgpu::BlendState::ALPHA_BLENDING))
        } else {
            None
        };
        let (constants, n) = key.constants();
        let vertex_buffers = [
            Some(wgpu::VertexBufferLayout {
                array_stride: VERTEX_SIZE,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &VERTEX_ATTRIBUTES,
            }),
            Some(wgpu::VertexBufferLayout {
                array_stride: layout.stride as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &inst_attrs,
            }),
            Some(wgpu::VertexBufferLayout {
                array_stride: EXT_VERTEX_SIZE,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &ext_attrs,
            }),
        ];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(layout.name),
            layout: Some(&self.layouts_by_texture[(kinds & (KIND_COLOR_LAYERED | KIND_DATA_LAYERED)) as usize]),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &vertex_buffers[..if ext { 3 } else { 2 }],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: if key.class & CLASS_DOUBLE_SIDED != 0 {
                    None
                } else {
                    Some(wgpu::Face::Back)
                },
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(blend.is_none() && !gbuffer),
                depth_compare: Some(if gbuffer {
                    wgpu::CompareFunction::LessEqual
                } else {
                    wgpu::CompareFunction::Less
                }),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: key.samples,
                alpha_to_coverage_enabled: mode == BlendMode::AlphaMask && key.samples > 1,
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some(if gbuffer { "fs_gbuffer" } else { "fs_main" }),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants[..n],
                    ..Default::default()
                },
                targets: &[Some(wgpu::ColorTargetState {
                    format: key.format,
                    blend,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let Some((validation, internal)) = scopes else {
            return Ok((pipeline, None));
        };
        // Scopes pop innermost first.
        let (internal, validation) = (internal.pop(), validation.pop());
        let mut error = Box::pin(async move {
            match internal.await {
                Some(e) => Some(e.to_string()),
                None => validation.await.map(|e| e.to_string()),
            }
        });
        // Native backends resolve the scopes immediately; in the browser the result arrives
        // later: the pipeline is used meanwhile and replaced if it turns out to be invalid.
        match futures::FutureExt::now_or_never(&mut error) {
            Some(Some(e)) => {
                self.shaders.remove(&module_key);
                Err(e)
            }
            Some(None) => Ok((pipeline, None)),
            None => Ok((pipeline, Some((module_key, space3d_tasks::spawn(error))))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::{
        InstanceAttribute, InstanceData, InstanceHalf, InstancePacked, InstanceSemantic, InstanceTransform,
        InstanceTransformColor,
    };
    use crate::material::{CLASS_MODE_MASK, ShadowMode};
    use wgpu::VertexFormat as F;

    /// 11 instance locations (a matrix, a color, custom data…): no room for the extension
    /// stream.
    const WIDE: &InstanceLayout = &InstanceLayout {
        name: "Wide",
        stride: 136,
        attributes: &[
            InstanceAttribute::new(InstanceSemantic::ModelMatrix, F::Float32x4, 0),
            InstanceAttribute::new(InstanceSemantic::Color, F::Float32x4, 64),
            InstanceAttribute::new(InstanceSemantic::UvRect, F::Float32x4, 80),
            InstanceAttribute::new(InstanceSemantic::TextureLayer, F::Uint32, 96),
            InstanceAttribute::new(InstanceSemantic::Custom(0), F::Float32x4, 100),
            InstanceAttribute::new(InstanceSemantic::Custom(1), F::Float32x2, 116),
            InstanceAttribute::new(InstanceSemantic::Custom(2), F::Float32, 124),
            InstanceAttribute::new(InstanceSemantic::Custom(3), F::Float32x2, 128),
        ],
    };

    /// Custom shaders reading the material-map helpers and the extension accessors.
    const FIXTURES: [&str; 2] = [
        "fn vertex_hook(v: VertexInput, inst: InstanceInput, out: VsOut) -> VsOut {
             var o = out;
             o.custom = s3d_vertex_tangent(v) + vec4<f32>(s3d_vertex_uv1(v), 0.0, 0.0);
             return o;
         }
         fn fragment_hook(in: VsOut, c: vec4<f32>) -> vec4<f32> {
             let n = s3d_sample_normal(in.uv, in.layer) + s3d_sample_mr(in.uv, in.layer);
             let e = s3d_sample_emissive(in.uv, in.layer) * s3d_sample_occlusion(in.uv, in.layer);
             return c * n * e + in.custom * material.maps.x + vec4<f32>(f32(material.flags.x));
         }",
        "fn fragment_hook(in: VsOut, c: vec4<f32>) -> vec4<f32> {
             let uv = s3d_frag_uv1(in) + material.uv_row0.xy;
             return vec4<f32>(c.rgb * s3d_sample_emissive(uv, in.layer).rgb, c.a) * s3d_frag_tangent(in).w;
         }",
    ];

    fn layouts() -> [&'static InstanceLayout; 5] {
        [
            InstanceTransform::LAYOUT,
            InstanceTransformColor::LAYOUT,
            InstancePacked::LAYOUT,
            InstanceHalf::LAYOUT,
            WIDE,
        ]
    }

    /// Every override combination of the forward shader: output encoding × alpha mask ×
    /// alpha to coverage × shading model × normal maps.
    fn override_sets() -> Vec<naga::back::PipelineConstants> {
        let mut out = Vec::new();
        for output in 0..3u32 {
            for bits in 0..16u32 {
                let mut c = naga::back::PipelineConstants::default();
                // Every shadow filter (and none) with several of the other combinations.
                let taps = [0u32, 1, 4, 9, 16][((bits + output) % 5) as usize];
                c.insert("S3D_SHADOW_TAPS".into(), taps as f64);
                c.insert("S3D_OUTPUT".into(), output as f64);
                c.insert("MANUAL_SRGB".into(), (output == 1) as u32 as f64);
                c.insert("S3D_ALPHA_MASK".into(), (bits & 1) as f64);
                c.insert("S3D_ALPHA_TO_COVERAGE".into(), (bits >> 1 & 1) as f64);
                c.insert("S3D_SHADING_STANDARD".into(), (bits >> 2 & 1) as f64);
                c.insert("S3D_NORMAL_MAPS".into(), (bits >> 3 & 1) as f64);
                out.push(c);
            }
        }
        out
    }

    /// GLSL ES 3.00 (WebGL2) translation of both entry points with `constants`.
    fn glsl_es_300(module: &naga::Module, info: &naga::valid::ModuleInfo, constants: &naga::back::PipelineConstants) {
        use naga::back::glsl;
        let options = glsl::Options {
            version: glsl::Version::Embedded {
                version: 300,
                is_webgl: true,
            },
            writer_flags: glsl::WriterFlags::ADJUST_COORDINATE_SPACE | glsl::WriterFlags::FORCE_POINT_SIZE,
            binding_map: Default::default(),
            zero_initialize_workgroup_memory: true,
        };
        for (stage, entry) in [
            (naga::ShaderStage::Vertex, "vs_main"),
            (naga::ShaderStage::Fragment, "fs_main"),
        ] {
            let (module, info) =
                naga::back::pipeline_constants::process_overrides(module, info, Some((stage, entry)), constants)
                    .unwrap_or_else(|e| panic!("{stage:?} with {constants:?}: {e}"));
            let pipeline = glsl::PipelineOptions {
                shader_stage: stage,
                entry_point: entry.to_string(),
                multiview: None,
            };
            let mut out = String::new();
            glsl::Writer::new(
                &mut out,
                &module,
                &info,
                &options,
                &pipeline,
                naga::proc::BoundsCheckPolicies::default(),
            )
            .and_then(|mut w| w.write().map(drop))
            .unwrap_or_else(|e| panic!("{stage:?} with {constants:?}: {e}"));
        }
    }

    /// Every forward permutation of the materials area — instance layouts × texture kinds ×
    /// extension stream × custom shaders — validates with naga (uniformity included: maps are
    /// sampled behind uniform flags) and translates to GLSL ES 3.00 with every override value.
    #[test]
    fn material_permutations_validate_and_translate_to_glsl_es_300() {
        let customs: Vec<Option<CustomShader>> = std::iter::once(None)
            .chain(FIXTURES.iter().map(|w| Some(CustomShader::new("fixture", *w))))
            .collect();
        let overrides = override_sets();
        let mut modules = 0;
        for layout in layouts() {
            for kinds in 0..4u8 {
                for ext in [false, true] {
                    if ext && !super::super::ext_fits(layout) {
                        continue;
                    }
                    for custom in &customs {
                        let src = forward_shader(layout, kinds, ext, custom.as_ref());
                        let (module, info) =
                            parse_and_validate(&src).unwrap_or_else(|e| panic!("{} {kinds} {ext}: {e}", layout.name));
                        if let Some(c) = custom {
                            assert_eq!(validate_custom_wgsl(&src, true), Ok(None), "{}", c.name);
                        }
                        // Every override combination on the built-in shader; the defaults with
                        // custom shaders.
                        let sets = if custom.is_some() {
                            &overrides[..1]
                        } else {
                            &overrides[..]
                        };
                        for constants in sets {
                            glsl_es_300(&module, &info, constants);
                        }
                        assert_eq!(gles3_missing_builtin(&module), None);
                        modules += 1;
                    }
                }
            }
        }
        assert_eq!(modules, (4 * 2 * 4 + 4) * 3, "WIDE has no ext variant");
        // The extension attributes follow the instance attributes.
        let src = forward_shader(InstanceTransform::LAYOUT, 0, true, None);
        assert!(src.contains("@location(8) s3d_tangent: vec4<f32>"), "{src}");
        assert!(src.contains("@location(9) s3d_uv1: vec2<f32>"));
        assert!(src.contains("@location(6) s3d_tangent: vec4<f32>"), "VsOut");
        assert!(!forward_shader(InstanceTransform::LAYOUT, 0, false, None).contains("s3d_tangent:"));
        assert_eq!(ext_location(InstancePacked::LAYOUT), 9);
        assert!(!super::super::ext_fits(WIDE) && super::super::ext_fits(InstanceTransformColor::LAYOUT));
    }

    /// `GpuMaterial` sits where `MaterialParams` of the template reads it.
    #[test]
    fn gpu_material_matches_wgsl() {
        use std::mem::offset_of;
        let module = naga::front::wgsl::parse_str(&forward_shader(InstanceTransform::LAYOUT, 0, false, None)).unwrap();
        let (_, t) = module
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some("MaterialParams"))
            .unwrap();
        let naga::TypeInner::Struct { members, span } = &t.inner else {
            panic!()
        };
        assert_eq!(*span as usize, std::mem::size_of::<GpuMaterial>());
        let rust = [
            ("base_color", offset_of!(GpuMaterial, base_color)),
            ("emissive", offset_of!(GpuMaterial, emissive)),
            ("params", offset_of!(GpuMaterial, params)),
            ("maps", offset_of!(GpuMaterial, maps)),
            ("flags", offset_of!(GpuMaterial, flags)),
            ("uv_row0", offset_of!(GpuMaterial, uv_row0)),
            ("uv_row1", offset_of!(GpuMaterial, uv_row1)),
        ];
        assert_eq!(members.len(), rust.len());
        for (m, (name, offset)) in members.iter().zip(rust) {
            assert_eq!((m.name.as_deref(), m.offset as usize), (Some(name), offset));
        }
        // The flag constants of the template mirror `material.rs`.
        let src = FORWARD_TEMPLATE;
        for (name, value) in [
            ("S3D_MAP_NORMAL", crate::material::FLAG_NORMAL),
            ("S3D_MAP_MR", crate::material::FLAG_METALLIC_ROUGHNESS),
            ("S3D_MAP_OCCLUSION", crate::material::FLAG_OCCLUSION),
            ("S3D_MAP_EMISSIVE", crate::material::FLAG_EMISSIVE),
            ("S3D_ORM_PACKED", crate::material::FLAG_ORM_PACKED),
            ("S3D_UV_TRANSFORM", crate::material::FLAG_UV_TRANSFORM),
            ("S3D_FIXUP_NORMAL", 1 << (crate::material::FLAG_FIXUP_SHIFT + 1)),
            ("S3D_FIXUP_MR", 1 << (crate::material::FLAG_FIXUP_SHIFT + 2)),
            ("S3D_FIXUP_OCCLUSION", 1 << (crate::material::FLAG_FIXUP_SHIFT + 3)),
            ("S3D_NO_SHADOW", crate::material::FLAG_NO_SHADOW),
        ] {
            assert!(src.contains(&format!("const {name}: u32 = {value}u;")), "{name}");
        }
    }

    /// Blend-mode class bits round-trip, and the overrides follow the class and the globals.
    #[test]
    fn class_bits_and_overrides() {
        use crate::material::BlendMode as B;
        for b in [B::Opaque, B::AlphaMask, B::AlphaBlend, B::Premultiplied, B::Additive] {
            assert_eq!(B::from_class(b.class_bits()), b);
            assert_eq!(b.class_bits() & !CLASS_MODE_MASK, 0);
        }
        assert_eq!(B::from_class(CLASS_BLEND), B::AlphaBlend, "legacy transparent class");
        let key = |class: u8, samples: u32, globals: u8| PipelineKey {
            layout: 0,
            class,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            samples,
            output: OutputEncoding::ManualSrgb,
            kinds: 0,
            shader: None,
            pass: super::super::PassKind::Forward,
            globals,
        };
        let names = |k: PipelineKey| {
            let (c, n) = k.constants();
            c[..n].iter().map(|(name, v)| format!("{name}={v}")).collect::<Vec<_>>()
        };
        assert_eq!(names(key(0, 1, 0)), ["S3D_OUTPUT=1", "MANUAL_SRGB=1"]);
        assert_eq!(
            names(key(B::AlphaMask.class_bits(), 4, 3)),
            [
                "S3D_OUTPUT=1",
                "MANUAL_SRGB=1",
                "S3D_ALPHA_MASK=1",
                "S3D_ALPHA_TO_COVERAGE=1",
                "S3D_SHADING_STANDARD=1",
                "S3D_NORMAL_MAPS=0"
            ]
        );
        assert!(
            !names(key(B::AlphaMask.class_bits(), 1, 0))
                .iter()
                .any(|n| n.contains("COVERAGE"))
        );
        assert!(key(CLASS_EXT, 1, 0).module_key().2 && !key(0, 1, 0).module_key().2);
        // The casting light's filter: `S3D_SHADOW_TAPS`, nothing without a casting light.
        use crate::lighting::api::{LightingSettings, ShadowFilter};
        let settings = LightingSettings::default();
        assert_eq!(lighting_globals(&settings, None), 0);
        for (filter, taps) in [
            (ShadowFilter::Hard, 1),
            (ShadowFilter::Pcf4, 4),
            (ShadowFilter::Pcf9, 9),
            (ShadowFilter::Pcf16, 16),
        ] {
            let globals = lighting_globals(&settings, Some(filter));
            assert_eq!(shadow_taps(globals), taps);
            assert_eq!(names(key(0, 1, globals))[2..], [format!("S3D_SHADOW_TAPS={taps}")]);
            assert_eq!(key(0, 1, globals).module_key(), key(0, 1, 0).module_key());
        }
        let _ = ShadowMode::Auto;
    }

    /// Every material pipeline variant builds on the device without errors: blend modes (and
    /// alpha to coverage with MSAA), texture kinds, the extension stream, globals, custom shaders.
    #[test]
    fn material_pipelines_build_on_the_device() {
        use crate::material::BlendMode as B;
        let Ok(mut r) = pollster::block_on(crate::Renderer::new_headless(4, 4)) else {
            eprintln!("skipping GPU test: no adapter available");
            return;
        };
        let mut assets = crate::RenderAssets::default();
        let custom = CustomShader::new("fixture", FIXTURES[0]);
        let handle = assets.add_shader(custom.clone());
        r.pipelines.sync_custom(handle, 0, &Arc::new(custom));
        let ids: Vec<u16> = [InstanceTransform::LAYOUT, WIDE]
            .into_iter()
            .map(|l| r.pipelines.layout_id(l))
            .collect();
        let mut built = 0;
        for &layout in &ids {
            for mode in [B::Opaque, B::AlphaMask, B::AlphaBlend, B::Premultiplied, B::Additive] {
                let transparent = if mode.is_transparent() { CLASS_BLEND } else { 0 };
                for ext in [0, CLASS_EXT] {
                    if ext != 0 && layout == ids[1] {
                        continue;
                    }
                    for kinds in 0..4u8 {
                        // Globals: none; Standard + no normal maps + the 9-tap shadow.
                        for (samples, globals) in [(1, 0), (4, 3 | 3 << GLOBAL_SHADOW_SHIFT)] {
                            for shader in [None, Some((handle, 0))] {
                                let key = PipelineKey {
                                    layout,
                                    class: mode.class_bits() | transparent | ext | CLASS_DOUBLE_SIDED,
                                    format: wgpu::TextureFormat::Rgba16Float,
                                    samples,
                                    output: OutputEncoding::Linear,
                                    kinds,
                                    shader,
                                    pass: super::super::PassKind::Forward,
                                    globals,
                                };
                                r.pipelines.forward(&r.device, key);
                                assert_eq!(r.pipelines.custom_error(handle), None, "{key:?}");
                                built += 1;
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(built, 5 * (2 + 1) * 4 * 2 * 2);
        assert_eq!(r.gpu_error_count(), 0);
    }
}
