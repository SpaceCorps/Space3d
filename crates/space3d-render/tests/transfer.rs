//! The sRGB transfer function, everywhere it is written down: the CPU pair
//! (`linear_to_srgb` / `srgb_to_linear`), the WGSL pair of `OUTPUT_WGSL` (`s3d_linear_to_srgb` /
//! `s3d_srgb_to_linear`, used by the forward tail, the overlays and the composite) and the GPU's
//! own encoding of an sRGB-format target. Golden values, the two against each other over every
//! 8-bit code, and one grey ramp through every way a pass can write its output
//! (`OutputEncoding::Linear` into an sRGB target, `OutputEncoding::ManualSrgb` into an 8-bit
//! target without one, `OutputEncoding::Linear` into a float target encoded on the CPU): the same
//! bytes. The composite's `S3D_POST_OUT` kinds end in the same three writes.

mod common;

use common::headless_app;
use space3d_render::*;

/// IEC 61966-2-1: the two segments meet at 0.0031308 linear = 0.04045 encoded, the linear
/// segment's slope is 12.92, and the well-known greys land on their codes.
#[test]
fn srgb_golden_values() {
    assert!((linear_to_srgb(0.003_130_8) - 0.040_45).abs() < 2e-6);
    assert!((srgb_to_linear(0.040_45) - 0.003_130_8).abs() < 2e-7);
    // Continuous at the knee: the two segments agree there (to 1e-5 of an encoded unit).
    let knee = 0.003_130_8f32;
    let upper = 1.055 * knee.powf(1.0 / 2.4) - 0.055;
    assert!((knee * 12.92 - upper).abs() < 1e-5, "{} vs {upper}", knee * 12.92);
    assert_eq!(linear_to_srgb(0.001), 0.001 * 12.92);
    assert_eq!(srgb_to_linear(0.02), 0.02 / 12.92);
    let code = |linear: f32| (linear_to_srgb(linear) * 255.0).round() as u8;
    assert_eq!(
        [0.0, 0.18, 0.215_861, 0.5, 1.0].map(code),
        [0, 118, 128, 188, 255],
        "black, middle grey, code 128, half, white"
    );
    assert!((srgb_to_linear(0.5) - 0.214_041).abs() < 1e-6);
    // Every 8-bit code survives the round trip, and the curve rises all the way.
    let mut last = -1.0;
    for c in 0..=255u8 {
        let linear = srgb_to_linear(f32::from(c) / 255.0);
        assert!(linear > last, "{c}");
        last = linear;
        assert_eq!(code(linear), c);
    }
    // The WGSL pair is written with the same constants.
    for constant in ["12.92", "0.04045", "0.0031308", "1.055", "0.055", "2.4"] {
        assert!(OUTPUT_WGSL.contains(constant), "{constant}");
    }
}

/// What a 256 × 1 pass writes: the ramp `RAMP` (0: grey code x decoded with
/// `s3d_srgb_to_linear`; 1: linear x / 255; 2: `s3d_linear_to_srgb(x / 255)` as it is), sent
/// through `s3d_output` with the encoding `ENCODING`.
const RAMP_WGSL: &str = r"
override RAMP: u32 = 0u;
override ENCODING: u32 = 0u;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let x = vec3<f32>(floor(pos.x) / 255.0);
    var linear = x;
    if (RAMP == 0u) {
        linear = s3d_srgb_to_linear(x);
    } else if (RAMP == 2u) {
        linear = s3d_linear_to_srgb(x);
    }
    return s3d_output(vec4<f32>(linear, 1.0), ENCODING, 1.0, S3D_TM_NONE);
}
";

fn f16_to_f32(h: u16) -> f32 {
    let (sign, exp, frac) = (h >> 15, (h >> 10) & 0x1f, f32::from(h & 0x3ff));
    let v = match exp {
        0 => frac * 2f32.powi(-24),
        31 => f32::INFINITY,
        e => (1.0 + frac / 1024.0) * 2f32.powi(i32::from(e) - 15),
    };
    if sign == 1 { -v } else { v }
}

/// The red channel of the 256 texels the ramp pass writes into `format`: bytes as stored (0..255
/// as `f32`), or the float value of a float target.
fn ramp(renderer: &Renderer, format: wgpu::TextureFormat, ramp: u32, encoding: OutputEncoding) -> Vec<f32> {
    let (device, queue) = (renderer.device(), renderer.queue());
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("transfer ramp"),
        source: wgpu::ShaderSource::Wgsl(format!("{OUTPUT_WGSL}\n{RAMP_WGSL}").into()),
    });
    let constants = [("RAMP", f64::from(ramp)), ("ENCODING", f64::from(encoding.id()))];
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("transfer ramp"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                ..Default::default()
            },
            targets: &[Some(format.into())],
        }),
        multiview_mask: None,
        cache: None,
    });
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("transfer ramp"),
        size: wgpu::Extent3d {
            width: 256,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    let texel = format.block_copy_size(None).expect("color format") as usize;
    let size = (256 * texel).next_multiple_of(256) as u64;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("transfer ramp"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("transfer ramp"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&pipeline);
        pass.draw(0..3, 0..1);
    }
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size as u32),
                rows_per_image: None,
            },
        },
        texture.size(),
    );
    queue.submit([encoder.finish()]);
    buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = buffer.slice(..).get_mapped_range().expect("mapped");
    (0..256)
        .map(|x| {
            let p = &data[x * texel..][..texel];
            match format {
                wgpu::TextureFormat::Rgba16Float => f16_to_f32(u16::from_le_bytes([p[0], p[1]])),
                _ => f32::from(p[0]),
            }
        })
        .collect()
}

/// Largest difference between two rows of codes, and where.
fn worst(a: &[f32], b: &[f32]) -> (f32, usize) {
    a.iter()
        .zip(b)
        .enumerate()
        .map(|(i, (x, y))| ((x - y).abs(), i))
        .fold((0.0, 0), |m, v| if v.0 > m.0 { v } else { m })
}

/// One grey ramp (every 8-bit code, decoded in the shader) through every way a pass writes its
/// output comes out as the same bytes, the codes themselves: the GPU's encoding of an sRGB
/// target, the shader's own encoding into an 8-bit target without an sRGB format, and a float
/// target encoded on the CPU.
#[test]
fn every_output_kind_writes_the_same_bytes() {
    let Some(app) = headless_app(4, 4) else { return };
    let renderer = app.resources.expect::<Renderer>();
    use wgpu::TextureFormat as F;
    let codes: Vec<f32> = (0..256).map(|c| c as f32).collect();
    let hardware = ramp(renderer, F::Rgba8UnormSrgb, 0, OutputEncoding::Linear);
    let manual = ramp(renderer, F::Rgba8Unorm, 0, OutputEncoding::ManualSrgb);
    let float: Vec<f32> = ramp(renderer, F::Rgba16Float, 0, OutputEncoding::Linear)
        .into_iter()
        .map(|linear| (linear_to_srgb(linear) * 255.0).round())
        .collect();
    assert_eq!(hardware, codes, "sRGB target (the GPU encodes)");
    assert_eq!(manual, codes, "8-bit target (the shader encodes)");
    // A half float holds 11 bits: the dark codes are a third of a code apart in it at worst.
    assert_eq!(float, codes, "float target, encoded on the CPU");
    assert_eq!(renderer.gpu_error_count(), 0);
}

/// The WGSL transfer functions against the CPU's over a 256-entry sweep, within 1 LSB (the
/// test prints the worst case; 1 code at a rounding boundary on the GPU this was written on):
/// * `s3d_linear_to_srgb(x / 255)` written by the shader into an 8-bit target is the CPU's
///   `linear_to_srgb(x / 255)` rounded;
/// * so is the GPU's own encoding of an sRGB target for the same linear ramp;
/// * `s3d_srgb_to_linear` is the CPU's `srgb_to_linear`, read from a float target (half-float
///   precision: 1/1024 relative).
#[test]
fn wgsl_transfer_functions_match_the_cpu() {
    let Some(app) = headless_app(4, 4) else { return };
    let renderer = app.resources.expect::<Renderer>();
    use wgpu::TextureFormat as F;
    let cpu: Vec<f32> = (0..256)
        .map(|x| (linear_to_srgb(x as f32 / 255.0) * 255.0).round())
        .collect();
    let manual = ramp(renderer, F::Rgba8Unorm, 1, OutputEncoding::ManualSrgb);
    let hardware = ramp(renderer, F::Rgba8UnormSrgb, 1, OutputEncoding::Linear);
    let (m, h) = (worst(&manual, &cpu), worst(&hardware, &cpu));
    eprintln!(
        "linear ramp, codes off the CPU's at worst: {} (shader encode, at {}), {} (sRGB target, at {})",
        m.0, m.1, h.0, h.1
    );
    assert!(m.0 <= 1.0, "s3d_linear_to_srgb: {m:?}");
    assert!(h.0 <= 1.0, "sRGB target: {h:?}");
    // The same function written as data (no second encoding): into a float target.
    let encoded = ramp(renderer, F::Rgba16Float, 2, OutputEncoding::Linear);
    for (x, got) in encoded.iter().enumerate() {
        let want = linear_to_srgb(x as f32 / 255.0);
        assert!(
            (got - want).abs() <= want / 1024.0 + 1e-6,
            "encode {x}: {got} vs {want}"
        );
    }
    let decoded = ramp(renderer, F::Rgba16Float, 0, OutputEncoding::Linear);
    for (x, got) in decoded.iter().enumerate() {
        let want = srgb_to_linear(x as f32 / 255.0);
        assert!(
            (got - want).abs() <= want / 1024.0 + 1e-6,
            "decode {x}: {got} vs {want}"
        );
    }
    assert_eq!(renderer.gpu_error_count(), 0);
}
