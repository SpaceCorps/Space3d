//! The post chain of a post group: pooled bloom levels, the exposure texture, the LDR
//! intermediate before FXAA, and the composite pass (graphics spec §1.9). Written by the frame
//! milestone F4 (composite only: exposure, tonemapping, dither, alpha, rect mapping and
//! upscale) and grown by the post area (PO1–PO6).

use super::PostGroupPlan;
use super::api::{BloomComposite, FilmGrain};
use super::grading;
use crate::node::FrameMode;
use crate::output::{Tonemapper, pcg};
use bytemuck::{Pod, Zeroable};

/// Effect bits of `PostParams::flags.x` (composite.wgsl `S3D_POST_*`). `GRADE`: white balance
/// × saturation and contrast (HDR); `GRADE_LDR`: lift/gamma/gain (display); `LUT`: the 3D LUT.
#[allow(dead_code, reason = "the post area (PO1–PO5) sets the effect bits")]
pub(crate) mod bits {
    pub const BLOOM: u32 = 1;
    pub const AUTO_EXPOSURE: u32 = 2;
    pub const VIGNETTE: u32 = 4;
    pub const CA: u32 = 8;
    pub const GRAIN: u32 = 16;
    pub const DITHER: u32 = 32;
    pub const GRADE: u32 = 64;
    pub const LUT: u32 = 128;
    pub const DISTORTION: u32 = 256;
    pub const TRANSPARENT: u32 = 512;
    pub const SCENE_ALPHA: u32 = 1024;
    pub const EXACT: u32 = 2048;
    pub const GRADE_LDR: u32 = 4096;
    /// Bloom level 0 is read with one bilinear tap (`Bloom::fast`).
    pub const BLOOM_BILINEAR: u32 = 8192;
    /// The display gamma is not 1 (`PostProcess::display_gamma`).
    pub const DISPLAY_GAMMA: u32 = 16384;
}

/// Matches `PostParams` in composite.wgsl: one 256-byte uniform slot per post group. Nothing in
/// it varies per frame: the per-frame seeds (animated grain, dither) are uploaded a window of
/// frames at a time ([`SeedKey`], `SeedWindow` in post/mod.rs), so a still scene uploads
/// nothing.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct GpuPost {
    /// The group's rect in `D` pixels (x, y, w, h), unclipped.
    pub out_rect: [f32; 4],
    /// Its rect in `S` pixels, unclipped.
    pub src_rect: [f32; 4],
    /// 1/S.w, 1/S.h, rect aspect (w/h), D height in pixels.
    pub src_texel: [f32; 4],
    /// Linear exposure multiplier (the compensation with auto exposure), 1 / display gamma,
    /// bloom intensity, bloom mode (0 additive, 1 energy).
    pub exposure: [f32; 4],
    pub vignette: [f32; 4],
    pub vignette_color: [f32; 4],
    pub lens: [f32; 4],
    pub grain: [f32; 4],
    /// White balance × saturation (three std140 columns).
    pub grade_matrix: [[f32; 4]; 3],
    /// Lift (rgb), contrast.
    pub lift_contrast: [f32; 4],
    /// 1 / gamma (rgb), LUT contribution.
    pub gamma_lut: [f32; 4],
    /// Gain (rgb), LUT size N.
    pub gain_lut: [f32; 4],
    /// Effect bits ([`bits`]), tonemapper id, 1 = last pass, 1 = the LUT texture is sRGB.
    pub flags: [u32; 4],
    /// Reserved (zero): the seeds moved to the seed window (`post_seeds` in composite.wgsl).
    pub frame: [u32; 4],
}

/// Dither seed of frame `frame` (graphics spec §4.3): `pcg(0x9E3779B9 ^ frame)`.
pub(crate) fn dither_seed(frame: u64) -> u32 {
    pcg(0x9E37_79B9 ^ frame as u32)
}

/// Second dither hash seed (`s3d_hash01(px, seed) − s3d_hash01(px, seed ^ DITHER_SECOND)`).
const DITHER_SECOND: u32 = 0x68E3_1DA4;

/// XORed into both pre-hashed dither seeds for the dither of the LDR intermediate before FXAA
/// (`S3D_POST_DITHER_L` in composite.wgsl), so it is independent of the last pass's.
#[cfg(test)]
const DITHER_INTERMEDIATE: u32 = 0xA511_E9B3;

/// CPU mirror of the composite's dither noise at pixel `px` of frame `frame`, in LSB (−1..1,
/// triangular): the last pass's (`post_finish`), or the LDR intermediate's before FXAA
/// (`post_dither_intermediate`).
#[cfg(test)]
pub(crate) fn dither_noise(px: [u32; 2], frame: u64, intermediate: bool) -> f32 {
    let x = if intermediate { DITHER_INTERMEDIATE } else { 0 };
    let hash = |hashed: u32| pcg(px[0] ^ pcg(px[1] ^ hashed)) as f32 * (1.0 / 4_294_967_295.0);
    let seed = dither_seed(frame);
    hash(pcg(seed) ^ x) - hash(pcg(seed ^ DITHER_SECOND) ^ x)
}

/// Grain seed of frame `frame` (graphics spec §4.3) for a grain of `seed`: `animated ?
/// pcg(seed ^ frame) : seed`.
pub(crate) fn grain_seed(seed: u32, animated: bool, frame: u64) -> u32 {
    if animated { pcg(seed ^ frame as u32) } else { seed }
}

/// What a post group's composite does besides the settings' own switches this frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct CompositePlan {
    /// Its bloom chain is recorded.
    pub bloom: bool,
    /// The composite writes the LDR intermediate and the FXAA pass finishes (grain, dither,
    /// encoding).
    pub fxaa: bool,
    /// The exposure is metered (the composite multiplies by 2^E).
    pub auto_exposure: bool,
    /// A usable LUT: size N, the texture is sRGB.
    pub lut: Option<(u32, bool)>,
}

impl GpuPost {
    /// Parameters of one post group (see [`CompositePlan`]). Its seeds: [`SeedKey::of`].
    pub(crate) fn new(plan: &PostGroupPlan, c: CompositePlan) -> Self {
        let CompositePlan {
            bloom,
            fxaa,
            auto_exposure,
            lut,
        } = c;
        let s = &plan.settings;
        // The passthrough group is a plain copy and upscale: no effect.
        let effects = !plan.passthrough;
        let mut flags = 0;
        if s.dither && effects {
            flags |= bits::DITHER;
        }
        if plan.transparent {
            flags |= bits::TRANSPARENT;
        }
        if plan.scene_alpha {
            flags |= bits::SCENE_ALPHA;
        }
        if plan.scene_rect[2..] == plan.output_rect[2..] {
            flags |= bits::EXACT;
        }
        // The distortion buffer is written this frame: the composite samples `S` (and the
        // bloom) where it points, `lens.z` view heights per encoded unit, with the chromatic
        // split `lens.w`.
        let d = &s.distortion;
        let distortion = plan.distortion && effects && d.enabled && d.strength.is_finite() && d.strength != 0.0;
        if distortion {
            flags |= bits::DISTORTION;
        }
        // The Ldr fallback exposes and tonemaps per fragment already; passthrough copies.
        let hdr = plan.mode == FrameMode::Hdr && effects;
        let (exposure, tonemapper) = if hdr {
            (s.exposure.exp2(), s.tonemapper)
        } else {
            (1.0, Tonemapper::None)
        };
        let rect = |r: [i32; 4]| [r[0] as f32, r[1] as f32, r[2] as f32, r[3] as f32];
        let (sw, sh) = (plan.scene_size.0.max(1) as f32, plan.scene_size.1.max(1) as f32);
        let out = rect(plan.output_rect);
        let src = rect(plan.scene_rect);
        let mut p = GpuPost::zeroed();
        if distortion {
            p.lens[2] = super::api::DISTORTION_RANGE * d.strength;
            p.lens[3] = if d.chromatic.is_finite() {
                d.chromatic.clamp(0.0, 1.0)
            } else {
                0.0
            };
        }
        p.out_rect = out;
        p.src_rect = src;
        p.src_texel = [1.0 / sw, 1.0 / sh, out[2] / out[3].max(1.0), plan.output_size.1 as f32];
        // The display gamma: a uniform bit, so moving it never builds a pipeline; 1 leaves the
        // composite exactly as it was (the passthrough group copies, and gets none).
        let display_gamma = crate::output::display_gamma_of(s.display_gamma);
        let inverse_display_gamma = if effects && display_gamma != 1.0 {
            flags |= bits::DISPLAY_GAMMA;
            1.0 / display_gamma
        } else {
            1.0
        };
        p.exposure = [exposure, inverse_display_gamma, 0.0, 0.0];
        if bloom {
            flags |= bits::BLOOM;
            if s.bloom.fast {
                flags |= bits::BLOOM_BILINEAR;
            }
            let energy = s.bloom.composite == BloomComposite::EnergyConserving;
            p.exposure[2] = if energy {
                s.bloom.intensity.clamp(0.0, 1.0)
            } else {
                s.bloom.intensity.max(0.0)
            };
            p.exposure[3] = f32::from(u8::from(energy));
        }
        let v = &s.vignette;
        if effects && v.enabled && v.intensity > 0.0 {
            flags |= bits::VIGNETTE;
            p.vignette = [
                v.intensity.clamp(0.0, 1.0),
                v.smoothness.clamp(0.01, 1.0),
                v.roundness.clamp(0.0, 1.0),
                v.center.x,
            ];
            p.vignette_color = [v.color.x, v.color.y, v.color.z, v.center.y];
        }
        let ca = &s.chromatic_aberration;
        if effects && ca.enabled && ca.intensity > 0.0 {
            flags |= bits::CA;
            // Red and blue sit `d · |d|² · lens.x` apart (d: S pixels from the rect's center):
            // intensity × 2 % of the rect's width at its corners.
            let corner = 0.5 * (src[2] * src[2] + src[3] * src[3]).sqrt().max(1.0);
            p.lens[0] = ca.intensity.clamp(0.0, 1.0) * 0.02 * src[2] / (corner * corner * corner);
        }
        let g = &s.film_grain;
        if effects && g.enabled && g.intensity > 0.0 {
            flags |= bits::GRAIN;
            // Cells of `size` pixels at 1080p, scaled with the output height, at least a pixel.
            let cell = (g.size.max(0.0) * plan.output_size.1 as f32 / 1080.0).max(1.0);
            p.grain = [
                g.intensity.clamp(0.0, 1.0),
                cell,
                g.luminance_response.clamp(0.0, 1.0),
                0.0,
            ];
        }
        if hdr && auto_exposure {
            flags |= bits::AUTO_EXPOSURE;
        }
        let g = &s.color_grading;
        let m = if effects && grading::hdr_active(g) {
            flags |= bits::GRADE;
            p.lift_contrast[3] = grading::contrast(g);
            grading::hdr_matrix(g)
        } else {
            p.lift_contrast[3] = 1.0;
            glam::Mat3::IDENTITY
        };
        p.grade_matrix = [m.x_axis, m.y_axis, m.z_axis].map(|c| [c.x, c.y, c.z, 0.0]);
        let inverse_gamma = grading::inverse_gamma(g);
        if effects && grading::ldr_active(g) {
            flags |= bits::GRADE_LDR;
            let lift = if g.lift.is_finite() { g.lift } else { glam::Vec3::ZERO };
            let gain = if g.gain.is_finite() { g.gain } else { glam::Vec3::ONE };
            p.lift_contrast[..3].copy_from_slice(&lift.to_array());
            p.gain_lut[..3].copy_from_slice(&gain.to_array());
            p.gamma_lut[..3].copy_from_slice(&inverse_gamma.to_array());
        } else {
            p.gain_lut[..3].copy_from_slice(&[1.0; 3]);
            p.gamma_lut[..3].copy_from_slice(&[1.0; 3]);
        }
        let mut lut_srgb = 0;
        if let (true, Some((n, srgb)), Some(l)) = (effects, lut, g.lut) {
            flags |= bits::LUT;
            p.gamma_lut[3] = l.contribution.clamp(0.0, 1.0);
            p.gain_lut[3] = n as f32;
            lut_srgb = u32::from(srgb);
        }
        p.flags = [flags, tonemapper.id(), u32::from(!fxaa), lut_srgb];
        p
    }
}

/// What a post group's per-frame seeds depend on besides the frame: whether it applies grain
/// (with the grain's seed and animation) and dither (its `GpuPost` flags).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SeedKey {
    grain: Option<(u32, bool)>,
    dither: bool,
}

impl SeedKey {
    /// The key of a group with effect bits `flags` and film grain `g`.
    pub(crate) fn of(flags: u32, g: &FilmGrain) -> Self {
        Self {
            grain: (flags & bits::GRAIN != 0).then_some((g.seed, g.animated)),
            dither: flags & bits::DITHER != 0,
        }
    }

    /// The group's seeds at `frame` (`post_seeds` in composite.wgsl: grain, dither, second
    /// dither, 0), hashed once here (`s3d_pcg(seed)`), not per pixel: the shader's
    /// `post_hash01(p, pcg(seed))` equals `s3d_hash01(p, seed)` bit for bit. Graphics spec
    /// §4.3's seeds, a pure function of the frame.
    pub(crate) fn seeds(&self, frame: u64) -> [u32; 4] {
        let grain = self
            .grain
            .map_or(0, |(seed, animated)| pcg(grain_seed(seed, animated, frame)));
        let (dither, second) = if self.dither {
            (pcg(dither_seed(frame)), pcg(dither_seed(frame) ^ DITHER_SECOND))
        } else {
            (0, 0)
        };
        [grain, dither, second, 0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Neutral grading and no auto exposure set no bits; each grading part sets its own, with
    /// sanitized values; auto exposure only in Hdr mode; nothing in the passthrough group.
    #[test]
    fn grading_and_exposure_bits() {
        use super::super::api::{ColorGrading, ColorLut, PostProcess};
        let plan = |settings: PostProcess, mode: FrameMode, passthrough: bool| PostGroupPlan {
            camera: space3d_core::World::new().spawn(()),
            settings,
            mode,
            scene_rect: [0, 0, 64, 64],
            output_rect: [0, 0, 64, 64],
            scene_size: (64, 64),
            output_size: (64, 64),
            scene_alpha: true,
            transparent: false,
            passthrough,
            distortion: false,
        };
        let grading_bits = bits::GRADE | bits::GRADE_LDR | bits::LUT | bits::AUTO_EXPOSURE;
        let plain = GpuPost::new(
            &plan(PostProcess::default(), FrameMode::Hdr, false),
            CompositePlan::default(),
        );
        assert_eq!(plain.flags[0] & grading_bits, 0);
        assert_eq!(plain.lift_contrast[3], 1.0);
        assert_eq!(plain.gamma_lut[..3], [1.0; 3]);
        let lut = ColorLut::new(crate::RenderAssets::default().add_texture(ColorLut::identity(16)), 16);
        let graded = PostProcess::default().with_color_grading(ColorGrading {
            saturation: 0.5,
            gamma: glam::Vec3::splat(2.0),
            gain: glam::Vec3::new(f32::NAN, 1.0, 1.0),
            lut: Some(lut),
            ..ColorGrading::default()
        });
        let with = CompositePlan {
            auto_exposure: true,
            lut: Some((16, true)),
            ..CompositePlan::default()
        };
        let p = GpuPost::new(&plan(graded, FrameMode::Hdr, false), with);
        assert_eq!(p.flags[0] & grading_bits, grading_bits);
        assert_eq!((p.gamma_lut, p.gain_lut), ([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 16.0]));
        assert_eq!(p.flags[3], 1, "sRGB strip");
        assert_eq!(p.grade_matrix[0][0], 0.5 + 0.5 * 0.2126);
        let ldr = GpuPost::new(&plan(graded, FrameMode::Ldr, false), with);
        assert_eq!(
            ldr.flags[0] & bits::AUTO_EXPOSURE,
            0,
            "no auto exposure in the Ldr fallback"
        );
        assert_eq!(ldr.flags[0] & bits::GRADE, bits::GRADE, "grading runs there");
        let passthrough = GpuPost::new(&plan(graded, FrameMode::Ldr, true), with);
        assert_eq!(passthrough.flags[0] & grading_bits, 0);
    }

    #[test]
    fn gpu_post_matches_wgsl() {
        let src = super::super::composite_source();
        let module = naga::front::wgsl::parse_str(&src).unwrap();
        let (_, params) = module
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some("PostParams"))
            .unwrap();
        let naga::TypeInner::Struct { members, span } = &params.inner else {
            panic!("PostParams is a struct")
        };
        assert_eq!(*span as usize, std::mem::size_of::<GpuPost>());
        assert_eq!(std::mem::size_of::<GpuPost>(), 256);
        let offset = |name: &str| members.iter().find(|m| m.name.as_deref() == Some(name)).unwrap().offset as usize;
        assert_eq!(offset("grade_matrix"), std::mem::offset_of!(GpuPost, grade_matrix));
        assert_eq!(offset("flags"), std::mem::offset_of!(GpuPost, flags));
        assert_eq!(offset("frame"), std::mem::offset_of!(GpuPost, frame));
        for (name, bit) in [
            ("S3D_POST_BLOOM", bits::BLOOM),
            ("S3D_POST_BLOOM_BILINEAR", bits::BLOOM_BILINEAR),
            ("S3D_POST_AUTO_EXPOSURE", bits::AUTO_EXPOSURE),
            ("S3D_POST_GRADE", bits::GRADE),
            ("S3D_POST_GRADE_LDR", bits::GRADE_LDR),
            ("S3D_POST_LUT", bits::LUT),
            ("S3D_POST_DITHER", bits::DITHER),
            ("S3D_POST_TRANSPARENT", bits::TRANSPARENT),
            ("S3D_POST_SCENE_ALPHA", bits::SCENE_ALPHA),
            ("S3D_POST_EXACT", bits::EXACT),
            ("S3D_POST_DISTORTION", bits::DISTORTION),
            ("S3D_POST_VIGNETTE", bits::VIGNETTE),
            ("S3D_POST_CA", bits::CA),
            ("S3D_POST_GRAIN", bits::GRAIN),
            ("S3D_POST_DISPLAY_GAMMA", bits::DISPLAY_GAMMA),
        ] {
            assert!(src.contains(&format!("const {name}: u32 = {bit}u;")), "{name}");
        }
        crate::pipeline::validate_wgsl(&src).unwrap_or_else(|e| panic!("{e}"));
    }

    /// The display gamma is a uniform bit and `1 / gamma`: at 1 (the default), with NaN, and
    /// in the passthrough group the composite's parameters are exactly those without it.
    #[test]
    fn display_gamma_is_off_at_one() {
        use super::super::api::PostProcess;
        let plan = |settings: PostProcess, passthrough: bool| PostGroupPlan {
            camera: space3d_core::World::new().spawn(()),
            settings,
            mode: FrameMode::Hdr,
            scene_rect: [0, 0, 64, 64],
            output_rect: [0, 0, 64, 64],
            scene_size: (64, 64),
            output_size: (64, 64),
            scene_alpha: true,
            transparent: false,
            passthrough,
            distortion: false,
        };
        let c = || CompositePlan {
            bloom: true,
            fxaa: false,
            auto_exposure: false,
            lut: None,
        };
        let params = |gamma: f32, passthrough: bool| {
            GpuPost::new(
                &plan(PostProcess::default().with_display_gamma(gamma), passthrough),
                c(),
            )
        };
        let plain = GpuPost::new(&plan(PostProcess::default(), false), c());
        assert_eq!(PostProcess::default().display_gamma, 1.0);
        assert_eq!((plain.flags[0] & bits::DISPLAY_GAMMA, plain.exposure[1]), (0, 1.0));
        assert_eq!(params(1.0, false), plain);
        assert_eq!(params(f32::NAN, false), plain);
        assert_eq!(params(f32::INFINITY, false), plain);
        let moved = params(1.25, false);
        assert_eq!(
            (moved.flags[0] & bits::DISPLAY_GAMMA, moved.exposure[1]),
            (bits::DISPLAY_GAMMA, 0.8)
        );
        // Nothing else moves: the exposure, the bloom and the grading are as they were.
        let mut same = moved;
        same.flags[0] &= !bits::DISPLAY_GAMMA;
        same.exposure[1] = 1.0;
        assert_eq!(same, plain);
        // Clamped to the range; the passthrough group copies and gets none.
        assert_eq!(
            (params(100.0, false).exposure[1], params(0.0, false).exposure[1]),
            (0.25, 4.0)
        );
        assert_eq!(params(1.25, true).flags[0] & bits::DISPLAY_GAMMA, 0);
        assert_eq!(params(1.25, true).exposure[1], 1.0);
    }

    /// The shader hashes pre-hashed seeds (`post_hash01(p, pcg(seed))`): the same numbers as
    /// `s3d_hash01(p, seed)`, and the grain seed follows spec §4.3.
    #[test]
    fn seeds_are_prehashed() {
        let hash = |p: [u32; 2], hashed: u32| pcg(p[0] ^ pcg(p[1] ^ hashed)) as f32 * (1.0 / 4_294_967_295.0);
        for (p, seed) in [([0, 0], 0), ([17, 3], 0x9E37_79B9), ([1919, 1079], u32::MAX)] {
            assert_eq!(hash(p, pcg(seed)), crate::output::hash01(p, seed));
        }
        let still = FilmGrain {
            animated: false,
            seed: 42,
            ..FilmGrain::default()
        };
        let seed = |g: &FilmGrain, frame| grain_seed(g.seed, g.animated, frame);
        assert_eq!((seed(&still, 1), seed(&still, 2)), (42, 42));
        let animated = FilmGrain {
            animated: true,
            ..still
        };
        assert_eq!(seed(&animated, 7), pcg(42 ^ 7));
        assert_ne!(seed(&animated, 7), seed(&animated, 8));
    }

    /// The LDR intermediate before FXAA is dithered (it is 8-bit: without the dither it holds
    /// the bands, and the last pass's dither only adds noise to them). A ramp over 8 codes
    /// (1 to 9: at black the clamp keeps the noise's upper half only, as in the last pass),
    /// 1024 px wide: quantized as it is, the mean of a 16-px column block is up to half a code
    /// off; with the composite's dither it is the ramp's own value within 0.1 codes. The noise
    /// is triangular (zero mean, variance 1/6 LSB²) and independent of the last pass's.
    #[test]
    fn intermediate_dither_removes_the_bands() {
        const W: u32 = 1024;
        const H: u32 = 64;
        let value = |x: u32| 1.0 + 8.0 * (x as f32 + 0.5) / W as f32;
        let worst = |dither: bool| {
            let mut worst = 0.0f32;
            for block in 0..W / 16 {
                let (mut sum, mut want) = (0.0f64, 0.0f64);
                for x in block * 16..block * 16 + 16 {
                    for y in 0..H {
                        let n = if dither { dither_noise([x, y], 3, true) } else { 0.0 };
                        sum += f64::from((value(x) + n).round().clamp(0.0, 255.0));
                        want += f64::from(value(x));
                    }
                }
                worst = worst.max(((sum - want) / f64::from(16 * H)).abs() as f32);
            }
            worst
        };
        let (banded, dithered) = (worst(false), worst(true));
        eprintln!("worst 16-px block mean error: {banded:.3} LSB quantized, {dithered:.3} LSB dithered");
        assert!(banded > 0.4, "{banded}");
        assert!(dithered < 0.1, "{dithered}");
        let (mut sum, mut sq, mut cross) = (0.0f64, 0.0f64, 0.0f64);
        let n = f64::from(W * H);
        for x in 0..W {
            for y in 0..H {
                let a = f64::from(dither_noise([x, y], 3, true));
                let b = f64::from(dither_noise([x, y], 3, false));
                assert!(a.abs() <= 1.0);
                sum += a;
                sq += a * a;
                cross += a * b;
            }
        }
        assert!((sum / n).abs() < 0.01, "mean {}", sum / n);
        assert!((sq / n - 1.0 / 6.0).abs() < 0.01, "variance {}", sq / n);
        assert!((cross / n).abs() < 0.01, "correlation with the last pass {}", cross / n);
        let wgsl = super::super::COMPOSITE_WGSL;
        assert!(wgsl.contains(&format!("S3D_POST_DITHER_L: u32 = 0x{DITHER_INTERMEDIATE:X}u")));
    }

    /// A group's window seeds are the spec's per-frame seeds (§4.3), pre-hashed, for any frame.
    #[test]
    fn seed_keys_give_the_spec_seeds() {
        let grain = FilmGrain {
            animated: true,
            seed: 9,
            ..FilmGrain::default()
        };
        let all = SeedKey::of(bits::GRAIN | bits::DITHER, &grain);
        for frame in [0, 1, 63, 64, 1 << 33] {
            assert_eq!(
                all.seeds(frame),
                [
                    pcg(grain_seed(9, true, frame)),
                    pcg(dither_seed(frame)),
                    pcg(dither_seed(frame) ^ DITHER_SECOND),
                    0
                ]
            );
        }
        let still = FilmGrain {
            animated: false,
            ..grain
        };
        assert_eq!(SeedKey::of(bits::GRAIN, &still).seeds(5), [pcg(9), 0, 0, 0]);
        assert_eq!(SeedKey::of(bits::BLOOM, &grain).seeds(5), [0; 4], "no grain, no dither");
        assert_ne!(SeedKey::of(bits::GRAIN, &grain), SeedKey::of(bits::GRAIN, &still));
    }

    #[test]
    fn dither_seed_is_the_spec_hash() {
        assert_eq!(dither_seed(0), pcg(0x9E37_79B9));
        assert_ne!(dither_seed(1), dither_seed(2));
    }
}
