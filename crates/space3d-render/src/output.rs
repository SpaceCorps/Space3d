//! Output encodings and tonemapping: [`OutputEncoding`], [`Tonemapper`] (one enum for the whole
//! engine, with CPU mirrors of the shader curves) and the [`OUTPUT_WGSL`] module every shader
//! drawing into a frame pass uses to encode its color (graphics spec §2.2.3, Appendix B).

use glam::{Mat3, Vec3};

/// How a fragment shader drawing into a pass must encode its color (see
/// [`crate::PassTarget::output`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OutputEncoding {
    /// Linear, unclamped (float targets; sRGB-format targets encode in hardware).
    #[default]
    Linear,
    /// Clamp to 0..1 and encode sRGB in the shader (8-bit non-sRGB targets without an sRGB
    /// view).
    ManualSrgb,
    /// LDR fallback of a post camera: exposure × tonemapper in the shader, then linear (sRGB
    /// target).
    Tonemapped,
}

impl OutputEncoding {
    /// Value of the `S3D_OUTPUT` pipeline override and of `s3d_output`'s `encoding`: 0, 1, 2.
    pub const fn id(self) -> u32 {
        match self {
            Self::Linear => 0,
            Self::ManualSrgb => 1,
            Self::Tonemapped => 2,
        }
    }
}

/// HDR → display mapping (one enum for the whole engine: post, the Ldr fallback, particles and
/// overlays).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Tonemapper {
    /// Clamp to 0..1.
    None,
    /// Luminance Reinhard `c / (1 + L)`: hue-preserving, never white.
    Reinhard,
    /// Stephen Hill's ACES fit.
    AcesFitted,
    /// Sobotka AgX (minimal fit, no look): saturated lights go to white without hue skew.
    #[default]
    AgX,
    /// Khronos PBR Neutral: faithful base colors (UI and product previews) with a highlight
    /// roll-off from 0.76. Its **toe** subtracts `x − 6.25·x²` from every channel of a color
    /// whose weakest channel `x` is below 0.08 (0.04 above): nearly all of a dim, saturated
    /// color's weakest channel — sRGB (21, 47, 65) becomes (1, 40, 60), a teal nebula a blue
    /// primary. For dim saturated art use [`Tonemapper::KhronosPbrNeutralNoToe`].
    KhronosPbrNeutral,
    /// AgX with Sobotka's "punchy" look (power 1.35, saturation 1.4 in AgX's log space):
    /// saturated emissives and effects keep their hue (a red laser stays red, blue exhaust
    /// blue) at the cost of darker mid tones (0.18 → 0.12 display-linear, 0.21 with `AgX`).
    /// A game-look choice for effect-heavy scenes; `AgX` stays the neutral default. The look
    /// also pushes **dim** saturated colors toward their strongest channel: a dim magenta
    /// nebula, linear (0.84, 0.10, 0.40) × 0.04 (R:B 2.1), comes out at +0.5 EV as sRGB
    /// (54, 0, 21) (R:B 4.8, green crushed). For dim saturated backdrops prefer `AgX` with
    /// [`crate::ColorGrading::vivid`] (docs/graphics/frame.md, "Choose the look explicitly").
    AgXPunchy,
    /// Khronos PBR Neutral without its toe: colors pass **unchanged** below 0.8 (where Neutral
    /// starts compressing, 0.76, plus the 0.04 the toe subtracts) and the highlights roll off
    /// to white like Neutral's. Display-authored art (skies, paint, UI previews) keeps its
    /// exact values, dim saturated colors included (sRGB (21, 47, 65) stays (21, 47, 65)),
    /// with no full-frame lift to hide Neutral's toe.
    KhronosPbrNeutralNoToe,
}

impl Tonemapper {
    /// Operator id of `s3d_tonemap` (`GpuView.output.y`): 0 None, 1 Reinhard, 2 AcesFitted,
    /// 3 AgX, 4 KhronosPbrNeutral, 5 AgXPunchy, 6 KhronosPbrNeutralNoToe.
    pub const fn id(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Reinhard => 1,
            Self::AcesFitted => 2,
            Self::AgX => 3,
            Self::KhronosPbrNeutral => 4,
            Self::AgXPunchy => 5,
            Self::KhronosPbrNeutralNoToe => 6,
        }
    }

    /// CPU mirror of `s3d_tonemap` (linear in → display-linear 0..1 out). Used by tests, by the
    /// Ldr clear-color path and by UI previews.
    pub fn apply(self, linear: Vec3) -> Vec3 {
        let c = linear.max(Vec3::ZERO);
        match self {
            Self::None => c.clamp(Vec3::ZERO, Vec3::ONE),
            Self::Reinhard => (c / (1.0 + luma(c))).clamp(Vec3::ZERO, Vec3::ONE),
            Self::AcesFitted => aces_fitted(c),
            Self::AgX => agx_display(agx_curve(c)),
            Self::KhronosPbrNeutral => pbr_neutral(c),
            Self::AgXPunchy => agx_punchy(c),
            Self::KhronosPbrNeutralNoToe => pbr_neutral_no_toe(c),
        }
    }
}

fn luma(c: Vec3) -> f32 {
    c.dot(Vec3::new(0.2126, 0.7152, 0.0722))
}

/// WGSL `v * M` (row vector times matrix, `M` given by columns).
fn row_mul(v: Vec3, m: Mat3) -> Vec3 {
    Vec3::new(v.dot(m.x_axis), v.dot(m.y_axis), v.dot(m.z_axis))
}

fn aces_fitted(c: Vec3) -> Vec3 {
    let i = Mat3::from_cols(
        Vec3::new(0.59719, 0.35458, 0.04823),
        Vec3::new(0.07600, 0.90834, 0.01566),
        Vec3::new(0.02840, 0.13383, 0.83777),
    );
    let o = Mat3::from_cols(
        Vec3::new(1.60475, -0.53108, -0.07367),
        Vec3::new(-0.10208, 1.10813, -0.00605),
        Vec3::new(-0.00327, -0.07276, 1.07602),
    );
    let v = row_mul(c, i);
    let r = (v * (v + 0.0245786) - 0.000_090_537) / (v * (0.983729 * v + 0.432_951) + 0.238081);
    row_mul(r, o).clamp(Vec3::ZERO, Vec3::ONE)
}

// The literals mirror `s3d_agx_curve` / `s3d_agx_display` in output.wgsl digit for digit.
#[allow(clippy::excessive_precision)]
fn agx_curve(c: Vec3) -> Vec3 {
    let m = Mat3::from_cols(
        Vec3::new(0.842479062253094, 0.0423282422610123, 0.0423756549057051),
        Vec3::new(0.0784335999999992, 0.878468636469772, 0.0784336),
        Vec3::new(0.0792237451477643, 0.0791661274605434, 0.879142973793104),
    );
    let (lo, hi) = (-12.47393f32, 4.026069f32);
    let log = (m * c).max(Vec3::splat(1e-10));
    let mut v = Vec3::new(log.x.log2(), log.y.log2(), log.z.log2()).clamp(Vec3::splat(lo), Vec3::splat(hi));
    v = (v - lo) / (hi - lo);
    let v2 = v * v;
    let v4 = v2 * v2;
    15.5 * v4 * v2 - 40.14 * v4 * v + 31.96 * v4 - 6.868 * v2 * v + 0.4298 * v2 + 0.1191 * v - 0.00232
}

#[allow(clippy::excessive_precision)]
fn agx_display(v: Vec3) -> Vec3 {
    let mi = Mat3::from_cols(
        Vec3::new(1.19687900512017, -0.0528968517574562, -0.0529716355144438),
        Vec3::new(-0.0980208811401368, 1.15190312990417, -0.0980434501171241),
        Vec3::new(-0.0990297440797205, -0.0989611768448433, 1.15107367264116),
    );
    let d = (mi * v).max(Vec3::ZERO);
    Vec3::new(d.x.powf(2.2), d.y.powf(2.2), d.z.powf(2.2))
}

/// `s3d_agx_punchy`: the ASC CDL look (power 1.35, saturation 1.4) between curve and display.
fn agx_punchy(c: Vec3) -> Vec3 {
    let v = agx_curve(c).max(Vec3::ZERO);
    let v = Vec3::new(v.x.powf(1.35), v.y.powf(1.35), v.z.powf(1.35));
    let l = luma(v);
    agx_display(Vec3::splat(l) + 1.4 * (v - Vec3::splat(l))).min(Vec3::ONE)
}

fn pbr_neutral(c_in: Vec3) -> Vec3 {
    let mut c = c_in;
    let x = c.min_element();
    let offset = if x < 0.08 { x - 6.25 * x * x } else { 0.04 };
    c -= offset;
    let peak = c.max_element();
    if peak < 0.76 {
        return c;
    }
    let d = 1.0 - 0.76;
    let new_peak = 1.0 - d * d / (peak + d - 0.76);
    c *= new_peak / peak;
    let g = 1.0 - 1.0 / (0.15 * (peak - new_peak) + 1.0);
    c.lerp(Vec3::splat(new_peak), g)
}

/// `s3d_pbr_neutral_no_toe`: Neutral's roll-off, starting at 0.8, with no toe.
fn pbr_neutral_no_toe(c_in: Vec3) -> Vec3 {
    let mut c = c_in;
    let start = 0.8;
    let peak = c.max_element();
    if peak < start {
        return c;
    }
    let d = 1.0 - start;
    let new_peak = 1.0 - d * d / (peak + d - start);
    c *= new_peak / peak;
    let g = 1.0 - 1.0 / (0.15 * (peak - new_peak) + 1.0);
    c.lerp(Vec3::splat(new_peak), g)
}

/// [`crate::PostProcess::display_gamma`] as the composite uses it: inside
/// [`crate::DISPLAY_GAMMA_RANGE`], 1 for NaN and infinities.
pub(crate) fn display_gamma_of(gamma: f32) -> f32 {
    let (lo, hi) = crate::DISPLAY_GAMMA_RANGE;
    if gamma.is_finite() { gamma.clamp(lo, hi) } else { 1.0 }
}

/// CPU mirror of the composite's display gamma ([`crate::PostProcess::display_gamma`],
/// `post_display_gamma` in composite.wgsl): the sRGB-encoded value `encoded` (0..1) as it is
/// displayed at `gamma`, `encoded^(1 / gamma)`. 0 and 1 stay; at 1 it is `encoded` itself. For
/// UI that previews the setting (a calibration strip) and for tests.
pub fn apply_display_gamma(encoded: f32, gamma: f32) -> f32 {
    let e = encoded.clamp(0.0, 1.0);
    let gamma = display_gamma_of(gamma);
    if gamma == 1.0 { e } else { e.powf(1.0 / gamma) }
}

/// PCG hash (`s3d_pcg` in `shaders/common/hash.wgsl`, bit-identical): the seed of every noise
/// (dither, grain, particles) — see graphics spec §4.3.
pub(crate) fn pcg(v: u32) -> u32 {
    let s = v.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let w = ((s >> ((s >> 28) + 4)) ^ s).wrapping_mul(277_803_737);
    (w >> 22) ^ w
}

/// Uniform 0..1 per pixel and seed (`s3d_hash01`).
#[allow(
    dead_code,
    reason = "graphics contract (spec §2.6): wired by the F4–F8 frame milestones"
)]
pub(crate) fn hash01(p: [u32; 2], seed: u32) -> f32 {
    pcg(p[0] ^ pcg(p[1] ^ pcg(seed))) as f32 * (1.0 / 4_294_967_295.0)
}

/// WGSL without bindings: `S3D_OUTPUT_LINEAR` / `S3D_OUTPUT_MANUAL_SRGB` /
/// `S3D_OUTPUT_TONEMAPPED`, `S3D_TM_*`, `s3d_linear_to_srgb`, `s3d_srgb_to_linear`,
/// `s3d_luma`, `s3d_tonemap(op: u32, c: vec3<f32>) -> vec3<f32>` and
/// `s3d_output(c: vec4<f32>, encoding: u32, exposure: f32, op: u32) -> vec4<f32>`.
pub const OUTPUT_WGSL: &str = include_str!("shaders/common/output.wgsl");

/// `s3d_pcg`, `s3d_hash01` (crate-internal; post and particles copy them).
pub(crate) const HASH_WGSL: &str = include_str!("shaders/common/hash.wgsl");

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [Tonemapper; 7] = [
        Tonemapper::None,
        Tonemapper::Reinhard,
        Tonemapper::AcesFitted,
        Tonemapper::AgX,
        Tonemapper::KhronosPbrNeutral,
        Tonemapper::AgXPunchy,
        Tonemapper::KhronosPbrNeutralNoToe,
    ];

    #[test]
    fn ids_are_the_shader_constants() {
        assert_eq!(ALL.map(Tonemapper::id), [0, 1, 2, 3, 4, 5, 6]);
        for (t, name) in [
            (Tonemapper::AgX, "S3D_TM_AGX: u32 = 3u"),
            (Tonemapper::KhronosPbrNeutral, "S3D_TM_PBR_NEUTRAL: u32 = 4u"),
            (Tonemapper::AgXPunchy, "S3D_TM_AGX_PUNCHY: u32 = 5u"),
            (
                Tonemapper::KhronosPbrNeutralNoToe,
                "S3D_TM_PBR_NEUTRAL_NO_TOE: u32 = 6u",
            ),
        ] {
            assert!(OUTPUT_WGSL.contains(name), "{t:?}: {name}");
        }
        assert_eq!(
            [
                OutputEncoding::Linear,
                OutputEncoding::ManualSrgb,
                OutputEncoding::Tonemapped
            ]
            .map(OutputEncoding::id),
            [0, 1, 2]
        );
        assert_eq!(Tonemapper::default(), Tonemapper::AgX);
        for (name, id) in [
            ("S3D_OUTPUT_LINEAR: u32 = 0u", 0),
            ("S3D_OUTPUT_MANUAL_SRGB: u32 = 1u", 1),
            ("S3D_OUTPUT_TONEMAPPED: u32 = 2u", 2),
        ] {
            assert!(OUTPUT_WGSL.contains(name), "{name} ({id})");
        }
    }

    #[test]
    fn tonemappers_map_into_display_range() {
        for t in ALL {
            assert!(t.apply(Vec3::ZERO).max_element() < 0.01, "{t:?}");
            for x in [0.05f32, 0.18, 0.5, 1.0, 2.0, 8.0, 64.0] {
                let y = t.apply(Vec3::splat(x));
                assert!(
                    y.min_element() >= 0.0 && y.max_element() <= 1.0 + 1e-4,
                    "{t:?}({x}) = {y}"
                );
            }
            // Negative input is clamped first.
            assert_eq!(t.apply(Vec3::splat(-1.0)), t.apply(Vec3::ZERO));
        }
        assert_eq!(
            Tonemapper::None.apply(Vec3::new(0.5, 2.0, 0.0)),
            Vec3::new(0.5, 1.0, 0.0)
        );
        let r = Tonemapper::Reinhard.apply(Vec3::ONE);
        assert!((r.x - 0.5).abs() < 1e-6, "{r}");
        // PBR Neutral keeps base colors below its compression start (0.76 − the toe offset).
        let n = Tonemapper::KhronosPbrNeutral.apply(Vec3::new(0.5, 0.3, 0.2));
        assert!((n - Vec3::new(0.46, 0.26, 0.16)).abs().max_element() < 1e-5, "{n}");
        // AgX sends a saturated bright light towards white.
        let a = Tonemapper::AgX.apply(Vec3::new(64.0, 4.0, 1.0));
        assert!(a.min_element() > 0.5, "{a}");
        // The punchy look keeps saturated emissives saturated (a 2.5× red laser: AgX's pale
        // pink (234, 187, 179) becomes (246, 162, 149)) and darkens mid grey.
        let to_u8 = |c: Vec3| c.to_array().map(|x| (crate::linear_to_srgb(x) * 255.0).round() as u8);
        let red = Vec3::new(2.5, 0.625, 0.5);
        assert_eq!(to_u8(Tonemapper::AgX.apply(red)), [234, 187, 179]);
        assert_eq!(to_u8(Tonemapper::AgXPunchy.apply(red)), [246, 162, 149]);
        let hot = Tonemapper::AgXPunchy.apply(Vec3::new(16.0, 7.2, 1.92));
        assert!(hot.max_element() <= 1.0 && hot.min_element() > 0.5, "{hot}");
        let grey = |t: Tonemapper| t.apply(Vec3::splat(0.18)).x;
        assert!((grey(Tonemapper::AgX) - 0.21).abs() < 0.01 && (grey(Tonemapper::AgXPunchy) - 0.125).abs() < 0.01);
    }

    /// game#9 and game#6: the dim, saturated colors of the reported captures through each
    /// look (sRGB in → sRGB out). Neutral's toe crushes the weakest channel; the toe-less
    /// Neutral passes them unchanged; AgX Punchy at +0.5 EV skews them toward the strongest
    /// channel, which plain AgX with the `vivid` grading does far less.
    #[test]
    fn dim_saturated_colors() {
        let to_u8 = |c: Vec3| c.to_array().map(|x| (crate::linear_to_srgb(x) * 255.0).round() as u8);
        let lin = |s: [u8; 3]| Vec3::from_array(s.map(|v| crate::srgb_to_linear(v as f32 / 255.0)));
        let teal = lin([21, 47, 65]);
        assert_eq!(to_u8(Tonemapper::KhronosPbrNeutral.apply(teal)), [1, 40, 60]);
        assert_eq!(to_u8(Tonemapper::KhronosPbrNeutralNoToe.apply(teal)), [21, 47, 65]);
        // Unchanged below 0.8, rolled off above like Neutral (never above 1, white at 64).
        for x in [0.0f32, 0.02, 0.18, 0.5, 0.79] {
            let c = Vec3::new(x, 0.5 * x, 0.25 * x);
            assert_eq!(Tonemapper::KhronosPbrNeutralNoToe.apply(c), c);
        }
        let hot = Tonemapper::KhronosPbrNeutralNoToe.apply(Vec3::new(64.0, 8.0, 1.0));
        assert!(hot.min_element() > 0.8 && hot.max_element() <= 1.0, "{hot}");
        // The M-1 magenta at 4 % (R:B 2.1), a dim teal and a red laser, at +0.5 EV (display
        // sRGB): AgX Punchy crushes the magenta's green and the teal's red; AgX keeps them
        // (milky); AgX with `ColorGrading::vivid` keeps them and is more saturated.
        let e = 0.5f32.exp2();
        let magenta = Vec3::new(0.84, 0.10, 0.40) * 0.04 * e;
        let laser = Vec3::new(2.5, 0.625, 0.5) * e;
        let vivid = crate::ColorGrading::vivid();
        let look = |t: Tonemapper, g: bool, c: Vec3| to_u8(t.apply(if g { vivid.grade_hdr(c) } else { c }));
        let sat = |c: [u8; 3]| {
            let (lo, hi) = (
                c.iter().min().copied().unwrap_or(0),
                c.iter().max().copied().unwrap_or(0),
            );
            (hi - lo) as f32 / hi.max(1) as f32
        };
        let (punchy, plain, graded) = (
            look(Tonemapper::AgXPunchy, false, magenta),
            look(Tonemapper::AgX, false, magenta),
            look(Tonemapper::AgX, true, magenta),
        );
        eprintln!("magenta: punchy {punchy:?}, AgX {plain:?}, AgX + vivid {graded:?}");
        assert_eq!(punchy, [54, 0, 21]);
        assert_eq!(plain, [70, 10, 41]);
        assert_eq!(graded, [78, 8, 47]);
        assert_eq!(look(Tonemapper::AgXPunchy, false, teal * e), [1, 36, 69]);
        assert_eq!(look(Tonemapper::AgX, true, teal * e), [19, 66, 95]);
        // The hue (R:B) stays within 5 % of plain AgX's; the saturation rises.
        let rb = |c: [u8; 3]| c[0] as f32 / c[2] as f32;
        assert!((rb(graded) / rb(plain) - 1.0).abs() < 0.05, "{graded:?} vs {plain:?}");
        assert!(sat(graded) > sat(plain));
        assert!(sat(look(Tonemapper::AgX, true, laser)) > sat(look(Tonemapper::AgX, false, laser)));
    }

    /// The display gamma's golden values: the encoded value to the power 1 / gamma. 0 and 1 are
    /// fixed points, 1 is the identity bit for bit, and out-of-range settings are clamped.
    #[test]
    fn display_gamma_golden_values() {
        for code in 0..=255u8 {
            let e = f32::from(code) / 255.0;
            assert_eq!(apply_display_gamma(e, 1.0).to_bits(), e.to_bits(), "{code}");
            assert!(
                apply_display_gamma(e, 1.3) >= e && apply_display_gamma(e, 0.8) <= e,
                "{code}"
            );
        }
        assert!((apply_display_gamma(0.5, 1.2) - 0.561_231).abs() < 1e-6);
        assert!((apply_display_gamma(0.5, 0.8) - 0.420_448).abs() < 1e-6);
        assert!((apply_display_gamma(0.5, 2.0) - 0.5f32.sqrt()).abs() < 1e-6);
        for g in [0.25, 0.7, 1.0, 1.5, 4.0] {
            assert_eq!(
                (apply_display_gamma(0.0, g), apply_display_gamma(1.0, g)),
                (0.0, 1.0),
                "{g}"
            );
        }
        let code = |c: u8, g: f32| (apply_display_gamma(f32::from(c) / 255.0, g) * 255.0).round() as u8;
        // Dark codes (a calibration strip's 4, 8 and 16) and middle grey's 118.
        assert_eq!([4, 8, 16, 118].map(|c| code(c, 0.7)), [1, 2, 5, 85]);
        assert_eq!([4, 8, 16, 118].map(|c| code(c, 1.3)), [10, 18, 30, 141]);
        assert_eq!([4, 8, 16, 118].map(|c| code(c, 1.5)), [16, 25, 40, 153]);
        // Clamped to the range; NaN and infinities are 1; the input is clamped to 0..1.
        let (lo, hi) = crate::DISPLAY_GAMMA_RANGE;
        assert_eq!(apply_display_gamma(0.5, 100.0), apply_display_gamma(0.5, hi));
        assert_eq!(apply_display_gamma(0.5, 0.0), apply_display_gamma(0.5, lo));
        assert_eq!(
            (
                apply_display_gamma(0.5, f32::NAN),
                apply_display_gamma(0.5, f32::INFINITY)
            ),
            (0.5, 0.5)
        );
        assert_eq!(
            (apply_display_gamma(-1.0, 1.3), apply_display_gamma(2.0, 1.3)),
            (0.0, 1.0)
        );
    }

    #[test]
    fn hashes_are_stable() {
        assert_eq!(pcg(0), 129_708_002);
        assert_ne!(pcg(1), pcg(2));
        let h = hash01([3, 4], 0x9E37_79B9);
        assert!((0.0..=1.0).contains(&h));
        assert_eq!(h, hash01([3, 4], 0x9E37_79B9));
    }

    #[test]
    fn wgsl_validates() {
        crate::pipeline::validate_wgsl(OUTPUT_WGSL).unwrap_or_else(|e| panic!("{e}"));
        crate::pipeline::validate_wgsl(HASH_WGSL).unwrap_or_else(|e| panic!("{e}"));
    }
}
