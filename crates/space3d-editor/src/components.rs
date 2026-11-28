//! Editor-side component support: scene-registry registrations for render components,
//! [`MeshSource`] (re-creatable meshes), [`EditorInternal`] and generic JSON access to
//! registered components.

use serde::{Deserialize, Serialize};
use space3d_core::hecs::{Entity, World};
use space3d_core::{MeshData, Name, Parent, Transform};
use space3d_render::{
    Aabb, AntiAliasing, AutoExposure, BlendMode, Bloom, BloomComposite, Camera3d, CameraFrame, ChromaticAberration,
    ColorGrading, DirectionalLight, DirectionalShadow, Distortion, DistortionResolution, EnvironmentLight, FilmGrain,
    FxaaQuality, HdrMode, Material, Mesh3d, Metering, PointLight, PostProcess, Projection, RenderTarget, ShadowFilter,
    ShadowFit, ShadowMode, Skybox, Tonemapper, Vignette, Visibility,
};
use space3d_scene::{ComponentRegistry, ComponentValue, Scene, SceneEntity, StableId, save_entities};

/// Marker for entities owned by the editor itself (model preview camera, ...): hidden from
/// the hierarchy and never saved.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EditorInternal;

/// How to (re)create an entity's mesh. Saved in scenes (as `"MeshSource"`), so meshes survive
/// save/load; the editor turns it into a [`Mesh3d`] when the entity has none.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum MeshSource {
    /// `MeshData::cube(size)`.
    Cube {
        /// Edge length.
        size: f32,
    },
    /// `MeshData::plane(size)`.
    Plane {
        /// Edge length.
        size: f32,
    },
    /// `MeshData::sphere(radius, sectors, stacks)`.
    Sphere {
        /// Radius.
        radius: f32,
        /// Longitudinal segments.
        sectors: u32,
        /// Latitudinal segments.
        stacks: u32,
    },
    /// Inline mesh data (e.g. from the model editor).
    Data(MeshData),
}

impl MeshSource {
    /// Build the mesh.
    pub fn mesh(&self) -> MeshData {
        match self {
            MeshSource::Cube { size } => MeshData::cube(*size),
            MeshSource::Plane { size } => MeshData::plane(*size),
            MeshSource::Sphere {
                radius,
                sectors,
                stacks,
            } => MeshData::sphere(*radius, *sectors, *stacks),
            MeshSource::Data(m) => m.clone(),
        }
    }

    /// Short label.
    pub fn label(&self) -> &'static str {
        match self {
            MeshSource::Cube { .. } => "cube",
            MeshSource::Plane { .. } => "plane",
            MeshSource::Sphere { .. } => "sphere",
            MeshSource::Data(_) => "mesh data",
        }
    }
}

#[derive(Serialize, Deserialize)]
struct MaterialRepr {
    base_color: [f32; 4],
    metallic: f32,
    roughness: f32,
    emissive: [f32; 3],
    #[serde(default)]
    unlit: bool,
    #[serde(default)]
    double_sided: bool,
    #[serde(default)]
    alpha_blend: bool,
    // Graphics upgrade (scalars only: texture and shader handles are session-local).
    #[serde(default)]
    blend: BlendRepr,
    #[serde(default = "half")]
    alpha_cutoff: f32,
    #[serde(default = "one")]
    normal_scale: f32,
    #[serde(default = "one")]
    occlusion_strength: f32,
    #[serde(default)]
    shadows: ShadowModeRepr,
    #[serde(default)]
    render_order: i16,
}

fn half() -> f32 {
    0.5
}

fn one() -> f32 {
    1.0
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
enum BlendRepr {
    #[default]
    Opaque,
    AlphaMask,
    AlphaBlend,
    Premultiplied,
    Additive,
}

impl BlendRepr {
    fn of(b: BlendMode) -> Self {
        match b {
            BlendMode::AlphaMask => Self::AlphaMask,
            BlendMode::AlphaBlend => Self::AlphaBlend,
            BlendMode::Premultiplied => Self::Premultiplied,
            BlendMode::Additive => Self::Additive,
            _ => Self::Opaque,
        }
    }
    fn mode(self) -> BlendMode {
        match self {
            Self::Opaque => BlendMode::Opaque,
            Self::AlphaMask => BlendMode::AlphaMask,
            Self::AlphaBlend => BlendMode::AlphaBlend,
            Self::Premultiplied => BlendMode::Premultiplied,
            Self::Additive => BlendMode::Additive,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
enum ShadowModeRepr {
    #[default]
    Auto,
    CastAndReceive,
    CastOnly,
    ReceiveOnly,
    Off,
}

impl ShadowModeRepr {
    fn of(m: ShadowMode) -> Self {
        match m {
            ShadowMode::CastAndReceive => Self::CastAndReceive,
            ShadowMode::CastOnly => Self::CastOnly,
            ShadowMode::ReceiveOnly => Self::ReceiveOnly,
            ShadowMode::None => Self::Off,
            _ => Self::Auto,
        }
    }
    fn mode(self) -> ShadowMode {
        match self {
            Self::Auto => ShadowMode::Auto,
            Self::CastAndReceive => ShadowMode::CastAndReceive,
            Self::CastOnly => ShadowMode::CastOnly,
            Self::ReceiveOnly => ShadowMode::ReceiveOnly,
            Self::Off => ShadowMode::None,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct DirectionalLightRepr {
    color: [f32; 3],
    intensity: f32,
}

#[derive(Serialize, Deserialize)]
struct PointLightRepr {
    color: [f32; 3],
    intensity: f32,
    range: f32,
}

#[derive(Serialize, Deserialize)]
enum ProjectionRepr {
    Perspective { fov_y_degrees: f32, near: f32, far: f32 },
    Orthographic { height: f32, near: f32, far: f32 },
}

#[derive(Serialize, Deserialize)]
struct CameraRepr {
    projection: ProjectionRepr,
    clear_color: Option<[f32; 4]>,
    #[serde(default)]
    order: i32,
    #[serde(default)]
    viewport: Option<[f32; 4]>,
    active: bool,
}

/// A string name for each variant of a `#[non_exhaustive]` render enum: `(value, name)` pairs
/// searched both ways; unknown variants (added later) save as the first entry.
fn name_of<T: PartialEq + Copy>(table: &[(T, &'static str)], v: T) -> String {
    table.iter().find(|(t, _)| *t == v).unwrap_or(&table[0]).1.to_owned()
}

fn value_of<T: Copy>(table: &[(T, &'static str)], name: &str) -> T {
    table.iter().find(|(_, n)| *n == name).unwrap_or(&table[0]).0
}

/// `HdrMode` names.
pub(crate) const HDR_MODES: [(HdrMode, &str); 3] = [
    (HdrMode::Auto, "Auto"),
    (HdrMode::Ldr, "Ldr"),
    (HdrMode::Compact, "Compact"),
];

/// `Tonemapper` names.
pub(crate) const TONEMAPPERS: [(Tonemapper, &str); 7] = [
    (Tonemapper::AgX, "AgX"),
    (Tonemapper::AgXPunchy, "AgXPunchy"),
    (Tonemapper::AcesFitted, "AcesFitted"),
    (Tonemapper::KhronosPbrNeutral, "KhronosPbrNeutral"),
    (Tonemapper::KhronosPbrNeutralNoToe, "KhronosPbrNeutralNoToe"),
    (Tonemapper::Reinhard, "Reinhard"),
    (Tonemapper::None, "None"),
];

/// `AntiAliasing` names.
pub(crate) const ANTI_ALIASING: [(AntiAliasing, &str); 5] = [
    (AntiAliasing::None, "None"),
    (AntiAliasing::Fxaa(FxaaQuality::Low), "FxaaLow"),
    (AntiAliasing::Fxaa(FxaaQuality::Medium), "FxaaMedium"),
    (AntiAliasing::Fxaa(FxaaQuality::High), "FxaaHigh"),
    (AntiAliasing::Fxaa(FxaaQuality::Ultra), "FxaaUltra"),
];

/// `ShadowFilter` names.
pub(crate) const SHADOW_FILTERS: [(ShadowFilter, &str); 4] = [
    (ShadowFilter::Pcf9, "Pcf9"),
    (ShadowFilter::Hard, "Hard"),
    (ShadowFilter::Pcf4, "Pcf4"),
    (ShadowFilter::Pcf16, "Pcf16"),
];

const METERING: [(Metering, &str); 2] = [
    (Metering::CenterWeighted, "CenterWeighted"),
    (Metering::Average, "Average"),
];

const BLOOM_COMPOSITE: [(BloomComposite, &str); 2] = [
    (BloomComposite::Additive, "Additive"),
    (BloomComposite::EnergyConserving, "EnergyConserving"),
];

const DISTORTION_RESOLUTION: [(DistortionResolution, &str); 2] = [
    (DistortionResolution::Half, "Half"),
    (DistortionResolution::Full, "Full"),
];

#[derive(Serialize, Deserialize)]
#[serde(default)]
struct CameraFrameRepr {
    hdr: String,
    render_scale: Option<f32>,
}

impl Default for CameraFrameRepr {
    fn default() -> Self {
        Self::of(&CameraFrame::default())
    }
}

impl CameraFrameRepr {
    fn of(c: &CameraFrame) -> Self {
        Self {
            hdr: name_of(&HDR_MODES, c.hdr),
            render_scale: c.render_scale,
        }
    }
    fn frame(&self) -> CameraFrame {
        CameraFrame {
            hdr: value_of(&HDR_MODES, &self.hdr),
            render_scale: self.render_scale,
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(default)]
struct AutoExposureRepr {
    enabled: bool,
    min_ev: f32,
    max_ev: f32,
    adapt_to_bright: f32,
    adapt_to_dark: f32,
    low_percentile: f32,
    high_percentile: f32,
    metering: String,
}

impl Default for AutoExposureRepr {
    fn default() -> Self {
        Self::of(&AutoExposure::default())
    }
}

impl AutoExposureRepr {
    fn of(a: &AutoExposure) -> Self {
        Self {
            enabled: a.enabled,
            min_ev: a.min_ev,
            max_ev: a.max_ev,
            adapt_to_bright: a.adapt_to_bright,
            adapt_to_dark: a.adapt_to_dark,
            low_percentile: a.low_percentile,
            high_percentile: a.high_percentile,
            metering: name_of(&METERING, a.metering),
        }
    }
    fn value(&self) -> AutoExposure {
        AutoExposure {
            enabled: self.enabled,
            min_ev: self.min_ev,
            max_ev: self.max_ev,
            adapt_to_bright: self.adapt_to_bright,
            adapt_to_dark: self.adapt_to_dark,
            low_percentile: self.low_percentile,
            high_percentile: self.high_percentile,
            metering: value_of(&METERING, &self.metering),
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(default)]
struct BloomRepr {
    enabled: bool,
    intensity: f32,
    threshold: f32,
    soft_knee: f32,
    scatter: f32,
    composite: String,
    mip_count: u32,
    max_mip_height: u32,
    min_rect_height: u32,
    fast: bool,
}

impl Default for BloomRepr {
    fn default() -> Self {
        Self::of(&Bloom::default())
    }
}

impl BloomRepr {
    fn of(b: &Bloom) -> Self {
        Self {
            enabled: b.enabled,
            intensity: b.intensity,
            threshold: b.threshold,
            soft_knee: b.soft_knee,
            scatter: b.scatter,
            composite: name_of(&BLOOM_COMPOSITE, b.composite),
            mip_count: b.mip_count,
            max_mip_height: b.max_mip_height,
            min_rect_height: b.min_rect_height,
            fast: b.fast,
        }
    }
    fn value(&self) -> Bloom {
        Bloom {
            enabled: self.enabled,
            intensity: self.intensity,
            threshold: self.threshold,
            soft_knee: self.soft_knee,
            scatter: self.scatter,
            composite: value_of(&BLOOM_COMPOSITE, &self.composite),
            mip_count: self.mip_count,
            max_mip_height: self.max_mip_height,
            min_rect_height: self.min_rect_height,
            fast: self.fast,
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(default)]
struct VignetteRepr {
    enabled: bool,
    intensity: f32,
    smoothness: f32,
    roundness: f32,
    center: [f32; 2],
    color: [f32; 3],
}

impl Default for VignetteRepr {
    fn default() -> Self {
        Self::of(&Vignette::default())
    }
}

impl VignetteRepr {
    fn of(v: &Vignette) -> Self {
        Self {
            enabled: v.enabled,
            intensity: v.intensity,
            smoothness: v.smoothness,
            roundness: v.roundness,
            center: v.center.to_array(),
            color: v.color.to_array(),
        }
    }
    fn value(&self) -> Vignette {
        Vignette {
            enabled: self.enabled,
            intensity: self.intensity,
            smoothness: self.smoothness,
            roundness: self.roundness,
            center: self.center.into(),
            color: self.color.into(),
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(default)]
struct FilmGrainRepr {
    enabled: bool,
    intensity: f32,
    size: f32,
    luminance_response: f32,
    animated: bool,
    seed: u32,
}

impl Default for FilmGrainRepr {
    fn default() -> Self {
        Self::of(&FilmGrain::default())
    }
}

impl FilmGrainRepr {
    fn of(g: &FilmGrain) -> Self {
        Self {
            enabled: g.enabled,
            intensity: g.intensity,
            size: g.size,
            luminance_response: g.luminance_response,
            animated: g.animated,
            seed: g.seed,
        }
    }
    fn value(&self) -> FilmGrain {
        FilmGrain {
            enabled: self.enabled,
            intensity: self.intensity,
            size: self.size,
            luminance_response: self.luminance_response,
            animated: self.animated,
            seed: self.seed,
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(default)]
struct DistortionRepr {
    enabled: bool,
    strength: f32,
    chromatic: f32,
    resolution: String,
}

impl Default for DistortionRepr {
    fn default() -> Self {
        Self::of(&Distortion::default())
    }
}

impl DistortionRepr {
    fn of(d: &Distortion) -> Self {
        Self {
            enabled: d.enabled,
            strength: d.strength,
            chromatic: d.chromatic,
            resolution: name_of(&DISTORTION_RESOLUTION, d.resolution),
        }
    }
    fn value(&self) -> Distortion {
        Distortion {
            enabled: self.enabled,
            strength: self.strength,
            chromatic: self.chromatic,
            resolution: value_of(&DISTORTION_RESOLUTION, &self.resolution),
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(default)]
struct ColorGradingRepr {
    temperature: f32,
    tint: f32,
    saturation: f32,
    contrast: f32,
    lift: [f32; 3],
    gamma: [f32; 3],
    gain: [f32; 3],
}

impl Default for ColorGradingRepr {
    fn default() -> Self {
        Self::of(&ColorGrading::default())
    }
}

impl ColorGradingRepr {
    fn of(g: &ColorGrading) -> Self {
        Self {
            temperature: g.temperature,
            tint: g.tint,
            saturation: g.saturation,
            contrast: g.contrast,
            lift: g.lift.to_array(),
            gamma: g.gamma.to_array(),
            gain: g.gain.to_array(),
        }
    }
    fn value(&self) -> ColorGrading {
        ColorGrading {
            temperature: self.temperature,
            tint: self.tint,
            saturation: self.saturation,
            contrast: self.contrast,
            lift: self.lift.into(),
            gamma: self.gamma.into(),
            gain: self.gain.into(),
            ..ColorGrading::default()
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(default)]
struct PostProcessRepr {
    enabled: bool,
    exposure: f32,
    auto_exposure: AutoExposureRepr,
    tonemapper: String,
    bloom: BloomRepr,
    anti_aliasing: String,
    vignette: VignetteRepr,
    chromatic_aberration: f32,
    film_grain: FilmGrainRepr,
    distortion: DistortionRepr,
    color_grading: ColorGradingRepr,
    dither: bool,
    display_gamma: f32,
}

impl Default for PostProcessRepr {
    fn default() -> Self {
        Self::of(&PostProcess::default()).expect("no handles")
    }
}

impl PostProcessRepr {
    /// `None` for settings with session-local handles (a LUT, a grading shader).
    fn of(p: &PostProcess) -> Option<Self> {
        if p.color_grading.lut.is_some() || p.color_grading.shader.is_some() {
            return None;
        }
        Some(Self {
            enabled: p.enabled,
            exposure: p.exposure,
            auto_exposure: AutoExposureRepr::of(&p.auto_exposure),
            tonemapper: name_of(&TONEMAPPERS, p.tonemapper),
            bloom: BloomRepr::of(&p.bloom),
            anti_aliasing: name_of(&ANTI_ALIASING, p.anti_aliasing),
            vignette: VignetteRepr::of(&p.vignette),
            chromatic_aberration: if p.chromatic_aberration.enabled {
                p.chromatic_aberration.intensity
            } else {
                0.0
            },
            film_grain: FilmGrainRepr::of(&p.film_grain),
            distortion: DistortionRepr::of(&p.distortion),
            color_grading: ColorGradingRepr::of(&p.color_grading),
            dither: p.dither,
            display_gamma: p.display_gamma,
        })
    }
    fn post(&self) -> PostProcess {
        PostProcess {
            enabled: self.enabled,
            exposure: self.exposure,
            auto_exposure: self.auto_exposure.value(),
            tonemapper: value_of(&TONEMAPPERS, &self.tonemapper),
            bloom: self.bloom.value(),
            anti_aliasing: value_of(&ANTI_ALIASING, &self.anti_aliasing),
            vignette: self.vignette.value(),
            chromatic_aberration: ChromaticAberration {
                enabled: self.chromatic_aberration > 0.0,
                intensity: self.chromatic_aberration,
            },
            film_grain: self.film_grain.value(),
            distortion: self.distortion.value(),
            color_grading: self.color_grading.value(),
            dither: self.dither,
            display_gamma: self.display_gamma,
            // Fog, SSAO and SSR are not editable here yet (city effects): they keep their defaults.
            ..PostProcess::default()
        }
    }
}

/// What a shadow map covers: `Camera` (the first active Primary camera) or a fixed box.
#[derive(Serialize, Deserialize)]
enum ShadowFitRepr {
    Camera {
        max_distance: f32,
        #[serde(default)]
        receivers: Option<[[f32; 3]; 2]>,
    },
    Bounds([[f32; 3]; 2]),
}

#[derive(Serialize, Deserialize)]
#[serde(default)]
struct DirectionalShadowRepr {
    map_size: u32,
    fit: ShadowFitRepr,
    filter: String,
    depth_bias: f32,
    normal_bias: f32,
    slope_bias: f32,
    strength: f32,
    caster_depth: f32,
    min_caster_pixels: f32,
}

impl Default for DirectionalShadowRepr {
    fn default() -> Self {
        Self::of(&DirectionalShadow::default()).expect("default fit")
    }
}

impl DirectionalShadowRepr {
    /// `None` for fits naming a camera entity (not representable across sessions).
    fn of(s: &DirectionalShadow) -> Option<Self> {
        let aabb = |b: &Aabb| [b.min.to_array(), b.max.to_array()];
        let fit = match &s.fit {
            ShadowFit::Camera {
                camera: None,
                max_distance,
                receivers,
            } => ShadowFitRepr::Camera {
                max_distance: *max_distance,
                receivers: receivers.as_ref().map(aabb),
            },
            ShadowFit::Bounds(b) => ShadowFitRepr::Bounds(aabb(b)),
            _ => return None,
        };
        Some(Self {
            map_size: s.map_size,
            fit,
            filter: name_of(&SHADOW_FILTERS, s.filter),
            depth_bias: s.depth_bias,
            normal_bias: s.normal_bias,
            slope_bias: s.slope_bias,
            strength: s.strength,
            caster_depth: s.caster_depth,
            min_caster_pixels: s.min_caster_pixels,
        })
    }
    fn shadow(&self) -> DirectionalShadow {
        let aabb = |b: &[[f32; 3]; 2]| Aabb::new(b[0].into(), b[1].into());
        DirectionalShadow {
            map_size: self.map_size,
            fit: match &self.fit {
                ShadowFitRepr::Camera {
                    max_distance,
                    receivers,
                } => ShadowFit::Camera {
                    camera: None,
                    max_distance: *max_distance,
                    receivers: receivers.as_ref().map(aabb),
                },
                ShadowFitRepr::Bounds(b) => ShadowFit::Bounds(aabb(b)),
            },
            filter: value_of(&SHADOW_FILTERS, &self.filter),
            depth_bias: self.depth_bias,
            normal_bias: self.normal_bias,
            slope_bias: self.slope_bias,
            strength: self.strength,
            caster_depth: self.caster_depth,
            min_caster_pixels: self.min_caster_pixels,
        }
    }
}

/// Registration names used by the editor.
pub mod names {
    /// [`space3d_render::Material`].
    pub const MATERIAL: &str = "Material";
    /// [`space3d_render::DirectionalLight`].
    pub const DIRECTIONAL_LIGHT: &str = "DirectionalLight";
    /// [`space3d_render::PointLight`].
    pub const POINT_LIGHT: &str = "PointLight";
    /// [`space3d_render::Camera3d`] (primary-target cameras only).
    pub const CAMERA: &str = "Camera3d";
    /// [`space3d_render::Visibility`].
    pub const VISIBILITY: &str = "Visibility";
    /// [`super::MeshSource`].
    pub const MESH_SOURCE: &str = "MeshSource";
    /// [`space3d_render::CameraFrame`].
    pub const CAMERA_FRAME: &str = "CameraFrame";
    /// [`space3d_render::PostProcess`] (cameras without a LUT or grading shader: their
    /// handles are session-local).
    pub const POST_PROCESS: &str = "PostProcess";
    /// [`space3d_render::DirectionalShadow`] (fits without an explicit camera entity).
    pub const DIRECTIONAL_SHADOW: &str = "DirectionalShadow";
    /// [`space3d_render::Skybox`] (snapshot-only: its cubemap handle is session-local).
    pub const SKYBOX: &str = "Skybox";
    /// [`space3d_render::EnvironmentLight`] (snapshot-only: its cubemap handle is
    /// session-local).
    pub const ENVIRONMENT_LIGHT: &str = "EnvironmentLight";
}

/// Register render components (+ [`MeshSource`]) in a scene registry so the editor can save,
/// load, snapshot and JSON-edit them. Types that are already registered are left alone.
/// `Mesh3d` is snapshot-only (GPU handles are session-local); `MeshSource` is saved instead.
pub fn register_editor_components(r: &mut ComponentRegistry) {
    if !r.is_registered::<Material>() {
        r.register_mapped::<Material, MaterialRepr>(
            names::MATERIAL,
            |m, _| {
                Some(MaterialRepr {
                    base_color: m.base_color.to_array(),
                    metallic: m.metallic,
                    roughness: m.roughness,
                    emissive: m.emissive.to_array(),
                    unlit: m.unlit,
                    double_sided: m.double_sided,
                    alpha_blend: m.alpha_blend,
                    blend: BlendRepr::of(m.blend),
                    alpha_cutoff: m.alpha_cutoff,
                    normal_scale: m.normal_scale,
                    occlusion_strength: m.occlusion_strength,
                    shadows: ShadowModeRepr::of(m.shadows),
                    render_order: m.render_order,
                })
            },
            |r, _| {
                Some(Material {
                    base_color: r.base_color.into(),
                    metallic: r.metallic,
                    roughness: r.roughness,
                    emissive: r.emissive.into(),
                    unlit: r.unlit,
                    double_sided: r.double_sided,
                    alpha_blend: r.alpha_blend,
                    blend: r.blend.mode(),
                    alpha_cutoff: r.alpha_cutoff,
                    normal_scale: r.normal_scale,
                    occlusion_strength: r.occlusion_strength,
                    shadows: r.shadows.mode(),
                    render_order: r.render_order,
                    // Session-local GPU handles (textures, maps, custom shader) are not
                    // serialized.
                    ..Default::default()
                })
            },
        );
    }
    if !r.is_registered::<DirectionalLight>() {
        r.register_mapped::<DirectionalLight, DirectionalLightRepr>(
            names::DIRECTIONAL_LIGHT,
            |l, _| {
                Some(DirectionalLightRepr {
                    color: l.color.to_array(),
                    intensity: l.intensity,
                })
            },
            |r, _| {
                Some(DirectionalLight {
                    color: r.color.into(),
                    intensity: r.intensity,
                })
            },
        );
    }
    if !r.is_registered::<PointLight>() {
        r.register_mapped::<PointLight, PointLightRepr>(
            names::POINT_LIGHT,
            |l, _| {
                Some(PointLightRepr {
                    color: l.color.to_array(),
                    intensity: l.intensity,
                    range: l.range,
                })
            },
            |r, _| {
                Some(PointLight {
                    color: r.color.into(),
                    intensity: r.intensity,
                    range: r.range,
                })
            },
        );
    }
    if !r.is_registered::<Camera3d>() {
        r.register_mapped::<Camera3d, CameraRepr>(
            names::CAMERA,
            |c, _| {
                if c.target != RenderTarget::Primary {
                    return None;
                }
                Some(CameraRepr {
                    projection: match c.projection {
                        Projection::Perspective { fov_y, near, far } => ProjectionRepr::Perspective {
                            fov_y_degrees: fov_y.to_degrees(),
                            near,
                            far,
                        },
                        Projection::Orthographic { height, near, far } => {
                            ProjectionRepr::Orthographic { height, near, far }
                        }
                    },
                    clear_color: c.clear_color.map(|c| c.to_array()),
                    order: c.order,
                    viewport: c.viewport,
                    active: c.active,
                })
            },
            |r, _| {
                Some(Camera3d {
                    projection: match r.projection {
                        ProjectionRepr::Perspective {
                            fov_y_degrees,
                            near,
                            far,
                        } => Projection::Perspective {
                            fov_y: fov_y_degrees.to_radians(),
                            near,
                            far,
                        },
                        ProjectionRepr::Orthographic { height, near, far } => {
                            Projection::Orthographic { height, near, far }
                        }
                    },
                    clear_color: r.clear_color.map(Into::into),
                    target: RenderTarget::Primary,
                    order: r.order,
                    viewport: r.viewport,
                    active: r.active,
                })
            },
        );
    }
    if !r.is_registered::<Visibility>() {
        r.register_mapped::<Visibility, bool>(
            names::VISIBILITY,
            |v, _| Some(*v == Visibility::Visible),
            |b, _| Some(if b { Visibility::Visible } else { Visibility::Hidden }),
        );
    }
    if !r.is_registered::<CameraFrame>() {
        r.register_mapped::<CameraFrame, CameraFrameRepr>(
            names::CAMERA_FRAME,
            |c, _| Some(CameraFrameRepr::of(c)),
            |r, _| Some(r.frame()),
        );
    }
    if !r.is_registered::<PostProcess>() {
        r.register_mapped::<PostProcess, PostProcessRepr>(
            names::POST_PROCESS,
            |p, _| PostProcessRepr::of(p),
            |r, _| Some(r.post()),
        );
    }
    if !r.is_registered::<DirectionalShadow>() {
        r.register_mapped::<DirectionalShadow, DirectionalShadowRepr>(
            names::DIRECTIONAL_SHADOW,
            |s, _| DirectionalShadowRepr::of(s),
            |r, _| Some(r.shadow()),
        );
    }
    if !r.is_registered::<Skybox>() {
        r.register_snapshot_only::<Skybox>(names::SKYBOX);
    }
    if !r.is_registered::<EnvironmentLight>() {
        r.register_snapshot_only::<EnvironmentLight>(names::ENVIRONMENT_LIGHT);
    }
    if !r.is_registered::<MeshSource>() {
        r.register::<MeshSource>(names::MESH_SOURCE);
    }
    if !r.is_registered::<Mesh3d>() {
        r.register_snapshot_only::<Mesh3d>("Mesh3d");
    }
}

/// The entity's [`StableId`], inserting a fresh one if it has none.
pub fn ensure_stable_id(world: &mut World, e: Entity) -> Option<StableId> {
    if let Ok(id) = world.get::<&StableId>(e) {
        return Some(*id);
    }
    let id = StableId::new();
    world.insert_one(e, id).ok()?;
    Some(id)
}

/// Entity carrying `id`.
pub fn find_by_id(world: &World, id: StableId) -> Option<Entity> {
    world
        .query::<(Entity, &StableId)>()
        .iter()
        .find(|(_, s)| **s == id)
        .map(|(e, _)| e)
}

/// Serializable registered components of `e` as JSON values (name → value).
///
/// Makes sure `e` and its parent have stable ids first so references can be expressed.
pub fn component_values(
    world: &mut World,
    registry: &ComponentRegistry,
    e: Entity,
) -> Option<std::collections::BTreeMap<String, ComponentValue>> {
    ensure_stable_id(world, e)?;
    if let Ok(p) = world.get::<&Parent>(e).map(|p| p.0)
        && world.contains(p)
    {
        ensure_stable_id(world, p);
    }
    let scene = save_entities(world, registry, &[e]).ok()?;
    scene.entities.into_iter().next().map(|se| se.components)
}

/// Deserialize `value` as the registered component `name` and insert it into `e`
/// (replacing any previous value).
///
/// Entity references are resolved against entities of `world` carrying [`StableId`]s.
pub fn set_component_value(
    world: &mut World,
    registry: &ComponentRegistry,
    e: Entity,
    name: &str,
    value: ComponentValue,
) -> Result<(), String> {
    if !world.contains(e) {
        return Err("entity no longer exists".into());
    }
    if name == space3d_scene::PARENT {
        // References must resolve in the real world; handled without a scratch world.
        let id: StableId = serde_json::from_value(value).map_err(|err| err.to_string())?;
        let parent = find_by_id(world, id).ok_or("parent not found")?;
        if parent == e {
            return Err("an entity cannot be its own parent".into());
        }
        world.insert_one(e, Parent(parent)).map_err(|err| err.to_string())?;
        return Ok(());
    }
    let mut se = SceneEntity::new(StableId::new());
    se.components.insert(name.to_owned(), value);
    let scene = Scene { entities: vec![se] };
    let mut scratch = World::new();
    let spawned = scene
        .spawn_into(&mut scratch, registry)
        .map_err(|err| err.to_string())?;
    let t = spawned[0];
    let _ = scratch.remove_one::<StableId>(t);
    let taken = scratch.take(t).map_err(|err| err.to_string())?;
    world.insert(e, taken).map_err(|err| err.to_string())
}

/// Remove a component by registration name — supported for the core and render types the
/// editor knows (`Name`, `Transform`, `Parent`, `Material`, lights, `Camera3d`,
/// `Visibility`, `MeshSource`, `Mesh3d`, and the graphics components `CameraFrame`,
/// `PostProcess`, `DirectionalShadow`, `Skybox`, `EnvironmentLight`).
pub fn remove_component(world: &mut World, e: Entity, name: &str) -> bool {
    fn rm<T: space3d_core::hecs::Component>(w: &mut World, e: Entity) -> bool {
        w.remove_one::<T>(e).is_ok()
    }
    match name {
        "Name" => rm::<Name>(world, e),
        space3d_scene::TRANSFORM => rm::<Transform>(world, e),
        space3d_scene::PARENT => rm::<Parent>(world, e),
        names::MATERIAL => rm::<Material>(world, e),
        names::DIRECTIONAL_LIGHT => rm::<DirectionalLight>(world, e),
        names::POINT_LIGHT => rm::<PointLight>(world, e),
        names::CAMERA => rm::<Camera3d>(world, e),
        names::VISIBILITY => rm::<Visibility>(world, e),
        names::MESH_SOURCE => rm::<MeshSource>(world, e),
        names::CAMERA_FRAME => rm::<CameraFrame>(world, e),
        names::POST_PROCESS => rm::<PostProcess>(world, e),
        names::DIRECTIONAL_SHADOW => rm::<DirectionalShadow>(world, e),
        names::SKYBOX => rm::<Skybox>(world, e),
        names::ENVIRONMENT_LIGHT => rm::<EnvironmentLight>(world, e),
        "Mesh3d" => rm::<Mesh3d>(world, e),
        _ => false,
    }
}

/// Whether [`remove_component`] can remove `name`.
pub fn can_remove(name: &str) -> bool {
    matches!(
        name,
        "Name"
            | space3d_scene::TRANSFORM
            | space3d_scene::PARENT
            | names::MATERIAL
            | names::DIRECTIONAL_LIGHT
            | names::POINT_LIGHT
            | names::CAMERA
            | names::VISIBILITY
            | names::MESH_SOURCE
            | names::CAMERA_FRAME
            | names::POST_PROCESS
            | names::DIRECTIONAL_SHADOW
            | names::SKYBOX
            | names::ENVIRONMENT_LIGHT
            | "Mesh3d"
    )
}

/// Display name of an entity: its [`Name`], else a description of what it is.
pub fn display_name(world: &World, e: Entity) -> String {
    if let Ok(n) = world.get::<&Name>(e) {
        return n.0.clone();
    }
    let kind = if world.satisfies::<&Camera3d>(e) {
        "Camera"
    } else if world.satisfies::<&DirectionalLight>(e) {
        "Directional light"
    } else if world.satisfies::<&PointLight>(e) {
        "Point light"
    } else if world.satisfies::<&Mesh3d>(e) {
        "Mesh"
    } else {
        "Entity"
    };
    format!("{kind} #{}", e.id())
}

#[cfg(test)]
mod tests {
    use super::*;
    use space3d_core::glam::Vec4;

    fn registry() -> ComponentRegistry {
        let mut r = ComponentRegistry::default();
        register_editor_components(&mut r);
        r
    }

    #[test]
    fn json_roundtrip_of_render_components() {
        let r = registry();
        let mut w = World::new();
        let e = w.spawn((
            Name("cube".into()),
            Transform::from_xyz(1.0, 2.0, 3.0),
            Material::color(Vec4::new(0.5, 0.25, 1.0, 1.0)),
            MeshSource::Cube { size: 2.0 },
            Visibility::Hidden,
        ));
        let values = component_values(&mut w, &r, e).unwrap();
        for k in ["Name", "Transform", "Material", "MeshSource", "Visibility"] {
            assert!(values.contains_key(k), "{k} missing from {values:?}");
        }
        let mut mat = values["Material"].clone();
        mat["metallic"] = serde_json::json!(0.75);
        set_component_value(&mut w, &r, e, "Material", mat).unwrap();
        assert_eq!(w.get::<&Material>(e).unwrap().metallic, 0.75);
        assert_eq!(w.get::<&Name>(e).unwrap().0, "cube", "other components untouched");
        assert!(set_component_value(&mut w, &r, e, "Material", serde_json::json!({"bad": 1})).is_err());
        assert!(remove_component(&mut w, e, "Visibility"));
        assert!(!w.satisfies::<&Visibility>(e));
    }

    /// The graphics components round-trip through their JSON reprs (and the inspector can
    /// edit them); components with session-local handles are snapshot-only or skipped.
    #[test]
    fn graphics_components_roundtrip() {
        let r = registry();
        let mut w = World::new();
        let post = PostProcess::cinematic()
            .with_tonemapper(Tonemapper::AgXPunchy)
            .with_exposure(0.5)
            .with_auto_exposure(AutoExposure::on());
        let frame = CameraFrame {
            hdr: HdrMode::Compact,
            render_scale: Some(0.75),
        };
        let shadow = DirectionalShadow::new()
            .with_fit(ShadowFit::top_down(0.0, 40.0, 500.0))
            .with_filter(ShadowFilter::Pcf16)
            .with_map_size(4096);
        let material = Material {
            blend: BlendMode::Additive,
            alpha_cutoff: 0.3,
            normal_scale: 0.5,
            occlusion_strength: 0.8,
            shadows: ShadowMode::CastOnly,
            render_order: 3,
            ..Material::srgb(0.2, 0.4, 0.6)
        };
        let e = w.spawn((post, frame, shadow, material.clone()));
        let values = component_values(&mut w, &r, e).unwrap();
        for k in ["PostProcess", "CameraFrame", "DirectionalShadow", "Material"] {
            assert!(values.contains_key(k), "{k} missing from {values:?}");
        }
        assert_eq!(values["PostProcess"]["tonemapper"], "AgXPunchy");
        assert_eq!(values["DirectionalShadow"]["filter"], "Pcf16");
        let mut w2 = World::new();
        let e2 = w2.spawn(());
        for (k, v) in &values {
            if k != "StableId" {
                set_component_value(&mut w2, &r, e2, k, v.clone()).unwrap();
            }
        }
        assert_eq!(*w2.get::<&PostProcess>(e2).unwrap(), post);
        assert_eq!(*w2.get::<&CameraFrame>(e2).unwrap(), frame);
        assert_eq!(*w2.get::<&DirectionalShadow>(e2).unwrap(), shadow);
        assert_eq!(*w2.get::<&Material>(e2).unwrap(), material);
        // Edits through JSON.
        let mut p = values["PostProcess"].clone();
        p["bloom"]["enabled"] = serde_json::json!(false);
        set_component_value(&mut w, &r, e, "PostProcess", p).unwrap();
        assert!(!w.get::<&PostProcess>(e).unwrap().bloom.enabled);
        assert!(remove_component(&mut w, e, "CameraFrame"));
        assert!(can_remove("Skybox") && can_remove("EnvironmentLight"));
        // An explicit camera in the fit cannot be saved: the component is skipped, not mangled.
        let cam = w.spawn(());
        let pinned = DirectionalShadow::new().with_fit(ShadowFit::Camera {
            camera: Some(cam),
            max_distance: 100.0,
            receivers: None,
        });
        let e3 = w.spawn((pinned,));
        assert!(
            !component_values(&mut w, &r, e3)
                .unwrap()
                .contains_key("DirectionalShadow")
        );
    }

    #[test]
    fn parent_references_resolve_in_the_real_world() {
        let r = registry();
        let mut w = World::new();
        let a = w.spawn((Name("a".into()),));
        let b = w.spawn((Name("b".into()),));
        let id = ensure_stable_id(&mut w, a).unwrap();
        set_component_value(&mut w, &r, b, "Parent", serde_json::to_value(id).unwrap()).unwrap();
        assert_eq!(w.get::<&Parent>(b).unwrap().0, a);
        let values = component_values(&mut w, &r, b).unwrap();
        assert_eq!(values["Parent"], serde_json::to_value(id).unwrap());
        assert!(set_component_value(&mut w, &r, a, "Parent", serde_json::to_value(id).unwrap()).is_err());
    }
}
