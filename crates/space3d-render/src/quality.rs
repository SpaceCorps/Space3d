//! [`GraphicsQuality`]: coherent quality tiers for graphics menus and device defaults (graphics
//! spec §2.2.6, preset table §4.4; docs/graphics.md "Quality presets").
//!
//! A tier is a set of *performance* knobs: render scale, MSAA / FXAA, bloom chain length, auto
//! exposure and lens effects on Low, the shadow map and its filter, the image-based lighting
//! bake and the shading model. It never changes a game's look otherwise (tonemapper, exposure,
//! grading, bloom intensity, sky, materials stay as authored). Particles take the same tier
//! through `space3d_particles::ParticleSettings::for_quality`.
//!
//! | knob | Low (WebGL2 / C-class) | Medium (iGPU, Apple, mobile) | High (dGPU) | Ultra (opt-in) |
//! |---|---|---|---|---|
//! | render scale (primary) | 0.75 | 1 | 1 | 1 |
//! | MSAA / AA | 1 + FXAA Low | 4 on tile-based GPUs, else 1 + FXAA Medium | 4 | 4 |
//! | bloom mips / max mip height | 4 / 270 | 6 / 540 | 7 / 720 | 8 / 720 |
//! | auto exposure | off | as authored | as authored | as authored |
//! | chromatic aberration, film grain | off | as authored | as authored | as authored |
//! | shadow map / filter | 1024 / Pcf4 | 2048 / Pcf9 | 2048 / Pcf16 | 4096 / Pcf16 |
//! | IBL specular / irradiance / samples / passes per frame | 128 / 16 / 32 / 6 | 256 / 32 / 64 / all | 512 / 32 / 128 / all | 512 / 32 / 256 / all |
//! | shading model | Classic | Standard | Standard | Standard |
//! | particles (`ParticleSettings::for_quality`) | CPU 16k (8k web) | GPU 262k (65k mobile) | GPU 262k | GPU 262k |
//! | recommended sky face size ([`GraphicsQuality::sky_face_size`]) | 512 | 1024 | 2048 | 2048 |
//!
//! Apply a tier in one call with [`GraphicsQuality::apply`] (every camera's `PostProcess`, every
//! `DirectionalShadow`, `FrameSettings`, `LightingSettings`, the primary target's MSAA), or
//! adjust single settings with the per-area adapters ([`GraphicsQuality::post`],
//! [`GraphicsQuality::shadow`], …). On loading screens, build the pipelines of the tier with
//! [`GraphicsQuality::prewarm_request`] (WebGL2 links a program in 50–150 ms).

use crate::caps::GraphicsCaps;
use crate::lighting::api::{DirectionalShadow, IblSettings, LightingSettings, ShadingModel, ShadowFilter};
use crate::post::api::{AntiAliasing, FxaaQuality, PostProcess};
use crate::renderer::{PrewarmRequest, RenderCapabilities, Renderer};
use crate::settings::FrameSettings;
use space3d_core::{Entity, Resources, World};

/// Coherent quality tiers (module docs: the preset table).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum GraphicsQuality {
    /// WebGL2 and devices without float render targets or compute.
    Low,
    /// Integrated, Apple and mobile GPUs.
    #[default]
    Medium,
    /// Discrete GPUs.
    High,
    /// High with a 4096² shadow map, an 8-level bloom chain (half-resolution first level up
    /// to 4K) and a finer environment bake. Never recommended: an explicit menu choice.
    Ultra,
}

impl GraphicsQuality {
    /// Every tier, lowest first (graphics menus).
    pub const ALL: [Self; 4] = [Self::Low, Self::Medium, Self::High, Self::Ultra];

    /// Display name ("Low", "Medium", "High", "Ultra").
    pub fn name(self) -> &'static str {
        match self {
            Self::Low => "Low",
            Self::Medium => "Medium",
            Self::High => "High",
            Self::Ultra => "Ultra",
        }
    }

    /// The tier for a device: WebGL2 / GL, no float render targets, or no compute (the GPU
    /// particle backend needs it) → Low; discrete GPUs → High; everything else (integrated,
    /// Apple, mobile, software) → Medium. `graphics` is after `FrameSettings::downlevel`
    /// masking, so `DownlevelOverride::webgl2()` recommends Low on any device.
    pub fn recommended(caps: &RenderCapabilities, graphics: &GraphicsCaps) -> Self {
        if caps.adapter.backend == wgpu::Backend::Gl || graphics.hdr_format.is_none() || !graphics.compute {
            Self::Low
        } else if caps.adapter.device_type == wgpu::DeviceType::DiscreteGpu {
            Self::High
        } else {
            Self::Medium
        }
    }

    /// [`Self::recommended`] for a renderer.
    pub fn recommended_for(renderer: &Renderer) -> Self {
        Self::recommended(renderer.capabilities(), renderer.graphics_caps())
    }

    /// `base` with this tier's frame options: the primary render scale (Low 0.75, else 1).
    pub fn frame_settings(self, base: FrameSettings) -> FrameSettings {
        FrameSettings {
            render_scale: if self == Self::Low { 0.75 } else { 1.0 },
            ..base
        }
    }

    /// MSAA sample count of the primary target for this tier on a device: Low 1; Medium 4 on
    /// tile-based GPUs (Apple, Android, iOS: resolves stay on chip), else 1 (FXAA instead);
    /// High and Ultra 4. Never more than the device offers.
    pub fn msaa_samples(self, caps: &RenderCapabilities) -> u32 {
        let four = if caps.max_msaa >= 4 { 4 } else { 1 };
        let tile_based =
            caps.adapter.backend == wgpu::Backend::Metal || cfg!(any(target_os = "android", target_os = "ios"));
        match self {
            Self::Low => 1,
            Self::Medium if tile_based => four,
            Self::Medium => 1,
            _ => four,
        }
    }

    /// `base` with this tier's post settings, for a primary target without MSAA on Low and
    /// Medium and with it on High and Ultra (the tiers' defaults; [`Self::post_for`] takes the
    /// actual sample count).
    pub fn post(self, base: PostProcess) -> PostProcess {
        let samples = if self >= Self::High { 4 } else { 1 };
        self.post_for(base, samples)
    }

    /// `base` with this tier's post settings when the camera renders with `msaa_samples`:
    /// * the bloom chain (`PostProcess::with_quality`: Low 4 / 270, Medium 6 / 540, High
    ///   7 / 720, Ultra 8 / 720: High's chain plus a wider level) and the FXAA quality when
    ///   FXAA is on;
    /// * anti-aliasing: Low FXAA Low; Medium FXAA Medium without MSAA; High and Ultra FXAA
    ///   High / Ultra without MSAA (a device without 4× MSAA), else as authored;
    /// * Low turns auto exposure, chromatic aberration and film grain off; the other tiers keep
    ///   them as authored (the spec table's "on": available, never forced — they change the
    ///   look).
    ///
    /// * the city effects (docs/graphics/post.md, "City effects"): Low drops SSAO and SSR and
    ///   the fog's sun in-scatter (the fog itself stays: one cheap pass); Medium caps SSAO at
    ///   1 slice x 4 steps at half resolution with a 2-tap blur and drops SSR; High caps SSAO at
    ///   2 x 6 and SSR at 32 steps; Ultra caps SSAO at 3 x 8. Caps only lower what was authored.
    ///
    /// Everything else (tonemapper, exposure, bloom intensity, grading, distortion, dither)
    /// stays as authored.
    pub fn post_for(self, base: PostProcess, msaa_samples: u32) -> PostProcess {
        let mut p = base.with_quality(self);
        let msaa = msaa_samples > 1;
        match self {
            Self::Low => {
                p.anti_aliasing = AntiAliasing::Fxaa(FxaaQuality::Low);
                p.auto_exposure.enabled = false;
                p.chromatic_aberration.enabled = false;
                p.film_grain.enabled = false;
            }
            Self::Medium if !msaa => p.anti_aliasing = AntiAliasing::Fxaa(FxaaQuality::Medium),
            Self::High if !msaa => p.anti_aliasing = AntiAliasing::Fxaa(FxaaQuality::High),
            Self::Ultra if !msaa => p.anti_aliasing = AntiAliasing::Fxaa(FxaaQuality::Ultra),
            _ => {}
        }
        // The city effects.
        let (slices, steps, ssr_steps) = match self {
            Self::Low | Self::Medium => (1, 4, 0),
            Self::High => (2, 6, 32),
            Self::Ultra => (3, 8, 128),
        };
        p.ssao.slices = p.ssao.slices.min(slices);
        p.ssao.steps = p.ssao.steps.min(steps);
        match self {
            Self::Low => {
                p.fog.sun_intensity = 0.0;
                p.ssao.enabled = false;
                p.ssr.enabled = false;
            }
            Self::Medium => {
                p.ssao.half_res = true;
                p.ssao.blur_radius = p.ssao.blur_radius.min(2);
                p.ssr.enabled = false;
            }
            _ => p.ssr.steps = p.ssr.steps.min(ssr_steps),
        }
        p
    }

    /// `base` with this tier's lighting settings: the shading model (Low `Classic`, else
    /// `Standard`) and the image-based lighting bake (specular / irradiance face sizes,
    /// specular samples, bake passes per frame). Normal maps and transparent sorting stay as
    /// authored.
    pub fn lighting(self, base: LightingSettings) -> LightingSettings {
        let (specular_size, irradiance_size, specular_samples, bake_passes_per_frame) = match self {
            Self::Low => (128, 16, 32, Some(6)),
            Self::Medium => (256, 32, 64, None),
            Self::High => (512, 32, 128, None),
            Self::Ultra => (512, 32, 256, None),
        };
        LightingSettings {
            shading: if self == Self::Low {
                ShadingModel::Classic
            } else {
                ShadingModel::Standard
            },
            ibl: IblSettings {
                specular_size,
                irradiance_size,
                specular_samples,
                bake_passes_per_frame,
                ..base.ibl
            },
            ..base
        }
    }

    /// `base` with this tier's shadow map size and filter (Low 1024 / Pcf4, Medium 2048 / Pcf9,
    /// High 2048 / Pcf16, Ultra 4096 / Pcf16); fit, biases and strength stay as authored.
    pub fn shadow(self, base: DirectionalShadow) -> DirectionalShadow {
        let (map_size, filter) = match self {
            Self::Low => (1024, ShadowFilter::Pcf4),
            Self::Medium => (2048, ShadowFilter::Pcf9),
            Self::High => (2048, ShadowFilter::Pcf16),
            Self::Ultra => (4096, ShadowFilter::Pcf16),
        };
        DirectionalShadow {
            map_size,
            filter,
            ..base
        }
    }

    /// Recommended face size of skybox / environment cubemaps (a documentation value: the
    /// content decides): Low 512, Medium 1024, High and Ultra 2048.
    pub fn sky_face_size(self) -> u32 {
        match self {
            Self::Low => 512,
            Self::Medium => 1024,
            _ => 2048,
        }
    }

    /// The tier's prewarm plan for `materials` (graphics spec §3.5): apply the tier, render one
    /// frame of the loading scene (it records the pass signatures), then
    /// `renderer.prewarm(&quality.prewarm_request(&materials))`. The request holds the
    /// materials (their forward, depth and shadow pipelines for the tier's shading model, filter
    /// and sample count); on Medium and up it adds the capture format
    /// ([`crate::RENDER_TARGET_FORMAT`]) so screenshots and render-target previews do not
    /// hitch. Low leaves it out: every WebGL2 program link is 50–150 ms.
    pub fn prewarm_request(self, materials: &[crate::Material]) -> PrewarmRequest {
        PrewarmRequest {
            materials: materials.to_vec(),
            layouts: Vec::new(),
            formats: if self == Self::Low {
                Vec::new()
            } else {
                vec![crate::RENDER_TARGET_FORMAT]
            },
        }
    }

    /// Apply the whole tier: `FrameSettings` ([`Self::frame_settings`]), `LightingSettings`
    /// ([`Self::lighting`]), the primary target's MSAA ([`Self::msaa_samples`], when a
    /// [`Renderer`] exists), every camera's `PostProcess` ([`Self::post_for`] with the sample
    /// count the camera will get) and every `DirectionalShadow` ([`Self::shadow`]). Cameras
    /// and lights spawned later take the tier by calling the adapters themselves (or `apply`
    /// again).
    ///
    /// Switching tiers is lossless: the post settings a camera had before the first `apply`
    /// (its authored look) are remembered, and each `apply` derives the tier's settings from
    /// them — Low turning chromatic aberration off does not lose it for High. Changing a
    /// camera's `PostProcess` after an `apply` makes the new value its authored look.
    /// Particles: `ParticleSettings::for_quality` (the `space3d-particles` crate).
    pub fn apply(self, world: &mut World, resources: &mut Resources) {
        let (msaa, hdr_msaa) = match resources.get_mut::<Renderer>() {
            Some(renderer) => {
                let samples = self.msaa_samples(renderer.capabilities());
                renderer.set_msaa_samples(samples);
                (renderer.msaa_samples(), renderer.graphics_caps().hdr_msaa)
            }
            None => (if self >= Self::High { 4 } else { 1 }, true),
        };
        // Hdr groups without MSAA-capable float targets drop to 1 sample (logged by the frame).
        let post_samples = if hdr_msaa { msaa } else { 1 };
        let frame = resources.init::<FrameSettings>();
        *frame = self.frame_settings(frame.clone());
        let lighting = resources.init::<LightingSettings>();
        *lighting = self.lighting(lighting.clone());
        let mut remembered = Vec::new();
        for (entity, post, base) in world.query_mut::<(Entity, &mut PostProcess, Option<&QualityAuthored>)>() {
            let authored = match base {
                Some(b) if b.applied == *post => b.authored,
                _ => *post,
            };
            let applied = self.post_for(authored, post_samples);
            *post = applied;
            remembered.push((entity, QualityAuthored { authored, applied }));
        }
        for (entity, base) in remembered {
            let _ = world.insert_one(entity, base);
        }
        for shadow in world.query_mut::<&mut DirectionalShadow>() {
            *shadow = self.shadow(*shadow);
        }
    }
}

/// A camera's post settings before [`GraphicsQuality::apply`] adjusted them (`authored`), and
/// what it set (`applied`): the next `apply` derives from `authored` unless the camera's
/// `PostProcess` changed since.
#[derive(Clone, Copy, Debug, PartialEq)]
struct QualityAuthored {
    authored: PostProcess,
    applied: PostProcess,
}

impl std::fmt::Display for GraphicsQuality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::post::api::{AutoExposure, FilmGrain};

    fn caps(backend: wgpu::Backend, device_type: wgpu::DeviceType, max_msaa: u32) -> RenderCapabilities {
        RenderCapabilities {
            adapter: wgpu::AdapterInfo::new(device_type, backend),
            compute: true,
            indirect: true,
            indirect_first_instance: true,
            multi_draw_indirect_count: false,
            base_vertex: true,
            texture_compression: (false, false, false),
            best_draw_path: crate::DrawPath::GpuMultiDraw,
            max_msaa,
            max_storage_binding: 1 << 27,
            max_buffer_size: 1 << 30,
        }
    }

    fn graphics(float: bool, compute: bool) -> GraphicsCaps {
        GraphicsCaps {
            hdr_format: float.then_some(wgpu::TextureFormat::Rgba16Float),
            hdr_msaa: float,
            compact_hdr_format: None,
            compact_hdr_storage: false,
            bloom_format: float.then_some(wgpu::TextureFormat::Rgba16Float),
            exposure_format: float.then_some(wgpu::TextureFormat::R32Float),
            read_only_depth: true,
            msaa_depth_resolve: true,
            compute,
            compute_histogram: compute,
            timestamps: false,
            depth_bias_clamp: true,
            max_texture_2d: 8192,
            max_texture_cube: 8192,
            max_bind_groups: 4,
        }
    }

    #[test]
    fn recommended_tiers() {
        use wgpu::{Backend as B, DeviceType as D};
        let full = graphics(true, true);
        assert_eq!(
            GraphicsQuality::recommended(&caps(B::Gl, D::IntegratedGpu, 4), &full),
            GraphicsQuality::Low
        );
        assert_eq!(
            GraphicsQuality::recommended(&caps(B::Metal, D::IntegratedGpu, 4), &graphics(false, true)),
            GraphicsQuality::Low,
            "no float targets"
        );
        assert_eq!(
            GraphicsQuality::recommended(&caps(B::Vulkan, D::DiscreteGpu, 4), &graphics(true, false)),
            GraphicsQuality::Low,
            "no compute: the particles need the CPU backend's budget"
        );
        assert_eq!(
            GraphicsQuality::recommended(&caps(B::Metal, D::IntegratedGpu, 4), &full),
            GraphicsQuality::Medium
        );
        assert_eq!(
            GraphicsQuality::recommended(&caps(B::Vulkan, D::DiscreteGpu, 4), &full),
            GraphicsQuality::High
        );
    }

    #[test]
    fn msaa_per_tier() {
        use wgpu::{Backend as B, DeviceType as D};
        let apple = caps(B::Metal, D::IntegratedGpu, 4);
        let pc = caps(B::Vulkan, D::DiscreteGpu, 4);
        let no_msaa = caps(B::Vulkan, D::DiscreteGpu, 1);
        let tbdr = cfg!(any(target_os = "android", target_os = "ios"));
        assert_eq!(GraphicsQuality::Low.msaa_samples(&apple), 1);
        assert_eq!(GraphicsQuality::Medium.msaa_samples(&apple), 4);
        assert_eq!(GraphicsQuality::Medium.msaa_samples(&pc), if tbdr { 4 } else { 1 });
        assert_eq!(GraphicsQuality::High.msaa_samples(&pc), 4);
        assert_eq!(GraphicsQuality::Ultra.msaa_samples(&apple), 4);
        assert_eq!(GraphicsQuality::High.msaa_samples(&no_msaa), 1);
    }

    #[test]
    fn preset_table() {
        use GraphicsQuality as Q;
        let fs = FrameSettings::default();
        assert_eq!(Q::Low.frame_settings(fs.clone()).render_scale, 0.75);
        for q in [Q::Medium, Q::High, Q::Ultra] {
            assert_eq!(q.frame_settings(fs.clone().with_render_scale(0.5)).render_scale, 1.0);
        }
        let s = |q: Q| {
            let s = q.shadow(DirectionalShadow::default().with_strength(0.7));
            assert_eq!(s.strength, 0.7, "authored fields stay");
            (s.map_size, s.filter)
        };
        assert_eq!(s(Q::Low), (1024, ShadowFilter::Pcf4));
        assert_eq!(s(Q::Medium), (2048, ShadowFilter::Pcf9));
        assert_eq!(s(Q::High), (2048, ShadowFilter::Pcf16));
        assert_eq!(s(Q::Ultra), (4096, ShadowFilter::Pcf16));
        let l = |q: Q| {
            let l = q.lighting(LightingSettings {
                normal_maps: false,
                ..LightingSettings::default()
            });
            assert!(!l.normal_maps, "authored fields stay");
            (
                l.shading,
                l.ibl.specular_size,
                l.ibl.irradiance_size,
                l.ibl.specular_samples,
                l.ibl.bake_passes_per_frame,
            )
        };
        assert_eq!(l(Q::Low), (ShadingModel::Classic, 128, 16, 32, Some(6)));
        assert_eq!(l(Q::Medium), (ShadingModel::Standard, 256, 32, 64, None));
        assert_eq!(l(Q::High), (ShadingModel::Standard, 512, 32, 128, None));
        assert_eq!(l(Q::Ultra), (ShadingModel::Standard, 512, 32, 256, None));
        assert_eq!(Q::ALL.map(Q::sky_face_size), [512, 1024, 2048, 2048]);
        assert_eq!(Q::ALL.map(Q::name), ["Low", "Medium", "High", "Ultra"]);
        assert_eq!(Q::Ultra.to_string(), "Ultra");
        assert!(Q::Low < Q::Medium && Q::High < Q::Ultra);
    }

    #[test]
    fn city_effects_follow_the_tier_and_stay_lossless() {
        use crate::post::api::{Fog, Ssao, Ssr};
        let authored = PostProcess::default()
            .with_fog(Fog::city_haze())
            .with_ssao(Ssao::on().with_quality(4, 12))
            .with_ssr(Ssr::on().with_steps(96));
        let low = Q::Low.post(authored);
        assert!(
            low.fog.enabled && low.fog.sun_intensity == 0.0,
            "fog stays, no sun lobe"
        );
        assert!(!low.ssao.enabled && !low.ssr.enabled);
        let medium = Q::Medium.post(authored);
        assert!(medium.ssao.enabled && !medium.ssr.enabled);
        assert_eq!((medium.ssao.slices, medium.ssao.steps), (1, 4));
        assert!(medium.ssao.half_res && medium.ssao.blur_radius <= 2);
        let high = Q::High.post(authored);
        assert_eq!((high.ssao.slices, high.ssao.steps, high.ssr.steps), (2, 6, 32));
        assert!(high.ssr.enabled);
        let ultra = Q::Ultra.post(authored);
        assert_eq!((ultra.ssao.slices, ultra.ssao.steps, ultra.ssr.steps), (3, 8, 96));
        assert_eq!(ultra.fog, authored.fog, "fog is never touched above Low");
        // Caps only lower: a cheap authored SSAO is kept on every tier that has it.
        let cheap = PostProcess::default().with_ssao(Ssao::on().with_quality(1, 3));
        let u = Q::Ultra.post(cheap);
        assert_eq!((u.ssao.slices, u.ssao.steps), (1, 3));
        // Nothing enabled stays nothing enabled.
        assert!(!Q::Ultra.post(PostProcess::default()).screen_effects());
    }

    #[test]
    fn post_adjusts_performance_knobs_only() {
        use GraphicsQuality as Q;
        let authored = PostProcess::cinematic()
            .with_exposure(0.5)
            .with_auto_exposure(AutoExposure::on());
        let bloom = |p: PostProcess| (p.bloom.mip_count, p.bloom.max_mip_height);
        let low = Q::Low.post(authored);
        assert_eq!(bloom(low), (4, 270));
        assert_eq!(low.anti_aliasing, AntiAliasing::Fxaa(FxaaQuality::Low));
        assert!(!low.auto_exposure.enabled && !low.chromatic_aberration.enabled && !low.film_grain.enabled);
        assert_eq!(
            (low.exposure, low.tonemapper, low.vignette),
            (0.5, authored.tonemapper, authored.vignette)
        );
        let medium = Q::Medium.post(PostProcess::default());
        assert_eq!(bloom(medium), (6, 540));
        assert_eq!(medium.anti_aliasing, AntiAliasing::Fxaa(FxaaQuality::Medium));
        assert_eq!(
            Q::Medium.post_for(PostProcess::default(), 4).anti_aliasing,
            AntiAliasing::None,
            "MSAA on a tile-based GPU instead of FXAA"
        );
        let high = Q::High.post(authored);
        assert_eq!(bloom(high), (7, 720));
        assert!(high.auto_exposure.enabled && high.chromatic_aberration.enabled && high.film_grain.enabled);
        assert_eq!(Q::High.post(PostProcess::default()).anti_aliasing, AntiAliasing::None);
        assert_eq!(
            Q::High.post_for(PostProcess::default(), 1).anti_aliasing,
            AntiAliasing::Fxaa(FxaaQuality::High),
            "no MSAA on the device: FXAA"
        );
        let ultra = Q::Ultra.post(authored);
        assert_eq!(bloom(ultra), (8, 720));
        assert_eq!(ultra.anti_aliasing, AntiAliasing::Fxaa(FxaaQuality::Ultra));
        let grain = FilmGrain {
            enabled: true,
            ..FilmGrain::default()
        };
        assert!(
            !Q::Low
                .post(PostProcess::default().with_film_grain(grain))
                .film_grain
                .enabled
        );
    }

    /// `apply` without a renderer: settings, cameras and lights take the tier, and switching
    /// down and up again restores what Low turned off.
    #[test]
    fn apply_is_lossless_across_tiers() {
        use GraphicsQuality as Q;
        let mut world = World::new();
        let mut resources = Resources::default();
        let authored = PostProcess::cinematic().with_auto_exposure(AutoExposure::on());
        let cam = world.spawn((authored,));
        let sun = world.spawn((DirectionalShadow::default(),));
        Q::Low.apply(&mut world, &mut resources);
        assert_eq!(resources.expect::<FrameSettings>().render_scale, 0.75);
        assert_eq!(resources.expect::<LightingSettings>().shading, ShadingModel::Classic);
        let low = *world.get::<&PostProcess>(cam).unwrap();
        assert!(!low.chromatic_aberration.enabled && !low.auto_exposure.enabled);
        assert_eq!(world.get::<&DirectionalShadow>(sun).unwrap().map_size, 1024);
        Q::High.apply(&mut world, &mut resources);
        let high = *world.get::<&PostProcess>(cam).unwrap();
        assert_eq!(high, Q::High.post(authored), "derived from the authored look");
        assert!(high.chromatic_aberration.enabled && high.auto_exposure.enabled);
        assert_eq!(resources.expect::<FrameSettings>().render_scale, 1.0);
        assert_eq!(
            *world.get::<&DirectionalShadow>(sun).unwrap(),
            Q::High.shadow(DirectionalShadow::default())
        );
        // A camera edited after an apply: the edit is the new authored look.
        world.get::<&mut PostProcess>(cam).unwrap().exposure = 1.5;
        Q::Medium.apply(&mut world, &mut resources);
        let medium = *world.get::<&PostProcess>(cam).unwrap();
        assert_eq!(medium.exposure, 1.5);
        assert!(medium.chromatic_aberration.enabled);
    }

    #[test]
    fn prewarm_request_per_tier() {
        let m = [crate::Material::srgb(1.0, 0.5, 0.2)];
        let low = GraphicsQuality::Low.prewarm_request(&m);
        assert_eq!(low.materials.len(), 1);
        assert!(low.formats.is_empty(), "WebGL2: no extra program links");
        let high = GraphicsQuality::High.prewarm_request(&m);
        assert_eq!(high.formats, vec![crate::RENDER_TARGET_FORMAT]);
    }
}
