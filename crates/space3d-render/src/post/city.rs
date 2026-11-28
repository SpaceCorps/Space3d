//! Settings of the city effects (fog, SSAO, SSR): plain `Copy` structs with `Default` (disabled),
//! fields of [`super::api::PostProcess`]. Re-exported from the crate root through `api`.
//! The passes are in `screenfx.rs`; docs/graphics/post.md, "City effects".

use glam::Vec3;

/// Exponential height and distance fog, applied to the linear HDR scene before bloom and
/// exposure.
///
/// The optical depth along the view ray integrates the density
/// `density * exp(-height_falloff * (y - base_height))` analytically (Y is up); the pixel keeps
/// `T = exp(-tau)` of the scene and gains `1 - T` of the fog color, capped at `max_opacity`.
/// The fog color is `color` plus a sun in-scatter lobe `sun_color * sun_intensity *
/// pow(max(dot(view_dir, -sun_dir), 0), sun_exponent)`. Sky pixels (depth at the far plane)
/// are fogged by `sky_amount`, strongest at the horizon.
///
/// Works in Hdr mode with scene depth; off in the Ldr fallback and in Direct mode (warned
/// once). Off, the camera renders as without it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fog {
    /// Default false.
    pub enabled: bool,
    /// Linear in-scatter color. Default (0.60, 0.70, 0.82).
    pub color: Vec3,
    /// Extinction per world unit at `base_height`. Default 0.004.
    pub density: f32,
    /// Density falloff with height, 1/units; 0 = uniform fog. Default 0.05.
    pub height_falloff: f32,
    /// World Y where `density` applies. Default 0.
    pub base_height: f32,
    /// No fog before this distance from the camera. Default 0.
    pub start_distance: f32,
    /// Maximum fog opacity, 0..=1. Default 1.
    pub max_opacity: f32,
    /// Sun in-scatter color (linear). Default (1.0, 0.80, 0.55).
    pub sun_color: Vec3,
    /// Sun in-scatter strength; 0 = none. Default 0.
    pub sun_intensity: f32,
    /// Sun lobe tightness. Default 8.
    pub sun_exponent: f32,
    /// Direction the sun light travels; `None` = the first directional light of the view
    /// (no sun in-scatter without one). Default None.
    pub sun_direction: Option<Vec3>,
    /// Fog on sky pixels: 0 none, 1 full at the horizon. Default 1.
    pub sky_amount: f32,
    /// How fast sky fog fades with elevation (per unit of the view direction's Y). Default 3.
    pub sky_falloff: f32,
}

impl Default for Fog {
    fn default() -> Self {
        Self {
            enabled: false,
            color: Vec3::new(0.60, 0.70, 0.82),
            density: 0.004,
            height_falloff: 0.05,
            base_height: 0.0,
            start_distance: 0.0,
            max_opacity: 1.0,
            sun_color: Vec3::new(1.0, 0.80, 0.55),
            sun_intensity: 0.0,
            sun_exponent: 8.0,
            sun_direction: None,
            sky_amount: 1.0,
            sky_falloff: 3.0,
        }
    }
}

impl Fog {
    /// Enabled light blue haze with a warm sun lobe (the city demo's look).
    pub fn city_haze() -> Self {
        Self {
            enabled: true,
            sun_intensity: 0.6,
            ..Self::default()
        }
    }
    /// Set [`Self::density`].
    pub fn with_density(mut self, d: f32) -> Self {
        self.density = d;
        self
    }
    /// Set [`Self::base_height`] and [`Self::height_falloff`].
    pub fn with_height(mut self, base_height: f32, falloff: f32) -> Self {
        self.base_height = base_height;
        self.height_falloff = falloff;
        self
    }
    /// Set [`Self::color`].
    pub fn with_color(mut self, c: Vec3) -> Self {
        self.color = c;
        self
    }
    /// Set the sun in-scatter.
    pub fn with_sun(mut self, color: Vec3, intensity: f32, exponent: f32) -> Self {
        self.sun_color = color;
        self.sun_intensity = intensity;
        self.sun_exponent = exponent;
        self
    }
    /// Set [`Self::start_distance`].
    pub fn with_start(mut self, d: f32) -> Self {
        self.start_distance = d;
        self
    }

    /// CPU mirror of the shader's fog opacity (`fog_opacity` in screenfx.wgsl) for a ray from a
    /// camera at height `camera_y` to a point `distance` away at height `point_y`:
    /// `1 - exp(-tau)`, capped at `max_opacity`.
    pub fn opacity(&self, camera_y: f32, point_y: f32, distance: f32) -> f32 {
        let d = (distance - self.start_distance.max(0.0)).max(0.0);
        let dh = point_y - camera_y;
        let dist = distance.max(1e-6);
        let k = self.height_falloff.max(0.0);
        let base = self.density.max(0.0) * (-k * (camera_y - self.base_height)).exp();
        // The ray's height rises by `dh` over `distance`: the integral of exp(-k t dh / dist)
        // over the distance is d * (1 - e^(-x)) / x with x = k dh d / dist.
        let x = k * dh * d / dist;
        let avg = if x.abs() < 1e-4 { 1.0 } else { (1.0 - (-x).exp()) / x };
        let tau = base * d * avg;
        (1.0 - (-tau).exp()).min(self.max_opacity.clamp(0.0, 1.0))
    }
}

/// Screen-space ambient occlusion (ground-truth style horizon search over a few direction
/// slices, at half resolution by default, then a depth-aware blur). The blurred occlusion `ao`
/// multiplies the HDR scene color by `mix(1, ao, intensity)`: a forward renderer has no
/// separate ambient buffer, so direct light is darkened in creases too (keep `intensity`
/// moderate on strongly lit scenes).
///
/// Needs scene depth (Hdr mode); view-space normals are reconstructed from it. Off in the Ldr
/// fallback and in Direct mode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ssao {
    /// Default false.
    pub enabled: bool,
    /// Search radius in world units. Default 1.
    pub radius: f32,
    /// 0..=2. Default 1.
    pub intensity: f32,
    /// Contrast exponent on the occlusion term. Default 1.5.
    pub power: f32,
    /// Horizon bias as a fraction of the radius (hides self-occlusion on flat surfaces).
    /// Default 0.05.
    pub bias: f32,
    /// The effect fades out beyond this view distance. Default 150.
    pub max_distance: f32,
    /// Direction slices, 1..=4. Default 2.
    pub slices: u32,
    /// Samples per slice side, 2..=12. Default 6.
    pub steps: u32,
    /// Compute at half resolution (upsampled by the blur). Default true.
    pub half_res: bool,
    /// Blur taps each side; 0 = no blur. Default 3.
    pub blur_radius: u32,
}

impl Default for Ssao {
    fn default() -> Self {
        Self {
            enabled: false,
            radius: 1.0,
            intensity: 1.0,
            power: 1.5,
            bias: 0.05,
            max_distance: 150.0,
            slices: 2,
            steps: 6,
            half_res: true,
            blur_radius: 3,
        }
    }
}

impl Ssao {
    /// Enabled with the defaults.
    pub fn on() -> Self {
        Self {
            enabled: true,
            ..Self::default()
        }
    }
    /// Set [`Self::radius`].
    pub fn with_radius(mut self, r: f32) -> Self {
        self.radius = r;
        self
    }
    /// Set [`Self::intensity`].
    pub fn with_intensity(mut self, i: f32) -> Self {
        self.intensity = i;
        self
    }
    /// Set [`Self::slices`] and [`Self::steps`].
    pub fn with_quality(mut self, slices: u32, steps: u32) -> Self {
        self.slices = slices;
        self.steps = steps;
        self
    }
}

/// Screen-space reflections: a linear ray march in view space against the scene depth with a
/// thickness test and a binary refinement; the hit color is the scene color of the same frame.
///
/// It needs each surface's normal, roughness and reflectivity: a G-buffer-lite target
/// (`Rgba8Unorm`: octahedral world normal, roughness, F0) that the renderer fills with an extra
/// depth-tested pass over the view's opaque draws after the scene pass, **only for cameras
/// with `ssr.enabled`**. The reflection weight is Fresnel (Schlick from the material's F0) ×
/// a roughness fade × a screen-edge fade. Misses and faded pixels keep the forward pass's
/// image-based (cubemap) specular; on a hit that specular term is replaced by the hit color,
/// so wet asphalt and glass show the scene where it is on screen and the environment elsewhere.
///
/// Needs scene depth and `GraphicsCaps::read_only_depth` (not on GL/WebGL2); Hdr mode only.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ssr {
    /// Default false.
    pub enabled: bool,
    /// World distance the ray travels. Default 80.
    pub max_distance: f32,
    /// Ray-march steps, 8..=128. Default 48.
    pub steps: u32,
    /// View-space depth tolerance of a hit, world units. Default 0.4.
    pub thickness: f32,
    /// 0..=2. Default 1.
    pub intensity: f32,
    /// No reflection from surfaces rougher than this; it fades in from 0.6× of it. Default 0.55.
    pub max_roughness: f32,
    /// Fraction of the screen near its border that fades to the environment. Default 0.1.
    pub edge_fade: f32,
    /// Blur of the reflection with roughness and ray length (1 = the default look, 0 = mirror
    /// sharp at every roughness). Default 1.
    pub roughness_blur: f32,
    /// Vertical stretch of that blur (>= 1): on wet, rough ground the reflection of a light
    /// smears into a vertical glare streak. Default 3.
    pub stretch: f32,
    /// Extra gain of reflected radiance above display white (lights, emissives), up to
    /// `1 + hdr_boost` at 4x white. Default 1.
    pub hdr_boost: f32,
    /// Depth-aware blur taps each side of the reflection (0 = none): hides the ray-march
    /// noise at silhouettes and rims, no temporal history. 0..=4. Default 2.
    pub denoise: u32,
}

impl Default for Ssr {
    fn default() -> Self {
        Self {
            enabled: false,
            max_distance: 80.0,
            steps: 48,
            thickness: 0.4,
            intensity: 1.0,
            max_roughness: 0.55,
            edge_fade: 0.1,
            roughness_blur: 1.0,
            stretch: 3.0,
            hdr_boost: 1.0,
            denoise: 2,
        }
    }
}

impl Ssr {
    /// Enabled with the defaults.
    pub fn on() -> Self {
        Self {
            enabled: true,
            ..Self::default()
        }
    }
    /// Set [`Self::max_distance`].
    pub fn with_distance(mut self, d: f32) -> Self {
        self.max_distance = d;
        self
    }
    /// Set [`Self::steps`].
    pub fn with_steps(mut self, s: u32) -> Self {
        self.steps = s;
        self
    }
    /// Set [`Self::intensity`].
    pub fn with_intensity(mut self, i: f32) -> Self {
        self.intensity = i;
        self
    }
    /// Set [`Self::max_roughness`].
    pub fn with_max_roughness(mut self, r: f32) -> Self {
        self.max_roughness = r;
        self
    }
    /// Set [`Self::roughness_blur`] and [`Self::stretch`].
    pub fn with_blur(mut self, roughness_blur: f32, stretch: f32) -> Self {
        self.roughness_blur = roughness_blur;
        self.stretch = stretch;
        self
    }
    /// Set [`Self::hdr_boost`].
    pub fn with_hdr_boost(mut self, b: f32) -> Self {
        self.hdr_boost = b;
        self
    }
    /// Set [`Self::denoise`].
    pub fn with_denoise(mut self, taps: u32) -> Self {
        self.denoise = taps;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_disabled() {
        assert!(!Fog::default().enabled);
        assert!(!Ssao::default().enabled);
        assert!(!Ssr::default().enabled);
        assert!(Fog::city_haze().enabled && Ssao::on().enabled && Ssr::on().enabled);
    }

    #[test]
    fn fog_opacity_grows_with_distance_and_falls_with_height() {
        let f = Fog::city_haze().with_density(0.01).with_height(0.0, 0.1);
        let near = f.opacity(1.0, 1.0, 10.0);
        let far = f.opacity(1.0, 1.0, 200.0);
        assert!(near > 0.0 && far > near && far < 1.0, "{near} {far}");
        // Looking up out of the layer: less fog than along the ground.
        let up = f.opacity(1.0, 150.0, 200.0);
        assert!(up < far, "{up} {far}");
        // Uniform fog (no falloff) is the plain exponential.
        let u = Fog::city_haze().with_density(0.01).with_height(0.0, 0.0);
        let want = 1.0 - (-0.01f32 * 100.0).exp();
        assert!((u.opacity(5.0, 50.0, 100.0) - want).abs() < 1e-5);
    }

    #[test]
    fn fog_start_and_cap() {
        let f = Fog {
            max_opacity: 0.5,
            start_distance: 50.0,
            ..Fog::city_haze().with_density(0.05).with_height(0.0, 0.0)
        };
        assert_eq!(f.opacity(0.0, 0.0, 40.0), 0.0);
        assert!((f.opacity(0.0, 0.0, 10_000.0) - 0.5).abs() < 1e-6);
    }
}
