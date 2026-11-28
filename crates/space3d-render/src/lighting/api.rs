//! Lighting API hub (graphics spec §2.2.9): cubemaps, [`Skybox`], [`EnvironmentLight`],
//! [`DirectionalShadow`] and the global [`LightingSettings`]. Re-exported from the crate root.
//!
//! Types are frozen at F2; the renderer consumes them with the materials milestones (PM3
//! skybox, PM4 image-based lighting, PM5 shadows). Until then they are plain data.

use crate::assets::TextureSource;
use crate::culling::Aabb;
use space3d_core::{HdrImageData, ImageData};

pub use crate::handles::CubemapHandle;

/// Pixel data of a cubemap for [`crate::RenderAssets::add_cubemap`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum CubemapSource {
    /// Six equal square faces as a 6-layer texture in wgpu/OpenGL order +X, −X, +Y, −Y, +Z, −Z
    /// (any sampled format; missing mips are generated).
    Faces(TextureSource),
    /// An equirectangular panorama converted on the GPU to `face_size`² faces.
    Equirect {
        /// The panorama.
        texture: TextureSource,
        /// Edge of each face in texels.
        face_size: u32,
    },
}

/// Default face size of a panorama: a quarter of its width rounded down to a power of two, at
/// most 2048.
fn equirect_face_size(width: u32) -> u32 {
    let quarter = (width / 4).max(1);
    (1u32 << (31 - quarter.leading_zeros())).min(2048)
}

impl CubemapSource {
    /// sRGB faces in wgpu order (+X, −X, +Y, −Y, +Z, −Z) and orientation (the cube-map face
    /// table of every graphics API, which is left-handed: seen from inside this right-handed
    /// world a face image appears mirrored); `None` unless all six are square and of equal
    /// size.
    pub fn from_faces(faces: [ImageData; 6]) -> Option<Self> {
        let size = faces[0].width;
        if size == 0 || faces.iter().any(|f| f.width != size || f.height != size) {
            return None;
        }
        TextureSource::from_layers(&faces).map(Self::Faces)
    }

    /// Faces authored for three.js' `CubeTextureLoader` (a mirrored convention): swaps ±X and
    /// flips every face horizontally, on the CPU, once.
    pub fn from_faces_threejs(faces: [ImageData; 6]) -> Option<Self> {
        let [px, nx, py, ny, pz, nz] = faces;
        let flip = |mut img: ImageData| {
            let (w, h) = (img.width as usize, img.height as usize);
            for row in img.pixels.chunks_exact_mut(w * 4).take(h) {
                for x in 0..w / 2 {
                    for c in 0..4 {
                        row.swap(x * 4 + c, (w - 1 - x) * 4 + c);
                    }
                }
            }
            img
        };
        Self::from_faces([nx, px, py, ny, pz, nz].map(flip))
    }

    /// An sRGB LDR panorama.
    pub fn from_equirect_image(image: ImageData) -> Self {
        let face_size = equirect_face_size(image.width);
        Self::Equirect {
            texture: TextureSource::from_image(image),
            face_size,
        }
    }

    /// A linear HDR panorama (`Rgba16Float`).
    pub fn from_equirect_hdr(image: &HdrImageData) -> Self {
        Self::Equirect {
            texture: TextureSource::from_hdr(image),
            face_size: equirect_face_size(image.width),
        }
    }

    /// Converted panoramas: faces of `size`² texels (face sources keep their own size).
    pub fn with_face_size(self, size: u32) -> Self {
        match self {
            Self::Equirect { texture, .. } => Self::Equirect {
                texture,
                face_size: size.max(1),
            },
            faces => faces,
        }
    }

    /// Edge of the cubemap's faces in texels.
    pub fn face_size(&self) -> u32 {
        match self {
            Self::Faces(t) => t.width,
            Self::Equirect { face_size, .. } => *face_size,
        }
    }
}

/// Camera component: cubemap background, drawn in the Skybox phase at the far plane (replaces
/// the clear color where nothing was drawn).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Skybox {
    /// The cubemap.
    pub cubemap: CubemapHandle,
    /// Linear multiplier (above 1 feeds bloom with post).
    pub intensity: f32,
    /// Rotation of the sky.
    pub rotation: glam::Quat,
    /// Lower-hemisphere control: `(linear color, blend half-width)`. The sky below the horizon
    /// (view direction Y < 0, world up) is replaced by the color, blended over
    /// `-blend..blend` of the direction's Y so a ground plane meets the sky without a seam.
    /// `None` = the cubemap all around (default; identical to before).
    pub ground: Option<(glam::Vec3, f32)>,
}

impl Skybox {
    /// Intensity 1, no rotation.
    pub fn new(cubemap: CubemapHandle) -> Self {
        Self {
            cubemap,
            intensity: 1.0,
            rotation: glam::Quat::IDENTITY,
            ground: None,
        }
    }
    /// Set [`Self::ground`]: sky color below the horizon, `blend_height` the half-width of the
    /// blend in the view direction's Y (0.05 is a soft horizon; match the fog color to hide it).
    pub fn with_ground(mut self, color: glam::Vec3, blend_height: f32) -> Self {
        self.ground = Some((color, blend_height));
        self
    }
    /// Set [`Self::intensity`].
    pub fn with_intensity(mut self, i: f32) -> Self {
        self.intensity = i;
        self
    }
    /// Set [`Self::rotation`].
    pub fn with_rotation(mut self, r: glam::Quat) -> Self {
        self.rotation = r;
        self
    }
}

/// Camera component: image-based lighting — diffuse irradiance and GGX prefiltered specular
/// from `cubemap` — on top of `AmbientLight`. Until the bake is done
/// ([`crate::Renderer::environment_state`]) the camera renders without it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnvironmentLight {
    /// The environment.
    pub cubemap: CubemapHandle,
    /// Diffuse (irradiance) multiplier.
    pub diffuse_intensity: f32,
    /// Specular (reflection) multiplier.
    pub specular_intensity: f32,
    /// Rotation of the environment.
    pub rotation: glam::Quat,
}

impl EnvironmentLight {
    /// Intensities 1, no rotation.
    pub fn new(cubemap: CubemapHandle) -> Self {
        Self {
            cubemap,
            diffuse_intensity: 1.0,
            specular_intensity: 1.0,
            rotation: glam::Quat::IDENTITY,
        }
    }
    /// Set both intensities.
    pub fn with_intensity(mut self, diffuse: f32, specular: f32) -> Self {
        self.diffuse_intensity = diffuse;
        self.specular_intensity = specular;
        self
    }
    /// Set [`Self::rotation`].
    pub fn with_rotation(mut self, r: glam::Quat) -> Self {
        self.rotation = r;
        self
    }
}

/// Light component next to a `DirectionalLight` (which is unchanged): the light casts shadows.
/// At most one light casts: among several, the lowest `Entity::to_bits` wins (logged once).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DirectionalShadow {
    /// Shadow map edge in texels (power of two, clamped to the device). Default 2048.
    pub map_size: u32,
    /// What the shadow map covers.
    pub fit: ShadowFit,
    /// Filter kernel.
    pub filter: ShadowFilter,
    /// World units along the light. Default 0.02.
    pub depth_bias: f32,
    /// Shadow texels along the geometric normal. Default 1.0.
    pub normal_bias: f32,
    /// Raster slope-scaled bias. Default 1.5.
    pub slope_bias: f32,
    /// 1 = full shadow, 0 = none. Default 1.
    pub strength: f32,
    /// Extra distance toward the light searched for casters. Default 500.
    pub caster_depth: f32,
    /// An instance whose bounding sphere is under this many pixels in radius on the screen of
    /// the camera the map is fitted to (`ShadowFit::Camera`'s, else the first primary scene
    /// view; perspective cameras only) is left out of the shadow map: it neither shadows
    /// others nor itself. Default 0: everything casts. A cut for the cheaper tiers, where the
    /// shadow pass is vertex work for shadows a few pixels wide; an instance crossing the size
    /// gains or loses its shadow at once (no hysteresis).
    pub min_caster_pixels: f32,
}

impl Default for DirectionalShadow {
    fn default() -> Self {
        Self {
            map_size: 2048,
            fit: ShadowFit::default(),
            filter: ShadowFilter::default(),
            depth_bias: 0.02,
            normal_bias: 1.0,
            slope_bias: 1.5,
            strength: 1.0,
            caster_depth: 500.0,
            min_caster_pixels: 0.0,
        }
    }
}

impl DirectionalShadow {
    /// The defaults.
    pub fn new() -> Self {
        Self::default()
    }
    /// Set [`Self::map_size`].
    pub fn with_map_size(mut self, s: u32) -> Self {
        self.map_size = s;
        self
    }
    /// Set [`Self::fit`].
    pub fn with_fit(mut self, f: ShadowFit) -> Self {
        self.fit = f;
        self
    }
    /// Set [`Self::filter`].
    pub fn with_filter(mut self, f: ShadowFilter) -> Self {
        self.filter = f;
        self
    }
    /// Set the three biases.
    pub fn with_bias(mut self, depth: f32, normal: f32, slope: f32) -> Self {
        self.depth_bias = depth;
        self.normal_bias = normal;
        self.slope_bias = slope;
        self
    }
    /// Set [`Self::strength`].
    pub fn with_strength(mut self, s: f32) -> Self {
        self.strength = s;
        self
    }
    /// Set [`Self::min_caster_pixels`].
    pub fn with_min_caster_pixels(mut self, pixels: f32) -> Self {
        self.min_caster_pixels = pixels;
        self
    }
}

/// What a shadow map covers.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum ShadowFit {
    /// A camera's frustum (`None` = the first active Primary camera by `order`) from its near
    /// plane to `max_distance`, clipped to `receivers` if given.
    Camera {
        /// The camera.
        camera: Option<space3d_core::Entity>,
        /// Farthest distance covered.
        max_distance: f32,
        /// Box receivers are known to lie in.
        receivers: Option<Aabb>,
    },
    /// A fixed world box (arenas, previews, stories).
    Bounds(Aabb),
}

impl Default for ShadowFit {
    fn default() -> Self {
        Self::Camera {
            camera: None,
            max_distance: 200.0,
            receivers: None,
        }
    }
}

impl ShadowFit {
    /// Top-down games: the camera frustum within the slab `|y − plane_y| ≤ half_thickness`.
    pub fn top_down(plane_y: f32, half_thickness: f32, max_distance: f32) -> Self {
        let h = half_thickness.abs();
        Self::Camera {
            camera: None,
            max_distance,
            receivers: Some(Aabb::new(
                glam::Vec3::new(f32::MIN, plane_y - h, f32::MIN),
                glam::Vec3::new(f32::MAX, plane_y + h, f32::MAX),
            )),
        }
    }
}

/// Shadow filter kernel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ShadowFilter {
    /// One tap.
    Hard,
    /// 2×2 PCF.
    Pcf4,
    /// 3×3 PCF.
    #[default]
    Pcf9,
    /// 4×4 PCF.
    Pcf16,
}

/// Global lighting options (resource, inserted by [`crate::RenderPlugin`]).
#[derive(Clone, Debug, PartialEq)]
pub struct LightingSettings {
    /// Shading model of lit materials. Default `Classic` (pixel-identical to the legacy
    /// shader).
    pub shading: ShadingModel,
    /// Global switch: false ignores normal maps (low-end). Default true.
    pub normal_maps: bool,
    /// Sorting of transparent instances. Default `None` (opt-in `PerInstance`).
    pub transparent_sort: TransparentSort,
    /// Image-based lighting bake settings.
    pub ibl: IblSettings,
}

impl Default for LightingSettings {
    fn default() -> Self {
        Self {
            shading: ShadingModel::Classic,
            normal_maps: true,
            transparent_sort: TransparentSort::None,
            ibl: IblSettings::default(),
        }
    }
}

/// Shading model of lit materials.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ShadingModel {
    /// The legacy Blinn–Phong-style model (IBL with it is approximate).
    #[default]
    Classic,
    /// GGX / Smith / Schlick (use with image-based lighting).
    Standard,
}

/// Sorting of transparent instances.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TransparentSort {
    /// Batch order (the legacy behaviour).
    #[default]
    None,
    /// Back to front per instance, up to `max_instances` per view.
    PerInstance {
        /// Cap per view (larger batches keep batch order).
        max_instances: u32,
    },
}

/// Image-based lighting bake settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IblSettings {
    /// Prefiltered specular cube face size (≤ the source's). Default 256.
    pub specular_size: u32,
    /// Irradiance cube face size. Default 32.
    pub irradiance_size: u32,
    /// Specular samples per texel (filtered importance sampling). Default 64.
    pub specular_samples: u32,
    /// Irradiance samples per texel. Default 256.
    pub irradiance_samples: u32,
    /// Most bake render passes per frame (`None` = all at once). Headless renderers ignore it.
    pub bake_passes_per_frame: Option<u32>,
}

impl Default for IblSettings {
    fn default() -> Self {
        Self {
            specular_size: 256,
            irradiance_size: 32,
            specular_samples: 64,
            irradiance_samples: 256,
            bake_passes_per_frame: None,
        }
    }
}

/// Where a cubemap is on its way to being usable as an environment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum EnvironmentState {
    /// No such cubemap (removed, or not seen by a rendered frame yet).
    Missing,
    /// Registered, not on the GPU yet.
    Uploading,
    /// Uploaded; bake steps remain (all of them until a camera's `EnvironmentLight` uses it).
    Baking {
        /// Steps left: render passes (WebGL2 / GLES 3.0), else compute dispatches.
        passes_left: u32,
    },
    /// Baked: cameras with an `EnvironmentLight` on it use it.
    Ready,
    /// Upload or bake failed (logged).
    Failed,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults() {
        let s = DirectionalShadow::new();
        assert_eq!(
            (s.map_size, s.filter, s.depth_bias, s.caster_depth),
            (2048, ShadowFilter::Pcf9, 0.02, 500.0)
        );
        assert!(matches!(
            s.fit,
            ShadowFit::Camera {
                camera: None,
                max_distance: 200.0,
                receivers: None
            }
        ));
        let l = LightingSettings::default();
        assert_eq!(
            (l.shading, l.normal_maps, l.transparent_sort),
            (ShadingModel::Classic, true, TransparentSort::None)
        );
        assert_eq!(l.ibl.specular_size, 256);
        let t = ShadowFit::top_down(1.0, -2.0, 100.0);
        let ShadowFit::Camera { receivers: Some(b), .. } = t else {
            panic!()
        };
        assert_eq!((b.min.y, b.max.y), (-1.0, 3.0));
    }

    #[test]
    fn cubemap_sources() {
        let face = |v: u8| ImageData::filled(4, 4, [v, 0, 0, 255]);
        let faces = std::array::from_fn(|i| face(i as u8));
        let CubemapSource::Faces(t) = CubemapSource::from_faces(faces).unwrap() else {
            panic!()
        };
        assert_eq!((t.layers, t.width), (6, 4));
        assert_eq!(t.data[0], 0);
        assert_eq!(t.data[4 * 4 * 4], 1, "layer order kept");
        let mut odd = std::array::from_fn(|i| face(i as u8));
        odd[3] = ImageData::new(4, 2);
        assert!(CubemapSource::from_faces(odd).is_none());
        // three.js: ±X swapped, faces mirrored.
        let mut three: [ImageData; 6] = std::array::from_fn(|i| face(i as u8));
        three[0].set(0, 0, [9, 9, 9, 255]);
        let CubemapSource::Faces(t) = CubemapSource::from_faces_threejs(three).unwrap() else {
            panic!()
        };
        assert_eq!(t.data[0], 1, "−X face first after the swap");
        let second = 4 * 4 * 4;
        assert_eq!(&t.data[second + 3 * 4..second + 4 * 4], &[9, 9, 9, 255], "mirrored");
        let e = CubemapSource::from_equirect_image(ImageData::new(1000, 500));
        assert_eq!(e.face_size(), 128);
        assert_eq!(e.with_face_size(64).face_size(), 64);
        let hdr = CubemapSource::from_equirect_hdr(&HdrImageData::filled(64, 32, [1.0; 4]));
        assert_eq!(hdr.face_size(), 16);
        assert_eq!(equirect_face_size(16384), 2048);
    }
}
