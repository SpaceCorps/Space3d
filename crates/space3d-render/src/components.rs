//! ECS components and resources understood by the renderer.

use crate::handles::{MeshHandle, ShaderHandle, TextureHandle};
use crate::material::{BlendMode, MapMask, ShadowMode, TextureTransform};
use glam::{Mat4, Vec3, Vec4};

/// Renders the entity's mesh with its [`Material`] at its `GlobalTransform`.
/// Entities sharing mesh + material are drawn with a single instanced draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Mesh3d(pub MeshHandle);

/// Optional visibility toggle; entities without it are visible.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Visibility {
    /// Drawn (subject to frustum culling).
    #[default]
    Visible,
    /// Skipped entirely.
    Hidden,
}

/// Convert one sRGB-encoded channel (0..1) to linear.
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert one linear channel (0..1) to sRGB encoding.
pub fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Surface description. All colors are **linear**; use [`Material::srgb`] for sRGB input.
///
/// Materials are plain values: the renderer interns identical materials, so many entities with
/// equal materials still batch into one draw.
#[derive(Clone, Debug, PartialEq)]
pub struct Material {
    /// Linear RGBA multiplier (and alpha when `alpha_blend`).
    pub base_color: Vec4,
    /// Optional sRGB color texture. Texture arrays pick their layer per instance
    /// ([`crate::InstanceSemantic::TextureLayer`]); single-layer textures ignore it.
    pub base_color_texture: Option<TextureHandle>,
    /// 0 = dielectric, 1 = metal.
    pub metallic: f32,
    /// 0 = mirror, 1 = fully rough.
    pub roughness: f32,
    /// Linear emitted radiance added after lighting.
    pub emissive: Vec3,
    /// Skip lighting: output `base_color * texture + emissive`.
    pub unlit: bool,
    /// Disable back-face culling.
    pub double_sided: bool,
    /// Alpha blending (drawn after opaque geometry, no depth writes).
    pub alpha_blend: bool,
    /// Custom WGSL hooks ([`crate::CustomShader`], registered with
    /// [`crate::RenderAssets::add_shader`]); `None` uses the built-in shader only.
    pub shader: Option<ShaderHandle>,
    /// Tangent-space normal map (linear; +Y green up). Default `None`.
    pub normal_texture: Option<TextureHandle>,
    /// Strength of the normal map's XY. Default 1.
    pub normal_scale: f32,
    /// Metallic-roughness map (linear; G roughness, B metallic, multiplying the factors).
    /// Default `None`.
    pub metallic_roughness_texture: Option<TextureHandle>,
    /// Ambient occlusion map (linear, R). The same handle as `metallic_roughness_texture` is an
    /// ORM-packed texture (one sample). Default `None`.
    pub occlusion_texture: Option<TextureHandle>,
    /// Occlusion strength (0 = none). Default 1.
    pub occlusion_strength: f32,
    /// Emissive map (sRGB), multiplied by `emissive`. Default `None`.
    pub emissive_texture: Option<TextureHandle>,
    /// Blend mode. Default `Opaque`; the legacy `alpha_blend` flag counts as `AlphaBlend`
    /// ([`Self::effective_blend`]).
    pub blend: BlendMode,
    /// Alpha below which `AlphaMask` discards. Default 0.5.
    pub alpha_cutoff: f32,
    /// UV transform of every map. Default identity.
    pub uv_transform: TextureTransform,
    /// Maps sampled with the second UV set (`MeshExtras::uvs1`). Default none.
    pub uv1_maps: MapMask,
    /// Shadow participation. Default `Auto`.
    pub shadows: ShadowMode,
    /// Draw order among transparent meshes (lower first; then blend class, pipeline, …).
    /// Default 0. Orders meshes only: particles have their own order.
    pub render_order: i16,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color: Vec4::ONE,
            base_color_texture: None,
            metallic: 0.0,
            roughness: 0.5,
            emissive: Vec3::ZERO,
            unlit: false,
            double_sided: false,
            alpha_blend: false,
            shader: None,
            normal_texture: None,
            normal_scale: 1.0,
            metallic_roughness_texture: None,
            occlusion_texture: None,
            occlusion_strength: 1.0,
            emissive_texture: None,
            blend: BlendMode::Opaque,
            alpha_cutoff: 0.5,
            uv_transform: TextureTransform::IDENTITY,
            uv1_maps: MapMask::NONE,
            shadows: ShadowMode::Auto,
            render_order: 0,
        }
    }
}

impl Material {
    /// Lit material with a linear RGBA color.
    pub fn color(base_color: Vec4) -> Self {
        Self {
            base_color,
            ..Default::default()
        }
    }
    /// Lit material from sRGB-encoded components in 0..1.
    pub fn srgb(r: f32, g: f32, b: f32) -> Self {
        Self::color(Vec4::new(srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b), 1.0))
    }
    /// Unlit material with a linear RGBA color.
    pub fn unlit(base_color: Vec4) -> Self {
        Self {
            base_color,
            unlit: true,
            ..Default::default()
        }
    }
    /// Set the base color texture.
    pub fn with_texture(mut self, texture: TextureHandle) -> Self {
        self.base_color_texture = Some(texture);
        self
    }
    /// Set metallic / roughness.
    pub fn with_metallic_roughness(mut self, metallic: f32, roughness: f32) -> Self {
        self.metallic = metallic;
        self.roughness = roughness;
        self
    }
    /// Set emissive radiance (linear).
    pub fn with_emissive(mut self, emissive: Vec3) -> Self {
        self.emissive = emissive;
        self
    }
    /// Draw with a [`crate::CustomShader`] (its hooks run inside the built-in shader).
    pub fn with_shader(mut self, shader: ShaderHandle) -> Self {
        self.shader = Some(shader);
        self
    }
    /// Set the normal map and its scale.
    pub fn with_normal_map(mut self, t: TextureHandle, scale: f32) -> Self {
        self.normal_texture = Some(t);
        self.normal_scale = scale;
        self
    }
    /// Set the metallic-roughness map.
    pub fn with_metallic_roughness_map(mut self, t: TextureHandle) -> Self {
        self.metallic_roughness_texture = Some(t);
        self
    }
    /// Set the occlusion map and its strength.
    pub fn with_occlusion_map(mut self, t: TextureHandle, strength: f32) -> Self {
        self.occlusion_texture = Some(t);
        self.occlusion_strength = strength;
        self
    }
    /// Set an ORM-packed map (R occlusion, G roughness, B metallic) as both the
    /// metallic-roughness and the occlusion map.
    pub fn with_orm_map(mut self, t: TextureHandle) -> Self {
        self.metallic_roughness_texture = Some(t);
        self.occlusion_texture = Some(t);
        self
    }
    /// Set the emissive map and the emissive factor it is multiplied by.
    pub fn with_emissive_map(mut self, t: TextureHandle, factor: Vec3) -> Self {
        self.emissive_texture = Some(t);
        self.emissive = factor;
        self
    }
    /// Set the blend mode.
    pub fn with_blend(mut self, b: BlendMode) -> Self {
        self.blend = b;
        self
    }
    /// Alpha-tested holes below `cutoff` (`BlendMode::AlphaMask`).
    pub fn with_alpha_mask(mut self, cutoff: f32) -> Self {
        self.blend = BlendMode::AlphaMask;
        self.alpha_cutoff = cutoff;
        self
    }
    /// Tile every map `u` x `v` times across the mesh's UV range (a `uv_transform` scale; the
    /// maps' addressing must be `Repeat`, the default). With `MeshData::cuboid_tiled` /
    /// `plane_tiled` (UVs in world tiles) a 1 k wall texture keeps its real-world scale.
    pub fn with_tiling(mut self, u: f32, v: f32) -> Self {
        self.uv_transform.scale = glam::Vec2::new(u, v);
        self
    }

    /// Set the UV transform of every map.
    pub fn with_uv_transform(mut self, t: TextureTransform) -> Self {
        self.uv_transform = t;
        self
    }
    /// Sample these maps with the second UV set.
    pub fn with_uv1_maps(mut self, m: MapMask) -> Self {
        self.uv1_maps = m;
        self
    }
    /// Set the shadow participation.
    pub fn with_shadows(mut self, m: ShadowMode) -> Self {
        self.shadows = m;
        self
    }
    /// Set the transparent draw order.
    pub fn with_render_order(mut self, o: i16) -> Self {
        self.render_order = o;
        self
    }
    /// The blend mode drawn: `blend`, or `AlphaBlend` for a legacy `alpha_blend` material.
    pub fn effective_blend(&self) -> BlendMode {
        if self.blend == BlendMode::Opaque && self.alpha_blend {
            BlendMode::AlphaBlend
        } else {
            self.blend
        }
    }
    /// Unlit, additive, double-sided glow of a linear color (`alpha_blend` is set too, so it
    /// also draws blended where blend modes are not supported yet).
    pub fn additive(color: Vec4) -> Self {
        Self {
            base_color: color,
            unlit: true,
            double_sided: true,
            alpha_blend: true,
            blend: BlendMode::Additive,
            ..Default::default()
        }
    }
    /// Unlit premultiplied color (a = 0 adds light; a > 0 also covers).
    pub fn premultiplied(color: Vec4) -> Self {
        Self {
            base_color: color,
            unlit: true,
            alpha_blend: true,
            blend: BlendMode::Premultiplied,
            ..Default::default()
        }
    }
}

/// Camera projection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Projection {
    /// Perspective projection.
    Perspective {
        /// Vertical field of view in radians.
        fov_y: f32,
        /// Near plane distance.
        near: f32,
        /// Far plane distance.
        far: f32,
    },
    /// Orthographic projection (width follows the aspect ratio).
    Orthographic {
        /// Visible world-space height.
        height: f32,
        /// Near plane distance.
        near: f32,
        /// Far plane distance.
        far: f32,
    },
}

impl Default for Projection {
    fn default() -> Self {
        Projection::Perspective {
            fov_y: 60f32.to_radians(),
            near: 0.1,
            far: 1000.0,
        }
    }
}

impl Projection {
    /// Right-handed projection matrix with wgpu's 0..1 depth range.
    pub fn matrix(&self, aspect: f32) -> Mat4 {
        match *self {
            Projection::Perspective { fov_y, near, far } => {
                glam::camera::rh::proj::directx::perspective(fov_y, aspect, near, far)
            }
            Projection::Orthographic { height, near, far } => {
                let h = height * 0.5;
                let w = h * aspect;
                glam::camera::rh::proj::directx::orthographic(-w, w, -h, h, near, far)
            }
        }
    }
}

/// Where a camera renders.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum RenderTarget {
    /// The window surface, or the offscreen capture target when headless.
    #[default]
    Primary,
    /// A texture created with [`crate::RenderAssets::add_render_target`].
    Texture(TextureHandle),
}

/// A 3D camera. Uses the entity's `GlobalTransform` as its world transform (looking down -Z).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera3d {
    /// Projection.
    pub projection: Projection,
    /// Linear clear color of the camera's viewport (the whole target without one); `None`
    /// draws over what cameras with a lower `order` drew into the same target this frame (for
    /// overlay cameras). The first camera drawing into a target in a frame always starts from
    /// a cleared target: black wherever its own clear color does not apply.
    pub clear_color: Option<Vec4>,
    /// Render target.
    pub target: RenderTarget,
    /// Cameras render in ascending `order`.
    pub order: i32,
    /// Normalized viewport `[x, y, width, height]` in 0..1, from the top left; `None` = full
    /// target. Its edges are rounded to whole pixels (the same on every backend; neighbouring
    /// viewports share edges exactly); one thinner than a pixel keeps the pixel holding its
    /// center. The camera clears and draws only inside it, so several cameras can share a
    /// target (split screen, picture-in-picture, minimaps).
    pub viewport: Option<[f32; 4]>,
    /// Inactive cameras are skipped.
    pub active: bool,
}

impl Default for Camera3d {
    fn default() -> Self {
        Self {
            projection: Projection::default(),
            clear_color: Some(Vec4::new(0.02, 0.02, 0.03, 1.0)),
            target: RenderTarget::Primary,
            order: 0,
            viewport: None,
            active: true,
        }
    }
}

impl Camera3d {
    /// Camera with a given clear color.
    pub fn with_clear_color(mut self, color: Vec4) -> Self {
        self.clear_color = Some(color);
        self
    }
    /// Camera with a given projection.
    pub fn with_projection(mut self, projection: Projection) -> Self {
        self.projection = projection;
        self
    }
    /// Camera rendering to a texture target.
    pub fn with_target(mut self, target: RenderTarget) -> Self {
        self.target = target;
        self
    }
}

/// Sun-like light shining along the entity's forward (-Z) axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DirectionalLight {
    /// Linear color.
    pub color: Vec3,
    /// Intensity multiplier.
    pub intensity: f32,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            color: Vec3::ONE,
            intensity: 1.0,
        }
    }
}

/// Omni light at the entity's position with smooth falloff to zero at `range`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointLight {
    /// Linear color.
    pub color: Vec3,
    /// Intensity multiplier.
    pub intensity: f32,
    /// Distance at which the light fades to zero.
    pub range: f32,
}

impl Default for PointLight {
    fn default() -> Self {
        Self {
            color: Vec3::ONE,
            intensity: 1.0,
            range: 10.0,
        }
    }
}

/// Global ambient term (resource). Defaults to a dim neutral light.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AmbientLight {
    /// Linear color.
    pub color: Vec3,
    /// Intensity multiplier.
    pub intensity: f32,
}

impl Default for AmbientLight {
    fn default() -> Self {
        Self {
            color: Vec3::ONE,
            intensity: 0.05,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn material_extensions_default_to_the_legacy_look() {
        let m = Material::default();
        assert_eq!(
            (m.normal_scale, m.occlusion_strength, m.alpha_cutoff, m.render_order),
            (1.0, 1.0, 0.5, 0)
        );
        assert_eq!(
            (m.blend, m.shadows, m.uv1_maps),
            (BlendMode::Opaque, ShadowMode::Auto, MapMask::NONE)
        );
        assert_eq!(m.effective_blend(), BlendMode::Opaque);
        let legacy = Material {
            alpha_blend: true,
            ..Material::default()
        };
        assert_eq!(legacy.effective_blend(), BlendMode::AlphaBlend);
        let t = TextureHandle::from_raw(crate::handles::RawHandle {
            index: 1,
            generation: 0,
        });
        let orm = Material::default().with_orm_map(t).with_alpha_mask(0.3);
        assert_eq!(
            (orm.metallic_roughness_texture, orm.occlusion_texture),
            (Some(t), Some(t))
        );
        assert_eq!((orm.blend, orm.alpha_cutoff), (BlendMode::AlphaMask, 0.3));
        let glow = Material::additive(Vec4::ONE);
        assert_eq!(glow.effective_blend(), BlendMode::Additive);
        assert!(glow.unlit && glow.double_sided);
        assert_eq!(
            Material::premultiplied(Vec4::ONE).effective_blend(),
            BlendMode::Premultiplied
        );
    }

    #[test]
    fn srgb_roundtrip() {
        for v in [0.0, 0.001, 0.2, 0.5, 1.0] {
            assert!((linear_to_srgb(srgb_to_linear(v)) - v).abs() < 1e-5);
        }
    }
}
