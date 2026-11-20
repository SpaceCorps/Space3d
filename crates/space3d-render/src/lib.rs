#![warn(missing_docs)]
//! Space3d renderer: a wgpu 30 forward renderer built around instancing and minimal
//! CPU→GPU traffic. Works windowed (via `space3d-platform`) and **headless** (tests,
//! storybook, CI screenshots). See `ARCHITECTURE.md` for the full data-flow description.
//!
//! * **ECS path** — entities with [`Mesh3d`] + [`Material`] + `GlobalTransform` are batched
//!   automatically: one instanced draw per (pipeline, material, mesh), sorted to minimize
//!   state changes.
//! * **Explicit path** — [`InstancedMesh`] + user-owned [`InstanceBuffer<T>`] of custom
//!   [`InstanceData`] for 100k–1M+ instances with dirty-range uploads.
//! * **GPU-driven culling** — compute-shader frustum culling writing indirect args, drawn with
//!   `multi_draw_indexed_indirect[_count]` where supported; CPU culling + direct draws on
//!   WebGL2/downlevel ([`DrawPath`]).
//! * **Custom shaders** — [`CustomShader`] WGSL hooks per [`Material`], with access to custom
//!   per-instance attributes ([`InstanceSemantic::Custom`]).
//! * **Headless capture** — [`Renderer::capture`] / [`render_to_image`].
//! * **Overlays** — [`RenderOverlay`]s registered in [`RenderOverlays`] draw after the 3D scene
//!   into the final target (window surface or capture target), e.g. the egui UI — or, with
//!   [`OverlayStage::HdrScene`], into the HDR scene color before post.
//! * **Render nodes** — [`RenderNode`]s registered in [`RenderNodes`] draw into the frame's own
//!   passes per phase (Opaque, Skybox, Transparent, Distortion), optionally reading scene depth.
//!
//! ```no_run
//! use space3d_core::prelude::*;
//! use space3d_render::*;
//! let mut app = App::new();
//! app.add_plugin(RenderPlugin::headless(256, 256));
//! let mesh = app.resources.expect_mut::<RenderAssets>().add_mesh(MeshData::cube(1.0));
//! app.world.spawn((Transform::default(), Mesh3d(mesh), Material::srgb(0.8, 0.2, 0.2)));
//! app.world.spawn((Transform::from_xyz(0.0, 0.0, 3.0), Camera3d::default()));
//! let image = render_to_image(&mut app).expect("GPU available");
//! assert_eq!((image.width, image.height), (256, 256));
//! ```

mod arena;
mod assets;
mod caps;
mod components;
mod culling;
mod dirty;
mod dynamic_resolution;
mod frame;
mod gpu;
mod handles;
mod instance;
mod lighting;
pub mod lod;
mod material;
mod node;
mod output;
mod overlay;
mod pipeline;
mod post;
mod quality;
mod readback;
mod renderer;
mod settings;
mod shader;
mod stats;
#[doc(hidden)]
pub mod testing;
mod texture;
mod timing;
mod unproject;

pub use assets::{AddressMode, FilterMode, RenderAssets, SamplerSettings, TextureSource};
pub use components::{
    AmbientLight, Camera3d, DirectionalLight, Material, Mesh3d, PointLight, Projection, RenderTarget, Visibility,
    linear_to_srgb, srgb_to_linear,
};
pub use culling::{Aabb, Frustum, Sphere, max_axis_scale};
pub use dirty::DirtyRanges;
pub use dynamic_resolution::DynamicResolution;
pub use frame::RenderSettings;
pub use handles::{InstanceBufferHandle, MeshHandle, ShaderHandle, TextureHandle};
pub use instance::{
    BufferUsage, CullFloat, CullSource, InstanceAttribute, InstanceBuffer, InstanceBuffers, InstanceCompact,
    InstanceData, InstanceHalf, InstanceLayout, InstancePacked, InstanceSemantic, InstanceTransform,
    InstanceTransformColor, InstancedMesh, MAX_INSTANCE_LOCATIONS, ScaleKind, f16_to_f32, f32_to_f16,
};
pub use overlay::{
    OverlayScene, OverlayStage, OverlayTarget, OverlayTargetKind, RENDER_TARGET_FORMAT, RenderOverlay, RenderOverlays,
    SharedOverlay, lock_overlay,
};
pub use renderer::{
    DisplayHandle, PrewarmRequest, RenderCapabilities, RenderError, Renderer, RendererConfig, RendererDevice,
};
pub use shader::CustomShader;
pub use stats::{DrawPath, RenderStats};

// Graphics API hubs (graphics spec §2.1): each area adds items to its hub, never to this file.
pub use caps::GraphicsCaps;
pub use lighting::api::*;
pub use lod::{LodSettings, MeshLod};
pub use material::*;
pub use node::{
    DepthAccess, DrawContext, FULLSCREEN_WGSL, FrameInfo, FrameKind, FrameLayouts, FrameMode, MESH_VERTEX_LAYOUT,
    MeshRef, PassTarget, PhaseSlot, PrepareContext, RenderNode, RenderNodes, RenderPhase, SceneDepthRef, SceneReads,
    SharedNode, SlotMask, TextureRef, VIEW_WGSL, ViewInfo, ViewPrepareContext, ViewRole, lock_node, scene_wgsl,
    scene_wgsl_for,
};
pub use output::{OUTPUT_WGSL, OutputEncoding, Tonemapper};
pub use post::api::*;
pub use quality::GraphicsQuality;
pub use settings::*;
pub use timing::{GpuScope, GpuTimings};
pub use unproject::{inverse_proj_rot, unproject_ray};

pub use glam;
pub use wgpu;

use space3d_core::{App, ImageData, Plugin, Stage, SystemCtx};

/// Registers render resources and the `Stage::Render` system.
///
/// The [`Renderer`] itself is inserted separately: by the platform runner once a window
/// exists, by [`RenderPlugin::headless`], or by [`init_headless`]. Without a renderer the
/// render system is a no-op, so simulation-only apps and tests keep working.
#[derive(Clone, Debug, Default)]
pub struct RenderPlugin {
    /// Create a headless renderer of this size during `build` (native only).
    pub headless: Option<(u32, u32)>,
}

impl RenderPlugin {
    /// Plugin that also creates a headless `width`x`height` renderer (skipped with a warning
    /// when no GPU adapter is available).
    pub fn headless(width: u32, height: u32) -> Self {
        Self {
            headless: Some((width, height)),
        }
    }
}

impl Plugin for RenderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RenderAssets>()
            .init_resource::<InstanceBuffers>()
            .init_resource::<AmbientLight>()
            .init_resource::<RenderStats>()
            .init_resource::<RenderSettings>()
            .init_resource::<RenderOverlays>()
            .init_resource::<FrameSettings>()
            .init_resource::<UploadBudget>()
            .init_resource::<LightingSettings>()
            .init_resource::<RenderNodes>()
            .init_resource::<GpuTimings>()
            .init_resource::<DynamicResolution>()
            .init_resource::<dynamic_resolution::FrameClock>();
        let nodes = app.resources.expect_mut::<RenderNodes>();
        lighting::Lighting::register_nodes(nodes);
        post::PostState::register_nodes(nodes);
        app.add_system_named(Stage::Render, "render::dynamic_resolution", |ctx| {
            dynamic_resolution::dynamic_resolution_system(ctx.resources)
        });
        // Before the frame is extracted: the meshes it draws are the levels picked here.
        app.add_system_named(Stage::Render, "render::select_lods", |ctx| {
            lod::select_lods(ctx.world, ctx.resources)
        });
        app.add_system_named(Stage::Render, "render::render_frame", render_system);
        if let Some((w, h)) = self.headless {
            #[cfg(not(target_arch = "wasm32"))]
            if let Err(e) = init_headless(app, w, h) {
                tracing::warn!("headless renderer unavailable: {e}");
            }
            #[cfg(target_arch = "wasm32")]
            tracing::warn!(
                "RenderPlugin::headless({w}, {h}) is not supported on the web; insert a Renderer asynchronously"
            );
        }
    }
}

/// The `Stage::Render` system: extract the world and draw a frame if a [`Renderer`] exists.
pub fn render_system(ctx: &mut SystemCtx) {
    if !ctx.resources.contains::<Renderer>() {
        return;
    }
    let world = &*ctx.world;
    // `scope` moves the resource's existing Box out and back: no per-frame allocation.
    ctx.resources
        .scope::<Renderer, _>(|resources, renderer| renderer.render_frame(world, resources));
}

/// Create a headless renderer (blocking on adapter/device creation) and insert it.
#[cfg(not(target_arch = "wasm32"))]
pub fn init_headless(app: &mut App, width: u32, height: u32) -> Result<(), RenderError> {
    let renderer = futures::executor::block_on(Renderer::new_headless(width, height))?;
    app.insert_resource(renderer);
    Ok(())
}

/// Render the app's current state (without advancing time) and read back the primary target.
///
/// Transforms are propagated first, then one frame is rendered and captured synchronously.
/// Returns `None` when there is no [`Renderer`] (e.g. no GPU) or on the web, where blocking
/// is impossible — use [`Renderer::capture`] there. Deterministic for identical inputs on the
/// same machine.
pub fn render_to_image(app: &mut App) -> Option<ImageData> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = app;
        None
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        space3d_core::transform::propagate_transforms(&mut app.world);
        let mut renderer = app.resources.remove::<Renderer>()?;
        renderer.render_frame(&app.world, &mut app.resources);
        let image = renderer.capture_blocking();
        app.resources.insert(renderer);
        match image {
            Ok(img) => Some(img),
            Err(e) => {
                tracing::error!("render_to_image: {e}");
                None
            }
        }
    }
}

/// Common imports.
///
/// Graphics additions are limited to a curated, collision-checked list (graphics spec §2.2.10:
/// `PostProcess`, `Tonemapper`, `Bloom`, `CameraFrame`, `Skybox`, `EnvironmentLight`,
/// `DirectionalShadow`, `CubemapHandle`); everything else is reachable through the crate root.
pub mod prelude {
    pub use crate::{
        AmbientLight, BufferUsage, Camera3d, CustomShader, DirectionalLight, DrawPath, InstanceBuffer, InstanceBuffers,
        InstanceCompact, InstanceData, InstanceHalf, InstanceTransform, InstanceTransformColor, InstancedMesh,
        Material, Mesh3d, MeshHandle, OverlayTarget, PointLight, Projection, RenderAssets, RenderOverlay,
        RenderOverlays, RenderPlugin, RenderSettings, RenderStats, RenderTarget, Renderer, ShaderHandle, TextureHandle,
        Visibility, render_to_image,
    };
    pub use crate::{
        Bloom, CameraFrame, CubemapHandle, DirectionalShadow, EnvironmentLight, PostProcess, Skybox, Tonemapper,
    };
}

/// `SPACE3D_NO_GPU=1` makes adapter acquisition fail as if the machine had no GPU: every GPU
/// test then takes its documented skip path. Used on hosted CI runners, whose software or
/// paravirtual GPUs (lavapipe, WARP) crash or render differently from real hardware; real GPU
/// runners and developers leave it unset.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn gpu_disabled_by_env() -> bool {
    std::env::var_os("SPACE3D_NO_GPU").is_some_and(|v| !v.is_empty() && v != "0")
}

/// Never set on the web.
#[cfg(target_arch = "wasm32")]
pub(crate) fn gpu_disabled_by_env() -> bool {
    false
}
