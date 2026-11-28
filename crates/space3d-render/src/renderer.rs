//! [`Renderer`]: device/queue ownership, surfaces, render targets, asset upload and capture.

use crate::assets::RenderAssets;
use crate::frame::FrameState;
use crate::gpu::{Uploader, align_up};
use crate::pipeline::{DEPTH_FORMAT, OFFSCREEN_FORMAT, Pipelines};
use crate::stats::{DrawPath, RenderStats};
use crate::texture::{SamplerCache, create_gpu_texture, format_supported, white_texture};
use space3d_core::ImageData;
use space3d_tasks::Task;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Renderer errors.
#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    /// No suitable GPU adapter (headless CI without GPU, unsupported browser...).
    #[error("no suitable GPU adapter: {0}")]
    NoAdapter(String),
    /// Device creation failed.
    #[error("failed to create device: {0}")]
    Device(String),
    /// Surface creation/configuration failed.
    #[error("surface error: {0}")]
    Surface(String),
    /// GPU readback failed.
    #[error("capture failed: {0}")]
    Capture(String),
}

/// Construction options.
#[derive(Clone, Debug)]
pub struct RendererConfig {
    /// Initial width of the primary target (window surface or offscreen).
    pub width: u32,
    /// Initial height of the primary target.
    pub height: u32,
    /// MSAA samples for the primary target (1 or 4; clamped to what the device supports).
    pub msaa_samples: u32,
    /// Wait for vblank when presenting.
    pub vsync: bool,
    /// Adapter preference.
    pub power_preference: wgpu::PowerPreference,
    /// Restrict backends (`None` = platform default, overridable with `WGPU_BACKEND`).
    pub backends: Option<wgpu::Backends>,
    /// Use a software adapter (e.g. lavapipe/WARP) if available.
    pub force_fallback_adapter: bool,
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 720,
            msaa_samples: 1,
            vsync: true,
            power_preference: wgpu::PowerPreference::HighPerformance,
            backends: None,
            force_fallback_adapter: false,
        }
    }
}

/// What the adapter can do and which draw path was selected.
#[derive(Clone, Debug)]
pub struct RenderCapabilities {
    /// Adapter description.
    pub adapter: wgpu::AdapterInfo,
    /// Compute shaders + enough storage buffers for GPU culling.
    pub compute: bool,
    /// Indirect draws supported.
    pub indirect: bool,
    /// `first_instance` honoured by indirect draws.
    pub indirect_first_instance: bool,
    /// `multi_draw_indexed_indirect_count` available.
    pub multi_draw_indirect_count: bool,
    /// Non-zero `base_vertex` in direct indexed draws (`DownlevelFlags::BASE_VERTEX`). Missing
    /// on WebGL2 and GLES < 3.2, where wgpu does not validate it and the GL backend calls an
    /// entry point that does not exist (a panic on the web). Without it, direct draws rebind
    /// the arena's vertex buffer at each mesh's first vertex and draw with `base_vertex = 0`,
    /// and GPU-driven paths are not offered.
    pub base_vertex: bool,
    /// BC / ASTC / ETC2 compressed textures.
    pub texture_compression: (bool, bool, bool),
    /// Best draw path for this device.
    pub best_draw_path: DrawPath,
    /// MSAA sample counts usable on the primary target.
    pub max_msaa: u32,
    /// Largest storage binding (bounds GPU-culled buffers).
    pub max_storage_binding: u64,
    /// Largest buffer the device can create (bounds the mesh arena and instance buffers).
    pub max_buffer_size: u64,
}

/// The device of a [`Renderer`] held out of the world between frames (resource): the
/// render-on-demand mode of `space3d_test::TestApp` keeps its renderer aside while the app
/// ticks and inserts this instead, so tick-time state sized by the device (the particle
/// budget, the off-screen LOD's aspect ratio) is the same as with the renderer in the world:
/// render-every-tick equals render-on-demand (graphics spec §4.3). Systems read the
/// [`Renderer`] first and fall back to this.
#[derive(Clone, Debug)]
pub struct RendererDevice {
    /// [`Renderer::capabilities`].
    pub capabilities: RenderCapabilities,
    /// [`Renderer::graphics_caps`].
    pub graphics: crate::caps::GraphicsCaps,
    /// [`Renderer::size`].
    pub size: (u32, u32),
}

impl RendererDevice {
    /// The device of `renderer`.
    pub fn of(renderer: &Renderer) -> Self {
        Self {
            capabilities: renderer.capabilities().clone(),
            graphics: renderer.graphics_caps().clone(),
            size: renderer.size(),
        }
    }
}

impl RenderCapabilities {
    fn detect(adapter: &wgpu::Adapter, device: &wgpu::Device) -> Self {
        let dl = adapter.get_downlevel_capabilities();
        let limits = device.limits();
        let features = device.features();
        let compute = dl.flags.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
            && limits.max_storage_buffers_per_shader_stage >= 4
            && limits.max_compute_workgroup_size_x >= 64
            && limits.max_compute_invocations_per_workgroup >= 64;
        let indirect = dl.flags.contains(wgpu::DownlevelFlags::INDIRECT_EXECUTION);
        let base_vertex = dl.flags.contains(wgpu::DownlevelFlags::BASE_VERTEX);
        let ifi = features.contains(wgpu::Features::INDIRECT_FIRST_INSTANCE);
        let mdic = features.contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT);
        let best_draw_path = best_draw_path(compute && indirect, base_vertex, ifi, mdic);
        let msaa4 = adapter
            .get_texture_format_features(OFFSCREEN_FORMAT)
            .flags
            .sample_count_supported(4)
            && adapter
                .get_texture_format_features(DEPTH_FORMAT)
                .flags
                .sample_count_supported(4);
        Self {
            adapter: adapter.get_info(),
            compute,
            indirect,
            indirect_first_instance: ifi,
            multi_draw_indirect_count: mdic,
            base_vertex,
            texture_compression: (
                features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC),
                features.contains(wgpu::Features::TEXTURE_COMPRESSION_ASTC),
                features.contains(wgpu::Features::TEXTURE_COMPRESSION_ETC2),
            ),
            best_draw_path,
            max_msaa: if msaa4 { 4 } else { 1 },
            // Storage bindings must be a multiple of 4 bytes and never exceed the buffer.
            max_storage_binding: limits.max_storage_buffer_binding_size.min(limits.max_buffer_size) & !3,
            max_buffer_size: limits.max_buffer_size,
        }
    }
}

/// Best draw path for a device. GPU-driven paths need compute + indirect draws, and are only
/// offered with `BASE_VERTEX`: devices without it (WebGL2, GLES 3.0/3.1) are the downlevel
/// targets `CpuDirect` exists for, and it is the path whose direct draws emulate `base_vertex`.
fn best_draw_path(compute_indirect: bool, base_vertex: bool, first_instance: bool, multi_count: bool) -> DrawPath {
    match (compute_indirect && base_vertex, first_instance, multi_count) {
        (true, true, true) => DrawPath::GpuMultiDrawCount,
        (true, true, false) => DrawPath::GpuMultiDraw,
        (true, false, _) => DrawPath::GpuIndirect,
        _ => DrawPath::CpuDirect,
    }
}

/// Color/MSAA/depth attachments for one render target size.
pub(crate) struct Attachments {
    pub width: u32,
    pub height: u32,
    pub samples: u32,
    pub format: wgpu::TextureFormat,
    /// Single-sample color texture (offscreen only; surfaces provide their own).
    pub color: Option<wgpu::Texture>,
    pub color_view: Option<wgpu::TextureView>,
    pub msaa_view: Option<wgpu::TextureView>,
    pub depth_view: wgpu::TextureView,
    /// A multisampled depth texture here is sampleable (`GraphicsCaps::msaa_depth_resolve`).
    pub msaa_depth_binding: bool,
}

impl Attachments {
    /// Color (offscreen only), MSAA color and depth attachments. The depth texture can be
    /// sampled (`TEXTURE_BINDING`, graphics spec §1.6) at 1 sample, and multisampled only with
    /// `msaa_depth_binding` (`GraphicsCaps::msaa_depth_resolve`: GL cannot sample multisampled
    /// depth, and a sampled depth texture there is a texture instead of a renderbuffer).
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        samples: u32,
        format: wgpu::TextureFormat,
        offscreen: bool,
        msaa_depth_binding: bool,
    ) -> Self {
        let size = wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        };
        let tex = |label, format, samples, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size,
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let color = offscreen.then(|| {
            tex(
                "space3d offscreen color",
                format,
                1,
                wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
            )
        });
        let msaa_view = (samples > 1).then(|| {
            tex(
                "space3d msaa color",
                format,
                samples,
                wgpu::TextureUsages::RENDER_ATTACHMENT,
            )
            .create_view(&Default::default())
        });
        let depth_usage = if samples == 1 || msaa_depth_binding {
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING
        } else {
            wgpu::TextureUsages::RENDER_ATTACHMENT
        };
        let depth = tex("space3d depth", DEPTH_FORMAT, samples, depth_usage);
        Self {
            width: size.width,
            height: size.height,
            samples,
            format,
            msaa_depth_binding,
            color_view: color.as_ref().map(|c| c.create_view(&Default::default())),
            color,
            msaa_view,
            depth_view: depth.create_view(&Default::default()),
        }
    }
}

pub(crate) struct SurfaceState {
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    /// Format of the views frames render into: `config.format`, or its sRGB twin (listed in
    /// `config.view_formats`) when the surface itself has no sRGB format.
    pub view_format: wgpu::TextureFormat,
    /// The shader encodes sRGB itself (no sRGB format or view available).
    pub manual_srgb: bool,
    pub attachments: Attachments,
    /// A frame was acquired with the current configuration.
    pub presented: bool,
}

impl SurfaceState {
    /// Render without the sRGB view format: reconfigure the surface without it and encode sRGB
    /// in the shader instead.
    pub(crate) fn drop_srgb_view(&mut self, device: &wgpu::Device) {
        self.config.view_formats.clear();
        self.view_format = self.config.format;
        self.manual_srgb = !self.view_format.is_srgb();
        let a = &self.attachments;
        self.attachments = Attachments::new(
            device,
            a.width,
            a.height,
            a.samples,
            self.view_format,
            false,
            a.msaa_depth_binding,
        );
        self.surface.configure(device, &self.config);
    }
}

/// Whether a surface that reported `Lost` should be reconfigured without its sRGB view format
/// instead of being given up. On the web, wgpu reports a canvas configuration the browser
/// rejected as `Lost` (native wgpu validates view formats when configuring instead), so a
/// canvas that never produced a frame with a view format may have been refused because of it.
/// Recreating the surface would configure it the same way again, forever.
pub(crate) fn retry_without_view_formats(web: bool, presented: bool, config: &wgpu::SurfaceConfiguration) -> bool {
    web && !presented && !config.view_formats.is_empty()
}

/// Surface texture format and the format frames are rendered in, from a surface's supported
/// formats (`None` if there are none). An sRGB format is preferred. Browser WebGPU canvases
/// offer only `Bgra8Unorm`/`Rgba8Unorm`(/`Rgba16Float`), but allow an sRGB *view* of the
/// canvas texture: rendering through it keeps blending and MSAA resolves in linear space,
/// like native sRGB surfaces. Only formats without an sRGB twin (or devices without
/// `SURFACE_VIEW_FORMATS`) fall back to encoding sRGB in the shader, where blending then
/// operates on encoded values.
pub(crate) fn surface_formats(
    formats: &[wgpu::TextureFormat],
    view_formats_supported: bool,
) -> Option<(wgpu::TextureFormat, wgpu::TextureFormat)> {
    let first = *formats.first()?;
    if let Some(srgb) = formats.iter().copied().find(|f| f.is_srgb()) {
        return Some((srgb, srgb));
    }
    let view = first.add_srgb_suffix();
    Some((first, if view_formats_supported { view } else { first }))
}

/// The GPU renderer. Insert it as a resource (the platform runner does this for windows; use
/// [`Renderer::new_headless`] for tests/tools). Rendering happens in `Stage::Render`.
pub struct Renderer {
    pub(crate) instance: wgpu::Instance,
    pub(crate) adapter: wgpu::Adapter,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) caps: RenderCapabilities,
    pub(crate) config: RendererConfig,
    pub(crate) surface: Option<SurfaceState>,
    /// Capture target; also the primary target when headless.
    pub(crate) offscreen: Attachments,
    pub(crate) pipelines: Pipelines,
    pub(crate) samplers: SamplerCache,
    pub(crate) white: wgpu::TextureView,
    pub(crate) white_sampler: wgpu::Sampler,
    pub(crate) uploader: Uploader,
    pub(crate) frame: FrameState,
    pub(crate) stats: RenderStats,
    pub(crate) frame_index: u64,
    /// Snapshot of the `RenderOverlays` resource (see `overlay.rs`).
    pub(crate) overlays: crate::overlay::OverlayCache,
    /// Uncaptured wgpu errors so far (see [`install_error_handler`]).
    pub(crate) gpu_errors: Arc<AtomicU64>,
    /// The device was lost (`FrameInfo::device_lost`).
    pub(crate) device_lost: Arc<AtomicBool>,
    /// Graphics capabilities of the device, unmasked (`frame_graphics` is what
    /// [`Renderer::graphics_caps`] reports).
    pub(crate) graphics: crate::caps::GraphicsCaps,
    /// Shared node / overlay layouts.
    pub(crate) layouts: crate::node::FrameLayouts,
    /// Unique id of this renderer (see `FrameInfo::renderer`).
    pub(crate) id: u64,
    /// GPU pass timings (see [`Renderer::gpu_timings`]).
    pub(crate) gpu_timings: crate::timing::GpuTimings,
    /// Timestamp queries of the frame's passes (`FrameSettings::gpu_timing`).
    pub(crate) timer: crate::timing::GpuTimer,
    /// `graphics` masked by the last frame's `FrameSettings::downlevel`.
    pub(crate) frame_graphics: crate::caps::GraphicsCaps,
    /// Post-processing state: composite pipelines, per-camera history.
    pub(crate) post: crate::post::PostState,
    /// Frame-plan warnings already logged.
    pub(crate) plan_warnings: crate::frame::PlanWarnings,
    /// Snapshot of the `RenderNodes` resource and its schedule.
    pub(crate) nodes: crate::frame::NodeCache,
    /// Scene-input bind groups (group 1 of node and overlay passes) and depth resolves.
    pub(crate) scene_inputs: crate::frame::SceneInputs,
    /// The city effects' pipelines and G-buffers, created by the first camera that enables one.
    pub(crate) screen_fx: Option<crate::post::ScreenFx>,
    /// Submissions of the last [`crate::frame::MAX_FRAMES_IN_FLIGHT`] frames (slot: frame index
    /// modulo the count): `render_frame` waits for the oldest before it is replaced.
    pub(crate) in_flight: [Option<wgpu::SubmissionIndex>; crate::frame::MAX_FRAMES_IN_FLIGHT],
    /// Cubemaps, image-based lighting and shadows (graphics spec §2.6): the frame calls it at
    /// fixed points of `render_frame` and draws the depth views it requests.
    pub(crate) lighting: crate::lighting::Lighting,
    /// Depth views injected by [`crate::testing`] (renderer tests), drawn after the lighting's.
    pub(crate) injected_depth_views: Vec<crate::lighting::DepthViewRequest>,
}

/// What [`Renderer::prewarm`] builds, in addition to the post pipelines of the current post
/// cameras and every node's `prewarm` for the pass targets reachable in the current modes.
#[derive(Clone, Debug, Default)]
pub struct PrewarmRequest {
    /// Materials whose forward pipelines to build.
    pub materials: Vec<crate::Material>,
    /// Instance layouts to build them for (the ECS layout is always included).
    pub layouts: Vec<&'static crate::InstanceLayout>,
    /// Extra color formats to build for (e.g. the capture format when windowed).
    pub formats: Vec<wgpu::TextureFormat>,
}

/// Renderer ids: unique per process.
fn next_renderer_id() -> u64 {
    static ID: AtomicU64 = AtomicU64::new(1);
    ID.fetch_add(1, Ordering::Relaxed)
}

impl std::fmt::Debug for Renderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Renderer")
            .field("adapter", &self.caps.adapter.name)
            .field("backend", &self.caps.adapter.backend)
            .field("size", &self.size())
            .field("draw_path", &self.caps.best_draw_path)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
thread_local! {
    /// `(max_buffer_size, max_storage_buffer_binding_size)` requested instead of the adapter's
    /// limits (when lower) by renderers created on this thread, so tests run against a device that really
    /// enforces smaller limits (wgpu validation fails past them) — see
    /// `frame::tests::limited_app`.
    pub(crate) static TEST_LIMITS: std::cell::Cell<Option<(u64, u64)>> = const { std::cell::Cell::new(None) };
}

/// Route wgpu errors that no error scope captured through `tracing` instead of wgpu's default
/// handler, which panics. With `panic` (debug builds and tests) the error still panics after
/// being logged so bugs surface immediately; release builds log and keep running — skipping a
/// broken command buffer is better than crashing the game. Errors tend to repeat every frame,
/// so only the first ones and then every 1000th are logged.
///
/// Once the device is lost (driver reset, GPU removed, `destroy`), every resource created
/// afterwards is invalid and each use of one is reported as an error. Those are the expected
/// consequence of the loss, not bugs, so from the device-lost callback on they are only
/// logged, in debug builds too: a lost device degrades (captures fail, frames are dropped)
/// instead of crashing.
pub(crate) fn install_error_handler(device: &wgpu::Device, panic: bool) -> (Arc<AtomicU64>, Arc<AtomicBool>) {
    let count = Arc::new(AtomicU64::new(0));
    let lost = Arc::new(AtomicBool::new(false));
    let (counter, lost_seen, lost_flag) = (count.clone(), lost.clone(), lost.clone());
    device.on_uncaptured_error(Arc::new(move |err: wgpu::Error| {
        let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
        let lost = lost_seen.load(Ordering::Acquire);
        if n <= 16 || n.is_multiple_of(1000) {
            if lost {
                tracing::warn!(occurrences = n, "wgpu error after the device was lost: {err}");
            } else {
                tracing::error!(occurrences = n, "wgpu error: {err}");
            }
        }
        if panic && !lost {
            panic!("wgpu error: {err}");
        }
    }));
    device.set_device_lost_callback(move |reason, message| {
        lost.store(true, Ordering::Release);
        match reason {
            wgpu::DeviceLostReason::Destroyed => tracing::debug!(%message, "GPU device destroyed"),
            _ => tracing::error!(?reason, %message, "GPU device lost"),
        }
    });
    (count, lost_flag)
}

/// Primary target size within `1..=max_dimension` (the device's `max_texture_dimension_2d`):
/// a larger surface or offscreen texture fails validation, which invalidates the frame's
/// whole command buffer, every frame. Oversized requests are clamped per axis and logged.
pub(crate) fn target_size(width: u32, height: u32, max_dimension: u32) -> (u32, u32) {
    let clamped = (width.clamp(1, max_dimension), height.clamp(1, max_dimension));
    if clamped.0 < width || clamped.1 < height {
        tracing::warn!(
            requested = ?(width, height),
            used = ?clamped,
            "primary target larger than the device's max_texture_dimension_2d; clamped"
        );
    }
    clamped
}

/// Display connection handed to wgpu (needed by GLES on Wayland/X11; unused elsewhere).
pub type DisplayHandle = Box<dyn wgpu::wgt::WgpuHasDisplayHandle>;

async fn create_instance(backends: Option<wgpu::Backends>, display: Option<DisplayHandle>) -> wgpu::Instance {
    let mut desc = match display {
        Some(d) => wgpu::InstanceDescriptor::new_with_display_handle_from_env(d),
        None => wgpu::InstanceDescriptor::new_without_display_handle_from_env(),
    };
    if let Some(b) = backends {
        desc.backends = b;
    }
    // Android: a debug build asks for wgpu's debug flags, which name every Vulkan object
    // (VK_EXT_debug_utils). The emulator's Vulkan driver crashes in that call
    // (vk_common_SetDebugUtilsObjectNameEXT, a null read while the device is created), and a
    // phone has no validation layers installed anyway. Off unless WGPU_DEBUG / WGPU_VALIDATION ask.
    #[cfg(target_os = "android")]
    {
        desc.flags = wgpu::InstanceFlags::empty().with_env();
    }
    #[cfg(target_arch = "wasm32")]
    {
        wgpu::util::new_instance_with_webgpu_detection(desc).await
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        wgpu::Instance::new(desc)
    }
}

impl Renderer {
    /// Headless renderer with an offscreen `width`x`height` target (default settings).
    pub async fn new_headless(width: u32, height: u32) -> Result<Self, RenderError> {
        Self::new_headless_with(RendererConfig {
            width,
            height,
            ..Default::default()
        })
        .await
    }

    /// Headless renderer with explicit configuration.
    pub async fn new_headless_with(config: RendererConfig) -> Result<Self, RenderError> {
        let instance = create_instance(config.backends, None).await;
        Self::from_instance(instance, None, config).await
    }

    /// Renderer presenting to a window/canvas. `target` is typically an `Arc<winit::Window>`
    /// (native) or a `wgpu::SurfaceTarget::Canvas` (web).
    pub async fn new_with_surface(
        target: impl Into<wgpu::SurfaceTarget<'static>>,
        config: RendererConfig,
    ) -> Result<Self, RenderError> {
        Self::new_with_surface_and_display(target, None, config).await
    }

    /// Like [`Self::new_with_surface`], also passing the platform display connection (e.g.
    /// winit's `OwnedDisplayHandle`), which the GLES backend needs on Wayland/X11.
    pub async fn new_with_surface_and_display(
        target: impl Into<wgpu::SurfaceTarget<'static>>,
        display: Option<DisplayHandle>,
        config: RendererConfig,
    ) -> Result<Self, RenderError> {
        let instance = create_instance(config.backends, display).await;
        let surface = instance
            .create_surface(target)
            .map_err(|e| RenderError::Surface(e.to_string()))?;
        Self::from_instance(instance, Some(surface), config).await
    }

    async fn from_instance(
        instance: wgpu::Instance,
        surface: Option<wgpu::Surface<'static>>,
        mut config: RendererConfig,
    ) -> Result<Self, RenderError> {
        if crate::gpu_disabled_by_env() {
            return Err(RenderError::NoAdapter("SPACE3D_NO_GPU is set".into()));
        }
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: config.power_preference,
                force_fallback_adapter: config.force_fallback_adapter,
                compatible_surface: surface.as_ref(),
                apply_limit_buckets: false,
            })
            .await
            .map_err(|e| RenderError::NoAdapter(e.to_string()))?;
        let wanted = wgpu::Features::INDIRECT_FIRST_INSTANCE
            | wgpu::Features::MULTI_DRAW_INDIRECT_COUNT
            | wgpu::Features::FLOAT32_FILTERABLE
            | wgpu::Features::TEXTURE_COMPRESSION_BC
            | wgpu::Features::TEXTURE_COMPRESSION_ASTC
            | wgpu::Features::TEXTURE_COMPRESSION_ETC2
            | crate::caps::WANTED_FEATURES;
        #[cfg_attr(not(test), expect(unused_mut))]
        let mut required_limits = adapter.limits();
        #[cfg(test)]
        if let Some((buffer, binding)) = TEST_LIMITS.get() {
            required_limits.max_buffer_size = buffer.min(required_limits.max_buffer_size);
            required_limits.max_storage_buffer_binding_size =
                binding.min(required_limits.max_storage_buffer_binding_size);
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("space3d device"),
                required_features: adapter.features() & wanted,
                required_limits,
                ..Default::default()
            })
            .await
            .map_err(|e| RenderError::Device(e.to_string()))?;
        let (gpu_errors, device_lost) = install_error_handler(&device, cfg!(debug_assertions));
        let caps = RenderCapabilities::detect(&adapter, &device);
        tracing::info!(
            adapter = %caps.adapter.name,
            backend = ?caps.adapter.backend,
            draw_path = ?caps.best_draw_path,
            "space3d renderer initialised"
        );
        config.msaa_samples = if config.msaa_samples > 1 { caps.max_msaa } else { 1 };
        (config.width, config.height) =
            target_size(config.width, config.height, device.limits().max_texture_dimension_2d);
        let graphics = crate::caps::GraphicsCaps::detect(&adapter, &device, &caps);
        let offscreen = Attachments::new(
            &device,
            config.width,
            config.height,
            config.msaa_samples,
            OFFSCREEN_FORMAT,
            true,
            graphics.msaa_depth_resolve,
        );
        // wgpu's GL backend compiles GLSL ES 3.00 on WebGL2 and GLES 3.0 (no compute shaders);
        // GLES 3.1+ (compute) has what custom shaders may use beyond it.
        let glsl_es_300 = caps.adapter.backend == wgpu::Backend::Gl
            && !adapter
                .get_downlevel_capabilities()
                .flags
                .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS);
        let mut pipelines = Pipelines::new(&device, caps.compute && caps.indirect, glsl_es_300);
        pipelines.depth_bias_clamp = graphics.depth_bias_clamp;
        let white = white_texture(&device, &queue);
        let mut samplers = SamplerCache::default();
        let white_sampler = samplers.get(&device, Default::default());
        let frame = FrameState::new(&caps, &device);
        let uploader = Uploader::new(&device);
        let layouts = crate::node::FrameLayouts::new(&device, &pipelines.view_bgl);
        let gpu_timings = crate::timing::GpuTimings {
            supported: graphics.timestamps,
            ..Default::default()
        };
        let post = crate::post::PostState::new(&device, &graphics);
        let scene_inputs = crate::frame::SceneInputs::new(&device, &queue, &layouts);
        let lighting = crate::lighting::Lighting::new(&device, &queue, &graphics);
        let timer = crate::timing::GpuTimer::new(graphics.timestamps, queue.get_timestamp_period());
        let mut renderer = Self {
            instance,
            adapter,
            device,
            queue,
            caps,
            config,
            surface: None,
            offscreen,
            pipelines,
            samplers,
            white,
            white_sampler,
            uploader,
            frame,
            stats: RenderStats::default(),
            frame_index: 0,
            overlays: Default::default(),
            gpu_errors,
            device_lost,
            layouts,
            id: next_renderer_id(),
            gpu_timings,
            timer,
            frame_graphics: graphics.clone(),
            graphics,
            post,
            plan_warnings: Default::default(),
            nodes: Default::default(),
            scene_inputs,
            screen_fx: None,
            in_flight: Default::default(),
            lighting,
            injected_depth_views: Vec::new(),
        };
        if let Some(surface) = surface {
            renderer.configure_surface(surface)?;
        }
        Ok(renderer)
    }

    fn configure_surface(&mut self, surface: wgpu::Surface<'static>) -> Result<(), RenderError> {
        let caps = surface.get_capabilities(&self.adapter);
        let view_formats_supported = self
            .adapter
            .get_downlevel_capabilities()
            .flags
            .contains(wgpu::DownlevelFlags::SURFACE_VIEW_FORMATS);
        let Some((format, view_format)) = surface_formats(&caps.formats, view_formats_supported) else {
            return Err(RenderError::Surface("surface not supported by adapter".into()));
        };
        let present_mode = if self.config.vsync {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        };
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: self.config.width,
            height: self.config.height,
            present_mode,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps
                .alpha_modes
                .first()
                .copied()
                .unwrap_or(wgpu::CompositeAlphaMode::Auto),
            view_formats: if view_format != format {
                vec![view_format]
            } else {
                vec![]
            },
        };
        surface.configure(&self.device, &config);
        let samples = if self.config.msaa_samples > 1
            && self
                .adapter
                .get_texture_format_features(view_format)
                .flags
                .sample_count_supported(self.config.msaa_samples)
        {
            self.config.msaa_samples
        } else {
            1
        };
        // The MSAA and depth attachments are separate textures: they use the view format.
        let attachments = Attachments::new(
            &self.device,
            config.width,
            config.height,
            samples,
            view_format,
            false,
            self.graphics.msaa_depth_resolve,
        );
        self.surface = Some(SurfaceState {
            surface,
            view_format,
            manual_srgb: !view_format.is_srgb(),
            config,
            attachments,
            presented: false,
        });
        Ok(())
    }

    /// Attach (or replace) the presentation surface — e.g. on Android/iOS resume.
    pub fn set_surface(&mut self, target: impl Into<wgpu::SurfaceTarget<'static>>) -> Result<(), RenderError> {
        let surface = self
            .instance
            .create_surface(target)
            .map_err(|e| RenderError::Surface(e.to_string()))?;
        if !self.adapter.is_surface_supported(&surface) {
            return Err(RenderError::Surface("adapter cannot present to this surface".into()));
        }
        self.configure_surface(surface)
    }

    /// Drop the presentation surface (Android suspend). Rendering continues offscreen.
    pub fn remove_surface(&mut self) {
        self.surface = None;
    }

    /// Whether a presentation surface is attached.
    pub fn has_surface(&self) -> bool {
        self.surface.is_some()
    }

    /// Resize the primary target. The window surface is reconfigured now; the headless
    /// offscreen target is reallocated by the next frame, so [`Self::capture`] until then
    /// still returns the last rendered frame. Each side is clamped to the device's
    /// `max_texture_dimension_2d` (see [`Self::size`]).
    pub fn resize(&mut self, width: u32, height: u32) {
        let (width, height) = target_size(width, height, self.device.limits().max_texture_dimension_2d);
        self.config.width = width;
        self.config.height = height;
        if let Some(s) = &mut self.surface {
            s.config.width = width;
            s.config.height = height;
            s.surface.configure(&self.device, &s.config);
            s.attachments = Attachments::new(
                &self.device,
                width,
                height,
                s.attachments.samples,
                s.view_format,
                false,
                self.graphics.msaa_depth_resolve,
            );
        }
    }

    /// Headless: (re)allocate the offscreen target to the configured size and sample count.
    /// Called at the start of a frame, never between a frame and its capture.
    pub(crate) fn ensure_offscreen(&mut self) {
        let windowed = self.surface.is_some();
        // A test renderer standing in for a window whose surface is lost this frame.
        #[cfg(test)]
        let windowed = windowed || self.frame.simulate_surface_loss;
        if windowed {
            // Windowed: the offscreen target only serves captures and frames whose surface
            // was lost, which size it themselves (`size_offscreen`).
            return;
        }
        self.size_offscreen((self.config.width, self.config.height));
    }

    /// (Re)allocate the offscreen target at `size` with the configured sample count, unless it
    /// already has them.
    pub(crate) fn size_offscreen(&mut self, (w, h): (u32, u32)) {
        let (w, h, samples) = (w.max(1), h.max(1), self.config.msaa_samples);
        let a = &self.offscreen;
        if (a.width, a.height, a.samples) != (w, h, samples) {
            self.offscreen = Attachments::new(
                &self.device,
                w,
                h,
                samples,
                OFFSCREEN_FORMAT,
                true,
                self.graphics.msaa_depth_resolve,
            );
        }
    }

    /// Change MSAA sample count (1 or 4; clamped to device support). Surface attachments are
    /// recreated now, the headless offscreen target on the next frame.
    pub fn set_msaa_samples(&mut self, samples: u32) {
        let samples = if samples > 1 { self.caps.max_msaa } else { 1 };
        if samples == self.config.msaa_samples {
            return;
        }
        self.config.msaa_samples = samples;
        if let Some(s) = &mut self.surface {
            let ok = self
                .adapter
                .get_texture_format_features(s.view_format)
                .flags
                .sample_count_supported(samples);
            let samples = if ok { samples } else { 1 };
            s.attachments = Attachments::new(
                &self.device,
                s.config.width,
                s.config.height,
                samples,
                s.view_format,
                false,
                self.graphics.msaa_depth_resolve,
            );
        }
    }

    /// MSAA sample count of the primary target.
    pub fn msaa_samples(&self) -> u32 {
        self.config.msaa_samples
    }

    /// Current primary target size (as requested, within the device's texture size limit).
    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// The wgpu device.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// The wgpu queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Adapter capabilities and chosen draw path.
    pub fn capabilities(&self) -> &RenderCapabilities {
        &self.caps
    }

    /// Stats of the last rendered frame.
    pub fn stats(&self) -> &RenderStats {
        &self.stats
    }

    /// Format frames are rendered in when windowed (for overlays drawing into the same view):
    /// the surface format, or its sRGB view format where the surface itself has no sRGB format
    /// (browser WebGPU canvases). It changes if a browser rejects the sRGB view format, so
    /// overlays should check it every frame.
    pub fn surface_format(&self) -> Option<wgpu::TextureFormat> {
        self.surface.as_ref().map(|s| s.view_format)
    }

    /// Whether textures of `format` can be uploaded and sampled by materials: compressed
    /// formats need device features, and the format must be filterable float (integer and
    /// depth formats never are; 32-bit float only with `FLOAT32_FILTERABLE`).
    pub fn supports_texture_format(&self, format: wgpu::TextureFormat) -> bool {
        format_supported(&self.adapter, &self.device, format)
    }

    /// Build error of the current version of a [`crate::CustomShader`], once a draw has used it
    /// (its draws then use the built-in shader) or a post camera graded with it
    /// (`ColorGrading::shader`: grading then runs without the hooks). `None` while it builds fine
    /// or is unused.
    pub fn shader_error(&self, shader: crate::ShaderHandle) -> Option<&str> {
        self.pipelines
            .custom_error(shader)
            .or_else(|| self.post.shader_error(shader))
    }

    /// Portability warning of the current version of a [`crate::CustomShader`], once a draw has
    /// used it: the shader builds and draws on this device, but uses what WebGL2 / GLES 3.0
    /// (GLSL ES 3.00) lacks, so on those devices it fails to build and its draws use the
    /// built-in shader (see [`crate::CustomShader`], "Portability"). Also logged once through
    /// `tracing`. `None` for portable shaders, and on WebGL2 / GLES 3.0 devices themselves,
    /// where such a shader is a [`Self::shader_error`].
    pub fn shader_warning(&self, shader: crate::ShaderHandle) -> Option<&str> {
        self.pipelines.custom_warning(shader)
    }

    /// Number of wgpu errors that reached the uncaptured-error handler so far. Release builds
    /// log these instead of panicking; tests and tools can assert that it stays 0.
    pub fn gpu_error_count(&self) -> u64 {
        self.gpu_errors.load(Ordering::Relaxed)
    }

    /// Upload all pending meshes/textures now (normally done at the start of each frame),
    /// whatever the [`crate::UploadBudget`]: tools and tests call it to have everything on the
    /// GPU before a frame.
    pub fn upload_assets(&mut self, assets: &mut RenderAssets) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("space3d asset upload"),
        });
        self.upload_assets_into(&mut encoder, assets, None);
        self.uploader.finish();
        self.queue.submit([encoder.finish()]);
        self.uploader.recall();
    }

    /// Encode pending asset uploads (textures, then the mesh arena): everything queued, or with a
    /// `budget` in bytes the first of it (see [`crate::UploadBudget`]). What is left stays in
    /// the queues in its order, and the stats say how much that is.
    pub(crate) fn upload_assets_into(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        assets: &mut RenderAssets,
        budget: Option<u64>,
    ) {
        let mut allowance = UploadAllowance::new(budget);
        // Free removed meshes first so their space can be reused this frame.
        for a in assets.freed_allocs.drain(..) {
            assets.arena.free(a);
        }
        // Textures before meshes: an entity is drawn as soon as its mesh is uploaded, so a mesh
        // must not arrive ahead of the maps queued with it (its material would draw without
        // them meanwhile). The frame stops at the first item that does not fit, so no new mesh
        // is uploaded while a texture still waits.
        if !assets.pending_textures.is_empty() {
            self.upload_textures(encoder, assets, &mut allowance);
        }
        if !assets.pending_meshes.is_empty() {
            self.upload_meshes(encoder, assets, &mut allowance);
        }
        let (meshes, textures, bytes) = assets.pending_uploads();
        self.stats.pending_meshes = meshes;
        self.stats.pending_textures = textures;
        self.stats.pending_upload_bytes = bytes;
    }

    /// Create the queued textures the frame's allowance admits, and every queued render target.
    fn upload_textures(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        assets: &mut RenderAssets,
        allowance: &mut UploadAllowance,
    ) {
        let mut pending = std::mem::take(&mut assets.pending_textures);
        for h in &pending {
            let Some(entry) = assets.textures.get_mut(h.raw()) else {
                continue;
            };
            if entry.render_target {
                // A render target holds no data, and cameras that draw into it are skipped
                // until it exists: it never waits behind queued images.
                entry.queued = false;
                if entry.gpu.is_some() {
                    continue;
                }
            } else {
                // Until its turn the texture keeps what it has: nothing for a new one (its
                // materials draw without it), the previous contents for a replaced one.
                if entry.source.is_some() && !allowance.admit(entry.pending_bytes()) {
                    assets.pending_textures.push(*h);
                    continue;
                }
                entry.queued = false;
                if entry.source.is_none() {
                    continue;
                }
            }
            let targets = crate::texture::TargetCaps {
                hdr_format: self.frame_graphics.hdr_format,
                hdr_msaa: self.frame_graphics.hdr_msaa,
                max_msaa: self.caps.max_msaa,
                msaa_depth_binding: self.graphics.msaa_depth_resolve,
            };
            match create_gpu_texture(
                &self.adapter,
                &self.device,
                &self.queue,
                encoder,
                &mut self.pipelines,
                &mut self.samplers,
                targets,
                entry,
            ) {
                Ok(bytes) => {
                    self.uploader.bytes += bytes;
                    self.stats.textures_uploaded += 1;
                }
                Err(e) => tracing::error!(texture = ?h, "texture upload failed: {e}"),
            }
        }
        if assets.pending_textures.is_empty() {
            // Nothing waits: keep the queue's memory for the next uploads.
            pending.clear();
            assets.pending_textures = pending;
        }
    }

    /// Upload the queued meshes the frame's allowance admits, and every replaced mesh that has
    /// a GPU copy.
    fn upload_meshes(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        assets: &mut RenderAssets,
        allowance: &mut UploadAllowance,
    ) {
        // The meshes uploaded in this call are moved to the front of `pending` (it then lists
        // exactly them); the ones that wait go back into the queue in their order.
        let mut pending = std::mem::take(&mut assets.pending_meshes);
        let mut uploading = 0;
        // Convert; keep the existing placement when the size is unchanged.
        let scratch = &mut self.frame.mesh_scratch;
        scratch.clear();
        for k in 0..pending.len() {
            let h = pending[k];
            let Some(entry) = assets.meshes.get_mut(h.raw()) else {
                continue;
            };
            // Decided from the data's size, before anything is converted or taken.
            let bytes = entry.pending_bytes();
            if entry.cpu.is_some() {
                if entry.gpu.is_some() {
                    // A replaced mesh that is being drawn: it shows its new data in this frame,
                    // as it does without a budget. Its bytes still count against the frame.
                    allowance.charge(bytes);
                } else if !allowance.admit(bytes) {
                    assets.pending_meshes.push(h);
                    continue;
                }
            }
            entry.queued = false;
            let Some(data) = entry.cpu.take() else {
                continue;
            };
            let mut v = Vec::new();
            let mut i = Vec::new();
            crate::arena::build_vertices(&data, &mut v, &mut i);
            let (vc, ic) = (v.len() as u32, i.len() as u32);
            let alloc = match entry.gpu {
                Some(g) if g.alloc.vertex_count == vc && g.alloc.index_count == ic => Some(g.alloc),
                Some(g) => {
                    assets.arena.free(g.alloc);
                    entry.gpu = None;
                    None
                }
                None => None,
            };
            scratch.push((h, alloc, v, i, data));
            pending[uploading] = h;
            uploading += 1;
        }
        pending.truncate(uploading);

        // Allocate, growing only the buffer(s) that are short, until everything fits or the
        // device's max_buffer_size is reached.
        let arena = &mut assets.arena;
        let old = (arena.vertices.capacity(), arena.indices.capacity());
        if arena.vertex.is_none() || arena.index.is_none() {
            arena.grow_capacity(0, 0, self.caps.max_buffer_size);
        }
        // A mesh larger than a whole buffer can ever be is not worth growing for.
        let (max_v, max_i) = crate::arena::MeshArena::max_capacity(self.caps.max_buffer_size);
        let fits = |v: &Vec<_>, i: &Vec<u32>| v.len() as u64 <= max_v as u64 && i.len() as u64 <= max_i as u64;
        loop {
            let (mut short_v, mut short_i) = (0u32, 0u32);
            for (_, alloc, v, i, _) in scratch.iter_mut().filter(|e| e.1.is_none() && fits(&e.2, &e.3)) {
                match arena.try_alloc(v.len() as u32, i.len() as u32) {
                    Ok(a) => *alloc = Some(a),
                    Err(short) => {
                        if short.vertices {
                            short_v = short_v.saturating_add(v.len() as u32);
                        }
                        if short.indices {
                            short_i = short_i.saturating_add(i.len() as u32);
                        }
                    }
                }
            }
            if (short_v, short_i) == (0, 0) {
                break;
            }
            let before = (arena.vertices.capacity(), arena.indices.capacity());
            arena.grow_capacity(short_v, short_i, self.caps.max_buffer_size);
            if (arena.vertices.capacity(), arena.indices.capacity()) == before {
                break; // At the device limit: what is left does not fit.
            }
        }
        if arena.vertex.is_none()
            || arena.index.is_none()
            || (arena.vertices.capacity(), arena.indices.capacity()) != old
        {
            // Recorded before the writes below, so they land after the old contents' copy.
            arena.realloc_gpu(&self.device, encoder, old, &mut self.uploader);
        }
        arena.write_all(
            encoder,
            scratch,
            |(_, alloc, v, i, _)| {
                alloc.map(|alloc| crate::arena::ArenaWrite {
                    alloc,
                    vertices: v,
                    indices: i,
                })
            },
            &mut self.uploader,
        );
        for (h, alloc, v, i, data) in scratch.drain(..) {
            let Some(entry) = assets.meshes.get_mut(h.raw()) else {
                continue;
            };
            let Some(alloc) = alloc else {
                tracing::error!(
                    mesh = ?h,
                    vertices = v.len(),
                    indices = i.len(),
                    max_buffer_size = self.caps.max_buffer_size,
                    "mesh does not fit the mesh arena; it is not drawn (update or remove it)"
                );
                entry.cpu = Some(data);
                continue;
            };
            entry.gpu = Some(crate::assets::GpuMesh {
                alloc,
                sphere: entry.sphere,
            });
            self.stats.meshes_uploaded += 1;
        }
        // The extension stream (tangents, second UV set) of the meshes just placed, and of no
        // other: a mesh placed in an earlier frame has given its extension vertices away, and
        // would have its range of the stream cleared.
        assets.upload_extras(&self.device, encoder, &pending, &mut self.uploader);
    }

    /// Read back the primary target of the last rendered frame as sRGB RGBA8.
    ///
    /// Headless: copies the offscreen target. Windowed: re-renders the last frame's primary
    /// views into the offscreen target first (surface textures are not readable). A
    /// post-processed primary target composites its persisted scene color again, and also
    /// re-renders its scene views when the frame skipped them (no surface texture: occluded or
    /// minimized window).
    /// Poll the returned task from a system, `.block()` it in tests, or `.wait().await` it.
    ///
    /// Never blocks: on native a background task resolves it with non-blocking device polls
    /// (each frame's own poll resolves it too). A blocking `device.poll` would hold wgpu's
    /// device-wide snatch lock for the whole GPU wait, stalling `queue.present` and buffer
    /// destruction on the main thread until the GPU caught up.
    pub fn capture(&mut self) -> Task<Result<ImageData, RenderError>> {
        #[cfg_attr(target_arch = "wasm32", allow(unused_mut))]
        let (readback, mut rx, _, error) = self.submit_capture();
        #[cfg(not(target_arch = "wasm32"))]
        {
            let device = self.device.clone();
            space3d_tasks::spawn(async move {
                if let Some(e) = error.await {
                    return Err(RenderError::Capture(e));
                }
                let mapped = loop {
                    let _ = device.poll(wgpu::PollType::Poll);
                    match rx.try_recv() {
                        Ok(Some(r)) => break Ok(r),
                        Ok(None) => tokio::time::sleep(std::time::Duration::from_millis(1)).await,
                        Err(_) => break Err(()),
                    }
                };
                readback.read(mapped)
            })
        }
        #[cfg(target_arch = "wasm32")]
        {
            // The browser (WebGPU) or the per-frame device poll (WebGL) resolves the mapping.
            space3d_tasks::spawn(async move {
                if let Some(e) = error.await {
                    return Err(RenderError::Capture(e));
                }
                readback.read(rx.await.map_err(|_| ()))
            })
        }
    }

    /// Record and submit the copy of the offscreen target into a `MAP_READ` buffer; returns
    /// the readback, the mapping result channel, the submission and the validation error of
    /// the capture's commands, if any.
    ///
    /// The capture runs inside an error scope: its failures are the capture's result, not
    /// uncaptured errors (which panic in debug builds). This matters once the device is lost —
    /// after `destroy` the device-lost callback only fires when in-flight work has finished,
    /// and until then mapping the (invalid) readback buffer is an error.
    fn submit_capture(
        &mut self,
    ) -> (
        CaptureReadback,
        MapReceiver,
        wgpu::SubmissionIndex,
        impl Future<Output = Option<String>> + use<>,
    ) {
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("space3d capture"),
        });
        if self.capture_replays() {
            self.encode_capture_replay(&mut encoder);
        }
        let (w, h) = (self.offscreen.width, self.offscreen.height);
        let padded = align_up(w as u64 * 4, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as u64);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("space3d capture readback"),
            size: padded * h as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let color = self.offscreen.color.as_ref().expect("offscreen color");
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded as u32),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        let (tx, rx) = futures::channel::oneshot::channel();
        encoder.map_buffer_on_submit(&buffer, wgpu::MapMode::Read, .., move |r| {
            let _ = tx.send(r);
        });
        let submission = self.queue.submit([encoder.finish()]);
        let error = scope.pop();
        // Native backends resolve the scope immediately; the browser answers asynchronously.
        #[cfg(not(target_arch = "wasm32"))]
        let error = std::future::ready(futures::FutureExt::now_or_never(error).flatten().map(|e| e.to_string()));
        #[cfg(target_arch = "wasm32")]
        let error = async move { error.await.map(|e| e.to_string()) };
        let readback = CaptureReadback {
            buffer,
            size: (w, h),
            padded,
        };
        (readback, rx, submission, error)
    }

    /// [`Self::capture`] and wait for the result on the calling thread (tests/tools only;
    /// never from a system).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn capture_blocking(&mut self) -> Result<ImageData, RenderError> {
        let (readback, mut rx, submission, error) = self.submit_capture();
        if let Some(e) = futures::FutureExt::now_or_never(error).flatten() {
            return Err(RenderError::Capture(e));
        }
        let mapped = loop {
            match rx.try_recv() {
                Ok(Some(r)) => break Ok(r),
                Ok(None) => {}
                Err(_) => break Err(()),
            }
            let waited = self.device.poll(wgpu::PollType::Wait {
                submission_index: Some(submission.clone()),
                timeout: None,
            });
            if waited.is_err() {
                // Device lost: the callback has fired with an error by now, or never will.
                break rx.try_recv().ok().flatten().ok_or(());
            }
        };
        readback.read(mapped)
    }

    /// Non-blocking device poll: fires pending map callbacks (captures, staging recycling).
    pub fn poll(&self) {
        let _ = self.device.poll(wgpu::PollType::Poll);
    }

    /// Graphics capabilities of the device (one probe for every graphics feature), masked by
    /// [`crate::FrameSettings::downlevel`] as of the last rendered frame (unmasked before the
    /// first frame).
    pub fn graphics_caps(&self) -> &crate::caps::GraphicsCaps {
        &self.frame_graphics
    }

    /// Shared layouts for render nodes and overlays (renderer lifetime).
    pub fn frame_layouts(&self) -> &crate::node::FrameLayouts {
        &self.layouts
    }

    /// Unique id of this renderer: a new renderer (or device) has a new one, so nodes rebuild
    /// their GPU state.
    pub fn renderer_id(&self) -> u64 {
        self.id
    }

    /// GPU pass timings of a recent frame ([`crate::FrameSettings::gpu_timing`]).
    pub fn gpu_timings(&self) -> &crate::timing::GpuTimings {
        &self.gpu_timings
    }

    /// The primary target group's scene color `S` of the last frame — after `HdrScene`
    /// overlays, before post (graphics spec §1.10) — as linear float pixels: `Rgba16Float`
    /// as stored, `Rg11b10Ufloat` with alpha 1, the Ldr fallback's `Rgba8UnormSrgb` decoded.
    /// `Err(RenderError::Capture)` when the primary group rendered in Direct mode (no post
    /// camera and render scale 1): there is no scene texture — or when the last frame had no
    /// primary view. A window whose last frame got no surface texture (occluded, minimized;
    /// always so in background mode on macOS) skipped its scene views: they are rendered into
    /// `S` first, like [`Self::capture`]'s replay does. For tests and tools; never blocks.
    pub fn capture_scene_hdr(&mut self) -> Task<Result<space3d_core::HdrImageData, RenderError>> {
        let submitted = self.submit_scene_capture();
        #[cfg(not(target_arch = "wasm32"))]
        let device = self.device.clone();
        space3d_tasks::spawn(async move {
            #[cfg_attr(target_arch = "wasm32", allow(unused_mut))]
            let (readback, mut rx, _, error) = submitted?;
            if let Some(e) = error.await {
                return Err(RenderError::Capture(e));
            }
            #[cfg(not(target_arch = "wasm32"))]
            let mapped = loop {
                let _ = device.poll(wgpu::PollType::Poll);
                match rx.try_recv() {
                    Ok(Some(r)) => break Ok(r),
                    Ok(None) => tokio::time::sleep(std::time::Duration::from_millis(1)).await,
                    Err(_) => break Err(()),
                }
            };
            #[cfg(target_arch = "wasm32")]
            let mapped = rx.await.map_err(|_| ());
            readback.read(mapped)
        })
    }

    /// [`Self::capture_scene_hdr`], waiting on the calling thread (tests and tools only).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn capture_scene_hdr_blocking(&mut self) -> Result<space3d_core::HdrImageData, RenderError> {
        let (readback, mut rx, submission, error) = self.submit_scene_capture()?;
        if let Some(e) = futures::FutureExt::now_or_never(error).flatten() {
            return Err(RenderError::Capture(e));
        }
        let mapped = loop {
            match rx.try_recv() {
                Ok(Some(r)) => break Ok(r),
                Ok(None) => {}
                Err(_) => break Err(()),
            }
            if self
                .device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(submission.clone()),
                    timeout: None,
                })
                .is_err()
            {
                break rx.try_recv().ok().flatten().ok_or(());
            }
        };
        readback.read(mapped)
    }

    /// Record and submit the copy of the primary group's `S` into a `MAP_READ` buffer (inside a
    /// validation error scope, like [`Self::submit_capture`]).
    #[allow(clippy::type_complexity)]
    fn submit_scene_capture(
        &mut self,
    ) -> Result<
        (
            SceneReadback,
            MapReceiver,
            wgpu::SubmissionIndex,
            impl Future<Output = Option<String>> + use<>,
        ),
        RenderError,
    > {
        let primary = crate::frame::GroupTarget::Primary;
        let Some(t) = self.frame.intermediate_textures(primary) else {
            return Err(RenderError::Capture(
                "the primary target rendered in Direct mode (no post camera, render scale 1): there \
                 is no scene texture"
                    .into(),
            ));
        };
        // A window whose surface texture was unavailable (occluded, minimized) skipped the scene
        // views: render them into `S` first, as a capture replay does (the windowed offscreen
        // target it also composites into only serves captures). Headless frames always render
        // them, unless the frame had no primary scene view: `S` is then an older frame's.
        let stale = t.scene_frame != self.frame_index;
        let rerender = stale && self.capture_replays() && self.frame.plans_primary_scene();
        if stale && !rerender {
            return Err(RenderError::Capture(
                "the last frame rendered no primary scene view: the scene texture is stale".into(),
            ));
        }
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("space3d scene capture"),
        });
        if rerender {
            self.encode_primary_to_offscreen(&mut encoder);
        }
        let texture = self
            .frame
            .intermediate_textures(primary)
            .expect("checked above")
            .scene
            .clone();
        let format = texture.format();
        let (w, h) = (texture.width(), texture.height());
        let texel = format.block_copy_size(None).unwrap_or(4) as u64;
        let padded = align_up(w as u64 * texel, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as u64);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("space3d scene capture readback"),
            size: padded * h as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded as u32),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        let (tx, rx) = futures::channel::oneshot::channel();
        encoder.map_buffer_on_submit(&buffer, wgpu::MapMode::Read, .., move |r| {
            let _ = tx.send(r);
        });
        let submission = self.queue.submit([encoder.finish()]);
        let error = scope.pop();
        #[cfg(not(target_arch = "wasm32"))]
        let error = std::future::ready(futures::FutureExt::now_or_never(error).flatten().map(|e| e.to_string()));
        #[cfg(target_arch = "wasm32")]
        let error = async move { error.await.map(|e| e.to_string()) };
        let readback = SceneReadback {
            buffer,
            size: (w, h),
            padded,
            format,
        };
        Ok((readback, rx, submission, error))
    }

    /// Last read-back exposure (stops, incl. compensation) of a post camera
    /// ([`crate::FrameSettings::read_exposure`]).
    pub fn camera_exposure(&self, camera: space3d_core::Entity) -> Option<f32> {
        self.post.camera_exposure(camera)
    }

    /// Forget a camera's temporal post history (auto exposure snaps next frame).
    pub fn reset_post_history(&mut self, camera: space3d_core::Entity) {
        self.post.reset_history(camera);
    }

    /// Build pipelines ahead of use (loading screens: WebGL2 links programs at creation, 50–150
    /// ms each): for the pass signatures of the last rendered frame (render one frame of the
    /// scene first) and `request.formats` on primary views (the capture format when windowed),
    /// the forward pipelines of `request.materials` × instance layouts, every render node's
    /// `prewarm` for the targets its slots draw with, and the post pipelines of the post groups'
    /// targets (the composite, FXAA and the bloom chain in both chain formats). Each material's
    /// forward pipeline gets the class bits it has on each layout, its maps' texture kinds (as
    /// the last frame resolved them; single-layer for maps no frame has drawn yet) and the last
    /// frame's frame-wide overrides (shading model, normal maps, the casting light's filter);
    /// casters also get their depth pipeline when the last frame drew a shadow.
    pub fn prewarm(&mut self, request: &PrewarmRequest) {
        self.prewarm_pipelines(request);
    }

    /// Where a cubemap is on its way to being usable as an environment, as of the last rendered
    /// frame: `Missing` for cubemaps that frame did not have (removed, or added since),
    /// `Uploading` until it is on the GPU, then `Baking` (the passes left of its environment
    /// bake, which starts in the first frame a camera's `EnvironmentLight` uses it) until
    /// `Ready`; `Failed` if the upload failed (logged).
    pub fn environment_state(&self, cubemap: crate::handles::CubemapHandle) -> crate::lighting::api::EnvironmentState {
        self.lighting.environment_state(cubemap)
    }
}

/// Receives the result of a capture buffer's mapping.
type MapReceiver = futures::channel::oneshot::Receiver<Result<(), wgpu::BufferAsyncError>>;

/// What is left of a frame's [`crate::UploadBudget`] while its uploads are chosen.
struct UploadAllowance {
    /// Bytes left (`None`: no limit).
    left: Option<u64>,
    /// An item was admitted in this frame: the next ones have to fit.
    any: bool,
    /// An item did not fit: everything after it waits too, so the order is kept.
    stopped: bool,
}

impl UploadAllowance {
    fn new(budget: Option<u64>) -> Self {
        Self {
            left: budget,
            any: false,
            stopped: false,
        }
    }

    /// Whether an item of `bytes` is uploaded in this frame: the first one always (an item
    /// larger than the budget would otherwise never be), a later one while it fits in what is
    /// left. After the first refusal everything is refused.
    fn admit(&mut self, bytes: u64) -> bool {
        let Some(left) = self.left else { return true };
        if self.stopped || (self.any && bytes > left) {
            self.stopped = true;
            return false;
        }
        self.any = true;
        self.left = Some(left.saturating_sub(bytes));
        true
    }

    /// Count the bytes of an item that is uploaded whatever the budget.
    fn charge(&mut self, bytes: u64) {
        if let Some(left) = &mut self.left {
            *left = left.saturating_sub(bytes);
        }
    }
}

/// A submitted capture copy: `MAP_READ` buffer with rows padded to `padded` bytes.
struct CaptureReadback {
    buffer: wgpu::Buffer,
    size: (u32, u32),
    padded: u64,
}

impl CaptureReadback {
    /// Strip the row padding of the mapped buffer (`Err(())`: the callback was dropped).
    fn read(self, mapped: Result<Result<(), wgpu::BufferAsyncError>, ()>) -> Result<ImageData, RenderError> {
        mapped
            .map_err(|()| RenderError::Capture("map callback dropped".into()))?
            .map_err(|e| RenderError::Capture(e.to_string()))?;
        let (w, h) = self.size;
        let view = self
            .buffer
            .slice(..)
            .get_mapped_range()
            .map_err(|e| RenderError::Capture(format!("{e:?}")))?;
        let row = w as usize * 4;
        let mut pixels = Vec::with_capacity(row * h as usize);
        for y in 0..h as usize {
            let start = y * self.padded as usize;
            pixels.extend_from_slice(&view[start..start + row]);
        }
        drop(view);
        self.buffer.unmap();
        Ok(ImageData {
            width: w,
            height: h,
            pixels,
        })
    }
}

/// A submitted copy of a scene texture: `MAP_READ` buffer with rows padded to `padded` bytes.
struct SceneReadback {
    buffer: wgpu::Buffer,
    size: (u32, u32),
    padded: u64,
    format: wgpu::TextureFormat,
}

impl SceneReadback {
    /// Decode the mapped rows into linear float RGBA.
    fn read(
        self,
        mapped: Result<Result<(), wgpu::BufferAsyncError>, ()>,
    ) -> Result<space3d_core::HdrImageData, RenderError> {
        mapped
            .map_err(|()| RenderError::Capture("map callback dropped".into()))?
            .map_err(|e| RenderError::Capture(e.to_string()))?;
        let (w, h) = self.size;
        let view = self
            .buffer
            .slice(..)
            .get_mapped_range()
            .map_err(|e| RenderError::Capture(format!("{e:?}")))?;
        let texel = self.format.block_copy_size(None).unwrap_or(4) as usize;
        let mut pixels = Vec::with_capacity(w as usize * h as usize);
        for y in 0..h as usize {
            let row = &view[y * self.padded as usize..][..w as usize * texel];
            for p in row.chunks_exact(texel) {
                pixels.push(decode_texel(self.format, p));
            }
        }
        drop(view);
        self.buffer.unmap();
        Ok(space3d_core::HdrImageData::new(w, h, pixels))
    }
}

/// Linear RGBA of one texel of a scene-texture format.
pub(crate) fn decode_texel(format: wgpu::TextureFormat, p: &[u8]) -> [f32; 4] {
    use crate::instance::f16_to_f32;
    match format {
        wgpu::TextureFormat::Rgba16Float => [0, 2, 4, 6].map(|i| f16_to_f32(u16::from_le_bytes([p[i], p[i + 1]]))),
        wgpu::TextureFormat::Rg11b10Ufloat => {
            let v = u32::from_le_bytes([p[0], p[1], p[2], p[3]]);
            // Unsigned floats: 5-bit exponent (bias 15), 6- or 5-bit mantissa.
            let small = |bits: u32, mantissa_bits: u32| {
                let m = bits & ((1 << mantissa_bits) - 1);
                let e = bits >> mantissa_bits;
                let scale = (1u32 << mantissa_bits) as f32;
                match e {
                    0 => m as f32 / scale * 2f32.powi(-14),
                    31 => f32::INFINITY,
                    e => 2f32.powi(e as i32 - 15) * (1.0 + m as f32 / scale),
                }
            };
            [small(v & 0x7ff, 6), small((v >> 11) & 0x7ff, 6), small(v >> 22, 5), 1.0]
        }
        wgpu::TextureFormat::Rgba32Float => {
            [0, 4, 8, 12].map(|i| f32::from_le_bytes([p[i], p[i + 1], p[i + 2], p[i + 3]]))
        }
        f if f.is_srgb() => {
            let c = |i: usize| crate::components::srgb_to_linear(p[i] as f32 / 255.0);
            [c(0), c(1), c(2), p[3] as f32 / 255.0]
        }
        _ => [0, 1, 2, 3].map(|i| p[i] as f32 / 255.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use space3d_core::MeshData;

    #[test]
    fn scene_texels_decode() {
        use crate::instance::f32_to_f16;
        let half: Vec<u8> = [8.0f32, 0.5, 0.0, 1.0]
            .iter()
            .flat_map(|v| f32_to_f16(*v).to_le_bytes())
            .collect();
        assert_eq!(
            decode_texel(wgpu::TextureFormat::Rgba16Float, &half),
            [8.0, 0.5, 0.0, 1.0]
        );
        // Rg11b10: 1.0 = exponent 15, mantissa 0; 8.0 = exponent 18; 0.5 = exponent 14.
        let packed = (15u32 << 6) | ((18u32 << 6) << 11) | ((14u32 << 5) << 22);
        assert_eq!(
            decode_texel(wgpu::TextureFormat::Rg11b10Ufloat, &packed.to_le_bytes()),
            [1.0, 8.0, 0.5, 1.0]
        );
        let srgb = decode_texel(wgpu::TextureFormat::Rgba8UnormSrgb, &[255, 0, 188, 128]);
        assert!((srgb[0] - 1.0).abs() < 1e-6 && srgb[1] == 0.0 && (srgb[2] - 0.5).abs() < 0.01);
        assert!((srgb[3] - 128.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn an_upload_allowance_admits_a_prefix_and_always_one_item() {
        // no budget: everything
        let mut all = UploadAllowance::new(None);
        assert!(all.admit(u64::MAX) && all.admit(u64::MAX));
        // 100 bytes: 60 fits, 50 does not, and nothing after it does either (the order is kept)
        let mut a = UploadAllowance::new(Some(100));
        assert_eq!([a.admit(60), a.admit(50), a.admit(10)], [true, false, false]);
        // an item that fills the budget exactly fits; an empty one after it does too
        let mut a = UploadAllowance::new(Some(100));
        assert_eq!(
            [a.admit(60), a.admit(40), a.admit(0), a.admit(1)],
            [true, true, true, false]
        );
        // the first item goes whatever its size, alone
        let mut a = UploadAllowance::new(Some(100));
        assert_eq!([a.admit(5000), a.admit(1)], [true, false]);
        // and so it does when exempt uploads have used the frame up
        let mut a = UploadAllowance::new(Some(100));
        a.charge(500);
        assert_eq!([a.admit(70), a.admit(1)], [true, false]);
    }

    fn renderer() -> Option<Renderer> {
        match pollster::block_on(Renderer::new_headless(16, 16)) {
            Ok(r) => Some(r),
            Err(e) => {
                eprintln!("skipping GPU test: no adapter available ({e})");
                None
            }
        }
    }

    /// Regression: arena growth doubled vertex and index buffers in lockstep and never looked
    /// at `max_buffer_size`, so create_buffer failed validation (a panic) long before the data
    /// reached the limit. Growth is now per buffer, clamped, and meshes that cannot fit are
    /// reported and kept on the CPU instead.
    #[test]
    fn mesh_arena_respects_max_buffer_size() {
        let Some(mut r) = renderer() else { return };
        r.caps.max_buffer_size = 36 * 16_384; // 16384 vertices
        let mut assets = RenderAssets::default();
        let small: Vec<_> = (0..10).map(|_| assets.add_mesh(MeshData::cube(1.0))).collect();
        let big = assets.add_mesh(MeshData::sphere(1.0, 200, 100)); // 20301 vertices
        r.upload_assets(&mut assets);
        assert!(small.iter().all(|m| assets.is_mesh_uploaded(*m)));
        assert!(!assets.is_mesh_uploaded(big), "does not fit");
        assert!(assets.contains_mesh(big));
        let (v, i) = (
            assets.arena.vertex.as_ref().unwrap(),
            assets.arena.index.as_ref().unwrap(),
        );
        assert!(v.size() <= r.caps.max_buffer_size && i.size() <= r.caps.max_buffer_size);
        assert_eq!(v.size(), 4096 * 36, "no growth for the small meshes");

        // Index-heavy meshes grow only the index buffer.
        let fan = MeshData {
            indices: (0..9000).map(|k| k % 3).collect(),
            ..MeshData::plane(1.0)
        };
        let m = assets.add_mesh(fan);
        r.upload_assets(&mut assets);
        assert!(assets.is_mesh_uploaded(m));
        assert_eq!(assets.arena.vertex.as_ref().unwrap().size(), 4096 * 36);
        assert!(assets.arena.index.as_ref().unwrap().size() > 8192 * 4);

        // The rejected mesh fits once the limit allows it.
        r.caps.max_buffer_size = u32::MAX as u64;
        assert!(assets.update_mesh(big, MeshData::sphere(1.0, 200, 100)));
        r.upload_assets(&mut assets);
        assert!(assets.is_mesh_uploaded(big));
        assert!(small.iter().all(|m| assets.is_mesh_uploaded(*m)));
        assert_eq!(r.gpu_error_count(), 0);
    }

    /// Regression: browser WebGPU canvases offer no sRGB format, so the renderer encoded sRGB in
    /// the shader and blending/MSAA resolves ran on encoded values (50% white over black gave
    /// 128 instead of 188). It now renders through an sRGB view of the canvas texture.
    #[test]
    fn surfaces_without_srgb_formats_render_through_an_srgb_view() {
        use wgpu::TextureFormat as F;
        let webgpu_canvas = [F::Bgra8Unorm, F::Rgba8Unorm, F::Rgba16Float];
        assert_eq!(
            surface_formats(&webgpu_canvas, true),
            Some((F::Bgra8Unorm, F::Bgra8UnormSrgb))
        );
        // Without SURFACE_VIEW_FORMATS: in-shader encoding.
        assert_eq!(
            surface_formats(&webgpu_canvas, false),
            Some((F::Bgra8Unorm, F::Bgra8Unorm))
        );
        // Native surfaces with an sRGB format use it directly.
        assert_eq!(
            surface_formats(&[F::Bgra8Unorm, F::Bgra8UnormSrgb], true),
            Some((F::Bgra8UnormSrgb, F::Bgra8UnormSrgb))
        );
        // Formats without an sRGB twin keep in-shader encoding.
        assert_eq!(
            surface_formats(&[F::Rgba16Float], true),
            Some((F::Rgba16Float, F::Rgba16Float))
        );
        assert_eq!(surface_formats(&[], true), None);
    }

    /// A canvas that rejects the sRGB view format (reported as `Lost` on the web) is
    /// reconfigured without it once, instead of being recreated with the same configuration
    /// in a loop. Native surfaces and surfaces that already presented are recreated.
    #[test]
    fn rejected_srgb_view_formats_fall_back_to_shader_encoding() {
        use wgpu::TextureFormat as F;
        let mut config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: F::Bgra8Unorm,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: 8,
            height: 8,
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![F::Bgra8UnormSrgb],
        };
        assert!(retry_without_view_formats(true, false, &config));
        assert!(!retry_without_view_formats(true, true, &config));
        assert!(!retry_without_view_formats(false, false, &config));
        config.view_formats.clear();
        assert!(!retry_without_view_formats(true, false, &config));
    }

    /// Without `DownlevelFlags::BASE_VERTEX` (WebGL2, GLES < 3.2) only `CpuDirect` is offered:
    /// it is the path whose direct draws emulate `base_vertex`.
    #[test]
    fn gpu_paths_require_base_vertex() {
        assert_eq!(best_draw_path(true, true, true, true), DrawPath::GpuMultiDrawCount);
        assert_eq!(best_draw_path(true, true, true, false), DrawPath::GpuMultiDraw);
        assert_eq!(best_draw_path(true, true, false, false), DrawPath::GpuIndirect);
        assert_eq!(best_draw_path(true, false, true, true), DrawPath::CpuDirect);
        assert_eq!(best_draw_path(false, true, true, true), DrawPath::CpuDirect);
    }

    fn invalid_buffer(device: &wgpu::Device) {
        // MAP_READ may only be combined with COPY_DST: a validation error.
        let _ = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("invalid on purpose"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
    }

    /// Release builds log uncaptured wgpu errors instead of wgpu's default panic; debug builds
    /// (and tests) still panic so bugs surface.
    #[test]
    fn uncaptured_wgpu_errors_are_logged_in_release_and_panic_in_debug() {
        let Some(r) = renderer() else { return };
        assert_eq!(r.gpu_error_count(), 0);
        let (count, _) = install_error_handler(&r.device, false);
        invalid_buffer(&r.device);
        invalid_buffer(&r.device);
        assert_eq!(count.load(Ordering::Relaxed), 2, "logged, not panicked");

        let (count, _) = install_error_handler(&r.device, true);
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| invalid_buffer(&r.device)));
        assert!(panicked.is_err());
        assert_eq!(count.load(Ordering::Relaxed), 1);
    }

    /// Once the device is lost every new resource is invalid and each use of one is an error;
    /// debug builds must log those instead of panicking (the loss is not a bug).
    #[test]
    fn errors_after_device_loss_do_not_panic_in_debug() {
        let Some(r) = renderer() else { return };
        let (count, _) = install_error_handler(&r.device, true);
        r.device.destroy();
        // `destroy` loses the device once its in-flight work is done: the callback fires here.
        let _ = r.device.poll(wgpu::PollType::wait_indefinitely());
        let buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("after loss"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        assert!(
            count.load(Ordering::Relaxed) >= 1,
            "the invalid buffer's error was reported"
        );
    }
}
