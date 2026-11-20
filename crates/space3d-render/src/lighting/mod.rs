//! Lighting: cubemaps, skybox, image-based lighting and directional shadows.
//!
//! [`api`] is the public hub (re-exported from the crate root). The rest is the crate-internal
//! contract between the frame and the materials area (graphics spec §2.6): the frame calls
//! [`Lighting`] at fixed points of `render_frame` (steps 4, 5 and 8 of §1.4) and draws the depth
//! views it requests; the materials area owns the bodies.
//!
//! * Step 4, [`Lighting::prepare`]: uploads new cubemaps (radiance cubes, see [`cubemap`]),
//!   finds the environments the frame's cameras use (`EnvironmentLight`, created on first use:
//!   skybox-only cubemaps never pay for a bake), and records bake passes — every pass when the
//!   renderer is headless or `IblSettings::bake_passes_per_frame` is `None`, else that many per
//!   frame, oldest environment first. An environment is used from the frame its last pass is
//!   recorded in; until then its camera renders with the previous environment of the same
//!   cubemap if one is baked (a settings change or a re-upload), else without image-based
//!   lighting.
//! * Step 8, [`Lighting::fill_view`] / [`Lighting::view_resources`]: the camera's sky and
//!   environment fields of `GpuView` and its group-0 bindings 1–4 (the BRDF LUT is always bound:
//!   `ShadingModel::Standard` uses it for multiple-scattering compensation).
//! * Environments unused for [`ENV_IDLE_FRAMES`] frames, those of removed cubemaps and those
//!   superseded by a baked newer one are dropped.
//! * Directional shadows ([`shadow`]): step 4 picks the casting light and keeps its shadow map,
//!   step 5 ([`Lighting::depth_views`]) fits the map to the frame's cameras and requests it as a
//!   depth view, step 8 gives every camera view the light matrix and binds the map (group 0
//!   bindings 5–6; a 1×1 far map without a shadow).

pub(crate) mod api;
pub(crate) mod cubemap;
mod environment;
pub(crate) mod shadow;
mod skybox;

use crate::assets::RenderAssets;
use crate::caps::GraphicsCaps;
use crate::components::Camera3d;
use crate::culling::Frustum;
use crate::node::{RenderNodes, ViewInfo};
use crate::pipeline::GpuView;
use crate::stats::RenderStats;
use api::{CubemapHandle, EnvironmentLight, EnvironmentState, LightingSettings, Skybox};
use environment::{Baker, Env, EnvKey, Job, Writes};
use glam::{Mat3, Mat4, Quat};
use space3d_core::{Entity, World};

/// Frames an environment stays cached after its last use.
pub(crate) const ENV_IDLE_FRAMES: u64 = 600;

/// Inputs of [`Lighting::prepare`] (step 4: cubemap uploads and conversion, IBL bake passes).
pub(crate) struct LightingPrepare<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// The frame's world (cameras with an `EnvironmentLight` request their environment).
    pub world: &'a World,
    pub assets: &'a mut RenderAssets,
    pub settings: &'a LightingSettings,
    /// The frame's capabilities (after `FrameSettings::downlevel` masking): without
    /// `hdr_format`, radiance cubes of HDR sources and bakes are `Rgba8UnormSrgb`.
    pub graphics: &'a GraphicsCaps,
    /// Headless renderers finish every pending bake in the frame it is requested.
    pub headless: bool,
    pub stats: &'a mut RenderStats,
    /// GPU timings: the frame's lighting passes are one `"ibl.bake"` scope.
    pub timing: &'a crate::timing::GpuTimer,
}

/// One depth-only view the frame renders before the camera views (a shadow map).
#[allow(
    dead_code,
    reason = "graphics contract (spec §2.6): `light` and `size` describe the view; the frame reads the rest"
)]
pub(crate) struct DepthViewRequest {
    pub light: Entity,
    pub view_proj: Mat4,
    pub frustum: Frustum,
    /// `Depth32Float`, `size`².
    pub target: wgpu::TextureView,
    pub size: u32,
    pub bias: wgpu::DepthBiasState,
}

/// Bindings 1–6 of group 0 for a camera, and a stable key for bind-group caching. The frame
/// keeps one bind group per key (and views-buffer generation): equal keys must mean identical
/// resources, and a new resource (a re-created shadow map, a baked environment) a new key. The
/// resources for `None` (depth views) must not include a shadow map a depth view renders into.
pub(crate) struct ViewResources<'a> {
    pub env_specular: &'a wgpu::TextureView,
    pub env_irradiance: &'a wgpu::TextureView,
    pub brdf_lut: &'a wgpu::TextureView,
    pub env_sampler: &'a wgpu::Sampler,
    pub shadow_map: &'a wgpu::TextureView,
    pub shadow_sampler: &'a wgpu::Sampler,
    pub key: u64,
}

/// A registered cubemap as of the last `prepare`.
#[derive(Clone, Copy)]
struct CubeStatus {
    handle: CubemapHandle,
    state: EnvironmentState,
    /// The radiance cube is on the GPU (a skybox can show it).
    uploaded: bool,
}

/// Lighting state: cubemaps, the environment cache, the bake queue and the shadow state.
pub(crate) struct Lighting {
    dummy_cube: wgpu::TextureView,
    dummy_depth: wgpu::TextureView,
    brdf_lut: wgpu::TextureView,
    sampler: wgpu::Sampler,
    comparison: wgpu::Sampler,
    depth_views: Vec<DepthViewRequest>,
    /// Bake pipelines (with the first cubemap).
    baker: Option<Baker>,
    /// Uploads planned this frame (recorded all at once).
    uploads: Vec<Job>,
    /// Environments, oldest first (the bake queue's order).
    envs: Vec<Env>,
    /// This frame's cameras with an environment: (camera, `envs` index of the one to bind).
    camera_envs: Vec<(Entity, usize)>,
    /// Registered cubemaps as of the last `prepare` (`environment_state`, sky readiness).
    states: Vec<CubeStatus>,
    frame: u64,
    next_id: u64,
    /// Warned that HDR cubemaps are stored as LDR (no float render targets).
    warned_ldr: bool,
    /// This frame's casting light (step 4).
    shadow_light: Option<shadow::CastingLight>,
    /// Its shadow map (kept for [`ENV_IDLE_FRAMES`] frames without a casting light).
    shadow_map: Option<shadow::ShadowMap>,
    /// This frame's fitted shadow (step 5), for the camera views.
    shadow: Option<shadow::ActiveShadow>,
    /// Warned that several lights asked to cast.
    warned_casters: bool,
    /// Tests: apply `bake_passes_per_frame` although the renderer is headless.
    pub(crate) budget_headless: bool,
}

/// A rotation's inverse as `GpuView` matrix columns (world → cubemap).
fn rotation_columns(q: Quat) -> [[f32; 4]; 3] {
    let q = if q.length_squared() > 1e-12 {
        q.normalize()
    } else {
        Quat::IDENTITY
    };
    let m = Mat3::from_quat(q).transpose();
    [
        m.x_axis.extend(0.0).to_array(),
        m.y_axis.extend(0.0).to_array(),
        m.z_axis.extend(0.0).to_array(),
    ]
}

impl Lighting {
    /// The BRDF LUT and neutral resources (1×1 black environment, far shadow map).
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, caps: &GraphicsCaps) -> Self {
        let _ = caps;
        let texture = |label, layers, format, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: layers,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let sampled = wgpu::TextureUsages::TEXTURE_BINDING;
        let cube = texture("space3d dummy cube", 6, wgpu::TextureFormat::Rgba8Unorm, sampled);
        let depth = texture(
            "space3d dummy shadow map",
            1,
            crate::pipeline::DEPTH_FORMAT,
            sampled | wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        Self {
            dummy_cube: cube.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::Cube),
                ..Default::default()
            }),
            dummy_depth: depth.create_view(&Default::default()),
            brdf_lut: environment::create_brdf_lut(device, queue),
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("space3d environment"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                ..Default::default()
            }),
            // Bilinear comparisons: each tap is a 2×2 PCF (the kernels of `S3D_SHADOW_TAPS`
            // weigh them).
            comparison: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("space3d shadow comparison"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                compare: Some(wgpu::CompareFunction::LessEqual),
                ..Default::default()
            }),
            depth_views: Vec::new(),
            baker: None,
            uploads: Vec::new(),
            envs: Vec::new(),
            camera_envs: Vec::new(),
            states: Vec::new(),
            frame: 0,
            next_id: 1,
            warned_ldr: false,
            shadow_light: None,
            shadow_map: None,
            shadow: None,
            warned_casters: false,
            budget_headless: false,
        }
    }

    fn next_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Step 4: cubemap uploads and conversion, IBL bake passes (budgeted; all at once when
    /// headless).
    pub fn prepare(&mut self, cx: &mut LightingPrepare<'_>) {
        self.frame += 1;
        let frame = self.frame;
        let hdr = cx.graphics.hdr_format == Some(wgpu::TextureFormat::Rgba16Float);
        // HDR skies in `Rg11b10Ufloat` where it renders and compute writes it (or nothing is
        // computed): half the memory of `Rgba16Float` (a 1024² sky with mips: 33.5 MB instead
        // of 67; `GraphicsCaps::hdr_sky_format`).
        let compact = cx.graphics.hdr_sky_format() == Some(wgpu::TextureFormat::Rg11b10Ufloat);
        // Compute dispatches (one compute pass) where compute shaders can write storage
        // textures; fragment passes elsewhere (WebGL2, `DownlevelOverride::no_compute`). The
        // compact HDR sky is written by compute where storage takes `Rg11b10Ufloat` (a 1024²
        // sky: 11 dispatches instead of 66 render passes).
        let compute = cx.graphics.compute && cx.device.limits().max_storage_textures_per_shader_stage > 0;
        let writes = Writes::new(compute, cx.graphics.compact_hdr_storage);
        let capacities = (
            self.uploads.capacity(),
            self.camera_envs.capacity(),
            self.states.capacity(),
        );
        self.prepare_shadow(cx.device, cx.world, cx.stats);

        // Uploads of new and updated cubemaps.
        let mut pending = std::mem::take(&mut cx.assets.pending_cubemaps);
        for &h in &pending {
            let Some(entry) = cx.assets.cubemaps.get_mut(h.raw()) else {
                continue;
            };
            let Some(source) = entry.source.take() else {
                continue;
            };
            let id = self.next_id();
            let baker = self.baker.get_or_insert_with(|| Baker::new(cx.device));
            match cubemap::plan_upload(baker, cx.device, cx.queue, &source, (hdr, compact), writes, id) {
                Ok(upload) => {
                    let format = match &source {
                        api::CubemapSource::Faces(t) | api::CubemapSource::Equirect { texture: t, .. } => t.format,
                    };
                    if !hdr && !self.warned_ldr && cubemap::is_hdr(format) {
                        self.warned_ldr = true;
                        tracing::warn!(
                            "no float render targets: HDR cubemaps are stored as Rgba8UnormSrgb \
                             (radiance above 1 clamps; environments bake in LDR)"
                        );
                    }
                    cx.stats.bytes_uploaded += upload.bytes;
                    cx.stats.textures_uploaded += 1;
                    cx.stats.textures_created += upload.textures;
                    entry.gpu = Some(upload.gpu);
                    entry.state = EnvironmentState::Ready;
                    self.uploads.push(upload.job);
                }
                Err(e) => {
                    tracing::warn!("cubemap {h:?} not uploaded: {e}");
                    entry.state = EnvironmentState::Failed;
                }
            }
        }
        pending.clear();
        cx.assets.pending_cubemaps = pending;

        // Environments: drop those of removed or re-uploaded cubemaps once nothing needs them,
        // idle ones, and those a baked newer one supersedes.
        let key = EnvKey::new(&cx.settings.ibl, hdr);
        let current = |e: &Env, assets: &RenderAssets| {
            e.key == key && assets.cubemap_gpu(e.cubemap).is_some_and(|g| g.id == e.source)
        };
        let assets = &*cx.assets;
        let mut i = 0;
        while i < self.envs.len() {
            let e = &self.envs[i];
            let live = assets.contains_cubemap(e.cubemap);
            let idle = e.last_used + ENV_IDLE_FRAMES < frame;
            let superseded = !current(e, assets)
                && self
                    .envs
                    .iter()
                    .any(|o| o.cubemap == e.cubemap && o.ready() && current(o, assets));
            if !live || idle || superseded {
                self.envs.remove(i);
            } else {
                i += 1;
            }
        }

        // The environments this frame's cameras use (created on first use).
        self.camera_envs.clear();
        for (camera, _, light) in cx.world.query::<(Entity, &Camera3d, &EnvironmentLight)>().iter() {
            let Some(cube) = assets.cubemap_gpu(light.cubemap) else {
                continue;
            };
            let found = self
                .envs
                .iter()
                .position(|e| e.cubemap == light.cubemap && e.source == cube.id && e.key == key);
            let index = match found {
                Some(i) => i,
                None => {
                    let id = self.next_id;
                    self.next_id += 1;
                    let baker = self.baker.get_or_insert_with(|| Baker::new(cx.device));
                    self.envs
                        .push(Env::new(baker, cx.device, light.cubemap, cube, key, writes, id, frame));
                    cx.stats.textures_created += 2;
                    self.envs.len() - 1
                }
            };
            self.envs[index].last_used = frame;
            self.camera_envs.push((camera, index));
        }
        cx.stats.lighting_bytes += self.envs.iter().map(Env::bytes).sum::<u64>()
            + cx.assets
                .cubemaps
                .iter()
                .filter_map(|(_, e)| e.gpu.as_ref())
                .map(|g| g.bytes)
                .sum::<u64>();

        // Record: every upload step, then bake steps within the budget (a render pass each on
        // the fragment path, dispatches in one compute pass on the compute path).
        let budget = match cx.settings.ibl.bake_passes_per_frame {
            Some(n) if !cx.headless || self.budget_headless => n,
            _ => u32::MAX,
        };
        let mut passes = 0u32;
        let mut steps = 0u32;
        for job in &self.uploads {
            let (p, s) = job.passes(u32::MAX);
            passes += p;
            steps += s;
        }
        let mut left = budget;
        for e in &self.envs {
            if left == 0 {
                break;
            }
            let (p, s) = e.job.passes(left);
            passes += p;
            steps += s;
            left -= s;
        }
        if passes > 0 {
            let timing = cx.timing;
            let mut timer = environment::PassTimer::new(timing.span("ibl.bake", None), passes);
            let baker = self.baker.as_mut().expect("jobs imply the baker");
            for job in &mut self.uploads {
                baker.record(cx.device, cx.encoder, job, u32::MAX, &mut timer);
            }
            let mut left = budget;
            for e in &mut self.envs {
                if left == 0 {
                    break;
                }
                left -= baker.record(cx.device, cx.encoder, &mut e.job, left, &mut timer);
            }
            cx.stats.ibl_bake_passes += steps;
            cx.stats.render_passes += timer.render_passes;
            cx.stats.fullscreen_passes += timer.render_passes;
            cx.stats.compute_passes += timer.compute_passes;
        }
        self.uploads.clear();

        // Which environment each camera binds: its own once baked, else a baked older one of the
        // same cubemap.
        for (_, index) in &mut self.camera_envs {
            let e = &self.envs[*index];
            if !e.ready()
                && let Some(older) = self.envs.iter().position(|o| o.cubemap == e.cubemap && o.ready())
            {
                *index = older;
            }
        }
        for &(_, index) in &self.camera_envs {
            self.envs[index].last_used = frame;
        }

        // Cubemap states for `environment_state` and the sky.
        self.states.clear();
        for (raw, entry) in assets.cubemaps.iter() {
            let handle = CubemapHandle::from_raw(raw);
            let state = match &entry.gpu {
                None if entry.state == EnvironmentState::Failed => EnvironmentState::Failed,
                None => EnvironmentState::Uploading,
                Some(g) => self
                    .envs
                    .iter()
                    .find(|e| e.cubemap == handle && e.source == g.id && e.key == key)
                    .map_or(
                        EnvironmentState::Baking {
                            passes_left: key.passes(g.size, compute),
                        },
                        Env::state,
                    ),
            };
            self.states.push(CubeStatus {
                handle,
                state,
                uploaded: entry.gpu.is_some(),
            });
        }
        let after = (
            self.uploads.capacity(),
            self.camera_envs.capacity(),
            self.states.capacity(),
        );
        if after != capacities {
            cx.stats.cpu_reallocations += 1;
        }
    }

    /// State of `cubemap` as of the last frame (`Missing` for cubemaps the renderer has not
    /// seen yet or that were removed).
    pub fn environment_state(&self, cubemap: CubemapHandle) -> EnvironmentState {
        self.states
            .iter()
            .find(|s| s.handle == cubemap)
            .map_or(EnvironmentState::Missing, |s| s.state)
    }

    /// The baked environment `camera` binds this frame.
    fn camera_env(&self, camera: Entity) -> Option<&Env> {
        self.camera_envs
            .iter()
            .find(|(c, _)| *c == camera)
            .map(|&(_, i)| &self.envs[i])
            .filter(|e| e.ready())
    }

    /// Step 5: this frame's shadow depth views (after camera planning).
    /// The casting light's map is fitted to `cameras` (see [`shadow`]); no request without a
    /// casting light, a camera to fit to, or with `strength` 0.
    pub fn depth_views(
        &mut self,
        world: &World,
        cameras: &[ViewInfo],
        settings: &LightingSettings,
    ) -> &[DepthViewRequest] {
        let _ = (world, settings);
        self.depth_views.clear();
        self.shadow = None;
        if let (Some(light), Some(map)) = (&self.shadow_light, &self.shadow_map)
            && light.shadow.strength > 0.0
            && let Some(fitted) = shadow::fit(light, map.size, cameras)
        {
            self.depth_views.push(DepthViewRequest {
                light: light.entity,
                view_proj: fitted.view_proj,
                frustum: Frustum {
                    small: shadow::small_casters(&light.shadow, cameras),
                    ..Frustum::from_view_proj(&fitted.view_proj)
                },
                target: map.view.clone(),
                size: map.size,
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: light.shadow.slope_bias.max(0.0) * shadow::filter_reach(light.shadow.filter),
                    clamp: 0.01,
                },
            });
            self.shadow = Some(shadow::ActiveShadow::new(light, &fitted, map));
        }
        &self.depth_views
    }

    /// Step 4 of the shadow: the casting light, and its map (created on first use and when its
    /// size changes; dropped after [`ENV_IDLE_FRAMES`] frames without a casting light).
    fn prepare_shadow(&mut self, device: &wgpu::Device, world: &World, stats: &mut RenderStats) {
        let (light, casters) = shadow::casting_light(world);
        if casters > 1 && !self.warned_casters {
            self.warned_casters = true;
            tracing::warn!("{casters} directional lights have a DirectionalShadow: only one casts (the lowest entity)");
        }
        self.shadow_light = light;
        match &light {
            Some(l) => {
                let size = shadow::map_size(l.shadow.map_size, device.limits().max_texture_dimension_2d);
                if self.shadow_map.as_ref().is_none_or(|m| m.size != size) {
                    let id = self.next_id();
                    self.shadow_map = Some(shadow::ShadowMap::new(device, size, id, self.frame));
                    stats.textures_created += 1;
                }
                if let Some(m) = &mut self.shadow_map {
                    m.last_used = self.frame;
                    // A render target of the frame like the others (`Depth32Float`).
                    stats.target_bytes += 4 * m.size as u64 * m.size as u64;
                }
            }
            None => {
                if self
                    .shadow_map
                    .as_ref()
                    .is_some_and(|m| m.last_used + ENV_IDLE_FRAMES < self.frame)
                {
                    self.shadow_map = None;
                }
            }
        }
    }

    /// Step 8: fill the environment, sky and shadow fields of a camera's `GpuView` (zero = off).
    /// With a shadow, the casting light moves to `dir_lights[0]`.
    pub fn fill_view(&self, world: &World, camera: Entity, view: &mut GpuView) {
        if let Some(s) = &self.shadow {
            s.fill(view);
        }
        if let Ok(sky) = world.get::<&Skybox>(camera) {
            let uploaded = self.states.iter().any(|s| s.handle == sky.cubemap && s.uploaded);
            view.sky_params = [sky.intensity, if uploaded { 1.0 } else { 0.0 }, 0.0, 0.0];
            view.sky_rotation = rotation_columns(sky.rotation);
        }
        if let Some(env) = self.camera_env(camera)
            && let Ok(light) = world.get::<&EnvironmentLight>(camera)
        {
            view.env_params = [
                light.diffuse_intensity,
                light.specular_intensity,
                (env.specular_mips - 1) as f32,
                1.0,
            ];
            view.env_rotation = rotation_columns(light.rotation);
        }
    }

    /// Bindings 1–6 of group 0 for `camera` (`None`: depth views, which never sample the map
    /// they may be rendering). Keys: the environment's id in the low 32 bits, the shadow map's
    /// above (ids are unique and far below 2³²).
    pub fn view_resources(&self, camera: Option<Entity>) -> ViewResources<'_> {
        let env = camera.and_then(|c| self.camera_env(c));
        let map = camera
            .and(self.shadow.as_ref())
            .and_then(|s| self.shadow_map.as_ref().filter(|m| m.id == s.map));
        ViewResources {
            env_specular: env.map_or(&self.dummy_cube, |e| &e.specular),
            env_irradiance: env.map_or(&self.dummy_cube, |e| &e.irradiance),
            brdf_lut: &self.brdf_lut,
            env_sampler: &self.sampler,
            shadow_map: map.map_or(&self.dummy_depth, |m| &m.view),
            shadow_sampler: &self.comparison,
            key: env.map_or(0, |e| e.id & 0xFFFF_FFFF) | map.map_or(0, |m| m.id << 32),
        }
    }

    /// This frame's shadow map, when a depth view renders it (tests).
    #[cfg(test)]
    pub(crate) fn shadow_map_size(&self) -> Option<u32> {
        self.shadow.and(self.shadow_map.as_ref()).map(|m| m.size)
    }

    /// Register the built-in lighting nodes (`"space3d/skybox"`).
    pub fn register_nodes(nodes: &mut RenderNodes) {
        skybox::register(nodes);
    }

    /// Environments cached (tests).
    #[cfg(test)]
    pub(crate) fn environment_count(&self) -> usize {
        self.envs.len()
    }

    /// The textures of cached environment `i`: (specular, irradiance) (tests).
    #[cfg(test)]
    pub(crate) fn environment_textures(&self, i: usize) -> (&wgpu::Texture, &wgpu::Texture) {
        (&self.envs[i].textures[0], &self.envs[i].textures[1])
    }
}

#[cfg(test)]
mod tests;
