//! A procedural city at golden hour after rain, built entirely from generated meshes and
//! textures: a grid of blocks with towers, brick and stone buildings and lit windows (base color,
//! normal, packed ORM and emissive maps), wet asphalt with puddles, pavements, street lamps and
//! traffic signals that really light the street (the 16 nearest), cars with headlights and
//! taillights on the lanes, steam, drizzle and mist (`space3d-particles`), a procedural sunset as
//! skybox and image-based light, one shadow-casting sun and a cinematic post stack (AgX, bloom,
//! grading, vignette, film grain, FXAA/MSAA) — under a [`GraphicsQuality`] tier that picks itself
//! for the device (Low on WebGL2) and can be cycled at runtime.
//!
//! ```text
//! cargo run -p city_demo --release
//! SPACE3D_QUALITY=ultra cargo run -p city_demo --release       # force a tier (low|medium|high|ultra)
//! SPACE3D_HEADLESS=1920x1080 SPACE3D_SHOTS=0,20,40 SPACE3D_SHOT_DIR=out \
//!     cargo run -p city_demo --release                         # PNGs of the flythrough at those seconds
//! # web: see examples/city_demo/web (`space3d build --target web`, then `space3d serve`)
//! ```
//!
//! Keys: `F` free-fly (WASD move, right mouse drag looks, `Shift` fast, `Space`/`Ctrl` up/down),
//! `P` pause the flythrough, `Q` next quality tier, `H` hide the panel, `Esc` quit. Everything
//! is generated off the frame (`spawn_compute`) once the tier is known; the city appears when it
//! is done.

#[macro_use]
mod files;

mod city;
mod effects;
mod flight;
mod mesh;
mod people;
mod procgen;
mod props;
mod real;

use city::{CityData, LightSpot};
use mesh::MeshBuilder;
use procgen::{Facade, TextureSet};
use space3d::assets::render::{RenderModel, spawn_model};
use space3d::particles::prelude::*;
use space3d::prelude::*;
use space3d::render::{
    Bloom, ColorGrading, CubemapHandle, DirectionalShadow, EnvironmentLight, FilmGrain, Fog, FrameSettings,
    FxaaQuality, GpuTimings, GraphicsQuality, IblSettings, LightingSettings, RenderStats, Renderer, ShadingModel,
    ShadowFit, Ssao, Ssr, Vignette,
};
use space3d::tasks::Task;

/// A tunable number: the environment variable `name` (native only) or `default`.
fn knob(name: &str, default: f32) -> f32 {
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(v) = std::env::var(name).ok().and_then(|v| v.trim().parse::<f32>().ok()) {
        return v;
    }
    let _ = name;
    default
}

struct Textures {
    facades: Vec<(Facade, TextureSet)>,
}

struct Rest {
    shop: TextureSet,
    asphalt: TextureSet,
    paving: TextureSet,
    roof: TextureSet,
    palette: procgen::Palette,
    walls: [Option<(TextureSet, f32)>; 4],
    glow: ImageData,
}

/// The generation tasks and what they have delivered.
#[derive(Default)]
struct Loading {
    sky: Option<Task<real::Sky>>,
    tex_a: Option<Task<Textures>>,
    tex_b: Option<Task<Textures>>,
    rest: Option<Task<Rest>>,
    models: Option<Task<real::Models>>,
    got_models: Option<real::Models>,
    city: Option<Task<CityData>>,
    got_sky: Option<real::Sky>,
    got_a: Option<Textures>,
    got_b: Option<Textures>,
    got_rest: Option<Rest>,
    got_city: Option<CityData>,
}

/// A pedestrian entity and the path it paces.
struct Pedestrian {
    entity: Entity,
    walker: people::Walker,
    /// The two stride meshes of its outfit.
    meshes: [MeshHandle; 2],
    parity: usize,
}

struct Car {
    entity: Entity,
    along_z: bool,
    dir: f32,
    /// Street line coordinate and lane offset from it.
    line: f32,
    lane: f32,
    base: f32,
    speed: f32,
    /// Half the vehicle's length in meters.
    half_len: f32,
}

struct Slot {
    entity: Entity,
    id: Option<u32>,
    fade: f32,
}

struct Shots {
    times: Vec<f32>,
    index: usize,
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    dir: String,
    /// Frames waited since the last capture.
    settle: u32,
    pending: Option<Task<Result<ImageData, space3d::render::RenderError>>>,
}

struct Demo {
    quality: Option<GraphicsQuality>,
    loading: Option<Loading>,
    camera: Option<Entity>,
    sky: Option<CubemapHandle>,
    /// Frames since the city was dressed.
    dressed: u64,
    time: f32,
    paused: bool,
    free: bool,
    yaw: f32,
    pitch: f32,
    pos: Vec3,
    spots: Vec<LightSpot>,
    cars: Vec<Car>,
    walkers: Vec<Pedestrian>,
    /// The web build fetching its assets, and the tier it will generate for.
    fetching: Option<(std::sync::Arc<files::Progress>, bool, u32)>,
    slots: Vec<Slot>,
    hud: bool,
    shots: Option<Shots>,
    timings: bool,
}

fn build(app: &mut App) {
    let headless = DefaultPlugins::headless_from_env((1920, 1080)).is_some();
    app.add_plugin(
        DefaultPlugins::new()
            .window(WindowPlugin {
                title: "Space3d — city demo".into(),
                canvas_id: Some("space3d".into()),
                ..WindowPlugin::default()
            })
            .without_physics()
            .without_net(),
    );
    add_game(app, headless);
}

fn shots_from_env() -> Option<Shots> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let v = std::env::var("SPACE3D_SHOTS").ok()?;
        let times: Vec<f32> = v.split(',').filter_map(|s| s.trim().parse().ok()).collect();
        if times.is_empty() {
            return None;
        }
        let dir = std::env::var("SPACE3D_SHOT_DIR").unwrap_or_else(|_| ".".into());
        let _ = std::fs::create_dir_all(&dir);
        Some(Shots {
            times,
            index: 0,
            dir,
            settle: 0,
            pending: None,
        })
    }
    #[cfg(target_arch = "wasm32")]
    None
}

/// The scene and its systems, without the engine plugins (the tests add them to a headless
/// app).
fn add_game(app: &mut App, headless: bool) {
    let initial = if cfg!(target_arch = "wasm32") {
        GraphicsQuality::Low
    } else {
        GraphicsQuality::High
    };
    app.add_plugin(ParticlesPlugin::with_settings(ParticleSettings::for_quality(initial)));
    app.insert_resource(Demo {
        quality: None,
        loading: None,
        camera: None,
        sky: None,
        dressed: 0,
        time: knob("CITY_START", 0.0),
        paused: false,
        free: false,
        yaw: 0.0,
        pitch: 0.0,
        pos: Vec3::ZERO,
        spots: Vec::new(),
        cars: Vec::new(),
        fetching: None,
        walkers: Vec::new(),
        slots: Vec::new(),
        hud: !headless,
        shots: shots_from_env(),
        timings: knob("CITY_REPORT", 0.0) > 0.5,
    });
    app.add_system(Stage::Startup, setup);
    app.add_system(Stage::PreUpdate, choose_quality);
    app.add_system(Stage::Update, load_content);
    app.add_system(Stage::Update, controls);
    app.add_system(Stage::Update, drive_camera);
    app.add_system(Stage::Update, traffic);
    app.add_system(Stage::Update, pedestrians);
    app.add_system(Stage::PostUpdate, street_lights);
    app.add_system(Stage::Last, screenshots);
    app.add_system(Stage::Last, report);
    app.add_ui_system(hud);
}

fn quality_from_env() -> Option<GraphicsQuality> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let v = std::env::var("SPACE3D_QUALITY").ok()?;
        GraphicsQuality::ALL
            .into_iter()
            .find(|q| q.name().eq_ignore_ascii_case(v.trim()))
    }
    #[cfg(target_arch = "wasm32")]
    None
}

fn set_quality(ctx: &mut SystemCtx, q: GraphicsQuality) {
    q.apply(ctx.world, ctx.resources);
    let particles = match ctx.resources.get::<Renderer>() {
        Some(r) => ParticleSettings::for_quality_on(q, r.graphics_caps()),
        None => ParticleSettings::for_quality(q),
    };
    *ctx.resources.expect_mut::<ParticleSettings>() = particles;
    ctx.resources.expect_mut::<Demo>().quality = Some(q);
}

/// The camera, the sun and the grade.
fn setup(ctx: &mut SystemCtx) {
    ctx.resources.insert(LightingSettings {
        shading: ShadingModel::Standard,
        ibl: IblSettings {
            bake_passes_per_frame: cfg!(target_arch = "wasm32").then_some(6),
            ..IblSettings::default()
        },
        ..LightingSettings::default()
    });
    ctx.resources.insert(AmbientLight {
        color: Vec3::new(0.62, 0.52, 0.6),
        intensity: knob("CITY_AMBIENT", 0.03),
    });
    let sun_dir = procgen::sun_direction();
    ctx.world.spawn((
        Name("sun".into()),
        Transform::IDENTITY.looking_at(-sun_dir, Vec3::Y),
        DirectionalLight {
            color: Vec3::new(1.0, 0.56, 0.26),
            intensity: knob("CITY_SUN", 4.2),
        },
        DirectionalShadow::new().with_fit(ShadowFit::Camera {
            camera: None,
            max_distance: knob("CITY_SHADOW_DISTANCE", 360.0),
            receivers: None,
        }),
    ));
    let grade = ColorGrading {
        saturation: knob("CITY_SATURATION", 1.12),
        contrast: knob("CITY_CONTRAST", 1.08),
        temperature: knob("CITY_TEMPERATURE", 0.06),
        tint: knob("CITY_TINT", 0.0),
        gain: Vec3::new(1.03, 1.0, 0.97),
        lift: Vec3::new(0.0, 0.0, 0.008),
        ..ColorGrading::default()
    };
    let post = PostProcess::default()
        .with_tonemapper(Tonemapper::AgX)
        .with_exposure(knob("CITY_EXPOSURE", -0.3))
        .with_bloom(
            Bloom::glow()
                .with_intensity(knob("CITY_BLOOM", 0.5))
                .with_threshold(knob("CITY_BLOOM_THRESHOLD", 1.0), 0.6)
                .with_scatter(0.78),
        )
        .with_color_grading(grade)
        .with_vignette(Vignette {
            enabled: true,
            intensity: 0.3,
            smoothness: 0.55,
            ..Vignette::default()
        })
        .with_film_grain(FilmGrain {
            enabled: true,
            intensity: 0.03,
            ..FilmGrain::default()
        })
        .with_chromatic_aberration(0.07)
        .with_fxaa(FxaaQuality::High);
    // Height fog, SSAO and screen-space reflections (opt-in post effects; the tiers drop them
    // where the device cannot run them).
    let post = if knob("CITY_EFFECTS", 1.0) > 0.5 {
        post.with_fog(Fog {
            enabled: true,
            color: Vec3::new(0.52, 0.46, 0.55),
            density: knob("CITY_FOG", 0.0007),
            height_falloff: 0.012,
            sky_amount: 0.95,
            sky_falloff: 7.0,
            sun_color: Vec3::new(1.0, 0.6, 0.3),
            sun_intensity: 1.3,
            sun_exponent: 8.0,
            ..Fog::default()
        })
        .with_ssao(Ssao::on().with_radius(1.5).with_intensity(1.0))
        .with_ssr(
            Ssr::on()
                .with_distance(120.0)
                .with_max_roughness(0.75)
                .with_blur(1.5, 4.0)
                .with_hdr_boost(1.0)
                .with_denoise(2),
        )
    } else {
        post
    };
    let (pos, target) = flight::sample(ctx.resources.expect::<Demo>().time);
    let camera = ctx.world.spawn((
        Name("camera".into()),
        Transform::from_xyz(pos.x, pos.y, pos.z).looking_at(target, Vec3::Y),
        Camera3d::default()
            .with_clear_color(Vec4::new(0.01, 0.01, 0.02, 1.0))
            .with_projection(Projection::Perspective {
                fov_y: knob("CITY_FOV", 46.0).to_radians(),
                near: 0.3,
                far: 6000.0,
            }),
        post,
    ));
    // The light pool: the lights the street lights and cars hand to the 16 nearest.
    let mut slots = Vec::new();
    for _ in 0..18 {
        let entity = ctx.world.spawn((
            Transform::from_xyz(0.0, -500.0, 0.0),
            PointLight {
                color: Vec3::ONE,
                intensity: 0.0,
                range: 1.0,
            },
        ));
        slots.push(Slot {
            entity,
            id: None,
            fade: 0.0,
        });
    }
    let d = ctx.resources.expect_mut::<Demo>();
    d.camera = Some(camera);
    d.pos = pos;
    d.slots = slots;
}

/// Once the renderer exists: pick the tier and start generating.
fn choose_quality(ctx: &mut SystemCtx) {
    if ctx.resources.expect::<Demo>().quality.is_some() {
        return;
    }
    let Some(renderer) = ctx.resources.get::<Renderer>() else {
        return;
    };
    let q = quality_from_env().unwrap_or_else(|| GraphicsQuality::recommended_for(renderer));
    println!(
        "city_demo: {} on {} ({:?}); keys: F free-fly, P pause, Q tier, H panel",
        q,
        renderer.capabilities().adapter.name,
        renderer.capabilities().adapter.backend
    );
    set_quality(ctx, q);
    let low = q == GraphicsQuality::Low;
    let face = match q {
        GraphicsQuality::Low => 512,
        GraphicsQuality::Medium => 768,
        _ => 1024,
    };
    match files::start(low) {
        Some(progress) => ctx.resources.expect_mut::<Demo>().fetching = Some((progress, low, face)),
        None => start_generation(ctx, low, face),
    }
}

/// The assets are in (or embedded): generate the city off the frame.
fn start_generation(ctx: &mut SystemCtx, low: bool, face: u32) {
    let facades = |kinds: [Facade; 2]| Textures {
        facades: kinds
            .into_iter()
            .map(|k| (k, procgen::facade_textures(k, 11 + k as u32 * 100)))
            .collect(),
    };
    let loading = Loading {
        sky: Some(space3d::tasks::spawn_compute(move || {
            real::sky(low, face).unwrap_or_else(|| real::Sky {
                source: procgen::sky_cubemap(face),
                sun: None,
            })
        })),
        tex_a: Some(space3d::tasks::spawn_compute(move || {
            facades([Facade::Concrete, Facade::Brick])
        })),
        tex_b: Some(space3d::tasks::spawn_compute(move || {
            facades([Facade::Glass, Facade::Stone])
        })),
        models: Some(space3d::tasks::spawn_compute(real::models)),
        rest: Some(space3d::tasks::spawn_compute(move || Rest {
            shop: procgen::shop_textures(5),
            asphalt: real::asphalt(low).unwrap_or_else(|| procgen::asphalt_textures(21)),
            paving: real::paving(low).unwrap_or_else(|| procgen::paving_textures(33)),
            roof: real::roof(low).unwrap_or_else(|| procgen::roof_textures(44)),
            palette: procgen::palette(),
            walls: [
                real::wall(Facade::Concrete, low),
                real::wall(Facade::Brick, low),
                None,
                real::wall(Facade::Stone, low),
            ],
            glow: procgen::glow_texture(),
        })),
        city: Some(space3d::tasks::spawn_compute(city::generate)),
        ..Loading::default()
    };
    ctx.resources.expect_mut::<Demo>().loading = Some(loading);
}

macro_rules! poll_into {
    ($l:expr, $task:ident, $got:ident) => {
        if let Some(r) = $l.$task.as_mut().and_then(Task::poll) {
            $l.$task = None;
            $l.$got = Some(r);
        }
    };
}

fn load_content(ctx: &mut SystemCtx) {
    let d = ctx.resources.expect_mut::<Demo>();
    if d.fetching.as_ref().is_some_and(|(p, ..)| p.finished()) {
        let (_, low, face) = d.fetching.take().unwrap();
        start_generation(ctx, low, face);
        return;
    }
    d.dressed += u64::from(d.sky.is_some());
    let Some(l) = d.loading.as_mut() else {
        return;
    };
    poll_into!(l, sky, got_sky);
    poll_into!(l, tex_a, got_a);
    poll_into!(l, tex_b, got_b);
    poll_into!(l, rest, got_rest);
    poll_into!(l, models, got_models);
    poll_into!(l, city, got_city);
    if l.got_sky.is_some()
        && l.got_a.is_some()
        && l.got_b.is_some()
        && l.got_rest.is_some()
        && l.got_models.is_some()
        && l.got_city.is_some()
    {
        let l = d.loading.take().unwrap();
        dress(ctx, l);
    }
}

struct Maps {
    base: TextureHandle,
    normal: TextureHandle,
    orm: TextureHandle,
    emissive: TextureHandle,
}

fn upload(assets: &mut RenderAssets, t: TextureSet) -> Maps {
    Maps {
        base: assets.add_image(t.base),
        normal: assets.add_image_linear(t.normal),
        orm: assets.add_image_linear(t.orm),
        emissive: assets.add_image(t.emissive),
    }
}

fn lit_material(m: &Maps, emissive: f32) -> Material {
    Material::color(Vec4::ONE)
        .with_texture(m.base)
        .with_normal_map(m.normal, 1.0)
        .with_orm_map(m.orm)
        .with_metallic_roughness(1.0, 1.0)
        .with_emissive_map(m.emissive, Vec3::splat(emissive))
}

fn spawn_mesh(world: &mut World, assets: &mut RenderAssets, data: MeshData, material: &Material) {
    if data.indices.is_empty() {
        return;
    }
    let h = assets.add_mesh(data);
    world.spawn((Transform::IDENTITY, Mesh3d(h), material.clone()));
}

/// Register the generated content and spawn the city.
fn dress(ctx: &mut SystemCtx, l: Loading) {
    let models = l.got_models.unwrap();
    let (sky, a, b, rest, city) = (
        l.got_sky.unwrap(),
        l.got_a.unwrap(),
        l.got_b.unwrap(),
        l.got_rest.unwrap(),
        l.got_city.unwrap(),
    );
    let quality0 = ctx.resources.expect::<Demo>().quality.unwrap_or_default();
    let assets = ctx.resources.expect_mut::<RenderAssets>();
    let sun = sky.sun;
    let sky = assets.add_cubemap(sky.source);
    let emissive = knob("CITY_WINDOWS", 1.8);
    // Real walls (tiled at world scale) under alpha-masked window layers; glass towers are the
    // atlas alone (opaque, smooth, dark, metallic).
    let mut wall_mats: [Option<Material>; 4] = [None, None, None, None];
    for (slot, w) in rest.walls.into_iter().enumerate() {
        if let Some((set, tile)) = w {
            let maps = upload(assets, set);
            wall_mats[slot] = Some(lit_material(&maps, 0.0).with_tiling(4.0 / tile, 4.0 / tile));
        }
    }
    let mut facade_mats: [Option<Material>; 4] = [None, None, None, None];
    for (kind, set) in a.facades.into_iter().chain(b.facades) {
        let maps = upload(assets, set);
        let slot = match kind {
            Facade::Concrete => 0,
            Facade::Brick => 1,
            Facade::Glass => 2,
            Facade::Stone => 3,
        };
        let mat = lit_material(&maps, emissive);
        facade_mats[slot] = Some(if wall_mats[slot].is_some() {
            mat.with_alpha_mask(0.5)
        } else {
            mat
        });
    }
    let shop_mat = lit_material(&upload(assets, rest.shop), knob("CITY_SHOPS", 1.4));
    let road_mat = lit_material(&upload(assets, rest.asphalt), 0.0);
    let paving_mat = lit_material(&upload(assets, rest.paving), 0.0);
    let roof_mat = lit_material(&upload(assets, rest.roof), 0.0);
    let (pb, po, pe) = (
        assets.add_texture(rest.palette.base),
        assets.add_texture(rest.palette.orm),
        assets.add_texture(rest.palette.emissive),
    );
    let props = |emissive: f32| {
        Material::color(Vec4::ONE)
            .with_texture(pb)
            .with_orm_map(po)
            .with_metallic_roughness(1.0, 1.0)
            .with_emissive_map(pe, Vec3::splat(emissive))
    };
    let hot = props(knob("CITY_PROPS_GLOW", 14.0));
    let cold = props(0.0);
    let glow_tex = assets.add_image(rest.glow);
    let glow_mat = Material::additive(Vec4::ONE)
        .with_texture(glow_tex)
        .with_render_order(-5);

    let puddle_mat = Material::color(Vec4::new(0.012, 0.014, 0.018, 1.0)).with_metallic_roughness(0.0, 0.03);
    spawn_mesh(ctx.world, assets, city.ground, &road_mat);
    let mut parked = Vec::new();
    let mut street_props = Vec::new();
    let mut walkers: Vec<people::Walker> = Vec::new();
    for chunk in city.chunks {
        parked.extend(chunk.parked);
        walkers.extend(chunk.walkers);
        street_props.extend(chunk.props);
        for (i, f) in chunk.facade.into_iter().enumerate() {
            spawn_mesh(ctx.world, assets, f, facade_mats[i].as_ref().unwrap());
        }
        for (i, w) in chunk.wall.into_iter().enumerate() {
            if let Some(m) = &wall_mats[i] {
                spawn_mesh(ctx.world, assets, w, m);
            }
        }
        spawn_mesh(ctx.world, assets, chunk.shop, &shop_mat);
        spawn_mesh(ctx.world, assets, chunk.roof, &roof_mat);
        spawn_mesh(ctx.world, assets, chunk.paving, &paving_mat);
        spawn_mesh(ctx.world, assets, chunk.props_hot, &hot);
        spawn_mesh(ctx.world, assets, chunk.props_cold, &cold);
        spawn_mesh(ctx.world, assets, chunk.glow, &glow_mat);
        spawn_mesh(ctx.world, assets, chunk.puddle, &puddle_mat);
    }
    let vehicle_handles: Vec<MeshHandle> = city.vehicles.into_iter().map(|m| assets.add_mesh(m)).collect();

    // Vehicles: the CC0 models (the renderer instances their parts), the procedural cars where
    // a model did not load.
    struct CarAsset {
        model: RenderModel,
        half_len: f32,
        scale: f32,
        weight: u32,
        lights: MeshHandle,
        streaks: MeshHandle,
    }
    let mut car_assets: Vec<CarAsset> = Vec::new();
    for m in models.cars {
        let mut rm = m.model.upload(assets);
        for part in &mut rm.parts {
            let wheel = part.name.as_deref().is_some_and(|n| n.contains("wheel"));
            part.material.roughness = if wheel { 0.75 } else { 0.28 };
            part.material.metallic = if wheel { 0.0 } else { 0.25 };
        }
        let mut lb = MeshBuilder::new();
        props::add_car_lights(&mut lb, m.half_w, m.half_len, 0.45);
        // the lights' reflections on the wet road, ahead of and behind the car
        let mut sb = MeshBuilder::new();
        let ys = 0.05 / m.scale;
        for side in [-1.0f32, 1.0] {
            let x = side * m.half_w * 0.62;
            city::streak(
                &mut sb,
                Vec3::new(x, ys, m.half_len + 3.5),
                Vec3::Z,
                4.0,
                0.3,
                Vec3::new(1.0, 0.92, 0.75) * 1.4,
            );
            city::streak(
                &mut sb,
                Vec3::new(x, ys, -m.half_len - 2.5),
                Vec3::Z,
                3.0,
                0.25,
                Vec3::new(1.0, 0.06, 0.03) * 1.4,
            );
        }
        car_assets.push(CarAsset {
            model: rm,
            half_len: m.half_len * m.scale,
            scale: m.scale,
            weight: m.weight,
            lights: assets.add_mesh(lb.finish()),
            streaks: assets.add_mesh(sb.finish()),
        });
    }
    let total_weight: u32 = car_assets.iter().map(|c| c.weight).sum();
    let pick = |r: &mut city::Rng| {
        let mut x = r.int(total_weight.max(1));
        for (i, c) in car_assets.iter().enumerate() {
            if x < c.weight {
                return i;
            }
            x -= c.weight;
        }
        0
    };

    // Traffic on the streets around the center.
    let mut cars = Vec::new();
    let mut rng = city::Rng::new(4242);
    let span = 600.0;
    for s in 6..=10 {
        for along_z in [true, false] {
            for dir in [1.0f32, -1.0] {
                for lane in [1.75f32, 5.25] {
                    let speed = rng.range(8.0, 14.0) * if lane > 3.0 { 1.15 } else { 1.0 };
                    let n = if s == 8 { 7 } else { 4 };
                    for k in 0..n {
                        let bus = rng.chance(0.07);
                        let (entity, half_len) = if bus || car_assets.is_empty() {
                            let variant = if bus {
                                vehicle_handles.len() - 1
                            } else {
                                rng.int(vehicle_handles.len() as u32 - 1) as usize
                            };
                            let e =
                                ctx.world
                                    .spawn((Transform::IDENTITY, Mesh3d(vehicle_handles[variant]), hot.clone()));
                            (e, if bus { 5.5 } else { 2.2 })
                        } else {
                            let a = &car_assets[pick(&mut rng)];
                            let e = spawn_model(
                                ctx.world,
                                &a.model,
                                Transform::IDENTITY.with_scale(Vec3::splat(a.scale)),
                            );
                            ctx.world
                                .spawn((Transform::IDENTITY, Mesh3d(a.lights), hot.clone(), Parent(e)));
                            ctx.world
                                .spawn((Transform::IDENTITY, Mesh3d(a.streaks), glow_mat.clone(), Parent(e)));
                            (e, a.half_len)
                        };
                        cars.push(Car {
                            entity,
                            along_z,
                            dir,
                            line: city::street(s),
                            lane,
                            base: -span * 0.5 + span / n as f32 * k as f32 + rng.range(-12.0, 12.0),
                            speed,
                            half_len,
                        });
                    }
                }
            }
        }
    }
    // Parked cars (no lights on).
    for (pos, yaw, seed) in parked.into_iter().take(1500) {
        let rot = Quat::from_rotation_y(yaw);
        if car_assets.is_empty() {
            let h = vehicle_handles[seed as usize % (vehicle_handles.len() - 1)];
            ctx.world.spawn((
                Transform::from_xyz(pos.x, pos.y, pos.z).with_rotation(rot),
                Mesh3d(h),
                cold.clone(),
            ));
        } else {
            let a = &car_assets[seed as usize % car_assets.len()];
            let t = Transform::from_xyz(pos.x, pos.y, pos.z)
                .with_rotation(rot)
                .with_scale(Vec3::splat(a.scale));
            spawn_model(ctx.world, &a.model, t);
        }
    }
    // Hydrants, bins and benches (high-detail scans: only near the flythrough).
    let path: Vec<Vec3> = (0..flight::duration() as i32)
        .map(|t| flight::sample(t as f32).0)
        .collect();
    let near_path = |p: Vec3| path.iter().any(|q| q.distance_squared(p) < 70.0 * 70.0);
    let prop_models = [models.hydrant, models.trash, models.bench].map(|m| m.map(|m| m.upload(assets)));
    let mut counts = [0u32; 3];
    let caps = [150, 150, 100];
    for (kind, pos, yaw) in street_props {
        let k = kind as usize;
        let Some(model) = &prop_models[k] else {
            continue;
        };
        if counts[k] >= caps[k] || !near_path(pos) {
            continue;
        }
        counts[k] += 1;
        let t = Transform::from_xyz(pos.x, pos.y, pos.z).with_rotation(Quat::from_rotation_y(yaw));
        spawn_model(ctx.world, model, t);
    }

    // Pedestrians near the flythrough: the walking human in 16 outfits, two stride meshes each.
    let ped_cap = match quality0 {
        GraphicsQuality::Low => 150,
        GraphicsQuality::Medium => 350,
        _ => 700,
    };
    let people_meshes: Vec<[MeshHandle; 2]> = (0..people::OUTFITS as u32)
        .map_while(|i| {
            let o = people::outfit(i);
            Some([
                assets.add_mesh(people::mesh(&o, false)?),
                assets.add_mesh(people::mesh(&o, true)?),
            ])
        })
        .collect();
    let mut pedestrians = Vec::new();
    // (no pedestrian file: no pedestrians)
    let walkers = if people_meshes.len() == people::OUTFITS {
        walkers
    } else {
        Vec::new()
    };
    for w in walkers
        .into_iter()
        .filter(|w| path.iter().any(|q| q.distance_squared(w.origin) < 75.0 * 75.0))
    {
        if pedestrians.len() >= ped_cap {
            break;
        }
        let meshes = people_meshes[w.outfit as usize];
        let (p, yaw, parity) = people::pose(&w, 0.0);
        let entity = ctx.world.spawn((
            Transform::from_xyz(p.x, p.y, p.z).with_rotation(Quat::from_rotation_y(yaw)),
            Mesh3d(meshes[parity]),
            cold.clone(),
        ));
        pedestrians.push(Pedestrian {
            entity,
            walker: w,
            meshes,
            parity,
        });
    }

    // Particle effects: steam on the streets and roofs, drizzle and mist around the camera.
    let mut handles = Vec::new();
    {
        let reg = ctx.resources.expect_mut::<ParticleEffects>();
        for src in [effects::STEAM, effects::RAIN, effects::MIST] {
            match ParticleEffect::from_ron(src)
                .map_err(|e| e.to_string())
                .and_then(|fx| reg.add(fx).map_err(|e| e.to_string()))
            {
                Ok(h) => handles.push(h),
                Err(e) => eprintln!("city_demo: effect failed: {e}"),
            }
        }
    }
    let quality = ctx.resources.expect::<Demo>().quality.unwrap_or_default();
    let camera = ctx.resources.expect::<Demo>().camera;
    if handles.len() == 3 {
        let steam_count = match quality {
            GraphicsQuality::Low => 20,
            GraphicsQuality::Medium => 40,
            _ => 70,
        };
        for p in city.steam.iter().filter(|p| far_from_path(**p)).take(steam_count) {
            ctx.world.spawn((
                Transform::from_xyz(p.x, p.y, p.z),
                ParticleEmitter::new(handles[0]).with_prewarm(5.0),
            ));
        }
        if let Some(cam) = camera {
            for (h, prewarm) in [(handles[1], 1.2), (handles[2], 8.0)] {
                let e = ctx
                    .world
                    .spawn((Transform::IDENTITY, ParticleEmitter::new(h).with_prewarm(prewarm)));
                let _ = ctx.world.insert_one(e, Parent(cam));
            }
        }
    }

    // The sky is the background and the light.
    // The HDRI's sun is turned to the streets' azimuth and lights the city with its elevation.
    let target_az = 18f32.to_radians();
    let (rot, sun_dir) = match sun {
        Some((az, el)) => (
            Quat::from_rotation_y(az - target_az),
            Vec3::new(el.cos() * target_az.sin(), el.sin(), -el.cos() * target_az.cos()),
        ),
        None => (Quat::IDENTITY, procgen::sun_direction()),
    };
    for (t, _) in ctx.world.query_mut::<(&mut Transform, &DirectionalLight)>() {
        *t = Transform::IDENTITY.looking_at(-sun_dir, Vec3::Y);
    }
    if let Some(cam) = camera {
        let mut env =
            EnvironmentLight::new(sky).with_intensity(knob("CITY_ENV_DIFFUSE", 0.9), knob("CITY_ENV_SPECULAR", 0.7));
        env.rotation = rot;
        let mut sb = Skybox::new(sky);
        sb.rotation = rot;
        sb.intensity = knob("CITY_SKY", 0.55);
        sb.ground = Some((Vec3::new(0.52, 0.46, 0.55) * knob("CITY_GROUND", 0.7), 0.07));
        let _ = ctx.world.insert(cam, (sb, env));
    }
    let d = ctx.resources.expect_mut::<Demo>();
    d.sky = Some(sky);
    d.spots = city.lights;
    d.cars = cars;
    d.walkers = pedestrians;
    d.dressed = 0;
}

/// Where a car is at `time`: position and heading (yaw about +Y, 0 = +Z).
fn car_pose(c: &Car, time: f32) -> (Vec3, f32) {
    let span = 600.0;
    let along = (c.base + c.dir * c.speed * time + span * 0.5).rem_euclid(span) - span * 0.5;
    if c.along_z {
        // Right-hand traffic: heading +Z drives on the −X side of the street line.
        (
            Vec3::new(c.line - c.dir * c.lane, 0.0, along),
            if c.dir > 0.0 { 0.0 } else { std::f32::consts::PI },
        )
    } else {
        (
            Vec3::new(along, 0.0, c.line + c.dir * c.lane),
            if c.dir > 0.0 {
                std::f32::consts::FRAC_PI_2
            } else {
                -std::f32::consts::FRAC_PI_2
            },
        )
    }
}

fn pedestrians(ctx: &mut SystemCtx) {
    let d = ctx.resources.expect_mut::<Demo>();
    let time = d.time;
    for p in &mut d.walkers {
        let (pos, yaw, parity) = people::pose(&p.walker, time);
        if let Ok(mut t) = ctx.world.get::<&mut Transform>(p.entity) {
            t.translation = pos;
            t.rotation = Quat::from_rotation_y(yaw);
        }
        if parity != p.parity {
            p.parity = parity;
            if let Ok(mut m) = ctx.world.get::<&mut Mesh3d>(p.entity) {
                m.0 = p.meshes[parity];
            }
        }
    }
}

fn traffic(ctx: &mut SystemCtx) {
    let d = ctx.resources.expect::<Demo>();
    for c in &d.cars {
        let (p, yaw) = car_pose(c, d.time);
        if let Ok(mut t) = ctx.world.get::<&mut Transform>(c.entity) {
            t.translation = p;
            t.rotation = Quat::from_rotation_y(yaw);
        }
    }
}

/// Hand the lamps, signals, signs and car headlights nearest the camera to the real lights,
/// fading them in and out so none pops.
fn street_lights(ctx: &mut SystemCtx) {
    let dt = ctx.time().delta_secs();
    let d = ctx.resources.expect_mut::<Demo>();
    let Some(cam) = d.camera else {
        return;
    };
    let Ok(cam_pos) = ctx.world.get::<&Transform>(cam).map(|t| t.translation) else {
        return;
    };
    let lamp_scale = knob("CITY_LAMPS", 1.0);
    // candidates: (id, distance², spot)
    let mut cands: Vec<(u32, f32, LightSpot)> = Vec::new();
    for (i, s) in d.spots.iter().enumerate() {
        let d2 = s.pos.distance_squared(cam_pos);
        if d2 < 110.0 * 110.0 {
            cands.push((i as u32, d2, *s));
        }
    }
    for (i, c) in d.cars.iter().enumerate() {
        let (p, yaw) = car_pose(c, d.time);
        if p.distance_squared(cam_pos) > 90.0 * 90.0 {
            continue;
        }
        let fwd = Vec3::new(yaw.sin(), 0.0, yaw.cos());
        let head = p + fwd * (c.half_len + 1.8) + Vec3::Y * 0.8;
        cands.push((
            1_000_000 + i as u32,
            head.distance_squared(cam_pos),
            LightSpot {
                pos: head,
                color: Vec3::new(1.0, 0.9, 0.72),
                intensity: 7.0,
                range: 17.0,
            },
        ));
        let tail = p - fwd * (c.half_len + 0.6) + Vec3::Y * 0.7;
        cands.push((
            2_000_000 + i as u32,
            tail.distance_squared(cam_pos) + 400.0,
            LightSpot {
                pos: tail,
                color: Vec3::new(1.0, 0.06, 0.03),
                intensity: 1.6,
                range: 6.0,
            },
        ));
    }
    // Rank by distance discounted by reach and keep the best 14.
    cands.sort_by(|a, b| {
        let ka = a.1.sqrt() - a.2.range * 0.6;
        let kb = b.1.sqrt() - b.2.range * 0.6;
        ka.total_cmp(&kb)
    });
    cands.truncate(14);
    let mut taken = vec![false; cands.len()];
    for slot in &mut d.slots {
        if let Some(id) = slot.id {
            match cands.iter().position(|c| c.0 == id) {
                Some(k) => taken[k] = true,
                None => {
                    slot.fade = (slot.fade - dt * 3.0).max(0.0);
                    if slot.fade <= 0.0 {
                        slot.id = None;
                    }
                }
            }
        }
    }
    for (k, c) in cands.iter().enumerate() {
        if taken[k] {
            continue;
        }
        if let Some(slot) = d.slots.iter_mut().find(|s| s.id.is_none()) {
            slot.id = Some(c.0);
            slot.fade = 0.0;
        }
    }
    for slot in &mut d.slots {
        let spot = slot.id.and_then(|id| cands.iter().find(|c| c.0 == id));
        let Ok((t, l)) = ctx
            .world
            .query_one_mut::<(&mut Transform, &mut PointLight)>(slot.entity)
        else {
            continue;
        };
        if let Some(c) = spot {
            slot.fade = (slot.fade + dt * 3.0).min(1.0);
            let f = slot.fade * slot.fade * (3.0 - 2.0 * slot.fade);
            t.translation = c.2.pos;
            l.color = c.2.color;
            l.intensity = c.2.intensity * lamp_scale * f;
            l.range = c.2.range;
        } else if slot.id.is_none() {
            l.intensity = 0.0;
            t.translation.y = -500.0;
        } else {
            // leaving: its fade runs down (the light stays where it was)
            l.intensity *= 0.92;
        }
    }
}

/// Steam is kept off the flythrough: no source within 18 m of the camera path.
fn far_from_path(p: Vec3) -> bool {
    let mut t = 0.0;
    while t < flight::duration() {
        if flight::sample(t).0.distance(p) < 18.0 {
            return false;
        }
        t += 0.5;
    }
    true
}

/// `CITY_REPORT=1`: frame rate and GPU time once a second on stdout.
fn report(ctx: &mut SystemCtx) {
    if knob("CITY_REPORT", 0.0) < 0.5 {
        return;
    }
    let now = ctx.time().elapsed_secs();
    let (frames, since) = {
        static STATE: std::sync::Mutex<(u64, f64)> = std::sync::Mutex::new((0, 0.0));
        let mut s = STATE.lock().unwrap();
        s.0 += 1;
        if now - s.1 < 2.0 {
            return;
        }
        let r = (s.0, now - s.1);
        *s = (0, now);
        r
    };
    let gpu = ctx.resources.get::<GpuTimings>().map(|t| t.total_ms).unwrap_or(0.0);
    let d = ctx.resources.expect::<Demo>();
    println!(
        "fps {:.1} ({:.2} ms/frame), GPU scopes {:.2} ms, t={:.0}s, quality {}",
        frames as f64 / since,
        since * 1000.0 / frames as f64,
        gpu,
        d.time,
        d.quality.map_or("?".into(), |q| q.to_string())
    );
}

fn controls(ctx: &mut SystemCtx) {
    let Some(input) = ctx.resources.get::<Input>() else {
        return;
    };
    let (quit, tier, free, pause, hud, timings) = (
        input.just_pressed(KeyCode::Escape),
        input.just_pressed(KeyCode::KeyQ),
        input.just_pressed(KeyCode::KeyF),
        input.just_pressed(KeyCode::KeyP),
        input.just_pressed(KeyCode::KeyH),
        input.just_pressed(KeyCode::KeyG),
    );
    if quit {
        ctx.send(AppExit);
    }
    if tier && let Some(q) = ctx.resources.expect::<Demo>().quality {
        let next = GraphicsQuality::ALL[(GraphicsQuality::ALL.iter().position(|&t| t == q).unwrap_or(0) + 1) % 4];
        println!("quality: {next}");
        set_quality(ctx, next);
    }
    let cam = ctx.resources.expect::<Demo>().camera;
    let pose = cam.and_then(|c| ctx.world.get::<&Transform>(c).ok().map(|t| (t.translation, t.rotation)));
    let d = ctx.resources.expect_mut::<Demo>();
    d.paused ^= pause;
    d.hud ^= hud;
    d.timings ^= timings;
    if free {
        d.free = !d.free;
        if d.free
            && let Some((p, r)) = pose
        {
            let (yaw, pitch, _) = r.to_euler(EulerRot::YXZ);
            d.pos = p;
            d.yaw = yaw;
            d.pitch = pitch;
        }
    }
    let on = d.timings;
    ctx.resources.expect_mut::<FrameSettings>().gpu_timing = on;
}

/// The flythrough or the free camera.
fn drive_camera(ctx: &mut SystemCtx) {
    let dt = (ctx.time().delta_secs()).min(0.1);
    let (fwd, back, left, right, up, down, fast, look, mouse) = {
        let Some(input) = ctx.resources.get::<Input>() else {
            return;
        };
        (
            input.pressed(KeyCode::KeyW),
            input.pressed(KeyCode::KeyS),
            input.pressed(KeyCode::KeyA),
            input.pressed(KeyCode::KeyD),
            input.pressed(KeyCode::Space),
            input.pressed(KeyCode::ControlLeft),
            input.pressed(KeyCode::ShiftLeft),
            input.mouse_pressed(MouseButton::Right),
            input.mouse_delta(),
        )
    };
    let d = ctx.resources.expect_mut::<Demo>();
    let Some(cam) = d.camera else {
        return;
    };
    let shot_time = d.shots.as_ref().and_then(|s| s.times.get(s.index).copied());
    if let Some(t) = shot_time {
        d.time = t;
    } else if !d.paused {
        d.time += dt;
    }
    let transform = if d.free && shot_time.is_none() {
        if look {
            d.yaw -= mouse.x * 0.003;
            d.pitch = (d.pitch - mouse.y * 0.003).clamp(-1.5, 1.5);
        }
        let rot = Quat::from_euler(EulerRot::YXZ, d.yaw, d.pitch, 0.0);
        let mut v = Vec3::ZERO;
        v += rot * Vec3::NEG_Z * (i32::from(fwd) - i32::from(back)) as f32;
        v += rot * Vec3::X * (i32::from(right) - i32::from(left)) as f32;
        v += Vec3::Y * (i32::from(up) - i32::from(down)) as f32;
        let speed = if fast { 120.0 } else { 25.0 };
        d.pos += v.normalize_or_zero() * speed * dt;
        d.pos.y = d.pos.y.max(0.6);
        Transform::from_xyz(d.pos.x, d.pos.y, d.pos.z).with_rotation(rot)
    } else {
        let (p, t) = flight::sample(d.time);
        // a touch of handheld drift
        let s = if shot_time.is_some() { 0.0 } else { d.time };
        let sway = Vec3::new((s * 0.7).sin() * 0.08, (s * 0.9).sin() * 0.05, (s * 0.53).cos() * 0.06);
        Transform::from_xyz(p.x + sway.x, p.y + sway.y, p.z + sway.z).looking_at(t, Vec3::Y)
    };
    if let Ok(mut t) = ctx.world.get::<&mut Transform>(cam) {
        *t = transform;
    }
}

/// Headless: capture the requested flythrough times, one PNG each, then exit.
fn screenshots(ctx: &mut SystemCtx) {
    let d = ctx.resources.expect_mut::<Demo>();
    let ready = d.sky.is_some() && d.dressed > 90;
    let Some(shots) = d.shots.as_mut() else {
        return;
    };
    if !ready {
        return;
    }
    if shots.pending.is_none() {
        if shots.settle < 45 {
            shots.settle += 1;
            return;
        }
        if let Some(renderer) = ctx.resources.get_mut::<Renderer>() {
            let task = renderer.capture();
            ctx.resources.expect_mut::<Demo>().shots.as_mut().unwrap().pending = Some(task);
        }
        return;
    }
    let Some(result) = shots.pending.as_mut().and_then(Task::poll) else {
        return;
    };
    shots.pending = None;
    #[cfg(not(target_arch = "wasm32"))]
    {
        let t = shots.times[shots.index];
        let path = format!("{}/city_t{:04}.png", shots.dir, (t * 10.0).round() as i32);
        match result {
            Ok(image) => match space3d::test::save_png(&path, &image) {
                Ok(()) => println!("wrote {path} ({} × {})", image.width, image.height),
                Err(e) => eprintln!("could not write {path}: {e}"),
            },
            Err(e) => eprintln!("capture failed: {e}"),
        }
    }
    #[cfg(target_arch = "wasm32")]
    let _ = result;
    shots.index += 1;
    shots.settle = 0;
    if shots.index >= shots.times.len() {
        ctx.send(AppExit);
    }
}

fn hud(ui: &egui::Context, ctx: &mut SystemCtx) {
    let d = ctx.resources.expect::<Demo>();
    // until the city is up: what it is waiting for (shown even with the panel hidden)
    if d.sky.is_none() && !d.paused {
        let what = match &d.fetching {
            Some((p, ..)) => format!("Loading the city: fetching assets, {}", p.text()),
            None if d.quality.is_some() => "Loading the city: generating…".to_string(),
            None => "Loading the city…".to_string(),
        };
        egui::Area::new(egui::Id::new("city_loading"))
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ui, |ui| {
                ui.heading(what);
            });
    }
    if !d.hud {
        return;
    }
    let stats = ctx.resources.get::<RenderStats>().cloned().unwrap_or_default();
    let particles = ctx.resources.get::<ParticleStats>().cloned().unwrap_or_default();
    let gpu = ctx.resources.get::<GpuTimings>().map(|t| t.total_ms).unwrap_or(0.0);
    let fps = 1.0 / ctx.time().delta_secs().max(1e-6);
    egui::Window::new("City")
        .default_pos([12.0, 12.0])
        .resizable(false)
        .show(ui, |ui| {
            egui::Grid::new("city_stats").num_columns(2).show(ui, |ui| {
                let mut row = |k: &str, v: String| {
                    ui.label(k);
                    ui.monospace(v);
                    ui.end_row();
                };
                row("quality", d.quality.map_or("…".into(), |q| q.to_string()));
                row("fps", format!("{fps:.0}"));
                if d.timings {
                    row("GPU scopes", format!("{gpu:.2} ms"));
                }
                row(
                    "city",
                    match (&d.fetching, d.sky.is_some()) {
                        (Some((p, ..)), _) => format!("fetching {}", p.text()),
                        (None, true) => "ready".into(),
                        _ => "generating…".into(),
                    },
                );
                row(
                    "camera",
                    if d.free {
                        "free-fly".into()
                    } else {
                        format!("flythrough {:.0} s", d.time)
                    },
                );
                row(
                    "draw calls",
                    format!("{} + {} particles", stats.draw_calls, particles.draw_calls),
                );
                row("particles", format!("{}", particles.alive));
            });
            ui.small("F free-fly · WASD + right-drag · P pause · Q quality · G timings · H hide");
        });
}

space3d::main!(build);

#[cfg(test)]
mod tests {
    use super::*;

    /// Headless: the tier is chosen, the city arrives off the frame and renders without GPU
    /// errors, with lit pixels.
    #[test]
    fn city_renders_headless() {
        let mut app = App::new();
        app.add_plugin(DefaultPlugins::headless(480, 270));
        if app.resources.get::<Renderer>().is_none() {
            eprintln!("skipping: no GPU adapter");
            return;
        }
        add_game(&mut app, true);
        app.step_ticks(1);
        for _ in 0..30_000 {
            if app.resources.expect::<Demo>().sky.is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
            app.step_ticks(1);
        }
        assert!(app.resources.expect::<Demo>().sky.is_some(), "content arrived");
        app.step_ticks(30);
        let img = render_to_image(&mut app).expect("capture");
        assert_eq!(app.resources.expect::<Renderer>().gpu_error_count(), 0);
        let lit = img.pixels.chunks(4).filter(|p| p[0].max(p[1]).max(p[2]) > 60).count();
        assert!(lit > 2000, "{lit} lit pixels");
    }
}
