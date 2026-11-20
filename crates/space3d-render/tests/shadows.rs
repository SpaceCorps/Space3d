//! The directional shadow (graphics spec §5.2 PM5): one light with a `DirectionalShadow` casts
//! into a fitted shadow map drawn as a depth view (culled like a camera on every draw path) and
//! sampled with PCF by every lit receiver.
//!
//! The scenes look straight down on a ground plane (orthographic camera, 8 px per world unit,
//! screen right = +X, down = +Z) under a sun travelling along (−1, −1, 0): a point at height h
//! shadows the ground h units toward −X.
//!
//! Every test skips without a GPU adapter. `WGPU_BACKEND=gl` (with ANGLE, see `tests/gl.rs`)
//! runs them on the WebGL2-class GL backend (`sampler2DShadow`, CPU culling).

mod common;
use common::*;
use space3d_core::prelude::*;
use space3d_render::*;

const S: u32 = 64;
/// World units the camera sees vertically (and horizontally: the target is square).
const VIEW: f32 = 8.0;

fn app_with(samples: u32) -> Option<App> {
    let mut app = App::new();
    app.add_plugin(RenderPlugin::default());
    match pollster::block_on(Renderer::new_headless_with(RendererConfig {
        width: S,
        height: S,
        msaa_samples: samples,
        ..Default::default()
    })) {
        Ok(r) => {
            app.insert_resource(r);
            Some(app)
        }
        Err(e) => {
            eprintln!("skipping GPU test: no adapter available ({e})");
            None
        }
    }
}

fn render(app: &mut App) -> ImageData {
    let img = render_to_image(app).expect("capture");
    assert_eq!(app.resources.expect::<Renderer>().gpu_error_count(), 0, "GPU errors");
    // `SPACE3D_TEST_OUT=<dir>` keeps every capture (named after the test) for inspection.
    if let Ok(dir) = std::env::var("SPACE3D_TEST_OUT") {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let name = std::thread::current().name().unwrap_or("x").replace("::", "_");
        space3d_test::save_png(format!("{dir}/{name}_{n}.png"), &img).unwrap();
    }
    img
}

/// Pixel showing ground point (x, z).
fn pixel(x: f32, z: f32) -> (u32, u32) {
    let px = |v: f32| ((v + VIEW * 0.5) * S as f32 / VIEW) as u32;
    (px(x).min(S - 1), px(z).min(S - 1))
}

/// Mean of RGB (sRGB bytes) at ground point (x, z).
fn value(img: &ImageData, x: f32, z: f32) -> f32 {
    let (px, py) = pixel(x, z);
    let p = img.get(px, py);
    (p[0] as f32 + p[1] as f32 + p[2] as f32) / 3.0
}

/// The top-down camera, the ground (a lit grey plane at y = 0) and a low ambient light.
fn ground_scene(app: &mut App, ground: Material) -> MeshHandle {
    app.world.spawn((
        Transform::from_xyz(0.0, 20.0, 0.0).looking_at(Vec3::ZERO, Vec3::NEG_Z),
        Camera3d::default()
            .with_clear_color(Vec4::new(0.0, 0.0, 0.0, 1.0))
            .with_projection(Projection::Orthographic {
                height: VIEW,
                near: 0.1,
                far: 40.0,
            }),
    ));
    app.insert_resource(AmbientLight {
        color: Vec3::ONE,
        intensity: 0.1,
    });
    let assets = app.resources.expect_mut::<RenderAssets>();
    let plane = assets.add_mesh(MeshData::plane(16.0));
    let cube = assets.add_mesh(MeshData::cube(1.0));
    app.world.spawn((Transform::IDENTITY, Mesh3d(plane), ground));
    cube
}

fn grey() -> Material {
    Material::srgb(0.8, 0.8, 0.8).with_metallic_roughness(0.0, 1.0)
}

/// The sun along `direction` with `shadow`.
fn sun(app: &mut App, direction: Vec3, shadow: Option<DirectionalShadow>) -> Entity {
    let e = app.world.spawn((
        Transform::IDENTITY.looking_at(direction, Vec3::Y),
        DirectionalLight {
            color: Vec3::ONE,
            intensity: 2.0,
        },
    ));
    if let Some(s) = shadow {
        app.world.insert_one(e, s).unwrap();
    }
    e
}

/// The shadow the tests use: fitted to the camera over the slab |y| ≤ 3, a 512² map.
fn shadow() -> DirectionalShadow {
    DirectionalShadow::new()
        .with_map_size(512)
        .with_fit(ShadowFit::top_down(0.0, 3.0, 100.0))
}

const SUN: Vec3 = Vec3::new(-1.0, -1.0, 0.0);

/// `lit` is clearly brighter than `dark` (the ground in the sun against its shadow).
fn assert_shadowed(img: &ImageData, dark: (f32, f32), lit: (f32, f32), what: &str) {
    let (d, l) = (value(img, dark.0, dark.1), value(img, lit.0, lit.1));
    assert!(d < 0.45 * l, "{what}: shadowed {d} vs lit {l}");
}

fn assert_lit(img: &ImageData, at: (f32, f32), lit: (f32, f32), what: &str) {
    let (a, l) = (value(img, at.0, at.1), value(img, lit.0, lit.1));
    assert!((a - l).abs() <= 3.0, "{what}: {a} vs lit {l}");
}

/// Two cubes and a sphere on the ground, drawn on `path` (or the adapter's best), optionally
/// under `DownlevelOverride::webgl2()`.
fn parity_scene(path: Option<DrawPath>, webgl2: bool) -> Option<ImageData> {
    let mut app = app_with(1)?;
    if let Some(p) = path {
        set_path(&mut app, p);
    }
    if webgl2 {
        app.insert_resource(FrameSettings::default().with_downlevel(DownlevelOverride::webgl2()));
    }
    let cube = ground_scene(&mut app, grey());
    let sphere = app
        .resources
        .expect_mut::<RenderAssets>()
        .add_mesh(MeshData::sphere(0.6, 24, 12));
    app.world.spawn((
        Transform::from_xyz(1.5, 1.0, 0.0),
        Mesh3d(cube),
        Material::srgb(0.9, 0.3, 0.2),
    ));
    app.world.spawn((
        Transform::from_xyz(-1.0, 0.5, 2.5).with_rotation(Quat::from_rotation_y(0.6)),
        Mesh3d(cube),
        Material::srgb(0.2, 0.5, 0.9),
    ));
    app.world.spawn((
        Transform::from_xyz(2.0, 2.0, -2.5),
        Mesh3d(sphere),
        Material::srgb(0.3, 0.9, 0.3),
    ));
    sun(&mut app, SUN, Some(shadow()));
    let img = render(&mut app);
    let s = stats(&app);
    assert_eq!(s.depth_views, 1, "{s:?}");
    assert!(s.shadow_casters >= 4 && s.shadow_draw_calls >= 1, "{s:?}");
    Some(img)
}

/// The same shadows on every draw path and under `DownlevelOverride::webgl2()`: the depth view
/// culls its casters on the CPU or as one more frustum of the GPU culling, and the image is
/// exactly the same.
#[test]
fn shadow_same_on_every_path() {
    let Some(first) = parity_scene(Some(DrawPath::CpuDirect), false) else {
        return;
    };
    // The red cube (x 1..2, top at 1.5) shadows x −0.5..1 on the ground.
    assert_shadowed(&first, (0.0, 0.0), (-2.5, 0.0), "cube shadow");
    // The sphere (center (2, 2, −2.5)) shadows around (0, −2.5).
    assert_shadowed(&first, (0.0, -2.5), (-2.5, -2.5), "sphere shadow");
    let best = {
        let app = app_with(1).unwrap();
        best_path(&app)
    };
    for path in paths_up_to(best).into_iter().skip(1) {
        let img = parity_scene(Some(path), false).unwrap();
        assert_eq!(diff_fraction(&img, &first, 0), 0.0, "{path:?}");
    }
    let webgl2 = parity_scene(None, true).unwrap();
    assert_eq!(diff_fraction(&webgl2, &first, 0), 0.0, "webgl2()");
}

/// A caster outside the camera's view (toward the light) still shadows what the camera sees,
/// on every draw path: the depth view culls against the light's frustum, which reaches
/// `caster_depth` toward the light.
#[test]
fn offscreen_caster_still_casts() {
    let Some(app) = app_with(1) else { return };
    let best = best_path(&app);
    drop(app);
    for path in paths_up_to(best) {
        let mut app = app_with(1).unwrap();
        set_path(&mut app, path);
        let cube = ground_scene(&mut app, grey());
        // At x = 6 the cube is outside the view (x within ±4); its shadow falls around x = 0.
        app.world.spawn((
            Transform::from_xyz(6.0, 6.0, 0.0),
            Mesh3d(cube),
            Material::srgb(0.9, 0.9, 0.9),
        ));
        sun(&mut app, SUN, Some(shadow()));
        let img = render(&mut app);
        assert_shadowed(&img, (0.0, 0.0), (0.0, 2.5), &format!("{path:?}"));
        let s = stats(&app);
        assert_eq!((s.depth_views, s.shadow_casters), (1, 2), "{path:?}: ground + cube");
    }
}

/// `DirectionalShadow::min_caster_pixels`: under a perspective camera a caster whose bounding
/// sphere is smaller on screen than that casts no shadow, a larger one still does, and 0 keeps
/// both — on every draw path (the CPU's frustum test and the GPU culling's).
#[test]
fn casters_under_the_pixel_size_cast_no_shadow() {
    let Some(app) = app_with(1) else { return };
    let best = best_path(&app);
    drop(app);
    for path in paths_up_to(best) {
        let scene = |min_pixels: f32| {
            let mut app = app_with(1).unwrap();
            set_path(&mut app, path);
            // The ground 20 units below the eye fills the same 8 units as the other scenes:
            // 160 px per unit of radius ÷ distance.
            app.world.spawn((
                Transform::from_xyz(0.0, 20.0, 0.0).looking_at(Vec3::ZERO, Vec3::NEG_Z),
                Camera3d::default()
                    .with_clear_color(Vec4::new(0.0, 0.0, 0.0, 1.0))
                    .with_projection(Projection::Perspective {
                        fov_y: 2.0 * (VIEW * 0.5 / 20.0).atan(),
                        near: 0.1,
                        far: 40.0,
                    }),
            ));
            app.insert_resource(AmbientLight {
                color: Vec3::ONE,
                intensity: 0.1,
            });
            let assets = app.resources.expect_mut::<RenderAssets>();
            let plane = assets.add_mesh(MeshData::plane(16.0));
            let cube = assets.add_mesh(MeshData::cube(1.0));
            app.world.spawn((Transform::IDENTITY, Mesh3d(plane), grey()));
            // About 7 px of radius: shadows the ground around (0, 0).
            app.world
                .spawn((Transform::from_xyz(1.5, 1.0, 0.0), Mesh3d(cube), grey()));
            // About 3 px: shadows the ground around (0.5, 2.5).
            app.world.spawn((
                Transform::from_xyz(1.5, 1.0, 2.5).with_scale(Vec3::splat(0.4)),
                Mesh3d(cube),
                grey(),
            ));
            sun(&mut app, SUN, Some(shadow().with_min_caster_pixels(min_pixels)));
            render(&mut app)
        };
        let all = scene(0.0);
        assert_shadowed(&all, (0.0, 0.0), (-2.5, -2.5), &format!("{path:?}: the large cube"));
        assert_shadowed(&all, (0.5, 2.5), (-2.5, -2.5), &format!("{path:?}: the small cube"));
        let cut = scene(5.0);
        assert_shadowed(
            &cut,
            (0.0, 0.0),
            (-2.5, -2.5),
            &format!("{path:?}: the large cube, cut at 5 px"),
        );
        assert_lit(
            &cut,
            (0.5, 2.5),
            (-2.5, -2.5),
            &format!("{path:?}: the small cube, cut at 5 px"),
        );
    }
}

/// A light at 45° and at 80° from the vertical over a flat ground: with the default biases no
/// texel of the ground shadows itself (no acne) for any filter, with a fine and a coarse map,
/// while a cube still casts its shadow.
#[test]
fn no_acne_at_grazing_angle() {
    for (angle, size, filter) in [45.0f32, 80.0]
        .into_iter()
        .flat_map(|a| [512, 64].map(move |s| (a, s)))
        .flat_map(|(a, s)| {
            [
                ShadowFilter::Hard,
                ShadowFilter::Pcf4,
                ShadowFilter::Pcf9,
                ShadowFilter::Pcf16,
            ]
            .map(move |f| (a, s, f))
        })
    {
        let what = format!("{angle}° {size}² {filter:?}");
        let Some(mut app) = app_with(1) else { return };
        let cube = ground_scene(&mut app, grey());
        app.world.spawn((
            Transform::from_xyz(2.5, 0.25, -2.5),
            Mesh3d(cube),
            Material::srgb(0.9, 0.9, 0.9),
        ));
        let dir = Vec3::new(-angle.to_radians().sin(), -angle.to_radians().cos(), 0.0);
        sun(&mut app, dir, Some(shadow().with_filter(filter).with_map_size(size)));
        let img = render(&mut app);
        // The ground away from the cube and its shadow: every pixel within 2 of the median.
        let mut values = Vec::new();
        for py in S / 2..S {
            for px in 0..S {
                let p = img.get(px, py);
                values.push(p[0].max(p[1]).max(p[2]));
            }
        }
        let mut sorted = values.clone();
        sorted.sort_unstable();
        let median = sorted[sorted.len() / 2];
        let acne = values.iter().filter(|&&v| v.abs_diff(median) > 2).count();
        assert_eq!(acne, 0, "{what}: {acne} ground pixels differ from {median}");
        // The cube (0.25 up, 0.5 half size: top at 0.75) casts 0.75 · tan θ toward −X (the
        // grazing sun lights the ground at only cos 80°: ambient is a larger share). The coarse
        // map's wide kernels blur that thin shadow away.
        let (dark, lit) = (value(&img, 1.5, -2.5), value(&img, 1.5, 2.5));
        assert!(size < 512 || dark < lit - 40.0, "{what}: shadow {dark} vs {lit}");
    }
}

/// With `ShadowMode::Auto`, unlit and transparent materials cast nothing, lit opaque ones
/// cast; `CastOnly` makes an unlit one cast, `None` stops a lit one; receivers with `CastOnly`
/// or `None` stay lit.
#[test]
fn unlit_and_transparent_do_not_cast_by_default() {
    let Some(mut app) = app_with(1) else { return };
    let cube = ground_scene(&mut app, grey());
    // Four cubes in a column along z, each shadowing the ground 1.5 units toward −X.
    let casters = [
        (-3.0, Material::unlit(Vec4::new(1.0, 1.0, 0.0, 1.0))),
        (
            -1.0,
            Material::color(Vec4::new(0.05, 0.8, 0.05, 0.5)).with_blend(BlendMode::AlphaBlend),
        ),
        (1.0, Material::srgb(0.9, 0.2, 0.2)),
        (3.0, Material::srgb(0.2, 0.2, 0.9).with_shadows(ShadowMode::None)),
    ];
    for (z, m) in casters {
        app.world.spawn((Transform::from_xyz(2.0, 1.0, z), Mesh3d(cube), m));
    }
    sun(&mut app, SUN, Some(shadow()));
    let img = render(&mut app);
    let lit = (-3.0, 0.0);
    assert_lit(&img, (0.5, -3.0), lit, "unlit caster");
    assert_lit(&img, (0.5, -1.0), lit, "transparent caster");
    assert_shadowed(&img, (0.5, 1.0), lit, "opaque caster");
    assert_lit(&img, (0.5, 3.0), lit, "ShadowMode::None caster");

    // An unlit caster with `CastOnly` casts.
    for (m,) in app.world.query_mut::<(&mut Material,)>() {
        if m.unlit {
            *m = m.clone().with_shadows(ShadowMode::CastOnly);
        }
    }
    let img = render(&mut app);
    assert_shadowed(&img, (0.5, -3.0), lit, "unlit CastOnly caster");

    // A ground that does not receive stays lit everywhere.
    for mode in [ShadowMode::CastOnly, ShadowMode::None] {
        for (m,) in app.world.query_mut::<(&mut Material,)>() {
            if m.base_color.x > 0.5 && m.base_color.y > 0.5 && m.base_color.z > 0.5 && !m.unlit {
                *m = grey().with_shadows(mode);
            }
        }
        let img = render(&mut app);
        assert_lit(&img, (0.5, 1.0), lit, &format!("ground {mode:?}"));
    }
}

/// Explicit `InstanceBuffer`s cast like ECS entities, on every draw path: the same cubes from
/// an instance buffer and from entities give the same image.
#[test]
fn explicit_instance_buffers_cast() {
    let Some(app) = app_with(1) else { return };
    let best = best_path(&app);
    drop(app);
    let positions = [
        Vec3::new(1.5, 1.0, 0.0),
        Vec3::new(-1.0, 0.8, 2.5),
        Vec3::new(2.5, 2.0, -2.5),
    ];
    for path in paths_up_to(best) {
        let render_with = |explicit: bool| {
            let mut app = app_with(1).unwrap();
            set_path(&mut app, path);
            let cube = ground_scene(&mut app, grey());
            let material = Material::srgb(0.9, 0.5, 0.2);
            if explicit {
                let mut buf = InstanceBuffer::with_capacity(4, BufferUsage::Static);
                for p in positions {
                    buf.push(InstanceCompact::new(p, Quat::IDENTITY, 1.0));
                }
                let instances = app.resources.expect_mut::<InstanceBuffers>().insert(buf);
                app.world.spawn((InstancedMesh {
                    mesh: cube,
                    material,
                    instances,
                },));
            } else {
                for p in positions {
                    app.world
                        .spawn((Transform::from_xyz(p.x, p.y, p.z), Mesh3d(cube), material.clone()));
                }
            }
            sun(&mut app, SUN, Some(shadow()));
            let img = render(&mut app);
            (img, stats(&app))
        };
        let (ecs, _) = render_with(false);
        let (explicit, s) = render_with(true);
        assert_shadowed(&explicit, (0.0, 0.0), (-2.5, 0.0), &format!("{path:?}"));
        assert!(s.shadow_casters >= 4, "{path:?}: {s:?}");
        assert!(
            diff_fraction(&ecs, &explicit, 2) < 0.002,
            "{path:?}: explicit vs ECS casters"
        );
    }
}

/// An alpha-masked caster casts the cut-out shape: a quad with a 2×2 checker of opaque and
/// cut-out texels shadows only the opaque quarters.
#[test]
fn alpha_mask_casts_masked_shadow() {
    let Some(mut app) = app_with(1) else { return };
    ground_scene(&mut app, grey());
    let mut checker = ImageData::new(2, 2);
    for (x, y) in [(0, 0), (1, 1)] {
        checker.set(x, y, [255, 255, 255, 255]);
    }
    for (x, y) in [(1, 0), (0, 1)] {
        checker.set(x, y, [255, 255, 255, 0]);
    }
    let a = app.resources.expect_mut::<RenderAssets>();
    let mut source = TextureSource::from_image(checker);
    source.sampler = SamplerSettings {
        filter: FilterMode::Nearest,
        address: AddressMode::Clamp,
    };
    let tex = a.add_texture(source);
    let quad = a.add_mesh(MeshData::plane(2.0));
    // A 2 × 2 horizontal quad 1.5 up over x 1..3 (its shadow lands on x −0.5..1.5), both sides.
    app.world.spawn((
        Transform::from_xyz(2.0, 1.5, 0.0),
        Mesh3d(quad),
        Material {
            double_sided: true,
            ..Material::srgb(0.9, 0.9, 0.9).with_texture(tex).with_alpha_mask(0.5)
        },
    ));
    sun(&mut app, SUN, Some(shadow()));
    let img = render(&mut app);
    // plane(2) UVs: u along +X, v = 0 at −Z. Texel (0, 0) (opaque) covers x 1..2, z −1..0 and
    // shadows x −0.5..0.5; (1, 0) (cut) x 2..3 → x 0.5..1.5 (visible up to x 1, where the quad
    // starts); (0, 1) (cut) and (1, 1) (opaque) likewise at z 0..1.
    let lit = (-3.0, 2.5);
    assert_shadowed(&img, (0.0, -0.5), lit, "opaque quarter (0, 0)");
    assert_lit(&img, (0.75, -0.5), lit, "cut quarter (1, 0)");
    assert_lit(&img, (0.0, 0.5), lit, "cut quarter (0, 1)");
    assert_shadowed(&img, (0.75, 0.5), lit, "opaque quarter (1, 1)");
}

/// The filters soften the edge (Hard has the fewest partly shadowed pixels, PCF16 the most),
/// and `strength` scales the shadow.
#[test]
fn filters_soften_and_strength_scales() {
    let render_with = |shadow: DirectionalShadow| -> Option<ImageData> {
        let mut app = app_with(1)?;
        let cube = ground_scene(&mut app, grey());
        app.world.spawn((
            Transform::from_xyz(1.5, 1.0, 0.0),
            Mesh3d(cube),
            Material::srgb(0.9, 0.9, 0.9),
        ));
        sun(&mut app, SUN, Some(shadow));
        Some(render(&mut app))
    };
    // A coarse map (64² over the ~17-unit fit): texels of about a quarter unit, 2 pixels.
    let coarse = shadow().with_map_size(64);
    let Some(hard) = render_with(coarse.with_filter(ShadowFilter::Hard)) else {
        return;
    };
    let lit = value(&hard, -2.5, 0.0);
    let dark = value(&hard, 0.0, 0.0);
    // Partly shadowed ground pixels along z = −1..1, x −1.5..1 (the shadow's edges).
    let penumbra = |img: &ImageData| {
        let mut n = 0;
        for z in -8..8 {
            for x in -12..8 {
                let v = value(img, x as f32 / 8.0, z as f32 / 8.0);
                if v > dark + 4.0 && v < lit - 4.0 {
                    n += 1;
                }
            }
        }
        n
    };
    let counts: Vec<usize> = [
        ShadowFilter::Hard,
        ShadowFilter::Pcf4,
        ShadowFilter::Pcf9,
        ShadowFilter::Pcf16,
    ]
    .into_iter()
    .map(|f| penumbra(&render_with(coarse.with_filter(f)).unwrap()))
    .collect();
    assert!(
        counts.windows(2).all(|w| w[0] <= w[1]) && counts[3] > counts[0],
        "penumbra pixels per filter: {counts:?}"
    );
    // Half strength: the shadowed ground halfway (in linear light) between lit and shadowed.
    let half = render_with(shadow().with_strength(0.5)).unwrap();
    let full = render_with(shadow()).unwrap();
    let lin = |v: f32| srgb_to_linear(v / 255.0);
    let (l, d, h) = (
        lin(value(&full, -2.5, 0.0)),
        lin(value(&full, 0.0, 0.0)),
        lin(value(&half, 0.0, 0.0)),
    );
    assert!((h - (l + d) * 0.5).abs() < 0.03, "half strength {h} vs {l}/{d}");
}

/// Shadows off is exactly the image without them: adding and removing the `DirectionalShadow`
/// returns to the same pixels (and pipelines), and a second light asking to cast is ignored
/// (the lowest entity casts).
#[test]
fn removing_the_shadow_restores_the_image() {
    let Some(mut app) = app_with(1) else { return };
    let cube = ground_scene(&mut app, grey());
    app.world.spawn((
        Transform::from_xyz(1.5, 1.0, 0.0),
        Mesh3d(cube),
        Material::srgb(0.9, 0.3, 0.2),
    ));
    let light = sun(&mut app, SUN, None);
    let before = render(&mut app);
    assert_eq!(stats(&app).depth_views, 0);
    app.world.insert_one(light, shadow()).unwrap();
    let with = render(&mut app);
    assert_shadowed(&with, (0.0, 0.0), (-2.5, 0.0), "with the shadow");
    // A second, dim shadowed light (spawned later: higher entity bits) does not cast.
    let other = app.world.spawn((
        Transform::IDENTITY.looking_at(Vec3::new(1.0, -1.0, 0.0), Vec3::Y),
        DirectionalLight {
            color: Vec3::ONE,
            intensity: 0.2,
        },
        shadow(),
    ));
    let two = render(&mut app);
    assert_eq!(stats(&app).depth_views, 1);
    assert_shadowed(&two, (0.0, 0.0), (-2.5, 0.0), "the first light still casts");
    app.world.despawn(other).unwrap();
    app.world.remove_one::<DirectionalShadow>(light).unwrap();
    let after = render(&mut app);
    assert_eq!(stats(&app).depth_views, 0);
    assert_eq!(diff_fraction(&before, &after, 0), 0.0, "without the shadow again");
}

/// A still scene with a shadow reaches a steady state: nothing uploaded, no texture created or
/// buffer reallocated, identical frames; changing the map size creates one texture.
#[test]
fn static_scene_uploads_nothing_with_shadows() {
    let Some(mut app) = app_with(4) else { return };
    let cube = ground_scene(&mut app, grey());
    for i in 0..5 {
        app.world.spawn((
            Transform::from_xyz(i as f32 - 2.0, 0.5 + i as f32 * 0.3, (i as f32 * 1.7) % 3.0 - 1.5),
            Mesh3d(cube),
            Material::srgb(0.9, 0.5, 0.2),
        ));
    }
    let light = sun(&mut app, SUN, Some(shadow()));
    let first = render(&mut app);
    assert!(stats(&app).textures_created > 0);
    for _ in 0..3 {
        render(&mut app);
    }
    for _ in 0..3 {
        let img = render(&mut app);
        let s = stats(&app);
        assert_eq!(
            (
                s.bytes_uploaded,
                s.textures_created,
                s.cpu_reallocations,
                s.buffer_reallocations
            ),
            (0, 0, 0, 0),
            "{s:?}"
        );
        assert_eq!(img, first);
    }
    app.world.get::<&mut DirectionalShadow>(light).unwrap().map_size = 256;
    render(&mut app);
    assert_eq!(stats(&app).textures_created, 1, "a new map");
    render(&mut app);
    assert_eq!(stats(&app).textures_created, 0);
}

/// Every material feature at once — maps (normal, ORM, emissive), a skybox, image-based
/// lighting with Standard shading, the shadow, alpha-masked, blended and additive materials,
/// post-processing — renders the same image twice in one app and in a second app built the
/// same way (graphics spec §4.3).
#[test]
fn render_to_image_is_deterministic_with_all_features() {
    let build = || -> Option<App> {
        let mut app = app_with(4)?;
        app.resources.expect_mut::<LightingSettings>().shading = ShadingModel::Standard;
        let cube = ground_scene(&mut app, grey());
        let a = app.resources.expect_mut::<RenderAssets>();
        let faces = std::array::from_fn(|i| ImageData::filled(16, 16, [30 + 30 * i as u8, 60, 120, 255]));
        let sky = a.add_cubemap(CubemapSource::from_faces(faces).unwrap());
        let normal = a.add_image_linear(ImageData::checker(8, 2, [150, 110, 240, 255], [110, 150, 240, 255]));
        let orm = a.add_image_linear(ImageData::filled(4, 4, [230, 120, 200, 255]));
        let glow = a.add_image(ImageData::checker(8, 2, [255, 140, 20, 255], [0, 0, 0, 255]));
        let mask = a.add_image(ImageData::checker(8, 2, [255, 255, 255, 255], [255, 255, 255, 0]));
        let hull = Material::srgb(0.7, 0.72, 0.75)
            .with_normal_map(normal, 1.0)
            .with_orm_map(orm)
            .with_emissive_map(glow, Vec3::splat(2.0));
        for (i, m) in [
            hull.clone(),
            Material::srgb(0.8, 0.8, 0.8).with_texture(mask).with_alpha_mask(0.5),
            Material::color(Vec4::new(0.2, 0.6, 1.0, 0.4)).with_blend(BlendMode::AlphaBlend),
            Material::additive(Vec4::new(1.0, 0.4, 0.1, 1.0)),
        ]
        .into_iter()
        .enumerate()
        {
            app.world
                .spawn((Transform::from_xyz(i as f32 * 1.6 - 2.4, 1.0, 0.5), Mesh3d(cube), m));
        }
        sun(&mut app, SUN, Some(shadow()));
        let camera = app
            .world
            .query::<(Entity, &Camera3d)>()
            .iter()
            .map(|(e, _)| e)
            .next()
            .unwrap();
        app.world
            .insert(
                camera,
                (Skybox::new(sky), EnvironmentLight::new(sky), PostProcess::default()),
            )
            .unwrap();
        Some(app)
    };
    let Some(mut a) = build() else { return };
    let first = render(&mut a);
    let again = render(&mut a);
    assert_eq!(diff_fraction(&first, &again, 0), 0.0, "twice in one app");
    let mut b = build().unwrap();
    let other = render(&mut b);
    assert_eq!(diff_fraction(&first, &other, 0), 0.0, "a second app");
    assert_eq!(stats(&b).depth_views, 1);
}

/// Perf gate "shadow 2048², 100k instanced casters ≤ 0.30 ms" (graphics spec §4.4): the
/// `shadow` pass (GPU timestamps) of 100 000 cube entities (one instanced batch) all inside the
/// map (a 400 × 400 field under a top-down camera), with the default filter; median of 120
/// frames rendered back to back. For the record, the same cubes from one explicit
/// `InstanceBuffer<InstanceCompact>` (as fast as the ECS buffer on the merged M2 tree, 0.275 vs
/// 0.277 ms; the 2.4× gap recorded at PM5 did not reproduce: docs/performance.md, "M2 merge").
/// Needs `TIMESTAMP_QUERY`; run alone:
/// `cargo test -p space3d-render --release --test shadows -- --ignored --nocapture shadow_map_gate`.
#[test]
#[ignore = "perf gate: run alone with --ignored"]
fn shadow_map_gate() {
    let mut gated = None;
    for explicit in [false, true] {
        let Some(mut app) = headless_app(1920, 1080) else {
            return;
        };
        if !app.resources.expect::<Renderer>().graphics_caps().timestamps {
            eprintln!("skipping: no TIMESTAMP_QUERY");
            return;
        }
        app.insert_resource(FrameSettings::default().with_gpu_timing(true));
        let a = app.resources.expect_mut::<RenderAssets>();
        let cube = a.add_mesh(MeshData::cube(1.0));
        let plane = a.add_mesh(MeshData::plane(420.0));
        app.world.spawn((Transform::IDENTITY, Mesh3d(plane), grey()));
        let at = |i: u32| {
            let (x, z) = ((i % 316) as f32, (i / 316) as f32);
            let h = (i.wrapping_mul(2_654_435_761) >> 28) as f32;
            (
                Vec3::new(x * 1.26 - 199.0, 1.0 + h * 0.3, z * 1.26 - 199.0),
                i as f32 * 0.37,
            )
        };
        let material = Material::srgb(0.7, 0.6, 0.5);
        if explicit {
            let mut buf = InstanceBuffer::with_capacity(100_000, BufferUsage::Static);
            for i in 0..100_000 {
                let (p, yaw) = at(i);
                buf.push(InstanceCompact::new(p, Quat::from_rotation_y(yaw), 0.8));
            }
            let instances = app.resources.expect_mut::<InstanceBuffers>().insert(buf);
            app.world.spawn((InstancedMesh {
                mesh: cube,
                material,
                instances,
            },));
        } else {
            for i in 0..100_000 {
                let (p, yaw) = at(i);
                app.world.spawn((
                    Transform::from_xyz(p.x, p.y, p.z)
                        .with_rotation(Quat::from_rotation_y(yaw))
                        .with_scale(Vec3::splat(0.8)),
                    Mesh3d(cube),
                    material.clone(),
                ));
            }
        }
        app.world.spawn((
            Transform::from_xyz(0.0, 260.0, 0.0).looking_at(Vec3::ZERO, Vec3::NEG_Z),
            Camera3d::default().with_projection(Projection::Perspective {
                fov_y: 70f32.to_radians(),
                near: 1.0,
                far: 600.0,
            }),
        ));
        sun(
            &mut app,
            Vec3::new(-0.3, -1.0, -0.2),
            Some(
                DirectionalShadow::new()
                    .with_map_size(2048)
                    .with_fit(ShadowFit::top_down(0.0, 10.0, 2000.0)),
            ),
        );
        let frame = |app: &mut App| {
            space3d_core::transform::propagate_transforms(&mut app.world);
            app.run_stage(Stage::Render);
            app.resources.expect::<RenderStats>().frame
        };
        for _ in 0..30 {
            frame(&mut app);
        }
        let mut times = Vec::new();
        let mut scene = Vec::new();
        let mut last = None;
        let mut i = 0;
        while times.len() < 120 {
            frame(&mut app);
            let t = app.resources.expect::<GpuTimings>();
            if t.frame != last
                && let Some(f) = t.frame
            {
                last = Some(f);
                times.push(t.scope_ms("shadow"));
                scene.push(t.scope_ms("scene"));
            }
            i += 1;
            assert!(i < 2000, "timings do not arrive");
        }
        let _ = app
            .resources
            .expect::<Renderer>()
            .device()
            .poll(wgpu::PollType::wait_indefinitely());
        let s = stats(&app);
        assert!(s.shadow_casters >= 100_000, "{s:?}");
        times.sort_by(f32::total_cmp);
        let median = times[times.len() / 2];
        scene.sort_by(f32::total_cmp);
        eprintln!(
            "shadow 2048², 100k {} casters: median {median:.3} ms (min {:.3}, max {:.3}); scene pass {:.3} ms; {} shadow draws, path {:?}",
            if explicit { "explicit-buffer" } else { "ECS" },
            times[0],
            times[times.len() - 1],
            scene[scene.len() / 2],
            s.shadow_draw_calls,
            s.draw_path
        );
        if !explicit {
            gated = Some(median);
        }
    }
    let scale = std::env::var("SPACE3D_PERF_BUDGET_SCALE")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(1.0);
    let median = gated.unwrap();
    assert!(median <= 0.30 * scale, "{median:.3} ms > 0.30 ms");
}
