//! Render a small wet-street scene (ground plane, buildings, a glassy slab, a metal sphere) with
//! each of the city effects off and on, and save one PNG per variant.
//!
//! `cargo run -p space3d-render --example city_effects -- out_dir [width height]`
//!
//! Writes `base.png`, `fog.png`, `ssao.png`, `ssr.png` and `all.png` into `out_dir`
//! (docs/graphics/post.md, "City effects").

use space3d_core::HdrImageData;
use space3d_core::prelude::*;
use space3d_render::*;

/// A vertical sky gradient panorama (linear HDR): bright warm horizon, blue zenith, dark ground.
fn sky_panorama() -> HdrImageData {
    let (w, h) = (128u32, 64u32);
    let mut pixels = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        let t = (y as f32 + 0.5) / h as f32; // 0 zenith, 1 nadir
        let elev = 1.0 - 2.0 * t; // +1 up .. -1 down
        let c = if elev >= 0.0 {
            let k = elev.powf(0.5);
            Vec3::new(0.95, 0.85, 0.75).lerp(Vec3::new(0.25, 0.45, 0.95), k) * 1.4
        } else {
            Vec3::new(0.12, 0.12, 0.13)
        };
        for _ in 0..w {
            pixels.push([c.x, c.y, c.z, 1.0]);
        }
    }
    HdrImageData::new(w, h, pixels)
}

fn build(app: &mut App, post: PostProcess) -> Entity {
    let (cube, sphere, plane) = {
        let a = app.resources.expect_mut::<RenderAssets>();
        (
            a.add_mesh(MeshData::cube(1.0)),
            a.add_mesh(MeshData::sphere(1.0, 48, 24)),
            a.add_mesh(MeshData::plane(120.0)),
        )
    };
    let sky = app
        .resources
        .expect_mut::<RenderAssets>()
        .add_cubemap(CubemapSource::from_equirect_hdr(&sky_panorama()));
    // Wet asphalt: dark, smooth.
    app.world.spawn((
        Transform::from_xyz(0.0, 0.0, 0.0),
        Mesh3d(plane),
        Material::color(Vec4::new(0.05, 0.05, 0.055, 1.0)).with_metallic_roughness(0.0, 0.25),
    ));
    // A window-grid texture tiled at real-world scale: `cuboid_tiled` UVs are in 4 m tiles,
    // `with_tiling(2, 2)` repeats the 256 px texture every 2 m on every facade, whatever its size.
    let grid = app.resources.expect_mut::<RenderAssets>().add_image(ImageData::checker(
        256,
        32,
        [200, 196, 188, 255],
        [70, 80, 96, 255],
    ));
    // Two rows of buildings along a street (x = +-7).
    for i in 0..9 {
        let z = -2.0 - i as f32 * 7.0;
        for (side, shade) in [(-1.0f32, 0.0f32), (1.0, 0.12)] {
            let height = 6.0 + ((i * 7 + side as i32 * 3).rem_euclid(5)) as f32 * 3.0;
            let tone = 0.35 + shade + 0.04 * (i % 3) as f32;
            let size = Vec3::new(6.0, height, 5.5);
            // Baked AO in the vertex colors: darker toward the street level.
            let mut data = MeshData::cuboid_tiled(size, 4.0);
            data.colors = data
                .positions
                .iter()
                .map(|p| {
                    let k = ((p[1] + size.y * 0.5) / 8.0).clamp(0.0, 1.0);
                    let v = 0.45 + 0.55 * k;
                    [v, v, v, 1.0]
                })
                .collect();
            let mesh = app.resources.expect_mut::<RenderAssets>().add_mesh(data);
            app.world.spawn((
                Transform::from_xyz(side * 9.5, height * 0.5, z),
                Mesh3d(mesh),
                Material::color(Vec4::new(tone, tone * 0.95, tone * 0.9, 1.0))
                    .with_texture(grid)
                    .with_tiling(2.0, 2.0)
                    .with_metallic_roughness(0.0, 0.85),
            ));
            // A street lamp: a hot emissive bulb above the sidewalk.
            app.world.spawn((
                Transform::from_xyz(side * 5.2, 4.5, z).with_scale(Vec3::splat(0.25)),
                Mesh3d(sphere),
                Material::unlit(Vec4::new(24.0, 18.0, 10.0, 1.0)),
            ));
        }
    }
    // A glassy slab and a metal sphere near the street, a red box.
    app.world.spawn((
        Transform::from_xyz(-3.2, 2.0, -9.0).with_scale(Vec3::new(0.3, 4.0, 3.5)),
        Mesh3d(cube),
        Material::color(Vec4::new(0.15, 0.25, 0.3, 1.0)).with_metallic_roughness(0.9, 0.04),
    ));
    app.world.spawn((
        Transform::from_xyz(2.2, 1.0, -7.0),
        Mesh3d(sphere),
        Material::color(Vec4::new(0.9, 0.9, 0.95, 1.0)).with_metallic_roughness(1.0, 0.08),
    ));
    app.world.spawn((
        Transform::from_xyz(-1.0, 0.6, -5.0).with_rotation(Quat::from_rotation_y(0.4)),
        Mesh3d(cube),
        Material::color(Vec4::new(0.8, 0.05, 0.04, 1.0)).with_metallic_roughness(0.0, 0.3),
    ));
    app.world.spawn((
        Transform::from_xyz(0.0, 0.0, 0.0).looking_at(Vec3::new(-0.4, -0.5, -0.6), Vec3::Y),
        DirectionalLight {
            color: Vec3::new(1.0, 0.92, 0.8),
            intensity: 3.0,
        },
    ));
    app.world.spawn((
        Transform::from_xyz(0.0, 2.2, 14.0).looking_at(Vec3::new(0.0, 3.0, -20.0), Vec3::Y),
        Camera3d::default().with_clear_color(Vec4::new(0.5, 0.65, 0.95, 1.0)),
        post,
        Skybox::new(sky).with_ground(Vec3::new(0.60, 0.70, 0.82) * 0.9, 0.08),
        EnvironmentLight::new(sky),
    ))
}

fn main() {
    let mut args = std::env::args().skip(1);
    let out = args.next().unwrap_or_else(|| "city_effects_out".into());
    let width: u32 = args.next().and_then(|v| v.parse().ok()).unwrap_or(960);
    let height: u32 = args.next().and_then(|v| v.parse().ok()).unwrap_or(540);
    std::fs::create_dir_all(&out).expect("output directory");

    let mut app = App::new();
    app.add_plugin(RenderPlugin::headless(width, height));
    if !app.resources.contains::<Renderer>() {
        eprintln!("no GPU adapter available");
        return;
    }
    // GPU timings (when the device has timestamp queries): rendered a few frames per variant.
    app.insert_resource(FrameSettings::default().with_gpu_timing(true));
    let camera = build(&mut app, PostProcess::default().with_fxaa(FxaaQuality::Medium));
    let fog = Fog::city_haze().with_density(0.012).with_height(0.0, 0.06);
    let variants = [
        ("base", PostProcess::default()),
        ("fog", PostProcess::default().with_fog(fog)),
        ("ssao", PostProcess::default().with_ssao(Ssao::on().with_radius(1.5))),
        ("ssr", PostProcess::default().with_ssr(Ssr::on())),
        (
            "all",
            PostProcess::default()
                .with_fog(fog)
                .with_ssao(Ssao::on().with_radius(1.5))
                .with_ssr(Ssr::on()),
        ),
    ];
    for (name, post) in variants {
        *app.world.get::<&mut PostProcess>(camera).expect("camera post") = post.with_fxaa(FxaaQuality::Medium);
        for _ in 0..8 {
            render_to_image(&mut app).expect("capture");
        }
        let img = render_to_image(&mut app).expect("capture");
        let timings = app.resources.expect::<GpuTimings>().clone();
        if timings.supported {
            let ms = |l: &str| timings.scope_ms(l);
            println!(
                "{name}: gpu ms  scene {:.3}  gbuffer {:.3}  ssao {:.3}  ssao_blur {:.3}  ssr {:.3}  ssr_blur {:.3}  apply {:.3}  bloom+composite {:.3}",
                ms("scene"),
                ms("gbuffer"),
                ms("post.ssao"),
                ms("post.ssao_blur"),
                ms("post.ssr"),
                ms("post.ssr_blur"),
                ms("post.screenfx"),
                ms("post.bloom") + ms("post.composite"),
            );
        }
        let stats = app.resources.expect::<RenderStats>();
        println!(
            "{name}: passes {} screen_fx_passes {} target_bytes {}",
            stats.render_passes, stats.screen_fx_passes, stats.target_bytes
        );
        let path = format!("{out}/{name}.png");
        image::RgbaImage::from_raw(img.width, img.height, img.pixels)
            .expect("size")
            .save(&path)
            .expect("write png");
        println!("wrote {path}");
    }
}
