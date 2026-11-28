//! The city effects (fog, SSAO, SSR; docs/graphics/post.md, "City effects") on the default
//! backend: they are off unless enabled (pixel-identical to a camera without them), each one
//! does what it promises on a measurable scene, and they are disabled coherently in the Ldr
//! fallback. GPU tests skip without an adapter.

mod common;
#[path = "common/post.rs"]
mod post_common;

use common::headless_app;
use post_common::{frame_settings, lin, max_diff, mean, render, set_post};
use space3d_core::prelude::*;
use space3d_render::*;

const W: u32 = 256;
const H: u32 = 144;

/// Tonemapper `None`, no dither, no bloom: display = radiance clamped to 1.
fn plain() -> PostProcess {
    PostProcess {
        tonemapper: Tonemapper::None,
        dither: false,
        ..PostProcess::default()
    }
    .without_bloom()
}

fn stats(app: &App) -> RenderStats {
    app.resources.expect::<RenderStats>().clone()
}

/// An unlit quad facing +Z at `center`, `size` world units.
fn wall(app: &mut App, color: Vec3, center: Vec3, size: Vec2) -> Entity {
    let mesh = app
        .resources
        .expect_mut::<RenderAssets>()
        .add_mesh(MeshData::plane(1.0));
    app.world.spawn((
        Transform::from_xyz(center.x, center.y, center.z)
            .with_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2))
            .with_scale(Vec3::new(size.x, 1.0, size.y)),
        Mesh3d(mesh),
        Material::unlit(color.extend(1.0)),
    ))
}

/// An upward-facing floor quad (`size` units square) at height `y`.
fn floor(app: &mut App, material: Material, y: f32, size: f32) -> Entity {
    let mesh = app
        .resources
        .expect_mut::<RenderAssets>()
        .add_mesh(MeshData::plane(1.0));
    app.world.spawn((
        Transform::from_xyz(0.0, y, 0.0).with_scale(Vec3::new(size, 1.0, size)),
        Mesh3d(mesh),
        material,
    ))
}

fn camera(app: &mut App, eye: Vec3, target: Vec3, post: PostProcess) -> Entity {
    app.world.spawn((
        Transform::from_xyz(eye.x, eye.y, eye.z).looking_at(target, Vec3::Y),
        Camera3d::default().with_clear_color(Vec4::new(0.0, 0.0, 0.0, 1.0)),
        post,
    ))
}

fn sun(app: &mut App) {
    app.world.spawn((
        Transform::from_xyz(0.0, 0.0, 0.0).looking_at(Vec3::new(-0.3, -1.0, -0.4), Vec3::Y),
        DirectionalLight {
            color: Vec3::ONE,
            intensity: 2.0,
        },
    ));
}

#[test]
fn disabled_effects_are_pixel_identical_and_record_nothing() {
    let Some(mut app) = headless_app(W, H) else { return };
    sun(&mut app);
    floor(&mut app, Material::color(Vec4::new(0.3, 0.3, 0.35, 1.0)), 0.0, 40.0);
    wall(
        &mut app,
        Vec3::new(0.8, 0.4, 0.2),
        Vec3::new(0.0, 2.0, -6.0),
        Vec2::new(8.0, 4.0),
    );
    let cam = camera(&mut app, Vec3::new(0.0, 2.0, 5.0), Vec3::new(0.0, 1.0, -6.0), plain());
    let base = render(&mut app);
    let base_stats = stats(&app);
    assert_eq!(base_stats.screen_fx_passes, 0);
    // Every knob changed, none enabled: the same image, the same passes, no extra memory.
    let tweaked = PostProcess {
        fog: Fog {
            density: 0.5,
            enabled: false,
            ..Fog::city_haze()
        },
        ssao: Ssao {
            radius: 9.0,
            enabled: false,
            ..Ssao::on()
        },
        ssr: Ssr {
            steps: 128,
            enabled: false,
            ..Ssr::on()
        },
        ..plain()
    };
    set_post(&mut app, cam, tweaked);
    let img = render(&mut app);
    assert_eq!(max_diff(&base, &img), 0, "disabled effects change nothing");
    let s = stats(&app);
    assert_eq!(
        (s.render_passes, s.screen_fx_passes, s.target_bytes),
        (base_stats.render_passes, 0, base_stats.target_bytes)
    );
    assert!(!tweaked.screen_effects());
}

#[test]
fn fog_blends_toward_its_color_with_distance() {
    let Some(mut app) = headless_app(W, H) else { return };
    // Red quads: a small near one at the center, a big far one behind it.
    wall(
        &mut app,
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, -5.0),
        Vec2::splat(1.0),
    );
    wall(
        &mut app,
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, -100.0),
        Vec2::splat(400.0),
    );
    let fog = Fog {
        enabled: true,
        color: Vec3::new(0.0, 0.0, 1.0),
        density: 0.02,
        height_falloff: 0.0,
        sun_intensity: 0.0,
        sky_amount: 0.0,
        ..Fog::default()
    };
    let cam = camera(&mut app, Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0), plain());
    let clear = render(&mut app);
    let off_near = lin(&clear, W / 2, H / 2);
    assert!(
        (off_near - Vec3::X).length() < 0.02,
        "unfogged near quad is red: {off_near}"
    );
    set_post(&mut app, cam, plain().with_fog(fog));
    let img = render(&mut app);
    assert_eq!(stats(&app).screen_fx_passes, 1, "one apply pass");
    // Center pixel: the near quad at distance 5 (and its small size: a few pixels around it).
    let near = mean(&img, [W / 2 - 1, H / 2 - 1, 2, 2]);
    let o = fog.opacity(0.0, 0.0, 5.0);
    assert!((near.z - o).abs() < 0.03, "near blue {} want {o}", near.z);
    assert!(
        (near.x - (1.0 - o)).abs() < 0.03,
        "near red {} want {}",
        near.x,
        1.0 - o
    );
    // Off to the side: the far quad at distance ~100.
    let far = mean(&img, [W * 3 / 4, H / 2 - 2, 4, 4]);
    let o = fog.opacity(0.0, 0.0, 101.0);
    assert!(o > 0.8);
    assert!((far.z - o).abs() < 0.04, "far blue {} want {o}", far.z);
    assert!((far.x - (1.0 - o)).abs() < 0.04, "far red {} want {}", far.x, 1.0 - o);
    assert!(far.z > near.z + 0.5, "farther is foggier");
}

#[test]
fn fog_height_falloff_thins_it_above_the_layer() {
    let Some(mut app) = headless_app(W, H) else { return };
    // A far wall seen from low down: its top is above the layer and clearer than its foot.
    wall(&mut app, Vec3::ONE, Vec3::new(0.0, 25.0, -60.0), Vec2::new(200.0, 60.0));
    let fog = Fog {
        enabled: true,
        color: Vec3::ZERO,
        density: 0.03,
        height_falloff: 0.15,
        base_height: 0.0,
        sky_amount: 0.0,
        ..Fog::default()
    };
    camera(
        &mut app,
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 12.0, -60.0),
        plain().with_fog(fog),
    );
    let img = render(&mut app);
    let (low, high) = (mean(&img, [W / 2 - 4, H - 24, 8, 8]), mean(&img, [W / 2 - 4, 8, 8, 8]));
    assert!(
        high.x > low.x + 0.1,
        "higher on the wall is clearer: low {low} high {high}"
    );
}

/// The row profile of the mean linear luminance of a column band.
fn profile(img: &ImageData) -> Vec<f32> {
    (0..img.height)
        .map(|y| {
            let m = mean(img, [W / 2 - 24, y, 48, 1]);
            (m.x + m.y + m.z) / 3.0
        })
        .collect()
}

#[test]
fn ssao_darkens_the_crease_and_leaves_open_floor_alone() {
    let Some(mut app) = headless_app(W, H) else { return };
    // An unlit floor meeting an unlit wall at z = -4.
    let up = Quat::from_rotation_x(0.0);
    let mesh = app
        .resources
        .expect_mut::<RenderAssets>()
        .add_mesh(MeshData::plane(1.0));
    app.world.spawn((
        Transform::from_xyz(0.0, 0.0, 0.0)
            .with_rotation(up)
            .with_scale(Vec3::new(30.0, 1.0, 30.0)),
        Mesh3d(mesh),
        Material::unlit(Vec4::new(0.7, 0.7, 0.7, 1.0)),
    ));
    wall(
        &mut app,
        Vec3::splat(0.7),
        Vec3::new(0.0, 4.0, -4.0),
        Vec2::new(30.0, 8.0),
    );
    let cam = camera(&mut app, Vec3::new(0.0, 3.0, 4.0), Vec3::new(0.0, 0.5, -4.0), plain());
    let off = render(&mut app);
    set_post(&mut app, cam, plain().with_ssao(Ssao::on().with_radius(2.0)));
    let on = render(&mut app);
    let s = stats(&app);
    assert!(s.screen_fx_passes >= 3, "ao, two blurs, apply: {}", s.screen_fx_passes);
    let (a, b) = (profile(&off), profile(&on));
    let ratio: Vec<f32> = a
        .iter()
        .zip(&b)
        .map(|(a, b)| if *a > 0.05 { b / a } else { 1.0 })
        .collect();
    let min = ratio.iter().cloned().fold(f32::MAX, f32::min);
    assert!(min < 0.85, "the crease is darkened: darkest row keeps {min}");
    // The nearest floor (the bottom rows, far from the wall) is untouched.
    let bottom = ratio[ratio.len() - 8..].iter().cloned().fold(f32::MAX, f32::min);
    assert!(bottom > 0.97, "open floor keeps {bottom}");
    // The effect is local: most rows stay within a few percent.
    let calm = ratio.iter().filter(|r| **r > 0.95).count();
    assert!(calm * 2 > ratio.len(), "{calm} of {} rows untouched", ratio.len());
}

#[test]
fn ssr_mirrors_a_bright_object_in_a_smooth_floor() {
    let Some(mut app) = headless_app(W, H) else { return };
    let caps_ok = app.resources.expect::<Renderer>().graphics_caps().read_only_depth;
    if !caps_ok {
        eprintln!("skipping: no read-only depth on this device");
        return;
    }
    sun(&mut app);
    // A glossy dark floor and a bright red unlit box standing on it.
    floor(
        &mut app,
        Material::color(Vec4::new(0.03, 0.03, 0.03, 1.0)).with_metallic_roughness(0.0, 0.05),
        0.0,
        60.0,
    );
    let cube = app.resources.expect_mut::<RenderAssets>().add_mesh(MeshData::cube(1.0));
    app.world.spawn((
        Transform::from_xyz(0.0, 1.0, -8.0).with_scale(Vec3::new(2.0, 2.0, 2.0)),
        Mesh3d(cube),
        Material::unlit(Vec4::new(6.0, 0.1, 0.1, 1.0)),
    ));
    let cam = camera(&mut app, Vec3::new(0.0, 1.5, 4.0), Vec3::new(0.0, 0.8, -8.0), plain());
    let off = render(&mut app);
    set_post(&mut app, cam, plain().with_ssr(Ssr::on().with_max_roughness(0.4)));
    let on = render(&mut app);
    assert!(stats(&app).screen_fx_passes >= 1);
    // Count the pixels whose red grew and by how much more than green/blue.
    let (mut changed, mut red_gain, mut other_gain) = (0u32, 0.0f32, 0.0f32);
    for y in 0..H {
        for x in 0..W {
            let (a, b) = (lin(&off, x, y), lin(&on, x, y));
            if b.x - a.x > 0.01 {
                changed += 1;
                red_gain += b.x - a.x;
                other_gain += (b.y - a.y) + (b.z - a.z);
            }
        }
    }
    assert!(changed > 100, "the box shows in the floor: {changed} pixels changed");
    assert!(
        red_gain > 3.0 * other_gain.max(0.0),
        "the reflection is red: {red_gain} vs {other_gain}"
    );
    // Only below the box (the floor): the top of the frame is unchanged.
    let top = max_diff_rows(&off, &on, 0..H / 3);
    assert_eq!(top, 0, "nothing changes above the horizon");
}

fn max_diff_rows(a: &ImageData, b: &ImageData, rows: std::ops::Range<u32>) -> u8 {
    let mut m = 0;
    for y in rows {
        for x in 0..a.width {
            let (p, q) = (a.get(x, y), b.get(x, y));
            for c in 0..4 {
                m = m.max(p[c].abs_diff(q[c]));
            }
        }
    }
    m
}

#[test]
fn effects_are_off_in_the_ldr_fallback() {
    let Some(mut app) = headless_app(W, H) else { return };
    wall(
        &mut app,
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, -50.0),
        Vec2::splat(200.0),
    );
    let all = plain()
        .with_fog(Fog::city_haze().with_density(0.05))
        .with_ssao(Ssao::on())
        .with_ssr(Ssr::on());
    let cam = camera(&mut app, Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0), all);
    frame_settings(&mut app).downlevel.no_float_targets = true;
    let ldr = render(&mut app);
    assert_eq!(stats(&app).screen_fx_passes, 0, "no effects without float targets");
    set_post(&mut app, cam, plain());
    assert_eq!(render(&mut app), ldr, "exactly the image without the effects");
    // HdrMode::Ldr on a device with float targets: the same.
    frame_settings(&mut app).downlevel.no_float_targets = false;
    set_post(&mut app, cam, all);
    app.world
        .insert_one(cam, CameraFrame::default().with_hdr(HdrMode::Ldr))
        .unwrap();
    assert_eq!(render(&mut app), ldr);
    assert_eq!(stats(&app).screen_fx_passes, 0);
}

#[test]
fn effects_work_with_msaa_depth_resolve() {
    let renderer = pollster::block_on(Renderer::new_headless_with(RendererConfig {
        width: W,
        height: H,
        msaa_samples: 4,
        ..Default::default()
    }));
    let Ok(renderer) = renderer else { return };
    let caps = renderer.graphics_caps();
    if renderer.capabilities().max_msaa < 4 || !(caps.msaa_depth_resolve && caps.hdr_msaa) {
        eprintln!("skipping: no resolvable multisampled depth");
        return;
    }
    let mut app = App::new();
    app.add_plugin(RenderPlugin::default());
    app.insert_resource(renderer);
    wall(
        &mut app,
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, -50.0),
        Vec2::splat(200.0),
    );
    let fog = Fog {
        enabled: true,
        density: 0.03,
        height_falloff: 0.0,
        ..Fog::default()
    };
    camera(&mut app, Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0), plain().with_fog(fog));
    let img = render(&mut app);
    assert_eq!(stats(&app).screen_fx_passes, 1);
    let m = mean(&img, [W / 2 - 4, H / 2 - 4, 8, 8]);
    let o = fog.opacity(0.0, 0.0, 50.0);
    assert!(
        (m.x - (1.0 - o) - o * fog.color.x).abs() < 0.04,
        "fogged through the resolved depth: {m}"
    );
}

fn solid_faces(rgba: [u8; 4]) -> [ImageData; 6] {
    std::array::from_fn(|_| ImageData::filled(8, 8, rgba))
}

#[test]
fn skybox_ground_replaces_the_lower_hemisphere() {
    let Some(mut app) = headless_app(W, H) else { return };
    let sky = app
        .resources
        .expect_mut::<RenderAssets>()
        .add_cubemap(CubemapSource::from_faces(solid_faces([255, 0, 0, 255])).unwrap());
    // Looking level at the horizon: sky above the middle row, ground below.
    let cam = camera(&mut app, Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0), plain());
    app.world.insert_one(cam, Skybox::new(sky)).unwrap();
    let plain_sky = render(&mut app);
    let (top, bottom) = (lin(&plain_sky, W / 2, 8), lin(&plain_sky, W / 2, H - 8));
    assert!(top.x > 0.9 && bottom.x > 0.9, "the cubemap all around: {top} {bottom}");
    app.world
        .insert_one(cam, Skybox::new(sky).with_ground(Vec3::new(0.0, 1.0, 0.0), 0.05))
        .unwrap();
    let grounded = render(&mut app);
    let (top, bottom) = (lin(&grounded, W / 2, 8), lin(&grounded, W / 2, H - 8));
    assert!(top.x > 0.9 && top.y < 0.05, "sky above the horizon is untouched: {top}");
    assert!(bottom.y > 0.9 && bottom.x < 0.05, "ground color below it: {bottom}");
    // The blend is soft: the rows around the horizon are in between.
    let mid = lin(&grounded, W / 2, H / 2);
    assert!(mid.x > 0.05 && mid.y > 0.05, "blended at the horizon: {mid}");
}

#[test]
fn tiled_meshes_keep_their_texture_scale() {
    // A 6 x 3 x 2 box with 2 m tiles: the +Z face (6 x 3 units) spans 3 x 1.5 tiles.
    let m = MeshData::cuboid_tiled(Vec3::new(6.0, 3.0, 2.0), 2.0);
    let front: Vec<[f32; 2]> = m
        .normals
        .iter()
        .zip(&m.uvs)
        .filter(|(n, _)| n[2] > 0.5)
        .map(|(_, uv)| *uv)
        .collect();
    let max_u = front.iter().map(|uv| uv[0]).fold(0.0, f32::max);
    let max_v = front.iter().map(|uv| uv[1]).fold(0.0, f32::max);
    assert!(
        (max_u - 3.0).abs() < 1e-5 && (max_v - 1.5).abs() < 1e-5,
        "{max_u} {max_v}"
    );
    assert!((m.positions.iter().map(|p| p[0]).fold(0.0, f32::max) - 3.0).abs() < 1e-6);
    assert_eq!(
        Material::default().with_tiling(4.0, 2.0).uv_transform.scale,
        Vec2::new(4.0, 2.0)
    );
    let p = MeshData::plane_tiled(10.0, 2.0);
    assert!(p.uvs.iter().all(|uv| uv[0] <= 5.0 && uv[1] <= 5.0) && p.uvs.iter().any(|uv| uv[0] == 5.0));
}
