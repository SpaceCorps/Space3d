//! The smallest complete Space3d game: a window with a lit, rotating cube.
//!
//! ```text
//! cargo run -p hello_cube
//! SPACE3D_EXIT_AFTER_FRAMES=120 cargo run -p hello_cube   # smoke test: quit after 120 frames
//! SPACE3D_HEADLESS=1 SPACE3D_EXIT_AFTER_FRAMES=60 cargo run -p hello_cube   # no window (CI)
//! ```
//! Controls: `Esc` quits, `Space` pauses the rotation.

use space3d::prelude::*;

/// Rotation speed in radians per second around a fixed axis.
struct Spin {
    axis: Vec3,
    speed: f32,
}

/// Whether the cube is currently rotating.
#[derive(Default)]
struct Paused(bool);

/// The whole game: the engine's plugins and the scene. `src/main.rs` and the phone build
/// (`examples/hello_cube_mobile`) both hand it to `space3d::main!`.
pub fn build(app: &mut App) {
    app.add_plugin(DefaultPlugins::new().title("Space3d — hello cube"));
    add_game(app);
}

/// The scene and its systems, without the engine plugins (the tests add them to a headless app).
fn add_game(app: &mut App) {
    app.init_resource::<Paused>();
    app.add_system(Stage::Startup, setup);
    app.add_system(Stage::Update, controls);
    app.add_system(Stage::Update, spin);
}

fn setup(ctx: &mut SystemCtx) {
    let (cube, ground) = {
        let assets = ctx.resources.expect_mut::<RenderAssets>();
        (
            assets.add_mesh(MeshData::cube(1.0)),
            assets.add_mesh(MeshData::plane(8.0)),
        )
    };
    ctx.world.spawn((
        Name("cube".into()),
        Transform::from_xyz(0.0, 0.75, 0.0),
        Mesh3d(cube),
        Material::srgb(0.95, 0.45, 0.2).with_metallic_roughness(0.1, 0.35),
        Spin {
            axis: Vec3::new(0.3, 1.0, 0.2).normalize(),
            speed: 1.0,
        },
    ));
    ctx.world.spawn((
        Name("ground".into()),
        Transform::default(),
        Mesh3d(ground),
        Material::srgb(0.35, 0.38, 0.42).with_metallic_roughness(0.0, 0.9),
    ));
    ctx.world.spawn((
        Transform::from_xyz(3.0, 5.0, 2.0).looking_at(Vec3::ZERO, Vec3::Y),
        DirectionalLight {
            color: Vec3::new(1.0, 0.96, 0.9),
            intensity: 2.5,
        },
    ));
    ctx.world.spawn((
        Transform::from_xyz(-2.0, 2.0, 2.0),
        PointLight {
            color: Vec3::new(0.3, 0.5, 1.0),
            intensity: 6.0,
            range: 8.0,
        },
    ));
    ctx.resources.insert(AmbientLight {
        color: Vec3::new(0.7, 0.8, 1.0),
        intensity: 0.15,
    });
    ctx.world.spawn((
        Transform::from_xyz(0.0, 2.2, 4.5).looking_at(Vec3::new(0.0, 0.6, 0.0), Vec3::Y),
        Camera3d::default().with_clear_color(Vec4::new(0.04, 0.05, 0.08, 1.0)),
    ));
}

fn controls(ctx: &mut SystemCtx) {
    let input = ctx.resources.expect::<Input>();
    let (quit, toggle) = (input.just_pressed(KeyCode::Escape), input.just_pressed(KeyCode::Space));
    if toggle {
        let paused = ctx.resources.expect_mut::<Paused>();
        paused.0 = !paused.0;
    }
    if quit {
        ctx.send(AppExit);
    }
}

fn spin(ctx: &mut SystemCtx) {
    if ctx.resources.expect::<Paused>().0 {
        return;
    }
    let dt = ctx.time().delta_secs();
    for (transform, spin) in ctx.world.query_mut::<(&mut Transform, &Spin)>() {
        transform.rotation = (Quat::from_axis_angle(spin.axis, spin.speed * dt) * transform.rotation).normalize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headless() -> App {
        let mut app = App::new();
        app.add_plugin(DefaultPlugins::headless(64, 64));
        add_game(&mut app);
        app
    }

    fn cube_rotation(app: &App) -> Quat {
        let mut query = app.world.query::<(&Transform, &Spin)>();
        query.iter().next().expect("the cube exists").0.rotation
    }

    #[test]
    fn cube_spins_space_pauses_escape_quits() {
        let mut app = headless();
        app.step_ticks(30);
        let spinning = cube_rotation(&app);
        assert!(spinning.angle_between(Quat::IDENTITY) > 0.3, "{spinning:?}");

        app.resources.expect_mut::<Input>().inject_key_tap(KeyCode::Space);
        app.step_ticks(1);
        let paused = cube_rotation(&app);
        app.step_ticks(30);
        assert_eq!(cube_rotation(&app), paused);

        assert!(!app.should_exit());
        app.resources.expect_mut::<Input>().inject_key_tap(KeyCode::Escape);
        app.step_ticks(1);
        assert!(app.should_exit());
    }
}
