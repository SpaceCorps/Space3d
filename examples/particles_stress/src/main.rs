//! Particle stress test: `PARTICLES` (default 200 000, up to about 1 000 000) additive glows
//! streaming from four emitters sweeping round into outward spirals (a vortex turns them as they
//! fly), 4 s lives — simulated, compacted and drawn on the GPU backend (one `simulate` and one
//! `emit` dispatch per tick, `mark` + `scatter` per frame, an indirect draw). The particle stats
//! are printed every two seconds.
//!
//! ```text
//! cargo run --release -p particles_stress
//! PARTICLES=1000000 cargo run --release -p particles_stress
//! SPACE3D_HEADLESS=1 SPACE3D_EXIT_AFTER_FRAMES=120 cargo run -p particles_stress   # CI smoke test
//! ```
//! Without compute shaders (WebGL2) the CPU backend's budget (16k, 8k on the web) refuses most of
//! it. `Esc` quits.

use space3d::particles::prelude::*;
use space3d::particles::{Curve, EmitShape, FadeCurve, FloatRange, Force, Gradient, SpriteRender, SpriteShape};
use space3d::prelude::*;

/// Particle count wanted (`PARTICLES`), and the emitters sweeping round.
struct Stress {
    count: u32,
    arms: Vec<Entity>,
    report_at: f64,
}

/// `PARTICLES` from the environment, clamped to 1 000 .. 2 000 000.
fn wanted() -> u32 {
    std::env::var("PARTICLES")
        .ok()
        .and_then(|v| v.replace('_', "").parse().ok())
        .unwrap_or(200_000u32)
        .clamp(1_000, 2_000_000)
}

fn build(app: &mut App) {
    app.add_plugin(DefaultPlugins::new().title("Space3d — particle stress"));
    add_game(app, wanted());
}

fn add_game(app: &mut App, count: u32) {
    app.add_plugin(ParticlesPlugin::with_settings(ParticleSettings {
        max_particles: (count + count / 4).next_power_of_two(),
        ..ParticleSettings::default()
    }));
    app.insert_resource(Stress {
        count,
        arms: Vec::new(),
        report_at: 0.0,
    });
    app.add_system(Stage::Startup, setup);
    app.add_system(Stage::FixedUpdate, sweep);
    app.add_system(Stage::Update, controls);
}

fn setup(ctx: &mut SystemCtx) {
    let count = ctx.resources.expect::<Stress>().count;
    ctx.world.spawn((
        Transform::from_xyz(0.0, 150.0, 30.0).looking_at(Vec3::ZERO, Vec3::Y),
        Camera3d::default().with_clear_color(Vec4::new(0.004, 0.005, 0.012, 1.0)),
        PostProcess::default(),
    ));
    // Four arms, each `count / 4` alive: a rate of that over the 4 s life.
    let arm = ParticleLayer {
        lod: false,
        shape: EmitShape::Ring {
            radius: 6.0,
            width: 4.0,
            arc: 0.9,
        },
        flatten: 1.0,
        spread: 0.15,
        speed: FloatRange::new(9.0, 16.0),
        forces: vec![Force::Vortex {
            axis: Vec3::Y,
            strength: 1.6,
            pull: 0.0,
        }],
        drag: 0.1,
        size: FloatRange::new(0.6, 1.2),
        color_over_life: Gradient::linear(Vec4::new(0.4, 0.2, 0.08, 1.0), Vec4::new(0.03, 0.06, 0.3, 1.0)),
        alpha_over_life: Curve::Preset(FadeCurve::InOut),
        render: SpriteRender::additive(SpriteShape::Glow {
            core: 0.2,
            hardness: 0.0,
        }),
        ..ParticleLayer::continuous(count as f32 / 4.0 / 4.0, 4.0)
    };
    let effects = ctx.resources.expect_mut::<ParticleEffects>();
    let Ok(fx) = effects.add(ParticleEffect {
        looping: true,
        ..ParticleEffect::new("stress").layer(arm)
    }) else {
        return;
    };
    let arms: Vec<Entity> = (0..4)
        .map(|i| ctx.spawn_effect(fx, EffectSpawn::at(Vec3::ZERO).seed(i).keep()))
        .collect();
    ctx.resources.expect_mut::<Stress>().arms = arms;
}

fn sweep(ctx: &mut SystemCtx) {
    let a = ctx.time().tick as f32 * 0.03;
    let arms = ctx.resources.expect::<Stress>().arms.clone();
    for (i, e) in arms.into_iter().enumerate() {
        if let Ok(mut t) = ctx.world.get::<&mut Transform>(e) {
            t.rotation = Quat::from_rotation_y(a + i as f32 * std::f32::consts::FRAC_PI_2);
        }
    }
}

fn controls(ctx: &mut SystemCtx) {
    if ctx.resources.expect::<Input>().just_pressed(KeyCode::Escape) {
        ctx.send(AppExit);
    }
    let now = ctx.time().elapsed_secs();
    let stress = ctx.resources.expect_mut::<Stress>();
    if now < stress.report_at {
        return;
    }
    stress.report_at = now + 2.0;
    let want = stress.count;
    if let Some(s) = ctx.resources.get::<ParticleStats>() {
        println!(
            "stress: {} alive of {want} ({:?} backend, pool {} of {}), {} dispatches, {} draws, {} B uploaded, {} refused",
            s.alive,
            s.backend,
            s.pool_reserved,
            s.pool_capacity,
            s.dispatches,
            s.draw_calls,
            s.bytes_uploaded,
            s.refused
        );
    }
}

space3d::main!(build);

#[cfg(test)]
mod tests {
    use super::*;

    /// 20 000 particles reserved and streaming after a second.
    #[test]
    fn stress_reserves_and_emits() {
        let mut app = App::new();
        app.add_plugin(DefaultPlugins::headless(64, 64));
        add_game(&mut app, 20_000);
        app.step_ticks(60);
        let s = app.resources.expect::<ParticleStats>().clone();
        if s.backend.is_none() {
            eprintln!("skipping: no GPU adapter (the particle backend needs a renderer)");
            return;
        }
        assert_eq!(s.effects, 4, "{s:?}");
        assert!(s.pool_reserved >= 20_000, "{s:?}");
        assert!(s.spawned >= 4_000, "{s:?}");
        assert_eq!(s.refused, 0, "{s:?}");
    }
}
