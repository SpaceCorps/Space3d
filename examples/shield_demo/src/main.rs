//! A 3D shield for SpaceCorps 2027 on Space3d: every ship has a shield hugging its hull, drawn by
//! a `RenderNode` (WGSL), with the game's weapons, gun positions and sounds, the engine's quality
//! tiers, and the Aegis: the carrier's shield stretched into a bubble over its group.
//!
//! The SpaceCorps ship models are committed in `assets/`; the game's sounds are not, so the demo
//! is silent unless they are added in `assets/sfx/`. `SPACE3D_SHIELD_DEMO_ASSETS`, or an `assets/`
//! folder next to the executable, takes precedence. A model that can't be read falls back to the
//! engine's test ship. See `README.md`.
//!
//! ```text
//! cargo run -p shield_demo --release
//! SPACE3D_HEADLESS=1 SHIELD_SHOT=out.png SHIELD_SHOT_FRAME=150 cargo run -p shield_demo   # capture
//! ```
//! Options for captures and scripted runs (environment variables; URL parameters on the web, for
//! example `?SHIELD_WEAPON=x4&SHIELD_LOG_FRAME=300`). Frame numbers are fixed ticks counted from
//! the moment the ships are on screen. `SHIELD_LOG_FRAME` reports the state (console,
//! `window.__shield`). `SHIELD_QUALITY=low|medium|high|ultra`, `SHIELD_MAIN`, `SHIELD_ALLY`,
//! `SHIELD_FOE` (hull names), `SHIELD_SOLO=1`, `SHIELD_LANG=en`,
//! `SHIELD_WEAPON=x1|x3|x4|am|gem|best|lancet|ember|siphon|custom`,
//! `SHIELD_TARGET=you|player|mob`, `SHIELD_WEAR=1`, `SHIELD_RATE=<volleys/s>`, and at a given
//! tick `SHIELD_FIRE_AT`, `SHIELD_BREAK_AT`, `SHIELD_AEGIS_AT`, `SHIELD_SWITCH_AT` (with
//! `SHIELD_SWITCH_TO=<hull>`).
//!
//! Controls: click in the view or `Space` fires a volley, `B` breaks your shield, `R` restores
//! it, `F` deploys the Aegis, `P` pauses, `1`–`4` pick the quality tier, `Esc` quits.

mod catalog;
mod envelope;
mod game;
mod node;
mod sbe;
mod ui;

use game::Game;
use node::{ShieldFrame, ShieldNode};
use space3d::particles::prelude::*;
use space3d::prelude::*;
use space3d::render::{RenderNodes, Renderer};

fn build(app: &mut App) {
    app.add_plugin(DefaultPlugins::new().window(WindowPlugin {
        title: "Space3d — bouclier 3D / 3D shield".into(),
        // Web: render into the page's <canvas id="space3d"> (ignored on desktop).
        canvas_id: Some("space3d".into()),
        ..WindowPlugin::default()
    }));
    add_game(app);
    if let Ok(path) = std::env::var("SHIELD_SHOT") {
        let at = std::env::var("SHIELD_SHOT_FRAME")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(150);
        screenshot_system(app, path, at);
    }
}

/// The demo without the engine plugins (the tests add them to a headless app).
fn add_game(app: &mut App) {
    app.add_plugin(ParticlesPlugin::default());
    app.init_resource::<Game>();
    app.init_resource::<ShieldFrame>();
    app.resources
        .init::<RenderNodes>()
        .insert("spacecorps/shields", ShieldNode::default());
    app.add_system(Stage::Startup, game::setup);
    app.add_system(Stage::Update, game::update);
    app.add_system(Stage::FixedUpdate, game::tick);
    app.add_ui_system(ui::panel);
}

/// Capture one frame to a PNG, then quit (tests and documentation).
fn screenshot_system(app: &mut App, path: String, at: u64) {
    let mut task: Option<Task<Result<ImageData, space3d::render::RenderError>>> = None;
    let mut requested = false;
    let mut capture_started = 0;
    let mut writing: Option<Task<Result<String, String>>> = None;
    app.add_system(Stage::Last, move |sys| {
        let frame = sys.time().frame;
        // counted from the moment the ships are on screen
        let scene = sys.resources.get::<Game>().map_or(0, |g| g.scene_frames);
        if !requested && scene >= at {
            requested = true;
            capture_started = frame;
            if let Some(g) = sys.resources.get::<Game>() {
                println!("{}", g.debug_line());
            }
            task = sys.resources.get_mut::<Renderer>().map(|r| r.capture());
            if task.is_none() {
                eprintln!("capture failed: no renderer");
                sys.send(AppExit);
                return;
            }
        }
        if let Some(w) = &mut writing
            && let Some(result) = w.poll()
        {
            writing = None;
            match result {
                Ok(line) => println!("{line}"),
                Err(line) => eprintln!("{line}"),
            }
            sys.send(AppExit);
        }
        if let Some(t) = &mut task {
            if let Some(r) = sys.resources.get::<Renderer>() {
                r.poll();
            }
            match t.poll() {
                Some(Ok(img)) => {
                    task = None;
                    // encoding and writing off the frame (rule 2)
                    let path = path.clone();
                    writing = Some(spawn_compute(move || {
                        let size = (img.width, img.height);
                        space3d::assets::edit::texture::encode_png(&img)
                            .map_err(|e| e.to_string())
                            .and_then(|png| std::fs::write(&path, png).map_err(|e| e.to_string()))
                            .map(|()| format!("capture saved to {path} ({}x{})", size.0, size.1))
                            .map_err(|e| format!("capture failed: {path}: {e}"))
                    }));
                }
                Some(Err(e)) => {
                    task = None;
                    eprintln!("capture failed: {e}");
                    sys.send(AppExit);
                }
                None if frame >= capture_started + 600 => {
                    eprintln!("capture failed: timeout");
                    sys.send(AppExit);
                }
                None => {}
            }
        }
    });
}

space3d::main!(build);

#[cfg(test)]
mod tests {
    use super::*;
    use game::{AegisState, Role, State, Target};
    use space3d::render::GraphicsQuality;

    /// A headless app with the demo, every hull already loaded (the engine's test ship), so the
    /// tests drive the simulation with `step_ticks` only. They need no GPU (with one, the frames
    /// are also rendered offscreen). Options from the environment are ignored in tests.
    fn demo() -> App {
        let mut app = App::new();
        app.add_plugin(DefaultPlugins::headless(64, 64));
        add_game(&mut app);
        game::preload_fallback(&mut app.resources);
        {
            let g = app.resources.expect_mut::<Game>();
            g.s.auto = false;
            g.s.spin = false;
            g.s.miss = false;
            g.s.target = Target::You;
            // the same tier on every machine (the rules don't depend on it, the particles do)
            g.set_quality(GraphicsQuality::High);
        }
        app.step_ticks(2);
        assert_eq!(
            app.resources.expect::<Game>().units.len(),
            3,
            "your ship, the other player, the mob"
        );
        app
    }

    fn game(app: &mut App) -> &mut Game {
        app.resources.expect_mut::<Game>()
    }

    /// A volley on your shield: every bolt makes a wave, and without wear the shield stays full.
    #[test]
    fn a_volley_makes_waves() {
        let mut app = demo();
        game(&mut app).fire(Some(0.0), Some(0));
        app.step_ticks(120);
        let g = game(&mut app);
        assert!(g.hits >= 1, "the volley hits ({} hits)", g.hits);
        assert!(!g.units[0].impacts.is_empty(), "a wave on your shield");
        assert_eq!(g.state, State::Up);
        assert_eq!(g.s.pct, 100.0);
    }

    /// With wear, a volley takes 7 % × the shield's share; at 0 the shield breaks from the last
    /// hit, goes down, and `restore_shield` brings it back.
    #[test]
    fn wear_breaks_and_restore_brings_it_back() {
        let mut app = demo();
        {
            let g = game(&mut app);
            g.s.wear = true;
            // no regeneration during the test (it starts after 15 s without damage)
            g.s.regen = 1000.0;
            g.s.pct = 3.0;
            g.fire(Some(0.0), Some(0));
        }
        app.step_ticks(240);
        let g = game(&mut app);
        assert_eq!(g.s.pct, 0.0);
        assert!(matches!(g.state, State::Breaking | State::Down), "{:?}", g.state);
        g.restore_shield(true);
        app.step_ticks(120);
        let g = game(&mut app);
        assert_eq!(g.state, State::Up);
        assert_eq!(g.s.pct, 100.0);
    }

    /// The Aegis: your shield becomes the bubble, hits on the ally wear its reserve, and after
    /// 8 s it ends and your shield is back.
    #[test]
    fn the_aegis_takes_hits_then_ends() {
        let mut app = demo();
        {
            let g = game(&mut app);
            g.s.target = Target::Player;
            g.aegis_at = Some(g.scene_frames + 1);
        }
        app.step_ticks(30);
        assert_eq!(game(&mut app).aegis.state, AegisState::On);
        let ally = game(&mut app).ally_index().expect("an ally");
        game(&mut app).fire(Some(std::f32::consts::PI), Some(ally));
        app.step_ticks(120);
        let g = game(&mut app);
        assert!(g.aegis.reserve < g.aegis.full, "the bubble took the volley");
        app.step_ticks(60 * 9);
        let g = game(&mut app);
        assert_eq!(g.aegis.state, AegisState::Off);
        assert_eq!(g.state, State::Up);
    }

    /// Fires your Siphon at the ship with `role`, with wear and no regeneration; `prepare` runs
    /// before the shot. Returns the target's index.
    fn your_siphon_at(app: &mut App, role: Role, prepare: impl FnOnce(&mut Game)) -> usize {
        let g = game(app);
        g.s.wear = true;
        g.s.regen = 1000.0;
        g.s.weapon = catalog::WEAPONS
            .iter()
            .position(|w| w.key == "siphon")
            .expect("the Siphon");
        prepare(g);
        let target = g.units.iter().position(|u| u.role == role).expect("the target");
        g.fire(None, Some(target));
        target
    }

    fn gain_waves(list: &[game::Impact]) -> Vec<Vec3> {
        list.iter().filter(|m| m.kind == 3.0).map(|m| m.p).collect()
    }

    /// Your Siphon leaves from your ship: the mob's shield is drained, and the packets come back
    /// into your shield on the side facing the mob, which takes in 7 %.
    #[test]
    fn your_siphon_refills_your_shield() {
        let mut app = demo();
        let mob = your_siphon_at(&mut app, Role::Foe, |g| g.s.pct = 50.0);
        app.step_ticks(120);
        let g = game(&mut app);
        assert!(
            g.units[mob].impacts.iter().any(|m| m.kind == 1.0),
            "a drain on the mob's shield"
        );
        assert_eq!(g.s.pct, 57.0);
        let waves = gain_waves(&g.units[0].impacts);
        assert_eq!(waves.len(), 1, "one gain wave on your shield");
        // the mob is on +x and the ships don't turn in the tests
        assert!(waves[0].x > 0.0, "the packets enter on the mob's side: {:?}", waves[0]);
        assert_eq!(g.state, State::Up);
    }

    /// Your shield at 0: the Siphon's gain brings it back with what it took in.
    #[test]
    fn your_siphon_brings_your_shield_back() {
        let mut app = demo();
        your_siphon_at(&mut app, Role::Foe, |g| {
            g.s.pct = 0.0;
            g.break_shield(None);
        });
        app.step_ticks(150);
        let g = game(&mut app);
        assert_eq!(g.s.pct, 7.0);
        assert_eq!(g.state, State::Up);
    }

    /// Under the Aegis your shield is the bubble: the gain wave is on the bubble, not on the
    /// hull, and the gain goes to its reserve.
    #[test]
    fn your_siphon_refills_the_aegis() {
        let mut app = demo();
        {
            let g = game(&mut app);
            g.s.pct = 50.0;
            g.aegis_at = Some(g.scene_frames + 1);
        }
        app.step_ticks(30);
        assert_eq!(game(&mut app).aegis.state, AegisState::On);
        your_siphon_at(&mut app, Role::Foe, |_| {});
        app.step_ticks(120);
        let g = game(&mut app);
        assert_eq!(gain_waves(&g.aegis.impacts).len(), 1, "the wave is on the bubble");
        assert!(gain_waves(&g.units[0].impacts).is_empty(), "and not on the hidden hull");
        assert_eq!(g.aegis.reserve, 57.0);
        assert_eq!(g.s.pct, 57.0);
    }

    /// The Aegis goes up while your Siphon is under way to the other player: right after the
    /// shot (the bubble lets your probe out and doesn't drain your own reserve), or after the
    /// drain, while the packets fly (they go to the bubble, the shield you have when they land).
    #[test]
    fn the_aegis_deployed_under_your_siphon() {
        for after in [1, 30] {
            let mut app = demo();
            let ally = your_siphon_at(&mut app, Role::Ally, |g| {
                g.s.pct = 50.0;
                g.aegis_at = Some(g.scene_frames + after);
            });
            app.step_ticks(120);
            let g = game(&mut app);
            assert_eq!(g.aegis.state, AegisState::On);
            assert!(
                g.units[ally].impacts.iter().any(|m| m.kind == 1.0),
                "the probe drained the other player (Aegis after {after} ticks)"
            );
            assert_eq!(
                gain_waves(&g.aegis.impacts).len(),
                1,
                "the wave is on the bubble (Aegis after {after} ticks)"
            );
            assert!(gain_waves(&g.units[0].impacts).is_empty());
            assert_eq!(g.aegis.reserve, 57.0, "nothing taken from the bubble, 7 % in");
        }
    }

    /// Same inputs, same result: the simulation only depends on the fixed ticks, not on the
    /// quality tier (which changes the number of particles drawn).
    #[test]
    fn the_simulation_is_deterministic() {
        let run = |q: GraphicsQuality| {
            let mut app = demo();
            {
                let g = game(&mut app);
                g.s.auto = true;
                g.set_quality(q);
            }
            app.step_ticks(300);
            let g = game(&mut app);
            (
                g.hits,
                g.s.pct,
                g.units.iter().map(|u| u.impacts.len()).collect::<Vec<_>>(),
            )
        };
        let high = run(GraphicsQuality::High);
        assert_eq!(high, run(GraphicsQuality::High));
        assert_eq!(high, run(GraphicsQuality::Low));
    }
}
