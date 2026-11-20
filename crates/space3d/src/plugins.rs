//! Plugin groups ([`DefaultPlugins`], [`MinimalPlugins`]) and small utility plugins
//! ([`LogPlugin`], [`AutoExitPlugin`]).

use space3d_core::{App, AppExit, Plugin, Stage, TickRate};
use space3d_tasks::TasksPlugin;

/// Environment variable: exit after this many frames (e.g. `SPACE3D_EXIT_AFTER_FRAMES=120`).
pub const ENV_EXIT_AFTER_FRAMES: &str = "SPACE3D_EXIT_AFTER_FRAMES";
/// Environment variable: exit after this many seconds of app time (e.g. `SPACE3D_EXIT_AFTER_SECS=2.5`).
pub const ENV_EXIT_AFTER_SECS: &str = "SPACE3D_EXIT_AFTER_SECS";
/// Environment variable: run [`DefaultPlugins`] without a window. `1`/`true` renders
/// offscreen at the configured window size, `WIDTHxHEIGHT` (e.g. `640x360`) at that size.
pub const ENV_HEADLESS: &str = "SPACE3D_HEADLESS";

fn env_var(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|v| v.trim().to_owned())
        .filter(|v| !v.is_empty())
}

// ------------------------------------------------------------------------------------------
// LogPlugin

/// Installs a global `tracing` subscriber: formatted stderr output filtered by `RUST_LOG`
/// (falling back to [`LogPlugin::filter`]) on native, the browser console (plus a panic hook
/// that prints Rust panics) on the web. Installing twice is harmless.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogPlugin {
    /// `tracing_subscriber::EnvFilter` directives used when `RUST_LOG` is unset (native only).
    pub filter: String,
}

impl Default for LogPlugin {
    fn default() -> Self {
        Self {
            // winit's iOS loop logs two lines of its own in every frame (a warning for its own
            // `AboutToWait`, an info for every redraw): never useful, and megabytes a minute.
            filter: if cfg!(target_os = "ios") {
                "info,wgpu_core=warn,wgpu_hal=warn,naga=warn,winit::platform_impl::ios::app_state=error".into()
            } else {
                "info,wgpu_core=warn,wgpu_hal=warn,naga=warn".into()
            },
        }
    }
}

impl Plugin for LogPlugin {
    fn build(&self, _app: &mut App) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use tracing_subscriber::EnvFilter;
            let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&self.filter));
            let _ = tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_writer(std::io::stderr)
                .try_init();
        }
        #[cfg(target_arch = "wasm32")]
        {
            console_error_panic_hook::set_once();
            let _ = tracing_wasm::try_set_as_global_default();
        }
    }
}

// ------------------------------------------------------------------------------------------
// AutoExitPlugin

/// Sends [`AppExit`] after a number of frames and/or seconds of app time.
///
/// [`AutoExitPlugin::from_env`] reads [`ENV_EXIT_AFTER_FRAMES`] / [`ENV_EXIT_AFTER_SECS`];
/// both plugin groups install it that way, so any Space3d app can be smoke-tested with
/// `SPACE3D_EXIT_AFTER_FRAMES=60 cargo run`. When it fires it prints one summary line
/// (`space3d: auto-exit after N frames ...`) to stderr.
///
/// An app runs at most one `AutoExitPlugin` (plugins are de-duplicated by type): with
/// [`DefaultPlugins`] / [`MinimalPlugins`] prefer their `.exit_after_frames(n)`, which also
/// works when the environment variables are set. Adding one yourself works too, unless the
/// group already installed one from the environment (then the environment wins).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AutoExitPlugin {
    /// Exit once `Time::frame` reaches this.
    pub after_frames: Option<u64>,
    /// Exit once `Time::elapsed` reaches this many seconds.
    pub after_secs: Option<f64>,
}

impl AutoExitPlugin {
    /// Exit after `frames` frames.
    pub fn frames(frames: u64) -> Self {
        Self {
            after_frames: Some(frames),
            after_secs: None,
        }
    }

    /// Exit after `secs` seconds of app time.
    pub fn secs(secs: f64) -> Self {
        Self {
            after_frames: None,
            after_secs: Some(secs),
        }
    }

    /// Limits from [`ENV_EXIT_AFTER_FRAMES`] / [`ENV_EXIT_AFTER_SECS`] (unset or unparsable
    /// values mean "no limit"; on the web there is no environment, so never).
    pub fn from_env() -> Self {
        Self {
            after_frames: env_var(ENV_EXIT_AFTER_FRAMES).and_then(|v| v.parse().ok()),
            after_secs: env_var(ENV_EXIT_AFTER_SECS).and_then(|v| v.parse().ok()),
        }
    }

    /// Whether any limit is configured.
    pub fn is_active(&self) -> bool {
        self.after_frames.is_some() || self.after_secs.is_some()
    }
}

impl Plugin for AutoExitPlugin {
    fn build(&self, app: &mut App) {
        if !self.is_active() {
            return;
        }
        let limits = *self;
        let mut fired = false;
        app.add_system_named(Stage::Last, "space3d::auto_exit", move |ctx| {
            let time = ctx.time();
            let (frame, elapsed) = (time.frame, time.elapsed_secs());
            let hit =
                limits.after_frames.is_some_and(|n| frame >= n) || limits.after_secs.is_some_and(|s| elapsed >= s);
            if hit && !fired {
                fired = true;
                let fps = if elapsed > 0.0 { frame as f64 / elapsed } else { 0.0 };
                eprintln!(
                    "space3d: auto-exit after {frame} frames / {elapsed:.2}s ({fps:.1} fps avg, {} fixed ticks)",
                    time.tick
                );
                ctx.send(AppExit);
            }
        });
    }
}

// ------------------------------------------------------------------------------------------
// MinimalPlugins

/// The smallest useful set: [`TasksPlugin`] (async completions), [`AutoExitPlugin::from_env`]
/// and an optional [`LogPlugin`]. No window, no renderer: `App::run()` uses the core headless
/// runner (real-time loop at the tick rate until [`AppExit`]). Use it for servers, tools and
/// simulation tests.
///
/// ```
/// use space3d::prelude::*;
/// let mut app = App::new();
/// app.add_plugin(MinimalPlugins::default().tick_rate(30.0));
/// app.step_ticks(3);
/// assert_eq!(app.resources.expect::<Time>().tick, 3);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct MinimalPlugins {
    /// Logging (`None` = leave the global subscriber alone). Default `None`.
    pub log: Option<LogPlugin>,
    /// Frame/time limit. Default: from the environment.
    pub auto_exit: AutoExitPlugin,
    /// Fixed simulation rate in Hz (`None` keeps the app's current [`TickRate`]).
    pub tick_rate: Option<f64>,
}

impl Default for MinimalPlugins {
    fn default() -> Self {
        Self {
            log: None,
            auto_exit: AutoExitPlugin::from_env(),
            tick_rate: None,
        }
    }
}

impl MinimalPlugins {
    /// Same as `default()`.
    pub fn new() -> Self {
        Self::default()
    }
    /// Also install a [`LogPlugin`].
    pub fn with_log(mut self) -> Self {
        self.log = Some(LogPlugin::default());
        self
    }
    /// Set the fixed tick rate (Hz).
    pub fn tick_rate(mut self, hz: f64) -> Self {
        self.tick_rate = Some(hz);
        self
    }
    /// Exit after `frames` frames (overrides the environment).
    pub fn exit_after_frames(mut self, frames: u64) -> Self {
        self.auto_exit.after_frames = Some(frames);
        self
    }
}

impl Plugin for MinimalPlugins {
    fn build(&self, app: &mut App) {
        if let Some(log) = &self.log {
            app.add_plugin(log.clone());
        }
        if let Some(hz) = self.tick_rate {
            app.set_tickrate(TickRate::new(hz));
        }
        app.add_plugin(TasksPlugin);
        // Only an active limit claims the plugin slot (`App::add_plugin` de-duplicates by type),
        // so an `AutoExitPlugin` the game adds itself still works.
        if self.auto_exit.is_active() {
            app.add_plugin(self.auto_exit);
        }
    }
}

// ------------------------------------------------------------------------------------------
// DefaultPlugins

/// Everything a typical game needs, depending on the enabled cargo features:
///
/// | piece | feature | default |
/// |---|---|---|
/// | [`LogPlugin`] | — | on |
/// | [`TasksPlugin`] + [`AutoExitPlugin::from_env`] | — | on |
/// | `PlatformPlugin` (window, input, renderer) | `platform` | 1280×720 window titled "Space3d" |
/// | `AssetsPlugin` (root `assets/`, hot reload on native) | `assets` | on |
/// | `AudioPlugin` (falls back to silent without a device) | `audio` | on |
/// | `PhysicsPlugin` (gravity −9.81) | `physics` | on |
/// | `ScenePlugin` | `scene` | on |
/// | `NetPlugin` (HTTP + WebSocket clients) | `net` | on |
/// | `DbPlugin` | `db` | **off** — opt in with [`DefaultPlugins::with_db`] (it creates a file) |
/// | `UiPlugin` (egui, F3 debug overlay, F4 profiler, console) | `ui` | on |
///
/// With the `ui` feature the [`LogPlugin`] settings are handed to `UiPlugin`, which installs
/// the global `tracing` subscriber itself (log output + profiler + in-game console layers).
///
/// Each piece can be replaced or switched off with the builder methods (or by setting the
/// public fields). Setting [`ENV_HEADLESS`] (`SPACE3D_HEADLESS=1` or `=640x360`) replaces the
/// window with an offscreen renderer, which together with `SPACE3D_EXIT_AFTER_FRAMES` makes
/// any game runnable in CI.
///
/// ```no_run
/// use space3d::prelude::*;
/// let mut app = App::new();
/// app.add_plugin(DefaultPlugins::new().title("My game").tick_rate(120.0));
/// app.run();
/// ```
#[derive(Clone, Debug)]
pub struct DefaultPlugins {
    /// Logging (`None` = leave the global subscriber alone).
    pub log: Option<LogPlugin>,
    /// Frame/time limit. Default: from the environment.
    pub auto_exit: AutoExitPlugin,
    /// Fixed simulation rate in Hz (`None` keeps the app's current [`TickRate`]).
    pub tick_rate: Option<f64>,
    /// Window settings; `None` = no window (input + renderer resources only, see `headless`).
    #[cfg(feature = "platform")]
    pub window: Option<space3d_platform::WindowPlugin>,
    /// Without a window: create an offscreen renderer of this size (native only).
    #[cfg(feature = "render")]
    pub headless: Option<(u32, u32)>,
    /// Asset server settings.
    #[cfg(feature = "assets")]
    pub assets: Option<space3d_assets::AssetsPlugin>,
    /// Audio settings.
    #[cfg(feature = "audio")]
    pub audio: Option<space3d_audio::AudioPlugin>,
    /// Physics settings.
    #[cfg(feature = "physics")]
    pub physics: Option<space3d_physics::PhysicsPlugin>,
    /// Scene loading.
    #[cfg(feature = "scene")]
    pub scene: bool,
    /// HTTP / WebSocket client settings.
    #[cfg(feature = "net")]
    pub net: Option<space3d_net::NetPlugin>,
    /// Database (off by default).
    #[cfg(feature = "db")]
    pub db: Option<space3d_db::DbPlugin>,
    /// egui integration (debug overlay, profiler, console).
    #[cfg(feature = "ui")]
    pub ui: Option<space3d_ui::UiPlugin>,
}

impl Default for DefaultPlugins {
    fn default() -> Self {
        Self {
            log: Some(LogPlugin::default()),
            auto_exit: AutoExitPlugin::from_env(),
            tick_rate: None,
            #[cfg(feature = "platform")]
            window: Some(space3d_platform::WindowPlugin::default()),
            #[cfg(feature = "render")]
            headless: None,
            #[cfg(feature = "assets")]
            assets: Some(space3d_assets::AssetsPlugin::default()),
            #[cfg(feature = "audio")]
            audio: Some(space3d_audio::AudioPlugin::default()),
            #[cfg(feature = "physics")]
            physics: Some(space3d_physics::PhysicsPlugin::default()),
            #[cfg(feature = "scene")]
            scene: true,
            #[cfg(feature = "net")]
            net: Some(space3d_net::NetPlugin::default()),
            #[cfg(feature = "db")]
            db: None,
            #[cfg(feature = "ui")]
            ui: Some(space3d_ui::UiPlugin::default()),
        }
    }
}

impl DefaultPlugins {
    /// Same as `default()`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Headless preset for tests, servers and CI: no window, an offscreen `width`×`height`
    /// renderer (skipped with a warning when there is no GPU), silent audio, no logging.
    pub fn headless(width: u32, height: u32) -> Self {
        #[allow(unused_mut)]
        let mut p = Self {
            log: None,
            ..Self::default()
        };
        #[cfg(feature = "platform")]
        {
            p.window = None;
        }
        #[cfg(feature = "render")]
        {
            p.headless = Some((width, height));
        }
        #[cfg(not(feature = "render"))]
        let _ = (width, height);
        #[cfg(feature = "audio")]
        {
            p.audio = Some(space3d_audio::AudioPlugin::disabled());
        }
        #[cfg(feature = "assets")]
        {
            p.assets = Some(space3d_assets::AssetsPlugin {
                hot_reload: false,
                ..Default::default()
            });
        }
        #[cfg(feature = "ui")]
        {
            // Reproducible pixels for screenshot tests.
            p.ui = Some(space3d_ui::UiPlugin::deterministic());
        }
        p
    }

    /// Set the window title (creates default window settings if there were none).
    #[cfg(feature = "platform")]
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.window.get_or_insert_with(Default::default).title = title.into();
        self
    }

    /// Replace the window settings.
    #[cfg(feature = "platform")]
    pub fn window(mut self, window: space3d_platform::WindowPlugin) -> Self {
        self.window = Some(window);
        self
    }

    /// Asset root directory (native) or base URL (web).
    #[cfg(feature = "assets")]
    pub fn assets_root(mut self, root: impl Into<String>) -> Self {
        self.assets = Some(space3d_assets::AssetsPlugin::with_root(root));
        self
    }

    /// Open a database (e.g. `DbPlugin::new("save.db")`).
    #[cfg(feature = "db")]
    pub fn with_db(mut self, db: space3d_db::DbPlugin) -> Self {
        self.db = Some(db);
        self
    }

    /// No audio plugin at all.
    #[cfg(feature = "audio")]
    pub fn without_audio(mut self) -> Self {
        self.audio = None;
        self
    }

    /// No physics plugin.
    #[cfg(feature = "physics")]
    pub fn without_physics(mut self) -> Self {
        self.physics = None;
        self
    }

    /// No network plugin.
    #[cfg(feature = "net")]
    pub fn without_net(mut self) -> Self {
        self.net = None;
        self
    }

    /// No egui integration.
    #[cfg(feature = "ui")]
    pub fn without_ui(mut self) -> Self {
        self.ui = None;
        self
    }

    /// Replace the egui settings.
    #[cfg(feature = "ui")]
    pub fn ui(mut self, ui: space3d_ui::UiPlugin) -> Self {
        self.ui = Some(ui);
        self
    }

    /// Show the debug overlay (FPS, frame graph, ticks, render stats; toggle with F3) from the
    /// first frame.
    #[cfg(feature = "ui")]
    pub fn with_debug_overlay(mut self) -> Self {
        self.ui = Some(self.ui.take().unwrap_or_default().with_debug_overlay(true));
        self
    }

    /// No global logger.
    pub fn without_log(mut self) -> Self {
        self.log = None;
        self
    }

    /// Set the fixed tick rate (Hz).
    pub fn tick_rate(mut self, hz: f64) -> Self {
        self.tick_rate = Some(hz);
        self
    }

    /// Exit after `frames` frames (overrides the environment).
    pub fn exit_after_frames(mut self, frames: u64) -> Self {
        self.auto_exit.after_frames = Some(frames);
        self
    }

    /// Offscreen size requested through [`ENV_HEADLESS`], if set. `fallback` is used for `1`/`true`.
    pub fn headless_from_env(fallback: (u32, u32)) -> Option<(u32, u32)> {
        parse_headless(&env_var(ENV_HEADLESS)?, fallback)
    }
}

/// Largest offscreen size accepted from [`ENV_HEADLESS`] per axis: wgpu's default
/// `max_texture_dimension_2d`, which every desktop adapter supports.
pub const MAX_HEADLESS_SIZE: u32 = 8192;

/// Parse a [`ENV_HEADLESS`] value: `WxH`, or a truthy flag (→ `fallback`). Falsy → `None`.
/// Sizes are clamped to `1..=`[`MAX_HEADLESS_SIZE`] (with a warning), so a typo cannot make
/// the renderer panic on an oversized texture.
pub(crate) fn parse_headless(v: &str, fallback: (u32, u32)) -> Option<(u32, u32)> {
    let v = v.trim().to_ascii_lowercase();
    match v.as_str() {
        "" | "0" | "false" | "no" | "off" => None,
        "1" | "true" | "yes" | "on" => Some(fallback),
        _ => {
            let (w, h) = v.split_once('x')?;
            let (w, h): (u32, u32) = (w.trim().parse().ok()?, h.trim().parse().ok()?);
            let clamped = (w.clamp(1, MAX_HEADLESS_SIZE), h.clamp(1, MAX_HEADLESS_SIZE));
            if clamped != (w, h) {
                eprintln!(
                    "space3d: {ENV_HEADLESS}={w}x{h} is outside 1..={MAX_HEADLESS_SIZE}; using {}x{}",
                    clamped.0, clamped.1
                );
            }
            Some(clamped)
        }
    }
}

impl Plugin for DefaultPlugins {
    fn build(&self, app: &mut App) {
        // With egui enabled, `UiPlugin` installs the subscriber (so its profiler and console
        // layers see every thread); otherwise `LogPlugin` does.
        #[cfg(feature = "ui")]
        let ui_owns_logging = self.ui.is_some();
        #[cfg(not(feature = "ui"))]
        let ui_owns_logging = false;
        if let Some(log) = &self.log
            && !ui_owns_logging
        {
            app.add_plugin(log.clone());
        }
        if let Some(hz) = self.tick_rate {
            app.set_tickrate(TickRate::new(hz));
        }
        app.add_plugin(TasksPlugin);
        // See `MinimalPlugins::build`: an inactive limit must not block a user-added one.
        if self.auto_exit.is_active() {
            app.add_plugin(self.auto_exit);
        }

        #[cfg(feature = "render")]
        {
            #[cfg(feature = "platform")]
            let window_size = self.window.as_ref().map_or((1280, 720), |w| (w.width, w.height));
            #[cfg(not(feature = "platform"))]
            let window_size = (1280, 720);
            let env_headless = Self::headless_from_env(window_size);
            #[cfg(feature = "platform")]
            let windowed = match (&self.window, env_headless) {
                (Some(window), None) => {
                    app.add_plugin(space3d_platform::PlatformPlugin { window: window.clone() });
                    true
                }
                _ => {
                    app.add_plugin(space3d_platform::InputPlugin);
                    false
                }
            };
            #[cfg(not(feature = "platform"))]
            let windowed = false;
            if !windowed {
                let headless = env_headless.or(self.headless);
                app.add_plugin(space3d_render::RenderPlugin { headless });
            }
        }
        #[cfg(feature = "assets")]
        if let Some(assets) = &self.assets {
            #[allow(unused_mut)]
            let mut assets = assets.clone();
            // Don't warn about watching a directory the game doesn't have.
            #[cfg(not(target_arch = "wasm32"))]
            if assets.hot_reload && !std::path::Path::new(&assets.root).is_dir() {
                assets.hot_reload = false;
            }
            app.add_plugin(assets);
        }
        #[cfg(feature = "audio")]
        if let Some(audio) = &self.audio {
            app.add_plugin(*audio);
        }
        #[cfg(feature = "physics")]
        if let Some(physics) = &self.physics {
            app.add_plugin(*physics);
        }
        #[cfg(feature = "scene")]
        if self.scene {
            app.add_plugin(space3d_scene::ScenePlugin);
        }
        #[cfg(feature = "net")]
        if let Some(net) = &self.net {
            app.add_plugin(net.clone());
        }
        #[cfg(feature = "db")]
        if let Some(db) = &self.db {
            app.add_plugin(db.clone());
        }
        #[cfg(feature = "ui")]
        if let Some(ui) = &self.ui {
            let mut ui = ui.clone();
            match &self.log {
                Some(log) => ui.log_filter = log.filter.clone(),
                // No logging requested: leave the global subscriber alone.
                None => ui.install_tracing = false,
            }
            app.add_plugin(ui);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use space3d_core::Time;

    #[test]
    fn headless_env_values() {
        assert_eq!(parse_headless("1", (8, 9)), Some((8, 9)));
        assert_eq!(parse_headless("true", (8, 9)), Some((8, 9)));
        assert_eq!(parse_headless("640x360", (8, 9)), Some((640, 360)));
        assert_eq!(parse_headless(" 32 x 16 ", (8, 9)), Some((32, 16)));
        assert_eq!(parse_headless("0", (8, 9)), None);
        assert_eq!(parse_headless("garbage", (8, 9)), None);
        assert_eq!(parse_headless("99999x99999", (8, 9)), Some((8192, 8192)));
        assert_eq!(parse_headless("640x0", (8, 9)), Some((640, 1)));
        assert_eq!(parse_headless("-5x10", (8, 9)), None);
    }

    #[test]
    fn user_auto_exit_works_next_to_the_groups() {
        // No limit in the environment: the group leaves the slot free for the game's own plugin.
        if AutoExitPlugin::from_env().is_active() {
            return;
        }
        let mut app = App::new();
        app.add_plugin(MinimalPlugins::new());
        app.add_plugin(AutoExitPlugin::frames(3));
        app.step_ticks(2);
        assert!(!app.should_exit());
        app.step_ticks(1);
        assert!(app.should_exit());
    }

    #[test]
    fn auto_exit_after_frames() {
        let mut app = App::new();
        app.add_plugin(AutoExitPlugin::frames(5));
        app.step_ticks(4);
        assert!(!app.should_exit());
        app.step_ticks(1);
        assert!(app.should_exit());
    }

    #[test]
    fn auto_exit_inactive_adds_nothing() {
        let mut app = App::new();
        let before = app.system_names(Stage::Last).len();
        app.add_plugin(AutoExitPlugin::default());
        assert_eq!(app.system_names(Stage::Last).len(), before);
    }

    #[test]
    fn minimal_plugins_run_headless_until_exit() {
        let mut app = App::new();
        app.add_plugin(MinimalPlugins::new().tick_rate(1000.0).exit_after_frames(3));
        app.run();
    }

    #[test]
    fn minimal_plugins_tick_rate() {
        let mut app = App::new();
        app.add_plugin(MinimalPlugins::new().tick_rate(20.0));
        assert_eq!(app.resources.expect::<TickRate>().hz, 20.0);
        app.step_ticks(2);
        assert_eq!(app.resources.expect::<Time>().tick, 2);
        assert!(app.resources.contains::<space3d_tasks::AsyncQueue>());
    }

    #[test]
    fn default_plugins_headless_registers_everything() {
        let mut app = App::new();
        app.add_plugin(DefaultPlugins::headless(64, 64).exit_after_frames(2));
        #[cfg(feature = "render")]
        assert!(app.resources.contains::<space3d_render::RenderAssets>());
        #[cfg(feature = "platform")]
        assert!(app.resources.contains::<space3d_platform::Input>());
        #[cfg(feature = "assets")]
        assert!(app.resources.contains::<space3d_assets::AssetServer>());
        #[cfg(feature = "audio")]
        assert!(app.resources.contains::<space3d_audio::Audio>());
        #[cfg(feature = "physics")]
        assert!(app.resources.contains::<space3d_physics::Physics>());
        #[cfg(feature = "scene")]
        assert!(app.resources.contains::<space3d_scene::SceneCommands>());
        #[cfg(feature = "net")]
        assert!(app.resources.contains::<space3d_net::Http>());
        #[cfg(feature = "db")]
        assert!(!app.resources.contains::<space3d_db::Db>());
        #[cfg(feature = "ui")]
        assert!(app.resources.contains::<space3d_ui::EguiContext>());
        app.step_ticks(2);
        assert!(app.should_exit());
    }
}
