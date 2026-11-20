#![warn(missing_docs)]
//! Space3d UI: [egui](https://github.com/emilk/egui) integration, debug overlay, profiler and
//! in-game console.
//!
//! * [`UiPlugin`] inserts the [`EguiContext`] resource and runs one egui pass per frame:
//!   input is translated from `space3d_platform::Input` in `Stage::PreUpdate` (mouse, keys,
//!   text, wheel, touch, scale factor — everything injectable in tests), UI systems run in
//!   `Stage::Update`, and the pass is ended + tessellated in `Stage::PostUpdate`. The result is
//!   drawn by `egui_wgpu::Renderer` through the renderer's overlay hook
//!   ([`space3d_render::RenderOverlay`]) into the final target: the window surface, or the
//!   headless target — so UI shows up in `render_to_image` screenshots.
//! * [`UiAppExt::add_ui_system`] registers `|ctx: &egui::Context, sys: &mut SystemCtx|` UI code;
//!   [`UiAppExt::add_root_ui_system`] gets the frame's root [`egui::Ui`] for panels.
//! * Pointer: mouse buttons reach egui in the order they happened
//!   (`space3d_platform::Input::mouse_button_events`), each where the cursor was; a release
//!   outside the window goes where egui last saw the pointer (no hover at the window's corner);
//!   `egui::Event::WindowFocused` follows the window's focus. A release that no device event
//!   reported — the window lost the focus, or the OS shows the button up with no release event —
//!   is a cancel: egui registers neither a click nor a drop (see `docs/platforms.md`, "Mouse
//!   buttons, focus and the UI"); a system-cancelled touch is no tap either.
//!   [`EguiContext::pointer_cancelled_this_frame`] / [`pointer_cancelled_this_frame`] tell game
//!   code that decides drops from `Response::drag_stopped` / `any_released` to skip such a frame.
//!   Buttons before any pointer position is known are dropped (not sent to (0, 0)).
//! * egui plugins get their pass hooks (`on_begin_pass` before the UI systems, `on_end_pass`
//!   after them), which egui itself only runs from `Context::run_ui`: the built-in ones (label
//!   text selection, drag-and-drop, `egui::debug_text`) and those added with
//!   [`EguiContext::add_plugin`].
//! * Debug overlay (**F3**, [`DebugOverlay`]): FPS, frame-time sparkline, tick / tick rate /
//!   dropped ticks, [`space3d_render::RenderStats`] and entity count in a small monospace grid.
//! * Profiler (**F4**, [`Profiler`]): per-system timings aggregated from the core's `system`
//!   trace spans by a `tracing` layer ([`ProfilerLayer`]), shown as a sorted table and a
//!   flame-style frame timeline.
//! * Console (**`**, [`Console`]): log lines captured from `tracing` ([`ConsoleLayer`]) and a
//!   command registry.
//! * The three tools are compact under any egui style: they size themselves (their own `Ui`
//!   only) and never change the global style (see `docs/ui.md`). They dock to the corners of the
//!   screen, or of the [`ToolDockArea`] a host with its own panels sets (the editor's viewport,
//!   a band a game's HUD leaves free: side by side when it is too short to stack them), follow
//!   it when it is resized, and stay above the game's windows.
//! * [`UiTextures`] shows renderer render targets (e.g. a `Camera3d` rendering to a texture) as
//!   egui images.
//! * Opt-in density ([`theme`]): `app.insert_resource(UiTheme::compact())` applies the compact
//!   preset's [`UiDensity`] tokens (spacing, control and text sizes, corner radii) to egui's
//!   style once, before the next pass; without a [`UiTheme`] resource the engine never touches
//!   egui's `Style`.
//! * Chinese and Japanese text wraps by the kinsoku rules (no line starts with 、。，」）？ or a
//!   small kana, none ends with 「（): Space3d builds egui with a patched `epaint`
//!   (`third_party/epaint`); a game gets it with the same `[patch.crates-io]` line — see
//!   `docs/ui.md`, "Chinese and Japanese line breaking".
//!
//! Headless determinism: with [`UiPlugin::deterministic`] (fixed `pixels_per_point`, no
//! animations, no dithering) and egui time taken from [`space3d_core::Time`], the same inputs
//! produce the same pixels, so UI can be screenshot-tested.
//!
//! ```no_run
//! use space3d_core::prelude::*;
//! use space3d_render::RenderPlugin;
//! use space3d_ui::{UiAppExt, UiPlugin, egui};
//!
//! let mut app = App::new();
//! app.add_plugin(RenderPlugin::headless(640, 360));
//! app.add_plugin(UiPlugin::deterministic());
//! app.add_ui_system(|ctx: &egui::Context, sys: &mut SystemCtx| {
//!     egui::Window::new("Stats").show(ctx, |ui| {
//!         ui.label(format!("entities: {}", sys.world.len()));
//!     });
//! });
//! app.step_ticks(3);
//! let screenshot = space3d_render::render_to_image(&mut app);
//! # let _ = screenshot;
//! ```

pub mod console;
mod context;
pub mod debug;
mod dock;
mod input;
mod overlay;
pub mod profiler;
mod textures;
pub mod theme;
mod tool;
mod tracing_setup;

pub use console::{Console, ConsoleCommand, ConsoleLayer, LogLevel, LogLine};
pub use context::{EguiContext, UiFrameStats, winit_cursor};
pub use debug::DebugOverlay;
pub use egui;
pub use egui_wgpu;
pub use input::{
    REPEAT_DELAY, REPEAT_INTERVAL, REPEAT_KEYS, egui_key, egui_logical_key, physical_key_for_egui_key,
    pointer_cancel_cause, pointer_cancelled_this_frame,
};
pub use profiler::{FrameTimeline, Profiler, ProfilerLayer, SystemStats, TimelineSpan};
pub use textures::UiTextures;
pub use theme::{
    ThemePreview, UiDensity, UiTheme, UiThemeSync, apply_density, apply_theme, compact_style, visible_scroll_bars,
};
pub use tool::{ToolDockArea, set_tool_dock_area};
pub use tracing_setup::{TracingInstall, install_tracing, tracing_install_state};

use overlay::EguiOverlay;
use space3d_core::{App, Plugin, Stage, SystemCtx};
use space3d_platform::{InputPlugin, KeyCode};
use space3d_render::RenderOverlays;
use std::sync::{Arc, Mutex};

/// Name of the egui overlay in [`RenderOverlays`].
pub const UI_OVERLAY_NAME: &str = "space3d_ui::egui";

/// egui integration + debug tools. See the [crate docs](crate).
#[derive(Clone, Debug)]
pub struct UiPlugin {
    /// Fixed physical pixels per egui point. `None` follows
    /// `space3d_platform::WindowState::scale_factor` (1.0 without a window).
    pub pixels_per_point: Option<f32>,
    /// A phone's rule for the scale, used while `pixels_per_point` is `None`: the scale at which
    /// the screen's short side is this many points (never below 1.0). A UI made for a laptop's
    /// window then fits a phone, with small text. `None`: the window's own scale.
    pub short_side_points: Option<f32>,
    /// Lay the UI out inside `space3d_platform::SafeArea` (a phone's notch and home bar) and
    /// not over the whole screen. Off by default. The insets are filled on Android and iOS (in
    /// landscape UIKit's values were the portrait ones in the Simulator, and the runner turns
    /// them: `space3d_platform::SafeArea::turned_for_landscape`).
    pub inside_safe_area: bool,
    /// UI size in physical pixels when there is neither a renderer nor a window.
    pub fallback_size: (u32, u32),
    /// Dither egui's output to reduce banding (position-based, so still deterministic).
    pub dithering: bool,
    /// egui animations (window fade-in, collapsing headers...). Off for screenshot tests.
    pub animations: bool,
    /// Draw order of the egui overlay among [`RenderOverlays`] (higher = on top).
    pub overlay_priority: i32,
    /// Show the debug overlay from the first frame.
    pub show_debug_overlay: bool,
    /// Key toggling the [`DebugOverlay`] (default F3).
    pub debug_overlay_key: Option<KeyCode>,
    /// Key toggling the [`Profiler`] window (default F4).
    pub profiler_key: Option<KeyCode>,
    /// Key toggling the [`Console`] (default backquote).
    pub console_key: Option<KeyCode>,
    /// Install the profiler/console `tracing` layers (see [`install_tracing`]).
    pub install_tracing: bool,
    /// Directives for the stderr log output when this plugin installs the global subscriber
    /// (`RUST_LOG` wins when set).
    pub log_filter: String,
}

impl Default for UiPlugin {
    fn default() -> Self {
        Self {
            pixels_per_point: None,
            short_side_points: None,
            inside_safe_area: false,
            fallback_size: (1280, 720),
            dithering: true,
            animations: true,
            overlay_priority: 1000,
            show_debug_overlay: false,
            debug_overlay_key: Some(KeyCode::F3),
            profiler_key: Some(KeyCode::F4),
            console_key: Some(KeyCode::Backquote),
            install_tracing: true,
            log_filter: tracing_setup::DEFAULT_FILTER.to_owned(),
        }
    }
}

impl UiPlugin {
    /// Settings for reproducible screenshots: 1 pixel per point, no animations, no dithering.
    pub fn deterministic() -> Self {
        Self {
            pixels_per_point: Some(1.0),
            dithering: false,
            animations: false,
            ..Default::default()
        }
    }

    /// Same settings with a fixed scale factor.
    pub fn with_pixels_per_point(mut self, pixels_per_point: f32) -> Self {
        self.pixels_per_point = Some(pixels_per_point);
        self
    }

    /// Show the debug overlay from the first frame.
    pub fn with_debug_overlay(mut self, visible: bool) -> Self {
        self.show_debug_overlay = visible;
        self
    }
}

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(InputPlugin);
        app.init_resource::<RenderOverlays>();
        if self.install_tracing {
            install_tracing(&self.log_filter);
        }

        let egui_ctx = egui::Context::default();
        egui_ctx.set_theme(egui::Theme::Dark);
        if !self.animations {
            egui_ctx.global_style_mut(|s| s.animation_time = 0.0);
        }
        app.insert_resource(EguiContext::new(egui_ctx, self));
        app.init_resource::<UiTextures>();
        app.insert_resource(DebugOverlay::new(self.show_debug_overlay));
        app.init_resource::<Profiler>();
        app.insert_resource(Console::with_builtin_commands());

        let overlay: Arc<Mutex<EguiOverlay>> = app.resources.expect_mut::<RenderOverlays>().insert(
            UI_OVERLAY_NAME,
            self.overlay_priority,
            EguiOverlay::new(self.dithering),
        );

        let mut translator = input::InputTranslator::default();
        let mut theme = UiThemeSync::default();
        let settings = self.clone();
        app.add_system_named(Stage::PreUpdate, "ui::begin_frame", move |sys| {
            context::begin_frame(sys, &mut translator, &mut theme, &settings);
        });
        let keys = (self.debug_overlay_key, self.profiler_key, self.console_key);
        app.add_system_named(Stage::PreUpdate, "ui::hotkeys", move |sys| debug::hotkeys(sys, keys));
        let hint = debug::tool_keys_hint(self.profiler_key, self.console_key);
        app.add_system_named(Stage::Update, "ui::debug_overlay", move |sys| {
            debug::debug_overlay_system(sys, &hint);
        });
        app.add_system_named(Stage::Update, "ui::profiler", profiler::profiler_system);
        app.add_system_named(Stage::Update, "ui::console", console::console_system);
        app.add_system_named(Stage::PostUpdate, "ui::end_frame", move |sys| {
            context::end_frame(sys, &overlay);
        });
    }
}

/// UI system registration on [`App`].
pub trait UiAppExt {
    /// Run `system` every frame in `Stage::Update` with the frame's egui context (skipped when
    /// no egui pass is active, e.g. without [`UiPlugin`] or while the UI is disabled).
    fn add_ui_system(&mut self, system: impl FnMut(&egui::Context, &mut SystemCtx) + Send + 'static) -> &mut Self;

    /// Like [`Self::add_ui_system`], but hands out the frame's root [`egui::Ui`] (the space
    /// not yet taken by panels of earlier root-ui systems), for `egui::Panel` /
    /// `egui::CentralPanel` layouts.
    fn add_root_ui_system(&mut self, system: impl FnMut(&mut egui::Ui, &mut SystemCtx) + Send + 'static) -> &mut Self;
}

impl UiAppExt for App {
    fn add_ui_system(&mut self, mut system: impl FnMut(&egui::Context, &mut SystemCtx) + Send + 'static) -> &mut Self {
        let name = std::any::type_name_of_val(&system);
        self.add_system_named(Stage::Update, name, move |sys| {
            let Some(ctx) = EguiContext::active_context(sys) else {
                return;
            };
            system(&ctx, sys);
        })
    }

    fn add_root_ui_system(
        &mut self,
        mut system: impl FnMut(&mut egui::Ui, &mut SystemCtx) + Send + 'static,
    ) -> &mut Self {
        let name = std::any::type_name_of_val(&system);
        self.add_system_named(Stage::Update, name, move |sys| {
            EguiContext::with_root_ui(sys, |ui, sys| system(ui, sys));
        })
    }
}

/// Common imports.
pub mod prelude {
    pub use crate::{
        Console, DebugOverlay, EguiContext, Profiler, ToolDockArea, UiAppExt, UiDensity, UiPlugin, UiTextures, UiTheme,
        egui,
    };
}
