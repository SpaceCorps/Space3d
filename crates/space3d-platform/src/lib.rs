#![warn(missing_docs)]
//! Space3d platform layer: the winit runner (desktop, web, Android, iOS) and input.
//!
//! * [`PlatformPlugin`] / [`WindowPlugin`] — open a window, create the renderer surface and
//!   run the frame loop (`App::run()`), handling resize, DPI, focus and mobile
//!   suspend/resume (surface recreation).
//! * [`Input`] — keyboard, mouse, multi-touch and gamepads ([`gamepad`]), fully injectable for tests
//!   (`Input::inject_*`); [`ActionMap`] + [`ActionState`] map named actions to bindings.
//!   Mouse buttons keep their order within a frame ([`Input::mouse_button_events`]); releases
//!   that no release event reported — a focus loss, or (macOS, Windows) the OS showing a held
//!   button up once per frame — are cancels ([`MouseButtonCause`]): the UI registers no click
//!   and no drop for them. A touch the system cancelled ([`TouchPhase::Cancelled`]) is no tap
//!   either ([`Input::just_cancelled_touches`]).
//! * [Keyboard layouts](#keyboard-layouts) — logical keys ([`KeyEvent`]) and key labels
//!   ([`Input::key_label`]) that follow the player's layout; see [`layout`].
//! * Web: `init_web_logging()` (wasm32 only), canvas attachment via [`WindowPlugin::canvas_id`].
//! * Mobile: `android_main_entry` (Android) and the iOS `start_app` pattern — see [`mobile`].
//!
//! ```no_run
//! use space3d_core::prelude::*;
//! use space3d_platform::*;
//! let mut app = App::new();
//! app.add_plugin(PlatformPlugin { window: WindowPlugin::titled("My game") });
//! app.add_system(Stage::Update, |ctx| {
//!     if ctx.resources.expect::<Input>().just_pressed(KeyCode::Escape) {
//!         ctx.send(AppExit);
//!     }
//! });
//! app.run();
//! ```
//!
//! # Keyboard layouts
//!
//! [`KeyCode`]s are **physical** keys — positions named after the US QWERTY key in that place.
//! Bind those for movement and other position-based controls ([`Binding::Key`]): WASD stays an
//! inverted T on AZERTY (`Z/Q/S/D`), QWERTZ, Dvorak or JIS. The **logical** key — what the key
//! means on the player's layout — is recorded on every [`KeyEvent`] ([`Input::key_events`],
//! [`Input::logical_key`]); the UI uses it, so text editing and shortcuts (`Ctrl/Cmd+Z/A/C/V/X`)
//! follow the layout as in native apps.
//!
//! To show bindings, use the layout's labels instead of code names: [`Input::key_label`]
//! (`"q"` for `KeyA` on AZERTY, `"Space"`, `"F1"`) or [`Input::key_cap`] (`"Q"`, never empty).
//! They come from the OS layout table (macOS, Windows, Chromium-based browsers) or are learned
//! from observed key events (everywhere; the only source on Linux, Firefox/Safari, Android and
//! iOS). Names of non-character keys are English — localize them in the game.
//! [`KeyboardLayout::generation`] changes whenever labels may have changed. Platform details
//! and limitations: [`layout`].
//!
//! ```
//! use space3d_platform::*;
//! let mut input = Input::default();
//! // Injected QWERTZ event: the key in QWERTY's `Y` position types "z".
//! input.inject_key_tap_logical(KeyCode::KeyY, LogicalKey::Character("z".into()));
//! assert_eq!(input.key_cap(KeyCode::KeyY), "Z");
//! assert_eq!(input.key_label(KeyCode::Space).as_deref(), Some("Space"));
//! ```

#[cfg(target_os = "android")]
mod android;
pub mod background;
pub mod gamepad;
mod ime;
mod input;
#[cfg(target_os = "ios")]
pub mod ios;
pub mod layout;
pub mod mobile;
mod os_buttons;
mod runner;
pub mod touch_pointer;
#[cfg(target_arch = "wasm32")]
mod web;
mod window;

pub use background::{background_mode, set_background_mode};
pub use ime::ImeEvent;
pub use input::{
    ActionMap, ActionState, ActionValue, Binding, ButtonState, Gamepad, GamepadAxis, GamepadButton, GamepadChange,
    Input, InputPlugin, KeyCode, KeyEvent, MouseButton, MouseButtonCause, MouseButtonEvent, Touch, TouchGesture,
    TouchPhase,
};
pub use layout::{
    KeyLabel, KeyLabelSource, KeyLocation, KeyboardLayout, LogicalKey, NamedKey, english_key_name,
    is_writing_system_key, us_qwerty_label, writing_system_keys,
};
#[cfg(target_os = "android")]
pub use mobile::{AndroidApp, android_app, android_main_entry};
pub use mobile::{immersive, set_immersive};
#[cfg(target_arch = "wasm32")]
pub use web::{find_canvas, init_web_logging};
pub use window::{
    AppliedCursor, CursorIcon, KeyboardInset, PlatformEvent, SafeArea, TextInput, WindowCursor, WindowIcon,
    WindowPlugin, WindowState,
};

pub use winit;

use space3d_core::{App, Plugin};

/// Everything needed for a windowed app: [`WindowPlugin`] (which adds [`InputPlugin`] and
/// `space3d_render::RenderPlugin`) with the given window settings.
#[derive(Clone, Debug, Default)]
pub struct PlatformPlugin {
    /// Window settings.
    pub window: WindowPlugin,
}

impl Plugin for PlatformPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(self.window.clone());
    }
}

/// Common imports.
pub mod prelude {
    pub use crate::{
        ActionMap, ActionState, Binding, Input, InputPlugin, KeyCode, MouseButton, PlatformEvent, PlatformPlugin,
        WindowPlugin, WindowState,
    };
}
