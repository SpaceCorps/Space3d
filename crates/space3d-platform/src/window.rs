//! Window configuration ([`WindowPlugin`]), runtime window state and platform events.

use crate::input::InputPlugin;
use glam::UVec2;
use space3d_core::{App, Plugin};
use space3d_render::RenderPlugin;

/// Opens a window, creates the renderer's surface and installs the winit runner.
///
/// Adds [`InputPlugin`] and [`RenderPlugin`]. `App::run()` then hands control to winit:
/// every redraw calls `app.update(dt)`; the loop ends on window close (when
/// `exit_on_close`) or when an `AppExit` event is sent.
#[derive(Clone, Debug)]
pub struct WindowPlugin {
    /// Window title (also settable at runtime via [`WindowState::title`]).
    pub title: String,
    /// Initial logical width.
    pub width: u32,
    /// Initial logical height.
    pub height: u32,
    /// Present with vsync.
    pub vsync: bool,
    /// User-resizable window.
    pub resizable: bool,
    /// MSAA samples (1 or 4).
    pub msaa_samples: u32,
    /// Web: id of an existing `<canvas>` to render into. `None` creates one and appends it to
    /// `<body>`.
    pub canvas_id: Option<String>,
    /// Exit when the window is closed (otherwise only [`PlatformEvent::CloseRequested`] is sent).
    pub exit_on_close: bool,
    /// Phones: fingers are also the mouse, for a game whose controls and UI read the mouse
    /// ([`crate::touch_pointer`]): a tap is a left click (pressed a frame after the pointer
    /// arrives, so a UI knows whose click it is), a drag a left drag, a long press a right
    /// click, a two-finger pinch the wheel; the pointer leaves when the finger lifts. The
    /// fingers still arrive as touches. Default `false`.
    pub touch_as_mouse: bool,
    /// Window icon (title bar, Alt-Tab) and, on Windows, the taskbar icon. `None` uses the
    /// platform default. Ignored on macOS (the Dock shows the app bundle's icon), the web and
    /// mobile.
    pub icon: Option<WindowIcon>,
    /// Desktops (Windows, macOS, Linux): turn the OS input method (IME, for typing Chinese,
    /// Japanese and Korean) on while a text field has the focus, and pass the composition and
    /// the caret's place on (`Input::ime_events`, `TextInput`). Default `true`; the environment
    /// variable `SPACE3D_NO_IME=1` switches it off whatever this says. Off, a desktop window is
    /// as it was before the engine knew input methods. Phones and the web are not affected.
    pub ime: bool,
    /// Content under the title bar: the app's picture fills the whole window, the title bar's
    /// strip included, and the window buttons sit on top of it. Default `false`.
    ///
    /// The [`crate::TitleBar`] resource says what the strip is and what the buttons cover, and
    /// takes the app's claims on it ([`crate::title_bar`]). **Windows**: the window keeps what
    /// a system title bar does (dragging, double click, Snap Layouts, Aero Snap, resizing from
    /// every edge); the three buttons are the app's to draw, which `space3d-ui` does (an app
    /// without it draws them from [`crate::TitleBar::buttons`]). **macOS**: a full-size content
    /// view under a transparent title bar without a title; the traffic lights stay. **Linux,
    /// web, Android, iOS**: nothing changes.
    ///
    /// A window whose caption is taken off (fullscreen: winit's, or an app's own) has no title
    /// bar while it is off.
    pub content_under_title_bar: bool,
}

/// An RGBA8 window icon (see [`WindowPlugin::icon`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowIcon {
    /// Row-major RGBA8 pixels, `width * height * 4` bytes.
    pub rgba: Vec<u8>,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl WindowIcon {
    /// An icon from an RGBA8 image.
    pub fn from_image(image: &space3d_core::ImageData) -> Self {
        Self {
            rgba: image.pixels.clone(),
            width: image.width,
            height: image.height,
        }
    }
}

pub use winit::window::CursorIcon;

/// The pointer shape shown over the window (resource, inserted by [`WindowPlugin`]).
///
/// `space3d-ui` writes egui's cursor here whenever egui asks for a different one (pointing hand
/// over buttons, I-beam over text fields, resize arrows...); a game may write it too, and the
/// last write wins. After every frame the runner applies it to the OS window with winit's
/// `Window::set_cursor`, which Windows re-applies on `WM_SETCURSOR` and macOS keeps through its
/// cursor rects. The pointer is visible only while [`Self::icon`] is `Some` and
/// [`WindowState::cursor_visible`] is `true`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowCursor {
    /// Requested shape; `None` hides the pointer (egui's `CursorIcon::None`).
    pub icon: Option<CursorIcon>,
    applied: Option<AppliedCursor>,
}

impl Default for WindowCursor {
    fn default() -> Self {
        Self {
            icon: Some(CursorIcon::Default),
            applied: None,
        }
    }
}

impl WindowCursor {
    /// What the runner last applied to a real OS window; `None` before the first frame with a
    /// window, and always without one (headless apps, tests without a window).
    pub fn applied(&self) -> Option<AppliedCursor> {
        self.applied
    }

    pub(crate) fn set_applied(&mut self, applied: AppliedCursor) {
        self.applied = Some(applied);
    }
}

/// A cursor the runner applied to the OS window (see [`WindowCursor::applied`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AppliedCursor {
    /// Shape set with `Window::set_cursor` (kept while hidden).
    pub icon: CursorIcon,
    /// Whether the pointer is shown.
    pub visible: bool,
}

impl Default for WindowPlugin {
    fn default() -> Self {
        Self {
            title: "Space3d".into(),
            width: 1280,
            height: 720,
            vsync: true,
            resizable: true,
            msaa_samples: 4,
            canvas_id: None,
            exit_on_close: true,
            touch_as_mouse: false,
            icon: None,
            ime: true,
            content_under_title_bar: false,
        }
    }
}

impl WindowPlugin {
    /// Settings with a title.
    pub fn titled(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            ..Default::default()
        }
    }
}

impl Plugin for WindowPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(InputPlugin);
        app.add_plugin(RenderPlugin::default());
        app.add_event::<PlatformEvent>();
        app.insert_resource(WindowState {
            physical_size: UVec2::new(self.width, self.height),
            scale_factor: 1.0,
            focused: true,
            suspended: false,
            title: self.title.clone(),
            cursor_visible: true,
            cursor_grab: false,
        });
        app.init_resource::<WindowCursor>();
        app.init_resource::<SafeArea>();
        app.init_resource::<crate::title_bar::TitleBar>();
        if self.touch_as_mouse {
            app.resources.init::<crate::touch_pointer::TouchMouse>().enabled = true;
        }
        let settings = self.clone();
        app.set_runner(move |app| crate::runner::run(app, settings));
    }
}

/// Live window state (resource). `title`, `cursor_visible` and `cursor_grab` are writable:
/// the runner applies changes after each frame.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowState {
    /// Framebuffer size in physical pixels.
    pub physical_size: UVec2,
    /// Physical pixels per logical pixel.
    pub scale_factor: f64,
    /// Has keyboard focus.
    pub focused: bool,
    /// Suspended by the OS (mobile background): no surface, no frames.
    pub suspended: bool,
    /// Window title.
    pub title: String,
    /// Show the OS cursor.
    pub cursor_visible: bool,
    /// Grab/lock the cursor (FPS-style camera control).
    pub cursor_grab: bool,
}

/// The edges of the window the system keeps for itself, in physical pixels (resource): a
/// phone's notch or camera island, rounded corners and the home bar. The picture covers them;
/// what must stay readable and tappable keeps out of them (`space3d-ui` lays its UI out inside).
/// Filled on iOS (the view's insets) and on Android (the activity's content rectangle: what the
/// status bar, the navigation bar and a cutout leave; for an edge-to-edge window, which is told
/// no rectangle of its own, [`Self::ANDROID_NAVIGATION_POINTS`] at the bottom); zero everywhere else.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SafeArea {
    /// Kept at the left edge.
    pub left: u32,
    /// Kept at the top edge.
    pub top: u32,
    /// Kept at the right edge.
    pub right: u32,
    /// Kept at the bottom edge.
    pub bottom: u32,
}

impl SafeArea {
    /// Android, when the content rectangle is the whole window (an edge-to-edge app is not told
    /// the bars): the height kept at the bottom for the navigation bar, in dp. 24 is the gesture
    /// bar's; the three-button bar (48) and a cutout are not covered.
    pub const ANDROID_NAVIGATION_POINTS: f64 = 24.0;

    /// The insets of a content rectangle `(left, top, right, bottom)` in a window of `size`
    /// pixels (Android's `onContentRectChanged`). A rectangle that is empty or not inside the
    /// window (not told yet, or of the other orientation during a rotation) gives no insets.
    pub fn from_content_rect(rect: (i32, i32, i32, i32), size: (u32, u32)) -> Self {
        let (left, top, right, bottom) = rect;
        let (w, h) = (size.0 as i32, size.1 as i32);
        if left < 0 || top < 0 || right <= left || bottom <= top || right > w || bottom > h {
            return Self::default();
        }
        Self {
            left: left as u32,
            top: top as u32,
            right: (w - right) as u32,
            bottom: (h - bottom) as u32,
        }
    }

    /// Insets that are a portrait screen's (something at the top, nothing at either side) on a
    /// landscape window of `size` pixels, turned: the top's at **both** sides (which side the
    /// notch is on is not known), nothing at the top, two thirds of the bottom's (a home bar is
    /// thinner along the long edge). Anything else is returned as it is.
    ///
    /// UIKit answers so in the iOS 27 Simulator for a landscape scene (`crate::ios`); on a
    /// device that answers right this changes nothing.
    pub fn turned_for_landscape(self, size: (u32, u32)) -> Self {
        if size.0 > size.1 && self.top > 0 && self.left == 0 && self.right == 0 {
            Self {
                left: self.top,
                right: self.top,
                top: 0,
                bottom: self.bottom * 2 / 3,
            }
        } else {
            self
        }
    }
}

/// What the on-screen keyboard covers of the window, measured from its bottom edge, in pixels
/// (resource; the runner keeps it on Android 11 and later, asking Java's `WindowInsets` while a
/// text field wants text). 0: no keyboard is up, or the platform does not say (iOS, a desktop).
/// A window that is not resized for the keyboard (a NativeActivity's is not) moves what is
/// being typed in above this line itself.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KeyboardInset {
    /// Pixels of the window's bottom under the keyboard.
    pub bottom: u32,
}

/// Whether a touch that went down at `down` and lifted at `up` (pixels) was a tap: it stayed
/// within `slop` pixels. A finger that scrolled or dragged is none.
#[cfg_attr(not(any(target_os = "android", target_os = "ios", test)), allow(dead_code))]
pub(crate) fn lift_is_a_tap(down: glam::Vec2, up: glam::Vec2, slop: f32) -> bool {
    down.distance(up) <= slop
}

/// Whether the game wants typed text now (resource). `space3d-ui` sets `wanted` while one of
/// its text fields has the focus; on Android and iOS the runner shows the on-screen keyboard
/// while it is set and hides it after. On a desktop the runner turns the OS input method (IME,
/// for Chinese, Japanese and Korean) on while it is set, unless the field is a password field.
/// A game with text fields of its own writes it itself.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TextInput {
    /// A text field has the focus.
    pub wanted: bool,
    /// The field with the focus is a password field: a desktop's input method stays off for it
    /// (what is typed into a password field is never composed or offered candidates). Phones show
    /// their keyboard all the same. It is what the last UI pass said of the field that was
    /// focused while the pass drew it, so the input method follows a focus that moves to or from
    /// a password field one frame later.
    pub password: bool,
    /// Where the text caret of that field is, in physical pixels of the window, for a desktop
    /// input method to open its candidate list next to (`space3d-ui` sets it from egui's
    /// `PlatformOutput::ime`). `None`: no field has the focus, or it is a password field.
    pub caret: Option<crate::ime::ImeArea>,
    /// The last UI pass asked to interrupt an input method composition that is going on
    /// (egui's `IMEOutput::should_interrupt_composition`: the player clicked into the field, or
    /// the game moved the focus with `request_focus`). egui then counts the preedit as plain text
    /// at once, so the OS composition must end too, or the next preedit would be written beside
    /// the old one. The runner acts on the rising edge of it while a composition is going on
    /// (`set_ime_allowed(false)`, then `true`): a game that asks for the focus every frame
    /// interrupts nothing after the first.
    pub interrupt: bool,
}

/// Window/application lifecycle events (`Events<PlatformEvent>`).
#[derive(Clone, Debug, PartialEq)]
pub enum PlatformEvent {
    /// Framebuffer resized (physical pixels).
    Resized(UVec2),
    /// DPI changed.
    ScaleFactorChanged(f64),
    /// Focus gained/lost.
    Focused(bool),
    /// User asked to close the window.
    CloseRequested,
    /// App went to background (mobile) — surface destroyed.
    Suspended,
    /// App returned to foreground — surface recreated.
    Resumed,
}

#[cfg(test)]
mod safe_area_tests {
    use super::SafeArea;

    #[test]
    fn a_content_rectangle_gives_the_insets() {
        // A 2400 x 1080 landscape phone: a cutout of 136 px at the left, a gesture bar of 63 below.
        let area = SafeArea::from_content_rect((136, 0, 2400, 1017), (2400, 1080));
        assert_eq!(
            area,
            SafeArea {
                left: 136,
                top: 0,
                right: 0,
                bottom: 63
            }
        );
        // Not told yet, or the portrait rectangle while the window is already landscape.
        assert_eq!(
            SafeArea::from_content_rect((0, 0, 0, 0), (2400, 1080)),
            SafeArea::default()
        );
        assert_eq!(
            SafeArea::from_content_rect((0, 136, 1080, 2337), (2400, 1080)),
            SafeArea::default()
        );
    }

    #[test]
    fn portrait_insets_on_a_landscape_window_are_turned() {
        let portrait = SafeArea {
            left: 0,
            top: 186,
            right: 0,
            bottom: 102,
        };
        assert_eq!(
            portrait.turned_for_landscape((2622, 1206)),
            SafeArea {
                left: 186,
                top: 0,
                right: 186,
                bottom: 68
            }
        );
        // A portrait window keeps them, and so do insets that already name a side.
        assert_eq!(portrait.turned_for_landscape((1206, 2622)), portrait);
        let right = SafeArea {
            left: 177,
            top: 0,
            right: 0,
            bottom: 63,
        };
        assert_eq!(right.turned_for_landscape((2622, 1206)), right);
    }
}

#[cfg(test)]
mod ime_tests {
    use super::*;

    #[test]
    fn the_input_method_support_is_on_unless_the_game_says_so() {
        assert!(WindowPlugin::default().ime);
        assert!(WindowPlugin::titled("a game").ime);
        // A game's literal with the rest defaulted keeps it.
        let plugin = WindowPlugin {
            width: 640,
            ..Default::default()
        };
        assert!(plugin.ime);
        assert!(!TextInput::default().wanted && !TextInput::default().password);
        assert_eq!(TextInput::default().caret, None);
    }
}

#[cfg(test)]
mod keyboard_tests {
    use super::*;

    /// Only a finger that stayed where it went down brings the on-screen keyboard back.
    #[test]
    fn a_lift_within_the_slop_is_a_tap() {
        let down = glam::Vec2::new(400.0, 300.0);
        assert!(lift_is_a_tap(down, down, 26.0));
        assert!(lift_is_a_tap(down, glam::Vec2::new(410.0, 320.0), 26.0));
        assert!(!lift_is_a_tap(down, glam::Vec2::new(400.0, 500.0), 26.0));
        assert_eq!(KeyboardInset::default().bottom, 0);
    }
}
