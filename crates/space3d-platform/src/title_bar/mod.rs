//! Content under the title bar ([`crate::WindowPlugin::content_under_title_bar`]): the app's
//! picture fills the whole window, the title bar's strip included, and the window buttons sit on
//! top of it.
//!
//! The [`TitleBar`] resource is the whole interface. The platform says there what the strip is
//! and what the window buttons cover; the app says which parts of the strip are its own, so a
//! click there reaches it. Everything else of the strip drags the window.
//!
//! | platform | what the option does | the window buttons |
//! |---|---|---|
//! | Windows | the title bar's area becomes part of the picture | drawn by the app (`space3d-ui` does it), at the top right; the system still knows where they are (Snap Layouts) |
//! | macOS | full-size content view under a transparent title bar without a title | the system's traffic lights, at the top left |
//! | Linux, web, Android, iOS | nothing: the window is as without the option | as before |
//!
//! In fullscreen there is no title bar, and [`TitleBar::active`] is `false`.
//!
//! All sizes and places are **physical pixels of the window's picture** (what
//! [`crate::WindowState::physical_size`] counts), like [`crate::SafeArea`].
//!
//! # Which clicks the app gets
//!
//! A point of the strip belongs to the app when it is in one of the app's rectangles:
//! [`TitleBar::claim`] (this frame's, for hit areas of the app's own), `space3d-ui`'s
//! ([`TitleBar::set_ui_claims`]: every egui widget that senses a click or a drag, floating
//! windows included, but not a backdrop that covers the whole screen), or while
//! [`TitleBar::keep_strip`] was called this frame (the whole strip). The window buttons always
//! belong to the window, and so does the thin band at the top edge that resizes a window that is
//! not maximised (Windows). What the platform asks about is the last frame's rectangles, so a
//! widget takes clicks from the frame after it first appears.

#[cfg(target_os = "macos")]
pub(crate) mod macos;
#[cfg(target_os = "windows")]
pub(crate) mod windows;

use glam::Vec2;

/// A rectangle in physical pixels of the window's picture; `x`, `y` are its top left corner.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TitleBarRect {
    /// Left edge.
    pub x: i32,
    /// Top edge.
    pub y: i32,
    /// Width (0 or less: empty).
    pub width: i32,
    /// Height (0 or less: empty).
    pub height: i32,
}

impl TitleBarRect {
    /// A rectangle from its top left corner and size.
    pub const fn new(x: i32, y: i32, width: i32, height: i32) -> Self {
        Self { x, y, width, height }
    }

    /// The smallest whole-pixel rectangle that covers `min`..`max` (physical pixels).
    pub fn covering(min: Vec2, max: Vec2) -> Self {
        let (x, y) = (min.x.floor() as i32, min.y.floor() as i32);
        let (right, bottom) = (max.x.ceil() as i32, max.y.ceil() as i32);
        Self::new(x, y, right.saturating_sub(x), bottom.saturating_sub(y))
    }

    /// Just past the right edge.
    pub const fn right(self) -> i32 {
        self.x.saturating_add(self.width)
    }

    /// Just past the bottom edge.
    pub const fn bottom(self) -> i32 {
        self.y.saturating_add(self.height)
    }

    /// No pixels.
    pub const fn is_empty(self) -> bool {
        self.width <= 0 || self.height <= 0
    }

    /// Whether the pixel at `(x, y)` is inside (the right and bottom edges are outside).
    pub const fn contains(self, x: i32, y: i32) -> bool {
        x >= self.x && y >= self.y && x < self.right() && y < self.bottom()
    }

    /// Whether the two share a pixel.
    pub const fn intersects(self, other: Self) -> bool {
        !self.is_empty()
            && !other.is_empty()
            && self.x < other.right()
            && other.x < self.right()
            && self.y < other.bottom()
            && other.y < self.bottom()
    }
}

/// One of the three window buttons.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WindowControl {
    /// Minimise.
    Minimize,
    /// Maximise, or restore a maximised window.
    Maximize,
    /// Close the window (what the system's close button does: `PlatformEvent::CloseRequested`).
    Close,
}

impl WindowControl {
    /// From left to right, as Windows has them.
    pub const ALL: [Self; 3] = [Self::Minimize, Self::Maximize, Self::Close];
}

/// A window button the app draws (Windows): where it is and how it looks right now. A click on
/// it is the window's own: the app does not handle it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowButton {
    /// Which button.
    pub control: WindowControl,
    /// Where it is (physical pixels).
    pub rect: TitleBarRect,
    /// Whether it does anything (a window that cannot be resized has a dimmed maximise button).
    pub enabled: bool,
    /// The pointer is over it.
    pub hovered: bool,
    /// The left button went down on it and is still held, with the pointer over it.
    pub pressed: bool,
}

/// The window's title bar when the app's picture fills it (resource, inserted by
/// [`crate::WindowPlugin`]; see the [module docs](self)).
///
/// The platform writes the fields above [`Self::draw_buttons`] before every frame. A test or a
/// headless run may insert one itself ([`Self::simulated_windows`], [`Self::simulated_macos`]).
#[derive(Clone, Debug, PartialEq)]
pub struct TitleBar {
    /// The picture is under the title bar now. `false`: the option is off, the platform keeps
    /// its title bar (Linux, web, phones), or the window is in fullscreen. Everything the
    /// platform says is then empty and the app has nothing to keep clear of.
    pub active: bool,
    /// The height of the strip at the top of the picture that drags the window, in physical
    /// pixels. It is the height of the window buttons.
    pub height: u32,
    /// What the window buttons cover: Windows' three at the top right, macOS' traffic lights at
    /// the top left. Nothing the app must show or click belongs under it.
    pub controls: Option<TitleBarRect>,
    /// Windows: the three buttons, for whoever draws them. `None` on macOS (the traffic lights
    /// are the system's) and when not [`Self::active`].
    pub buttons: Option<[WindowButton; 3]>,
    /// Windows: the window is maximised (the maximise button then restores). `false` on macOS.
    pub maximized: bool,
    /// Windows: physical pixels per logical pixel the buttons are sized with (the window's
    /// scale, whatever zoom the app's UI has).
    pub scale: f32,
    /// `space3d-ui` draws [`Self::buttons`]. Default `true`; set it `false` to draw your own.
    /// An app without `space3d-ui` draws them itself: the system draws none (Windows).
    pub draw_buttons: bool,
    /// `space3d-ui`'s rectangles (replaced by every UI pass).
    ui_claims: Vec<TitleBarRect>,
    /// The app's rectangles of this frame (cleared before every frame).
    app_claims: Vec<TitleBarRect>,
    /// The app keeps the whole strip this frame (cleared before every frame).
    app_keeps_strip: bool,
}

impl Default for TitleBar {
    fn default() -> Self {
        Self {
            active: false,
            height: 0,
            controls: None,
            buttons: None,
            maximized: false,
            scale: 1.0,
            draw_buttons: true,
            ui_claims: Vec::new(),
            app_claims: Vec::new(),
            app_keeps_strip: false,
        }
    }
}

impl TitleBar {
    /// Clicks and drags in `rect` are the app's this frame (call it every frame the app wants
    /// it). `space3d-ui`'s widgets need no call.
    pub fn claim(&mut self, rect: TitleBarRect) {
        if !rect.is_empty() {
            self.app_claims.push(rect);
        }
    }

    /// The whole strip is the app's this frame: it does not drag the window (a game that needs
    /// every click while the pointer steers). The window buttons still work.
    pub fn keep_strip(&mut self) {
        self.app_keeps_strip = true;
    }

    /// `space3d-ui`'s rectangles of a UI pass. They take the place of the last pass's; the old
    /// buffer comes back in `rects` (empty), so the UI allocates nothing from pass to pass.
    pub fn set_ui_claims(&mut self, rects: &mut Vec<TitleBarRect>) {
        std::mem::swap(&mut self.ui_claims, rects);
        rects.clear();
    }

    /// Every rectangle of the strip that is the app's: the UI's and the app's own.
    pub fn claims(&self) -> impl Iterator<Item = TitleBarRect> + '_ {
        self.ui_claims.iter().chain(&self.app_claims).copied()
    }

    /// Whether the app keeps the whole strip this frame ([`Self::keep_strip`]).
    pub fn strip_kept(&self) -> bool {
        self.app_keeps_strip
    }

    /// The strip at the top of a window `width` pixels wide (empty when not [`Self::active`]).
    pub fn strip(&self, width: u32) -> TitleBarRect {
        if !self.active {
            return TitleBarRect::default();
        }
        TitleBarRect::new(0, 0, width as i32, self.height as i32)
    }

    /// Pixels at the left end of the strip that the window buttons cover (macOS), else 0.
    pub fn left_inset(&self) -> u32 {
        match self.controls {
            Some(c) if self.active && c.x <= 0 => c.right().max(0) as u32,
            _ => 0,
        }
    }

    /// Pixels at the right end of the strip of a window `width` pixels wide that the window
    /// buttons cover (Windows), else 0.
    pub fn right_inset(&self, width: u32) -> u32 {
        match self.controls {
            Some(c) if self.active && c.x > 0 => (width as i32 - c.x).max(0) as u32,
            _ => 0,
        }
    }

    /// What a Windows window `width` pixels wide at `scale` has (nothing hovered or pressed):
    /// what the runner writes, for tests and headless pictures of a UI that must keep clear of
    /// the buttons.
    pub fn simulated_windows(width: u32, scale: f32, maximized: bool) -> Self {
        let mut bar = Self::default();
        bar.set_windows(
            &WindowsFrame {
                width: width as i32,
                scale,
                maximized,
                resizable: true,
                minimizable: true,
                maximizable: true,
            },
            None,
            None,
        );
        bar
    }

    /// What a macOS window at `scale` has: a strip of 28 points whose left 78 are the traffic
    /// lights (the runner measures the real ones).
    pub fn simulated_macos(scale: f32) -> Self {
        let mut bar = Self::default();
        bar.set_traffic_lights(
            (MACOS_HEIGHT * scale).round() as u32,
            (MACOS_LIGHTS_WIDTH * scale).round() as i32,
            scale,
        );
        bar
    }

    /// No title bar under the picture (fullscreen, or a platform without the option).
    pub(crate) fn set_inactive(&mut self) {
        self.active = false;
        self.height = 0;
        self.controls = None;
        self.buttons = None;
        self.maximized = false;
    }

    /// Windows' strip and buttons for `frame`, with the button under the pointer and the one
    /// held down.
    pub(crate) fn set_windows(
        &mut self,
        frame: &WindowsFrame,
        hovered: Option<WindowControl>,
        pressed: Option<WindowControl>,
    ) {
        let rects = frame.button_rects();
        let buttons = std::array::from_fn(|i| {
            let control = WindowControl::ALL[i];
            let enabled = frame.enabled(control);
            WindowButton {
                control,
                rect: rects[i],
                enabled,
                hovered: enabled && hovered == Some(control),
                pressed: enabled && hovered == Some(control) && pressed == Some(control),
            }
        });
        self.active = true;
        self.height = frame.strip_height().max(0) as u32;
        self.controls = Some(TitleBarRect::new(
            rects[0].x,
            0,
            rects[2].right() - rects[0].x,
            frame.strip_height(),
        ));
        self.buttons = Some(buttons);
        self.maximized = frame.maximized;
        self.scale = frame.scale;
    }

    /// macOS' strip: `height` pixels, the traffic lights `lights` pixels wide from the left.
    pub(crate) fn set_traffic_lights(&mut self, height: u32, lights: i32, scale: f32) {
        self.active = true;
        self.height = height;
        self.controls = Some(TitleBarRect::new(0, 0, lights, height as i32));
        self.buttons = None;
        self.maximized = false;
        self.scale = scale;
    }

    /// macOS: whether a press at `(x, y)` in a window `width` pixels wide drags it: in the
    /// strip, not on the traffic lights and not the app's. (Windows asks its own question,
    /// `windows_hit`.)
    #[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
    pub(crate) fn drags_at(&self, x: i32, y: i32, width: u32) -> bool {
        self.strip(width).contains(x, y)
            && !self.controls.is_some_and(|c| c.contains(x, y))
            && !self.app_keeps_strip
            && !self.claims().any(|c| c.contains(x, y))
    }

    /// A new frame: what the app claimed for the last one is gone.
    pub(crate) fn clear_app_claims(&mut self) {
        self.app_claims.clear();
        self.app_keeps_strip = false;
    }
}

/// Logical pixels: Windows 11's title bar is 32 high and its buttons 46 wide.
pub(crate) const WINDOWS_HEIGHT: f32 = 32.0;
pub(crate) const WINDOWS_BUTTON_WIDTH: f32 = 46.0;
/// Logical pixels of the top edge that resize a window that is not maximised: the sizing frame
/// and its padding (4 + 4 at 100 %), what a system title bar has.
#[cfg_attr(not(any(target_os = "windows", test)), allow(dead_code))]
pub(crate) const WINDOWS_RESIZE_BAND: f32 = 8.0;

/// macOS' title bar in points, and what its three traffic lights take of it from the left.
pub(crate) const MACOS_HEIGHT: f32 = 28.0;
pub(crate) const MACOS_LIGHTS_WIDTH: f32 = 78.0;

/// A Windows window as the strip's geometry sees it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct WindowsFrame {
    /// The picture's width in pixels.
    pub width: i32,
    /// Pixels per logical pixel.
    pub scale: f32,
    pub maximized: bool,
    pub resizable: bool,
    pub minimizable: bool,
    pub maximizable: bool,
}

impl WindowsFrame {
    fn px(&self, logical: f32) -> i32 {
        (logical * self.scale.max(0.1)).round() as i32
    }

    pub fn strip_height(&self) -> i32 {
        self.px(WINDOWS_HEIGHT)
    }

    /// The top band that resizes: none on a window that is maximised or cannot be resized.
    #[cfg_attr(not(any(target_os = "windows", test)), allow(dead_code))]
    pub fn resize_band(&self) -> i32 {
        if self.maximized || !self.resizable {
            0
        } else {
            self.px(WINDOWS_RESIZE_BAND)
        }
    }

    /// Minimise, maximise, close: side by side, the close button in the corner.
    pub fn button_rects(&self) -> [TitleBarRect; 3] {
        let (w, h) = (self.px(WINDOWS_BUTTON_WIDTH), self.strip_height());
        std::array::from_fn(|i| TitleBarRect::new(self.width - w * (3 - i as i32), 0, w, h))
    }

    pub fn enabled(&self, control: WindowControl) -> bool {
        match control {
            WindowControl::Minimize => self.minimizable,
            WindowControl::Maximize => self.maximizable,
            WindowControl::Close => true,
        }
    }
}

/// What a point inside the picture of a window with content under its title bar is.
#[cfg_attr(not(any(target_os = "windows", test)), allow(dead_code))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Hit {
    /// The app's: a normal click.
    App,
    /// Drags the window (a double click maximises it).
    Drag,
    /// Resizes from the top edge or one of the top corners.
    ResizeTop,
    ResizeTopLeft,
    ResizeTopRight,
    /// A window button.
    Control(WindowControl),
}

/// What the point `(x, y)` of the picture is (Windows). `claims` are the app's rectangles and
/// `keep` says it keeps the whole strip.
///
/// Order: the top resize band (not on a maximised window, whose top row belongs to what is
/// under it, so the corner of the screen closes it), then the buttons, which win over anything
/// the app claims (a window must always close), then the app's, then the strip's drag. A
/// disabled button is nothing to press, and does not drag either.
#[cfg_attr(not(any(target_os = "windows", test)), allow(dead_code))]
pub(crate) fn windows_hit(
    frame: &WindowsFrame,
    x: i32,
    y: i32,
    claims: impl IntoIterator<Item = TitleBarRect>,
    keep: bool,
) -> Hit {
    if x < 0 || y < 0 || x >= frame.width {
        return Hit::App;
    }
    let band = frame.resize_band();
    if y < band {
        return if x < band {
            Hit::ResizeTopLeft
        } else if x >= frame.width - band {
            Hit::ResizeTopRight
        } else {
            Hit::ResizeTop
        };
    }
    if y >= frame.strip_height() {
        return Hit::App;
    }
    for (rect, control) in frame.button_rects().into_iter().zip(WindowControl::ALL) {
        if rect.contains(x, y) {
            return if frame.enabled(control) {
                Hit::Control(control)
            } else {
                Hit::App
            };
        }
    }
    if keep || claims.into_iter().any(|c| c.contains(x, y)) {
        Hit::App
    } else {
        Hit::Drag
    }
}

#[cfg(test)]
mod tests;
