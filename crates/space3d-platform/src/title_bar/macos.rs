//! macOS: a full-size content view under a transparent title bar without a title. The traffic
//! lights stay, the system's own; what is left for the platform is to say where they are and to
//! drag the window from the strip.

use objc2::rc::Retained;
use objc2_app_kit::{NSView, NSWindow, NSWindowButton, NSWindowStyleMask};
use winit::platform::macos::WindowAttributesExtMacOS;
use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::window::{Window, WindowAttributes};

/// Points left free right of the zoom button.
const LIGHTS_MARGIN: f64 = 8.0;

pub(crate) fn window_attributes(attrs: WindowAttributes) -> WindowAttributes {
    attrs
        .with_titlebar_transparent(true)
        .with_fullsize_content_view(true)
        .with_title_hidden(true)
}

fn ns_window(window: &Window) -> Option<Retained<NSWindow>> {
    let handle = window.window_handle().ok()?;
    let RawWindowHandle::AppKit(handle) = handle.as_raw() else {
        return None;
    };
    // SAFETY: winit's own view, alive as long as the window; the runner is on the main thread,
    // which AppKit asks for.
    let view: &NSView = unsafe { handle.ns_view.cast::<NSView>().as_ref() };
    view.window()
}

/// The title bar's height and what the traffic lights take of it from the left, in points.
/// `None` without a title bar over the picture: native fullscreen, or a window whose title bar
/// the app took off.
pub(crate) fn measure(window: &Window) -> Option<(f64, f64)> {
    let window = ns_window(window)?;
    let style = window.styleMask();
    if style.contains(NSWindowStyleMask::FullScreen) || !style.contains(NSWindowStyleMask::Titled) {
        return None;
    }
    // SAFETY: a plain getter, on the main thread. With a full-size content view it is the part
    // of the window the title bar does not cover.
    let free = unsafe { window.contentLayoutRect() };
    let height = (window.frame().size.height - free.size.height).max(0.0);
    let zoom = window.standardWindowButton(NSWindowButton::NSWindowZoomButton)?;
    let zoom = zoom.frame();
    Some((height, zoom.origin.x + zoom.size.width + LIGHTS_MARGIN))
}
