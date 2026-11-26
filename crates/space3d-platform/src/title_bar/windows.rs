//! Windows: the title bar's area becomes part of the picture.
//!
//! The window keeps every style a decorated window has, so everything the system does for one
//! stays the system's: dragging and Aero Snap, double click to maximise, the system menu, the
//! minimise and maximise animations, the shadow, Win+arrows, Snap Layouts. Two things change, in
//! a window procedure put in front of winit's:
//!
//! 1. `WM_NCCALCSIZE`: the client area begins at the window's top, not under a title bar. The
//!    sizing frame at the left, the right and below stays the system's. A maximised window
//!    hangs over its monitor by that frame, so its client area begins that far down, and it
//!    leaves [`AUTOHIDE_GAP`] pixels at an edge with an auto-hide taskbar, which would not come
//!    out from behind a window that covers the whole monitor.
//! 2. `WM_NCHITTEST`: inside the client area the top rows resize, the strip's buttons answer as
//!    the system's (`HTMAXBUTTON` is what shows Windows 11's Snap Layouts), the app's
//!    rectangles are the client's and the rest of the strip is the caption.
//!
//! Over a button the system sends non-client mouse messages: the procedure keeps what is under
//! the pointer and what is held (the app draws it), and on a release over the button it went
//! down on posts the system command. The system's own button tracking never runs (it would
//! paint a classic button over the picture).
//!
//! A window without a caption (fullscreen: winit's, or an app's own that takes `WS_CAPTION` off)
//! is left entirely alone.
//!
//! winit sizes a window as if it had its title bar (`AdjustWindowRectExForDpi`), at creation and
//! when the monitor's scale changes: [`Chrome::fit_client`] puts the asked size right once, and
//! `WM_DPICHANGED` does after winit.

use super::{Hit, TitleBar, TitleBarRect, WindowControl, WindowsFrame, windows_hit};
use std::mem::size_of;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex, MutexGuard};
use windows_sys::Win32::Foundation::{GetLastError, HWND, LPARAM, LRESULT, POINT, RECT, SetLastError, WPARAM};
use windows_sys::Win32::Graphics::Gdi::{
    GetMonitorInfoW, MONITOR_DEFAULTTONEAREST, MONITORINFO, MonitorFromRect, ScreenToClient,
};
use windows_sys::Win32::UI::Input::KeyboardAndMouse::{TME_LEAVE, TME_NONCLIENT, TRACKMOUSEEVENT, TrackMouseEvent};
use windows_sys::Win32::UI::Shell::{
    ABE_BOTTOM, ABE_LEFT, ABE_RIGHT, ABE_TOP, ABM_GETAUTOHIDEBAREX, APPBARDATA, SHAppBarMessage,
};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    CallWindowProcW, DefWindowProcW, GWL_STYLE, GWLP_WNDPROC, GetClientRect, GetPropW, GetWindowLongW, GetWindowRect,
    HTBOTTOM, HTCAPTION, HTCLIENT, HTCLOSE, HTLEFT, HTMAXBUTTON, HTMINBUTTON, HTRIGHT, HTTOP, HTTOPLEFT, HTTOPRIGHT,
    IsIconic, IsZoomed, NCCALCSIZE_PARAMS, PostMessageW, RemovePropW, SC_CLOSE, SC_MAXIMIZE, SC_MINIMIZE, SC_RESTORE,
    SWP_FRAMECHANGED, SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOOWNERZORDER, SWP_NOSIZE, SWP_NOZORDER, SetPropW, SetWindowPos,
    WM_DPICHANGED, WM_LBUTTONUP, WM_MOUSEMOVE, WM_NCCALCSIZE, WM_NCDESTROY, WM_NCHITTEST, WM_NCLBUTTONDBLCLK,
    WM_NCLBUTTONDOWN, WM_NCLBUTTONUP, WM_NCMOUSEMOVE, WM_SYSCOMMAND, WNDPROC, WS_CAPTION, WS_MAXIMIZEBOX,
    WS_MINIMIZEBOX, WS_THICKFRAME,
};

/// `WM_NCMOUSELEAVE` (windows-sys files it with the common controls, which this crate has no
/// other use for).
const WM_NCMOUSELEAVE: u32 = 0x02A2;

/// The window property that holds the procedure's state.
const PROPERTY: &[u16] = &{
    let ascii = *b"space3d.title_bar\0";
    let mut wide = [0u16; 18];
    let mut i = 0;
    while i < ascii.len() {
        wide[i] = ascii[i] as u16;
        i += 1;
    }
    wide
};

/// Pixels a maximised window leaves free at an edge with an auto-hide taskbar.
pub(crate) const AUTOHIDE_GAP: i32 = 2;

/// What the procedure and the runner share. Never locked across a call into the system: a
/// message sent from inside one comes back to the procedure on the same thread.
struct Shared {
    /// The procedure that was the window's (winit's).
    previous: isize,
    /// Pixels per logical pixel, as the runner knows it.
    scale: f32,
    /// The app's rectangles of the strip, and whether it keeps all of it.
    claims: Vec<TitleBarRect>,
    keep: bool,
    /// The button under the pointer and the one held down.
    hovered: Option<WindowControl>,
    pressed: Option<WindowControl>,
    /// The system was asked to say when the pointer leaves the buttons.
    tracking: bool,
}

type Handle = Arc<Mutex<Shared>>;

fn lock(shared: &Handle) -> MutexGuard<'_, Shared> {
    shared.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The runner's end of a window with content under its title bar.
pub(crate) struct Chrome {
    /// The window (a number, so that the runner stays `Send`; used on its own thread only).
    hwnd: isize,
    shared: Handle,
}

impl Chrome {
    /// Puts the procedure in front of the window's and has the frame computed again. `None`
    /// (logged) leaves the window with its system title bar.
    ///
    /// # Safety
    ///
    /// `hwnd` is a live window of the calling thread.
    pub unsafe fn install(hwnd: isize, scale: f32) -> Option<Self> {
        let window = hwnd as HWND;
        // SAFETY: a live window and a NUL-terminated name (the caller's contract; as below).
        if !unsafe { GetPropW(window, PROPERTY.as_ptr()) }.is_null() {
            tracing::warn!("title bar: the window has its content under it already");
            return None;
        }
        // SAFETY: a live window and a valid index.
        let previous = unsafe { window_proc(window) };
        if previous == 0 {
            tracing::warn!("title bar: no window procedure to put ours in front of; the system keeps the title bar");
            return None;
        }
        let shared: Handle = Arc::new(Mutex::new(Shared {
            previous,
            scale,
            claims: Vec::new(),
            keep: false,
            hovered: None,
            pressed: None,
            tracking: false,
        }));
        let stored = Arc::into_raw(shared.clone());
        // SAFETY: a live window, a NUL-terminated name; the pointer is the property's own
        // reference, given back at WM_NCDESTROY.
        if unsafe { SetPropW(window, PROPERTY.as_ptr(), stored as *mut core::ffi::c_void) } == 0 {
            // SAFETY: the reference `into_raw` made above, which nothing else has.
            drop(unsafe { Arc::from_raw(stored) });
            tracing::warn!("title bar: SetPropW failed; the system keeps the title bar");
            return None;
        }
        // SAFETY: no preconditions. (0 is also a previous value: the error code tells.)
        unsafe { SetLastError(0) };
        // SAFETY: a live window of this thread; the new procedure has WNDPROC's signature.
        let replaced = unsafe { set_window_proc(window, procedure as *const () as isize) };
        // SAFETY: no preconditions.
        if replaced == 0 && unsafe { GetLastError() } != 0 {
            // SAFETY: the property set above, and its reference.
            unsafe {
                RemovePropW(window, PROPERTY.as_ptr());
                drop(Arc::from_raw(stored));
            }
            tracing::warn!("title bar: the window procedure could not be replaced; the system keeps the title bar");
            return None;
        }
        // SAFETY: a live window; only the frame is computed again.
        unsafe { frame_changed(window) };
        Some(Self { hwnd, shared })
    }

    /// The runner's scale of the window.
    pub fn set_scale(&self, scale: f32) {
        lock(&self.shared).scale = scale;
    }

    /// What the strip is now, for the app: inactive while the window has no caption
    /// (fullscreen).
    pub fn sync(&self, bar: &mut TitleBar) {
        let (scale, hovered, pressed) = {
            let s = lock(&self.shared);
            (s.scale, s.hovered, s.pressed)
        };
        // SAFETY: the window this runner made; a dead one answers 0 to everything below.
        match unsafe { frame_of(self.hwnd as HWND, scale) } {
            Some(frame) => bar.set_windows(&frame, hovered, pressed),
            None => bar.set_inactive(),
        }
    }

    /// Takes the app's rectangles of the frame that was just drawn, for the hit tests to come.
    pub fn publish(&self, bar: &TitleBar) {
        let mut s = lock(&self.shared);
        s.keep = bar.strip_kept();
        s.claims.clear();
        s.claims.extend(bar.claims());
    }

    /// Resizes the window so that its client area is `width` x `height` pixels. Not a
    /// maximised, minimised or fullscreen window's: the system sizes those.
    pub fn fit_client(&self, width: i32, height: i32) {
        // SAFETY: the window this runner made, on its thread.
        unsafe { fit_client(self.hwnd as HWND, width, height) }
    }
}

#[cfg(target_pointer_width = "64")]
unsafe fn window_proc(hwnd: HWND) -> isize {
    // SAFETY: the caller's.
    unsafe { windows_sys::Win32::UI::WindowsAndMessaging::GetWindowLongPtrW(hwnd, GWLP_WNDPROC) }
}

#[cfg(target_pointer_width = "64")]
unsafe fn set_window_proc(hwnd: HWND, procedure: isize) -> isize {
    // SAFETY: the caller's.
    unsafe { windows_sys::Win32::UI::WindowsAndMessaging::SetWindowLongPtrW(hwnd, GWLP_WNDPROC, procedure) }
}

#[cfg(target_pointer_width = "32")]
unsafe fn window_proc(hwnd: HWND) -> isize {
    // SAFETY: the caller's.
    unsafe { GetWindowLongW(hwnd, GWLP_WNDPROC) as isize }
}

#[cfg(target_pointer_width = "32")]
unsafe fn set_window_proc(hwnd: HWND, procedure: isize) -> isize {
    // SAFETY: the caller's.
    unsafe {
        windows_sys::Win32::UI::WindowsAndMessaging::SetWindowLongW(hwnd, GWLP_WNDPROC, procedure as i32) as isize
    }
}

/// Has `WM_NCCALCSIZE` asked again; the window does not move, grow or get the focus.
unsafe fn frame_changed(hwnd: HWND) {
    // SAFETY: the caller's; the position, size and order are ignored by the flags.
    unsafe {
        SetWindowPos(
            hwnd,
            std::ptr::null_mut(),
            0,
            0,
            0,
            0,
            SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_NOACTIVATE,
        );
    }
}

unsafe fn style_of(hwnd: HWND) -> u32 {
    // SAFETY: any handle; a dead one answers 0.
    unsafe { GetWindowLongW(hwnd, GWL_STYLE) as u32 }
}

/// The window has a title bar to put the picture under (`WS_CAPTION`, both its bits).
fn has_caption(style: u32) -> bool {
    style & WS_CAPTION == WS_CAPTION
}

unsafe fn client_size(hwnd: HWND) -> (i32, i32) {
    let mut r = RECT::default();
    // SAFETY: any handle and a RECT to write; it stays empty when the call fails.
    unsafe { GetClientRect(hwnd, &mut r) };
    (r.right - r.left, r.bottom - r.top)
}

/// The window as the strip's geometry sees it; `None` without a caption.
unsafe fn frame_of(hwnd: HWND, scale: f32) -> Option<WindowsFrame> {
    // SAFETY (all): any handle.
    let style = unsafe { style_of(hwnd) };
    has_caption(style).then(|| WindowsFrame {
        // SAFETY: any handle.
        width: unsafe { client_size(hwnd) }.0,
        scale,
        // SAFETY: any handle.
        maximized: unsafe { IsZoomed(hwnd) } != 0,
        resizable: style & WS_THICKFRAME != 0,
        minimizable: style & WS_MINIMIZEBOX != 0,
        maximizable: style & WS_MAXIMIZEBOX != 0,
    })
}

unsafe fn fit_client(hwnd: HWND, width: i32, height: i32) {
    // SAFETY (all): the caller's window; RECTs are ours to write.
    unsafe {
        if !has_caption(style_of(hwnd)) || IsZoomed(hwnd) != 0 || IsIconic(hwnd) != 0 {
            return;
        }
        let (have_w, have_h) = client_size(hwnd);
        let (grow_w, grow_h) = (width - have_w, height - have_h);
        let mut window = RECT::default();
        if (grow_w, grow_h) == (0, 0) || width <= 0 || height <= 0 || GetWindowRect(hwnd, &mut window) == 0 {
            return;
        }
        SetWindowPos(
            hwnd,
            std::ptr::null_mut(),
            0,
            0,
            window.right - window.left + grow_w,
            window.bottom - window.top + grow_h,
            SWP_NOMOVE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_NOACTIVATE,
        );
    }
}

/// The client area, from the window rectangle the system proposes and the client area the
/// standard frame would have in it. `autohide`: the monitor has an auto-hide taskbar at its
/// left, top, right, bottom edge (asked for a maximised window only).
pub(crate) fn client_rect(proposed: RECT, standard: RECT, maximized: bool, autohide: [bool; 4]) -> RECT {
    let mut client = standard;
    client.top = proposed.top;
    if maximized {
        // A maximised window hangs over its monitor by the sizing frame on every side; the
        // standard frame's left side says by how much.
        client.top += (standard.left - proposed.left).max(0);
        let [left, top, right, bottom] = autohide.map(|there| if there { AUTOHIDE_GAP } else { 0 });
        client.left += left;
        client.top += top;
        client.right -= right;
        client.bottom -= bottom;
    }
    client
}

/// Which edges of the monitor a window at `window` is mostly on have an auto-hide taskbar:
/// left, top, right, bottom.
unsafe fn autohide_edges(window: &RECT) -> [bool; 4] {
    let mut info = MONITORINFO {
        cbSize: size_of::<MONITORINFO>() as u32,
        ..Default::default()
    };
    // SAFETY: a RECT to read; MONITOR_DEFAULTTONEAREST always answers with a monitor.
    let monitor = unsafe { MonitorFromRect(window, MONITOR_DEFAULTTONEAREST) };
    // SAFETY: `info` has its size set.
    if unsafe { GetMonitorInfoW(monitor, &mut info) } == 0 {
        return [false; 4];
    }
    [ABE_LEFT, ABE_TOP, ABE_RIGHT, ABE_BOTTOM].map(|edge| {
        // SAFETY: an all-zero APPBARDATA is valid; size, edge and monitor are set.
        let mut data: APPBARDATA = unsafe { std::mem::zeroed() };
        data.cbSize = size_of::<APPBARDATA>() as u32;
        data.uEdge = edge;
        data.rc = info.rcMonitor;
        // SAFETY: the message reads `data`; it answers the bar's window or 0.
        unsafe { SHAppBarMessage(ABM_GETAUTOHIDEBAREX, &mut data) != 0 }
    })
}

/// The button a non-client hit code names.
fn control_of(code: u32) -> Option<WindowControl> {
    match code {
        HTMINBUTTON => Some(WindowControl::Minimize),
        HTMAXBUTTON => Some(WindowControl::Maximize),
        HTCLOSE => Some(WindowControl::Close),
        _ => None,
    }
}

fn hit_code(hit: Hit) -> u32 {
    match hit {
        Hit::App => HTCLIENT,
        Hit::Drag => HTCAPTION,
        Hit::ResizeTop => HTTOP,
        Hit::ResizeTopLeft => HTTOPLEFT,
        Hit::ResizeTopRight => HTTOPRIGHT,
        Hit::Control(WindowControl::Minimize) => HTMINBUTTON,
        Hit::Control(WindowControl::Maximize) => HTMAXBUTTON,
        Hit::Control(WindowControl::Close) => HTCLOSE,
    }
}

/// What a click on `control` asks the window to do: the system command, so that it animates,
/// obeys the system menu and closes through `WM_CLOSE` as the system's buttons do.
unsafe fn command(hwnd: HWND, control: WindowControl) {
    let command = match control {
        WindowControl::Minimize => SC_MINIMIZE,
        // SAFETY: any handle.
        WindowControl::Maximize if unsafe { IsZoomed(hwnd) } != 0 => SC_RESTORE,
        WindowControl::Maximize => SC_MAXIMIZE,
        WindowControl::Close => SC_CLOSE,
    };
    // SAFETY: the window that was clicked; posted, so it runs after this message.
    unsafe { PostMessageW(hwnd, WM_SYSCOMMAND, command as WPARAM, 0) };
}

/// The state the window holds, with a reference of its own for as long as the message takes.
unsafe fn shared_of(hwnd: HWND) -> Option<Handle> {
    // SAFETY: any handle and a NUL-terminated name.
    let stored = unsafe { GetPropW(hwnd, PROPERTY.as_ptr()) } as *const Mutex<Shared>;
    (!stored.is_null()).then(|| {
        // SAFETY: the pointer `install` stored from `Arc::into_raw`; its reference lives until
        // WM_NCDESTROY removes the property, on this same thread.
        unsafe {
            Arc::increment_strong_count(stored);
            Arc::from_raw(stored)
        }
    })
}

unsafe fn call(previous: isize, hwnd: HWND, message: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    // SAFETY: `previous` is the value GetWindowLongPtrW(GWLP_WNDPROC) gave, which is what
    // CallWindowProcW takes.
    unsafe {
        let previous: WNDPROC = std::mem::transmute::<isize, WNDPROC>(previous);
        CallWindowProcW(previous, hwnd, message, wparam, lparam)
    }
}

unsafe extern "system" fn procedure(hwnd: HWND, message: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    // SAFETY: Windows calls the procedure with its own window.
    let Some(shared) = (unsafe { shared_of(hwnd) }) else {
        // SAFETY: Windows' own arguments.
        return unsafe { DefWindowProcW(hwnd, message, wparam, lparam) };
    };
    let previous = lock(&shared).previous;
    if message == WM_NCDESTROY {
        // SAFETY: the window is going away: give back its procedure and the property's
        // reference (`shared` keeps its own until the end of the call).
        unsafe {
            let stored = RemovePropW(hwnd, PROPERTY.as_ptr()) as *const Mutex<Shared>;
            if window_proc(hwnd) == procedure as *const () as isize {
                set_window_proc(hwnd, previous);
            }
            if !stored.is_null() {
                drop(Arc::from_raw(stored));
            }
            return call(previous, hwnd, message, wparam, lparam);
        }
    }
    // A panic must not cross into Windows: the message then goes to winit as it came.
    // SAFETY: Windows' own arguments, passed on.
    catch_unwind(AssertUnwindSafe(|| unsafe {
        handle(&shared, previous, hwnd, message, wparam, lparam)
    }))
    .unwrap_or_else(|_| {
        // SAFETY: as above.
        unsafe { call(previous, hwnd, message, wparam, lparam) }
    })
}

unsafe fn handle(
    shared: &Handle,
    previous: isize,
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // SAFETY (every use): the message Windows sent, passed on unchanged.
    let pass = || unsafe { call(previous, hwnd, message, wparam, lparam) };
    match message {
        WM_NCCALCSIZE => {
            // SAFETY: this thread's window.
            if !has_caption(unsafe { style_of(hwnd) }) || unsafe { IsIconic(hwnd) } != 0 || lparam == 0 {
                return pass();
            }
            // With `wparam` the message carries NCCALCSIZE_PARAMS, whose first RECT is the one;
            // without it, a RECT.
            let rect: *mut RECT = if wparam != 0 {
                // SAFETY: WM_NCCALCSIZE with TRUE: `lparam` points to NCCALCSIZE_PARAMS.
                unsafe { &raw mut (*(lparam as *mut NCCALCSIZE_PARAMS)).rgrc[0] }
            } else {
                lparam as *mut RECT
            };
            // SAFETY: the RECT of the message, before and after the standard frame is taken off.
            let proposed = unsafe { *rect };
            let result = pass();
            // SAFETY: as above.
            let standard = unsafe { *rect };
            // SAFETY: this thread's window (the style says maximised before this message).
            let maximized = unsafe { IsZoomed(hwnd) } != 0;
            let autohide = if maximized {
                // SAFETY: a RECT to read.
                unsafe { autohide_edges(&proposed) }
            } else {
                [false; 4]
            };
            // SAFETY: as above.
            unsafe { *rect = client_rect(proposed, standard, maximized, autohide) };
            result
        }
        WM_NCHITTEST => {
            let result = pass();
            let scale = lock(shared).scale;
            // SAFETY: this thread's window.
            let Some(frame) = (unsafe { frame_of(hwnd, scale) }) else {
                return result;
            };
            // Screen coordinates, signed: a monitor left of or above the first has negative ones.
            let mut point = POINT {
                x: i32::from(lparam as u16 as i16),
                y: i32::from((lparam >> 16) as u16 as i16),
            };
            // SAFETY: this thread's window and a POINT to convert.
            if unsafe { ScreenToClient(hwnd, &mut point) } == 0 {
                return result;
            }
            // SAFETY: this thread's window.
            let (width, height) = unsafe { client_size(hwnd) };
            if point.x >= 0 && point.y >= 0 && point.x < width && point.y < height {
                let s = lock(shared);
                return hit_code(windows_hit(&frame, point.x, point.y, s.claims.iter().copied(), s.keep)) as LRESULT;
            }
            // The system's sizing frame at the sides: its top ends are corners now.
            if point.y >= 0 && point.y < frame.resize_band() {
                match result as u32 {
                    HTLEFT => return HTTOPLEFT as LRESULT,
                    HTRIGHT => return HTTOPRIGHT as LRESULT,
                    _ => {}
                }
            }
            // Above the client area (a maximised window's overhang) nothing answers as a
            // button or a caption of a title bar that is not drawn.
            if point.y < 0 && !matches!(result as u32, HTLEFT | HTRIGHT | HTBOTTOM) {
                return HTCLIENT as LRESULT;
            }
            result
        }
        WM_NCMOUSEMOVE => {
            let control = control_of(wparam as u32);
            let ask = {
                let mut s = lock(shared);
                s.hovered = control;
                !std::mem::replace(&mut s.tracking, true)
            };
            if ask {
                let mut track = TRACKMOUSEEVENT {
                    cbSize: size_of::<TRACKMOUSEEVENT>() as u32,
                    dwFlags: TME_LEAVE | TME_NONCLIENT,
                    hwndTrack: hwnd,
                    dwHoverTime: 0,
                };
                // SAFETY: a TRACKMOUSEEVENT with its size set, for this window.
                if unsafe { TrackMouseEvent(&mut track) } == 0 {
                    lock(shared).tracking = false;
                }
            }
            pass()
        }
        WM_NCMOUSELEAVE => {
            let mut s = lock(shared);
            (s.hovered, s.pressed, s.tracking) = (None, None, false);
            drop(s);
            pass()
        }
        WM_MOUSEMOVE => {
            lock(shared).hovered = None;
            pass()
        }
        WM_LBUTTONUP => {
            lock(shared).pressed = None;
            pass()
        }
        WM_NCLBUTTONDOWN | WM_NCLBUTTONDBLCLK => {
            let control = control_of(wparam as u32);
            let mut s = lock(shared);
            s.pressed = control;
            if control.is_some() {
                s.hovered = control;
                // Not passed on: the system would track and paint a classic button.
                return 0;
            }
            drop(s);
            pass()
        }
        WM_NCLBUTTONUP => {
            let control = control_of(wparam as u32);
            let pressed = lock(shared).pressed.take();
            let Some(control) = control else {
                return pass();
            };
            if pressed == Some(control) {
                // SAFETY: this thread's window.
                unsafe { command(hwnd, control) };
            }
            0
        }
        WM_DPICHANGED => {
            // The scale and the picture's size before winit answers: it keeps the picture's
            // logical size, then makes the window big enough for a title bar on top of it.
            let old_scale = lock(shared).scale.max(0.1);
            // SAFETY: this thread's window.
            let (width, height) = unsafe { client_size(hwnd) };
            let result = pass();
            let new_scale = f32::from(wparam as u16) / 96.0;
            if new_scale > 0.0 {
                lock(shared).scale = new_scale;
                if new_scale != old_scale {
                    let scaled = |v: i32| (f64::from(v) / f64::from(old_scale) * f64::from(new_scale)).round() as i32;
                    // SAFETY: this thread's window.
                    unsafe { fit_client(hwnd, scaled(width), scaled(height)) };
                }
            }
            result
        }
        _ => pass(),
    }
}

#[cfg(test)]
mod tests;
