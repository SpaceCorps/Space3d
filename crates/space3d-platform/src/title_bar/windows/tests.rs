//! The window procedure on a real window of the test's thread that is **never shown**: nothing
//! appears on the screen and nothing takes the focus. A maximised window is one that has the
//! style bit (`IsZoomed` reads it), so the geometry is the system's own without a window ever
//! covering anything.

use super::*;
use windows_sys::Win32::Graphics::Gdi::ClientToScreen;
use windows_sys::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DestroyWindow, MSG, PM_REMOVE, PeekMessageW, RegisterClassW, SendMessageW, SetWindowLongW,
    WNDCLASSW, WS_MAXIMIZE, WS_OVERLAPPEDWINDOW,
};

struct TestWindow(HWND);

impl TestWindow {
    fn new() -> Self {
        let class: Vec<u16> = "space3d-title-bar-test\0".encode_utf16().collect();
        let definition = WNDCLASSW {
            lpfnWndProc: Some(DefWindowProcW),
            lpszClassName: class.as_ptr(),
            ..Default::default()
        };
        // SAFETY: a complete class; registering it again (another test) fails harmlessly.
        unsafe { RegisterClassW(&definition) };
        // SAFETY: the registered class, no parent, menu or data; the window is not visible.
        let hwnd = unsafe {
            CreateWindowExW(
                0,
                class.as_ptr(),
                class.as_ptr(),
                WS_OVERLAPPEDWINDOW,
                100,
                100,
                816,
                639,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null(),
            )
        };
        assert!(!hwnd.is_null(), "no test window");
        Self(hwnd)
    }

    fn install(&self, scale: f32) -> Chrome {
        // SAFETY: a live window of this thread.
        unsafe { Chrome::install(self.0 as isize, scale) }.expect("the procedure goes in")
    }

    fn window(&self) -> RECT {
        let mut r = RECT::default();
        // SAFETY: a live window.
        assert_ne!(unsafe { GetWindowRect(self.0, &mut r) }, 0);
        r
    }

    /// How far the client area is in from the window at the left, top, right and bottom.
    fn insets(&self) -> [i32; 4] {
        let window = self.window();
        let mut origin = POINT::default();
        // SAFETY: a live window and a POINT to convert.
        assert_ne!(unsafe { ClientToScreen(self.0, &mut origin) }, 0);
        let (w, h) = self.client();
        [
            origin.x - window.left,
            origin.y - window.top,
            window.right - (origin.x + w),
            window.bottom - (origin.y + h),
        ]
    }

    fn client(&self) -> (i32, i32) {
        // SAFETY: a live window.
        unsafe { client_size(self.0) }
    }

    fn send(&self, message: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
        // SAFETY: a live window of this thread; the messages sent here carry what they need.
        unsafe { SendMessageW(self.0, message, wparam, lparam) }
    }

    /// `WM_NCHITTEST` for the point `(x, y)` of the client area (which may be outside it).
    fn hit(&self, x: i32, y: i32) -> u32 {
        let mut point = POINT { x, y };
        // SAFETY: as above.
        unsafe { ClientToScreen(self.0, &mut point) };
        let lparam = (point.x as u16 as u32 | (point.y as u16 as u32) << 16) as i32 as LPARAM;
        self.send(WM_NCHITTEST, 0, lparam) as u32
    }

    /// Takes the system commands the window posted to itself, so that none of them runs.
    fn commands(&self) -> Vec<u32> {
        let mut commands = Vec::new();
        let mut message = MSG::default();
        // SAFETY: a MSG to write and this thread's window.
        while unsafe { PeekMessageW(&mut message, self.0, WM_SYSCOMMAND, WM_SYSCOMMAND, PM_REMOVE) } != 0 {
            commands.push(message.wParam as u32);
        }
        commands
    }

    fn set_style(&self, style: u32) {
        // SAFETY: a live window; then the frame is computed again.
        unsafe {
            SetWindowLongW(self.0, GWL_STYLE, style as i32);
            frame_changed(self.0);
        }
    }

    fn style(&self) -> u32 {
        // SAFETY: a live window.
        unsafe { style_of(self.0) }
    }

    fn autohide(&self) -> [i32; 4] {
        let window = self.window();
        // SAFETY: a RECT to read.
        unsafe { autohide_edges(&window) }.map(|there| if there { AUTOHIDE_GAP } else { 0 })
    }
}

impl Drop for TestWindow {
    fn drop(&mut self) {
        // SAFETY: this thread's window.
        unsafe { DestroyWindow(self.0) };
    }
}

fn synced(chrome: &Chrome) -> TitleBar {
    let mut bar = TitleBar::default();
    chrome.sync(&mut bar);
    bar
}

#[test]
fn the_client_area_goes_up_to_the_top_of_the_window() {
    let window = TestWindow::new();
    let [left, top, right, bottom] = window.insets();
    assert!(left > 0 && right == left && bottom == left, "a sizing frame all round");
    assert!(top > left + 10, "and a title bar above it: {top}");
    let outer = window.window();
    let _chrome = window.install(1.0);
    assert_eq!(window.insets(), [left, 0, right, bottom], "only the title bar went");
    let after = window.window();
    assert_eq!(
        (after.left, after.top, after.right, after.bottom),
        (outer.left, outer.top, outer.right, outer.bottom),
        "the window did not move or change size"
    );
}

#[test]
fn a_second_install_is_refused() {
    let window = TestWindow::new();
    let _chrome = window.install(1.0);
    // SAFETY: a live window of this thread.
    assert!(unsafe { Chrome::install(window.0 as isize, 1.0) }.is_none());
}

#[test]
fn every_part_of_the_strip_answers_for_itself() {
    let window = TestWindow::new();
    let _chrome = window.install(1.0);
    let (w, h) = window.client();
    assert_eq!(window.hit(200, 20), HTCAPTION);
    assert_eq!(window.hit(200, 32), HTCLIENT);
    assert_eq!(window.hit(200, h - 1), HTCLIENT);
    assert_eq!(window.hit(200, 2), HTTOP);
    assert_eq!(window.hit(2, 2), HTTOPLEFT);
    assert_eq!(window.hit(w - 2, 2), HTTOPRIGHT);
    assert_eq!(window.hit(w - 3 * 46 + 5, 20), HTMINBUTTON);
    assert_eq!(window.hit(w - 2 * 46 + 5, 20), HTMAXBUTTON, "what shows Snap Layouts");
    assert_eq!(window.hit(w - 46 + 5, 20), HTCLOSE);
    // The system's own sizing frame, whose top ends are corners now.
    assert_eq!(window.hit(-2, 200), HTLEFT);
    assert_eq!(window.hit(w + 1, 200), HTRIGHT);
    assert_eq!(window.hit(200, h + 1), HTBOTTOM);
    assert_eq!(window.hit(-2, 2), HTTOPLEFT);
    assert_eq!(window.hit(w + 1, 2), HTTOPRIGHT);
}

#[test]
fn the_apps_rectangles_are_the_clients_once_published() {
    let window = TestWindow::new();
    let chrome = window.install(1.0);
    assert_eq!(window.hit(120, 20), HTCAPTION);
    let mut bar = synced(&chrome);
    bar.claim(TitleBarRect::new(100, 10, 50, 20));
    chrome.publish(&bar);
    assert_eq!(window.hit(120, 20), HTCLIENT);
    assert_eq!(window.hit(160, 20), HTCAPTION);
    bar.clear_app_claims();
    bar.keep_strip();
    chrome.publish(&bar);
    assert_eq!(window.hit(160, 20), HTCLIENT);
    bar.clear_app_claims();
    chrome.publish(&bar);
    assert_eq!(window.hit(120, 20), HTCAPTION, "a frame without claims takes them back");
}

#[test]
fn the_scale_sizes_the_strip() {
    let window = TestWindow::new();
    let chrome = window.install(1.0);
    chrome.set_scale(1.5);
    let (w, _) = window.client();
    assert_eq!(window.hit(200, 47), HTCAPTION);
    assert_eq!(window.hit(200, 48), HTCLIENT);
    assert_eq!(window.hit(200, 11), HTTOP);
    assert_eq!(window.hit(w - 68, 30), HTCLOSE);
    assert_eq!(synced(&chrome), TitleBar::simulated_windows(w as u32, 1.5, false));
}

#[test]
fn a_maximised_window_is_not_cut_off_and_leaves_an_auto_hide_taskbar_its_edge() {
    let window = TestWindow::new();
    let [frame, ..] = window.insets();
    let chrome = window.install(1.0);
    window.set_style(window.style() | WS_MAXIMIZE);
    let [gap_left, gap_top, gap_right, gap_bottom] = window.autohide();
    // The system hangs a maximised window over its monitor by the frame on every side: the
    // client area starts that far in at the top too, so nothing of the picture is off screen.
    assert_eq!(
        window.insets(),
        [frame + gap_left, frame + gap_top, frame + gap_right, frame + gap_bottom]
    );
    let bar = synced(&chrome);
    assert!(bar.active && bar.maximized);
    // The top row is a button's or the caption's, not a resize edge.
    let (w, _) = window.client();
    assert_eq!(window.hit(w - 1, 0), HTCLOSE);
    assert_eq!(window.hit(200, 0), HTCAPTION);
}

#[test]
fn the_client_rectangle_of_a_maximised_window_is_the_work_area() {
    // A 1920 x 1080 monitor with a 48 pixel taskbar below: the system proposes the work area
    // plus an 8 pixel frame all round, and the standard frame takes the frame and the caption.
    let proposed = RECT {
        left: -8,
        top: -8,
        right: 1928,
        bottom: 1040,
    };
    let standard = RECT {
        left: 0,
        top: 23,
        right: 1920,
        bottom: 1032,
    };
    let sides = |r: RECT| (r.left, r.top, r.right, r.bottom);
    assert_eq!(
        sides(client_rect(proposed, standard, true, [false; 4])),
        (0, 0, 1920, 1032)
    );
    // An auto-hide taskbar below, then at the top, left and right.
    assert_eq!(
        sides(client_rect(proposed, standard, true, [false, false, false, true])),
        (0, 0, 1920, 1030)
    );
    assert_eq!(
        sides(client_rect(proposed, standard, true, [true, true, true, false])),
        (2, 2, 1918, 1032)
    );
    // Restored: the top of the window, whatever the taskbar does.
    assert_eq!(
        sides(client_rect(proposed, standard, false, [true; 4])),
        (0, -8, 1920, 1032)
    );
}

#[test]
fn a_window_without_a_caption_is_left_alone() {
    let window = TestWindow::new();
    let [frame, ..] = window.insets();
    let chrome = window.install(1.0);
    let decorated = window.style();
    // What borderless fullscreen takes off.
    window.set_style(decorated & !(WS_CAPTION | WS_THICKFRAME | WS_MAXIMIZEBOX));
    assert_eq!(window.insets(), [0; 4], "the picture is the whole window");
    let bar = synced(&chrome);
    assert!(!bar.active && bar.buttons.is_none() && bar.height == 0);
    let (w, _) = window.client();
    for (x, y) in [(200, 0), (200, 20), (w - 10, 20), (w - 1, 0)] {
        assert_eq!(window.hit(x, y), HTCLIENT, "({x}, {y}) in fullscreen");
    }
    // And back.
    window.set_style(decorated);
    assert_eq!(window.insets(), [frame, 0, frame, frame]);
    assert!(synced(&chrome).active);
}

#[test]
fn a_click_on_a_button_posts_the_systems_command() {
    let window = TestWindow::new();
    let _chrome = window.install(1.0);
    let click = |down: u32, up: u32| {
        window.send(WM_NCLBUTTONDOWN, down as WPARAM, 0);
        window.send(WM_NCLBUTTONUP, up as WPARAM, 0);
        window.commands()
    };
    assert_eq!(click(HTMINBUTTON, HTMINBUTTON), [SC_MINIMIZE]);
    assert_eq!(click(HTMAXBUTTON, HTMAXBUTTON), [SC_MAXIMIZE]);
    assert_eq!(click(HTCLOSE, HTCLOSE), [SC_CLOSE]);
    assert_eq!(click(HTCLOSE, HTMAXBUTTON), [] as [u32; 0], "released elsewhere");
    // A release without a press is nobody's click.
    window.send(WM_NCLBUTTONUP, HTCLOSE as WPARAM, 0);
    assert!(window.commands().is_empty());
    // A maximised window's maximise button restores.
    window.set_style(window.style() | WS_MAXIMIZE);
    assert_eq!(click(HTMAXBUTTON, HTMAXBUTTON), [SC_RESTORE]);
}

#[test]
fn the_pointer_over_a_button_is_told_to_the_app() {
    let window = TestWindow::new();
    let chrome = window.install(1.0);
    let lit = |chrome: &Chrome| synced(chrome).buttons.unwrap().map(|b| (b.hovered, b.pressed));
    window.send(WM_NCMOUSEMOVE, HTMAXBUTTON as WPARAM, 0);
    assert_eq!(lit(&chrome), [(false, false), (true, false), (false, false)]);
    window.send(WM_NCLBUTTONDOWN, HTMAXBUTTON as WPARAM, 0);
    assert_eq!(lit(&chrome), [(false, false), (true, true), (false, false)]);
    window.send(WM_NCMOUSELEAVE, 0, 0);
    assert_eq!(lit(&chrome), [(false, false); 3]);
    window.send(WM_NCMOUSEMOVE, HTCLOSE as WPARAM, 0);
    window.send(WM_MOUSEMOVE, 0, (100 | (100 << 16)) as LPARAM);
    assert_eq!(lit(&chrome), [(false, false); 3], "back in the client area");
}

#[test]
fn the_picture_gets_the_size_that_was_asked_for() {
    let window = TestWindow::new();
    let chrome = window.install(1.0);
    chrome.fit_client(1000, 500);
    assert_eq!(window.client(), (1000, 500));
    chrome.fit_client(640, 400);
    assert_eq!(window.client(), (640, 400));
    // Not a maximised window's.
    window.set_style(window.style() | WS_MAXIMIZE);
    let before = window.client();
    chrome.fit_client(900, 600);
    assert_eq!(window.client(), before);
}

#[test]
fn a_new_scale_keeps_the_pictures_logical_size() {
    let window = TestWindow::new();
    let chrome = window.install(1.0);
    chrome.fit_client(640, 400);
    // (a plain window does not resize itself for WM_DPICHANGED, as winit does before this
    // runs: what is held here is the size the procedure puts right afterwards)
    let mut suggested = window.window();
    let wparam = (144 | (144 << 16)) as WPARAM;
    window.send(WM_DPICHANGED, wparam, (&raw mut suggested) as LPARAM);
    assert_eq!(window.client(), (960, 600));
    let (w, _) = window.client();
    assert_eq!(window.hit(w - 68, 40), HTCLOSE, "the buttons follow the new scale");
}
