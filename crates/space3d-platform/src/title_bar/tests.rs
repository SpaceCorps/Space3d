use super::*;

/// A restored window 1000 pixels wide at `scale`.
fn frame(scale: f32) -> WindowsFrame {
    WindowsFrame {
        width: 1000,
        scale,
        maximized: false,
        resizable: true,
        minimizable: true,
        maximizable: true,
    }
}

fn hit(frame: &WindowsFrame, x: i32, y: i32) -> Hit {
    windows_hit(frame, x, y, [], false)
}

#[test]
fn the_buttons_sit_side_by_side_in_the_top_right_corner() {
    for (scale, w, h) in [(1.0, 46, 32), (1.25, 58, 40), (1.5, 69, 48), (2.0, 92, 64)] {
        let [min, max, close] = frame(scale).button_rects();
        assert_eq!(close, TitleBarRect::new(1000 - w, 0, w, h), "scale {scale}");
        assert_eq!(max, TitleBarRect::new(1000 - 2 * w, 0, w, h), "scale {scale}");
        assert_eq!(min, TitleBarRect::new(1000 - 3 * w, 0, w, h), "scale {scale}");
        assert_eq!(frame(scale).strip_height(), h);
    }
}

#[test]
fn the_strip_drags_and_what_is_under_it_is_the_apps() {
    let f = frame(1.0);
    assert_eq!(hit(&f, 400, 20), Hit::Drag);
    assert_eq!(hit(&f, 400, 31), Hit::Drag);
    assert_eq!(hit(&f, 400, 32), Hit::App, "below the strip");
    assert_eq!(hit(&f, 2, 20), Hit::Drag, "below the corner's resize band");
    // Outside the picture is not this function's to answer.
    assert_eq!(hit(&f, -1, 20), Hit::App);
    assert_eq!(hit(&f, 1000, 20), Hit::App);
    assert_eq!(hit(&f, 400, -1), Hit::App);
}

#[test]
fn the_top_edge_and_corners_resize_a_window_that_is_not_maximised() {
    let f = frame(1.0);
    assert_eq!(hit(&f, 400, 0), Hit::ResizeTop);
    assert_eq!(hit(&f, 400, 7), Hit::ResizeTop);
    assert_eq!(hit(&f, 400, 8), Hit::Drag);
    assert_eq!(hit(&f, 7, 0), Hit::ResizeTopLeft);
    assert_eq!(hit(&f, 8, 0), Hit::ResizeTop);
    assert_eq!(hit(&f, 992, 3), Hit::ResizeTopRight, "above the close button too");
    assert_eq!(hit(&f, 991, 3), Hit::ResizeTop);
    // The band follows the scale: 8 logical pixels.
    assert_eq!(hit(&frame(1.5), 400, 11), Hit::ResizeTop);
    assert_eq!(hit(&frame(1.5), 400, 12), Hit::Drag);
    let fixed = WindowsFrame { resizable: false, ..f };
    assert_eq!(hit(&fixed, 400, 0), Hit::Drag, "a window that cannot be resized");
}

#[test]
fn a_maximised_windows_corner_closes_it_and_its_top_row_drags() {
    let f = WindowsFrame {
        maximized: true,
        ..frame(1.0)
    };
    assert_eq!(hit(&f, 999, 0), Hit::Control(WindowControl::Close));
    assert_eq!(hit(&f, 400, 0), Hit::Drag);
    assert_eq!(hit(&f, 0, 0), Hit::Drag);
}

#[test]
fn each_button_answers_as_itself_and_wins_over_the_apps_claims() {
    let f = frame(1.0);
    let everything = TitleBarRect::new(0, 0, 1000, 32);
    for (x, control) in [
        (1000 - 3 * 46, WindowControl::Minimize),
        (1000 - 2 * 46, WindowControl::Maximize),
        (1000 - 46, WindowControl::Close),
    ] {
        assert_eq!(hit(&f, x, 20), Hit::Control(control));
        assert_eq!(hit(&f, x + 45, 31), Hit::Control(control));
        assert_eq!(windows_hit(&f, x + 10, 20, [everything], true), Hit::Control(control));
    }
    assert_eq!(hit(&f, 1000 - 3 * 46 - 1, 20), Hit::Drag, "just left of minimise");
}

#[test]
fn a_disabled_button_does_nothing_and_does_not_drag() {
    let f = WindowsFrame {
        maximizable: false,
        ..frame(1.0)
    };
    assert_eq!(hit(&f, 1000 - 2 * 46 + 10, 20), Hit::App);
    assert_eq!(hit(&f, 1000 - 46 + 10, 20), Hit::Control(WindowControl::Close));
}

#[test]
fn the_apps_rectangles_take_their_own_clicks() {
    let f = frame(1.0);
    let claims = [TitleBarRect::new(100, 10, 50, 40), TitleBarRect::new(300, 0, 10, 10)];
    assert_eq!(windows_hit(&f, 100, 10, claims, false), Hit::App);
    assert_eq!(windows_hit(&f, 149, 31, claims, false), Hit::App);
    assert_eq!(windows_hit(&f, 150, 20, claims, false), Hit::Drag);
    assert_eq!(windows_hit(&f, 99, 20, claims, false), Hit::Drag);
    // The resize band stays the window's above a widget.
    assert_eq!(windows_hit(&f, 305, 4, claims, false), Hit::ResizeTop);
    // Keeping the strip: nothing drags.
    assert_eq!(windows_hit(&f, 600, 20, [], true), Hit::App);
    assert_eq!(windows_hit(&f, 600, 4, [], true), Hit::ResizeTop);
}

#[test]
fn the_app_is_told_what_the_buttons_cover() {
    let bar = TitleBar::simulated_windows(1600, 1.5, false);
    assert!(bar.active);
    assert_eq!(bar.height, 48);
    assert_eq!(bar.controls, Some(TitleBarRect::new(1600 - 3 * 69, 0, 207, 48)));
    assert_eq!((bar.left_inset(), bar.right_inset(1600)), (0, 207));
    assert_eq!(bar.strip(1600), TitleBarRect::new(0, 0, 1600, 48));
    let buttons = bar.buttons.expect("Windows has buttons to draw");
    assert_eq!(buttons.map(|b| b.control), WindowControl::ALL);
    assert!(buttons.iter().all(|b| b.enabled && !b.hovered && !b.pressed));

    let mac = TitleBar::simulated_macos(2.0);
    assert_eq!(mac.height, 56);
    assert_eq!((mac.left_inset(), mac.right_inset(1600)), (156, 0));
    assert!(mac.buttons.is_none(), "the traffic lights are the system's");
}

#[test]
fn without_a_title_bar_under_the_picture_there_is_nothing_to_keep_clear_of() {
    let off = TitleBar::default();
    assert!(!off.active && off.draw_buttons);
    assert_eq!((off.height, off.left_inset(), off.right_inset(800)), (0, 0, 0));
    assert!(off.strip(800).is_empty());

    let mut bar = TitleBar::simulated_windows(800, 1.0, true);
    bar.set_inactive();
    assert!(!bar.active && bar.controls.is_none() && bar.buttons.is_none() && !bar.maximized);
    assert_eq!((bar.height, bar.right_inset(800)), (0, 0));
}

#[test]
fn a_button_is_pressed_only_while_the_pointer_is_on_it() {
    let mut bar = TitleBar::default();
    let f = frame(1.0);
    let close = WindowControl::Close;
    let state = |bar: &TitleBar| bar.buttons.unwrap().map(|b| (b.hovered, b.pressed));
    bar.set_windows(&f, Some(close), Some(close));
    assert_eq!(state(&bar), [(false, false), (false, false), (true, true)]);
    bar.set_windows(&f, Some(WindowControl::Maximize), Some(close));
    assert_eq!(state(&bar), [(false, false), (true, false), (false, false)]);
    bar.set_windows(&f, None, Some(close));
    assert_eq!(state(&bar), [(false, false); 3]);
    // A disabled button never lights up.
    let f = WindowsFrame {
        maximizable: false,
        ..f
    };
    bar.set_windows(&f, Some(WindowControl::Maximize), Some(WindowControl::Maximize));
    assert_eq!(state(&bar), [(false, false); 3]);
    assert!(!bar.buttons.unwrap()[1].enabled);
}

#[test]
fn the_apps_claims_last_a_frame_and_the_uis_until_the_next_pass() {
    let mut bar = TitleBar::default();
    let (a, b) = (TitleBarRect::new(0, 0, 10, 10), TitleBarRect::new(20, 0, 10, 10));
    bar.claim(a);
    bar.claim(TitleBarRect::new(5, 5, 0, 10));
    let mut ui = vec![b];
    bar.set_ui_claims(&mut ui);
    assert!(ui.is_empty());
    assert_eq!(bar.claims().collect::<Vec<_>>(), [b, a], "an empty rectangle is none");
    bar.keep_strip();
    assert!(bar.strip_kept());
    bar.clear_app_claims();
    assert!(!bar.strip_kept());
    assert_eq!(bar.claims().collect::<Vec<_>>(), [b]);
    // The next pass takes the place of the last one, and hands its buffer back.
    ui.push(a);
    bar.set_ui_claims(&mut ui);
    assert!(ui.is_empty() && ui.capacity() >= 1);
    assert_eq!(bar.claims().collect::<Vec<_>>(), [a]);
}

#[test]
fn macos_drags_from_the_strip_that_is_neither_the_lights_nor_the_apps() {
    let mut bar = TitleBar::simulated_macos(1.0);
    assert!(bar.drags_at(400, 10, 800));
    assert!(!bar.drags_at(40, 10, 800), "the traffic lights");
    assert!(!bar.drags_at(400, 28, 800), "below the strip");
    assert!(!bar.drags_at(800, 10, 800), "outside the window");
    bar.claim(TitleBarRect::new(390, 0, 20, 28));
    assert!(!bar.drags_at(400, 10, 800));
    assert!(bar.drags_at(420, 10, 800));
    bar.keep_strip();
    assert!(!bar.drags_at(420, 10, 800));
    bar.clear_app_claims();
    bar.set_inactive();
    assert!(!bar.drags_at(420, 10, 800), "fullscreen");
}

#[test]
fn rectangles_cover_contain_and_intersect() {
    let r = TitleBarRect::covering(Vec2::new(10.4, 0.0), Vec2::new(20.2, 31.5));
    assert_eq!(r, TitleBarRect::new(10, 0, 11, 32));
    assert!(r.contains(10, 0) && r.contains(20, 31));
    assert!(!r.contains(21, 31) && !r.contains(20, 32) && !r.contains(9, 0));
    assert!(r.intersects(TitleBarRect::new(20, 31, 5, 5)));
    assert!(!r.intersects(TitleBarRect::new(21, 0, 5, 5)));
    assert!(!r.intersects(TitleBarRect::default()));
    assert!(TitleBarRect::new(3, 3, 0, 5).is_empty());
}
