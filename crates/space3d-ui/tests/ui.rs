//! Headless UI tests: egui output shows up in captures, injected input reaches widgets, the
//! profiler sees system spans and render targets can be displayed as egui images.
//! GPU tests skip (return early) when no adapter is available.

use space3d_core::prelude::*;
use space3d_platform::{Input, KeyCode, LogicalKey, MouseButton};
use space3d_render::*;
use space3d_ui::*;

const BG: [u8; 4] = [0, 0, 255, 255];

/// Headless app (camera clearing to pure blue) with the deterministic UI, or `None` without GPU.
fn ui_app(w: u32, h: u32) -> Option<App> {
    let mut app = App::new();
    app.add_plugin(RenderPlugin::default());
    match pollster::block_on(Renderer::new_headless(w, h)) {
        Ok(r) => {
            app.insert_resource(r);
        }
        Err(e) => {
            eprintln!("skipping GPU test: no adapter available ({e})");
            return None;
        }
    }
    app.add_plugin(UiPlugin::deterministic());
    app.world.spawn((
        Transform::from_xyz(0.0, 0.0, 3.0),
        Camera3d::default().with_clear_color(Vec4::new(0.0, 0.0, 1.0, 1.0)),
    ));
    Some(app)
}

fn non_background(img: &ImageData, x0: u32, y0: u32, x1: u32, y1: u32) -> usize {
    let mut n = 0;
    for y in y0..y1.min(img.height) {
        for x in x0..x1.min(img.width) {
            if img.get(x, y) != BG {
                n += 1;
            }
        }
    }
    n
}

#[derive(Default)]
struct Clicks {
    count: u32,
    button: Option<egui::Rect>,
}

fn window_system(ctx: &egui::Context, sys: &mut SystemCtx) {
    egui::Window::new("Test window")
        .fixed_pos([20.0, 20.0])
        .fixed_size([140.0, 60.0])
        .show(ctx, |ui| {
            let r = ui.button("Click me");
            let clicks = sys.resources.init::<Clicks>();
            clicks.button = Some(r.rect);
            if r.clicked() {
                clicks.count += 1;
            }
        });
}

#[test]
fn egui_window_appears_in_headless_capture() {
    let Some(mut app) = ui_app(256, 192) else { return };
    app.add_ui_system(window_system);
    app.step_ticks(3);
    let img = render_to_image(&mut app).expect("capture");
    let inside = non_background(&img, 20, 20, 160, 80);
    assert!(inside > 2000, "window pixels drawn over the 3D clear color ({inside})");
    assert_eq!(
        non_background(&img, 200, 120, 256, 192),
        0,
        "outside the window stays clear"
    );
    let stats = app.resources.expect::<EguiContext>().stats();
    assert!(stats.primitives > 0 && stats.vertices > 0);

    // Same inputs, same pixels.
    let again = render_to_image(&mut app).expect("capture");
    assert_eq!(img, again);
}

#[test]
fn ui_frames_are_deterministic_across_apps() {
    let capture = || -> Option<ImageData> {
        let mut app = ui_app(160, 120)?;
        app.add_ui_system(window_system);
        app.step_ticks(4);
        render_to_image(&mut app)
    };
    let (Some(a), Some(b)) = (capture(), capture()) else {
        return;
    };
    assert_eq!(a, b);
}

#[test]
fn injected_click_reaches_button() {
    let Some(mut app) = ui_app(256, 192) else { return };
    app.add_ui_system(window_system);
    app.step_ticks(3);
    let rect = app.resources.expect::<Clicks>().button.expect("button laid out");
    let center = app.resources.expect::<EguiContext>().to_physical(rect.center());

    let input = app.resources.expect_mut::<Input>();
    input.inject_mouse_move(center);
    app.step_ticks(1);
    app.resources
        .expect_mut::<Input>()
        .inject_mouse_button_down(MouseButton::Left);
    app.step_ticks(1);
    assert!(app.resources.expect::<EguiContext>().wants_pointer_input());
    app.resources
        .expect_mut::<Input>()
        .inject_mouse_button_up(MouseButton::Left);
    app.step_ticks(1);
    assert_eq!(app.resources.expect::<Clicks>().count, 1);

    // A click outside the button does nothing.
    let input = app.resources.expect_mut::<Input>();
    input.inject_mouse_move(Vec2::new(230.0, 170.0));
    input.inject_mouse_button_down(MouseButton::Left);
    app.step_ticks(1);
    app.resources
        .expect_mut::<Input>()
        .inject_mouse_button_up(MouseButton::Left);
    app.step_ticks(1);
    assert_eq!(app.resources.expect::<Clicks>().count, 1);
    assert!(!app.resources.expect::<EguiContext>().is_pointer_over_ui());
}

#[test]
fn typed_text_reaches_focused_text_edit() {
    let mut app = App::new();
    app.add_plugin(UiPlugin::deterministic());
    app.init_resource::<String>();
    app.add_ui_system(|ctx, sys| {
        egui::Window::new("Text").fixed_pos([10.0, 10.0]).show(ctx, |ui| {
            let text = sys.resources.expect_mut::<String>();
            let r = ui.text_edit_singleline(text);
            if ctx.cumulative_pass_nr() == 1 {
                r.request_focus();
            }
        });
    });
    app.step_ticks(2);
    let input = app.resources.expect_mut::<Input>();
    input.inject_text("hi!");
    app.step_ticks(1);
    app.resources.expect_mut::<Input>().inject_key_tap(KeyCode::Backspace);
    app.step_ticks(1);
    assert_eq!(app.resources.expect::<String>(), "hi");
    assert!(app.resources.expect::<EguiContext>().wants_keyboard_input());
}

#[test]
fn debug_overlay_toggles_with_f3() {
    let Some(mut app) = ui_app(320, 320) else { return };
    app.step_ticks(2);
    let before = render_to_image(&mut app).expect("capture");
    assert_eq!(non_background(&before, 0, 0, 320, 320), 0, "no UI yet");
    app.resources.expect_mut::<Input>().inject_key_tap(KeyCode::F3);
    app.step_ticks(2);
    assert!(app.resources.expect::<DebugOverlay>().visible);
    let after = render_to_image(&mut app).expect("capture");
    assert!(non_background(&after, 0, 0, 240, 200) > 1000, "overlay drawn top-left");
    assert!(app.resources.expect::<DebugOverlay>().frame_times().count() > 0);
}

#[test]
fn profiler_times_systems() {
    let mut app = App::new();
    app.add_plugin(UiPlugin::deterministic());
    app.add_system_named(Stage::Update, "test::busy_system", |sys| {
        let mut x = 0u64;
        for i in 0..10_000u64 {
            x = x.wrapping_mul(31).wrapping_add(i);
        }
        sys.resources.insert(x);
    });
    app.step_ticks(5);
    let p = app.resources.expect::<Profiler>();
    let s = p.get("test::busy_system").expect("system timed");
    assert_eq!(s.calls, 1);
    assert!(s.total_calls >= 3);
    assert!(p.get("ui::begin_frame").is_some());
    assert!(!p.timeline().spans.is_empty());
    assert!(tracing_install_state() != TracingInstall::None);
}

#[test]
fn console_commands_and_log_capture() {
    let mut app = App::new();
    app.add_plugin(UiPlugin::deterministic());
    app.step_ticks(1);
    tracing::info!("hello-from-ui-test");
    app.resources.expect_mut::<Console>().submit("entities");
    app.step_ticks(1);
    let c = app.resources.expect::<Console>();
    assert!(c.lines().any(|l| l.message.contains("hello-from-ui-test")));
    assert!(
        c.lines()
            .any(|l| l.level == LogLevel::Output && l.message.ends_with("entities"))
    );
}

#[test]
fn render_target_is_shown_as_egui_image() {
    let Some(mut app) = ui_app(200, 200) else { return };
    // A red-clearing camera renders into a texture; the UI shows it at (10,10) 64x64.
    let target = app.resources.expect_mut::<RenderAssets>().add_render_target(64, 64);
    app.world.spawn((
        Transform::from_xyz(0.0, 0.0, 3.0),
        Camera3d::default()
            .with_clear_color(Vec4::new(1.0, 0.0, 0.0, 1.0))
            .with_target(RenderTarget::Texture(target)),
    ));
    let ctx = app.resources.expect::<EguiContext>().ctx().clone();
    let id = app
        .resources
        .expect_mut::<UiTextures>()
        .register_render_target(&ctx, target);
    app.add_ui_system(move |ctx, _| {
        egui::Area::new(egui::Id::new("rt"))
            .fixed_pos([10.0, 10.0])
            .show(ctx, |ui| ui.image((id, egui::vec2(64.0, 64.0))));
    });
    app.step_ticks(3);
    let img = render_to_image(&mut app).expect("capture");
    let px = img.get(40, 40);
    assert!(
        px[0] > 240 && px[1] < 10 && px[2] < 10,
        "render target visible in the UI: {px:?}"
    );
    assert_eq!(img.get(150, 150), BG);
}

#[test]
fn ui_without_renderer_or_window_still_runs() {
    let mut app = App::new();
    app.add_plugin(UiPlugin::default());
    let mut shown = 0;
    app.add_ui_system(move |ctx, sys| {
        egui::Window::new("w").show(ctx, |ui| ui.label("x"));
        shown += 1;
        sys.resources.insert(shown);
    });
    app.step_ticks(3);
    assert_eq!(*app.resources.expect::<i32>(), 3);
    app.resources.expect_mut::<EguiContext>().set_enabled(false);
    app.step_ticks(2);
    assert_eq!(*app.resources.expect::<i32>(), 3, "disabled UI skips ui systems");
}

/// App with one focused single-line text field whose content is the `String` resource.
fn text_field_app(initial: &str) -> App {
    let mut app = App::new();
    app.add_plugin(UiPlugin::deterministic());
    app.insert_resource(initial.to_owned());
    app.add_ui_system(|ctx, sys| {
        egui::Window::new("Text").fixed_pos([10.0, 10.0]).show(ctx, |ui| {
            let text = sys.resources.expect_mut::<String>();
            let r = ui.text_edit_singleline(text);
            if ctx.cumulative_pass_nr() == 1 {
                r.request_focus();
            }
        });
    });
    app.step_ticks(2);
    app
}

#[test]
fn held_backspace_repeats_in_text_fields() {
    let mut app = text_field_app("");
    let alphabet = "abcdefghijklmnopqrstuvwxyz";
    app.resources.expect_mut::<Input>().inject_text(alphabet);
    app.step_ticks(1);
    assert_eq!(app.resources.expect::<String>(), alphabet);
    // Hold Backspace for one second (60 ticks at the default 60 Hz): one press, then repeats
    // after the delay.
    app.resources.expect_mut::<Input>().inject_key_down(KeyCode::Backspace);
    app.step_ticks(60);
    app.resources.expect_mut::<Input>().inject_key_up(KeyCode::Backspace);
    app.step_ticks(1);
    let left = app.resources.expect::<String>().len();
    assert!(left <= 26 - 10, "holding Backspace deleted {} chars", 26 - left);
    assert!(left > 0, "repeat rate is bounded");
}

#[test]
fn console_key_types_into_focused_text_fields() {
    let mut app = text_field_app("");
    let tap_backquote = |app: &mut App| {
        let input = app.resources.expect_mut::<Input>();
        input.inject_key_tap(KeyCode::Backquote);
        input.inject_text("`");
        app.step_ticks(1);
    };
    tap_backquote(&mut app);
    assert_eq!(app.resources.expect::<String>(), "`", "typed into the field");
    assert!(
        !app.resources.expect::<Console>().visible,
        "console not toggled while typing"
    );
    // F3 is not a text key: it still works while typing.
    app.resources.expect_mut::<Input>().inject_key_tap(KeyCode::F3);
    app.step_ticks(1);
    assert!(app.resources.expect::<DebugOverlay>().visible);

    // Leave the field: the key toggles the console again.
    app.resources.expect_mut::<Input>().inject_key_tap(KeyCode::Escape);
    app.step_ticks(1);
    assert!(!app.resources.expect::<EguiContext>().wants_keyboard_input());
    tap_backquote(&mut app);
    assert!(app.resources.expect::<Console>().visible);
    // The console focuses its input; the key then closes it again.
    app.step_ticks(2);
    assert!(app.resources.expect::<Console>().input_has_focus());
    tap_backquote(&mut app);
    assert!(!app.resources.expect::<Console>().visible);
    assert_eq!(app.resources.expect::<String>(), "`", "field untouched meanwhile");
}

/// The platform's command modifier (Cmd on Apple platforms, Ctrl elsewhere).
const COMMAND: KeyCode = if cfg!(any(target_os = "macos", target_os = "ios")) {
    KeyCode::SuperLeft
} else {
    KeyCode::ControlLeft
};

fn ch(s: &str) -> LogicalKey {
    LogicalKey::Character(s.into())
}

/// Tap physical `key` typing `c` on the player's layout, one frame.
fn tap(app: &mut App, key: KeyCode, c: &str) {
    app.resources.expect_mut::<Input>().inject_key_tap_logical(key, ch(c));
    app.step_ticks(1);
}

/// Command + physical `key` (typing `c` on the layout), one frame each for down/tap/up.
fn command(app: &mut App, key: KeyCode, c: &str) {
    app.resources.expect_mut::<Input>().inject_key_down(COMMAND);
    app.step_ticks(1);
    tap(app, key, c);
    app.resources.expect_mut::<Input>().inject_key_up(COMMAND);
    app.step_ticks(1);
}

/// AZERTY: "azerty" is typed with QWERTY's Q W E R T Y positions.
const AZERTY_WORD: [(KeyCode, &str); 6] = [
    (KeyCode::KeyQ, "a"),
    (KeyCode::KeyW, "z"),
    (KeyCode::KeyE, "e"),
    (KeyCode::KeyR, "r"),
    (KeyCode::KeyT, "t"),
    (KeyCode::KeyY, "y"),
];

#[test]
fn azerty_typing_and_undo_follow_the_layout() {
    let mut app = text_field_app("");
    for (key, c) in AZERTY_WORD {
        tap(&mut app, key, c);
    }
    assert_eq!(app.resources.expect::<String>(), "azerty");
    // Let egui's undoer settle (new undo point after 1 s without changes).
    app.step_ticks(90);
    // Command + QWERTY's Z position types "w" on AZERTY: not undo.
    // Only where Command is the Command key: elsewhere it is Ctrl, and egui's Ctrl+W deletes the
    // word before the cursor (emacs binding), which is not what this step is about.
    if cfg!(any(target_os = "macos", target_os = "ios")) {
        command(&mut app, KeyCode::KeyZ, "w");
        assert_eq!(app.resources.expect::<String>(), "azerty", "Cmd+W is not undo");
    }
    // Command + the key labelled Z (QWERTY's W position): undo, as in native apps.
    command(&mut app, KeyCode::KeyW, "z");
    assert_eq!(app.resources.expect::<String>(), "", "Cmd+Z undid the typing");
    assert_eq!(
        app.resources.expect::<Input>().key_cap(KeyCode::KeyW),
        "Z",
        "key label learned from the events"
    );
}

#[test]
fn azerty_select_all_and_copy_follow_the_layout() {
    let mut app = text_field_app("");
    for (key, c) in AZERTY_WORD {
        tap(&mut app, key, c);
    }
    // Command + QWERTY's A position ("q" on AZERTY) does not select all...
    command(&mut app, KeyCode::KeyA, "q");
    command(&mut app, KeyCode::KeyC, "c");
    assert_eq!(app.resources.expect::<EguiContext>().clipboard(), "");
    // ...Command + the key labelled A (QWERTY's Q position) does.
    command(&mut app, KeyCode::KeyQ, "a");
    command(&mut app, KeyCode::KeyC, "c");
    assert_eq!(app.resources.expect::<EguiContext>().clipboard(), "azerty");
    assert_eq!(app.resources.expect::<String>(), "azerty", "shortcuts typed nothing");
}

#[test]
fn dvorak_paste_follows_the_layout() {
    let mut app = text_field_app("");
    app.resources.expect_mut::<EguiContext>().set_clipboard("pasted");
    // Dvorak: QWERTY's V position types "k"; "v" is on QWERTY's period position.
    command(&mut app, KeyCode::KeyV, "k");
    assert_eq!(app.resources.expect::<String>(), "");
    command(&mut app, KeyCode::Period, "v");
    assert_eq!(app.resources.expect::<String>(), "pasted");
}
