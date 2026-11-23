//! Input methods (IME) through the UI, headless: what the UI tells the platform about the text
//! field that has the focus (`TextInput`), and what a real window's `Ime` events do to it.
//!
//! The window itself (winit's `set_ime_allowed`, the OS's candidate list) cannot be driven
//! without a real input method: see "Input methods (IME)" in docs/platforms.md for the manual test.

use space3d_core::prelude::*;
use space3d_platform::{ImeEvent, Input, KeyCode, TextInput};
use space3d_ui::*;

/// A one-line field bound to the `String` resource, focused on the first pass (a password field
/// shows dots and must never get an input method).
fn field_app(password: bool) -> App {
    let mut app = App::new();
    app.add_plugin(UiPlugin::deterministic());
    app.insert_resource(String::new());
    app.add_ui_system(move |ctx, sys| {
        egui::Window::new("Text").fixed_pos([10.0, 10.0]).show(ctx, |ui| {
            let text = sys.resources.expect_mut::<String>();
            let r = ui.add(egui::TextEdit::singleline(text).password(password));
            if ctx.cumulative_pass_nr() == 1 {
                r.request_focus();
            }
        });
    });
    app.step_ticks(3);
    app
}

#[test]
fn the_ui_tells_the_platform_that_a_text_field_has_the_focus() {
    let app = field_app(false);
    let at = *app.resources.expect::<TextInput>();
    assert!(at.wanted, "a field has the focus");
    assert!(!at.password, "an ordinary one: the input method may be on");
}

#[test]
fn a_password_field_is_told() {
    let app = field_app(true);
    let at = *app.resources.expect::<TextInput>();
    assert!(at.wanted, "it is a text field (a phone's keyboard comes up)");
    assert!(at.password, "but no input method");
}

#[test]
fn leaving_a_password_field_clears_the_flags() {
    let mut app = field_app(true);
    assert!(app.resources.expect::<TextInput>().password);
    app.resources.expect_mut::<Input>().inject_key_tap(KeyCode::Escape);
    app.step_ticks(2);
    let at = *app.resources.expect::<TextInput>();
    assert!(!at.wanted && !at.password, "no field has the focus: {at:?}");
}

fn preedit(text: &str) -> ImeEvent {
    ImeEvent::Preedit {
        text: text.into(),
        cursor: None,
    }
}

/// One frame with these input method events.
fn frame(app: &mut App, events: impl IntoIterator<Item = ImeEvent>) {
    let input = app.resources.expect_mut::<Input>();
    for e in events {
        input.inject_ime(e);
    }
    app.step_ticks(1);
}

fn text(app: &App) -> String {
    app.resources.expect::<String>().clone()
}

#[test]
fn a_composition_is_shown_in_the_field_and_the_commit_replaces_it() {
    let mut app = field_app(false);
    // Windows' order: Enabled, Preedit..., Preedit(""), Commit, Disabled.
    frame(&mut app, [ImeEvent::Enabled, preedit("n")]);
    assert_eq!(text(&app), "n", "the preedit is in the field while it is composed");
    frame(&mut app, [preedit("ni")]);
    frame(&mut app, [preedit("nihao")]);
    assert_eq!(text(&app), "nihao");
    frame(
        &mut app,
        [preedit(""), ImeEvent::Commit("你好".into()), ImeEvent::Disabled],
    );
    assert_eq!(text(&app), "你好", "the commit once, no pinyin left");
    // The next word goes after it.
    frame(&mut app, [preedit("sh"), ImeEvent::Commit("世界".into())]);
    assert_eq!(text(&app), "你好世界");
}

#[test]
fn a_commit_is_entered_once_and_is_no_typed_text() {
    let mut app = field_app(false);
    frame(&mut app, [ImeEvent::Commit("你好".into())]);
    assert_eq!(text(&app), "你好", "a commit with no preedit before it works");
    // (the runner never also types it: `Input::text` is empty)
    assert_eq!(app.resources.expect::<Input>().text(), "");
    // The way macOS sends them: an empty preedit, then the commit.
    frame(&mut app, [preedit("a")]);
    frame(&mut app, [preedit(""), ImeEvent::Commit("啊".into())]);
    assert_eq!(text(&app), "你好啊");
}

#[test]
fn an_empty_preedit_ends_the_composition() {
    let mut app = field_app(false);
    frame(&mut app, [preedit("ni")]);
    assert_eq!(text(&app), "ni");
    frame(&mut app, [preedit("")]);
    assert_eq!(text(&app), "", "cancelled (Esc in the IME): the preedit is gone");
    // Typing goes on where it was; an empty preedit with no composition (Wayland sends them) is nothing.
    frame(&mut app, [ImeEvent::Commit("好".into()), preedit("")]);
    assert_eq!(text(&app), "好");
}

#[test]
fn disabled_takes_a_live_preedit_out_and_otherwise_does_nothing() {
    let mut app = field_app(false);
    frame(&mut app, [ImeEvent::Commit("好".into())]);
    frame(&mut app, [ImeEvent::Disabled]);
    assert_eq!(text(&app), "好", "no composition: nothing to clear");
    frame(&mut app, [preedit("ni")]);
    assert_eq!(text(&app), "好ni");
    frame(&mut app, [ImeEvent::Disabled]);
    assert_eq!(
        text(&app),
        "好",
        "the IME went away mid-composition: its text goes with it"
    );
}

#[test]
fn a_preedit_with_a_cursor_in_bytes_enters_the_field() {
    // winit counts the cursor of 你好 in bytes; the field takes the characters.
    let mut app = field_app(false);
    let winit = space3d_platform::winit::event::Ime::Preedit("你好".into(), Some((3, 6)));
    frame(&mut app, [ImeEvent::from_winit(winit)]);
    assert_eq!(text(&app), "你好");
}

#[test]
fn a_page_s_ime_events_enter_once_and_the_engine_adds_none() {
    // The web path: a game's hidden input makes egui's IME events itself (`push_event`), and the
    // engine's `Input` has none (winit's web backend sends no `Ime` event).
    let mut app = field_app(false);
    app.resources
        .expect_mut::<EguiContext>()
        .push_event(egui::Event::Ime(egui::ImeEvent::Preedit {
            text: "ni".into(),
            active_range_chars: None,
        }));
    app.step_ticks(1);
    assert_eq!(text(&app), "ni");
    assert!(app.resources.expect::<Input>().ime_events().is_empty());
    app.resources
        .expect_mut::<EguiContext>()
        .push_event(egui::Event::Ime(egui::ImeEvent::Commit("你".into())));
    app.step_ticks(2);
    assert_eq!(text(&app), "你", "entered once");
}

#[test]
fn typed_text_and_keys_are_what_they_were() {
    // The phones' keyboards and the key events with text of every platform: no input method
    // event is made of them and they enter as before.
    let mut app = field_app(false);
    app.resources.expect_mut::<Input>().inject_text("hi!");
    app.step_ticks(1);
    app.resources.expect_mut::<Input>().inject_key_tap(KeyCode::Backspace);
    app.step_ticks(1);
    assert_eq!(text(&app), "hi");
    assert!(app.resources.expect::<Input>().ime_events().is_empty());
}
