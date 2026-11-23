//! Input methods (IME) through the UI, headless: what the UI tells the platform about the text
//! field that has the focus (`TextInput`), and what a real window's `Ime` events do to it.
//!
//! The window itself (winit's `set_ime_allowed`, the OS's candidate list) cannot be driven
//! without a real input method: see "Input methods (IME)" in docs/platforms.md for the manual test.

use space3d_core::prelude::*;
use space3d_platform::{Input, KeyCode, TextInput};
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
