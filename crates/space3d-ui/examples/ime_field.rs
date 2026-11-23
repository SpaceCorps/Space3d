//! A window with a chat line and a password line, to try the OS input method (IME: Chinese,
//! Japanese and Korean typing) by hand, and a scripted run that shows the runner turning it on and
//! off in a real window. See "Input methods (IME)" in docs/platforms.md.
//!
//! ```text
//! cargo run -p space3d-ui --example ime_field                 # by hand: type pinyin into the chat
//! cargo run -p space3d-ui --example ime_field -- --frames 120 # scripted: focus moves, then exits
//! RUST_LOG=ime=debug ...                                      # logs every `set_ime_allowed`
//! SPACE3D_NO_IME=1 ...                                        # the kill switch
//! ```
//!
//! By hand: click into the chat, switch to your input method, type. The pinyin shows underlined in
//! the field and the candidate list opens at the caret. Click outside the fields (the IME goes off:
//! "W held" follows the key whatever mode the IME is in) and into the password field (no candidate
//! list, plain letters). The window lists what the input method sent.
//!
//! Scripted (`--frames N`, or `SPACE3D_EXIT_AFTER_FRAMES`): the window opens in the background (no
//! focus stolen), focus goes to the chat, then to the password field, then nowhere, and the run
//! ends with one line of what `TextInput` said at each step.

use space3d_core::prelude::*;
use space3d_platform::*;
use space3d_render::*;
use space3d_ui::*;

#[derive(Default)]
struct Fields {
    chat: String,
    password: String,
    /// What the input method sent, newest last.
    events: Vec<String>,
    /// `TextInput` as the scripted run saw it at each step.
    steps: Vec<String>,
    /// Which field to focus next (scripted run): 0 the chat, 1 the password, 2 none.
    focus: Option<u8>,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut max_frames: Option<u64> = std::env::var("SPACE3D_EXIT_AFTER_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok());
    while let Some(a) = args.next() {
        if a == "--frames" {
            max_frames = args.next().and_then(|v| v.parse().ok());
        }
    }
    if max_frames.is_some() {
        // A scripted run: don't steal focus from whatever the developer is doing.
        space3d_platform::set_background_mode(true);
    }

    let mut app = App::new();
    app.add_plugin(PlatformPlugin {
        window: WindowPlugin::titled("Space3d: input methods"),
    });
    app.add_plugin(UiPlugin::default());
    app.init_resource::<Fields>();
    app.add_system(Stage::Startup, |ctx| {
        ctx.world
            .spawn((Transform::from_xyz(0.0, 0.0, 3.0), Camera3d::default()));
        ctx.resources.expect_mut::<Fields>().focus = Some(0);
    });
    app.add_ui_system(ui);
    app.add_system(Stage::Last, move |ctx| {
        let frame = ctx.time().frame;
        let events: Vec<String> = ctx
            .resources
            .expect::<Input>()
            .ime_events()
            .iter()
            .map(|e| format!("{e:?}"))
            .collect();
        let text = ctx.resources.get::<TextInput>().copied().unwrap_or_default();
        let fields = ctx.resources.expect_mut::<Fields>();
        fields.events.extend(events);
        let keep = fields.events.len().saturating_sub(12);
        fields.events.drain(..keep);
        let Some(max) = max_frames else { return };
        // The script: the chat has the focus from the start, the password field from a third of
        // the run, nothing from two thirds. What `TextInput` says is noted a few frames later.
        let at = |n: u64| frame == max * n / 3;
        if at(1) {
            fields.focus = Some(1);
        } else if at(2) {
            fields.focus = Some(2);
        }
        if [max / 3 - 3, 2 * max / 3 - 3, max - 3].contains(&frame) {
            fields.steps.push(format!(
                "frame {frame}: wanted={} password={} caret={}",
                text.wanted,
                text.password,
                text.caret
                    .map_or("none".into(), |c| format!("{},{} {}x{}", c.x, c.y, c.width, c.height))
            ));
        }
        if frame >= max {
            for s in &fields.steps {
                println!("{s}");
            }
            ctx.send(AppExit);
        }
    });
    app.run();
}

fn ui(ctx: &egui::Context, sys: &mut SystemCtx) {
    let w_held = sys.resources.expect::<Input>().pressed(KeyCode::KeyW);
    let text = sys.resources.get::<TextInput>().copied().unwrap_or_default();
    let fields = sys.resources.expect_mut::<Fields>();
    let focus = fields.focus.take();
    egui::Window::new("Input methods")
        .default_pos([20.0, 20.0])
        .show(ctx, |ui| {
            ui.label("Chat (the input method is on while it has the focus):");
            let chat = ui.add(egui::TextEdit::singleline(&mut fields.chat).hint_text("type here, nihao"));
            ui.label("Password (never an input method):");
            let password = ui.add(egui::TextEdit::singleline(&mut fields.password).password(true));
            match focus {
                Some(0) => chat.request_focus(),
                Some(1) => password.request_focus(),
                Some(_) => ui.memory_mut(|m| {
                    if let Some(id) = m.focused() {
                        m.surrender_focus(id);
                    }
                }),
                None => {}
            }
            ui.separator();
            ui.label(format!(
                "W held: {w_held}   (click outside the fields; the IME is off there)"
            ));
            ui.label(format!(
                "TextInput: wanted={} password={} caret={:?}",
                text.wanted, text.password, text.caret
            ));
            ui.label("What the input method sent:");
            for e in &fields.events {
                ui.monospace(e);
            }
        });
}
