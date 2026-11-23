//! Translation of `space3d_platform::Input` state into egui events.
//!
//! `Input` is state-based (held keys + per-frame edges), so events are reconstructed from the
//! edges every frame: pointer moves, button presses/releases (a press and release in the same
//! frame become a click), wheel, keys, typed text and touches. Because tests drive the same
//! `Input` resource through `inject_*`, UI interaction is fully scriptable.
//!
//! Keys follow the player's keyboard layout like egui-winit: `Event::Key::key` is the
//! **logical** key of the press ([`egui_logical_key`] of winit's logical key, then of the
//! modifier-free key) and `physical_key` the physical one ([`egui_key`]). When the layout's key
//! has no egui equivalent (non-Latin layouts, AZERTY's digit row) or no logical key was
//! recorded (legacy `inject_key_*` events), the physical key is used for both — so `Ctrl/Cmd+Z`
//! undoes with the key labelled Z (physical `KeyW` on AZERTY), and on a Cyrillic layout with the
//! key in Z's position. A key's release reuses the egui key of its press. Hotkeys read from
//! egui (`key_pressed(egui::Key::M)`) therefore fire on the key labelled M; to compare them
//! with physical-key bindings, translate with [`physical_key_for_egui_key`].
//!
//! **Pointer.** Mouse buttons are sent in the order they happened
//! ([`Input::mouse_button_events`]), each where the cursor was at the time: egui's pointer first
//! moves there (so a release after a move beyond egui's click distance is no click), then the
//! button event; after the frame's buttons the pointer moves to where the cursor is now. A fast
//! double click's release and second press in one frame therefore leave the button down, as in
//! `Input`. A button event while the cursor is outside the window (the release of a drag that
//! left it) is sent at the last position egui saw, and the pointer is gone again right after
//! (`PointerGone`), so it re-enters no hover: nothing at (0, 0) reacts.
//!
//! **Focus.** `Event::WindowFocused` is sent when the window's focus changes; egui then forgets
//! the keys and modifiers it holds.
//!
//! **Cancelled releases.** A release that no device event reported — the window lost the
//! focus ([`Input::release_all`]) or the OS shows the button up although its release never
//! arrived ([`space3d_platform::MouseButtonCause`]) — cancels the press, like switching away
//! from a native app mid-drag: egui registers **no click and no drop**. The pointer first
//! detours far off-screen (beyond any click distance, so egui's press can no longer be a click),
//! comes back, and the button is released where the cursor is: drags end (`drag_stopped`), a
//! dragged slider stays at its value. The drag-and-drop payload is cleared before the pass, so
//! no drop zone receives it. Any button egui still holds that `Input` does not (egui was
//! disabled meanwhile, `Input` was replaced) is released the same way. A system-cancelled touch
//! ([`space3d_platform::TouchPhase::Cancelled`]) is a cancelled release of the primary button,
//! not a tap. A cancel only concerns its own button: it is ordered ahead of the other buttons'
//! clicks of its frame, so it cannot end one of them. A release outside the window is no click
//! and drops nothing either (egui's pointer detours off-screen first, as for a cancel).
//!
//! Game code that decides drops or "release over X" from egui (`Response::drag_stopped`,
//! `hovered()` / `contains_pointer()` with `any_released()`, ...) sees a cancelled release as an ordinary one, since
//! egui knows no other: ask [`pointer_cancelled_this_frame`] /
//! [`pointer_cancel_cause`] (or the same on `EguiContext`) whether this frame's release was a
//! cancel.
//!
//! **No pointer position yet.** A button event before any pointer position is known (nothing
//! moved the cursor into the window yet, e.g. a test that presses first) is dropped rather than
//! sent to (0, 0): a press is ignored and so is its release, whenever it comes. Buttons work as
//! soon as one position was seen.
//!
//! `Input` drops the OS's key repeats (a held key is simply "pressed"), but egui text fields
//! only act on key *presses*, so held editing/navigation keys ([`REPEAT_KEYS`]: Backspace,
//! Delete, arrows, Home/End, ...) get synthesized repeat presses: after [`REPEAT_DELAY`]
//! seconds, one every [`REPEAT_INTERVAL`]. The timing uses `space3d_core::Time`, so it is
//! deterministic under `App::step_ticks`.

use space3d_core::glam::Vec2;
use space3d_platform::{
    ImeEvent, Input, KeyCode, LogicalKey, MouseButton, MouseButtonCause, MouseButtonEvent, NamedKey,
    is_writing_system_key,
};

/// Map a logical (layout) key to an egui key — the mapping egui-winit uses: characters by
/// [`egui::Key::from_name`] (`"z"` → `Z`, `"?"` → `Questionmark`), named keys (Enter, arrows,
/// F-keys, ...) directly; `None` for dead keys, modifiers and characters egui has no key for
/// (`"é"`, `"я"`).
pub fn egui_logical_key(key: &LogicalKey) -> Option<egui::Key> {
    match key {
        LogicalKey::Named(named) => egui_named_key(*named),
        LogicalKey::Character(s) => egui::Key::from_name(s.as_str()),
        LogicalKey::Unidentified(_) | LogicalKey::Dead(_) => None,
    }
}

fn egui_named_key(key: NamedKey) -> Option<egui::Key> {
    use NamedKey as N;
    use egui::Key as E;
    Some(match key {
        N::Enter => E::Enter,
        N::Tab => E::Tab,
        N::ArrowDown => E::ArrowDown,
        N::ArrowLeft => E::ArrowLeft,
        N::ArrowRight => E::ArrowRight,
        N::ArrowUp => E::ArrowUp,
        N::End => E::End,
        N::Home => E::Home,
        N::PageDown => E::PageDown,
        N::PageUp => E::PageUp,
        N::Backspace => E::Backspace,
        N::Delete => E::Delete,
        N::Insert => E::Insert,
        N::Escape => E::Escape,
        N::Cut => E::Cut,
        N::Copy => E::Copy,
        N::Paste => E::Paste,
        N::Space => E::Space,
        N::F1 => E::F1,
        N::F2 => E::F2,
        N::F3 => E::F3,
        N::F4 => E::F4,
        N::F5 => E::F5,
        N::F6 => E::F6,
        N::F7 => E::F7,
        N::F8 => E::F8,
        N::F9 => E::F9,
        N::F10 => E::F10,
        N::F11 => E::F11,
        N::F12 => E::F12,
        N::F13 => E::F13,
        N::F14 => E::F14,
        N::F15 => E::F15,
        N::F16 => E::F16,
        N::F17 => E::F17,
        N::F18 => E::F18,
        N::F19 => E::F19,
        N::F20 => E::F20,
        N::F21 => E::F21,
        N::F22 => E::F22,
        N::F23 => E::F23,
        N::F24 => E::F24,
        N::F25 => E::F25,
        N::F26 => E::F26,
        N::F27 => E::F27,
        N::F28 => E::F28,
        N::F29 => E::F29,
        N::F30 => E::F30,
        N::F31 => E::F31,
        N::F32 => E::F32,
        N::F33 => E::F33,
        N::F34 => E::F34,
        N::F35 => E::F35,
        N::BrowserBack => E::BrowserBack,
        _ => return None,
    })
}

/// The egui key for a press of physical `code`: the logical key of the frame's last press
/// event of `code` (winit's logical key first, like egui-winit, then the modifier-free key —
/// which still names the layout's letter when Option/AltGr changed the character), else the
/// physical key.
fn press_key(input: &Input, code: KeyCode) -> Option<egui::Key> {
    let event = input
        .key_events()
        .iter()
        .rev()
        .find(|e| e.physical == code && e.pressed);
    event
        .and_then(|e| {
            egui_logical_key(&e.logical_key).or_else(|| e.key_without_modifiers.as_ref().and_then(egui_logical_key))
        })
        .or_else(|| egui_key(code))
}

/// The physical key that sends `key` to egui on the player's current layout when pressed
/// without modifiers — the inverse of the translation above. Relates egui key reads
/// (`i.key_pressed(egui::Key::M)`) to physical-key bindings ([`space3d_platform::Binding::Key`],
/// [`Input::pressed`]): on AZERTY egui's `M` is `KeyCode::Semicolon`, on Dvorak `G` is
/// `KeyCode::KeyU`, on QWERTY and QWERTZ it is `KeyCode::KeyM`.
///
/// It is the writing-system key whose layout label ([`Input::key_label`]) egui reads as
/// `key` (the first in keyboard order when several are); else the physical key [`egui_key`]
/// maps to `key`, unless the layout makes that key send another egui key (a label egui has no
/// key for, such as Cyrillic `я` in Z's position, still sends the physical key's). `None` when
/// the layout moved `key` to a position whose label is still unknown (no OS table, key not
/// pressed yet), or no key sends it.
///
/// Cheap enough for every frame: it walks the layout's cached reverse map
/// ([`KeyboardLayout::key_for_label_matching`](space3d_platform::KeyboardLayout::key_for_label_matching)),
/// which is rebuilt only when the labels change, and does not allocate otherwise.
pub fn physical_key_for_egui_key(input: &Input, key: egui::Key) -> Option<KeyCode> {
    // What egui reads for a typed character: `egui_logical_key(&LogicalKey::Character(label))`.
    let sends = egui::Key::from_name;
    let layout = input.keyboard_layout();
    if let Some(code) = layout.key_for_label_matching(|label| sends(label) == Some(key)) {
        return Some(code);
    }
    KEYS.iter().copied().find(|&code| {
        egui_key(code) == Some(key)
            && (!is_writing_system_key(code) || layout.label_text(code).and_then(sends).is_none())
    })
}

/// Map a physical key to the egui key it produces (US layout for punctuation).
pub fn egui_key(code: KeyCode) -> Option<egui::Key> {
    use KeyCode as K;
    use egui::Key as E;
    Some(match code {
        K::ArrowDown => E::ArrowDown,
        K::ArrowLeft => E::ArrowLeft,
        K::ArrowRight => E::ArrowRight,
        K::ArrowUp => E::ArrowUp,
        K::Escape => E::Escape,
        K::Tab => E::Tab,
        K::Backspace => E::Backspace,
        K::Enter | K::NumpadEnter => E::Enter,
        K::Space => E::Space,
        K::Insert => E::Insert,
        K::Delete => E::Delete,
        K::Home => E::Home,
        K::End => E::End,
        K::PageUp => E::PageUp,
        K::PageDown => E::PageDown,
        K::Copy => E::Copy,
        K::Cut => E::Cut,
        K::Paste => E::Paste,
        K::Comma => E::Comma,
        K::Backslash => E::Backslash,
        // The ISO key right of left Shift (`<` on AZERTY/QWERTZ), as egui-winit maps it.
        K::IntlBackslash => E::IntlBackslash,
        K::Slash | K::NumpadDivide => E::Slash,
        K::BracketLeft => E::OpenBracket,
        K::BracketRight => E::CloseBracket,
        K::Backquote => E::Backtick,
        K::Minus | K::NumpadSubtract => E::Minus,
        K::Period | K::NumpadDecimal => E::Period,
        K::NumpadAdd => E::Plus,
        K::Equal | K::NumpadEqual => E::Equals,
        K::Semicolon => E::Semicolon,
        K::Quote => E::Quote,
        K::Digit0 | K::Numpad0 => E::Num0,
        K::Digit1 | K::Numpad1 => E::Num1,
        K::Digit2 | K::Numpad2 => E::Num2,
        K::Digit3 | K::Numpad3 => E::Num3,
        K::Digit4 | K::Numpad4 => E::Num4,
        K::Digit5 | K::Numpad5 => E::Num5,
        K::Digit6 | K::Numpad6 => E::Num6,
        K::Digit7 | K::Numpad7 => E::Num7,
        K::Digit8 | K::Numpad8 => E::Num8,
        K::Digit9 | K::Numpad9 => E::Num9,
        K::KeyA => E::A,
        K::KeyB => E::B,
        K::KeyC => E::C,
        K::KeyD => E::D,
        K::KeyE => E::E,
        K::KeyF => E::F,
        K::KeyG => E::G,
        K::KeyH => E::H,
        K::KeyI => E::I,
        K::KeyJ => E::J,
        K::KeyK => E::K,
        K::KeyL => E::L,
        K::KeyM => E::M,
        K::KeyN => E::N,
        K::KeyO => E::O,
        K::KeyP => E::P,
        K::KeyQ => E::Q,
        K::KeyR => E::R,
        K::KeyS => E::S,
        K::KeyT => E::T,
        K::KeyU => E::U,
        K::KeyV => E::V,
        K::KeyW => E::W,
        K::KeyX => E::X,
        K::KeyY => E::Y,
        K::KeyZ => E::Z,
        K::F1 => E::F1,
        K::F2 => E::F2,
        K::F3 => E::F3,
        K::F4 => E::F4,
        K::F5 => E::F5,
        K::F6 => E::F6,
        K::F7 => E::F7,
        K::F8 => E::F8,
        K::F9 => E::F9,
        K::F10 => E::F10,
        K::F11 => E::F11,
        K::F12 => E::F12,
        K::BrowserBack => E::BrowserBack,
        K::ShiftLeft => E::ShiftLeft,
        K::ShiftRight => E::ShiftRight,
        K::ControlLeft => E::ControlLeft,
        K::ControlRight => E::ControlRight,
        K::AltLeft => E::AltLeft,
        K::AltRight => E::AltRight,
        K::SuperLeft => E::SuperLeft,
        K::SuperRight => E::SuperRight,
        _ => return None,
    })
}

/// Every key [`egui_key`] maps (scanned for edges each frame).
const KEYS: &[KeyCode] = {
    use KeyCode as K;
    &[
        K::ArrowDown,
        K::ArrowLeft,
        K::ArrowRight,
        K::ArrowUp,
        K::Escape,
        K::Tab,
        K::Backspace,
        K::Enter,
        K::NumpadEnter,
        K::Space,
        K::Insert,
        K::Delete,
        K::Home,
        K::End,
        K::PageUp,
        K::PageDown,
        K::Copy,
        K::Cut,
        K::Paste,
        K::Comma,
        K::Backslash,
        K::IntlBackslash,
        K::Slash,
        K::NumpadDivide,
        K::BracketLeft,
        K::BracketRight,
        K::Backquote,
        K::Minus,
        K::NumpadSubtract,
        K::Period,
        K::NumpadDecimal,
        K::NumpadAdd,
        K::Equal,
        K::NumpadEqual,
        K::Semicolon,
        K::Quote,
        K::Digit0,
        K::Digit1,
        K::Digit2,
        K::Digit3,
        K::Digit4,
        K::Digit5,
        K::Digit6,
        K::Digit7,
        K::Digit8,
        K::Digit9,
        K::Numpad0,
        K::Numpad1,
        K::Numpad2,
        K::Numpad3,
        K::Numpad4,
        K::Numpad5,
        K::Numpad6,
        K::Numpad7,
        K::Numpad8,
        K::Numpad9,
        K::KeyA,
        K::KeyB,
        K::KeyC,
        K::KeyD,
        K::KeyE,
        K::KeyF,
        K::KeyG,
        K::KeyH,
        K::KeyI,
        K::KeyJ,
        K::KeyK,
        K::KeyL,
        K::KeyM,
        K::KeyN,
        K::KeyO,
        K::KeyP,
        K::KeyQ,
        K::KeyR,
        K::KeyS,
        K::KeyT,
        K::KeyU,
        K::KeyV,
        K::KeyW,
        K::KeyX,
        K::KeyY,
        K::KeyZ,
        K::F1,
        K::F2,
        K::F3,
        K::F4,
        K::F5,
        K::F6,
        K::F7,
        K::F8,
        K::F9,
        K::F10,
        K::F11,
        K::F12,
        K::BrowserBack,
        K::ShiftLeft,
        K::ShiftRight,
        K::ControlLeft,
        K::ControlRight,
        K::AltLeft,
        K::AltRight,
        K::SuperLeft,
        K::SuperRight,
    ]
};

/// Held keys that auto-repeat (editing and navigation keys; printable characters repeat
/// through the OS text events instead).
pub const REPEAT_KEYS: &[KeyCode] = {
    use KeyCode as K;
    &[
        K::Backspace,
        K::Delete,
        K::ArrowLeft,
        K::ArrowRight,
        K::ArrowUp,
        K::ArrowDown,
        K::Home,
        K::End,
        K::PageUp,
        K::PageDown,
        K::Enter,
        K::NumpadEnter,
        K::Tab,
    ]
};
/// Seconds a key must be held before it starts repeating.
pub const REPEAT_DELAY: f64 = 0.5;
/// Seconds between synthesized repeats.
pub const REPEAT_INTERVAL: f64 = 1.0 / 30.0;
/// Upper bound on repeats emitted per key and frame (long frames do not burst).
const MAX_REPEATS_PER_FRAME: u32 = 4;

const BUTTONS: [(MouseButton, egui::PointerButton); 5] = [
    (MouseButton::Left, egui::PointerButton::Primary),
    (MouseButton::Right, egui::PointerButton::Secondary),
    (MouseButton::Middle, egui::PointerButton::Middle),
    (MouseButton::Back, egui::PointerButton::Extra1),
    (MouseButton::Forward, egui::PointerButton::Extra2),
];

/// Where egui's pointer detours before a cancelled release: far off-screen, beyond any click
/// distance from a press, so the release that follows is no click.
const CANCEL_DETOUR: egui::Pos2 = egui::pos2(-1.0e6, -1.0e6);

/// The pointer releases of the last egui pass that were cancels, by button (kept in egui's
/// temporary data by `ui::begin_frame`, so code that only has an [`egui::Context`] can read
/// them: [`pointer_cancel_cause`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PointerCancels([Option<MouseButtonCause>; BUTTONS.len()]);

fn pointer_cancels_id() -> egui::Id {
    egui::Id::new("space3d_ui::pointer_cancels")
}

/// Record which releases of the pass that begins now are cancels.
pub(crate) fn publish_pointer_cancels(ctx: &egui::Context, cancels: [Option<MouseButtonCause>; BUTTONS.len()]) {
    ctx.data_mut(|d| d.insert_temp(pointer_cancels_id(), PointerCancels(cancels)));
}

/// Why `button`'s release in this frame's egui pass was cancelled ([`MouseButtonCause`]:
/// [`FocusLost`](MouseButtonCause::FocusLost) or [`OsReconciled`](MouseButtonCause::OsReconciled)),
/// or `None`: it was not released this frame, or it was released by the device.
///
/// A cancelled release is registered by egui as no click and no drop, but it still ends the
/// press: `Response::drag_stopped`, `PointerState::button_released` and `any_released` are
/// `true` in that frame. Logic that acts on the end of a drag or press should check this and
/// undo instead:
///
/// ```no_run
/// # use space3d_ui::{egui, pointer_cancelled_this_frame};
/// # fn drop_item_here() {}
/// # fn ui(ctx: &egui::Context, zone: &egui::Response) {
/// if zone.contains_pointer() && ctx.input(|i| i.pointer.any_released()) && !pointer_cancelled_this_frame(ctx) {
///     drop_item_here();
/// }
/// # }
/// ```
///
/// Valid from `Stage::PreUpdate` (`ui::begin_frame`) of the frame until the next one's; a
/// system-cancelled touch counts as the primary button, cause `FocusLost` (the app lost the
/// finger to the system). Nothing is cancelled without a running `UiPlugin`.
pub fn pointer_cancel_cause(ctx: &egui::Context, button: egui::PointerButton) -> Option<MouseButtonCause> {
    let cancels = ctx.data(|d| d.get_temp::<PointerCancels>(pointer_cancels_id()))?;
    cancels.0[button as usize]
}

/// Whether any pointer button's release in this frame's egui pass was a cancel
/// ([`pointer_cancel_cause`]).
pub fn pointer_cancelled_this_frame(ctx: &egui::Context) -> bool {
    ctx.data(|d| d.get_temp::<PointerCancels>(pointer_cancels_id()))
        .is_some_and(|c| c.0.iter().any(Option::is_some))
}

/// Modifier state derived from held keys.
pub(crate) fn modifiers(input: &Input) -> egui::Modifiers {
    use KeyCode as K;
    let shift = input.any_pressed([K::ShiftLeft, K::ShiftRight]);
    let ctrl = input.any_pressed([K::ControlLeft, K::ControlRight]);
    let alt = input.any_pressed([K::AltLeft, K::AltRight]);
    let sup = input.any_pressed([K::SuperLeft, K::SuperRight]);
    let mac = cfg!(any(target_os = "macos", target_os = "ios"));
    egui::Modifiers {
        alt,
        ctrl,
        shift,
        mac_cmd: mac && sup,
        command: if mac {
            sup
        } else if cfg!(target_arch = "wasm32") {
            ctrl || sup
        } else {
            ctrl
        },
    }
}

fn printable(c: char) -> bool {
    let private_use = ('\u{e000}'..='\u{f8ff}').contains(&c)
        || ('\u{f0000}'..='\u{ffffd}').contains(&c)
        || ('\u{100000}'..='\u{10fffd}').contains(&c);
    !private_use && !c.is_control()
}

/// Per-frame translator (keeps the pointer position, buttons, focus, modifiers and key-repeat
/// timers egui was told about).
#[derive(Default)]
pub(crate) struct InputTranslator {
    /// egui's pointer position as last sent (`None`: gone, `PointerGone` sent).
    last_pos: Option<egui::Pos2>,
    /// The last position sent, kept through `PointerGone`: where button events outside the
    /// window go.
    known_pos: Option<egui::Pos2>,
    /// Mouse buttons egui holds ([`BUTTONS`] order).
    down: [bool; BUTTONS.len()],
    /// Buttons whose press was dropped because no pointer position was known yet: their release
    /// is dropped too ([`BUTTONS`] order).
    ignored: [bool; BUTTONS.len()],
    /// Buttons whose release was cancelled in the last translated frame, and why ([`BUTTONS`]
    /// order; the primary one also for a cancelled pointer touch).
    cancels: [Option<MouseButtonCause>; BUTTONS.len()],
    /// The order the last frame's button events went to egui (indices into
    /// `Input::mouse_button_events`; reused, so a steady state allocates nothing).
    order: Vec<usize>,
    /// egui was told the window has no focus (egui starts focused).
    unfocused: bool,
    last_modifiers: egui::Modifiers,
    /// Touch that emulates the mouse pointer (first finger down).
    pointer_touch: Option<u64>,
    /// Held [`REPEAT_KEYS`] and the time of their next synthesized repeat.
    repeats: Vec<(KeyCode, f64)>,
    /// Held keys and the egui key their press was sent as (the release and repeats reuse it).
    held: Vec<(KeyCode, egui::Key)>,
    /// An input method's composition is going on as far as egui was told: a preedit with text
    /// was sent and no commit, empty preedit or `Disabled` since.
    ime_composing: bool,
}

impl InputTranslator {
    /// `Event::WindowFocused` if the window's focus changed since egui was last told.
    pub(crate) fn focus(&mut self, focused: bool, out: &mut Vec<egui::Event>) {
        if focused == self.unfocused {
            self.unfocused = !focused;
            out.push(egui::Event::WindowFocused(focused));
        }
    }

    /// An input method's events as egui's `Event::Ime`, which a focused `TextEdit` takes: a
    /// preedit with text shows it (underlined, in the bound string) and replaces the one before;
    /// a commit replaces the preedit with the finished text. An empty preedit, a commit and a
    /// `Disabled` end a composition. egui 0.36 ignores its deprecated `Enabled`/`Disabled` and
    /// ignores an empty preedit when no composition is going on, so what is sent is only what
    /// it follows: a `Disabled` that finds a composition becomes an empty preedit (the text the
    /// IME was composing is taken out of the field), one that does not is nothing.
    fn ime(&mut self, events: &[ImeEvent], out: &mut Vec<egui::Event>) {
        for event in events {
            match event {
                ImeEvent::Enabled => {}
                ImeEvent::Preedit { text, cursor } if !text.is_empty() => {
                    // (the cursor is in characters, within the text: a script's event may not be)
                    let chars = text.chars().count();
                    let active = cursor.as_ref().map(|c| c.start.min(chars)..c.end.min(chars));
                    self.ime_composing = true;
                    out.push(egui::Event::Ime(egui::ImeEvent::Preedit {
                        text: text.clone(),
                        active_range_chars: active.map(|r| r.start.min(r.end)..r.end),
                    }));
                }
                ImeEvent::Commit(text) => {
                    // (a line break is the Enter key's, not text an input method commits)
                    let text: String = text.chars().filter(|c| !c.is_control()).collect();
                    if text.is_empty() {
                        self.end_ime_composition(out);
                    } else {
                        // egui takes the preedit out itself, then enters the text
                        self.ime_composing = false;
                        out.push(egui::Event::Ime(egui::ImeEvent::Commit(text)));
                    }
                }
                ImeEvent::Preedit { .. } | ImeEvent::Disabled => self.end_ime_composition(out),
            }
        }
    }

    /// An empty preedit if a composition is going on (egui takes its text out), else nothing.
    fn end_ime_composition(&mut self, out: &mut Vec<egui::Event>) {
        if std::mem::take(&mut self.ime_composing) {
            out.push(egui::Event::Ime(egui::ImeEvent::Preedit {
                text: String::new(),
                active_range_chars: None,
            }));
        }
    }

    /// Append this frame's egui events for `input` (physical pixels → points with `ppp`).
    /// `time` is the app time in seconds (drives key repeat). Returns whether egui's
    /// drag-and-drop payload must go, undelivered: a release was cancelled (see the module
    /// docs) or a button came up outside the window. [`Self::cancelled_buttons`] tells which
    /// releases were cancels.
    pub(crate) fn translate(
        &mut self,
        input: &Input,
        ppp: f32,
        time: f64,
        clipboard: &str,
        out: &mut Vec<egui::Event>,
    ) -> bool {
        let to_points = |p: Vec2| egui::pos2(p.x / ppp, p.y / ppp);
        let mods = modifiers(input);
        if mods != self.last_modifiers {
            out.push(egui::Event::ModifiersChanged(mods));
            self.last_modifiers = mods;
        }

        // Pointer: the frame's button events in order (cancels early: `button_order`), each
        // where the cursor was; then where the cursor is now.
        let pos = input.mouse_position().map(to_points);
        let mut cancelled = false;
        self.cancels = [None; BUTTONS.len()];
        let events = input.mouse_button_events();
        self.button_order(events);
        for k in 0..self.order.len() {
            let e = &events[self.order[k]];
            let Some(i) = BUTTONS.iter().position(|(b, _)| *b == e.button) else {
                continue;
            };
            let at = e.position.map(to_points);
            if e.pressed {
                self.button_press(i, at, mods, out);
            } else {
                let cause = e.cancelled().then_some(e.cause);
                cancelled |= self.button_release(i, cause, at, mods, out);
            }
        }
        // A button egui holds that `Input` does not (egui was disabled when it came up, `Input`
        // was replaced): released as a cancel.
        for (i, (b, _)) in BUTTONS.iter().enumerate() {
            if self.down[i] && !input.mouse_pressed(*b) {
                self.detour(out);
                cancelled = true;
                self.cancels[i] = Some(MouseButtonCause::FocusLost);
                self.pointer_button(i, false, pos, mods, out);
            }
        }
        match pos {
            Some(p) if self.last_pos != Some(p) => {
                out.push(egui::Event::PointerMoved(p));
                self.last_pos = Some(p);
                self.known_pos = Some(p);
            }
            None if self.last_pos.is_some() && self.pointer_touch.is_none() => {
                out.push(egui::Event::PointerGone);
                self.last_pos = None;
            }
            _ => {}
        }
        let delta = input.mouse_delta();
        if delta != Vec2::ZERO {
            out.push(egui::Event::MouseMoved(egui::vec2(delta.x / ppp, delta.y / ppp)));
        }

        // Wheel (Input reports lines).
        let wheel = input.wheel_delta();
        if wheel != Vec2::ZERO {
            out.push(egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Line,
                delta: egui::vec2(wheel.x, wheel.y),
                phase: egui::TouchPhase::Move,
                modifiers: mods,
            });
        }

        // The input method (desktops: `Input::ime_events`), before the keys: a commit and the
        // Enter that follows it in one frame must enter the text first.
        self.ime(input.ime_events(), out);

        // Keys (+ clipboard shortcuts): the physical keys egui knows, then any other key that
        // produced an event this frame (its logical key may still be one egui knows).
        let mut extra: Vec<KeyCode> = Vec::new();
        for e in input.key_events() {
            if !KEYS.contains(&e.physical) && !extra.contains(&e.physical) {
                extra.push(e.physical);
            }
        }
        for code in KEYS.iter().copied().chain(extra) {
            let (down, up) = (input.just_pressed(code), input.just_released(code));
            if !down && !up {
                continue;
            }
            let physical_key = egui_key(code);
            let held = self.held.iter().position(|(k, _)| *k == code);
            let key = if down {
                let key = press_key(input, code);
                if let Some(i) = held {
                    self.held.remove(i);
                }
                if let Some(key) = key
                    && !up
                {
                    self.held.push((code, key));
                }
                key
            } else {
                held.map(|i| self.held.remove(i).1).or_else(|| press_key(input, code))
            };
            let Some(key) = key else { continue };
            if down && mods.command {
                match key {
                    egui::Key::C => {
                        out.push(egui::Event::Copy);
                        continue;
                    }
                    egui::Key::X => {
                        out.push(egui::Event::Cut);
                        continue;
                    }
                    egui::Key::V => {
                        if !clipboard.is_empty() {
                            out.push(egui::Event::Paste(clipboard.to_owned()));
                        }
                        continue;
                    }
                    _ => {}
                }
            }
            for pressed in [true, false] {
                if (pressed && down) || (!pressed && up) {
                    out.push(egui::Event::Key {
                        key,
                        physical_key,
                        pressed,
                        repeat: false,
                        modifiers: mods,
                    });
                }
            }
        }

        self.key_repeats(input, time, mods, out);

        // Typed text (not while a command shortcut is held).
        let text: String = input.text().chars().filter(|&c| printable(c)).collect();
        if !text.is_empty() && !(mods.command || (mods.ctrl && !mods.alt)) {
            out.push(egui::Event::Text(text));
        }

        // Touches (the first finger also drives the pointer, like egui-winit).
        let device_id = egui::TouchDeviceId(0);
        for &id in input.just_started_touches() {
            if let Some(t) = input.touch(id) {
                let p = to_points(t.position);
                out.push(egui::Event::Touch {
                    device_id,
                    id: egui::TouchId(id),
                    phase: egui::TouchPhase::Start,
                    pos: p,
                    force: None,
                });
                if self.pointer_touch.is_none() && input.mouse_position().is_none() {
                    self.pointer_touch = Some(id);
                    out.push(egui::Event::PointerMoved(p));
                    out.push(egui::Event::PointerButton {
                        pos: p,
                        button: egui::PointerButton::Primary,
                        pressed: true,
                        modifiers: mods,
                    });
                    self.last_pos = Some(p);
                    self.known_pos = Some(p);
                }
            }
        }
        for t in input.touches() {
            if t.delta != Vec2::ZERO && !input.just_started_touches().contains(&t.id) {
                let p = to_points(t.position);
                out.push(egui::Event::Touch {
                    device_id,
                    id: egui::TouchId(t.id),
                    phase: egui::TouchPhase::Move,
                    pos: p,
                    force: None,
                });
                if self.pointer_touch == Some(t.id) {
                    out.push(egui::Event::PointerMoved(p));
                    self.last_pos = Some(p);
                    self.known_pos = Some(p);
                }
            }
        }
        for t in input.just_ended_touches() {
            let p = to_points(t.position);
            // A touch the system cancelled is no tap: the pointer's release is a cancel.
            let cancel = input.just_cancelled_touches().iter().any(|c| c.id == t.id);
            out.push(egui::Event::Touch {
                device_id,
                id: egui::TouchId(t.id),
                phase: if cancel {
                    egui::TouchPhase::Cancel
                } else {
                    egui::TouchPhase::End
                },
                pos: p,
                force: None,
            });
            if self.pointer_touch == Some(t.id) {
                self.pointer_touch = None;
                if cancel {
                    self.detour(out);
                    cancelled = true;
                    self.cancels[0] = Some(MouseButtonCause::FocusLost);
                }
                out.push(egui::Event::PointerButton {
                    pos: p,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: mods,
                });
                out.push(egui::Event::PointerGone);
                self.last_pos = None;
                self.known_pos = Some(p);
            }
        }
        cancelled
    }
}

impl InputTranslator {
    /// The buttons whose release the last [`Self::translate`] sent as a cancel, and why
    /// ([`BUTTONS`] order: primary, secondary, middle, extra 1, extra 2).
    pub(crate) fn cancelled_buttons(&self) -> [Option<MouseButtonCause>; BUTTONS.len()] {
        self.cancels
    }

    /// Sets [`Self::order`], the order this frame's button events go to egui: `Input`'s order,
    /// except that a **cancel goes as early as it can**, so it does not land inside another
    /// button's click. egui follows one press at a time and any release ends the click in
    /// progress, so a cancel between another button's press and release would swallow that
    /// click. A cancel therefore moves before the presses and clicks of other buttons in its
    /// frame, but stays after its own button's earlier events, after other cancels, and after
    /// the release that closes a click begun in an earlier frame (that click is complete when
    /// the cancel comes: egui's one shared press is over by then, so it judges the cancelled
    /// release alone, and may count it as a click, which the detour cannot prevent).
    fn button_order(&mut self, events: &[MouseButtonEvent]) {
        self.order.clear();
        self.order.extend(0..events.len());
        if !events.iter().any(MouseButtonEvent::cancelled) {
            return;
        }
        // The first event of a button egui holds is its release: it closes an earlier click.
        let down = self.down;
        let closes_a_click = |j: usize| {
            let e = &events[j];
            !e.pressed
                && BUTTONS
                    .iter()
                    .position(|(b, _)| *b == e.button)
                    .is_some_and(|i| down[i])
                && events[..j].iter().all(|o| o.button != e.button)
        };
        let order = &mut self.order;
        for (k, e) in events.iter().enumerate() {
            if !e.cancelled() {
                continue;
            }
            let from = order.iter().position(|&o| o == k).unwrap_or(k);
            let mut to = from;
            while to > 0 {
                let before = order[to - 1];
                let o = &events[before];
                if o.button == e.button || o.cancelled() || closes_a_click(before) {
                    break;
                }
                to -= 1;
            }
            order[to..=from].rotate_right(1);
        }
    }

    /// A button press at `at` (see [`Self::pointer_button`]). With no pointer position known
    /// yet (`at` is `None`, and egui has never seen one) the press is dropped, and its release
    /// with it: nothing can tell what was pressed, and (0, 0) is not it.
    fn button_press(
        &mut self,
        i: usize,
        at: Option<egui::Pos2>,
        modifiers: egui::Modifiers,
        out: &mut Vec<egui::Event>,
    ) {
        if at.is_none() && self.known_pos.is_none() {
            self.ignored[i] = true;
            return;
        }
        self.ignored[i] = false;
        self.pointer_button(i, true, at, modifiers, out);
    }

    /// A button release at `at`; `cause`: why it was cancelled (no click, no drop). A release
    /// outside the window (`at` is `None`) is no click on a widget at the window's edge either: egui
    /// only sees the pointer where it last was, which may still be over such a widget. Both detour
    /// egui's pointer off-screen first. Returns whether egui's drag-and-drop payload must go:
    /// a cancel, or a release outside the window (nothing is dropped there).
    fn button_release(
        &mut self,
        i: usize,
        cause: Option<MouseButtonCause>,
        at: Option<egui::Pos2>,
        modifiers: egui::Modifiers,
        out: &mut Vec<egui::Event>,
    ) -> bool {
        if std::mem::take(&mut self.ignored[i]) || (at.is_none() && self.known_pos.is_none()) {
            return false;
        }
        let drops_nothing = cause.is_some() || at.is_none();
        if drops_nothing {
            self.detour(out);
        }
        self.cancels[i] = self.cancels[i].or(cause);
        self.pointer_button(i, false, at, modifiers, out);
        drops_nothing
    }

    /// A button event at `at` (points; `None`: the cursor is outside the window). Inside, egui's
    /// pointer moves there first. Outside, the event goes to the last position egui saw and the
    /// pointer is gone again right after, so it re-enters no hover.
    fn pointer_button(
        &mut self,
        i: usize,
        pressed: bool,
        at: Option<egui::Pos2>,
        modifiers: egui::Modifiers,
        out: &mut Vec<egui::Event>,
    ) {
        let button = BUTTONS[i].1;
        match at {
            Some(pos) => {
                if self.last_pos != Some(pos) {
                    out.push(egui::Event::PointerMoved(pos));
                }
                out.push(egui::Event::PointerButton {
                    pos,
                    button,
                    pressed,
                    modifiers,
                });
                self.last_pos = Some(pos);
                self.known_pos = Some(pos);
            }
            None => {
                out.push(egui::Event::PointerButton {
                    pos: self.known_pos.unwrap_or(egui::Pos2::ZERO), // callers drop it before
                    button,
                    pressed,
                    modifiers,
                });
                out.push(egui::Event::PointerGone);
                self.last_pos = None;
            }
        }
        self.down[i] = pressed;
    }

    /// Before a cancelled release: egui's pointer moves far off-screen (not remembered as a
    /// known position), so egui's press has moved beyond the click distance and the release
    /// that follows is no click.
    fn detour(&mut self, out: &mut Vec<egui::Event>) {
        if self.last_pos != Some(CANCEL_DETOUR) {
            out.push(egui::Event::PointerMoved(CANCEL_DETOUR));
            self.last_pos = Some(CANCEL_DETOUR);
        }
    }

    /// Synthesized repeat presses for held [`REPEAT_KEYS`].
    fn key_repeats(&mut self, input: &Input, time: f64, mods: egui::Modifiers, out: &mut Vec<egui::Event>) {
        self.repeats
            .retain(|(k, _)| input.pressed(*k) && !input.just_pressed(*k));
        for &code in REPEAT_KEYS {
            if !input.pressed(code) {
                continue;
            }
            let physical_key = egui_key(code);
            let held = self.held.iter().find(|(k, _)| *k == code).map(|h| h.1);
            let Some(key) = held.or(physical_key) else { continue };
            let Some(i) = self.repeats.iter().position(|(k, _)| *k == code) else {
                // Newly held (or held before the UI started): repeat after the delay.
                self.repeats.push((code, time + REPEAT_DELAY));
                continue;
            };
            let next = &mut self.repeats[i].1;
            let mut n = 0;
            while time >= *next && n < MAX_REPEATS_PER_FRAME {
                out.push(egui::Event::Key {
                    key,
                    physical_key,
                    pressed: true,
                    repeat: true,
                    modifiers: mods,
                });
                *next += REPEAT_INTERVAL;
                n += 1;
            }
            if time >= *next {
                // Too far behind (long frame): skip the backlog instead of bursting.
                *next = time + REPEAT_INTERVAL;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use space3d_platform::NamedKey;

    fn repeats(ev: &[egui::Event], key: egui::Key) -> usize {
        ev.iter()
            .filter(|e| matches!(e, egui::Event::Key { key: k, pressed: true, repeat: true, .. } if *k == key))
            .count()
    }

    #[test]
    fn held_editing_keys_repeat_after_a_delay() {
        let mut t = InputTranslator::default();
        let mut input = Input::default();
        input.inject_key_down(KeyCode::Backspace);
        input.inject_key_down(KeyCode::KeyA);
        let mut ev = Vec::new();
        t.translate(&input, 1.0, 10.0, "", &mut ev);
        assert_eq!(repeats(&ev, egui::Key::Backspace), 0, "first frame: only the press");
        assert!(ev.iter().any(|e| matches!(
            e,
            egui::Event::Key {
                key: egui::Key::Backspace,
                pressed: true,
                repeat: false,
                ..
            }
        )));
        let mut total = 0;
        let mut frame_time = 10.0;
        for _ in 0..60 {
            input.end_frame();
            frame_time += 1.0 / 60.0;
            ev.clear();
            t.translate(&input, 1.0, frame_time, "", &mut ev);
            if frame_time - 10.0 < REPEAT_DELAY - 1e-9 {
                assert_eq!(repeats(&ev, egui::Key::Backspace), 0, "no repeat before the delay");
            }
            total += repeats(&ev, egui::Key::Backspace);
            assert_eq!(repeats(&ev, egui::Key::A), 0, "letters repeat through text events");
        }
        // 1 s held: repeats during the last 0.5 s at 30 Hz.
        assert!((13..=17).contains(&total), "{total} repeats");
        // A long frame does not burst.
        input.end_frame();
        ev.clear();
        t.translate(&input, 1.0, frame_time + 5.0, "", &mut ev);
        assert!(repeats(&ev, egui::Key::Backspace) <= MAX_REPEATS_PER_FRAME as usize);
        // Released: no more repeats; pressed again: delay starts over.
        input.end_frame();
        input.inject_key_up(KeyCode::Backspace);
        ev.clear();
        t.translate(&input, 1.0, frame_time + 6.0, "", &mut ev);
        assert_eq!(repeats(&ev, egui::Key::Backspace), 0);
        input.end_frame();
        input.inject_key_down(KeyCode::Backspace);
        ev.clear();
        t.translate(&input, 1.0, frame_time + 7.0, "", &mut ev);
        input.end_frame();
        ev.clear();
        t.translate(&input, 1.0, frame_time + 7.1, "", &mut ev);
        assert_eq!(repeats(&ev, egui::Key::Backspace), 0);
    }

    fn ch(s: &str) -> LogicalKey {
        LogicalKey::Character(s.into())
    }

    /// `(key, physical_key, pressed)` of every `Event::Key` in `ev`.
    fn keys(ev: &[egui::Event]) -> Vec<(egui::Key, Option<egui::Key>, bool)> {
        ev.iter()
            .filter_map(|e| match e {
                egui::Event::Key {
                    key,
                    physical_key,
                    pressed,
                    ..
                } => Some((*key, *physical_key, *pressed)),
                _ => None,
            })
            .collect()
    }

    fn translate_frame(t: &mut InputTranslator, input: &mut Input, clipboard: &str) -> Vec<egui::Event> {
        let mut ev = Vec::new();
        t.translate(input, 1.0, 0.0, clipboard, &mut ev);
        input.end_frame();
        ev
    }

    #[test]
    fn logical_key_mapping_matches_egui_winit() {
        use egui::Key as E;
        assert_eq!(egui_logical_key(&ch("z")), Some(E::Z));
        assert_eq!(egui_logical_key(&ch("Z")), Some(E::Z));
        assert_eq!(egui_logical_key(&ch("?")), Some(E::Questionmark));
        assert_eq!(egui_logical_key(&ch(":")), Some(E::Colon));
        assert_eq!(egui_logical_key(&ch("+")), Some(E::Plus));
        assert_eq!(egui_logical_key(&ch("é")), None);
        assert_eq!(egui_logical_key(&ch("я")), None);
        assert_eq!(egui_logical_key(&LogicalKey::Named(NamedKey::Enter)), Some(E::Enter));
        assert_eq!(egui_logical_key(&LogicalKey::Named(NamedKey::F5)), Some(E::F5));
        assert_eq!(egui_logical_key(&LogicalKey::Named(NamedKey::Shift)), None);
        assert_eq!(egui_logical_key(&LogicalKey::Dead(Some('^'))), None);
    }

    #[test]
    fn egui_gets_the_logical_key_and_the_physical_key() {
        use egui::Key as E;
        let mut t = InputTranslator::default();
        let mut input = Input::default();
        // (physical key, modifier-free key, logical key if different, egui key, egui physical key)
        type Case = (KeyCode, LogicalKey, Option<LogicalKey>, E, Option<E>);
        let cases: [Case; 11] = [
            // AZERTY: QWERTY's W position is labelled Z.
            (KeyCode::KeyW, ch("z"), None, E::Z, Some(E::W)),
            // QWERTZ: QWERTY's Y position is labelled Z.
            (KeyCode::KeyY, ch("z"), None, E::Z, Some(E::Y)),
            // Dvorak: QWERTY's period position types "v".
            (KeyCode::Period, ch("v"), None, E::V, Some(E::Period)),
            // JIS: QWERTY's quote position types ":".
            (KeyCode::Quote, ch(":"), None, E::Colon, Some(E::Quote)),
            // AZERTY digit row types "&": egui has no such key, the physical key is used.
            (KeyCode::Digit1, ch("&"), None, E::Num1, Some(E::Num1)),
            // Cyrillic: non-Latin letters fall back to the physical key (Ctrl+Z still works).
            (KeyCode::KeyZ, ch("я"), None, E::Z, Some(E::Z)),
            // US Shift+/: the shifted logical key, like egui-winit.
            (KeyCode::Slash, ch("/"), Some(ch("?")), E::Questionmark, Some(E::Slash)),
            // macOS AZERTY Option+(labelled W): logical "Ω", the modifier-free key names W.
            (KeyCode::KeyZ, ch("w"), Some(ch("Ω")), E::W, Some(E::Z)),
            // A key egui has no physical key for, whose character it knows (JIS ろ).
            (KeyCode::IntlRo, ch("\\"), None, E::Backslash, None),
            // The ISO key right of left Shift: "`" on US/British Macs, "<" (no egui key: the
            // physical key, like egui-winit) on French/German keyboards.
            (
                KeyCode::IntlBackslash,
                ch("`"),
                None,
                E::Backtick,
                Some(E::IntlBackslash),
            ),
            (
                KeyCode::IntlBackslash,
                ch("<"),
                None,
                E::IntlBackslash,
                Some(E::IntlBackslash),
            ),
        ];
        for (code, key, logical, want, want_physical) in cases {
            let mut press = space3d_platform::KeyEvent::press(code, key.clone());
            if let Some(l) = logical {
                press = press.with_logical_key(l);
            }
            input.inject_key_event(press);
            input.inject_key_up_logical(code, key);
            let ev = translate_frame(&mut t, &mut input, "");
            assert_eq!(
                keys(&ev),
                vec![(want, want_physical, true), (want, want_physical, false)],
                "{code:?}"
            );
        }
        // Legacy injection (no logical key): physical for both.
        input.inject_key_tap(KeyCode::KeyW);
        let ev = translate_frame(&mut t, &mut input, "");
        assert_eq!(keys(&ev), vec![(E::W, Some(E::W), true), (E::W, Some(E::W), false)]);
    }

    #[test]
    fn release_matches_its_press() {
        let mut t = InputTranslator::default();
        let mut input = Input::default();
        // US: Shift, then "/" pressed as "?"; Shift released before the key.
        input.inject_key_down(KeyCode::ShiftLeft);
        input.inject_key_event(space3d_platform::KeyEvent::press(KeyCode::Slash, ch("/")).with_logical_key(ch("?")));
        let ev = translate_frame(&mut t, &mut input, "");
        assert_eq!(
            keys(&ev),
            vec![
                (egui::Key::Questionmark, Some(egui::Key::Slash), true),
                (egui::Key::ShiftLeft, Some(egui::Key::ShiftLeft), true)
            ]
        );
        input.inject_key_up(KeyCode::ShiftLeft);
        translate_frame(&mut t, &mut input, "");
        input.inject_key_up_logical(KeyCode::Slash, ch("/"));
        let ev = translate_frame(&mut t, &mut input, "");
        assert_eq!(
            keys(&ev),
            vec![(egui::Key::Questionmark, Some(egui::Key::Slash), false)],
            "egui sees the key it saw pressed go up"
        );
    }

    #[test]
    fn clipboard_shortcuts_follow_the_layout() {
        let cmd = if cfg!(any(target_os = "macos", target_os = "ios")) {
            KeyCode::SuperLeft
        } else {
            KeyCode::ControlLeft
        };
        let mut t = InputTranslator::default();
        let mut input = Input::default();
        input.inject_key_down(cmd);
        // Dvorak: "v" is on QWERTY's period position; QWERTY's V position types "k".
        input.inject_key_tap_logical(KeyCode::Period, ch("v"));
        let ev = translate_frame(&mut t, &mut input, "clip");
        assert!(ev.contains(&egui::Event::Paste("clip".into())), "{ev:?}");
        assert!(
            !ev.iter().any(|e| matches!(e, egui::Event::Text(_))),
            "no text under Cmd"
        );
        input.inject_key_tap_logical(KeyCode::KeyV, ch("k"));
        let ev = translate_frame(&mut t, &mut input, "clip");
        assert!(!ev.iter().any(|e| matches!(e, egui::Event::Paste(_))));
        assert_eq!(keys(&ev)[0].0, egui::Key::K);
        // AZERTY: "c" and "x" keep their QWERTY positions.
        input.inject_key_tap_logical(KeyCode::KeyC, ch("c"));
        input.inject_key_tap_logical(KeyCode::KeyX, ch("x"));
        let ev = translate_frame(&mut t, &mut input, "");
        assert!(ev.contains(&egui::Event::Copy) && ev.contains(&egui::Event::Cut));
    }

    #[test]
    fn physical_key_for_egui_key_inverts_the_translation() {
        use egui::Key as E;
        type Layout = &'static [(KeyCode, &'static str)];
        type Expected = &'static [(E, Option<KeyCode>)];
        // (layout, keys taught, egui key → expected physical key)
        let cases: [(&str, Layout, Expected); 8] = [
            (
                "unknown (US assumed)",
                &[],
                &[(E::M, Some(KeyCode::KeyM)), (E::G, Some(KeyCode::KeyG))],
            ),
            (
                "AZERTY",
                &[(KeyCode::Semicolon, "m"), (KeyCode::KeyM, ","), (KeyCode::KeyW, "z")],
                &[
                    (E::M, Some(KeyCode::Semicolon)),
                    (E::Comma, Some(KeyCode::KeyM)),
                    (E::Z, Some(KeyCode::KeyW)),
                    (E::L, Some(KeyCode::KeyL)),
                ],
            ),
            (
                "AZERTY, M's position not seen yet",
                &[(KeyCode::KeyM, ",")],
                &[(E::M, None)],
            ),
            (
                "Dvorak",
                &[
                    (KeyCode::KeyU, "g"),
                    (KeyCode::KeyP, "l"),
                    (KeyCode::KeyG, "i"),
                    (KeyCode::KeyL, "n"),
                ],
                &[(E::G, Some(KeyCode::KeyU)), (E::L, Some(KeyCode::KeyP))],
            ),
            (
                "Cyrillic",
                &[(KeyCode::KeyM, "ь"), (KeyCode::KeyZ, "я")],
                &[(E::M, Some(KeyCode::KeyM)), (E::Z, Some(KeyCode::KeyZ))],
            ),
            (
                "US on an ISO Mac: ` right of left Shift",
                &[(KeyCode::Backquote, "§"), (KeyCode::IntlBackslash, "`")],
                &[(E::Backtick, Some(KeyCode::IntlBackslash))],
            ),
            (
                "German on an ISO Mac",
                &[(KeyCode::Backquote, "^"), (KeyCode::IntlBackslash, "<")],
                &[
                    (E::IntlBackslash, Some(KeyCode::IntlBackslash)),
                    (E::Backtick, Some(KeyCode::Backquote)),
                ],
            ),
            (
                "named keys",
                &[],
                &[
                    (E::Enter, Some(KeyCode::Enter)),
                    (E::Escape, Some(KeyCode::Escape)),
                    (E::F1, Some(KeyCode::F1)),
                ],
            ),
        ];
        for (name, taught, expected) in cases {
            let mut input = Input::default();
            for &(code, c) in taught {
                input.inject_key_tap_logical(code, ch(c));
            }
            input.end_frame();
            for &(key, want) in expected {
                assert_eq!(physical_key_for_egui_key(&input, key), want, "{name}: {key:?}");
                // Pressing that key sends `key` to egui.
                let Some(code) = want else { continue };
                let label = input.key_label(code).filter(|_| is_writing_system_key(code));
                let mut t = InputTranslator::default();
                match label {
                    Some(l) => input.inject_key_tap_logical(code, ch(&l)),
                    None => input.inject_key_tap(code),
                }
                let ev = translate_frame(&mut t, &mut input, "");
                assert_eq!(keys(&ev)[0].0, key, "{name}: {code:?} sends {key:?}");
            }
        }
    }

    #[test]
    fn keys_table_is_fully_mapped() {
        for &k in KEYS {
            assert!(egui_key(k).is_some(), "{k:?} unmapped");
        }
        assert_eq!(egui_key(KeyCode::KeyA), Some(egui::Key::A));
        assert_eq!(egui_key(KeyCode::CapsLock), None);
    }

    #[test]
    fn mouse_click_text_and_wheel() {
        let mut t = InputTranslator::default();
        let mut input = Input::default();
        input.inject_mouse_move(Vec2::new(20.0, 40.0));
        input.inject_mouse_button_down(MouseButton::Left);
        input.inject_mouse_button_up(MouseButton::Left);
        input.inject_text("hi\u{8}");
        input.inject_wheel(Vec2::new(0.0, -1.0));
        let mut ev = Vec::new();
        t.translate(&input, 2.0, 0.0, "", &mut ev);
        assert!(ev.contains(&egui::Event::PointerMoved(egui::pos2(10.0, 20.0))));
        let presses: Vec<bool> = ev
            .iter()
            .filter_map(|e| match e {
                egui::Event::PointerButton { pressed, .. } => Some(*pressed),
                _ => None,
            })
            .collect();
        assert_eq!(presses, vec![true, false], "tap = press then release");
        assert!(ev.contains(&egui::Event::Text("hi".into())), "control chars filtered");
        assert!(ev.iter().any(|e| matches!(e, egui::Event::MouseWheel { .. })));
        // Unchanged position: no new move event.
        input.end_frame();
        ev.clear();
        t.translate(&input, 2.0, 0.0, "", &mut ev);
        assert!(ev.is_empty(), "{ev:?}");
        input.inject_mouse_left();
        t.translate(&input, 2.0, 0.0, "", &mut ev);
        assert_eq!(ev, vec![egui::Event::PointerGone]);
    }

    /// The pointer events of `ev`: moves, gone, `(button, pressed)` at a position.
    #[derive(Debug, PartialEq)]
    enum P {
        Moved(egui::Pos2),
        Gone,
        Button(egui::PointerButton, bool, egui::Pos2),
        Focused(bool),
    }

    fn pointer(ev: &[egui::Event]) -> Vec<P> {
        ev.iter()
            .filter_map(|e| match *e {
                egui::Event::PointerMoved(p) => Some(P::Moved(p)),
                egui::Event::PointerGone => Some(P::Gone),
                egui::Event::PointerButton {
                    pos, button, pressed, ..
                } => Some(P::Button(button, pressed, pos)),
                egui::Event::WindowFocused(f) => Some(P::Focused(f)),
                _ => None,
            })
            .collect()
    }

    const PRIMARY: egui::PointerButton = egui::PointerButton::Primary;

    #[test]
    fn button_events_are_sent_in_order_where_they_happened() {
        let mut t = InputTranslator::default();
        let mut input = Input::default();
        let (a, b, c) = (egui::pos2(10.0, 10.0), egui::pos2(40.0, 10.0), egui::pos2(60.0, 30.0));
        let at = |p: egui::Pos2| Vec2::new(p.x, p.y);
        input.inject_mouse_move(at(a));
        input.inject_mouse_button_down(MouseButton::Left);
        let ev = translate_frame(&mut t, &mut input, "");
        assert_eq!(pointer(&ev), [P::Moved(a), P::Button(PRIMARY, true, a)]);
        // Released after a move, pressed again (double click), then dragged on: all in one frame.
        input.inject_mouse_move(at(b));
        input.inject_mouse_button_up(MouseButton::Left);
        input.inject_mouse_button_down(MouseButton::Left);
        input.inject_mouse_move(at(c));
        let ev = translate_frame(&mut t, &mut input, "");
        assert_eq!(
            pointer(&ev),
            [
                P::Moved(b),
                P::Button(PRIMARY, false, b),
                P::Button(PRIMARY, true, b),
                P::Moved(c),
            ]
        );
        assert!(t.down[0], "egui holds the button, like Input");
        // A tap within one frame.
        input.inject_mouse_button_up(MouseButton::Left);
        let ev = translate_frame(&mut t, &mut input, "");
        assert_eq!(pointer(&ev), [P::Button(PRIMARY, false, c)]);
        input.inject_mouse_button_down(MouseButton::Right);
        input.inject_mouse_button_up(MouseButton::Right);
        let ev = translate_frame(&mut t, &mut input, "");
        let secondary = egui::PointerButton::Secondary;
        assert_eq!(
            pointer(&ev),
            [P::Button(secondary, true, c), P::Button(secondary, false, c)]
        );
    }

    #[test]
    fn buttons_outside_the_window_go_to_the_last_known_position() {
        let mut t = InputTranslator::default();
        let mut input = Input::default();
        let a = egui::pos2(5.0, 7.0);
        input.inject_mouse_move(Vec2::new(10.0, 14.0));
        input.inject_mouse_button_down(MouseButton::Left);
        let mut ev = Vec::new();
        t.translate(&input, 2.0, 0.0, "", &mut ev);
        input.end_frame();
        assert_eq!(pointer(&ev), [P::Moved(a), P::Button(PRIMARY, true, a)]);
        input.inject_mouse_left();
        let mut ev = Vec::new();
        t.translate(&input, 2.0, 0.0, "", &mut ev);
        input.end_frame();
        assert_eq!(pointer(&ev), [P::Gone]);
        // Released outside: where egui last saw the pointer, which is gone again right after,
        // and never a click there (egui's pointer detours first): not a cancel either.
        input.inject_mouse_button_up(MouseButton::Left);
        let mut ev = Vec::new();
        assert!(t.translate(&input, 2.0, 0.0, "", &mut ev), "drops what is dragged");
        assert_eq!(t.cancelled_buttons(), [None; 5], "a device release");
        input.end_frame();
        assert_eq!(
            pointer(&ev),
            [P::Moved(CANCEL_DETOUR), P::Button(PRIMARY, false, a), P::Gone]
        );
        let mut ev = Vec::new();
        t.translate(&input, 2.0, 0.0, "", &mut ev);
        assert!(ev.is_empty(), "{ev:?}");
        // Back inside.
        input.inject_mouse_move(Vec2::new(20.0, 20.0));
        let mut ev = Vec::new();
        t.translate(&input, 2.0, 0.0, "", &mut ev);
        assert_eq!(pointer(&ev), [P::Moved(egui::pos2(10.0, 10.0))]);
    }

    #[test]
    fn cancelled_releases_detour_off_screen_first() {
        for cause in [MouseButtonCause::FocusLost, MouseButtonCause::OsReconciled] {
            let mut t = InputTranslator::default();
            let mut input = Input::default();
            let a = egui::pos2(30.0, 40.0);
            input.inject_mouse_move(Vec2::new(a.x, a.y));
            input.inject_mouse_button_down(MouseButton::Left);
            input.inject_mouse_button_down(MouseButton::Middle);
            translate_frame(&mut t, &mut input, "");
            if cause == MouseButtonCause::FocusLost {
                input.release_all();
            } else {
                input.release_mouse_button(MouseButton::Left, cause);
                input.release_mouse_button(MouseButton::Middle, cause);
            }
            let mut ev = Vec::new();
            assert!(t.translate(&input, 1.0, 0.0, "", &mut ev), "{cause:?}: a cancel");
            input.end_frame();
            let middle = egui::PointerButton::Middle;
            assert_eq!(
                pointer(&ev),
                [
                    P::Moved(CANCEL_DETOUR),
                    P::Moved(a),
                    P::Button(PRIMARY, false, a),
                    P::Moved(CANCEL_DETOUR),
                    P::Moved(a),
                    P::Button(middle, false, a),
                ],
                "{cause:?}"
            );
            // Then nothing: the pointer is where it was.
            let mut ev = Vec::new();
            assert!(!t.translate(&input, 1.0, 0.0, "", &mut ev));
            assert!(ev.is_empty(), "{ev:?}");
        }
        // The cursor outside: released at the last known position, gone again after.
        let mut t = InputTranslator::default();
        let mut input = Input::default();
        input.inject_mouse_move(Vec2::new(3.0, 4.0));
        input.inject_mouse_button_down(MouseButton::Left);
        input.inject_mouse_left();
        translate_frame(&mut t, &mut input, "");
        input.release_all();
        let ev = translate_frame(&mut t, &mut input, "");
        assert_eq!(
            pointer(&ev),
            [
                P::Moved(CANCEL_DETOUR),
                P::Button(PRIMARY, false, egui::pos2(3.0, 4.0)),
                P::Gone
            ]
        );
        // Device releases inside the window are not cancels.
        input.inject_mouse_move(Vec2::new(3.0, 4.0));
        input.inject_mouse_button_down(MouseButton::Left);
        input.inject_mouse_button_up(MouseButton::Left);
        let mut ev = Vec::new();
        assert!(!t.translate(&input, 1.0, 0.0, "", &mut ev));
        assert!(!ev.contains(&egui::Event::PointerMoved(CANCEL_DETOUR)));
        assert_eq!(t.cancelled_buttons(), [None; 5]);
    }

    /// Which buttons a frame cancelled is told apart (a cancel of one is no cancel of another),
    /// and the flags are those of the last translated frame only.
    #[test]
    fn cancelled_buttons_tell_which_button() {
        let mut t = InputTranslator::default();
        let mut input = Input::default();
        input.inject_mouse_move(Vec2::new(30.0, 40.0));
        input.inject_mouse_button_down(MouseButton::Left);
        input.inject_mouse_button_down(MouseButton::Right);
        input.inject_mouse_button_down(MouseButton::Middle);
        translate_frame(&mut t, &mut input, "");
        assert_eq!(t.cancelled_buttons(), [None; 5]);
        input.release_mouse_button(MouseButton::Right, MouseButtonCause::OsReconciled);
        input.inject_mouse_button_up(MouseButton::Left);
        let mut ev = Vec::new();
        assert!(t.translate(&input, 1.0, 0.0, "", &mut ev));
        input.end_frame();
        assert_eq!(
            t.cancelled_buttons(),
            [None, Some(MouseButtonCause::OsReconciled), None, None, None]
        );
        // The next frame: the middle button by a focus loss.
        input.release_all();
        translate_frame(&mut t, &mut input, "");
        assert_eq!(
            t.cancelled_buttons(),
            [None, None, Some(MouseButtonCause::FocusLost), None, None]
        );
        translate_frame(&mut t, &mut input, "");
        assert_eq!(t.cancelled_buttons(), [None; 5], "nothing this frame");
        // A cancelled pointer touch is the primary button's.
        let mut input2 = Input::default();
        input2.inject_touch(1, space3d_platform::TouchPhase::Started, Vec2::new(5.0, 5.0));
        let mut t2 = InputTranslator::default();
        translate_frame(&mut t2, &mut input2, "");
        input2.inject_touch(1, space3d_platform::TouchPhase::Cancelled, Vec2::new(5.0, 5.0));
        let mut ev = Vec::new();
        assert!(t2.translate(&input2, 1.0, 0.0, "", &mut ev));
        assert_eq!(
            t2.cancelled_buttons(),
            [Some(MouseButtonCause::FocusLost), None, None, None, None]
        );
        assert!(ev.iter().any(|e| matches!(
            e,
            egui::Event::Touch {
                phase: egui::TouchPhase::Cancel,
                ..
            }
        )));
    }

    /// A cancel goes ahead of other buttons' clicks in its frame but keeps its place after its
    /// own button's events, other cancels, and the release that closes an earlier click.
    #[test]
    fn a_cancel_is_ordered_before_other_buttons_clicks() {
        let (l, r, m) = (MouseButton::Left, MouseButton::Right, MouseButton::Middle);
        let order = |t: &mut InputTranslator, input: &Input| {
            t.button_order(input.mouse_button_events());
            t.order.clone()
        };
        let cancel = MouseButtonCause::OsReconciled;
        // Left held from an earlier frame; right clicks; left cancelled (index 2 of 3).
        let mut t = InputTranslator::default();
        let mut input = Input::default();
        input.inject_mouse_move(Vec2::new(10.0, 10.0));
        input.inject_mouse_button_down(l);
        translate_frame(&mut t, &mut input, "");
        input.inject_mouse_button_down(r);
        input.inject_mouse_button_up(r);
        input.release_mouse_button(l, cancel);
        assert_eq!(order(&mut t, &input), [2, 0, 1], "cancel first");
        // Its own button's earlier events stay before it.
        let mut t = InputTranslator::default();
        let mut input = Input::default();
        input.inject_mouse_move(Vec2::new(10.0, 10.0));
        input.inject_mouse_button_down(m);
        input.inject_mouse_button_down(l);
        input.inject_mouse_button_down(r);
        input.release_mouse_button(l, cancel);
        input.inject_mouse_button_up(r);
        assert_eq!(order(&mut t, &input), [0, 1, 3, 2, 4], "after its own press only");
        // The release that closes a click begun in an earlier frame comes first.
        let mut t = InputTranslator::default();
        let mut input = Input::default();
        input.inject_mouse_move(Vec2::new(10.0, 10.0));
        input.inject_mouse_button_down(l);
        input.inject_mouse_button_down(r);
        translate_frame(&mut t, &mut input, "");
        input.inject_mouse_button_up(r);
        input.release_mouse_button(l, cancel);
        assert_eq!(order(&mut t, &input), [0, 1], "right's click is complete");
        // Two cancels keep their order, both ahead of another button's click.
        let mut t = InputTranslator::default();
        let mut input = Input::default();
        input.inject_mouse_move(Vec2::new(10.0, 10.0));
        input.inject_mouse_button_down(l);
        input.inject_mouse_button_down(m);
        translate_frame(&mut t, &mut input, "");
        input.inject_mouse_button_down(r);
        input.inject_mouse_button_up(r);
        input.release_mouse_button(m, cancel);
        input.release_mouse_button(l, cancel);
        assert_eq!(order(&mut t, &input), [2, 3, 0, 1]);
        // Nothing to move: the frame's own order.
        input.end_frame();
        input.inject_mouse_button_down(l);
        input.inject_mouse_button_up(l);
        assert_eq!(order(&mut t, &input), [0, 1]);
    }

    /// Before any pointer position is known a button event is dropped, its release with it.
    #[test]
    fn buttons_before_a_position_is_known_are_dropped_as_a_pair() {
        let mut t = InputTranslator::default();
        let mut input = Input::default();
        input.inject_mouse_button_down(MouseButton::Left);
        input.inject_mouse_button_down(MouseButton::Right);
        let ev = translate_frame(&mut t, &mut input, "");
        assert!(pointer(&ev).is_empty(), "{ev:?}");
        assert_eq!(t.down, [false; 5]);
        // The position becomes known while both are held: the release of a dropped press goes
        // nowhere, a new press works.
        input.inject_mouse_move(Vec2::new(4.0, 4.0));
        input.inject_mouse_button_up(MouseButton::Left);
        let ev = translate_frame(&mut t, &mut input, "");
        assert_eq!(pointer(&ev), [P::Moved(egui::pos2(4.0, 4.0))], "no release, no press");
        input.inject_mouse_button_up(MouseButton::Right);
        input.inject_mouse_button_down(MouseButton::Left);
        input.inject_mouse_button_up(MouseButton::Left);
        let ev = translate_frame(&mut t, &mut input, "");
        let at = egui::pos2(4.0, 4.0);
        assert_eq!(
            pointer(&ev),
            [P::Button(PRIMARY, true, at), P::Button(PRIMARY, false, at)]
        );
        // A press before the pointer is known but after egui saw one is not dropped (outside the
        // window: at the last position).
        input.inject_mouse_left();
        input.inject_mouse_button_down(MouseButton::Left);
        let ev = translate_frame(&mut t, &mut input, "");
        assert_eq!(pointer(&ev), [P::Button(PRIMARY, true, at), P::Gone]);
    }

    #[test]
    fn buttons_egui_still_holds_are_released_as_cancels() {
        let mut t = InputTranslator::default();
        let mut input = Input::default();
        input.inject_mouse_move(Vec2::new(8.0, 9.0));
        input.inject_mouse_button_down(MouseButton::Left);
        translate_frame(&mut t, &mut input, "");
        // `Input` replaced mid-press (or egui was not told of the release).
        let mut input = Input::default();
        input.inject_mouse_move(Vec2::new(8.0, 9.0));
        let mut ev = Vec::new();
        assert!(t.translate(&input, 1.0, 0.0, "", &mut ev));
        let a = egui::pos2(8.0, 9.0);
        assert_eq!(
            pointer(&ev),
            [P::Moved(CANCEL_DETOUR), P::Moved(a), P::Button(PRIMARY, false, a)]
        );
        assert!(!t.down[0]);
    }

    #[test]
    fn focus_changes_reach_egui_once() {
        let mut t = InputTranslator::default();
        let mut ev = Vec::new();
        t.focus(true, &mut ev);
        assert!(ev.is_empty(), "egui starts focused");
        t.focus(false, &mut ev);
        t.focus(false, &mut ev);
        assert_eq!(pointer(&ev), [P::Focused(false)]);
        ev.clear();
        t.focus(true, &mut ev);
        t.focus(true, &mut ev);
        assert_eq!(pointer(&ev), [P::Focused(true)]);
    }

    #[test]
    fn keys_modifiers_and_clipboard() {
        let mut t = InputTranslator::default();
        let mut input = Input::default();
        let cmd = if cfg!(any(target_os = "macos", target_os = "ios")) {
            KeyCode::SuperLeft
        } else {
            KeyCode::ControlLeft
        };
        input.inject_key_down(cmd);
        input.inject_key_tap(KeyCode::KeyV);
        let mut ev = Vec::new();
        t.translate(&input, 1.0, 0.0, "pasted", &mut ev);
        assert!(
            ev.iter()
                .any(|e| matches!(e, egui::Event::ModifiersChanged(m) if m.command))
        );
        assert!(ev.contains(&egui::Event::Paste("pasted".into())));
        assert!(
            !ev.iter()
                .any(|e| matches!(e, egui::Event::Key { key: egui::Key::V, .. }))
        );
        input.end_frame();
        input.inject_key_up(cmd);
        input.inject_key_tap(KeyCode::Enter);
        ev.clear();
        t.translate(&input, 1.0, 0.0, "", &mut ev);
        let enter: Vec<bool> = ev
            .iter()
            .filter_map(|e| match e {
                egui::Event::Key {
                    key: egui::Key::Enter,
                    pressed,
                    ..
                } => Some(*pressed),
                _ => None,
            })
            .collect();
        assert_eq!(enter, vec![true, false]);
    }

    #[test]
    fn touch_drives_pointer() {
        let mut t = InputTranslator::default();
        let mut input = Input::default();
        input.inject_touch(7, space3d_platform::TouchPhase::Started, Vec2::new(10.0, 10.0));
        let mut ev = Vec::new();
        t.translate(&input, 1.0, 0.0, "", &mut ev);
        assert!(ev.iter().any(|e| matches!(
            e,
            egui::Event::Touch {
                phase: egui::TouchPhase::Start,
                ..
            }
        )));
        assert!(
            ev.iter()
                .any(|e| matches!(e, egui::Event::PointerButton { pressed: true, .. }))
        );
        input.end_frame();
        input.inject_touch(7, space3d_platform::TouchPhase::Ended, Vec2::new(12.0, 10.0));
        ev.clear();
        t.translate(&input, 1.0, 0.0, "", &mut ev);
        assert!(
            ev.iter()
                .any(|e| matches!(e, egui::Event::PointerButton { pressed: false, .. }))
        );
        assert_eq!(ev.last(), Some(&egui::Event::PointerGone));
    }

    // ---- the input method ----------------------------------------------------------------

    fn preedit(text: &str) -> ImeEvent {
        ImeEvent::Preedit {
            text: text.into(),
            cursor: None,
        }
    }

    /// egui's IME events of one frame's `Input::ime_events` (a translator that keeps what it knows
    /// between frames: pass it in).
    fn ime_frame(t: &mut InputTranslator, events: &[ImeEvent]) -> Vec<egui::Event> {
        let mut input = Input::default();
        for e in events {
            input.inject_ime(e.clone());
        }
        let mut ev = Vec::new();
        t.translate(&input, 1.0, 0.0, "", &mut ev);
        ev
    }

    fn egui_preedit(text: &str, active: Option<std::ops::Range<usize>>) -> egui::Event {
        egui::Event::Ime(egui::ImeEvent::Preedit {
            text: text.into(),
            active_range_chars: active,
        })
    }

    fn egui_commit(text: &str) -> egui::Event {
        egui::Event::Ime(egui::ImeEvent::Commit(text.into()))
    }

    #[test]
    fn a_preedit_cursor_in_bytes_reaches_egui_in_characters() {
        let mut t = InputTranslator::default();
        // winit: 你好 is 6 bytes; the IME converts the second character (bytes 3..6).
        let winit = space3d_platform::winit::event::Ime::Preedit("你好".into(), Some((3, 6)));
        let ev = ime_frame(&mut t, &[ImeEvent::from_winit(winit)]);
        assert_eq!(ev, vec![egui_preedit("你好", Some(1..2))], "characters, not bytes");
    }

    #[test]
    fn a_preedit_cursor_outside_the_text_is_cut_to_it() {
        let mut t = InputTranslator::default();
        let ev = ime_frame(
            &mut t,
            &[ImeEvent::Preedit {
                text: "ni".into(),
                cursor: Some(5..9),
            }],
        );
        assert_eq!(ev, vec![egui_preedit("ni", Some(2..2))]);
    }

    #[test]
    fn a_commit_is_one_ime_event_and_no_text() {
        let mut t = InputTranslator::default();
        let ev = ime_frame(
            &mut t,
            &[preedit("nihao"), preedit(""), ImeEvent::Commit("你好".into())],
        );
        assert_eq!(
            ev,
            vec![egui_preedit("nihao", None), egui_preedit("", None), egui_commit("你好")]
        );
        assert!(
            !ev.iter().any(|e| matches!(e, egui::Event::Text(_))),
            "egui would enter it twice"
        );
    }

    #[test]
    fn a_commit_without_a_preedit_works() {
        let mut t = InputTranslator::default();
        assert_eq!(
            ime_frame(&mut t, &[ImeEvent::Commit("你好".into())]),
            vec![egui_commit("你好")]
        );
        // (control characters are no text; a commit of nothing else is nothing)
        assert_eq!(ime_frame(&mut t, &[ImeEvent::Commit("\n".into())]), vec![]);
        assert_eq!(
            ime_frame(&mut t, &[ImeEvent::Commit("好\r".into())]),
            vec![egui_commit("好")]
        );
    }

    #[test]
    fn an_empty_preedit_ends_the_composition_once() {
        let mut t = InputTranslator::default();
        assert_eq!(ime_frame(&mut t, &[preedit("ni")]), vec![egui_preedit("ni", None)]);
        assert_eq!(ime_frame(&mut t, &[preedit("")]), vec![egui_preedit("", None)], "ended");
        // Wayland sends them with no composition (and Windows twice around a commit).
        assert_eq!(ime_frame(&mut t, &[preedit(""), preedit("")]), vec![]);
        // The composition's end is remembered across frames.
        assert_eq!(ime_frame(&mut t, &[preedit("a")]), vec![egui_preedit("a", None)]);
        assert_eq!(
            ime_frame(&mut t, &[ImeEvent::Commit("啊".into()), preedit("")]),
            vec![egui_commit("啊")],
            "a commit ended it: the empty preedit after it is nothing"
        );
    }

    #[test]
    fn disabled_clears_a_live_composition_and_otherwise_does_nothing() {
        let mut t = InputTranslator::default();
        assert_eq!(ime_frame(&mut t, &[ImeEvent::Disabled]), vec![]);
        assert_eq!(ime_frame(&mut t, &[preedit("ni")]), vec![egui_preedit("ni", None)]);
        assert_eq!(
            ime_frame(&mut t, &[ImeEvent::Disabled]),
            vec![egui_preedit("", None)],
            "the preedit text goes out of the field"
        );
        assert_eq!(ime_frame(&mut t, &[ImeEvent::Disabled, ImeEvent::Enabled]), vec![]);
        // Windows: a commit, then Disabled.
        ime_frame(&mut t, &[preedit("h")]);
        assert_eq!(
            ime_frame(&mut t, &[ImeEvent::Commit("好".into()), ImeEvent::Disabled]),
            vec![egui_commit("好")]
        );
    }

    #[test]
    fn the_input_method_s_events_come_before_the_frame_s_keys() {
        // A commit and the Enter after it in one frame: the text goes in first.
        let mut t = InputTranslator::default();
        let mut input = Input::default();
        input.inject_ime(ImeEvent::Commit("你好".into()));
        input.inject_key_tap(KeyCode::Enter);
        let mut ev = Vec::new();
        t.translate(&input, 1.0, 0.0, "", &mut ev);
        let commit = ev.iter().position(|e| *e == egui_commit("你好")).expect("commit");
        let enter = ev
            .iter()
            .position(|e| {
                matches!(
                    e,
                    egui::Event::Key {
                        key: egui::Key::Enter,
                        ..
                    }
                )
            })
            .expect("enter");
        assert!(commit < enter, "{ev:?}");
    }

    /// The phones' keyboards and the web's hidden input send key events with text, and a desktop
    /// with no input method (or no `Ime` event in the frame) has nothing else: what egui gets is
    /// what it got before the input method support, and nothing of an input method.
    #[test]
    fn a_frame_without_input_method_events_is_what_it_was() {
        let mut t = InputTranslator::default();
        let mut input = Input::default();
        input.inject_key_event(space3d_platform::KeyEvent::press(
            KeyCode::KeyA,
            LogicalKey::Character("a".into()),
        ));
        input.inject_text("é");
        let mut ev = Vec::new();
        t.translate(&input, 1.0, 0.0, "", &mut ev);
        assert_eq!(
            ev,
            vec![
                egui::Event::Key {
                    key: egui::Key::A,
                    physical_key: Some(egui::Key::A),
                    pressed: true,
                    repeat: false,
                    modifiers: egui::Modifiers::NONE,
                },
                egui::Event::Text("aé".into()),
            ]
        );
        assert!(!t.ime_composing);
    }
}
