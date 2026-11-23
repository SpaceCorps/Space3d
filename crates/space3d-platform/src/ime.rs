//! Input methods (IME) on the desktop: the events of the OS input method and the decision when
//! the window asks for one.
//!
//! A player who types Chinese, Japanese or Korean does it through an input method (IME): the
//! keys go to the IME, which shows what is being composed (the *preedit*, `nihao` for 你好) and
//! a list of candidates, and sends the finished text (the *commit*) when the player picks one.
//! winit gives a window an input method only when the app asks for it
//! (`Window::set_ime_allowed`); without it Windows removes the IME from the window (the
//! language bar's mode switch does nothing, pinyin keys arrive as Latin letters), macOS and X11
//! bypass the IME and Wayland never enables text input.
//!
//! What the engine does, on Windows, macOS, Linux and the BSDs (not on phones, which have their
//! own on-screen keyboard path in the runner, and not on the web, where winit has no IME):
//!
//! 1. **On while a text field has the focus** (`TextInput::wanted`), off otherwise, and off for
//!    a password field (`TextInput::password`): flight keys and hotkeys stay raw while the player
//!    flies, even when his IME is in Chinese mode, and a password is never composed
//!    (`ime_switch`).
//! 2. **The composition is forwarded** ([`ImeEvent`], `Input::ime_events`): `space3d-ui` makes
//!    egui's `Event::Ime` of them, so a `TextEdit` shows the preedit underlined and takes the
//!    commit once. A commit is **not** also typed text (`Input::text`): egui would enter it twice.
//! 3. **The candidate list opens at the text caret** (`TextInput::caret`, [`ImeArea`], physical
//!    pixels): the runner passes it to `Window::set_ime_cursor_area` when it changes, and again
//!    when a composition starts or the IME comes back on.
//! 4. **Keys the IME processed are not keys of the app**: on Windows the IME gets a key before
//!    the app and winit still reports it, with the logical key [`NamedKey::Process`]; the Esc that
//!    cancels a composition, the Enter that commits the letters and the Backspace that deletes
//!    one of them must not also close a chat, send it or delete the text before the composition
//!    (`ImeState::swallows_key`).
//!
//! **Kill switch:** the environment variable [`ENV_NO_IME`] (`SPACE3D_NO_IME=1`) or
//! `WindowPlugin::ime = false` leaves the window as it was before the engine knew input methods:
//! no IME on a desktop, no event of it, a commit (there is none then) typed text.

use crate::input::Input;
use crate::layout::{LogicalKey, NamedKey};
use crate::window::TextInput;
use std::ops::Range;

/// Environment variable that switches the desktop input method support off (`1`, `true`,
/// `yes` or `on`; anything else, or unset, leaves it on). The other switch is
/// `WindowPlugin::ime`.
pub const ENV_NO_IME: &str = "SPACE3D_NO_IME";

/// What an input method tells the app (`Input::ime_events`, one frame's worth in order).
///
/// winit's `Ime` with the cursor in characters. The usual order on Windows is
/// `Enabled`, `Preedit` (growing), then `Preedit("")` and `Commit`, then `Disabled`; macOS
/// sends an empty `Preedit` before every `Commit` and Wayland empty ones with no composition, so
/// a consumer must not rely on the order beyond this: a `Preedit` with text starts or updates a
/// composition; an empty one, a `Commit` or a `Disabled` ends it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImeEvent {
    /// The input method started (winit `Ime::Enabled`).
    Enabled,
    /// What is being composed. An empty `text` ends the composition (nothing is committed).
    Preedit {
        /// The composition string.
        text: String,
        /// The part of `text` the IME is converting now (the active clause, or the caret as an
        /// empty range) as **character** offsets into `text`; `None`: it shows none. (winit
        /// gives byte offsets: [`ImeEvent::from_winit`] converts them.)
        cursor: Option<Range<usize>>,
    },
    /// The finished text.
    Commit(String),
    /// The input method ended (winit `Ime::Disabled`), or the window turned it off.
    Disabled,
}

impl ImeEvent {
    /// winit's event as ours: the preedit's cursor from byte offsets to character offsets
    /// (`你好` is 6 bytes and 2 characters), clamped to the text and to character boundaries.
    pub fn from_winit(ime: winit::event::Ime) -> Self {
        use winit::event::Ime;
        match ime {
            Ime::Enabled => Self::Enabled,
            Ime::Preedit(text, cursor) => {
                let cursor = cursor.map(|(start, end)| char_range(&text, start, end));
                Self::Preedit { text, cursor }
            }
            Ime::Commit(text) => Self::Commit(text),
            Ime::Disabled => Self::Disabled,
        }
    }
}

/// The byte range `start..end` of `text` as a range of characters. An offset past the end or
/// inside a character (an IME that counts in another unit) moves back to the boundary before it;
/// a backwards range is turned around.
fn char_range(text: &str, start: usize, end: usize) -> Range<usize> {
    let to_chars = |mut byte: usize| {
        byte = byte.min(text.len());
        while !text.is_char_boundary(byte) {
            byte -= 1;
        }
        text[..byte].chars().count()
    };
    let (a, b) = (to_chars(start), to_chars(end));
    a.min(b)..a.max(b)
}

/// Where the text caret is, in physical pixels of the window (`TextInput::caret`): the
/// rectangle an input method keeps clear and opens its candidate list next to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ImeArea {
    /// Left edge.
    pub x: i32,
    /// Top edge.
    pub y: i32,
    /// Width, at least 1.
    pub width: u32,
    /// Height, at least 1.
    pub height: u32,
}

impl ImeArea {
    /// A rectangle of `size` at `min` in UI points as physical pixels: every number times
    /// `pixels_per_point`, rounded (so a caret that moves by less than a pixel is no change).
    /// `None` for a number that is not finite or a scale that is not positive.
    pub fn from_points(min: [f32; 2], size: [f32; 2], pixels_per_point: f32) -> Option<Self> {
        let all = [min[0], min[1], size[0], size[1], pixels_per_point];
        if !all.iter().all(|v| v.is_finite()) || pixels_per_point <= 0.0 {
            return None;
        }
        let px = |v: f32| (v * pixels_per_point).round();
        // (`as` saturates: a rectangle far outside the window is still a rectangle)
        Some(Self {
            x: px(min[0]) as i32,
            y: px(min[1]) as i32,
            width: px(size[0].max(0.0)).max(1.0) as u32,
            height: px(size[1].max(0.0)).max(1.0) as u32,
        })
    }
}

/// Whether the input method support is on: `flag` (`WindowPlugin::ime`) and no `SPACE3D_NO_IME`
/// in the environment (`get` reads a variable).
pub(crate) fn resolve_enabled(flag: bool, get: impl Fn(&str) -> Option<String>) -> bool {
    let off =
        get(ENV_NO_IME).is_some_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"));
    flag && !off
}

/// Whether the window's input method should change: `Some(on)` when a text field wants text
/// (`wanted`) that is not a password (`password`) and that differs from `current`, `None` when
/// nothing changes. A pure function of three booleans, so the same frame twice asks nothing
/// twice (toggling the OS input method ends a composition that is under way).
pub(crate) fn ime_switch(wanted: bool, password: bool, current: bool) -> Option<bool> {
    let on = wanted && !password;
    (on != current).then_some(on)
}

/// The area to pass to the window: `wanted` when it is one and differs from the `applied` one.
pub(crate) fn ime_area_to_send(wanted: Option<ImeArea>, applied: Option<ImeArea>) -> Option<ImeArea> {
    wanted.filter(|w| applied != Some(*w))
}

/// What the runner does to the window after a frame: see `ImeState::step`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ImeStep {
    /// `Window::set_ime_allowed(on)`.
    pub allow: Option<bool>,
    /// `Window::set_ime_cursor_area` here.
    pub area: Option<ImeArea>,
    /// The composition is to end at the OS while the field goes on: `set_ime_allowed(false)`,
    /// then `set_ime_allowed(true)` (egui interrupted it: see `TextInput::interrupt`).
    pub restart: bool,
    /// The input method was turned off, or restarted, while it composed: the app is told with an
    /// `ImeEvent::Disabled`, so that what the UI's translator remembers of the composition ends
    /// with it. egui takes the preedit text out of the field for it only while that field still
    /// has the focus (a `TextEdit` takes no event of an input method otherwise): a field that
    /// lost the focus keeps what was composed as plain text.
    pub cancel: bool,
}

/// What the window runner remembers of the input method between frames.
#[derive(Clone, Debug)]
pub(crate) struct ImeState {
    /// The support is on (a desktop; not switched off).
    enabled: bool,
    /// What the window was last told (`set_ime_allowed`).
    on: bool,
    /// A composition is going on (a preedit with text came, nothing ended it yet).
    composing: bool,
    /// What the window was last told as the caret's area (`None`: nothing since it was turned on).
    area: Option<ImeArea>,
    /// `TextInput::interrupt` of the last step: only a change from false to true interrupts.
    interrupted: bool,
}

impl ImeState {
    /// The state of a window whose input method support is `enabled` (or not).
    pub(crate) fn new(enabled: bool) -> Self {
        Self {
            enabled,
            on: false,
            composing: false,
            area: None,
            interrupted: false,
        }
    }

    /// What to tell the window after a frame, given what the game asks (`TextInput`). Nothing
    /// when the support is off. The window's area is forgotten whenever the input method is
    /// turned on (a context made new knows none), so the next call sends it.
    pub(crate) fn step(&mut self, text: &TextInput) -> ImeStep {
        let mut step = ImeStep::default();
        if !self.enabled {
            return step;
        }
        // egui asks for a composition to be interrupted in the pass where the player clicks into
        // the field, and in every pass where a game calls `request_focus`: only the first of a
        // run of them counts, or a game that asks every frame would never let a composition live.
        let interrupt = text.interrupt && !self.interrupted;
        self.interrupted = text.interrupt;
        if let Some(on) = ime_switch(text.wanted, text.password, self.on) {
            self.on = on;
            self.area = None;
            step.allow = Some(on);
            if !on && self.composing {
                self.composing = false;
                step.cancel = true;
            }
        } else if interrupt && self.on && self.composing {
            // egui already counts the preedit as plain text, so the input method must stop
            // composing it; a context made new knows no area.
            self.composing = false;
            self.area = None;
            step.restart = true;
            step.cancel = true;
        }
        if self.on {
            step.area = ime_area_to_send(text.caret, self.area);
            if step.area.is_some() {
                self.area = step.area;
            }
        }
        step
    }

    /// Note an event of the input method. A composition that starts has the area sent again:
    /// an input method may place its windows by what it was last told and reset it when its
    /// mode or language changes.
    pub(crate) fn event(&mut self, event: &ImeEvent) {
        match event {
            ImeEvent::Enabled => self.area = None,
            ImeEvent::Preedit { text, .. } => {
                let composing = !text.is_empty();
                if composing && !self.composing {
                    self.area = None;
                }
                self.composing = composing;
            }
            ImeEvent::Commit(_) | ImeEvent::Disabled => self.composing = false,
        }
    }

    /// Hand winit's `ime` to the app: with the support on as an `ImeEvent` for
    /// `Input::ime_events` (never also as text), else as before the engine knew input methods:
    /// a commit is typed text (winit 0.30 sends one on desktops that allow it only; the
    /// phones' and the web's keyboards send none), the rest is dropped.
    pub(crate) fn deliver(&mut self, ime: winit::event::Ime, input: &mut Input) {
        if self.enabled {
            let event = ImeEvent::from_winit(ime);
            self.event(&event);
            input.inject_ime(event);
        } else if let winit::event::Ime::Commit(text) = ime {
            input.inject_text(&text);
        }
    }

    /// Whether a key event is one the input method processed and the app must not see: its
    /// logical key is [`NamedKey::Process`] (Windows' `VK_PROCESSKEY`), and it is a press, or a
    /// release of a key the app never saw pressed (the release of a key that is down always
    /// goes through, or the key would stay held).
    pub(crate) fn swallows_key(&self, logical: &LogicalKey, pressed: bool, held: bool) -> bool {
        self.enabled && *logical == LogicalKey::Named(NamedKey::Process) && (pressed || !held)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use winit::event::Ime;

    /// 你好: 2 characters, 6 bytes.
    const NIHAO: &str = "你好";

    fn field(wanted: bool, password: bool) -> TextInput {
        TextInput {
            wanted,
            password,
            ..TextInput::default()
        }
    }

    fn area(x: i32) -> ImeArea {
        ImeArea {
            x,
            y: 700,
            width: 2,
            height: 20,
        }
    }

    fn caret(caret: ImeArea) -> TextInput {
        TextInput {
            wanted: true,
            caret: Some(caret),
            ..TextInput::default()
        }
    }

    fn preedit(text: &str) -> ImeEvent {
        ImeEvent::Preedit {
            text: text.into(),
            cursor: None,
        }
    }

    // ---- the decision -------------------------------------------------------------------

    #[test]
    fn the_input_method_is_on_while_a_text_field_has_the_focus() {
        assert_eq!(ime_switch(true, false, false), Some(true), "a field got the focus");
        assert_eq!(ime_switch(false, false, true), Some(false), "the focus left it");
        assert_eq!(ime_switch(false, false, false), None, "flying: stays off");
        assert_eq!(ime_switch(true, false, true), None, "typing: stays on");
    }

    #[test]
    fn a_password_field_never_gets_an_input_method() {
        assert_eq!(ime_switch(true, true, false), None, "off stays off");
        assert_eq!(
            ime_switch(true, true, true),
            Some(false),
            "the chat's IME goes off at the password"
        );
        assert_eq!(ime_switch(false, true, true), Some(false));
    }

    #[test]
    fn frames_that_change_nothing_ask_nothing() {
        // The chat gets the focus, stays, goes to a password, back, and leaves.
        let frames = [
            (false, false),
            (true, false),
            (true, false),
            (true, false),
            (true, true),
            (true, true),
            (true, false),
            (false, false),
            (false, false),
        ];
        let mut state = ImeState::new(true);
        let asked: Vec<Option<bool>> = frames
            .iter()
            .map(|&(wanted, password)| state.step(&field(wanted, password)).allow)
            .collect();
        assert_eq!(
            asked,
            [
                None,
                Some(true),
                None,
                None,
                Some(false),
                None,
                Some(true),
                Some(false),
                None
            ]
        );
    }

    #[test]
    fn turning_it_off_while_composing_cancels_the_composition() {
        let mut state = ImeState::new(true);
        state.step(&field(true, false));
        state.event(&preedit("ni"));
        let step = state.step(&field(false, false));
        assert_eq!((step.allow, step.cancel), (Some(false), true));
        // Said once.
        assert_eq!(state.step(&field(false, false)), ImeStep::default());
        // Not composing: nothing to cancel.
        state.step(&field(true, false));
        assert!(!state.step(&field(false, false)).cancel);
    }

    // ---- egui interrupts a composition ----------------------------------------------------

    fn interrupting(text: TextInput) -> TextInput {
        TextInput {
            interrupt: true,
            ..text
        }
    }

    #[test]
    fn an_interrupt_while_composing_restarts_the_input_method_once() {
        let mut state = ImeState::new(true);
        let at = caret(area(10));
        state.step(&at);
        state.event(&preedit("nihao"));
        // The player clicked into the field: egui counts the preedit as text from now on.
        let step = state.step(&interrupting(at));
        assert_eq!((step.allow, step.restart, step.cancel), (None, true, true));
        // A context made new knows no area: the caret is told again in the same step.
        assert_eq!(step.area, Some(area(10)));
        assert!(!state.composing, "the composition ended at the OS");
        // The pass after still says so (a `request_focus` every frame): nothing happens twice.
        state.event(&preedit("ni"));
        let step = state.step(&interrupting(at));
        assert_eq!((step.restart, step.cancel), (false, false), "no edge: not again");
        assert!(state.composing, "a composition that began after it goes on");
    }

    #[test]
    fn an_interrupt_with_no_composition_changes_nothing() {
        let mut state = ImeState::new(true);
        let at = caret(area(10));
        state.step(&at);
        // The first click into a field, a `request_focus`: nothing is being composed.
        assert_eq!(state.step(&interrupting(at)), ImeStep::default());
        // It was seen: a composition that starts while the flag stays up is not interrupted.
        state.event(&preedit("n"));
        assert!(!state.step(&interrupting(at)).restart);
        // The edge counts again after the flag went down.
        state.step(&at);
        assert!(state.step(&interrupting(at)).restart);
    }

    #[test]
    fn an_interrupt_with_the_input_method_off_or_going_off_is_no_restart() {
        // Off (a password field, no field): there is no context to restart.
        let mut state = ImeState::new(true);
        let password = TextInput {
            interrupt: true,
            ..field(true, true)
        };
        assert_eq!(state.step(&password), ImeStep::default());
        // Going off in the same step: that is the cancel, once.
        let mut state = ImeState::new(true);
        state.step(&field(true, false));
        state.event(&preedit("ni"));
        let step = state.step(&TextInput {
            interrupt: true,
            ..field(false, false)
        });
        assert_eq!((step.allow, step.restart, step.cancel), (Some(false), false, true));
        // Support off (the kill switch): nothing, whatever egui says.
        let mut state = ImeState::new(false);
        assert_eq!(state.step(&interrupting(caret(area(10)))), ImeStep::default());
    }

    // ---- the caret's area ---------------------------------------------------------------

    #[test]
    fn the_area_is_in_physical_pixels_at_every_scale() {
        // A caret 2 x 20 points at (100, 300).
        let of = |ppp| ImeArea::from_points([100.0, 300.0], [2.0, 20.0], ppp);
        let at = |x, y, width, height| Some(ImeArea { x, y, width, height });
        assert_eq!(of(1.0), at(100, 300, 2, 20));
        assert_eq!(of(1.5), at(150, 450, 3, 30));
        assert_eq!(of(2.0), at(200, 600, 4, 40));
        // Rounded, not cut; never empty.
        assert_eq!(ImeArea::from_points([100.5, 10.0], [0.0, 0.0], 1.5), at(151, 15, 1, 1));
        // Nothing to say for a number that is no number.
        assert_eq!(ImeArea::from_points([f32::NAN, 0.0], [1.0, 1.0], 1.0), None);
        assert_eq!(ImeArea::from_points([0.0, 0.0], [1.0, f32::INFINITY], 1.0), None);
        assert_eq!(ImeArea::from_points([0.0, 0.0], [1.0, 1.0], 0.0), None);
    }

    #[test]
    fn the_area_goes_to_the_window_only_when_it_changed() {
        let mut state = ImeState::new(true);
        let steps: Vec<Option<ImeArea>> = [10, 10, 10, 30, 30, 10]
            .into_iter()
            .map(|x| state.step(&caret(area(x))).area)
            .collect();
        assert_eq!(
            steps,
            [Some(area(10)), None, None, Some(area(30)), None, Some(area(10))]
        );
        // No caret known: nothing to say.
        assert_eq!(state.step(&field(true, false)).area, None);
    }

    #[test]
    fn the_area_goes_again_when_the_input_method_comes_back_on_and_a_composition_starts() {
        let mut state = ImeState::new(true);
        let at = caret(area(10));
        assert_eq!(state.step(&at).area, Some(area(10)));
        assert_eq!(state.step(&at).area, None);
        // Off and on: a context made new knows no area.
        let off = TextInput { wanted: false, ..at };
        assert_eq!(state.step(&off).area, None);
        assert_eq!(state.step(&at).area, Some(area(10)));
        assert_eq!(state.step(&at).area, None);
        // A composition starts: sent again once, not for every preedit that follows.
        state.event(&ImeEvent::Enabled);
        assert_eq!(state.step(&at).area, Some(area(10)));
        state.event(&preedit("n"));
        assert_eq!(state.step(&at).area, Some(area(10)));
        state.event(&preedit("ni"));
        state.event(&preedit("nih"));
        assert_eq!(state.step(&at).area, None);
        // It ended, and another starts.
        state.event(&preedit(""));
        state.event(&preedit("h"));
        assert_eq!(state.step(&at).area, Some(area(10)));
    }

    #[test]
    fn no_area_while_the_input_method_is_off() {
        let mut state = ImeState::new(true);
        // A password field's caret is no business of the input method.
        let password = TextInput {
            password: true,
            ..caret(area(10))
        };
        assert_eq!(state.step(&password), ImeStep::default());
        let flying = TextInput {
            wanted: false,
            ..caret(area(10))
        };
        assert_eq!(state.step(&flying), ImeStep::default());
    }

    // ---- the kill switch ----------------------------------------------------------------

    #[test]
    fn the_kill_switch_is_the_flag_or_the_environment() {
        let env = |v: Option<&'static str>| move |k: &str| (k == ENV_NO_IME).then(|| v.map(String::from)).flatten();
        assert!(resolve_enabled(true, env(None)), "on by default");
        assert!(!resolve_enabled(false, env(None)), "WindowPlugin::ime = false");
        for off in ["1", "true", "TRUE", " yes ", "on"] {
            assert!(!resolve_enabled(true, env(Some(off))), "SPACE3D_NO_IME={off:?}");
        }
        for on in ["0", "", "false", "no", "off", "2"] {
            assert!(
                resolve_enabled(true, env(Some(on))),
                "SPACE3D_NO_IME={on:?} leaves it on"
            );
        }
        // Another variable is none of its business.
        assert!(resolve_enabled(true, |k: &str| (k == "CI").then(|| "1".to_string())));
    }

    #[test]
    fn a_state_that_is_off_asks_for_nothing() {
        let mut state = ImeState::new(false);
        assert_eq!(state.step(&caret(area(10))), ImeStep::default());
        assert!(!state.swallows_key(&LogicalKey::Named(NamedKey::Process), true, false));
    }

    // ---- winit's events -----------------------------------------------------------------

    #[test]
    fn a_preedit_cursor_in_bytes_is_in_characters() {
        let cursor = |c| match ImeEvent::from_winit(Ime::Preedit(NIHAO.into(), c)) {
            ImeEvent::Preedit { text, cursor } => {
                assert_eq!(text, NIHAO);
                cursor
            }
            other => panic!("{other:?}"),
        };
        // 3 bytes are one character, 6 two.
        assert_eq!(cursor(Some((0, 6))), Some(0..2));
        assert_eq!(cursor(Some((3, 6))), Some(1..2));
        assert_eq!(cursor(Some((3, 3))), Some(1..1), "the caret alone");
        assert_eq!(cursor(None), None, "no cursor shown");
        // Mixed: nihao is one byte a character.
        assert_eq!(
            ImeEvent::from_winit(Ime::Preedit("ni你好".into(), Some((2, 8)))),
            ImeEvent::Preedit {
                text: "ni你好".into(),
                cursor: Some(2..4)
            }
        );
    }

    #[test]
    fn a_preedit_cursor_that_does_not_fit_the_text_is_made_to() {
        let cursor = |start, end| match ImeEvent::from_winit(Ime::Preedit(NIHAO.into(), Some((start, end)))) {
            ImeEvent::Preedit { cursor, .. } => cursor,
            other => panic!("{other:?}"),
        };
        assert_eq!(cursor(0, 99), Some(0..2), "past the end");
        assert_eq!(cursor(4, 6), Some(1..2), "inside a character: before it");
        assert_eq!(cursor(5, 2), Some(0..1), "turned around, one inside a character");
        assert_eq!(cursor(99, 100), Some(2..2));
    }

    #[test]
    fn the_other_events_pass_as_they_are() {
        assert_eq!(ImeEvent::from_winit(Ime::Enabled), ImeEvent::Enabled);
        assert_eq!(ImeEvent::from_winit(Ime::Disabled), ImeEvent::Disabled);
        assert_eq!(
            ImeEvent::from_winit(Ime::Commit(NIHAO.into())),
            ImeEvent::Commit(NIHAO.into())
        );
        assert_eq!(ImeEvent::from_winit(Ime::Preedit(String::new(), None)), preedit(""));
    }

    // ---- where the events go ------------------------------------------------------------

    #[test]
    fn with_the_support_on_a_commit_is_an_ime_event_and_no_typed_text() {
        let mut state = ImeState::new(true);
        let mut input = Input::default();
        state.deliver(Ime::Enabled, &mut input);
        state.deliver(Ime::Preedit("ni".into(), Some((2, 2))), &mut input);
        state.deliver(Ime::Preedit(String::new(), None), &mut input);
        state.deliver(Ime::Commit(NIHAO.into()), &mut input);
        state.deliver(Ime::Disabled, &mut input);
        assert_eq!(input.text(), "", "egui would enter a commit twice");
        assert_eq!(
            input.ime_events(),
            [
                ImeEvent::Enabled,
                ImeEvent::Preedit {
                    text: "ni".into(),
                    cursor: Some(2..2)
                },
                preedit(""),
                ImeEvent::Commit(NIHAO.into()),
                ImeEvent::Disabled,
            ]
        );
        // A frame's worth: gone at the end of it.
        input.end_frame();
        assert!(input.ime_events().is_empty());
    }

    /// The phones, the web and a desktop with the support off keep what they had: a commit is
    /// typed text, no event of an input method exists.
    #[test]
    fn without_the_support_a_commit_is_typed_text_as_before() {
        let mut state = ImeState::new(false);
        let mut input = Input::default();
        state.deliver(Ime::Enabled, &mut input);
        state.deliver(Ime::Preedit("ni".into(), None), &mut input);
        state.deliver(Ime::Commit(NIHAO.into()), &mut input);
        state.deliver(Ime::Disabled, &mut input);
        assert_eq!(input.text(), NIHAO);
        assert!(input.ime_events().is_empty());
        // And a key's text (the phones' keyboards, the web's hidden input) is untouched.
        input.inject_text("!");
        assert_eq!(input.text(), format!("{NIHAO}!"));
    }

    #[test]
    fn the_state_follows_the_composition() {
        let mut state = ImeState::new(true);
        assert!(!state.composing);
        state.event(&ImeEvent::Enabled);
        assert!(!state.composing, "enabled is no composition yet");
        state.event(&preedit("n"));
        assert!(state.composing);
        state.event(&preedit("ni"));
        assert!(state.composing);
        state.event(&preedit(""));
        assert!(!state.composing, "an empty preedit ends it");
        state.event(&preedit("h"));
        state.event(&ImeEvent::Commit(NIHAO.into()));
        assert!(!state.composing, "so does a commit");
        state.event(&preedit("h"));
        state.event(&ImeEvent::Disabled);
        assert!(!state.composing, "and the input method going away");
    }

    #[test]
    fn keys_the_input_method_processed_are_not_the_app_s() {
        let state = ImeState::new(true);
        let process = LogicalKey::Named(NamedKey::Process);
        let backspace = LogicalKey::Named(NamedKey::Backspace);
        // Esc, Enter, Backspace, Space... pressed while composing.
        assert!(state.swallows_key(&process, true, false));
        assert!(state.swallows_key(&process, true, true), "a repeat too");
        // The release of one the app never saw go down is dropped, one it holds goes through.
        assert!(state.swallows_key(&process, false, false));
        assert!(!state.swallows_key(&process, false, true), "or the key would stay held");
        // Every other key is the app's.
        assert!(!state.swallows_key(&backspace, true, false));
        assert!(!state.swallows_key(&LogicalKey::Character("a".into()), true, false));
    }
}
