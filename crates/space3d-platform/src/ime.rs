//! Input methods (IME) on the desktop: when the window asks the OS for one.
//!
//! A player who types Chinese, Japanese or Korean does it through an input method (IME): the
//! keys go to the IME, which shows what is being composed (the *preedit*, `nihao` for 你好) and
//! a list of candidates, and sends the finished text (the *commit*) when the player picks one.
//! winit gives a window an input method only when the app asks for it
//! (`Window::set_ime_allowed`); without it Windows removes the IME from the window (the
//! language bar's mode switch does nothing, pinyin keys arrive as Latin letters), macOS and X11
//! bypass the IME and Wayland never enables text input.
//!
//! On Windows, macOS, Linux and the BSDs (not on phones, which have their own on-screen
//! keyboard path in the runner, and not on the web, where winit has no IME) the IME is on
//! **only while a text field has the focus** (`TextInput::wanted`) and **never for a password
//! field** (`TextInput::password`): flight keys and hotkeys stay raw while the player flies,
//! even when his IME is in Chinese mode, and a password is never composed.

use crate::window::TextInput;

/// Whether the window's input method should change: `Some(on)` when a text field wants text
/// (`wanted`) that is not a password (`password`) and that differs from `current`, `None` when
/// nothing changes. A pure function of three booleans, so the same frame twice asks nothing
/// twice (toggling the OS input method ends a composition that is under way).
pub(crate) fn ime_switch(wanted: bool, password: bool, current: bool) -> Option<bool> {
    let on = wanted && !password;
    (on != current).then_some(on)
}

/// What the runner does to the window after a frame: see `ImeState::step`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ImeStep {
    /// `Window::set_ime_allowed(on)`.
    pub allow: Option<bool>,
}

/// What the window runner remembers of the input method between frames.
#[derive(Clone, Debug, Default)]
pub(crate) struct ImeState {
    /// What the window was last told (`set_ime_allowed`).
    on: bool,
}

impl ImeState {
    /// What to tell the window after a frame, given what the game asks (`TextInput`).
    pub(crate) fn step(&mut self, text: &TextInput) -> ImeStep {
        let mut step = ImeStep::default();
        if let Some(on) = ime_switch(text.wanted, text.password, self.on) {
            self.on = on;
            step.allow = Some(on);
        }
        step
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(wanted: bool, password: bool) -> TextInput {
        TextInput { wanted, password }
    }

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
        let mut state = ImeState::default();
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
}
