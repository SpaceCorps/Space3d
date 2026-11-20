//! Gamepads: how the pads of the OS reach [`Input`].
//!
//! What a game reads is in [`Input`]: [`Input::gamepad`], [`Input::gamepad_ids`],
//! [`Input::gamepad_changes`] (plugged in / unplugged this frame) and the `any_gamepad_*`
//! helpers; [`crate::Binding::GamepadButton`] and [`crate::Binding::GamepadAxis`] bind actions.
//! Buttons are named by their place ([`GamepadButton`]); a stick's axes are -1..1 with **right
//! and up positive**; a trigger is an axis 0..1 and a button
//! ([`GamepadButton::LeftTrigger`]).
//!
//! Where the pads come from:
//!
//! * **Windows, macOS, Linux**: the `gilrs` crate, polled by the window runner before every
//!   frame (Windows.Gaming.Input, IOKit, evdev). Xbox, PlayStation and most other pads come
//!   with the standard layout (gilrs's SDL mapping table). Only the **focused** window obeys
//!   a pad: on a focus loss every held pad button is released and the axes go to rest
//!   ([`Input::release_all`]), and while the window has no focus (or the run is a background
//!   one) only a pad's coming and going is taken ([`Pads::apply_unfocused`]). Windows itself
//!   gives pad input to the focused window only; evdev and IOKit give it to every process, so
//!   the runner holds it back. gilrs is not started at all in a background run
//!   (`background_mode`) or with `SPACE3D_GAMEPAD=off` ([`ENV_GAMEPAD`]): its Windows
//!   backend is a thread that asks the OS for the pads every 8 ms. Linux needs `libudev` (and
//!   `pkg-config` to build).
//! * **Android**: winit 0.30 has no gamepad events. A pad's buttons arrive as key events of no
//!   place and are mapped here ([`AndroidPads::key`]): A B X Y, L1 R1, L3 R3, Start, Select,
//!   Mode; L2 R2 where the pad sends them as keys too (PlayStation pads are said to, Xbox
//!   pads send them as axes only); the d-pad only of a pad that sends it as arrow keys (Xbox
//!   and PlayStation pads send a hat axis, so theirs does nothing). A device is a pad from
//!   its first **button**. winit reports every motion event as a touch, so a stick arrives as
//!   a "finger" that never went down, at the place (`AXIS_X`, `AXIS_Y`):
//!   [`AndroidPads::touch`] takes those of a known pad as the **left stick** and keeps every
//!   such move away from the fingers. The right stick, analog triggers and a hat d-pad are
//!   lost until winit is patched. winit has no event for a pad that went away either: the
//!   runner asks Android for its gamepads once a second ([`AndroidPads::os_pads`]), which
//!   also gives the pad its name while it is the only one. Read from winit's source; not yet
//!   run on a device.
//! * **iOS, the web**: nothing yet (`GCController`, `navigator.getGamepads`).
//!
//! Every backend goes through [`Pads`], which owns the filtering: a radial dead zone on each
//! stick ([`dead_zone`], [`STICK_DEAD_ZONE`]) and the trigger's button with hysteresis
//! ([`trigger_down`]). [`Pads::apply`] is plain code over [`PadEvent`]s, tested on any host.

use crate::input::{GamepadAxis, GamepadButton, Input, TouchPhase};
use glam::Vec2;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

/// A stick closer to its rest than this is at rest (the worn sticks of real pads never
/// return to 0).
pub const STICK_DEAD_ZONE: f32 = 0.15;
/// A trigger pulled this far is its button, down.
pub const TRIGGER_PRESS: f32 = 0.5;
/// A trigger that was down is up again once it is let go this far.
pub const TRIGGER_RELEASE: f32 = 0.4;
/// Env var: `off` (or `0`, `false`, `no`) leaves the desktop backend (gilrs and its thread)
/// unstarted: no pads.
pub const ENV_GAMEPAD: &str = "SPACE3D_GAMEPAD";

/// Whether [`ENV_GAMEPAD`]'s `value` turns the pads off.
pub fn env_turns_off(value: Option<&str>) -> bool {
    value.is_some_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "off" | "0" | "false" | "no"))
}

/// A stick with the radial dead zone taken out: inside `dead` it is at rest, outside the rest
/// of the way is stretched to 0..1, so the stick still reaches every value and does not jump
/// at the edge. The direction is kept (unlike a dead zone on each axis, which pulls slow
/// diagonal pushes to the axes).
pub fn dead_zone(stick: Vec2, dead: f32) -> Vec2 {
    let len = stick.length();
    if len <= dead || len <= f32::EPSILON {
        return Vec2::ZERO;
    }
    let out = ((len - dead) / (1.0 - dead)).min(1.0);
    stick / len * out
}

/// Whether a trigger at `value` (0..1) is its button down, given that it `was`: down from
/// [`TRIGGER_PRESS`], up again below [`TRIGGER_RELEASE`], so a finger resting near the middle
/// does not flicker.
pub fn trigger_down(was: bool, value: f32) -> bool {
    if was {
        value > TRIGGER_RELEASE
    } else {
        value >= TRIGGER_PRESS
    }
}

/// One of a pad's two sticks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stick {
    /// The left one.
    Left,
    /// The right one.
    Right,
}

/// One of a pad's two triggers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Trigger {
    /// LT / L2.
    Left,
    /// RT / R2.
    Right,
}

/// What a backend saw of one pad, before filtering.
#[derive(Clone, Debug, PartialEq)]
pub enum PadEvent {
    /// Plugged in, with the name the OS gives it (may be empty).
    Connected(String),
    /// Unplugged.
    Disconnected,
    /// A button other than a trigger went down or up.
    Button(GamepadButton, bool),
    /// The raw place of a stick along x (right positive), -1..1.
    StickX(Stick, f32),
    /// The raw place of a stick along y (**up** positive), -1..1.
    StickY(Stick, f32),
    /// How far a trigger is pulled, 0..1.
    Trigger(Trigger, f32),
}

/// The filter between a backend and [`Input`]: keeps each stick's raw place (the dead zone is
/// radial, so one axis needs the other) and turns triggers into buttons.
#[derive(Debug, Default)]
pub struct Pads {
    raw: HashMap<u32, [Vec2; 2]>,
}

impl Pads {
    /// `event` of the pad `id`, into `input`.
    pub fn apply(&mut self, id: u32, event: PadEvent, input: &mut Input) {
        match event {
            PadEvent::Connected(name) => {
                self.raw.insert(id, [Vec2::ZERO; 2]);
                input.inject_gamepad_connected_named(id, &name);
            }
            PadEvent::Disconnected => {
                self.raw.remove(&id);
                // Gone with what it held: no button stays down, no stick stays pushed.
                input.inject_gamepad_disconnected(id);
            }
            PadEvent::Button(button, pressed) => input.inject_gamepad_button(id, button, pressed),
            PadEvent::StickX(stick, v) => self.stick(id, stick, input, |raw| raw.x = v),
            PadEvent::StickY(stick, v) => self.stick(id, stick, input, |raw| raw.y = v),
            PadEvent::Trigger(trigger, value) => {
                let (axis, button) = match trigger {
                    Trigger::Left => (GamepadAxis::LeftTrigger, GamepadButton::LeftTrigger),
                    Trigger::Right => (GamepadAxis::RightTrigger, GamepadButton::RightTrigger),
                };
                let value = value.clamp(0.0, 1.0);
                let was = input.gamepad(id).is_some_and(|p| p.buttons.pressed(button));
                input.inject_gamepad_axis(id, axis, value);
                let down = trigger_down(was, value);
                if down != was {
                    input.inject_gamepad_button(id, button, down);
                }
            }
        }
    }

    /// `event` of the pad `id` while the window does not have the focus: a pad still comes
    /// and goes, and a stick's raw place is remembered (its next event after the focus is
    /// back needs both axes), but nothing it holds or pushes reaches `input`.
    pub fn apply_unfocused(&mut self, id: u32, event: PadEvent, input: &mut Input) {
        match event {
            PadEvent::Connected(_) | PadEvent::Disconnected => self.apply(id, event, input),
            PadEvent::StickX(stick, v) => self.raw.entry(id).or_default()[stick as usize].x = v,
            PadEvent::StickY(stick, v) => self.raw.entry(id).or_default()[stick as usize].y = v,
            PadEvent::Button(..) | PadEvent::Trigger(..) => {}
        }
    }

    fn stick(&mut self, id: u32, stick: Stick, input: &mut Input, set: impl FnOnce(&mut Vec2)) {
        let raw = &mut self.raw.entry(id).or_default()[stick as usize];
        set(raw);
        let v = dead_zone(raw.clamp(Vec2::NEG_ONE, Vec2::ONE), STICK_DEAD_ZONE);
        let (x, y) = match stick {
            Stick::Left => (GamepadAxis::LeftStickX, GamepadAxis::LeftStickY),
            Stick::Right => (GamepadAxis::RightStickX, GamepadAxis::RightStickY),
        };
        input.inject_gamepad_axis(id, x, v.x);
        input.inject_gamepad_axis(id, y, v.y);
    }
}

// ---- Android: key events and stray motion events (winit 0.30) ---------------------------------

/// Android's `KEYCODE_BUTTON_*` of a gamepad as a pad button (`None`: not one we use).
pub fn android_button(keycode: u32) -> Option<GamepadButton> {
    Some(match keycode {
        96 => GamepadButton::South,          // BUTTON_A
        97 => GamepadButton::East,           // BUTTON_B
        99 => GamepadButton::West,           // BUTTON_X
        100 => GamepadButton::North,         // BUTTON_Y
        102 => GamepadButton::LeftShoulder,  // BUTTON_L1
        103 => GamepadButton::RightShoulder, // BUTTON_R1
        104 => GamepadButton::LeftTrigger,   // BUTTON_L2
        105 => GamepadButton::RightTrigger,  // BUTTON_R2
        106 => GamepadButton::LeftThumb,     // BUTTON_THUMBL
        107 => GamepadButton::RightThumb,    // BUTTON_THUMBR
        108 => GamepadButton::Start,         // BUTTON_START
        109 => GamepadButton::Select,        // BUTTON_SELECT
        110 => GamepadButton::Mode,          // BUTTON_MODE
        _ => return None,
    })
}

/// The pads of an Android device, told apart by the input device `D` their events name
/// (winit's `DeviceId`). A device is a pad from its first pad **button**; the id it gets then
/// is the pad's id in [`Input`]. (Not from a stick: winit shows every motion event as a touch
/// without its source, and Android has other devices with absolute axes.)
#[derive(Debug)]
pub struct AndroidPads<D> {
    pads: Pads,
    ids: HashMap<D, u32>,
    /// The next pad's id: never one given before, so a pad that comes back is a new pad.
    next_id: u32,
    /// Devices a finger went down on: touch screens (and mice), never pads.
    fingers: HashSet<D>,
    /// The gamepads Android itself listed last ([`Self::os_pads`]): device number and name.
    os: Vec<(i32, String)>,
}

impl<D> Default for AndroidPads<D> {
    fn default() -> Self {
        Self {
            pads: Pads::default(),
            ids: HashMap::new(),
            next_id: 0,
            fingers: HashSet::new(),
            os: Vec::new(),
        }
    }
}

impl<D: Copy + Eq + Hash> AndroidPads<D> {
    fn id(&mut self, device: D, input: &mut Input) -> u32 {
        if let Some(id) = self.ids.get(&device) {
            return *id;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.ids.insert(device, id);
        // winit hides the device's number, so which of Android's pads this is cannot be asked:
        // it has a name only while Android lists one pad.
        let name = match self.os.as_slice() {
            [(_, name)] => name.clone(),
            _ => String::new(),
        };
        self.pads.apply(id, PadEvent::Connected(name), input);
        id
    }

    /// The gamepads Android lists now (device number and name; the runner asks once a
    /// second). When one that was listed is gone, a pad went away: winit does not say which
    /// device an event is from in Android's numbers, so **every** pad known here is
    /// disconnected with what it held (no stick stays pushed), and one still there is a pad
    /// again from its next button. Returns whether pads were disconnected.
    pub fn os_pads(&mut self, now: Vec<(i32, String)>, input: &mut Input) -> bool {
        let gone = self.os.iter().any(|(was, _)| !now.iter().any(|(n, _)| n == was));
        self.os = now;
        if !gone || self.ids.is_empty() {
            return false;
        }
        for (_, id) in self.ids.drain() {
            self.pads.apply(id, PadEvent::Disconnected, input);
        }
        true
    }

    /// A key event of no place with Android key code `keycode` from `device`: `true` when it
    /// was a pad's button (and is now in `input`).
    pub fn key(&mut self, device: D, keycode: u32, pressed: bool, input: &mut Input) -> bool {
        let Some(button) = android_button(keycode) else {
            return false;
        };
        let id = self.id(device, input);
        self.pads.apply(id, PadEvent::Button(button, pressed), input);
        true
    }

    /// An arrow key from `device`, as the d-pad `button` it would be: `true` when the device is
    /// a known pad, whose d-pad it then is. A keyboard's arrows stay keys.
    pub fn dpad_key(&mut self, device: D, button: GamepadButton, pressed: bool, input: &mut Input) -> bool {
        let Some(&id) = self.ids.get(&device) else {
            return false;
        };
        self.pads.apply(id, PadEvent::Button(button, pressed), input);
        true
    }

    /// A touch event as winit reports it. `true`: it was no finger but a motion event of a pad
    /// or another device with axes (a move from a device no finger ever went down on) and
    /// must not reach the touches. From a device already known as a pad by a button
    /// ([`Self::key`]) its place, when within -1..1, is the left stick (`AXIS_X`, `AXIS_Y`;
    /// Android's y points down, [`Input`]'s up); from any other device it is dropped: a
    /// sensor that reports axes must not become a controller that steers.
    pub fn touch(&mut self, device: D, phase: TouchPhase, position: Vec2, input: &mut Input) -> bool {
        match phase {
            TouchPhase::Started => {
                self.fingers.insert(device);
                false
            }
            TouchPhase::Moved if !self.fingers.contains(&device) => {
                if let Some(&id) = self.ids.get(&device)
                    && position.x.abs() <= 1.0
                    && position.y.abs() <= 1.0
                {
                    self.pads.apply(id, PadEvent::StickX(Stick::Left, position.x), input);
                    self.pads.apply(id, PadEvent::StickY(Stick::Left, -position.y), input);
                }
                true
            }
            _ => false,
        }
    }
}

// ---- desktop: gilrs ---------------------------------------------------------------------------

#[cfg(any(target_os = "windows", target_os = "macos", target_os = "linux"))]
pub(crate) use desktop::Backend;

#[cfg(any(target_os = "windows", target_os = "macos", target_os = "linux"))]
mod desktop {
    use super::{PadEvent, Pads, Stick, Trigger};
    use crate::input::{GamepadButton, Input};
    use gilrs::ev::filter::{Filter, Jitter, axis_dpad_to_button};
    use gilrs::{Axis, Button, EventType, Gilrs, GilrsBuilder};

    /// The gilrs context, made at the first poll. A failure is logged once and leaves no pads.
    #[derive(Default)]
    pub(crate) struct Backend {
        gilrs: Option<Gilrs>,
        tried: bool,
        pads: Pads,
    }

    impl Backend {
        /// Everything the pads did since the last call, into `input`. `active`: the window
        /// has the focus and the run is not a background one; otherwise only a pad's coming
        /// and going is taken (evdev and IOKit give a pad's input to every process, focused
        /// or not). gilrs is started at the first poll outside a background run.
        pub(crate) fn poll(&mut self, input: &mut Input, active: bool) {
            if !self.tried && !crate::background::background_mode() {
                self.tried = true;
                if super::env_turns_off(std::env::var(super::ENV_GAMEPAD).ok().as_deref()) {
                    tracing::info!(target: "gamepad", "{}: gamepads are off", super::ENV_GAMEPAD);
                    return;
                }
                // gilrs's own per-axis dead zone is left out: `Pads` takes a radial one.
                match GilrsBuilder::new().with_default_filters(false).build() {
                    Ok(gilrs) => {
                        for (id, pad) in gilrs.gamepads() {
                            let id = usize::from(id) as u32;
                            tracing::info!(target: "gamepad", "pad {id} is there: {}", pad.name());
                            self.pads.apply(id, PadEvent::Connected(pad.name().to_string()), input);
                        }
                        self.gilrs = Some(gilrs);
                    }
                    Err(e) => tracing::warn!(target: "gamepad", "no gamepad support: {e}"),
                }
            }
            let Some(gilrs) = &mut self.gilrs else {
                return;
            };
            let jitter = Jitter::new();
            while let Some(ev) = gilrs
                .next_event()
                .filter_ev(&axis_dpad_to_button, gilrs)
                .filter_ev(&jitter, gilrs)
            {
                let id = usize::from(ev.id) as u32;
                let event = match ev.event {
                    EventType::Connected => {
                        let name = gilrs.gamepad(ev.id).name().to_string();
                        tracing::info!(target: "gamepad", "pad {id} connected: {name}");
                        Some(PadEvent::Connected(name))
                    }
                    EventType::Disconnected => {
                        tracing::info!(target: "gamepad", "pad {id} disconnected");
                        Some(PadEvent::Disconnected)
                    }
                    other => pad_event(&other),
                };
                match event {
                    Some(event) if active => self.pads.apply(id, event, input),
                    Some(event) => self.pads.apply_unfocused(id, event, input),
                    None => {}
                }
            }
        }
    }

    /// A gilrs event as ours (`None`: nothing we use). The triggers come by their value
    /// (`ButtonChanged`), every other button by its press and release; gilrs's sticks already
    /// have up positive.
    fn pad_event(event: &EventType) -> Option<PadEvent> {
        match *event {
            EventType::ButtonChanged(b, v, _) => trigger(b, v),
            EventType::ButtonPressed(b, _) => button(b).map(|b| PadEvent::Button(b, true)),
            EventType::ButtonReleased(b, _) => button(b).map(|b| PadEvent::Button(b, false)),
            EventType::AxisChanged(a, v, _) => axis(a, v),
            _ => None,
        }
    }

    /// A button's value: only the triggers' is used.
    fn trigger(b: Button, v: f32) -> Option<PadEvent> {
        match b {
            Button::LeftTrigger2 => Some(PadEvent::Trigger(Trigger::Left, v)),
            Button::RightTrigger2 => Some(PadEvent::Trigger(Trigger::Right, v)),
            _ => None,
        }
    }

    fn axis(a: Axis, v: f32) -> Option<PadEvent> {
        match a {
            Axis::LeftStickX => Some(PadEvent::StickX(Stick::Left, v)),
            Axis::LeftStickY => Some(PadEvent::StickY(Stick::Left, v)),
            Axis::RightStickX => Some(PadEvent::StickX(Stick::Right, v)),
            Axis::RightStickY => Some(PadEvent::StickY(Stick::Right, v)),
            _ => None,
        }
    }

    /// A gilrs button as ours. gilrs calls the shoulder buttons `LeftTrigger`/`RightTrigger`
    /// and the triggers `...Trigger2`; the triggers are not here (they come by value).
    fn button(b: Button) -> Option<GamepadButton> {
        Some(match b {
            Button::South => GamepadButton::South,
            Button::East => GamepadButton::East,
            Button::North => GamepadButton::North,
            Button::West => GamepadButton::West,
            Button::LeftTrigger => GamepadButton::LeftShoulder,
            Button::RightTrigger => GamepadButton::RightShoulder,
            Button::Select => GamepadButton::Select,
            Button::Start => GamepadButton::Start,
            Button::Mode => GamepadButton::Mode,
            Button::LeftThumb => GamepadButton::LeftThumb,
            Button::RightThumb => GamepadButton::RightThumb,
            Button::DPadUp => GamepadButton::DPadUp,
            Button::DPadDown => GamepadButton::DPadDown,
            Button::DPadLeft => GamepadButton::DPadLeft,
            Button::DPadRight => GamepadButton::DPadRight,
            _ => return None,
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Every gilrs value we use becomes the right event of ours; the rest nothing. (An
        /// `EventType` with a button cannot be built here: gilrs's `Code` has no public
        /// constructor.)
        #[test]
        fn gilrs_values_map_to_pad_events() {
            assert_eq!(
                trigger(Button::LeftTrigger2, 0.7),
                Some(PadEvent::Trigger(Trigger::Left, 0.7))
            );
            assert_eq!(
                trigger(Button::RightTrigger2, 0.2),
                Some(PadEvent::Trigger(Trigger::Right, 0.2))
            );
            assert_eq!(trigger(Button::South, 1.0), None, "a plain button comes by its press");
            assert_eq!(trigger(Button::LeftTrigger, 1.0), None, "the shoulder button too");
            assert_eq!(axis(Axis::LeftStickX, -0.5), Some(PadEvent::StickX(Stick::Left, -0.5)));
            assert_eq!(axis(Axis::LeftStickY, 0.25), Some(PadEvent::StickY(Stick::Left, 0.25)));
            assert_eq!(axis(Axis::RightStickX, 1.0), Some(PadEvent::StickX(Stick::Right, 1.0)));
            assert_eq!(axis(Axis::RightStickY, 0.9), Some(PadEvent::StickY(Stick::Right, 0.9)));
            for none in [Axis::LeftZ, Axis::RightZ, Axis::DPadX, Axis::DPadY, Axis::Unknown] {
                assert_eq!(axis(none, 0.9), None, "{none:?}");
            }
            assert_eq!(pad_event(&EventType::Dropped), None);
            assert_eq!(pad_event(&EventType::Connected), None, "the poll names the pad itself");
        }

        #[test]
        fn every_named_button_has_a_place() {
            let ours = [
                (Button::South, GamepadButton::South),
                (Button::East, GamepadButton::East),
                (Button::North, GamepadButton::North),
                (Button::West, GamepadButton::West),
                (Button::LeftTrigger, GamepadButton::LeftShoulder),
                (Button::RightTrigger, GamepadButton::RightShoulder),
                (Button::Select, GamepadButton::Select),
                (Button::Start, GamepadButton::Start),
                (Button::Mode, GamepadButton::Mode),
                (Button::LeftThumb, GamepadButton::LeftThumb),
                (Button::RightThumb, GamepadButton::RightThumb),
                (Button::DPadUp, GamepadButton::DPadUp),
                (Button::DPadDown, GamepadButton::DPadDown),
                (Button::DPadLeft, GamepadButton::DPadLeft),
                (Button::DPadRight, GamepadButton::DPadRight),
            ];
            for (theirs, want) in ours {
                assert_eq!(button(theirs), Some(want), "{theirs:?}");
            }
            for none in [
                Button::LeftTrigger2,
                Button::RightTrigger2,
                Button::C,
                Button::Z,
                Button::Unknown,
            ] {
                assert_eq!(button(none), None, "{none:?}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::GamepadChange;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn dead_zone_rests_inside_and_reaches_one() {
        assert_eq!(dead_zone(Vec2::ZERO, 0.15), Vec2::ZERO);
        assert_eq!(dead_zone(Vec2::new(0.1, 0.1), 0.15), Vec2::ZERO, "length 0.141");
        assert_eq!(
            dead_zone(Vec2::new(0.15, 0.0), 0.15),
            Vec2::ZERO,
            "the edge itself rests"
        );
        let full = dead_zone(Vec2::new(1.0, 0.0), 0.15);
        assert!(close(full.x, 1.0) && full.y == 0.0);
        // Half way between the edge and full is half.
        let half = dead_zone(Vec2::new(0.0, -0.575), 0.15);
        assert!(close(half.y, -0.5) && half.x == 0.0, "{half:?}");
        // Just outside the edge it starts near 0: no jump.
        assert!(dead_zone(Vec2::new(0.16, 0.0), 0.15).x < 0.02);
        // A corner of a square stick (length 1.41) is clamped to the circle.
        let corner = dead_zone(Vec2::new(1.0, 1.0), 0.15);
        assert!(close(corner.length(), 1.0), "{corner:?}");
    }

    #[test]
    fn dead_zone_keeps_the_direction() {
        // Radial, not per axis: a slow diagonal push stays diagonal.
        let v = dead_zone(Vec2::new(0.2, 0.12), 0.15);
        assert!(v.length() > 0.0);
        assert!(close(v.y / v.x, 0.12 / 0.2), "{v:?}");
    }

    #[test]
    fn trigger_has_hysteresis() {
        assert!(!trigger_down(false, 0.49));
        assert!(trigger_down(false, 0.5));
        assert!(trigger_down(true, 0.45), "held between the two marks");
        assert!(!trigger_down(false, 0.45), "but not pressed there");
        assert!(!trigger_down(true, 0.4));
    }

    #[test]
    fn stick_events_are_filtered_as_a_pair() {
        let mut pads = Pads::default();
        let mut input = Input::default();
        pads.apply(3, PadEvent::Connected("Test Pad".into()), &mut input);
        assert_eq!(input.gamepad_changes(), &[GamepadChange::Connected(3)]);
        assert_eq!(input.gamepad(3).expect("pad").name(), "Test Pad");
        // x alone inside the dead zone: rest.
        pads.apply(3, PadEvent::StickX(Stick::Left, 0.1), &mut input);
        assert_eq!(input.any_gamepad_axis(GamepadAxis::LeftStickX), 0.0);
        // With y the length (0.608) is outside: both axes move, the direction is the raw one.
        pads.apply(3, PadEvent::StickY(Stick::Left, 0.6), &mut input);
        let pad = input.gamepad(3).expect("pad");
        let (x, y) = (pad.axis(GamepadAxis::LeftStickX), pad.axis(GamepadAxis::LeftStickY));
        assert!(x > 0.0 && y > 0.0);
        assert!(close(x / y, 0.1 / 0.6), "{x} {y}");
        assert_eq!(pad.axis(GamepadAxis::RightStickX), 0.0, "the right stick is its own");
        // Back to rest.
        pads.apply(3, PadEvent::StickX(Stick::Left, 0.0), &mut input);
        pads.apply(3, PadEvent::StickY(Stick::Left, 0.05), &mut input);
        let pad = input.gamepad(3).expect("pad");
        assert_eq!(pad.axis(GamepadAxis::LeftStickX), 0.0);
        assert_eq!(pad.axis(GamepadAxis::LeftStickY), 0.0);
    }

    #[test]
    fn trigger_is_an_axis_and_a_button() {
        let mut pads = Pads::default();
        let mut input = Input::default();
        pads.apply(0, PadEvent::Trigger(Trigger::Right, 0.3), &mut input);
        assert!(close(input.any_gamepad_axis(GamepadAxis::RightTrigger), 0.3));
        assert!(!input.any_gamepad_pressed(GamepadButton::RightTrigger));
        pads.apply(0, PadEvent::Trigger(Trigger::Right, 0.6), &mut input);
        assert!(input.any_gamepad_just_pressed(GamepadButton::RightTrigger));
        input.end_frame();
        pads.apply(0, PadEvent::Trigger(Trigger::Right, 0.45), &mut input);
        assert!(input.any_gamepad_pressed(GamepadButton::RightTrigger), "still held");
        assert!(
            !input.any_gamepad_just_pressed(GamepadButton::RightTrigger),
            "pressed once"
        );
        pads.apply(0, PadEvent::Trigger(Trigger::Right, 0.1), &mut input);
        assert!(!input.any_gamepad_pressed(GamepadButton::RightTrigger));
        assert!(!input.any_gamepad_pressed(GamepadButton::LeftTrigger));
    }

    /// A pad pulled out while a button is held and a stick pushed leaves nothing behind, and
    /// the same id plugged in again starts clean.
    #[test]
    fn disconnect_takes_everything_held() {
        let mut pads = Pads::default();
        let mut input = Input::default();
        pads.apply(1, PadEvent::Connected("A".into()), &mut input);
        pads.apply(1, PadEvent::Button(GamepadButton::South, true), &mut input);
        pads.apply(1, PadEvent::StickX(Stick::Left, 1.0), &mut input);
        pads.apply(1, PadEvent::Trigger(Trigger::Left, 1.0), &mut input);
        input.end_frame();
        assert!(input.any_gamepad_pressed(GamepadButton::South));
        pads.apply(1, PadEvent::Disconnected, &mut input);
        assert_eq!(input.gamepad_changes(), &[GamepadChange::Disconnected(1)]);
        assert!(input.gamepad(1).is_none());
        assert_eq!(input.gamepad_ids().count(), 0);
        assert!(!input.any_gamepad_pressed(GamepadButton::South));
        assert!(!input.any_gamepad_pressed(GamepadButton::LeftTrigger));
        assert_eq!(input.any_gamepad_axis(GamepadAxis::LeftStickX), 0.0);
        input.end_frame();
        assert!(input.gamepad_changes().is_empty());
        pads.apply(1, PadEvent::Connected("A".into()), &mut input);
        let pad = input.gamepad(1).expect("back");
        assert!(!pad.buttons.pressed(GamepadButton::South));
        assert_eq!(pad.axis(GamepadAxis::LeftStickX), 0.0);
        // The raw stick was forgotten too: y alone decides now.
        pads.apply(1, PadEvent::StickY(Stick::Left, 0.1), &mut input);
        assert_eq!(input.any_gamepad_axis(GamepadAxis::LeftStickX), 0.0);
        assert_eq!(input.any_gamepad_axis(GamepadAxis::LeftStickY), 0.0);
        // A second unplug of a pad that is not there says nothing.
        input.end_frame();
        pads.apply(7, PadEvent::Disconnected, &mut input);
        assert!(input.gamepad_changes().is_empty());
    }

    #[test]
    fn focus_loss_rests_the_pad() {
        let mut pads = Pads::default();
        let mut input = Input::default();
        pads.apply(0, PadEvent::Button(GamepadButton::West, true), &mut input);
        pads.apply(0, PadEvent::StickX(Stick::Left, 1.0), &mut input);
        input.release_all();
        assert!(!input.any_gamepad_pressed(GamepadButton::West));
        assert_eq!(input.any_gamepad_axis(GamepadAxis::LeftStickX), 0.0);
        assert!(input.gamepad(0).is_some(), "still connected");
    }

    /// A window without the focus: the pad's buttons, triggers and sticks do nothing (Linux
    /// and macOS deliver them all the same), it still comes and goes, and the stick's place
    /// is not stale when the focus is back.
    #[test]
    fn an_unfocused_window_does_not_obey_the_pad() {
        let mut pads = Pads::default();
        let mut input = Input::default();
        pads.apply_unfocused(2, PadEvent::Connected("Pad".into()), &mut input);
        assert_eq!(input.gamepad_changes(), &[GamepadChange::Connected(2)]);
        pads.apply(2, PadEvent::StickX(Stick::Left, 1.0), &mut input);
        // The focus goes: everything rests, and what the pad does now is not taken.
        input.release_all();
        pads.apply_unfocused(2, PadEvent::Button(GamepadButton::South, true), &mut input);
        pads.apply_unfocused(2, PadEvent::Trigger(Trigger::Right, 1.0), &mut input);
        pads.apply_unfocused(2, PadEvent::StickY(Stick::Right, 1.0), &mut input);
        pads.apply_unfocused(2, PadEvent::StickX(Stick::Left, 0.0), &mut input);
        let pad = input.gamepad(2).expect("still connected");
        assert!(pad.buttons.iter_pressed().next().is_none());
        for axis in [
            GamepadAxis::LeftStickX,
            GamepadAxis::RightStickY,
            GamepadAxis::RightTrigger,
        ] {
            assert_eq!(pad.axis(axis), 0.0, "{axis:?}");
        }
        // The focus is back and the left stick goes up: x was let go meanwhile, so it is 0.
        pads.apply(2, PadEvent::StickY(Stick::Left, 1.0), &mut input);
        let pad = input.gamepad(2).expect("pad");
        assert_eq!(
            pad.axis(GamepadAxis::LeftStickX),
            0.0,
            "not the x of before the focus loss"
        );
        assert!(close(pad.axis(GamepadAxis::LeftStickY), 1.0));
        // Pulled out while unfocused: gone.
        pads.apply_unfocused(2, PadEvent::Disconnected, &mut input);
        assert!(input.gamepad(2).is_none());
    }

    /// Two pads off their rest: the stick read is one pad's, whole.
    #[test]
    fn the_stick_of_the_pad_pushed_furthest_is_whole() {
        let mut input = Input::default();
        let (x, y) = (GamepadAxis::LeftStickX, GamepadAxis::LeftStickY);
        assert_eq!(input.any_gamepad_stick(x, y), Vec2::ZERO);
        input.inject_gamepad_axis(0, x, 0.9);
        input.inject_gamepad_axis(0, y, 0.1);
        input.inject_gamepad_axis(1, x, -0.2);
        input.inject_gamepad_axis(1, y, -0.6);
        assert_eq!(
            input.any_gamepad_stick(x, y),
            Vec2::new(0.9, 0.1),
            "pad 0's, not (0.9, -0.6)"
        );
        input.inject_gamepad_axis(0, x, 0.0);
        assert_eq!(input.any_gamepad_stick(x, y), Vec2::new(-0.2, -0.6));
    }

    #[test]
    fn the_env_switch() {
        for off in ["off", "0", "false", "no", " OFF "] {
            assert!(env_turns_off(Some(off)), "{off:?}");
        }
        for on in ["", "1", "on", "yes"] {
            assert!(!env_turns_off(Some(on)), "{on:?}");
        }
        assert!(!env_turns_off(None));
    }

    /// Android lists its gamepads once a second: a pad that is no longer listed takes every
    /// pad known with it (a stick held when a Bluetooth pad dropped does not fly on), the pad
    /// has the OS's name while it is the only one, and a pad that comes back is a new one.
    #[test]
    fn android_a_pad_that_went_away_is_disconnected() {
        let mut pads = AndroidPads::<i32>::default();
        let mut input = Input::default();
        assert!(!pads.os_pads(vec![(41, "Xbox Wireless Controller".into())], &mut input));
        assert_eq!(
            input.gamepad_ids().count(),
            0,
            "listed is not yet a pad: a button makes it one"
        );
        assert!(pads.key(7, 96, true, &mut input));
        assert_eq!(input.gamepad(0).expect("pad").name(), "Xbox Wireless Controller");
        assert!(pads.touch(7, TouchPhase::Moved, Vec2::new(1.0, 0.0), &mut input));
        assert!(close(input.any_gamepad_axis(GamepadAxis::LeftStickX), 1.0));
        input.end_frame();
        // Still listed (and another device came): nothing changes.
        assert!(!pads.os_pads(
            vec![(41, "Xbox Wireless Controller".into()), (52, "Other".into())],
            &mut input
        ));
        assert!(close(input.any_gamepad_axis(GamepadAxis::LeftStickX), 1.0));
        // The pad drops with the stick held.
        assert!(pads.os_pads(vec![(52, "Other".into())], &mut input));
        assert_eq!(input.gamepad_changes(), &[GamepadChange::Disconnected(0)]);
        assert_eq!(input.gamepad_ids().count(), 0);
        assert_eq!(input.any_gamepad_axis(GamepadAxis::LeftStickX), 0.0);
        assert!(!input.any_gamepad_pressed(GamepadButton::South));
        input.end_frame();
        // Its late stick moves are no pad's; it is a pad again from a button, with a new id.
        assert!(pads.touch(7, TouchPhase::Moved, Vec2::new(1.0, 0.0), &mut input));
        assert_eq!(input.gamepad_ids().count(), 0);
        assert!(!pads.os_pads(vec![(52, "Other".into()), (60, "Pad".into())], &mut input));
        assert!(pads.key(8, 97, true, &mut input));
        assert_eq!(input.gamepad_changes(), &[GamepadChange::Connected(1)]);
        assert_eq!(
            input.gamepad(1).expect("pad").name(),
            "",
            "two listed: which one is not known"
        );
        // The other device goes: the pad known goes with it (which went is not known) ...
        assert!(pads.os_pads(vec![(60, "Pad".into())], &mut input));
        // ... and with no pad known, a device going away changes nothing.
        assert!(!pads.os_pads(Vec::new(), &mut input));
    }

    /// A motion event within -1..1 from a device that never sent a pad button (a sensor with
    /// axes) is no controller: it connects nothing and steers nothing, and is no finger.
    #[test]
    fn android_axes_alone_are_no_pad() {
        let mut pads = AndroidPads::<i32>::default();
        let mut input = Input::default();
        assert!(
            pads.touch(3, TouchPhase::Moved, Vec2::new(0.6, -0.4), &mut input),
            "kept from the touches"
        );
        assert_eq!(input.gamepad_ids().count(), 0);
        assert!(input.gamepad_changes().is_empty());
        assert_eq!(input.any_gamepad_axis(GamepadAxis::LeftStickX), 0.0);
    }

    #[test]
    fn android_key_codes() {
        assert_eq!(android_button(96), Some(GamepadButton::South));
        assert_eq!(android_button(97), Some(GamepadButton::East));
        assert_eq!(android_button(98), None, "BUTTON_C");
        assert_eq!(android_button(99), Some(GamepadButton::West));
        assert_eq!(android_button(100), Some(GamepadButton::North));
        assert_eq!(android_button(101), None, "BUTTON_Z");
        assert_eq!(android_button(102), Some(GamepadButton::LeftShoulder));
        assert_eq!(android_button(103), Some(GamepadButton::RightShoulder));
        assert_eq!(android_button(104), Some(GamepadButton::LeftTrigger));
        assert_eq!(android_button(105), Some(GamepadButton::RightTrigger));
        assert_eq!(android_button(106), Some(GamepadButton::LeftThumb));
        assert_eq!(android_button(107), Some(GamepadButton::RightThumb));
        assert_eq!(android_button(108), Some(GamepadButton::Start));
        assert_eq!(android_button(109), Some(GamepadButton::Select));
        assert_eq!(android_button(110), Some(GamepadButton::Mode));
        assert_eq!(android_button(4), None, "Back");
        assert_eq!(android_button(111), None, "Escape");
    }

    #[test]
    fn android_first_button_connects_the_pad() {
        let mut pads = AndroidPads::<i32>::default();
        let mut input = Input::default();
        assert!(!pads.key(7, 29, true, &mut input), "KEYCODE_A of a keyboard");
        assert_eq!(input.gamepad_ids().count(), 0);
        // Arrow keys of a device that is no pad stay keys.
        assert!(!pads.dpad_key(7, GamepadButton::DPadUp, true, &mut input));
        assert!(pads.key(7, 96, true, &mut input));
        assert_eq!(input.gamepad_changes(), &[GamepadChange::Connected(0)]);
        assert!(input.any_gamepad_just_pressed(GamepadButton::South));
        assert!(pads.key(7, 96, false, &mut input));
        assert!(!input.any_gamepad_pressed(GamepadButton::South));
        // Now its arrows are the d-pad.
        assert!(pads.dpad_key(7, GamepadButton::DPadUp, true, &mut input));
        assert!(input.any_gamepad_pressed(GamepadButton::DPadUp));
        // A second device is a second pad.
        assert!(pads.key(9, 108, true, &mut input));
        assert!(
            input
                .gamepad(1)
                .is_some_and(|p| p.buttons.pressed(GamepadButton::Start))
        );
        assert!(
            !input
                .gamepad(0)
                .is_some_and(|p| p.buttons.pressed(GamepadButton::Start))
        );
    }

    /// winit reports a stick's motion event as a touch move with pointer id 0, the id of the
    /// first finger: it must not drag that finger to the corner, and it is the left stick.
    #[test]
    fn android_stray_move_is_the_stick_not_a_finger() {
        const SCREEN: i32 = 2;
        const PAD: i32 = 5;
        let mut pads = AndroidPads::<i32>::default();
        let mut input = Input::default();
        let touch = |pads: &mut AndroidPads<i32>, input: &mut Input, dev, phase, p: Vec2| {
            if !pads.touch(dev, phase, p, input) {
                input.inject_touch(0, phase, p);
            }
        };
        // A finger goes down and moves: a touch as ever.
        touch(
            &mut pads,
            &mut input,
            SCREEN,
            TouchPhase::Started,
            Vec2::new(400.0, 300.0),
        );
        touch(
            &mut pads,
            &mut input,
            SCREEN,
            TouchPhase::Moved,
            Vec2::new(410.0, 300.0),
        );
        assert_eq!(input.touch(0).expect("finger").position, Vec2::new(410.0, 300.0));
        assert_eq!(input.gamepad_ids().count(), 0);
        // The pad is a pad from its first button.
        assert!(pads.key(PAD, 96, true, &mut input));
        assert!(pads.key(PAD, 96, false, &mut input));
        assert_eq!(input.gamepad_changes(), &[GamepadChange::Connected(0)]);
        // The stick moves: right and up (Android's y points down).
        touch(&mut pads, &mut input, PAD, TouchPhase::Moved, Vec2::new(1.0, -0.0));
        assert_eq!(
            input.touch(0).expect("finger").position,
            Vec2::new(410.0, 300.0),
            "the finger stays where it is"
        );
        assert!(close(input.any_gamepad_axis(GamepadAxis::LeftStickX), 1.0));
        touch(&mut pads, &mut input, PAD, TouchPhase::Moved, Vec2::new(0.0, -0.8));
        assert!(input.any_gamepad_axis(GamepadAxis::LeftStickY) > 0.7, "up is positive");
        assert_eq!(input.any_gamepad_axis(GamepadAxis::LeftStickX), 0.0);
        // Rest.
        touch(&mut pads, &mut input, PAD, TouchPhase::Moved, Vec2::ZERO);
        assert_eq!(input.any_gamepad_axis(GamepadAxis::LeftStickY), 0.0);
        // A stray move far outside -1..1 (some other pointer device) is dropped, not a stick.
        touch(&mut pads, &mut input, 11, TouchPhase::Moved, Vec2::new(500.0, 20.0));
        assert_eq!(input.gamepad_ids().count(), 1);
        assert_eq!(input.touch(0).expect("finger").position, Vec2::new(410.0, 300.0));
        // The finger still moves and lifts.
        touch(&mut pads, &mut input, SCREEN, TouchPhase::Moved, Vec2::new(0.5, 0.5));
        assert_eq!(
            input.touch(0).expect("finger").position,
            Vec2::new(0.5, 0.5),
            "a finger in the screen's corner is a finger"
        );
        touch(&mut pads, &mut input, SCREEN, TouchPhase::Ended, Vec2::new(0.5, 0.5));
        assert!(input.touch(0).is_none());
    }
}
