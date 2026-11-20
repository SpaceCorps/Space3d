//! Scripted input: timed [`InputEvent`]s injected into `space3d_platform::Input`, gesture
//! builders (taps, holds, clicks, drags, swipes, pinches, gamepad), RON persistence and an
//! [`InputRecorder`] that turns a live session back into a replayable [`InputScript`].
//!
//! Tick semantics: an event at tick `t` is injected right before the frame that runs fixed
//! tick `t` (relative to the moment the script starts), so `PreUpdate`, every `FixedUpdate`
//! system and `Update` of that frame see it. A script lasts [`InputScript::duration`] ticks.
//!
//! ```
//! use space3d_platform::KeyCode;
//! use space3d_test::{InputEvent, InputScript};
//! let script = InputScript::new()
//!     .hold_key(0, KeyCode::ArrowRight, 30)
//!     .tap_key(40, KeyCode::Space);
//! assert_eq!(script.duration(), 42);
//! let ron = script.to_ron().unwrap();
//! assert_eq!(InputScript::from_ron(&ron).unwrap(), script);
//! assert_eq!(script.events()[0].event, InputEvent::KeyDown(KeyCode::ArrowRight));
//! ```

use crate::keys::ALL_KEYS;
use serde::{Deserialize, Serialize};
use space3d_core::glam::Vec2;
use space3d_core::{App, Plugin, Stage, Time};
use space3d_platform::{GamepadAxis, GamepadButton, Input, KeyCode, MouseButton, MouseButtonCause, TouchPhase};
use std::collections::{HashMap, HashSet};
use std::path::Path;

#[derive(Serialize, Deserialize)]
#[serde(remote = "MouseButton")]
enum MouseButtonDef {
    Left,
    Right,
    Middle,
    Back,
    Forward,
    Other(u16),
}

/// Wire form of a [`MouseButtonCause`] (a foreign `#[non_exhaustive]` enum: serde's remote
/// derive cannot cover it). A cause this version does not know is written as the nearest one.
#[derive(Serialize, Deserialize)]
enum MouseButtonCauseWire {
    Device,
    FocusLost,
    OsReconciled,
}

mod mouse_button_cause {
    use super::{MouseButtonCause, MouseButtonCauseWire};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(cause: &MouseButtonCause, s: S) -> Result<S::Ok, S::Error> {
        match cause {
            MouseButtonCause::Device => MouseButtonCauseWire::Device,
            MouseButtonCause::OsReconciled => MouseButtonCauseWire::OsReconciled,
            // `FocusLost`, and any cancel added later.
            _ if cause.is_cancel() => MouseButtonCauseWire::FocusLost,
            _ => MouseButtonCauseWire::Device,
        }
        .serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<MouseButtonCause, D::Error> {
        Ok(match MouseButtonCauseWire::deserialize(d)? {
            MouseButtonCauseWire::Device => MouseButtonCause::Device,
            MouseButtonCauseWire::FocusLost => MouseButtonCause::FocusLost,
            MouseButtonCauseWire::OsReconciled => MouseButtonCause::OsReconciled,
        })
    }
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "TouchPhase")]
enum TouchPhaseDef {
    Started,
    Moved,
    Ended,
    Cancelled,
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "GamepadButton")]
enum GamepadButtonDef {
    South,
    East,
    North,
    West,
    LeftShoulder,
    RightShoulder,
    Start,
    Select,
    DPadUp,
    DPadDown,
    DPadLeft,
    DPadRight,
    LeftTrigger,
    RightTrigger,
    LeftThumb,
    RightThumb,
    Mode,
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "GamepadAxis")]
enum GamepadAxisDef {
    LeftStickX,
    LeftStickY,
    RightStickX,
    RightStickY,
    LeftTrigger,
    RightTrigger,
}

const MOUSE_BUTTONS: [MouseButton; 5] = [
    MouseButton::Left,
    MouseButton::Right,
    MouseButton::Middle,
    MouseButton::Back,
    MouseButton::Forward,
];

const GAMEPAD_BUTTONS: [GamepadButton; 17] = [
    GamepadButton::South,
    GamepadButton::East,
    GamepadButton::North,
    GamepadButton::West,
    GamepadButton::LeftShoulder,
    GamepadButton::RightShoulder,
    GamepadButton::Start,
    GamepadButton::Select,
    GamepadButton::DPadUp,
    GamepadButton::DPadDown,
    GamepadButton::DPadLeft,
    GamepadButton::DPadRight,
    GamepadButton::LeftTrigger,
    GamepadButton::RightTrigger,
    GamepadButton::LeftThumb,
    GamepadButton::RightThumb,
    GamepadButton::Mode,
];

const GAMEPAD_AXES: [GamepadAxis; 6] = [
    GamepadAxis::LeftStickX,
    GamepadAxis::LeftStickY,
    GamepadAxis::RightStickX,
    GamepadAxis::RightStickY,
    GamepadAxis::LeftTrigger,
    GamepadAxis::RightTrigger,
];

/// One input injection. Positions are physical pixels (origin top-left).
///
/// `#[non_exhaustive]`: variants get added ([`InputEvent::MouseCancel`] was), so a `match` needs
/// a wildcard arm. Scripts written before still load and play as they did.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum InputEvent {
    /// Key goes down.
    KeyDown(KeyCode),
    /// Key goes up.
    KeyUp(KeyCode),
    /// Key goes down and up within one frame (both edges visible in that frame).
    KeyTap(KeyCode),
    /// Committed text.
    Text(String),
    /// Cursor moves to a position (produces a mouse delta).
    MouseMove(Vec2),
    /// Raw relative mouse motion (grabbed cursor).
    MouseMotion(Vec2),
    /// Cursor leaves the window.
    MouseLeft,
    /// Mouse button goes down.
    MouseDown(#[serde(with = "MouseButtonDef")] MouseButton),
    /// Mouse button goes up.
    MouseUp(#[serde(with = "MouseButtonDef")] MouseButton),
    /// Mouse button is released *without a release event of the device*: the window lost the
    /// focus ([`MouseButtonCause::FocusLost`]) or the OS showed the button up
    /// ([`MouseButtonCause::OsReconciled`]). What [`InputRecorder`] records for such a release
    /// instead of a plain [`InputEvent::MouseUp`], and what a script plays back with
    /// `Input::release_mouse_button`: the UI registers no click and no drop, and
    /// `MouseButtonEvent::cancelled()` is `true`. With [`MouseButtonCause::Device`] it is a plain
    /// release.
    MouseCancel {
        /// The button.
        #[serde(with = "MouseButtonDef")]
        button: MouseButton,
        /// Why it was released.
        #[serde(with = "mouse_button_cause")]
        cause: MouseButtonCause,
    },
    /// Scroll in lines.
    Wheel(Vec2),
    /// Touch event for one finger.
    Touch {
        /// Finger id.
        id: u64,
        /// Phase.
        #[serde(with = "TouchPhaseDef")]
        phase: TouchPhase,
        /// Position.
        position: Vec2,
    },
    /// Gamepad connected.
    GamepadConnected(u32),
    /// Gamepad disconnected.
    GamepadDisconnected(u32),
    /// Gamepad button changes.
    GamepadButton {
        /// Pad id.
        pad: u32,
        /// Button.
        #[serde(with = "GamepadButtonDef")]
        button: GamepadButton,
        /// Down (`true`) or up.
        pressed: bool,
    },
    /// Gamepad axis value.
    GamepadAxis {
        /// Pad id.
        pad: u32,
        /// Axis.
        #[serde(with = "GamepadAxisDef")]
        axis: GamepadAxis,
        /// Value (-1..1 sticks, 0..1 triggers).
        value: f32,
    },
    /// Window resized (physical pixels; used by `Binding::TouchRegion`).
    WindowSize(Vec2),
    /// Release every held key and button (focus loss).
    ReleaseAll,
}

impl InputEvent {
    /// Inject this event into `input`.
    pub fn apply(&self, input: &mut Input) {
        match self {
            InputEvent::KeyDown(k) => input.inject_key_down(*k),
            InputEvent::KeyUp(k) => input.inject_key_up(*k),
            InputEvent::KeyTap(k) => input.inject_key_tap(*k),
            InputEvent::Text(t) => input.inject_text(t),
            InputEvent::MouseMove(p) => input.inject_mouse_move(*p),
            InputEvent::MouseMotion(d) => input.inject_mouse_motion(*d),
            InputEvent::MouseLeft => input.inject_mouse_left(),
            InputEvent::MouseDown(b) => input.inject_mouse_button_down(*b),
            InputEvent::MouseUp(b) => input.inject_mouse_button_up(*b),
            InputEvent::MouseCancel { button, cause } => input.release_mouse_button(*button, *cause),
            InputEvent::Wheel(d) => input.inject_wheel(*d),
            InputEvent::Touch { id, phase, position } => input.inject_touch(*id, *phase, *position),
            InputEvent::GamepadConnected(id) => input.inject_gamepad_connected(*id),
            InputEvent::GamepadDisconnected(id) => input.inject_gamepad_disconnected(*id),
            InputEvent::GamepadButton { pad, button, pressed } => input.inject_gamepad_button(*pad, *button, *pressed),
            InputEvent::GamepadAxis { pad, axis, value } => input.inject_gamepad_axis(*pad, *axis, *value),
            InputEvent::WindowSize(s) => input.set_window_size(*s),
            InputEvent::ReleaseAll => input.release_all(),
        }
    }
}

/// An [`InputEvent`] at a tick relative to the start of its script.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TimedInput {
    /// Tick offset from the script start.
    pub tick: u64,
    /// The event.
    pub event: InputEvent,
}

/// Errors loading or saving an [`InputScript`].
#[derive(Debug, thiserror::Error)]
pub enum ScriptError {
    /// File access failed.
    #[error("input script I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// RON (de)serialization failed.
    #[error("input script RON error: {0}")]
    Ron(String),
}

/// A replayable list of timed input events, kept sorted by tick (stable for equal ticks).
///
/// Build it with the gesture helpers, load it from RON, or record it with [`InputRecorder`].
/// Play it with [`crate::TestApp::play`] (steps [`Self::duration`] ticks) or
/// [`crate::TestApp::schedule`] (queues without stepping).
///
/// Deserializing (RON, JSON, or as a field of a larger config) re-sorts the events by tick.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(from = "UnsortedInputScript")]
pub struct InputScript {
    events: Vec<TimedInput>,
}

/// Wire form of [`InputScript`]; events may come in any order.
#[derive(Deserialize)]
#[serde(rename = "InputScript")]
struct UnsortedInputScript {
    events: Vec<TimedInput>,
}

impl From<UnsortedInputScript> for InputScript {
    fn from(raw: UnsortedInputScript) -> Self {
        let mut events = raw.events;
        // Stable: events of the same tick keep their order.
        events.sort_by_key(|e| e.tick);
        Self { events }
    }
}

fn lerp(a: Vec2, b: Vec2, t: f32) -> Vec2 {
    a + (b - a) * t
}

impl InputScript {
    /// Empty script.
    pub fn new() -> Self {
        Self::default()
    }

    /// Events in playback order.
    pub fn events(&self) -> &[TimedInput] {
        &self.events
    }

    /// Number of events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// No events.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Ticks needed to play every event: last event tick + 1 (0 when empty).
    pub fn duration(&self) -> u64 {
        self.events.last().map_or(0, |e| e.tick + 1)
    }

    /// Insert an event (after existing events of the same tick).
    pub fn push(&mut self, tick: u64, event: InputEvent) -> &mut Self {
        let at = self.events.partition_point(|e| e.tick <= tick);
        self.events.insert(at, TimedInput { tick, event });
        self
    }

    /// Builder form of [`Self::push`].
    pub fn at(mut self, tick: u64, event: InputEvent) -> Self {
        self.push(tick, event);
        self
    }

    /// Key down at `tick`.
    pub fn key_down(self, tick: u64, key: KeyCode) -> Self {
        self.at(tick, InputEvent::KeyDown(key))
    }

    /// Key up at `tick`.
    pub fn key_up(self, tick: u64, key: KeyCode) -> Self {
        self.at(tick, InputEvent::KeyUp(key))
    }

    /// Press at `tick`, release at `tick + 1`: held for exactly one fixed tick.
    pub fn tap_key(self, tick: u64, key: KeyCode) -> Self {
        self.hold_key(tick, key, 1)
    }

    /// Press at `tick`, release at `tick + ticks`: held for `ticks` fixed ticks.
    pub fn hold_key(self, tick: u64, key: KeyCode, ticks: u64) -> Self {
        self.key_down(tick, key).key_up(tick + ticks.max(1), key)
    }

    /// Press and release within one frame (`Input::inject_key_tap`); `pressed()` is never true.
    pub fn key_tap_same_frame(self, tick: u64, key: KeyCode) -> Self {
        self.at(tick, InputEvent::KeyTap(key))
    }

    /// Committed text at `tick`.
    pub fn type_text(self, tick: u64, text: &str) -> Self {
        self.at(tick, InputEvent::Text(text.to_string()))
    }

    /// Cursor to `position` at `tick`.
    pub fn mouse_move(self, tick: u64, position: Vec2) -> Self {
        self.at(tick, InputEvent::MouseMove(position))
    }

    /// Mouse button down at `tick`.
    pub fn mouse_down(self, tick: u64, button: MouseButton) -> Self {
        self.at(tick, InputEvent::MouseDown(button))
    }

    /// Mouse button up at `tick`.
    pub fn mouse_up(self, tick: u64, button: MouseButton) -> Self {
        self.at(tick, InputEvent::MouseUp(button))
    }

    /// Mouse button released as a cancel at `tick` ([`InputEvent::MouseCancel`]): no click, no
    /// drop.
    pub fn mouse_cancel(self, tick: u64, button: MouseButton, cause: MouseButtonCause) -> Self {
        self.at(tick, InputEvent::MouseCancel { button, cause })
    }

    /// Move to `position` and press at `tick`, release at `tick + 1`.
    pub fn click(self, tick: u64, button: MouseButton, position: Vec2) -> Self {
        self.mouse_move(tick, position)
            .mouse_down(tick, button)
            .mouse_up(tick + 1, button)
    }

    /// Press at `from` on `tick`, move linearly to `to` over `ticks` ticks, release one tick
    /// after the last move.
    pub fn drag(mut self, tick: u64, button: MouseButton, from: Vec2, to: Vec2, ticks: u64) -> Self {
        let ticks = ticks.max(1);
        self = self.mouse_move(tick, from).mouse_down(tick, button);
        for i in 1..=ticks {
            self = self.mouse_move(tick + i, lerp(from, to, i as f32 / ticks as f32));
        }
        self.mouse_up(tick + ticks + 1, button)
    }

    /// Scroll `delta` lines at `tick`.
    pub fn wheel(self, tick: u64, delta: Vec2) -> Self {
        self.at(tick, InputEvent::Wheel(delta))
    }

    /// Raw touch event.
    pub fn touch(self, tick: u64, id: u64, phase: TouchPhase, position: Vec2) -> Self {
        self.at(tick, InputEvent::Touch { id, phase, position })
    }

    /// Finger `id` down at `tick`, up at `tick + 1`.
    pub fn touch_tap(self, tick: u64, id: u64, position: Vec2) -> Self {
        self.touch(tick, id, TouchPhase::Started, position)
            .touch(tick + 1, id, TouchPhase::Ended, position)
    }

    /// Finger `id` down at `from`, moved linearly to `to` over `ticks` ticks, lifted one tick
    /// after the last move.
    pub fn swipe(mut self, tick: u64, id: u64, from: Vec2, to: Vec2, ticks: u64) -> Self {
        let ticks = ticks.max(1);
        self = self.touch(tick, id, TouchPhase::Started, from);
        for i in 1..=ticks {
            self = self.touch(tick + i, id, TouchPhase::Moved, lerp(from, to, i as f32 / ticks as f32));
        }
        self.touch(tick + ticks + 1, id, TouchPhase::Ended, to)
    }

    /// Two fingers (`id`, `id + 1`) placed horizontally around `center`, `start_distance`
    /// apart, moved symmetrically to `end_distance` over `ticks` ticks (pinch in/out), lifted
    /// one tick after the last move.
    pub fn pinch(
        mut self,
        tick: u64,
        id: u64,
        center: Vec2,
        start_distance: f32,
        end_distance: f32,
        ticks: u64,
    ) -> Self {
        let ticks = ticks.max(1);
        let at = |d: f32| (center - Vec2::new(d * 0.5, 0.0), center + Vec2::new(d * 0.5, 0.0));
        let (a, b) = at(start_distance);
        self = self
            .touch(tick, id, TouchPhase::Started, a)
            .touch(tick, id + 1, TouchPhase::Started, b);
        let mut last = (a, b);
        for i in 1..=ticks {
            let d = start_distance + (end_distance - start_distance) * (i as f32 / ticks as f32);
            let (a, b) = at(d);
            self = self
                .touch(tick + i, id, TouchPhase::Moved, a)
                .touch(tick + i, id + 1, TouchPhase::Moved, b);
            last = (a, b);
        }
        self.touch(tick + ticks + 1, id, TouchPhase::Ended, last.0).touch(
            tick + ticks + 1,
            id + 1,
            TouchPhase::Ended,
            last.1,
        )
    }

    /// Gamepad button change at `tick`.
    pub fn gamepad_button(self, tick: u64, pad: u32, button: GamepadButton, pressed: bool) -> Self {
        self.at(tick, InputEvent::GamepadButton { pad, button, pressed })
    }

    /// Gamepad button held from `tick` for `ticks` ticks.
    pub fn gamepad_press(self, tick: u64, pad: u32, button: GamepadButton, ticks: u64) -> Self {
        self.gamepad_button(tick, pad, button, true)
            .gamepad_button(tick + ticks.max(1), pad, button, false)
    }

    /// Gamepad axis value at `tick`.
    pub fn gamepad_axis(self, tick: u64, pad: u32, axis: GamepadAxis, value: f32) -> Self {
        self.at(tick, InputEvent::GamepadAxis { pad, axis, value })
    }

    /// Gamepad axis held at `value` from `tick` for `ticks` ticks, then returned to 0.
    pub fn gamepad_axis_hold(self, tick: u64, pad: u32, axis: GamepadAxis, value: f32, ticks: u64) -> Self {
        self.gamepad_axis(tick, pad, axis, value)
            .gamepad_axis(tick + ticks.max(1), pad, axis, 0.0)
    }

    /// Copy of the script with every event shifted by `ticks`.
    pub fn offset(&self, ticks: u64) -> Self {
        Self {
            events: self
                .events
                .iter()
                .map(|e| TimedInput {
                    tick: e.tick + ticks,
                    event: e.event.clone(),
                })
                .collect(),
        }
    }

    /// Merge `other` into this script (same time base).
    pub fn merge(mut self, other: &InputScript) -> Self {
        for e in &other.events {
            self.push(e.tick, e.event.clone());
        }
        self
    }

    /// Append `other` after this script ends ([`Self::duration`]).
    pub fn then(self, other: &InputScript) -> Self {
        let d = self.duration();
        self.merge(&other.offset(d))
    }

    /// Events due at exactly `tick`.
    pub fn events_at(&self, tick: u64) -> impl Iterator<Item = &InputEvent> {
        let start = self.events.partition_point(|e| e.tick < tick);
        self.events[start..]
            .iter()
            .take_while(move |e| e.tick == tick)
            .map(|e| &e.event)
    }

    /// Pretty RON text.
    pub fn to_ron(&self) -> Result<String, ScriptError> {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::new()).map_err(|e| ScriptError::Ron(e.to_string()))
    }

    /// Parse RON text (events are re-sorted by tick).
    pub fn from_ron(text: &str) -> Result<Self, ScriptError> {
        ron::from_str(text).map_err(|e| ScriptError::Ron(e.to_string()))
    }

    /// Write as RON (creating parent directories).
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), ScriptError> {
        let path = path.as_ref();
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, self.to_ron()?)?;
        Ok(())
    }

    /// Read a RON file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ScriptError> {
        Self::from_ron(&std::fs::read_to_string(path)?)
    }
}

/// Records the [`Input`] resource frame by frame into an [`InputScript`] (resource installed
/// by [`InputRecorderPlugin`]).
///
/// Works in real windowed apps too: add the plugin, call [`InputRecorder::start`], and save
/// [`InputRecorder::stop`]'s script on exit — then replay it in a test. Events are stamped
/// with the fixed tick at the start of the frame they were observed in, which is exactly
/// when [`crate::TestApp`] re-injects them. With a variable frame rate several frames can
/// share one tick; record at a fixed frame rate for bit-exact replays.
#[derive(Clone, Debug, Default)]
pub struct InputRecorder {
    recording: bool,
    start_tick: u64,
    script: InputScript,
    keys: HashSet<KeyCode>,
    mouse: HashSet<MouseButton>,
    mouse_position: Option<Vec2>,
    window_size: Vec2,
    pads: HashMap<u32, PadState>,
}

#[derive(Clone, Debug, Default)]
struct PadState {
    buttons: HashSet<GamepadButton>,
    axes: [f32; 6],
}

impl InputRecorder {
    /// Start a fresh recording; ticks are relative to `current_tick`. Keys already held are
    /// recorded as pressed at tick 0.
    pub fn start(&mut self, current_tick: u64) {
        *self = Self {
            recording: true,
            start_tick: current_tick,
            ..Default::default()
        };
    }

    /// Stop and return the recorded script.
    pub fn stop(&mut self) -> InputScript {
        self.recording = false;
        std::mem::take(&mut self.script)
    }

    /// Currently recording.
    pub fn is_recording(&self) -> bool {
        self.recording
    }

    /// Script recorded so far.
    pub fn script(&self) -> &InputScript {
        &self.script
    }

    /// Diff `input` against the previous frame and record the changes at `tick` (absolute).
    pub fn observe(&mut self, tick: u64, input: &Input) {
        if !self.recording {
            return;
        }
        let t = tick.saturating_sub(self.start_tick);
        let mut out: Vec<InputEvent> = Vec::new();

        if input.window_size() != self.window_size {
            self.window_size = input.window_size();
            out.push(InputEvent::WindowSize(self.window_size));
        }

        // Keys.
        let keys: HashSet<KeyCode> = input.keys().iter_pressed().collect();
        diff_buttons(
            &self.keys,
            &keys,
            ALL_KEYS.iter().copied(),
            |k| (input.just_pressed(k), input.just_released(k)),
            &mut out,
            InputEvent::KeyDown,
            InputEvent::KeyUp,
            |k| vec![InputEvent::KeyTap(k)],
        );
        self.keys = keys;

        // Mouse.
        let pos = input.mouse_position();
        let mut expected_delta = Vec2::ZERO;
        if pos != self.mouse_position {
            match pos {
                Some(p) => {
                    if let Some(prev) = self.mouse_position {
                        expected_delta = p - prev;
                    }
                    out.push(InputEvent::MouseMove(p));
                }
                None => out.push(InputEvent::MouseLeft),
            }
            self.mouse_position = pos;
        }
        let extra = input.mouse_delta() - expected_delta;
        if extra != Vec2::ZERO {
            out.push(InputEvent::MouseMotion(extra));
        }
        let mouse: HashSet<MouseButton> = MOUSE_BUTTONS
            .iter()
            .copied()
            .chain(self.mouse.iter().copied())
            .filter(|b| input.mouse_pressed(*b))
            .collect();
        let mouse_from = out.len();
        diff_buttons(
            &self.mouse,
            &mouse,
            MOUSE_BUTTONS.iter().copied(),
            |b| (input.mouse_just_pressed(b), input.mouse_just_released(b)),
            &mut out,
            InputEvent::MouseDown,
            InputEvent::MouseUp,
            |b| vec![InputEvent::MouseDown(b), InputEvent::MouseUp(b)],
        );
        cancel_releases(&mut out[mouse_from..], input);
        self.mouse = mouse;
        if input.wheel_delta() != Vec2::ZERO {
            out.push(InputEvent::Wheel(input.wheel_delta()));
        }
        if !input.text().is_empty() {
            out.push(InputEvent::Text(input.text().to_string()));
        }

        // Touches.
        let started = input.just_started_touches();
        for &id in started {
            let touch = input
                .touch(id)
                .copied()
                .or_else(|| input.just_ended_touches().iter().find(|t| t.id == id).copied());
            if let Some(t) = touch {
                out.push(InputEvent::Touch {
                    id,
                    phase: TouchPhase::Started,
                    position: t.start,
                });
                if input.touch(id).is_some() && t.position != t.start {
                    out.push(InputEvent::Touch {
                        id,
                        phase: TouchPhase::Moved,
                        position: t.position,
                    });
                }
            }
        }
        for t in input.touches() {
            if !started.contains(&t.id) && t.delta != Vec2::ZERO {
                out.push(InputEvent::Touch {
                    id: t.id,
                    phase: TouchPhase::Moved,
                    position: t.position,
                });
            }
        }
        for t in input.just_ended_touches() {
            // A touch the system cancelled is no tap: it replays as cancelled.
            let cancelled = input.just_cancelled_touches().iter().any(|c| c.id == t.id);
            out.push(InputEvent::Touch {
                id: t.id,
                phase: if cancelled {
                    TouchPhase::Cancelled
                } else {
                    TouchPhase::Ended
                },
                position: t.position,
            });
        }

        // Gamepads (sorted for determinism).
        let mut ids: Vec<u32> = input.gamepad_ids().collect();
        ids.sort_unstable();
        let mut gone: Vec<u32> = self.pads.keys().copied().filter(|id| !ids.contains(id)).collect();
        gone.sort_unstable();
        for id in gone {
            self.pads.remove(&id);
            out.push(InputEvent::GamepadDisconnected(id));
        }
        for id in ids {
            let pad = input.gamepad(id).expect("listed gamepad");
            let prev = match self.pads.get(&id) {
                Some(p) => p.clone(),
                None => {
                    out.push(InputEvent::GamepadConnected(id));
                    PadState::default()
                }
            };
            let buttons: HashSet<GamepadButton> = pad.buttons.iter_pressed().collect();
            diff_buttons(
                &prev.buttons,
                &buttons,
                GAMEPAD_BUTTONS.iter().copied(),
                |b| (pad.buttons.just_pressed(b), pad.buttons.just_released(b)),
                &mut out,
                |button| InputEvent::GamepadButton {
                    pad: id,
                    button,
                    pressed: true,
                },
                |button| InputEvent::GamepadButton {
                    pad: id,
                    button,
                    pressed: false,
                },
                |button| {
                    vec![
                        InputEvent::GamepadButton {
                            pad: id,
                            button,
                            pressed: true,
                        },
                        InputEvent::GamepadButton {
                            pad: id,
                            button,
                            pressed: false,
                        },
                    ]
                },
            );
            let mut axes = prev.axes;
            for (i, axis) in GAMEPAD_AXES.iter().enumerate() {
                let v = pad.axis(*axis);
                if v != axes[i] {
                    axes[i] = v;
                    out.push(InputEvent::GamepadAxis {
                        pad: id,
                        axis: *axis,
                        value: v,
                    });
                }
            }
            self.pads.insert(id, PadState { buttons, axes });
        }

        for e in out {
            self.script.push(t, e);
        }
    }
}

/// Turn the `MouseUp`s of a frame's diff whose release was a cancel (the cause in
/// [`Input::mouse_button_events`]: focus loss, OS state) into [`InputEvent::MouseCancel`]s. A
/// button's last `MouseUp`s pair with its last release events, so what leaves the button up is
/// what is recorded; a `MouseUp` without a release event (state changed behind `Input`'s back)
/// stays one.
fn cancel_releases(events: &mut [InputEvent], input: &Input) {
    let mut buttons: Vec<MouseButton> = Vec::new();
    for e in events.iter() {
        if let InputEvent::MouseUp(b) = e
            && !buttons.contains(b)
        {
            buttons.push(*b);
        }
    }
    for b in buttons {
        let is_up = |e: &InputEvent| matches!(e, InputEvent::MouseUp(x) if *x == b);
        let ups = events.iter().filter(|e| is_up(e)).count();
        let causes: Vec<MouseButtonCause> = input
            .mouse_button_events()
            .iter()
            .filter(|e| !e.pressed && e.button == b)
            .map(|e| e.cause)
            .collect();
        let unpaired = ups.saturating_sub(causes.len());
        let mut causes = causes[causes.len().saturating_sub(ups)..].iter();
        for (n, e) in events.iter_mut().filter(|e| is_up(e)).enumerate() {
            if n < unpaired {
                continue;
            }
            if let Some(&cause) = causes.next()
                && cause.is_cancel()
            {
                *e = InputEvent::MouseCancel { button: b, cause };
            }
        }
    }
}

/// Emit down/up/tap events for one kind of button, in a deterministic order.
#[allow(clippy::too_many_arguments)]
fn diff_buttons<T: Copy + Eq + std::hash::Hash + std::fmt::Debug>(
    prev: &HashSet<T>,
    now: &HashSet<T>,
    all: impl Iterator<Item = T>,
    edges: impl Fn(T) -> (bool, bool),
    out: &mut Vec<InputEvent>,
    down: impl Fn(T) -> InputEvent,
    up: impl Fn(T) -> InputEvent,
    tap: impl Fn(T) -> Vec<InputEvent>,
) {
    let sorted = |set: &mut Vec<T>| set.sort_by_cached_key(|b| format!("{b:?}"));
    let mut released: Vec<T> = prev.difference(now).copied().collect();
    sorted(&mut released);
    for b in released {
        out.push(up(b));
    }
    let mut repressed: Vec<T> = prev.intersection(now).copied().filter(|b| edges(*b).1).collect();
    sorted(&mut repressed);
    for b in repressed {
        out.push(up(b));
        out.push(down(b));
    }
    let mut pressed: Vec<T> = now.difference(prev).copied().collect();
    sorted(&mut pressed);
    for b in pressed {
        out.push(down(b));
    }
    for b in all {
        if !prev.contains(&b) && !now.contains(&b) {
            let (jp, jr) = edges(b);
            if jp && jr {
                out.extend(tap(b));
            }
        }
    }
}

/// Installs the [`InputRecorder`] resource and its `PreUpdate` observer system.
#[derive(Clone, Copy, Debug, Default)]
pub struct InputRecorderPlugin {
    /// Start recording immediately (at the app's current tick).
    pub start_immediately: bool,
}

impl Plugin for InputRecorderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Input>();
        let mut recorder = InputRecorder::default();
        if self.start_immediately {
            recorder.start(app.resources.get::<Time>().map_or(0, |t| t.tick));
        }
        app.insert_resource(recorder);
        app.add_system_named(Stage::PreUpdate, "test::record_input", |ctx| {
            let tick = ctx.time().tick;
            ctx.resources.scope::<InputRecorder, _>(|res, rec| {
                if rec.is_recording() {
                    rec.observe(tick, res.expect::<Input>());
                }
            });
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_orders_events_and_duration() {
        let s = InputScript::new()
            .key_up(5, KeyCode::KeyA)
            .key_down(0, KeyCode::KeyA)
            .at(5, InputEvent::Text("x".into()));
        assert_eq!(s.duration(), 6);
        assert_eq!(s.events()[0].tick, 0);
        let at5: Vec<_> = s.events_at(5).cloned().collect();
        assert_eq!(
            at5,
            vec![InputEvent::KeyUp(KeyCode::KeyA), InputEvent::Text("x".into())]
        );
        assert_eq!(s.events_at(3).count(), 0);
        let later = s.then(&InputScript::new().tap_key(0, KeyCode::Space));
        assert_eq!(later.duration(), 8);
        assert_eq!(InputScript::new().duration(), 0);
    }

    #[test]
    fn gestures_shape() {
        let s = InputScript::new().swipe(2, 7, Vec2::ZERO, Vec2::new(100.0, 0.0), 4);
        assert_eq!(s.len(), 6);
        assert_eq!(s.duration(), 2 + 4 + 2);
        assert_eq!(
            s.events()[4].event,
            InputEvent::Touch {
                id: 7,
                phase: TouchPhase::Moved,
                position: Vec2::new(100.0, 0.0)
            }
        );
        let p = InputScript::new().pinch(0, 1, Vec2::new(50.0, 50.0), 10.0, 30.0, 2);
        assert_eq!(p.len(), 2 + 4 + 2);
        let d = InputScript::new().drag(0, MouseButton::Left, Vec2::ZERO, Vec2::ONE, 3);
        assert_eq!(d.duration(), 5);
        assert_eq!(d.events().last().unwrap().event, InputEvent::MouseUp(MouseButton::Left));
    }

    #[test]
    fn ron_roundtrip_all_variants() {
        let s = InputScript::new()
            .key_down(0, KeyCode::ArrowLeft)
            .key_tap_same_frame(1, KeyCode::Enter)
            .type_text(1, "hi")
            .click(2, MouseButton::Other(9), Vec2::new(1.5, 2.5))
            .at(3, InputEvent::MouseMotion(Vec2::X))
            .at(3, InputEvent::MouseLeft)
            .wheel(4, Vec2::new(0.0, -1.0))
            .touch_tap(5, 3, Vec2::splat(4.0))
            .at(6, InputEvent::GamepadConnected(2))
            .gamepad_press(6, 2, GamepadButton::DPadUp, 2)
            .gamepad_axis(7, 2, GamepadAxis::RightTrigger, 0.5)
            .at(9, InputEvent::GamepadDisconnected(2))
            .at(9, InputEvent::WindowSize(Vec2::new(640.0, 480.0)))
            .at(10, InputEvent::ReleaseAll);
        let text = s.to_ron().unwrap();
        assert!(text.contains("ArrowLeft"), "{text}");
        assert_eq!(InputScript::from_ron(&text).unwrap(), s);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested/script.ron");
        s.save(&path).unwrap();
        assert_eq!(InputScript::load(&path).unwrap(), s);
        assert!(matches!(InputScript::from_ron("nope"), Err(ScriptError::Ron(_))));
    }

    #[test]
    fn any_deserializer_sorts_events() {
        let text = r#"(events: [
            (tick: 5, event: KeyUp(KeyA)),
            (tick: 0, event: KeyDown(KeyA)),
            (tick: 5, event: Text("x")),
            (tick: 2, event: Wheel((0.0, 1.0))),
        ])"#;
        // Plain `ron::from_str` (not just `from_ron`) and JSON, also nested in another struct.
        let s: InputScript = ron::from_str(text).unwrap();
        let ticks: Vec<u64> = s.events().iter().map(|e| e.tick).collect();
        assert_eq!(ticks, [0, 2, 5, 5]);
        assert_eq!(s.duration(), 6);
        assert_eq!(
            s.events_at(5).cloned().collect::<Vec<_>>(),
            [InputEvent::KeyUp(KeyCode::KeyA), InputEvent::Text("x".into())],
            "same-tick order is kept"
        );
        #[derive(Deserialize)]
        struct Level {
            intro: InputScript,
        }
        let json = format!(
            r#"{{"intro": {}}}"#,
            serde_json::to_string(&s.clone().reversed_for_test()).unwrap()
        );
        let level: Level = serde_json::from_str(&json).unwrap();
        assert_eq!(level.intro.events().first().unwrap().tick, 0);
        assert_eq!(level.intro.duration(), 6);
        let mut pushed = level.intro;
        pushed.push(3, InputEvent::ReleaseAll);
        assert_eq!(pushed.events()[2].tick, 3);
    }

    impl InputScript {
        /// Events in reverse order (bypasses the sorted invariant to build unsorted input).
        fn reversed_for_test(mut self) -> Self {
            self.events.reverse();
            self
        }
    }

    #[test]
    fn recorder_diffs_input_state() {
        let mut input = Input::default();
        let mut rec = InputRecorder::default();
        rec.start(10);
        input.inject_key_down(KeyCode::KeyW);
        input.inject_key_tap(KeyCode::Space);
        input.inject_mouse_move(Vec2::new(5.0, 5.0));
        input.inject_touch(1, TouchPhase::Started, Vec2::ONE);
        input.inject_gamepad_axis(0, GamepadAxis::LeftStickX, 0.25);
        rec.observe(10, &input);
        input.end_frame();
        input.inject_key_up(KeyCode::KeyW);
        input.inject_mouse_move(Vec2::new(7.0, 5.0));
        input.inject_mouse_motion(Vec2::new(1.0, 0.0));
        input.inject_touch(1, TouchPhase::Moved, Vec2::new(3.0, 1.0));
        rec.observe(11, &input);
        input.end_frame();
        input.inject_touch(1, TouchPhase::Ended, Vec2::new(3.0, 1.0));
        input.inject_gamepad_disconnected(0);
        rec.observe(12, &input);
        let s = rec.stop();
        assert!(!rec.is_recording());
        let at = |t| s.events_at(t).cloned().collect::<Vec<_>>();
        assert_eq!(
            at(0),
            vec![
                InputEvent::KeyDown(KeyCode::KeyW),
                InputEvent::KeyTap(KeyCode::Space),
                InputEvent::MouseMove(Vec2::new(5.0, 5.0)),
                InputEvent::Touch {
                    id: 1,
                    phase: TouchPhase::Started,
                    position: Vec2::ONE
                },
                InputEvent::GamepadConnected(0),
                InputEvent::GamepadAxis {
                    pad: 0,
                    axis: GamepadAxis::LeftStickX,
                    value: 0.25
                },
            ]
        );
        assert_eq!(
            at(1),
            vec![
                InputEvent::KeyUp(KeyCode::KeyW),
                InputEvent::MouseMove(Vec2::new(7.0, 5.0)),
                InputEvent::MouseMotion(Vec2::new(1.0, 0.0)),
                InputEvent::Touch {
                    id: 1,
                    phase: TouchPhase::Moved,
                    position: Vec2::new(3.0, 1.0)
                },
            ]
        );
        assert_eq!(
            at(2),
            vec![
                InputEvent::Touch {
                    id: 1,
                    phase: TouchPhase::Ended,
                    position: Vec2::new(3.0, 1.0)
                },
                InputEvent::GamepadDisconnected(0),
            ]
        );
    }

    /// A release the device did not report (focus loss, OS state) is recorded as a cancel with
    /// its cause, replays as one (RON included), and device releases stay plain `MouseUp`s.
    #[test]
    fn recorder_keeps_the_cause_of_a_cancelled_release() {
        let (l, r, m) = (MouseButton::Left, MouseButton::Right, MouseButton::Middle);
        let mut input = Input::default();
        let mut rec = InputRecorder::default();
        rec.start(0);
        input.inject_mouse_move(Vec2::new(5.0, 6.0));
        input.inject_mouse_button_down(l);
        input.inject_mouse_button_down(r);
        input.inject_mouse_button_down(m);
        rec.observe(0, &input);
        input.end_frame();
        input.inject_mouse_button_up(l);
        input.release_mouse_button(r, MouseButtonCause::OsReconciled);
        input.release_mouse_button(m, MouseButtonCause::FocusLost);
        rec.observe(1, &input);
        input.end_frame();
        // A press and its cancel within one frame, and a cancel followed by a device press.
        input.inject_mouse_button_down(l);
        input.release_mouse_button(l, MouseButtonCause::FocusLost);
        input.inject_mouse_button_down(r);
        rec.observe(2, &input);
        input.end_frame();
        input.release_all();
        rec.observe(3, &input);
        let script = rec.stop();
        let at = |t| script.events_at(t).cloned().collect::<Vec<_>>();
        // Released buttons come sorted by name (Left, Middle, Right); a press comes before a tap.
        assert_eq!(
            at(1),
            [
                InputEvent::MouseUp(l),
                InputEvent::MouseCancel {
                    button: m,
                    cause: MouseButtonCause::FocusLost
                },
                InputEvent::MouseCancel {
                    button: r,
                    cause: MouseButtonCause::OsReconciled
                },
            ]
        );
        assert_eq!(
            at(2),
            [
                InputEvent::MouseDown(r),
                InputEvent::MouseDown(l),
                InputEvent::MouseCancel {
                    button: l,
                    cause: MouseButtonCause::FocusLost
                },
            ]
        );
        assert_eq!(
            at(3),
            [InputEvent::MouseCancel {
                button: r,
                cause: MouseButtonCause::FocusLost
            }]
        );

        // Through RON, into a fresh `Input`: the same releases with the same causes.
        let script = InputScript::from_ron(&script.to_ron().unwrap()).unwrap();
        let mut replay = Input::default();
        for tick in 0..=3 {
            replay.end_frame();
            for e in script.events_at(tick) {
                e.apply(&mut replay);
            }
            let events: Vec<(MouseButton, bool, MouseButtonCause)> = replay
                .mouse_button_events()
                .iter()
                .map(|e| (e.button, e.pressed, e.cause))
                .collect();
            match tick {
                1 => assert_eq!(
                    events,
                    [
                        (l, false, MouseButtonCause::Device),
                        (m, false, MouseButtonCause::FocusLost),
                        (r, false, MouseButtonCause::OsReconciled),
                    ]
                ),
                3 => assert_eq!(events, [(r, false, MouseButtonCause::FocusLost)]),
                _ => {}
            }
        }
        assert!(!replay.mouse_pressed(r) && !replay.mouse_pressed(l));
    }

    /// The old wire format (plain `MouseUp`) still loads, and a recorded cancel round-trips
    /// through RON and JSON.
    #[test]
    fn mouse_cancel_serializes_and_old_scripts_load() {
        let old = r#"(events: [(tick: 0, event: MouseDown(Left)), (tick: 1, event: MouseUp(Left))])"#;
        let s = InputScript::from_ron(old).unwrap();
        assert_eq!(s.events()[1].event, InputEvent::MouseUp(MouseButton::Left));
        let s = InputScript::new()
            .mouse_down(0, MouseButton::Left)
            .mouse_cancel(1, MouseButton::Left, MouseButtonCause::FocusLost)
            .mouse_cancel(2, MouseButton::Other(7), MouseButtonCause::OsReconciled);
        let ron = s.to_ron().unwrap();
        assert!(ron.contains("MouseCancel") && ron.contains("FocusLost"), "{ron}");
        assert_eq!(InputScript::from_ron(&ron).unwrap(), s);
        let json = serde_json::to_string(&s).unwrap();
        assert_eq!(serde_json::from_str::<InputScript>(&json).unwrap(), s);
        // A device cause is a plain release.
        let mut input = Input::default();
        input.inject_mouse_button_down(MouseButton::Left);
        InputEvent::MouseCancel {
            button: MouseButton::Left,
            cause: MouseButtonCause::Device,
        }
        .apply(&mut input);
        assert!(!input.mouse_pressed(MouseButton::Left));
        assert!(!input.mouse_button_events().last().unwrap().cancelled());
    }

    /// A system-cancelled touch is recorded as cancelled, so a replay is no tap either.
    #[test]
    fn recorder_keeps_a_cancelled_touch_cancelled() {
        let mut input = Input::default();
        let mut rec = InputRecorder::default();
        rec.start(0);
        input.inject_touch(1, TouchPhase::Started, Vec2::new(4.0, 4.0));
        input.inject_touch(2, TouchPhase::Started, Vec2::new(9.0, 9.0));
        rec.observe(0, &input);
        input.end_frame();
        input.inject_touch(1, TouchPhase::Cancelled, Vec2::new(4.0, 4.0));
        input.inject_touch(2, TouchPhase::Ended, Vec2::new(9.0, 9.0));
        rec.observe(1, &input);
        let script = rec.stop();
        let phases: Vec<TouchPhase> = script
            .events_at(1)
            .filter_map(|e| match e {
                InputEvent::Touch { phase, .. } => Some(*phase),
                _ => None,
            })
            .collect();
        assert_eq!(phases, [TouchPhase::Cancelled, TouchPhase::Ended]);
        let mut replay = Input::default();
        for tick in 0..=1 {
            replay.end_frame();
            script.events_at(tick).for_each(|e| e.apply(&mut replay));
        }
        assert_eq!(replay.just_ended_touches().len(), 2);
        assert_eq!(replay.just_cancelled_touches().len(), 1);
    }

    /// End to end, with nothing but what the crate always had: a session with a cancelled
    /// release is recorded, saved, loaded and replayed, and the replayed release is still a
    /// cancel (not a plain device release, which the UI would count as a click).
    #[test]
    fn a_recorded_cancel_replays_as_a_cancel() {
        let mut input = Input::default();
        let mut rec = InputRecorder::default();
        rec.start(0);
        input.inject_mouse_move(Vec2::new(8.0, 8.0));
        input.inject_mouse_button_down(MouseButton::Left);
        rec.observe(0, &input);
        input.end_frame();
        input.release_mouse_button(MouseButton::Left, MouseButtonCause::OsReconciled);
        rec.observe(1, &input);
        let script = InputScript::from_ron(&rec.stop().to_ron().unwrap()).unwrap();

        let mut replay = Input::default();
        for tick in 0..=1 {
            replay.end_frame();
            script.events_at(tick).for_each(|e| e.apply(&mut replay));
        }
        let release = replay
            .mouse_button_events()
            .last()
            .copied()
            .expect("the release replayed");
        assert!(!release.pressed && release.button == MouseButton::Left);
        assert!(release.cancelled(), "replayed as {:?}", release.cause);
    }
}
