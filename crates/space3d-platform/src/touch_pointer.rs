//! Fingers as a mouse ([`WindowPlugin::touch_as_mouse`](crate::WindowPlugin::touch_as_mouse)):
//! what a game whose controls and UI read the mouse gets from a touch screen.
//!
//! | gesture | the mouse |
//! |---|---|
//! | finger down | the pointer is there (a hover); the left button goes down on the **next frame** |
//! | tap | left press and release, a frame each, then the pointer leaves |
//! | drag | a left drag |
//! | long press (still for [`TouchPointerConfig::long_press`]) | a right click where the finger is; the left button stays down and is let go without a click when the finger lifts |
//! | two fingers, pinch | the wheel: one line per [`TouchPointerConfig::pinch_step`] of spread (apart is up), the pointer between the fingers |
//! | two fingers, anything | a gesture ([`Input::gesture`]): this frame's zoom, turn and pan |
//!
//! Two options for a game that steers by holding the button ([`TouchPointerConfig`]):
//! `press_on_move` holds the left press back until the finger leaves the slop (the press then
//! lands where the finger went down) or lifts (a tap), so a still finger presses nothing and a
//! long press is the right click alone; `pinch_wheel` off leaves two fingers to the gesture.
//!
//! The press waits a frame because a mouse hovers before it clicks and a finger does not: a UI
//! that decides whose click it is from where the pointer was at its last pass (egui's "is the
//! pointer over a window") would hand a tap on a button to the world behind it. For the same
//! reason a tap's release comes a frame after its press, and the pointer leaves a frame after
//! that, so nothing stays hovered where the finger was.
//!
//! A second finger ends the first one's press as a cancel (no click). When one finger of a
//! pinch lifts the pointer leaves; the other finger does nothing alone, and a finger put down
//! beside it begins a new gesture with it.
//!
//! `side_tap` is for a game played with two thumbs, one steering and one on the buttons: a
//! second finger that lands on a [`TouchPointer::tap_zones`] rectangle, or anywhere while the
//! first has been dragging for [`TouchPointerConfig::SIDE_TAP_AFTER`], is a tap of its own and
//! no gesture. There is one mouse, so when it lifts the mouse goes there for three frames
//! (the first finger's button is let go without a click, then hover, press, release) and comes
//! back to the first finger, whose press is made again where the finger is by then. Outside a
//! zone, a side finger that leaves the slop is a pinch with the first after all.
//!
//! [`TouchPointer`] is the state machine alone (no window, no clock). [`TouchMouse`] is the
//! resource around it that the window runner feeds every finger through, and that a test or a
//! script feeds the same way ([`TouchMouse::touch`]): it applies the [`PointerAction`]s to
//! [`Input`] and hands the raw touch on. Its per-frame call is a `PreUpdate` system of
//! [`InputPlugin`](crate::InputPlugin) at [`TouchMouse::ORDER`], so it also runs without a window.

use crate::input::{Input, MouseButton, MouseButtonCause, TouchPhase};
use glam::Vec2;
use web_time::Instant;

/// The thresholds of [`TouchPointer`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TouchPointerConfig {
    /// Physical pixels a finger may wander and still be a tap or a long press. The runner sets
    /// it to [`Self::SLOP_POINTS`] of the window's scale.
    pub slop: f32,
    /// Seconds a still finger is down before the right click. Default 0.5.
    pub long_press: f64,
    /// The ratio of the fingers' distance that is one wheel line. Default 1.05 (twice as far
    /// apart: about 14 lines).
    pub pinch_step: f32,
    /// Whether a pinch turns the wheel. Default `true`; a game that reads
    /// [`Input::gesture`] instead turns it off, so a pinch is not also a scroll.
    pub pinch_wheel: bool,
    /// The left press waits until the finger leaves the slop or lifts. Default `false`: the
    /// press comes on the frame after the finger.
    pub press_on_move: bool,
    /// A second finger can be a tap of its own while the first keeps its drag (see the
    /// module). Default `false`: a second finger is always a gesture.
    pub side_tap: bool,
}

impl TouchPointerConfig {
    /// Seconds the first finger must have been down before a second one, off every
    /// [`TouchPointer::tap_zones`], counts as a side tap and not as the second finger of a pinch
    /// (the fingers of a pinch land together).
    pub const SIDE_TAP_AFTER: f64 = 0.2;

    /// The slop in the window's own points (multiply by the scale factor for pixels).
    pub const SLOP_POINTS: f32 = 10.0;
}

impl Default for TouchPointerConfig {
    fn default() -> Self {
        Self {
            slop: Self::SLOP_POINTS,
            long_press: 0.5,
            pinch_step: 1.05,
            pinch_wheel: true,
            press_on_move: false,
            side_tap: false,
        }
    }
}

/// What the mouse does ([`TouchPointer`]), in order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PointerAction {
    /// The pointer appears at a position (no movement from where it was).
    Place(Vec2),
    /// The pointer moves to a position.
    Move(Vec2),
    /// The pointer is gone.
    Leave,
    /// A button goes down.
    Down(MouseButton),
    /// A button comes up.
    Up(MouseButton),
    /// A button comes up without a click (a second finger, a cancelled touch).
    Cancel(MouseButton),
    /// The wheel, in lines (up is positive).
    Wheel(f32),
    /// Two fingers are down, with this point between them.
    GestureStart(Vec2),
    /// The two fingers moved ([`crate::input::TouchGesture`]: the distance ratio, the turn in
    /// radians, the movement of the point between them and where it is now).
    Gesture {
        /// The fingers' distance over what it was.
        zoom: f32,
        /// Radians the line between the fingers turned (clockwise on the screen is positive).
        rotate: f32,
        /// Pixels the point between the fingers moved.
        pan: Vec2,
        /// The point between the fingers.
        center: Vec2,
    },
    /// One of the two fingers lifted.
    GestureEnd,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum State {
    /// No finger that counts.
    Idle,
    /// One finger is the pointer.
    One {
        id: u64,
        start: Vec2,
        since: f64,
        /// It left the slop: a drag, no long press any more.
        moved: bool,
        /// The long press's right click was sent.
        long: bool,
    },
    /// Two fingers pinch.
    Pinch { a: (u64, Vec2), b: (u64, Vec2) },
    /// A gesture is over but fingers are still down: no pointer. `rest` is the finger of the
    /// gesture that stayed, which a new finger makes a gesture with.
    Lifting { rest: Option<(u64, Vec2)> },
}

/// A second finger that may be a tap of its own ([`TouchPointerConfig::side_tap`]).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Side {
    id: u64,
    start: Vec2,
    /// It went down on a tap zone: a tap wherever it wanders.
    zone: bool,
}

/// The mouse's visit to a side tap: `next` is 1 before the press, 2 before the release, 3
/// before the way back to the first finger.
#[derive(Clone, Copy, Debug, PartialEq)]
struct SideTap {
    next: u8,
    /// The first finger held the left button: it is pressed again afterwards.
    resume: bool,
}

/// The state machine of fingers as a mouse (see the module).
#[derive(Clone, Debug)]
pub struct TouchPointer {
    /// The thresholds.
    pub config: TouchPointerConfig,
    /// With [`TouchPointerConfig::side_tap`]: rectangles (minimum and maximum corner, physical
    /// pixels) where a second finger is always a tap, whatever the first one does. The game
    /// keeps them at its buttons.
    pub tap_zones: Vec<[Vec2; 2]>,
    side: Option<Side>,
    tap: Option<SideTap>,
    /// The press that waits is the first finger's again after a side tap.
    resume_due: bool,
    /// This frame's press was that one ([`Self::press_resumed`]).
    resumed: bool,
    state: State,
    /// Fingers down, whatever they do.
    fingers: Vec<u64>,
    /// Where the one finger that is the pointer was last.
    last: Option<Vec2>,
    /// The left button is down for the game.
    left: bool,
    /// The left press waits for the next frame.
    press_due: bool,
    /// The left release waits for the frame after the press (`true`: a cancel).
    release_due: Option<bool>,
    /// The right button of a long press comes up on the next frame.
    right_due: bool,
    /// The pointer leaves on the next frame with nothing else to do.
    leave_due: bool,
    /// `press_on_move`: where the finger is by now, for the move right after the press that
    /// still waits at the finger's start.
    after_press: Option<Vec2>,
}

impl Default for TouchPointer {
    fn default() -> Self {
        Self::new(TouchPointerConfig::default())
    }
}

impl TouchPointer {
    /// A machine with `config`.
    pub fn new(config: TouchPointerConfig) -> Self {
        Self {
            config,
            tap_zones: Vec::new(),
            side: None,
            tap: None,
            resume_due: false,
            resumed: false,
            state: State::Idle,
            fingers: Vec::new(),
            last: None,
            left: false,
            press_due: false,
            release_due: None,
            right_due: false,
            leave_due: false,
            after_press: None,
        }
    }

    /// A touch event at `now` (seconds on any clock that [`Self::frame`] shares).
    pub fn touch(&mut self, id: u64, phase: TouchPhase, p: Vec2, now: f64, out: &mut Vec<PointerAction>) {
        match phase {
            TouchPhase::Started => {
                if !self.fingers.contains(&id) {
                    self.fingers.push(id);
                }
                match self.state {
                    State::Idle => {
                        // A tap right after a tap: the first one's press and release are not lost.
                        self.flush(out);
                        self.resume_due = false;
                        self.leave_due = false;
                        out.push(PointerAction::Place(p));
                        self.press_due = !self.config.press_on_move;
                        self.state = State::One {
                            id,
                            start: p,
                            since: now,
                            moved: false,
                            long: false,
                        };
                    }
                    State::One {
                        id: first,
                        start,
                        since,
                        moved,
                        ..
                    } if first != id => {
                        if self.side.is_some() {
                            // A third finger.
                            return;
                        }
                        let zone = self.in_zone(p);
                        let dragging = moved
                            && (self.left || self.press_due || self.tap.is_some())
                            && now - since >= TouchPointerConfig::SIDE_TAP_AFTER;
                        if self.config.side_tap && (zone || dragging) {
                            self.side = Some(Side { id, start: p, zone });
                        } else {
                            self.pinch((first, self.last.unwrap_or(start)), (id, p), out);
                        }
                    }
                    State::Lifting { rest: Some(a) } if a.0 != id && self.fingers.len() == 2 => {
                        self.pinch(a, (id, p), out);
                    }
                    _ => {}
                }
            }
            TouchPhase::Moved => {
                if let (Some(side), State::One { id: first, start, .. }) = (self.side, self.state)
                    && side.id == id
                {
                    // Off the buttons a side finger that travels is a pinch after all.
                    if !side.zone && side.start.distance(p) > self.config.slop {
                        self.side = None;
                        self.pinch((first, self.last.unwrap_or(start)), (id, p), out);
                    }
                    return;
                }
                let visiting = self.tap.is_some();
                match &mut self.state {
                    State::One {
                        id: first,
                        start,
                        moved,
                        ..
                    } if *first == id => {
                        let was_still = !*moved;
                        if start.distance(p) > self.config.slop {
                            *moved = true;
                        }
                        self.last = Some(p);
                        if visiting {
                            // The mouse is at a side tap: it comes back to where this finger is.
                        } else if self.config.press_on_move && *moved && !self.left {
                            // The held-back press: it lands where the finger went down, on the
                            // next frame, and the pointer follows the finger from there.
                            // (after a long press too: a finger that rested and then goes is a drag)
                            if was_still {
                                out.push(PointerAction::Move(*start));
                                self.press_due = true;
                            }
                            self.after_press = Some(p);
                        } else {
                            out.push(PointerAction::Move(p));
                        }
                    }
                    State::Pinch { a, b } if a.0 == id || b.0 == id => {
                        let (line, middle) = (b.1 - a.1, (a.1 + b.1) * 0.5);
                        let before = line.length();
                        if a.0 == id {
                            a.1 = p;
                        } else {
                            b.1 = p;
                        }
                        let (line_now, center) = (b.1 - a.1, (a.1 + b.1) * 0.5);
                        let after = line_now.length();
                        out.push(PointerAction::Place(center));
                        let apart = before > 1.0 && after > 1.0;
                        out.push(PointerAction::Gesture {
                            zoom: if apart { after / before } else { 1.0 },
                            // The angle between the two lines, the short way round.
                            rotate: if apart {
                                line.perp_dot(line_now).atan2(line.dot(line_now))
                            } else {
                                0.0
                            },
                            pan: center - middle,
                            center,
                        });
                        if apart && self.config.pinch_wheel && self.config.pinch_step > 1.0 {
                            let lines = (after / before).ln() / self.config.pinch_step.ln();
                            if lines != 0.0 {
                                out.push(PointerAction::Wheel(lines));
                            }
                        }
                    }
                    State::Lifting { rest: Some(rest) } if rest.0 == id => rest.1 = p,
                    _ => {}
                }
            }
            TouchPhase::Ended | TouchPhase::Cancelled => {
                self.fingers.retain(|f| *f != id);
                if let Some(side) = self.side.filter(|s| s.id == id) {
                    self.side = None;
                    if phase == TouchPhase::Ended {
                        // The mouse leaves the first finger for the tap; a visit still under
                        // way ends at once.
                        let resume = self.finish_tap(out) || self.left || self.press_due;
                        self.press_due = false;
                        self.after_press = None;
                        if self.left {
                            self.left = false;
                            out.push(PointerAction::Cancel(MouseButton::Left));
                        }
                        out.push(PointerAction::Place(side.start));
                        self.tap = Some(SideTap { next: 1, resume });
                    }
                    return;
                }
                match self.state {
                    State::One {
                        id: first, moved, long, ..
                    } if first == id => {
                        self.last = None;
                        // A side tap's click is not lost; its way back is.
                        self.finish_tap(out);
                        // A drag after the long press's right click ends as a drag does.
                        let long = long && !(self.config.press_on_move && moved);
                        if let Some(side) = self.side.take() {
                            // The side finger is still down: this one ends now, and the side
                            // finger is the pointer from here.
                            if self.press_due {
                                self.press_due = false;
                                if phase == TouchPhase::Ended {
                                    self.left = true;
                                    out.push(PointerAction::Down(MouseButton::Left));
                                }
                            }
                            self.after_press = None;
                            out.push(PointerAction::Move(p));
                            self.release(phase == TouchPhase::Cancelled || long, out);
                            out.push(PointerAction::Place(side.start));
                            self.last = Some(side.start);
                            self.press_due = !self.config.press_on_move;
                            self.state = State::One {
                                id: side.id,
                                start: side.start,
                                since: now,
                                moved: false,
                                long: false,
                            };
                            return;
                        }
                        if self.press_due && self.after_press.is_some() {
                            // The held-back press has not happened yet: the lift's place
                            // comes after it.
                            self.after_press = Some(p);
                        } else {
                            out.push(PointerAction::Move(p));
                        }
                        if self.config.press_on_move
                            && phase == TouchPhase::Ended
                            && !self.left
                            && !self.press_due
                            && !moved
                            && !long
                        {
                            // A tap: the press it held back, then the release.
                            self.press_due = true;
                        }
                        if phase == TouchPhase::Cancelled {
                            // Not a click: a press that was still waiting never happens.
                            if self.press_due {
                                self.press_due = false;
                                self.after_press = None;
                            } else {
                                self.release_due = Some(true);
                            }
                        } else {
                            // After a long press's right click the lift is no left click too.
                            self.release_due = Some(long);
                        }
                        self.state = State::Idle;
                        self.leave_due = true;
                    }
                    State::Pinch { a, b } if a.0 == id || b.0 == id => {
                        self.last = None;
                        out.push(PointerAction::GestureEnd);
                        let rest = if a.0 == id { b } else { a };
                        self.state = if self.fingers.is_empty() {
                            State::Idle
                        } else {
                            State::Lifting {
                                rest: Some(rest).filter(|r| self.fingers.contains(&r.0)),
                            }
                        };
                        self.leave_due = true;
                    }
                    State::Lifting { .. } if self.fingers.is_empty() => self.state = State::Idle,
                    State::Lifting { rest: Some(rest) } if rest.0 == id => {
                        self.state = State::Lifting { rest: None };
                    }
                    _ => {}
                }
            }
        }
    }

    /// Once a frame, before the game runs: what waited for this frame, and the long press.
    pub fn frame(&mut self, now: f64, out: &mut Vec<PointerAction>) {
        self.resumed = false;
        if self.right_due {
            self.right_due = false;
            out.push(PointerAction::Up(MouseButton::Right));
        }
        if let Some(tap) = &mut self.tap {
            match tap.next {
                1 => out.push(PointerAction::Down(MouseButton::Left)),
                2 => out.push(PointerAction::Up(MouseButton::Left)),
                _ => {
                    // Back to the first finger: a hover where it is now, and its press again
                    // on the next frame if it had one (or earned one meanwhile).
                    let resume = tap.resume;
                    self.tap = None;
                    if let State::One { start, moved, .. } = self.state {
                        out.push(PointerAction::Place(self.last.unwrap_or(start)));
                        self.press_due = resume || (self.config.press_on_move && moved);
                        self.resume_due = self.press_due;
                    }
                    return;
                }
            }
            tap.next += 1;
            return;
        }
        if self.press_due {
            self.press_due = false;
            self.left = true;
            self.resumed = std::mem::take(&mut self.resume_due);
            out.push(PointerAction::Down(MouseButton::Left));
            if let Some(p) = self.after_press.take() {
                out.push(PointerAction::Move(p));
            }
            // Its release, if the finger is up already, is the next frame's.
            return;
        }
        if let Some(cancel) = self.release_due.take() {
            self.release(cancel, out);
            return;
        }
        if let State::One { since, moved, long, .. } = &mut self.state
            && (self.left || self.config.press_on_move)
            && !*moved
            && !*long
            && now - *since >= self.config.long_press
        {
            *long = true;
            self.right_due = true;
            out.push(PointerAction::Down(MouseButton::Right));
            return;
        }
        // (with fingers left over from a gesture too: the pointer is not parked between them)
        if self.leave_due && matches!(self.state, State::Idle | State::Lifting { .. }) {
            self.leave_due = false;
            out.push(PointerAction::Leave);
        }
    }

    /// Forget every finger (the app went to the background or lost the focus, and the ends of
    /// the touches that were down may never come): held buttons are let go without a click
    /// and the pointer leaves, now.
    pub fn reset(&mut self, out: &mut Vec<PointerAction>) {
        let was_active = self.active() || !self.fingers.is_empty();
        if std::mem::take(&mut self.right_due) {
            out.push(PointerAction::Up(MouseButton::Right));
        }
        if std::mem::take(&mut self.left) || self.tap.is_some_and(|t| t.next == 2) {
            out.push(PointerAction::Cancel(MouseButton::Left));
        }
        self.tap = None;
        self.side = None;
        self.resume_due = false;
        self.resumed = false;
        if matches!(self.state, State::Pinch { .. }) {
            out.push(PointerAction::GestureEnd);
        }
        if was_active {
            out.push(PointerAction::Leave);
        }
        self.after_press = None;
        self.state = State::Idle;
        self.fingers.clear();
        self.last = None;
        self.press_due = false;
        self.release_due = None;
        self.leave_due = false;
    }

    /// Whether the left press of this frame ([`Self::frame`]) is the first finger's own, made
    /// again after the mouse's visit to a side tap: the drag it had goes on, and a game does
    /// not take it for a new click on what is under the finger.
    pub fn press_resumed(&self) -> bool {
        self.resumed
    }

    /// Whether a finger is the pointer or a gesture is going on.
    pub fn active(&self) -> bool {
        self.state != State::Idle || self.press_due || self.release_due.is_some() || self.right_due || self.leave_due
    }

    fn in_zone(&self, p: Vec2) -> bool {
        self.tap_zones
            .iter()
            .any(|[min, max]| p.x >= min.x && p.y >= min.y && p.x <= max.x && p.y <= max.y)
    }

    /// Two fingers become a gesture: whatever one of them pressed is no click.
    fn pinch(&mut self, a: (u64, Vec2), b: (u64, Vec2), out: &mut Vec<PointerAction>) {
        self.finish_tap(out);
        self.resume_due = false;
        self.press_due = false;
        self.release_due = None;
        self.after_press = None;
        if self.left {
            self.left = false;
            out.push(PointerAction::Cancel(MouseButton::Left));
        }
        self.state = State::Pinch { a, b };
        out.push(PointerAction::Place((a.1 + b.1) * 0.5));
        out.push(PointerAction::GestureStart((a.1 + b.1) * 0.5));
    }

    /// The rest of a side tap's click, now, without the way back; whether the first finger's
    /// press was to come again.
    fn finish_tap(&mut self, out: &mut Vec<PointerAction>) -> bool {
        let Some(tap) = self.tap.take() else {
            return false;
        };
        if tap.next <= 1 {
            out.push(PointerAction::Down(MouseButton::Left));
        }
        if tap.next <= 2 {
            out.push(PointerAction::Up(MouseButton::Left));
        }
        tap.resume
    }

    fn release(&mut self, cancel: bool, out: &mut Vec<PointerAction>) {
        if self.left {
            self.left = false;
            out.push(if cancel {
                PointerAction::Cancel(MouseButton::Left)
            } else {
                PointerAction::Up(MouseButton::Left)
            });
        }
    }

    /// Everything that waited for a frame, now (a new finger is down before that frame came).
    fn flush(&mut self, out: &mut Vec<PointerAction>) {
        if self.right_due {
            self.right_due = false;
            out.push(PointerAction::Up(MouseButton::Right));
        }
        if self.press_due {
            self.press_due = false;
            self.left = true;
            out.push(PointerAction::Down(MouseButton::Left));
            if let Some(p) = self.after_press.take() {
                out.push(PointerAction::Move(p));
            }
        }
        if let Some(cancel) = self.release_due.take() {
            self.release(cancel, out);
        }
    }
}

/// What the [`PointerAction`]s do to the mouse and the gesture of `input`.
pub fn apply(input: &mut Input, actions: &[PointerAction]) {
    use PointerAction as A;
    for action in actions {
        match *action {
            A::Place(p) => input.set_mouse_position(Some(p)),
            A::Move(p) => input.inject_mouse_move(p),
            A::Leave => input.inject_mouse_left(),
            A::Down(b) => input.inject_mouse_button_down(b),
            A::Up(b) => input.inject_mouse_button_up(b),
            A::Cancel(b) => input.release_mouse_button(b, MouseButtonCause::FocusLost),
            A::Wheel(lines) => input.inject_wheel(Vec2::new(0.0, lines)),
            A::GestureStart(center) => input.inject_gesture_start(center),
            A::Gesture {
                zoom,
                rotate,
                pan,
                center,
            } => input.inject_gesture(zoom, rotate, pan, center),
            A::GestureEnd => input.inject_gesture_end(),
        }
    }
}

/// Fingers as the mouse, as a resource ([`InputPlugin`](crate::InputPlugin) adds it, off):
/// the one way a finger gets into [`Input`]. The window runner calls [`Self::touch`] for
/// every touch the system reports, with [`WindowPlugin::touch_as_mouse`](crate::WindowPlugin)
/// as [`Self::enabled`]; a test or a scripted run calls the same, so what it sees is what a
/// finger does.
#[derive(Clone, Debug)]
pub struct TouchMouse {
    /// Whether fingers move the mouse and make gestures. Off: [`Self::touch`] only records
    /// the touch ([`Input::touches`]).
    pub enabled: bool,
    /// The state machine and its thresholds (but the slop: see [`Self::scale`]).
    pub pointer: TouchPointer,
    /// Physical pixels a point of the screen: the slop is
    /// [`TouchPointerConfig::SLOP_POINTS`] of it. The window runner keeps it at the window's
    /// scale factor.
    pub scale: f32,
    clock: Instant,
}

impl Default for TouchMouse {
    fn default() -> Self {
        Self {
            enabled: false,
            pointer: TouchPointer::default(),
            scale: 1.0,
            clock: Instant::now(),
        }
    }
}

impl TouchMouse {
    /// Where [`Self::frame`] runs in `Stage::PreUpdate`. A system that feeds touches for the
    /// same frame, as the window's events do, has a lower order.
    pub const ORDER: i32 = -1000;

    /// Seconds on the clock of [`Self::touch`] and [`Self::frame`].
    pub fn now(&self) -> f64 {
        self.clock.elapsed().as_secs_f64()
    }

    /// A finger's event, now: the mouse and the gesture follow it (when [`Self::enabled`]),
    /// and `input` gets the touch itself.
    pub fn touch(&mut self, input: &mut Input, id: u64, phase: TouchPhase, p: Vec2) {
        self.touch_at(input, id, phase, p, self.now());
    }

    /// [`Self::touch`] at `now` seconds (a test's own clock, shared with [`Self::frame_at`]).
    pub fn touch_at(&mut self, input: &mut Input, id: u64, phase: TouchPhase, p: Vec2, now: f64) {
        // The mouse first: the UI takes a finger for its pointer only while there is no mouse.
        if self.enabled {
            let mut actions = Vec::new();
            self.pointer.config.slop = TouchPointerConfig::SLOP_POINTS * self.scale.max(1.0);
            self.pointer.touch(id, phase, p, now, &mut actions);
            apply(input, &actions);
        }
        input.inject_touch(id, phase, p);
    }

    /// Once a frame before the game reads `input`: the press a frame after its hover, a long
    /// press's right click.
    pub fn frame(&mut self, input: &mut Input) {
        self.frame_at(input, self.now());
    }

    /// [`Self::frame`] at `now` seconds.
    pub fn frame_at(&mut self, input: &mut Input, now: f64) {
        if !self.enabled || !self.pointer.active() {
            return;
        }
        let mut actions = Vec::new();
        self.pointer.frame(now, &mut actions);
        apply(input, &actions);
    }

    /// Forget every finger ([`TouchPointer::reset`]) and every touch of `input`.
    pub fn reset(&mut self, input: &mut Input) {
        input.clear_touches();
        let mut actions = Vec::new();
        self.pointer.reset(&mut actions);
        apply(input, &actions);
    }
}

#[cfg(test)]
mod tests {
    use super::PointerAction::*;
    use super::*;
    use crate::input::MouseButton::{Left, Right};
    use TouchPhase::{Cancelled, Ended, Moved, Started};

    const P: Vec2 = Vec2::new(100.0, 200.0);

    /// A time and what happens at it: a touch, or `None` for a frame.
    type Step = (f64, Option<(u64, TouchPhase, Vec2)>);

    /// The actions of `steps` (a touch, or `None` for a frame at that time), one list per step.
    fn run(t: &mut TouchPointer, steps: &[Step]) -> Vec<Vec<PointerAction>> {
        steps
            .iter()
            .map(|(now, touch)| {
                let mut out = Vec::new();
                match touch {
                    Some((id, phase, p)) => t.touch(*id, *phase, *p, *now, &mut out),
                    None => t.frame(*now, &mut out),
                }
                out
            })
            .collect()
    }

    /// A tap whose down and up arrive between two frames (`adb shell input tap`, a quick finger):
    /// the pointer is placed, then the press, the release and the leave take a frame each.
    #[test]
    fn a_tap_is_a_hover_then_a_click_over_three_frames() {
        let mut t = TouchPointer::default();
        let got = run(
            &mut t,
            &[
                (0.0, Some((1, Started, P))),
                (0.001, Some((1, Ended, P))),
                (0.016, None),
                (0.033, None),
                (0.05, None),
                (0.066, None),
            ],
        );
        assert_eq!(
            got,
            vec![
                vec![Place(P)],
                vec![Move(P)],
                vec![Down(Left)],
                vec![Up(Left)],
                vec![Leave],
                vec![]
            ]
        );
        assert!(!t.active());
    }

    /// The press of a finger that stays down comes on the first frame after it, not with it.
    #[test]
    fn the_press_waits_for_the_next_frame_and_a_drag_is_a_left_drag() {
        let mut t = TouchPointer::default();
        let to = P + Vec2::new(80.0, 0.0);
        let got = run(
            &mut t,
            &[
                (0.0, Some((7, Started, P))),
                (0.016, None),
                (0.02, Some((7, Moved, to))),
                (0.033, None),
                // Long after: a drag is no long press.
                (2.0, None),
                (2.1, Some((7, Ended, to))),
                (2.116, None),
                (2.133, None),
            ],
        );
        assert_eq!(
            got,
            vec![
                vec![Place(P)],
                vec![Down(Left)],
                vec![Move(to)],
                vec![],
                vec![],
                vec![Move(to)],
                vec![Up(Left)],
                vec![Leave]
            ]
        );
    }

    /// A still finger: one right click after the long-press time, the left button still down;
    /// the lift lets it go without a left click.
    #[test]
    fn a_long_press_is_one_right_click() {
        let mut t = TouchPointer::default();
        let near = P + Vec2::new(3.0, -4.0);
        let got = run(
            &mut t,
            &[
                (0.0, Some((1, Started, P))),
                (0.016, None),
                (0.2, Some((1, Moved, near))),
                (0.4, None),
                (0.52, None),
                (0.54, None),
                (1.5, None),
                (1.6, Some((1, Ended, near))),
                (1.62, None),
            ],
        );
        assert_eq!(
            got,
            vec![
                vec![Place(P)],
                vec![Down(Left)],
                vec![Move(near)],
                vec![],
                vec![Down(Right)],
                vec![Up(Right)],
                vec![],
                vec![Move(near)],
                vec![Cancel(Left)]
            ]
        );
    }

    /// The app is suspended with a finger down and its end never comes: after the reset
    /// nothing is held, and the next finger (a new id, as on iOS) is a tap, not a pinch.
    #[test]
    fn a_reset_forgets_the_fingers_that_were_down() {
        let mut t = TouchPointer::default();
        let mut out = Vec::new();
        t.touch(1, Started, P, 0.0, &mut out);
        t.frame(0.016, &mut out);
        out.clear();
        t.reset(&mut out);
        assert_eq!(out, vec![Cancel(Left), Leave]);
        assert!(!t.active());
        // Long after (the time in the background): no long press of the forgotten finger.
        out.clear();
        t.frame(60.0, &mut out);
        assert_eq!(out, vec![]);
        let got = run(
            &mut t,
            &[
                (60.1, Some((2, Started, P))),
                (60.101, Some((2, Ended, P))),
                (60.116, None),
                (60.133, None),
                (60.15, None),
            ],
        );
        assert_eq!(
            got,
            vec![
                vec![Place(P)],
                vec![Move(P)],
                vec![Down(Left)],
                vec![Up(Left)],
                vec![Leave]
            ]
        );
        // A reset in a pinch, and one with nothing down, which does nothing.
        t.touch(3, Started, P, 61.0, &mut out);
        t.touch(4, Started, P + Vec2::X * 100.0, 61.001, &mut out);
        out.clear();
        t.reset(&mut out);
        assert_eq!(out, vec![GestureEnd, Leave]);
        out.clear();
        t.reset(&mut out);
        assert_eq!(out, vec![]);
    }

    /// Spreading two fingers turns the wheel up, closing them down; the first finger's press is
    /// cancelled (no click), and the finger left after the pinch does nothing.
    #[test]
    fn a_pinch_is_the_wheel_and_never_a_click() {
        let mut t = TouchPointer::default();
        let (a, b) = (Vec2::new(400.0, 300.0), Vec2::new(600.0, 300.0));
        let wide = Vec2::new(820.5, 300.0);
        let mut out = Vec::new();
        t.touch(1, Started, a, 0.0, &mut out);
        t.frame(0.016, &mut out);
        t.touch(2, Started, b, 0.02, &mut out);
        assert_eq!(
            out,
            vec![
                Place(a),
                Down(Left),
                Cancel(Left),
                Place((a + b) * 0.5),
                GestureStart((a + b) * 0.5)
            ]
        );
        out.clear();
        // 200 px apart to 420.5: 1.05^15.2.
        t.touch(2, Moved, wide, 0.05, &mut out);
        let lines: f32 = out
            .iter()
            .filter_map(|a| match a {
                Wheel(l) => Some(*l),
                _ => None,
            })
            .sum();
        assert!((lines - 15.23).abs() < 0.05, "{lines}");
        assert!(out.contains(&Place((a + wide) * 0.5)));
        out.clear();
        t.touch(2, Moved, b, 0.08, &mut out);
        assert!(matches!(out[2], Wheel(l) if (l + 15.23).abs() < 0.05), "{out:?}");
        out.clear();
        // One finger lifts; the other moves about and lifts: no button at all.
        t.touch(2, Ended, b, 0.1, &mut out);
        t.frame(0.116, &mut out);
        t.touch(1, Moved, a + Vec2::splat(90.0), 0.2, &mut out);
        t.frame(1.0, &mut out);
        t.touch(1, Ended, a, 1.1, &mut out);
        t.frame(1.116, &mut out);
        t.frame(1.133, &mut out);
        assert_eq!(out, vec![GestureEnd, Leave]);
        assert!(!t.active());
    }

    /// A second finger before the first frame: the press that waited never happens.
    #[test]
    fn a_second_finger_drops_a_press_that_was_still_waiting() {
        let mut t = TouchPointer::default();
        let mut out = Vec::new();
        t.touch(1, Started, P, 0.0, &mut out);
        t.touch(2, Started, P + Vec2::X * 100.0, 0.001, &mut out);
        t.frame(0.016, &mut out);
        assert!(!out.iter().any(|a| matches!(a, Down(_) | Up(_) | Cancel(_))), "{out:?}");
    }

    /// Two taps faster than the frames: both clicks arrive, each press before its release.
    #[test]
    fn a_double_tap_keeps_both_clicks_in_order() {
        let mut t = TouchPointer::default();
        let mut out = Vec::new();
        t.touch(1, Started, P, 0.0, &mut out);
        t.touch(1, Ended, P, 0.001, &mut out);
        t.touch(2, Started, P, 0.002, &mut out);
        t.touch(2, Ended, P, 0.003, &mut out);
        for frame in 1..=4 {
            t.frame(frame as f64 * 0.016, &mut out);
        }
        let buttons: Vec<_> = out.iter().filter(|a| matches!(a, Down(_) | Up(_))).copied().collect();
        assert_eq!(buttons, vec![Down(Left), Up(Left), Down(Left), Up(Left)]);
        assert_eq!(out.last(), Some(&Leave));
    }

    /// A touch the system takes away (a gesture of its own) lets go without a click.
    #[test]
    fn a_cancelled_touch_is_no_click() {
        let mut t = TouchPointer::default();
        let got = run(
            &mut t,
            &[
                (0.0, Some((1, Started, P))),
                (0.016, None),
                (0.1, Some((1, Cancelled, P))),
                (0.116, None),
                (0.133, None),
            ],
        );
        assert_eq!(got[3], vec![Cancel(Left)]);
        assert_eq!(got[4], vec![Leave]);
        // Cancelled before its press: nothing but the pointer coming and going.
        let got = run(
            &mut t,
            &[
                (1.0, Some((2, Started, P))),
                (1.001, Some((2, Cancelled, P))),
                (1.016, None),
            ],
        );
        assert_eq!(got, vec![vec![Place(P)], vec![Move(P)], vec![Leave]]);
    }

    /// What two fingers did, summed over `out`: (zoom, turn in radians, pan).
    fn gesture(out: &[PointerAction]) -> (f32, f32, Vec2) {
        out.iter().fold((1.0, 0.0, Vec2::ZERO), |(z, r, p), a| match a {
            Gesture { zoom, rotate, pan, .. } => (z * zoom, r + rotate, p + *pan),
            _ => (z, r, p),
        })
    }

    /// Two fingers down at `a` and `b`, the first one's frame between them.
    fn two_fingers(t: &mut TouchPointer, a: Vec2, b: Vec2) {
        let mut out = Vec::new();
        t.touch(1, Started, a, 0.0, &mut out);
        t.frame(0.016, &mut out);
        t.touch(2, Started, b, 0.02, &mut out);
        assert_eq!(out.last(), Some(&GestureStart((a + b) * 0.5)), "{out:?}");
    }

    /// Both fingers moving apart: the zoom is the ratio of the distances, with no turn and no
    /// pan left over; without `pinch_wheel` the wheel stays still.
    #[test]
    fn a_pinch_is_a_zoom_gesture_and_the_wheel_only_when_asked() {
        let mut t = TouchPointer::new(TouchPointerConfig {
            pinch_wheel: false,
            ..Default::default()
        });
        let (a, b) = (Vec2::new(400.0, 300.0), Vec2::new(600.0, 300.0));
        two_fingers(&mut t, a, b);
        let mut out = Vec::new();
        for step in 1..=10 {
            let d = Vec2::X * 10.0 * step as f32;
            t.touch(1, Moved, a - d, 0.02 + step as f64 * 0.01, &mut out);
            t.touch(2, Moved, b + d, 0.025 + step as f64 * 0.01, &mut out);
        }
        let (zoom, turn, pan) = gesture(&out);
        assert!((zoom - 2.0).abs() < 1e-3, "{zoom}");
        assert!(turn.abs() < 1e-4 && pan.length() < 1e-3, "{turn} {pan}");
        assert!(!out.iter().any(|a| matches!(a, Wheel(_))), "{out:?}");
        out.clear();
        t.touch(1, Ended, a, 0.5, &mut out);
        assert_eq!(out, vec![GestureEnd]);
    }

    /// The fingers turn a quarter round their middle, clockwise on the screen: a quarter turn,
    /// positive, with the zoom and the pan as they were.
    #[test]
    fn a_twist_is_a_turn_gesture() {
        let mut t = TouchPointer::default();
        let c = Vec2::new(500.0, 300.0);
        let at = |angle: f32| Vec2::new(angle.cos(), angle.sin()) * 100.0;
        two_fingers(&mut t, c - at(0.0), c + at(0.0));
        let mut out = Vec::new();
        for step in 1..=9 {
            // y points down: a growing angle is clockwise to the eye.
            let angle = (step as f32 * 10.0).to_radians();
            t.touch(1, Moved, c - at(angle), step as f64 * 0.01 + 0.02, &mut out);
            t.touch(2, Moved, c + at(angle), step as f64 * 0.01 + 0.025, &mut out);
        }
        let (zoom, turn, pan) = gesture(&out);
        assert!((turn - std::f32::consts::FRAC_PI_2).abs() < 1e-3, "{turn}");
        assert!((zoom - 1.0).abs() < 1e-3 && pan.length() < 1e-2, "{zoom} {pan}");
    }

    /// Both fingers move the same way: the pan is that way, with no zoom and no turn.
    #[test]
    fn two_fingers_moving_together_are_a_pan_gesture() {
        let mut t = TouchPointer::default();
        let (a, b) = (Vec2::new(400.0, 300.0), Vec2::new(600.0, 320.0));
        two_fingers(&mut t, a, b);
        let mut out = Vec::new();
        for step in 1..=8 {
            let d = Vec2::new(0.0, -15.0 * step as f32);
            t.touch(1, Moved, a + d, step as f64 * 0.01 + 0.02, &mut out);
            t.touch(2, Moved, b + d, step as f64 * 0.01 + 0.025, &mut out);
        }
        let (zoom, turn, pan) = gesture(&out);
        assert!((pan - Vec2::new(0.0, -120.0)).length() < 1e-2, "{pan}");
        assert!((zoom - 1.0).abs() < 1e-3 && turn.abs() < 1e-3, "{zoom} {turn}");
        assert_eq!(
            out.iter().rev().find_map(|a| match a {
                Gesture { center, .. } => Some(*center),
                _ => None,
            }),
            Some((a + b) * 0.5 + Vec2::new(0.0, -120.0))
        );
    }

    fn on_move() -> TouchPointer {
        TouchPointer::new(TouchPointerConfig {
            press_on_move: true,
            ..Default::default()
        })
    }

    /// `press_on_move`: a finger that stays where it is presses nothing. After the long-press
    /// time it is one right click, and the lift is no click either.
    #[test]
    fn with_press_on_move_a_still_finger_is_the_right_click_alone() {
        let mut t = on_move();
        let near = P + Vec2::new(3.0, -4.0);
        let got = run(
            &mut t,
            &[
                (0.0, Some((1, Started, P))),
                (0.016, None),
                (0.2, Some((1, Moved, near))),
                (0.4, None),
                (0.52, None),
                (0.54, None),
                (1.5, None),
                (1.6, Some((1, Ended, near))),
                (1.62, None),
                (1.64, None),
            ],
        );
        assert_eq!(
            got,
            vec![
                vec![Place(P)],
                vec![],
                vec![Move(near)],
                vec![],
                vec![Down(Right)],
                vec![Up(Right)],
                vec![],
                vec![Move(near)],
                vec![],
                vec![Leave]
            ]
        );
        assert!(!t.active());
    }

    /// `press_on_move`: a tap is still a click, made when the finger lifts.
    #[test]
    fn with_press_on_move_a_tap_clicks_on_the_lift() {
        let mut t = on_move();
        let got = run(
            &mut t,
            &[
                (0.0, Some((1, Started, P))),
                (0.016, None),
                (0.033, None),
                (0.1, Some((1, Ended, P))),
                (0.116, None),
                (0.133, None),
                (0.15, None),
            ],
        );
        assert_eq!(
            got,
            vec![
                vec![Place(P)],
                vec![],
                vec![],
                vec![Move(P)],
                vec![Down(Left)],
                vec![Up(Left)],
                vec![Leave]
            ]
        );
        // Two taps faster than the frames: both clicks, in order.
        let mut out = Vec::new();
        t.touch(1, Started, P, 1.0, &mut out);
        t.touch(1, Ended, P, 1.001, &mut out);
        t.touch(2, Started, P, 1.002, &mut out);
        t.touch(2, Ended, P, 1.003, &mut out);
        for frame in 1..=4 {
            t.frame(1.0 + frame as f64 * 0.016, &mut out);
        }
        let buttons: Vec<_> = out.iter().filter(|a| matches!(a, Down(_) | Up(_))).copied().collect();
        assert_eq!(buttons, vec![Down(Left), Up(Left), Down(Left), Up(Left)]);
    }

    /// `press_on_move`: a finger that leaves the slop presses where it went down, on the next
    /// frame, and drags from there; however long it then rests, that is no long press.
    #[test]
    fn with_press_on_move_a_drag_presses_where_the_finger_went_down() {
        let mut t = on_move();
        let (near, far, farther) = (P + Vec2::X * 6.0, P + Vec2::X * 40.0, P + Vec2::X * 90.0);
        let got = run(
            &mut t,
            &[
                (0.0, Some((7, Started, P))),
                (0.016, None),
                (0.02, Some((7, Moved, near))),
                (0.03, Some((7, Moved, far))),
                (0.031, Some((7, Moved, farther))),
                (0.033, None),
                (0.04, Some((7, Moved, far))),
                (2.0, None),
                (2.1, Some((7, Ended, far))),
                (2.116, None),
                (2.133, None),
            ],
        );
        assert_eq!(
            got,
            vec![
                vec![Place(P)],
                vec![],
                vec![Move(near)],
                vec![Move(P)],
                vec![],
                vec![Down(Left), Move(farther)],
                vec![Move(far)],
                vec![],
                vec![Move(far)],
                vec![Up(Left)],
                vec![Leave]
            ]
        );
        // A flick that is over before its press: the press, the place it ended, the release.
        let got = run(
            &mut t,
            &[
                (3.0, Some((8, Started, P))),
                (3.001, Some((8, Moved, far))),
                (3.002, Some((8, Ended, farther))),
                (3.016, None),
                (3.033, None),
            ],
        );
        assert_eq!(
            got,
            vec![
                vec![Place(P)],
                vec![Move(P)],
                vec![],
                vec![Down(Left), Move(farther)],
                vec![Up(Left)]
            ]
        );
        // A second finger before the press: no button at all.
        let mut out = Vec::new();
        t.touch(1, Started, P, 4.0, &mut out);
        t.touch(1, Moved, far, 4.001, &mut out);
        t.touch(2, Started, farther, 4.002, &mut out);
        t.frame(4.016, &mut out);
        assert!(!out.iter().any(|a| matches!(a, Down(_) | Up(_) | Cancel(_))), "{out:?}");
    }

    /// [`TouchMouse`] is the path of a finger into [`Input`]: off, a touch is only a touch;
    /// on, the same calls move the mouse, and two fingers are [`Input::gesture`] with this
    /// frame's zoom, until the frame after they lift.
    #[test]
    fn touch_mouse_feeds_the_mouse_and_the_gesture_of_input() {
        let mut input = Input::default();
        let mut mouse = TouchMouse::default();
        mouse.touch_at(&mut input, 1, Started, P, 0.0);
        assert_eq!(input.touches().len(), 1);
        assert_eq!(input.mouse_position(), None);
        mouse.touch_at(&mut input, 1, Ended, P, 0.01);
        input.end_frame();

        mouse.enabled = true;
        mouse.scale = 2.0;
        mouse.pointer.config.pinch_wheel = false;
        let (a, b) = (Vec2::new(400.0, 300.0), Vec2::new(600.0, 300.0));
        mouse.touch_at(&mut input, 1, Started, a, 1.0);
        assert_eq!(mouse.pointer.config.slop, 20.0);
        assert_eq!(input.mouse_position(), Some(a));
        mouse.frame_at(&mut input, 1.016);
        assert!(input.mouse_pressed(MouseButton::Left));
        input.end_frame();
        mouse.touch_at(&mut input, 2, Started, b, 1.02);
        assert!(!input.mouse_pressed(MouseButton::Left));
        let g = *input.gesture().expect("two fingers are a gesture");
        assert!(g.just_started && g.start == (a + b) * 0.5 && g.zoom == 1.0);
        mouse.frame_at(&mut input, 1.033);
        input.end_frame();
        mouse.touch_at(&mut input, 2, Moved, Vec2::new(700.0, 300.0), 1.04);
        mouse.touch_at(&mut input, 2, Moved, Vec2::new(800.0, 300.0), 1.045);
        let g = *input.gesture().unwrap();
        assert!(!g.just_started && (g.zoom - 2.0).abs() < 1e-4, "{g:?}");
        assert_eq!((g.pan, g.center), (Vec2::new(100.0, 0.0), Vec2::new(600.0, 300.0)));
        assert_eq!(input.wheel_delta(), Vec2::ZERO);
        input.end_frame();
        // A frame with still fingers: the gesture is there and did nothing.
        let g = *input.gesture().unwrap();
        assert_eq!((g.zoom, g.rotate, g.pan), (1.0, 0.0, Vec2::ZERO));
        mouse.touch_at(&mut input, 2, Moved, Vec2::new(600.0, 300.0), 1.06);
        mouse.touch_at(&mut input, 2, Ended, Vec2::new(600.0, 300.0), 1.07);
        // The frame of the lift still has what the fingers did last.
        assert!((input.gesture().unwrap().zoom - 0.5).abs() < 1e-4);
        input.end_frame();
        assert_eq!(input.gesture(), None);
        // A reset in a gesture ends it too.
        mouse.touch_at(&mut input, 1, Ended, a, 1.1);
        mouse.touch_at(&mut input, 3, Started, a, 2.0);
        mouse.touch_at(&mut input, 4, Started, b, 2.01);
        assert!(input.gesture().is_some());
        mouse.reset(&mut input);
        input.end_frame();
        assert_eq!(input.gesture(), None);
        assert!(input.touches().is_empty() && input.mouse_position().is_none());
    }

    /// `press_on_move`: a finger that rests through the long press and then goes is a drag
    /// all the same (the right click came first): the press where it went down, the release
    /// when it lifts.
    #[test]
    fn with_press_on_move_a_hold_and_then_a_drag_is_a_drag() {
        let mut t = on_move();
        let far = P + Vec2::X * 60.0;
        let got = run(
            &mut t,
            &[
                (0.0, Some((1, Started, P))),
                (0.52, None),
                (0.54, None),
                (0.8, Some((1, Moved, far))),
                (0.816, None),
                (0.9, Some((1, Moved, P + Vec2::X * 90.0))),
                (1.6, Some((1, Ended, far))),
                (1.62, None),
                (1.64, None),
            ],
        );
        assert_eq!(
            got,
            vec![
                vec![Place(P)],
                vec![Down(Right)],
                vec![Up(Right)],
                vec![Move(P)],
                vec![Down(Left), Move(far)],
                vec![Move(P + Vec2::X * 90.0)],
                vec![Move(far)],
                vec![Up(Left)],
                vec![Leave]
            ]
        );
        assert!(!t.active());
    }

    fn two_thumbs() -> TouchPointer {
        let mut t = TouchPointer::new(TouchPointerConfig {
            press_on_move: true,
            pinch_wheel: false,
            side_tap: true,
            ..Default::default()
        });
        t.tap_zones = vec![[Vec2::new(1400.0, 600.0), Vec2::new(1600.0, 700.0)]];
        t
    }

    const BUTTON: Vec2 = Vec2::new(1500.0, 650.0);

    /// One thumb steers (a drag), the other taps a button: the mouse lets go without a click,
    /// clicks the button, and comes back to the steering thumb, pressed again where that thumb
    /// is by then. No gesture, and the steering goes on.
    #[test]
    fn a_side_tap_clicks_and_the_first_fingers_drag_goes_on() {
        let mut t = two_thumbs();
        let (a, b, c, d) = (
            Vec2::new(300.0, 400.0),
            Vec2::new(400.0, 400.0),
            Vec2::new(420.0, 380.0),
            Vec2::new(450.0, 380.0),
        );
        let got = run(
            &mut t,
            &[
                (0.0, Some((1, Started, a))),
                (0.01, Some((1, Moved, b))),
                (0.016, None),
                (1.0, Some((2, Started, BUTTON))),
                (1.01, Some((2, Moved, BUTTON + Vec2::splat(30.0)))),
                (1.02, Some((1, Moved, c))),
                (1.08, Some((2, Ended, BUTTON + Vec2::splat(30.0)))),
                (1.09, Some((1, Moved, d))),
                (1.1, None),
                (1.116, None),
                (1.133, None),
                (1.14, Some((1, Moved, c))),
                (1.15, None),
            ],
        );
        assert!(t.press_resumed(), "the press after the visit is the old drag's");
        let mut got = got;
        got.extend(run(
            &mut t,
            &[
                (1.2, Some((1, Moved, d))),
                (2.0, Some((1, Ended, d))),
                (2.016, None),
                (2.033, None),
            ],
        ));
        assert!(!t.press_resumed());
        assert_eq!(
            got,
            vec![
                vec![Place(a)],
                vec![Move(a)],
                vec![Down(Left), Move(b)],
                vec![],
                vec![],
                vec![Move(c)],
                vec![Cancel(Left), Place(BUTTON)],
                vec![],
                vec![Down(Left)],
                vec![Up(Left)],
                vec![Place(d)],
                vec![],
                vec![Down(Left), Move(c)],
                vec![Move(d)],
                vec![Move(d)],
                vec![Up(Left)],
                vec![Leave]
            ]
        );
        assert!(!t.active());
    }

    /// Off the buttons: while the first finger drags, a second that taps is a click there (a
    /// target picked while steering); one that travels is a pinch after all, and two fingers
    /// that land together are a pinch as before.
    #[test]
    fn off_the_zones_a_side_finger_taps_only_while_the_first_drags() {
        let world = Vec2::new(900.0, 300.0);
        let (a, b) = (Vec2::new(300.0, 400.0), Vec2::new(400.0, 400.0));
        let drag = |t: &mut TouchPointer| {
            let mut out = Vec::new();
            t.touch(1, Started, a, 0.0, &mut out);
            t.touch(1, Moved, b, 0.01, &mut out);
            t.frame(0.016, &mut out);
        };
        // A tap on the world.
        let mut t = two_thumbs();
        drag(&mut t);
        let mut out = Vec::new();
        t.touch(2, Started, world, 1.0, &mut out);
        t.touch(2, Ended, world, 1.05, &mut out);
        for frame in 1..=5 {
            t.frame(1.05 + frame as f64 * 0.016, &mut out);
        }
        assert_eq!(
            out,
            vec![Cancel(Left), Place(world), Down(Left), Up(Left), Place(b), Down(Left)]
        );
        // It travels: a pinch.
        let mut t = two_thumbs();
        drag(&mut t);
        let mut out = Vec::new();
        t.touch(2, Started, world, 1.0, &mut out);
        assert_eq!(out, vec![]);
        t.touch(2, Moved, world + Vec2::X * 40.0, 1.02, &mut out);
        let mid = (b + world + Vec2::X * 40.0) * 0.5;
        assert_eq!(out, vec![Cancel(Left), Place(mid), GestureStart(mid)]);
        // Together, the first one still: a pinch at once.
        let mut t = two_thumbs();
        let mut out = Vec::new();
        t.touch(1, Started, a, 0.0, &mut out);
        t.touch(2, Started, world, 0.03, &mut out);
        assert_eq!(out.last(), Some(&GestureStart((a + world) * 0.5)));
        // A still first finger and a second on a button: the button's click, and no press of
        // the first.
        let mut t = two_thumbs();
        let mut out = Vec::new();
        t.touch(1, Started, a, 0.0, &mut out);
        t.frame(0.016, &mut out);
        t.touch(2, Started, BUTTON, 0.1, &mut out);
        t.touch(2, Ended, BUTTON, 0.15, &mut out);
        for frame in 1..=5 {
            t.frame(0.15 + frame as f64 * 0.016, &mut out);
        }
        assert_eq!(out, vec![Place(a), Place(BUTTON), Down(Left), Up(Left), Place(a)]);
        // Without `side_tap` a second finger is a gesture wherever it lands.
        let mut t = on_move();
        t.tap_zones = vec![[Vec2::new(1400.0, 600.0), Vec2::new(1600.0, 700.0)]];
        drag(&mut t);
        let mut out = Vec::new();
        t.touch(2, Started, BUTTON, 1.0, &mut out);
        assert_eq!(out.last(), Some(&GestureStart((b + BUTTON) * 0.5)));
    }

    /// The ends of a side tap that come in another order: the first finger lifts while the
    /// second is down (the second is the pointer from there and its lift a tap), the first
    /// lifts while the mouse is at the tap (the click is made, nothing is pressed again, and
    /// the lift of a drag is no click on the world), and a reset in the middle holds nothing.
    #[test]
    fn a_side_tap_survives_the_first_finger_lifting() {
        let (a, b) = (Vec2::new(300.0, 400.0), Vec2::new(400.0, 400.0));
        let mut t = two_thumbs();
        let mut out = Vec::new();
        t.touch(1, Started, a, 0.0, &mut out);
        t.touch(1, Moved, b, 0.01, &mut out);
        t.frame(0.016, &mut out);
        t.touch(2, Started, BUTTON, 1.0, &mut out);
        out.clear();
        t.touch(1, Ended, b, 1.02, &mut out);
        t.frame(1.033, &mut out);
        t.touch(2, Ended, BUTTON, 1.05, &mut out);
        for frame in 1..=3 {
            t.frame(1.05 + frame as f64 * 0.016, &mut out);
        }
        assert_eq!(
            out,
            vec![
                Move(b),
                Up(Left),
                Place(BUTTON),
                Move(BUTTON),
                Down(Left),
                Up(Left),
                Leave
            ]
        );
        assert!(!t.active());

        let mut out = Vec::new();
        t.touch(1, Started, a, 2.0, &mut out);
        t.touch(1, Moved, b, 2.01, &mut out);
        t.frame(2.016, &mut out);
        t.touch(2, Started, BUTTON, 3.0, &mut out);
        t.touch(2, Ended, BUTTON, 3.05, &mut out);
        out.clear();
        t.touch(1, Ended, b, 3.06, &mut out);
        for frame in 1..=3 {
            t.frame(3.06 + frame as f64 * 0.016, &mut out);
        }
        assert_eq!(out, vec![Down(Left), Up(Left), Move(b), Leave]);
        assert!(!t.active());

        let mut out = Vec::new();
        t.touch(1, Started, a, 4.0, &mut out);
        t.touch(1, Moved, b, 4.01, &mut out);
        t.frame(4.016, &mut out);
        t.touch(2, Started, BUTTON, 5.0, &mut out);
        t.touch(2, Ended, BUTTON, 5.05, &mut out);
        t.frame(5.066, &mut out);
        out.clear();
        t.reset(&mut out);
        assert_eq!(out, vec![Cancel(Left), Leave]);
        out.clear();
        t.frame(6.0, &mut out);
        assert_eq!(out, vec![]);
        assert!(!t.active());
    }

    /// One finger of a pinch lifts: the pointer leaves on the next frame though a finger is
    /// still down, and a finger put down again makes a new gesture with the one that stayed,
    /// from where that one is by now.
    #[test]
    fn a_finger_put_back_after_a_pinch_is_a_new_gesture() {
        let mut t = TouchPointer::default();
        let (a, b) = (Vec2::new(400.0, 300.0), Vec2::new(600.0, 300.0));
        two_fingers(&mut t, a, b);
        let mut out = Vec::new();
        t.touch(2, Ended, b, 0.1, &mut out);
        t.frame(0.116, &mut out);
        assert_eq!(out, vec![GestureEnd, Leave]);
        out.clear();
        let moved = a + Vec2::Y * 50.0;
        t.touch(1, Moved, moved, 0.2, &mut out);
        t.touch(3, Started, b, 0.3, &mut out);
        assert_eq!(out, vec![Place((moved + b) * 0.5), GestureStart((moved + b) * 0.5)]);
        out.clear();
        t.touch(1, Ended, moved, 0.4, &mut out);
        t.touch(3, Ended, b, 0.41, &mut out);
        t.frame(0.416, &mut out);
        assert_eq!(out, vec![GestureEnd, Leave]);
        assert!(!t.active());
    }
}
