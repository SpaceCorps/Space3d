//! The real gamepad backend, checked without a pad (Windows): a virtual Xbox 360 pad is
//! plugged in through ViGEmBus, its buttons, sticks and triggers are driven by a script, it is
//! pulled out while a button is held and plugged in again, and every step is compared with
//! what [`Input`] shows.
//!
//! ```text
//! SPACE3D_BACKGROUND=0 cargo run -p space3d-platform --example virtual_pad
//! ```
//! Needs the ViGEmBus driver (<https://github.com/nefarius/ViGEmBus>) and a **focused** window:
//! Windows gives pad input to the focused window only, so the run takes the focus for about
//! 20 seconds (`SPACE3D_BACKGROUND=0` where background mode is automatic). Exit code 0: every
//! step matched; 1: a step did not; 2: no driver, or the window never had the focus.
//!
//! `--log-only` opens the window and prints what any pad does (a real pad on the desk)
//! until the window is closed.

#[cfg(not(target_os = "windows"))]
fn main() {
    eprintln!("virtual_pad: Windows only (ViGEmBus)");
}

#[cfg(target_os = "windows")]
fn main() {
    win::main();
}

#[cfg(target_os = "windows")]
mod win {
    use space3d_core::prelude::*;
    use space3d_platform::gamepad::{STICK_DEAD_ZONE, dead_zone};
    use space3d_platform::*;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};
    use vigem_client::{XButtons, XGamepad};

    /// What the app saw last frame.
    #[derive(Clone, Debug, Default)]
    struct Seen {
        frames: u64,
        focused: bool,
        ever_focused: bool,
        /// (id, name, held buttons, axes in `AXES` order) of every pad.
        pads: Vec<(u32, String, Vec<GamepadButton>, [f32; 6])>,
        /// Set by the script when it is through: the exit code.
        done: Option<i32>,
    }

    const AXES: [GamepadAxis; 6] = [
        GamepadAxis::LeftStickX,
        GamepadAxis::LeftStickY,
        GamepadAxis::RightStickX,
        GamepadAxis::RightStickY,
        GamepadAxis::LeftTrigger,
        GamepadAxis::RightTrigger,
    ];

    const BUTTONS: [GamepadButton; 17] = [
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

    pub(super) fn main() {
        let log_only = std::env::args().any(|a| a == "--log-only");
        let seen = Arc::new(Mutex::new(Seen::default()));
        let mut app = App::new();
        app.add_plugin(PlatformPlugin {
            window: WindowPlugin::titled("Space3d: virtual pad (needs the focus)"),
        });
        let shared = seen.clone();
        let mut last = String::from("?");
        app.add_system(Stage::Update, move |ctx| {
            let focused = ctx.resources.expect::<WindowState>().focused;
            let input = ctx.resources.expect::<Input>();
            for change in input.gamepad_changes() {
                println!("[input] {change:?}");
            }
            let mut ids: Vec<u32> = input.gamepad_ids().collect();
            ids.sort_unstable();
            let pads: Vec<_> = ids
                .into_iter()
                .filter_map(|id| {
                    let pad = input.gamepad(id)?;
                    let held: Vec<GamepadButton> = BUTTONS.into_iter().filter(|b| pad.buttons.pressed(*b)).collect();
                    Some((id, pad.name().to_string(), held, AXES.map(|a| pad.axis(a))))
                })
                .collect();
            let line = pads
                .iter()
                .map(|(id, name, held, axes)| format!("pad {id} \"{name}\" held {held:?} axes {axes:.2?}"))
                .collect::<Vec<_>>()
                .join(" | ");
            if line != last {
                println!(
                    "[input] focused={focused} {}",
                    if line.is_empty() { "no pad" } else { &line }
                );
                last = line;
            }
            let mut seen = shared.lock().expect("seen");
            seen.frames += 1;
            seen.focused = focused;
            seen.ever_focused |= focused;
            seen.pads = pads;
            if let Some(code) = seen.done {
                println!("[example] exit code {code}");
                // The code leaves with the process: the runner's own exit is 0.
                std::process::exit(code);
            }
        });
        if !log_only {
            std::thread::spawn(move || {
                let code = script(&seen);
                seen.lock().expect("seen").done = Some(code);
            });
        }
        app.run();
    }

    fn sleep(ms: u64) {
        std::thread::sleep(Duration::from_millis(ms));
    }

    /// Wait until `ok(seen)` holds (at most `ms`), then say whether it did.
    fn wait(seen: &Mutex<Seen>, ms: u64, ok: impl Fn(&Seen) -> bool) -> (bool, Seen) {
        let until = Instant::now() + Duration::from_millis(ms);
        loop {
            let now = seen.lock().expect("seen").clone();
            if ok(&now) {
                return (true, now);
            }
            if Instant::now() >= until {
                return (false, now);
            }
            sleep(10);
        }
    }

    struct Run<'a> {
        seen: &'a Mutex<Seen>,
        failed: u32,
        passed: u32,
    }

    impl Run<'_> {
        fn check(&mut self, what: &str, ms: u64, ok: impl Fn(&Seen) -> bool) {
            let (good, now) = wait(self.seen, ms, ok);
            if good {
                self.passed += 1;
                println!("[step] PASS {what}");
            } else {
                self.failed += 1;
                println!("[step] FAIL {what}: saw focused={} pads={:?}", now.focused, now.pads);
            }
        }
    }

    fn stick(v: f32) -> i16 {
        (v.clamp(-1.0, 1.0) * 32767.0) as i16
    }

    fn script(seen: &Mutex<Seen>) -> i32 {
        let client = match vigem_client::Client::connect() {
            Ok(c) => c,
            Err(e) => {
                println!("[example] no ViGEmBus driver: {e:?}");
                return 2;
            }
        };
        // The window first, and its focus: without it Windows delivers nothing.
        let (focused, _) = wait(seen, 15_000, |s| s.frames > 30 && s.focused);
        if !focused {
            println!(
                "[example] UNVERIFIED: the window did not get the focus in 15 s; Windows gives pad input to the focused window only"
            );
            return 2;
        }
        let before = seen.lock().expect("seen").pads.len();
        println!("[example] window focused; {before} pad(s) there before the virtual one");
        let mut target = vigem_client::Xbox360Wired::new(client, vigem_client::TargetId::XBOX360_WIRED);
        let mut run = Run {
            seen,
            failed: 0,
            passed: 0,
        };
        if let Err(e) = target.plugin().and_then(|()| target.wait_ready()) {
            println!("[example] cannot plug the virtual pad in: {e:?}");
            return 2;
        }
        run.check("plugged in: a pad with a name appears", 8_000, |s| {
            s.pads.len() == before + 1 && s.pads.iter().any(|p| !p.1.is_empty())
        });
        // The virtual pad is the one with the highest id (the newest).
        let pad = |s: &Seen| s.pads.last().cloned();
        sleep(300);

        let buttons = [
            (XButtons::A, GamepadButton::South, "A"),
            (XButtons::B, GamepadButton::East, "B"),
            (XButtons::X, GamepadButton::West, "X"),
            (XButtons::Y, GamepadButton::North, "Y"),
            (XButtons::LB, GamepadButton::LeftShoulder, "LB"),
            (XButtons::RB, GamepadButton::RightShoulder, "RB"),
            (XButtons::START, GamepadButton::Start, "Start"),
            (XButtons::BACK, GamepadButton::Select, "Back/View"),
            (XButtons::LTHUMB, GamepadButton::LeftThumb, "L3"),
            (XButtons::RTHUMB, GamepadButton::RightThumb, "R3"),
            (XButtons::UP, GamepadButton::DPadUp, "d-pad up"),
            (XButtons::DOWN, GamepadButton::DPadDown, "d-pad down"),
            (XButtons::LEFT, GamepadButton::DPadLeft, "d-pad left"),
            (XButtons::RIGHT, GamepadButton::DPadRight, "d-pad right"),
        ];
        for (raw, want, name) in buttons {
            let state = XGamepad {
                buttons: XButtons { raw },
                ..Default::default()
            };
            let _ = target.update(&state);
            run.check(&format!("{name} down is {want:?} alone"), 1_500, |s| {
                pad(s).is_some_and(|p| p.2 == [want])
            });
            let _ = target.update(&XGamepad::default());
            run.check(&format!("{name} up: nothing held"), 1_500, |s| {
                pad(s).is_some_and(|p| p.2.is_empty())
            });
        }

        // Sticks: (lx, ly, rx, ry), up positive on both sides of the driver.
        let sticks: [(f32, f32, f32, f32); 7] = [
            (0.5, 0.0, 0.0, 0.0),
            (0.0, -1.0, 0.0, 0.0),
            (0.7, 0.7, 0.0, 0.0),
            (0.1, 0.05, 0.0, 0.0),
            (0.0, 0.0, -1.0, 0.0),
            (0.0, 0.0, 0.0, 0.6),
            (-0.3, 0.4, 0.8, -0.2),
        ];
        for (lx, ly, rx, ry) in sticks {
            let state = XGamepad {
                thumb_lx: stick(lx),
                thumb_ly: stick(ly),
                thumb_rx: stick(rx),
                thumb_ry: stick(ry),
                ..Default::default()
            };
            let _ = target.update(&state);
            let l = dead_zone(glam::Vec2::new(lx, ly), STICK_DEAD_ZONE);
            let r = dead_zone(glam::Vec2::new(rx, ry), STICK_DEAD_ZONE);
            let want = [l.x, l.y, r.x, r.y];
            run.check(
                &format!("sticks raw ({lx}, {ly}) ({rx}, {ry}) read {want:.2?} within 0.05"),
                1_500,
                |s| pad(s).is_some_and(|p| (0..4).all(|i| (p.3[i] - want[i]).abs() <= 0.05)),
            );
        }
        let _ = target.update(&XGamepad::default());
        run.check("sticks at rest read 0", 1_500, |s| {
            pad(s).is_some_and(|p| p.3[..4].iter().all(|v| *v == 0.0))
        });

        // Triggers: the value within 0.05, the button from one half.
        for (lt, rt) in [(0.25f32, 0.0f32), (1.0, 0.0), (0.0, 0.6), (0.0, 0.0)] {
            let state = XGamepad {
                left_trigger: (lt * 255.0) as u8,
                right_trigger: (rt * 255.0) as u8,
                ..Default::default()
            };
            let _ = target.update(&state);
            let mut held = Vec::new();
            if lt >= 0.5 {
                held.push(GamepadButton::LeftTrigger);
            }
            if rt >= 0.5 {
                held.push(GamepadButton::RightTrigger);
            }
            run.check(
                &format!("triggers ({lt}, {rt}) within 0.05, buttons {held:?}"),
                1_500,
                |s| pad(s).is_some_and(|p| (p.3[4] - lt).abs() <= 0.05 && (p.3[5] - rt).abs() <= 0.05 && p.2 == held),
            );
        }

        // Pulled out while A is held and the stick is pushed: nothing stays behind.
        let state = XGamepad {
            buttons: XButtons { raw: XButtons::A },
            thumb_lx: stick(1.0),
            ..Default::default()
        };
        let _ = target.update(&state);
        run.check("A held and the stick right before the unplug", 1_500, |s| {
            pad(s).is_some_and(|p| p.2 == [GamepadButton::South] && p.3[0] > 0.95)
        });
        if let Err(e) = target.unplug() {
            println!("[example] unplug failed: {e:?}");
        }
        run.check("unplugged mid-hold: the pad is gone, nothing held", 8_000, |s| {
            s.pads.len() == before
        });
        sleep(500);

        // Plugged in again: input comes again, and starts clean.
        if let Err(e) = target.plugin().and_then(|()| target.wait_ready()) {
            println!("[example] cannot plug the virtual pad in again: {e:?}");
            run.failed += 1;
        }
        run.check("plugged in again: a pad, nothing held, sticks at rest", 8_000, |s| {
            s.pads.len() == before + 1 && pad(s).is_some_and(|p| p.2.is_empty() && p.3.iter().all(|v| *v == 0.0))
        });
        sleep(300);
        let state = XGamepad {
            buttons: XButtons { raw: XButtons::X },
            thumb_ly: stick(1.0),
            ..Default::default()
        };
        let _ = target.update(&state);
        run.check("after the second plug: X and the stick up arrive", 1_500, |s| {
            pad(s).is_some_and(|p| p.2 == [GamepadButton::West] && p.3[1] > 0.95)
        });
        let _ = target.update(&XGamepad::default());
        sleep(200);
        let _ = target.unplug();
        run.check("unplugged at the end", 8_000, |s| s.pads.len() == before);

        let ever = seen.lock().expect("seen").ever_focused;
        println!(
            "[example] {} passed, {} failed (window had the focus: {ever})",
            run.passed, run.failed
        );
        if run.failed == 0 { 0 } else { 1 }
    }
}
