//! What the runner asks the Android activity through JNI: winit 0.30 and `android-activity`'s
//! NativeActivity glue do not pass these on.
//!
//! * **Whether the activity is destroyed.** `onDestroy` runs on Java's UI thread and waits for
//!   `android_main` to return; winit only logs the event and its loop goes on, so the UI thread
//!   waits for ever, the process stays and the next start hangs. The runner asks
//!   `Activity.isDestroyed()` while the app is in the background and leaves the loop when it is.
//! * **Finishing the activity** when the game itself exits, so that the task ends with it.
//! * **The window's insets** (system bars and display cutout): an edge-to-edge window's content
//!   rectangle is the whole window, the insets are Java's to give.
//!
//! * **The on-screen keyboard's height** (`WindowInsets.Type.ime()`, Android 11 on): a
//!   NativeActivity's window is not resized for the keyboard and nothing tells it that one is up.
//!
//! * **The system bars hidden** (sticky immersive, [`crate::set_immersive`]): asked on Java's
//!   main thread, since a view's methods throw on any other.
//!
//! * **The gamepads connected** ([`gamepads`]): winit has no event for a pad that went away (and
//!   hides the device number of the events it does pass on), so the runner asks
//!   `InputDevice.getDeviceIds()` once a second ([`crate::gamepad::AndroidPads::os_pads`]).
//!
//! Every call attaches the calling thread to the VM (cheap once attached) and treats a Java
//! exception as "no answer".

use crate::mobile::AndroidApp;
use crate::window::SafeArea;
use jni::objects::{JIntArray, JObject, JString};
use jni::refs::Global;
use jni::{JValue, JavaVM, jni_sig, jni_str};

/// Run `f` with the activity's object. `None`: no VM, or Java threw.
fn with_activity<T>(app: &AndroidApp, f: impl FnOnce(&mut jni::Env, &JObject) -> jni::errors::Result<T>) -> Option<T> {
    let vm = app.vm_as_ptr();
    let raw = app.activity_as_ptr() as jni::sys::jobject;
    if vm.is_null() || raw.is_null() {
        return None;
    }
    // SAFETY: the pointer is the process's `JavaVM`, valid for the process's life.
    let vm = unsafe { JavaVM::from_raw(vm.cast()) };
    vm.attach_current_thread(|env| -> jni::errors::Result<T> {
        // SAFETY: a global reference `app` owns and keeps while it lives; not deleted here.
        let activity = unsafe { env.as_cast_raw::<Global<JObject>>(&raw)? };
        f(env, activity.as_ref())
    })
    .ok()
}

/// `Activity.isDestroyed()`: true from the start of `onDestroy` on. `false` when not known.
pub(crate) fn activity_destroyed(app: &AndroidApp) -> bool {
    with_activity(app, |env, activity| {
        env.call_method(activity, jni_str!("isDestroyed"), jni_sig!("()Z"), &[])?
            .z()
    })
    .unwrap_or(false)
}

/// `Activity.finish()`: the activity goes away as after Back (pause, stop, destroy).
pub(crate) fn finish_activity(app: &AndroidApp) -> bool {
    with_activity(app, |env, activity| {
        env.call_method(activity, jni_str!("finish"), jni_sig!("()V"), &[])?;
        Ok(())
    })
    .is_some()
}

/// What the on-screen keyboard covers of the window's bottom, in pixels
/// (`getRootWindowInsets().getInsets(ime()).bottom`; 0 while it is hidden). `None`: before
/// Android 11, no insets yet, or Java threw.
pub(crate) fn keyboard_inset(app: &AndroidApp) -> Option<u32> {
    if AndroidApp::sdk_version() < 30 {
        return None;
    }
    with_activity(app, |env, activity| {
        let window = env
            .call_method(
                activity,
                jni_str!("getWindow"),
                jni_sig!("()Landroid/view/Window;"),
                &[],
            )?
            .l()?;
        let decor = env
            .call_method(
                &window,
                jni_str!("getDecorView"),
                jni_sig!("()Landroid/view/View;"),
                &[],
            )?
            .l()?;
        let insets = env
            .call_method(
                &decor,
                jni_str!("getRootWindowInsets"),
                jni_sig!("()Landroid/view/WindowInsets;"),
                &[],
            )?
            .l()?;
        if insets.is_null() {
            return Ok(None);
        }
        let ime = env
            .call_static_method(
                jni_str!("android/view/WindowInsets$Type"),
                jni_str!("ime"),
                jni_sig!("()I"),
                &[],
            )?
            .i()?;
        let covered = env
            .call_method(
                &insets,
                jni_str!("getInsets"),
                jni_sig!("(I)Landroid/graphics/Insets;"),
                &[JValue::Int(ime)],
            )?
            .l()?;
        Ok(Some(
            env.get_field(&covered, jni_str!("bottom"), jni_sig!(jint))?.i()?.max(0) as u32,
        ))
    })
    .flatten()
}

/// What the system bars that are shown and the display cutout cover of the window, in pixels
/// (`getRootWindowInsets().getInsets(systemBars | displayCutout)`, Android 11 on; the system
/// window insets before). `None`: the window has no insets yet, or Java threw.
pub(crate) fn window_insets(app: &AndroidApp) -> Option<SafeArea> {
    with_activity(app, |env, activity| {
        let window = env
            .call_method(
                activity,
                jni_str!("getWindow"),
                jni_sig!("()Landroid/view/Window;"),
                &[],
            )?
            .l()?;
        let decor = env
            .call_method(
                &window,
                jni_str!("getDecorView"),
                jni_sig!("()Landroid/view/View;"),
                &[],
            )?
            .l()?;
        let insets = env
            .call_method(
                &decor,
                jni_str!("getRootWindowInsets"),
                jni_sig!("()Landroid/view/WindowInsets;"),
                &[],
            )?
            .l()?;
        if insets.is_null() {
            return Ok(None);
        }
        if AndroidApp::sdk_version() >= 30 {
            let class = jni_str!("android/view/WindowInsets$Type");
            let bars = env
                .call_static_method(class, jni_str!("systemBars"), jni_sig!("()I"), &[])?
                .i()?;
            let cutout = env
                .call_static_method(class, jni_str!("displayCutout"), jni_sig!("()I"), &[])?
                .i()?;
            // (immersive: the bars are hidden, and one an edge swipe shows for a while covers
            // nothing the UI should move for; a cutout always covers, seen or not)
            let (method, sig, mask) = if crate::mobile::immersive() {
                (
                    jni_str!("getInsetsIgnoringVisibility"),
                    jni_sig!("(I)Landroid/graphics/Insets;"),
                    cutout,
                )
            } else {
                (
                    jni_str!("getInsets"),
                    jni_sig!("(I)Landroid/graphics/Insets;"),
                    bars | cutout,
                )
            };
            let covered = env.call_method(&insets, method, sig, &[JValue::Int(mask)])?.l()?;
            let mut side = |name| -> jni::errors::Result<u32> {
                Ok(env.get_field(&covered, name, jni_sig!(jint))?.i()?.max(0) as u32)
            };
            Ok(Some(SafeArea {
                left: side(jni_str!("left"))?,
                top: side(jni_str!("top"))?,
                right: side(jni_str!("right"))?,
                bottom: side(jni_str!("bottom"))?,
            }))
        } else if crate::mobile::immersive() {
            // Immersive before Android 11: the cutout's safe insets (Android 9 and 10), none
            // before that (no cutouts)
            if AndroidApp::sdk_version() < 28 {
                return Ok(Some(SafeArea::default()));
            }
            let cutout = env
                .call_method(
                    &insets,
                    jni_str!("getDisplayCutout"),
                    jni_sig!("()Landroid/view/DisplayCutout;"),
                    &[],
                )?
                .l()?;
            if cutout.is_null() {
                return Ok(Some(SafeArea::default()));
            }
            let mut side = |name| -> jni::errors::Result<u32> {
                Ok(env.call_method(&cutout, name, jni_sig!("()I"), &[])?.i()?.max(0) as u32)
            };
            Ok(Some(SafeArea {
                left: side(jni_str!("getSafeInsetLeft"))?,
                top: side(jni_str!("getSafeInsetTop"))?,
                right: side(jni_str!("getSafeInsetRight"))?,
                bottom: side(jni_str!("getSafeInsetBottom"))?,
            }))
        } else {
            let mut side = |name| -> jni::errors::Result<u32> {
                Ok(env.call_method(&insets, name, jni_sig!("()I"), &[])?.i()?.max(0) as u32)
            };
            Ok(Some(SafeArea {
                left: side(jni_str!("getSystemWindowInsetLeft"))?,
                top: side(jni_str!("getSystemWindowInsetTop"))?,
                right: side(jni_str!("getSystemWindowInsetRight"))?,
                bottom: side(jni_str!("getSystemWindowInsetBottom"))?,
            }))
        }
    })
    .flatten()
}

/// Hide the status and navigation bars, sticky (an edge swipe shows them for a while), and let
/// the window draw into a display cutout on the short edges. Posted to Java's main thread
/// (`AndroidApp::run_on_java_main_thread`): `View` and `WindowInsetsController` throw
/// `CalledFromWrongThreadException` on the game's thread. A Java exception is logged, not raised.
pub(crate) fn hide_system_bars(app: &AndroidApp) {
    let posted = app.clone();
    app.run_on_java_main_thread(Box::new(move || {
        let sdk = AndroidApp::sdk_version();
        let done = with_activity(&posted, |env, activity| {
            let window = env
                .call_method(
                    activity,
                    jni_str!("getWindow"),
                    jni_sig!("()Landroid/view/Window;"),
                    &[],
                )?
                .l()?;
            // Android 9 on: draw under a cutout on the short edges (landscape: the side the
            // camera is on); the safe area keeps the UI clear of it.
            if sdk >= 28 {
                let attrs = env
                    .call_method(
                        &window,
                        jni_str!("getAttributes"),
                        jni_sig!("()Landroid/view/WindowManager$LayoutParams;"),
                        &[],
                    )?
                    .l()?;
                // LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES
                env.set_field(
                    &attrs,
                    jni_str!("layoutInDisplayCutoutMode"),
                    jni_sig!(jint),
                    JValue::Int(1),
                )?;
                env.call_method(
                    &window,
                    jni_str!("setAttributes"),
                    jni_sig!("(Landroid/view/WindowManager$LayoutParams;)V"),
                    &[JValue::Object(&attrs)],
                )?;
            }
            if sdk >= 30 {
                env.call_method(
                    &window,
                    jni_str!("setDecorFitsSystemWindows"),
                    jni_sig!("(Z)V"),
                    &[JValue::Bool(false)],
                )?;
                let controller = env
                    .call_method(
                        &window,
                        jni_str!("getInsetsController"),
                        jni_sig!("()Landroid/view/WindowInsetsController;"),
                        &[],
                    )?
                    .l()?;
                if controller.is_null() {
                    return Ok(false);
                }
                // BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
                env.call_method(
                    &controller,
                    jni_str!("setSystemBarsBehavior"),
                    jni_sig!("(I)V"),
                    &[JValue::Int(2)],
                )?;
                let bars = env
                    .call_static_method(
                        jni_str!("android/view/WindowInsets$Type"),
                        jni_str!("systemBars"),
                        jni_sig!("()I"),
                        &[],
                    )?
                    .i()?;
                env.call_method(&controller, jni_str!("hide"), jni_sig!("(I)V"), &[JValue::Int(bars)])?;
            } else {
                let decor = env
                    .call_method(
                        &window,
                        jni_str!("getDecorView"),
                        jni_sig!("()Landroid/view/View;"),
                        &[],
                    )?
                    .l()?;
                // IMMERSIVE_STICKY | FULLSCREEN | HIDE_NAVIGATION | LAYOUT_FULLSCREEN
                // | LAYOUT_HIDE_NAVIGATION | LAYOUT_STABLE
                let flags = 0x1000 | 0x4 | 0x2 | 0x400 | 0x200 | 0x100;
                env.call_method(
                    &decor,
                    jni_str!("setSystemUiVisibility"),
                    jni_sig!("(I)V"),
                    &[JValue::Int(flags)],
                )?;
            }
            Ok(true)
        });
        match done {
            Some(true) => tracing::info!("Android: the system bars are hidden (sticky immersive, API {sdk})"),
            _ => tracing::warn!("Android: the system bars could not be hidden (API {sdk})"),
        }
    }));
}

/// `InputDevice.SOURCE_GAMEPAD`: the device has gamepad buttons.
const SOURCE_GAMEPAD: i32 = 0x0000_0401;

/// The gamepads Android has now: the device number and name of every input device that is not
/// virtual and has gamepad buttons (`InputDevice.getSources() & SOURCE_GAMEPAD`). `None`: no VM,
/// or Java threw. Compiled for Android, **not yet run on a device**.
pub(crate) fn gamepads(app: &AndroidApp) -> Option<Vec<(i32, String)>> {
    with_activity(app, |env, _activity| {
        let class = jni_str!("android/view/InputDevice");
        let ids = env
            .call_static_method(class, jni_str!("getDeviceIds"), jni_sig!("()[I"), &[])?
            .l()?;
        if ids.is_null() {
            return Ok(Vec::new());
        }
        let ids = JIntArray::cast_local(env, ids)?;
        let mut numbers = vec![0i32; ids.len(env)?];
        ids.get_region(env, 0, &mut numbers)?;
        let mut pads = Vec::new();
        for number in numbers {
            let device = env
                .call_static_method(
                    class,
                    jni_str!("getDevice"),
                    jni_sig!("(I)Landroid/view/InputDevice;"),
                    &[JValue::Int(number)],
                )?
                .l()?;
            // (gone between the two calls)
            if device.is_null() {
                continue;
            }
            let sources = env
                .call_method(&device, jni_str!("getSources"), jni_sig!("()I"), &[])?
                .i()?;
            let is_virtual = env
                .call_method(&device, jni_str!("isVirtual"), jni_sig!("()Z"), &[])?
                .z()?;
            if is_virtual || sources & SOURCE_GAMEPAD != SOURCE_GAMEPAD {
                continue;
            }
            let name = env
                .call_method(&device, jni_str!("getName"), jni_sig!("()Ljava/lang/String;"), &[])?
                .l()?;
            let name = if name.is_null() {
                String::new()
            } else {
                env.cast_local::<JString>(name)?.try_to_string(env)?
            };
            pads.push((number, name));
        }
        Ok(pads)
    })
}
