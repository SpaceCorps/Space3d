# Platforms

One `build(app: &mut App)` function runs everywhere; `space3d::main!(build)` generates the
platform entry points:

| target | generated | what runs |
|---|---|---|
| Windows / macOS / Linux | `pub fn main()` | `App::new()` → `build` → `app.run()` (winit event loop) |
| Web (`wasm32-unknown-unknown`) | `#[wasm_bindgen(start)]` fn (+ a no-op `main`) | panic hook + console logging, then the same; `run()` returns and the browser drives frames |
| Android | `#[no_mangle] fn android_main(AndroidApp)` | `space3d_platform::android_main_entry` (NativeActivity) |
| iOS | `#[no_mangle] extern "C" fn start_app()` | called from the Xcode project's `main.m`; never returns |

Android and iOS load a *library*, so projects created by `space3d new` keep the game in
`src/lib.rs` and have two thin entry crates: `src/main.rs` (desktop + web binary) and `mobile/`
(`crate-type = ["cdylib", "staticlib"]`). Desktop and web builds therefore never link the mobile
library types.

`space3d doctor` checks everything below and prints the fix for anything missing.

## Desktop

```sh
space3d run [--release]
space3d build --release            # binary + assets/ in dist/desktop/
```

Linux needs `libasound2-dev libudev-dev libxkbcommon-dev libwayland-dev` (audio, input,
keyboard, Wayland). Graphics: DX12/Vulkan (Windows), Metal (macOS), Vulkan with GLES fallback
(Linux). Typing Chinese, Japanese or Korean into a text field goes through the OS input method:
see [Input methods (IME)](#input-methods-ime).

## Web

```sh
rustup target add wasm32-unknown-unknown
space3d serve                      # build + serve http://localhost:8080 (COOP/COEP, no-cache)
space3d serve --watch              # rebuild on changes to src/, assets/, web/, Cargo.toml
space3d serve --host               # listen on 0.0.0.0 (open it on a phone on the same network)
space3d build --target web --release   # dist/web/
```

What `space3d build --target web` does:

1. `cargo build --target wasm32-unknown-unknown --bin <game>`.
2. Reads the `wasm-bindgen` version from `Cargo.lock` and checks the installed
   `wasm-bindgen` CLI. They must match exactly; if not it prints
   `cargo install wasm-bindgen-cli --version <x.y.z> --locked` (or installs it with
   `--install-tools`).
3. `wasm-bindgen --target web --no-typescript --out-name <game> --out-dir dist/web …`.
4. Release builds: `wasm-opt -Oz` when binaryen is installed.
5. Copies `web/` (your `index.html`, icons, …) and `assets/` into `dist/web`.

The page loads the bundle with `import init from "./<game>.js"; init();`. Set
`WindowPlugin::canvas_id` to render into an existing `<canvas id="…">` (the templates use
`space3d`); otherwise a canvas is appended to `<body>`. winit gives the canvas an inline CSS
size from `WindowPlugin::width`/`height`; to let the page decide (e.g. fill the viewport), use
`!important` in your stylesheet as the templates do
(`canvas#space3d { width: 100vw !important; height: 100vh !important; }`) — the surface follows
the canvas' real size.

Web specifics: the renderer is created asynchronously (WebGPU when available, WebGL2 otherwise —
WebGL2 uses the CPU draw path); assets are fetched over HTTP relative to the page; the database
uses `localStorage` (KV, save slots, leaderboards; raw SQL returns `Unsupported`); WebSocket and
HTTP clients use the browser APIs; servers are native-only; audio starts after a user gesture.

Graphics on the web ([graphics.md](graphics.md#webgl2-and-other-downlevel-devices)): WebGPU
browsers get every feature as natively (`GraphicsQuality::recommended` picks Medium or High);
WebGL2 gets `GraphicsQuality::Low` — render scale 0.75, FXAA instead of MSAA, a 4-level bloom
(where `EXT_color_buffer_float` exists; without it the Ldr fallback tonemaps per fragment and has
no bloom), a 1024² PCF4 shadow, the Classic shading model with a small environment bake spread
over frames, and the CPU particle backend (8 192 particles, thinner spawns). Every feature has a
WebGL2 path (no compute, no storage buffers in vertex shaders). WebGL2 links each shader
program when a pipeline is first built (50–150 ms): prewarm on loading screens
(`GraphicsQuality::prewarm_request`). `examples/graphics_showcase` builds for the web
(`cargo run -p space3d-cli -- serve` from its directory; `web/index.html`), and
`DownlevelOverride::webgl2()` shows the WebGL2 look on a desktop GPU (the
`Engine/Showcase/WebGL2 *` stories).

## Android

One-time setup (the exact commands for a Windows PC without Android Studio are in
SpaceCorps2027's `docs/MOBILE.md`):

```sh
# SDK: platform-tools, a platform, build-tools, the NDK (sdkmanager or Android Studio)
export ANDROID_HOME="$HOME/Library/Android/sdk"        # Linux: $HOME/Android/Sdk
export ANDROID_NDK_ROOT="$ANDROID_HOME/ndk/<version>"
cargo install cargo-apk --locked
# rust-toolchain.toml already lists aarch64-linux-android (phones) and x86_64-linux-android (PC emulators)
```

Build and install:

```sh
space3d build --target android [--release]     # = cargo apk build -p <game>-mobile --lib --target aarch64-linux-android
adb install -r dist/android/<game>.apk
adb logcat -s RustStdoutStderr
```

The engine's own proof is `examples/hello_cube_mobile` (first run on a device: 2026-11-12, the
Android 16 x86_64 emulator, cargo-apk 0.10.0, NDK 30):

```sh
cargo apk build -p hello_cube_mobile --lib --target x86_64-linux-android
adb install -r target/debug/apk/hello_cube.apk
adb shell am start -n org.spacecorps.space3d.hellocube/android.app.NativeActivity
```

Packaging metadata (application id, SDK levels, orientation, permissions, bundled `assets/`)
lives in `[package.metadata.android]` of `mobile/Cargo.toml`; add `strip = "strip"` there, or a
debug build's library carries hundreds of MB of debug info into the APK. What the engine does
on Android:

- **Entry**: `space3d::main!` generates `android_main` (NativeActivity; no Java code).
  `space3d::platform::android_app()` is the activity's handle from the game's `build` on, and
  `space3d::platform::mobile::data_dir()` the folder the app may write in.
- **Assets**: the default asset source is `AndroidAssetSource`, the APK's `assets/` read through
  the activity's asset manager (they are zip entries, not files). Hot reload is off.
- **Rendering**: Vulkan (GLES fallback). wgpu's debug flags are off even in a debug build: the
  emulator's Vulkan driver crashes when objects are named (set `WGPU_DEBUG=1` to ask for them).
- **Lifecycle**: in the background the surface is dropped and the loop sleeps (no frames, no
  systems); on resume the surface is recreated, meshes and textures survive. What must stop at
  once hangs on `space3d_core::BackgroundHooks` (no frame runs to read an event):
  `AudioPlugin` pauses everything that plays there and resumes it on return.
- **The activity's end**: when the activity is destroyed (Back, the task swiped away, a
  re-creation `configChanges` does not cover) the runner leaves the loop and
  `android_main_entry` ends the process with `_exit`. winit 0.30 only logs `onDestroy`, whose
  Java side waits for `android_main` to return: without this the process stayed and the next
  start hung (seen on the emulator; `platform/src/android.rs` asks `Activity.isDestroyed()`
  through JNI while the app is in the background). A game that exits by itself finishes the
  activity first (`Activity.finish()`), then the same happens. List every change the game
  handles itself in the manifest's `configChanges` (the example's `Cargo.toml` has the list),
  or a split screen, a keyboard plugged in or a new system language restarts the process.
- **Input**: multi-touch `Touch`es, `Binding::TouchRegion`; `WindowPlugin::touch_as_mouse` makes
  fingers the mouse too, for controls and a UI written for one (`platform::touch_pointer`): a
  tap is a left click whose press comes a frame after the pointer (so a UI knows whose click it
  is), a drag a left drag, a long press (0.5 s, still) a right click, a two-finger pinch the
  wheel, and the pointer leaves when the finger lifts. The lift after a long press is no left
  click. Going to the background or losing the focus forgets every finger
  (`TouchPointer::reset`, `Input::clear_touches`). While a `space3d-ui` text field
  has the focus (`TextInput::wanted`) the on-screen keyboard is up; it is asked for again a
  few frames after every tap (a lift within the slop; a finger that scrolled or dragged does
  not bring it back, and each finger has its own slot, so a second finger does not lose the
  first one's tap) and after a resume, because nothing tells the app that the player put it
  away.
  Two fingers are also a gesture, `Input::gesture()`: this frame's zoom (the ratio of the
  fingers' distance), turn (radians, clockwise on the screen positive) and pan (pixels), with
  the point between the fingers. The machine is the resource `touch_pointer::TouchMouse`
  (`InputPlugin` adds it, off; `touch_as_mouse` turns it on): the window runner sends every
  finger through `TouchMouse::touch`, and a test or a scripted run calls the same, so it works
  without a window. Its per-frame step is a `PreUpdate` system at `TouchMouse::ORDER`. Two
  options in `TouchMouse::pointer.config`: `press_on_move` holds the left press back until
  the finger leaves the slop (the press lands where the finger went down) or lifts (a tap),
  so a still finger presses nothing and a long press is the right click alone, for a game
  that steers by holding the button; `pinch_wheel = false` leaves two fingers to the gesture.
- **Typing is ASCII only.** A NativeActivity has no `InputConnection`, and winit 0.30's
  Android backend sends no `Ime` events: text arrives only as key events, which is what a
  keyboard sends for Latin letters, digits and punctuation. Cyrillic, CJK, swipe typing and
  autocorrect text do not arrive (read in winit's and android-activity's source; only ASCII
  was typed on the emulator). The way out is GameActivity (winit's `android-game-activity`,
  which has GameTextInput) and a Gradle build, or a Java activity with an `InputConnection`.
- **HTTPS**: `space3d_net::http_client_builder()` checks certificates against the compiled-in
  Mozilla roots (reqwest's platform verifier needs a Java class a NativeActivity app has not).

- **`SafeArea`** (resource): what the status bar, the navigation bar and a cutout cover of the
  window; `UiPlugin::inside_safe_area` lays the UI out inside it. Read from Java through JNI
  (`getRootWindowInsets().getInsets(systemBars | displayCutout)`, Android 11 on; the system
  window insets before), again when the window's size changes and every 30 frames. Seen on
  the emulator (Pixel 7 profile, landscape, 2400 x 1080): left 136 (the camera cutout), top 74
  (the status bar), bottom 63 (the gesture bar). Only when Java gives no answer: the
  activity's content rectangle, and for an edge-to-edge window 24 dp at the bottom.

- **`KeyboardInset`** (resource): how many pixels of the window's bottom the on-screen
  keyboard covers (`bottom`), measured from Java (`getRootWindowInsets().getInsets(ime())`,
  Android 11 on) every few frames while a text field wants text, and 0 when none does. A
  NativeActivity's window is not resized for the keyboard, so the game moves what is being
  typed in above that line itself. iOS and the desktop do not report it (always 0).

Not done: the engine does not move the view when the keyboard covers a text field (it only
measures the keyboard, `KeyboardInset`, and only on Android), stopping the audio stream
in the background (the sounds are paused, the stream stays open), compressed textures (ASTC),
and `space3d build --target android` still builds arm64 only.

## iOS

**What was built and run (2026-11-12, Mac mini, Xcode 27.2, iPhone 18 Pro Simulator on iOS
27.2):** `hello_cube_mobile` as a program bundled by `scripts/ios-bundle.sh`, no Xcode project.
It draws the cube on Metal (`adapter=Apple iOS simulator GPU`), loads `assets/proof.png` from
the bundle, and comes back from the background in the same process. `cargo build` for
`aarch64-apple-ios` (a device) was not run for this example; nothing ran on a device. The
XcodeGen route further down was **not** run (xcodegen is not installed on the Mac).

```sh
rustup target add aarch64-apple-ios-sim aarch64-apple-ios   # rust-toolchain.toml lists them
# The mobile library built as a program: a Rust `main` runs the app (`space3d::main!` makes
# one), and cargo links the frameworks itself.
cargo rustc -p hello_cube_mobile --lib --crate-type bin --target aarch64-apple-ios-sim
scripts/ios-bundle.sh target/aarch64-apple-ios-sim/debug/hello_cube_mobile HelloCube   org.spacecorps.space3d.hellocube target/ios examples/hello_cube_mobile/assets
SIM="iPhone 18 Pro"                                         # a device name or UDID
xcrun simctl boot "$SIM"; xcrun simctl bootstatus "$SIM"
xcrun simctl install "$SIM" target/ios/HelloCube.app
xcrun simctl launch --stdout=/tmp/hello.log --stderr=/tmp/hello.log "$SIM" org.spacecorps.space3d.hellocube
xcrun simctl io "$SIM" screenshot hello.png
```

What iOS needed from the engine (all in `space3d-platform` unless said):

- **Scenes.** An app linked with the iOS 27 SDK that has not adopted the scene life cycle is
  stopped at launch (a trap in `UIApplicationEvaluateRuntimeIssueForNoSceneLifecycleAdoption`;
  winit 0.30 has no scenes). `src/ios.rs` registers a scene delegate class,
  `Space3dSceneDelegate`, and puts winit's window into the scene when it connects; the app's
  `Info.plist` must name that class in `UIApplicationSceneManifest` (the bundle script and the
  template's `project.yml` do).
- **The window is the screen.** `WindowPlugin`'s width and height are not passed to winit on
  iOS (they made a 1280 x 720 point window, a third of it off screen), and the renderer and
  `Input` take `outer_size` (winit's `inner_size` is the safe area only).
- **The loop waits.** `ControlFlow::Wait` on iOS: a redraw request becomes `drawRect` at the
  display's pace. With `Poll` the main run loop span hundreds of thousands of times a second.
- **Log.** winit's iOS loop logs a warning and an info line of its own in every frame;
  `LogPlugin`'s default filter silences that target on iOS (a game with its own filter adds
  `winit::platform_impl::ios=error`).
- **Assets** (`space3d-assets`): a relative root is the bundle's folder. **Files:**
  `mobile::data_dir()` is `Library/Application Support` in the app's container.
- **`SafeArea`** (resource): the view's `safeAreaInsets` in pixels; `UiPlugin::inside_safe_area`
  lays the UI out inside it (off by default). **Not right in landscape in the Simulator:**
  UIKit answers the portrait insets (62 points top, 34 bottom) for a landscape scene, so the
  runner turns such insets (`SafeArea::turned_for_landscape`: the top's at both sides, two
  thirds of the bottom's). A guess until a device shows what UIKit answers there.
- **`UiPlugin::short_side_points`**: a UI scale from the screen's short side, for a UI made for
  a laptop's window.

The on-screen keyboard's Return arrives from winit as the text "
" of an unidentified key;
the runner makes it the Enter key, pressed and released (unit test in `runner.rs`; not tried
in the Simulator).

Not done or not seen: touches and the on-screen keyboard (the code paths are winit's; `simctl`
cannot tap), rotation, audio, a device, signing, `space3d build --target ios`.

### The Xcode route (`space3d build --target ios`; not run)

One-time setup (macOS only):

```sh
# Install Xcode from the App Store (the Command Line Tools alone are not enough: C dependencies
# such as SQLite and rustls' aws-lc need the iOS SDK), then:
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
rustup target add aarch64-apple-ios aarch64-apple-ios-sim
brew install xcodegen
```

Build:

```sh
space3d build --target ios --sim        # simulator: cargo build -p <game>-mobile --target aarch64-apple-ios-sim,
                                        # then xcodegen + xcodebuild
space3d build --target ios --release    # device (needs signing in Xcode)
```

The Xcode project is generated from `ios/project.yml` by XcodeGen: an app target with
`main.m` (calls the Rust `start_app()`), `assets/` as a resource folder, the Rust static library
linked from `target/aarch64-apple-ios[-sim]/<profile>/`, and a pre-build phase that runs
`cargo build -p <game>-mobile` for the SDK and configuration being built — so ⌘R in Xcode
always runs current Rust code. To run on a device, open `ios/<game>.xcodeproj` and pick your
signing team.

Manual equivalent:

```sh
cargo build -p <game>-mobile --lib --target aarch64-apple-ios-sim
(cd ios && xcodegen generate)
xcodebuild -project ios/<game>.xcodeproj -scheme <game> -sdk iphonesimulator \
  -configuration Debug -derivedDataPath target/xcode CODE_SIGNING_ALLOWED=NO build
xcrun simctl install booted target/xcode/Build/Products/Debug-iphonesimulator/<game>.app
xcrun simctl launch booted <bundle id>
```

## Keyboard layouts

Players use AZERTY, QWERTZ, Dvorak, JIS and many other layouts. Every keyboard event carries
two identities, and the engine keeps both:

* **Physical key** (`KeyCode`) — the key's *position*, named after the US QWERTY key in that
  place. `Input::pressed`, `just_pressed` and `Binding::Key` use it. `KeyCode::KeyW` is the key
  right of Tab on every keyboard, labelled W on QWERTY and Z on AZERTY.
* **Logical key** (`LogicalKey`, winit's `Key`) — what the key *means* on the player's layout.
  The window runner records it on a `KeyEvent` per OS event: `physical`, `logical_key` (with
  Shift/AltGr/Option applied), `key_without_modifiers` (desktop only), `text`, `location`,
  `pressed`, `repeat`. Read this frame's events with `Input::key_events()`, or the last logical
  key of a physical key with `Input::logical_key(KeyCode)`.

**Bindings.** Bind physical keys for movement and anything position-based: WASD bound as
`KeyW/KeyA/KeyS/KeyD` is `Z/Q/S/D` on AZERTY — still an inverted T under the left hand. What
the player must see is the label on *their* keyboard, not the code name:

```rust
let input = ctx.resources.expect::<Input>();
let cap = input.key_cap(KeyCode::KeyW);           // "W" on QWERTY, "Z" on AZERTY — never empty
let label = input.key_label(KeyCode::KeyW);       // Some("z") / None while unknown
let space = input.key_label(KeyCode::Space);      // Some("Space") — English: localize it
let gen = input.keyboard_layout().generation();   // changes when labels may have changed
```

`key_label` returns what the key types **without modifiers** (dead keys: their base character,
`^`), or an English name for non-character keys (`Space`, `Enter`, `Shift`, `F1`, `Numpad 1`;
`Command`/`Option` on Apple platforms, `Windows` on Windows). `key_label_info` adds the shifted
character and the source (`Os`, `Learned`, `Named`); `key_cap` upper-cases letters as keyboards
print them (`Q`, but AZERTY's `é`/`2` key stays `é`) and falls back to the US QWERTY character
of the position while the layout's is unknown.

**UI.** egui receives the logical key (`Event::Key::key`) and the physical one
(`physical_key`), like egui-winit, so text fields and shortcuts follow the layout as in native
apps: `Ctrl/Cmd+Z/A/C/V/X` act on the keys labelled Z/A/C/V/X (on AZERTY, undo is the key in
QWERTY's W position). Layouts without Latin letters (Cyrillic, Greek, …) fall back to the
physical key, so shortcuts still work in the QWERTY positions.

Hotkeys read from egui follow the layout too: `i.key_pressed(egui::Key::M)` fires on the key
*labelled* M — `KeyCode::Semicolon` on AZERTY, `KeyU` for `G` on Dvorak — where it used to fire
on physical `KeyM`. Code that compares such a hotkey with physical-key bindings (skip the hotkey
while an action is bound to the same key) must translate it first, or the two disagree on
other layouts:

```rust
let input = ctx.resources.expect::<Input>();
// The physical key egui reads as M right now (None: the key labelled M is not known yet).
let m_key = space3d_ui::physical_key_for_egui_key(input, egui::Key::M);
let m_free = m_key.is_none_or(|k| !bound_to_an_action(k));
```

Alternatively read position-based hotkeys from `Input::just_pressed(KeyCode::KeyM)` and show
`Input::key_cap(KeyCode::KeyM)`. `physical_key_for_egui_key` is cheap enough for every frame:
it walks a reverse map (label → physical key) that the layout caches per `generation()` and
does not allocate while the labels stay the same. Use the map directly for "which key types
this character":

```rust
let layout = ctx.resources.expect::<Input>().keyboard_layout();
let m = layout.key_for_label("m");                     // Some(KeyCode::Semicolon) on AZERTY
let digit = layout.key_for_label_matching(|t| t.chars().all(|c| c.is_ascii_digit()));
let text: Option<&str> = layout.label_text(KeyCode::KeyA); // borrowed, no allocation
```

When several keys print the same text, the first in keyboard order (`writing_system_keys()`)
wins.

**Fake and sample layouts.** `KeyboardLayout::set_os_layout(id, labels)` sets a layout table
directly (custom platforms, tests, screenshots of another layout's key caps). On macOS,
Windows and the web the window runner re-reads the OS table (once per second, on focus), which
replaces such a table; pin it to keep it:

```rust
let layout = ctx.resources.expect_mut::<Input>().keyboard_layout_mut();
layout.set_os_layout(Some("sample-azerty".into()), azerty_labels);
layout.set_pinned(true);  // no OS re-reads, key events teach nothing, until unpinned
// ...
layout.set_pinned(false); // the runner re-reads the OS table on its next frame
```

While pinned, explicit calls (`set_os_layout`, `learn`, `clear_learned`, `clear_os_layout`)
still apply; `refresh_from_os` / `Input::refresh_keyboard_layout` do nothing.

**Where labels come from.** Learned labels (from key events on the current layout) win over the
OS table; the OS table is read by the window runner on the event-loop thread at start-up, on
focus gain and when the layout id changes (checked at most once per second) — never per frame.

| platform | OS table | layout switch detected | notes |
|---|---|---|---|
| macOS | `TISCopyCurrentKeyboardLayoutInputSource` + `UCKeyTranslate`, main thread only | yes (input source id, keyboard type) | first query at start-up initializes Text Input Services (~35 ms, once); IMEs report the layout they type through; ISO and JIS keys: see below |
| Windows | `MapVirtualKeyExW` + `ToUnicodeEx` for the thread's `HKL` | yes (`HKL`) | |
| Web | `navigator.keyboard.getLayoutMap()` (Chromium-based browsers), async | on focus | Firefox/Safari, or iframes without the `keyboard-map` permission: learned only; no shifted characters |
| Linux X11/Wayland | none (no hard `libxkbcommon` dependency) | no | learned from winit's `key_without_modifiers` |
| Android, iOS | none | no | learned from unmodified presses (Shift/Alt presses only teach the shifted character) |

Limitations:

* Without an OS table a character key has no label until the player presses it once; after an
  undetected layout switch the old learned labels stay until each key is pressed again (a key
  that types a new character forgets the shifted character learned on the old layout).
* Key-cap capitalization without a known shifted character is a heuristic (letter rows only).
* Caps Lock on the web and mobile: letters are learned in lower case (`"A"` → `"a"`), but where
  Caps Lock also locks Shift on other keys (the digit row of Windows AZERTY) a press teaches the
  shifted character until the key is pressed without Caps Lock.
* macOS Text Input Sources are main-thread only (macOS 14+ aborts the process otherwise): the
  runner reads them on the event-loop thread, and `Input::refresh_keyboard_layout` reads nothing
  on other threads (headless tests on libtest threads use learned labels only).
* macOS ISO keyboards (most European Macs): winit 0.30 reports both the key left of `1` (`§`
  on US/British, `@` on French, `^` on German Macs; virtual key `0x0A`) and the key right of
  left Shift (`` ` `` on US/British, `<` on French/German; `0x32`) as `KeyCode::Backquote`. The
  runner reads the virtual key of the event AppKit is dispatching (`NSApp.currentEvent`) and
  reports the key right of left Shift as `KeyCode::IntlBackslash` (its W3C position, as on
  Windows); each key has its own OS label (egui gets its character's key, else
  `egui::Key::IntlBackslash`, like egui-winit). A `Backquote` binding (such as the UI
  console's default key) now fires on the key left of `1` only. If winit delivers a key event
  late (it queues events that arrive while it is handling another), the virtual key is unknown
  and either key reads `Backquote` (a release still releases the key its press named), so on
  ISO keyboards `key_label(Backquote)` comes from the OS table only and is not learned from
  presses.
* macOS JIS keyboards: winit 0.30 has no code for the Ro key (`_`, left of right Shift; virtual
  key `0x5E`); the runner reports it as `KeyCode::IntlRo` with its OS label.
* A table set with `set_os_layout` lasts until the next OS check unless pinned (see above).

## Input methods (IME)

A player who types Chinese, Japanese or Korean does it through an input method (IME): the keys go
to the IME, which shows what is being composed (the *preedit*: `nihao`) and a list of candidates,
and sends the finished text (the *commit*: 你好) when one is picked. winit gives a window an input
method only when the app asks for it (`Window::set_ime_allowed`); without it Windows takes the IME
out of the window (the language bar's mode switch does nothing, pinyin keys arrive as Latin
letters), macOS and X11 bypass the IME, and Wayland never enables text input. The runner used to
ask on phones only, so a game with a text field could not be typed Chinese into on a desktop.

**What is on where**

| | input method | how |
|---|---|---|
| Windows | yes | `ImmAssociateContextEx` (winit); `Enabled`, `Preedit`, `Commit`, `Disabled` from `WM_IME_*`; the IME's keys still arrive, with the logical key `Process` |
| macOS | yes | `interpretKeyEvents` while allowed. winit sends a commit only for text that was **composed** (pinyin, kana, dead keys: there is marked text); a key with no composition (plain Latin letters) still arrives as a key event with its text, so `Input::text()` has those and `ime_events()` the composed ones. Switching the input source (the layout) while the IME is on sends a `Disabled` |
| Linux (X11) | yes | XIM/ibus (winit); an allowed input context loses dead keys during a composition; the candidate list gets the caret's position only (winit ignores the size on X11: the XIM spot is the caret's top-left, where the list opens is not verified) |
| Linux (Wayland) | yes | text-input-v3 (winit); an empty `Preedit` may come with no composition: it is ignored |
| Web | **no** | winit's web backend has no IME and sends no `Ime` event. A game with a touch keyboard (a hidden `<input>`) makes egui's IME events itself; the engine adds none, so the page's text is entered once. A desktop browser with a mouse has no IME path at all |
| Android | **no** | a NativeActivity has no `InputConnection`: the on-screen keyboard's text arrives as key events (ASCII). See [Android](#android) |
| iOS | as before | the on-screen keyboard arrives as key events with text; nothing changed |

Linux has no CJK font bundled either: the game ships one or loads the system's
([UI: fonts](ui.md#fonts-for-chinese-japanese-and-korean)).

**What the engine does** (`WindowPlugin::ime`, on by default; desktops only):

1. **The IME is on only while a text field has the focus.** `space3d-ui` sets
   `TextInput::wanted` from egui (`text_edit_focused()`); the runner calls
   `set_ime_allowed(wanted)` after the frame when that changes (never twice for the same state).
   Outside a text field the IME is off, so flight keys and hotkeys stay raw even when the
   player's IME is in Chinese mode. A **password field never gets one** (`TextInput::password`,
   from egui's `IMEOutput::purpose`): the IME goes off when it takes the focus. (winit's
   `set_ime_purpose` does nothing on Windows and X11, so "off" is the only way to keep a
   password out of an IME.) The IME follows the focus **one frame later**: what `TextInput`
   says is what the pass drew, and a pass in which the focus moves (`request_focus` after both
   fields were drawn, the sign-in form's Enter) still reports the field the focus left; a
   composition cannot start in 16 ms.
2. **The composition is forwarded.** The runner puts winit's `Ime` events in
   `Input::ime_events()` (`ImeEvent::{Enabled, Preedit { text, cursor }, Commit, Disabled}`; a
   frame's worth, in order, cleared in `end_frame`). The preedit's cursor is in **characters**:
   winit counts bytes (你好 is 6 bytes, 2 characters) and `ImeEvent::from_winit` converts, clamped
   to the text. `space3d-ui` turns them into egui's `Event::Ime`, **before** the frame's key
   events: a preedit with text starts or updates the composition (egui writes it into the bound
   `String`, underlined, until it ends); an empty preedit, a commit or a `Disabled` ends it. egui
   0.36 ignores its (deprecated) `Enabled` and `Disabled`, so the translator sends an empty
   preedit for a `Disabled` that finds a composition, which takes the preedit text out of the
   field **while that field still has the focus**: a `TextEdit` takes no event of an input method
   without it. A field that loses the focus in the middle of a composition (a click on the flight
   view, the chat closing) keeps the letters as plain text, and the IME is off after it; a game
   that wants them gone drops them itself when its field loses the focus (egui's `Event::Ime`
   tells it a composition was going on). A commit is **not also `Input::text()`**: egui would
   enter it twice.
3. **The candidate list opens at the caret.** `space3d-ui` reads the focused field's
   `PlatformOutput::ime` at the end of the pass (the caret's rectangle, `IMEOutput::cursor_rect`,
   in points) and sets `TextInput::caret` in physical pixels (points times the pass's pixels
   per point; `ImeArea`). The runner calls `Window::set_ime_cursor_area` when it changes, and
   once more when a composition starts or the IME comes back on (an IME may reset its windows
   when its mode or language changes). **While a composition is going on the area is not
   followed:** egui puts the caret at the *end* of the preedit, so it moves right with every
   letter, and a candidate list that followed it would slide along the pinyin. The window keeps
   the area from before the composition (the caret where the preedit starts) and is told the new
   caret when the composition ends. The caret's rectangle and not the field's, because X11
   and Wayland take a position and a cursor rectangle; the field's (`IMEOutput::rect`) is a
   one-line change.
4. **Keys the IME processed are not the app's.** On Windows the IME sees a key first and winit
   still reports it, with the logical key `Process` (`VK_PROCESSKEY`): the Esc that cancels a
   composition, the Enter that commits the letters, the Backspace that deletes one of them, the
   arrows and Space that pick a candidate. The runner drops those presses (and the release of a
   key it never saw go down; the release of a held key goes through, or the key would stay
   held), so Esc does not close a chat and Backspace does not delete the text before the
   composition. These keys come from the OS's input method; the runner drops them whenever the
   support is enabled (not only while the IME is on), so a key the IME processed just as it
   went off is no ghost key press either.

5. **A click into the field ends a composition at the OS.** egui counts the preedit as plain text
   from the moment the player clicks into the field (its caret moves, `Memory::request_focus`
   interrupts the composition): the letters stay in the field, as they do anywhere. The OS
   composition would go on, and its next preedit would be written beside the old one. So
   `space3d-ui` passes egui's `IMEOutput::should_interrupt_composition` on (`TextInput::interrupt`)
   and the runner, **while a composition is going on**, turns the IME off and on again
   (`set_ime_allowed(false)`, `true`) and tells the UI with a `Disabled`. Only the first pass of a
   run of them counts (a game that calls `request_focus` every frame would otherwise never let a
   composition live), and a request with no composition going on does nothing. What the IME does
   with the letters it was composing when its context goes away (drops them, or commits them) is
   its own: see "Not done".

The order of events differs by platform. Windows: `Enabled`, `Preedit("n")`, `Preedit("ni")`...,
then `Preedit("")`, `Commit("你好")`, `Disabled`. macOS: an empty `Preedit` before each `Commit`.
Wayland: empty `Preedit`s with no composition. Rely only on this: a `Preedit` with text starts or
updates a composition; an empty one, a `Commit` or a `Disabled` ends it; a `Commit` needs no
`Preedit` before it.

**The kill switch.** `SPACE3D_NO_IME=1` (also `true`, `yes`, `on`) in the environment, or
`WindowPlugin { ime: false, .. }`, leaves the window as it was before: no IME on a desktop, no
`ImeEvent`, and a commit (there is none then) would be typed text. The engine has no setting
for a player: set the variable for the process (a launcher, a shortcut), or have the game read
a setting or a command-line flag into `WindowPlugin::ime` where it builds the plugin. Phones and the web never
change: the on-screen keyboard path (`TextInput::wanted` -> `set_ime_allowed`, the key events
with text, the page's hidden input) is the code it was.

**Not done.**

* Android has no input method for the app (see above): the way out is GameActivity or a Java
  activity with an `InputConnection`.
* The desktop web with a mouse has no IME path. A game's hidden `<input>` is shown after a touch
  only.
* Linux has no bundled CJK font; the game loads one.
* What an input method does with the letters it was composing when the engine ends the
  composition (a click into the field) has not been seen on a real one: they may be dropped, or
  committed once more, which would put them in the field a second time.
* egui's `ImeEvent::DeleteSurrounding` (Wayland's surrounding-text deletion) is not produced:
  winit 0.30 has no such event.
* A `space3d-ui` that is disabled (`EguiContext::set_enabled(false)`) reads no `Input`, and
  `TextInput::wanted` keeps what the last pass said: a focused field then leaves the IME on.

**Manual test (a real IME; headless tests cannot show it).** Windows 10 or 11, display language
and keyboard Chinese (Simplified) with Microsoft Pinyin:

1. Start the game, log in, fly.
2. Press Enter to open the chat. Switch to the Pinyin IME with Win+Space. Shift toggles Chinese
   and English mode: the toggle must work now (the indicator in the taskbar changes).
3. Type `nihao`: the pinyin is shown underlined inside the field and the candidate list opens
   right under the field (not at the screen's corner). Space or a number picks a candidate,
   Enter sends, the line shows the glyphs.
4. Type `nihao` again and press Backspace: it removes pinyin letters, not earlier chat text.
   Press Esc: the composition is gone and the chat is still open.
5. Leave the chat and fly with the IME still in Chinese mode: WASD, Q, E and Space work as
   before (the IME is off outside a text field). The sign-in password field takes plain Latin
   letters (no candidate list).
6. Repeat 3 in windowed mode and in borderless fullscreen (the candidate window must be
   visible), and on a display scaled to 150 %.
7. With `SPACE3D_NO_IME=1` the chat takes no Chinese, as before.
8. Type `nihao` and click into the chat line (not on the end of the text): the letters stay as
   text, the candidate list is gone, and the next `nihao` starts a new composition at the click
   (no second run of pinyin beside the first, and nothing committed twice).

On a Mac (Pinyin, Kotoeri) and on Linux (ibus or fcitx5, X11 and Wayland) do step 3; on X11 dead
keys do not work while a field has the focus, and note where the candidate list opens: winit
passes X11 the caret's top-left only (the XIM spot, conventionally the baseline), so a list that
covers the chat line wants the caret's height added to `y` on X11 (one line in
`apply_window_state`, `set_ime_cursor_area`). None of this has been seen with a real IME yet.

The same test without a game: `cargo run -p space3d-ui --example ime_field` opens a window with a
chat line, a password line, a "W held" line and the list of what the input method sent. With
`RUST_LOG=ime=debug` the runner logs each `set_ime_allowed` it makes (on at the chat, off at the
password and outside the fields, once each). `-- --frames 150` runs it scripted in the background
(no focus stolen) and prints what `TextInput` said with the chat, the password field and nothing
focused: that shows the runner switching the real window's IME, but composes nothing.

## Mouse buttons, focus and the UI

`Input` holds the mouse buttons (`mouse_pressed`, per-frame edges `mouse_just_pressed` /
`mouse_just_released`) and also records this frame's presses and releases **in order**:
`Input::mouse_button_events()` lists a `MouseButtonEvent` per change with the button, `pressed`,
the cursor `position` at the time (`None`: outside the window) and a `cause`. The edges alone
lose the order — a release and a new press in one frame (the second click of a fast double
click) show both edges with the button held; the events say it went up first. Every injection
records events (`inject_mouse_button_down` / `inject_mouse_button_up` as `Device` events), so
tests and replays drive the same path as the window runner.

**Releases without a release event.** A button normally comes up with its release event. Two
more things release it, as **cancels** (`MouseButtonEvent::cancelled()`,
`MouseButtonCause::is_cancel()`):

* `MouseButtonCause::FocusLost` — the window lost the focus (or the app was suspended): the
  runner calls `Input::release_all()`, which releases every held key and button so game state
  never sticks.
* `MouseButtonCause::OsReconciled` — the OS reports the button up although its release event
  never arrived. On macOS the release goes elsewhere when the system takes the pointer between
  press and release (Mission Control, App Exposé, a Spaces swipe, a hot corner, the screenshot
  overlay) while the window stays key; nothing else would ever release the button. Before each
  frame the runner asks the OS which buttons it holds — one call, only while a button is held —
  and releases a button the OS showed held during this press and has shown up since, on
  consecutive checks spanning at least 50 ms (two at least, with a frame's events delivered in
  between: the OS state can be ahead of the event queue, and a real release event still on its
  way stays a real release). Only a press the window itself received (an event of the OS window)
  is checked: a press a test, a replay or a script injected (`Input::inject_mouse_button_down`)
  was never held by the OS, so the developer's own physical button — held in another window
  and let go later — cannot release it. There is no check either while the window has no focus
  or in background mode (windowed runs of tests, CI and agents: `space3d_platform::background`):
  the OS's buttons then belong to another app, and what was seen of a press is forgotten. A
  press the OS never showed held (a click shorter than a frame, a synthetic press) is left to
  its own release. It never presses anything.

| platform | OS button state | a lost release |
|---|---|---|
| macOS | `+[NSEvent pressedMouseButtons]` (window server state) | released as `OsReconciled` |
| Windows | `GetAsyncKeyState` (physical buttons, swapped back for left-handed mice: `SM_SWAPBUTTON`) | released as `OsReconciled` |
| Linux, web, Android, iOS | none | waits for the next click or a focus loss |

Leaving the window releases nothing: macOS, Windows and X11 deliver the release of a drag that
ends outside, and a drag may cross the window's edge (a slider, a window, hold-to-steer).

**What the UI does with them.** `space3d-ui` sends egui the button events in order, each where
the cursor was (egui's pointer moves there first, so a release after a move beyond egui's click
distance is no click), then the cursor's current position. A button event while the cursor is
outside the window goes to the last position egui saw, and the pointer is gone again right after
(`PointerGone`), so nothing at the window's corner gets hovered. `Event::WindowFocused` is sent
when `WindowState::focused` changes (egui then forgets the keys and modifiers it held).

* **A release outside the window is not a click on a widget at the window's edge**, even in
  the frame the pointer leaves: it is delivered at the last position egui saw, after egui's
  pointer detoured off-screen (as for a cancel), so the widget under that position sees a
  release but no click. A drag-and-drop payload is dropped nowhere. (The one exception is the
  case described under "A cancel only concerns its own button": when egui has no press in
  progress for the button, it has nothing to withhold a click from.)
* **No pointer position yet.** A button event before any pointer position is known (nothing has
  moved the cursor into the window; a test that presses first) is *dropped*, not sent to
  (0, 0): the press is ignored and so is its release, whenever it comes (a release after the
  cursor moved in is not delivered to a press egui never saw). Buttons work as soon as one
  position was seen; `Input` itself still records them (`mouse_pressed`, `mouse_button_events`).
* **A cancel only concerns its own button.** egui follows one press at a time and any release
  ends the click in progress, so a cancel between another button's press and release would
  swallow that click. The UI therefore sends a cancel as early as it can: ahead of the presses
  and clicks of the other buttons in its frame (after its own button's earlier events, after
  other cancels, and after the release that ends a click begun in an earlier frame).
  A right click that starts and ends in the frame the held left button is cancelled is a
  right click. (When a click begun earlier ends in the same frame as another button's cancel,
  egui has no press in progress when the cancel comes and judges it alone: the click of the
  other button is kept, and the cancelled release may count as a click there.)
* **Touches.** `TouchPhase::Cancelled` (an incoming call, a rejected palm, a system gesture
  that took the finger) ends the touch like `Ended` but is no tap: `Input::just_ended_touches`
  lists it, `Input::just_cancelled_touches` tells which of those were cancelled, egui gets
  `TouchPhase::Cancel`, and the pointer touch's release is a cancelled release of the primary
  button (no click). Games should not fire on the release of a cancelled touch.

A cancelled release registers **neither a click nor a drop** in egui — like switching away from
a native app mid-drag. Drags end (`Response::drag_stopped`), a dragged slider keeps its value,
a button pressed before the focus loss does not fire, and egui's drag-and-drop payload is
dropped without reaching any drop zone (`dnd_release_payload` returns `None`). Game code that
implements its own drag-and-drop or "release over X" logic from `Input` should skip releases
with `cancelled()`:

```rust
for e in ctx.resources.expect::<Input>().mouse_button_events() {
    if e.button == MouseButton::Left && !e.pressed {
        if e.cancelled() {
            cancel_drag();           // focus lost / release only the OS saw: no drop
        } else {
            drop_here(e.position);   // a real release
        }
    }
}
```

**Deciding drops and commits from egui.** egui has no notion of a cancel: a cancelled release
still ends the press, so `Response::drag_stopped()`, `PointerState::button_released` and
`any_released()` are `true` in that frame, and "drop here" logic written as `hovered()` (or
`contains_pointer()`) with `any_released()` would act on a focus loss. Ask whether the release
was a cancel:
`EguiContext::pointer_cancelled_this_frame()` (any button) and
`EguiContext::button_cancelled_this_frame(button)`, or, when only an `egui::Context` is at
hand, the free functions `space3d_ui::pointer_cancelled_this_frame(ctx)` and
`space3d_ui::pointer_cancel_cause(ctx, button)` (the `MouseButtonCause`: `FocusLost` or
`OsReconciled`; a cancelled touch is the primary button's, `FocusLost`). They describe the
frame's pass, from `Stage::PreUpdate` (`ui::begin_frame`) until the next frame's, and are
`false` without a running `UiPlugin` or while the UI is disabled.

```rust
app.add_root_ui_system(|ui: &mut egui::Ui, sys: &mut SystemCtx| {
    let cancelled = sys
        .resources
        .expect::<EguiContext>()
        .button_cancelled_this_frame(egui::PointerButton::Primary);
    let (_, zone) = ui.allocate_exact_size(egui::vec2(60.0, 30.0), egui::Sense::hover());
    let released = ui.input(|i| i.pointer.any_released());
    if zone.contains_pointer() && released {
        if cancelled {
            put_it_back();           // focus lost / a release only the OS saw: no drop
        } else {
            drop_here();             // a real release
        }
    }
});
```

egui's own drag-and-drop (`Response::dnd_release_payload`) needs none of this: the engine drops
the payload before the pass on a cancel. Code that runs outside the UI systems (a game
system reading `Input`) uses `MouseButtonEvent::cancelled()` as above.

**Sub-apps in the editor.** A `LiveView` (the editor's sub-app viewport, the storybook's story
view) forwards its host UI's pointer to the sub-app's `Input`: a press as a device press, a
release as a device release — or, when the host cancelled it (`pointer_cancel_cause`), as
`Input::release_mouse_button(button, cause)` with the host's cause, so the story or sub-app's
own `MouseButtonEvent::cancelled()` and UI behave as in a window of their own.

**egui plugins.** egui 0.36 runs its plugins' pass hooks (`Plugin::on_begin_pass` /
`on_end_pass`) only from `Context::run_ui`; the engine's pass spans `Stage::PreUpdate` to
`Stage::PostUpdate` (`begin_pass` / `end_pass`), so the engine runs the hooks itself, in
`run_ui`'s order and with a top-level `Ui` like `run_ui`'s: the begin hooks before the UI systems,
the end hooks after them. That covers egui's built-in plugins — label text selection (a drag to
select ends with the release), drag-and-drop (a payload nobody took ends with the release or
Esc) and `egui::debug_text` — and plugins added with `EguiContext::add_plugin(plugin)` (or made
known with `EguiContext::register_pass_hooks::<P>()` when a library already added them with
`egui::Context::add_plugin`). egui cannot list a context's plugins, so hooks of plugins the engine
does not know about do not run, and `egui::Context::on_begin_pass` / `on_end_pass` callbacks
(kept in a private egui plugin) never run: register a plugin instead.

**Tests** drive all of this headless: `WindowState::focused = false` plus `Input::release_all()`
is what the runner does on a focus loss, and `Input::release_mouse_button(b,
MouseButtonCause::OsReconciled)` is what the OS check does. Injected presses
(`inject_mouse_button_down`) are never subject to the OS check; only the runner's own window
events are (crate-internal `Input::os_mouse_button`).

## Content under the title bar

`WindowPlugin::content_under_title_bar` (default `false`) lets the app's picture fill the whole
window, the title bar's strip included. The window buttons sit on top of the picture.

```rust
app.add_plugin(WindowPlugin {
    title: "My game".into(),
    content_under_title_bar: true,
    ..WindowPlugin::default()
});
```

| platform | what changes | the window buttons |
|---|---|---|
| Windows | the client area begins at the window's top; the sizing frame at the left, right and bottom stays the system's | `space3d-ui` draws them at the top right; the system still knows where they are, so Windows 11 shows Snap Layouts over maximise |
| macOS | full-size content view under a transparent title bar without a title | the system's traffic lights, at the top left |
| Linux, web, Android, iOS | nothing | as without the option |

In fullscreen (winit's, or an app's own that takes `WS_CAPTION` off) there is no title bar, and
`TitleBar::active` is `false`.

**What the app has to do.** The `TitleBar` resource says what the strip is, in physical pixels of
the window's picture:

* `active`, `height`: whether the picture is under the title bar now, and how high the strip is;
* `controls`, `left_inset()`, `right_inset(width)`: what the window buttons cover. Put nothing
  there that the player must see or click.

The rest of the strip drags the window, and a double click maximises it. A click reaches the app
instead where the app claims it:

* `space3d-ui` claims every egui widget in the strip that senses a click or a drag, floating
  windows included. A backdrop that covers the whole screen is not claimed, so the strip still
  drags behind a modal sheet. There is nothing to call.
* An app with hit areas of its own calls `TitleBar::claim(rect)` every frame it wants them.
* `TitleBar::keep_strip()` keeps the whole strip for one frame: for a game that needs every
  click while the pointer steers. The window buttons still work.

The platform asks about the last frame's rectangles, so a new widget takes clicks from the frame
after it first appears. The window buttons beat every claim: a window must always close.

**Windows.** The window keeps the styles of a decorated window and gets a window procedure in
front of winit's (`space3d_platform::title_bar`):

* dragging, Aero Snap, double click, the system menu, Win+arrows and the minimise and maximise
  animations stay the system's;
* it resizes from every edge; at the top, from a band 8 logical pixels high (none while
  maximised, so the corner of the screen closes the window);
* a maximised window is not cut off, and it leaves 2 pixels free at an edge with an auto-hide
  taskbar;
* a window that cannot be resized has a dimmed maximise button;
* the asked window size is the picture's size, at creation and after a move to a monitor with
  another scale (winit alone would add a title bar's height).

`space3d-ui` paints the buttons above everything it draws, and its docked tools stay below the
strip. Set `TitleBar::draw_buttons` to `false` to draw your own from `TitleBar::buttons` (each
says where it is and whether it is enabled, hovered or pressed). An app without `space3d-ui` must
draw them: the system draws none.

**macOS.** A press in the strip that nothing claims drags the window, and a second one within half
a second maximises or restores it. (It does not read the "double-click a window's title bar to"
setting of System Settings.)

**Tests and pictures** need no window: insert `TitleBar::simulated_windows(width, scale,
maximized)` or `TitleBar::simulated_macos(scale)`, step the app, and read `claims()`
(`crates/space3d-ui/tests/title_bar.rs`). The Windows window procedure has tests on real windows
that are never shown (`crates/space3d-platform/src/title_bar/windows/tests.rs`).

## The mouse cursor and the window icon

The pointer shape over the window is the `WindowCursor` resource (inserted by `WindowPlugin`).
`space3d-ui` writes egui's cursor into it whenever egui asks for a different one (the I-beam
over text fields, resize arrows, a pointing hand where the game asks for it with
`on_hover_cursor` or egui's `interact_cursor` style); a game may set `WindowCursor::icon`
itself (a crosshair in flight, say), and that stays until egui's cursor changes. After every
frame the runner applies it with winit's `Window::set_cursor`, so it survives Windows
re-applying the window cursor on `WM_SETCURSOR` and macOS's cursor rects. `icon: None` (egui's
`CursorIcon::None`) hides the pointer, as does `WindowState::cursor_visible = false`; grabbing
(`cursor_grab`) is independent. `WindowCursor::applied()` reports what was last applied to a
real window (shape and visibility; `None` headless), for tests that run a window in background
mode.

`WindowPlugin::icon` sets the window icon (title bar, Alt-Tab) from RGBA8 pixels
(`WindowIcon { rgba, width, height }` or `WindowIcon::from_image`); on Windows it is also the
taskbar icon, which Windows 11 otherwise shows blank for an exe without an embedded icon. macOS
ignores it (the Dock shows the app bundle's icon); the web and mobile use their own manifests.

## Gamepads

Xbox, PlayStation and other USB or Bluetooth controllers (not HOTAS flight sticks). A game
reads them from `Input` (`space3d_platform::gamepad` has the whole story in its module docs):

```rust
let input = ctx.resources.expect::<Input>();
for change in input.gamepad_changes() { /* GamepadChange::Connected(id) / Disconnected(id), this frame */ }
if input.any_gamepad_just_pressed(GamepadButton::South) { /* A / Cross on any pad */ }
let stick = input.any_gamepad_stick(GamepadAxis::LeftStickX, GamepadAxis::LeftStickY); // the pad pushed furthest, whole; x right, y UP
let lt = input.any_gamepad_axis(GamepadAxis::LeftTrigger); // one axis: 0..1 (a stick's: -1..1)
let name = input.gamepad(id).map(|p| p.name());            // "Xbox 360 Controller"; may be empty
```

Buttons are named by their place (`South` = A / Cross, `East` = B / Circle, `West` = X / Square,
`North` = Y / Triangle, `LeftShoulder`, `LeftTrigger`, `LeftThumb`, `Start`, `Select`, `Mode`,
the d-pad). A trigger is an axis 0..1 and a button (down from 0.5, up again below 0.4). Sticks
have a radial dead zone of 0.15 taken out and the rest stretched to 0..1 (`gamepad::dead_zone`).
`Binding::GamepadButton` / `Binding::GamepadAxis` bind actions. Tests and scripts inject:
`Input::inject_gamepad_connected_named`, `inject_gamepad_button`, `inject_gamepad_axis`,
`inject_gamepad_disconnected` (and `space3d-test`'s `InputEvent::Gamepad*`).

| Platform | Backend | What works | Notes |
|---|---|---|---|
| Windows | `gilrs` 0.11 (Windows.Gaming.Input) | everything, hot plug: as built and unit-tested; **not yet seen with a pad, real or virtual** (below) | Windows gives pad input to the **focused** window only. |
| macOS | `gilrs` (IOKit) | compiles in the dependency graph; **not built or run on a Mac yet** | IOKit gives pad input to every process: the runner holds it back while the window has no focus (below). |
| Linux | `gilrs` (evdev) | **not built or run on Linux yet** | evdev gives pad input to every process, as IOKit. Needs `libudev` at run time and `libudev-dev` + `pkg-config` to build (the game's build images must have them). |
| Android | winit key events (`runner.rs`, `gamepad::AndroidPads`), `InputDevice` over JNI (`android::gamepads`) | **The left stick and the face buttons**: A B X Y, L1 R1, L3 R3, Start, Select, Mode, and the left stick once a button was pressed. L2 R2 only where the pad sends them as keys too (PlayStation pads are said to; Xbox pads send axes only). **Not**: the d-pad of an Xbox or PlayStation pad (a hat axis), the right stick, analog triggers. Unplug: seen within a second. | winit 0.30 has no gamepad events and marks every motion event handled, so Android's own stick-to-d-pad fallback never runs. A stick arrives as a "touch" move from a device no finger went down on, at (`AXIS_X`, `AXIS_Y`); from a device already known as a pad **by a button** it is the left stick, from any other it is dropped (a sensor with axes is no controller), and none reaches the fingers. A d-pad works only on a pad that sends arrow keys. The rest needs winit patched (third_party). winit has no device-removed event and hides the device number, so the runner asks `InputDevice.getDeviceIds()` once a second: when a listed gamepad is gone, **every** pad known is disconnected with what it held (one still there is a pad again from its next button), and a pad has the OS's name only while Android lists exactly one. Compiles for both Android targets; **not run on a device** (the JNI calls included). |
| iOS | none | nothing | `GCController` through objc2: not done. |
| Web | none | nothing | gilrs has a `navigator.getGamepads` backend; the dependency is desktop-only for now. |

**Focus.** Only the focused window obeys a pad, on every desktop OS: a focus loss releases every
pad button and rests the axes (`Input::release_all`), and while the window has no focus, or the
run is a background one, the runner takes only a pad's coming and going from gilrs
(`Pads::apply_unfocused`; a stick's raw place is still remembered, so its first move after the
focus is back is right). Without this a pad used in another game or a browser would fly the ship
of a window in the background on Linux and macOS.

**The backend's cost and its switch.** gilrs's Windows backend is a thread that asks the OS for
the controllers every 8 ms for the whole run, pad or no pad (Linux and macOS start a thread each
too). It is **not started in a background run** (`background_mode()`: tests, smoke runs, headless
QA clients; a scripted pad through `Input::inject_gamepad_*` still works there), and
`SPACE3D_GAMEPAD=off` (or `0`, `false`, `no`) leaves it unstarted in any run. Its cost in a
focused game was not measured.

`cargo run -p space3d-platform --example virtual_pad` (Windows) checks the real backend
without a pad: it plugs a virtual Xbox 360 pad in through the ViGEmBus driver, drives every
button, both sticks and the triggers, pulls it out mid-hold and plugs it in again, and compares
each step with `Input` (exit code 0 / 1; 2 without the driver or the focus). `--log-only`
prints what any real pad does.

On the build PC (2026-11-15) the example exited 2: the `ViGEmBus` driver service runs, but there is
no ViGEm bus device for the client to open (`BusNotFound`), and no pad was connected (`--log-only`:
"no pad"). So the gilrs path has compiled and its mapping is unit-tested, and a real or virtual pad
has not driven it yet.

## CI and releases

`space3d ci init --targets desktop,web,android,ios` (also run by `space3d new`) writes
`.github/workflows/space3d.yml`: tests with a software GPU, a desktop matrix (Linux, Windows,
macOS), the web bundle (deployed to GitHub Pages), an APK and an unsigned `.ipa`, and a GitHub
Release with all of them on `v*` tags. The file goes to the repository root (`git rev-parse
--show-toplevel`, else the cargo workspace root), the only place GitHub reads workflows; for a
package nested inside a bigger repository the command says which `-p` / `working-directory`
to add to the jobs. The `wasm-bindgen-cli` version it installs is taken from `Cargo.lock`.

Locally, `space3d publish --tag v1.2.0 [--targets desktop,web,android] [--draft] [--dry-run]`
builds release artifacts into `dist/`, archives them (`<game>-v1.2.0-<platform>.tar.gz`, `.apk`)
in `dist/release/`, creates the GitHub release (or reuses the existing one for that tag) and
uploads the archives. It needs `GITHUB_TOKEN`/`GH_TOKEN` or `gh auth login`; the repository is
taken from `git remote get-url origin` unless `--repo owner/name` is given.
