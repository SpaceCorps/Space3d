//! The winit 0.30 `ApplicationHandler` runner shared by desktop, web, Android and iOS.
//!
//! Lifecycle:
//! * `resumed` — create the window once; create the [`Renderer`] (blocking on native,
//!   asynchronously on the web) or, on mobile resume, recreate its surface.
//! * `suspended` — drop the surface (Android destroys the native window in background).
//! * `window_event` — feed [`Input`] (keyboard events with their logical keys; physical keys
//!   winit 0.30 misses or conflates are corrected: macOS ISO `IntlBackslash`, JIS `IntlRo`),
//!   track size/DPI/focus, render on `RedrawRequested`.
//!   Before rendering, a surface the renderer dropped as lost (Vulkan `SURFACE_LOST`, DX12)
//!   is recreated, at most once per `SurfaceRetry::INTERVAL`.
//! * `about_to_wait` — request the next redraw (continuous rendering).
//!
//! The keyboard layout's OS table ([`crate::layout`]) is read on the event-loop thread at the
//! first resume and on focus gain, and its id is re-checked at most once per second (cheap:
//! no per-frame OS calls).
//!
//! Held mouse buttons are checked against the OS before each frame (macOS, Windows; one OS
//! call per frame, none while no button is held): a button whose release event never arrived is
//! released as a cancel once the OS has shown it up for a moment (`crate::os_buttons`; only presses
//! the window itself received, and only while it has the focus outside background mode). A focus
//! loss or suspension releases every held key and button ([`Input::release_all`]; mouse buttons
//! as cancels).

use crate::ime::ImeState;
use crate::input::{Input, KeyEvent, MouseButton, TouchPhase};
use crate::layout::LayoutWatch;
use crate::os_buttons::ButtonReconciler;
use crate::window::{PlatformEvent, WindowPlugin, WindowState};
use glam::{UVec2, Vec2};
use space3d_core::{App, Events};
use space3d_render::{RenderError, Renderer, RendererConfig};
use std::sync::Arc;
use std::time::Duration;
use web_time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::PhysicalKey;
use winit::window::{CursorGrabMode, Window, WindowId};

/// Pixels per "line" when converting pixel-precise scrolling (trackpads) to lines.
const PIXELS_PER_LINE: f32 = 40.0;

/// A window of a desktop (Windows, macOS, Linux, the BSDs): not a phone, not the web. The input
/// method support (`crate::ime`) is for these; the phones' keyboard path below is the old one.
const DESKTOP: bool = cfg!(not(any(
    target_os = "android",
    target_os = "ios",
    target_arch = "wasm32"
)));

pub(crate) struct WinitRunner {
    app: App,
    settings: WindowPlugin,
    window: Option<Arc<Window>>,
    last_frame: Option<Instant>,
    renderer_requested: bool,
    raw_mouse_motion: bool,
    exiting: bool,
    applied_title: String,
    applied_cursor: (bool, bool),
    /// Shape last passed to `Window::set_cursor` (`None`: not set yet).
    applied_icon: Option<crate::window::CursorIcon>,
    surface_retry: SurfaceRetry,
    layout_watch: LayoutWatch,
    buttons: ButtonReconciler,
    /// The OS input method (IME) of a desktop window (`crate::ime`).
    ime: ImeState,
    /// The pads of the OS (`crate::gamepad`), polled before every frame.
    #[cfg(any(target_os = "windows", target_os = "macos", target_os = "linux"))]
    pads: crate::gamepad::Backend,
    /// Android: pads by the input device their key and motion events name.
    #[cfg(target_os = "android")]
    pads: crate::gamepad::AndroidPads<DeviceId>,
    /// Android: when the OS was last asked for its gamepads.
    #[cfg(target_os = "android")]
    pads_asked: Option<Instant>,
    /// Whether the on-screen keyboard was last asked for (`TextInput`, phones).
    #[cfg(any(target_os = "android", target_os = "ios"))]
    keyboard_shown: bool,
    /// Frames until the on-screen keyboard is asked for again (a finger lifted; see `Touch`).
    #[cfg(any(target_os = "android", target_os = "ios"))]
    keyboard_again: u8,
    /// Where each finger on the screen went down, any of which may bring the keyboard back
    /// (`Touch`): one slot per finger, so a second finger does not lose the first one's tap.
    #[cfg(any(target_os = "android", target_os = "ios"))]
    touch_down: Vec<(u64, Vec2)>,
    /// Android: frames since the keyboard's height was asked (`KeyboardInset`).
    #[cfg(target_os = "android")]
    keyboard_asked: u32,
    /// Android: the window's insets as Java gave them last, the window size they were asked
    /// for and the frames since.
    #[cfg(target_os = "android")]
    insets: (Option<crate::window::SafeArea>, (u32, u32), u32),
    #[cfg(target_arch = "wasm32")]
    pending: std::rc::Rc<std::cell::RefCell<Option<Result<Renderer, space3d_render::RenderError>>>>,
}

fn build_event_loop() -> Result<EventLoop<()>, winit::error::EventLoopError> {
    #[cfg(target_os = "android")]
    {
        use winit::platform::android::EventLoopBuilderExtAndroid;
        let Some(android_app) = crate::mobile::android_app() else {
            tracing::error!("Android: start the app through space3d_platform::android_main_entry");
            // (winit's NotSupportedError has no public constructor.)
            return Err(winit::error::EventLoopError::ExitFailure(1));
        };
        EventLoop::builder().with_android_app(android_app).build()
    }
    #[cfg(target_os = "macos")]
    {
        use winit::platform::macos::{ActivationPolicy, EventLoopBuilderExtMacOS};
        let mut builder = EventLoop::builder();
        if crate::background::background_mode() {
            // Never come to front / take focus: no Dock icon, no activation on launch.
            builder
                .with_activation_policy(ActivationPolicy::Accessory)
                .with_activate_ignoring_other_apps(false);
        }
        builder.build()
    }
    #[cfg(not(any(target_os = "android", target_os = "macos")))]
    {
        EventLoop::new()
    }
}

/// Throttles re-attaching a presentation surface after the renderer dropped it (a lost
/// surface can only be replaced, and on a window that really cannot present every attempt
/// fails). Also rate-limits a lose/recreate cycle, since a successful attempt counts too.
#[derive(Debug, Default)]
struct SurfaceRetry {
    last_attempt: Option<Instant>,
    failing: bool,
}

impl SurfaceRetry {
    /// Minimum time between two recreation attempts.
    const INTERVAL: Duration = Duration::from_secs(1);

    /// Whether to try attaching a new surface now. `renderer_has_surface` is `None` while no
    /// renderer exists. Never while suspended: Android drops the surface on purpose then.
    fn due(&self, now: Instant, suspended: bool, has_window: bool, renderer_has_surface: Option<bool>) -> bool {
        !suspended
            && has_window
            && renderer_has_surface == Some(false)
            && self
                .last_attempt
                .is_none_or(|last| now.duration_since(last) >= Self::INTERVAL)
    }

    /// Record a failed attempt. Returns `true` for the first failure of a streak (log it).
    fn failed(&mut self, now: Instant) -> bool {
        self.last_attempt = Some(now);
        !std::mem::replace(&mut self.failing, true)
    }

    /// Record a successful attempt. Returns `true` if it ends a failure streak (log it).
    fn succeeded(&mut self, now: Instant) -> bool {
        self.last_attempt = Some(now);
        std::mem::replace(&mut self.failing, false)
    }
}

/// Attach a new presentation surface for `window` to `r` and size it to the window.
fn attach_surface(r: &mut Renderer, window: &Arc<Window>) -> Result<(), RenderError> {
    #[cfg(not(target_arch = "wasm32"))]
    r.set_surface(window.clone())?;
    #[cfg(target_arch = "wasm32")]
    match crate::web::canvas_of(window) {
        Some(c) => r.set_surface(space3d_render::wgpu::SurfaceTarget::Canvas(c))?,
        None => return Err(RenderError::Surface("window has no canvas".into())),
    }
    let size = drawable_size(window);
    r.resize(size.width.max(1), size.height.max(1));
    Ok(())
}

/// The size of what the window draws to, in physical pixels. winit's `inner_size` everywhere
/// but iOS: there it is the safe area only (the screen without the notch and the home bar),
/// while the Metal layer and the touches cover the whole screen, which `outer_size` is.
fn drawable_size(window: &Window) -> winit::dpi::PhysicalSize<u32> {
    #[cfg(target_os = "ios")]
    {
        window.outer_size()
    }
    #[cfg(not(target_os = "ios"))]
    {
        window.inner_size()
    }
}

/// How the loop runs while frames are drawn. `Poll` (a frame as soon as the last is done; the
/// surface's vsync paces it) everywhere but iOS: there a redraw request becomes the view's
/// `drawRect` at the display's pace, and `Poll` only spins the main run loop between two of
/// them (seen in the Simulator: hundreds of thousands of empty turns a second).
const FRAME_FLOW: ControlFlow = if cfg!(target_os = "ios") {
    ControlFlow::Wait
} else {
    ControlFlow::Poll
};

/// Runner installed by [`WindowPlugin`]. Falls back to the headless runner when no event loop
/// can be created (e.g. CI without a display server).
pub(crate) fn run(app: App, settings: WindowPlugin) {
    let event_loop = match build_event_loop() {
        Ok(el) => el,
        Err(e) => {
            tracing::error!("cannot create event loop ({e}); running headless");
            space3d_core::app::headless_runner(app);
            return;
        }
    };
    event_loop.set_control_flow(FRAME_FLOW);
    // iOS: the scene delegate's class must exist before UIKit starts (`crate::ios`).
    #[cfg(target_os = "ios")]
    crate::ios::register();
    let runner = WinitRunner {
        applied_title: settings.title.clone(),
        app,
        settings,
        window: None,
        last_frame: None,
        renderer_requested: false,
        raw_mouse_motion: false,
        exiting: false,
        applied_cursor: (true, false),
        applied_icon: None,
        surface_retry: SurfaceRetry::default(),
        layout_watch: LayoutWatch::default(),
        buttons: ButtonReconciler::default(),
        ime: ImeState::new(DESKTOP),
        #[cfg(any(
            target_os = "windows",
            target_os = "macos",
            target_os = "linux",
            target_os = "android"
        ))]
        pads: Default::default(),
        #[cfg(target_os = "android")]
        pads_asked: None,
        #[cfg(any(target_os = "android", target_os = "ios"))]
        keyboard_shown: false,
        #[cfg(any(target_os = "android", target_os = "ios"))]
        keyboard_again: 0,
        #[cfg(any(target_os = "android", target_os = "ios"))]
        touch_down: Vec::new(),
        #[cfg(target_os = "android")]
        keyboard_asked: 0,
        #[cfg(target_os = "android")]
        insets: (None, (0, 0), 0),
        #[cfg(target_arch = "wasm32")]
        pending: Default::default(),
    };
    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut runner = runner;
        if let Err(e) = event_loop.run_app(&mut runner) {
            tracing::error!("event loop error: {e}");
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::EventLoopExtWebSys;
        // Returns immediately; the browser drives the loop.
        event_loop.spawn_app(runner);
    }
}

impl WinitRunner {
    fn send(&mut self, event: PlatformEvent) {
        self.app.resources.init::<Events<PlatformEvent>>().send(event);
    }

    fn input(&mut self) -> &mut Input {
        self.app.resources.init::<Input>()
    }

    fn state(&mut self) -> &mut WindowState {
        self.app.resources.expect_mut::<WindowState>()
    }

    /// Keep the keyboard layout's OS table current (`force`: start-up, focus gain).
    fn poll_layout(&mut self, now: Instant, force: bool) {
        let layout = self.app.resources.init::<Input>().keyboard_layout_mut();
        self.layout_watch.poll(now, force, layout);
    }

    fn renderer_config(&self, size: UVec2) -> RendererConfig {
        RendererConfig {
            width: size.x.max(1),
            height: size.y.max(1),
            msaa_samples: self.settings.msaa_samples,
            vsync: self.settings.vsync,
            ..Default::default()
        }
    }

    fn create_window(&mut self, event_loop: &ActiveEventLoop) -> Option<Arc<Window>> {
        let s = &self.settings;
        #[allow(unused_mut)]
        let mut attrs = Window::default_attributes()
            .with_title(s.title.clone())
            .with_resizable(s.resizable);
        // iOS: the window is the screen. A size asked for makes a `UIWindow` of that size in
        // points, which is not the screen's (seen: a 1280 x 720 window, a third of it off screen).
        if !cfg!(target_os = "ios") {
            attrs = attrs.with_inner_size(winit::dpi::LogicalSize::new(s.width as f64, s.height as f64));
        }
        if crate::background::background_mode() {
            attrs = attrs.with_active(false);
        }
        // iOS: the status bar hidden and the home indicator faded until touched
        // ([`crate::set_immersive`]).
        #[cfg(target_os = "ios")]
        if crate::mobile::immersive() {
            use winit::platform::ios::{ScreenEdge, WindowAttributesExtIOS};
            // (and the bottom edge's swipe home asks twice: the stick and the fire button sit on it)
            attrs = attrs
                .with_prefers_status_bar_hidden(true)
                .with_prefers_home_indicator_hidden(true)
                .with_preferred_screen_edges_deferring_system_gestures(ScreenEdge::BOTTOM);
        }
        if let Some(icon) = &s.icon {
            match winit::window::Icon::from_rgba(icon.rgba.clone(), icon.width, icon.height) {
                Ok(icon) => {
                    #[cfg(target_os = "windows")]
                    {
                        use winit::platform::windows::WindowAttributesExtWindows;
                        attrs = attrs.with_taskbar_icon(Some(icon.clone()));
                    }
                    attrs = attrs.with_window_icon(Some(icon));
                }
                Err(e) => tracing::warn!("window icon ignored: {e}"),
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            attrs = crate::web::window_attributes(attrs, s.canvas_id.as_deref());
        }
        match event_loop.create_window(attrs) {
            Ok(w) => Some(Arc::new(w)),
            Err(e) => {
                tracing::error!("failed to create window: {e}");
                None
            }
        }
    }

    fn on_resize(&mut self, size: UVec2) {
        let size = size.max(UVec2::ONE);
        self.state().physical_size = size;
        self.input().set_window_size(size.as_vec2());
        if let Some(r) = self.app.resources.get_mut::<Renderer>() {
            r.resize(size.x, size.y);
        }
        self.send(PlatformEvent::Resized(size));
    }

    /// Create the renderer on first resume, or reattach the surface on later resumes.
    fn ensure_renderer(&mut self, event_loop: &ActiveEventLoop, window: &Arc<Window>) {
        if let Some(r) = self.app.resources.get::<Renderer>() {
            if !r.has_surface() {
                // Resume attaches immediately, regardless of the retry interval.
                self.reattach_surface(window, Instant::now());
            }
            return;
        }
        if self.renderer_requested {
            return;
        }
        let size = drawable_size(window);
        let size = UVec2::new(size.width, size.height).max(UVec2::ONE);
        self.renderer_requested = true;
        let config = self.renderer_config(size);
        #[cfg(not(target_arch = "wasm32"))]
        {
            let display: space3d_render::DisplayHandle = Box::new(event_loop.owned_display_handle());
            match pollster::block_on(Renderer::new_with_surface_and_display(
                window.clone(),
                Some(display),
                config,
            )) {
                Ok(r) => {
                    self.app.insert_resource(r);
                }
                Err(e) => {
                    tracing::error!("failed to create renderer: {e}");
                    event_loop.exit();
                }
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = event_loop;
            let Some(canvas) = crate::web::canvas_of(window) else {
                tracing::error!("window has no canvas");
                return;
            };
            let pending = self.pending.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let target = space3d_render::wgpu::SurfaceTarget::Canvas(canvas);
                let result = Renderer::new_with_surface(target, config).await;
                *pending.borrow_mut() = Some(result);
            });
        }
    }

    /// Attach a new surface to the existing renderer. Only the first failure of a streak is
    /// logged as an error, so a window that cannot present does not spam every retry.
    fn reattach_surface(&mut self, window: &Arc<Window>, now: Instant) {
        let Some(r) = self.app.resources.get_mut::<Renderer>() else {
            return;
        };
        match attach_surface(r, window) {
            Ok(()) => {
                if self.surface_retry.succeeded(now) {
                    tracing::info!("presentation surface recreated");
                }
            }
            Err(e) => {
                if self.surface_retry.failed(now) {
                    tracing::error!(
                        "failed to recreate surface: {e}; retrying every {:?} (further failures logged at debug)",
                        SurfaceRetry::INTERVAL
                    );
                } else {
                    tracing::debug!("failed to recreate surface: {e}");
                }
            }
        }
    }

    /// Recreate a surface the renderer dropped as lost (desktop never resumes, so
    /// `ensure_renderer` would not). Throttled by [`SurfaceRetry`].
    fn recover_surface(&mut self, now: Instant) {
        let suspended = self.state().suspended;
        let has_surface = self.app.resources.get::<Renderer>().map(Renderer::has_surface);
        if !self
            .surface_retry
            .due(now, suspended, self.window.is_some(), has_surface)
        {
            return;
        }
        if let Some(window) = self.window.clone() {
            self.reattach_surface(&window, now);
        }
    }

    /// Web: install the renderer once its async creation finished.
    #[cfg(target_arch = "wasm32")]
    fn poll_pending_renderer(&mut self) {
        let Some(result) = self.pending.borrow_mut().take() else {
            return;
        };
        match result {
            Ok(mut r) => {
                if let Some(w) = &self.window {
                    let s = w.inner_size();
                    r.resize(s.width.max(1), s.height.max(1));
                }
                self.app.insert_resource(r);
            }
            Err(e) => tracing::error!("failed to create renderer: {e}"),
        }
    }

    fn frame(&mut self, event_loop: &ActiveEventLoop) {
        #[cfg(target_arch = "wasm32")]
        self.poll_pending_renderer();
        if self.exiting || self.state().suspended {
            return;
        }
        let now = Instant::now();
        let dt = self.last_frame.map_or(Duration::ZERO, |last| now - last);
        self.last_frame = Some(now);
        self.recover_surface(now);
        self.poll_layout(now, false);
        // Held buttons the OS has let go of without a release event (see `os_buttons`); only
        // while this window has the focus and the run is not a background (unattended) one.
        let active = ButtonReconciler::active(self.state().focused, crate::background::background_mode());
        let input = self.app.resources.init::<Input>();
        self.buttons.frame(active, input, now, crate::os_buttons::query);
        // What the pads did since the last frame (desktop; Android's come as window events).
        #[cfg(any(target_os = "windows", target_os = "macos", target_os = "linux"))]
        self.pads.poll(input, active);
        // Android: winit says nothing of a pad that went away; Android is asked once a second.
        #[cfg(target_os = "android")]
        if self
            .pads_asked
            .is_none_or(|at| now.duration_since(at) >= Duration::from_secs(1))
        {
            self.pads_asked = Some(now);
            if let Some(listed) = crate::mobile::android_app().and_then(|a| crate::android::gamepads(&a))
                && self.pads.os_pads(listed, input)
            {
                tracing::info!(target: "gamepad", "a gamepad went away: every pad is disconnected");
            }
        }
        if let Some(w) = &self.window {
            w.pre_present_notify();
        }
        self.app.update(dt);
        self.apply_window_state();
        if self.app.should_exit() {
            self.exiting = true;
            // Android: the activity is finished first, and the loop ends when it is destroyed
            // (`about_to_wait`); a process that ended under a live activity would be started again.
            #[cfg(target_os = "android")]
            if crate::mobile::android_app().is_some_and(|a| crate::android::finish_activity(&a)) {
                return;
            }
            event_loop.exit();
        }
    }

    /// A key event of a gamepad (Android): `true` when it went to the pad and is no key.
    #[cfg(target_os = "android")]
    fn android_pad_key(&mut self, device: DeviceId, key: &PhysicalKey, pressed: bool) -> bool {
        use winit::keyboard::{KeyCode, NativeKeyCode};
        let input = self.app.resources.init::<Input>();
        match *key {
            PhysicalKey::Unidentified(NativeKeyCode::Android(code)) => self.pads.key(device, code, pressed, input),
            PhysicalKey::Code(code) => {
                let button = match code {
                    KeyCode::ArrowUp => crate::input::GamepadButton::DPadUp,
                    KeyCode::ArrowDown => crate::input::GamepadButton::DPadDown,
                    KeyCode::ArrowLeft => crate::input::GamepadButton::DPadLeft,
                    KeyCode::ArrowRight => crate::input::GamepadButton::DPadRight,
                    _ => return false,
                };
                self.pads.dpad_key(device, button, pressed, input)
            }
            _ => false,
        }
    }

    /// The app goes to the background or loses the focus: no key, button or finger stays held.
    fn release_input(&mut self) {
        self.input().release_all();
        self.app
            .resources
            .scope::<crate::touch_pointer::TouchMouse, _>(|res, mouse| mouse.reset(res.init::<Input>()));
    }

    /// A finger: through [`crate::touch_pointer::TouchMouse`] to the mouse, the gesture and
    /// the touches of [`Input`] (the path a scripted touch takes too).
    fn touch(&mut self, id: u64, phase: TouchPhase, p: Vec2) {
        let scale = self.state().scale_factor.max(1.0) as f32;
        self.app
            .resources
            .scope::<crate::touch_pointer::TouchMouse, _>(|res, mouse| {
                mouse.scale = scale;
                mouse.touch(res.init::<Input>(), id, phase, p);
            });
    }

    /// Push game-side changes of [`WindowState`] to the OS window.
    fn apply_window_state(&mut self) {
        let Some(window) = self.window.clone() else {
            return;
        };
        // Phones: the on-screen keyboard is up while a text field wants text (`TextInput`).
        #[cfg(any(target_os = "android", target_os = "ios"))]
        {
            let wanted = self
                .app
                .resources
                .get::<crate::window::TextInput>()
                .is_some_and(|t| t.wanted);
            if wanted != self.keyboard_shown {
                self.keyboard_shown = wanted;
                window.set_ime_allowed(wanted);
            }
            // A finger lifted a few frames ago and a text field (still) has the focus: the
            // player may have put the keyboard away himself, which nothing tells the app.
            // Asking for a keyboard that is up changes nothing.
            if self.keyboard_again > 0 {
                self.keyboard_again -= 1;
                if self.keyboard_again == 0 && wanted {
                    window.set_ime_allowed(true);
                }
            }
        }
        // Desktops: the OS input method (IME) is on while a text field has the focus, and not for
        // a password field (`crate::ime`).
        if DESKTOP {
            let text = self
                .app
                .resources
                .get::<crate::window::TextInput>()
                .copied()
                .unwrap_or_default();
            let step = self.ime.step(&text);
            if let Some(on) = step.allow {
                tracing::debug!(target: "ime", on, "set_ime_allowed");
                window.set_ime_allowed(on);
            }
            if step.cancel {
                // Turned off in the middle of a composition: the preedit is no more.
                self.app
                    .resources
                    .init::<Input>()
                    .inject_ime(crate::ime::ImeEvent::Disabled);
            }
            // The candidate list opens next to the caret (not at the window's corner).
            if let Some(a) = step.area {
                window.set_ime_cursor_area(
                    winit::dpi::PhysicalPosition::new(a.x, a.y),
                    winit::dpi::PhysicalSize::new(a.width, a.height),
                );
            }
        }
        // Android: how much of the window the keyboard covers, asked every few frames while a
        // text field wants text (the keyboard slides in over some), and 0 otherwise.
        #[cfg(target_os = "android")]
        {
            let wanted = self.keyboard_shown;
            self.keyboard_asked += 1;
            let known = self.app.resources.get::<crate::window::KeyboardInset>().copied();
            let now = if !wanted {
                Some(crate::window::KeyboardInset::default())
            } else if self.keyboard_asked >= 6 {
                self.keyboard_asked = 0;
                crate::mobile::android_app()
                    .and_then(|android| crate::android::keyboard_inset(&android))
                    .map(|bottom| crate::window::KeyboardInset { bottom })
            } else {
                None
            };
            if let Some(now) = now
                && known != Some(now)
            {
                tracing::info!("Android: the on-screen keyboard covers {} px", now.bottom);
                // The keyboard gone: it may have brought the navigation bar back, and nothing
                // else (no focus change, no resume) hides it again.
                if now.bottom == 0
                    && known.is_some_and(|k| k.bottom > 0)
                    && crate::mobile::immersive()
                    && let Some(android) = crate::mobile::android_app()
                {
                    crate::android::hide_system_bars(&android);
                }
                self.app.insert_resource(now);
            }
        }
        // iOS: the safe area, asked every frame (it changes with the rotation, after the resize).
        #[cfg(target_os = "ios")]
        {
            let size = window.outer_size();
            let area = crate::ios::safe_area(&window).turned_for_landscape((size.width, size.height));
            if self.app.resources.get::<crate::window::SafeArea>() != Some(&area) {
                tracing::info!("iOS: the safe area is {area:?} of {size:?}");
                self.app.insert_resource(area);
            }
        }
        // Android: what the system bars and a cutout cover of the window. Java's insets
        // (`crate::android`), asked again when the window's size changes and every 30 frames
        // (nothing tells a NativeActivity app that they changed).
        #[cfg(target_os = "android")]
        if let Some(android) = crate::mobile::android_app() {
            let size = window.inner_size();
            let key = (size.width, size.height);
            self.insets.2 += 1;
            if self.insets.1 != key || self.insets.2 >= 30 {
                self.insets = (crate::android::window_insets(&android), key, 0);
            }
            let r = android.content_rect();
            let area = self.insets.0.unwrap_or_else(|| {
                // No answer from Java: the content rectangle, which for an edge-to-edge window
                // (every app from Android 15 on) is the whole window; then keep the UI off the
                // navigation bar's pill at least.
                let mut area = crate::window::SafeArea::from_content_rect(
                    (r.left, r.top, r.right, r.bottom),
                    (size.width, size.height),
                );
                if area == crate::window::SafeArea::default() {
                    area.bottom =
                        (crate::window::SafeArea::ANDROID_NAVIGATION_POINTS * window.scale_factor()).round() as u32;
                }
                area
            });
            if self.app.resources.get::<crate::window::SafeArea>() != Some(&area) {
                let from = if self.insets.0.is_some() {
                    "the window's insets"
                } else {
                    "a guess"
                };
                tracing::info!("Android: the safe area is {area:?} of {size:?} ({from}; content {r:?})");
                self.app.insert_resource(area);
            }
        }
        let requested = self
            .app
            .resources
            .get::<crate::window::WindowCursor>()
            .map_or(Some(crate::window::CursorIcon::Default), |c| c.icon);
        let (title, cursor) = {
            let s = self.state();
            // The pointer shows only while both the game and the UI want it.
            (
                s.title.clone(),
                (s.cursor_visible && requested.is_some(), s.cursor_grab),
            )
        };
        if title != self.applied_title {
            window.set_title(&title);
            self.applied_title = title;
        }
        if cursor != self.applied_cursor {
            window.set_cursor_visible(cursor.0);
            let mode = if cursor.1 {
                CursorGrabMode::Locked
            } else {
                CursorGrabMode::None
            };
            if window.set_cursor_grab(mode).is_err() && cursor.1 {
                let _ = window.set_cursor_grab(CursorGrabMode::Confined);
            }
            self.applied_cursor = cursor;
        }
        // Through winit (not a one-off OS call): Windows re-applies the window's cursor on every
        // WM_SETCURSOR and macOS resets it from the view's cursor rects.
        if let Some(icon) = requested
            && self.applied_icon != Some(icon)
        {
            window.set_cursor(icon);
            self.applied_icon = Some(icon);
        }
        if let Some(icon) = self.applied_icon
            && let Some(c) = self.app.resources.get_mut::<crate::window::WindowCursor>()
        {
            let applied = crate::window::AppliedCursor {
                icon,
                visible: cursor.0,
            };
            if c.applied() != Some(applied) {
                c.set_applied(applied);
            }
        }
    }
}

impl ApplicationHandler for WinitRunner {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let Some(w) = self.create_window(event_loop) else {
                event_loop.exit();
                return;
            };
            // iOS: the window joins the app's scene (`crate::ios`).
            #[cfg(target_os = "ios")]
            crate::ios::attach(&w);
            self.window = Some(w);
        }
        let window = self.window.clone().expect("window");
        let scale = window.scale_factor();
        {
            let s = self.state();
            s.suspended = false;
            s.scale_factor = scale;
        }
        self.ensure_renderer(event_loop, &window);
        self.poll_layout(Instant::now(), true);
        let size = drawable_size(&window);
        self.on_resize(UVec2::new(size.width, size.height));
        self.send(PlatformEvent::Resumed);
        // Android: the system bars hidden again (the system shows them while the app is away).
        #[cfg(target_os = "android")]
        if crate::mobile::immersive()
            && let Some(android) = crate::mobile::android_app()
        {
            crate::android::hide_system_bars(&android);
        }
        space3d_core::BackgroundHooks::run(&mut self.app.resources, false);
        // Avoid a huge dt after returning from background.
        self.last_frame = None;
        window.request_redraw();
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        // Android destroys the native window while in background: drop the surface now and
        // recreate it in `resumed`. Harmless elsewhere.
        if let Some(r) = self.app.resources.get_mut::<Renderer>() {
            r.remove_surface();
        }
        self.state().suspended = true;
        self.release_input();
        // The system hides the on-screen keyboard with the app: ask again after the resume.
        #[cfg(any(target_os = "android", target_os = "ios"))]
        {
            self.keyboard_shown = false;
        }
        self.send(PlatformEvent::Suspended);
        // No frame runs until the resume, so what must stop now (sound) hangs here.
        space3d_core::BackgroundHooks::run(&mut self.app.resources, true);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                self.send(PlatformEvent::CloseRequested);
                if self.settings.exit_on_close {
                    self.exiting = true;
                    event_loop.exit();
                }
            }
            WindowEvent::Resized(size) => {
                // iOS: the whole screen (`drawable_size`), not the event's safe area. Elsewhere
                // the event's own size, which is the one the system announced.
                let size = if cfg!(target_os = "ios") {
                    self.window.as_deref().map_or(size, drawable_size)
                } else {
                    size
                };
                self.on_resize(UVec2::new(size.width, size.height));
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.state().scale_factor = scale_factor;
                self.send(PlatformEvent::ScaleFactorChanged(scale_factor));
            }
            WindowEvent::Focused(focused) => {
                self.state().focused = focused;
                if focused {
                    // The layout may have been switched while another app had focus.
                    self.poll_layout(Instant::now(), true);
                    // Android: a dialog or the notification shade that took the focus may have
                    // shown the bars; hide them again.
                    #[cfg(target_os = "android")]
                    if crate::mobile::immersive()
                        && let Some(android) = crate::mobile::android_app()
                    {
                        crate::android::hide_system_bars(&android);
                    }
                } else {
                    self.release_input();
                }
                self.send(PlatformEvent::Focused(focused));
            }
            WindowEvent::KeyboardInput {
                event,
                #[cfg(target_os = "android")]
                device_id,
                ..
            } => {
                let pressed = event.state == ElementState::Pressed;
                // Android: a gamepad's buttons are keys of no place, and a known pad's arrow
                // keys are its d-pad (`crate::gamepad`).
                #[cfg(target_os = "android")]
                if self.android_pad_key(device_id, &event.physical_key, pressed) {
                    return;
                }
                // winit 0.30 leaves some keys unidentified or reports two keys as one (macOS
                // ISO and JIS keys): see `layout::physical_key`.
                match crate::layout::physical_key(event.physical_key) {
                    PhysicalKey::Code(code) => {
                        #[cfg(target_os = "macos")]
                        let code = if pressed {
                            code
                        } else {
                            let input = self.input();
                            crate::layout::macos_released_key(code, |k| input.pressed(k))
                        };
                        // An input method processed the key (Windows: the logical key `Process`):
                        // it is the IME's, not the game's (`crate::ime`).
                        let held = self.input().pressed(code);
                        if self.ime.swallows_key(&event.logical_key, pressed, held) {
                            return;
                        }
                        let key_event = KeyEvent {
                            physical: code,
                            key_without_modifiers: key_without_modifiers(&event),
                            logical_key: event.logical_key,
                            text: event.text.map(|t| t.to_string()),
                            location: event.location,
                            pressed,
                            repeat: event.repeat,
                        };
                        // Presses/releases the physical key, appends the text, learns the label.
                        self.input().inject_key_event(key_event);
                    }
                    PhysicalKey::Unidentified(_) => {
                        // Android: Back is a key of no place (winit 0.30: `Unidentified`, logical
                        // `BrowserBack`).
                        if let Some(keys) = back_as_escape(&event.logical_key, pressed) {
                            for key_event in keys {
                                self.input().inject_key_event(key_event);
                            }
                        } else if pressed && let Some(text) = &event.text {
                            // iOS: the on-screen keyboard's Return is the text "\n" of no key.
                            if let Some(enter) = return_as_enter(text) {
                                for key_event in enter {
                                    self.input().inject_key_event(key_event);
                                }
                            } else {
                                self.input().inject_text(text);
                            }
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let b = MouseButton::from(button);
                match state {
                    ElementState::Pressed => self.input().os_mouse_button(b, true),
                    ElementState::Released => self.input().os_mouse_button(b, false),
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let p = Vec2::new(position.x as f32, position.y as f32);
                if self.raw_mouse_motion {
                    self.input().set_mouse_position(Some(p));
                } else {
                    self.input().inject_mouse_move(p);
                }
            }
            WindowEvent::CursorLeft { .. } => self.input().inject_mouse_left(),
            WindowEvent::MouseWheel { delta, .. } => {
                let d = match delta {
                    MouseScrollDelta::LineDelta(x, y) => Vec2::new(x, y),
                    MouseScrollDelta::PixelDelta(p) => Vec2::new(p.x as f32, p.y as f32) / PIXELS_PER_LINE,
                };
                self.input().inject_wheel(d);
            }
            WindowEvent::Touch(t) => {
                let p = Vec2::new(t.location.x as f32, t.location.y as f32);
                let phase = TouchPhase::from(t.phase);
                // Android: a pad's stick comes as a "touch" that never went down; it is the
                // stick, not a finger (`crate::gamepad::AndroidPads::touch`).
                #[cfg(target_os = "android")]
                {
                    let input = self.app.resources.init::<Input>();
                    if self.pads.touch(t.device_id, phase, p, input) {
                        return;
                    }
                }
                self.touch(t.id, phase, p);
                #[cfg(any(target_os = "android", target_os = "ios"))]
                match phase {
                    TouchPhase::Started => {
                        self.touch_down.retain(|(id, _)| *id != t.id);
                        self.touch_down.push((t.id, p));
                    }
                    TouchPhase::Cancelled => self.touch_down.retain(|(id, _)| *id != t.id),
                    TouchPhase::Ended => {
                        // Once the tap's click is through the UI (`touch_pointer`: three
                        // frames). Only a tap asks: a finger that scrolled a list or dragged a
                        // window beside a focused field does not bring back the keyboard the
                        // player put away.
                        let slop = crate::touch_pointer::TouchPointerConfig::SLOP_POINTS
                            * self.state().scale_factor.max(1.0) as f32;
                        let down = self.touch_down.iter().position(|(id, _)| *id == t.id);
                        if down
                            .map(|i| self.touch_down.swap_remove(i).1)
                            .is_some_and(|down| crate::window::lift_is_a_tap(down, p, slop))
                        {
                            self.keyboard_again = 5;
                        }
                    }
                    _ => {}
                }
            }
            // What an input method does. winit 0.30 sends it on a desktop only, when the game
            // allows IME (`apply_window_state`): iOS's on-screen keyboard arrives as key events
            // with text, and Android's only as far as the keyboard sends key events (ASCII;
            // docs/platforms.md). A desktop's goes to `Input::ime_events` (and a commit is not
            // also typed text: egui would enter it twice); anywhere else a commit is typed text.
            WindowEvent::Ime(ime) => {
                let input = self.app.resources.init::<Input>();
                self.ime.deliver(ime, input);
            }
            WindowEvent::RedrawRequested => self.frame(event_loop),
            _ => {}
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event {
            // Raw motion is unaccelerated and keeps working with a locked cursor; once seen,
            // cursor moves only update the position.
            self.raw_mouse_motion = true;
            self.input()
                .inject_mouse_motion(Vec2::new(delta.0 as f32, delta.1 as f32));
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        #[cfg(target_arch = "wasm32")]
        self.poll_pending_renderer();
        // A phone app in the background has no surface to draw to: sleep until the system
        // wakes it (measured on the emulator: a whole core spent polling otherwise). Frames,
        // and with them the game's systems, stand still until it is resumed.
        #[cfg(any(target_os = "android", target_os = "ios"))]
        {
            let suspended = self.state().suspended;
            // Android: the activity was destroyed (`crate::android`). Java's `onDestroy` waits
            // for this thread, so the loop ends here and `android_main_entry` ends the process.
            #[cfg(target_os = "android")]
            if (suspended || self.exiting)
                && !_event_loop.exiting()
                && crate::mobile::android_app().is_some_and(|a| crate::android::activity_destroyed(&a))
            {
                tracing::info!("Android: the activity is destroyed, leaving the event loop");
                self.exiting = true;
                _event_loop.exit();
                return;
            }
            _event_loop.set_control_flow(if suspended { ControlFlow::Wait } else { FRAME_FLOW });
            if suspended {
                return;
            }
        }
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

/// The Enter key's press and release for a key event that is only the text of Return. iOS's
/// on-screen keyboard sends Return so (winit 0.30: `PhysicalKey::Unidentified`, text `"\n"`),
/// and a UI drops control characters from text, so the form would never be sent.
fn return_as_enter(text: &str) -> Option<[KeyEvent; 2]> {
    use crate::layout::{LogicalKey, NamedKey};
    matches!(text, "\n" | "\r" | "\r\n").then(|| {
        let enter = LogicalKey::Named(NamedKey::Enter);
        [
            KeyEvent::press(winit::keyboard::KeyCode::Enter, enter.clone()).with_text(None),
            KeyEvent::release(winit::keyboard::KeyCode::Enter, enter),
        ]
    })
}

/// Android's Back as two keys an app can read: `BrowserBack` (it was Back) and `Escape` (what
/// Back means in a UI: close what is open), pressed or released as Back is. winit 0.30 gives
/// Back as a key of no place (`PhysicalKey::Unidentified`) whose logical key is
/// `NamedKey::BrowserBack`, and reports it as handled, so the system does nothing more with
/// it: leaving the app on Back is the app's to do. (An app that targets Android 16 is not sent
/// the key at all unless its manifest opts out of the predictive back gesture.)
fn back_as_escape(logical: &winit::keyboard::Key, pressed: bool) -> Option<[KeyEvent; 2]> {
    use crate::layout::{LogicalKey, NamedKey};
    use winit::keyboard::KeyCode;
    (*logical == LogicalKey::Named(NamedKey::BrowserBack)).then(|| {
        let of = |code: KeyCode, named: NamedKey| {
            let key = LogicalKey::Named(named);
            if pressed {
                KeyEvent::press(code, key).with_text(None)
            } else {
                KeyEvent::release(code, key)
            }
        };
        [
            of(KeyCode::BrowserBack, NamedKey::BrowserBack),
            of(KeyCode::Escape, NamedKey::Escape),
        ]
    })
}

/// winit's `key_without_modifiers` where the platform has it (Windows, macOS, X11, Wayland).
#[cfg(any(
    target_os = "windows",
    target_os = "macos",
    all(
        unix,
        not(any(
            target_os = "ios",
            target_os = "android",
            target_os = "emscripten",
            target_os = "redox"
        ))
    )
))]
fn key_without_modifiers(event: &winit::event::KeyEvent) -> Option<winit::keyboard::Key> {
    use winit::platform::modifier_supplement::KeyEventExtModifierSupplement;
    Some(event.key_without_modifiers())
}

/// Web, Android, iOS: not reported (the logical key is used instead).
#[cfg(not(any(
    target_os = "windows",
    target_os = "macos",
    all(
        unix,
        not(any(
            target_os = "ios",
            target_os = "android",
            target_os = "emscripten",
            target_os = "redox"
        ))
    )
)))]
fn key_without_modifiers(_event: &winit::event::KeyEvent) -> Option<winit::keyboard::Key> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// iOS's Return (the text "\n" of an unidentified key) is the Enter key, pressed and
    /// released, with no text; any other text stays text.
    #[test]
    fn the_text_of_return_is_the_enter_key() {
        use winit::keyboard::KeyCode;
        for text in ["\n", "\r", "\r\n"] {
            let [down, up] = return_as_enter(text).expect("Return");
            assert_eq!(
                (down.physical, down.pressed, down.text.as_deref()),
                (KeyCode::Enter, true, None)
            );
            assert_eq!(
                (up.physical, up.pressed, up.text.as_deref()),
                (KeyCode::Enter, false, None)
            );
            let mut input = Input::default();
            input.inject_key_event(down);
            assert!(input.just_pressed(KeyCode::Enter));
            assert_eq!(input.text(), "");
            input.inject_key_event(up);
            assert!(!input.pressed(KeyCode::Enter));
        }
        assert!(return_as_enter("a").is_none());
        assert!(return_as_enter("\n\n").is_none());
        assert!(return_as_enter("").is_none());
    }

    /// Android's Back is the Back key and the Escape key, down with it and up with it; no
    /// other key of no place is.
    #[test]
    fn back_is_also_escape() {
        use crate::layout::{LogicalKey, NamedKey};
        use winit::keyboard::KeyCode;
        let back = LogicalKey::Named(NamedKey::BrowserBack);
        let mut input = Input::default();
        for key_event in back_as_escape(&back, true).expect("Back") {
            assert_eq!(key_event.text, None);
            input.inject_key_event(key_event);
        }
        assert!(input.just_pressed(KeyCode::BrowserBack) && input.just_pressed(KeyCode::Escape));
        assert_eq!(input.text(), "");
        for key_event in back_as_escape(&back, false).expect("Back") {
            input.inject_key_event(key_event);
        }
        assert!(!input.pressed(KeyCode::BrowserBack) && !input.pressed(KeyCode::Escape));
        assert!(back_as_escape(&LogicalKey::Named(NamedKey::BrowserForward), true).is_none());
        assert!(back_as_escape(&LogicalKey::Character("a".into()), true).is_none());
    }

    #[test]
    fn surface_retry_needs_window_renderer_and_missing_surface() {
        let r = SurfaceRetry::default();
        let now = Instant::now();
        assert!(r.due(now, false, true, Some(false)));
        assert!(
            !r.due(now, true, true, Some(false)),
            "suspended: Android dropped it on purpose"
        );
        assert!(!r.due(now, false, false, Some(false)), "no window");
        assert!(!r.due(now, false, true, None), "no renderer yet");
        assert!(!r.due(now, false, true, Some(true)), "surface attached");
    }

    #[test]
    fn surface_retry_backs_off_after_any_attempt() {
        let t0 = Instant::now();
        let almost = t0 + SurfaceRetry::INTERVAL - Duration::from_millis(1);
        let later = t0 + SurfaceRetry::INTERVAL;

        let mut r = SurfaceRetry::default();
        r.failed(t0);
        assert!(!r.due(t0, false, true, Some(false)));
        assert!(!r.due(almost, false, true, Some(false)));
        assert!(r.due(later, false, true, Some(false)));

        // A success throttles too: a surface lost again right away waits for the interval.
        let mut r = SurfaceRetry::default();
        r.succeeded(t0);
        assert!(!r.due(almost, false, true, Some(false)));
        assert!(r.due(later, false, true, Some(false)));
    }

    #[test]
    fn surface_retry_logs_once_per_failure_streak() {
        let mut r = SurfaceRetry::default();
        let t = Instant::now();
        assert!(!r.succeeded(t), "success without a prior failure is not a recovery");
        assert!(r.failed(t), "first failure is logged");
        assert!(!r.failed(t), "repeated failures are not");
        assert!(!r.failed(t));
        assert!(r.succeeded(t), "recovery after a failure is logged");
        assert!(r.failed(t), "a new streak logs again");
    }
}
