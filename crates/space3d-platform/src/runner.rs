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
    #[cfg(target_arch = "wasm32")]
    pending: std::rc::Rc<std::cell::RefCell<Option<Result<Renderer, space3d_render::RenderError>>>>,
}

fn build_event_loop() -> Result<EventLoop<()>, winit::error::EventLoopError> {
    #[cfg(target_os = "android")]
    {
        use winit::platform::android::EventLoopBuilderExtAndroid;
        let Some(android_app) = crate::mobile::android_app() else {
            tracing::error!("Android: start the app through space3d_platform::android_main_entry");
            // `NotSupportedError::new` is private in winit 0.30; `ExitFailure` is the public error
            // for "the loop cannot run".
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
    let size = window.inner_size();
    r.resize(size.width.max(1), size.height.max(1));
    Ok(())
}

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
    event_loop.set_control_flow(ControlFlow::Poll);
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
            .with_inner_size(winit::dpi::LogicalSize::new(s.width as f64, s.height as f64))
            .with_resizable(s.resizable);
        if crate::background::background_mode() {
            attrs = attrs.with_active(false);
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
        let size = window.inner_size();
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
        if let Some(w) = &self.window {
            w.pre_present_notify();
        }
        self.app.update(dt);
        self.apply_window_state();
        if self.app.should_exit() {
            self.exiting = true;
            event_loop.exit();
        }
    }

    /// Push game-side changes of [`WindowState`] to the OS window.
    fn apply_window_state(&mut self) {
        let Some(window) = self.window.clone() else {
            return;
        };
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
        let size = window.inner_size();
        self.on_resize(UVec2::new(size.width, size.height));
        self.send(PlatformEvent::Resumed);
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
        self.input().release_all();
        self.send(PlatformEvent::Suspended);
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
            WindowEvent::Resized(size) => self.on_resize(UVec2::new(size.width, size.height)),
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.state().scale_factor = scale_factor;
                self.send(PlatformEvent::ScaleFactorChanged(scale_factor));
            }
            WindowEvent::Focused(focused) => {
                self.state().focused = focused;
                if focused {
                    // The layout may have been switched while another app had focus.
                    self.poll_layout(Instant::now(), true);
                } else {
                    self.input().release_all();
                }
                self.send(PlatformEvent::Focused(focused));
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
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
                        if pressed && let Some(text) = &event.text {
                            self.input().inject_text(text);
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
                self.input().inject_touch(t.id, TouchPhase::from(t.phase), p);
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
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
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
