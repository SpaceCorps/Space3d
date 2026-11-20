//! [`EguiContext`] resource and the per-frame begin/end systems.
//!
//! Each frame is one egui pass, begun in `Stage::PreUpdate` (`begin_frame`) and ended in
//! `Stage::PostUpdate` (`end_frame`) with `egui::Context::begin_pass` / `end_pass`, so UI
//! systems in between use the context directly. egui 0.36 calls its plugins' pass hooks
//! (`Plugin::on_begin_pass` / `on_end_pass`) only from `Context::run_ui`, so the engine runs them
//! itself in `run_ui`'s order: the begin hooks right after the pass begins, the end hooks right
//! before it ends, both with the same top-level [`egui::Ui`] over the screen (id
//! `(viewport_id, "__top_ui")`, like `run_ui`'s). See [`EguiContext::add_plugin`].

use crate::input::InputTranslator;
use crate::overlay::EguiOverlay;
use crate::{UiPlugin, UiTheme, UiThemeSync};
use space3d_core::glam::Vec2;
use space3d_core::{SystemCtx, Time};
use space3d_platform::{Input, WindowState};
use space3d_render::{RenderOverlays, Renderer};
use std::any::TypeId;
use std::sync::{Arc, Mutex};

/// Which egui plugin pass hook to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PassHook {
    Begin,
    End,
}

/// Runs one plugin's hook (if that plugin is registered with the context).
type HookRunner = fn(&egui::Context, &mut egui::Ui, PassHook);

fn run_hook<P: egui::Plugin>(ctx: &egui::Context, ui: &mut egui::Ui, hook: PassHook) {
    ctx.with_plugin::<P, _>(|plugin| match hook {
        PassHook::Begin => plugin.on_begin_pass(ui),
        PassHook::End => plugin.on_end_pass(ui),
    });
}

fn hook_of<P: egui::Plugin>() -> (TypeId, HookRunner) {
    (TypeId::of::<P>(), run_hook::<P>)
}

/// egui's built-in plugins with pass hooks, in egui's registration order (`Context::default`).
/// egui's first plugin, the private `CallbackPlugin` behind `Context::on_begin_pass` /
/// `Context::on_end_pass`, cannot be reached from outside egui.
fn builtin_hooks() -> Vec<(TypeId, HookRunner)> {
    vec![
        hook_of::<egui::debug_text::DebugTextPlugin>(),
        hook_of::<egui::text_selection::LabelSelectionState>(),
        hook_of::<egui::DragAndDrop>(),
    ]
}

// The pass's top-level `Ui` lives in the resource between `begin_frame` and `end_frame`.
const _: () = {
    const fn send_sync<T: Send + Sync>() {}
    send_sync::<egui::Ui>();
};

/// Statistics of the last finished egui pass.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct UiFrameStats {
    /// Completed egui passes since startup.
    pub passes: u64,
    /// Tessellated clipped primitives.
    pub primitives: usize,
    /// Vertices uploaded for the frame.
    pub vertices: usize,
    /// Indices uploaded for the frame.
    pub indices: usize,
    /// egui-managed textures alive.
    pub textures: usize,
    /// Physical pixels per point used for tessellation.
    pub pixels_per_point: f32,
}

/// The egui context of the app (resource inserted by [`UiPlugin`]).
///
/// Between `Stage::PreUpdate` and `Stage::PostUpdate` an egui pass is active: UI code in
/// `Stage::Update` (see [`crate::UiAppExt`]) can use [`EguiContext::ctx`] directly.
pub struct EguiContext {
    ctx: egui::Context,
    in_pass: bool,
    enabled: bool,
    available_rect: egui::Rect,
    root_uis: u64,
    pixels_per_point: f32,
    screen_size_px: [u32; 2],
    wants_pointer: bool,
    wants_keyboard: bool,
    pointer_over: bool,
    injected: Vec<egui::Event>,
    clipboard: String,
    cursor_icon: egui::CursorIcon,
    /// `cursor_icon` has been written to `space3d_platform::WindowCursor` at least once.
    cursor_sent: bool,
    stats: UiFrameStats,
    /// Plugins whose pass hooks the engine runs, in order.
    plugin_hooks: Vec<(TypeId, HookRunner)>,
    /// The pass's top-level `Ui`, handed to the plugins' begin and end hooks.
    top_ui: Option<egui::Ui>,
}

impl std::fmt::Debug for EguiContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EguiContext")
            .field("in_pass", &self.in_pass)
            .field("enabled", &self.enabled)
            .field("pixels_per_point", &self.pixels_per_point)
            .field("screen_size_px", &self.screen_size_px)
            .finish()
    }
}

impl EguiContext {
    pub(crate) fn new(ctx: egui::Context, settings: &UiPlugin) -> Self {
        let (w, h) = settings.fallback_size;
        Self {
            ctx,
            in_pass: false,
            enabled: true,
            available_rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(w as f32, h as f32)),
            root_uis: 0,
            pixels_per_point: settings.pixels_per_point.unwrap_or(1.0),
            screen_size_px: [w, h],
            wants_pointer: false,
            wants_keyboard: false,
            pointer_over: false,
            injected: Vec::new(),
            clipboard: String::new(),
            cursor_icon: egui::CursorIcon::Default,
            cursor_sent: false,
            stats: UiFrameStats::default(),
            plugin_hooks: builtin_hooks(),
            top_ui: None,
        }
    }

    /// Add an egui [`egui::Plugin`] to the context (`egui::Context::add_plugin`) and run its
    /// pass hooks every pass: `on_begin_pass` right after the pass begins (`Stage::PreUpdate`,
    /// before UI systems), `on_end_pass` right before it ends (`Stage::PostUpdate`, after them),
    /// both with the pass's top-level [`egui::Ui`], as `egui::Context::run_ui` does. Hooks run
    /// in registration order, after egui's built-in plugins (debug text, label selection,
    /// drag-and-drop). Adding a plugin type twice does nothing.
    ///
    /// Use this instead of `egui::Context::add_plugin`: egui runs pass hooks only from
    /// `run_ui`, which the engine's pass does not use, and egui has no way to list the plugins
    /// of a context, so the engine only runs the hooks of plugins it knows about. For a plugin
    /// that is already registered with the context (e.g. by a library), see
    /// [`Self::register_pass_hooks`]. egui's `Context::on_begin_pass` / `on_end_pass` callbacks
    /// live in a private egui plugin and never run; register a plugin instead.
    ///
    /// The input and output hooks (`Plugin::input_hook`, `output_hook`) are run by egui itself.
    pub fn add_plugin<P: egui::Plugin>(&mut self, plugin: P) {
        self.ctx.add_plugin(plugin);
        self.register_pass_hooks::<P>();
    }

    /// Run the pass hooks of plugin type `P` every pass, like [`Self::add_plugin`], for a plugin
    /// registered with `egui::Context::add_plugin` (nothing runs while no `P` is registered).
    /// Registering a type twice does nothing.
    pub fn register_pass_hooks<P: egui::Plugin>(&mut self) {
        let hook = hook_of::<P>();
        if !self.plugin_hooks.iter().any(|(id, _)| *id == hook.0) {
            self.plugin_hooks.push(hook);
        }
    }

    /// Run every registered plugin's `hook` with `ui`.
    fn run_pass_hooks(&self, ui: &mut egui::Ui, hook: PassHook) {
        for (_, run) in &self.plugin_hooks {
            run(&self.ctx, ui, hook);
        }
    }

    /// End the running pass: the plugins' end hooks, then `egui::Context::end_pass`.
    fn end_pass(&mut self) -> egui::FullOutput {
        if let Some(mut top) = self.top_ui.take() {
            self.run_pass_hooks(&mut top, PassHook::End);
        }
        self.in_pass = false;
        self.ctx.end_pass()
    }

    /// The egui context (cheap to clone).
    pub fn ctx(&self) -> &egui::Context {
        &self.ctx
    }

    /// Whether an egui pass is running (between `PreUpdate` and `PostUpdate`).
    pub fn in_pass(&self) -> bool {
        self.in_pass
    }

    /// Enable/disable the whole UI: while disabled no pass runs and nothing is drawn.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// See [`Self::set_enabled`].
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Physical pixels per egui point of the current/last pass.
    pub fn pixels_per_point(&self) -> f32 {
        self.pixels_per_point
    }

    /// Size of the UI area in physical pixels.
    pub fn screen_size_px(&self) -> [u32; 2] {
        self.screen_size_px
    }

    /// Convert a physical pixel position (e.g. `Input::mouse_position`) to egui points.
    pub fn to_points(&self, physical: Vec2) -> egui::Pos2 {
        egui::pos2(physical.x / self.pixels_per_point, physical.y / self.pixels_per_point)
    }

    /// Convert egui points to physical pixels (e.g. for `Input::inject_mouse_move`).
    pub fn to_physical(&self, points: egui::Pos2) -> Vec2 {
        Vec2::new(points.x * self.pixels_per_point, points.y * self.pixels_per_point)
    }

    /// egui is using the pointer (hovering a window, dragging a slider...) as of the last
    /// finished pass — games should ignore mouse input then.
    pub fn wants_pointer_input(&self) -> bool {
        self.wants_pointer
    }

    /// A text field has keyboard focus (as of the last finished pass).
    pub fn wants_keyboard_input(&self) -> bool {
        self.wants_keyboard
    }

    /// The pointer is over any egui area (as of the last finished pass).
    pub fn is_pointer_over_ui(&self) -> bool {
        self.pointer_over
    }

    /// Whether a pointer release in this frame's pass was a **cancel**: the window lost the
    /// focus mid-press, the OS showed the button up with no release event, or the system
    /// cancelled a touch (see `docs/platforms.md`, "Mouse buttons, focus and the UI"). egui
    /// registers no click and no drop for it, but it still ends the press, so
    /// `Response::drag_stopped()` and `hovered()` + `any_released()` are `true` in that frame:
    /// code that decides a drop or a commit from them should skip a frame for which this is
    /// `true` (or [`Self::button_cancelled_this_frame`] for its button) and undo instead.
    /// Valid from `Stage::PreUpdate` of the frame (the pass begins) until the next frame's.
    /// [`crate::pointer_cancelled_this_frame`] is the same for code with only an
    /// [`egui::Context`].
    pub fn pointer_cancelled_this_frame(&self) -> bool {
        crate::input::pointer_cancelled_this_frame(&self.ctx)
    }

    /// Like [`Self::pointer_cancelled_this_frame`] for one pointer button (a cancelled touch is
    /// the primary button's).
    pub fn button_cancelled_this_frame(&self, button: egui::PointerButton) -> bool {
        crate::input::pointer_cancel_cause(&self.ctx, button).is_some()
    }

    /// Queue a raw egui event for the next pass (tests, custom platforms, IME...).
    pub fn push_event(&mut self, event: egui::Event) {
        self.injected.push(event);
    }

    /// The UI clipboard: text copied in egui lands here; `Ctrl/Cmd+V` pastes it.
    pub fn clipboard(&self) -> &str {
        &self.clipboard
    }

    /// Replace the clipboard contents (pasted with `Ctrl/Cmd+V`).
    pub fn set_clipboard(&mut self, text: impl Into<String>) {
        self.clipboard = text.into();
    }

    /// Cursor icon egui asked for in the last pass.
    pub fn cursor_icon(&self) -> egui::CursorIcon {
        self.cursor_icon
    }

    /// Statistics of the last finished pass.
    pub fn stats(&self) -> UiFrameStats {
        self.stats
    }

    /// The egui context if a pass is active (for systems that draw UI).
    pub fn active_context(sys: &SystemCtx) -> Option<egui::Context> {
        let e = sys.resources.get::<EguiContext>()?;
        (e.in_pass && e.enabled).then(|| e.ctx.clone())
    }

    /// Run `f` with the frame's root [`egui::Ui`], i.e. the screen area not yet claimed by
    /// panels added through earlier root-ui calls this frame. Returns `None` when no pass is
    /// active.
    pub fn with_root_ui<R>(sys: &mut SystemCtx, f: impl FnOnce(&mut egui::Ui, &mut SystemCtx) -> R) -> Option<R> {
        let (ctx, rect, id) = {
            let e = sys.resources.get_mut::<EguiContext>()?;
            if !e.in_pass || !e.enabled {
                return None;
            }
            e.root_uis += 1;
            (
                e.ctx.clone(),
                e.available_rect,
                egui::Id::new(("space3d_root_ui", e.root_uis)),
            )
        };
        let mut ui = egui::Ui::new(
            ctx,
            id,
            egui::UiBuilder::new()
                .layer_id(egui::LayerId::background())
                .max_rect(rect),
        );
        let out = f(&mut ui, sys);
        if let Some(e) = sys.resources.get_mut::<EguiContext>() {
            e.available_rect = ui.available_rect_before_wrap();
        }
        Some(out)
    }

    /// Screen area not yet claimed by root-ui panels this frame (points).
    pub fn available_rect(&self) -> egui::Rect {
        self.available_rect
    }
}

/// Screen size (physical px), native pixels-per-point and focus for this frame.
fn screen_info(sys: &SystemCtx, settings: &UiPlugin) -> ([u32; 2], f32, bool, usize) {
    let window = sys.resources.get::<WindowState>();
    let renderer = sys.resources.get::<Renderer>();
    let size = match (renderer, window) {
        (Some(r), _) => {
            let (w, h) = r.size();
            [w, h]
        }
        (None, Some(w)) => [w.physical_size.x, w.physical_size.y],
        (None, None) => [settings.fallback_size.0, settings.fallback_size.1],
    };
    let native_ppp = settings
        .pixels_per_point
        .or_else(|| {
            let points = settings.short_side_points.filter(|p| *p > 0.0)?;
            Some((size[0].min(size[1]) as f32 / points).max(1.0))
        })
        .unwrap_or_else(|| window.map_or(1.0, |w| w.scale_factor as f32))
        .max(0.1);
    let focused = window.is_none_or(|w| w.focused);
    let max_texture_side = renderer.map_or(2048, |r| r.device().limits().max_texture_dimension_2d as usize);
    (size, native_ppp, focused, max_texture_side)
}

/// A part of the UI's CPU work in a frame, timed while the returned guard lives: a trace span of the kind the schedule opens round every
/// system (`"system"`, with a `name`), so a profiler lists it beside them ("ui::end_frame.tessellate"). Costs nothing while no
/// subscriber takes trace spans.
fn part(name: &'static str) -> tracing::span::EnteredSpan {
    tracing::trace_span!("system", name = name).entered()
}

/// `Stage::PreUpdate`: translate input and begin the egui pass.
pub(crate) fn begin_frame(
    sys: &mut SystemCtx,
    translator: &mut InputTranslator,
    theme: &mut UiThemeSync,
    settings: &UiPlugin,
) {
    let (size, native_ppp, focused, max_texture_side) = screen_info(sys, settings);
    let ui_theme = sys.resources.get::<UiTheme>().copied();
    let (time, dt) = sys
        .resources
        .get::<Time>()
        .map_or((0.0, 0.0), |t| (t.elapsed_secs(), t.delta_secs()));
    let Some(e) = sys.resources.get_mut::<EguiContext>() else {
        return;
    };
    if e.in_pass {
        // The previous pass was never ended (e.g. PostUpdate was skipped): drop it (its
        // plugins still see it end).
        let _ = e.end_pass();
    }
    if !e.enabled {
        // No pass, so no release is a cancel of this frame's.
        crate::input::publish_pointer_cancels(&e.ctx, Default::default());
        return;
    }
    let ppp = native_ppp * e.ctx.zoom_factor();
    let clipboard = e.clipboard.clone();
    let injected = std::mem::take(&mut e.injected);

    let mut events = Vec::new();
    translator.focus(focused, &mut events);
    let (cancelled, cancels) = match sys.resources.get::<Input>() {
        Some(input) => {
            let cancelled = translator.translate(input, ppp, time, &clipboard, &mut events);
            (cancelled, translator.cancelled_buttons())
        }
        None => (false, Default::default()),
    };
    events.extend(injected);
    if let Some(o) = sys.resources.get_mut::<RenderOverlays>() {
        o.set_scale_factor(ppp);
    }

    // `UiPlugin::inside_safe_area`: the UI keeps out of a phone's notch and home bar.
    let safe = sys
        .resources
        .get::<space3d_platform::SafeArea>()
        .copied()
        .filter(|_| settings.inside_safe_area)
        .unwrap_or_default();
    let e = sys.resources.expect_mut::<EguiContext>();
    let full = egui::vec2(size[0] as f32 / ppp, size[1] as f32 / ppp);
    let (left, top) = (safe.left as f32 / ppp, safe.top as f32 / ppp);
    let screen = egui::Rect::from_min_max(
        egui::pos2(left, top),
        egui::pos2(
            (full.x - safe.right as f32 / ppp).max(left + 1.0),
            (full.y - safe.bottom as f32 / ppp).max(top + 1.0),
        ),
    );
    let mut raw = egui::RawInput {
        screen_rect: Some(screen),
        max_texture_side: Some(max_texture_side),
        time: Some(time),
        predicted_dt: if dt > 0.0 {
            dt.clamp(1.0 / 1000.0, 0.25)
        } else {
            1.0 / 60.0
        },
        events,
        focused,
        ..Default::default()
    };
    raw.viewports.insert(
        egui::ViewportId::ROOT,
        egui::ViewportInfo {
            native_pixels_per_point: Some(native_ppp),
            inner_rect: Some(screen),
            focused: Some(focused),
            ..Default::default()
        },
    );
    crate::input::publish_pointer_cancels(&e.ctx, cancels);
    e.pixels_per_point = ppp;
    e.screen_size_px = size;
    e.available_rect = screen;
    e.root_uis = 0;
    if cancelled && e.ctx.plugin_opt::<egui::DragAndDrop>().is_some() {
        // A release was cancelled (focus loss, a release only the OS saw): whatever was being
        // dragged is dropped nowhere.
        egui::DragAndDrop::clear_payload(&e.ctx);
    }
    // An inserted or changed `UiTheme` restyles egui once, between passes (no resource: the
    // style is never touched).
    theme.sync(&e.ctx, ui_theme.as_ref());
    // (egui drops here what the last pass laid out and this one has not asked for yet: the galleys)
    let span = part("ui::begin_frame.pass");
    e.ctx.begin_pass(raw);
    drop(span);
    e.in_pass = true;
    // `run_ui`'s top-level ui, handed to the plugins' begin hooks now and end hooks in
    // `end_frame`. What they claim (a panel...) is not available to root-ui systems.
    let ctx = e.ctx.clone();
    let mut top = egui::Ui::new(
        ctx.clone(),
        egui::Id::new((ctx.viewport_id(), "__top_ui")),
        egui::UiBuilder::new()
            .layer_id(egui::LayerId::background())
            .max_rect(ctx.viewport_rect()),
    );
    e.run_pass_hooks(&mut top, PassHook::Begin);
    e.available_rect = top.available_rect_before_wrap();
    e.top_ui = Some(top);
}

/// `Stage::PostUpdate`: end the pass, tessellate and hand the frame to the render overlay.
pub(crate) fn end_frame(sys: &mut SystemCtx, overlay: &Arc<Mutex<EguiOverlay>>) {
    let natives = crate::textures::resolve_native_views(sys.resources);
    // A focused text field asks the platform for typed text: on a phone, the on-screen
    // keyboard (`space3d_platform::TextInput`; the last pass's answer, one frame late).
    // (a text field, not any focus: a focused button takes Enter and wants no keyboard on screen)
    if let Some(wanted) = sys.resources.get::<EguiContext>().map(|e| e.ctx().text_edit_focused()) {
        let text_input = sys.resources.init::<space3d_platform::TextInput>();
        if text_input.wanted != wanted {
            text_input.wanted = wanted;
        }
    }
    let Some(e) = sys.resources.get_mut::<EguiContext>() else {
        return;
    };
    if !e.in_pass {
        if !e.enabled {
            // Nothing to draw while disabled.
            crate::overlay::lock(overlay).clear_frame();
        }
        return;
    }
    // The plugins' end hooks first (they may still show UI, end a drag-to-select, drop a
    // drag-and-drop payload...), as in `run_ui`.
    if let Some(mut top) = e.top_ui.take() {
        e.run_pass_hooks(&mut top, PassHook::End);
    }
    // Query interaction state while the pass state is still the current one.
    // (egui's own `is_pointer_over_egui` relies on `run_ui` bookkeeping, so the background
    // layer is tested against our root-ui available rect instead.)
    let pointer = e.ctx.input(|i| i.pointer.interact_pos());
    e.pointer_over = pointer.is_some_and(|p| {
        e.ctx
            .layer_id_at(p)
            .is_some_and(|l| l.order != egui::Order::Background || !e.available_rect.contains(p))
    });
    let any_down = e.ctx.input(|i| i.pointer.any_down());
    e.wants_pointer = e.ctx.egui_is_using_pointer() || (e.pointer_over && !any_down);
    e.wants_keyboard = e.ctx.egui_wants_keyboard_input();
    let span = part("ui::end_frame.pass");
    let output = e.end_pass();
    drop(span);
    for cmd in &output.platform_output.commands {
        if let egui::OutputCommand::CopyText(text) = cmd {
            e.clipboard.clone_from(text);
        }
    }
    // Only a change of egui's cursor is passed on, so a cursor a game sets itself (e.g. a
    // crosshair in flight) stays until egui asks for a different one.
    let new_cursor = output.platform_output.cursor_icon;
    let cursor_changed = new_cursor != e.cursor_icon || !e.cursor_sent;
    e.cursor_icon = new_cursor;
    let ppp = output.pixels_per_point;
    let span = part("ui::end_frame.tessellate");
    let primitives = e.ctx.tessellate(output.shapes, ppp);
    drop(span);
    let (mut vertices, mut indices) = (0, 0);
    for p in &primitives {
        if let egui::epaint::Primitive::Mesh(m) = &p.primitive {
            vertices += m.vertices.len();
            indices += m.indices.len();
        }
    }
    e.stats = UiFrameStats {
        passes: e.stats.passes + 1,
        primitives: primitives.len(),
        vertices,
        indices,
        textures: e.ctx.tex_manager().read().num_allocated(),
        pixels_per_point: ppp,
    };
    crate::overlay::lock(overlay).set_frame(primitives, ppp, output.textures_delta, natives);
    if cursor_changed && let Some(window_cursor) = sys.resources.get_mut::<space3d_platform::WindowCursor>() {
        window_cursor.icon = winit_cursor(new_cursor);
        if let Some(e) = sys.resources.get_mut::<EguiContext>() {
            e.cursor_sent = true;
        }
    }
}

/// egui's cursor as the window's (`None` hides the pointer), as egui-winit maps it.
pub fn winit_cursor(icon: egui::CursorIcon) -> Option<space3d_platform::CursorIcon> {
    use egui::CursorIcon as E;
    use space3d_platform::CursorIcon as W;
    Some(match icon {
        E::None => return None,
        E::Default => W::Default,
        E::ContextMenu => W::ContextMenu,
        E::Help => W::Help,
        E::PointingHand => W::Pointer,
        E::Progress => W::Progress,
        E::Wait => W::Wait,
        E::Cell => W::Cell,
        E::Crosshair => W::Crosshair,
        E::Text => W::Text,
        E::VerticalText => W::VerticalText,
        E::Alias => W::Alias,
        E::Copy => W::Copy,
        E::Move => W::Move,
        E::NoDrop => W::NoDrop,
        E::NotAllowed => W::NotAllowed,
        E::Grab => W::Grab,
        E::Grabbing => W::Grabbing,
        E::AllScroll => W::AllScroll,
        E::ResizeHorizontal => W::EwResize,
        E::ResizeNeSw => W::NeswResize,
        E::ResizeNwSe => W::NwseResize,
        E::ResizeVertical => W::NsResize,
        E::ResizeEast => W::EResize,
        E::ResizeSouthEast => W::SeResize,
        E::ResizeSouth => W::SResize,
        E::ResizeSouthWest => W::SwResize,
        E::ResizeWest => W::WResize,
        E::ResizeNorthWest => W::NwResize,
        E::ResizeNorth => W::NResize,
        E::ResizeNorthEast => W::NeResize,
        E::ResizeColumn => W::ColResize,
        E::ResizeRow => W::RowResize,
        E::ZoomIn => W::ZoomIn,
        E::ZoomOut => W::ZoomOut,
    })
}
