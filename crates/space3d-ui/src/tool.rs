//! Shared look and placement of the engine's in-game tools (debug overlay, profiler, console).
//!
//! The tools show up inside games that style egui themselves, so they never read their sizes
//! from the global style and never change it: each tool applies [`TOOL_DENSITY`] to its own
//! `Ui` (a scoped copy of the style), keeps the context's colors, and draws its own compact
//! header instead of egui's title bar (whose height follows the global heading size).
//!
//! They dock to the corners of the [`ToolDockArea`] (the whole screen by default; side by side
//! in a short one: [`crate::dock`]), follow it when it changes (the window or an editor panel is
//! resized) until the user drags them away, keep their default size for the current area until
//! the user resizes them (then the size the user gave them, as far as the area and the other
//! tools leave room), and sit on egui's foreground layer, above the game's own windows.

use crate::theme::{UiDensity, apply_density, visible_scroll_bars};
use space3d_core::Resources;
use std::sync::Arc;

/// Density of the engine tools: the compact preset with slightly smaller text.
pub(crate) const TOOL_DENSITY: UiDensity = UiDensity {
    body_size: 12.0,
    button_size: 12.0,
    monospace_size: 11.5,
    small_size: 10.0,
    heading_size: 12.5,
    ..UiDensity::compact()
};

/// Gap between a docked tool window and the dock area's edges (and between two tools).
pub(crate) const EDGE: f32 = 6.0;

/// Smallest outer size of a tool window.
pub(crate) const MIN_SIZE: egui::Vec2 = egui::vec2(160.0, 60.0);

/// Inner margin of a tool window (more at the bottom than at the top, so the last row never
/// touches the border and stays clear of the resize grip).
const MARGIN: egui::Margin = egui::Margin {
    left: 6,
    right: 6,
    top: 4,
    bottom: 8,
};

/// Screen area (egui points) the engine tools dock inside (resource; [`set_tool_dock_area`]): the debug overlay (F3) in
/// its top-left corner, the profiler (F4) top-right and the console bottom-left, each 6 pt in
/// from its edges. In an area too short to stack them (a band a HUD leaves free) the console
/// docks between the overlay and the profiler instead, so they stay inside and apart
/// (`docs/ui.md`, "Short dock areas"). A docked tool follows the area when it moves or changes
/// size; one the user dragged away stays where it was put (double-click its title to dock it
/// again).
///
/// Without the resource the tools dock to the whole screen (games). Hosts that cover part of the
/// screen with their own panels set it every frame to what is left, so the tools never cover
/// their menu bar or side panels: the standalone editor ([`space3d_editor::EditorPlugin`]) and
/// the storybook viewer set it to their central viewport. An area outside the screen is clipped
/// to it; one too small to hold a tool window is ignored (the tools use the whole screen).
///
/// The tools draw before most UI systems of a frame, so they use the area set in the previous
/// frame.
///
/// ```
/// use space3d_core::prelude::*;
/// use space3d_ui::{UiAppExt, egui, set_tool_dock_area};
///
/// # let mut app = App::new();
/// // A game with a side panel of its own keeps the engine tools off it.
/// app.add_root_ui_system(|ui: &mut egui::Ui, sys: &mut SystemCtx| {
///     egui::Panel::right("inventory").show(ui, |ui| {
///         ui.label("Salvage: 12 t");
///     });
///     set_tool_dock_area(sys.resources, ui.available_rect_before_wrap());
/// });
/// ```
///
/// [`space3d_editor::EditorPlugin`]: https://docs.rs/space3d-editor
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ToolDockArea(pub egui::Rect);

/// Set the [`ToolDockArea`] to `rect` (in place when it exists: hosts call this every frame).
pub fn set_tool_dock_area(resources: &mut Resources, rect: egui::Rect) {
    match resources.get_mut::<ToolDockArea>() {
        Some(area) => area.0 = rect,
        None => {
            resources.insert(ToolDockArea(rect));
        }
    }
}

/// The rect the tools dock inside this frame: the [`ToolDockArea`] within the screen, or the
/// screen (no resource, or an area smaller than one tool window with its gaps). With content
/// under the title bar (`space3d_platform::TitleBar`) the screen begins under its strip, so
/// that no tool lies under the window buttons.
pub(crate) fn dock_rect(resources: &Resources, ctx: &egui::Context) -> egui::Rect {
    let mut screen = ctx.content_rect();
    if let Some(bar) = resources.get::<space3d_platform::TitleBar>().filter(|bar| bar.active) {
        let under = bar.height as f32 / ctx.pixels_per_point();
        screen.min.y = screen.min.y.max(under).min(screen.max.y);
    }
    let min = MIN_SIZE + egui::Vec2::splat(2.0 * EDGE);
    resources
        .get::<ToolDockArea>()
        .map(|area| area.0.intersect(screen))
        .filter(|r| r.width() >= min.x && r.height() >= min.y)
        .unwrap_or(screen)
}

/// A style derived from the `base` style by a tool, cached with the `base` it came from.
#[derive(Clone)]
struct DerivedStyle {
    base: Arc<egui::Style>,
    derived: Arc<egui::Style>,
}

/// Give `ui` its current style changed by `derive`, from a cache keyed by `key`: no style clone
/// per frame while `ui`'s style is unchanged (`ui.spacing_mut()` / `ui.style_mut()` would copy
/// the whole style every frame).
pub(crate) fn derived_style(ui: &mut egui::Ui, key: &'static str, derive: fn(&mut egui::Style)) {
    let id = egui::Id::new(("space3d_ui::tool_style", key));
    let base = ui.style().clone();
    let cached = ui
        .ctx()
        .data(|d| d.get_temp::<DerivedStyle>(id))
        .filter(|c| Arc::ptr_eq(&c.base, &base));
    let derived = match cached {
        Some(c) => c.derived,
        None => {
            let mut style = (*base).clone();
            derive(&mut style);
            let derived = Arc::new(style);
            let entry = DerivedStyle {
                base,
                derived: derived.clone(),
            };
            ui.ctx().data_mut(|d| d.insert_temp(id, entry));
            derived
        }
    };
    ui.set_style(derived);
}

/// Apply the tool density, visible scroll bars and readable weak text to `ui` only (the global
/// style is untouched; colors stay the context's). Text keeps egui's per-layout wrapping; rows
/// that must stay on one line say so.
pub(crate) fn scope_style(ui: &mut egui::Ui) {
    derived_style(ui, "tool", |style| {
        apply_density(style, &TOOL_DENSITY);
        style.spacing.scroll = visible_scroll_bars(TOOL_DENSITY.scroll_bar_width);
        style.visuals.weak_text_alpha = weak_text_alpha(&style.visuals);
    });
}

/// Most opacity weak text gets in the tools: above it weak and normal text look alike.
const MAX_WEAK_ALPHA: f32 = 0.85;

/// Opacity of weak text in the tools (trace / debug log lines, placeholders), derived from the
/// palette: the least, from the palette's own `weak_text_alpha` up to [`MAX_WEAK_ALPHA`], at
/// which weak text on the window fill reaches 65 % of the normal text's contrast, at least 3:1
/// and at most 4.5:1. Weak text stays readable on egui's grey defaults (whose 0.6 is about
/// 2.4:1) and clearly dimmer than normal text under a bright game palette. An explicit
/// `weak_text_color` is left to the palette.
pub(crate) fn weak_text_alpha(visuals: &egui::Visuals) -> f32 {
    let base = visuals.weak_text_alpha;
    if visuals.weak_text_color.is_some() || !base.is_finite() {
        return base;
    }
    let text = visuals.text_color();
    let [r, g, b, _] = visuals.window_fill.to_srgba_unmultiplied();
    let fill = egui::Color32::from_rgb(r, g, b);
    let target = (0.65 * contrast(text, fill, 1.0)).clamp(3.0, 4.5);
    let mut alpha = base.max(0.0);
    while alpha < MAX_WEAK_ALPHA && contrast(text, fill, alpha) < target {
        alpha += 0.01;
    }
    alpha.min(MAX_WEAK_ALPHA).max(base)
}

/// Contrast ratio (WCAG) of `text` drawn at `alpha` over the opaque `fill`, blended as the
/// renderer does: egui's premultiplied gamma color, added in linear space.
fn contrast(text: egui::Color32, fill: egui::Color32, alpha: f32) -> f32 {
    let lin = |c: u8| egui::ecolor::linear_f32_from_gamma_u8(c);
    let lum = |rgb: [f32; 3]| 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
    let weak = text.gamma_multiply(alpha);
    let over = |i: usize| lin(weak[i]) + lin(fill[i]) * (1.0 - alpha);
    let (a, b) = (
        lum([over(0), over(1), over(2)]),
        lum([lin(fill[0]), lin(fill[1]), lin(fill[2])]),
    );
    (a.max(b) + 0.05) / (a.min(b) + 0.05)
}

/// A per-window flag of the tools (`false` until set).
fn flag(ctx: &egui::Context, id: &str, name: &'static str) -> bool {
    ctx.data(|d| d.get_temp::<bool>(egui::Id::new(id).with(name)))
        .unwrap_or(false)
}

fn set_flag(ctx: &egui::Context, id: &str, name: &'static str, on: bool) {
    ctx.data_mut(|d| d.insert_temp(egui::Id::new(id).with(name), on));
}

const FLOATING: &str = "space3d_ui::floating";
const USER_SIZED: &str = "space3d_ui::user_sized";
const FREE_EDGE_PRESS: &str = "space3d_ui::free_edge_press";
/// Where the held press started, when it started on the window.
const PRESS: &str = "space3d_ui::press";
/// The axes ([x, y]) whose edges the held press grabbed.
const GRABBED: &str = "space3d_ui::grabbed";
/// The height a press that grabbed no top or bottom edge holds the window at.
const HELD_HEIGHT: &str = "space3d_ui::held_height";
/// The size the user gave the window (see [`user_size`]).
const USER_SIZE: &str = "space3d_ui::user_size";
/// The user's size as last applied to the docked window (capped to the dock area and the
/// width left beside the other tools).
const APPLIED_USER_SIZE: &str = "space3d_ui::applied_user_size";
const DEFAULT_SIZE: &str = "space3d_ui::default_size";

fn data<T: Clone + Send + Sync + 'static>(ctx: &egui::Context, id: &str, name: &'static str) -> Option<T> {
    ctx.data(|d| d.get_temp::<T>(egui::Id::new(id).with(name)))
}

fn set_data<T: Clone + Send + Sync + 'static>(ctx: &egui::Context, id: &str, name: &'static str, value: T) {
    ctx.data_mut(|d| d.insert_temp(egui::Id::new(id).with(name), value));
}

fn remove_data<T: Clone + Default + Send + Sync + 'static>(ctx: &egui::Context, id: &str, name: &'static str) {
    ctx.data_mut(|d| d.remove_temp::<T>(egui::Id::new(id).with(name)));
}

/// The default size a docked tool window `id` last took (see [`window`]).
fn applied_default(ctx: &egui::Context, id: &str) -> Option<egui::Vec2> {
    data(ctx, id, DEFAULT_SIZE)
}

fn set_applied_default(ctx: &egui::Context, id: &str, size: egui::Vec2) {
    set_data(ctx, id, DEFAULT_SIZE, size);
}

/// The size the user gave the tool window `id` by resizing it (outer, for each axis the last
/// size a drag of that axis' edge left it at; the other axis as it was then), when they did. A
/// docked window gets it back when the dock area and the other tools leave room for it again
/// (egui itself keeps only the size a window was last squeezed to). For a window drawn shorter
/// than it is given (the console's whole log lines), the height it is given.
pub(crate) fn user_size(ctx: &egui::Context, id: &str) -> Option<egui::Vec2> {
    data(ctx, id, USER_SIZE).filter(|_| is_user_sized(ctx, id))
}

/// Whether the tool window `id` was dragged away from its dock (it stays where the user put it).
pub(crate) fn is_floating(ctx: &egui::Context, id: &str) -> bool {
    flag(ctx, id, FLOATING)
}

/// Whether the user resized the tool window `id` (it keeps that size; otherwise a docked tool
/// takes its default size for the current dock area).
pub(crate) fn is_user_sized(ctx: &egui::Context, id: &str) -> bool {
    flag(ctx, id, USER_SIZED)
}

/// Where tool window `id` is this frame: where it floats, or docked with its `anchor` corner at
/// `dock_pos`, at most `bounds`' size: the size it had last frame (`default_size` before its first
/// frame). With `follow_default`, a tool the user has not resized takes `default_size` when that
/// is new (the dock area changed: [`window`] resizes it this frame); its size can end up smaller
/// than the default (the console is a whole number of log lines tall).
pub(crate) fn placed_rect(
    ctx: &egui::Context,
    id: &str,
    anchor: egui::Align2,
    dock_pos: egui::Pos2,
    default_size: egui::Vec2,
    bounds: egui::Rect,
    follow_default: bool,
) -> egui::Rect {
    let last = ctx.memory(|m| m.area_rect(egui::Id::new(id)));
    match last {
        Some(rect) if is_floating(ctx, id) => rect,
        _ => {
            let new_default = follow_default
                && !is_user_sized(ctx, id)
                && applied_default(ctx, id).is_some_and(|applied| applied != default_size);
            let size = match last {
                Some(r) if !new_default => r.size(),
                _ => default_size,
            };
            anchor.anchor_size(dock_pos, size.min(bounds.size()))
        }
    }
}

/// The frame of a tool window under `style` (the context's colors and stroke, the tools' margin).
fn frame(style: &egui::Style) -> egui::Frame {
    egui::Frame::window(style).inner_margin(MARGIN).corner_radius(4)
}

/// Size a tool window's frame adds around its contents (margins and stroke).
pub(crate) fn chrome(ctx: &egui::Context) -> egui::Vec2 {
    frame(&ctx.global_style()).total_margin().sum()
}

/// How a tool window is placed and sized.
pub(crate) struct ToolWindow<'a> {
    /// Stable id (egui remembers where the user moved / resized the window).
    pub id: &'a str,
    /// Header title.
    pub title: &'a str,
    /// Corner of the window that docks (its pivot).
    pub anchor: egui::Align2,
    /// Where the `anchor` corner goes while the window is docked (screen points).
    pub dock_pos: egui::Pos2,
    /// Rect a docked window stays inside: the dock area [`EDGE`] in from its edges. A window the
    /// user dragged away stays inside the screen instead.
    pub bounds: egui::Rect,
    /// Default outer size (frame included): the size of a docked window until the user resizes
    /// it (it follows the dock area as that changes). With [`Self::fit_height`], the height the
    /// contents should stay within.
    pub size: egui::Vec2,
    /// The window is exactly as tall as its contents (up to `size.y`) instead of user-resizable
    /// in height. The contents get the bottom they must stay above.
    pub fit_height: bool,
    /// Outer width the window must stay within this frame (another tool is in the way), or
    /// `f32::INFINITY`. Unlike `size.x`, applied every frame.
    pub max_width: f32,
    /// How much taller than it is drawn the window must be given to be drawn as tall again (the
    /// console, a whole number of log lines tall, is drawn up to a line shorter than it is
    /// given); 0 for a window as tall as it is given.
    pub height_slack: f32,
}

/// A tool window on the foreground layer: compact frame and header row (title, `summary`,
/// close button) under any global style. `open` is cleared by the close button. While docked it
/// keeps its `anchor` corner at `dock_pos` (so it follows the dock area) and, until the user
/// resizes it, its default size `spec.size` (which follows the dock area too); once resized, the
/// size the user gave it ([`user_size`]) within `bounds` and `spec.max_width`, which it gets back
/// when those grow again. Only the edges a press grabbed resize it: a press that grabbed no top
/// or bottom edge holds its height. Dragging it (or resizing it by a docked edge) makes it
/// float, double-clicking its title docks it again. The contents run with the tool density and
/// get the screen y their last row must end above (the window's largest content bottom).
pub(crate) fn window<R>(
    ctx: &egui::Context,
    spec: ToolWindow<'_>,
    open: &mut bool,
    summary: impl FnOnce(&mut egui::Ui),
    contents: impl FnOnce(&mut egui::Ui, f32) -> R,
) -> Option<R> {
    let frame = frame(&ctx.global_style());
    let chrome = frame.total_margin().sum();
    let floating = is_floating(ctx, spec.id);
    let user_sized = is_user_sized(ctx, spec.id);
    let bounds = if floating {
        ctx.content_rect().shrink(EDGE)
    } else {
        spec.bounds
    };
    let size = spec
        .size
        .min(bounds.size())
        .min(egui::vec2(spec.max_width, f32::INFINITY))
        .max(MIN_SIZE);
    let id = egui::Id::new(spec.id);
    let fit_id = id.with("fit_height");
    let last_rect = ctx.memory(|m| m.area_rect(id));
    let mut window = egui::Window::new(spec.title)
        .id(id)
        .title_bar(false)
        .frame(frame)
        .order(egui::Order::Foreground)
        .pivot(spec.anchor)
        .default_pos(spec.dock_pos)
        .default_size(size)
        .constrain_to(bounds)
        .min_size(MIN_SIZE);
    if !floating {
        window = window.current_pos(spec.dock_pos);
    }
    if spec.max_width.is_finite() {
        // (Not `size.x`: the user may widen the window up to what is in the way.)
        window = window.max_width(spec.max_width.max(size.x));
    }
    if last_rect.is_none() {
        // egui caps a new window's size by the *global* window margin; asking for the size as
        // the minimum on the first frame keeps the tool's size independent of the game's style.
        window = window.min_size(size);
    }
    if spec.fit_height {
        // The contents keep below `size.y` whatever fits; when not even their smallest layout
        // does (a tiny screen), the window grows rather than cutting a row.
        let height = ctx
            .data(|d| d.get_temp::<f32>(fit_id))
            .unwrap_or(size.y)
            .max(MIN_SIZE.y);
        window = window.resizable([true, false]).min_height(height).max_height(height);
    }
    // A press that started on the window (or its resize margin), held until the frame it is
    // released: only the user moves a docked window's corner, or resizes it, while one is held.
    // Whether it did is decided where it started, on the rect of that frame: while an edge is
    // dragged egui remembers the window where the drag would move it next frame (a docked one
    // stays put), which a narrowing drag moves away from the press.
    let (pressed, down, released) =
        ctx.input(|i| (i.pointer.any_pressed(), i.pointer.any_down(), i.pointer.any_released()));
    if pressed {
        let origin = ctx.input(|i| i.pointer.press_origin());
        let grab = ctx.global_style().interaction.resize_grab_radius_corner;
        let on = origin.zip(last_rect).filter(|(p, r)| r.expand(grab).contains(*p));
        set_data(ctx, spec.id, PRESS, on.map(|(p, _)| p));
        if let Some((p, r)) = on {
            let resizable = [true, !spec.fit_height];
            let edges = grabbed_edges(&ctx.global_style().interaction, r, p, resizable);
            // Resizing by a free edge (one away from the docked corner) leaves the window
            // docked. egui moves the window's top-left corner before it applies the new size, so
            // while the edge is dragged the docked corner seems to move with it; such a press
            // must not undock it.
            set_flag(ctx, spec.id, FREE_EDGE_PRESS, only_free_edges(edges, spec.anchor));
            let grabbed = edges.map(|e| e.contains(&true));
            set_data(ctx, spec.id, GRABBED, grabbed);
            // egui resizes a window by an edge to the rect it has as drawn, in both directions:
            // a window drawn shorter than it is given (the console) would lose a line to a drag
            // of its side edge, or to a move.
            let held = (!grabbed[1] && !spec.fit_height).then(|| r.height() + spec.height_slack);
            set_data(ctx, spec.id, HELD_HEIGHT, held);
        }
    }
    let press = data::<Option<egui::Pos2>>(ctx, spec.id, PRESS).flatten();
    // (egui forgets the press origin on the frame it is released.)
    let pressed_on_window =
        press.is_some_and(|p| (down && ctx.input(|i| i.pointer.press_origin()) == Some(p)) || (released && !down));
    if press.is_some() && !down && !released {
        set_data::<Option<egui::Pos2>>(ctx, spec.id, PRESS, None);
    }
    let free_edge_press = pressed_on_window && flag(ctx, spec.id, FREE_EDGE_PRESS);
    let grabbed = if pressed_on_window {
        data::<[bool; 2]>(ctx, spec.id, GRABBED).unwrap_or_default()
    } else {
        [false; 2]
    };
    if pressed_on_window && let Some(held) = data::<Option<f32>>(ctx, spec.id, HELD_HEIGHT).flatten() {
        let held = held.min(bounds.height());
        window = window.min_height(held).max_height(held);
    }
    // Docked and never resized by the user: when its default size changes (the dock area did,
    // or a host set one after the first frame), the window takes the new one, as if it had
    // started with it (egui keeps the size of the window's first frame). Its contents may keep
    // it smaller (the console's whole lines).
    if !floating && !user_sized && !pressed_on_window {
        let applied = applied_default(ctx, spec.id);
        if last_rect.is_some() && applied.is_some_and(|a| a.x != spec.size.x) {
            window = window.min_width(size.x).max_width(size.x);
        }
        if last_rect.is_some() && applied.is_some_and(|a| a.y != spec.size.y) && !spec.fit_height {
            window = window.min_height(size.y).max_height(size.y);
        }
        if applied != Some(spec.size) {
            set_applied_default(ctx, spec.id, spec.size);
        }
    }
    // Docked and resized by the user: the size the user gave it, within the dock area and the
    // width left beside the other tools. egui keeps only the size a window was last squeezed
    // to, so when the room comes back (the area got larger again, or a tool beside it went
    // away) the window takes the user's size again.
    if floating {
        // (Docked again, it takes the user's size once.)
        remove_data::<egui::Vec2>(ctx, spec.id, APPLIED_USER_SIZE);
    } else if user_sized
        && !pressed_on_window
        && let Some(last) = last_rect
    {
        let user =
            data::<egui::Vec2>(ctx, spec.id, USER_SIZE).unwrap_or(last.size() + egui::vec2(0.0, spec.height_slack));
        let target = user
            .min(bounds.size())
            .min(egui::vec2(spec.max_width, f32::INFINITY))
            .max(MIN_SIZE);
        let applied = data::<egui::Vec2>(ctx, spec.id, APPLIED_USER_SIZE);
        if applied.is_none_or(|a| a.x != target.x) {
            window = window.min_width(target.x).max_width(target.x);
        }
        if applied.is_none_or(|a| a.y != target.y) && !spec.fit_height {
            window = window.min_height(target.y).max_height(target.y);
        }
        if applied != Some(target) {
            set_data(ctx, spec.id, APPLIED_USER_SIZE, target);
        }
    }
    let mut close = false;
    let mut dock = false;
    let out = window.show(ctx, |ui| {
        scope_style(ui);
        let top = ui.max_rect().top();
        let max_bottom = top + size.y - chrome.y;
        if spec.fit_height {
            // Lay out as if the window had its largest height, so the measured height below
            // is the contents' own (the window follows next frame).
            ui.set_max_height(max_bottom - top);
        }
        ui.horizontal(|ui| {
            let title = ui.add(
                egui::Label::new(egui::RichText::new(spec.title).strong())
                    .selectable(false)
                    .sense(egui::Sense::click()),
            );
            // (egui counts a double-click within 0.6 s of an earlier click as a triple-click.)
            dock = floating && (title.double_clicked() || title.triple_clicked());
            if floating {
                title.on_hover_text("Double-click to dock");
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                close = close_button(ui).clicked();
                summary(ui);
            });
        });
        ui.separator();
        let r = contents(ui, max_bottom);
        if spec.fit_height {
            let height = (ui.min_rect().bottom() - top + chrome.y).round();
            if ctx.data(|d| d.get_temp::<f32>(fit_id)) != Some(height) {
                ctx.data_mut(|d| d.insert_temp(fit_id, height));
                ctx.request_repaint();
            }
        }
        r
    });
    if let Some(shown) = &out {
        let rect = shown.response.rect;
        let corner = spec.anchor.pos_in_rect(&rect);
        if !floating && pressed_on_window && !free_edge_press && corner.distance(spec.dock_pos) > 0.5 {
            set_flag(ctx, spec.id, FLOATING, true);
        } else if dock {
            set_flag(ctx, spec.id, FLOATING, false);
            ctx.request_repaint();
        }
        // Resized by the user by an edge the press grabbed (a height that follows the contents
        // does not count, nor what the other tools or the area squeeze meanwhile): the window
        // keeps that size from now on.
        if let Some(last) = last_rect {
            let resized = [
                grabbed[0] && (rect.width() - last.width()).abs() > 0.5,
                grabbed[1] && !spec.fit_height && (rect.height() - last.height()).abs() > 0.5,
            ];
            if resized.contains(&true) {
                let mut user = data::<egui::Vec2>(ctx, spec.id, USER_SIZE)
                    .filter(|_| user_sized)
                    .unwrap_or(last.size() + egui::vec2(0.0, spec.height_slack));
                if resized[0] {
                    user.x = rect.width();
                }
                if resized[1] {
                    user.y = rect.height() + spec.height_slack;
                }
                set_data(ctx, spec.id, USER_SIZE, user);
                set_flag(ctx, spec.id, USER_SIZED, true);
                remove_data::<egui::Vec2>(ctx, spec.id, APPLIED_USER_SIZE);
            }
        }
    }
    if close {
        *open = false;
    }
    out.and_then(|r| r.inner)
}

/// The edges of the window `rect` whose resize handle a press at `p` grabs (as egui places them:
/// a corner square, else a side strip, for the `resizable` [x, y] directions): for each axis,
/// [min edge, max edge].
fn grabbed_edges(
    interaction: &egui::style::Interaction,
    rect: egui::Rect,
    p: egui::Pos2,
    resizable: [bool; 2],
) -> [[bool; 2]; 2] {
    let (side, corner) = (
        interaction.resize_grab_radius_side,
        interaction.resize_grab_radius_corner,
    );
    // For each axis: which edge (0 = min, 1 = max) the press is within `radius` of.
    let near = |radius: f32| {
        [
            [
                (p.x - rect.left()).abs() <= radius,
                (p.x - rect.right()).abs() <= radius,
            ],
            [
                (p.y - rect.top()).abs() <= radius,
                (p.y - rect.bottom()).abs() <= radius,
            ],
        ]
    };
    let (in_corner, on_side) = (near(corner), near(side));
    let is_corner = in_corner[0].contains(&true) && in_corner[1].contains(&true);
    let mut grabbed = [[false; 2]; 2];
    for axis in 0..2 {
        let other = 1 - axis;
        // A side strip leaves the corner squares to the corners.
        let edges = if is_corner {
            in_corner[axis]
        } else if rect.expand(side).contains(p)
            && (rect.min[other] + corner..=rect.max[other] - corner).contains(&p[other])
        {
            on_side[axis]
        } else {
            [false; 2]
        };
        grabbed[axis] = edges.map(|hit| hit && resizable[axis]);
    }
    grabbed
}

/// Whether the `grabbed` edges ([`grabbed_edges`]) are some and all away from the window's
/// docked `anchor` corner.
fn only_free_edges(grabbed: [[bool; 2]; 2], anchor: egui::Align2) -> bool {
    let docked = [anchor.x() == egui::Align::Max, anchor.y() == egui::Align::Max].map(usize::from);
    grabbed.iter().flatten().any(|&g| g) && (0..2).all(|axis| !grabbed[axis][docked[axis]])
}

/// An 18 pt "×" button (drawn, so it needs no glyph from the game's fonts).
fn close_button(ui: &mut egui::Ui) -> egui::Response {
    let side = ui.spacing().interact_size.y;
    let (rect, response) = ui.allocate_exact_size(egui::vec2(side, side), egui::Sense::click());
    let visuals = ui.style().interact(&response);
    if response.hovered() {
        ui.painter().rect_filled(rect.shrink(1.0), 2.0, visuals.weak_bg_fill);
    }
    let r = rect.shrink((side - 8.0) / 2.0);
    let stroke = egui::Stroke::new(1.2, visuals.fg_stroke.color);
    ui.painter().line_segment([r.left_top(), r.right_bottom()], stroke);
    ui.painter().line_segment([r.right_top(), r.left_bottom()], stroke);
    response.on_hover_text("Close")
}

/// Case-insensitive `haystack.contains(needle_lower)` (`needle_lower` already lowercase),
/// without allocating for ASCII needles.
pub(crate) fn contains_ignore_case(haystack: &str, needle_lower: &str) -> bool {
    if needle_lower.is_empty() {
        return true;
    }
    if needle_lower.is_ascii() {
        let (h, n) = (haystack.as_bytes(), needle_lower.as_bytes());
        h.len() >= n.len() && h.windows(n.len()).any(|w| w.eq_ignore_ascii_case(n))
    } else {
        haystack.to_lowercase().contains(needle_lower)
    }
}

/// Write `n` with a k / M / G suffix (`950`, `12.3k`, `4.56M`).
pub(crate) fn write_si(out: &mut String, n: u64) {
    use std::fmt::Write as _;
    let _ = match n {
        n if n >= 1_000_000_000 => write!(out, "{:.2}G", n as f64 / 1e9),
        n if n >= 1_000_000 => write!(out, "{:.2}M", n as f64 / 1e6),
        n if n >= 10_000 => write!(out, "{:.1}k", n as f64 / 1e3),
        n => write!(out, "{n}"),
    };
}

/// Write a byte count in at most 6 characters, binary units (`512B`, `2.0K`, `598K`, `1.5M`,
/// `2.3G`).
pub(crate) fn write_short_bytes(out: &mut String, n: u64) {
    use std::fmt::Write as _;
    let n = n as f64;
    let (v, suffix) = match n {
        n if n >= (1u64 << 30) as f64 => (n / (1u64 << 30) as f64, "G"),
        n if n >= (1u64 << 20) as f64 => (n / (1u64 << 20) as f64, "M"),
        n if n >= 1024.0 => (n / 1024.0, "K"),
        n => {
            let _ = write!(out, "{n}B");
            return;
        }
    };
    let _ = if v < 10.0 {
        write!(out, "{v:.1}{suffix}")
    } else {
        write!(out, "{v:.0}{suffix}")
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    fn si(n: u64) -> String {
        let mut s = String::new();
        write_si(&mut s, n);
        s
    }

    fn short_bytes(n: u64) -> String {
        let mut s = String::new();
        write_short_bytes(&mut s, n);
        s
    }

    #[test]
    fn number_formats() {
        assert_eq!(si(950), "950");
        assert_eq!(si(9_999), "9999");
        assert_eq!(si(12_345), "12.3k");
        assert_eq!(si(4_560_000), "4.56M");
        assert_eq!(short_bytes(512), "512B");
        assert_eq!(short_bytes(2048), "2.0K");
        assert_eq!(short_bytes(612_400), "598K");
        assert_eq!(short_bytes(3 << 20), "3.0M");
        assert_eq!(short_bytes(1000 << 20), "1000M");
    }

    /// Weak text follows the palette: readable on egui's grey defaults, clearly dimmer than
    /// normal text under a bright game palette, never as strong as normal text.
    #[test]
    fn weak_text_alpha_follows_the_palette() {
        let dark = egui::Visuals::dark();
        let a = weak_text_alpha(&dark);
        let fill = dark.window_fill;
        assert!(a > dark.weak_text_alpha && a <= MAX_WEAK_ALPHA, "{a}");
        assert!(contrast(dark.text_color(), fill, a) >= 3.0, "readable on egui's greys");
        assert!(contrast(dark.text_color(), fill, a) < contrast(dark.text_color(), fill, 1.0));

        // A game's bright text on a dark translucent fill: egui's 0.6 already reads well above
        // 4.5:1, so weak text stays at 0.6, well apart from normal text.
        let mut game = egui::Visuals::dark();
        game.override_text_color = Some(egui::Color32::from_rgb(245, 244, 239));
        game.window_fill = egui::Color32::from_rgba_unmultiplied(24, 23, 21, 235);
        let a = weak_text_alpha(&game);
        assert_eq!(a, game.weak_text_alpha);
        let [r, g, b, _] = game.window_fill.to_srgba_unmultiplied();
        let fill = egui::Color32::from_rgb(r, g, b);
        let (weak, normal) = (
            contrast(game.text_color(), fill, a),
            contrast(game.text_color(), fill, 1.0),
        );
        assert!(weak >= 4.5 && weak < 0.7 * normal, "{weak} vs {normal}");

        // Light theme and an explicit weak color.
        let light = egui::Visuals::light();
        let a = weak_text_alpha(&light);
        assert!(contrast(light.text_color(), light.window_fill, a) >= 3.0);
        let mut explicit = egui::Visuals::dark();
        explicit.weak_text_color = Some(egui::Color32::GRAY);
        assert_eq!(weak_text_alpha(&explicit), explicit.weak_text_alpha);
    }

    #[test]
    fn case_insensitive_filter() {
        assert!(contains_ignore_case("Shader CACHE corrupt", "cache"));
        assert!(contains_ignore_case("anything", ""));
        assert!(!contains_ignore_case("abc", "abcd"));
        assert!(contains_ignore_case("ÄRGER über", "über"));
        assert!(!contains_ignore_case("render", "physics"));
    }
}
