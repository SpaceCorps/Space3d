//! A window with content under its title bar (`space3d_platform::WindowPlugin::content_under_title_bar`,
//! `space3d_platform::TitleBar`): after a pass's UI, what of the strip is the UI's, and the
//! window buttons over everything.
//!
//! **Claims.** Every widget that senses a click or a drag and reaches into the strip takes its
//! own clicks there: buttons, sliders, text fields, floating windows and areas (egui's own move
//! response covers them), disabled ones too. A widget that covers (nearly) the whole screen is
//! no control of the strip but a backdrop (a modal's, a click catcher under a page): the strip
//! over it drags the window.
//!
//! **Buttons** (Windows: `TitleBar::buttons`, unless the app set `TitleBar::draw_buttons` to
//! `false`) are painted on egui's topmost layer, in the system's proportions: glyphs of 10
//! logical pixels, a faint outline in the opposite shade under them so that they read over any
//! picture, the close button red under the pointer, everything dimmed while the window does not
//! have the focus. They are paint only: the system, not egui, answers their clicks. With the UI
//! disabled nothing is painted.

use egui::{Color32, LayerId, Pos2, Rect, Shape, Stroke, StrokeKind, pos2, vec2};
use space3d_platform::{TitleBarRect, WindowButton, WindowControl};

/// A widget that covers this share of the screen in both directions is a backdrop.
const BACKDROP: f32 = 0.9;

/// The title bar's state between two passes of the UI.
#[derive(Default)]
pub(crate) struct TitleBarPass {
    /// Widgets that reach into the strip, in their layers' own coordinates (kept to allocate
    /// nothing from pass to pass).
    widgets: Vec<(LayerId, Rect)>,
    /// What is claimed, in physical pixels (swapped into the resource).
    pub claims: Vec<TitleBarRect>,
}

impl TitleBarPass {
    /// Fills [`Self::claims`] from the widgets of the pass under way; `height` is the strip's in
    /// physical pixels.
    pub fn collect(&mut self, ctx: &egui::Context, height: u32, pixels_per_point: f32) {
        self.claims.clear();
        self.widgets.clear();
        let screen = ctx.viewport_rect();
        let found = &mut self.widgets;
        ctx.viewport(|viewport| {
            for (layer, widgets) in viewport.this_pass.widgets.layers() {
                found.extend(
                    widgets
                        .iter()
                        .filter(|w| w.sense.interactive() && w.interact_rect.is_positive())
                        .map(|w| (*layer, w.interact_rect)),
                );
            }
        });
        let strip_bottom = height as f32 / pixels_per_point;
        for &(layer, rect) in &self.widgets {
            // A layer with a transform (a zoomed or moved area) says where it is on screen.
            let rect = ctx
                .layer_transform_to_global(layer)
                .map_or(rect, |to_global| to_global * rect);
            if rect.top() >= strip_bottom || rect.bottom() <= 0.0 {
                continue;
            }
            if rect.width() >= screen.width() * BACKDROP && rect.height() >= screen.height() * BACKDROP {
                continue;
            }
            let claim = TitleBarRect::covering(
                space3d_core::glam::Vec2::new(rect.left(), rect.top().max(0.0)) * pixels_per_point,
                space3d_core::glam::Vec2::new(rect.right(), rect.bottom().min(strip_bottom)) * pixels_per_point,
            );
            if !claim.is_empty() {
                self.claims.push(claim);
            }
        }
    }
}

/// Paints the window buttons on the topmost layer.
pub(crate) fn paint_buttons(
    ctx: &egui::Context,
    buttons: &[WindowButton; 3],
    maximized: bool,
    focused: bool,
    pixels_per_point: f32,
) {
    let painter = ctx.layer_painter(LayerId::new(
        egui::Order::Debug,
        egui::Id::new("space3d_window_buttons"),
    ));
    let dark = ctx.global_style().visuals.dark_mode;
    let (ink, paper) = if dark {
        (Color32::WHITE, Color32::BLACK)
    } else {
        (Color32::BLACK, Color32::WHITE)
    };
    for button in buttons {
        let r = button.rect;
        let rect = Rect::from_min_max(
            pos2(r.x as f32, r.y as f32) / pixels_per_point,
            pos2(r.right() as f32, r.bottom() as f32) / pixels_per_point,
        );
        let close = button.control == WindowControl::Close;
        let fill = match (close, button.pressed, button.hovered) {
            (true, true, _) => Some(Color32::from_rgb(0xB2, 0x2A, 0x1C)),
            (true, false, true) => Some(Color32::from_rgb(0xC4, 0x2B, 0x1C)),
            (false, true, _) => Some(ink.gamma_multiply(0.08)),
            (false, false, true) => Some(ink.gamma_multiply(0.13)),
            _ => None,
        };
        if let Some(fill) = fill {
            painter.rect_filled(rect, 0.0, fill);
        }
        let red = close && fill.is_some();
        let strength = if !button.enabled {
            0.35
        } else if !focused && fill.is_none() {
            0.5
        } else {
            1.0
        };
        let color = if red { Color32::WHITE } else { ink }.gamma_multiply(strength);
        // A logical pixel of the system's, in points: the buttons are 32 high.
        let unit = rect.height() / 32.0;
        // The centre on the middle of a physical pixel, so that straight lines stay sharp.
        let snap = |v: f32| ((v * pixels_per_point).floor() + 0.5) / pixels_per_point;
        let center = pos2(snap(rect.center().x), snap(rect.center().y));
        if fill.is_none() {
            let halo = paper.gamma_multiply(0.35 * strength);
            glyph(
                &painter,
                center,
                unit,
                button.control,
                maximized,
                Stroke::new(3.0 * unit, halo),
            );
        }
        glyph(
            &painter,
            center,
            unit,
            button.control,
            maximized,
            Stroke::new(unit, color),
        );
    }
}

/// A window button's sign, 10 units square round `center`.
fn glyph(painter: &egui::Painter, center: Pos2, unit: f32, control: WindowControl, maximized: bool, stroke: Stroke) {
    let at = |x: f32, y: f32| center + vec2(x, y) * unit;
    let square = |min: Pos2, max: Pos2| {
        painter.rect_stroke(Rect::from_min_max(min, max), 1.5 * unit, stroke, StrokeKind::Middle);
    };
    match control {
        WindowControl::Minimize => {
            painter.line_segment([at(-5.0, 0.0), at(5.0, 0.0)], stroke);
        }
        WindowControl::Maximize if maximized => {
            // Two windows: the front one, and what shows of the one behind it.
            square(at(-5.0, -3.0), at(3.0, 5.0));
            painter.add(Shape::line(
                vec![
                    at(-3.0, -3.0),
                    at(-3.0, -5.0),
                    at(5.0, -5.0),
                    at(5.0, 3.0),
                    at(3.0, 3.0),
                ],
                stroke,
            ));
        }
        WindowControl::Maximize => square(at(-5.0, -5.0), at(5.0, 5.0)),
        WindowControl::Close => {
            painter.line_segment([at(-5.0, -5.0), at(5.0, 5.0)], stroke);
            painter.line_segment([at(-5.0, 5.0), at(5.0, -5.0)], stroke);
        }
    }
}
