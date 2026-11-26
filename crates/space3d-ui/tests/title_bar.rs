//! A window with content under its title bar (`space3d_platform::TitleBar`): what of the strip
//! the UI claims, and the window buttons it paints. Headless: no window; the title bar is the
//! one a Windows window would have (`TitleBar::simulated_windows`). GPU tests skip (return
//! early) when no adapter is available.

use space3d_core::prelude::*;
use space3d_platform::{TitleBar, TitleBarRect};
use space3d_render::*;
use space3d_ui::*;

const W: u32 = 800;
const H: u32 = 600;

/// What the UI systems drew where, and what they show.
#[derive(Default)]
struct Drawn {
    button: Option<egui::Rect>,
    low_button: Option<egui::Rect>,
    show_button: bool,
    show_window: bool,
    show_modal: bool,
}

/// A page: a button in the strip, text in the strip, a button below it and a click catcher over
/// the whole screen behind them all.
fn page(ctx: &egui::Context, sys: &mut SystemCtx) {
    let drawn = sys.resources.init::<Drawn>();
    egui::Area::new(egui::Id::new("catcher"))
        .order(egui::Order::Background)
        .fixed_pos(egui::Pos2::ZERO)
        .show(ctx, |ui| {
            let screen = ui.ctx().viewport_rect();
            ui.allocate_exact_size(screen.size(), egui::Sense::click_and_drag());
        });
    if drawn.show_button {
        egui::Area::new(egui::Id::new("menu"))
            .fixed_pos([100.0, 4.0])
            .show(ctx, |ui| drawn.button = Some(ui.button("Menu").rect));
    }
    egui::Area::new(egui::Id::new("label"))
        .fixed_pos([300.0, 6.0])
        .interactable(false)
        .show(ctx, |ui| ui.add(egui::Label::new("A title").selectable(false)));
    egui::Area::new(egui::Id::new("low"))
        .fixed_pos([100.0, 100.0])
        .show(ctx, |ui| drawn.low_button = Some(ui.button("Below").rect));
    if drawn.show_window {
        egui::Window::new("Floating")
            .fixed_pos([420.0, 0.0])
            .fixed_size([100.0, 80.0])
            .show(ctx, |ui| ui.label("body"));
    }
    if drawn.show_modal {
        egui::Modal::new(egui::Id::new("sheet")).show(ctx, |ui| ui.label("A sheet"));
    }
}

fn app_with(bar: TitleBar, plugin: UiPlugin) -> App {
    let mut app = App::new();
    app.add_plugin(UiPlugin {
        fallback_size: (W, H),
        ..plugin
    });
    app.insert_resource(bar);
    app.insert_resource(Drawn {
        show_button: true,
        ..Drawn::default()
    });
    app.add_ui_system(page);
    app
}

fn app() -> App {
    app_with(TitleBar::simulated_windows(W, 1.0, false), UiPlugin::deterministic())
}

fn claims(app: &App) -> Vec<TitleBarRect> {
    app.resources.expect::<TitleBar>().claims().collect()
}

fn claimed(app: &App, x: i32, y: i32) -> bool {
    claims(app).iter().any(|c| c.contains(x, y))
}

fn centre(r: egui::Rect) -> (i32, i32) {
    (r.center().x as i32, r.center().y as i32)
}

#[test]
fn a_widget_in_the_strip_keeps_its_clicks_and_the_rest_of_the_strip_drags() {
    let mut app = app();
    app.step_ticks(3);
    let button = app.resources.expect::<Drawn>().button.expect("drawn");
    assert!(button.top() < 32.0, "the button is in the strip: {button:?}");
    let (x, y) = centre(button);
    assert!(claimed(&app, x, y), "{:?}", claims(&app));
    assert!(!claimed(&app, 330, 14), "text that takes no clicks");
    assert!(!claimed(&app, 700, 14), "nothing but the full-screen click catcher");
    assert!(!claimed(&app, 40, 14));
    // Only what is in the strip is claimed, cut to it.
    assert!(
        claims(&app).iter().all(|c| c.y >= 0 && c.bottom() <= 32),
        "{:?}",
        claims(&app)
    );
    let low = app.resources.expect::<Drawn>().low_button.expect("drawn");
    assert!(!claimed(&app, low.center().x as i32, low.center().y as i32));
}

#[test]
fn claims_go_when_the_widget_goes() {
    let mut app = app();
    app.step_ticks(3);
    let (x, y) = centre(app.resources.expect::<Drawn>().button.expect("drawn"));
    assert!(claimed(&app, x, y));
    app.resources.expect_mut::<Drawn>().show_button = false;
    app.step_ticks(2);
    assert!(!claimed(&app, x, y), "{:?}", claims(&app));
}

#[test]
fn a_floating_window_in_the_strip_takes_its_own_clicks() {
    let mut app = app();
    app.resources.expect_mut::<Drawn>().show_window = true;
    app.step_ticks(3);
    // Its title bar and body: egui drags the window by them.
    assert!(claimed(&app, 470, 10), "{:?}", claims(&app));
    assert!(!claimed(&app, 650, 10));
}

#[test]
fn a_modals_backdrop_does_not_stop_the_window_being_dragged() {
    let mut app = app();
    app.resources.expect_mut::<Drawn>().show_modal = true;
    app.step_ticks(3);
    assert!(!claimed(&app, 700, 14), "{:?}", claims(&app));
}

#[test]
fn claims_are_in_physical_pixels_whatever_the_uis_zoom() {
    // A window at 150 % whose UI is drawn at 2 pixels a point.
    let mut app = app_with(
        TitleBar::simulated_windows(W, 1.5, false),
        UiPlugin::deterministic().with_pixels_per_point(2.0),
    );
    app.step_ticks(3);
    let button = app.resources.expect::<Drawn>().button.expect("drawn");
    let (x, y) = ((button.center().x * 2.0) as i32, (button.center().y * 2.0) as i32);
    assert!(claimed(&app, x, y), "{button:?} {:?}", claims(&app));
    assert!(claims(&app).iter().all(|c| c.bottom() <= 48), "{:?}", claims(&app));
}

#[test]
fn nothing_is_claimed_without_a_title_bar_under_the_picture() {
    let mut app = app_with(TitleBar::default(), UiPlugin::deterministic());
    app.step_ticks(3);
    assert!(claims(&app).is_empty());
}

/// The page over a plain background, rendered, or `None` without a GPU.
fn capture(bar: TitleBar) -> Option<ImageData> {
    let mut app = App::new();
    app.add_plugin(RenderPlugin::default());
    match pollster::block_on(Renderer::new_headless(W, H)) {
        Ok(r) => {
            app.insert_resource(r);
        }
        Err(e) => {
            eprintln!("skipping GPU test: no adapter available ({e})");
            return None;
        }
    }
    app.add_plugin(UiPlugin::deterministic());
    app.insert_resource(bar);
    app.world.spawn((
        Transform::from_xyz(0.0, 0.0, 3.0),
        Camera3d::default().with_clear_color(Vec4::new(0.0, 0.0, 1.0, 1.0)),
    ));
    app.step_ticks(3);
    render_to_image(&mut app)
}

/// How many pixels of `rect` differ between the two pictures.
fn changed(a: &ImageData, b: &ImageData, rect: TitleBarRect) -> usize {
    let mut n = 0;
    for y in rect.y..rect.bottom() {
        for x in rect.x..rect.right() {
            if a.get(x as u32, y as u32) != b.get(x as u32, y as u32) {
                n += 1;
            }
        }
    }
    n
}

#[test]
fn the_window_buttons_are_painted_where_the_platform_says_they_are() {
    let bar = TitleBar::simulated_windows(W, 1.0, false);
    let own = {
        let mut bar = bar.clone();
        bar.draw_buttons = false;
        bar
    };
    let (Some(with), Some(without)) = (capture(bar.clone()), capture(own)) else {
        return;
    };
    for button in bar.buttons.expect("Windows has them") {
        let pixels = changed(&with, &without, button.rect);
        assert!(pixels > 8, "{:?}: {pixels} pixels of its sign", button.control);
    }
    // And nothing anywhere else: the picture is the app's up to its edges.
    let controls = bar.controls.expect("covered");
    let everything = changed(&with, &without, TitleBarRect::new(0, 0, W as i32, H as i32));
    assert_eq!(everything, changed(&with, &without, controls));
}

#[test]
fn a_button_under_the_pointer_lights_up_and_close_turns_red() {
    let idle = TitleBar::simulated_windows(W, 1.0, false);
    let hovered = {
        let mut bar = idle.clone();
        let buttons = bar.buttons.as_mut().expect("Windows has them");
        buttons[2].hovered = true;
        bar
    };
    let (Some(a), Some(b)) = (capture(idle.clone()), capture(hovered)) else {
        return;
    };
    let close = idle.buttons.expect("Windows has them")[2].rect;
    let corner = b.get((close.x + 3) as u32, 3);
    assert!(
        corner[0] > 150 && corner[2] < 80,
        "red behind the close button: {corner:?}"
    );
    assert_eq!(changed(&a, &b, TitleBarRect::new(0, 0, close.x, 32)), 0);
}
