//! Headless editor + storybook viewer tests. GPU tests skip without an adapter.

use space3d_core::prelude::*;
use space3d_editor::*;
use space3d_platform::{Input, KeyCode, MouseButton};
use space3d_render::*;
use space3d_stories::{KnobValue, Story, Storybook};
use space3d_ui::{EguiContext, UiDensity, UiTheme, egui};
use std::time::Duration;

/// The platform's "command" modifier key (Cmd on macOS, Ctrl elsewhere).
const COMMAND: KeyCode = if cfg!(any(target_os = "macos", target_os = "ios")) {
    KeyCode::SuperLeft
} else {
    KeyCode::ControlLeft
};

/// Press `COMMAND` + `key` for one frame.
fn shortcut(app: &mut App, key: KeyCode) {
    let input = app.resources.expect_mut::<Input>();
    input.inject_key_down(COMMAND);
    input.inject_key_tap(key);
    app.step_ticks(1);
    app.resources.expect_mut::<Input>().inject_key_up(COMMAND);
    app.step_ticks(1);
}

/// Run frames until `done` holds (background I/O and GPU readbacks finish asynchronously).
fn pump_until(app: &mut App, what: &str, mut done: impl FnMut(&mut App) -> bool) {
    for _ in 0..600 {
        if done(app) {
            return;
        }
        app.update(Duration::from_millis(16));
        std::thread::sleep(Duration::from_millis(2));
    }
    panic!("timed out waiting for {what}");
}

fn entity_named(app: &App, name: &str) -> Option<Entity> {
    app.world
        .query::<(Entity, &Name)>()
        .iter()
        .find(|(_, n)| n.0 == name)
        .map(|(e, _)| e)
}

/// Click the left mouse button at physical position `p` (one frame per edge).
fn click_at(app: &mut App, p: Vec2) {
    app.resources.expect_mut::<Input>().inject_mouse_move(p);
    app.step_ticks(1);
    app.resources
        .expect_mut::<Input>()
        .inject_mouse_button_down(MouseButton::Left);
    app.step_ticks(1);
    app.resources
        .expect_mut::<Input>()
        .inject_mouse_button_up(MouseButton::Left);
    app.step_ticks(1);
}

#[derive(Default)]
struct FixedTicks(u64);

fn scene(app: &mut App) {
    app.world.spawn((
        Name("Main camera".into()),
        Transform::from_xyz(0.0, 0.0, 6.0).looking_at(Vec3::ZERO, Vec3::Y),
        Camera3d::default().with_clear_color(Vec4::new(0.0, 0.0, 1.0, 1.0)),
    ));
    app.world.spawn((
        Name("Cube".into()),
        Transform::IDENTITY,
        MeshSource::Cube { size: 1.5 },
        Material::unlit(Vec4::new(1.0, 1.0, 0.0, 1.0)),
    ));
    app.init_resource::<FixedTicks>();
    app.add_system(Stage::FixedUpdate, |c| c.resources.expect_mut::<FixedTicks>().0 += 1);
}

fn headless_editor(w: u32, h: u32) -> Option<App> {
    let config = EditorConfig {
        headless: Some((w, h)),
        ..EditorConfig::new("test")
    }
    .with_setup(scene);
    let app = editor_app(&config);
    if !app.resources.contains::<Renderer>() {
        eprintln!("skipping GPU test: no adapter available");
        return None;
    }
    Some(app)
}

/// Write `img` to `$SPACE3D_EDITOR_SNAPSHOT_DIR/<name>.png` when the variable is set (for
/// looking at the UI by eye).
fn maybe_save(name: &str, img: &ImageData) {
    if let Ok(dir) = std::env::var("SPACE3D_EDITOR_SNAPSHOT_DIR") {
        let png = space3d_assets::edit::texture::encode_png(img).unwrap();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(format!("{dir}/{name}.png"), png).unwrap();
    }
}

fn count(img: &ImageData, x0: u32, y0: u32, x1: u32, y1: u32, pred: impl Fn([u8; 4]) -> bool) -> usize {
    let mut n = 0;
    for y in y0..y1.min(img.height) {
        for x in x0..x1.min(img.width) {
            if pred(img.get(x, y)) {
                n += 1;
            }
        }
    }
    n
}

#[test]
fn editor_draws_panels_over_the_scene_and_pauses_simulation() {
    let Some(mut app) = headless_editor(900, 520) else {
        return;
    };
    app.step_ticks(4);
    assert_eq!(app.resources.expect::<FixedTicks>().0, 0, "editor starts paused");
    assert_eq!(app.resources.expect::<Time>().frame, 4);
    let img = render_to_image(&mut app).expect("capture");
    maybe_save("editor", &img);
    let blue = |p: [u8; 4]| p == [0, 0, 255, 255];
    let yellow = |p: [u8; 4]| p[0] > 240 && p[1] > 240 && p[2] < 20;
    // Hierarchy panel on the left covers the blue clear color; the cube shows in the middle.
    assert_eq!(count(&img, 5, 60, 200, 400, blue), 0, "left panel drawn");
    assert!(
        count(&img, 350, 150, 550, 370, yellow) > 1000,
        "cube visible in the viewport"
    );
    assert!(count(&img, 250, 60, 600, 500, blue) > 1000, "viewport shows the scene");

    // Step exactly 5 ticks, then play.
    app.resources.expect_mut::<SimControl>().step(5);
    app.step_ticks(3);
    assert_eq!(app.resources.expect::<FixedTicks>().0, 5);
    app.resources.expect_mut::<SimControl>().play();
    app.step_ticks(3);
    assert_eq!(app.resources.expect::<FixedTicks>().0, 8);
}

#[test]
fn clicking_the_cube_selects_it_and_undo_works() {
    let Some(mut app) = headless_editor(900, 520) else {
        return;
    };
    app.step_ticks(3);
    let cube = app
        .world
        .query::<(Entity, &Name)>()
        .iter()
        .find(|(_, n)| n.0 == "Cube")
        .map(|(e, _)| e)
        .unwrap();
    // Project the cube center to the screen and click it.
    let cam = space3d_editor::viewport::primary_camera(&app.world, 900.0 / 520.0).unwrap();
    let ppp = app.resources.expect::<EguiContext>().pixels_per_point();
    let screen = Vec2::new(900.0, 520.0) / ppp;
    let p = cam.project(Vec3::ZERO, screen).unwrap() * ppp;
    app.resources.expect_mut::<Input>().inject_mouse_move(p);
    app.step_ticks(1);
    app.resources
        .expect_mut::<Input>()
        .inject_mouse_button_down(MouseButton::Left);
    app.step_ticks(1);
    app.resources
        .expect_mut::<Input>()
        .inject_mouse_button_up(MouseButton::Left);
    app.step_ticks(1);
    assert_eq!(app.resources.expect::<EditorState>().selected, Some(cube));

    // Edit through the history and undo.
    let id = components::ensure_stable_id(&mut app.world, cube).unwrap();
    let registry = app.resources.remove::<space3d_scene::ComponentRegistry>().unwrap();
    let mut history = app.resources.remove::<EditorHistory>().unwrap();
    history.execute(
        EditCommand::set(
            "Move",
            id,
            Some(Transform::IDENTITY),
            Some(Transform::from_xyz(2.0, 0.0, 0.0)),
        ),
        &mut app.world,
        &registry,
    );
    assert_eq!(app.world.get::<&Transform>(cube).unwrap().translation.x, 2.0);
    history.undo(&mut app.world, &registry);
    assert_eq!(app.world.get::<&Transform>(cube).unwrap().translation.x, 0.0);
    app.insert_resource(registry);
    app.insert_resource(history);

    // Delete via the editor state + keyboard shortcut path is covered by unit tests; here the
    // selection survives more frames.
    app.step_ticks(2);
    assert_eq!(app.resources.expect::<EditorState>().selected, Some(cube));
}

#[test]
fn model_editor_preview_renders_into_the_ui() {
    let Some(mut app) = headless_editor(1200, 700) else {
        return;
    };
    {
        let model = app.resources.expect_mut::<ModelEditor>();
        model.open = true;
        model.wireframe = false;
        model.color = [1.0, 0.0, 1.0, 1.0];
    }
    app.step_ticks(12);
    assert!(app.resources.expect::<ModelEditor>().preview_texture().is_some());
    let img = render_to_image(&mut app).expect("capture");
    maybe_save("model_editor", &img);
    // Only the offscreen preview (render target → egui image) can show the magenta model.
    let magenta = count(&img, 0, 0, 1200, 700, |p| {
        let (r, g, b) = (p[0] as i32, p[1] as i32, p[2] as i32);
        r > 60 && b > 60 && g * 3 < r && (r - b).abs() < 40
    });
    assert!(magenta > 1500, "model visible in the preview image ({magenta} px)");
    // Selected faces are highlighted (orange) over the preview.
    app.resources.expect_mut::<ModelEditor>().select_facing(Axis::PosY);
    app.step_ticks(2);
    let img = render_to_image(&mut app).expect("capture");
    maybe_save("model_editor_selection", &img);
    let orange = |p: [u8; 4]| p[0] > 180 && (45..120).contains(&p[1]) && p[2] < 170;
    let highlighted = count(&img, 0, 0, 1200, 700, orange);
    assert!(highlighted > 300, "selection highlight drawn ({highlighted} px)");
    // Closing destroys the preview entities.
    let before = app.world.len();
    app.resources.expect_mut::<ModelEditor>().open = false;
    app.step_ticks(1);
    assert_eq!(app.world.len(), before - 3);
}

#[test]
fn texture_editor_canvas_shows_the_image() {
    let Some(mut app) = headless_editor(1000, 700) else {
        return;
    };
    {
        let tex = app.resources.expect_mut::<TextureEditor>();
        tex.open = true;
        tex.open_image(ImageData::filled(64, 64, [0, 255, 0, 255]));
    }
    app.step_ticks(12);
    let img = render_to_image(&mut app).expect("capture");
    maybe_save("texture_editor", &img);
    let green = count(&img, 0, 0, 1000, 700, |p| p[1] > 240 && p[0] < 20 && p[2] < 20);
    assert!(green > 20_000, "image drawn on the canvas ({green} px)");
    // A background operation goes through the UI loop and lands as one undo step.
    app.resources
        .expect_mut::<TextureEditor>()
        .apply_async("Invert", space3d_assets::edit::texture::invert);
    pump_until(&mut app, "invert", |app| {
        !app.resources.expect::<TextureEditor>().is_busy()
    });
    let tex = app.resources.expect_mut::<TextureEditor>();
    assert_eq!(tex.image().unwrap().get(0, 0), [255, 0, 255, 255]);
    assert!(tex.undo());
    assert_eq!(tex.image().unwrap().get(0, 0), [0, 255, 0, 255]);
}

#[test]
fn gizmo_drag_moves_the_selection_and_cmd_z_undoes_it() {
    let Some(mut app) = headless_editor(900, 520) else {
        return;
    };
    app.step_ticks(3);
    let cube = entity_named(&app, "Cube").unwrap();
    let cam = space3d_editor::viewport::primary_camera(&app.world, 900.0 / 520.0).unwrap();
    let ppp = app.resources.expect::<EguiContext>().pixels_per_point();
    let screen = Vec2::new(900.0, 520.0) / ppp;
    let center = cam.project(Vec3::ZERO, screen).unwrap();
    click_at(&mut app, center * ppp);
    assert_eq!(app.resources.expect::<EditorState>().selected, Some(cube));
    assert_eq!(
        app.resources.expect::<EditorHistory>().undo_len(),
        0,
        "showing the inspector records nothing"
    );

    // Same geometry as the gizmo: the +X arrow is 15% of the camera distance long.
    let len = (Vec3::ZERO - cam.eye).length() * 0.15;
    let tip = cam.project(Vec3::X * len, screen).unwrap();
    let grab = center + (tip - center) * 0.8;
    let along = (tip - center).normalize();
    app.resources.expect_mut::<Input>().inject_mouse_move(grab * ppp);
    app.step_ticks(1);
    app.resources
        .expect_mut::<Input>()
        .inject_mouse_button_down(MouseButton::Left);
    app.step_ticks(1);
    for i in 1..=4 {
        let p = grab + along * (10.0 * i as f32);
        app.resources.expect_mut::<Input>().inject_mouse_move(p * ppp);
        app.step_ticks(1);
    }
    app.resources
        .expect_mut::<Input>()
        .inject_mouse_button_up(MouseButton::Left);
    app.step_ticks(1);
    let moved = *app.world.get::<&Transform>(cube).unwrap();
    assert!(moved.translation.x > 0.2, "dragged along +X: {:?}", moved.translation);
    assert!(moved.translation.y.abs() < 1e-4 && moved.translation.z.abs() < 1e-4);
    assert_eq!(
        app.resources.expect::<EditorState>().selected,
        Some(cube),
        "drag keeps selection"
    );
    let history = app.resources.expect::<EditorHistory>();
    assert_eq!(history.undo_len(), 1, "one undo step for the whole drag");
    assert_eq!(history.undo_label(), Some("Move"));

    shortcut(&mut app, KeyCode::KeyZ);
    assert_eq!(app.world.get::<&Transform>(cube).unwrap().translation, Vec3::ZERO);
    assert_eq!(app.resources.expect::<EditorState>().status(), "undo: Move");
    // Cmd+Shift+Z redoes.
    let input = app.resources.expect_mut::<Input>();
    input.inject_key_down(COMMAND);
    input.inject_key_down(KeyCode::ShiftLeft);
    input.inject_key_tap(KeyCode::KeyZ);
    app.step_ticks(1);
    let input = app.resources.expect_mut::<Input>();
    input.inject_key_up(COMMAND);
    input.inject_key_up(KeyCode::ShiftLeft);
    app.step_ticks(1);
    assert_eq!(
        app.world.get::<&Transform>(cube).unwrap().translation,
        moved.translation
    );
}

#[test]
fn save_and_open_through_the_editor_shortcuts() {
    let Some(mut app) = headless_editor(640, 400) else {
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("level.ron");
    app.resources.expect_mut::<EditorState>().scene_path = path.to_string_lossy().into_owned();
    app.step_ticks(2);
    let cube = entity_named(&app, "Cube").unwrap();
    // A game mesh without MeshSource survives a reload in the same session.
    let game_mesh = app
        .resources
        .expect_mut::<RenderAssets>()
        .add_mesh(MeshData::sphere(0.5, 8, 6));
    app.world
        .spawn((Name("Game mesh".into()), Transform::IDENTITY, Mesh3d(game_mesh)));
    app.step_ticks(1);

    shortcut(&mut app, KeyCode::KeyS);
    pump_until(&mut app, "save", |app| {
        !app.resources.expect::<EditorState>().io_pending()
    });
    let status = app.resources.expect::<EditorState>().status().to_owned();
    assert!(status.starts_with("saved"), "{status}");
    assert!(
        status.contains("1 entities"),
        "warns about the session-only mesh: {status}"
    );
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.contains("Cube") && text.contains("MeshSource"));

    // Change the world, then reopen the file.
    app.world.despawn(cube).unwrap();
    app.world.spawn((Name("Extra".into()), Transform::IDENTITY));
    app.resources
        .expect_mut::<EditorHistory>()
        .push(EditCommand::new("dummy", |_, _| {}, |_, _| {}));
    app.step_ticks(1);
    shortcut(&mut app, KeyCode::KeyO);
    pump_until(&mut app, "open", |app| {
        !app.resources.expect::<EditorState>().io_pending()
    });
    let status = app.resources.expect::<EditorState>().status().to_owned();
    assert!(status.starts_with("loaded"), "{status}");
    app.step_ticks(1);
    assert!(entity_named(&app, "Extra").is_none(), "scene replaced");
    let cube = entity_named(&app, "Cube").expect("cube restored");
    assert!(app.world.satisfies::<&Mesh3d>(cube), "mesh rebuilt from MeshSource");
    let game = entity_named(&app, "Game mesh").unwrap();
    assert_eq!(app.world.get::<&Mesh3d>(game).unwrap().0, game_mesh);
    assert!(entity_named(&app, "Main camera").is_some());
    assert_eq!(app.resources.expect::<EditorHistory>().undo_len(), 0, "history cleared");
    assert!(app.resources.expect::<SimControl>().is_paused());
}

fn demo_book() -> Storybook {
    // (story "Demo/Red": camera clearing to the "clear" knob color)
    let mut b = Storybook::new();
    b.add(
        Story::new("Demo/Red", |app, k| {
            let c = k.color("clear");
            app.world.spawn((
                Transform::from_xyz(0.0, 0.0, 3.0),
                Camera3d::default().with_clear_color(Vec4::from(c)),
            ));
        })
        .knob_color("clear", [1.0, 0.0, 0.0, 1.0])
        .size(96, 64)
        .ticks(2),
    );
    b
}

#[test]
fn storybook_viewer_streams_the_story_into_its_viewport() {
    let shots = tempfile::tempdir().unwrap();
    let config = StorybookConfig {
        headless: Some((900, 500)),
        screenshot_dir: shots.path().to_string_lossy().into_owned(),
        ..Default::default()
    };
    let mut app = storybook_app(demo_book(), config);
    if !app.resources.contains::<Renderer>() {
        eprintln!("skipping GPU test: no adapter available");
        return;
    }
    // The story renderer is created and read back asynchronously: pump frames (bounded).
    let mut frames = 0;
    for _ in 0..400 {
        app.update(Duration::from_millis(16));
        frames = app
            .resources
            .expect::<StorybookViewer>()
            .view()
            .map_or(0, LiveView::frames);
        if frames >= 2 {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(frames >= 2, "story frames reached the viewer");
    let viewer = app.resources.expect::<StorybookViewer>();
    assert_eq!(viewer.selected().unwrap().id, "Demo/Red");
    let frame = viewer.view().unwrap().image().unwrap();
    assert_eq!((frame.width, frame.height), (96, 64));
    assert_eq!(frame.get(10, 10), [255, 0, 0, 255]);
    let img = render_to_image(&mut app).expect("capture");
    maybe_save("storybook", &img);
    let red = count(&img, 0, 0, 900, 500, |p| p[0] > 240 && p[1] < 20 && p[2] < 20);
    assert!(red > 1000, "story frame shown in the host UI ({red} red pixels)");

    // "Capture screenshot" writes the latest story frame as PNG.
    let path = app
        .resources
        .expect_mut::<StorybookViewer>()
        .capture_screenshot()
        .unwrap();
    pump_until(&mut app, "screenshot", |app| {
        !app.resources.expect::<StorybookViewer>().io_pending()
    });
    let status = app.resources.expect::<StorybookViewer>().status().to_owned();
    assert!(status.starts_with("saved"), "{status}");
    let png = std::fs::read(&path).unwrap();
    let shot = space3d_assets::edit::texture::decode_image_with_hint(&png, Some("png")).unwrap();
    assert_eq!((shot.width, shot.height), (96, 64));
    assert_eq!(shot.get(50, 30), [255, 0, 0, 255]);

    // Changing a knob rebuilds the story on its own (no explicit rebuild call).
    app.resources
        .expect_mut::<StorybookViewer>()
        .set_knob("clear", KnobValue::Color([0.0, 1.0, 0.0, 1.0]));
    pump_until(&mut app, "rebuilt green story", |app| {
        app.resources
            .expect::<StorybookViewer>()
            .view()
            .and_then(LiveView::image)
            .is_some_and(|img| img.get(10, 10) == [0, 255, 0, 255])
    });

    // "Step N" runs over frames without blocking and ends exactly on the requested tick.
    let viewer = app.resources.expect_mut::<StorybookViewer>();
    let start = viewer.story_tick().unwrap();
    viewer.step(500);
    pump_until(&mut app, "500 steps", |app| {
        app.resources.expect::<StorybookViewer>().pending_steps() == 0
    });
    assert_eq!(
        app.resources.expect::<StorybookViewer>().story_tick(),
        Some(start + 500)
    );
}

/// A book whose story has one knob of every kind (the viewer's side panel shows all widgets).
fn knob_book() -> Storybook {
    let mut b = Storybook::new();
    b.add(
        Story::new("Demo/Scene/Knobs", |app, k| {
            let c = k.color("clear");
            app.world.spawn((
                Transform::from_xyz(0.0, 0.0, 3.0),
                Camera3d::default().with_clear_color(Vec4::from(c)),
            ));
        })
        .description("One knob of every kind.")
        .knob_color("clear", [0.1, 0.3, 0.6, 1.0])
        .knob_f32("roughness", 0.5, 0.0..=1.0)
        .knob_i64("count", 12, 1..=100)
        .knob_bool("wireframe", false)
        .knob_choice("projection", &["perspective", "orthographic"], 0)
        .knob_text("label", "hello")
        .tag("demo")
        .size(320, 180)
        .ticks(2),
    );
    b.add(Story::new("Demo/Scene/Other", |_, _| {}).size(64, 64));
    b.add(Story::new("Demo/UI/Panel", |_, _| {}).size(64, 64));
    b
}

/// The standalone tools opt into the compact density; `EditorPlugin` added to a game leaves the
/// game's egui style alone.
#[test]
fn standalone_tools_are_compact_and_the_plugin_keeps_a_games_style() {
    let config = EditorConfig {
        headless: Some((320, 200)),
        ..EditorConfig::new("test")
    };
    let mut editor = editor_app(&config);
    assert_eq!(editor.resources.get::<UiTheme>(), Some(&UiTheme::compact()));
    editor.step_ticks(2);
    let ctx = editor.resources.expect::<EguiContext>().ctx().clone();
    assert_eq!(UiDensity::of(&ctx), Some(UiDensity::compact()));
    assert_eq!(
        ctx.global_style().spacing.item_spacing,
        UiDensity::compact().item_spacing
    );
    let viewer = storybook_app(
        demo_book(),
        StorybookConfig {
            headless: Some((320, 200)),
            ..Default::default()
        },
    );
    assert_eq!(viewer.resources.get::<UiTheme>(), Some(&UiTheme::compact()));

    // A game with its own style adds the in-game editor: nothing restyles egui.
    let mut game = App::new();
    game.add_plugin(space3d_ui::UiPlugin::deterministic());
    let ctx = game.resources.expect::<EguiContext>().ctx().clone();
    ctx.global_style_mut(|s| s.spacing.item_spacing = egui::vec2(12.0, 8.0));
    let style = ctx.global_style();
    game.add_plugin(EditorPlugin::default());
    game.step_ticks(3);
    assert!(game.resources.get::<UiTheme>().is_none());
    assert!(
        std::sync::Arc::ptr_eq(&style, &ctx.global_style()),
        "the game's style is untouched"
    );
}

/// Width of an egui side panel after the last frame.
fn panel_width(app: &App, id: &str) -> f32 {
    let ctx = app.resources.expect::<EguiContext>().ctx().clone();
    egui::containers::panel::PanelState::load(&ctx, egui::Id::new(id))
        .unwrap_or_else(|| panic!("panel {id} shown"))
        .outer_rect
        .width()
}

/// The inspector's rows (sliders with their value boxes, vec3 fields, checkboxes, combo boxes)
/// never ask for more than the panel's width, whatever the style: the panels keep their width
/// frame after frame (they used to grow every frame until the viewport was gone) and it does not
/// depend on the selection — also with every graphics section open (camera frame,
/// post-processing, skybox, environment light, directional shadow).
#[test]
fn editor_panels_keep_their_width_under_any_style() {
    let styles: [(&str, Option<UiTheme>, bool); 4] = [
        ("egui default", None, false),
        ("compact", Some(UiTheme::compact()), false),
        (
            "custom 14 pt",
            Some(UiTheme::Custom(UiDensity {
                body_size: 14.0,
                button_size: 14.0,
                ..UiDensity::compact()
            })),
            false,
        ),
        ("roomy game style", None, true),
    ];
    for (name, theme, roomy) in styles {
        let mut app = App::new();
        app.add_plugin(space3d_ui::UiPlugin {
            install_tracing: false,
            ..space3d_ui::UiPlugin::deterministic()
        });
        if let Some(t) = theme {
            app.insert_resource(t);
        }
        if roomy {
            let ctx = app.resources.expect::<EguiContext>().ctx().clone();
            ctx.all_styles_mut(|s| {
                s.spacing.item_spacing = egui::vec2(14.0, 10.0);
                s.spacing.button_padding = egui::vec2(12.0, 6.0);
                s.spacing.interact_size = egui::vec2(48.0, 32.0);
                for font in s.text_styles.values_mut() {
                    font.size += 4.0;
                }
            });
        }
        app.add_plugin(EditorPlugin::default());
        scene(&mut app);
        let cube = entity_named(&app, "Cube").unwrap();
        let camera = entity_named(&app, "Main camera").unwrap();
        let _ = app.world.insert_one(cube, Transform::from_xyz(-12.345, 0.5, 1234.5));
        let sky = RenderAssets::default().add_cubemap(CubemapSource::from_equirect_image(ImageData::new(8, 4)));
        let mut post = PostProcess::default();
        post.chromatic_aberration.enabled = true;
        let graphics = app.world.spawn((
            Name("Graphics".into()),
            Transform::IDENTITY,
            Camera3d {
                active: false,
                ..Camera3d::default()
            },
            CameraFrame {
                render_scale: Some(0.75),
                ..CameraFrame::default()
            },
            post,
            DirectionalLight::default(),
            DirectionalShadow::default(),
            Skybox::new(sky),
            EnvironmentLight::new(sky),
        ));
        app.resources.expect_mut::<EditorState>().selected = Some(cube);
        app.step_ticks(6);
        let widths = |app: &App| {
            (
                panel_width(app, "space3d_editor_hierarchy"),
                panel_width(app, "space3d_editor_inspector"),
            )
        };
        let first = widths(&app);
        app.step_ticks(60);
        assert_eq!(widths(&app), first, "{name}: panels keep their width");
        assert!(first.1 <= 480.0 && first.0 <= 400.0, "{name}: {first:?}");
        for selection in [Some(camera), Some(graphics), None, Some(cube)] {
            app.resources.expect_mut::<EditorState>().selected = selection;
            app.step_ticks(6);
            // Different selections lay out different text, and glyph metrics differ by a
            // fraction of a pixel between platforms (0.1875 px on macOS): the width must not
            // depend on the selection beyond that. Growth over frames stays exact (above).
            let now = widths(&app);
            assert!(
                (now.0 - first.0).abs() < 1.0 && (now.1 - first.1).abs() < 1.0,
                "{name}: same width with {selection:?} selected: {now:?} vs {first:?}"
            );
        }
    }
}

/// The menu bar is as tall as egui's own top panel (one control plus 2 pt above and below), and
/// the texture and model editors open over the viewport: never over the menu bar or the side
/// panels.
#[test]
fn tool_windows_stay_over_the_viewport() {
    let mut app = App::new();
    app.add_plugin(space3d_ui::UiPlugin {
        install_tracing: false,
        fallback_size: (1280, 720),
        ..space3d_ui::UiPlugin::deterministic()
    });
    app.insert_resource(UiTheme::compact());
    app.add_plugin(EditorPlugin::default());
    scene(&mut app);
    app.resources.expect_mut::<TextureEditor>().open = true;
    app.resources.expect_mut::<ModelEditor>().open = true;
    app.step_ticks(6);
    let ctx = app.resources.expect::<EguiContext>().ctx().clone();
    let panel = |id: &str| {
        egui::containers::panel::PanelState::load(&ctx, egui::Id::new(id))
            .unwrap_or_else(|| panic!("panel {id} shown"))
            .outer_rect
    };
    let (menu, hierarchy, inspector) = (
        panel("space3d_editor_menu"),
        panel("space3d_editor_hierarchy"),
        panel("space3d_editor_inspector"),
    );
    // One control, 2 pt above and below, and the panel's 1 pt separator line.
    let control = UiDensity::compact().control_height;
    assert!((menu.height() - (control + 5.0)).abs() < 0.5, "menu bar {menu:?}");
    let viewport = egui::Rect::from_min_max(
        egui::pos2(hierarchy.right(), menu.bottom()),
        egui::pos2(inspector.left(), ctx.content_rect().bottom()),
    );
    for id in ["space3d_texture_editor", "space3d_model_editor"] {
        let window = ctx
            .memory(|m| m.area_rect(egui::Id::new(id)))
            .unwrap_or_else(|| panic!("{id} shown"));
        // 6 pt clear of the panels (the model editor's 820 pt used to touch both at 1280 x 720).
        assert!(
            viewport.shrink(5.5).contains_rect(window),
            "{id} {window:?} within the viewport {viewport:?}, 6 pt in"
        );
    }
}

/// The texture and model editors share the engine tools' foreground layer (the tools sit there
/// so a game's windows never cover them): the window opened or clicked last is in front, so an
/// editor window is never stuck under F3 / F4 / the console (they docked over its title bar,
/// toolbar and canvas for good).
#[test]
fn editor_windows_and_engine_tools_come_to_the_front_when_clicked() {
    let mut app = App::new();
    app.add_plugin(space3d_ui::UiPlugin {
        install_tracing: false,
        fallback_size: (1280, 720),
        ..space3d_ui::UiPlugin::deterministic()
    });
    app.insert_resource(UiTheme::compact());
    app.add_plugin(EditorPlugin::default());
    scene(&mut app);
    app.resources.expect_mut::<space3d_ui::DebugOverlay>().visible = true;
    app.resources.expect_mut::<space3d_ui::Profiler>().visible = true;
    app.resources.expect_mut::<space3d_ui::Console>().visible = true;
    app.step_ticks(6);
    let ctx = app.resources.expect::<EguiContext>().ctx().clone();
    let rect = |id: &str| {
        ctx.memory(|m| m.area_rect(egui::Id::new(id)))
            .unwrap_or_else(|| panic!("{id} shown"))
    };
    let front = || {
        ctx.memory(|m| m.areas().top_layer_id(egui::Order::Foreground))
            .map(|l| l.id)
    };
    let at = |p: egui::Pos2| ctx.layer_id_at(p).map(|l| l.id);
    let (texture_editor, profiler) = (
        egui::Id::new("space3d_texture_editor"),
        egui::Id::new("space3d_profiler"),
    );

    // Opened after the tools: in front of all of them.
    app.resources.expect_mut::<TextureEditor>().open = true;
    app.step_ticks(4);
    assert_eq!(front(), Some(texture_editor), "opens in front");
    let (tex, tools) = (
        rect("space3d_texture_editor"),
        [
            rect("space3d_debug_overlay"),
            rect("space3d_profiler"),
            rect("space3d_console"),
        ],
    );
    let both = tex.intersect(tools[1]);
    assert!(
        both.is_positive(),
        "the texture editor {tex:?} overlaps the profiler {:?}",
        tools[1]
    );
    assert_eq!(at(both.center()), Some(texture_editor));

    // A click on the profiler (where the texture editor leaves it free) brings it in front.
    let y = tools[1].center().y;
    let free = (tools[1].left() as i32..tools[1].right() as i32)
        .rev()
        .map(|x| egui::pos2(x as f32, y))
        .find(|&p| tools[1].shrink(4.0).contains(p) && !tex.expand(1.0).contains(p))
        .expect("a part of the profiler outside the texture editor");
    click_at(&mut app, Vec2::new(free.x, free.y));
    assert_eq!(front(), Some(profiler));
    assert_eq!(at(both.center()), Some(profiler));
    // Docked = same position. Its height follows the profiler's content (frame history), which
    // grows between frames on a slow machine, so the size is not compared.
    assert_eq!(rect("space3d_profiler").min, tools[1].min, "a click leaves it docked");

    // A click on the texture editor's title bar (clear of the tools) brings it back in front,
    // above the overlay too.
    let title = (tex.left() as i32 + 30..tex.right() as i32 - 30)
        .map(|x| egui::pos2(x as f32, tex.top() + 8.0))
        .find(|p| tools.iter().all(|t| !t.expand(2.0).contains(*p)))
        .expect("a part of the title bar clear of the tools");
    click_at(&mut app, Vec2::new(title.x, title.y));
    assert_eq!(front(), Some(texture_editor));
    assert_eq!(at(both.center()), Some(texture_editor));

    // The model editor too.
    app.resources.expect_mut::<TextureEditor>().open = false;
    app.resources.expect_mut::<ModelEditor>().open = true;
    app.step_ticks(4);
    assert_eq!(front(), Some(egui::Id::new("space3d_model_editor")));
}

/// The engine tools (F3 overlay, F4 profiler, console) dock inside the editor's viewport: never
/// over the menu bar, the hierarchy or the inspector, at any window size, and they follow the
/// viewport when the window is resized. With the editor UI hidden (F1) they use the whole screen.
#[test]
fn engine_tools_dock_inside_the_viewport() {
    let mut app = App::new();
    app.add_plugin(space3d_ui::UiPlugin {
        install_tracing: false,
        ..space3d_ui::UiPlugin::deterministic()
    });
    app.insert_resource(space3d_platform::WindowState {
        physical_size: space3d_core::glam::UVec2::new(1280, 720),
        scale_factor: 1.0,
        focused: true,
        suspended: false,
        title: String::new(),
        cursor_visible: true,
        cursor_grab: false,
    });
    app.insert_resource(UiTheme::compact());
    app.add_plugin(EditorPlugin::default());
    scene(&mut app);
    app.resources.expect_mut::<space3d_ui::DebugOverlay>().visible = true;
    app.resources.expect_mut::<space3d_ui::Profiler>().visible = true;
    app.resources.expect_mut::<space3d_ui::Console>().visible = true;
    let tools = |app: &App| {
        let ctx = app.resources.expect::<EguiContext>().ctx().clone();
        ["space3d_debug_overlay", "space3d_profiler", "space3d_console"].map(|id| {
            ctx.memory(|m| m.area_rect(egui::Id::new(id)))
                .unwrap_or_else(|| panic!("{id} shown"))
        })
    };
    for size in [(1280, 720), (1920, 1080), (1280, 720)] {
        app.resources
            .expect_mut::<space3d_platform::WindowState>()
            .physical_size = space3d_core::glam::UVec2::new(size.0, size.1);
        app.step_ticks(6);
        let ctx = app.resources.expect::<EguiContext>().ctx().clone();
        let panel = |id: &str| {
            egui::containers::panel::PanelState::load(&ctx, egui::Id::new(id))
                .unwrap_or_else(|| panic!("panel {id} shown"))
                .outer_rect
        };
        let (menu, hierarchy, inspector) = (
            panel("space3d_editor_menu"),
            panel("space3d_editor_hierarchy"),
            panel("space3d_editor_inspector"),
        );
        let screen = ctx.content_rect();
        assert_eq!(screen.size(), egui::vec2(size.0 as f32, size.1 as f32));
        let viewport = egui::Rect::from_min_max(
            egui::pos2(hierarchy.right(), menu.bottom()),
            egui::pos2(inspector.left(), screen.bottom()),
        );
        let [overlay, profiler, console] = tools(&app);
        let label =
            format!("{size:?}: viewport {viewport:?}, overlay {overlay:?}, profiler {profiler:?}, console {console:?}");
        for tool in [overlay, profiler, console] {
            assert!(viewport.shrink(5.5).contains_rect(tool), "inside the viewport: {label}");
        }
        assert!(
            overlay.left_top().distance(viewport.left_top() + egui::vec2(6.0, 6.0)) < 1.0,
            "{label}"
        );
        assert!(
            profiler
                .right_top()
                .distance(viewport.right_top() + egui::vec2(-6.0, 6.0))
                < 1.0,
            "{label}"
        );
        assert!(
            console
                .left_bottom()
                .distance(viewport.left_bottom() + egui::vec2(6.0, -6.0))
                < 1.0,
            "{label}"
        );
    }
    // F1 hides the editor UI: the tools take the screen's corners.
    app.resources.expect_mut::<EditorState>().visible = false;
    app.step_ticks(3);
    let screen = app.resources.expect::<EguiContext>().ctx().content_rect();
    let [overlay, profiler, _] = tools(&app);
    assert!(overlay.left_top().distance(egui::pos2(6.0, 6.0)) < 1.0, "{overlay:?}");
    assert!(
        profiler
            .right_top()
            .distance(screen.right_top() + egui::vec2(-6.0, 6.0))
            < 1.0,
        "{profiler:?}"
    );
}

/// Editor, texture editor, model editor and storybook viewer captures, written to
/// `$SPACE3D_EDITOR_SNAPSHOT_DIR` (for reviewing the tools' layout by eye).
#[test]
fn tool_layouts_for_review() {
    let Some(mut app) = headless_editor(1280, 720) else {
        return;
    };
    let cube = entity_named(&app, "Cube").unwrap();
    let child = app.world.spawn((
        Name("Cube child".into()),
        Transform::from_xyz(1.0, 0.0, 0.0),
        space3d_core::Parent(cube),
    ));
    let _ = child;
    app.world.spawn((
        Name("Sun".into()),
        Transform::IDENTITY,
        DirectionalLight {
            color: Vec3::ONE,
            intensity: 2.0,
        },
    ));
    app.resources.expect_mut::<EditorState>().selected = Some(cube);
    app.step_ticks(4);
    let img = render_to_image(&mut app).expect("capture");
    maybe_save("review_editor", &img);

    {
        let tex = app.resources.expect_mut::<TextureEditor>();
        tex.open = true;
        tex.open_image(ImageData::filled(64, 64, [0, 200, 90, 255]));
    }
    app.step_ticks(12);
    let img = render_to_image(&mut app).expect("capture");
    maybe_save("review_texture_editor", &img);
    app.resources.expect_mut::<TextureEditor>().open = false;
    app.resources.expect_mut::<ModelEditor>().open = true;
    app.step_ticks(12);
    let img = render_to_image(&mut app).expect("capture");
    maybe_save("review_model_editor", &img);
    app.resources.expect_mut::<ModelEditor>().open = false;

    // The inspector's graphics sections: camera frame and post-processing (on an inactive camera:
    // the viewport keeps its look), then skybox, environment light and directional shadow.
    let camera = app.world.spawn((
        Name("Post camera".into()),
        Transform::IDENTITY,
        Camera3d {
            active: false,
            ..Camera3d::default()
        },
        CameraFrame::default(),
        PostProcess::default(),
    ));
    app.resources.expect_mut::<EditorState>().selected = Some(camera);
    app.step_ticks(4);
    let img = render_to_image(&mut app).expect("capture");
    maybe_save("review_inspector_post", &img);
    let sky = app
        .resources
        .expect_mut::<RenderAssets>()
        .add_cubemap(CubemapSource::from_equirect_image(ImageData::filled(
            8,
            4,
            [40, 60, 90, 255],
        )));
    let environment = app.world.spawn((
        Name("Environment".into()),
        DirectionalLight::default(),
        DirectionalShadow::default(),
        Skybox::new(sky),
        EnvironmentLight::new(sky),
    ));
    app.resources.expect_mut::<EditorState>().selected = Some(environment);
    app.step_ticks(4);
    let img = render_to_image(&mut app).expect("capture");
    maybe_save("review_inspector_lighting", &img);

    let config = StorybookConfig {
        headless: Some((1280, 720)),
        ..Default::default()
    };
    let mut app = storybook_app(knob_book(), config);
    for _ in 0..400 {
        app.update(Duration::from_millis(16));
        let frames = app
            .resources
            .expect::<StorybookViewer>()
            .view()
            .map_or(0, LiveView::frames);
        if frames >= 2 {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    let img = render_to_image(&mut app).expect("capture");
    maybe_save("review_storybook", &img);
}
