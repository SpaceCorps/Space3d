//! The UI's parts of a frame are trace spans of the kind the schedule opens round every system (`"system"` with a `name`), so the
//! profiler lists them beside the systems: the pass begun (`ui::begin_frame.pass`), the pass ended (`ui::end_frame.pass`) and the
//! tessellation (`ui::end_frame.tessellate`), each inside its system's span. Headless: no GPU, no window.

use space3d_core::glam::UVec2;
use space3d_core::prelude::*;
use space3d_platform::WindowState;
use space3d_ui::*;

#[test]
fn the_parts_of_the_ui_frame_are_spans_inside_their_systems() {
    let mut app = App::new();
    // (the plugin installs the profiler's layer: on this thread, or for the process)
    app.add_plugin(UiPlugin::deterministic());
    app.insert_resource(WindowState {
        physical_size: UVec2::new(640, 360),
        scale_factor: 1.0,
        focused: true,
        suspended: false,
        title: String::new(),
        cursor_visible: true,
        cursor_grab: false,
    });
    app.add_root_ui_system(|ui: &mut egui::Ui, _: &mut SystemCtx| {
        ui.label("a label to lay out and tessellate");
    });
    Profiler::set_enabled(true);
    app.step_ticks(3);
    let profiler = app.resources.expect_mut::<Profiler>();
    profiler.update();
    let frame = profiler.timeline().clone();
    let of = |name: &str| {
        let hits: Vec<&TimelineSpan> = frame.spans.iter().filter(|s| &*s.name == name).collect();
        assert_eq!(hits.len(), 1, "{name} once in a frame: {:?}", frame.spans);
        hits[0].clone()
    };
    // A part runs within its system's time, one level under it
    let inside = |part: &str, system: &str| {
        let (p, s) = (of(part), of(system));
        assert_eq!(p.depth, s.depth + 1, "{part}");
        assert!(
            p.start >= s.start && p.start + p.duration <= s.start + s.duration + 1e-9,
            "{part}: {p:?} in {s:?}"
        );
        p
    };
    inside("ui::begin_frame.pass", "ui::begin_frame");
    let pass = inside("ui::end_frame.pass", "ui::end_frame");
    let tessellate = inside("ui::end_frame.tessellate", "ui::end_frame");
    assert!(
        pass.start + pass.duration <= tessellate.start + 1e-9,
        "the pass ends before its shapes are tessellated"
    );
    // ... and each has its row in the table, like a system
    for name in ["ui::begin_frame.pass", "ui::end_frame.pass", "ui::end_frame.tessellate"] {
        assert!(profiler.get(name).is_some_and(|s| s.calls == 1), "{name}");
    }
}
