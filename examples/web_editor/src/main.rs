//! The Space3d editor in the browser (and on the desktop): scene hierarchy, inspector,
//! translate gizmo and picking, play / pause / step, undo / redo, scene save / load (download /
//! upload on the web), plus the texture and model editors. Starts paused on a small sample scene.
//!
//! ```text
//! cargo run -p web_editor                       # desktop
//! cargo build -p web_editor --release --target wasm32-unknown-unknown   # web (see site/scripts/build-play.sh)
//! ```

use space3d::editor::{MeshSource, editor_app};
use space3d::prelude::*;

/// The scene the editor opens with.
fn sample_scene(app: &mut App) {
    let w = &mut app.world;
    let mut thing = |name: &str, pos: Vec3, mesh: MeshSource, color: (f32, f32, f32)| {
        w.spawn((
            Name(name.into()),
            Transform::from_xyz(pos.x, pos.y, pos.z),
            mesh,
            Material::srgb(color.0, color.1, color.2).with_metallic_roughness(0.1, 0.4),
        ));
    };
    thing(
        "Ground",
        Vec3::ZERO,
        MeshSource::Plane { size: 12.0 },
        (0.35, 0.38, 0.42),
    );
    thing(
        "Crate",
        Vec3::new(-1.5, 0.5, 0.0),
        MeshSource::Cube { size: 1.0 },
        (0.95, 0.45, 0.2),
    );
    thing(
        "Ball",
        Vec3::new(1.2, 0.6, 0.5),
        MeshSource::Sphere {
            radius: 0.6,
            sectors: 32,
            stacks: 16,
        },
        (0.2, 0.7, 0.9),
    );
    w.spawn((
        Name("Sun".into()),
        Transform::from_xyz(3.0, 5.0, 2.0).looking_at(Vec3::ZERO, Vec3::Y),
        DirectionalLight {
            color: Vec3::new(1.0, 0.96, 0.9),
            intensity: 2.5,
        },
    ));
    w.spawn((
        Name("Camera".into()),
        Transform::from_xyz(0.0, 2.5, 5.0).looking_at(Vec3::new(0.0, 0.5, 0.0), Vec3::Y),
        Camera3d::default().with_clear_color(Vec4::new(0.04, 0.05, 0.08, 1.0)),
    ));
    app.resources.insert(AmbientLight {
        color: Vec3::new(0.7, 0.8, 1.0),
        intensity: 0.2,
    });
}

fn build(app: &mut App) {
    *app = editor_app(&space3d::editor::EditorConfig::new("Space3d Editor").with_setup(sample_scene));
}

space3d::main!(build);
