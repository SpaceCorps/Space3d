//! Feature policy: no new weak dependency features (`dep?/feature`) in the facade. Cargo resolves
//! a lockfile as if every package such a feature could reach were enabled, so a game using the
//! facade writes them all into its `Cargo.lock` without ever building them: `ui` used to hold
//! `space3d-stories?/ui`, which listed the storybook's whole tree (space3d-stories,
//! space3d-physics, space3d-test, rapier3d, nalgebra, parry3d, three older glam versions, ...:
//! 24 packages) in SpaceCorps2027's lockfile. A feature that needs another crate's feature
//! enables that crate too (`stories = ["dep:space3d-stories", "space3d-stories/ui"]`).
//!
//! (Dependencies' own weak features are theirs: with `physics`, nalgebra's `std` names
//! `glam030?/std` .. `glam033?/std` and parry names `spade?/..`, so glam 0.30–0.32, robust and
//! spade stay listed unbuilt in lockfiles with the default features.)

const MANIFEST: &str = include_str!("../Cargo.toml");

/// The one weak feature left: `render` hands `space3d-assets` its renderer integration. It only
/// lists space3d-assets' tree (image, gltf, tobj, notify) in the lockfiles of crates with
/// `render` but without `assets`; without it `assets` would have to imply `render` (wgpu), or
/// builds with both would lose the integration.
const ALLOWED: [&str; 1] = ["space3d-assets?/render"];

#[test]
fn the_facade_has_no_new_weak_dependency_features() {
    // Windows checkouts may have CRLF line endings.
    let manifest = MANIFEST.replace("\r\n", "\n");
    let features = manifest
        .split("\n[features]\n")
        .nth(1)
        .expect("a [features] table")
        .split("\n[")
        .next()
        .unwrap_or_default();
    let weak: Vec<&str> = features
        .lines()
        .map(str::trim)
        .filter(|l| !l.starts_with('#'))
        .flat_map(|l| l.split('"').skip(1).step_by(2))
        .filter(|f| f.contains("?/") && !ALLOWED.contains(f))
        .collect();
    assert!(weak.is_empty(), "weak dependency features: {weak:?}");
    assert!(
        features.contains("stories = [\"dep:space3d-stories\", \"space3d-stories/ui\"]"),
        "the storybook holds the UI stories"
    );
}
