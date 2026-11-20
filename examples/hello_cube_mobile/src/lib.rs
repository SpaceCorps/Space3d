//! hello_cube as a library a phone can load: `space3d::main!` generates Android's
//! `android_main` and iOS's `start_app`.
//!
//! It also loads one image through the asset server (`assets/proof.png`, packed into the APK
//! or copied into the iOS bundle) and says in the log whether it arrived: the smallest check
//! that the platform's default asset source works on a device.

use space3d::assets::{AssetServer, Handle, LoadState};
use space3d::core::ImageData;
use space3d::prelude::*;
use space3d::tracing;

/// The proof image's handle, until its load ends one way or the other.
struct Proof(Option<Handle<ImageData>>);

fn build(app: &mut App) {
    hello_cube::build(app);
    let handle = app.resources.expect_mut::<AssetServer>().load("proof.png");
    app.insert_resource(Proof(Some(handle)));
    app.add_system(Stage::Update, |ctx| {
        let Some(handle) = ctx.resources.expect::<Proof>().0.clone() else {
            return;
        };
        let state = ctx.resources.expect::<AssetServer>().load_state(&handle);
        match state {
            LoadState::Loaded => tracing::info!("hello_cube_mobile: asset proof.png loaded"),
            LoadState::Failed(e) => tracing::error!("hello_cube_mobile: asset proof.png failed: {e}"),
            _ => return,
        }
        ctx.resources.expect_mut::<Proof>().0 = None;
    });
}

space3d::main!(build);
