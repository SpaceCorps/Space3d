//! Space3d assets: asset server, typed handles, image/glTF/OBJ loading, hot reload, and
//! texture & mesh editing operations.
//!
//! # Loading
//! Add [`AssetsPlugin`], then call [`AssetServer::load`] from anywhere with access to
//! resources. Loading never blocks the frame: bytes are read on the `space3d_tasks` I/O
//! runtime (tokio fs natively, `fetch` on the web), decoded on the compute pool, and
//! inserted into [`Assets<T>`] during `Stage::PreUpdate`, which also emits
//! [`AssetEvent<T>`]s.
//!
//! ```
//! use space3d_assets::prelude::*;
//! use space3d_assets::{MemorySource, edit::texture};
//! use space3d_core::prelude::*;
//!
//! let files = MemorySource::new();
//! files.insert("red.png", texture::encode_png(&ImageData::filled(2, 2, [255, 0, 0, 255])).unwrap());
//!
//! let mut app = App::new();
//! app.add_plugin(AssetsPlugin::default());
//! app.resources.expect_mut::<AssetServer>().add_source("mem", files);
//! let handle: Handle<ImageData> = app.resources.expect_mut::<AssetServer>().load("mem://red.png");
//!
//! // The frame loop keeps ticking while the asset loads in the background.
//! for _ in 0..1000 {
//!     app.step_ticks(1);
//!     if app.resources.expect::<AssetServer>().is_loaded(&handle) { break; }
//!     std::thread::sleep(std::time::Duration::from_millis(1));
//! }
//! let img = app.resources.expect::<Assets<ImageData>>().get(&handle).unwrap();
//! assert_eq!(img.get(0, 0), [255, 0, 0, 255]);
//! ```
//!
//! # Editing
//! [`edit::texture`] and [`edit::mesh`] hold pure functions on
//! [`ImageData`] / [`MeshData`], plus
//! undo/redo histories and OBJ/glTF export.

#![warn(missing_docs)]

mod assets;
pub mod edit;
mod error;
mod handle;
#[cfg(all(feature = "hot-reload", not(target_arch = "wasm32")))]
mod hot_reload;
mod loader;
pub mod loaders;
mod model;
mod par;
mod path;
#[cfg(feature = "render")]
pub mod render;
mod server;
mod source;

pub use assets::{Asset, AssetEvent, Assets};
pub use error::AssetError;
pub use handle::{AssetId, AssetIndex, Handle, UntypedAssetId};
#[cfg(all(feature = "hot-reload", not(target_arch = "wasm32")))]
pub use hot_reload::HotReloadWatcher;
pub use loader::{AssetLoader, LoadContext, MAX_DEPENDENCY_ROUNDS};
pub use loaders::{
    CubeConvention, CubemapData, GltfLoader, HdrImageLoader, ImageLoader, MeshLoader, ObjLoader, SkyboxLoader,
};
pub use model::{
    AlphaMode, MaterialData, MaterialTextureInfo, ModelData, ModelMesh, NodeData, Primitive, PrimitiveExtras,
    SamplerData, TextureInfo, UvTransformData, WrapMode,
};
pub use par::PAR_ROW_THRESHOLD;
pub use path::AssetPath;
pub use server::{AssetServer, LoadState};
#[cfg(target_os = "android")]
pub use source::{AndroidAssetSource, set_android_asset_manager};
pub use source::{AssetSource, BoxedFuture, FileSource, HttpSource, MemorySource, default_source};

use space3d_core::{App, Events, ImageData, MeshData, Plugin, Stage, SystemCtx};
use std::any::TypeId;
use std::marker::PhantomData;

/// Common imports.
pub mod prelude {
    pub use crate::{
        AssetApp, AssetEvent, AssetId, AssetLoader, AssetPath, AssetServer, Assets, AssetsPlugin, Handle, LoadState,
        ModelData,
    };
}

/// Registers the [`AssetServer`] (with the built-in loaders), `Assets<ImageData>`,
/// `Assets<MeshData>`, `Assets<ModelData>`, their events, and (native, `hot-reload`
/// feature) the file watcher.
#[derive(Clone, Debug)]
pub struct AssetsPlugin {
    /// Asset root: a directory on native ([`FileSource`]) or a base URL on the web
    /// ([`HttpSource`]). Default `"assets"`.
    pub root: String,
    /// Watch `root` and reload changed files. Only effective on native with the
    /// `hot-reload` feature (default on there).
    pub hot_reload: bool,
}

impl Default for AssetsPlugin {
    fn default() -> Self {
        Self {
            root: "assets".into(),
            // A phone has no asset folder to watch (an APK's assets are zip entries).
            hot_reload: cfg!(all(
                feature = "hot-reload",
                not(any(target_arch = "wasm32", target_os = "android", target_os = "ios"))
            )),
        }
    }
}

impl AssetsPlugin {
    /// Plugin with a custom root.
    pub fn with_root(root: impl Into<String>) -> Self {
        Self {
            root: root.into(),
            ..Default::default()
        }
    }
}

impl Plugin for AssetsPlugin {
    fn build(&self, app: &mut App) {
        let source = default_source(&self.root);
        match app.resources.get_mut::<AssetServer>() {
            Some(server) => server.set_default_source_arc(source),
            None => {
                app.insert_resource(AssetServer::from_arc(source));
            }
        }
        loaders::register_builtin_loaders(app.resources.expect_mut::<AssetServer>());
        #[cfg(all(feature = "hot-reload", not(target_arch = "wasm32")))]
        if self.hot_reload {
            hot_reload::install(app, &self.root);
        }
        app.add_system_named(Stage::PreUpdate, "assets::process_loads", |ctx| {
            ctx.resources
                .scope::<AssetServer, _>(|resources, server| server.process(resources));
        });
        // Every type a built-in loader produces (the `.hdr` and `.skybox` loaders too: loading
        // a `CubemapData` or `HdrImageData` needs no extra setup).
        app.init_asset::<ImageData>()
            .init_asset::<MeshData>()
            .init_asset::<ModelData>()
            .init_asset::<space3d_core::HdrImageData>()
            .init_asset::<CubemapData>();
    }
}

/// `App` extension for registering asset types and loaders.
pub trait AssetApp {
    /// Register asset type `T`: inserts `Assets<T>`, `Events<AssetEvent<T>>` and the systems
    /// that release dropped handles and flush events (`PreUpdate` and `PostUpdate`).
    /// Idempotent.
    fn init_asset<T: Asset>(&mut self) -> &mut Self;
    /// Register a loader with the [`AssetServer`].
    fn register_asset_loader<L: AssetLoader>(&mut self, loader: L) -> &mut Self;
}

struct Maintain<T>(PhantomData<T>);
struct Flush<T>(PhantomData<T>);

impl AssetApp for App {
    fn init_asset<T: Asset>(&mut self) -> &mut Self {
        if self.resources.contains::<Assets<T>>() {
            return self;
        }
        let assets = Assets::<T>::new();
        let provider = assets.provider().clone();
        self.insert_resource(assets);
        self.resources.init::<AssetServer>().register_type::<T>(provider);
        self.add_event::<AssetEvent<T>>();
        self.add_system_named(
            Stage::PreUpdate,
            std::any::type_name::<Maintain<T>>(),
            maintain_assets::<T>,
        );
        self.add_system_named(Stage::PostUpdate, std::any::type_name::<Flush<T>>(), flush_events::<T>);
        self
    }

    fn register_asset_loader<L: AssetLoader>(&mut self, loader: L) -> &mut Self {
        self.resources.init::<AssetServer>().register_loader(loader);
        self
    }
}

fn maintain_assets<T: Asset>(ctx: &mut SystemCtx) {
    let freed = ctx.resources.expect_mut::<Assets<T>>().release_dropped();
    if !freed.is_empty()
        && let Some(server) = ctx.resources.get_mut::<AssetServer>()
    {
        for index in freed {
            server.forget(UntypedAssetId {
                type_id: TypeId::of::<T>(),
                index,
            });
        }
    }
    flush_events::<T>(ctx);
}

fn flush_events<T: Asset>(ctx: &mut SystemCtx) {
    let events = ctx.resources.expect_mut::<Assets<T>>().take_events();
    if events.is_empty() {
        return;
    }
    let queue = ctx.resources.init::<Events<AssetEvent<T>>>();
    for e in events {
        queue.send(e);
    }
}
