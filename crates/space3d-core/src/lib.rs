//! Space3d core: the [`App`], plugins, ECS world, resources, events and the
//! fixed-tickrate scheduler that every other engine crate builds on.
//!
//! Frame model (see [`App::update`]):
//! 1. `Startup` (once)
//! 2. `PreUpdate` — async task completions, input, network inbox
//! 3. `FixedUpdate` × N — deterministic simulation at [`TickRate`]
//! 4. `Update` — variable-rate game logic
//! 5. `PostUpdate` — transforms, physics sync, audio
//! 6. `Render` — renderer extraction/submission
//! 7. `Last` — event buffer swap, cleanup

pub mod app;
pub mod background;
pub mod data;
pub mod events;
pub mod resources;
pub mod schedule;
pub mod time;
pub mod transform;

pub use app::{App, AppExit, ENV_HEADLESS_UNPACED, HeadlessPacing, Pacer, Plugin};
pub use background::BackgroundHooks;
pub use data::{HdrImageData, ImageData, MeshData};
pub use events::{EventReader, Events};
pub use resources::Resources;
pub use schedule::{Stage, SystemCtx};
pub use time::{TickRate, Time};
pub use transform::{GlobalTransform, Name, Parent, Transform};

pub use glam;
pub use hecs;
pub use hecs::{Entity, World};

pub mod prelude {
    pub use crate::{
        App, AppExit, Entity, EventReader, Events, GlobalTransform, ImageData, MeshData, Name, Parent, Plugin,
        Resources, Stage, SystemCtx, TickRate, Time, Transform, World,
    };
    pub use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
}
