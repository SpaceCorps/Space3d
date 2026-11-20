//! Space3d networking that never blocks the frame.
//!
//! | piece | native | web (wasm32) |
//! |---|---|---|
//! | [`Http`] client | reqwest + rustls on tokio | reqwest's `fetch` backend |
//! | [`WsClient`] / [`WsConnection`] | tokio-tungstenite + rustls | `web_sys::WebSocket` (text + arraybuffer) |
//! | [`HttpServer`] / [`WsServer`] | axum + tower-http | — (native only) |
//! | [`Channel`] / [`Codec`] / [`Frame`] | ✓ | ✓ |
//!
//! All I/O runs on the background runtime from [`space3d_tasks`]. The engine side
//! only polls [`space3d_tasks::Task`]s and lock-free channels, so systems never wait
//! on the network. [`NetPlugin`] wires everything into the [`space3d_core::App`]:
//!
//! * inserts the [`Http`] and [`WsClient`] resources,
//! * runs [`Http`] `*_then` callbacks in `PreUpdate` (full `SystemCtx` access),
//! * emits `Events<WsEvent>` for engine-managed client connections (`PreUpdate`),
//! * native: emits `Events<ServerEvent>` for a [`WsServer`] inserted as a resource.
//!
//! ```no_run
//! use space3d_core::prelude::*;
//! use space3d_net::prelude::*;
//!
//! #[derive(serde::Deserialize)]
//! struct Motd { text: String }
//!
//! let mut app = App::new();
//! app.add_plugin(NetPlugin::default());
//! app.resources.expect::<Http>().get_json_then("https://example.com/motd.json", |ctx, r: Result<Motd, NetError>| {
//!     if let Ok(motd) = r {
//!         ctx.resources.insert(motd.text);
//!     }
//! });
//! let chat = app.resources.expect_mut::<WsClient>().open_with(
//!     "wss://example.com/chat",
//!     WsConfig::default().reconnect(ReconnectPolicy::default()),
//! );
//! app.resources.expect::<WsClient>().send(chat, "hello").unwrap(); // queued until open
//! app.run();
//! ```
//!
//! See [`channel`] for typed, sequenced, tick-stamped messages and a
//! server-authoritative snapshot/input replication example.

#![warn(missing_docs)]

mod backoff;
pub mod channel;
pub mod client;
mod error;
mod rt;
#[cfg(not(target_arch = "wasm32"))]
pub mod server;
#[cfg(not(target_arch = "wasm32"))]
mod tls;
pub mod ws;

pub use backoff::{Backoff, ReconnectPolicy, RetryPolicy};
pub use channel::{Channel, Codec, Frame};
pub use client::{Http, HttpConfig, HttpRequest, HttpResponse, Method};
pub use error::NetError;
#[cfg(not(target_arch = "wasm32"))]
pub use server::{
    AcceptHook, ClientId, ClientSender, ConnectMeta, ConnectionLimits, ConnectionStats, CorsLayer, HttpServer, PeerKey,
    ServerEvent, ServerHandle, ShutdownTrigger, WsServer, WsServerConfig, joined_header,
};
#[cfg(not(target_arch = "wasm32"))]
pub use tls::http_client_builder;
pub use ws::{
    ConnectionId, Message, WsClient, WsConfig, WsConnection, WsEvent, WsEventKind, WsState, ws_events_system,
};

use space3d_core::{App, Plugin, Stage};

/// Installs the HTTP client, WebSocket client and their per-frame systems.
///
/// Also adds [`space3d_tasks::TasksPlugin`]. An [`Http`] resource inserted before
/// the plugin is kept (so you can pre-configure it); otherwise one is built from
/// [`NetPlugin::http`].
#[derive(Clone, Debug, Default)]
pub struct NetPlugin {
    /// Defaults for the [`Http`] resource.
    pub http: HttpConfig,
}

impl Plugin for NetPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(space3d_tasks::TasksPlugin);
        if !app.resources.contains::<Http>() {
            app.insert_resource(Http::with_config(self.http.clone()));
        }
        app.init_resource::<WsClient>();
        app.add_event::<WsEvent>();
        app.add_system_named(Stage::PreUpdate, "net::http_callbacks", Http::run_callbacks);
        app.add_system_named(Stage::PreUpdate, "net::WsEvents", ws_events_system);
        #[cfg(not(target_arch = "wasm32"))]
        {
            app.add_event::<ServerEvent>();
            app.add_system_named(Stage::PreUpdate, "net::ws_server_events", |ctx| {
                let Some(server) = ctx.resources.get::<WsServer>() else {
                    return;
                };
                let events = server.drain();
                if !events.is_empty() {
                    let out = ctx.resources.init::<space3d_core::Events<ServerEvent>>();
                    for e in events {
                        out.send(e);
                    }
                }
            });
        }
    }
}

/// `use space3d_net::prelude::*;` for the common types.
pub mod prelude {
    pub use crate::{
        Backoff, Channel, Codec, ConnectionId, Frame, Http, HttpConfig, HttpRequest, HttpResponse, Message, NetError,
        NetPlugin, ReconnectPolicy, RetryPolicy, WsClient, WsConfig, WsConnection, WsEvent, WsEventKind, WsState,
    };
    #[cfg(not(target_arch = "wasm32"))]
    pub use crate::{ClientId, HttpServer, ServerEvent, ServerHandle, WsServer, WsServerConfig};
}

#[cfg(test)]
mod tests {
    use super::*;
    use space3d_core::Events;

    #[test]
    fn plugin_registers_resources_and_systems() {
        let mut app = App::new();
        app.add_plugin(NetPlugin::default()).add_plugin(NetPlugin::default());
        assert!(app.resources.contains::<Http>());
        assert!(app.resources.contains::<WsClient>());
        assert!(app.resources.contains::<Events<WsEvent>>());
        let pre = app.system_names(Stage::PreUpdate);
        assert!(pre.contains(&"net::WsEvents"));
        assert_eq!(pre.iter().filter(|n| **n == "net::http_callbacks").count(), 1);
        app.step_ticks(2);
    }

    #[test]
    fn resources_are_send_sync() {
        fn assert_send_sync<T: Send + Sync + 'static>() {}
        assert_send_sync::<Http>();
        assert_send_sync::<WsClient>();
        assert_send_sync::<WsConnection>();
        #[cfg(not(target_arch = "wasm32"))]
        assert_send_sync::<WsServer>();
    }
}
