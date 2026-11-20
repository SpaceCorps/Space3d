//! WebSocket client: [`WsClient`], [`WsConnection`] and the [`WsEvent`] stream.
//!
//! A connection is driven by a background task (tokio-tungstenite natively,
//! `web_sys::WebSocket` in the browser). The engine side only ever touches
//! lock-free channels, so [`WsConnection::send`], [`WsConnection::try_recv`]
//! and [`WsConnection::state`] never block the frame.
//!
//! Two ways to consume a connection:
//! * **Standalone**: keep the [`WsConnection`] yourself and poll
//!   [`WsConnection::try_recv`] / [`WsConnection::drain`].
//! * **Engine-managed**: register it in the [`WsClient`] resource
//!   ([`WsClient::open`]); [`crate::NetPlugin`] then drains it every
//!   `PreUpdate` into `Events<WsEvent>` tagged with the [`ConnectionId`].

use crate::backoff::ReconnectPolicy;
use crate::{NetError, rt};
use parking_lot::Mutex;
use space3d_core::{Events, SystemCtx};
use std::collections::BTreeMap;
use std::future::Future;
use std::pin::pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::Poll;
use std::time::Duration;

#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(not(target_arch = "wasm32"))]
use native::Socket;
#[cfg(target_arch = "wasm32")]
mod web;
#[cfg(target_arch = "wasm32")]
use web::Socket;

/// A WebSocket data message.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Message {
    /// UTF-8 text frame.
    Text(String),
    /// Binary frame.
    Binary(Vec<u8>),
}

impl Message {
    /// The text, if this is a text message.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Message::Text(s) => Some(s),
            Message::Binary(_) => None,
        }
    }
    /// Raw payload bytes (UTF-8 bytes for text).
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Message::Text(s) => s.as_bytes(),
            Message::Binary(b) => b,
        }
    }
    /// Payload length in bytes.
    pub fn len(&self) -> usize {
        self.as_bytes().len()
    }
    /// Whether the payload is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Whether this is a text message.
    pub fn is_text(&self) -> bool {
        matches!(self, Message::Text(_))
    }
    /// Consume into payload bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        match self {
            Message::Text(s) => s.into_bytes(),
            Message::Binary(b) => b,
        }
    }
}

impl From<String> for Message {
    fn from(s: String) -> Self {
        Message::Text(s)
    }
}
impl From<&str> for Message {
    fn from(s: &str) -> Self {
        Message::Text(s.to_owned())
    }
}
impl From<Vec<u8>> for Message {
    fn from(b: Vec<u8>) -> Self {
        Message::Binary(b)
    }
}
impl From<&[u8]> for Message {
    fn from(b: &[u8]) -> Self {
        Message::Binary(b.to_vec())
    }
}

/// Lifecycle state of a [`WsConnection`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WsState {
    /// Handshake in progress, or waiting to reconnect.
    Connecting,
    /// Connected; messages flow.
    Open,
    /// Permanently closed (by [`WsConnection::close`], or the peer with no reconnect policy).
    Closed(String),
}

/// Identifies a client-side connection in [`WsEvent`]s and [`WsClient`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ConnectionId(pub u64);

/// Connection options.
#[derive(Clone, PartialEq)]
pub struct WsConfig {
    /// Reconnect automatically after the connection drops or an attempt fails.
    pub reconnect: Option<ReconnectPolicy>,
    /// Limit for a single connect + handshake attempt (default 10 s).
    pub connect_timeout: Option<Duration>,
    /// Largest accepted inbound message in bytes; a bigger one closes the connection
    /// (`1009` on the web). `None` keeps the backend default (64 MiB natively, unlimited on the web).
    pub max_message_size: Option<usize>,
    /// Extra upgrade-request headers (e.g. `Authorization`), sent on every attempt. Native
    /// only: browsers cannot set WebSocket headers, so the web backend ignores them. Values
    /// are redacted in `Debug`.
    pub headers: Vec<(String, String)>,
}

impl Default for WsConfig {
    fn default() -> Self {
        Self {
            reconnect: None,
            connect_timeout: Some(Duration::from_secs(10)),
            max_message_size: None,
            headers: Vec::new(),
        }
    }
}

impl std::fmt::Debug for WsConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WsConfig")
            .field("reconnect", &self.reconnect)
            .field("connect_timeout", &self.connect_timeout)
            .field("max_message_size", &self.max_message_size)
            .field(
                "headers",
                &self.headers.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl WsConfig {
    /// Enable automatic reconnects.
    pub fn reconnect(mut self, policy: ReconnectPolicy) -> Self {
        self.reconnect = Some(policy);
        self
    }
    /// Set the per-attempt connect timeout.
    pub fn connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = Some(timeout);
        self
    }
    /// Cap the size of inbound messages (and frames) in bytes.
    pub fn max_message_size(mut self, bytes: usize) -> Self {
        self.max_message_size = Some(bytes);
        self
    }
    /// Add an upgrade-request header (native only; see [`WsConfig::headers`]).
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }
}

/// What happened on a connection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WsEventKind {
    /// The handshake completed (also after each successful reconnect).
    Opened,
    /// A message arrived.
    Message(Message),
    /// The connection closed. `reconnecting` is `true` when a [`ReconnectPolicy`]
    /// will try again; a final `Closed { reconnecting: false }` is always sent last.
    Closed {
        /// Human-readable reason (close frame code/reason or transport error).
        reason: String,
        /// Whether an automatic reconnect will follow.
        reconnecting: bool,
    },
}

/// An event from an engine-managed connection (see [`WsClient`]), emitted as `Events<WsEvent>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WsEvent {
    /// Which connection.
    pub connection: ConnectionId,
    /// What happened.
    pub kind: WsEventKind,
}

/// Handle to one WebSocket connection. Dropping it closes the connection.
///
/// Messages sent while `Connecting` (including during a reconnect backoff) are
/// buffered and delivered once the socket opens.
pub struct WsConnection {
    id: ConnectionId,
    url: String,
    state: Arc<Mutex<WsState>>,
    outgoing: async_channel::Sender<Message>,
    close: async_channel::Sender<()>,
    events: async_channel::Receiver<WsEventKind>,
}

impl std::fmt::Debug for WsConnection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WsConnection")
            .field("id", &self.id)
            .field("url", &self.url)
            .field("state", &self.state())
            .finish()
    }
}

static NEXT_CONNECTION: AtomicU64 = AtomicU64::new(1);

impl WsConnection {
    /// This connection's id.
    pub fn id(&self) -> ConnectionId {
        self.id
    }
    /// The URL passed to connect.
    pub fn url(&self) -> &str {
        &self.url
    }
    /// Current lifecycle state.
    pub fn state(&self) -> WsState {
        self.state.lock().clone()
    }
    /// Whether the socket is currently open.
    pub fn is_open(&self) -> bool {
        matches!(*self.state.lock(), WsState::Open)
    }
    /// Whether the connection is permanently closed.
    pub fn is_closed(&self) -> bool {
        matches!(*self.state.lock(), WsState::Closed(_))
    }

    /// Queue a message. Never blocks; fails only once the connection is closed for good.
    pub fn send(&self, msg: impl Into<Message>) -> Result<(), NetError> {
        if self.is_closed() {
            return Err(NetError::Closed);
        }
        self.outgoing.try_send(msg.into()).map_err(|_| NetError::Closed)
    }

    /// Messages queued by [`Self::send`] that the socket has not taken yet (e.g. while
    /// connecting or when the network is congested). Useful to skip optional traffic.
    pub fn queued_outgoing(&self) -> usize {
        self.outgoing.len()
    }

    /// Next received message, skipping lifecycle events (see [`Self::state`]). Never blocks.
    pub fn try_recv(&self) -> Option<Message> {
        loop {
            match self.events.try_recv().ok()? {
                WsEventKind::Message(m) => return Some(m),
                _ => continue,
            }
        }
    }

    /// All messages received so far.
    pub fn drain(&self) -> Vec<Message> {
        std::iter::from_fn(|| self.try_recv()).collect()
    }

    /// Next lifecycle or message event. Never blocks.
    pub fn poll_event(&self) -> Option<WsEventKind> {
        self.events.try_recv().ok()
    }

    /// Close gracefully (buffered messages are flushed first). Idempotent.
    pub fn close(&self) {
        let _ = self.close.try_send(());
    }
}

impl Drop for WsConnection {
    fn drop(&mut self) {
        self.close();
    }
}

/// WebSocket client resource. Owns engine-managed connections whose traffic is
/// published as `Events<WsEvent>` by [`crate::NetPlugin`].
///
/// ```no_run
/// use space3d_core::prelude::*;
/// use space3d_net::{NetPlugin, WsClient, WsEvent, WsEventKind};
///
/// let mut app = App::new();
/// app.add_plugin(NetPlugin::default());
/// let id = app.resources.expect_mut::<WsClient>().open("ws://127.0.0.1:9000/");
/// let mut reader = EventReader::default();
/// app.add_system(Stage::Update, move |ctx| {
///     let events = ctx.resources.expect::<Events<WsEvent>>();
///     for ev in reader.read(events) {
///         if let WsEventKind::Message(msg) = &ev.kind {
///             println!("{:?}: {:?}", ev.connection, msg);
///         }
///     }
/// });
/// # let _ = id;
/// ```
#[derive(Default, Debug)]
pub struct WsClient {
    connections: BTreeMap<ConnectionId, WsConnection>,
}

impl WsClient {
    /// Open a standalone connection that you poll yourself.
    pub fn connect(url: impl Into<String>) -> WsConnection {
        Self::connect_with(url, WsConfig::default())
    }

    /// Open a standalone connection with options.
    pub fn connect_with(url: impl Into<String>, config: WsConfig) -> WsConnection {
        let url = url.into();
        let id = ConnectionId(NEXT_CONNECTION.fetch_add(1, Ordering::Relaxed));
        let (out_tx, out_rx) = async_channel::unbounded();
        let (close_tx, close_rx) = async_channel::bounded(1);
        let (ev_tx, ev_rx) = async_channel::unbounded();
        let state = Arc::new(Mutex::new(WsState::Connecting));
        let driver = Driver {
            url: url.clone(),
            config,
            state: state.clone(),
            outgoing: out_rx,
            close: close_rx,
            events: ev_tx,
        };
        // The task handle is intentionally dropped: the driver owns its lifetime
        // and exits when the connection handle is dropped or closed.
        let _ = space3d_tasks::spawn(driver.run());
        WsConnection {
            id,
            url,
            state,
            outgoing: out_tx,
            close: close_tx,
            events: ev_rx,
        }
    }

    /// Open an engine-managed connection; its traffic arrives as `Events<WsEvent>`.
    pub fn open(&mut self, url: impl Into<String>) -> ConnectionId {
        self.insert(Self::connect(url))
    }

    /// Open an engine-managed connection with options.
    pub fn open_with(&mut self, url: impl Into<String>, config: WsConfig) -> ConnectionId {
        self.insert(Self::connect_with(url, config))
    }

    /// Hand an existing connection over to the engine.
    pub fn insert(&mut self, conn: WsConnection) -> ConnectionId {
        let id = conn.id();
        self.connections.insert(id, conn);
        id
    }

    /// Look up a managed connection.
    pub fn get(&self, id: ConnectionId) -> Option<&WsConnection> {
        self.connections.get(&id)
    }

    /// State of a managed connection.
    pub fn state(&self, id: ConnectionId) -> Option<WsState> {
        self.get(id).map(WsConnection::state)
    }

    /// Queue a message on a managed connection.
    pub fn send(&self, id: ConnectionId, msg: impl Into<Message>) -> Result<(), NetError> {
        self.get(id).ok_or(NetError::UnknownPeer(id.0))?.send(msg)
    }

    /// Close a managed connection (it stays registered until its final event is emitted and it is removed).
    pub fn close(&self, id: ConnectionId) {
        if let Some(c) = self.get(id) {
            c.close();
        }
    }

    /// Unregister (and thereby close) a managed connection.
    pub fn remove(&mut self, id: ConnectionId) -> Option<WsConnection> {
        self.connections.remove(&id)
    }

    /// All managed connections, in id order.
    pub fn iter(&self) -> impl Iterator<Item = &WsConnection> {
        self.connections.values()
    }

    /// Number of managed connections.
    pub fn len(&self) -> usize {
        self.connections.len()
    }

    /// Whether no connections are managed.
    pub fn is_empty(&self) -> bool {
        self.connections.is_empty()
    }
}

/// The `WsEvents` system: drains every [`WsClient`] connection into `Events<WsEvent>`.
/// Registered by [`crate::NetPlugin`] in `Stage::PreUpdate`.
pub fn ws_events_system(ctx: &mut SystemCtx) {
    let Some(client) = ctx.resources.get::<WsClient>() else {
        return;
    };
    let mut out = Vec::new();
    for conn in client.connections.values() {
        while let Some(kind) = conn.poll_event() {
            out.push(WsEvent {
                connection: conn.id,
                kind,
            });
        }
    }
    if !out.is_empty() {
        let events = ctx.resources.init::<Events<WsEvent>>();
        for e in out {
            events.send(e);
        }
    }
}

/// What the socket backend reports.
pub(crate) enum Incoming {
    Message(Message),
    Closed(String),
}

enum Step {
    Close,
    Outgoing(Option<Message>),
    Incoming(Incoming),
}

enum PumpEnd {
    ClosedByClient,
    Lost(String),
}

const CLOSED_BY_CLIENT: &str = "closed by client";

/// The web: unread events a connection keeps before it is dropped (see [`Driver::pump`]); at a
/// game's 20 to 30 messages a second, about half a minute of a hidden tab.
#[cfg(target_arch = "wasm32")]
const WEB_MAX_UNREAD: usize = 1024;

/// Background state machine shared by native and web: connect, pump, reconnect.
struct Driver {
    url: String,
    config: WsConfig,
    state: Arc<Mutex<WsState>>,
    outgoing: async_channel::Receiver<Message>,
    close: async_channel::Receiver<()>,
    events: async_channel::Sender<WsEventKind>,
}

impl Driver {
    fn set_state(&self, s: WsState) {
        *self.state.lock() = s;
    }

    fn emit(&self, e: WsEventKind) {
        let _ = self.events.try_send(e);
    }

    fn finish(&self, reason: String) {
        self.set_state(WsState::Closed(reason.clone()));
        self.emit(WsEventKind::Closed {
            reason,
            reconnecting: false,
        });
    }

    async fn run(self) {
        let mut attempt = 0u32;
        loop {
            self.set_state(WsState::Connecting);
            let connect = rt::timeout(
                self.config.connect_timeout,
                Socket::connect(&self.url, self.config.max_message_size, &self.config.headers),
            );
            let (reason, was_open) = match rt::race(connect, self.close.recv()).await {
                rt::Race::Stopped => return self.finish(CLOSED_BY_CLIENT.into()),
                rt::Race::Done(Err(e)) => (e.to_string(), false),
                rt::Race::Done(Ok(Err(e))) => (e, false),
                rt::Race::Done(Ok(Ok(mut sock))) => {
                    attempt = 0;
                    self.set_state(WsState::Open);
                    self.emit(WsEventKind::Opened);
                    match self.pump(&mut sock).await {
                        PumpEnd::ClosedByClient => {
                            sock.close().await;
                            return self.finish(CLOSED_BY_CLIENT.into());
                        }
                        PumpEnd::Lost(reason) => (reason, true),
                    }
                }
            };
            let policy = match self.config.reconnect {
                Some(p) if p.allows(attempt) => p,
                _ => return self.finish(reason),
            };
            tracing::debug!(url = %self.url, %reason, attempt, "websocket reconnecting");
            if was_open {
                self.emit(WsEventKind::Closed {
                    reason,
                    reconnecting: true,
                });
            }
            self.set_state(WsState::Connecting);
            let delay = policy.backoff.delay(attempt);
            attempt = attempt.saturating_add(1);
            if let rt::Race::Stopped = rt::race(rt::sleep(delay), self.close.recv()).await {
                return self.finish(CLOSED_BY_CLIENT.into());
            }
        }
    }

    async fn pump(&self, sock: &mut Socket) -> PumpEnd {
        let mut incoming_first = true;
        loop {
            let step = {
                let mut close = pin!(self.close.recv());
                let mut out = pin!(self.outgoing.recv());
                let mut inc = pin!(sock.recv());
                // Alternate priority so neither direction can starve the other.
                incoming_first = !incoming_first;
                std::future::poll_fn(|cx| {
                    if close.as_mut().poll(cx).is_ready() {
                        return Poll::Ready(Step::Close);
                    }
                    if incoming_first && let Poll::Ready(i) = inc.as_mut().poll(cx) {
                        return Poll::Ready(Step::Incoming(i));
                    }
                    if let Poll::Ready(o) = out.as_mut().poll(cx) {
                        return Poll::Ready(Step::Outgoing(o.ok()));
                    }
                    if !incoming_first && let Poll::Ready(i) = inc.as_mut().poll(cx) {
                        return Poll::Ready(Step::Incoming(i));
                    }
                    Poll::Pending
                })
                .await
            };
            match step {
                Step::Close | Step::Outgoing(None) => {
                    // Flush what the game queued before asking to close.
                    while let Ok(m) = self.outgoing.try_recv() {
                        if sock.send(m).await.is_err() {
                            break;
                        }
                    }
                    return PumpEnd::ClosedByClient;
                }
                Step::Outgoing(Some(m)) => {
                    if let Err(e) = sock.send(m).await {
                        return PumpEnd::Lost(e);
                    }
                }
                Step::Incoming(Incoming::Message(m)) => {
                    // The web: a hidden tab stops the frames that read the events, while the
                    // socket's callbacks keep filling them. Past a cap the connection is dropped
                    // (the game reconnects when it reads again) instead of growing without end.
                    #[cfg(target_arch = "wasm32")]
                    if self.events.len() >= WEB_MAX_UNREAD {
                        return PumpEnd::Lost(format!(
                            "{WEB_MAX_UNREAD} messages unread (the page stopped reading: a hidden tab?)"
                        ));
                    }
                    self.emit(WsEventKind::Message(m))
                }
                Step::Incoming(Incoming::Closed(reason)) => return PumpEnd::Lost(reason),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_conversions() {
        let t: Message = "hi".into();
        assert_eq!(t.as_text(), Some("hi"));
        assert_eq!(t.len(), 2);
        let b: Message = vec![1u8, 2, 3].into();
        assert!(!b.is_text());
        assert_eq!(b.as_bytes(), &[1, 2, 3]);
        assert_eq!(Message::Text("ab".into()).into_bytes(), b"ab");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn failed_connect_without_reconnect_closes() {
        // Port 1 on localhost is essentially never listening → immediate refusal.
        let conn = WsClient::connect("ws://127.0.0.1:1/");
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while !conn.is_closed() {
            assert!(std::time::Instant::now() < deadline, "never closed");
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(conn.send("x"), Err(NetError::Closed));
        let last = std::iter::from_fn(|| conn.poll_event()).last();
        assert!(matches!(
            last,
            Some(WsEventKind::Closed {
                reconnecting: false,
                ..
            })
        ));
    }
}
