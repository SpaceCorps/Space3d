//! WebSocket server: accept clients, per-client ids, broadcast, inbound event queue.

use crate::{Message, NetError};
use axum::extract::ws::{CloseFrame, Message as AxMessage, WebSocket, WebSocketUpgrade, close_code};
use axum::extract::{ConnectInfo, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::MethodRouter;
use futures::{SinkExt, StreamExt};
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// Server-assigned id of a connected WebSocket client (unique per [`WsServer`], never reused).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ClientId(pub u64);

/// Decides whether an upgrade request may become a WebSocket client.
/// `Err(status)` rejects it with that HTTP status (e.g. `403`, `429`).
pub type AcceptHook = Arc<dyn Fn(&ConnectMeta) -> Result<(), u16> + Send + Sync>;

/// Limits and admission policy of a [`WsServer`] (see [`WsServer::with_config`]).
///
/// Every queue is bounded so a slow or hostile peer cannot grow server memory without
/// limit. The defaults suit general-purpose servers; game endpoints should use much
/// tighter values (the `space3d-server` gateway uses a 16 KiB `max_message_size`).
#[derive(Clone)]
pub struct WsServerConfig {
    /// Largest accepted inbound message in bytes. A bigger message closes the
    /// connection with `1009 message too big`. Default 1 MiB (tungstenite's own default is 64 MiB).
    pub max_message_size: usize,
    /// Largest accepted inbound frame in bytes. Default: `max_message_size`.
    pub max_frame_size: usize,
    /// Bytes queued for one client but not yet written to its socket. A send that
    /// would exceed it fails with [`NetError::Closed`] and closes the client with
    /// `1008 slow consumer`. Default 16 MiB.
    pub outbound_limit_bytes: usize,
    /// Inbound messages waiting to be polled (all clients together). A message arriving
    /// while the inbox is full is dropped and its sender is closed with `1008 inbox full`.
    /// Lifecycle events never count. Default 65 536.
    pub inbound_queue: usize,
    /// Concurrent clients; upgrades beyond it get `503`. Default 100 000.
    pub max_connections: usize,
    /// Accept text frames. When `false`, a text frame closes the client with `1003`. Default `true`.
    pub accept_text: bool,
    /// Optional admission hook run before the upgrade (IP bans, per-IP caps, `Origin` checks).
    pub accept: Option<AcceptHook>,
}

impl Default for WsServerConfig {
    fn default() -> Self {
        Self {
            max_message_size: 1 << 20,
            max_frame_size: 1 << 20,
            outbound_limit_bytes: 16 << 20,
            inbound_queue: 65_536,
            max_connections: 100_000,
            accept_text: true,
            accept: None,
        }
    }
}

impl std::fmt::Debug for WsServerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WsServerConfig")
            .field("max_message_size", &self.max_message_size)
            .field("max_frame_size", &self.max_frame_size)
            .field("outbound_limit_bytes", &self.outbound_limit_bytes)
            .field("inbound_queue", &self.inbound_queue)
            .field("max_connections", &self.max_connections)
            .field("accept_text", &self.accept_text)
            .field("accept", &self.accept.is_some())
            .finish()
    }
}

impl WsServerConfig {
    /// Set both the message and the frame size limit.
    pub fn max_message_size(mut self, bytes: usize) -> Self {
        self.max_message_size = bytes;
        self.max_frame_size = bytes;
        self
    }
    /// Set the per-client outbound backlog limit.
    pub fn outbound_limit_bytes(mut self, bytes: usize) -> Self {
        self.outbound_limit_bytes = bytes;
        self
    }
    /// Set the shared inbound queue bound.
    pub fn inbound_queue(mut self, messages: usize) -> Self {
        self.inbound_queue = messages;
        self
    }
    /// Set the concurrent connection limit.
    pub fn max_connections(mut self, n: usize) -> Self {
        self.max_connections = n;
        self
    }
    /// Accept or reject text frames.
    pub fn accept_text(mut self, yes: bool) -> Self {
        self.accept_text = yes;
        self
    }
    /// Install an admission hook.
    pub fn accept(mut self, hook: impl Fn(&ConnectMeta) -> Result<(), u16> + Send + Sync + 'static) -> Self {
        self.accept = Some(Arc::new(hook));
        self
    }
}

/// What the server knows about a client from its upgrade request.
#[derive(Clone, PartialEq, Eq)]
pub struct ConnectMeta {
    /// TCP peer address (a proxy's address when behind one; see `forwarded_for`).
    pub addr: SocketAddr,
    /// `Origin` header (browsers always send it; use it against cross-site WebSocket hijacking).
    pub origin: Option<String>,
    /// `X-Forwarded-For` header, verbatim. Only trust it when `addr` is a known proxy.
    pub forwarded_for: Option<String>,
    /// `User-Agent` header.
    pub user_agent: Option<String>,
    /// `Authorization` header (native clients can authenticate the upgrade itself; browsers
    /// cannot set it). Redacted in `Debug`.
    pub authorization: Option<String>,
}

impl std::fmt::Debug for ConnectMeta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConnectMeta")
            .field("addr", &self.addr)
            .field("origin", &self.origin)
            .field("forwarded_for", &self.forwarded_for)
            .field("user_agent", &self.user_agent)
            .field("authorization", &self.authorization.as_ref().map(|_| "<redacted>"))
            .finish()
    }
}

/// Lock-free sender for one connected client (see [`WsServer::client_sender`]): sending,
/// backlog and liveness checks touch only this client's queue and counters, never the
/// server-wide client table, so senders on many threads do not contend.
#[derive(Clone)]
pub struct ClientSender {
    tx: mpsc::UnboundedSender<Outgoing>,
    queued: Arc<AtomicUsize>,
    closed: Arc<AtomicBool>,
    outbound_limit: usize,
}

impl std::fmt::Debug for ClientSender {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClientSender")
            .field("queued", &self.queued.load(Ordering::Relaxed))
            .field("closed", &self.is_closed())
            .finish()
    }
}

impl ClientSender {
    /// Queue `msg` (same limits as [`WsServer::send`]).
    pub fn send(&self, msg: impl Into<Message>) -> Result<(), NetError> {
        if self.is_closed() {
            return Err(NetError::Closed);
        }
        enqueue(&self.tx, &self.queued, self.outbound_limit, msg.into())
    }

    /// Bytes queued but not yet written to the socket.
    pub fn backlog_bytes(&self) -> usize {
        self.queued.load(Ordering::Acquire)
    }

    /// Close the connection with `code` (after flushing queued messages). From now on the
    /// sender reports closed and refuses further sends.
    pub fn close_with(&self, code: u16, reason: impl Into<String>) {
        let _ = self.tx.send(Outgoing::Close(code, reason.into()));
        self.closed.store(true, Ordering::Release);
    }

    /// Whether the connection ended (or is ending: a close was queued by the server, the
    /// socket errored, or the client left).
    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire) || self.tx.is_closed()
    }

    /// Whether the client is still connected (`!is_closed`).
    pub fn is_connected(&self) -> bool {
        !self.is_closed()
    }
}

/// Queue one message against a client's outbound limit.
fn enqueue(
    tx: &mpsc::UnboundedSender<Outgoing>,
    queued: &AtomicUsize,
    limit: usize,
    msg: Message,
) -> Result<(), NetError> {
    let len = msg.len();
    let before = queued.fetch_add(len, Ordering::AcqRel);
    if before + len > limit {
        queued.fetch_sub(len, Ordering::AcqRel);
        let _ = tx.send(Outgoing::Close(CLOSE_POLICY, "slow consumer".into()));
        return Err(NetError::Closed);
    }
    tx.send(Outgoing::Message(msg)).map_err(|_| {
        queued.fetch_sub(len, Ordering::AcqRel);
        NetError::Closed
    })
}

/// Inbound traffic of a [`WsServer`], polled by engine systems
/// ([`WsServer::try_recv`] / [`WsServer::drain`], or `Events<ServerEvent>` when the
/// server is inserted as a resource alongside [`crate::NetPlugin`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ServerEvent {
    /// A client finished the handshake. It can be sent to immediately.
    /// `Connected` events are delivered in [`ClientId`] order.
    Connected {
        /// The new client.
        client: ClientId,
        /// Its remote address.
        addr: SocketAddr,
    },
    /// A client sent a message.
    Message {
        /// Sender.
        client: ClientId,
        /// Payload.
        message: Message,
    },
    /// A client left (closed, errored, kicked, or server shutdown).
    Disconnected {
        /// The client that left.
        client: ClientId,
        /// Human-readable reason.
        reason: String,
    },
}

impl ServerEvent {
    /// The client this event is about.
    pub fn client(&self) -> ClientId {
        match self {
            ServerEvent::Connected { client, .. }
            | ServerEvent::Message { client, .. }
            | ServerEvent::Disconnected { client, .. } => *client,
        }
    }
}

enum Outgoing {
    Message(Message),
    Close(u16, String),
}

struct Slot {
    tx: mpsc::UnboundedSender<Outgoing>,
    meta: ConnectMeta,
    /// Bytes queued in `tx` but not yet written to the socket.
    queued: Arc<AtomicUsize>,
    /// Set when the socket task ends (shared with [`ClientSender`]s).
    closed: Arc<AtomicBool>,
}

struct Inner {
    config: WsServerConfig,
    clients: Mutex<BTreeMap<ClientId, Slot>>,
    events_tx: async_channel::Sender<ServerEvent>,
    events_rx: async_channel::Receiver<ServerEvent>,
    /// `ServerEvent::Message`s currently queued (bounded by `config.inbound_queue`).
    inbound: AtomicUsize,
    next_id: AtomicU64,
}

const CLOSE_TOO_BIG: u16 = 1009;
const CLOSE_POLICY: u16 = 1008;
const CLOSE_UNSUPPORTED: u16 = 1003;

/// WebSocket server state, shareable between the network runtime and engine systems.
///
/// Mount it on an [`crate::HttpServer`] with [`crate::HttpServer::ws`], or run it
/// standalone with [`WsServer::listen`]. Cloning shares the same clients and inbox;
/// it can also be inserted as a resource (see [`ServerEvent`]).
///
/// All methods are non-blocking. Inbound events queue until polled.
#[derive(Clone)]
pub struct WsServer {
    inner: Arc<Inner>,
}

impl Default for WsServer {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for WsServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WsServer")
            .field("clients", &self.client_count())
            .field("queued_events", &self.inner.events_rx.len())
            .finish()
    }
}

impl WsServer {
    /// A server with no clients yet and default [`WsServerConfig`] limits.
    pub fn new() -> Self {
        Self::with_config(WsServerConfig::default())
    }

    /// A server with explicit limits and admission policy.
    pub fn with_config(config: WsServerConfig) -> Self {
        // Lifecycle events are few (bounded by connections); message events are counted
        // against `config.inbound_queue` separately, so the channel itself is unbounded.
        let (events_tx, events_rx) = async_channel::unbounded();
        Self {
            inner: Arc::new(Inner {
                config,
                clients: Mutex::new(BTreeMap::new()),
                events_tx,
                events_rx,
                inbound: AtomicUsize::new(0),
                next_id: AtomicU64::new(1),
            }),
        }
    }

    /// The limits this server was created with.
    pub fn config(&self) -> &WsServerConfig {
        &self.inner.config
    }

    /// Serve this WebSocket server on `addr` (any path). Port `0` picks a free port.
    pub fn listen(&self, addr: impl Into<SocketAddr>) -> Result<super::ServerHandle, NetError> {
        super::HttpServer::new()
            .bind(addr)
            .ws("/", self)
            .ws("/{*path}", self)
            .start()
    }

    /// Queue `msg` for one client. Never blocks.
    ///
    /// Fails with [`NetError::Closed`] (and closes the client with `1008 slow consumer`)
    /// if the message would push its backlog over [`WsServerConfig::outbound_limit_bytes`].
    pub fn send(&self, client: ClientId, msg: impl Into<Message>) -> Result<(), NetError> {
        let clients = self.inner.clients.lock();
        let slot = clients.get(&client).ok_or(NetError::UnknownPeer(client.0))?;
        self.enqueue(slot, msg.into())
    }

    fn enqueue(&self, slot: &Slot, msg: Message) -> Result<(), NetError> {
        enqueue(&slot.tx, &slot.queued, self.inner.config.outbound_limit_bytes, msg)
    }

    /// A lock-free sender for `client` (one table lookup now; none per send). `None` if the
    /// client is not connected.
    pub fn client_sender(&self, client: ClientId) -> Option<ClientSender> {
        self.inner.clients.lock().get(&client).map(|slot| ClientSender {
            tx: slot.tx.clone(),
            queued: slot.queued.clone(),
            closed: slot.closed.clone(),
            outbound_limit: self.inner.config.outbound_limit_bytes,
        })
    }

    /// Queue `msg` for every connected client; returns how many it was queued for.
    pub fn broadcast(&self, msg: impl Into<Message>) -> usize {
        self.broadcast_filtered(msg.into(), |_| true)
    }

    /// Queue `msg` for every client except `except`.
    pub fn broadcast_except(&self, except: ClientId, msg: impl Into<Message>) -> usize {
        self.broadcast_filtered(msg.into(), |id| id != except)
    }

    fn broadcast_filtered(&self, msg: Message, keep: impl Fn(ClientId) -> bool) -> usize {
        let clients = self.inner.clients.lock();
        clients
            .iter()
            .filter(|(id, _)| keep(**id))
            .filter(|(_, slot)| self.enqueue(slot, msg.clone()).is_ok())
            .count()
    }

    /// Close one client's connection (after flushing its queued messages).
    pub fn disconnect(&self, client: ClientId, reason: impl Into<String>) {
        self.disconnect_with(client, close_code::NORMAL, reason);
    }

    /// Close one client's connection with an explicit WebSocket close code
    /// (e.g. `1008` policy violation, `1001` going away).
    pub fn disconnect_with(&self, client: ClientId, code: u16, reason: impl Into<String>) {
        if let Some(slot) = self.inner.clients.lock().get(&client) {
            let _ = slot.tx.send(Outgoing::Close(code, reason.into()));
        }
    }

    /// Close every client connection. The server keeps accepting new ones.
    pub fn disconnect_all(&self, reason: impl Into<String>) {
        let reason = reason.into();
        for slot in self.inner.clients.lock().values() {
            let _ = slot.tx.send(Outgoing::Close(close_code::NORMAL, reason.clone()));
        }
    }

    /// Currently connected clients, in id order.
    pub fn clients(&self) -> Vec<ClientId> {
        self.inner.clients.lock().keys().copied().collect()
    }

    /// Number of connected clients.
    pub fn client_count(&self) -> usize {
        self.inner.clients.lock().len()
    }

    /// Whether `client` is connected.
    pub fn is_connected(&self, client: ClientId) -> bool {
        self.inner.clients.lock().contains_key(&client)
    }

    /// Remote address of a connected client.
    pub fn client_addr(&self, client: ClientId) -> Option<SocketAddr> {
        self.inner.clients.lock().get(&client).map(|s| s.meta.addr)
    }

    /// Upgrade-request metadata (address, `Origin`, `X-Forwarded-For`, `User-Agent`) of a connected client.
    pub fn client_meta(&self, client: ClientId) -> Option<ConnectMeta> {
        self.inner.clients.lock().get(&client).map(|s| s.meta.clone())
    }

    /// Bytes queued for `client` that its socket has not accepted yet (flow control:
    /// skip optional traffic while this grows). `None` if the client is not connected.
    pub fn backlog_bytes(&self, client: ClientId) -> Option<usize> {
        self.inner
            .clients
            .lock()
            .get(&client)
            .map(|s| s.queued.load(Ordering::Acquire))
    }

    /// Inbound messages currently waiting to be polled.
    pub fn inbound_len(&self) -> usize {
        self.inner.inbound.load(Ordering::Acquire)
    }

    /// Next inbound event, if any. Never blocks.
    pub fn try_recv(&self) -> Option<ServerEvent> {
        let ev = self.inner.events_rx.try_recv().ok()?;
        self.note_dequeued(&ev);
        Some(ev)
    }

    /// Wait for the next inbound event (for consumers running on the async runtime, e.g. a
    /// gateway task). `None` once every handle to the server is gone. Never call from a system.
    pub async fn recv(&self) -> Option<ServerEvent> {
        let ev = self.inner.events_rx.recv().await.ok()?;
        self.note_dequeued(&ev);
        Some(ev)
    }

    fn note_dequeued(&self, ev: &ServerEvent) {
        if matches!(ev, ServerEvent::Message { .. }) {
            self.inner.inbound.fetch_sub(1, Ordering::AcqRel);
        }
    }

    /// Queue an inbound message; `false` if the inbox is full (the caller closes the sender).
    fn push_inbound(&self, client: ClientId, message: Message) -> bool {
        let prev = self.inner.inbound.fetch_add(1, Ordering::AcqRel);
        if prev >= self.inner.config.inbound_queue {
            self.inner.inbound.fetch_sub(1, Ordering::AcqRel);
            return false;
        }
        if self
            .inner
            .events_tx
            .try_send(ServerEvent::Message { client, message })
            .is_err()
        {
            self.inner.inbound.fetch_sub(1, Ordering::AcqRel);
        }
        true
    }

    /// All queued inbound events, oldest first.
    pub fn drain(&self) -> Vec<ServerEvent> {
        std::iter::from_fn(|| self.try_recv()).collect()
    }

    /// axum route that upgrades to a WebSocket and runs the client until it
    /// leaves or `shutdown` (the owning HTTP server's token) fires.
    pub(crate) fn upgrade_route(&self, shutdown: CancellationToken) -> MethodRouter {
        axum::routing::get(upgrade).with_state((self.clone(), shutdown))
    }

    async fn run_client(self, socket: WebSocket, meta: ConnectMeta, shutdown: CancellationToken) {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let queued = Arc::new(AtomicUsize::new(0));
        let closed = Arc::new(AtomicBool::new(false));
        let addr = meta.addr;
        let id = {
            // Allocate, register and announce atomically: `Connected` events arrive in
            // id order, and the client is sendable by the time anyone sees the event.
            let mut clients = self.inner.clients.lock();
            let id = ClientId(self.inner.next_id.fetch_add(1, Ordering::Relaxed));
            clients.insert(
                id,
                Slot {
                    tx,
                    meta,
                    queued: queued.clone(),
                    closed: closed.clone(),
                },
            );
            let _ = self
                .inner
                .events_tx
                .try_send(ServerEvent::Connected { client: id, addr });
            id
        };
        tracing::debug!(client = id.0, %addr, "websocket client connected");

        let (mut sink, mut stream) = socket.split();
        // Set when this side sent the Close frame: the socket then lingers (see `linger`).
        let mut sent_close = false;
        let reason = loop {
            tokio::select! {
                _ = shutdown.cancelled() => {
                    let frame = CloseFrame { code: close_code::AWAY, reason: "server shutting down".into() };
                    let _ = sink.send(AxMessage::Close(Some(frame))).await;
                    sent_close = true;
                    break "server shutting down".to_owned();
                }
                out = rx.recv() => match out {
                    Some(Outgoing::Message(m)) => {
                        let len = m.len();
                        let m = match m {
                            Message::Text(s) => AxMessage::Text(s.into()),
                            Message::Binary(b) => AxMessage::Binary(b.into()),
                        };
                        let sent = sink.send(m).await;
                        queued.fetch_sub(len, Ordering::AcqRel);
                        if let Err(e) = sent {
                            break format!("send failed: {e}");
                        }
                    }
                    Some(Outgoing::Close(code, reason)) => {
                        let frame = CloseFrame { code, reason: reason.clone().into() };
                        let _ = sink.send(AxMessage::Close(Some(frame))).await;
                        sent_close = true;
                        break reason;
                    }
                    None => break "server dropped".to_owned(),
                },
                inc = stream.next() => match inc {
                    Some(Ok(AxMessage::Text(t))) => {
                        if !self.inner.config.accept_text {
                            let frame = CloseFrame { code: CLOSE_UNSUPPORTED, reason: "text frames not accepted".into() };
                            let _ = sink.send(AxMessage::Close(Some(frame))).await;
                            sent_close = true;
                            break "text frame rejected".to_owned();
                        }
                        if !self.push_inbound(id, Message::Text(t.as_str().to_owned())) {
                            let frame = CloseFrame { code: CLOSE_POLICY, reason: "inbox full".into() };
                            let _ = sink.send(AxMessage::Close(Some(frame))).await;
                            sent_close = true;
                            break "inbox full".to_owned();
                        }
                    }
                    Some(Ok(AxMessage::Binary(b))) => {
                        if !self.push_inbound(id, Message::Binary(b.to_vec())) {
                            let frame = CloseFrame { code: CLOSE_POLICY, reason: "inbox full".into() };
                            let _ = sink.send(AxMessage::Close(Some(frame))).await;
                            sent_close = true;
                            break "inbox full".to_owned();
                        }
                    }
                    Some(Ok(AxMessage::Ping(_) | AxMessage::Pong(_))) => {}
                    Some(Ok(AxMessage::Close(frame))) => {
                        break match frame {
                            Some(f) if !f.reason.is_empty() => format!("closed by client ({}): {}", f.code, f.reason),
                            Some(f) => format!("closed by client ({})", f.code),
                            None => "closed by client".to_owned(),
                        };
                    }
                    Some(Err(e)) => {
                        let text = crate::error::error_chain(&e);
                        if text.contains("too long") || text.contains("Space limit") {
                            let frame = CloseFrame { code: CLOSE_TOO_BIG, reason: "message too big".into() };
                            let _ = sink.send(AxMessage::Close(Some(frame))).await;
                            sent_close = true;
                            break format!("message too big: {text}");
                        }
                        break format!("connection error: {text}");
                    }
                    None => break "connection closed".to_owned(),
                },
            }
        };
        // From here on the connection reports closed and refuses sends, but the socket stays open
        // until the peer has answered the Close frame (or `CLOSE_LINGER` ran out).
        closed.store(true, Ordering::Release);
        if sent_close {
            linger(&mut stream).await;
        }
        tracing::debug!(client = id.0, %reason, "websocket client disconnected");
        let mut clients = self.inner.clients.lock();
        clients.remove(&id);
        let _ = self
            .inner
            .events_tx
            .try_send(ServerEvent::Disconnected { client: id, reason });
    }
}

/// How long a socket that this side closed waits for the peer's answer.
const CLOSE_LINGER: Duration = Duration::from_secs(1);

/// After this side sent its Close frame: read (and drop) what the peer still sends until its own
/// Close frame, EOF or an error. Closing a TCP socket that holds unread data makes the OS send a
/// reset instead of an orderly FIN, and a reset discards what the peer has received but not yet
/// read: a client that keeps sending inputs (every client of a game server does) lost the very
/// `Disconnect` that the server queued before closing, and saw `connection aborted/reset` instead.
async fn linger<S>(stream: &mut S)
where
    S: futures::Stream<Item = Result<AxMessage, axum::Error>> + Unpin,
{
    let _ = tokio::time::timeout(CLOSE_LINGER, async {
        while let Some(Ok(message)) = stream.next().await {
            if matches!(message, AxMessage::Close(_)) {
                break;
            }
        }
    })
    .await;
}

fn header_string(headers: &HeaderMap, name: header::HeaderName) -> Option<String> {
    headers.get(name).and_then(|v| v.to_str().ok()).map(str::to_owned)
}

/// Every line of a list-valued header (e.g. `X-Forwarded-For`), in order, joined with `", "`
/// (RFC 9110: repeated field lines equal one comma-separated list). Proxies may add their own
/// line instead of appending to the client's, so reading only the first line would hand the
/// client the choice of address. Non-UTF-8 bytes are replaced (an unparseable hop, never a
/// dropped one).
pub fn joined_header(headers: &HeaderMap, name: &str) -> Option<String> {
    let lines: Vec<String> = headers
        .get_all(name)
        .iter()
        .map(|v| String::from_utf8_lossy(v.as_bytes()).into_owned())
        .collect();
    (!lines.is_empty()).then(|| lines.join(", "))
}

async fn upgrade(
    State((server, shutdown)): State<(WsServer, CancellationToken)>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    upgrade: WebSocketUpgrade,
) -> Response {
    if shutdown.is_cancelled() {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    }
    let config = &server.inner.config;
    if server.client_count() >= config.max_connections {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    }
    let meta = ConnectMeta {
        addr,
        origin: header_string(&headers, header::ORIGIN),
        forwarded_for: joined_header(&headers, "x-forwarded-for"),
        user_agent: header_string(&headers, header::USER_AGENT),
        authorization: header_string(&headers, header::AUTHORIZATION),
    };
    if let Some(hook) = &config.accept
        && let Err(status) = hook(&meta)
    {
        return StatusCode::from_u16(status)
            .unwrap_or(StatusCode::FORBIDDEN)
            .into_response();
    }
    upgrade
        .max_message_size(config.max_message_size)
        .max_frame_size(config.max_frame_size)
        .on_upgrade(move |socket| server.run_client(socket, meta, shutdown))
}
