//! Full-stack, in-process test cluster (feature `testing`): **one player service and N game
//! servers as separate supervisor `App`s** (the split deployment, wired with in-process fleet
//! links instead of the fleet WebSocket), rooms stepped directly on the test thread (no shard
//! threads), clients as real `ClientReplicationPlugin` apps over one [`MemoryNetwork`] per game
//! server, a shared [`ManualClock`] + [`ManualUnixClock`], and in-memory databases.
//!
//! It exercises the real path end to end: guest login and matchmaking over the player-service
//! HTTP API (loopback, through `space3d_replication::services::PlayerServiceClient`) → signed
//! join ticket → gatekeeper → lazy room creation from the ticket's `RoomSpec` → replication →
//! results through the outbox and the fleet link → store; plus room crashes (`catch_unwind` +
//! rebuild from the in-memory checkpoint) and game-server loss (lease expiry → rejoin →
//! restore on another server from the checkpoint stored by the player service).
//!
//! ```ignore
//! let mut c = TestCluster::new(ArenaGame, TestClusterConfig { game_servers: 2, ..Default::default() });
//! let alice = c.guest("alice");
//! let bob = c.guest("bob");
//! let found = c.matchmake(&[&alice, &bob], "duel");         // runs matchmaker ticks
//! let ca = c.connect(&found[0], |app| { app.add_plugin(ArenaProtocol); });
//! let cb = c.connect(&found[1], |app| { app.add_plugin(ArenaProtocol); });
//! assert!(c.run_until(600, |c| c.client_net(ca).is_connected() && c.client_net(cb).is_connected()));
//! c.crash_room(found[0].room_id);                           // panic → rebuilt from checkpoint
//! ```
//!
//! Blocking (`Task::block`) is used for HTTP calls and checkpoint lookups: this is a test tool,
//! not frame code. Background work (database writes, checkpoint uploads, outbox acks) completes
//! asynchronously; wait for it with [`TestCluster::run_until`] / [`TestCluster::settle`].

use crate::clock::{ManualUnixClock, UnixClock};
use crate::config::{QueueConfig, Role, ServerConfig};
use crate::game::{GameDefinition, RoomSpec};
use crate::gs::GameServerHandle;
use crate::gs::checkpoint::{CheckpointStore, PlayerServiceCheckpoints, StoreError};
use crate::gs::fleet::{FleetLink, FleetOutage, in_process_pair};
use crate::gs::room::{build_room_app, panic_message, run_tick_guarded};
use crate::gs::rooms::{RoomEnv, RoomPhase, RoomStatus};
use crate::ps::{FleetHub, PlayerService, PlayerServiceHandle};
use crate::security::Secrets;
use crate::{BuildOptions, ServerMain};
use parking_lot::Mutex;
use space3d_core::{App, Stage, TickRate};
use space3d_net::Http;
use space3d_replication::api::{Assignment, MatchmakeRequest, QuickPlayRequest, SessionTokens, TicketStatus};
use space3d_replication::client::{ClientConfig, ClientReplicationPlugin, ConnectTarget, NetClient};
use space3d_replication::harness::{ClientHandle, ClientSpec};
use space3d_replication::protocol::handshake::HelloAuth;
use space3d_replication::services::{DeviceCredentials, OnlineError, PlayerServiceClient};
use space3d_replication::time::{ManualClock, NetClockHandle};
use space3d_replication::transport::conditioner::{LinkProfile, Semantics};
use space3d_replication::transport::memory::{MemoryNetwork, MemoryServerEndpoint};
use space3d_replication::transport::{ClientTransport, Connector, LinkEvent, TransportError};
use space3d_replication::{PlayerId, RoomId};
use space3d_tasks::Task;
use std::collections::BTreeMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Admin bearer token accepted by every cluster node.
pub const TEST_ADMIN_TOKEN: &str = "test-admin-token";

/// Cluster settings.
#[derive(Clone, Debug)]
pub struct TestClusterConfig {
    /// Base server config. Roles, binds (`127.0.0.1:0`), server ids and public URLs are set
    /// per node; everything else (queues, limits, tick rate, checkpoint cadence) is used as is.
    pub server: ServerConfig,
    /// Number of game servers (`gs-1`, `gs-2`, ...).
    pub game_servers: usize,
    /// Seed (memory networks, resume secrets).
    pub seed: u64,
    /// Default link of new clients.
    pub link: LinkProfile,
    /// Stream (WebSocket-like) or datagram semantics.
    pub semantics: Semantics,
    /// Client template (`game_id` and `connector` are filled in).
    pub client: ClientConfig,
    /// Start of the wall clock (unix ms).
    pub start_ms: i64,
    /// Run the supervisor apps (game servers and player service: fleet links, lifecycle,
    /// matchmaking) every this many steps, not every step. Production supervisors run at
    /// 10 Hz while rooms tick at 60 Hz (`6`): messages between the rooms, the fleet link and
    /// the player service then take as long to cross as they do for real, while clients react
    /// to their room at once.
    pub supervisor_every: u32,
}

impl Default for TestClusterConfig {
    fn default() -> Self {
        let mut server = ServerConfig::default();
        server.matchmaking.queues = vec![QueueConfig {
            name: "duel".into(),
            mode: "duel".into(),
            ..Default::default()
        }];
        // Every HTTP call comes from one loopback address: keep the per-IP HTTP limits out of
        // the way (tests of the limits set their own). Memory links get distinct synthetic IPs.
        server.player_service.http_rps_per_ip = 100_000;
        server.player_service.login_attempts_per_min = 100_000;
        server.persistence.checkpoint_interval = Duration::from_secs(1);
        Self {
            server,
            game_servers: 1,
            seed: 1,
            link: LinkProfile::default(),
            semantics: Semantics::Stream,
            client: ClientConfig::default(),
            start_ms: 1_800_000_000_000,
            supervisor_every: 1,
        }
    }
}

/// Memory connector that refuses connections once its game server was killed.
#[derive(Clone)]
struct NodeConnector {
    network: MemoryNetwork,
    profile: LinkProfile,
    alive: Arc<AtomicBool>,
}

impl Connector for NodeConnector {
    fn connect(&self, url: &str) -> Result<Box<dyn ClientTransport>, TransportError> {
        if !self.alive.load(Ordering::Acquire) {
            return Err(TransportError::Connect(format!("{url}: connection refused")));
        }
        self.network.connector(self.profile.clone()).connect(url)
    }
}

struct HostedRoom {
    app: App,
    spec: RoomSpec,
    env: RoomEnv,
    status: Arc<Mutex<RoomStatus>>,
    created_ms: i64,
    step: Duration,
    crashes: crate::gs::shard::CrashGuard,
}

struct GsNode {
    id: String,
    app: Option<App>,
    handle: GameServerHandle,
    network: MemoryNetwork,
    endpoint: MemoryServerEndpoint,
    alive: Arc<AtomicBool>,
    rooms: BTreeMap<RoomId, HostedRoom>,
    next_ip: u32,
    outage: FleetOutage,
}

struct ClientSlot {
    app: App,
    server: String,
}

/// The cluster. See the module docs.
pub struct TestCluster {
    config: TestClusterConfig,
    game_id: String,
    clock: ManualClock,
    net_clock: NetClockHandle,
    unix: ManualUnixClock,
    step: Duration,
    now: Duration,
    ps_app: App,
    ps: PlayerServiceHandle,
    http: Http,
    nodes: Vec<GsNode>,
    clients: Vec<ClientSlot>,
    crashes: u64,
    steps: u64,
    /// While set, checkpoint uploads of every game server wait ([`Self::hold_checkpoint_uploads`]).
    upload_hold: Arc<AtomicBool>,
}

/// Checkpoint store that holds its `put`s while the cluster's upload hold is set. The store
/// itself completes on background threads, in real time; a test that must observe the state
/// *during* a handoff cannot race that, so it holds the final upload instead and releases it
/// when it is done looking (a room frozen for a handoff closes once its capture is stored).
struct HeldCheckpoints {
    inner: Arc<dyn CheckpointStore>,
    hold: Arc<AtomicBool>,
}

impl CheckpointStore for HeldCheckpoints {
    fn put(&self, room: RoomId, bytes: Vec<u8>) -> Task<Result<(), StoreError>> {
        if !self.hold.load(Ordering::SeqCst) {
            return self.inner.put(room, bytes);
        }
        let (inner, hold) = (self.inner.clone(), self.hold.clone());
        space3d_tasks::spawn(async move {
            while hold.load(Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
            match inner.put(room, bytes).wait().await {
                Some(r) => r,
                None => Err(StoreError("worker gone".into())),
            }
        })
    }
    fn get(&self, room: RoomId) -> Task<Result<Option<Vec<u8>>, StoreError>> {
        self.inner.get(room)
    }
    fn recent(&self, min_saved_ms: i64) -> Task<Result<Vec<RoomId>, StoreError>> {
        self.inner.recent(min_saved_ms)
    }
    fn delete(&self, room: RoomId, through_epoch: u32) -> Task<Result<(), StoreError>> {
        self.inner.delete(room, through_epoch)
    }
    fn prune(&self, before_ms: i64) -> Task<Result<u64, StoreError>> {
        self.inner.prune(before_ms)
    }
}

fn loopback() -> SocketAddr {
    SocketAddr::from((Ipv4Addr::LOCALHOST, 0))
}

impl TestCluster {
    /// Build the player service and `config.game_servers` game servers, and step until every
    /// game server is registered.
    pub fn new<G: GameDefinition>(game: G, config: TestClusterConfig) -> Self {
        let game = Arc::new(game);
        let clock = ManualClock::new();
        let net_clock = NetClockHandle(Arc::new(clock.clone()));
        let unix = ManualUnixClock::at(config.start_ms);
        let secrets = Secrets::for_tests(crate::security::DEV_FLEET_TOKEN, &[TEST_ADMIN_TOKEN]);

        let mut ps_config = config.server.clone();
        ps_config.role = Role::PlayerService;
        ps_config.server_id = "ps".into();
        ps_config.player_service.bind = loopback();
        ps_config.admin.bind = loopback();
        let ps_app = ServerMain::with_shared_game(game.clone(), ps_config)
            .unix_clock(Arc::new(unix.clone()))
            .options(BuildOptions {
                net_clock: Some(net_clock.clone()),
                secrets: Some(secrets.clone()),
                db: Some(space3d_db::Db::open_in_memory()),
                ..Default::default()
            })
            .build()
            .expect("player service builds");
        let ps = ps_app.resources.expect::<PlayerServiceHandle>().clone();
        let tick_hz = config.server.game.tick_hz;
        let mut cluster = Self {
            game_id: game.game_id().to_owned(),
            clock,
            net_clock,
            unix,
            step: Duration::from_secs_f64(1.0 / tick_hz),
            now: Duration::ZERO,
            http: Http::new(),
            ps_app,
            ps,
            nodes: Vec::new(),
            clients: Vec::new(),
            crashes: 0,
            steps: 0,
            upload_hold: Arc::new(AtomicBool::new(false)),
            config,
        };
        for i in 0..cluster.config.game_servers.max(1) {
            let id = format!("gs-{}", i + 1);
            let seed = cluster.config.seed ^ (i as u64 + 1).wrapping_mul(0x9E37_79B9);
            cluster.add_game_server(game.clone(), &id, seed, &secrets);
        }
        let ids = cluster.game_servers();
        let registered = cluster.run_until(600, |c| {
            let fleet = c.ps.service.fleet.lock();
            ids.iter().all(|id| fleet.server(id).is_some())
                && c.live().all(|n| {
                    n.app
                        .as_ref()
                        .is_some_and(|a| a.resources.expect::<crate::health::HealthState>().is_registered())
                })
        });
        assert!(registered, "game servers did not register with the player service");
        cluster
    }

    fn add_game_server<G: GameDefinition>(&mut self, game: Arc<G>, id: &str, seed: u64, secrets: &Secrets) {
        let mut config = self.config.server.clone();
        config.role = Role::GameServer;
        config.server_id = id.to_owned();
        config.game.bind = loopback();
        config.admin.bind = loopback();
        config.game.public_url = format!("mem://{id}/play");
        let (link, peer) = in_process_pair();
        let outage = link.outage();
        self.ps_app.resources.expect_mut::<FleetHub>().add_local(peer);
        let checkpoints: Arc<dyn CheckpointStore> = Arc::new(HeldCheckpoints {
            inner: Arc::new(PlayerServiceCheckpoints {
                store: self.ps.service.store.clone(),
                clock: Arc::new(self.unix.clone()),
            }),
            hold: self.upload_hold.clone(),
        });
        let fleet: Arc<dyn FleetLink> = Arc::new(link);
        let app = ServerMain::with_shared_game(game, config)
            .unix_clock(Arc::new(self.unix.clone()))
            .options(BuildOptions {
                manual_rooms: true,
                net_clock: Some(self.net_clock.clone()),
                secret_seed: Some(seed),
                secrets: Some(secrets.clone()),
                db: Some(space3d_db::Db::open_in_memory()),
                fleet_link: Some(fleet),
                checkpoints: Some(checkpoints),
            })
            .build()
            .expect("game server builds");
        let handle = app.resources.expect::<GameServerHandle>().clone();
        let network = MemoryNetwork::new(seed, self.net_clock.clone(), self.config.semantics);
        self.nodes.push(GsNode {
            id: id.to_owned(),
            app: Some(app),
            handle,
            endpoint: network.server_endpoint(),
            network,
            alive: Arc::new(AtomicBool::new(true)),
            rooms: BTreeMap::new(),
            next_ip: 1,
            outage,
        });
    }

    fn node(&self, server_id: &str) -> &GsNode {
        self.nodes
            .iter()
            .find(|n| n.id == server_id)
            .unwrap_or_else(|| panic!("unknown game server {server_id}"))
    }

    fn live(&self) -> impl Iterator<Item = &GsNode> {
        self.nodes.iter().filter(|n| n.app.is_some())
    }

    // ---- player service ------------------------------------------------------------------------

    /// Base URL of the player-service HTTP API (loopback).
    pub fn player_service_url(&self) -> String {
        self.ps.url()
    }

    /// The player service's shared state (store, matchmaker, fleet registry).
    pub fn player_service(&self) -> &Arc<PlayerService> {
        &self.ps.service
    }

    /// Player-service supervisor app.
    pub fn player_service_app(&mut self) -> &mut App {
        &mut self.ps_app
    }

    /// An HTTP client of the player service holding `tokens`' session.
    pub fn api(&self, tokens: &SessionTokens) -> PlayerServiceClient {
        let client = PlayerServiceClient::new(&self.http, &self.ps.url());
        client.set_session(Some(tokens.clone()));
        client
    }

    /// Guest login (fresh device) over the HTTP API.
    pub fn guest(&mut self, name: &str) -> SessionTokens {
        let client = PlayerServiceClient::new(&self.http, &self.ps.url());
        let device = DeviceCredentials::generate();
        wait(client.login_guest(&device, Some(name))).expect("guest login")
    }

    /// Queue `players` and step (matchmaker ticks run in the player-service supervisor) until
    /// every ticket is `Found`; assignments in `players` order. Panics after 60 s of cluster time.
    pub fn matchmake(&mut self, players: &[&SessionTokens], queue: &str) -> Vec<Assignment> {
        let apis: Vec<PlayerServiceClient> = players.iter().map(|t| self.api(t)).collect();
        let tickets: Vec<_> = apis
            .iter()
            .map(|api| {
                wait(api.matchmake(&MatchmakeRequest {
                    queue: queue.to_owned(),
                    ..Default::default()
                }))
                .expect("matchmaking ticket")
                .ticket_id
            })
            .collect();
        let mut found: Vec<Option<Assignment>> = vec![None; players.len()];
        let max_steps = (60.0 / self.step.as_secs_f64()) as u64;
        for i in 0..max_steps {
            if i % 4 == 0 {
                for (k, api) in apis.iter().enumerate() {
                    if found[k].is_some() {
                        continue;
                    }
                    match wait(api.poll_ticket(&tickets[k], 0)).expect("poll ticket") {
                        TicketStatus::Found { assignment } => found[k] = Some(assignment),
                        TicketStatus::Searching { .. } => {}
                        other => panic!("ticket {:?} ended as {other:?}", tickets[k]),
                    }
                }
                if found.iter().all(Option::is_some) {
                    return found.into_iter().map(|a| a.expect("checked")).collect();
                }
            }
            self.step();
        }
        panic!("matchmaking did not complete: {found:?}");
    }

    /// `POST /v1/play/quick`.
    pub fn quick_play(&mut self, player: &SessionTokens, mode: &str) -> Result<Assignment, OnlineError> {
        wait(self.api(player).quick_play(&QuickPlayRequest {
            mode: mode.to_owned(),
            ..Default::default()
        }))
    }

    /// `POST /v1/sessions/rejoin` (after a room crash or a server loss).
    pub fn rejoin(&mut self, player: &SessionTokens) -> Result<Option<Assignment>, OnlineError> {
        wait(self.api(player).rejoin())
    }

    // ---- clients -------------------------------------------------------------------------------

    fn connector(&self, server_id: &str, link: Option<LinkProfile>) -> Arc<dyn Connector> {
        let node = self.node(server_id);
        Arc::new(NodeConnector {
            network: node.network.clone(),
            profile: link.unwrap_or_else(|| self.config.link.clone()),
            alive: node.alive.clone(),
        })
    }

    /// Connect a new client with an assignment (the closure adds the game's protocol plugin,
    /// input sampler / bot).
    pub fn connect(&mut self, assignment: &Assignment, build: impl FnOnce(&mut App)) -> ClientHandle {
        self.connect_with(ClientSpec::named("client"), assignment, build)
    }

    /// [`Self::connect`] with the link profile of `spec`.
    pub fn connect_with(
        &mut self,
        spec: ClientSpec,
        assignment: &Assignment,
        build: impl FnOnce(&mut App),
    ) -> ClientHandle {
        let mut app = App::new().with_tickrate(self.config.server.game.tick_hz);
        app.insert_resource(self.net_clock.clone());
        let connector = self.connector(&assignment.server_id, spec.link.clone());
        let mut cc = self.config.client.clone();
        cc.connector = connector.clone();
        if cc.game_id.is_empty() {
            cc.game_id = self.game_id.clone();
        }
        app.add_plugin(ClientReplicationPlugin { config: cc });
        build(&mut app);
        app.resources.expect_mut::<NetClient>().connect(ConnectTarget::Custom {
            connector,
            url: assignment.server_url.clone(),
            auth: HelloAuth::Ticket(assignment.join_ticket.clone()),
        });
        self.clients.push(ClientSlot {
            app,
            server: assignment.server_id.clone(),
        });
        ClientHandle(self.clients.len() - 1)
    }

    /// Point an existing client at a new assignment (after [`Self::rejoin`]).
    pub fn reconnect(&mut self, c: ClientHandle, assignment: &Assignment) {
        let connector = self.connector(&assignment.server_id, None);
        let slot = &mut self.clients[c.0];
        slot.server = assignment.server_id.clone();
        slot.app
            .resources
            .expect_mut::<NetClient>()
            .connect(ConnectTarget::Custom {
                connector,
                url: assignment.server_url.clone(),
                auth: HelloAuth::Ticket(assignment.join_ticket.clone()),
            });
    }

    /// A client's app.
    pub fn client(&self, c: ClientHandle) -> &App {
        &self.clients[c.0].app
    }

    /// A client's app (mutable).
    pub fn client_mut(&mut self, c: ClientHandle) -> &mut App {
        &mut self.clients[c.0].app
    }

    /// A client's `NetClient`.
    pub fn client_net(&self, c: ClientHandle) -> &NetClient {
        self.clients[c.0].app.resources.expect()
    }

    /// Game server a client was last pointed at.
    pub fn client_server(&self, c: ClientHandle) -> &str {
        &self.clients[c.0].server
    }

    // ---- stepping ------------------------------------------------------------------------------

    /// Advance one tick: clocks, network delivery + gatekeepers, client frames, room builds and
    /// ticks, game-server supervisors, player-service supervisor.
    pub fn step(&mut self) {
        self.now += self.step;
        self.steps += 1;
        self.clock.set(self.now);
        self.unix.set_ms(self.config.start_ms + self.now.as_millis() as i64);
        let every = self.config.supervisor_every.max(1);
        let supervise = self.steps.is_multiple_of(u64::from(every));
        let supervisor_step = self.step * every;
        for i in 0..self.nodes.len() {
            self.route(i);
        }
        for c in &mut self.clients {
            c.app.update(self.step);
        }
        let now = self.net_clock.now();
        for i in 0..self.nodes.len() {
            self.route(i);
            self.build_rooms(i);
            self.tick_rooms(i);
            let node = &mut self.nodes[i];
            if let Some(app) = &mut node.app {
                let mut gate = node.handle.gate.lock();
                gate.sweep(now);
                gate.enforce_ip_bans(self.unix.now_ms());
                drop(gate);
                if supervise {
                    app.update(supervisor_step);
                }
            }
        }
        if supervise {
            self.ps_app.update(supervisor_step);
        }
    }

    /// `step()` × n.
    pub fn step_ticks(&mut self, n: u64) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Hold every game server's checkpoint uploads (they complete after
    /// [`Self::release_checkpoint_uploads`]). A room handed off on drain closes only once its
    /// freeze capture is stored, so while uploads are held it stays frozen and the player
    /// service keeps reporting it as moving, however many ticks the test steps: tests of what
    /// happens *during* a handoff use this instead of racing the background upload.
    pub fn hold_checkpoint_uploads(&self) {
        self.upload_hold.store(true, Ordering::SeqCst);
    }

    /// Let held checkpoint uploads complete.
    pub fn release_checkpoint_uploads(&self) {
        self.upload_hold.store(false, Ordering::SeqCst);
    }

    /// Step until `pred` holds (checked before each step) or `max_ticks` elapse.
    pub fn run_until(&mut self, max_ticks: u64, mut pred: impl FnMut(&Self) -> bool) -> bool {
        for _ in 0..max_ticks {
            if pred(self) {
                return true;
            }
            self.step();
        }
        pred(self)
    }

    /// Wait in real time, **without advancing the clocks**, for background work (database
    /// writes, checkpoint uploads, outbox acks) until `pred` holds; supervisors keep running.
    pub fn settle(&mut self, timeout: Duration, mut pred: impl FnMut(&Self) -> bool) -> bool {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if pred(self) {
                return true;
            }
            if std::time::Instant::now() >= deadline {
                return false;
            }
            for node in &mut self.nodes {
                if let Some(app) = &mut node.app {
                    app.update(Duration::ZERO);
                }
            }
            self.ps_app.update(Duration::ZERO);
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    fn route(&mut self, i: usize) {
        let now = self.net_clock.now();
        let now_ms = self.unix.now_ms();
        let node = &mut self.nodes[i];
        node.network.pump();
        let events = node.endpoint.drain();
        if node.app.is_none() {
            return;
        }
        let metrics = node.handle.rooms.env().metrics.clone();
        for mut ev in events {
            if let LinkEvent::Opened { peer, .. } = &mut ev {
                // Distinct synthetic client IPs (10.x.y.z): per-IP limits behave as in production.
                let n = node.next_ip;
                node.next_ip += 1;
                peer.client_ip = Some(IpAddr::V4(Ipv4Addr::new(10, (n >> 16) as u8, (n >> 8) as u8, n as u8)));
            }
            crate::gs::ingress::handle_event(&node.handle.gate, &node.handle.rooms, ev, now, now_ms, &metrics);
        }
    }

    fn build_rooms(&mut self, i: usize) {
        let node = &mut self.nodes[i];
        if node.app.is_none() {
            return;
        }
        for mut pending in node.handle.rooms.take_pending_builds() {
            if let Some(lookup) = pending.lookup.take()
                && let Some(Some(cp)) = lookup.block()
            {
                pending.build.restore.get_or_insert(cp);
            }
            let build = pending.build;
            let (id, spec, env, status, created_ms) = (
                build.spec.room_id,
                build.spec.clone(),
                build.env.clone(),
                build.status.clone(),
                build.created_ms,
            );
            match catch_unwind(AssertUnwindSafe(|| build_room_app(build))) {
                Ok(Ok(app)) => {
                    let step = app.resources.expect::<TickRate>().step();
                    node.rooms.insert(
                        id,
                        HostedRoom {
                            app,
                            spec,
                            env,
                            status,
                            created_ms,
                            step,
                            crashes: Default::default(),
                        },
                    );
                }
                Ok(Err(e)) => {
                    tracing::error!(room = %id, error = %e, "test cluster: room build failed");
                    status.lock().phase = RoomPhase::Closed;
                }
                Err(p) => {
                    tracing::error!(room = %id, panic = %panic_message(p), "test cluster: room build panicked");
                    status.lock().phase = RoomPhase::Closed;
                }
            }
        }
    }

    fn tick_rooms(&mut self, i: usize) {
        let now = self.now;
        let node = &mut self.nodes[i];
        let mut crashed = 0;
        node.rooms.retain(|id, room| {
            let _span = crate::logging::room_tick_span(&room.app, &room.env.server_id, *id).entered();
            if let Err(why) = run_tick_guarded(&mut room.app, room.step) {
                crashed += 1;
                // The shards' path: rebuild from the in-memory checkpoint, or close the room
                // when it crash-loops.
                crate::gs::shard::recover_crashed(
                    &mut room.app,
                    &room.spec,
                    &room.env,
                    &room.status,
                    room.created_ms,
                    &mut room.crashes,
                    now,
                    &why,
                );
            }
            room.status.lock().phase != RoomPhase::Closed
        });
        self.crashes += crashed;
    }

    // ---- rooms, servers, faults ----------------------------------------------------------------

    /// A room's app (on whichever live game server hosts it).
    pub fn room(&self, id: RoomId) -> Option<&App> {
        self.live().find_map(|n| n.rooms.get(&id)).map(|r| &r.app)
    }

    /// A room's app (mutable).
    pub fn room_mut(&mut self, id: RoomId) -> Option<&mut App> {
        self.nodes
            .iter_mut()
            .find_map(|n| n.rooms.get_mut(&id))
            .map(|r| &mut r.app)
    }

    /// The copy of a room on game server `server_id` (after a partition, a room can have a
    /// stale copy on one server and the live one on another).
    pub fn room_on(&self, server_id: &str, id: RoomId) -> Option<&App> {
        self.node(server_id).rooms.get(&id).map(|r| &r.app)
    }

    /// [`Self::room_on`], mutable.
    pub fn room_on_mut(&mut self, server_id: &str, id: RoomId) -> Option<&mut App> {
        self.nodes
            .iter_mut()
            .find(|n| n.id == server_id)
            .and_then(|n| n.rooms.get_mut(&id))
            .map(|r| &mut r.app)
    }

    /// Id of the live game server hosting a room.
    pub fn room_server(&self, id: RoomId) -> Option<&str> {
        self.live().find(|n| n.rooms.contains_key(&id)).map(|n| n.id.as_str())
    }

    /// Rooms hosted by game server `server_id`.
    pub fn rooms_of(&self, server_id: &str) -> Vec<RoomId> {
        self.node(server_id).rooms.keys().copied().collect()
    }

    /// Room panics contained so far.
    pub fn room_crashes(&self) -> u64 {
        self.crashes
    }

    /// Make a room's next tick panic (exercises `catch_unwind` + rebuild from the room's
    /// in-memory checkpoint at its next epoch; clients are disconnected and rejoin).
    pub fn crash_room(&mut self, id: RoomId) {
        let app = self.room_mut(id).unwrap_or_else(|| panic!("no room {id}"));
        app.add_system_named(Stage::Update, "test::crash_room", |_| {
            panic!("TestCluster::crash_room: injected panic")
        });
    }

    /// [`Self::crash_room`] for the copy of a room on game server `server_id`.
    pub fn crash_room_on(&mut self, server_id: &str, id: RoomId) {
        let app = self
            .room_on_mut(server_id, id)
            .unwrap_or_else(|| panic!("no room {id} on {server_id}"));
        app.add_system_named(Stage::Update, "test::crash_room", |_| {
            panic!("TestCluster::crash_room: injected panic")
        });
    }

    /// Kill game server `server_id` (process loss): its rooms and supervisor vanish at once,
    /// every link to it is severed and new connections are refused. Its fleet link closes, so
    /// the player service expires its lease after `fleet.lease_ttl` of cluster time and
    /// orphans its rooms; clients then `rejoin` and are restored on another game server from
    /// the last checkpoint the player service stored.
    pub fn kill_game_server(&mut self, server_id: &str) {
        let i = self
            .nodes
            .iter()
            .position(|n| n.id == server_id)
            .unwrap_or_else(|| panic!("unknown game server {server_id}"));
        let node = &mut self.nodes[i];
        assert!(node.app.is_some(), "{server_id} is already dead");
        node.alive.store(false, Ordering::Release);
        for link in node.network.links() {
            node.network.sever(link, "server lost");
        }
        for (_, room) in std::mem::take(&mut node.rooms) {
            let _ = catch_unwind(AssertUnwindSafe(move || drop(room)));
        }
        // Dropping the supervisor closes its fleet link: no more heartbeats.
        drop(node.app.take());
    }

    /// Take the fleet link of game server `server_id` down (player service unreachable for it)
    /// or bring it back. Rooms keep running meanwhile; the outbox buffers results and replays
    /// them when the link returns.
    pub fn set_fleet_outage(&mut self, server_id: &str, down: bool) {
        self.node(server_id).outage.set(down);
    }

    /// Lose every player-service → game-server message of `server_id`'s fleet link while set
    /// (its sends still succeed): answers to requests that already went out never arrive.
    pub fn set_fleet_swallow(&mut self, server_id: &str, yes: bool) {
        self.node(server_id).outage.swallow_downlink(yes);
    }

    /// Ids of live game servers.
    pub fn game_servers(&self) -> Vec<String> {
        self.live().map(|n| n.id.clone()).collect()
    }

    /// Supervisor app of a live game server.
    pub fn game_server_app(&self, server_id: &str) -> Option<&App> {
        self.node(server_id).app.as_ref()
    }

    /// Supervisor app of a live game server (mutable).
    pub fn game_server_app_mut(&mut self, server_id: &str) -> Option<&mut App> {
        self.nodes
            .iter_mut()
            .find(|n| n.id == server_id)
            .and_then(|n| n.app.as_mut())
    }

    /// Handle of a live game server (rooms, gatekeeper, bans, addresses).
    pub fn game_server(&self, server_id: &str) -> Option<&GameServerHandle> {
        self.live().find(|n| n.id == server_id).map(|n| &n.handle)
    }

    /// Admin API base URL of `"ps"` (the player service) or a live game server.
    pub fn admin_url(&self, node: &str) -> String {
        let app = if node == "ps" {
            &self.ps_app
        } else {
            self.game_server_app(node)
                .unwrap_or_else(|| panic!("{node} is not a live game server"))
        };
        format!("http://{}", app.resources.expect::<crate::AdminHandle>().addr)
    }

    /// Memory network of a game server (link conditions, partitions, wire digest).
    pub fn network(&self, server_id: &str) -> &MemoryNetwork {
        &self.node(server_id).network
    }

    /// Monotonic cluster clock.
    pub fn clock(&self) -> &ManualClock {
        &self.clock
    }

    /// Wall clock.
    pub fn unix_clock(&self) -> &ManualUnixClock {
        &self.unix
    }

    /// Cluster time since start.
    pub fn elapsed(&self) -> Duration {
        self.now
    }

    /// Player id of a session.
    pub fn player_of(tokens: &SessionTokens) -> PlayerId {
        tokens.profile.player_id
    }
}

/// Block on a player-service call (test tool).
fn wait<T>(task: space3d_tasks::Task<Result<T, OnlineError>>) -> Result<T, OnlineError> {
    task.block()
        .unwrap_or(Err(OnlineError::Net(space3d_net::NetError::Closed)))
}
