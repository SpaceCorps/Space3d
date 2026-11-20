//! End-to-end over real sockets: an all-in-one server (player service + game server sharing one
//! loopback listener, rooms on shard threads) and real WebSocket clients. Real time is used here
//! on purpose (it is the production wiring); every wait has a deadline.

mod common;

use common::{DuelProtocol, GAME_ID, Move, remote_count};
use space3d_core::{App, EventReader, Events};
use space3d_replication::bots::scripted;
use space3d_replication::client::{ClientConfig, ClientReplicationPlugin, ConnectTarget, ConnectionEvent, NetClient};
use space3d_replication::protocol::control::Notice;
use space3d_replication::protocol::handshake::DisconnectCode;
use space3d_replication::services::{DeviceCredentials, PlayerServiceClient};
use space3d_replication::{RoomId, api::QuickPlayRequest};
use space3d_server::config::{Role, ServerConfig};
use space3d_server::gs::GameServerHandle;
use space3d_server::health::HealthState;
use space3d_server::lifecycle::LifecycleHandle;
use space3d_server::loadtest::{self, LoadTarget, LoadTestConfig, Slo};
use space3d_server::ps::PlayerServiceHandle;
use space3d_server::security::Secrets;
use space3d_server::{AdminHandle, BuildOptions, ServerMain};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn all_in_one(dir: &std::path::Path) -> ServerConfig {
    let port = free_port();
    let mut c = ServerConfig {
        role: Role::AllInOne,
        server_id: "e2e-1".into(),
        ..Default::default()
    };
    c.game.bind = format!("127.0.0.1:{port}").parse().unwrap();
    c.player_service.bind = c.game.bind;
    c.admin.bind = "127.0.0.1:0".parse().unwrap();
    c.game.public_url = format!("ws://127.0.0.1:{port}/play");
    c.game.shards = 2;
    c.game.allow_dev_auth = true;
    c.game.drain_timeout = Duration::from_secs(1);
    c.player_service.db_path = dir.join("players.db");
    c.persistence.data_dir = dir.to_path_buf();
    c.persistence.checkpoint_interval = Duration::from_secs(1);
    c.player_service.http_rps_per_ip = 10_000;
    c.player_service.login_attempts_per_min = 10_000;
    c
}

fn client() -> App {
    let mut app = App::new().with_tickrate(60.0);
    app.add_plugin(ClientReplicationPlugin {
        config: ClientConfig {
            game_id: GAME_ID.into(),
            ..Default::default()
        },
    });
    app.add_plugin(DuelProtocol);
    app
}

/// Step the server and clients in real time until `pred` or the deadline.
fn pump_until(
    server: &mut App,
    clients: &mut [App],
    deadline: Duration,
    mut pred: impl FnMut(&App, &[App]) -> bool,
) -> bool {
    let end = Instant::now() + deadline;
    let mut last = Instant::now();
    while Instant::now() < end {
        let now = Instant::now();
        let dt = now - last;
        last = now;
        server.update(dt);
        for c in clients.iter_mut() {
            c.update(dt);
        }
        if pred(server, clients) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(4));
    }
    false
}

#[test]
fn all_in_one_quick_play_three_clients_then_drain_checkpoints_and_closes() {
    let dir = tempfile::tempdir().unwrap();
    let mut server = ServerMain::with_config(common::DuelGame, all_in_one(dir.path()))
        .options(BuildOptions {
            secrets: Some(Secrets::for_tests("fleet", &["admin"])),
            ..Default::default()
        })
        .build()
        .expect("server starts");
    let ps = server.resources.expect::<PlayerServiceHandle>().clone();
    let gs = server.resources.expect::<GameServerHandle>().clone();
    assert_eq!(ps.addr, gs.game_addr, "equal binds share one listener");
    let health = server.resources.expect::<HealthState>().clone();
    assert!(pump_until(&mut server, &mut [], Duration::from_secs(5), |_, _| health
        .is_registered()));

    // Three players: guest login → quick play → the same room over real WebSockets.
    let http = space3d_net::Http::new();
    let mut clients = Vec::new();
    let mut room = None;
    for name in ["ann", "ben", "cat"] {
        let api = PlayerServiceClient::new(&http, &ps.url());
        api.login_guest(&DeviceCredentials::generate(), Some(name))
            .block()
            .unwrap()
            .unwrap();
        let a = api
            .quick_play(&QuickPlayRequest {
                mode: "ffa".into(),
                ..Default::default()
            })
            .block()
            .unwrap()
            .unwrap();
        assert_eq!(
            *room.get_or_insert(a.room_id),
            a.room_id,
            "quick play fills the open room"
        );
        assert!(a.server_url.starts_with("ws://127.0.0.1:"));
        let mut app = client();
        app.resources.expect_mut::<NetClient>().connect(ConnectTarget::Ticket {
            url: a.server_url.clone(),
            ticket: a.join_ticket.clone(),
        });
        clients.push(app);
    }
    let room: RoomId = room.unwrap();
    let everyone = pump_until(&mut server, &mut clients, Duration::from_secs(10), |_, cs| {
        cs.iter()
            .all(|c| c.resources.expect::<NetClient>().is_connected() && remote_count(c) == 3)
    });
    assert!(everyone, "three clients connected and see each other");
    let status = gs.rooms.get(room).expect("room listed").status();
    assert_eq!(status.room_id, room);

    // Graceful drain: notice, deadline, final checkpoint, disconnect, exit.
    let mut readers: Vec<EventReader> = clients.iter().map(|_| EventReader::default()).collect();
    let mut notices = vec![false; clients.len()];
    let mut closed = vec![false; clients.len()];
    server.resources.expect::<LifecycleHandle>().drain();
    let exited = pump_until(&mut server, &mut clients, Duration::from_secs(30), |s, cs| {
        for (i, c) in cs.iter().enumerate() {
            if let Some(events) = c.resources.get::<Events<ConnectionEvent>>() {
                for e in readers[i].read(events) {
                    match e {
                        ConnectionEvent::Notice(Notice::ShuttingDown { .. }) => notices[i] = true,
                        ConnectionEvent::Disconnected { code, .. } if *code == Some(DisconnectCode::ServerShutdown) => {
                            closed[i] = true
                        }
                        _ => {}
                    }
                }
            }
        }
        s.should_exit()
    });
    assert!(exited, "drain completed");
    assert!(notices.iter().all(|n| *n), "every client was warned: {notices:?}");
    assert!(
        closed.iter().all(|c| *c),
        "every client got Disconnect(ServerShutdown): {closed:?}"
    );
    let cp = ps.service.store.get_checkpoint(room).block().unwrap().unwrap();
    assert!(cp.is_some(), "the shutdown checkpoint was stored for restore elsewhere");
}

/// A second signal (forced stop) skips the graceful drain but loses nothing: every room is
/// captured into the game server's local database and its clients get
/// `Disconnect(ServerShutdown)`; the next start moves those checkpoints into the configured
/// store, from where a rejoin restores them.
#[test]
fn second_signal_checkpoints_rooms_locally_and_disconnects_clients() {
    use space3d_server::gs::checkpoint::{Checkpoint, CheckpointStore, LocalCheckpoints};
    let dir = tempfile::tempdir().unwrap();
    let config = all_in_one(dir.path());
    let build = |config: ServerConfig| {
        ServerMain::with_config(common::DuelGame, config)
            .options(BuildOptions {
                secrets: Some(Secrets::for_tests("fleet", &["admin"])),
                ..Default::default()
            })
            .build()
            .expect("server starts")
    };
    let mut server = build(config.clone());
    let health = server.resources.expect::<HealthState>().clone();
    assert!(pump_until(&mut server, &mut [], Duration::from_secs(5), |_, _| health
        .is_registered()));
    let gs = server.resources.expect::<GameServerHandle>().clone();
    let mut clients: Vec<App> = ["ann", "ben"]
        .iter()
        .map(|name| {
            let mut app = client();
            app.resources.expect_mut::<NetClient>().connect(ConnectTarget::Dev {
                url: gs.play_url("/play"),
                name: (*name).into(),
            });
            app
        })
        .collect();
    let everyone = pump_until(&mut server, &mut clients, Duration::from_secs(10), |_, cs| {
        cs.iter()
            .all(|c| c.resources.expect::<NetClient>().is_connected() && remote_count(c) == 2)
    });
    assert!(everyone, "two dev clients in the dev room");
    let room = RoomId(1);
    let mut readers: Vec<EventReader> = clients.iter().map(|_| EventReader::default()).collect();
    let mut closed = vec![false; clients.len()];
    let lifecycle = server.resources.expect::<LifecycleHandle>().clone();
    lifecycle.drain();
    lifecycle.drain();
    let exited = pump_until(&mut server, &mut clients, Duration::from_secs(20), |s, cs| {
        for (i, c) in cs.iter().enumerate() {
            if let Some(events) = c.resources.get::<Events<ConnectionEvent>>() {
                for e in readers[i].read(events) {
                    if matches!(e, ConnectionEvent::Disconnected { code, .. } if *code == Some(DisconnectCode::ServerShutdown))
                    {
                        closed[i] = true;
                    }
                }
            }
        }
        s.should_exit() && closed.iter().all(|c| *c)
    });
    assert!(exited, "forced stop exits; clients told: {closed:?}");
    assert_eq!(
        *server.resources.expect::<space3d_server::lifecycle::ExitStatus>(),
        space3d_server::lifecycle::ExitStatus(1)
    );
    drop(clients);
    drop(server);

    // The room was checkpointed into the game server's own database.
    let db = space3d_db::Db::open(&config.player_service.db_path);
    let local = LocalCheckpoints::open(db.clone(), Arc::new(space3d_server::clock::SystemUnixClock))
        .block()
        .unwrap()
        .unwrap();
    let bytes = local
        .get(room)
        .block()
        .unwrap()
        .unwrap()
        .expect("local rescue checkpoint");
    let rescued = Checkpoint::from_bytes(&bytes).unwrap();
    assert_eq!(rescued.room_id, room);
    drop(local);
    drop(db);

    // Next start: the rescue checkpoint moves to the configured (player-service) store.
    let mut server = build(config);
    let ps = server.resources.expect::<PlayerServiceHandle>().clone();
    let moved = pump_until(&mut server, &mut [], Duration::from_secs(5), |_, _| {
        ps.service
            .store
            .get_checkpoint(room)
            .block()
            .unwrap()
            .unwrap()
            .is_some_and(|(_, b)| Checkpoint::from_bytes(&b).is_ok_and(|c| c.saved_ms == rescued.saved_ms))
    });
    assert!(moved, "rescue checkpoint uploaded on the next start");
}

#[test]
fn load_test_smoke_over_real_sockets() {
    let dir = tempfile::tempdir().unwrap();
    let config = all_in_one(dir.path());
    let (tx, rx) = std::sync::mpsc::channel();
    let stop = Arc::new(AtomicBool::new(false));
    let stop2 = stop.clone();
    // The supervisor App is !Send: it lives on its own thread for the (blocking) load test.
    let server = std::thread::spawn(move || {
        let mut app = ServerMain::with_config(common::DuelGame, config)
            .options(BuildOptions {
                secrets: Some(Secrets::for_tests("fleet", &["admin"])),
                ..Default::default()
            })
            .build()
            .expect("server starts");
        let ps = app.resources.expect::<PlayerServiceHandle>().url();
        let play = app.resources.expect::<GameServerHandle>().play_url("/play");
        let admin = app.resources.expect::<AdminHandle>().addr;
        let health = app.resources.expect::<HealthState>().clone();
        let mut sent = false;
        let mut last = Instant::now();
        while !stop2.load(Ordering::Acquire) {
            let now = Instant::now();
            app.update(now - last);
            last = now;
            if !sent && health.is_registered() {
                sent = true;
                tx.send((ps.clone(), play.clone(), format!("http://{admin}/metrics")))
                    .unwrap();
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    });
    let (ps, play, metrics) = rx.recv_timeout(Duration::from_secs(10)).expect("server ready");
    // Shared CI runners (macOS: 100+ ms RTT, skipped ticks) deliver fewer snapshots: relax the
    // rate by the documented `SPACE3D_PERF_BUDGET_SCALE`.
    let scale: f64 = std::env::var("SPACE3D_PERF_BUDGET_SCALE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let slo = Slo {
        connect_success: 1.0,
        tick_p99_ms: Some(50.0 * scale),
        snapshot_rate: 0.5 / scale,
        max_unexpected_disconnects: 0,
    };
    let protocol: fn(&mut App) = |app| {
        app.add_plugin(DuelProtocol);
    };
    let direct = loadtest::run(
        LoadTestConfig {
            target: LoadTarget::DirectDev { url: play },
            clients: 6,
            ramp_per_sec: 30.0,
            duration: Duration::from_secs(2),
            bots_per_thread: 3,
            tick_hz: 60.0,
            metrics_url: Some(metrics.clone()),
            slo: slo.clone(),
            game_id: GAME_ID.into(),
        },
        protocol,
        |i| {
            scripted(move |t| Move {
                x: if (t.0 / 60 + i as u32).is_multiple_of(2) { 1 } else { -1 },
                z: 0,
            })
        },
    )
    .unwrap();
    assert_eq!(direct.connected, 6, "{direct:#?}");
    assert!(direct.slo_passed, "{direct:#?}");
    assert!(
        direct.server_tick_p99_ms.is_some(),
        "tick histogram scraped: {direct:#?}"
    );
    assert!(direct.down_bytes_per_sec.0 > 0.0);

    let via_ps = loadtest::run(
        LoadTestConfig {
            target: LoadTarget::PlayerService {
                url: ps,
                queue: "ffa".into(),
                quick_play: true,
            },
            clients: 4,
            ramp_per_sec: 20.0,
            duration: Duration::from_secs(1),
            bots_per_thread: 2,
            tick_hz: 60.0,
            metrics_url: None,
            slo: Slo {
                tick_p99_ms: None,
                ..slo
            },
            game_id: GAME_ID.into(),
        },
        protocol,
        |_| scripted(|_| Move::default()),
    )
    .unwrap();
    stop.store(true, Ordering::Release);
    server.join().unwrap();
    assert_eq!(via_ps.connected, 4, "{via_ps:#?}");
    assert!(via_ps.slo_passed, "{via_ps:#?}");
}
