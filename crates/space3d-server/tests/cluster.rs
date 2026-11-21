//! Split-topology integration tests on the in-process `TestCluster`: one player service and
//! separate game servers (fleet links, tickets, gatekeepers, rooms, outbox, checkpoints),
//! clients over the memory network, manual clocks.

mod common;

use common::{DuelGame, DuelProtocol, Finish, SetHealth, healths, pawn_health, pawns, remote_count};
use space3d_net::{Http, HttpRequest};
use space3d_replication::api::SessionTokens;
use space3d_replication::client::{ConnectionState, FailReason, NetClient};
use space3d_replication::harness::ClientHandle;
use space3d_replication::protocol::handshake::{DisconnectCode, RejectCode};
use space3d_replication::{PlayerId, RoomId};
use space3d_server::gs::checkpoint::Checkpoint;
use space3d_server::ps::fleet::ServerState;
use space3d_server::testing::{TEST_ADMIN_TOKEN, TestCluster, TestClusterConfig};
use std::time::Duration;

fn cluster(game_servers: usize) -> TestCluster {
    TestCluster::new(
        DuelGame,
        TestClusterConfig {
            game_servers,
            ..Default::default()
        },
    )
}

fn player(t: &SessionTokens) -> PlayerId {
    t.profile.player_id
}

fn connected_pair(c: &mut TestCluster, a: &SessionTokens, b: &SessionTokens) -> (ClientHandle, ClientHandle, RoomId) {
    let found = c.matchmake(&[a, b], "duel");
    assert_eq!(found[0].room_id, found[1].room_id, "a duel shares one room");
    assert_eq!(found[0].server_id, found[1].server_id);
    let ca = c.connect(&found[0], |app| {
        app.add_plugin(DuelProtocol);
    });
    let cb = c.connect(&found[1], |app| {
        app.add_plugin(DuelProtocol);
    });
    let ok = c.run_until(600, |c| {
        c.client_net(ca).is_connected()
            && c.client_net(cb).is_connected()
            && remote_count(c.client(ca)) == 2
            && remote_count(c.client(cb)) == 2
    });
    assert!(ok, "both players connected and see both pawns");
    (ca, cb, found[0].room_id)
}

fn send<M: space3d_replication::NetMessage>(c: &mut TestCluster, client: ClientHandle, msg: &M) {
    c.client_mut(client)
        .resources
        .expect_mut::<NetClient>()
        .send(msg)
        .expect("send");
}

fn failed_with(c: &TestCluster, client: ClientHandle, code: DisconnectCode) -> bool {
    matches!(
        c.client_net(client).state(),
        ConnectionState::Failed(FailReason::Disconnected(got, _)) if *got == code
    )
}

fn admin(
    c: &TestCluster,
    node: &str,
    method: &str,
    path: &str,
    body: Option<serde_json::Value>,
    token: bool,
) -> (u16, String) {
    let mut req = HttpRequest::new(method.parse().unwrap(), format!("{}{path}", c.admin_url(node)));
    if token {
        req = req.header("authorization", format!("Bearer {TEST_ADMIN_TOKEN}"));
    }
    if let Some(b) = body {
        req = req.json(&b).unwrap();
    }
    let res = Http::new().request(req).block().unwrap().unwrap();
    (res.status, res.text())
}

#[test]
fn matchmaking_play_and_results_flow_to_the_player_service() {
    let mut c = cluster(2);
    let alice = c.guest("alice");
    let bob = c.guest("bob");
    let (ca, _cb, room) = connected_pair(&mut c, &alice, &bob);
    let server = c.room_server(room).expect("room hosted").to_owned();
    assert!(c.game_servers().contains(&server));
    assert_eq!(pawns(c.room(room).unwrap()), 2);

    // alice wins: the room reports the result through the outbox → fleet link → store.
    send(&mut c, ca, &Finish);
    let board = c.api(&alice);
    let applied = c.run_until(600, |_| {
        board
            .leaderboard("wins", 10)
            .block()
            .unwrap()
            .is_ok_and(|rows| rows.len() == 1)
    });
    assert!(applied, "match result reached the leaderboard");
    let rows = board.leaderboard("wins", 10).block().unwrap().unwrap();
    assert_eq!(rows[0].player_id, player(&alice));
    assert_eq!(rows[0].score, 1);
    let alice_me = c.api(&alice).me().block().unwrap().unwrap();
    let bob_me = c.api(&bob).me().block().unwrap().unwrap();
    assert!(
        alice_me.rating > bob_me.rating,
        "Elo moved: {} vs {}",
        alice_me.rating,
        bob_me.rating
    );
    assert!((alice_me.rating + bob_me.rating - 2.0 * alice.profile.rating).abs() < 1e-6);
    let settled = c.settle(Duration::from_secs(5), |c| {
        c.api(&alice)
            .get_data("progress")
            .block()
            .unwrap()
            .is_ok_and(|e| e.value == serde_json::json!({ "wins": 1 }))
    });
    assert!(settled, "room-written player data persisted");
}

// Restored rooms: `full_state::restore` pre-maps the captured `NetId`s and the change cache
// adopts those entities on its first collect, so (re)joining clients receive the restored state.

#[test]
fn room_crash_is_contained_and_players_rejoin_restored_state() {
    room_crash_scenario();
}

fn room_crash_scenario() {
    let mut c = cluster(1);
    let alice = c.guest("alice");
    let bob = c.guest("bob");
    let (ca, cb, room) = connected_pair(&mut c, &alice, &bob);
    send(&mut c, ca, &SetHealth(42));
    assert!(c.run_until(120, |c| pawn_health(c.room(room).unwrap(), player(&alice)) == Some(42)));
    // Let the 1 s checkpoint cadence capture the new state.
    c.step_ticks(90);

    c.crash_room(room);
    // The crashed room disconnects everyone (`RoomCrashed`, no resume): clients give up on the
    // link and rejoin through the player service.
    let dropped = c.run_until(120, |c| {
        failed_with(c, ca, DisconnectCode::RoomCrashed) && failed_with(c, cb, DisconnectCode::RoomCrashed)
    });
    assert!(
        dropped,
        "clients told to rejoin: {:?} / {:?}",
        c.client_net(ca).state(),
        c.client_net(cb).state()
    );
    assert_eq!(c.room_crashes(), 1);
    let restored = c.room(room).expect("room rebuilt, not closed");
    assert_eq!(pawns(restored), 2, "pawns restored from the checkpoint");
    assert_eq!(pawn_health(restored, player(&alice)), Some(42));

    for (client, tokens) in [(ca, &alice), (cb, &bob)] {
        let a = c.rejoin(tokens).unwrap().expect("room still exists");
        assert_eq!(a.room_id, room);
        c.reconnect(client, &a);
    }
    let ok = c.run_until(600, |c| {
        c.client_net(ca).is_connected() && c.client_net(cb).is_connected()
    });
    assert!(ok, "players back in the rebuilt room");
    assert_eq!(pawns(c.room(room).unwrap()), 2, "rejoin re-used the restored pawns");
    let synced = c.run_until(300, |c| {
        healths(c.client(ca)) == vec![42, 100] && healths(c.client(cb)) == vec![42, 100]
    });
    assert!(synced, "clients see the restored state: {:?}", healths(c.client(ca)));
}

#[test]
fn game_server_loss_restores_the_room_on_another_server() {
    server_loss_scenario();
}

fn server_loss_scenario() {
    let mut c = cluster(2);
    let alice = c.guest("alice");
    let bob = c.guest("bob");
    let (ca, cb, room) = connected_pair(&mut c, &alice, &bob);
    let lost = c.room_server(room).unwrap().to_owned();
    send(&mut c, ca, &SetHealth(7));
    assert!(c.run_until(120, |c| pawn_health(c.room(room).unwrap(), player(&alice)) == Some(7)));
    let set_at = c.unix_clock_ms();
    // A checkpoint captured after the change must reach the player service's store. Judge it
    // by the blob's own capture time: an older capture that was still uploading may land
    // later (regression: the store used to stamp arrival time, so a stale blob passed this
    // wait and the room was restored with health 100).
    let stored = c.run_until(600, |c| {
        c.player_service()
            .store
            .get_checkpoint(room)
            .block()
            .unwrap()
            .unwrap()
            .is_some_and(|(saved, bytes)| {
                let cp = Checkpoint::from_bytes(&bytes).expect("stored checkpoint decodes");
                assert_eq!(saved, cp.saved_ms, "row stamped with the capture time");
                cp.saved_ms > set_at
            })
    });
    assert!(stored, "checkpoint uploaded to the player service");

    c.kill_game_server(&lost);
    assert!(c.room(room).is_none());
    let lease = c.player_service().config.fleet.lease_ttl;
    let ticks = (lease.as_secs_f64() * 60.0) as u64 + 180;
    let dead = c.run_until(ticks, |c| {
        c.player_service()
            .fleet
            .lock()
            .server(&lost)
            .is_none_or(|s| s.state == ServerState::Dead)
    });
    assert!(dead, "lease expired");

    let mut target = None;
    for (client, tokens) in [(ca, &alice), (cb, &bob)] {
        let a = c.rejoin(tokens).unwrap().expect("orphaned room can be restored");
        assert_eq!(a.room_id, room);
        assert_ne!(a.server_id, lost);
        target.get_or_insert(a.server_id.clone());
        assert_eq!(
            target.as_deref(),
            Some(a.server_id.as_str()),
            "both players go to the same server"
        );
        c.reconnect(client, &a);
    }
    let ok = c.run_until(600, |c| {
        c.client_net(ca).is_connected() && c.client_net(cb).is_connected()
    });
    assert!(ok, "players reconnected to the restored room");
    assert_eq!(c.room_server(room), target.as_deref());
    let restored = c.room(room).unwrap();
    assert_eq!(pawns(restored), 2);
    assert_eq!(pawn_health(restored, player(&alice)), Some(7));
    let synced = c.run_until(300, |c| {
        healths(c.client(ca)) == vec![7, 100] && healths(c.client(cb)) == vec![7, 100]
    });
    assert!(synced, "clients see the restored state: {:?}", healths(c.client(ca)));
}

#[test]
fn draining_server_gets_no_new_rooms_and_admin_api_is_guarded() {
    let mut c = cluster(2);
    // Admin API: token required, metrics public.
    assert_eq!(admin(&c, "gs-1", "GET", "/admin/v1/status", None, false).0, 401);
    let (status, body) = admin(&c, "gs-1", "GET", "/admin/v1/status", None, true);
    assert_eq!(status, 200, "{body}");
    assert!(body.contains("\"registered\":true"), "{body}");
    let (status, metrics) = admin(&c, "gs-1", "GET", "/metrics", None, false);
    assert_eq!(status, 200);
    assert!(metrics.contains("space3d_rooms"), "{metrics}");
    let (status, schema) = admin(&c, "gs-2", "GET", "/admin/v1/schema", None, true);
    assert_eq!(status, 200);
    assert!(schema.contains("duel.health"), "{schema}");

    // Drain gs-1 through its admin API; the player service stops allocating to it.
    let (status, _) = admin(&c, "gs-1", "POST", "/admin/v1/drain", None, true);
    assert_eq!(status, 202);
    // It reports draining; with no rooms the drain completes at once and it deregisters
    // (dead right away, not after the lease).
    let draining = c.run_until(600, |c| {
        c.player_service()
            .fleet
            .lock()
            .server("gs-1")
            .is_none_or(|s| s.state != ServerState::Ready)
    });
    assert!(draining, "heartbeat reported draining");
    let lease_ticks = (c.player_service().config.fleet.lease_ttl.as_secs_f64() * 60.0) as u64;
    let left = c.run_until(lease_ticks / 2, |c| {
        c.player_service()
            .fleet
            .lock()
            .server("gs-1")
            .is_none_or(|s| s.state == ServerState::Dead && s.left)
    });
    assert!(left, "deregistered (dead and forgotten) before its lease ran out");
    let (status, _) = admin(&c, "gs-1", "GET", "/readyz", None, false);
    assert_eq!(status, 503, "a draining server is not ready");
    for name in ["p1", "p2", "p3"] {
        let t = c.guest(name);
        let a = c.quick_play(&t, "ffa").expect("quick play");
        assert_eq!(a.server_id, "gs-2");
    }
}

#[test]
fn bans_kick_players_and_block_their_tickets() {
    let mut c = cluster(1);
    let alice = c.guest("alice");
    let bob = c.guest("bob");
    let (ca, cb, room) = connected_pair(&mut c, &alice, &bob);

    let (status, body) = admin(
        &c,
        "ps",
        "POST",
        "/admin/v1/bans",
        Some(serde_json::json!({ "player": player(&bob), "reason": "cheating" })),
        true,
    );
    assert_eq!(status, 201, "{body}");
    // The room removes a banned player itself once its server's ban list changes.
    let kicked = c.run_until(300, |c| {
        failed_with(c, cb, DisconnectCode::Banned) && pawns(c.room(room).unwrap()) == 1
    });
    assert!(
        kicked,
        "the ban reached the game server and removed bob: {:?}",
        c.client_net(cb).state()
    );
    assert!(c.client_net(ca).is_connected(), "alice unaffected");
    let err = c.api(&bob).me().block().unwrap().unwrap_err();
    assert_eq!(err.status(), Some(403));
    assert_eq!(err.code(), Some("banned"));
    let err = c.rejoin(&bob).unwrap_err();
    assert_eq!(err.code(), Some("banned"));
    // Unknown admin token.
    let res = Http::new()
        .request(
            HttpRequest::get(format!("{}/admin/v1/bans", c.admin_url("ps"))).header("authorization", "Bearer nope"),
        )
        .block()
        .unwrap()
        .unwrap();
    assert_eq!(res.status, 401);
}

trait UnixMs {
    fn unix_clock_ms(&self) -> i64;
}

impl UnixMs for TestCluster {
    fn unix_clock_ms(&self) -> i64 {
        use space3d_server::clock::UnixClock;
        self.unix_clock().now_ms()
    }
}

#[test]
fn network_partition_resumes_the_same_slot_through_the_gatekeeper() {
    use space3d_core::{EventReader, Events};
    use space3d_replication::client::ConnectionEvent;
    let mut c = cluster(1);
    let alice = c.guest("alice");
    let bob = c.guest("bob");
    let (ca, cb, room) = connected_pair(&mut c, &alice, &bob);
    let server = c.client_server(ca).to_owned();
    // Alice's link is the first one opened on her game server's network.
    let link = c.network(&server).links()[0];
    let mut reader = EventReader::default();
    c.network(&server).partition(link, Duration::from_secs(2));
    let mut resumed = false;
    let back = c.run_until(1200, |c| {
        if let Some(events) = c.client(ca).resources.get::<Events<ConnectionEvent>>() {
            for e in reader.read(events) {
                if let ConnectionEvent::Connected { resumed: true, .. } = e {
                    resumed = true;
                }
            }
        }
        resumed && c.client_net(ca).is_connected()
    });
    assert!(back, "alice resumed: {:?}", c.client_net(ca).state());
    assert!(c.client_net(cb).is_connected(), "bob unaffected");
    assert_eq!(pawns(c.room(room).unwrap()), 2, "same slot and pawn, no duplicate");
    assert!(c.run_until(120, |c| remote_count(c.client(ca)) == 2));
}

#[test]
fn concurrent_quick_plays_fill_one_room() {
    use space3d_replication::api::QuickPlayRequest;
    let mut c = cluster(1);
    let players: Vec<SessionTokens> = (0..8).map(|i| c.guest(&format!("q{i}"))).collect();
    // All eight requests are in flight at once: finding an open room and allocating a new one
    // happen under one fleet lock, so they share one room (capacity 16) instead of racing
    // into several.
    let tasks: Vec<_> = players
        .iter()
        .map(|t| {
            c.api(t).quick_play(&QuickPlayRequest {
                mode: "ffa".into(),
                ..Default::default()
            })
        })
        .collect();
    let rooms: std::collections::BTreeSet<RoomId> = tasks
        .into_iter()
        .map(|t| t.block().unwrap().expect("quick play").room_id)
        .collect();
    assert_eq!(rooms.len(), 1, "{rooms:?}");
}

/// [`TestCluster::run_until`] that also gives background work real time (≥ 2 ms per tick):
/// checkpoint uploads, database writes and the player service's async answers finish in real
/// time, while cluster time can race ahead of them.
fn run_until_bg(c: &mut TestCluster, max_ticks: u64, mut pred: impl FnMut(&TestCluster) -> bool) -> bool {
    for _ in 0..max_ticks {
        if pred(c) {
            return true;
        }
        c.step();
        std::thread::sleep(Duration::from_millis(2));
    }
    pred(c)
}

fn persistent_cluster(game_servers: usize) -> TestCluster {
    use space3d_server::config::QueueConfig;
    let mut config = TestClusterConfig {
        game_servers,
        ..Default::default()
    };
    // Only the drain checkpoint captures state changed right before the drain.
    config.server.persistence.checkpoint_interval = Duration::from_secs(600);
    config.server.matchmaking.queues.push(QueueConfig {
        name: "world".into(),
        mode: "world".into(),
        teams: 1,
        team_size: 8,
        min_players: 1,
        backfill: true,
        params: [("persistent".to_owned(), "true".to_owned())].into(),
        ..Default::default()
    });
    TestCluster::new(DuelGame, config)
}

fn saw_notice(
    c: &TestCluster,
    client: ClientHandle,
    reader: &mut space3d_core::EventReader,
    want: &space3d_replication::protocol::control::Notice,
) -> bool {
    use space3d_replication::client::ConnectionEvent;
    let events = c
        .client(client)
        .resources
        .expect::<space3d_core::Events<ConnectionEvent>>();
    reader
        .read(events)
        .any(|e| matches!(e, ConnectionEvent::Notice(n) if n == want))
}

/// §2.4.2 regression: draining a server moves its persistent room. The room checkpoints at the
/// drain, closes once that checkpoint is stored (`Notice::Reconnect`, then
/// `Disconnect(ServerShutdown)`), and its players' rejoin restores exactly that state on
/// another server — never an older checkpoint, never two live copies.
#[test]
fn draining_hands_a_persistent_room_to_another_server() {
    use space3d_replication::protocol::control::Notice;
    let mut c = persistent_cluster(2);
    let alice = c.guest("alice");
    let bob = c.guest("bob");
    let a = c.quick_play(&alice, "world").expect("quick play");
    let b = c.quick_play(&bob, "world").expect("quick play");
    assert_eq!(a.room_id, b.room_id, "one open world room");
    let (room, host) = (a.room_id, a.server_id.clone());
    let ca = c.connect(&a, |app| {
        app.add_plugin(DuelProtocol);
    });
    let cb = c.connect(&b, |app| {
        app.add_plugin(DuelProtocol);
    });
    assert!(c.run_until(600, |c| {
        c.client_net(ca).is_connected() && c.client_net(cb).is_connected() && remote_count(c.client(ca)) == 2
    }));
    let spec = c
        .room(room)
        .unwrap()
        .resources
        .expect::<space3d_server::gs::room::RoomContext>()
        .spec
        .clone();
    assert!(spec.persistent);
    // Let the join checkpoint land, then change state only the drain checkpoint captures.
    c.step_ticks(30);
    send(&mut c, ca, &SetHealth(9));
    assert!(c.run_until(120, |c| pawn_health(c.room(room).unwrap(), player(&alice)) == Some(9)));

    let (status, _) = admin(&c, &host, "POST", "/admin/v1/drain", None, true);
    assert_eq!(status, 202);
    let mut reader = space3d_core::EventReader::default();
    let mut warned = false;
    let moved = run_until_bg(&mut c, 900, |c| {
        warned |= saw_notice(c, ca, &mut reader, &Notice::Reconnect);
        failed_with(c, ca, DisconnectCode::ServerShutdown) && failed_with(c, cb, DisconnectCode::ServerShutdown)
    });
    assert!(
        moved,
        "room handed off: {:?} / {:?}",
        c.client_net(ca).state(),
        c.client_net(cb).state()
    );
    assert!(warned, "clients were told the room moves");
    // The frozen room closes once its final checkpoint is stored.
    let handed_off = run_until_bg(&mut c, 900, |c| {
        c.room(room).is_none() && c.player_service().fleet.lock().room(room).is_some_and(|e| e.orphaned)
    });
    assert!(handed_off, "no live copy left on the draining server");
    let entry = c.player_service().fleet.lock().room(room).cloned().expect("room kept");
    assert!(entry.orphaned && !entry.moving, "kept restorable");

    let mut target = None;
    for (client, tokens) in [(ca, &alice), (cb, &bob)] {
        let a = c.rejoin(tokens).unwrap().expect("the world can be restored");
        assert_eq!(a.room_id, room);
        assert_ne!(a.server_id, host, "restored on another server");
        target.get_or_insert(a.server_id.clone());
        assert_eq!(target.as_deref(), Some(a.server_id.as_str()));
        c.reconnect(client, &a);
    }
    assert!(c.run_until(600, |c| {
        c.client_net(ca).is_connected() && c.client_net(cb).is_connected()
    }));
    let restored = c.room(room).unwrap();
    assert_eq!(
        pawn_health(restored, player(&alice)),
        Some(9),
        "the drain checkpoint's state"
    );
    assert!(c.run_until(300, |c| healths(c.client(ca)) == vec![9, 100]));
}

/// Regression: the drain handoff kept a persistent room playable while its drain checkpoint
/// uploaded, then restored that earlier capture, so every change made during the upload — which
/// the players had seen — was lost. The room now freezes (clients disconnected) at the capture,
/// and the player service holds rejoins until that capture is stored.
#[test]
fn a_drain_handoff_keeps_every_change_the_players_saw() {
    let mut c = persistent_cluster(2);
    let alice = c.guest("alice");
    let bob = c.guest("bob");
    let a = c.quick_play(&alice, "world").expect("quick play");
    let b = c.quick_play(&bob, "world").expect("quick play");
    let (room, host) = (a.room_id, a.server_id.clone());
    let ca = c.connect(&a, |app| {
        app.add_plugin(DuelProtocol);
    });
    let cb = c.connect(&b, |app| {
        app.add_plugin(DuelProtocol);
    });
    assert!(c.run_until(600, |c| {
        c.client_net(ca).is_connected() && c.client_net(cb).is_connected() && remote_count(c.client(ca)) == 2
    }));
    c.step_ticks(30);
    let (status, _) = admin(&c, &host, "POST", "/admin/v1/drain", None, true);
    assert_eq!(status, 202);
    // Alice changes her pawn every tick, as fast as the room accepts it, until the room
    // freezes (her link is closed).
    let mut last_in_room = 0;
    let mut last_seen = 0;
    for _ in 0..600 {
        if let Some(app) = c.room(room) {
            last_in_room = pawn_health(app, player(&alice)).unwrap_or(last_in_room);
        }
        if !c.client_net(ca).is_connected() {
            break;
        }
        let seen = healths(c.client(ca)).into_iter().max().unwrap_or(0);
        last_seen = last_seen.max(seen);
        send(&mut c, ca, &SetHealth(seen + 1));
        c.step();
    }
    assert!(!c.client_net(ca).is_connected(), "the room froze");
    // The frozen room closes once its freeze capture is stored (serialization and upload run
    // in the background: real time, not ticks — and however many ticks that takes, the room
    // never closes restorable before that capture is stored).
    let handed_off = run_until_bg(&mut c, 3000, |c| {
        if let Some(app) = c.room(room) {
            last_in_room = pawn_health(app, player(&alice)).unwrap_or(last_in_room);
        }
        c.room(room).is_none() && c.player_service().fleet.lock().room(room).is_some_and(|e| e.orphaned)
    });
    assert!(handed_off, "handed off");
    assert!(last_seen > 100, "alice changed her pawn during the drain");
    let target = c.rejoin(&alice).unwrap().expect("the world can be restored");
    assert_ne!(target.server_id, host);
    c.reconnect(ca, &target);
    assert!(c.run_until(600, |c| c.client_net(ca).is_connected()));
    let restored = pawn_health(c.room(room).unwrap(), player(&alice)).unwrap();
    assert_eq!(
        restored, last_in_room,
        "restored {restored}, but the old room reached {last_in_room} (alice saw {last_seen})"
    );
    assert!(restored >= last_seen);
}

/// A rejoin that arrives while a persistent room is frozen for its handoff waits until the
/// final checkpoint is stored, then restores the room elsewhere (it neither goes back to the
/// frozen room nor restores an older checkpoint).
#[test]
fn a_rejoin_during_a_handoff_waits_for_the_final_checkpoint() {
    let mut c = persistent_cluster(2);
    let alice = c.guest("alice");
    let a = c.quick_play(&alice, "world").expect("quick play");
    let (room, host) = (a.room_id, a.server_id.clone());
    let ca = c.connect(&a, |app| {
        app.add_plugin(DuelProtocol);
    });
    assert!(c.run_until(600, |c| c.client_net(ca).is_connected()));
    c.step_ticks(30);
    send(&mut c, ca, &SetHealth(42));
    assert!(c.run_until(120, |c| pawn_health(c.room(room).unwrap(), player(&alice)) == Some(42)));
    // The final capture is stored in the background, in real time: hold it, so that the room
    // is still frozen however many ticks the steps below take (released once the rejoin was
    // seen to wait).
    c.hold_checkpoint_uploads();
    let (status, _) = admin(&c, &host, "POST", "/admin/v1/drain", None, true);
    assert_eq!(status, 202);
    let moving = c.run_until(600, |c| {
        c.player_service().fleet.lock().room(room).is_some_and(|e| e.moving)
    });
    assert!(moving, "the player service knows the room is moving");
    assert!(c.run_until(120, |c| failed_with(c, ca, DisconnectCode::ServerShutdown)));
    // The room only closes when the test steps it: the rejoin must wait meanwhile.
    let api = c.api(&alice);
    let rejoin = std::thread::spawn(move || api.rejoin().block());
    std::thread::sleep(Duration::from_millis(300));
    assert!(!rejoin.is_finished(), "the rejoin waits for the handoff");
    c.release_checkpoint_uploads();
    assert!(
        run_until_bg(&mut c, 900, |_| rejoin.is_finished()),
        "answered after the handoff"
    );
    let target = rejoin
        .join()
        .unwrap()
        .expect("task")
        .expect("rejoin")
        .expect("restorable");
    assert_eq!(target.room_id, room);
    assert_ne!(target.server_id, host, "restored on another server");
    c.reconnect(ca, &target);
    assert!(c.run_until(600, |c| c.client_net(ca).is_connected()));
    assert_eq!(pawn_health(c.room(room).unwrap(), player(&alice)), Some(42));
}

/// Regression (split brain): a game server whose fleet link was cut for longer than its lease
/// keeps running its rooms, so a player who rejoined meanwhile got the room restored on another
/// server — and once the link healed, the stale copy kept running forever (its heartbeats even
/// flipped the room's entry back and forth). The player service now closes the stale copy
/// (`RoomSuperseded`) as soon as its server is heard from again, and its players rejoin the
/// live one.
#[test]
fn a_partitioned_server_closes_its_stale_copy_of_a_restored_room() {
    let mut c = cluster(2);
    let alice = c.guest("alice");
    let bob = c.guest("bob");
    let (ca, cb, room) = connected_pair(&mut c, &alice, &bob);
    let stale = c.room_server(room).unwrap().to_owned();
    send(&mut c, cb, &SetHealth(7));
    assert!(c.run_until(120, |c| pawn_health(c.room(room).unwrap(), player(&bob)) == Some(7)));
    let set_at = c.unix_clock_ms();
    let stored = c.run_until(600, |c| {
        c.player_service()
            .store
            .get_checkpoint(room)
            .block()
            .unwrap()
            .unwrap()
            .is_some_and(|(saved, _)| saved > set_at)
    });
    assert!(stored, "checkpoint stored");

    c.set_fleet_outage(&stale, true);
    let lease = c.player_service().config.fleet.lease_ttl;
    let dead = c.run_until((lease.as_secs_f64() * 60.0) as u64 + 180, |c| {
        c.player_service()
            .fleet
            .lock()
            .server(&stale)
            .is_none_or(|s| s.state == ServerState::Dead)
    });
    assert!(dead, "lease expired");
    assert!(
        c.rooms_of(&stale).contains(&room),
        "the partitioned server still runs the room"
    );
    // Bob's client restarted: the player service restores the room elsewhere.
    let moved = c.rejoin(&bob).unwrap().expect("restorable");
    assert_ne!(moved.server_id, stale);
    c.reconnect(cb, &moved);
    assert!(c.run_until(600, |c| c.client_net(cb).is_connected()));
    // Alice plays on in the stale copy until the partition heals.
    send(&mut c, ca, &SetHealth(55));
    c.step_ticks(60);
    c.set_fleet_outage(&stale, false);
    let closed = c.run_until(600, |c| !c.rooms_of(&stale).contains(&room));
    assert!(closed, "the stale copy was closed");
    assert!(
        c.run_until(120, |c| failed_with(c, ca, DisconnectCode::ServerShutdown)),
        "{:?}",
        c.client_net(ca).state()
    );
    let fleet = c.player_service().fleet.lock().room(room).cloned().unwrap();
    assert_eq!(fleet.server, moved.server_id, "the entry stays with the live copy");
    let back = c.run_until(600, |c| {
        c.player_service()
            .fleet
            .lock()
            .server(&stale)
            .is_some_and(|s| s.state == ServerState::Ready)
    });
    assert!(back, "the partitioned server registered again");
    let a = c.rejoin(&alice).unwrap().expect("rejoin");
    assert_eq!((a.room_id, a.server_id.as_str()), (room, moved.server_id.as_str()));
    c.reconnect(ca, &a);
    let one = c.run_until(600, |c| {
        c.client_net(ca).is_connected()
            && healths(c.client(ca)) == healths(c.client(cb))
            && healths(c.client(ca)).len() == 2
    });
    assert!(one, "{:?} vs {:?}", healths(c.client(ca)), healths(c.client(cb)));
    assert_eq!(c.rooms_of(&stale), Vec::<RoomId>::new());
}

/// Regression (split brain): the stale copy's match result, scores and player data sat in its
/// durable outbox during the partition and were applied when the link healed, next to the live
/// copy's — Elo applied twice with opposite winners, both players credited a win, the stale
/// copy's progress written. What a stale copy did after the point the live copy was restored
/// from is now dropped (its ops are acknowledged, not applied).
#[test]
fn a_stale_copy_s_results_are_dropped_when_the_partition_heals() {
    let mut c = cluster(2);
    let alice = c.guest("alice");
    let bob = c.guest("bob");
    let (ca, cb, room) = connected_pair(&mut c, &alice, &bob);
    let stale = c.room_server(room).unwrap().to_owned();
    let stored = c.run_until(600, |c| {
        c.player_service()
            .store
            .get_checkpoint(room)
            .block()
            .and_then(Result::ok)
            .flatten()
            .is_some()
    });
    assert!(stored, "a restore point");
    c.set_fleet_outage(&stale, true);
    let lease = c.player_service().config.fleet.lease_ttl;
    let dead = c.run_until((lease.as_secs_f64() * 60.0) as u64 + 180, |c| {
        c.player_service()
            .fleet
            .lock()
            .server(&stale)
            .is_none_or(|s| s.state == ServerState::Dead)
    });
    assert!(dead, "lease expired");
    let moved = c.rejoin(&bob).unwrap().expect("restorable");
    assert_ne!(moved.server_id, stale);
    c.reconnect(cb, &moved);
    assert!(c.run_until(600, |c| c.client_net(cb).is_connected()));
    // Stale copy: alice "wins" (result, score and progress wait in its outbox).
    send(&mut c, ca, &Finish);
    c.step_ticks(60);
    // Live copy: bob wins.
    send(&mut c, cb, &Finish);
    let board = |c: &TestCluster| {
        let mut rows: Vec<(PlayerId, i64)> = c
            .api(&alice)
            .leaderboard("wins", 10)
            .block()
            .unwrap()
            .unwrap()
            .into_iter()
            .map(|r| (r.player_id, r.score))
            .collect();
        rows.sort();
        rows
    };
    let bob_won = run_until_bg(&mut c, 600, |c| board(c) == vec![(player(&bob), 1)]);
    assert!(bob_won, "{:?}", board(&c));
    let rating = |c: &TestCluster, t: &SessionTokens| c.api(t).me().block().unwrap().unwrap().rating;
    let before = (rating(&c, &alice), rating(&c, &bob));
    assert!(before.1 > before.0, "bob won once: {before:?}");
    // The partition heals: the stale copy is closed and its outbox replayed.
    c.set_fleet_outage(&stale, false);
    assert!(
        c.run_until(600, |c| !c.rooms_of(&stale).contains(&room)),
        "stale copy closed"
    );
    let pending = c
        .game_server(&stale)
        .unwrap()
        .rooms
        .env()
        .metrics
        .outbox_pending
        .clone();
    let flushed = run_until_bg(&mut c, 1_800, |_| pending.get() == 0.0);
    assert!(
        flushed,
        "the stale copy's ops were answered ({} pending)",
        pending.get()
    );
    c.step_ticks(60);
    let _ = c.settle(Duration::from_millis(500), |_| false);
    assert_eq!(board(&c), vec![(player(&bob), 1)], "one win, bob's");
    assert_eq!((rating(&c, &alice), rating(&c, &bob)), before, "Elo applied once");
    assert!(
        c.api(&alice).get_data("progress").block().unwrap().is_err(),
        "the stale copy's progress write was dropped"
    );
}

/// Regression: a rejoin while the hosting server drains goes back to the match that still
/// runs there instead of restoring a second copy on another server.
#[test]
fn rejoin_during_a_drain_returns_to_the_running_match() {
    let mut c = cluster(2);
    let alice = c.guest("alice");
    let bob = c.guest("bob");
    let (ca, cb, room) = connected_pair(&mut c, &alice, &bob);
    let host = c.room_server(room).unwrap().to_owned();
    let (status, _) = admin(&c, &host, "POST", "/admin/v1/drain", None, true);
    assert_eq!(status, 202);
    assert!(c.run_until(600, |c| {
        c.player_service()
            .fleet
            .lock()
            .server(&host)
            .is_some_and(|s| s.state == ServerState::Draining)
    }));
    // Bob's client restarted (no resume token): rejoin through the player service.
    let a = c.rejoin(&bob).unwrap().expect("rejoin");
    assert_eq!(
        (a.room_id, a.server_id.as_str()),
        (room, host.as_str()),
        "same room, same server"
    );
    c.reconnect(cb, &a);
    assert!(
        c.run_until(600, |c| c.client_net(cb).is_connected()),
        "{:?}",
        c.client_net(cb).state()
    );
    assert!(c.client_net(ca).is_connected(), "alice unaffected");
    assert_eq!(pawns(c.room(room).unwrap()), 2, "bob took his own slot back");
    let other = c.game_servers().into_iter().find(|s| *s != host).unwrap();
    assert!(c.rooms_of(&other).is_empty(), "no second copy of the match");
    assert_eq!(
        c.player_service().fleet.lock().room(room).map(|e| e.server.clone()),
        Some(host.clone())
    );
}

/// §12 regression: a room crashing 3 times within 5 minutes is closed (clients get
/// `RoomCrashed`, no rebuild) — the in-process cluster uses the shards' crash-loop guard.
#[test]
fn a_crash_looping_room_is_closed() {
    let mut c = cluster(1);
    let alice = c.guest("alice");
    let bob = c.guest("bob");
    let (ca, _cb, room) = connected_pair(&mut c, &alice, &bob);
    let stored = |c: &TestCluster| {
        c.player_service()
            .store
            .get_checkpoint(room)
            .block()
            .and_then(Result::ok)
            .flatten()
            .is_some()
    };
    assert!(c.settle(Duration::from_secs(10), stored), "the room checkpointed");
    for crash in 1..=3u64 {
        c.crash_room(room);
        assert!(
            c.run_until(60, |c| c.room_crashes() == crash),
            "crash {crash} contained"
        );
        if crash < 3 {
            assert!(c.room(room).is_some(), "rebuilt after crash {crash}");
        }
        c.step_ticks(5);
    }
    assert!(c.room(room).is_none(), "closed after the third crash");
    assert!(failed_with(&c, ca, DisconnectCode::RoomCrashed));
    let gone = c.run_until(120, |c| c.player_service().fleet.lock().room(room).is_none());
    assert!(gone, "the player service forgot the room");
    assert!(!matches!(c.rejoin(&alice), Ok(Some(_))), "nothing to rejoin");
    // Regression: the crash-looping room's checkpoint stayed stored forever (and a join ticket
    // that was still valid recreated the room from it).
    assert!(
        c.settle(Duration::from_secs(10), |c| !stored(c)),
        "the crash-looping room's checkpoint is deleted"
    );
}

/// Regression: checkpoints of rooms that ended without a normal close (orphans nobody
/// rejoined, abandoned rooms, handed-off rooms nobody came back to) were never deleted; the
/// table grew with every lost server and deploy. Checkpoints too old to restore are now
/// garbage-collected.
#[test]
fn checkpoints_too_old_to_restore_are_garbage_collected() {
    use space3d_server::gs::checkpoint::CheckpointKey;
    let mut c = cluster(1);
    let store = c.player_service().store.clone();
    let now = {
        use space3d_server::clock::UnixClock;
        c.unix_clock().now_ms()
    };
    let put = |room: u64, saved_ms: i64| {
        let spec = space3d_server::game::RoomSpec {
            room_id: RoomId(room),
            mode: "duel".into(),
            ..Default::default()
        };
        let state = space3d_replication::server::full_state::FullState {
            schema_hash: 1,
            tick: space3d_replication::Tick(10),
            epoch: 1,
            next_net_id: 1,
            entities: Vec::new(),
        };
        let cp = Checkpoint::new(spec, state, None, None, saved_ms);
        let key = CheckpointKey { epoch: 1, saved_ms };
        store
            .put_checkpoint(RoomId(room), cp.to_bytes(), key)
            .block()
            .unwrap()
            .unwrap();
    };
    let max_age = c.player_service().config.persistence.recovery_max_age.as_millis() as i64;
    // A lost room's checkpoint (nobody rejoined) and a fresh one.
    put(901, now - max_age - 5 * 60_000);
    put(902, now);
    let has = |c: &TestCluster, room: u64| {
        c.player_service()
            .store
            .get_checkpoint(RoomId(room))
            .block()
            .and_then(Result::ok)
            .flatten()
            .is_some()
    };
    // Supervisors prune once a minute.
    c.step_ticks(61 * 60);
    assert!(
        c.settle(Duration::from_secs(10), |c| !has(c, 901)),
        "expired checkpoint pruned"
    );
    assert!(has(&c, 902), "a restorable checkpoint stays");
}

/// §12 regression: `RoomControl::load_player_data` reaches the player service and back
/// (`PlayerDataLoaded`), a fetch during a fleet outage is answered empty, and
/// `RoomControl::submit_score` reaches the leaderboard through the outbox (replayed after an
/// outage, applied once).
#[test]
fn rooms_load_player_data_and_submit_scores() {
    use common::{LoadData, Loaded, Score};
    let mut c = cluster(1);
    let alice = c.guest("alice");
    let bob = c.guest("bob");
    let (ca, _cb, room) = connected_pair(&mut c, &alice, &bob);
    let server = c.room_server(room).unwrap().to_owned();
    c.api(&alice)
        .put_data(
            "settings",
            &space3d_replication::api::PutPlayerDataRequest {
                value: serde_json::json!({ "fov": 90 }),
                expected_version: None,
            },
        )
        .block()
        .unwrap()
        .unwrap();
    let loaded = |c: &TestCluster| c.room(room).unwrap().resources.expect::<Loaded>().0.clone();
    send(&mut c, ca, &LoadData(vec!["settings".into(), "missing".into()]));
    assert!(run_until_bg(&mut c, 500, |c| !loaded(c).is_empty()), "answer arrived");
    let got = loaded(&c);
    assert_eq!(got[0].player, player(&alice));
    assert_eq!(got[0].entries.len(), 1, "missing keys are absent");
    assert_eq!(got[0].entries[0].key, "settings");
    assert_eq!(got[0].entries[0].value, serde_json::json!({ "fov": 90 }));

    // Player service unreachable: the game still gets an (empty) answer.
    c.set_fleet_outage(&server, true);
    send(&mut c, ca, &LoadData(vec!["settings".into()]));
    assert!(
        run_until_bg(&mut c, 300, |c| loaded(c).len() == 2),
        "answered during the outage"
    );
    assert!(loaded(&c)[1].entries.is_empty());

    // Score submitted during the outage: buffered, replayed, on the board once.
    send(&mut c, ca, &Score(7));
    c.step_ticks(30);
    c.set_fleet_outage(&server, false);
    let board = c.api(&alice);
    let on_board = c.run_until(600, |_| {
        board
            .leaderboard("kills", 10)
            .block()
            .unwrap()
            .is_ok_and(|rows| rows.len() == 1)
    });
    assert!(on_board, "score reached the leaderboard");
    let rows = board.leaderboard("kills", 10).block().unwrap().unwrap();
    assert_eq!((rows[0].player_id, rows[0].score), (player(&alice), 7));
}

/// A fetch the player service never answers (the link dropped after the request went out) is
/// answered empty after `FETCH_TIMEOUT`, so the game is never left waiting.
#[test]
fn unanswered_player_data_fetches_time_out() {
    use common::{LoadData, Loaded};
    let mut c = cluster(1);
    let alice = c.guest("alice");
    let bob = c.guest("bob");
    let (ca, _cb, room) = connected_pair(&mut c, &alice, &bob);
    let server = c.room_server(room).unwrap().to_owned();
    c.set_fleet_swallow(&server, true);
    send(&mut c, ca, &LoadData(vec!["settings".into()]));
    let loaded = |c: &TestCluster| c.room(room).unwrap().resources.expect::<Loaded>().0.clone();
    let ticks = (space3d_server::gs::FETCH_TIMEOUT.as_secs_f64() * 60.0) as u64 + 120;
    assert!(
        c.run_until(ticks, |c| !loaded(c).is_empty()),
        "answered after the timeout"
    );
    assert!(loaded(&c)[0].entries.is_empty());
}

/// Regression (drain handoff between real processes: supervisors at 10 Hz, rooms at 60 Hz):
/// the room froze and disconnected its clients in its drain tick, but its `RoomMoving` reached
/// the player service only a supervisor tick or two later. A client's rejoin — sent the moment
/// the disconnect arrives — came first and got a ticket straight back to the draining server,
/// which refused it (`Reject(ServerDraining)`), and the example client gave up. The room now
/// keeps its clients until the player service acknowledged the move, so every rejoin is held
/// until the room is stored and then restores it on another server.
#[test]
fn a_rejoin_sent_the_moment_the_handoff_disconnects_is_held_for_the_restore() {
    use space3d_server::config::QueueConfig;
    let mut config = TestClusterConfig {
        game_servers: 2,
        // The production cadence: supervisors (fleet links, player service) at 10 Hz.
        supervisor_every: 6,
        ..Default::default()
    };
    config.server.persistence.checkpoint_interval = Duration::from_secs(600);
    config.server.matchmaking.queues.push(QueueConfig {
        name: "world".into(),
        mode: "world".into(),
        teams: 1,
        team_size: 8,
        min_players: 1,
        backfill: true,
        params: [("persistent".to_owned(), "true".to_owned())].into(),
        ..Default::default()
    });
    let mut c = TestCluster::new(DuelGame, config);
    let alice = c.guest("alice");
    let bob = c.guest("bob");
    let a = c.quick_play(&alice, "world").expect("quick play");
    let b = c.quick_play(&bob, "world").expect("quick play");
    assert_eq!(a.room_id, b.room_id);
    let (room, host) = (a.room_id, a.server_id.clone());
    let ca = c.connect(&a, |app| {
        app.add_plugin(DuelProtocol);
    });
    let cb = c.connect(&b, |app| {
        app.add_plugin(DuelProtocol);
    });
    assert!(c.run_until(900, |c| {
        c.client_net(ca).is_connected() && c.client_net(cb).is_connected() && remote_count(c.client(ca)) == 2
    }));
    send(&mut c, ca, &SetHealth(42));
    assert!(c.run_until(120, |c| pawn_health(c.room(room).unwrap(), player(&alice)) == Some(42)));
    c.step_ticks(30);

    // The final capture is stored in the background, in real time: hold it so that the room
    // is still moving whenever the assertion below looks (released right after).
    c.hold_checkpoint_uploads();
    let (status, _) = admin(&c, &host, "POST", "/admin/v1/drain", None, true);
    assert_eq!(status, 202);
    // Each client asks the player service to rejoin as soon as its disconnect arrives — before
    // the cluster moves on (a real client does so within milliseconds).
    let mut rejoins = [None, None];
    for _ in 0..1200 {
        c.step();
        for (i, (client, tokens)) in [(ca, &alice), (cb, &bob)].into_iter().enumerate() {
            if rejoins[i].is_none() && failed_with(&c, client, DisconnectCode::ServerShutdown) {
                let api = c.api(tokens);
                rejoins[i] = Some(std::thread::spawn(move || api.rejoin().block()));
            }
        }
        if rejoins.iter().all(Option::is_some) {
            break;
        }
    }
    let rejoins = rejoins.map(|r| r.expect("the handoff disconnected the client"));
    // The cluster stands still: nothing more reaches the player service meanwhile.
    std::thread::sleep(Duration::from_millis(300));
    assert!(
        rejoins.iter().all(|r| !r.is_finished()),
        "the player service answered a rejoin while the room was still moving (a ticket back to the draining server)"
    );
    c.release_checkpoint_uploads();
    assert!(
        run_until_bg(&mut c, 1800, |_| rejoins
            .iter()
            .all(std::thread::JoinHandle::is_finished)),
        "answered once the room was stored"
    );
    for (client, rejoin) in [ca, cb].into_iter().zip(rejoins) {
        let target = rejoin
            .join()
            .unwrap()
            .expect("task")
            .expect("rejoin")
            .expect("restorable");
        assert_eq!(target.room_id, room);
        assert_ne!(target.server_id, host, "restored on another server");
        c.reconnect(client, &target);
    }
    assert!(c.run_until(900, |c| {
        c.client_net(ca).is_connected() && c.client_net(cb).is_connected() && remote_count(c.client(cb)) == 2
    }));
    assert_eq!(pawn_health(c.room(room).unwrap(), player(&alice)), Some(42));
}

/// Rooms of `mode` the player service knows (live, moving or waiting for a restore).
fn rooms_of_mode(c: &TestCluster, mode: &str) -> Vec<RoomId> {
    c.player_service()
        .fleet
        .lock()
        .rooms()
        .filter(|(_, e)| e.spec.as_ref().is_some_and(|s| s.mode == mode))
        .map(|(id, _)| *id)
        .collect()
}

fn quick_play_in_background(
    c: &TestCluster,
    tokens: &SessionTokens,
    mode: &str,
) -> std::thread::JoinHandle<
    Option<Result<space3d_replication::api::Assignment, space3d_replication::services::OnlineError>>,
> {
    let api = c.api(tokens);
    let req = space3d_replication::api::QuickPlayRequest {
        mode: mode.to_owned(),
        ..Default::default()
    };
    std::thread::spawn(move || api.quick_play(&req).block())
}

/// Regression (verifier runs 03/15/21, real processes): a player whose quick play was answered
/// in the moment a persistent room's server began to drain got a ticket into that room, which
/// refused it as closed for good (`Reject(RoomClosed)`, no retry delay). Their rejoin found
/// nothing (they had never joined), and their next quick play — the draining server no longer
/// taking players — opened a second copy of the world on another server, next to the restored
/// one, for good. The moving room now refuses the ticket as transient (`ServerDraining` with a
/// retry delay), and a quick play into a mode whose persistent room is moving waits for the
/// handoff, then joins the room where it was restored.
#[test]
fn a_quick_play_during_a_handoff_joins_the_moved_room() {
    let mut c = persistent_cluster(2);
    let alice = c.guest("alice");
    let bob = c.guest("bob");
    let a = c.quick_play(&alice, "world").expect("quick play");
    let (room, host) = (a.room_id, a.server_id.clone());
    let ca = c.connect(&a, |app| {
        app.add_plugin(DuelProtocol);
    });
    assert!(c.run_until(600, |c| c.client_net(ca).is_connected()));
    c.step_ticks(30);
    // Bob's quick play is answered just before the player service hears of the drain.
    let b = c.quick_play(&bob, "world").expect("quick play");
    assert_eq!((b.room_id, b.server_id.as_str()), (room, host.as_str()));
    // The final capture is stored in the background, in real time: hold it, so that the room
    // stays frozen and `moving` however many ticks the steps below take (released before the
    // handoff is waited for).
    c.hold_checkpoint_uploads();
    let (status, _) = admin(&c, &host, "POST", "/admin/v1/drain", None, true);
    assert_eq!(status, 202);
    assert!(c.run_until(600, |c| {
        c.player_service().fleet.lock().room(room).is_some_and(|e| e.moving)
    }));
    let cb = c.connect(&b, |app| {
        app.add_plugin(DuelProtocol);
    });
    assert!(c.run_until(300, |c| matches!(c.client_net(cb).state(), ConnectionState::Failed(_))));
    let ConnectionState::Failed(why) = c.client_net(cb).state().clone() else {
        unreachable!()
    };
    assert!(
        matches!(&why, FailReason::Rejected(r) if r.code == RejectCode::ServerDraining),
        "{why:?}"
    );
    assert!(why.retry_after().is_some(), "a transient refusal: ask again ({why:?})");
    // Bob asks the player service again: there is nothing to rejoin (he never joined), and his
    // quick play waits while the room moves — no second copy of the world meanwhile.
    assert_eq!(c.rejoin(&bob).expect("rejoin"), None);
    assert!(
        c.player_service().fleet.lock().room(room).is_some_and(|e| e.moving),
        "the room is still moving"
    );
    let quick = quick_play_in_background(&c, &bob, "world");
    std::thread::sleep(Duration::from_millis(300));
    assert!(!quick.is_finished(), "quick play waits for the handoff");
    assert_eq!(rooms_of_mode(&c, "world"), vec![room]);
    c.release_checkpoint_uploads();
    assert!(
        run_until_bg(&mut c, 1800, |_| quick.is_finished()),
        "answered after the handoff"
    );
    let b = quick.join().unwrap().expect("task").expect("quick play");
    assert_eq!(b.room_id, room, "bob joins the world, not a new room");
    assert_ne!(b.server_id, host, "where it was restored");
    assert!(failed_with(&c, ca, DisconnectCode::ServerShutdown));
    let a = c.rejoin(&alice).unwrap().expect("restorable");
    assert_eq!((a.room_id, a.server_id.as_str()), (room, b.server_id.as_str()));
    c.reconnect(ca, &a);
    c.reconnect(cb, &b);
    assert!(c.run_until(900, |c| {
        c.client_net(ca).is_connected()
            && c.client_net(cb).is_connected()
            && remote_count(c.client(ca)) == 2
            && remote_count(c.client(cb)) == 2
    }));
    assert_eq!(rooms_of_mode(&c, "world"), vec![room], "one world");
}

/// A quick play into a mode whose persistent room was lost with its server (its lease
/// expired, its players not back yet) restores that room on another server and places the
/// player in it, as its players' rejoin would — instead of opening a second world that the
/// returning players would never see.
#[test]
fn a_quick_play_restores_a_lost_world_instead_of_opening_another() {
    let mut c = persistent_cluster(2);
    let alice = c.guest("alice");
    let carol = c.guest("carol");
    let a = c.quick_play(&alice, "world").expect("quick play");
    let (room, lost) = (a.room_id, a.server_id.clone());
    let ca = c.connect(&a, |app| {
        app.add_plugin(DuelProtocol);
    });
    assert!(c.run_until(600, |c| c.client_net(ca).is_connected()));
    send(&mut c, ca, &SetHealth(9));
    assert!(c.run_until(120, |c| pawn_health(c.room(room).unwrap(), player(&alice)) == Some(9)));
    let set_at = c.unix_clock_ms();
    let (status, _) = admin(
        &c,
        &lost,
        "POST",
        &format!("/admin/v1/rooms/{room}/checkpoint"),
        None,
        true,
    );
    assert_eq!(status, 202);
    let stored = run_until_bg(&mut c, 600, |c| {
        c.player_service()
            .store
            .get_checkpoint(room)
            .block()
            .unwrap()
            .unwrap()
            .is_some_and(|(saved, _)| saved > set_at)
    });
    assert!(stored, "a checkpoint with the change reached the player service");
    c.kill_game_server(&lost);
    let lease = c.player_service().config.fleet.lease_ttl;
    let ticks = (lease.as_secs_f64() * 60.0) as u64 + 180;
    assert!(c.run_until(ticks, |c| {
        c.player_service().fleet.lock().room(room).is_some_and(|e| e.orphaned)
    }));
    let k = c.quick_play(&carol, "world").expect("quick play");
    assert_eq!(k.room_id, room, "the world, restored");
    assert_ne!(k.server_id, lost);
    let a = c.rejoin(&alice).unwrap().expect("restorable");
    assert_eq!((a.room_id, a.server_id.as_str()), (room, k.server_id.as_str()));
    c.reconnect(ca, &a);
    let ck = c.connect(&k, |app| {
        app.add_plugin(DuelProtocol);
    });
    assert!(c.run_until(900, |c| {
        c.client_net(ca).is_connected() && c.client_net(ck).is_connected() && remote_count(c.client(ck)) == 2
    }));
    assert_eq!(
        pawn_health(c.room(room).unwrap(), player(&alice)),
        Some(9),
        "restored state"
    );
    assert_eq!(rooms_of_mode(&c, "world"), vec![room], "one world");
}

/// Regression (verifier, rolling restart of a single game server): a rejoin that found no
/// game server able to take the room answered `503 no_capacity` without a retry delay (quick
/// play had one), so the example client gave up at once instead of waiting for the
/// replacement server. Both now say when to ask again.
#[test]
fn no_capacity_for_a_restore_says_when_to_ask_again() {
    let mut c = persistent_cluster(1);
    let alice = c.guest("alice");
    let carol = c.guest("carol");
    let a = c.quick_play(&alice, "world").expect("quick play");
    let (room, host) = (a.room_id, a.server_id.clone());
    let ca = c.connect(&a, |app| {
        app.add_plugin(DuelProtocol);
    });
    assert!(c.run_until(600, |c| c.client_net(ca).is_connected()));
    c.step_ticks(30);
    let (status, _) = admin(&c, &host, "POST", "/admin/v1/drain", None, true);
    assert_eq!(status, 202);
    assert!(run_until_bg(&mut c, 1800, |c| {
        c.player_service().fleet.lock().room(room).is_some_and(|e| e.orphaned)
    }));
    for e in [
        c.rejoin(&alice).unwrap_err(),
        c.quick_play(&carol, "world").unwrap_err(),
    ] {
        assert_eq!((e.status(), e.code()), (Some(503), Some("no_capacity")), "{e}");
        assert!(e.retry_after_ms().is_some(), "{e}");
    }
    assert_eq!(rooms_of_mode(&c, "world"), vec![room], "still restorable");
}

/// Regression: a room that panicked while frozen for its drain handoff was rebuilt as a
/// fresh, joinable room that knew nothing of the handoff: it never stored its freeze capture
/// nor closed restorable, so the player service kept it "moving" and every rejoin failed
/// (`503 room_moving`) until the server finally went away. The rebuilt room now continues the
/// handoff.
#[test]
fn a_room_crash_during_its_drain_handoff_still_hands_it_off() {
    use space3d_replication::server::NetServer;
    let mut c = persistent_cluster(2);
    let alice = c.guest("alice");
    let a = c.quick_play(&alice, "world").expect("quick play");
    let (room, host) = (a.room_id, a.server_id.clone());
    let ca = c.connect(&a, |app| {
        app.add_plugin(DuelProtocol);
    });
    assert!(c.run_until(600, |c| c.client_net(ca).is_connected()));
    c.step_ticks(30);
    send(&mut c, ca, &SetHealth(42));
    assert!(c.run_until(120, |c| pawn_health(c.room(room).unwrap(), player(&alice)) == Some(42)));
    // The room panics on the first tick after its clients were sent away: while frozen, before
    // the room ever sees its freeze capture stored.
    c.room_mut(room)
        .unwrap()
        .add_system(space3d_core::Stage::Update, |ctx| {
            if ctx.resources.expect::<NetServer>().clients().is_empty() {
                panic!("cluster test: injected panic while frozen");
            }
        });
    let (status, _) = admin(&c, &host, "POST", "/admin/v1/drain", None, true);
    assert_eq!(status, 202);
    assert!(c.run_until(600, |c| failed_with(c, ca, DisconnectCode::ServerShutdown)));
    assert!(c.run_until(60, |c| c.room_crashes() == 1), "the frozen room crashed");
    let api = c.api(&alice);
    let rejoin = std::thread::spawn(move || api.rejoin().block());
    assert!(
        run_until_bg(&mut c, 1500, |_| rejoin.is_finished()),
        "the rejoin is answered once the rebuilt room finished the handoff"
    );
    let target = rejoin
        .join()
        .unwrap()
        .expect("task")
        .expect("rejoin")
        .expect("restorable");
    assert_ne!(target.server_id, host, "restored on another server");
    c.reconnect(ca, &target);
    assert!(c.run_until(600, |c| c.client_net(ca).is_connected()));
    assert_eq!(pawn_health(c.room(room).unwrap(), player(&alice)), Some(42));
}

/// Regression (split brain): a stale copy of a room — left running behind a fleet partition
/// after the room was restored elsewhere — that closed normally (its match ended, it emptied)
/// deleted the stored checkpoint, which by then was the live copy's. Deletes now only remove
/// a checkpoint of the deleting copy's epoch or older.
#[test]
fn a_stale_copy_closing_during_a_partition_keeps_the_live_copy_s_checkpoint() {
    use space3d_server::gs::rooms::RoomCommand;
    let stored_epoch = |c: &TestCluster, room: RoomId| {
        c.player_service()
            .store
            .get_checkpoint(room)
            .block()
            .unwrap()
            .unwrap()
            .map(|(_, bytes)| Checkpoint::peek(&bytes).unwrap().1.epoch)
    };
    let mut c = cluster(2);
    let alice = c.guest("alice");
    let bob = c.guest("bob");
    let (_ca, cb, room) = connected_pair(&mut c, &alice, &bob);
    let stale = c.room_server(room).unwrap().to_owned();
    assert!(
        c.run_until(600, |c| stored_epoch(c, room).is_some()),
        "checkpoint stored"
    );
    let stale_epoch = stored_epoch(&c, room).unwrap();

    c.set_fleet_outage(&stale, true);
    let lease = c.player_service().config.fleet.lease_ttl;
    let dead = c.run_until((lease.as_secs_f64() * 60.0) as u64 + 180, |c| {
        c.player_service()
            .fleet
            .lock()
            .server(&stale)
            .is_none_or(|s| s.state == ServerState::Dead)
    });
    assert!(dead, "lease expired");
    let moved = c.rejoin(&bob).unwrap().expect("restorable");
    assert_ne!(moved.server_id, stale);
    c.reconnect(cb, &moved);
    assert!(c.run_until(600, |c| c.client_net(cb).is_connected()));
    let live = run_until_bg(&mut c, 900, |c| stored_epoch(c, room).is_some_and(|e| e > stale_epoch));
    assert!(live, "the live copy stored its own checkpoint");
    let live_epoch = stored_epoch(&c, room).unwrap();

    // Still partitioned, the stale copy closes normally.
    let handle = c.game_server(&stale).unwrap().rooms.get(room).expect("stale copy");
    handle
        .command(RoomCommand::Close {
            reason: "match over".into(),
        })
        .unwrap();
    assert!(c.run_until(120, |c| !c.rooms_of(&stale).contains(&room)));
    // Deletes run in the background: give them time, then the live copy's checkpoint is there.
    let _ = c.settle(Duration::from_millis(500), |_| false);
    assert_eq!(
        stored_epoch(&c, room),
        Some(live_epoch),
        "the live copy's checkpoint survives the stale copy's close"
    );
}

/// Regression (split brain, follow-up to the stale-copy delete): the epoch fences assume every
/// copy of a room has an epoch of its own, but a crash rebuild bumped the epoch by one just as
/// a restore does. A stale copy at epoch e (behind a fleet partition) that crashed was rebuilt
/// at e + 1 — the epoch of the live copy restored elsewhere from its last stored checkpoint
/// (epoch e). Its checkpoints then replaced the live copy's (same epoch, later capture), and
/// its normal close deleted the live copy's checkpoint (`through_epoch` = e + 1). A restore now
/// starts a new placement of the room (epoch generation) that crash rebuilds of an earlier one
/// never reach.
#[test]
fn a_stale_copy_rebuilt_after_a_crash_never_takes_the_live_copy_s_epoch() {
    use space3d_replication::server::NetServer;
    use space3d_server::gs::rooms::RoomCommand;
    let stored = |c: &TestCluster, room: RoomId| {
        c.player_service()
            .store
            .get_checkpoint(room)
            .block()
            .unwrap()
            .unwrap()
            .map(|(_, bytes)| Checkpoint::from_bytes(&bytes).unwrap())
    };
    let stored_epoch = |c: &TestCluster, room: RoomId| stored(c, room).map(|cp| cp.epoch);
    // The stale copy's state: a pawn at health 55 (the live copy never saw it).
    let holds_55 = |cp: &Checkpoint| {
        let v = postcard::to_stdvec(&55u16).unwrap();
        cp.state
            .entities
            .iter()
            .any(|(_, comps)| comps.iter().any(|(name, bytes)| name == "duel.health" && *bytes == v))
    };
    let mut c = cluster(2);
    let alice = c.guest("alice");
    let bob = c.guest("bob");
    let (ca, cb, room) = connected_pair(&mut c, &alice, &bob);
    let stale = c.room_server(room).unwrap().to_owned();
    assert!(
        c.run_until(600, |c| stored_epoch(c, room).is_some()),
        "checkpoint stored"
    );
    let stale_epoch = stored_epoch(&c, room).unwrap();

    c.set_fleet_outage(&stale, true);
    let lease = c.player_service().config.fleet.lease_ttl;
    let dead = c.run_until((lease.as_secs_f64() * 60.0) as u64 + 180, |c| {
        c.player_service()
            .fleet
            .lock()
            .server(&stale)
            .is_none_or(|s| s.state == ServerState::Dead)
    });
    assert!(dead, "lease expired");
    let moved = c.rejoin(&bob).unwrap().expect("restorable");
    assert_ne!(moved.server_id, stale);
    c.reconnect(cb, &moved);
    assert!(c.run_until(600, |c| c.client_net(cb).is_connected()));
    let live = run_until_bg(&mut c, 900, |c| stored_epoch(c, room).is_some_and(|e| e > stale_epoch));
    assert!(live, "the live copy stored its own checkpoint");
    let live_epoch = stored_epoch(&c, room).unwrap();

    // Alice plays on in the stale copy, which captures that state, then crashes and is rebuilt
    // from it (still partitioned).
    send(&mut c, ca, &SetHealth(55));
    assert!(c.run_until(120, |c| {
        c.room_on(&stale, room)
            .is_some_and(|app| pawn_health(app, player(&alice)) == Some(55))
    }));
    let stale_room = c.game_server(&stale).unwrap().rooms.get(room).expect("stale copy");
    stale_room.command(RoomCommand::Checkpoint).unwrap();
    c.step_ticks(2);
    c.crash_room_on(&stale, room);
    assert!(c.run_until(60, |c| c.room_crashes() == 1), "the stale copy crashed");
    let rebuilt_epoch = c
        .room_on(&stale, room)
        .expect("the stale copy was rebuilt")
        .resources
        .expect::<NetServer>()
        .config()
        .room_epoch;
    assert!(
        rebuilt_epoch > stale_epoch && rebuilt_epoch < live_epoch,
        "rebuilt stale copy at epoch {rebuilt_epoch}: between its own ({stale_epoch}) and the live copy's ({live_epoch})"
    );
    // Its checkpoints never replace the live copy's.
    stale_room.command(RoomCommand::Checkpoint).unwrap();
    c.step_ticks(2);
    let _ = c.settle(Duration::from_millis(500), |_| false);
    let cp = stored(&c, room).expect("stored");
    assert!(
        cp.epoch >= live_epoch && !holds_55(&cp),
        "the stale copy's checkpoint (epoch {}) replaced the live copy's",
        cp.epoch
    );
    // Its normal close keeps the live copy's checkpoint.
    stale_room
        .command(RoomCommand::Close {
            reason: "match over".into(),
        })
        .unwrap();
    assert!(c.run_until(120, |c| !c.rooms_of(&stale).contains(&room)));
    let _ = c.settle(Duration::from_millis(500), |_| false);
    let cp = stored(&c, room);
    assert!(
        cp.as_ref().is_some_and(|cp| cp.epoch >= live_epoch && !holds_55(cp)),
        "the live copy's checkpoint survives the rebuilt stale copy's close: {:?}",
        cp.map(|cp| cp.key())
    );
}
