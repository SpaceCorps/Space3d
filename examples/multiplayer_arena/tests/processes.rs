//! The shipped binaries as separate processes over real localhost sockets and real time (this
//! is what a developer or an orchestrator runs); every wait has a deadline.
//!
//! * All-in-one: the `server` (player service + game server) and two `client --headless --bot`
//!   processes that log in as guests, get join tickets through quick play and play in the same
//!   room; SIGTERM drains the server (exit code 0).
//! * Split drain handoff: a player service and two game servers (`server --role ...`), a
//!   persistent arena queue and two bots; SIGTERM the game server hosting their room mid-play:
//!   the room moves to the other game server and both bots play on there. Bots that quick play
//!   while the room moves join it there too (one world, never a second copy).
//! * Split crash: SIGKILL the game server hosting the room; its bots are back in the room,
//!   restored on the other game server, about a lease later.
//!
//! Every process's whole output (tracing goes to stdout) is captured and must hold no ERROR
//! line and no panic.
#![cfg(all(unix, feature = "client"))]

mod common;

use common::{Proc, Server, bot, free_port, http_get, server_cmd, wait_ready};
use std::time::{Duration, Instant};

#[test]
fn server_and_two_headless_bot_clients_over_localhost() {
    let mut server = Server::start(&[]);
    let ps = format!("http://127.0.0.1:{}", server.port);
    let mut bots = [bot("bot-1", &ps, 4), bot("bot-2", &ps, 4)];
    for bot in &mut bots {
        let status = bot.wait(Duration::from_secs(20)).expect("client exits on its own");
        let out = bot.output();
        assert!(status.success(), "client failed ({status}):\n{out}");
        assert!(out.contains("state=Connected"), "{out}");
        assert!(out.contains("cubes=2"), "both bots see both cubes: {out}");
        bot.assert_no_errors();
    }

    // Graceful shutdown: SIGTERM → drain (3 s in the dev config) → exit 0.
    let status = server.stop();
    assert!(status.success(), "server exit {status}:\n{}", server.proc.output());
    server.proc.assert_no_errors();
}

/// Regression: `client --help` (and any option it did not know) printed `unknown option` and went
/// on to start the windowed client: renderer, window and a connection attempt, on a developer's
/// machine when an agent only wanted to read the options. It now answers and exits before
/// anything is initialised. Should this ever regress, the environment keeps the stray client
/// in the background and ends it after two frames instead of leaving it open.
#[test]
fn client_help_and_bad_options_exit_before_starting_anything() {
    let run = |args: &[&str]| {
        let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_client"));
        cmd.args(args)
            .env("SPACE3D_BACKGROUND", "1")
            .env("SPACE3D_EXIT_AFTER_FRAMES", "2")
            .env("RUST_LOG", "info");
        let mut client = Proc::spawn("client", cmd);
        let status = client.wait(Duration::from_secs(20)).expect("exits at once");
        let out = client.output();
        assert!(
            !out.contains("renderer initialised") && !out.contains("state="),
            "{args:?} started the client:\n{out}"
        );
        (status.code(), out)
    };
    for flag in ["--help", "-h"] {
        let (code, out) = run(&[flag]);
        assert_eq!(code, Some(0), "{flag}:\n{out}");
        assert!(out.contains("usage: client") && out.contains("--ps"), "{flag}:\n{out}");
    }
    for (args, why) in [
        (&["--bogus"][..], "unknown option"),
        (&["--ps"], "needs a value"),
        (&["--seconds", "soon"], "--seconds"),
    ] {
        let (code, out) = run(args);
        assert_eq!(code, Some(2), "{args:?}:\n{out}");
        assert!(out.contains(why) && out.contains("usage: client"), "{args:?}:\n{out}");
    }
}

/// Regression (test harness): the process tests piped stderr only and looked for "ERROR" there,
/// but every tracing line — ERROR included — goes to stdout, so a process logging errors
/// passed. The harness now captures both streams; this checks it sees an error the server
/// only reports through tracing (stderr gets a plain `startup failed: ...`).
#[test]
fn errors_a_process_logs_are_seen() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("not-a-dir");
    std::fs::write(&file, b"").unwrap();
    // Its database would live under a regular file: startup fails.
    let mut cmd = server_cmd(&file.join("data"));
    cmd.env("SPACE3D__GAME__BIND", format!("127.0.0.1:{}", free_port()))
        .env("SPACE3D__PLAYER_SERVICE__BIND", format!("127.0.0.1:{}", free_port()))
        .env("SPACE3D__ADMIN__BIND", format!("127.0.0.1:{}", free_port()));
    let mut server = Proc::spawn("server", cmd);
    let status = server.wait(Duration::from_secs(20)).expect("exits");
    assert!(!status.success());
    let errors = server.errors();
    assert!(
        errors.iter().any(|l| l.contains("startup failed")),
        "the ERROR line (stdout) is seen: {errors:?}\n{}",
        server.output()
    );
}

/// One game server of the split deployment (`server --role game-server`).
struct GameServer {
    proc: Proc,
    admin: u16,
}

/// Rooms a game server reports on its admin API: `(room id, players)`.
fn rooms(admin: u16) -> Vec<(String, u64)> {
    let Some((true, body)) = http_get(admin, "/admin/v1/rooms") else {
        return Vec::new();
    };
    let rooms: Vec<serde_json::Value> = serde_json::from_str(&body).unwrap_or_default();
    rooms
        .iter()
        .filter_map(|r| Some((r["room_id"].as_str()?.to_owned(), r["players"].as_u64()?)))
        .collect()
}

fn wait_for<T>(what: &str, limit: Duration, mut f: impl FnMut() -> Option<T>) -> T {
    let end = Instant::now() + limit;
    loop {
        if let Some(v) = f() {
            return v;
        }
        assert!(Instant::now() < end, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// The split deployment: a player service and two game servers (`server --role ...`) with a
/// persistent arena queue (one world per mode, which moves on drain).
struct Split {
    ps: Proc,
    ps_url: String,
    game_servers: Vec<GameServer>,
    _dir: tempfile::TempDir,
}

impl Split {
    /// Start every process, `extra_env` set on each, and wait until all are ready.
    fn start(extra_env: &[(&str, &str)]) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let queues = r#"[{ name = "arena", mode = "arena", teams = 1, team_size = 16, min_players = 1, max_wait = "5s", backfill = true, params = { persistent = "true" } }]"#;
        let fleet_token = "process-test-fleet-token-0123456789";
        let common = |cmd: &mut std::process::Command| {
            cmd.env("SPACE3D__MATCHMAKING__QUEUES", queues)
                .env("SPACE3D__LOG__FILTER", "info")
                .env("SPACE3D_FLEET_TOKEN", fleet_token)
                // Every bot comes from 127.0.0.1: the stock per-IP limits are for public clients.
                .env("SPACE3D__PLAYER_SERVICE__HTTP_RPS_PER_IP", "1000")
                .env("SPACE3D__PLAYER_SERVICE__LOGIN_ATTEMPTS_PER_MIN", "600");
            for (k, v) in extra_env {
                cmd.env(k, v);
            }
        };
        let (ps_port, ps_admin) = (free_port(), free_port());
        let mut cmd = server_cmd(&dir.path().join("ps"));
        cmd.args(["--role", "player-service"])
            .env("SPACE3D__PLAYER_SERVICE__BIND", format!("127.0.0.1:{ps_port}"))
            .env("SPACE3D__ADMIN__BIND", format!("127.0.0.1:{ps_admin}"));
        common(&mut cmd);
        let mut ps = Proc::spawn("player-service", cmd);
        wait_ready(&mut ps, ps_admin, "player service");
        let mut game_servers: Vec<GameServer> = ["gs-1", "gs-2"]
            .into_iter()
            .map(|id| {
                let (port, admin) = (free_port(), free_port());
                let mut cmd = server_cmd(&dir.path().join(id));
                cmd.args(["--role", "game-server"])
                    .env("SPACE3D__SERVER_ID", id)
                    .env("SPACE3D__GAME__BIND", format!("127.0.0.1:{port}"))
                    .env("SPACE3D__GAME__PUBLIC_URL", format!("ws://127.0.0.1:{port}/play"))
                    .env("SPACE3D__ADMIN__BIND", format!("127.0.0.1:{admin}"))
                    .env(
                        "SPACE3D__FLEET__PLAYER_SERVICE_URL",
                        format!("ws://127.0.0.1:{ps_port}/internal/v1/fleet"),
                    )
                    .env(
                        "SPACE3D__FLEET__PLAYER_SERVICE_HTTP",
                        format!("http://127.0.0.1:{ps_port}"),
                    );
                common(&mut cmd);
                GameServer {
                    proc: Proc::spawn(id, cmd),
                    admin,
                }
            })
            .collect();
        for gs in &mut game_servers {
            wait_ready(&mut gs.proc, gs.admin, "game server");
        }
        Self {
            ps,
            ps_url: format!("http://127.0.0.1:{ps_port}"),
            game_servers,
            _dir: dir,
        }
    }

    /// Wait until one room holds `players`: `(index of its game server, room id)`.
    fn room_with(&self, players: u64) -> (usize, String) {
        wait_for(&format!("{players} bots in one room"), Duration::from_secs(15), || {
            self.game_servers.iter().enumerate().find_map(|(i, gs)| {
                rooms(gs.admin)
                    .into_iter()
                    .find(|(_, n)| *n == players)
                    .map(|(room, _)| (i, room))
            })
        })
    }

    /// SIGTERM the game servers in `running`, then the player service: each drains and exits 0.
    fn stop(&mut self, running: &[usize]) {
        for &i in running {
            let gs = &mut self.game_servers[i];
            gs.proc.term();
            let status = gs.proc.wait(Duration::from_secs(20)).expect("game server exits");
            assert!(status.success(), "game server exit {status}:\n{}", gs.proc.output());
        }
        self.ps.term();
        let status = self.ps.wait(Duration::from_secs(20)).expect("player service exits");
        assert!(status.success(), "player service exit {status}:\n{}", self.ps.output());
    }

    /// Panic if any server process logged an error.
    fn assert_no_errors(&self) {
        for p in self.game_servers.iter().map(|g| &g.proc).chain([&self.ps]) {
            p.assert_no_errors();
        }
    }
}

/// Each bot exits 0 on its own, connected, seeing `cubes` cubes, with no error logged.
fn assert_bots_play_on(bots: &mut [Proc], cubes: usize) {
    for bot in bots {
        let status = bot.wait(Duration::from_secs(30)).expect("client exits on its own");
        let out = bot.output();
        assert!(status.success(), "client failed ({status}):\n{out}");
        assert!(out.contains("state=Connected"), "{out}");
        assert!(
            out.contains(&format!("cubes={cubes} ")),
            "every bot sees {cubes} cubes: {out}"
        );
        bot.assert_no_errors();
    }
}

/// Regression (drain handoff over real processes, 3 of 4 runs failed): the draining game server
/// froze the room and disconnected its clients before the player service knew the room was
/// moving, so their rejoin got a ticket straight back to the draining server
/// (`Reject(ServerDraining)`), and the client burnt its rejoin budget in ~170 ms. Now the room
/// keeps its clients until the player service acknowledged the move, the player service holds
/// their rejoins until the room is stored, and the client honours retry hints with backoff.
/// Also checks that a drained game server leaves the fleet with a clean close (the player
/// service used to warn about a connection reset).
#[test]
fn a_draining_game_server_hands_its_persistent_room_to_another_over_real_processes() {
    let mut split = Split::start(&[]);
    let mut bots = [bot("bot-1", &split.ps_url, 10), bot("bot-2", &split.ps_url, 10)];
    let (host, room) = split.room_with(2);
    // Let them play a moment, then drain the hosting game server (SIGTERM).
    std::thread::sleep(Duration::from_millis(500));
    let other = 1 - host;
    split.game_servers[host].proc.term();
    let status = split.game_servers[host]
        .proc
        .wait(Duration::from_secs(20))
        .expect("the drained game server exits");
    assert!(
        status.success(),
        "drained game server exit {status}:\n{}",
        split.game_servers[host].proc.output()
    );
    // The room comes back on the other game server with both players in it.
    wait_for("the room restored with both bots", Duration::from_secs(10), || {
        rooms(split.game_servers[other].admin)
            .contains(&(room.clone(), 2))
            .then_some(())
    });
    assert_bots_play_on(&mut bots, 2);
    split.stop(&[other]);
    split.assert_no_errors();
    let ps_out = split.ps.output();
    assert!(
        !ps_out.contains("reset without closing handshake")
            && !split.ps.warnings().iter().any(|l| l.contains("fleet link closed")),
        "drained game servers leave the fleet with a clean close:\n{ps_out}"
    );
    assert!(ps_out.contains("fleet link closed (server left the fleet)"), "{ps_out}");
    let host_out = split.game_servers[host].proc.output();
    assert!(
        !host_out.contains("did not acknowledge the move"),
        "the player service acknowledged the move:\n{host_out}"
    );
}

/// Regression (verifier, real processes; 3 of 3 runs whose quick play came in the handoff
/// window): a player whose quick play arrived the moment the game server hosting the persistent
/// world began to drain got a ticket into the room being moved, which refused it as closed for
/// good (`Reject(RoomClosed)`). With nothing to rejoin, their next quick play opened a second
/// world on the other game server, next to the restored one, and the players stayed split for
/// good. Now the moving room refuses such a ticket as transient (`ServerDraining` with a retry
/// delay) and the player service holds a quick play into a moving world until it has moved:
/// bots that start while the world moves end up in it with the others.
///
/// (Test fix: the one-world check used to list the rooms after the bots had left on their own,
/// when the world reported 0 players, so it failed most runs; it now checks the room list seen
/// at the moment all four bots were in the moved world.)
#[test]
fn players_who_quick_play_during_a_handoff_join_the_moved_room_over_real_processes() {
    let mut split = Split::start(&[]);
    let mut bots = vec![bot("bot-1", &split.ps_url, 10), bot("bot-2", &split.ps_url, 10)];
    let (host, room) = split.room_with(2);
    std::thread::sleep(Duration::from_millis(300));
    let other = 1 - host;
    split.game_servers[host].proc.term();
    // Quick plays in the handoff window: one at once, one while the room freezes and moves.
    bots.push(bot("bot-3", &split.ps_url, 9));
    std::thread::sleep(Duration::from_millis(250));
    bots.push(bot("bot-4", &split.ps_url, 9));
    let status = split.game_servers[host]
        .proc
        .wait(Duration::from_secs(20))
        .expect("the drained game server exits");
    assert!(
        status.success(),
        "drained game server exit {status}:\n{}",
        split.game_servers[host].proc.output()
    );
    // Observed while the bots are in it (they leave on their own a few seconds later): the
    // game server's whole room list at the moment all four are in the moved world.
    let listed = wait_for("the world restored with all four bots", Duration::from_secs(15), || {
        let listed = rooms(split.game_servers[other].admin);
        listed.contains(&(room.clone(), 4)).then_some(listed)
    });
    assert_eq!(
        listed,
        vec![(room.clone(), 4)],
        "one world, not a second copy next to it"
    );
    assert_bots_play_on(&mut bots, 4);
    // Still that one world once they left (by id: its player count drops as they go).
    let ids: Vec<String> = rooms(split.game_servers[other].admin)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    assert_eq!(ids, vec![room], "one world after the bots left");
    split.stop(&[other]);
    split.assert_no_errors();
}

/// Regression (verifier, real processes): after a game server crashed (SIGKILL), its players
/// kept resuming against the dead server for the client's whole reconnect schedule (10
/// attempts over ~56 s, each refused at once) before they asked the player service, which had
/// the room restorable one lease later. Now a client stops resuming when the resume window is
/// over, and the example client asks the player service every 2 s while its reconnect keeps
/// failing: with a 4 s lease both bots play on in the restored room a few seconds after the
/// crash.
#[test]
fn a_crashed_game_server_s_room_is_back_about_a_lease_later_over_real_processes() {
    let mut split = Split::start(&[("SPACE3D__FLEET__LEASE_TTL", "4s"), ("SPACE3D__FLEET__HEARTBEAT", "1s")]);
    let mut bots = [bot("bot-1", &split.ps_url, 14), bot("bot-2", &split.ps_url, 14)];
    let (host, room) = split.room_with(2);
    // Let the room store a checkpoint with both players, then crash its game server.
    std::thread::sleep(Duration::from_millis(1_000));
    let other = 1 - host;
    split.game_servers[host].proc.kill();
    let crashed = Instant::now();
    split.game_servers[host]
        .proc
        .wait(Duration::from_secs(5))
        .expect("the killed game server is gone");
    wait_for("the room restored with both bots", Duration::from_secs(20), || {
        rooms(split.game_servers[other].admin)
            .contains(&(room.clone(), 2))
            .then_some(())
    });
    // The lease (4 s, swept) plus at most one 2 s probe period, with slack for a loaded
    // machine; resuming until the 20 s resume window was over would take 20 s, the old
    // reconnect schedule ~57 s.
    let took = crashed.elapsed();
    assert!(took < Duration::from_secs(12), "room back {took:?} after the crash");
    assert_bots_play_on(&mut bots, 2);
    split.stop(&[other]);
    split.assert_no_errors();
}
