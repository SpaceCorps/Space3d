//! A room and a client over a real loopback WebSocket (`space3d_net::WsServer` ↔
//! `WsConnector`), with the system clock: validates the WS client transport end to end. The
//! production gateway (`space3d-server`'s `WsServerTransport` + gatekeeper) is tested there;
//! this test bridges `WsServer` events to the room inbox directly, the way the harness does.
#![cfg(not(target_arch = "wasm32"))]

mod common;

use common::*;
use space3d_core::prelude::*;
use space3d_net::{ClientId, Message, ServerEvent, WsServer};
use space3d_replication::bots::{install_bot, scripted};
use space3d_replication::prelude::*;
use space3d_replication::server::handshake::{parse_hello, reject_packet};
use space3d_replication::server::{JoinRequest, RoomInbound, room_channel};
use space3d_replication::transport::{CloseCode, Lane, LinkTx, PeerInfo, TransportCaps, TransportError};
use space3d_replication::{LinkId, NetEntity};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Server end of one WebSocket link.
struct Tx {
    server: WsServer,
    client: ClientId,
}

impl LinkTx for Tx {
    fn caps(&self) -> TransportCaps {
        TransportCaps::stream(1 << 20)
    }

    fn send(&self, _lane: Lane, packet: Vec<u8>) -> Result<(), TransportError> {
        self.server
            .send(self.client, Message::Binary(packet))
            .map_err(|_| TransportError::Closed)
    }

    fn backlog_bytes(&self) -> usize {
        self.server.backlog_bytes(self.client).unwrap_or(0)
    }

    fn close(&self, code: CloseCode, reason: &str) {
        self.server.disconnect_with(self.client, code.0, reason);
    }

    fn is_open(&self) -> bool {
        self.server.is_connected(self.client)
    }
}

#[test]
fn room_and_client_over_a_real_websocket() {
    let ws = WsServer::new();
    let handle = ws.listen(([127, 0, 0, 1], 0)).expect("listen");
    let url = format!("ws://{}/play", handle.local_addr());

    let mut room = App::new().with_tickrate(60.0);
    let (sender, inbox) = room_channel(1024);
    room.insert_resource(inbox);
    room.add_plugin(ServerReplicationPlugin {
        config: ServerNetConfig {
            game_id: "ws-test".into(),
            ..Default::default()
        },
    })
    .add_plugin(TestProtocol)
    .add_plugin(TestRules);

    let mut client = App::new().with_tickrate(60.0);
    client
        .add_plugin(ClientReplicationPlugin {
            config: ClientConfig {
                game_id: "ws-test".into(),
                ..Default::default()
            },
        })
        .add_plugin(TestProtocol);
    install_bot(&mut client, scripted(|_| Move { x: 1, z: 0 }));
    client.resources.expect_mut::<NetClient>().connect(ConnectTarget::Dev {
        url,
        name: "alice".into(),
    });

    let mut joined: HashSet<ClientId> = HashSet::new();
    let step = Duration::from_secs_f64(1.0 / 60.0);
    let started = Instant::now();
    let mut last = Instant::now();
    let mut connected_for = 0u32;
    while started.elapsed() < Duration::from_secs(10) {
        for ev in ws.drain() {
            let link = LinkId(ev.client().0);
            match ev {
                ServerEvent::Connected { .. } => {}
                ServerEvent::Message { client: c, message } => {
                    let Message::Binary(bytes) = message else { continue };
                    if joined.contains(&c) {
                        let _ = sender.try_send(RoomInbound::Packet { link, bytes });
                        continue;
                    }
                    let tx: Arc<dyn LinkTx> = Arc::new(Tx {
                        server: ws.clone(),
                        client: c,
                    });
                    match parse_hello(&bytes) {
                        Ok(hello) => {
                            joined.insert(c);
                            let req = JoinRequest {
                                link,
                                tx,
                                peer: PeerInfo {
                                    transport: "ws",
                                    ..Default::default()
                                },
                                hello,
                                identity: None, // dev auth: the room derives it
                            };
                            let _ = sender.try_send(RoomInbound::Join(Box::new(req)));
                        }
                        Err(code) => {
                            let _ = tx.send(Lane::Reliable, reject_packet(code, "bad hello", None));
                            tx.close(CloseCode::POLICY, "bad hello");
                        }
                    }
                }
                ServerEvent::Disconnected { client: c, reason } => {
                    if joined.remove(&c) {
                        let _ = sender.try_send(RoomInbound::Closed { link, reason });
                    }
                }
            }
        }
        let now = Instant::now();
        let dt = now - last;
        last = now;
        room.update(dt);
        client.update(dt);
        let net = client.resources.expect::<NetClient>();
        if net.is_connected() && controlled(&client).is_some() {
            connected_for += 1;
            if connected_for > 90 {
                break;
            }
        }
        std::thread::sleep(step.saturating_sub(now.elapsed()));
    }
    let net = client.resources.expect::<NetClient>();
    assert!(net.is_connected(), "state {:?}", net.state());
    // Client and room run on this one thread in a sleeping loop, so the measured RTT includes the
    // test's own scheduling (201.8 ms on a shared macOS runner, over loopback): the bound only
    // checks that the estimate is a sane loopback value, not a latency budget.
    assert!(net.stats().rtt < Duration::from_secs(2), "{:?}", net.stats());
    // The pawn moved on the server, and the client predicts it close to the server's position
    // (plus its lead of a few ticks).
    let server_pawn = pawn(&room, net.player_id().unwrap()).unwrap();
    let sx = room.world.get::<&Transform>(server_pawn).unwrap().translation.x;
    let ce = controlled(&client).unwrap();
    let cx = client.world.get::<&Transform>(ce).unwrap().translation.x;
    assert!(sx > 0.5, "server pawn did not move: {sx}");
    assert!((cx - sx).abs() < 1.5, "client {cx} vs server {sx}");
    assert!(client.world.get::<&NetEntity>(ce).is_ok());
    // Orderly leave: the room frees the slot.
    client.resources.expect_mut::<NetClient>().disconnect();
    let deadline = Instant::now() + Duration::from_secs(5);
    // Until the slot is freed, not merely until the link dropped.
    while !room.resources.expect::<NetServer>().clients().is_empty() && Instant::now() < deadline {
        for ev in ws.drain() {
            let link = LinkId(ev.client().0);
            match ev {
                ServerEvent::Message {
                    message: Message::Binary(bytes),
                    ..
                } => {
                    let _ = sender.try_send(RoomInbound::Packet { link, bytes });
                }
                ServerEvent::Disconnected { reason, .. } => {
                    let _ = sender.try_send(RoomInbound::Closed { link, reason });
                }
                _ => {}
            }
        }
        room.update(step);
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(room.resources.expect::<NetServer>().connected_count(), 0);
    assert!(room.resources.expect::<NetServer>().clients().is_empty());
    drop(handle);
}
