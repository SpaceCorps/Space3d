# Space3d — contributor & agent guide

Rust cross-platform game engine (desktop, web/wasm, Android, iOS). Cargo workspace, edition 2024, Rust 1.98.1.

## Rules
1. **Latest stable versions only.** Every dependency is pinned to its newest stable release
   (no alpha/beta/rc). Before adding a crate, check `cargo search --limit 1 <crate>` /
   crates.io `max_stable_version`. Prefer `[workspace.dependencies]` entries in the root
   `Cargo.toml` via `foo.workspace = true`.
2. **Never block the frame.** I/O goes through `space3d_tasks::{spawn, AsyncQueue}`; CPU work
   through `spawn_compute`. Systems poll `Task::poll()` — never `.block()` in engine code.
3. **Everything is a `Plugin`.** Each crate exposes `XxxPlugin: space3d_core::Plugin`.
   Systems are `FnMut(&mut SystemCtx)` registered on a `Stage`.
4. **Deterministic simulation** runs in `Stage::FixedUpdate` at `TickRate` (default 60 Hz).
   Tests drive time with `App::step_ticks(n)` — never wall-clock.
5. **Cross-platform.** All runtime crates must `cargo check --target wasm32-unknown-unknown`.
   Native-only pieces are gated with `#[cfg(not(target_arch = "wasm32"))]` and provide a web
   fallback or a clear `Unsupported` error.
6. **Testable.** Unit tests next to code, integration tests in `tests/`. GPU tests must skip
   (not fail) when no adapter is available. Visual output is verified with golden screenshots
   via `space3d-test`.
7. **Never steal focus.** Windowed runs from tests, scripts, CI or agents must not take the
   developer's focus. The platform runner's background mode (`space3d_platform::background`)
   is on automatically when `CI`, `CLAUDECODE`, `SPACE3D_EXIT_AFTER_FRAMES`/`_SECS` is set, or
   via `SPACE3D_BACKGROUND=1` / `set_background_mode(true)`. Prefer headless runs
   (`SPACE3D_HEADLESS=1`, `RenderPlugin::headless`) whenever a window isn't needed.
8. `cargo fmt`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`
   must pass.
9. **Patched crates live in `third_party/`** (`[patch.crates-io]`, not workspace members):
   `epaint` carries kinsoku line breaking for Chinese and Japanese and three UAX #14 rules for all
   scripts (no break before U+00A0, none after an ASCII opening bracket, none next to a word
   joiner U+2060). After an egui bump run
   `python3 scripts/update-epaint.py`; after editing `third_party/epaint`, `--refresh` the patch
   (`third_party/README.md`).

## Crate map
| crate | role |
|---|---|
| `space3d-core` | App, Plugin, ECS (hecs), Resources, Events, Stage schedule, fixed tickrate, Transform |
| `space3d-tasks` | async runtime bridge (tokio native / wasm-bindgen-futures web), `Task`, `AsyncQueue` |
| `space3d-render` | wgpu renderer: meshes, materials, cameras, lights, instancing, HDR frame modes + tonemapping, post (bloom, FXAA, vignette, CA, grain), RenderNode extension API, overlay stages, headless capture; see docs/graphics/ |
| `space3d-particles` | GPU/CPU particle system: emitters, effect prefabs (RON), curves, flipbooks, soft particles, bridges; see docs/graphics/particles.md |
| `space3d-assets` | asset server, handles, image/glTF/OBJ loading, hot reload, texture & mesh editing ops |
| `space3d-net` | HTTP client/server, WebSocket client/server, native + web |
| `space3d-replication` | multiplayer netcode shared by client + server (wasm-clean): transports, protocol, delta snapshots, interest, prediction/reconciliation, interpolation, test harness |
| `space3d-server` | dedicated authoritative game server + player service (auth, matchmaking, lobbies, fleet, checkpoints, admin/metrics); see docs/server.md |
| `space3d-db` | SQLite (rusqlite, native), KV + save slots + migrations, web fallback |
| `space3d-github` | GitHub REST: releases, issues (crash reports), CI templates |
| `space3d-audio` | kira-based audio, spatial emitters |
| `space3d-physics` | rigid bodies, colliders, raycasts |
| `space3d-scene` | scene/prefab serialization (RON), save/load |
| `space3d-platform` | winit runner: desktop, web, Android, iOS; input |
| `space3d-ui` | egui integration, debug overlay, profiler |
| `space3d-editor` | scene inspector, texture editor, model editor |
| `space3d-stories` | storybook: registry + viewer + batch screenshotting |
| `space3d-test` | test harness: headless app, input simulation, golden screenshot diffing |
| `space3d-cli` | `space3d` binary: new/run/build/test/stories/snapshot/publish |
| `space3d` | facade + prelude |
