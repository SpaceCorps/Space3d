# Space3d for Claude Code

This plugin teaches Claude Code how to build games with Space3d, a cross-platform 3D game engine
written in Rust (desktop, web, Android and iOS). It contains one skill, `space3d-game`, which
guides Claude through creating a game, editing it, testing it with simulated input and golden
screenshots, looking at the result headlessly, and shipping it. It also carries the engine's hard
rules, such as a deterministic fixed tick and never blocking the frame, so that the code Claude
writes fits the engine.

## Before you start

The skill drives the `space3d` command-line tool, which you install once from source. It needs
Rust 1.98 or newer (https://rustup.rs) and git:

```sh
cargo install --git https://github.com/SpaceCorps/Space3d space3d-cli
space3d doctor
```

`space3d doctor` lists what else your machine needs for each platform and prints the exact command
for anything missing. There are no prebuilt binaries yet, so the first install compiles for a few
minutes.

## Install the plugin

```text
/plugin marketplace add SpaceCorps/Space3d
/plugin install space3d@space3d
```

Then ask Claude for a game, for example: "Make a top-down asteroid shooter. Add tests, run
space3d test, and show me a screenshot." Setup from zero, as plain text:
https://space3d.dev/claude/SETUP.md

## What this plugin runs, sends and fetches

The plugin is only text: a skill and an `AGENTS.md` template. It contains no scripts, no hooks, no
MCP server and no code that runs by itself.

- It runs: only what Claude runs on your behalf while following the skill, which are your local
  `space3d` and `cargo` commands (`space3d new`, `space3d test`, `space3d run --headless`,
  `space3d build`) inside your project folder. Publishing a release (`space3d publish`) is done
  only after Claude asks you.
- It sends: nothing. The plugin makes no network calls of its own and sends no data anywhere. It
  reads and stores no personal data.
- It fetches: nothing by itself. When you or Claude install the CLI, `cargo install` downloads the
  Space3d source from GitHub and its dependencies from crates.io, and a new game's dependencies
  are fetched the same way. The skill and docs are also published at https://space3d.dev.

## Links

- Documentation: https://space3d.dev/docs
- Source and issues: https://github.com/SpaceCorps/Space3d
- Licence: Apache-2.0
