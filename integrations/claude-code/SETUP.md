# Build a game with Space3d and Claude Code: setup from zero

Space3d is a cross-platform 3D game engine in Rust, made to be driven by Claude Code. This page is
plain text on purpose, so that an agent can read it. The same steps are on
https://space3d.dev/claude-code.html.

- Source (public, no login): https://github.com/SpaceCorps/Space3d
- Whole documentation in one file: https://space3d.dev/llms-full.txt
- Index for agents: https://space3d.dev/llms.txt

There are no prebuilt binaries yet. The `space3d` command-line tool is built from source on your
machine, which takes a few minutes the first time.

## Paste this into Claude Code

Replace `<YOUR GAME IDEA>`, then paste the whole block into Claude Code in an empty folder.

```text
I want to build a game with the Space3d engine (Rust). My idea: <YOUR GAME IDEA>

Do this, and tell me what you ran and what happened:

1. Learn the engine. Read https://space3d.dev/llms-full.txt (the README and all docs in one
   file). If you cannot fetch URLs, use the public repository instead:
   https://github.com/SpaceCorps/Space3d (clone it, read README.md, docs/getting-started.md and
   integrations/claude-code/).
2. Check the tools. You need git and Rust 1.98 or newer (`rustup update stable`). If
   `space3d --version` fails, install the CLI (a few minutes, it compiles from source):
   cargo install --git https://github.com/SpaceCorps/Space3d space3d-cli
   Then run `space3d doctor` and apply the fixes it prints.
3. Scaffold. Run `space3d new <snake_case_name> --template 3d` (templates: basic, 3d,
   multiplayer) and cd into it. Save https://space3d.dev/claude/AGENTS.md as AGENTS.md and
   create CLAUDE.md containing the single line @AGENTS.md
4. Build my idea in src/lib.rs. After every change add or extend a test in tests/ and run
   `space3d test`. Finish with `space3d run --headless --frames 120` and look at a screenshot.

Never open a window that steals my focus: use headless runs. Do not say it is done until
`space3d test` is green; show me the output.
```

## By hand

### 1. Prerequisites

- Rust 1.98 or newer: install https://rustup.rs, then `rustup update stable`.
- git.
- A C linker: Windows, Visual Studio Build Tools ("Desktop development with C++"); macOS,
  `xcode-select --install`; Linux, gcc or clang and pkg-config.

### 2. Install the CLI

```sh
cargo install --git https://github.com/SpaceCorps/Space3d space3d-cli
space3d --version
```

### 3. Check the machine

```sh
space3d doctor
```

It lists what is installed for each platform (desktop, web, Android, iOS) and prints the exact
command for anything missing, such as the wasm target.

### 4. Optional: the Claude Code plugin

```text
/plugin marketplace add SpaceCorps/Space3d
/plugin install space3d@space3d
```

It adds the `space3d-game` skill: scaffold, the edit, test and screenshot loop, and the engine's
hard rules. The skill text is at https://space3d.dev/claude/SKILL.md. Without the plugin, the
`AGENTS.md` below carries the same rules.

### 5. Scaffold a game

```sh
space3d new asteroids --template 3d
cd asteroids
curl -fsSL https://space3d.dev/claude/AGENTS.md -o AGENTS.md
echo '@AGENTS.md' > CLAUDE.md
```

On Windows PowerShell, use `Invoke-WebRequest https://space3d.dev/claude/AGENTS.md -OutFile AGENTS.md`
and `Set-Content CLAUDE.md '@AGENTS.md'`.

### 6. Ask Claude

```text
claude "Make a top-down asteroid shooter. Add tests, run space3d test, show me a screenshot."
```

Claude can run, test and screenshot headless, so no window pops up. There is no MCP server yet:
the `space3d` CLI is the tool surface.

## If something fails

- `space3d: command not found`: add `~/.cargo/bin` (Windows: `%USERPROFILE%\.cargo\bin`) to PATH.
- A compile error while installing: check `rustc --version` is 1.98 or newer.
- Anything else: run `space3d doctor`, and open an issue with its output at
  https://github.com/SpaceCorps/Space3d/issues
