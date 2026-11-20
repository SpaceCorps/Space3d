//! Pass-through commands: `run`, `stories`, `editor`, `bench`.

use super::cargo_flags;
use crate::Ctx;
use crate::args::{BenchArgs, EditorArgs, RunArgs, StoriesArgs};
use crate::process::Cmd;
use crate::project::Project;
use anyhow::{Result, bail};

/// `cargo run` for the game binary; returns the game's exit code.
pub fn run(ctx: &Ctx, args: &RunArgs) -> Result<i32> {
    let project = Project::load(&ctx.cwd, args.cargo.package.as_deref())?;
    let bin = project.package().main_bin(args.cargo.bin.as_deref())?;
    let cmd = run_cmd(ctx, &project, args, &bin);
    ctx.exec.status(&cmd)
}

/// The `cargo run` command line for [`run`].
pub fn run_cmd(ctx: &Ctx, project: &Project, args: &RunArgs, bin: &str) -> Cmd {
    let mut sel = args.cargo.clone();
    sel.package.get_or_insert_with(|| project.package().name.clone());
    let mut cmd = cargo_flags(Cmd::cargo("run").cwd(&ctx.cwd), &sel, Some(bin));
    if let Some(n) = args.frames {
        cmd = cmd.env("SPACE3D_EXIT_AFTER_FRAMES", n.to_string());
    }
    if args.headless {
        cmd = cmd.env("SPACE3D_HEADLESS", "1");
    }
    if !args.args.is_empty() {
        cmd = cmd.arg("--").args(&args.args);
    }
    cmd
}

/// `cargo run --bin stories -- [--headless] [--out DIR] [--filter PAT] [--list]`.
pub fn stories(ctx: &Ctx, args: &StoriesArgs) -> Result<i32> {
    let project = Project::load(&ctx.cwd, args.cargo.package.as_deref())?;
    let bin = args.cargo.bin.clone().unwrap_or_else(|| "stories".into());
    if !project.package().has_bin(&bin) {
        bail!(
            "package `{}` has no `{bin}` binary.\n\
             Add `src/bin/stories.rs` that builds your `Storybook` and calls the viewer \
             (projects created with `space3d new` have one), or pass --bin <name>.",
            project.package().name
        );
    }
    let cmd = stories_cmd(ctx, &project, args, &bin);
    ctx.exec.status(&cmd)
}

/// The command line for [`stories`].
pub fn stories_cmd(ctx: &Ctx, project: &Project, args: &StoriesArgs, bin: &str) -> Cmd {
    let mut sel = args.cargo.clone();
    sel.package.get_or_insert_with(|| project.package().name.clone());
    let out = args
        .out
        .as_ref()
        .map(|o| if o.is_absolute() { o.clone() } else { ctx.cwd.join(o) });
    let mut cmd = cargo_flags(Cmd::cargo("run").cwd(&ctx.cwd), &sel, Some(bin)).arg("--");
    cmd = cmd
        .arg_if(args.headless, "--headless")
        .arg_if(args.list, "--list")
        .opt("--out", out.as_deref())
        .opt("--filter", args.filter.as_deref());
    cmd.args(&args.args)
}

/// `cargo run --bin editor` (the project's editor binary wraps `space3d::editor`).
pub fn editor(ctx: &Ctx, args: &EditorArgs) -> Result<i32> {
    let project = Project::load(&ctx.cwd, args.cargo.package.as_deref())?;
    let bin = args.cargo.bin.clone().unwrap_or_else(|| "editor".into());
    if !project.package().has_bin(&bin) {
        bail!(
            "package `{}` has no `{bin}` binary. Create `src/bin/editor.rs`:\n\n    \
             use space3d::prelude::*;\n    \
             fn main() {{ run_editor(EditorConfig::new(\"Editor\").with_setup({}::editor_scene)); }}\n\n\
             where `editor_scene(app: &mut App)` adds your game content, and enable the facade's `editor` \
             feature (`space3d = {{ ..., features = [\"editor\"] }}`) — projects created by `space3d new` \
             have both. Or pass --bin <name>.",
            project.package().name,
            project.package().name.replace('-', "_"),
        );
    }
    let mut sel = args.cargo.clone();
    sel.package.get_or_insert_with(|| project.package().name.clone());
    let mut cmd = cargo_flags(Cmd::cargo("run").cwd(&ctx.cwd), &sel, Some(&bin));
    if !args.args.is_empty() {
        cmd = cmd.arg("--").args(&args.args);
    }
    ctx.exec.status(&cmd)
}

/// `cargo bench [args]`.
pub fn bench(ctx: &Ctx, args: &BenchArgs) -> Result<i32> {
    let mut cmd = Cmd::cargo("bench")
        .cwd(&ctx.cwd)
        .opt("-p", args.cargo.package.as_deref())
        .opt("--features", args.cargo.features.as_deref());
    if let Some(bin) = &args.cargo.bin {
        cmd = cmd.args(["--bench", bin]);
    }
    if !args.args.is_empty() {
        cmd = cmd.arg("--").args(&args.args);
    }
    ctx.exec.status(&cmd)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::CargoSel;
    use crate::process::Exec;
    use std::path::{Path, PathBuf};

    fn ctx() -> Ctx {
        Ctx {
            cwd: PathBuf::from("/games/demo"),
            exec: Exec::default(),
        }
    }

    fn project() -> Project {
        let json = serde_json::json!({
            "packages": [{
                "name": "demo", "version": "0.1.0", "manifest_path": "/games/demo/Cargo.toml",
                "targets": [
                    {"name": "demo", "kind": ["bin"], "crate_types": ["bin"]},
                    {"name": "stories", "kind": ["bin"], "crate_types": ["bin"]}
                ],
                "metadata": null
            }],
            "workspace_root": "/games/demo",
            "target_directory": "/games/demo/target"
        });
        Project::from_metadata_json(&json.to_string(), Path::new("/games/demo"), None).unwrap()
    }

    #[test]
    fn run_command_line() {
        let args = RunArgs {
            cargo: CargoSel {
                release: true,
                ..Default::default()
            },
            frames: Some(60),
            headless: true,
            args: vec!["--level".into(), "3".into()],
        };
        let cmd = run_cmd(&ctx(), &project(), &args, "demo");
        let shown = cmd.display();
        assert!(
            shown.starts_with("SPACE3D_EXIT_AFTER_FRAMES=60 SPACE3D_HEADLESS=1 "),
            "{shown}"
        );
        assert!(
            shown.ends_with("run --release -p demo --bin demo -- --level 3"),
            "{shown}"
        );
    }

    #[test]
    fn stories_command_line() {
        let args = StoriesArgs {
            headless: true,
            out: Some(PathBuf::from("shots")),
            filter: Some("Cube".into()),
            list: false,
            cargo: CargoSel::default(),
            args: vec![],
        };
        // Windows joins paths with a backslash.
        let shown = stories_cmd(&ctx(), &project(), &args, "stories")
            .display()
            .replace('\\', "/");
        assert!(
            shown.ends_with("run -p demo --bin stories -- --headless --out /games/demo/shots --filter Cube"),
            "{shown}"
        );
    }
}
