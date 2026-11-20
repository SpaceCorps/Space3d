//! The deployment files' documented commands must produce a config the production image
//! accepts (regression: compose.yaml told users to run with `SPACE3D_FLEET_TOKEN=dev-fleet`,
//! which the image's `SPACE3D__ENV=production` validation refuses, so every container exited).

use space3d_server::ServerConfig;
use space3d_server::security::Secrets;
use std::collections::BTreeMap;

/// `KEY=value` assignments of the documented command in `compose.yaml`'s header, with
/// `$(openssl rand -hex N)` / `$(openssl rand -base64 N)` expanded.
fn documented_env() -> BTreeMap<String, String> {
    let compose = include_str!("../deploy/compose.yaml");
    let line = compose
        .lines()
        .find(|l| l.starts_with('#') && l.contains("SPACE3D_FLEET_TOKEN="))
        .expect("compose.yaml documents the command");
    let mut env = BTreeMap::new();
    let mut rest = line.trim_start_matches('#').trim();
    while let Some(eq) = rest.find('=') {
        let key = rest[..eq].split_whitespace().last().unwrap_or_default().to_owned();
        let after = &rest[eq + 1..];
        let (prefix, after) = after.split_at(
            after
                .find("$(")
                .unwrap_or(0)
                .min(after.find(' ').unwrap_or(after.len())),
        );
        let (value, tail) = if let Some(cmd) = after.strip_prefix("$(") {
            let end = cmd.find(')').expect("closing paren");
            let args: Vec<&str> = cmd[..end].split_whitespace().collect();
            let n: usize = args.last().unwrap().parse().unwrap();
            let bytes: Vec<u8> = (0..n).map(|i| (i * 37 + 11) as u8).collect();
            let generated = match args[2] {
                "-hex" => bytes.iter().map(|b| format!("{b:02x}")).collect::<String>(),
                "-base64" => {
                    use base64::Engine;
                    base64::engine::general_purpose::STANDARD.encode(&bytes)
                }
                other => panic!("unexpected openssl flag {other}"),
            };
            (format!("{prefix}{generated}"), &cmd[end + 1..])
        } else {
            let end = after.find(' ').unwrap_or(after.len());
            (format!("{prefix}{}", &after[..end]), &after[end..])
        };
        if key.starts_with("SPACE3D") {
            env.insert(key, value);
        }
        rest = tail;
    }
    env
}

#[test]
fn the_compose_command_passes_production_validation() {
    let env = documented_env();
    for key in ["SPACE3D_KEYS", "SPACE3D_FLEET_TOKEN", "SPACE3D_ADMIN_TOKENS"] {
        assert!(env.contains_key(key), "{key} documented: {env:?}");
    }
    // What the image and compose set for the player service.
    let overlay = vec![
        ("SPACE3D__ENV".to_owned(), "production".to_owned()),
        ("SPACE3D__ROLE".to_owned(), "player-service".to_owned()),
    ];
    let config = ServerConfig::load_layered(Some(include_str!("../deploy/server.example.toml")), None, overlay)
        .expect("config loads");
    config.validate().expect("valid");
    let lookup = |k: &str| env.get(k).cloned();
    Secrets::load(&config, &lookup).expect("the documented secrets are accepted in production");
}

// ---- Dockerfile, compose and Kubernetes manifests stay consistent -----------------------------

const DOCKERFILE: &str = include_str!("../deploy/Dockerfile");
const COMPOSE: &str = include_str!("../deploy/compose.yaml");
const K8S_CONFIGMAP: &str = include_str!("../deploy/k8s/configmap.yaml");
const K8S_GAME_SERVER: &str = include_str!("../deploy/k8s/game-server.yaml");
const K8S_PLAYER_SERVICE: &str = include_str!("../deploy/k8s/player-service.yaml");
const K8S_INGRESS: &str = include_str!("../deploy/k8s/ingress.example.yaml");
const K8S_SECRET: &str = include_str!("../deploy/k8s/secret.example.yaml");
const TOOLCHAIN: &str = include_str!("../../../rust-toolchain.toml");

/// The text of a block scalar (`key: |`) in a manifest, un-indented.
fn block_scalar(yaml: &str, key: &str) -> String {
    let mut lines = yaml.lines();
    let header = lines
        .by_ref()
        .find(|l| l.trim_end() == format!("{}{key}: |", " ".repeat(l.len() - l.trim_start().len())))
        .unwrap_or_else(|| panic!("`{key}: |` in the manifest"));
    let key_indent = header.len() - header.trim_start().len();
    let mut out = String::new();
    let mut body_indent = None;
    for l in lines {
        if l.trim().is_empty() {
            out.push('\n');
            continue;
        }
        let indent = l.len() - l.trim_start().len();
        if indent <= key_indent {
            break;
        }
        let strip = *body_indent.get_or_insert(indent);
        out.push_str(&l[strip.min(indent)..]);
        out.push('\n');
    }
    out
}

/// The integer of the first `key: N` line (`  terminationGracePeriodSeconds: 960`).
fn yaml_int(yaml: &str, key: &str) -> u64 {
    yaml.lines()
        .find_map(|l| l.trim().strip_prefix(&format!("{key}: "))?.trim().parse().ok())
        .unwrap_or_else(|| panic!("`{key}: <integer>` in the manifest"))
}

fn channel() -> String {
    TOOLCHAIN
        .lines()
        .find_map(|l| l.strip_prefix("channel = "))
        .expect("rust-toolchain.toml channel")
        .trim_matches('"')
        .to_owned()
}

/// The image is built with the toolchain the repository pins (the Dockerfile also refuses a
/// mismatch at build time; this catches it without a Docker daemon).
#[test]
fn the_dockerfile_builds_with_the_pinned_toolchain() {
    let arg = DOCKERFILE
        .lines()
        .find_map(|l| l.strip_prefix("ARG RUST_VERSION="))
        .expect("ARG RUST_VERSION");
    assert_eq!(arg.trim(), channel(), "rust-toolchain.toml and the Dockerfile agree");
    assert!(
        DOCKERFILE.contains(" rust:${RUST_VERSION}-slim-bookworm AS build"),
        "the build stage uses that version"
    );
    assert!(
        DOCKERFILE.contains("ENV RUSTUP_TOOLCHAIN=${RUST_VERSION}"),
        "rustup does not follow rust-toolchain.toml's components / targets (they would be downloaded)"
    );
}

/// The instruction of a Dockerfile that contains `needle`, continuation lines joined.
fn dockerfile_instruction(needle: &str) -> String {
    let mut lines = DOCKERFILE.lines();
    while let Some(line) = lines.next() {
        let mut text = line.trim_end().to_owned();
        while text.ends_with('\\') {
            text.pop();
            match lines.next() {
                Some(next) => {
                    text.push(' ');
                    text.push_str(next.trim());
                }
                None => break,
            }
        }
        if !text.trim_start().starts_with('#') && text.contains(needle) {
            return text;
        }
    }
    panic!("no Dockerfile instruction with `{needle}`");
}

/// Regression: `docker compose build` builds the player service and the game server at the same
/// time, both running `cargo fetch` against the same cache mounts. Cargo's package-cache lock
/// files live in /usr/local/cargo, outside the mounted `registry` and `git` directories, so each
/// container has its own lock and two cold fetches unpacked the same crate at once (`failed to
/// unpack package ... File exists`): the documented `up --build` failed on a cold builder. The
/// fetch step must hold every cache mount it uses with `sharing=locked`.
#[test]
fn the_dockerfile_fetch_step_locks_its_cache_mounts() {
    let fetch = dockerfile_instruction("cargo fetch");
    let mounts: Vec<&str> = fetch
        .split_whitespace()
        .filter(|w| w.starts_with("--mount=type=cache"))
        .collect();
    assert!(
        mounts.iter().any(|m| m.contains("cargo-registry")) && mounts.iter().any(|m| m.contains("cargo-git")),
        "the fetch step mounts the registry and git caches: {mounts:?}"
    );
    for m in &mounts {
        assert!(
            m.contains("sharing=locked"),
            "{m}: concurrent cold fetches race without it"
        );
    }
    // The compile step reads the registry offline and cargo locks `target/` itself: it must not
    // be serialised across builds (a lock held for the whole compile).
    let compile = dockerfile_instruction("cargo build");
    assert!(
        !compile.contains("sharing=locked"),
        "the compile step stays concurrent: {compile}"
    );
}

/// The site config in the Kubernetes ConfigMap is accepted by the production image for both
/// roles, given the env the manifests add (role, server id, public URL, secrets).
#[test]
fn the_kubernetes_config_passes_production_validation() {
    let toml = block_scalar(K8S_CONFIGMAP, "server.toml");
    let secrets: BTreeMap<String, String> = [
        ("SPACE3D_KEYS", "k1:AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8="),
        ("SPACE3D_FLEET_TOKEN", "0123456789abcdef0123456789abcdef"),
        ("SPACE3D_ADMIN_TOKENS", "fedcba9876543210fedcba9876543210"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v.to_owned()))
    .collect();
    for (role, id) in [("game-server", "space3d-gs-0"), ("player-service", "players")] {
        let overlay = vec![
            ("SPACE3D__ENV".to_owned(), "production".to_owned()),
            ("SPACE3D__ROLE".to_owned(), role.to_owned()),
            ("SPACE3D__SERVER_ID".to_owned(), id.to_owned()),
            (
                "SPACE3D__GAME__PUBLIC_URL".to_owned(),
                "wss://{server_id}.play.example.com/play".to_owned(),
            ),
        ];
        let config = ServerConfig::load_layered(Some(&toml), None, overlay)
            .unwrap_or_else(|e| panic!("{role}: the ConfigMap's server.toml loads: {e}"));
        config.validate().unwrap_or_else(|e| panic!("{role}: valid: {e}"));
        Secrets::load(&config, &|k| secrets.get(k).cloned())
            .unwrap_or_else(|e| panic!("{role}: secrets accepted in production: {e}"));
    }
}

/// A pod that is deleted gets its preStop sleep, then SIGTERM and the whole drain, then SIGKILL:
/// the grace period has to cover all three (docs/server.md §6: drain_timeout + 30 s).
#[test]
fn the_kubernetes_grace_period_covers_the_drain() {
    let toml = block_scalar(K8S_CONFIGMAP, "server.toml");
    let config = ServerConfig::load_layered(Some(&toml), None, Vec::<(String, String)>::new()).expect("loads");
    let drain = config.game.drain_timeout.as_secs();
    let grace = yaml_int(K8S_GAME_SERVER, "terminationGracePeriodSeconds");
    let sleep: u64 = K8S_GAME_SERVER
        .split("sleep: { seconds: ")
        .nth(1)
        .and_then(|s| s.split(|c: char| !c.is_ascii_digit()).next()?.parse().ok())
        .expect("preStop `sleep: { seconds: N }`");
    assert!(sleep > 0, "the game server has a preStop sleep");
    assert!(
        grace >= drain + sleep + 30,
        "terminationGracePeriodSeconds {grace} < drain_timeout {drain} + preStop {sleep} + 30"
    );
    // The player service does not wait for matches.
    assert!(yaml_int(K8S_PLAYER_SERVICE, "terminationGracePeriodSeconds") >= 30);
}

/// The compose file's stop grace period covers its (shortened) drain timeout the same way.
#[test]
fn the_compose_grace_period_covers_the_drain() {
    let value = |prefix: &str| -> u64 {
        COMPOSE
            .lines()
            .find_map(|l| l.trim().strip_prefix(prefix))
            .and_then(|v| v.trim().trim_end_matches('s').parse().ok())
            .unwrap_or_else(|| panic!("`{prefix}` in compose.yaml"))
    };
    let drain = value("SPACE3D__GAME__DRAIN_TIMEOUT:");
    let grace = value("stop_grace_period:");
    assert!(
        grace >= drain + 30,
        "stop_grace_period {grace} s < drain {drain} s + 30 s"
    );
}

/// Every dotted key path a TOML document sets (tables are walked; arrays are leaves).
fn key_paths(value: &toml::Value, prefix: &str, out: &mut Vec<String>) {
    if let toml::Value::Table(t) = value {
        for (k, v) in t {
            let path = if prefix.is_empty() {
                k.clone()
            } else {
                format!("{prefix}.{k}")
            };
            if matches!(v, toml::Value::Table(_)) {
                key_paths(v, &path, out);
            } else {
                out.push(path);
            }
        }
    }
}

/// The ConfigMap sits over the game binary's built-in defaults, which are dev-oriented (the
/// arena example turns dev auth on and relaxes the per-IP limits): a key it leaves out leaks
/// through. So it must set every key the shipped example does, except what the pod env adds.
#[test]
fn the_kubernetes_config_restates_every_key_of_the_example() {
    let parse = |text: &str| -> Vec<String> {
        let mut keys = Vec::new();
        key_paths(
            &toml::Value::Table(text.parse::<toml::Table>().expect("valid TOML")),
            "",
            &mut keys,
        );
        keys
    };
    let ours = parse(&block_scalar(K8S_CONFIGMAP, "server.toml"));
    let per_pod = ["role", "server_id", "game.public_url"];
    let missing: Vec<String> = parse(include_str!("../deploy/server.example.toml"))
        .into_iter()
        .filter(|k| !per_pod.contains(&k.as_str()) && !ours.contains(k))
        .collect();
    assert!(missing.is_empty(), "k8s/configmap.yaml does not set {missing:?}");
    // The shipped defaults for production.
    assert!(!ours.is_empty());
    let value = block_scalar(K8S_CONFIGMAP, "server.toml")
        .parse::<toml::Table>()
        .unwrap();
    assert_eq!(value["env"].as_str(), Some("production"));
    assert_eq!(value["game"]["allow_dev_auth"].as_bool(), Some(false));
}

/// Documents of a multi-document manifest.
fn documents(yaml: &str) -> Vec<&str> {
    // `\n---` without the trailing newline: Windows checkouts have CRLF line endings.
    yaml.split("\n---").collect()
}

/// Regression: a draining game server is NotReady (`/readyz` 503) and Terminating, and Kubernetes
/// drops such a pod from a Service's endpoints unless the Service sets
/// `publishNotReadyAddresses: true`. The per-pod Services behind the ingress lacked it, so for the
/// whole drain (up to `drain_timeout`) the ingress answered 503 to every WebSocket, including a
/// resume after a link blip and a rejoin into a match still running on the draining server, both
/// of which the server accepts (docs/server.md §6).
#[test]
fn the_per_pod_services_stay_routable_while_the_pod_drains() {
    let per_pod: Vec<&str> = documents(K8S_INGRESS)
        .into_iter()
        .filter(|d| d.contains("kind: Service") && d.contains("statefulset.kubernetes.io/pod-name"))
        .collect();
    assert!(per_pod.len() >= 2, "the example has a Service per game-server pod");
    for doc in per_pod {
        assert!(
            doc.lines().any(|l| l.trim() == "publishNotReadyAddresses: true"),
            "a per-pod Service without publishNotReadyAddresses: true drops the draining pod:\n{doc}"
        );
    }
    // The headless Service is not used by players, and its DNS entries follow readiness.
    assert!(
        K8S_GAME_SERVER
            .lines()
            .any(|l| l.trim() == "publishNotReadyAddresses: false"),
        "the headless Service keeps readiness"
    );
}

/// Regression: the production image refused only the template's placeholder key (not valid
/// base64); with a real key and the template's 35-character fleet and admin tokens (above the
/// 16-character minimum) `check-config` passed, so an operator who replaced only the key ran with
/// publicly known tokens. Each placeholder of `secret.example.yaml` has to be refused on its own.
#[test]
fn every_placeholder_of_the_secret_template_is_refused_in_production() {
    let placeholders: BTreeMap<String, String> = K8S_SECRET
        .lines()
        .skip_while(|l| l.trim() != "stringData:")
        .skip(1)
        .filter_map(|l| {
            let (k, v) = l.trim().split_once(": ")?;
            Some((k.to_owned(), v.trim_matches('"').to_owned()))
        })
        .collect();
    let keys = ["SPACE3D_KEYS", "SPACE3D_FLEET_TOKEN", "SPACE3D_ADMIN_TOKENS"];
    for key in keys {
        assert!(
            placeholders.contains_key(key),
            "{key} in the template: {placeholders:?}"
        );
    }
    let real: BTreeMap<String, String> = [
        ("SPACE3D_KEYS", "k1:AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8="),
        ("SPACE3D_FLEET_TOKEN", "0123456789abcdef0123456789abcdef"),
        ("SPACE3D_ADMIN_TOKENS", "fedcba9876543210fedcba9876543210"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v.to_owned()))
    .collect();
    let toml = block_scalar(K8S_CONFIGMAP, "server.toml");
    let config_for = |role: &str| {
        let overlay = vec![
            ("SPACE3D__ENV".to_owned(), "production".to_owned()),
            ("SPACE3D__ROLE".to_owned(), role.to_owned()),
            ("SPACE3D__SERVER_ID".to_owned(), "space3d-gs-0".to_owned()),
            (
                "SPACE3D__GAME__PUBLIC_URL".to_owned(),
                "wss://{server_id}.play.example.com/play".to_owned(),
            ),
        ];
        ServerConfig::load_layered(Some(&toml), None, overlay).expect("loads")
    };
    for role in ["game-server", "player-service"] {
        let config = config_for(role);
        Secrets::load(&config, &|k| real.get(k).cloned()).expect("real secrets are accepted");
        for replaced in keys {
            let mut env = real.clone();
            env.insert(replaced.to_owned(), placeholders[replaced].clone());
            assert!(
                Secrets::load(&config, &|k| env.get(k).cloned()).is_err(),
                "{role}: the template's placeholder {replaced} is accepted in production"
            );
        }
    }
}
