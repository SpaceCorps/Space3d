#!/usr/bin/env python3
"""Maintain Space3d's patched copy of epaint (egui's text layout) in third_party/epaint.

third_party/epaint is the crates.io epaint crate plus the kinsoku patch in
third_party/epaint.kinsoku.patch (Chinese and Japanese line breaking and three rules of Unicode
line breaking for every script, see third_party/README.md). The root Cargo.toml applies it with
`[patch.crates-io]`, which only takes effect while the copy has the version egui asks for: after
an egui bump, cargo warns "patch `epaint` was not used" and silently builds crates.io's epaint,
without kinsoku.

    scripts/update-epaint.py              # re-vendor epaint at the egui version of Cargo.toml
    scripts/update-epaint.py 0.37.0       # ... at this version
    scripts/update-epaint.py --check      # CI: version, patch and resolution are consistent
    scripts/update-epaint.py --refresh    # rewrite the .patch after editing third_party/epaint

Updating fetches the pristine crate (from the local cargo registry, else crates.io with its
checksum verified against the index), applies the patch, replaces third_party/epaint, rewrites
the .patch for the new version, and runs the kinsoku tests. When the patch doesn't apply, the
new crate is left in a temporary directory with the rejected hunks (*.rej): port them by hand
into third_party/epaint (after copying the new crate there), then run --refresh and the tests.

Needs python3 (3.9+), git and cargo.
"""

import argparse
import filecmp
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDORED = ROOT / "third_party" / "epaint"
PATCH = ROOT / "third_party" / "epaint.kinsoku.patch"
README = ROOT / "third_party" / "README.md"
# Files Space3d adds to the vendored crate (the published crate lacks them).
OWN_FILES = {"LICENSE-MIT", "LICENSE-APACHE"}
IGNORED = {"target", ".cargo-ok"}
PATCH_HEADER = """\
Space3d patch to epaint {version}: kinsoku (禁則処理) line breaking for Chinese and Japanese,
and three rules of Unicode line breaking (UAX #14) for every script: no break before a no-break
space, none after an ASCII opening bracket, none next to a word joiner. See third_party/README.md. Applied to the published
crate with `git apply -p1` from its root (scripts/update-epaint.py does it); regenerate with
`scripts/update-epaint.py --refresh`.

"""


def fail(message):
    print(f"error: {message}", file=sys.stderr)
    sys.exit(1)


def run(cmd, cwd=ROOT, check=True, capture=False):
    print("+ " + " ".join(str(c) for c in cmd), file=sys.stderr)
    result = subprocess.run(
        [str(c) for c in cmd],
        cwd=cwd,
        text=True,
        encoding="utf-8",  # git's output (the patch has Chinese and Japanese text) on Windows too
        capture_output=capture,
        check=False,
    )
    if check and result.returncode != 0:
        if capture:
            sys.stderr.write(result.stdout + result.stderr)
        fail(f"`{' '.join(str(c) for c in cmd)}` failed ({result.returncode})")
    return result


def egui_version():
    text = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    m = re.search(r'^egui\s*=\s*"=?([^"]+)"', text, re.M)
    if not m:
        fail("no `egui = \"...\"` in the root Cargo.toml [workspace.dependencies]")
    return m.group(1)


def crate_version(crate_dir):
    text = (Path(crate_dir) / "Cargo.toml").read_text(encoding="utf-8")
    package = text.split("[package]", 1)[1]
    m = re.search(r'^version\s*=\s*"([^"]+)"', package, re.M)
    if not m:
        fail(f"no version in {crate_dir}/Cargo.toml")
    return m.group(1)


def copy_tree(src, dest, skip=()):
    """Copy src into the (new) directory dest, without IGNORED and `skip` entries."""
    def ignore(_dir, names):
        return [n for n in names if n in IGNORED or n in skip]

    shutil.copytree(src, dest, ignore=ignore)


def fetch_pristine(version, dest):
    """Unpacked epaint `version` as published on crates.io, copied to `dest`."""
    cargo_home = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo"))
    for candidate in sorted((cargo_home / "registry" / "src").glob(f"*/epaint-{version}")):
        if (candidate / ".cargo-ok").exists() and (candidate / "Cargo.toml").exists():
            print(f"using {candidate}", file=sys.stderr)
            copy_tree(candidate, dest)
            return
    index_url = "https://index.crates.io/ep/ai/epaint"
    print(f"downloading epaint {version} from crates.io", file=sys.stderr)
    with urllib.request.urlopen(index_url, timeout=60) as r:
        entries = [json.loads(line) for line in r.read().decode().splitlines() if line.strip()]
    entry = next((e for e in entries if e["vers"] == version), None)
    if entry is None:
        fail(f"epaint {version} is not on crates.io")
    url = f"https://static.crates.io/crates/epaint/epaint-{version}.crate"
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    if hashlib.sha256(data).hexdigest() != entry["cksum"]:
        fail(f"checksum mismatch for {url}")
    prefix = f"epaint-{version}/"
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            for member in tar.getmembers():
                name = member.name
                if not name.startswith(prefix) or ".." in Path(name).parts or not (
                    member.isfile() or member.isdir()
                ):
                    fail(f"unexpected entry {name!r} in {url}")
            if hasattr(tarfile, "data_filter"):
                tar.extractall(tmp, filter="data")
            else:  # Python < 3.12 without the backport; names were checked above
                tar.extractall(tmp)
        copy_tree(Path(tmp) / prefix.rstrip("/"), dest)


def tree_diff(pristine, patched):
    """git-style diff of two crate trees (paths a/... b/...), without Space3d's own files."""
    with tempfile.TemporaryDirectory() as tmp:
        copy_tree(pristine, Path(tmp) / "a")
        copy_tree(patched, Path(tmp) / "b", skip=OWN_FILES)
        result = run(
            [
                "git", "-c", "core.quotepath=false", "diff", "--no-index", "--no-prefix",
                "--no-color", "--no-ext-diff", "--diff-algorithm=myers", "--indent-heuristic",
                "-U3", "a", "b",
            ],
            cwd=tmp, check=False, capture=True,
        )
        if result.returncode not in (0, 1):
            sys.stderr.write(result.stderr)
            fail("git diff failed")
        return result.stdout


def apply_patch(tree, reject=False):
    cmd = ["git", "apply", "-p1", "--whitespace=nowarn"]
    if reject:
        cmd.append("--reject")
    return run(cmd + [PATCH], cwd=tree, check=False).returncode == 0


def differing_files(a, b):
    """Files that differ between two trees (Space3d's own files and target/ ignored)."""
    out = []

    def walk(cmp, rel):
        out.extend(str(rel / n) for n in cmp.left_only + cmp.right_only + cmp.diff_files
                   if n not in OWN_FILES and n not in IGNORED)
        out.extend(str(rel / n) for n in cmp.funny_files)
        for name, sub in cmp.subdirs.items():
            if name not in IGNORED:
                walk(sub, rel / name)

    walk(filecmp.dircmp(a, b, ignore=[]), Path("."))
    # dircmp compares by stat signature first; compare contents of every file to be sure.
    for path in sorted(Path(a).rglob("*")):
        rel = path.relative_to(a)
        other = Path(b) / rel
        if path.is_file() and other.is_file() and not (set(rel.parts) & IGNORED):
            if path.read_bytes() != other.read_bytes() and str(Path(".") / rel) not in out:
                out.append(str(Path(".") / rel))
    return sorted(set(out))


def write_patch(version, diff):
    # LF on Windows too: a patch with CRLF line ends does not apply to the published crate.
    PATCH.write_text(PATCH_HEADER.format(version=version) + diff, encoding="utf-8", newline="\n")
    print(f"wrote {PATCH.relative_to(ROOT)}", file=sys.stderr)


def check_resolution():
    """`cargo metadata` resolves epaint to third_party/epaint (the [patch] is used)."""
    result = run(["cargo", "metadata", "--format-version", "1", "--locked"], capture=True)
    meta = json.loads(result.stdout)
    used = {p["id"] for p in meta["packages"]} if meta.get("resolve") is None else {
        n["id"] for n in meta["resolve"]["nodes"]
    }
    epaints = [p for p in meta["packages"] if p["name"] == "epaint" and p["id"] in used]
    expected = (VENDORED / "Cargo.toml").resolve()
    if len(epaints) != 1 or Path(epaints[0]["manifest_path"]).resolve() != expected:
        found = ", ".join(f'{p["version"]} ({p["manifest_path"]})' for p in epaints) or "none"
        fail(
            f"epaint resolves to {found}, not {expected}: the [patch.crates-io] entry is "
            "missing or unused (egui and third_party/epaint versions differ?). "
            "Run scripts/update-epaint.py."
        )
    print(f"epaint {epaints[0]['version']} resolves to third_party/epaint", file=sys.stderr)


def check():
    egui, vendored = egui_version(), crate_version(VENDORED)
    if egui != vendored:
        fail(f"egui is {egui} but third_party/epaint is {vendored}: run scripts/update-epaint.py")
    with tempfile.TemporaryDirectory() as tmp:
        tree = Path(tmp) / "epaint"
        fetch_pristine(vendored, tree)
        if not apply_patch(tree):
            fail(f"{PATCH.name} does not apply to epaint {vendored}")
        diff = differing_files(tree, VENDORED)
        if diff:
            fail(
                f"third_party/epaint differs from epaint {vendored} + {PATCH.name} in: "
                + ", ".join(diff)
                + " (after editing third_party/epaint, run scripts/update-epaint.py --refresh)"
            )
    for own in OWN_FILES:
        if not (VENDORED / own).is_file():
            fail(f"third_party/epaint/{own} is missing")
    check_resolution()
    print(f"ok: third_party/epaint = epaint {vendored} + {PATCH.name}, used by the workspace")


def refresh():
    version = crate_version(VENDORED)
    with tempfile.TemporaryDirectory() as tmp:
        pristine = Path(tmp) / "epaint"
        fetch_pristine(version, pristine)
        write_patch(version, tree_diff(pristine, VENDORED))


def run_tests():
    # A path dependency's warnings are not capped as a crates.io crate's are: CI's
    # `-D warnings` applies to the vendored code, so check it here too (third_party/README.md).
    rustflags = (os.environ.get("RUSTFLAGS", "") + " -D warnings").strip()
    print(f"+ RUSTFLAGS={rustflags!r}", file=sys.stderr)
    result = subprocess.run(
        ["cargo", "test", "--manifest-path", str(VENDORED / "Cargo.toml"),
         "--target-dir", str(ROOT / "target" / "third_party"), "--lib", "text::"],
        cwd=ROOT, env={**os.environ, "RUSTFLAGS": rustflags}, check=False,
    )
    if result.returncode != 0:
        fail("the vendored epaint's tests failed or it has warnings (third_party/README.md)")
    run(["cargo", "test", "-p", "space3d-ui", "--test", "kinsoku", "--test", "line_breaks"])


def update(version, test):
    old = crate_version(VENDORED)
    work = Path(tempfile.mkdtemp(prefix="epaint-update-"))
    tree = work / f"epaint-{version}"
    fetch_pristine(version, tree)
    if not apply_patch(tree):
        apply_patch(tree, reject=True)
        fail(
            f"{PATCH.name} does not apply cleanly to epaint {version}. The new crate with the "
            f"hunks that applied and the rejected ones (*.rej) is in {tree}. Port the patch by "
            "hand: copy that crate over third_party/epaint (keep LICENSE-*; delete the .rej and "
            ".orig files), finish the kinsoku hook in src/text/text_layout.rs "
            "(RowBreakCandidates) and src/text/kinsoku.rs, then run "
            "scripts/update-epaint.py --refresh and the tests (third_party/README.md)."
        )
    own = {name: (VENDORED / name).read_bytes() for name in OWN_FILES if (VENDORED / name).exists()}
    shutil.rmtree(VENDORED)
    copy_tree(tree, VENDORED)
    for name, data in own.items():
        (VENDORED / name).write_bytes(data)
    shutil.rmtree(work)
    with tempfile.TemporaryDirectory() as tmp:
        pristine = Path(tmp) / "epaint"
        fetch_pristine(version, pristine)
        write_patch(version, tree_diff(pristine, VENDORED))
    if README.exists():
        text = README.read_text(encoding="utf-8")
        vcs = VENDORED / ".cargo_vcs_info.json"
        sha = json.loads(vcs.read_text())["git"]["sha1"] if vcs.exists() else "unknown"
        text = re.sub(
            r"^Vendored version: .*$",
            f"Vendored version: **epaint {version}** (egui commit `{sha}`, `crates/epaint`).",
            text, count=1, flags=re.M,
        )
        README.write_text(text, encoding="utf-8", newline="\n")
    print(f"third_party/epaint: epaint {old} -> {version}", file=sys.stderr)
    if egui_version() != version:
        print(
            f"note: the root Cargo.toml asks for egui {egui_version()}; bump egui (and egui-wgpu, "
            f"egui-winit, egui_extras) to {version} too, or the patch stays unused",
            file=sys.stderr,
        )
        return
    run(["cargo", "metadata", "--format-version", "1"], capture=True)  # refreshes Cargo.lock
    check_resolution()
    if test:
        run_tests()
    print("done: review `git diff`, then commit third_party/ and Cargo.lock together")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("version", nargs="?", help="epaint version (default: egui's in Cargo.toml)")
    parser.add_argument("--check", action="store_true", help="verify, change nothing (CI)")
    parser.add_argument("--refresh", action="store_true", help="rewrite the .patch from third_party/epaint")
    parser.add_argument("--no-test", action="store_true", help="skip the kinsoku tests after updating")
    args = parser.parse_args()
    if args.check:
        check()
    elif args.refresh:
        refresh()
    else:
        update(args.version or egui_version(), test=not args.no_test)


if __name__ == "__main__":
    main()
