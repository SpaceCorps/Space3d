#!/usr/bin/env bash
# Builds the engine examples for the web into site/public/play/<id>/ (wasm + wasm-bindgen glue).
# Used by site/Dockerfile and locally:  bash site/scripts/build-play.sh   (needs the
# wasm32-unknown-unknown target and a wasm-bindgen CLI matching Cargo.lock; wasm-opt optional).
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
ids=$(node -e 'console.log(require("'"$root"'/site/examples.json").map(e=>e.id).join(" "))')
pkgs=""; for id in $ids; do pkgs="$pkgs -p $id"; done
cd "$root"
cargo build --release --locked --target wasm32-unknown-unknown $pkgs
tdir="$(cargo metadata --format-version 1 --no-deps | node -e 'let s="";process.stdin.on("data",d=>s+=d).on("end",()=>console.log(JSON.parse(s).target_directory))')"
for id in $ids; do
  dir=$(node -e 'const e=require("'"$root"'/site/examples.json").find(e=>e.id==="'"$id"'");console.log(e.path||e.id)')
  out="$root/site/public/play/$dir"; mkdir -p "$out"
  wasm-bindgen --target web --no-typescript --out-name "$id" --out-dir "$out" "$tdir/wasm32-unknown-unknown/release/$id.wasm"
  # Opt-in: wasm-opt -Oz currently breaks startup (externref table cannot grow); gzip covers size.
  if [ -n "${WASM_OPT:-}" ] && command -v wasm-opt >/dev/null; then wasm-opt -Oz --enable-bulk-memory --enable-nontrapping-float-to-int --enable-sign-ext --enable-mutable-globals --enable-simd --enable-reference-types -o "$out/${id}_bg.wasm" "$out/${id}_bg.wasm"; fi
done
