# Build context: repository root (the site bundles docs/, integrations/ and the engine examples).
#   docker build -f site/Dockerfile -t space3d-site .
# Stage 1 builds the examples for the web (wasm), stage 2 the static site, stage 3 serves it.
FROM rust:1-bookworm AS wasm
RUN apt-get update && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /repo
COPY rust-toolchain.toml Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY examples ./examples
COPY third_party ./third_party
COPY site/examples.json site/scripts/build-play.sh ./site/
# `tr -d '\r'`: checkouts made on Windows have CRLF line endings.
RUN tr -d '\r' < site/build-play.sh > /tmp/build-play.sh \
    && rustup target add wasm32-unknown-unknown \
    && v="$(tr -d '\r' < Cargo.lock | grep -A1 '^name = "wasm-bindgen"$' | sed -n 's/^version = "\(.*\)"/\1/p')" \
    && echo "wasm-bindgen $v" && cargo install wasm-bindgen-cli --locked --version "$v" \
    && mkdir -p site/scripts && cp /tmp/build-play.sh site/scripts/build-play.sh \
    && bash site/scripts/build-play.sh

FROM node:24-slim AS build
RUN corepack enable
WORKDIR /repo
COPY README.md ./
COPY docs ./docs
COPY integrations ./integrations
COPY site ./site
COPY --from=wasm /repo/site/public/play ./site/public/play
WORKDIR /repo/site
RUN corepack install && pnpm install --frozen-lockfile && pnpm run build

FROM nginxinc/nginx-unprivileged:stable-alpine
COPY site/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /repo/site/dist /usr/share/nginx/html
EXPOSE 8080
HEALTHCHECK CMD wget -qO- http://127.0.0.1:8080/ >/dev/null || exit 1
