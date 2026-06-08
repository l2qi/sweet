#!/usr/bin/env bash
set -euo pipefail

cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo build -p sweet-mcp-mock-server
cargo test --workspace
cargo doc --workspace --no-deps --all-features
