#!/usr/bin/env bash
# polars-ds dev environment — source this file, do not execute it.
# Usage: source scripts/dev-env.sh

export RUSTC_WRAPPER=/opt/homebrew/bin/sccache
# sccache is incompatible with incremental compilation; set 0 to satisfy sccache.
export CARGO_INCREMENTAL=0
export CARGO_TARGET_DIR="$HOME/.cargo-target-shared"

echo "polars-ds dev env loaded — sccache enabled"

if command -v sccache >/dev/null 2>&1; then
    sccache --show-stats 2>/dev/null | head -3
fi
