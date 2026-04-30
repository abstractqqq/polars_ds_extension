#!/usr/bin/env bash
set -euo pipefail

# Usage: benches/run_ab.sh [baseline_tag]
# Default baseline_tag: baseline-perf-current
BASELINE_TAG="${1:-baseline-perf-current}"
RESULTS_DIR="benches/results"
mkdir -p "$RESULTS_DIR"

echo "=== Building HEAD (release) ==="
maturin develop --release

HEAD_REV=$(git rev-parse --short HEAD)
HEAD_JSON="$RESULTS_DIR/head-$HEAD_REV.json"

echo "=== Running HEAD benchmarks → $HEAD_JSON ==="
.venv/bin/pytest benches/ \
    --benchmark-json="$HEAD_JSON" \
    --benchmark-min-rounds=20 \
    --benchmark-max-time=60 \
    --benchmark-warmup=on \
    --benchmark-warmup-iterations=3 \
    -p no:xdist \
    -q

BASELINE_JSON="$RESULTS_DIR/${BASELINE_TAG}.json"
if [[ ! -f "$BASELINE_JSON" ]]; then
    echo "WARNING: $BASELINE_JSON missing. Saving HEAD as baseline."
    cp "$HEAD_JSON" "$BASELINE_JSON"
    echo "Baseline saved. Run again after a change to compare."
    exit 0
fi

REPORT="$RESULTS_DIR/comparison-$(date +%Y%m%d-%H%M%S).md"
.venv/bin/python benches/compare.py "$BASELINE_JSON" "$HEAD_JSON" --output "$REPORT"
echo "=== Report: $REPORT ==="
