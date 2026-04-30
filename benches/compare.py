#!/usr/bin/env python3
"""compare.py — compare two pytest-benchmark JSON outputs.

Usage:
    compare.py BASELINE.json HEAD.json [--output report.md] [--regression-threshold 5]

Exit codes:
    0  no regressions
    1  one or more regression flags set
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Core stats helpers
# ---------------------------------------------------------------------------


def _bootstrap_ci(
    baseline_data: list[float],
    head_data: list[float],
    n_resamples: int = 10_000,
    rng_seed: int = 0,
) -> tuple[float, float]:
    """Return (lower, upper) 95% CI for (median_head - median_baseline)/median_baseline*100."""
    rng = np.random.default_rng(rng_seed)
    b = np.asarray(baseline_data, dtype=np.float64)
    h = np.asarray(head_data, dtype=np.float64)

    b_resamples = rng.choice(b, size=(n_resamples, len(b)), replace=True)
    h_resamples = rng.choice(h, size=(n_resamples, len(h)), replace=True)

    b_medians = np.median(b_resamples, axis=1)
    h_medians = np.median(h_resamples, axis=1)

    # guard against zero-median baselines
    safe_b = np.where(b_medians == 0, np.nan, b_medians)
    deltas = (h_medians - b_medians) / safe_b * 100.0

    lo = float(np.nanpercentile(deltas, 2.5))
    hi = float(np.nanpercentile(deltas, 97.5))
    return lo, hi


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_benchmarks(path: Path) -> dict[str, dict[str, Any]]:
    """Return {name: benchmark_entry} from a pytest-benchmark JSON file."""
    data = json.loads(path.read_text())
    result: dict[str, dict[str, Any]] = {}
    for entry in data.get("benchmarks", []):
        name = entry["name"]
        result[name] = entry
    return result


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------


def compare(
    baseline_path: Path,
    head_path: Path,
    regression_threshold: float = 5.0,
) -> tuple[list[dict[str, Any]], bool]:
    """Run comparison; return (rows, any_regression)."""
    baseline_map = load_benchmarks(baseline_path)
    head_map = load_benchmarks(head_path)

    common = sorted(set(baseline_map) & set(head_map))
    rows: list[dict[str, Any]] = []
    any_regression = False

    for name in common:
        b_entry = baseline_map[name]
        h_entry = head_map[name]

        b_stats = b_entry["stats"]
        h_stats = h_entry["stats"]

        # times are in seconds inside pytest-benchmark; convert to ms for display
        b_median_s: float = b_stats["median"]
        h_median_s: float = h_stats["median"]

        b_median_ms = b_median_s * 1_000
        h_median_ms = h_median_s * 1_000

        b_median_ns = b_median_s * 1_000_000_000
        h_median_ns = h_median_s * 1_000_000_000

        if b_median_ns == 0:
            delta_pct = 0.0
        else:
            delta_pct = (h_median_ns - b_median_ns) / b_median_ns * 100.0

        # Bootstrap CI requires raw data arrays
        b_data = b_stats.get("data", [b_median_s])
        h_data = h_stats.get("data", [h_median_s])

        ci_lo, ci_hi = _bootstrap_ci(b_data, h_data)

        h_stddev: float = h_stats.get("stddev", 0.0)
        cv_head = (h_stddev / h_median_s) if h_median_s != 0 else 0.0

        regression_flag = (delta_pct > regression_threshold) and (ci_lo > 0)
        improvement_flag = (delta_pct < -regression_threshold) and (ci_hi < 0)

        if regression_flag:
            any_regression = True

        flag_str = ""
        if regression_flag:
            flag_str = "REGRESSION"
        elif improvement_flag:
            flag_str = "improved"

        rows.append(
            {
                "name": name,
                "baseline_ms": b_median_ms,
                "head_ms": h_median_ms,
                "delta_pct": delta_pct,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "cv_head": cv_head,
                "flag": flag_str,
            }
        )

    return rows, any_regression


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _fmt_ms(v: float) -> str:
    return f"{v:.3f}"


def _fmt_pct(v: float) -> str:
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.1f}%"


def _fmt_ci(lo: float, hi: float) -> str:
    return f"[{lo:+.1f}%, {hi:+.1f}%]"


def _fmt_cv(v: float) -> str:
    return f"{v:.3f}"


def _flag_cell(flag: str) -> str:
    if flag == "REGRESSION":
        return "REGRESSION"
    elif flag == "improved":
        return "improved"
    return ""


def render_markdown(rows: list[dict[str, Any]]) -> str:
    header = (
        "| Benchmark | Baseline (ms) | Head (ms) | Delta % | 95% CI | CV head | Flag |\n"
        "|-----------|---------------|-----------|---------|--------|---------|------|\n"
    )
    lines = [header]
    for r in rows:
        line = (
            f"| {r['name']} "
            f"| {_fmt_ms(r['baseline_ms'])} "
            f"| {_fmt_ms(r['head_ms'])} "
            f"| {_fmt_pct(r['delta_pct'])} "
            f"| {_fmt_ci(r['ci_lo'], r['ci_hi'])} "
            f"| {_fmt_cv(r['cv_head'])} "
            f"| {_flag_cell(r['flag'])} |\n"
        )
        lines.append(line)
    return "".join(lines)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _self_test() -> None:
    import tempfile

    # Build minimal pytest-benchmark-shaped JSON
    def _make_json(times_a: list[float], times_b: list[float]) -> dict:
        def _entry(name: str, data: list[float]) -> dict:
            arr = np.array(data)
            return {
                "name": name,
                "stats": {
                    "data": data,
                    "median": float(np.median(arr)),
                    "mean": float(np.mean(arr)),
                    "stddev": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                    "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
                },
            }

        return {
            "benchmarks": [
                _entry("bench_fast", times_a),
                _entry("bench_slow", times_b),
            ]
        }

    # baseline: bench_fast ~10 ms, bench_slow ~10 ms
    rng = np.random.default_rng(42)
    baseline_fast = (rng.normal(0.010, 0.0005, 30)).tolist()
    baseline_slow = (rng.normal(0.010, 0.0005, 30)).tolist()

    # head: bench_fast ~7 ms (improved), bench_slow ~14 ms (regression)
    head_fast = (rng.normal(0.007, 0.0005, 30)).tolist()
    head_slow = (rng.normal(0.014, 0.0005, 30)).tolist()

    baseline_data = _make_json(baseline_fast, baseline_slow)
    head_data = _make_json(head_fast, head_slow)

    with tempfile.TemporaryDirectory() as tmpdir:
        bl_path = Path(tmpdir) / "baseline.json"
        hd_path = Path(tmpdir) / "head.json"
        bl_path.write_text(json.dumps(baseline_data))
        hd_path.write_text(json.dumps(head_data))

        rows, any_regression = compare(bl_path, hd_path, regression_threshold=5.0)

    assert len(rows) == 2, f"Expected 2 rows, got {len(rows)}"

    fast_row = next(r for r in rows if r["name"] == "bench_fast")
    slow_row = next(r for r in rows if r["name"] == "bench_slow")

    assert fast_row["delta_pct"] < -5.0, (
        f"bench_fast should show improvement: {fast_row['delta_pct']:.1f}%"
    )
    assert fast_row["flag"] == "improved", f"Expected 'improved', got {fast_row['flag']!r}"

    assert slow_row["delta_pct"] > 5.0, (
        f"bench_slow should show regression: {slow_row['delta_pct']:.1f}%"
    )
    assert slow_row["flag"] == "REGRESSION", f"Expected 'REGRESSION', got {slow_row['flag']!r}"

    assert any_regression is True, "any_regression should be True"

    md = render_markdown(rows)
    assert "| Benchmark |" in md
    assert "bench_fast" in md
    assert "bench_slow" in md
    assert "REGRESSION" in md
    assert "improved" in md

    print("Self-test PASSED")
    print()
    print(md)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare two pytest-benchmark JSON outputs and produce a Markdown report."
    )
    parser.add_argument("baseline", type=Path, help="Baseline JSON file")
    parser.add_argument("head", type=Path, help="HEAD JSON file")
    parser.add_argument(
        "--output", type=Path, default=None, help="Write Markdown to this file (default: stdout)"
    )
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=5.0,
        metavar="PCT",
        help="Delta %% threshold for regression flag (default: 5)",
    )

    args = parser.parse_args(argv)

    rows, any_regression = compare(args.baseline, args.head, args.regression_threshold)

    md = render_markdown(rows)

    if args.output:
        args.output.write_text(md)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(md, end="")

    return 1 if any_regression else 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--self-test":
        _self_test()
        sys.exit(0)
    sys.exit(main())
