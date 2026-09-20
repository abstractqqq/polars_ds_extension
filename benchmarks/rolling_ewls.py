"""Run EWLS and per-window WLS in fresh processes; print one JSON record per run.

Examples (build with maturin develop --release first):
    python benchmarks/rolling_ewls.py --groups 256 --rows 4096 --windows 504 1040 \
        --predictors 1 --patterns random --methods ewls closed_form_1d --repeats 5 --threads 4
    python benchmarks/rolling_ewls.py --groups 5000 --rows 1500 --repeats 3
    python benchmarks/rolling_ewls.py --groups 8 --rows 1500 --methods ewls reference
    python benchmarks/rolling_ewls.py --groups 1 --rows 200000 --windows 252 1040

Timing excludes input generation and includes materializing coeffs and pred.
Peak RSS includes imports, inputs and outputs. Each case runs in a fresh process
so an earlier case cannot contaminate its peak. Compare speed ratios only on
matching cases; the large run and the smaller reference run are separate results.
"""

import argparse
import itertools
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def run_case(args):
    import resource
    import time

    import numpy as np
    import polars as pl

    import polars_ds as pds

    group_count, rows, window, p, pattern, method = args.case
    group_count, rows, window, p = map(int, [group_count, rows, window, p])
    rng = np.random.default_rng(42)
    n = group_count * rows
    columns = [f"x{i}" for i in range(p)]
    x = rng.normal(size=(n, p))
    y = x @ np.linspace(0.2, 0.8, p) + 0.5 + rng.normal(scale=0.2, size=n)
    frame = pl.DataFrame(
        {
            **{c: x[:, i] for i, c in enumerate(columns)},
            "y": y,
            "asset": np.repeat(np.arange(group_count, dtype=np.int32), rows),
        }
    )
    if pattern != "complete":
        missing = rng.random(n) < 0.1 if pattern == "random" else np.arange(n) % rows % 450 < 80
        frame = frame.with_columns(
            pl.when(pl.Series(missing)).then(None).otherwise(pl.col("y")).alias("y")
        )
    del x, y
    min_rows = max(p + 1, min(126, window))
    plugin_expr = (
        pds.rolling_lin_reg(
            *columns,
            target="y",
            window_size=window,
            half_life=args.half_life,
            min_valid_rows=min_rows,
            add_bias=True,
            null_policy="skip",
        )
        .over("asset")
        .alias("fit")
    )
    closed_form_expr = (
        pds.rolling_lin_reg_1d(
            columns[0],
            target="y",
            window_size=window,
            half_life=args.half_life,
            min_valid_rows=min_rows,
        )
        .over("asset")
        .alias("fit")
        if p == 1
        else None
    )
    expr = closed_form_expr if method == "closed_form_1d" else plugin_expr
    # Initialize expression dispatch outside the measurement on a small independent fit.
    frame.head(min(rows, 2 * window)).select(expr)
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    if method in ("ewls", "closed_form_1d"):
        result = frame.select(expr).unnest("fit")
    else:
        weights = pl.Series("w", np.exp2(-np.arange(window - 1, -1, -1) / args.half_life))
        coeffs, predictions = [], []
        for group in frame.partition_by("asset", maintain_order=True):
            for t in range(len(group)):
                if t < window - 1:
                    coeffs.append(None)
                    predictions.append(None)
                    continue
                subset = group.slice(t - window + 1, window).with_columns(weights).drop_nulls()
                if len(subset) < min_rows:
                    coeffs.append(None)
                    predictions.append(None)
                    continue
                beta = (
                    subset.select(pds.lin_reg(*columns, target="y", weights="w", add_bias=True))
                    .item()
                    .to_list()
                )
                coeffs.append(beta)
                row = group.select(columns).row(t)
                predictions.append(sum(a * b for a, b in zip(row, beta)) + beta[-1])
        result = pl.DataFrame(
            {
                "coeffs": pl.Series(coeffs, dtype=pl.List(pl.Float64)),
                "pred": pl.Series(predictions, dtype=pl.Float64),
            }
        )
    elapsed = time.perf_counter() - start_wall
    cpu_elapsed = time.process_time() - start_cpu
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mib = peak / (1024**2 if sys.platform == "darwin" else 1024)
    model = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        model = next(
            (
                line.split(":", 1)[1].strip()
                for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")
            ),
            model,
        )
    print(
        json.dumps(
            {
                "method": method,
                "groups": group_count,
                "rows_per_group": rows,
                "rows": n,
                "window": window,
                "half_life": args.half_life,
                "predictors": p,
                "dtype": str(result.schema["pred"]),
                "pattern": pattern,
                "seconds": elapsed,
                "cpu_seconds": cpu_elapsed,
                "peak_rss_mib": peak_mib,
                "valid_fits": len(result) - result["coeffs"].null_count(),
                "full_window_positions": group_count * max(0, rows - window + 1),
                "threads": pl.thread_pool_size(),
                "rayon_thread_limit": os.environ.get("RAYON_NUM_THREADS"),
                "blas_thread_limit": os.environ.get("OPENBLAS_NUM_THREADS"),
                "logical_cpus": os.cpu_count(),
                "cpu": model,
                "platform": platform.platform(),
                "python": platform.python_version(),
                "polars": pl.__version__,
                "polars_ds": pds.__version__,
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, default=5000)
    parser.add_argument("--rows", type=int, default=1500)
    parser.add_argument("--windows", type=int, nargs="+", default=[252, 504, 1040])
    parser.add_argument("--predictors", type=int, nargs="+", default=[1, 3])
    parser.add_argument(
        "--patterns",
        nargs="+",
        choices=["complete", "random", "block"],
        default=["complete", "random", "block"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["ewls", "closed_form_1d", "reference"],
        default=["ewls", "closed_form_1d"],
    )
    parser.add_argument("--half-life", type=float, default=126.0)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--case", nargs=6, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.case:
        if args.case[5] == "closed_form_1d" and int(args.case[3]) != 1:
            parser.error("closed_form_1d requires exactly one predictor")
        run_case(args)
        return
    env = dict(
        os.environ,
        POLARS_MAX_THREADS=str(args.threads),
        RAYON_NUM_THREADS=str(args.threads),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    for window, p, pattern, method, _ in itertools.product(
        args.windows, args.predictors, args.patterns, args.methods, range(args.repeats)
    ):
        if method == "closed_form_1d" and p != 1:
            continue
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--half-life",
                str(args.half_life),
                "--case",
                str(args.groups),
                str(args.rows),
                str(window),
                str(p),
                pattern,
                method,
            ],
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()
