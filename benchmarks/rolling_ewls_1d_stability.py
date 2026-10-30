"""Compare main, PR #481 and scalar 1d EWLS wheels on Linux in fresh processes.

Both environments must use the same Python/Polars/NumPy versions. Timing includes
both output fields and excludes input generation, warm-up and snapshot writing.
"""

import argparse
import itertools
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess


def cases(rows, groups, panel_rows):
    result = []

    def add(
        name,
        precision="f64",
        window=504,
        pattern="complete",
        p=1,
        bias=True,
        group_count=1,
        group_rows=rows,
    ):
        result.append(
            dict(
                name=name,
                precision=precision,
                window=window,
                pattern=pattern,
                p=p,
                bias=bias,
                groups=group_count,
                rows=group_rows,
            )
        )

    for precision in ["f64", "f32"]:
        for window in [504, 1040]:
            for pattern in ["complete", "random", "block"]:
                add(f"long-{precision}-w{window}-{pattern}", precision, window, pattern)
                add(
                    f"panel-{precision}-w{window}-{pattern}",
                    precision,
                    window,
                    pattern,
                    group_count=groups,
                    group_rows=panel_rows,
                )
    add("control-two-no-bias", p=2, bias=False)
    add("control-three-bias", p=3)
    add("expired-outliers", pattern="outliers")
    add("overflowed-outliers", pattern="overflow")
    add("all-missing", pattern="all-missing")
    add("short-groups", group_count=groups, group_rows=200)
    return result


def arrays(case):
    import numpy as np

    rng = np.random.default_rng(42)
    n = case["groups"] * case["rows"]
    x = rng.normal(size=(n, case["p"]))
    y = x @ np.linspace(0.2, 0.8, case["p"]) + 0.5 + rng.normal(scale=0.2, size=n)
    pattern = case["pattern"]
    if pattern == "random":
        y[rng.random(n) < 0.1] = np.nan
    elif pattern == "block":
        y[np.arange(n) % case["rows"] % 450 < 80] = np.nan
    elif pattern == "all-missing":
        y[:] = np.nan
    elif pattern in {"outliers", "overflow"}:
        x[25::50_000, 0] = 1e12 if pattern == "outliers" else 1e200
    return x, y


def worker(case, snapshot):
    import time

    import numpy as np
    import polars as pl
    import polars_ds as pds
    import polars_ds.config as cfg

    cfg.LIN_REG_EXPR_F64 = case["precision"] == "f64"
    x, y = arrays(case)
    columns = [f"x{i}" for i in range(case["p"])]
    frame = pl.DataFrame(
        {
            **{c: x[:, i] for i, c in enumerate(columns)},
            "y": y,
            "asset": np.repeat(np.arange(case["groups"]), case["rows"]),
        }
    )
    del x, y
    expr = pds.rolling_lin_reg(
        *columns,
        target="y",
        window_size=case["window"],
        half_life=126,
        min_valid_rows=126,
        add_bias=case["bias"],
        null_policy="skip",
    )
    if case["groups"] > 1:
        expr = expr.over("asset")
    query = frame.lazy().select(expr.alias("fit")).unnest("fit")
    frame.lazy().head(min(case["rows"], 2 * case["window"])).select(expr).collect()
    start = time.perf_counter()
    result = query.collect()
    elapsed = time.perf_counter() - start
    # Unlike ru_maxrss, this is the peak of this executable's address space and
    # cannot inherit memory peaks from the parent's snapshot comparisons.
    peak = next(
        float(s.split()[1]) / 1024
        for s in Path("/proc/self/status").read_text().splitlines()
        if s.startswith("VmHWM:")
    )
    record = dict(
        seconds=elapsed,
        peak_rss_mib=peak,
        valid_fits=len(result) - result["coeffs"].null_count(),
        schema=str(result.schema),
        python=platform.python_version(),
        polars=pl.__version__,
        polars_ds=pds.__version__,
        threads=pl.thread_pool_size(),
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        rss_source="/proc/self/status:VmHWM",
    )
    if snapshot:
        np.savez(
            snapshot,
            coeffs=result["coeffs"].list.to_array(case["p"] + int(case["bias"])).to_numpy(),
            pred=result["pred"].to_numpy(),
            coeff_null=result["coeffs"].is_null().to_numpy(),
            pred_null=result["pred"].is_null().to_numpy(),
        )
    print(json.dumps(record))


def reference_check(case, snapshot):
    """Check sampled whole windows with independent weighted NumPy SVD solves."""
    import numpy as np

    x, y = arrays(case)
    if case["precision"] == "f32":
        x, y = x.astype(np.float32).astype(float), y.astype(np.float32).astype(float)
    if case["bias"]:
        x = np.column_stack([x, np.ones(len(y))])
    n, w = len(y), case["window"]
    positions = set(np.linspace(0, n - 1, min(n, 64), dtype=int))
    positions.update(i for i in [w - 1, w, 2 * w - 1, 25 + w, 25 + w + 40, n - 1] if i < n)
    errors, checked = 0, 0
    rtol, atol = (1e-8, 1e-10) if case["precision"] == "f64" else (1e-4, 1e-5)
    with np.load(snapshot) as out:
        coeffs, pred = out["coeffs"], out["pred"]
        coeff_null, pred_null = out["coeff_null"], out["pred_null"]
    for t in sorted(positions):
        # Stress checks deliberately exclude windows still containing an extreme
        # observation: recovery is judged on ordinary, well-conditioned windows.
        if case["pattern"] in {"outliers", "overflow"} and t >= w - 1:
            if np.any(np.abs(x[t - w + 1 : t + 1, 0]) > 1e10):
                continue
        checked += 1
        expected = None
        if t % case["rows"] >= w - 1:
            xx, yy = x[t - w + 1 : t + 1], y[t - w + 1 : t + 1]
            valid = np.isfinite(xx).all(axis=1) & np.isfinite(yy)
            if valid.sum() >= max(126, x.shape[1]):
                ages = np.arange(w - 1, -1, -1, dtype=float)[valid]
                weights = np.exp2(-(ages - ages.min()) / 126)
                beta, _, rank, _ = np.linalg.lstsq(
                    xx[valid] * np.sqrt(weights)[:, None], yy[valid] * np.sqrt(weights), rcond=None
                )
                if rank == x.shape[1]:
                    expected = beta
        if expected is None:
            errors += int(not coeff_null[t] or not pred_null[t])
        elif coeff_null[t] or not np.allclose(coeffs[t], expected, rtol=rtol, atol=atol):
            errors += 1
        else:
            value = x[t] @ expected
            if np.isfinite(value):
                errors += int(pred_null[t] or not np.isclose(pred[t], value, rtol=rtol, atol=atol))
            else:
                errors += int(not pred_null[t])
    return dict(checked=checked, failures=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-python")
    parser.add_argument("--pr481-python")
    parser.add_argument("--scalar-python")
    parser.add_argument("--output", type=Path, default=Path("/tmp/polars-ds-ewls-comparison"))
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--rows", type=int, default=1_048_576)
    parser.add_argument("--groups", type=int, default=1000)
    parser.add_argument("--panel-rows", type=int, default=1500)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--snapshot", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(json.loads(args.worker), args.snapshot)
        return
    if not all([args.main_python, args.pr481_python, args.scalar_python]):
        parser.error("--main-python, --pr481-python and --scalar-python are required")
    if args.repeats < 1 or args.threads < 1:
        parser.error("repeats and threads must be positive")
    import numpy as np

    selected = cases(args.rows, args.groups, args.panel_rows)
    if args.only:
        selected = [case for case in selected if case["name"] in args.only]
    if not selected:
        parser.error("No benchmark cases selected")
    variants = ["main", "pr481", "scalar"]
    orders = list(itertools.permutations(variants))
    env = dict(
        os.environ,
        POLARS_MAX_THREADS=str(args.threads),
        RAYON_NUM_THREADS=str(args.threads),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    with (args.output / "runs.jsonl").open("w") as log:
        for case in selected:
            records = {v: [] for v in variants}
            for repeat in range(args.repeats):
                for variant in orders[repeat % len(orders)]:
                    snapshot = (
                        str(args.output / f"{case['name']}-{variant}.npz") if repeat == 0 else ""
                    )
                    completed = subprocess.run(
                        [
                            getattr(args, variant + "_python"),
                            str(Path(__file__).resolve()),
                            "--worker",
                            json.dumps(case),
                            "--snapshot",
                            snapshot,
                        ],
                        env=env,
                        text=True,
                        capture_output=True,
                    )
                    if completed.returncode:
                        raise RuntimeError((variant, case, completed.stderr))
                    record = json.loads(completed.stdout)
                    records[variant].append(record)
                    log.write(
                        json.dumps(dict(case=case, variant=variant, repeat=repeat, **record)) + "\n"
                    )
                    log.flush()
            for key in [
                "schema",
                "python",
                "polars",
                "polars_ds",
                "threads",
                "cpu_affinity",
                "rss_source",
            ]:
                assert all(
                    r[key] == records["main"][0][key] for rr in records.values() for r in rr
                ), key
            for rr in records.values():
                assert all(r["valid_fits"] == rr[0]["valid_fits"] for r in rr)
            paths = {v: args.output / f"{case['name']}-{v}.npz" for v in variants}
            stress = case["pattern"] in {"outliers", "overflow"}
            pairs = [("pr481", "scalar")]
            if not stress:
                pairs += [("main", "pr481"), ("main", "scalar")]
            differences = {}
            for left, right in pairs:
                errors = {}
                with np.load(paths[left]) as a, np.load(paths[right]) as b:
                    for key in ["coeff_null", "pred_null"]:
                        np.testing.assert_array_equal(a[key], b[key])
                    for key in ["coeffs", "pred"]:
                        rtol, atol = (1e-8, 1e-10) if case["precision"] == "f64" else (1e-4, 1e-5)
                        np.testing.assert_allclose(
                            a[key], b[key], rtol=rtol, atol=atol, equal_nan=True
                        )
                        finite = np.isfinite(a[key]) & np.isfinite(b[key])
                        errors[key] = (
                            float(np.max(np.abs(a[key][finite] - b[key][finite])))
                            if finite.any()
                            else 0.0
                        )
                        if case["name"].startswith("control-"):
                            np.testing.assert_array_equal(a[key], b[key])
                differences[left + "_vs_" + right] = errors
            checks = {v: reference_check(case, paths[v]) for v in variants}
            assert checks["pr481"]["failures"] == 0, (case, checks)
            assert checks["scalar"]["failures"] == 0, (case, checks)
            medians = {v: statistics.median(r["seconds"] for r in rr) for v, rr in records.items()}
            row = dict(
                case=case,
                median_seconds=medians,
                scalar_speedup_vs_main=medians["main"] / medians["scalar"],
                scalar_speedup_vs_pr481=medians["pr481"] / medians["scalar"],
                peak_rss_mib={v: max(r["peak_rss_mib"] for r in rr) for v, rr in records.items()},
                valid_fits={v: rr[0]["valid_fits"] for v, rr in records.items()},
                max_abs_difference=differences,
                numpy_checks=checks,
            )
            summary.append(row)
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
