"""Matched release-wheel rcond benchmarks; generated artifacts stay outside Git."""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


def cases():
    result = []
    for precision in ["f64", "f32"]:
        for name, groups, rows, features, batch, l2 in [
            ("small-fit", 1, 64, 3, 200, 0.0),
            ("medium-fit", 1, 1000, 20, 10, 0.0),
            ("long-3", 1, 100000, 3, 1, 0.0),
            ("long-20", 1, 100000, 20, 1, 0.0),
            ("million-3", 1, 1000000, 3, 1, 0.0),
            ("small-groups", 2000, 32, 3, 1, 0.0),
            ("daily-panels", 250, 1000, 20, 1, 0.0),
            ("ridge-small-groups", 2000, 32, 3, 1, 0.2),
        ]:
            result.append(
                dict(
                    name=precision + "-" + name,
                    precision=precision,
                    groups=groups,
                    rows=rows,
                    p=features,
                    batch=batch,
                    l2=l2,
                )
            )
    return result


def arrays(case):
    import numpy as np

    rng = np.random.default_rng(20261103)
    n = case["groups"] * case["rows"]
    x = rng.normal(size=(n, case["p"]))
    y = 0.7 + x @ (np.arange(1, case["p"] + 1) / case["p"]) + rng.normal(scale=0.1, size=n)
    return x, y


def worker(case, snapshot, solver="default"):
    import numpy as np
    import polars as pl
    import polars_ds as pds
    import polars_ds.config as cfg

    cfg.LIN_REG_EXPR_F64 = case["precision"] == "f64"
    x, y = arrays(case)
    columns = [f"x{i}" for i in range(case["p"])]
    frame = pl.DataFrame(
        {**dict(zip(columns, x.T)), "y": y, "g": np.repeat(np.arange(case["groups"]), case["rows"])}
    )
    del x, y
    solver_kwargs = {} if solver == "default" else {"stable": solver == "stable"}
    fit = pds.lin_reg_w_rcond(
        *columns, target="y", add_bias=True, rcond=1e-12, l2_reg=case["l2"], **solver_kwargs
    ).alias("fit")
    query = (
        frame.lazy().select(fit)
        if case["groups"] == 1
        else frame.lazy().group_by("g").agg(fit.first()).sort("g")
    )
    query = query.unnest("fit")
    query.collect()
    start = time.perf_counter()
    for _ in range(case["batch"]):
        result = query.collect()
    seconds = (time.perf_counter() - start) / case["batch"]
    rss = next(
        float(line.split()[1]) / 1024
        for line in Path("/proc/self/status").read_text().splitlines()
        if line.startswith("VmHWM:")
    )
    assert result.height == case["groups"]
    assert result["coeffs"].null_count() == result["singular_values"].null_count() == 0
    if snapshot:
        np.savez(
            snapshot,
            coeffs=result["coeffs"].list.to_array(case["p"] + 1).to_numpy(),
            sv=result["singular_values"].list.to_array(case["p"] + 1).to_numpy(),
        )
    print(
        json.dumps(
            dict(
                seconds=seconds,
                peak_rss_mib=rss,
                polars=pl.__version__,
                numpy=np.__version__,
                polars_ds=pds.__version__,
                threads=pl.thread_pool_size(),
                affinity=sorted(os.sched_getaffinity(0)),
            )
        )
    )


def reference(case, snapshot):
    import numpy as np

    x, y = arrays(case)
    if case["precision"] == "f32":
        x, y = x.astype("float32").astype(float), y.astype("float32").astype(float)
    positions = sorted(set([0, case["groups"] // 2, case["groups"] - 1]))
    with np.load(snapshot) as data:
        for g in positions:
            start = g * case["rows"]
            end = start + case["rows"]
            design = np.column_stack([x[start:end], np.ones(case["rows"])])
            target = y[start:end]
            if case["l2"]:
                penalty = np.zeros((case["p"], case["p"] + 1))
                value = (
                    np.sqrt(np.float32(case["l2"]))
                    if case["precision"] == "f32"
                    else np.sqrt(case["l2"])
                )
                penalty[np.arange(case["p"]), np.arange(case["p"])] = value
                design = np.vstack([design, penalty])
                target = np.r_[target, np.zeros(case["p"])]
            eps = np.finfo(np.float64 if case["precision"] == "f64" else np.float32).eps
            effective = max(1e-12, eps * max(case["rows"], case["p"] + 1))
            beta, _, rank, sv = np.linalg.lstsq(design, target, rcond=effective)
            assert rank == case["p"] + 1
            rtol, atol = (1e-8, 1e-10) if case["precision"] == "f64" else (2e-4, 1e-5)
            np.testing.assert_allclose(data["coeffs"][g], beta, rtol=rtol, atol=atol)
            np.testing.assert_allclose(data["sv"][g], sv, rtol=rtol, atol=atol)
    return len(positions)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-python")
    p.add_argument("--fixed-python")
    p.add_argument("--baseline-solver", choices=["default", "normal", "stable"], default="default")
    p.add_argument("--fixed-solver", choices=["default", "normal", "stable"], default="default")
    p.add_argument("--solver", choices=["default", "normal", "stable"], default="default")
    p.add_argument("--output", type=Path)
    p.add_argument("--repeats", type=int, default=6)
    p.add_argument("--only", nargs="+")
    p.add_argument("--worker")
    p.add_argument("--snapshot", default="")
    args = p.parse_args()
    if args.worker:
        worker(json.loads(args.worker), args.snapshot, args.solver)
        return
    if not all([args.baseline_python, args.fixed_python, args.output]):
        p.error("--baseline-python, --fixed-python and --output are required")
    if args.repeats < 1:
        p.error("--repeats must be positive")
    import numpy as np

    selected = [c for c in cases() if not args.only or c["name"] in args.only]
    variants = ["baseline", "fixed"]
    env = dict(
        os.environ,
        POLARS_MAX_THREADS="4",
        RAYON_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    with (args.output / "runs.jsonl").open("w") as raw:
        for case in selected:
            records = {v: [] for v in variants}
            for repeat in range(args.repeats):
                order = variants if repeat % 2 == 0 else list(reversed(variants))
                for variant in order:
                    snapshot = (
                        str(args.output / f"{case['name']}-{variant}.npz") if repeat == 0 else ""
                    )
                    run = subprocess.run(
                        [
                            getattr(args, variant + "_python"),
                            str(Path(__file__).resolve()),
                            "--worker",
                            json.dumps(case),
                            "--solver",
                            getattr(args, variant + "_solver"),
                            "--snapshot",
                            snapshot,
                        ],
                        env=env,
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    record = json.loads(run.stdout)
                    records[variant].append(record)
                    raw.write(
                        json.dumps(dict(case=case, variant=variant, repeat=repeat, **record)) + "\n"
                    )
                    raw.flush()
            for key in ["polars", "numpy", "polars_ds", "threads", "affinity"]:
                assert all(
                    r[key] == records["baseline"][0][key] for rr in records.values() for r in rr
                )
            snapshots = {v: args.output / f"{case['name']}-{v}.npz" for v in records}
            for variant in variants[1:]:
                with np.load(snapshots["baseline"]) as a, np.load(snapshots[variant]) as b:
                    for key in ["coeffs", "sv"]:
                        rtol, atol = (1e-8, 1e-10) if case["precision"] == "f64" else (2e-4, 1e-5)
                        np.testing.assert_allclose(a[key], b[key], rtol=rtol, atol=atol)
            checks = {v: reference(case, path) for v, path in snapshots.items()}
            times = {v: statistics.median(r["seconds"] for r in rr) for v, rr in records.items()}
            row = dict(
                case=case,
                solvers={v: getattr(args, v + "_solver") for v in variants},
                median_seconds=times,
                fixed_time_ratio=times["fixed"] / times["baseline"],
                peak_rss_mib={v: max(r["peak_rss_mib"] for r in rr) for v, rr in records.items()},
                numpy_reference_groups=checks,
            )
            summary.append(row)
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
