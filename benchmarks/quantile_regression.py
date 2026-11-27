"""Compare quantile_reg with statsmodels; requires tests/requirements-test.txt.

Run after an optimized build, without other builds/tests running:
OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=1 RAYON_NUM_THREADS=1 \
    python benchmarks/quantile_regression.py

Reports median milliseconds from five alternating repeats after warming both
implementations. Data generation and independent LP validation are not timed.
The solvers have different stopping rules: pds uses a normalized loss gap,
statsmodels uses p_tol (coefficient change). Compare the reported LP errors too.
"""

import json
import os
import platform
import statistics
import time
import warnings

import numpy as np
import polars as pl
import polars_ds as pds
import scipy
import statsmodels
from scipy.optimize import linprog
from scipy.sparse import csc_matrix, eye, hstack
from statsmodels.regression.quantile_regression import QuantReg


def timings(ours, other):
    ours()
    other()
    samples = [[], []]
    for repeat in range(5):
        for index in [0, 1] if repeat % 2 == 0 else [1, 0]:
            start = time.perf_counter()
            (ours, other)[index]()
            samples[index].append(time.perf_counter() - start)
    return [1000 * statistics.median(s) for s in samples]


def objective(x, y, beta, q):
    r = y - x @ beta
    return np.maximum(q * r, (q - 1) * r).mean()


def reference(x, y, q):
    n, p = x.shape
    solution = linprog(
        np.r_[np.zeros(p), np.full(n, q), np.full(n, 1 - q)],
        A_eq=hstack([csc_matrix(x), eye(n), -eye(n)], format="csc"),
        b_eq=y,
        bounds=[(None, None)] * p + [(0, None)] * (2 * n),
        method="highs",
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.fun / n


def dense_case(n, p, q):
    rng = np.random.default_rng(390 + n)
    a = rng.normal(size=(n, p))
    x = np.c_[a, np.ones(n)]
    y = a @ rng.normal(size=p) + 2 + rng.standard_t(4, size=n)
    cols = [f"x{i}" for i in range(p)]
    df = pl.from_numpy(a, schema=cols).with_columns(y=pl.Series(y))
    expr = pds.quantile_reg(*cols, target="y", quantile=q, add_bias=True)

    def ours():
        return df.select(expr).item().to_numpy()

    def other():
        return QuantReg(y, x).fit(q=q, p_tol=1e-7, max_iter=10_000).params

    with warnings.catch_warnings(record=True) as caught:
        beta, expected = ours(), other()
        best = reference(x, y, q)
        ours_ms, other_ms = timings(ours, other)
    return {
        "n": n,
        "predictors": p,
        "q": q,
        "polars_ds_ms": ours_ms,
        "statsmodels_ms": other_ms,
        "polars_ds_lp_gap": objective(x, y, beta, q) - best,
        "statsmodels_lp_gap": objective(x, y, expected, q) - best,
        "statsmodels_warnings": sorted({str(w.message) for w in caught}),
    }


def grouped_case():
    rng = np.random.default_rng(390)
    parts = []
    for group in range(20):
        a = rng.normal(size=(100, 3))
        y = a @ [0.4, -1.1, 2.1] + group + rng.standard_t(4, size=100)
        parts.append(
            pl.from_numpy(a, schema=["x0", "x1", "x2"]).with_columns(
                y=pl.Series(y), group=pl.lit(group)
            )
        )
    df = pl.concat(parts)
    expr = pds.quantile_reg("x0", "x1", "x2", target="y", quantile=0.9, add_bias=True)
    arrays = [
        (np.c_[part.select("x0", "x1", "x2").to_numpy(), np.ones(len(part))], part["y"].to_numpy())
        for part in parts
    ]

    def ours():
        return df.group_by("group").agg(expr)

    def other():
        return [QuantReg(y, x).fit(q=0.9, p_tol=1e-7, max_iter=10_000).params for x, y in arrays]

    maximum_gap = 0.0
    for group, beta in ours().iter_rows():
        x, y = arrays[group]
        maximum_gap = max(
            maximum_gap, objective(x, y, np.asarray(beta), 0.9) - reference(x, y, 0.9)
        )
    with warnings.catch_warnings(record=True) as caught:
        ours_ms, other_ms = timings(ours, other)
    return {
        "groups": 20,
        "n_per_group": 100,
        "predictors": 3,
        "q": 0.9,
        "polars_ds_ms": ours_ms,
        "statsmodels_ms": other_ms,
        "polars_ds_max_lp_gap": maximum_gap,
        "statsmodels_warnings": sorted({str(w.message) for w in caught}),
    }


if __name__ == "__main__":
    print(
        json.dumps(
            {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "polars": pl.__version__,
                "scipy": scipy.__version__,
                "statsmodels": statsmodels.__version__,
                "thread_settings": {
                    name: os.environ.get(name)
                    for name in ["OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"]
                },
            }
        ),
        flush=True,
    )
    for n, p in [(200, 3), (2000, 5), (10000, 5)]:
        for q in [0.5, 0.9]:
            print(json.dumps(dense_case(n, p, q)), flush=True)
    print(json.dumps(grouped_case()), flush=True)
