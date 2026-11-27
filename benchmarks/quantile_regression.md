# Non-rolling quantile regression: correctness and timing

The expression minimizes mean pinball loss with a smoothed MM/IRLS update,
column-pivoted QR solves and a numerical primal-dual gap stopping rule. The
surrogate, majorization inequality, dual and normalization are derived in
[the expression documentation](../docs/expr_linear.md). There are no new runtime
dependencies. SciPy and statsmodels are already test dependencies.

Tests use an independent equality-constrained SciPy HiGHS linear program for
pinball optimality, and statsmodels for well-conditioned identifiable
coefficients. Non-unique fits are checked by objective, rather than demanding
one arbitrary coefficient vector. Other cases cover grouped/lazy predictions,
null alignment and fills, scaling, rank deficiency, invalid/non-finite data,
iteration exhaustion, and Float64 coefficient range errors.

## Local measurements

Measured 2026-11-28 on macOS 26.2 arm64, Python 3.12.13, Polars 2.2.0,
NumPy 2.5.5, SciPy 1.17.0 and statsmodels 0.15.0. BLAS, Polars and Rayon thread
limits were all one. No concurrent build or test was run for these measurements;
the process was not CPU-pinned.

The local optimized build used opt-level 2, 16 codegen units and no LTO, with
the project's pinned nightly-2026-04-01. Cargo.toml is unchanged. This is not the
default wheel's opt-level 3 / fat-LTO build, so these results should not be
extrapolated to all builds or machines.

Times are median milliseconds from five alternating repeats, after warming both
implementations. Data generation and LP checks are outside timing. The PDS
expression uses default `tol=1e-6`, `max_iter=10_000`, Float64 and a bias.
Statsmodels uses `p_tol=1e-7`, `max_iter=10_000`. These stopping rules differ:
PDS checks normalized loss gap; statsmodels checks coefficient change.

| Workload | Quantile | PDS (ms) | statsmodels (ms) |
| --- | ---: | ---: | ---: |
| 200 rows, 3 predictors | 0.5 | 1.93 | 9.34 |
| 200 rows, 3 predictors | 0.9 | 3.30 | 4.67 |
| 2,000 rows, 5 predictors | 0.5 | 21.02 | 10.61 |
| 2,000 rows, 5 predictors | 0.9 | 41.85 | 13.90 |
| 10,000 rows, 5 predictors | 0.5 | 157.04 | 48.85 |
| 10,000 rows, 5 predictors | 0.9 | 182.07 | 56.36 |
| 20 groups of 100 rows, 3 predictors | 0.9 | 113.21 | 984.26 |

PDS includes dispatch and coefficient extraction; grouped PDS includes grouping.
Statsmodels receives pre-partitioned arrays for the grouped workload. Dense
mean-loss errors against LP were below 8.1e-10 for PDS and 2.5e-9 for statsmodels.
The largest PDS group loss error was 7.3e-9. Statsmodels emitted no warnings in
these workloads. These are workload-specific measurements.

Repeated weighted QR factorizations and the dual projection add work on dense
matrices; their individual costs have not been separately profiled. PDS is
about 2–3.3 times slower on the larger dense cases here. Its expression/grouped
integration can still be useful. The small and grouped workloads here favor
PDS; they do not establish a universal speedup.

## Convergence budget

An additional deterministic 50-fit audit (10 designs, quantiles 0.01, 0.1, 0.5,
0.9, 0.99) completed 49 fits within the default budget, all within the declared
LP objective tolerance. A rounded-response 60-row, 4-predictor 1% fit exhausted
10,000 and 20,000 iterations; `max_iter=100_000` solved it with a mean LP loss
gap of 2.08e-6, within the declared scaled tolerance. This case is retained in
[the tests](../tests/test_quantile_reg.py). The solver raises on exhaustion;
it does not silently return the unfinished coefficients. Extreme quantiles
and difficult designs may need a larger budget or a looser tolerance.

## Reproduction

Install the project's test requirements and build locally, for example:

```sh
CARGO_BUILD_JOBS=2 CARGO_PROFILE_RELEASE_LTO=false \
  CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 CARGO_PROFILE_RELEASE_OPT_LEVEL=2 \
  maturin develop --release --locked
OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=1 RAYON_NUM_THREADS=1 \
  python benchmarks/quantile_regression.py
pytest tests/test_quantile_reg.py
```

The script prints JSON with package versions, actual thread environment values,
timings, LP errors and statsmodels warnings. Random seeds and workload generation
are in the script. LP reference construction is independent of the IRLS solver.
