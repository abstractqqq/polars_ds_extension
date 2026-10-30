# One-dimensional EWLS: scope, rationale and validation

Candidate: `fix/ewls-1d-stability`, based on upstream main
`b28b46aea5dafc1516063f61190ef9fa848bb48d` (0.14.0).
Locally validated on 2026-10-30.

## Problem and intended change

The one-predictor/intercept/skip EWLS plugin currently subtracts shifted prefix
EWM sums. A large observation can erase small contributions; subtracting that
observation after expiry cannot recover the lost contributions. Overflow can
also poison subsequent prefix sums. In the seed-42, 180-row fixture with
`x[25]=1e12`, W=23 and half-life=6, the last window contains ordinary values:
main returns slope 0.27741610796240085, but independent NumPy returns
0.9980337601821238. The candidate returns 0.9980337601821249.

The numerical approach builds on [PR #481](https://github.com/abstractqqq/polars_ds_extension/pull/481),
contributed by `huangbogeng`, head `27440198ef30b88e7e3eaaad927bb9ee3d712e59`.
That PR was merged into `ewls_1d_path` after #480 had already merged into main;
its fix is absent from this main revision. Transplanting #481 is sufficient for
correctness. The candidate retains a specialized implementation because the
matched measurements below show substantial additional speed and memory gains.

## Files and necessity

| Files | Change | Necessity |
| --- | --- | --- |
| `src/linear/online_lr/ewls_1d.rs` | Five scalar moments, normalized finite-window updates, instability rebuilds and scaled scalar Cholesky; includes 3 Rust tests | Restores the current-window solution after expiry/overflow and preserves identifiable fits through missing tails |
| `src/num_ext/rolling_ewls_1d.rs` | Shared typed adapter and direct Arrow builders | Shares Float32/Float64 validation/output code and avoids a matrix allocation for every fit |
| `src/num_ext/linear_regression.rs`, `linear_regression_f32.rs` | Replace the two duplicated staged-plan bodies with typed adapter calls | Wires both existing plugin entrypoints to the corrected implementation |
| `src/linear/online_lr/mod.rs`, `src/num_ext/mod.rs` | One module declaration each | Registers the two private implementation modules |
| `tests/test_rolling_ewls.py` | 84 added regression parameter combinations | Locks down expiry recovery, overflow, missing tails, rank gates, prediction nulls, both APIs/precisions, groups/chunks/slices and execution engines |
| `maths/rolling_ewls.md`, `python/polars_ds/exprs/expr_linear.py` | Numerical explanation and API docstring | Documents rebuilding, normalization and finite-prediction behavior |
| `benchmarks/rolling_ewls_1d_stability.py`, this report | One three-way benchmark entrypoint and review evidence | Makes the decision to retain specialization reproducible |
| `uv.lock` | Correct only project version 0.13.1 → 0.14.0 | Repairs stale project metadata; does not change dependency resolution |

There are **12 changed/new files**, of which **6 are runtime Rust files**.
The Rust diff is +360/-263 lines including 82 lines of kernel tests:
production code alone is +278/-263, a net increase of 15 lines. Two large
Float32/Float64 bodies are replaced by one kernel and one typed adapter.

## Behavioral boundary and tradeoff

The affected path requires one predictor, an intercept, a positive half-life,
no regularization and `null_policy="skip"`. The explicit `rolling_lin_reg_1d`
API and existing automatic routing both use it. Signatures, plugin names,
coefficient order, schemas, precision configuration, warm-up and row-age/group
semantics stay the same. Nonfinite predictions now become null, and extreme
observations no longer contaminate later ordinary windows.

Generic EWLS (`src/linear/online_lr/ewls.rs` and
`src/num_ext/rolling_ewls.rs`), multiple predictors, no-intercept/other-policy
routes, non-EWLS regression, dependencies, Cargo features, build flags and CI
are outside this change. Older generic-EWLS optimizations and the comparison-only
#481 transplant are not part of the candidate.

The specialized kernel duplicates a small part of the generic window-state
logic. That is the maintenance cost being accepted for the measured benefit.
There is no new solver framework or public abstraction. Scheduled rebuilds cost
O(W) every W rows, giving O(N) usual work for this fixed-dimensional model;
exceptional rebuilds can cost more. Normal equations and the existing rank gate
do not guarantee arbitrary-scale or ill-conditioned accuracy.

## Validation and matched performance

| Python 3.12.12 runtime | Main with added regressions | Original #481 on main | Candidate |
| --- | --- | --- | --- |
| Polars 2.0.0 | 1229 passed / 30 failed | 1259 passed | 1259 passed |
| Polars 1.44.2 | 1229 passed / 30 failed | 1259 passed | 1259 passed |

All 30 main failures are added defect regressions. Both corrected variants pass
2,066 independent NumPy weighted-SVD window checks each. Full benchmark output
null masks match between them, and values agree within f64 rtol=1e-8/atol=1e-10
and f32 rtol=1e-4/atol=1e-5. Generic controls are bitwise equal across variants.
Three focused Rust kernel tests also passed before this organization pass.

Intel Xeon Platinum 8352V/Linux x86_64, Python 3.12.12, Polars 2.0.0, NumPy 2.5.4;
identical dependencies/Cargo manifests/locks and release flags. Thirty cases,
six fresh processes per variant (540 collections), all six run-order
permutations, CPUs 24–27, four Polars/Rayon threads and one BLAS/OpenMP thread.
Times are medians excluding setup/warm-up/snapshots. RSS is the maximum process
`/proc/self/status:VmHWM`, sampled before snapshot conversion. Long inputs have
1,048,576 rows; panels have 1,000 groups × 1,500 rows.

| Case | Main ms | #481 ms | Candidate ms | #481 RSS MiB | Candidate RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Long f64 / W504 / complete | 174.67 | 459.18 | 160.89 | 290.9 | 160.6 |
| Long f32 / W504 / complete | 93.42 | 254.63 | 84.88 | 275.9 | 149.8 |
| Panel f64 / W504 / complete | 400.64 | 185.87 | 143.11 | 242.5 | 241.4 |
| Panel f32 / W1040 / complete | 379.40 | 118.91 | 91.88 | 201.6 | 200.2 |

Across ordinary cases, the candidate is 2.39–3.81x faster than #481 on long
inputs and 1.14–1.42x on panels. Against main, the ranges are 1.03–1.48x and
2.41–4.13x respectively. This is a shared host: generic controls also fluctuate,
so modest gains need caution. Short groups show no established gain over #481.
Large long-input speed/memory differences provide the strongest justification.
On overflow stress main produces zero valid fits, while both fixes produce
1,037,967; incorrect/null output is not a meaningful speed baseline.

## Reproduction and local evidence

Install the three corresponding release wheels in separate environments with
identical Python/dependency versions, then run this single benchmark script:

```bash
taskset -c 24-27 "$MAIN_PYTHON" benchmarks/rolling_ewls_1d_stability.py \
  --main-python "$MAIN_PYTHON" \
  --pr481-python "$PR481_PYTHON" \
  --scalar-python "$SCALAR_PYTHON" \
  --output /tmp/ewls-three-way --repeats 6
```

Use `--rows 4096 --groups 4 --repeats 1 --only long-f64-w504-complete
expired-outliers` for a correctness smoke run. Performance output defaults to
`/tmp/polars-ds-ewls-comparison`; keep generated JSON/NPZ outside the repository.

The consolidated runner was smoke-checked on seven cases (21 fresh processes),
including both outlier-recovery cases and generic controls. Its data generation,
worker measurements and NumPy references are unchanged from the full run.
Generated raw records and snapshots are kept outside the source tree.
