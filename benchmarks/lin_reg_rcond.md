# rcond: normal-equations and stable QR-SVD paths

## Scope

The default `lin_reg_w_rcond(..., stable=False)` keeps upstream main's
normal-equations solver. Its only numerical change is comparing
`sqrt(gram_singular_value)` with `effective_rcond * sigma_max`. Values at or
below the threshold are discarded, including zero values. This fixes the
unit mismatch without replacing the existing factorization or matrix products.

`stable=True` selects the bounded Householder QR followed by SVD implementation
copied from the previous PR revision, with its public Rust entry point renamed
`faer_solve_lr_rcond_stable`. The arithmetic and tuning are unchanged.
Both paths share Python input preparation, null handling, cutoff floors,
Float32/Float64 dispatch and output construction. The selection occurs once
per fit, before the numerical solve; it does not run both solvers or switch
based on an implicit condition estimate.

Both paths support Ridge without penalizing the intercept. The stable path
uses the augmented design `[X; sqrt(lambda) D]`. Returned singular values
correspond to this augmented design when regularization is enabled. The
normal-equations path estimates them from the regularized Gram matrix.

Forming `X.T @ X` still squares the condition number and may lose weak
directions or overflow at extreme scales. Correcting its cutoff does not
recover information already lost in the Gram matrix. The stable path avoids
this loss, uses bounded Float64 workspaces for both input precisions, and
rejects nonfinite inputs after null handling. It retains the configured output
precision and cutoff floor `eps * max(M, N)`, using the original post-null
handling dimensions including the intercept.

## Analytic regression

For `z = [-2, -1, 0, 1, 2]`, `x = 1e-7*z`, `y = 0.01+0.001*z`, an intercept
and `rcond=1e-12`, upstream main returns slope zero instead of 10000. The
Float32 version uses `x=1e-4*z` and expects slope 10. The tests first reproduce
both failures against the unchanged upstream wheel, then exercise the default,
explicit normal and explicit stable paths.

Other tests cover rescaling, exact cutoff boundaries, zero designs, both
output precisions, Ridge/intercept behavior, grouped null handling and first
order optimality. The stable-specific suite additionally checks ill-conditioned
coefficients, predictions and objectives, extreme magnitudes, block boundaries
and late-arriving rank. Six native checks exercise block-size invariance,
cutoff boundaries, wide designs and scaling against independent direct SVD.

## Validation of the dual-path revision

The release wheel passes 549 tests on each of Polars 1.44.2 and 2.0.0,
using Python 3.10.12 and NumPy 2.2.6. This includes the two dedicated rcond
modules and existing expression, model and rolling/EWLS tests. All six native
checks pass; the manual kernel benchmark remains ignored in ordinary runs.

The retained Float32 normal-equations path has a larger roundoff error in the
scaled, null-skipping group case: its slope differs from the independent
reference by about 2.3e-4 relatively. That test uses a 1e-3 coefficient bound
for this path and also checks predictions, spectrum and rank; the stable path
retains its original 2e-4 coefficient bound. This is a measured numerical
trade-off, not a change to the default solver's arithmetic.

Maximum dimensionless first-order residuals across the untruncated OLS/Ridge
cases are identical on both Polars versions:

| Precision | Normal equations | Stable QR-SVD |
| --- | ---: | ---: |
| Float64 | 4.281e-16 | 4.872e-16 |
| Float32 | 2.187e-7 | 1.561e-8 |

For Float32 the normal path allows eight machine epsilons for the Gram product
and factorization; the stable path retains its 2e-7 bound. The dedicated
ill-conditioned coefficient/prediction/objective tests retain their original
strict stable-path bounds. These diagnostics describe the tested cases and
are not universal error guarantees.

## Performance evidence and reproduction

The previous matched release-wheel measurements found the stable implementation
took 1.5-2.4 times as long as upstream main for long fits. Its optimized input
copy/scaling reduced runtime by 10-20% compared with the earlier stable
implementation. Full measurements and numerical limitations are retained in the
[previous report](https://github.com/huangbogeng/polars_ds_extension/blob/23737a34c4b7e576feca943d80b882c58222f2c9/benchmarks/lin_reg_rcond.md#direct-comparison-with-upstream-main-and-the-final-implementation).
These historical ratios describe those workloads, not universal bounds.

The benchmark runner now accepts `--baseline-solver` and `--fixed-solver`, each
one of `default`, `normal`, or `stable`. `default` omits the new keyword, which
also permits comparison against older wheels. Use two release-wheel environments
with the same dependencies, four Polars/Rayon threads and one BLAS thread.
For example, compare both paths in one installed release wheel:

```sh
python benchmarks/lin_reg_rcond.py \
  --baseline-python .venv/bin/python --baseline-solver normal \
  --fixed-python .venv/bin/python --fixed-solver stable \
  --output .cache/rcond-dual-path-benchmark --repeats 5
```

Workers alternate execution order, warm up before timing, compare complete
outputs and independently check sampled groups against NumPy/LAPACK. Raw
results stay in the ignored `.cache` directory. All benchmark cases are
well-conditioned and full-rank so a broken cutoff cannot win by discarding work.

## Measurements of this revision (2026-11-06)

Both paths in the same release wheel were compared across all 16 cases with
five alternating fresh-process repeats per path (160 runs). The host is an AMD
EPYC 7H12; affinity is CPUs 0-3, Polars/Rayon use four threads, and BLAS uses
one. No builds or tests overlap timing. All complete-output comparisons and
independent NumPy reference checks pass.

| Long-fit workload | Float64 stable/normal | Float32 stable/normal |
| --- | ---: | ---: |
| 100,000 rows, 3 predictors | 1.548 | 1.685 |
| 100,000 rows, 20 predictors | 1.988 | 2.165 |
| 1,000,000 rows, 3 predictors | 2.169 | 2.027 |

An additional 60 fresh-process runs compare the corrected normal path against
an unchanged upstream-main wheel on those six long-fit cases, with the same
settings. All output and reference checks pass. Corrected/main median-time
ratios range from 0.773 to 1.106; there is no consistent slowdown across the
cases. The numerical change is only the cutoff comparison, so these timings
support retaining the default performance characteristics rather than a new
algorithmic speedup.

## References

- [LAPACK DGELSD](https://netlib.org/lapack/explore-html/d9/d67/group__gelsd_ga0bee7e1b9e7e43f59ecf2419b2759c42.html): minimum-norm least squares and the singular-value cutoff.
- [Demmel et al., Communication-avoiding parallel and sequential QR factorizations](https://arxiv.org/abs/0806.2159): bounded QR reduction and numerical stability.
