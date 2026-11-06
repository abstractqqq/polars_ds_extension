# rcond: correctness and performance

`lin_reg_w_rcond` retains the normal-equations solver by default and fixes its
cutoff to compare design singular values rather than squared values.
`stable=True` uses bounded Householder QR followed by SVD. Both paths preserve
output precision, the cutoff floor and the unpenalized Ridge intercept.
The stable path uses Float64 workspaces, avoiding precision loss from forming
`X.T @ X`, at a higher runtime cost.

For `z=[-2,-1,0,1,2]`, `x=1e-7*z`, `y=0.01+0.001*z`, an intercept and
`rcond=1e-12`, upstream main returns slope zero instead of 10000. The tests
cover this regression in both paths, cutoff boundaries, rescaling, null
handling, Ridge and first-order optimality. Stable-specific tests also check
ill-conditioned coefficients, predictions, objectives and extreme scales.
Six native checks verify block-size invariance and scaling against direct SVD.

## Measurements

Measured on 2026-11-06: AMD EPYC 7H12, CPU affinity 0-3, four Polars/Rayon
threads and one BLAS thread; Python 3.10.12, Polars 2.0.0 and NumPy 2.2.6.
The same release wheel was tested with both solvers across 16 workloads,
five alternating fresh-process repeats per solver (160 runs), without concurrent
builds or tests. Complete-output comparisons and NumPy reference checks pass.

| Workload | Float64 stable/normal time | Float32 stable/normal time |
| --- | ---: | ---: |
| 100,000 rows, 3 predictors | 1.548 | 1.685 |
| 100,000 rows, 20 predictors | 1.988 | 2.165 |
| 1,000,000 rows, 3 predictors | 2.169 | 2.027 |

Another 60 runs compared the corrected normal path with unchanged upstream
main on these six workloads. All reference checks pass; median time ratios
range from 0.773 to 1.106, with no consistent slowdown across cases.
These measurements are workload-specific, not universal performance bounds.

The retained Float32 normal path has about 2.3e-4 relative slope error in the
scaled null-skipping group test. That test checks coefficients, predictions,
spectrum and rank with a separate normal-path tolerance; stable-path bounds
remain unchanged. Correcting the cutoff does not recover directions already
lost when forming the Gram matrix.

## Reproduction

```sh
python benchmarks/lin_reg_rcond.py \
  --baseline-python .venv/bin/python --baseline-solver normal \
  --fixed-python .venv/bin/python --fixed-solver stable \
  --output .cache/rcond-benchmark --repeats 5
```

Use `--baseline-solver default` to compare against an older wheel without the
`stable` keyword. Raw results remain in the ignored output directory.

References: [LAPACK DGELSD](https://netlib.org/lapack/explore-html/d9/d67/group__gelsd_ga0bee7e1b9e7e43f59ecf2419b2759c42.html);
[Demmel et al., TSQR](https://arxiv.org/abs/0806.2159).
