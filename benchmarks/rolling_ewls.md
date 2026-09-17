# Rolling EWLS benchmark

Measured 2026-09-17 on Intel(R) Xeon(R) Platinum 8352V CPU @ 2.10GHz (144 visible logical CPUs, approximately 3.94 TiB RAM).
Environment: Linux-5.15.0-181-generic-x86_64-with-glibc2.35, Python 3.12.12, Polars 1.44.2, polars-ds 0.12.1 plus this change.
Built with `maturin develop --release --locked`, the pinned `nightly-2026-04-01` toolchain and `RUSTFLAGS="-C debuginfo=0"`; the repository release profile uses fat LTO and one codegen unit.

Regression inputs and outputs are Float64, with an intercept, `half_life=126`, `min_valid_rows=126` and `null_policy="skip"`.
Random missingness masks each target with probability 0.1. Block missingness removes 80 consecutive targets every 450 rows within each group. Seed: 42.
Times are medians of three fresh processes per case; peak RSS is the maximum of those runs. Timing excludes imports, input generation and a small warm-up, and includes materializing both `coeffs` and `pred`.
RSS includes imports, input generation, inputs and outputs. The host was shared and CPU affinity was not pinned.

## 5,000 groups x 1,500 rows

7.5 million input rows. Polars and Rayon thread pools are each limited to 16; BLAS/OpenMP/MKL are limited to one thread.
Each cell shows median seconds / peak MiB.

| Window | Predictors | Complete | Random missing | Block missing |
| ---: | ---: | ---: | ---: | ---: |
| 252 | 1 | 5.982 / 1548 | 5.792 / 1566 | 5.750 / 1569 |
| 252 | 3 | 6.989 / 1995 | 9.261 / 1994 | 8.202 / 1994 |
| 504 | 1 | 4.988 / 1351 | 4.965 / 1332 | 4.844 / 1361 |
| 504 | 3 | 5.806 / 1754 | 8.030 / 1765 | 6.382 / 1766 |
| 1040 | 1 | 2.937 / 1038 | 2.824 / 1002 | 2.379 / 1012 |
| 1040 | 3 | 2.929 / 1319 | 3.713 / 1351 | 4.992 / 1361 |

There are 6,245,000, 4,985,000 and 2,305,000 full-window positions for windows 252, 504 and 1040 respectively; all produce valid fits in these datasets. Larger windows have more warm-up rows, so the long-group comparison below is needed to assess steady-state window scaling.

## Matched per-window baseline

8 groups x 1,500 rows, with the same data generation, output fields and thread limits for both methods. The baseline assigns weights by original position, filters aligned rows, and calls `lin_reg(weights=...)` once per complete window in a Python loop.
The baseline timing includes repeated Python/Polars query setup and input handling as well as regression. These ratios apply to this 12,000-row comparison; they are not extrapolated to the large panel.

| Window | Predictors | Missingness | EWLS (ms) | Per-window WLS (ms) | WLS / EWLS | Peak MiB (EWLS / WLS) |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 252 | 1 | complete | 16.94 | 15703.54 | 926.9 | 82 / 95 |
| 252 | 1 | random | 16.29 | 15427.63 | 946.9 | 84 / 97 |
| 252 | 1 | block | 17.19 | 14604.16 | 849.5 | 85 / 97 |
| 252 | 3 | complete | 19.10 | 22773.13 | 1192.4 | 84 / 97 |
| 252 | 3 | random | 25.45 | 23175.23 | 910.5 | 86 / 98 |
| 252 | 3 | block | 26.24 | 21823.99 | 831.8 | 86 / 99 |
| 504 | 1 | complete | 13.57 | 13621.50 | 1004.1 | 84 / 94 |
| 504 | 1 | random | 14.40 | 13071.85 | 907.8 | 84 / 96 |
| 504 | 1 | block | 14.41 | 12890.31 | 894.7 | 83 / 96 |
| 504 | 3 | complete | 16.81 | 17562.26 | 1044.8 | 84 / 97 |
| 504 | 3 | random | 21.95 | 18882.15 | 860.1 | 85 / 98 |
| 504 | 3 | block | 23.69 | 18976.21 | 800.9 | 86 / 98 |
| 1040 | 1 | complete | 8.81 | 5710.74 | 648.0 | 83 / 91 |
| 1040 | 1 | random | 8.38 | 6551.20 | 782.2 | 83 / 93 |
| 1040 | 1 | block | 7.73 | 6042.61 | 781.6 | 84 / 94 |
| 1040 | 3 | complete | 10.43 | 8680.42 | 832.3 | 83 / 93 |
| 1040 | 3 | random | 12.95 | 9188.71 | 709.6 | 85 / 97 |
| 1040 | 3 | block | 12.85 | 9441.03 | 734.7 | 86 / 96 |

## Long-group scaling

One complete group, with Polars and Rayon each limited to one thread. Each cell shows median milliseconds / peak MiB. This reduces the difference in warm-up fractions between window sizes.

| Rows | 1 predictor, W=252 | 1 predictor, W=1040 | 3 predictors, W=252 | 3 predictors, W=1040 |
| ---: | ---: | ---: | ---: | ---: |
| 50,000 | 64.35 / 93 | 62.95 / 93 | 80.94 / 96 | 80.35 / 97 |
| 100,000 | 129.85 / 104 | 127.92 / 104 | 160.93 / 112 | 158.81 / 111 |
| 200,000 | 259.42 / 130 | 236.38 / 129 | 324.75 / 142 | 324.57 / 142 |

Increasing rows from 50,000 to 200,000 (4x) changes time by 3.76–4.04x across these cases. At 200,000 rows, increasing the window from 252 to 1040 changes time by 0.91x with one predictor and 1.00x with three predictors.
The recurrence and amortized rebuilding cost are derived in [the EWLS notes](../maths/rolling_ewls.md). Numerically unstable inputs may trigger additional rebuilds; these timings use well-conditioned synthetic data.

## Reproduce

```bash
python benchmarks/rolling_ewls.py --groups 5000 --rows 1500 --repeats 3 > large.jsonl
python benchmarks/rolling_ewls.py --groups 8 --rows 1500 --methods ewls reference --repeats 3 > matched.jsonl
for n in 50000 100000 200000; do
  python benchmarks/rolling_ewls.py --groups 1 --rows "$n" --windows 252 1040 --patterns complete --threads 1 --repeats 3
done > scaling.jsonl
```

The script emits raw JSON records with timings, peak RSS, valid-fit counts, hardware, software versions, dtype and thread settings.
