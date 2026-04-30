# polars-ds Tier-1 benchmarks

Performance benchmarks for the four Tier-1 workloads identified in the
overhead-reduction project.

## Requirements

```
pip install pytest-benchmark
```

The benchmark suite lives under `benches/`.  Results are written to `results/`.

## Running

```bash
# Quick run (pytest-benchmark defaults, results JSON saved)
pytest benches/ --benchmark-json=results/baseline.json

# Recommended for stable numbers (20 rounds minimum)
pytest benches/ --benchmark-json=results/baseline.json --benchmark-min-rounds=20

# Compare a branch against saved baseline
pytest benches/ --benchmark-compare=results/baseline.json
```

## Memory budget

| Fixture | Rows | Approx RAM |
|---|---|---|
| `glm_irls_df` (default, scaled-down) | 3 M | ~240 MB |
| `glm_irls_df` (full, `GLM_FULL=1`) | 30 M | ~2 GB |
| `knn_radius_df` | 1 M | ~48 MB |
| `entropy_series` | 1 M | ~8 MB |
| `rolling_lr_df` | 500 k | ~28 MB |

Running all four fixtures together uses **~330 MB** by default.
Set `GLM_FULL=1` to use the full 30 M-row GLM fixture; total RAM then rises
to **~2.1 GB** (roughly 4 GB with pytest-benchmark's internal copies).

```bash
# Full 30 M-row GLM run
GLM_FULL=1 pytest benches/ --benchmark-json=results/full.json --benchmark-min-rounds=5
```

## Fixture details

| Fixture | Workload | Key parameters |
|---|---|---|
| `glm_irls_df` | group_by GLM IRLS (Binomial) | 100 k groups × 30 rows, 5 features, binary target |
| `knn_radius_df` | KNN radius search | 1 M pts, 5 dims, uniform [0,1), r=0.3 SQL2 |
| `entropy_series` | Approximate/sample entropy, rolling ts features | 1 M standard-normal f64 |
| `rolling_lr_df` | Rolling LR with null prefix | 500 k rows, first 100 k have null x1..x3 |

All fixtures are seeded (`SEED=42`) via `numpy.random.default_rng(42)` for
reproducibility.
