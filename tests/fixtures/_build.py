"""
Fixture builder for polars-ds parity oracle.
Run: python tests/fixtures/_build.py
SEED = 42
"""
from pathlib import Path
import numpy as np
import polars as pl

SEED = 42
OUT = Path(__file__).parent
rs = np.random.RandomState(SEED)

CATS = ["alpha", "beta", "gamma", "delta", "epsilon"]


# ── tiny_clean (8 rows, 1 chunk, no nulls) ──────────────────────────────────
def build_tiny_clean() -> pl.DataFrame:
    n = 8
    x = rs.randn(n)
    y = rs.randn(n)
    weight = rs.uniform(0.1, 2.0, n)
    label = rs.randint(0, 2, n).astype(np.int32)
    cat = [CATS[i % len(CATS)] for i in range(n)]
    return pl.DataFrame(
        {
            "x": pl.Series(x, dtype=pl.Float64),
            "y": pl.Series(y, dtype=pl.Float64),
            "weight": pl.Series(weight, dtype=pl.Float64),
            "label": pl.Series(label, dtype=pl.Int32),
            "cat": pl.Series(cat, dtype=pl.String),
        }
    )


# ── tiny_with_nulls (20 rows, 1 chunk, ~20% nulls at fixed rows) ────────────
def build_tiny_with_nulls() -> pl.DataFrame:
    n = 20
    x_raw = rs.randn(n).tolist()
    y_raw = rs.randn(n).tolist()
    label = rs.randint(0, 2, n).astype(np.int32).tolist()
    cat_raw = [CATS[i % len(CATS)] for i in range(n)]

    # Fixed null positions (approximately 20% = 4 rows)
    null_x_rows = [2, 7, 13, 18]
    null_y_rows = [4, 9, 14, 19]
    null_cat_rows = [1, 6, 11, 16]

    for i in null_x_rows:
        x_raw[i] = None
    for i in null_y_rows:
        y_raw[i] = None
    for i in null_cat_rows:
        cat_raw[i] = None

    return pl.DataFrame(
        {
            "x": pl.Series(x_raw, dtype=pl.Float64),
            "y": pl.Series(y_raw, dtype=pl.Float64),
            "label": pl.Series(label, dtype=pl.Int32),
            "cat": pl.Series(cat_raw, dtype=pl.String),
        }
    )


# ── tiny_with_specials (16 rows, 1 chunk, NaN/Inf/-Inf at fixed positions) ──
def build_tiny_with_specials() -> pl.DataFrame:
    n = 16
    x = rs.randn(n)
    # row 1 = NaN, row 2 = +Inf, row 3 = -Inf
    x[1] = float("nan")
    x[2] = float("inf")
    x[3] = float("-inf")
    y = rs.randn(n)
    label = rs.randint(0, 2, n).astype(np.int32)
    return pl.DataFrame(
        {
            "x": pl.Series(x, dtype=pl.Float64),
            "y": pl.Series(y, dtype=pl.Float64),
            "label": pl.Series(label, dtype=pl.Int32),
        }
    )


# ── medium_multichunk (2000 rows, 4 chunks) ──────────────────────────────────
# Parquet round-trips to single-chunk; we store 4 separate files and
# reconstruct via pl.concat(..., rechunk=False) in the loader.
def build_medium_multichunk() -> list[pl.DataFrame]:
    n_total = 2000
    chunk_size = n_total // 4
    chunks = []
    for _ in range(4):
        x1 = rs.randn(chunk_size)
        x2 = rs.randn(chunk_size)
        x3 = rs.randn(chunk_size)
        label = rs.randint(0, 2, chunk_size).astype(np.int32)
        cat = [CATS[i % len(CATS)] for i in range(chunk_size)]
        chunks.append(
            pl.DataFrame(
                {
                    "x1": pl.Series(x1, dtype=pl.Float64),
                    "x2": pl.Series(x2, dtype=pl.Float64),
                    "x3": pl.Series(x3, dtype=pl.Float64),
                    "label": pl.Series(label, dtype=pl.Int32),
                    "cat": pl.Series(cat, dtype=pl.String),
                }
            )
        )
    return chunks


# ── medium_with_nulls (2000 rows, 1 chunk, ~5% nulls in x1) ─────────────────
def build_medium_with_nulls() -> pl.DataFrame:
    n = 2000
    x1_raw = rs.randn(n).tolist()
    x2 = rs.randn(n)
    label = rs.randint(0, 2, n).astype(np.int32)
    weight = rs.uniform(0.1, 2.0, n)

    # ~5% nulls in x1 at fixed seeded positions
    rng2 = np.random.RandomState(SEED + 1)
    null_mask = rng2.rand(n) < 0.05
    for i in range(n):
        if null_mask[i]:
            x1_raw[i] = None

    return pl.DataFrame(
        {
            "x1": pl.Series(x1_raw, dtype=pl.Float64),
            "x2": pl.Series(x2, dtype=pl.Float64),
            "label": pl.Series(label, dtype=pl.Int32),
            "weight": pl.Series(weight, dtype=pl.Float64),
        }
    )


def main() -> None:
    print("Building fixtures ...")

    df = build_tiny_clean()
    df.write_parquet(OUT / "tiny_clean.parquet")
    print(f"  tiny_clean.parquet          {df.shape}")

    df = build_tiny_with_nulls()
    df.write_parquet(OUT / "tiny_with_nulls.parquet")
    print(f"  tiny_with_nulls.parquet     {df.shape}")

    df = build_tiny_with_specials()
    df.write_parquet(OUT / "tiny_with_specials.parquet")
    print(f"  tiny_with_specials.parquet  {df.shape}")

    chunks = build_medium_multichunk()
    for i, chunk in enumerate(chunks):
        chunk.write_parquet(OUT / f"medium_multichunk_p{i}.parquet")
    combined = pl.concat(chunks, rechunk=False)
    print(
        f"  medium_multichunk_p0..p3    {combined.shape}  n_chunks={combined['x1'].n_chunks()}"
    )

    df = build_medium_with_nulls()
    df.write_parquet(OUT / "medium_with_nulls.parquet")
    print(f"  medium_with_nulls.parquet   {df.shape}")

    print("Done.")


if __name__ == "__main__":
    main()
