from __future__ import annotations

import polars as pl
import pytest

import polars_ds as pds


def test_random_functions_in_streaming():
    df = pl.DataFrame({"x": list(range(0, 50))}).lazy()

    try:
        df_test = df.with_columns(
            x1=pds.random(len_ref="x"),
            x2=pds.random_normal(0.0, 1.0, len_ref="x"),
            x3=pds.random_int(0, 10, len_ref="x"),
            x4=pds.random_exp(0.5, len_ref="x"),
            x5=pds.random_str(1, 3, len_ref="x"),
            x6=pds.random_binomial(5, 0.3, len_ref="x"),
        )
        _ = df_test.collect(engine="streaming")
        for batch in df_test.collect_batches(chunk_size=10):
            assert batch.shape == (10, 7)

        assert True
    except Exception as e:
        raise e


@pytest.mark.parametrize("dtype", [pl.Int64, pl.UInt32, pl.Float64])
def test_random_len_ref_single_row_any_dtype(dtype):
    # A reference column of length 1 must produce 1 sample, whatever its dtype or value.
    df = pl.DataFrame({"x": pl.Series([7], dtype=dtype)})
    out = df.lazy().select(pds.random(len_ref="x").alias("r")).collect(engine="in-memory")
    assert out.shape == (1, 1)


def test_random_len_ref_expr_and_str():
    df = pl.DataFrame({"x": list(range(5))})
    for ref in ("x", pl.col("x")):
        out = df.lazy().select(pds.random_int(0, 10, len_ref=ref).alias("r")).collect()
        assert out.shape == (5, 1)

    with pytest.raises(TypeError):
        pds.random(len_ref=1)  # type: ignore


def test_random_len_ref_multi_row_streaming_length():
    df = pl.DataFrame({"x": list(range(1000))}).lazy()
    out = df.select(pds.random_normal(0.0, 1.0, len_ref="x").alias("r")).collect(
        engine="streaming"
    )
    assert out.shape == (1000, 1)


@pytest.mark.parametrize("engine", ["streaming", "in-memory"])
def test_seeded_noise_not_repeated_across_batches(engine):
    # Seeded calls are not streamed (Polars v1 and v2), otherwise each streaming batch would
    # restart the RNG from the same seed and repeat the same random numbers.
    for n in (50, 300_000):
        df = pl.DataFrame({"x": [0.0] * n}).lazy()
        out = df.select(
            pds.jitter("x", std=1.0, seed=42).alias("j"),
            pds.perturb("x", seed=42).alias("p"),
            pds.random(seed=42, len_ref="x").alias("r"),
        ).collect(engine=engine)
        assert out["j"].n_unique() == n
        assert out["p"].n_unique() == n
        assert out["r"].n_unique() == n
