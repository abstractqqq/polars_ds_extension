from __future__ import annotations

import polars as pl
import random
import math
from typing import List, Tuple
from itertools import combinations, islice
# Internal dependency
from polars_ds.typing import PolarsFrame

__all__ = [
    "sample",
    "volume_neutral",
    "downsample",
    "random_cols",
    "split_by_ratio"
]


def _sampler_expr(value: float | int, seed: int | None = None) -> pl.Expr:
    if isinstance(value, float):
        if value >= 1.0 or value <= 0.0:
            raise ValueError("Sample rate must be in (0, 1) range.")
        return pl.int_range(0, pl.len(), dtype=pl.UInt32).shuffle(seed) < (pl.len() * value).cast(
            pl.UInt32
        )
    elif isinstance(value, int):
        if value <= 0:
            raise ValueError("Sample count must be > 0.")
        return pl.int_range(0, pl.len(), dtype=pl.UInt32).shuffle(seed) < pl.lit(
            value, dtype=pl.UInt32
        )
    elif isinstance(value, pl.Expr):
        return NotImplemented
    else:
        raise ValueError("Sample value must be either int or float.")


def sample(df: PolarsFrame, value: float | int, seed: int | None = None) -> pl.DataFrame:
    """
    Samples the dataframe.

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    value
        Either an int, in which case it means get n random rows from the df, or a float in the range (0, 1), in which
        case it means sample x% of the dataframe.
    seed
        A random seed
    """
    return df.lazy().filter(_sampler_expr(value, seed)).collect()


def volume_neutral(
    df: PolarsFrame,
    by: pl.Expr,
    control: pl.Expr | List[pl.Expr] | None = None,
    target_volume: int | None = None,
    seed: int | None = None,
) -> pl.DataFrame:
    """
    Select volume neutral many population from each segment in `by`, with optional control categories.

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    by
        A Polars expression represeting a column with discrete values. Volume neutral by the values
        in this column.
    control
        Additional level(s). If not none, the volume neutral selection will happen at the
        sublevel of the control column(s). See example.
    target_volume
        If none, it will select min(a, b, c) rows.
    seed
        A random seed
    """
    if isinstance(target_volume, int):
        target = pl.min_horizontal(by.value_counts().struct.field("count").min(), target_volume)
    else:
        target = by.value_counts().struct.field("count").min()

    if isinstance(control, pl.Expr):
        ctrl = [control]
    elif isinstance(control, list):
        ctrl = [c for c in control if isinstance(c, pl.Expr)]
    else:
        ctrl = []

    if len(ctrl) > 0:
        target = target.over(ctrl)
        final_ref = ctrl + [by]
    else:
        final_ref = by

    return (
        df.lazy().filter(pl.int_range(0, pl.len()).shuffle(seed).over(final_ref) < target).collect()
    )


def downsample(
    df: PolarsFrame,
    *conditions: Tuple[pl.Expr, float | int],
    seed: int | None = None,
) -> pl.DataFrame:
    """
    Downsamples data on the subsets where condition is true.

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    conditions
        Tuple[pl.Expr, float|int] or a sequence of such tuples as positional arguments.
        The first entry in the tuple should be a boolean expression and the second entry means we sample
        either n or x% on the part where the boolean is true.
    seed
        A random seed

    Example
    -------
    >>> import polars_ds.sample as sa
    >>> df.group_by("y").len().sort("y")
    shape: (4, 2)
    ┌─────┬───────┐
    │ y   ┆ len   │
    │ --- ┆ ---   │
    │ i32 ┆ u32   │
    ╞═════╪═══════╡
    │ 0   ┆ 24875 │
    │ 1   ┆ 25172 │
    │ 2   ┆ 25018 │
    │ 3   ┆ 24935 │
    └─────┴───────┘
    >>> downsampled = sa.down_sample(
    >>>     df,
    >>>     [(pl.col("y") == 1, 0.5), (pl.col("y") == 2, 0.5)]
    >>> )
    >>> downsampled.group_by("y").len().sort("y")
    shape: (4, 2)
    ┌─────┬───────┐
    │ y   ┆ len   │
    │ --- ┆ ---   │
    │ i32 ┆ u32   │
    ╞═════╪═══════╡
    │ 0   ┆ 24875 │
    │ 1   ┆ 12586 │
    │ 2   ┆ 12509 │
    │ 3   ┆ 24935 │
    └─────┴───────┘
    >>> # And this is equivalent to
    >>> downsampled = sa.down_sample(
    >>>     df,
    >>>     [(pl.col("y").is_between(1, 2, closed="both"), 0.5)]
    >>> )
    """
    all_filters = ((_sampler_expr(r, seed).over(c) | (~c)) for c, r in conditions)
    return df.lazy().filter(pl.lit(True).and_(*all_filters)).collect()


def random_cols(
    all_columns:List[str],
    k: int,
    keep: List[str] | None = None,
    seed: int | None = None,
) -> List[str]:
    """
    Selects random columns from the given pool of columns. Returns the selected columns in a list. 
    Note, it is impossible for this to randomly select both ["x", "y"] and ["y", "x"].

    Parameters
    ----------
    all_columns
        All column names
    k
        Select k random columns from all columns outside of `keep`.
    keep
        Columns to always keep
    seed
        A random seed
    """
    if keep is None:
        out = []
        to_sample = combinations(all_columns, k)
    else:
        out = keep
        to_sample = combinations((c for c in all_columns if c not in keep), k)

    pool_size = len(all_columns) - len(out)
    if pool_size < k:
        raise ValueError("Not enough columns to select from.")

    n = random.randrange(0, math.comb(pool_size, k))
    rand_cols = next(islice(to_sample, n, None), None)
    return out + list(rand_cols)


def split_by_ratio(
    df: PolarsFrame, split_ratio: float | List[float], seed: int | None = None
) -> List[pl.DataFrame]:
    """
    Split the dataframe into multiple. If split_ratio is a single number, it is treated as the
    ratio for the "train" set and the second is always the "test" set. If split_ratio is a list
    of floats, then they must sum to 1 and the return will be dataframes split into the corresponding
    ratios. Please avoid using floating point values with too many decimal places, which may cause
    the splits to be off by a 1 row.

    This will collect your LazyFrames.

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    split_ratio
        Either a single float or a list of floats.
    seed
        The random seed
    """

    if isinstance(split_ratio, float):
        if split_ratio <= 0.0 or split_ratio >= 1:
            raise ValueError("Split ratio must be > 0 and < 1.")

        frames = (
            df.lazy()
            .with_row_index(name="__id")
            .collect()
            .with_columns(
                (pl.col("__id").shuffle(seed=seed) < (pl.len() * split_ratio).cast(pl.Int64)).alias(
                    "__tt"
                )
            )
            .partition_by("__tt", as_dict=True)
        )
        train = frames[(True,)].select(pl.col("*").exclude(["__id", "__tt"]))
        test = frames[(False,)].select(pl.col("*").exclude(["__id", "__tt"]))
        return [train, test]
    else: # Should work with iterable (with a length), not just list
        if len(split_ratio) == 1:
            return split_by_ratio(df, split_ratio[0], seed)
        else:
            if sum(split_ratio) != 1:
                raise ValueError("Sum of the ratios is not 1.")

            df_eager = (
                df.with_row_index(name="__id")
                .with_columns(pl.col("__id").shuffle(seed=seed).alias("__tt"))
                .sort("__tt")
                .lazy()
                .collect()
            )

            n = len(df_eager)
            start = 0
            dfs = []
            for v in split_ratio:
                length = int(n * v)
                dfs.append(
                    df_eager.slice(start, length=length).select(pl.col("*").exclude(["__id", "__tt"]))
                )
                start += length

            return dfs
