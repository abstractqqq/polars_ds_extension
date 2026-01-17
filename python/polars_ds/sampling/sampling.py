from __future__ import annotations

import polars as pl
import random
import math
from typing import List, Tuple, Dict
from itertools import combinations, islice

# Internal dependency
from polars_ds.typing import PolarsFrame

__all__ = ["sample", "volume_neutral", "downsample", "random_cols", "split_by_ratio"]


def _sampler_expr(value: float | int, seed: int | None = None) -> pl.Expr:
    r"""
    _sampler_expr
    ===========
    Construct a Polars expression that selects a random sample of rows.

    Parameters
    ----------    
    value : int or float
        If an integer is provided, `value` rows are selected.  Otherwise, a proportion of `value` over the `df` is selected.
    
    seed : int, optional, default=None
        The seed value for the random number generator. The same seed will produce the same output each time.
    
    Returns
    ----------
    polars.Expr
        Returns a boolean Polars expression indicating which rows are selected.
    """
    # Input(s)
    if not isinstance(value, (int, float)):
        raise TypeError("'value' is neither an integer or a float.")
    elif isinstance(value, int):
        if value <= 0:
            raise ValueError("'value' must be greater than zero.")
    elif isinstance(value, float):
        if value >= 1.0 or value <= 0.0:
            raise ValueError("'value' must be in the range (0, 1)")

    if seed is not None and not isinstance(seed, int):
        raise TypeError("'seed' must be an integer or None.")

    # Engine
    if isinstance(value, int):
        sample_expr = (
            pl.int_range(0, pl.len(), dtype=pl.UInt32)
            .shuffle(seed) < pl.lit(value, dtype = pl.UInt32)
        )
    else:
        sample_expr = (
            pl.int_range(0, pl.len(), dtype=pl.UInt32)
            .shuffle(seed) < (pl.len() * value)
            .cast(pl.UInt32)
        )

    # Output(s)
    return sample_expr

def sample(df: PolarsFrame, value: float | int, seed: int | None = None, return_df: bool = False) -> PolarsFrame:
    r"""
    sample
    ===========
    Extracts a random sample from a Polars DataFrame or LazyFrame. 

    Parameters
    ----------
    df : PolarsFrame
        It may be either a polars.DataFrame or a polars.LazyFrame.
    
    value : int or float
        If an integer is provided, `value` observations are selected from `df`. Otherwise, a proportion of `value` over the `df` is selected.
    
    seed : int, optional, default=None
        The seed value for the random number generator. The same seed will produce the same output each time.

    return_df : bool, optional, default=False
        Determines whether the output should always be a polars.DataFrame or not.
    
    Returns
    ----------
    PolarsFrame
        Returns either a polars.DataFrame or a polars.LazyFrame depending on the `df` provided.
    
    Example
    ----------
    >>> import polars as pl
    >>> import polars_ds.sampling as pds_sa
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> lf = pl.LazyFrame(
            data = {
                "id": range(1, 1001)
                ,"value": np.random.rand(1000) * 100
                ,"category": np.random.choice(["A", "B", "C"], size = 1000)
            }
        )
    >>> pds_sa.sample(lf, 100, 101, True).head(3)
    shape: (3, 3)
    id	value	category
    i64	f64	str
    7	5.808361	"C"
    33	6.505159	"C"
    42	49.517691	"B"

    >>> pds_sa.sample(lf, 0.5, 101, True).head(3)
    shape: (3, 3)
    ┌─────┬───────────┬──────────┐
    │ id  ┆	value	  ┆ category │
    │ --- ┆ ---       ┆ ---      │
    │ i64 ┆	f64	      ┆ str      │
    │ 3   ┆	73.199394 ┆	"C"      │
    │ 4   ┆	59.865848 ┆	"C"      │
    │ 5   ┆	15.601864 ┆	"A"      │
    └─────┴───────────┴──────────┘
    """
    # Input(s)
    if not isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        raise TypeError("'df' is neither a polars.DataFrame or a polars.LazyFrame")

    if not isinstance(return_df, bool):
        raise TypeError("'return_df' is not a boolean.")

    # Engine
    sample = df.filter(_sampler_expr(value, seed))

    # Output(s)
    if isinstance(df, pl.LazyFrame) and return_df:
        sample = sample.collect()
    return sample


def volume_neutral(
    df: PolarsFrame,
    by: pl.Expr,
    control: pl.Expr | List[pl.Expr] | None = None,
    target_volume: int | None = None,
    seed: int | None = None,
    return_df: bool = False
) -> PolarsFrame:
    r"""
    volume_neutral
    ===========
    Subsample a polars.DataFrame or polars.LazyFrame to achieve volume neutrality per group,
    optionally controlling for additional grouping variables.  

    This function reduces each group defined by `by` (and optionally `control`) to a 
    target number of rows, ensuring that all groups have the same number of observations. 
    The selection within groups is randomized, with an optional seed for reproducibility. 

    Parameters
    ----------
    df : PolarsFrame
        It may be either a polars.DataFrame or a polars.LazyFrame.

    by : pl.Expr
        Expression defining the primary grouping discrete variable for volume balancing.

    control : pl.Expr or list of pl.Expr, optional, default=None
        Additional expressions to control grouping. Subsampling is done within each 
        combination of `control` and `by`.

    target_volume : int, optional, default=None
        Maximum number of rows to retain per group. If None, the size of the smallest 
        group is used.

    seed : int, optional, default=None
        The seed value for the random number generator. The same seed will produce the same output each time.

    return_df : bool, default=False
        Determines whether the output should always be a polars.DataFrame or not.
    
    Returns
    ----------
    PolarsFrame
        Returns either a polars.DataFrame or a polars.LazyFrame depending on the `df` provided.
    
    Example
    ----------
    >>> import polars as pl
    >>> import polars_ds.sampling as pds_sa
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> lf = pl.LazyFrame(
            data = {
                "id": range(1, 1001)
                ,"value": np.random.rand(1000) * 100
                ,"category": np.random.choice(["A", "B", "C"], size = 1000)
            }
        )
    >>> pds_sa.volume_neutral(lf, pl.col("category"), None, 2, 101, True)
    shape: (6, 3)
    ┌─────┬───────────┬──────────┐
    │ id  ┆	value	  ┆ category │
    │ --- ┆ ---       ┆ ---      │
    │ i64 ┆	f64	      ┆ str      │
    │ 817 ┆	59.127544 ┆	"A"      │
    │ 825 ┆	53.73956  ┆	"B"      │
    │ 874 ┆	40.873417 ┆	"C"      │
    │ 909 ┆	25.942343 ┆	"A"      │
    │ 923 ┆	89.455223 ┆	"B"      │
    │ 990 ┆	81.910232 ┆	"C"      │
    └─────┴───────────┴──────────┘
    """
    # Input(s)
    if not isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        raise TypeError("'df' is neither a polars.DataFrame or a polars.LazyFrame")

    if not isinstance(by, pl.Expr):
        raise TypeError("'by' is not a polars.Expression")

    if control is not None and not isinstance(control, (pl.Expr, list)):
        raise TypeError("'control' is not a polars.Expression, list, or None.")

    if isinstance(control, list):
        ctrl = [True if isinstance(c, pl.Expr) else False for c in control]
        if sum(ctrl) < len(control):
            raise TypeError("'control' contais elements that are not a polars.Expression.")

    if target_volume is not None and not isinstance(target_volume, int):
        raise TypeError("'target_volume' must be an integer or None.")

    if seed is not None and not isinstance(seed, int):
        raise TypeError("'seed' must be an integer or None.")

    if not isinstance(return_df, bool):
        raise TypeError("'return_df' is not a boolean.")

    # Engine
    if target_volume is not None:
        target = pl.min_horizontal(by.value_counts().struct.field("count").min(), target_volume)
    else:
        target = by.value_counts().struct.field("count").min()

    if isinstance(control, (pl.Expr, list)):
        ctrl = [control]
    else:
        ctrl = []

    if len(ctrl) > 0:
        target = target.over(ctrl)
        final_ref = ctrl + [by]
    else:
        final_ref = by
    
    volume_neutral = df.filter(pl.int_range(0, pl.len()).shuffle(seed).over(final_ref) < target)

    # Output
    if isinstance(df, pl.LazyFrame) and return_df:
        volume_neutral = volume_neutral.collect()
    return volume_neutral


def downsample(
    df: PolarsFrame,
    *conditions: Tuple[pl.Expr, float | int],
    seed: int | None = None,
    return_df: bool = False
) -> PolarsFrame:
    """
    downsample
    ===========
    Downsamples data on the subsets where condition is true.

    Parameters
    ----------
    df : PolarsFrame
        It may be either a polars.DataFrame or a polars.LazyFrame.

    conditions : 
        Tuple[pl.Expr, float|int] or a sequence of such tuples as positional arguments.
        The first entry in the tuple should be a boolean expression and the second entry means we sample
        either n or x% on the part where the boolean is true.

    seed : int, optional, default=None
        The seed value for the random number generator. The same seed will produce the same output each time.

    return_df : bool, default=False
        Determines whether the output should always be a polars.DataFrame or not.

    Returns
    ----------
    PolarsFrame
        Returns either a polars.DataFrame or a polars.LazyFrame depending on the `df` provided.

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
    # Input(s)

    # Engine

    # Output(s)

    all_filters = ((_sampler_expr(r, seed).over(c) | (~c)) for c, r in conditions)
    return df.lazy().filter(pl.lit(True).and_(*all_filters)).collect()


def random_cols(
    all_columns: List[str],
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
    df: PolarsFrame,
    split_ratio: float | List[float] | Dict[str, float],
    seed: int | None = None,
    split_col: str = "__split",
    by: str | list[str] | None = None,
    default_split_1: str = "train",
    default_split_2: str = "test",
) -> PolarsFrame:
    """
    Creates a random `split` column in the dataframe.

    If split_ratio is a single number, it is treated as the ratio for the default_split_1 set, which by
    default has the name 'train', and (1-ratio) is always the ratio for the default_split_2 set, which by
    default has the name 'test'.

    If split_ratio is a list of floats, then they must sum to 1 and the return will be the corresponding ratios
    and the split column values will be "split_i".

    If the split_ratio is a dict, then the dict values will be the ratios and the dict keys will be the value
    in the split column.

    Please avoid using floating point values with too many decimal places, which may cause
    the splits to be off by a 1 row.

    This will return lazy frames if input is lazy, and eager frames if input is eager. But if `by` is given,
    then the dataframe will be collected and will only return eager dataframe in the end.

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    split_ratio
        Either a single float or a list of floats.
    split_col
        The column name of the split
    seed
        The random seed
    by
        Optional str of list of str. The column(s) to stratify by.
    default_split_1
        Name of the first split when split_ratio is a scalar
    default_split_2
        Name of the second split when split_ratio is a scalar
    """

    if by is not None:
        results = [
            split_by_ratio(
                frame,
                split_ratio=split_ratio,
                seed=seed,
                by=None,
                split_col=split_col,
                default_split_1=default_split_1,
                default_split_2=default_split_2,
            )
            for frame in df.lazy().collect().partition_by(by)
        ]
        return pl.concat(results, how="vertical")

    if isinstance(split_ratio, float):
        if split_ratio <= 0.0 or split_ratio >= 1:
            raise ValueError("Split ratio must be > 0 and < 1.")

        return (
            df.with_row_index(name="__id")
            .with_columns(
                pl.when(pl.col("__id").shuffle(seed=seed) < (pl.len() * split_ratio).cast(pl.Int64))
                .then(pl.lit(default_split_1, dtype=pl.String))
                .otherwise(pl.lit(default_split_2, dtype=pl.String))
                .alias(split_col)
            )
            .select(pl.all().exclude("__id"))
        )

    else:
        if len(split_ratio) == 1:
            raise ValueError("If split_ratio is not a scalar, it must have length > 1.")
        else:
            if isinstance(split_ratio, dict):
                ratios: pl.Series = pl.Series(split_ratio.values())
                split_names = [str(k) for k in split_ratio.keys()]
            else:  # should work with other iterables like tuple
                ratios: pl.Series = pl.Series(split_ratio)
                split_names = [f"split_{i}" for i in range(len(split_ratio))]

            if ratios.sum() != 1:
                raise ValueError("Sum of the ratios is not 1.")

            pct = ratios.cum_sum()
            expr = pl.when(pl.lit(False)).then(None)
            for p, k in zip(pct, split_names):
                expr = expr.when(pl.col("__pct") < p).then(pl.lit(k, dtype=pl.String))

            return (
                df.with_row_index(name="__id")
                .with_columns(pl.col("__id").shuffle(seed=seed).alias("__tt"))
                .sort("__tt")
                .with_columns((pl.col("__tt") / pl.len()).alias("__pct"))
                .select(expr.alias(split_col), pl.all().exclude(["__id", "__pct", "__tt"]))
            )
