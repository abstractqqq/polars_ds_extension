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
        if not 0 < value < 1:
            raise ValueError("'value' must be in the range (0, 1).")

    if seed is not None and not isinstance(seed, int):
        raise TypeError("'seed' is neither an integer or None.")

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
    >>> import polars_ds.sampling as pds_samp
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> lf = pl.LazyFrame(
            data = {
                "id": range(1, 1001)
                ,"value": np.random.rand(1000) * 100
                ,"category": np.random.choice(["A", "B", "C"], size = 1000)
            }
        )
    >>> print(pds_sa.sample(lf, 100, 101, True).head(3))
    shape: (3, 3)
    ┌─────┬───────────┬──────────┐
    │ id  ┆ value     ┆ category │
    │ --- ┆ ---       ┆ ---      │
    │ i64 ┆ f64       ┆ str      │
    ╞═════╪═══════════╪══════════╡
    │ 7   ┆ 5.808361  ┆ C        │
    │ 33  ┆ 6.505159  ┆ C        │
    │ 42  ┆ 49.517691 ┆ B        │
    └─────┴───────────┴──────────┘

    >>> print(pds_samp.sample(lf, 0.5, 101, True).head(3))
    shape: (3, 3)
    ┌─────┬───────────┬──────────┐
    │ id  ┆ value     ┆ category │
    │ --- ┆ ---       ┆ ---      │
    │ i64 ┆ f64       ┆ str      │
    ╞═════╪═══════════╪══════════╡
    │ 3   ┆ 73.199394 ┆ C        │
    │ 4   ┆ 59.865848 ┆ C        │
    │ 5   ┆ 15.601864 ┆ A        │
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
    >>> import polars_ds.sampling as pds_samp
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> lf = pl.LazyFrame(
            data = {
                "id": range(1, 1001)
                ,"value": np.random.rand(1000) * 100
                ,"category": np.random.choice(["A", "B", "C"], size = 1000)
            }
        )
    >>> print(pds_samp.volume_neutral(lf, pl.col("category"), None, 2, 101, True))
    shape: (6, 3)
    ┌─────┬───────────┬──────────┐
    │ id  ┆ value     ┆ category │
    │ --- ┆ ---       ┆ ---      │
    │ i64 ┆ f64       ┆ str      │
    ╞═════╪═══════════╪══════════╡
    │ 817 ┆ 59.127544 ┆ A        │
    │ 825 ┆ 53.73956  ┆ B        │
    │ 874 ┆ 40.873417 ┆ C        │
    │ 909 ┆ 25.942343 ┆ A        │
    │ 923 ┆ 89.455223 ┆ B        │
    │ 990 ┆ 81.910232 ┆ C        │
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
    conditions: List[Tuple[pl.Expr, float | int]] | Tuple[pl.Expr, float | int],
    seed: int | None = None,
    return_df: bool = False
) -> PolarsFrame:
    """
    downsample
    ===========
    Downsamples subsets of a Polars DataFrame or LazyFrame based on specified conditions.

    This function applies downsampling to rows where each boolean condition is true.
    For each condition, you can specify either a fixed number of rows to keep (int)
    or a fraction of rows to keep (float). The downsampling is performed using a
    random sampling strategy, which can be made reproducible using a seed.

    Parameters
    ----------
    df : PolarsFrame
        It may be either a polars.DataFrame or a polars.LazyFrame.

    conditions : List[Tuple[pl.Expr, float | int]] | Tuple[pl.Expr, float | int]
        One or more tuples, each containing:
        - A boolean Polars expression (`polars.Expr`) defining the subset of rows to downsample.
        - A float (fraction of rows to keep, e.g., 0.5 for 50%) or an integer (fixed number of rows to keep).

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
    >>> import polars as pl
    >>> import polars_ds.sampling as pds_samp
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> lf = pl.LazyFrame(
            data = {
                "id": range(1, 1001)
                ,"value": np.random.rand(1000) * 100
                ,"category": np.random.choice(["A", "B", "C"], size = 1000)
            }
        )
    >>> print(lf.group_by("category").len().sort("category").collect())
    shape: (3, 2)
    ┌──────────┬─────┐
    │ category ┆ len │
    │ ---      ┆ --- │
    │ str      ┆ u32 │
    ╞══════════╪═════╡
    │ A        ┆ 341 │
    │ B        ┆ 343 │
    │ C        ┆ 316 │
    └──────────┴─────┘
    >>> print(pds_samp.downsample(
    >>>     lf,
    >>>     [
    >>>         (pl.col("category") == "A", 0.25),
    >>>         (pl.col("category") == "B", 10)
    >>>     ]
    >>> ).group_by("category").len().sort("category").collect())
    shape: (3, 2)
    ┌──────────┬─────┐
    │ category ┆ len │
    │ ---      ┆ --- │
    │ str      ┆ u32 │
    ╞══════════╪═════╡
    │ A        ┆ 85  │
    │ B        ┆ 10  │
    │ C        ┆ 316 │
    └──────────┴─────┘
    """
    # Input(s)
    if not isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        raise TypeError("'df' is neither a polars.DataFrame or a polars.LazyFrame")

    if seed is not None and not isinstance(seed, int):
        raise TypeError("'seed' is neither an integer or None.")

    if not isinstance(return_df, bool):
        raise TypeError("'return_df' is not a boolean.")
    
    if isinstance(conditions, tuple):
        conditions = [conditions]
    for condition in conditions:
        if not isinstance(condition, tuple) or len(condition) != 2:
            raise ValueError("Each condition must be a tuple of length 2.")

        expr, value = condition
        if not isinstance(expr, pl.Expr):
            raise TypeError("The first element of each condition must be a polars expression (pl.Expr).")

        if not isinstance(value, (float, int)):
            raise TypeError("The second element of each condition must be a float or an integer.")

        if isinstance(value, float) and not (0 <= value <= 1):
            raise ValueError("If the second element is a float, must be in the range [0, 1].")

    # Engine
    all_filters = ((_sampler_expr(r, seed).over(c) | (~c)) for c, r in conditions)
    downsample = df.filter(pl.lit(True).and_(*all_filters))

    # Output(s)
    if isinstance(df, pl.LazyFrame) and return_df:
        downsample = downsample.collect()
    return downsample


def random_cols(
    all_columns: List[str],
    k: int,
    keep: List[str] | None = None,
    seed: int | None = None,
) -> List[str]:
    """
    random_cols
    ===========
    Randomly select columns from the provided list of column names.

    Parameters
    ----------
    all_columns : List[str]
        List with the name of the columns from which to drawn randomly.

    k : int
        Number of columns to select randomly outside of the list provided in `keep`.

    keep : List[str], optional, default=None
        List of values to always include in the list of randomly drawn columns.

    seed : int, optional, default=None
        The seed value for the random number generator. The same seed will produce the same output each time.

    Returns
    ----------
    List[str]
        Returns a list with the name of the columns that were randomly drawn.

    Note(s)
    ----------
    - It is impossible to randomly select both ["x", "y"] and ["y", "x"].

    Example
    -------
    >>> import polars as pl
    >>> import polars_ds.sampling as pds_samp
    >>> pds_samp.random_cols(["a", "b", "c", "d", "e", "f"], 2, seed = 101)
    ['c', 'd']
    """
    # Input(s)
    if not isinstance(all_columns, list):
        raise TypeError("'all_columns' must be a list.")
    for element in all_columns:
        if not isinstance(element, str):
            raise ValueError("All values provided in 'all_columns' must be strings.")

    if not isinstance(k, int) or k <= 0:
        raise TypeError("'k' must be an integer greater than 0.")

    if keep is not None and not isinstance(keep, list):
        raise TypeError("'keep' must be a list or None.")
    if keep is not None:
        for element in keep:
            if not isinstance(element, str):
                raise ValueError("All values provided in 'keep' must be strings.")

    if seed is not None and not isinstance(seed, int):
        raise TypeError("'seed' is neither an integer or None.")

    # Engine
    if seed is not None:
        random.seed(seed)

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
    random_cols = out + list(rand_cols)

    # Output(s)
    return random_cols

def split_by_ratio(
    df: PolarsFrame,
    split_ratio: float | List[float] | Dict[str, float],
    split_col: str = "__split",
    by: str | list[str] | None = None,
    default_split_1: str = "train",
    default_split_2: str = "test",
    seed: int | None = None,
    return_df: bool = False
) -> PolarsFrame:
    """
    split_by_ratio
    ===========
    Randomly splits a Polars DataFrame or LazyFrame into subsets based on specified ratios.

    The function adds a new column (`split_col`) to the DataFrame/LazyFrame, assigning each row to a subset
    according to the provided `split_ratio`. The splitting can be stratified by one or more columns
    if the `by` parameter is specified.

    Parameters
    ----------
    df : PolarsFrame
        It may be either a polars.DataFrame or a polars.LazyFrame.

    split_ratio : float | List[float] | Dict[str, float]
       - **Float**: The ratio for the first subset (default: "train"), with the remainder assigned
        to the second subset (default: "test").
        - **List of floats**: Each float represents the ratio for a subset, and the list must sum to 1.
        Subsets are named "split_0", "split_1", etc.
        - **Dictionary**: Keys are subset names, and values are their respective ratios. The values must
        sum to 1.

    split_col : str, optional, default="__split"
        Name of the column to store the split assignments.

    by : str | list[str], optional, default=None
        Column(s) to stratify by. If specified, the DataFrame is collected and split within each stratum.

    default_split_1 : str, optional, default="train"
        Name of the first subset when `split_ratio` is a float.

    default_split_2 : str, optional, default="test"
        Name of the second subset when `split_ratio` is a float.

    seed : int, optional, default=None
        The seed value for the random number generator. The same seed will produce the same output each time.

    return_df : bool, default=False
        Determines whether the output should always be a polars.DataFrame or not.

    Returns
    ----------
    PolarsFrame
        Returns either a polars.DataFrame or a polars.LazyFrame depending on the `df` provided.

    Note(s)
    ----------
    - Avoid using floating-point values with too many decimal places, as this may cause the
    splits to be off by one row due to rounding errors.

    Example
    -------
    >>> import polars as pl
    >>> import polars_ds.sampling as pds_samp
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> lf = pl.LazyFrame(
            data = {
                "id": range(1, 1001)
                ,"value": np.random.rand(1000) * 100
                ,"category": np.random.choice(["A", "B", "C"], size = 1000)
            }
        )
    >>> print(pds_samp.split_by_ratio(
    >>>     df = lf,
    >>>     split_ratio = 0.75,
    >>>     seed = 101
    >>> ).group_by(["__split", "category"]).len().sort(["__split", "category"]).collect())
    shape: (6, 3)
    ┌─────────┬──────────┬─────┐
    │ __split ┆ category ┆ len │
    │ ---     ┆ ---      ┆ --- │
    │ str     ┆ str      ┆ u32 │
    ╞═════════╪══════════╪═════╡
    │ test    ┆ A        ┆ 98  │
    │ test    ┆ B        ┆ 63  │
    │ test    ┆ C        ┆ 89  │
    │ train   ┆ A        ┆ 243 │
    │ train   ┆ B        ┆ 280 │
    │ train   ┆ C        ┆ 227 │
    └─────────┴──────────┴─────┘

    >>> print(pds_samp.split_by_ratio(
    >>>     df = lf,
    >>>     split_ratio = 0.75,
    >>>     split_col = "sample",
    >>>     by = "category",
    >>>     seed = 101
    >>> ).group_by(["sample", "category"]).len().sort(["sample", "category"]).collect())
    shape: (6, 3)
    ┌────────┬──────────┬─────┐
    │ sample ┆ category ┆ len │
    │ ---    ┆ ---      ┆ --- │
    │ str    ┆ str      ┆ u32 │
    ╞════════╪══════════╪═════╡
    │ test   ┆ A        ┆ 86  │
    │ test   ┆ B        ┆ 86  │
    │ test   ┆ C        ┆ 79  │
    │ train  ┆ A        ┆ 255 │
    │ train  ┆ B        ┆ 257 │
    │ train  ┆ C        ┆ 237 │
    └────────┴──────────┴─────┘
    """
    # Input(s)
    if not isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        raise TypeError("'df' is neither a polars.DataFrame or a polars.LazyFrame")

    if not isinstance(split_ratio, (float, list, dict)):
        raise TypeError("'split_ratio' is neither a float, list or dictionary.")
    if isinstance(split_ratio, float):
        if not 0 < split_ratio < 1:
            raise ValueError("All values provided in 'split_ratio' must be in the range (0, 1).")
    if isinstance(split_ratio, list):
        for element in split_ratio:
            if not isinstance(element, float):
                raise ValueError("All values provided in 'split_ratio' must be floats.")
            else:
                if not 0 < element < 1:
                    raise ValueError("All values provided in 'split_ratio' must be in the range (0, 1).")
        if sum(split_ratio) != 1.0:
            raise ValueError("The sum of the values in 'split_ratio' must equal 1.")
    if isinstance(split_ratio, dict):
        for key, value in zip(split_ratio.keys(), split_ratio.values()):
            if not isinstance(key, str):
                raise ValueError("All key values provided in 'split_ratio' must be strings.")
            if not isinstance(value, float):
                raise ValueError("All values provided in 'split_ratio' must be floats.")
            else:
                if not 0 < value < 1:
                    raise ValueError("All values provided in 'split_ratio' must be in the range (0, 1).")
        if sum(split_ratio.values()) != 1.0:
            raise ValueError("The sum of the values in 'split_ratio' must equal 1.")

    if not isinstance(split_col, str):
        raise TypeError("'split_col' is not a string.")

    if by is not None and not isinstance(by, (str, list)):
        raise TypeError("'by' is not a string, list or None.")
    if isinstance(by, list):
        for element in by:
            if not isinstance(element, str):
                raise ValueError("All values provided in 'by' must be strings.")

    if not isinstance(default_split_1, str):
        raise TypeError("'default_split_1' is not a string.")

    if not isinstance(default_split_2, str):
        raise TypeError("'default_split_2' is not a string.")

    if seed is not None and not isinstance(seed, int):
        raise TypeError("'seed' is neither an integer or None.")

    if not isinstance(return_df, bool):
        raise TypeError("'return_df' is not a boolean.")

    # Engine
    ## Stratified Sampling
    if by is not None:
        results = []
        for cat in df.select(pl.col(by).unique()).collect().to_series().to_list():
            frame = df.filter(
                pl.col(by) == cat
            )
            results.append(
                split_by_ratio(
                    frame,
                    split_ratio = split_ratio,
                    seed = seed,
                    by = None,
                    split_col = split_col,
                    default_split_1 = default_split_1,
                    default_split_2 = default_split_2
                )
            )
            split_sample = pl.concat(results, how = "vertical")

    ## Simple Sampling
    else:
        if isinstance(split_ratio, float):
            split_sample = (
                df.with_row_index(name="__id")
                .with_columns(
                    pl.when(pl.col("__id").shuffle(seed = seed) < (pl.len() * split_ratio).cast(pl.Int64))
                    .then(pl.lit(default_split_1, dtype = pl.String))
                    .otherwise(pl.lit(default_split_2, dtype = pl.String))
                    .alias(split_col)
                )
                .select(pl.all().exclude("__id"))
            )

        else:
            if isinstance(split_ratio, dict):
                ratios: pl.Series = pl.Series(split_ratio.values())
                split_names = [str(k) for k in split_ratio.keys()]
            else:
                ratios: pl.Series = pl.Series(split_ratio)
                split_names = [f"split_{i}" for i in range(len(split_ratio))]

            pct = ratios.cum_sum()
            expr = pl.when(pl.lit(False)).then(None)
            for p, k in zip(pct, split_names):
                expr = expr.when(pl.col("__pct") < p).then(pl.lit(k, dtype = pl.String))
            
            split_sample = (
                    df.with_row_index(name="__id")
                    .with_columns(pl.col("__id").shuffle(seed = seed).alias("__tt"))
                    .sort("__tt")
                    .with_columns((pl.col("__tt") / pl.len()).alias("__pct"))
                    .select(expr.alias(split_col), pl.all().exclude(["__id", "__pct", "__tt"]))
                )

    # Output(s)
    if isinstance(df, pl.LazyFrame) and return_df:
        split_sample = split_sample.collect()
    return split_sample
    
