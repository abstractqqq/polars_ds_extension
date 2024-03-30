import polars as pl
import random
import math
from typing import Union, Optional, List, Tuple
from itertools import combinations, islice


def _sampler_expr(value: Union[float, int], seed: Optional[int] = None) -> pl.Expr:
    if isinstance(value, float):
        if value >= 1.0 or value <= 0.0:
            raise ValueError("Sample rate must be in (0, 1) range.")
        return pl.int_range(0, pl.len()).shuffle(seed) < pl.len() * value
    elif isinstance(value, int):
        if value <= 0:
            raise ValueError("Sample count must be > 0.")
        return pl.int_range(0, pl.len()).shuffle(seed) < value
    elif isinstance(value, pl.Expr):
        return NotImplemented
    else:
        raise ValueError("Sample value must be either int or float.")


def sample(
    df: Union[pl.DataFrame, pl.LazyFrame], value: Union[float, int], seed: Optional[int] = None
) -> pl.DataFrame:
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
    df: Union[pl.DataFrame, pl.LazyFrame],
    condition: pl.Expr,
    target_volume: Optional[int] = None,
    seed: Optional[int] = None,
) -> pl.DataFrame:
    """
    Given the condition, define the # of trues as N and the # of falses as M, then randomly select
    min(M, N, target_volume) rows on the two groups.

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    condition
        A Polars expression represeting a boolean condition
    target_volume
        If none, use min(M, N), this means that one group is always fully selected. If int,
        this will randomly select min(M, N, target_volume) for both groups.
    seed
        A random seed
    """
    trues = condition.sum()
    falses = pl.len() - trues.sum()
    if isinstance(target_volume, int):
        target = pl.min_horizontal(trues, falses, target_volume)
    else:
        target = pl.min_horizontal(trues, falses)

    return (
        df.lazy().filter(pl.int_range(0, pl.len()).shuffle(seed).over(condition) < target).collect()
    )


def down_sample(
    df: Union[pl.DataFrame, pl.LazyFrame],
    conditions: Union[Tuple[pl.Expr, Union[float, int]], List[Tuple[pl.Expr, Union[float, int]]]],
    seed: Optional[int] = None,
) -> pl.DataFrame:
    """
    Downsamples data on the subsets where condition is true.

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    conditions
        Either a Tuple[pl.Expr, float|int] or a list of such tuples. The first entry in the tuple should be a
        boolean expression and the second entry means we sample either n or x% on the part where the boolean is true.
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
    if isinstance(conditions, Tuple):
        all_conds = [conditions]
    else:
        all_conds = conditions

    all_filters = ((_sampler_expr(r, seed).over(c) | (~c)) for c, r in all_conds)
    return df.lazy().filter(pl.lit(True).and_(*all_filters)).collect()


def random_cols(
    df: Union[pl.DataFrame, pl.LazyFrame],
    k: Optional[int] = None,
    keep: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> List[str]:
    """
    Selects random columns in the dataframe. Returns the selected columns in a list. Note, it is
    impossible for this to randomly select both ["x", "y"] and ["y", "x"].

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    k
        Select k random columns from all columns outside of `keep`.
    keep
        Columns to always keep
    seed
        A random seed
    """
    if keep is None:
        out = []
        to_sample = combinations(df.columns, k)
    else:
        out = keep
        to_sample = combinations((c for c in df.columns if c not in keep), k)

    pool_size = len(df.columns) - len(out)
    if pool_size < k:
        raise ValueError("Not enough columns to select from.")

    n = random.randrange(0, math.comb(pool_size, k))
    rand_cols = next(islice(to_sample, n, None), None)
    return out + list(rand_cols)
