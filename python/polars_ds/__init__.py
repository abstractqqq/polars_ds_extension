import polars as pl
import logging
from typing import Optional

from polars_ds.num import *  # noqa: F403
from polars_ds.graph import *  # noqa: F403
from polars_ds.metrics import *  # noqa: F403
from polars_ds.complex import ComplexExt  # noqa: E402, F401
from polars_ds.str2 import StrExt  # noqa: E402, F401
from polars_ds.stats import StatsExt  # noqa: E402, F401
from polars_ds.graph import GraphExt  # noqa: E402, F401

logging.basicConfig(level=logging.INFO)

__version__ = "0.3.4"

# __all__ = ["NumExt", "ComplexExt", "StrExt", "StatsExt", "MetricExt", "GraphExt"]


def l_inf_horizontal(*v: pl.Expr, normalize: bool = False) -> pl.Expr:
    """
    Horizontally L inf norm. Shorthand for pl.max_horizontal(pl.col(x).abs() for x in exprs).

    Parameters
    ----------
    *v
        Expressions to compute horizontal L infinity.
    normalize
        Whether to divide by the dimension
    """
    if normalize:
        exprs = list(v)
        return pl.max_horizontal(pl.col(x).abs() for x in exprs) / len(exprs)
    else:
        return pl.max_horizontal(pl.col(x).abs() for x in v)


def l2_sq_horizontal(*v: pl.Expr, normalize: bool = False) -> pl.Expr:
    """
    Horizontally computes L2 norm squared. Shorthand for pl.sum_horizontal(pl.col(x).pow(2) for x in exprs).

    Parameters
    ----------
    *v
        Expressions to compute horizontal L2.
    normalize
        Whether to divide by the dimension
    """
    if normalize:
        exprs = list(v)
        return pl.sum_horizontal(pl.col(x).pow(2) for x in exprs) / len(exprs)
    else:
        return pl.sum_horizontal(pl.col(x).pow(2) for x in v)


def l1_horizontal(*v: pl.Expr, normalize: bool = False) -> pl.Expr:
    """
    Horizontally computes L1 norm. Shorthand for pl.sum_horizontal(pl.col(x).abs() for x in exprs).

    Parameters
    ----------
    *v
        Expressions to compute horizontal L1.
    normalize
        Whether to divide by the dimension
    """
    if normalize:
        exprs = list(v)
        return pl.sum_horizontal(pl.col(x).abs() for x in exprs) / len(exprs)
    else:
        return pl.sum_horizontal(pl.col(x).abs() for x in v)


def random_data(
    size: int = 2_000, n_cols: int = 3, null_pct: Optional[float] = None
) -> pl.DataFrame:
    """
    Generates a random eager Polars Dataframe with 1 column as row_num, and `n_cols` columns
    random features. Random features will be uniformly generated.

    Parameters
    ----------
    size
        The total number of rows in this dataframe
    n_cols
        The total number of uniformly (range = [0, 1)) generated features
    null_pct
        If none, no null values will be present. If it is a float, then each feature column
        will have this much percentage of nulls.
    """
    if null_pct is None:
        rand_cols = (
            pl.col("row_num").stats.sample_uniform(low=0.0, high=1.0).alias(f"feature_{i+1}")
            for i in range(n_cols)
        )
    else:
        rand_cols = (
            pl.col("row_num")
            .stats.sample_uniform(low=0.0, high=1.0)
            .stats.rand_null(null_pct)
            .alias(f"feature_{i+1}")
            for i in range(n_cols)
        )

    return pl.DataFrame(
        {
            "row_num": pl.Series(values=range(size), dtype=pl.UInt64),
        }
    ).with_columns(*rand_cols)
