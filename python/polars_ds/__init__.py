from __future__ import annotations
import polars as pl
from ._utils import str_to_expr

from polars_ds.num import *  # noqa: F403
from polars_ds.metrics import *  # noqa: F403
from polars_ds.stats import *  # noqa: F403
from polars_ds.string import *  # noqa: F403
from polars_ds.ts_features import *  # noqa: F403
from polars_ds.expr_knn import *  # noqa: F403
from polars_ds.expr_linear import *  # noqa: F403

__version__ = "0.6.3"


def l_inf_horizontal(*v: str | pl.Expr, normalize: bool = False) -> pl.Expr:
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
        return pl.max_horizontal(str_to_expr(x).abs() for x in exprs) / len(exprs)
    else:
        return pl.max_horizontal(str_to_expr(x).abs() for x in v)


def l2_sq_horizontal(*v: str | pl.Expr, normalize: bool = False) -> pl.Expr:
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
        return pl.sum_horizontal(str_to_expr(x).pow(2) for x in exprs) / len(exprs)
    else:
        return pl.sum_horizontal(str_to_expr(x).pow(2) for x in v)


def l1_horizontal(*v: str | pl.Expr, normalize: bool = False) -> pl.Expr:
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
        return pl.sum_horizontal(str_to_expr(x).abs() for x in exprs) / len(exprs)
    else:
        return pl.sum_horizontal(str_to_expr(x).abs() for x in v)

def frame(size: int = 2_000, index_name: str = "row_num") -> pl.DataFrame:
    """
    Generates a frame with only an index (row number) column.
    This is a convenience function to be chained with pds.random(...) when running simulations and tests.

    Parameters
    ----------
    size
        The total number of rows in this dataframe
    index_name
        The name of the index column
    """
    return pl.DataFrame({index_name: range(size)})
