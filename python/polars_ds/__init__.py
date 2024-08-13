from __future__ import annotations
import polars as pl
from .type_alias import str_to_expr
# import logging
# logging.basicConfig(level=logging.INFO)

from polars_ds.num import *  # noqa: F403
from polars_ds.metrics import *  # noqa: F403
from polars_ds.stats import *  # noqa: F403
from polars_ds.string import *  # noqa: F403
from polars_ds.features import *  # noqa: F403
from polars_ds.query_knn import *  # noqa: F403
from polars_ds.query_linear import *  # noqa: F403


__version__ = "0.5.3"


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


def eval_series(*series: pl.Series, expr: str, **kwargs) -> pl.DataFrame:
    """
    Evaluates a Polars DS expression on a series.

    Note: currently this doesn't support all Polars DS expressions. E.g. It may not work
    for least square related expressions. It doesn't work for 2D NumPy matrices either, and you
    have to pass column by column if you are using NumPy as input. This is also not tested for
    lower versions of Polars and also not on every expression.

    Parameters
    ----------
    series
        A sequence of series or NumPy arrays
    expr
        The name of the Polars DS expression
    kwargs
        Keyword arguments
    """

    if expr.startswith("_") or expr.endswith("_"):
        raise ValueError("Special underscored functions are not allowed here.")

    inputs = list(pl.lit(pl.Series(name=str(i), values=s)) for i, s in enumerate(series))
    if len(inputs) == 0:
        raise ValueError("This currently doesn't support expressions without a positonal argument.")

    func = globals()[expr]
    return pl.select(func(*inputs, **kwargs).alias(expr.replace("query_", "")))


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
