"""Iteration related helper expressions"""

from __future__ import annotations

import polars as pl

# Internal dependencies
from polars_ds._utils import pl_plugin, to_expr

__all__ = ["smooth_spline"]


def smooth_spline(x: str | pl.Expr, y: str | pl.Expr, lambda_: float) -> pl.Expr:
    """
    Fits a smoothing cubic spline f and returns f(x). The user must make sure
    that x is sorted and strictly increasing.

    For more details, see the maths/ folder in the repo.

    Parameters
    ----------
    x
        The x values of the points
    y
        The y values of the points
    lambda_
        The regularization factor. The larger, the smoother and less kinks
        the curve will have.
    """
    if lambda_ < 0.0:
        raise ValueError("Input `lambda_` must be nonnegative.")

    xx, yy = to_expr(x), to_expr(y)
    return pl_plugin(
        symbol="pl_smooth_spline",
        args=[xx.cast(pl.Float64).rechunk(), yy.cast(pl.Float64).rechunk()],
        kwargs={"lambda": lambda_},
    )
