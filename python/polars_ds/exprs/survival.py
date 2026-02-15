from __future__ import annotations
import polars as pl
import warnings

# Internal dependencies
from polars_ds._utils import pl_plugin, to_expr

__all__ = ["query_kaplan_meier_prob"]


def query_kaplan_meier_prob(status: str | pl.Expr, time_exit: str | pl.Expr) -> pl.Expr:
    """
    Computes probabilities given by the Kaplan Meier estimator. This returns a time column and the corresponding probabilities.

    Parameters
    ----------
    status
        Status column. Can be booleans or 0s and 1s. True or 1 indicates an event and False or 0 indicates right-censoring.
    time_exit
        Time of event or censoring.
    """
    warnings.warn(
        "This function's API is considered unstable and might have breaking changes in the future.",
        FutureWarning,
        stacklevel=2,
    )

    return pl_plugin(
        symbol="pl_kaplan_meier",
        args=[to_expr(status).cast(pl.UInt32), to_expr(time_exit)],
    )
