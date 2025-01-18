"""Not meant for outside use."""

from __future__ import annotations

import polars as pl
from typing import Sequence
from pathlib import Path
from polars.plugins import register_plugin_function

# Only need this
_PLUGIN_PATH = Path(__file__).parent

# V1.18 Introduces a Int128 dtype
# _IS_POLARS_V1_18 = pl.__version__.startswith("1.18.")


def pl_plugin(
    *,
    symbol: str,
    args: Sequence[pl.Series | pl.Expr],
    kwargs: dict[str, str | int | float | bool] | None = None,
    is_elementwise: bool = False,
    returns_scalar: bool = False,
    changes_length: bool = False,
    cast_to_supertype: bool = False,
    pass_name_to_apply: bool = False,
) -> pl.Expr:
    return register_plugin_function(
        plugin_path=_PLUGIN_PATH,
        args=args,
        function_name=symbol,
        kwargs=kwargs,
        is_elementwise=is_elementwise,
        returns_scalar=returns_scalar,
        changes_length=changes_length,
        cast_to_supertype=cast_to_supertype,
        pass_name_to_apply=pass_name_to_apply,
    )


# Auxiliary functions for type conversions
def str_to_expr(e: str | pl.Expr | int | float) -> pl.Expr:
    """
    Turns a string into an expression

    Parameters
    ----------
    e
        Either a str represeting a column name or an expression
    """
    if isinstance(e, str):
        return pl.col(e)
    elif isinstance(e, (int, float)):
        return pl.lit(e)
    elif isinstance(e, pl.Expr):
        return e
    else:
        raise ValueError("Input must either be a string or a Polars expression.")
