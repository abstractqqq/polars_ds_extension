from __future__ import annotations
from typing import Literal, Union
import sys
import polars as pl

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:  # 3.9, 3.8
    from typing_extensions import TypeAlias

# Custom Enum Types
DetrendMethod: TypeAlias = Literal["linear", "mean"]
Alternative: TypeAlias = Literal["two-sided", "less", "greater"]
ROCAUCStrategy: TypeAlias = Literal["macro", "weighted"]
Distance: TypeAlias = Literal["l1", "l2", "inf", "h", "cosine", "haversine"]
ConvMode: TypeAlias = Literal["same", "left", "right", "full", "valid"]

# Other Custom Types
PolarsFrame: TypeAlias = Union[pl.DataFrame, pl.LazyFrame]


# Auxiliary functions for type conversions
def str_to_expr(e: Union[pl.Expr, str]) -> pl.Expr:
    """
    Turns a string into an expression

    Parameters
    ----------
    e
        Either a str represeting a column name or an expression
    """
    if isinstance(e, pl.Expr):
        return e
    elif isinstance(e, str):
        return pl.col(e)
    else:
        raise ValueError("Input must either be a string or a Polars expression.")
