from __future__ import annotations
from typing import Literal, Union, List, Callable
import sys
import polars as pl

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:  # 3.9, 3.8
    from typing_extensions import TypeAlias

# Custom "Enum" Types
DetrendMethod: TypeAlias = Literal["linear", "mean"]
Alternative: TypeAlias = Literal["two-sided", "less", "greater"]
ROCAUCStrategy: TypeAlias = Literal["macro", "weighted"]
Distance: TypeAlias = Literal["l1", "l2", "inf", "h", "cosine", "haversine"]
ConvMode: TypeAlias = Literal["same", "left", "right", "full", "valid"]
ConvMethod: TypeAlias = Literal["fft", "direct"]
CorrMethod: TypeAlias = Literal["pearson", "spearman", "xi", "kendall"]
SimpleImputeMethod: TypeAlias = Literal["mean", "median", "mode"]
SimpleScaleMethod: TypeAlias = Literal["min_max", "standard", "abs_max"]

# Other Custom Types
PolarsFrame: TypeAlias = Union[pl.DataFrame, pl.LazyFrame]
StrOrExpr: TypeAlias = Union[str, pl.Expr]
ExprTransform: TypeAlias = Union[pl.Expr, List[pl.Expr]]
FitTransformFunc: TypeAlias = Callable[[PolarsFrame, List[str]], ExprTransform]


# Auxiliary functions for type conversions
def str_to_expr(e: StrOrExpr) -> pl.Expr:
    """
    Turns a string into an expression

    Parameters
    ----------
    e
        Either a str represeting a column name or an expression
    """
    if isinstance(e, str):
        return pl.col(e)
    elif isinstance(e, pl.Expr):
        return e
    else:
        raise ValueError("Input must either be a string or a Polars expression.")
