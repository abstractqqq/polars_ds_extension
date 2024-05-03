import polars as pl
from .type_alias import PolarsFrame, SimpleImputeMethod
from typing import List


def impute(
    df: PolarsFrame, cols: List[str], method: SimpleImputeMethod = "mean", const: float = 0.0
) -> List[pl.Expr]:
    """
    Create a missing indicator for the columns. This transform will collect if input is lazy.

    Parameters
    ----------
    df
        Either a lazy or an eager dataframe
    cols
        List of strings of column names.
    method
        One of `mean`, `median`, `mode`, `const`. If `mode`, a random value will be chosen if there is
        a tie and it may not be . If `const`, the value passed in const will be used.
    """
    if method == "mean":
        temp = df.lazy().select(pl.col(cols).mean()).collect().row(0)
        return [pl.col(c).fill_null(m) for c, m in zip(cols, temp)]
    elif method == "median":
        temp = df.lazy().select(pl.col(cols).median()).collect().row(0)
        return [pl.col(c).fill_null(m) for c, m in zip(cols, temp)]
    elif method == "mode":
        temp = df.lazy().select(pl.col(cols).mode().list.first()).collect().row(0)
        return [pl.col(c).fill_null(m) for c, m in zip(cols, temp)]
    elif method == "const":
        return [pl.col(c).fill_null(const) for c in cols]
    else:
        raise ValueError(f"Unknown input method: {method}")
