import polars as pl
from .type_alias import PolarsFrame, SimpleImputeMethod, SimpleScaleMethod
from typing import List


def impute(df: PolarsFrame, cols: List[str], method: SimpleImputeMethod = "mean") -> List[pl.Expr]:
    """
    Impute null values in the given columns. This transform will collect if input is lazy.

    Parameters
    ----------
    df
        Either a lazy or an eager dataframe
    cols
        A list of strings representing column names
    method
        One of `mean`, `median`, `mode`. If `mode`, a random value will be chosen if there is
        a tie.
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
    else:
        raise ValueError(f"Unknown input method: {method}")


def center(df: PolarsFrame, cols: List[str]) -> List[pl.Expr]:
    """
    Center the given columns so that they will have 0 mean.

    Parameters
    ----------
    df
        Either a lazy or an eager dataframe
    cols
        A list of strings representing column names
    """
    means = df.lazy().select(pl.col(cols).mean()).collect().row(0)
    return [pl.col(c) - m for c, m in zip(cols, means)]


def scale(
    df: PolarsFrame,
    cols: List[str],
    method: SimpleScaleMethod = "standard",
) -> List[pl.Expr]:
    """
    Impute null values in the given columns. This transform will collect if input is lazy.

    Parameters
    ----------
    df
        Either a lazy or an eager dataframe
    cols
        A list of strings representing column names
    method
        One of `standard`, `min_max`, `abs_max`
    """
    if method == "standard":
        temp = (
            df.lazy()
            .select(
                pl.col(cols).mean().name.prefix("mean:"),
                pl.col(cols).mean().name.prefix("std:"),
            )
            .collect()
            .row(0)
        )
        n = len(cols)
        return [(pl.col(c) - temp[i]) / temp[i + n] for i, c in enumerate(cols)]
    elif method == "min_max":
        temp = (
            df.lazy()
            .select(
                pl.col(cols).min().name.prefix("min:"),
                pl.col(cols).max().name.prefix("max:"),
            )
            .collect()
            .row(0)
        )
        n = len(cols)
        return [(pl.col(c) - temp[i]) / (temp[n + i] - temp[i]) for i, c in enumerate(cols)]
    elif method == "abs_max":
        temp = (
            df.lazy()
            .select(pl.max_horizontal(pl.col(c).min().abs(), pl.col(c).max().abs()) for c in cols)
            .collect()
            .row(0)
        )
        return [pl.col(c) / m for c, m in zip(cols, temp)]
    else:
        raise ValueError(f"Unknown input method: {method}")


def robust_scale(
    df: PolarsFrame, cols: List[str], q1: float = 0.25, q2: float = 0.75
) -> List[pl.Expr]:
    """
    Like min-max scaling, but scales each column by the quantile value at q1 and q2.

    Parameters
    ----------
    df
        Either a lazy or an eager dataframe
    cols
        A list of strings representing column names
    q1
        The lower quantile value
    q2
        The higher quantile value
    """
    if q1 > 1.0 or q1 < 0.0 or q2 > 1.0 or q2 < 0.0 or q1 > q2:
        raise ValueError(
            "Input `q1` and `q2` must be between 0 and 1 and q1 must be smaller than q2."
        )

    temp = (
        df.lazy()
        .select(
            pl.col(cols).quantile(q1).name.prefix("q1:"),
            pl.col(cols).quantile(q2).name.prefix("q2:"),
        )
        .collect()
        .row(0)
    )
    n = len(cols)
    return [(pl.col(c) - temp[i]) / (temp[n + i] - temp[i]) for i, c in enumerate(cols)]
