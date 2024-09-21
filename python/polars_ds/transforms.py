"""
This module provides classic ML dataset transforms. Note all functions here are single-use only, meaning
that the data learned (e.g. mean value in mean imputation) will not be preserved. For pipeline usage, which
preserves the learned values and optimizes the transform query, see pipeline.py.
"""
from __future__ import annotations

import polars as pl
import polars.selectors as cs
from .type_alias import (
    PolarsFrame,
    SimpleImputeMethod,
    SimpleScaleMethod,
    ExprTransform,
    QuantileMethod,
    EncoderDefaultStrategy,
)
from . import num as pds_num
from . import query_linear as pds_linear
from ._utils import _IS_POLARS_V1
from typing import List


def impute(df: PolarsFrame, cols: List[str], method: SimpleImputeMethod = "mean") -> ExprTransform:
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


def linear_impute(
    df: PolarsFrame, features: List[str], target: str | pl.Expr, add_bias: bool = False
) -> ExprTransform:
    """
    Imputes the target column by training a simple linear regression using the other features. This will
    cast the target column to f64.

    Note: The linear regression will skip nulls whenever there is a null in the features or in the target.
    Additionally, if NaN or Inf exists in data, the linear regression result may be invalid or an error
    will be thrown. It is recommended to use this only after imputing and dealing with NaN and Infs for
    all feature columns first.

    Parameters
    ----------
    df
        Either a lazy or an eager dataframe
    features
        A list of strings representing column names that will be used as features in the linear regression
    target
        The target column
    add_bias
        Whether to add a bias term to the linear regression
    """
    if _IS_POLARS_V1:
        target_name = df.lazy().select(target).collect_schema().names()[0]
    else:
        target_name = df.select(target).columns[0]

    features_as_expr = [pl.col(f) for f in features]
    target_as_expr = pl.col(target_name)
    temp = (
        df.lazy()
        .select(
            pds_linear.query_lstsq(
                *features_as_expr, target=target_as_expr, add_bias=add_bias, null_policy="skip"
            )
        )
        .collect()
    )  # Add streaming config
    coeffs = temp.item(0, 0)
    linear_eq = [f * coeffs[i] for i, f in enumerate(features_as_expr)]
    if add_bias:
        linear_eq.append(pl.lit(coeffs[-1], dtype=pl.Float64))

    return [
        pl.when(target_as_expr.is_null())
        .then(pl.sum_horizontal(linear_eq))
        .otherwise(target_as_expr.cast(pl.Float64))
        .alias(target_name)
    ]


def center(df: PolarsFrame, cols: List[str]) -> ExprTransform:
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
) -> ExprTransform:
    """
    Scales values in the given columns. This transform will collect if input is lazy.

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
                pl.col(cols).std(ddof=0).name.prefix("std:"),
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
        # If input is constant, this will return all NaNs.
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
    df: PolarsFrame,
    cols: List[str],
    q1: float = 0.25,
    q2: float = 0.75,
    method: QuantileMethod = "midpoint",
) -> ExprTransform:
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
    method
        Method to compute quantile. One of `nearest`, `higher`, `lower`, `midpoint`, `linear`.
    """
    if q1 > 1.0 or q1 < 0.0 or q2 > 1.0 or q2 < 0.0 or q1 >= q2:
        raise ValueError("Input `q1` and `q2` must be between 0 and 1 and q1 must be < than q2.")

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


def winsorize(
    df: PolarsFrame,
    cols: List[str],
    lower: float = 0.05,
    upper: float = 0.95,
    method: QuantileMethod = "nearest",
) -> ExprTransform:
    """
    Learns the lower and upper percentile from the columns, then clip each end at those values.

    Parameters
    ----------
    df
        Either a lazy or an eager dataframe
    cols
        A list of strings representing column names
    lower
        The lower quantile value
    upper
        The higher quantile value
    method
        Method to compute quantile. One of `nearest`, `higher`, `lower`, `midpoint`, `linear`.
    """
    if lower >= 1.0 or lower <= 0.0 or upper >= 1.0 or upper <= 0.0 or lower >= upper:
        raise ValueError("Lower and upper must be with in (0, 1) and upper should be > lower")

    temp = (
        df.lazy()
        .select(
            pl.col(cols).quantile(lower).name.prefix("l"),
            pl.col(cols).quantile(upper).name.prefix("u"),
        )
        .collect()
        .row(0)
    )
    n = len(cols)
    return [pl.col(c).clip(temp[i], temp[n + i]) for i, c in enumerate(cols)]


def one_hot_encode(
    df: PolarsFrame, cols: List[str], separator: str = "_", drop_first: bool = False
) -> ExprTransform:
    """
    Find the unique values in the string/categorical columns and one-hot encode them. This will NOT
    consider nulls as one of the unique values. Append a one-hot null indicator if you want to encode nulls.

    Parameters
    ----------
    df
        Either a lazy or an eager dataframe
    cols
        A list of strings representing column names. Columns of type != string/categorical will not produce any expression.
    separator
        E.g. if column name is `col` and `a` is an elemenet in it, then the one-hot encoded column will be called
        `col_a` where the separator `_` is used.
    drop_first
        Whether to drop the first distinct value (in terms of str/categorical order). This helps with reducing
        dimension and prevents some issues from linear dependency.
    """

    temp = (
        df.lazy()
        .select(cols)
        .select(
            (cs.string() | cs.categorical())
            .unique()
            .drop_nulls()
            .cast(pl.String)
            .implode()
            .list.sort()
        )
    )
    exprs: list[pl.Expr] = []
    for t in temp.collect().get_columns():
        u: pl.Series = t[0]  # t is a Series which contains a single series, so u is a series
        if len(u) > 1:
            # Need to take care of the case where null == 1 is null. Need only True and False, not null
            exprs.extend(
                pl.col(t.name).eq_missing(u[i]).cast(pl.UInt8).alias(t.name + separator + u[i])
                for i in range(int(drop_first), len(u))
            )

    if len(exprs) == 0:
        raise ValueError(
            "Provided columns either do not exist or are not string/categorical types."
        )

    return exprs


def rank_hot_encode(
    col: str,
    ranking: List[str],
) -> ExprTransform:
    """
    Given a ranking, e.g. ["bad", "neutral", "good"], where "bad", "neutral" and "good" are values coming
    from the column `col`, this will create 2 additional columns, where a row of [0, 0] will represent
    "bad", and a row of [1, 0] will represent "neutral", and a row of [1,1] will represent "good". The meaning
    of each rows is that the value is at least this rank. This currently only works on string columns.

    Values not in the provided ranking will have -1 in all the new columns.

    Parameters
    ----------
    col
        The name of a single column
    ranking
        A list of string representing the ranking of the values
    """

    n_ranks = len(ranking)
    if n_ranks <= 1:
        raise ValueError("Rank hot encoding does not work with single value ranking.")

    if not _IS_POLARS_V1:
        raise ValueError("Unavailable for Polars < v1.")

    number_rank = list(range(n_ranks))
    ranked_expr = pl.col(col).replace_strict(
        old=ranking, new=number_rank, default=None, return_dtype=pl.Int32
    )
    return [
        (ranked_expr >= i).cast(pl.Int8).fill_null(-1).alias(f"{col}>={c}")
        for i, c in zip(range(1, n_ranks), ranking[1:])
    ]


def _encoder_default_value(
    temp: PolarsFrame,
    default: EncoderDefaultStrategy | float | None,
    target: str | pl.Expr | pl.Series,
) -> float | None:
    """
    Finds the default value for encoders (Target, WOE, IV encoders) for null and unknown values.
    """
    if default is None or isinstance(default, (int, float)):
        return default
    elif isinstance(default, str):
        if default == "null":
            return None
        elif default == "zero":
            return 0.0
        elif default == "mean":
            if isinstance(target, str):
                return temp.lazy().select(pl.col(target).mean()).collect().item(0, 0)
            elif isinstance(target, pl.Expr):
                return temp.lazy().select(target.mean()).collect().item(0, 0)
            elif isinstance(target, pl.Series):
                return target.mean()
            else:
                raise ValueError("Target's type is not supported.")
        else:
            raise ValueError(
                "When input `default` is string, it can only be `mean` or `null` or `zero`."
            )
    else:
        raise ValueError("Invalid type for `default`")


def target_encode(
    df: PolarsFrame,
    cols: List[str],
    /,
    target: str | pl.Expr | pl.Series,
    min_samples_leaf: int = 20,
    smoothing: float = 10.0,
    default: EncoderDefaultStrategy | float | None = "null",
) -> ExprTransform:
    """
    Target encode the given variables. This will overwrite the columns that will be encoded.

    Note: Nulls will always be mapped to the default.

    Parameters
    ----------
    df
        Either a lazy or an eager dataframe
    cols
        A list of strings representing column names. Columns of type != string/categorical will not produce any expression.
    target
        The target column
    min_samples_leaf
        A regularization factor
    smoothing
        Smoothing effect to balance categorical average vs prior
    default
        If a new value is encountered during transform (unseen in training dataset), it will be mapped to default.
        If this is a string, it can be `null`, `zero`, or `mean`, where `mean` means map them to the mean of the target.

    Reference
    ---------
    https://contrib.scikit-learn.org/category_encoders/targetencoder.html
    """
    temp = df.lazy()
    if _IS_POLARS_V1:
        valid_cols = (
            temp.select(cols).select(cs.string() | cs.categorical()).collect_schema().names()
        )

    else:
        valid_cols = temp.select(cols).select(cs.string() | cs.categorical()).columns

    if len(valid_cols) == 0:
        raise ValueError(
            "The provided columns are either not string/categorical type, or are not in df."
        )

    default_value = _encoder_default_value(temp, default=default, target=target)

    temp = temp.select(
        pds_num.target_encode(
            c, target, min_samples_leaf=min_samples_leaf, smoothing=smoothing
        ).implode()
        for c in valid_cols
    ).collect()  # add collect config..
    # POLARS_V1
    if _IS_POLARS_V1:
        exprs = [
            # c[0] will be a series of struct because of the implode above.
            pl.col(c.name).replace_strict(
                old=c[0].struct.field("value"), new=c[0].struct.field("to"), default=default_value
            )
            for c in temp.get_columns()
        ]
    else:
        exprs = [
            # c[0] will be a series of struct because of the implode above.
            pl.col(c.name).replace(
                old=c[0].struct.field("value"), new=c[0].struct.field("to"), default=default_value
            )
            for c in temp.get_columns()
        ]
    return exprs


def woe_encode(
    df: PolarsFrame,
    cols: List[str],
    /,
    target: str | pl.Expr | pl.Series,
    default: EncoderDefaultStrategy | float | None = "null",
) -> ExprTransform:
    """
    Use Weight of Evidence to encode a discrete variable x with respect to target. This assumes x
    is discrete and castable to String. A value of 1 is added to all events/non-events
    (goods/bads) to smooth the computation. This is -1 * output of the package category_encoder's WOEEncoder.

    Note: Nulls will always be mapped to the default.

    Parameters
    ----------
    df
        Either a lazy or an eager dataframe
    cols
        A list of strings representing column names. Columns of type != string/categorical will not produce any expression.
    target
        The target column
    default
        If a new value is encountered during transform (unseen in training dataset), it will be mapped to default.
        If this is a string, it can be `null`, `zero`, or `mean`, where `mean` means map them to the mean of the target.

    Reference
    ---------
    https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html
    """
    temp = df.lazy()
    if _IS_POLARS_V1:
        valid_cols = (
            temp.select(cols).select(cs.string() | cs.categorical()).collect_schema().names()
        )
    else:
        valid_cols = temp.select(cols).select(cs.string() | cs.categorical()).columns

    if len(valid_cols) == 0:
        raise ValueError(
            "The provided columns are either not string/categorical type, or are not in df."
        )

    default_value = _encoder_default_value(temp, default=default, target=target)

    temp = temp.select(
        pds_num.query_woe_discrete(c, target).implode() for c in valid_cols
    ).collect()  # add collect config..
    # POLARS_V1
    if _IS_POLARS_V1:
        exprs = [
            # c[0] will be a series of struct because of the implode above.
            pl.col(c.name).replace_strict(
                old=c[0].struct.field("value"), new=c[0].struct.field("woe"), default=default_value
            )
            for c in temp.get_columns()
        ]
    else:
        exprs = [
            # c[0] will be a series of struct because of the implode above.
            pl.col(c.name).replace(
                old=c[0].struct.field("value"), new=c[0].struct.field("woe"), default=default_value
            )
            for c in temp.get_columns()
        ]

    return exprs


def iv_encode(
    df: PolarsFrame,
    cols: List[str],
    /,
    target: str | pl.Expr | pl.Series,
    default: EncoderDefaultStrategy | float | None = "null",
) -> ExprTransform:
    """
    Use Information Value to encode a discrete variable x with respect to target. This assumes x
    is discrete and castable to String. A value of 1 is added to all events/non-events
    (goods/bads) to smooth the computation.

    Note: Nulls will always be mapped to the default.

    Parameters
    ----------
    df
        Either a lazy or an eager dataframe
    cols
        A list of strings representing column names. Columns of type != string/categorical will not produce any expression.
    target
        The target column
    default
        If a new value is encountered during transform (unseen in training dataset), it will be mapped to default.
        If this is a string, it can be `null`, `zero`, or `mean`, where `mean` means map them to the mean of the target.

    Reference
    ---------
    https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html
    """
    temp = df.lazy()
    if _IS_POLARS_V1:
        valid_cols = (
            temp.select(cols).select(cs.string() | cs.categorical()).collect_schema().names()
        )
    else:
        valid_cols = temp.select(cols).select(cs.string() | cs.categorical()).columns

    if len(valid_cols) == 0:
        raise ValueError(
            "The provided columns are either not string/categorical type, or are not in df."
        )

    default_value = _encoder_default_value(temp, default=default, target=target)

    temp = temp.select(
        pds_num.query_iv_discrete(c, target, return_sum=False).implode() for c in valid_cols
    ).collect()  # add collect config..
    # POLARS_V1
    if _IS_POLARS_V1:
        exprs = [
            # c[0] will be a series of struct because of the implode above.
            pl.col(c.name).replace_strict(
                old=c[0].struct.field("value"), new=c[0].struct.field("iv"), default=default_value
            )
            for c in temp.get_columns()
        ]
    else:
        exprs = [
            # c[0] will be a series of struct because of the implode above.
            pl.col(c.name).replace(
                old=c[0].struct.field("value"), new=c[0].struct.field("iv"), default=default_value
            )
            for c in temp.get_columns()
        ]
    return exprs


def polynomial_features(
    cols: List[str],
    /,
    degree: int,
    interaction_only: bool = False,
) -> ExprTransform:
    """
    Generates polynomial combinations out of the features given, at the given degree.

    Parameters
    ----------
    cols
        A list of strings representing column names.
    degree
        The degree of the polynomial combination
    interaction_only
        It true, only combinations that involve 2 or more variables will be used.
    """
    from itertools import combinations_with_replacement

    if degree <= 1:
        raise ValueError("Degree should be > 1.")

    return list(
        pl.reduce(function=lambda acc, x: acc * x, exprs=list(comb)).alias("*".join(comb))
        for comb in combinations_with_replacement(cols, degree)
        if ((not interaction_only) or len(set(comb)) > 1)
    )
