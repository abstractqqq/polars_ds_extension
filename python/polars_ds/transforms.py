"""
This module provides classic ML dataset transforms. Note all functions here are single-use only, meaning
that the data learned (e.g. mean value in mean imputation) will not be preserved. For pipeline usage, which
preserves the learned values and optimizes the transform query, see pipeline.py.
"""

import polars as pl
import polars.selectors as cs
from .type_alias import (
    PolarsFrame,
    SimpleImputeMethod,
    SimpleScaleMethod,
    ExprTransform,
    StrOrExpr,
    QuantileMethod,
)
from . import num as pds_num
from ._utils import _IS_POLARS_V1
from typing import List, Union, Optional


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


def impute_nan(
    df: PolarsFrame, cols: List[str], method: SimpleImputeMethod = "mean"
) -> ExprTransform:
    """
    Impute NaN values in the given columns. NaN is not the same as null in Polars. In most Polars dataframes,
    NaN should occur only because of numerical problems, such as log(-1). This transformation
    also only applies to float columns and non-float columns will be ignored despite being passed in cols.

    This transform will collect if input is lazy.

    Parameters
    ----------
    df
        Either a lazy or an eager dataframe
    cols
        A list of strings representing column names
    method
        One of `mean` or `median`. `mode` will result in error.
    """
    if method == "mean":
        temp = df.lazy().select(cols).select(cs.float().mean()).collect()
        return [pl.col(c).fill_nan(m) for c, m in zip(temp.columns, temp.row(0))]
    elif method == "median":
        temp = df.lazy().select(cols).select(cs.float().median()).collect()
        return [pl.col(c).fill_nan(m) for c, m in zip(temp.columns, temp.row(0))]
    else:
        raise ValueError(f"Unknown input method: {method}")


def linear_impute(
    df: PolarsFrame, features: List[str], target: Union[str, pl.Expr], add_bias: bool = False
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
    temp = (
        df.lazy()
        .select(pds_num.query_lstsq(*features, target=target, add_bias=add_bias, skip_null=True))
        .collect()
    )  # Add streaming config
    coeffs = temp.item(0, 0)
    linear_eq = [pl.col(f) * coeffs[i] for i, f in enumerate(features)]
    if add_bias:
        linear_eq.append(pl.lit(coeffs[-1], dtype=pl.Float64))

    return [
        pl.when(pl.col(target_name).is_null())
        .then(pl.sum_horizontal(linear_eq))
        .otherwise(pl.col(target_name).cast(pl.Float64))
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
    consider nulls as one of the unique values.

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
            exprs.extend(
                pl.col(t.name)
                .eq(u[i])
                .fill_null(False)  # In the EQ comparison, None will result in None
                .cast(pl.UInt8)
                .alias(t.name + separator + u[i])
                for i in range(int(drop_first), len(u))
            )

    return exprs


def target_encode(
    df: PolarsFrame,
    cols: List[str],
    /,
    target: Union[StrOrExpr, pl.Series],
    min_samples_leaf: int = 20,
    smoothing: float = 10.0,
    default: Optional[float] = None,
) -> ExprTransform:
    """
    Target encode the given variables. This will overwrite the columns that will be encoded.

    Note: nulls will be encoded as well.

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
        If new value is encountered during transform, it will be mapped to default

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
                old=c[0].struct.field("value"), new=c[0].struct.field("to"), default=default
            )
            for c in temp.get_columns()
        ]
    else:
        exprs = [
            # c[0] will be a series of struct because of the implode above.
            pl.col(c.name).replace(
                old=c[0].struct.field("value"), new=c[0].struct.field("to"), default=default
            )
            for c in temp.get_columns()
        ]
    return exprs


def woe_encode(
    df: PolarsFrame,
    cols: List[str],
    /,
    target: Union[StrOrExpr, pl.Series],
    default: Optional[float] = None,
) -> ExprTransform:
    """
    Use Weight of Evidence to encode a discrete variable x with respect to target. This assumes x
    is discrete and castable to String. A value of 1 is added to all events/non-events
    (goods/bads) to smooth the computation. This is -1 * output of the package category_encoder's WOEEncoder.

    Note: nulls will be encoded as well.

    Parameters
    ----------
    df
        Either a lazy or an eager dataframe
    cols
        A list of strings representing column names. Columns of type != string/categorical will not produce any expression.
    target
        The target column
    default
        If new value is encountered during transform, it will be mapped to default

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

    temp = temp.select(
        pds_num.query_woe_discrete(c, target).implode() for c in valid_cols
    ).collect()  # add collect config..
    # POLARS_V1
    if _IS_POLARS_V1:
        exprs = [
            # c[0] will be a series of struct because of the implode above.
            pl.col(c.name).replace_strict(
                old=c[0].struct.field("value"), new=c[0].struct.field("woe"), default=default
            )
            for c in temp.get_columns()
        ]
    else:
        exprs = [
            # c[0] will be a series of struct because of the implode above.
            pl.col(c.name).replace(
                old=c[0].struct.field("value"), new=c[0].struct.field("woe"), default=default
            )
            for c in temp.get_columns()
        ]

    return exprs


def iv_encode(
    df: PolarsFrame,
    cols: List[str],
    /,
    target: Union[StrOrExpr, pl.Series],
    default: Optional[float] = None,
) -> ExprTransform:
    """
    Use Information Value to encode a discrete variable x with respect to target. This assumes x
    is discrete and castable to String. A value of 1 is added to all events/non-events
    (goods/bads) to smooth the computation.

    Note: nulls will be encoded as well.

    Parameters
    ----------
    df
        Either a lazy or an eager dataframe
    cols
        A list of strings representing column names. Columns of type != string/categorical will not produce any expression.
    target
        The target column
    default
        If new value is encountered during transform, it will be mapped to default

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
    temp = temp.select(
        pds_num.query_iv_discrete(c, target, return_sum=False).implode() for c in valid_cols
    ).collect()  # add collect config..
    # POLARS_V1
    if _IS_POLARS_V1:
        exprs = [
            # c[0] will be a series of struct because of the implode above.
            pl.col(c.name).replace_strict(
                old=c[0].struct.field("value"), new=c[0].struct.field("iv"), default=default
            )
            for c in temp.get_columns()
        ]
    else:
        exprs = [
            # c[0] will be a series of struct because of the implode above.
            pl.col(c.name).replace(
                old=c[0].struct.field("value"), new=c[0].struct.field("iv"), default=default
            )
            for c in temp.get_columns()
        ]
    return exprs
