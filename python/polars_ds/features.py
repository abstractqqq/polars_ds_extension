"""
Common feature engineering queries and time series features as Polars queries. 
"""
from __future__ import annotations

import math
import polars as pl
from .type_alias import str_to_expr, Distance
from ._utils import pl_plugin
from typing import Iterable, Literal

__all__ = [
    "query_abs_energy",
    "symmetry_ratio",
    "query_mean_abs_change",
    "query_cv",
    "query_mean_n_abs_max",
    "query_longest_streak",
    "query_avg_streak",
    "query_streak",
    "query_count_uniques",
    "query_range_count",
    "query_lempel_ziv",
    "query_first_digit_cnt",
    "query_c3_stats",
    "query_cid_ce",
    "query_time_reversal_asymmetry_stats",
    "query_entropy",
    "query_approx_entropy",
    "query_sample_entropy",
    "query_knn_entropy",
    "query_approx_entropy",
    "query_cond_entropy",
    "query_copula_entropy",
    "query_cond_indep",
    "query_transfer_entropy",
    "query_permute_entropy",
    "query_similar_count",
]

#################################################
# Time series features, some from tsfresh       #
#################################################

# index_mass_quantile


def symmetry_ratio(x: str | pl.Expr) -> pl.Expr:
    """
    Returns |mean - median| / (max - min). Note the closer to 0 this value is, the more symmetric
    the series is.
    """
    y = str_to_expr(x)
    return (y.mean() - y.median()).abs() / (y.max() - y.min())


def query_abs_energy(x: str | pl.Expr) -> pl.Expr:
    """
    Absolute energy is defined as Sum(x_i^2).
    """
    y = str_to_expr(x)
    return y.dot(y)


def query_mean_abs_change(x: str | pl.Expr) -> pl.Expr:
    """
    Returns the mean of all successive differences |X_i - X_i-1|
    """
    return str_to_expr(x).diff(null_behavior="drop").abs().mean()


def query_mean_n_abs_max(x: str | pl.Expr, n_maxima: int) -> pl.Expr:
    """
    Returns the average of the top `n_maxima` of |x|.
    """
    if n_maxima <= 0:
        raise ValueError("The number of maxima should be > 0.")
    return str_to_expr(x).abs().top_k(n_maxima).mean()


def query_cv(x: str | pl.Expr, ddof: int = 1) -> pl.Expr:
    """
    Returns the coefficient of variation for the variable. This is a shorthand for std / mean.

    Parameters
    ----------
    x
        The variable
    ddof
        The delta degree of frendom used in std computation
    """
    xx = str_to_expr(x)
    return xx.std(ddof=ddof) / xx.mean()


def query_count_uniques(x: str | pl.Expr) -> pl.Expr:
    """
    Returns the count of unique values.
    """
    return str_to_expr(x).is_unique().sum()


def query_range_count(x: str | pl.Expr, lower: float, upper: float) -> pl.Expr:
    """
    Returns the number of values inside [`lower`, `upper`].
    """
    return str_to_expr(x).is_between(lower_bound=lower, upper_bound=upper).sum()


def query_longest_streak(where: str | pl.Expr) -> pl.Expr:
    """
    Finds the longest streak length where the condition `where` is true.

    Note: the query is still runnable when `where` doesn't represent boolean column / boolean expressions.
    However, if that is the case the answer will not be easily interpretable.

    Parameters
    ----------
    where
        If where is string, the string must represent the name of a string column. If where is
        an expression, the expression must evaluate to some boolean expression.
    """

    if isinstance(where, str):
        condition = pl.col(where)
    else:
        condition = where

    y = condition.rle().struct.rename_fields(
        ["len", "value"]
    )  # POLARS V1 rename fields can be removed when polars hit v1.0
    return (
        y.filter(y.struct.field("value"))
        .struct.field("len")
        .max()
        .fill_null(0)
        .alias("longest_streak")
    )


def query_avg_streak(where: str | pl.Expr) -> pl.Expr:
    """
    Finds the average streak length where the condition `where` is true. The average is taken on
    the true set.

    Note: the query is still runnable when `where` doesn't represent boolean column / boolean expressions.
    However, if that is the case the answer will not be easily interpretable.

    Parameters
    ----------
    where
        If where is string, the string must represent the name of a string column. If where is
        an expression, the expression must evaluate to some boolean expression.
    """

    if isinstance(where, str):
        condition = pl.col(where)
    else:
        condition = where

    y = condition.rle().struct.rename_fields(
        ["len", "value"]
    )  # POLARS V1 rename fields can be removed when polars hit v1.0
    return (
        y.filter(y.struct.field("value"))
        .struct.field("len")
        .mean()
        .fill_null(0)
        .alias("avg_streak")
    )


def query_streak(where: str | pl.Expr) -> pl.Expr:
    """
    Finds the streak length where the condition `where` is true. This returns a full column of streak lengths.

    Note: the query is still runnable when `where` doesn't represent boolean column / boolean expressions.
    However, if that is the case the answer will not be easily interpretable.

    Parameters
    ----------
    where
        If where is string, the string must represent the name of a boolean column. If where is
        an expression, the expression must evaluate to some boolean series.
    """

    if isinstance(where, str):
        condition = pl.col(where)
    else:
        condition = where

    y = condition.rle().struct.rename_fields(
        ["len", "value"]
    )  # POLARS V1 rename fields can be removed when polars hit v1.0
    return y.struct.field("len").alias("streak_len")


def query_first_digit_cnt(var: str | pl.Expr) -> pl.Expr:
    """
    Finds the first digit count in the data. This is closely related to Benford's law,
    which states that the the first digits (1-9) follow a certain distribution.

    The output is a single element column of type list[u32]. The first value represents the count of 1s
    that are the first digit, the second value represents the count of 2s that are the first digit, etc.

    E.g. first digit of 12 is 1, of 0.0312 is 3. For integers, it is possible to have value = 0, and this
    will not be counted as a first digit.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Benford%27s_law
    """
    return pl_plugin(
        symbol="pl_benford_law",
        args=[str_to_expr(var)],
        returns_scalar=True,
    )


def query_similar_count(
    query: Iterable[float],
    target: str | pl.Expr,
    threshold: float,
    metric: Literal["sql2", "sqzl2"] = "sqzl2",
    parallel: bool = False,
    return_ratio: bool = False,
) -> pl.Expr:
    """
    Given a query subsequence, find the number of same-sized subsequences (windows) in target
    series that have distance < threshold from it.

    Note: If target is largely null, errors may occur. If metric is sqzl2, a mininum variance
    of 1e-10 is applied to all variance calculations to avoid division by 0.

    Parameters
    ----------
    query
        The query subsequence. Must not contain nulls.
    target
        The target time series.
    threshold
        The distance threshold
    metric
        Either 'sql2' or 'sqzl2', which stands for squared l2 and squared z-normalized l2.
    parallel
        Only applies when method is `direct`. Whether to compute the convulotion in parallel. Note that this may not
        have the expected performance when you are in group_by or other parallel context already. It is recommended
        to use this in select/with_columns context, when few expressions are being run at the same time.
    return_ratio
        If true, return # of similar subseuqnces / total number of subsequences.
    """

    q = pl.Series(name="", values=query, dtype=pl.Float64)
    if q.null_count() > 0:
        raise ValueError("Nulls found in the query subsequence.")
    if len(q) <= 1:
        raise ValueError("Length of the query should be > 1.")

    t = str_to_expr(target)
    kwargs = {"threshold": threshold, "parallel": parallel}
    if metric == "sql2":
        result = pl_plugin(
            symbol="pl_subseq_sim_cnt_l2",
            args=[t.cast(pl.Float64).rechunk(), q],
            kwargs=kwargs,
            returns_scalar=True,
        )
    elif metric == "sqzl2":  # pl_subseq_sim_cnt_zl2
        rolling_mean = t.rolling_mean(window_size=len(q)).slice(len(q) - 1, None)
        rolling_var = pl.max_horizontal(
            t.rolling_var(window_size=len(q)).slice(len(q) - 1, None).fill_nan(1e-10),
            pl.lit(1e-10, dtype=pl.Float64),
        )
        qq = pl.lit(q)
        args = [
            t.cast(pl.Float64).rechunk(),
            (qq - qq.mean()) / qq.std(),
            rolling_mean,
            rolling_var,
        ]
        result = pl_plugin(
            symbol="pl_subseq_sim_cnt_zl2",
            args=args,
            kwargs=kwargs,
            returns_scalar=True,
        )
    else:
        raise ValueError(f"Unsupported metric {metric}.")

    if return_ratio:
        return result / (t.len() - len(q) + 1)
    return result


def query_lempel_ziv(b: str | pl.Expr, as_ratio: bool = True) -> pl.Expr:
    """
    Computes Lempel Ziv complexity on a boolean column. Null will be mapped to False.

    Parameters
    ----------
    b
        A boolean column
    as_ratio : bool
        If true, return complexity / length.
    """
    x = str_to_expr(b)
    out = pl_plugin(
        symbol="pl_lempel_ziv_complexity",
        args=[x],
        returns_scalar=True,
    )
    if as_ratio:
        return out / x.len()
    return out


def query_c3_stats(x: str | pl.Expr, lag: int) -> pl.Expr:
    """
    Measure of non-linearity in the time series using c3 statistics.

    Parameters
    ----------
    x : pl.Expr
        Either the name of the column or a Polars expression
    lag : int
        The lag that should be used in the calculation of the feature.

    Reference
    ---------
    https://arxiv.org/pdf/chao-dyn/9909043
    """
    two_lags = 2 * lag
    xx = str_to_expr(x)
    return ((xx.mul(xx.shift(lag)).mul(xx.shift(two_lags))).sum()).truediv(xx.len() - two_lags)


def query_cid_ce(x: str | pl.Expr, normalize: bool = False) -> pl.Expr:
    """
    Estimates the time series complexity.

    Parameters
    ----------
    x : pl.Expr
        Either the name of the column or a Polars expression
    normalize : bool, optional
        If True, z-normalizes the time-series before computing the feature.
        Default is False.

    Reference
    ---------
    https://www.cs.ucr.edu/~eamonn/Complexity-Invariant%20Distance%20Measure.pdf
    """
    xx = str_to_expr(x)
    if normalize:
        y = (xx - xx.mean()) / xx.std()
    else:
        y = xx

    z = y - y.shift(-1)
    return z.dot(z).sqrt()


def query_time_reversal_asymmetry_stats(x: str | pl.Expr, n_lags: int) -> pl.Expr:
    """
    Queries the Time Reversal Asymmetry Statistic, which is the average of
    (L^2(x) * L(x) - L(x) * x^2), where L is the lag operator.
    """
    y = str_to_expr(x)
    one_lag = y.shift(-n_lags)
    two_lag = y.shift(-2 * n_lags)  # Nulls won't be in the mean calculation
    return (one_lag * (two_lag + y) * (two_lag - y)).mean()


#################################################
# Entropies | Entropy related features          #
#################################################


def query_entropy(x: str | pl.Expr, base: float = math.e, normalize: bool = True) -> pl.Expr:
    """
    Computes the entropy of any discrete column. This is shorthand for x.unique_counts().entropy()

    Parameters
    ----------
    x
        Either a string or a polars expression
    base
        Base for the log in the entropy computation
    normalize
        Normalize if the probabilities don't sum to 1.
    """
    return str_to_expr(x).unique_counts().entropy(base=base, normalize=normalize)


def query_cond_entropy(x: str | pl.Expr, y: str | pl.Expr) -> pl.Expr:
    """
    Queries the conditional entropy of x on y, aka. H(x|y).

    Parameters
    ----------
    x
        Either a string or a polars expression
    y
        Either a string or a polars expression
    """
    return pl_plugin(
        symbol="pl_conditional_entropy",
        args=[str_to_expr(x), str_to_expr(y)],
        returns_scalar=True,
        pass_name_to_apply=True,
    )


def query_sample_entropy(
    ts: str | pl.Expr, ratio: float = 0.2, m: int = 2, parallel: bool = False
) -> pl.Expr:
    """
    Calculate the sample entropy of this column. It is highly
    recommended that the user impute nulls before calling this.

    If NaN/some error is returned/thrown, it is likely that:
    (1) Too little data, e.g. m + 1 > length
    (2) ratio or (ratio * std) is too close to or below 0 or std is null/NaN.

    Parameters
    ----------
    ts : str | pl.Expr
        A time series
    ratio : float
        The tolerance parameter. Default is 0.2.
    m : int
        Length of a run of data. Most common run length is 2.
    parallel : bool
        Whether to run this in parallel or not. This is recommended when you
        are running only this expression, and not in group_by context.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Sample_entropy
    """
    if m <= 1:
        raise ValueError("Input `m` must be > 1.")

    t = str_to_expr(ts)
    r = ratio * t.std(ddof=0)
    rows = t.len() - m + 1

    data = [r, t.slice(0, length=rows)]
    # See rust code for more comment on why I put m + 1 here.
    data.extend(
        t.shift(-i).slice(0, length=rows).alias(str(i)) for i in range(1, m + 1)
    )  # More errors are handled in Rust
    return pl_plugin(
        symbol="pl_sample_entropy",
        args=data,
        kwargs={
            "k": 0,
            "metric": "inf",
            "parallel": parallel,
        },
        returns_scalar=True,
        pass_name_to_apply=True,
    )


def query_approx_entropy(
    ts: str | pl.Expr,
    m: int,
    filtering_level: float,
    scale_by_std: bool = True,
    parallel: bool = True,
) -> pl.Expr:
    """
    Approximate sample entropies of a time series given the filtering level. It is highly
    recommended that the user impute nulls before calling this.

    If NaN/some error is returned/thrown, it is likely that:
    (1) Too little data, e.g. m + 1 > length
    (2) filtering_level or (filtering_level * std) is too close to 0 or std is null/NaN.

    Parameters
    ----------
    ts : str | pl.Expr
        A time series
    m : int
        Length of compared runs of data. This is `m` in the wikipedia article.
    filtering_level : float
        Filtering level, must be positive. This is `r` in the wikipedia article.
    scale_by_std : bool
        Whether to scale filter level by std of data. In most applications, this is the default
        behavior, but not in some other cases.
    parallel : bool
        Whether to run this in parallel or not. This is recommended when you
        are running only this expression, and not in group_by context.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Approximate_entropy
    """

    if filtering_level <= 0 or m <= 1:
        raise ValueError("Filter level must be positive and m must be > 1.")

    t = str_to_expr(ts)
    if scale_by_std:
        r: pl.Expr = filtering_level * t.std()
    else:
        r: pl.Expr = pl.lit(filtering_level, dtype=pl.Float64)

    rows = t.len() - m + 1
    data = [r, t.slice(0, length=rows).cast(pl.Float64)]
    # See rust code for more comment on why I put m + 1 here.
    data.extend(
        t.shift(-i).slice(0, length=rows).cast(pl.Float64).alias(str(i)) for i in range(1, m + 1)
    )
    # More errors are handled in Rust
    return pl_plugin(
        symbol="pl_approximate_entropy",
        args=data,
        kwargs={
            "k": 0,
            "metric": "inf",
            "parallel": parallel,
        },
        returns_scalar=True,
        pass_name_to_apply=True,
    )


def query_knn_entropy(
    *features: str | pl.Expr,
    k: int = 3,
    dist: Distance = "l2",
    parallel: bool = False,
) -> pl.Expr:
    """
    Computes KNN entropy among all the rows.

    Note if rows <= k, NaN will be returned.

    Parameters
    ----------
    *features
        Columns used as features
    k
        The number of nearest neighbor to consider. Usually 2 or 3.
    dist : Literal[`l2`, `inf`]
        Note `l2` here has to be `l2` with square root.
    parallel : bool
        Whether to run the distance query in parallel. This is recommended when you
        are running only this expression, and not in group_by context.

    Reference
    ---------
    https://arxiv.org/pdf/1506.06501v1.pdf
    """
    if k <= 0:
        raise ValueError("Input `k` must be > 0.")
    if dist not in ["l2", "inf"]:
        raise ValueError("Invalid metric for KNN entropy.")

    return pl_plugin(
        symbol="pl_knn_entropy",
        args=[str_to_expr(e).alias(str(i)) for i, e in enumerate(features)],
        kwargs={
            "k": k,
            "metric": dist,
            "parallel": parallel,
            "skip_eval": False,
            "skip_data": False,
        },
        returns_scalar=True,
        pass_name_to_apply=True,
    )


def query_copula_entropy(*features: str | pl.Expr, k: int = 3, parallel: bool = False) -> pl.Expr:
    """
    Estimates Copula Entropy via rank statistics.

    Reference
    ---------
    Jian Ma and Zengqi Sun. Mutual information is copula entropy. Tsinghua Science & Technology, 2011, 16(1): 51-54.
    """
    ranks = [x.rank() / x.len() for x in (str_to_expr(f) for f in features)]
    return -query_knn_entropy(*ranks, k=k, dist="l2", parallel=parallel)


def query_cond_indep(
    x: str | pl.Expr, y: str | pl.Expr, z: str | pl.Expr, k: int = 3, parallel: bool = False
) -> pl.Expr:
    """
    Computes the conditional independance of `x`  and `y`, conditioned on `z`

    Reference
    ---------
    Jian Ma. Multivariate Normality Test with Copula Entropy. arXiv preprint arXiv:2206.05956, 2022.
    """
    # We can likely optimize this by going into Rust.
    # Here we are
    # (1) computing rank multiple times
    # (2) creating 3 separate kd-trees, and copying the data 3 times. Might just need to copy once.
    xyz = query_copula_entropy(x, y, z, k=k, parallel=parallel)
    yz = query_copula_entropy(y, z, k=k, parallel=parallel)
    xz = query_copula_entropy(x, z, k=k, parallel=parallel)
    return xyz - yz - xz


def query_transfer_entropy(
    x: str | pl.Expr, source: str | pl.Expr, lag: int = 1, k: int = 3, parallel: bool = False
) -> pl.Expr:
    """
    Estimating transfer entropy from `source` to `x` with a lag

    Reference
    ---------
    Jian Ma. Estimating Transfer Entropy via Copula Entropy. arXiv preprint arXiv:1910.04375, 2019.
    """
    if lag < 1:
        raise ValueError("Input `lag` must be >= 1.")

    xx = str_to_expr(x)
    x1 = xx.slice(0, pl.len() - lag)
    x2 = xx.slice(lag, pl.len() - lag)  # (equivalent to slice(lag, None), but will break in v1.0)
    s = str_to_expr(source).slice(0, pl.len() - lag)
    return query_cond_indep(x2, s, x1, k=k, parallel=parallel)


def query_permute_entropy(
    ts: str | pl.Expr,
    tau: int = 1,
    n_dims: int = 3,
    base: float = math.e,
) -> pl.Expr:
    """
    Computes permutation entropy.

    Parameters
    ----------
    ts : str | pl.Expr
        A time series
    tau : int
        The embedding time delay which controls the number of time periods between elements
        of each of the new column vectors.
    n_dims : int, > 1
        The embedding dimension which controls the length of each of the new column vectors
    base : float
        The base for log in the entropy computation

    Reference
    ---------
    https://www.aptech.com/blog/permutation-entropy/
    """
    if n_dims <= 1:
        raise ValueError("Input `n_dims` has to be > 1.")
    if tau < 1:
        raise ValueError("Input `tau` has to be >= 1.")

    t = str_to_expr(ts)
    if tau == 1:  # Fast track the most common use case
        return (
            pl.concat_list(t, *(t.shift(-i) for i in range(1, n_dims)))
            .head(t.len() - n_dims + 1)
            .list.eval(pl.element().arg_sort())
            .value_counts()  # groupby and count, but returns a struct
            .struct.field("count")  # extract the field named "count"
            .entropy(base=base, normalize=True)
        )
    else:
        return (
            pl.concat_list(
                t.gather_every(tau),
                *(t.shift(-i).gather_every(tau) for i in range(1, n_dims)),
            )
            .slice(0, length=(t.len() // tau) + 1 - (n_dims // tau))
            .list.eval(pl.element().arg_sort())
            .value_counts()
            .struct.field("count")
            .entropy(base=base, normalize=True)
        )
