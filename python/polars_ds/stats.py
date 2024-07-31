from __future__ import annotations

import polars as pl
import math
from .type_alias import Alternative, str_to_expr, CorrMethod, Noise, QuantileMethod
from typing import Union
from ._utils import pl_plugin

__all__ = [
    "query_ttest_ind",
    "query_ttest_1samp",
    "query_ttest_ind_from_stats",
    "query_ks_2samp",
    "query_f_test",
    "query_mann_whitney_u",
    "query_chi2",
    "perturb",
    "jitter",
    "add_noise",
    "normal_test",
    "random",
    "random_null",
    "random_int",
    "random_str",
    "random_binomial",
    "random_exp",
    "random_normal",
    "hmean",
    "gmean",
    "weighted_gmean",
    "weighted_mean",
    "weighted_var",
    "weighted_cov",
    "weighted_corr",
    "cosine_sim",
    "weighted_cosine_sim",
    "xi_corr",
    "kendall_tau",
    "bicor",
    "corr",
]


def query_ttest_ind(
    var1: str | pl.Expr,
    var2: str | pl.Expr,
    alternative: Alternative = "two-sided",
    equal_var: bool = False,
) -> pl.Expr:
    """
    Performs 2 sample student's t test or Welch's t test. Functionality-wise this is desgined
    to be equivalent to SciPy's ttest_ind, with fewer options. The result is not exact but
    within 1e-10 precision from SciPy's.

    In the case of student's t test, the data is assumed to have no nulls, and n = expr.count()
    is used. Note expr.count() only counts non-null elements after polars 0.20.
    The degree of freedom will be 2n - 2. As a result, nulls might cause problems.

    In the case of Welch's t test, data will be sanitized (nulls, NaNs, Infs will be dropped
    before the test), and df will be counted based on the length of sanitized data.

    Parameters
    ----------
    var1
        Variable 1
    var2
        Variable 2
    alternative : {"two-sided", "less", "greater"}
        Alternative of the hypothesis test
    equal_var
        If true, perform standard student t 2 sample test. Otherwise, perform Welch's
        t test.
    """
    y1, y2 = str_to_expr(var1), str_to_expr(var2)
    if equal_var:
        m1 = y1.mean()
        m2 = y2.mean()
        v1 = y1.var()
        v2 = y2.var()
        cnt = y1.count().cast(pl.UInt64)
        return pl_plugin(
            symbol="pl_ttest_2samp",
            args=[m1, m2, v1, v2, cnt, pl.lit(alternative, dtype=pl.String)],
            returns_scalar=True,
        )
    else:
        s1 = y1.filter(y1.is_finite())
        s2 = y2.filter(y2.is_finite())
        m1 = s1.mean()
        m2 = s2.mean()
        v1 = s1.var()
        v2 = s2.var()
        n1 = s1.len().cast(pl.UInt64)
        n2 = s2.len().cast(pl.UInt64)
        return pl_plugin(
            symbol="pl_welch_t",
            args=[m1, m2, v1, v2, n1, n2, pl.lit(alternative, dtype=pl.String)],
            returns_scalar=True,
        )


def query_ttest_1samp(
    var1: str | pl.Expr, pop_mean: float, alternative: Alternative = "two-sided"
) -> pl.Expr:
    """
    Performs a standard 1 sample t test using reference column and expected mean. This function
    sanitizes the self column first. The df is the count of valid values.

    Parameters
    ----------
    var1
        Variable 1
    pop_mean
        The expected population mean in the hypothesis test
    alternative : {"two-sided", "less", "greater"}
        Alternative of the hypothesis test
    """
    y = str_to_expr(var1)
    s1 = y.filter(y.is_finite())
    sm = s1.mean()
    pm = pl.lit(pop_mean, dtype=pl.Float64)
    var = s1.var()
    cnt = s1.len().cast(pl.UInt64)
    alt = pl.lit(alternative, dtype=pl.String)
    return pl_plugin(
        symbol="pl_ttest_1samp",
        args=[sm, pm, var, cnt, alt],
        returns_scalar=True,
    )


def query_ttest_ind_from_stats(
    var1: str | pl.Expr,
    mean: float,
    var: float,
    cnt: int,
    alternative: Alternative = "two-sided",
    equal_var: bool = False,
) -> pl.Expr:
    """
    Performs 2 sample student's t test or Welch's t test, using only scalar statistics from other.
    This is more suitable for t-tests between rolling data and some other fixed data, from which you
    can compute the mean, var, and count only once.

    Parameters
    ----------
    var1
        The variable 1
    mean
        The mean of var2
    var
        The var of var2
    cnt
        The count of var2, used only in welch's t test
    alternative : {"two-sided", "less", "greater"}
        Alternative of the hypothesis test
    equal_var
        If true, perform standard student t 2 sample test. Otherwise, perform Welch's
        t test.
    """
    y = str_to_expr(var1)
    if equal_var:
        m1 = y.mean()
        m2 = pl.lit(mean, pl.Float64)
        v1 = y.var()
        v2 = pl.lit(var, pl.Float64)
        cnt = y.count().cast(pl.UInt64)
        return pl_plugin(
            symbol="pl_ttest_2samp",
            args=[m1, m2, v1, v2, cnt, pl.lit(alternative, dtype=pl.String)],
            returns_scalar=True,
        )
    else:
        s1 = y.filter(y.is_finite())
        m1 = s1.mean()
        m2 = pl.lit(mean, pl.Float64)
        v1 = s1.var()
        v2 = pl.lit(var, pl.Float64)
        n1 = s1.len().cast(pl.UInt64)
        n2 = pl.lit(cnt, pl.UInt64)
        return pl_plugin(
            symbol="pl_welch_t",
            args=[m1, m2, v1, v2, n1, n2, pl.lit(alternative, dtype=pl.String)],
            returns_scalar=True,
        )


def query_ks_2samp(
    var1: str | pl.Expr,
    var2: str | pl.Expr,
    alpha: float = 0.05,
    is_binary: bool = False,
) -> pl.Expr:
    """
    Computes two-sided KS statistics between var1 and var2. This will
    sanitize data (only non-null finite values are used) before doing the computation. If
    is_binary is true, it will compare the statistics by comparing var2(var1=0) and var2(var1=1).

    Note, this returns a stastics and a threshold value. The threshold is not the p-value, but
    rather it is used in the following way: if the statistic is > the threshold value, then the null
    hypothesis should be rejected. This is suitable only for large sameple sizes. See more details
    in the reference.

    If either var1 or var2 has less than 20 values, a ks stats of INFINITY will be returned.

    Parameters
    ----------
    var1
        Variable 1
    var2
        Variable 2
    alpha
        The confidence level used to estimate p-value
    is_binary
        If true, instead of running ks(var1, var2), it runs ks(var2(var1=0), var2(var1=1))

    Reference
    ---------
    https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test#Two-sample_Kolmogorov%E2%80%93Smirnov_test
    """
    y1, y2 = str_to_expr(var1), str_to_expr(var2)
    if is_binary:
        z = y2.filter(y2.is_finite()).cast(pl.Float64)
        z1 = z.filter(y1 == 1).sort()
        z2 = z.filter(y1 == 0).sort()
        return pl_plugin(
            symbol="pl_ks_2samp",
            args=[z1, z2, pl.lit(alpha, pl.Float64)],
            returns_scalar=True,
        )
    else:
        z1 = y1.filter(y1.is_finite()).sort()
        z2 = y2.filter(y2.is_finite()).sort()
        return pl_plugin(
            symbol="pl_ks_2samp",
            args=[z1, z2, pl.lit(alpha, pl.Float64)],
            returns_scalar=True,
        )


def query_f_test(*variables: str | pl.Expr, group: str | pl.Expr) -> pl.Expr:
    """
    Performs the ANOVA F-test.

    Parameters
    ----------
    variables
        The columns (variables) to run ANOVA F-test on
    group
        The "target" column used to group the variables
    """
    vars_ = [str_to_expr(group)]
    vars_.extend(str_to_expr(x) for x in variables)
    if len(vars_) <= 1:
        raise ValueError("No input feature column to run F-test on.")
    elif len(vars_) == 2:
        return pl_plugin(symbol="pl_f_test", args=vars_, returns_scalar=True)
    else:
        return pl_plugin(symbol="pl_f_test", args=vars_, changes_length=True)


def query_chi2(var1: str | pl.Expr, var2: str | pl.Expr) -> pl.Expr:
    """
    Computes the Chi Squared statistic and p value between two categorical values.

    Note that it is up to the user to make sure that the two columns contain categorical
    values. This method is equivalent to SciPy's chi2_contingency, except that it also
    computes the contingency table internally for the user.

    Parameters
    ----------
    var1
        Either the name of the column or a Polars expression
    var2
        Either the name of the column or a Polars expression
    """
    return pl_plugin(
        symbol="pl_chi2",
        args=[str_to_expr(var1), str_to_expr(var2)],
        returns_scalar=True,
    )


def query_mann_whitney_u(
    var1: str | pl.Expr,
    var2: str | pl.Expr,
    alternative: Alternative = "two-sided",
) -> pl.Expr:
    """
    Computes the Mann-Whitney U statistic and the p-value. Note: this function will sanitize data (drop
    all non-finite values) before computing the statistic. This implementation follows method 2 in reference.
    This always applies tie correction, which may slow down computation by a little.

    WIP. PVALUE NOT DONE YET.

    Parameters
    ----------
    var1 : pl.Expr
        Either the name of the column or a Polars expression
    var2 : pl.Expr
        Either the name of the column or a Polars expression
    alternative: str
        The alternative for the test. `two-sided`, `greater` or `less`

    Reference
    ---------
    https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test
    """
    x = str_to_expr(var1)
    y = str_to_expr(var2)
    xx = x.filter(x.is_finite())
    yy = y.filter(y.is_finite())
    n1 = xx.len().cast(pl.Float64)
    n2 = yy.len().cast(pl.Float64)

    ranks = (xx.append(yy)).rank()

    u1 = ranks.slice(0, length=xx.len()).sum() - (n1 * (n1 + 1)) / 2
    u2 = (n1 * n2) - u1

    mean = (n1 * n2) / 2
    return pl_plugin(
        symbol="pl_mann_whitney_u",
        args=[u1, u2, mean, ranks.sort(), pl.lit(alternative, dtype=pl.String)],
    )


def winsorize(
    x: str | pl.Expr,
    lower: float = 0.05,
    upper: float = 0.95,
    method: QuantileMethod = "nearest",
) -> pl.Expr:
    """
    Winsorize the data by clipping by percentiles at the lower and upper ends.

    Parameters
    ----------
    x
        Either the name of the column or a Polars expression
    lower
        The lower percentile value to clip the data. E.g everything < x.quantile(lower)
        will be mapped to x.quantile(lower)
    upper
        The upper percentile value to clip the data. E.g everything > x.quantile(upper)
        will be mapped to x.quantile(upper)
    method
        Method for quantile estimate. One of "nearest", "higher", "lower", "midpoint", "linear".
    """
    if lower <= 0.0 or lower >= 1.0 or upper <= 0.0 or upper >= 1.0 or upper <= lower:
        raise ValueError("Lower and upper must be with in (0, 1) and upper should be > lower")

    xx = str_to_expr(x)
    return xx.clip(
        xx.quantile(lower, interpolation=method), xx.quantile(upper, interpolation=method)
    )


def perturb(x: str | pl.Expr, epsilon: float, positive: bool = False):
    """
    Perturb the var by a small amount. This only applies to float columns.

    Parameters
    ----------
    x
        Either the name of the column or a Polars expression
    epsilon
        The small amount to perturb.
    positive
        If true, randomly add a small amount in [0, epsilon). If false, it will use the range
        [-epsilon/2, epsilon/2)
    """
    if math.isinf(epsilon) or math.isnan(epsilon):
        raise ValueError("Input `epsilon should be a valid finite value.`")

    ep = abs(epsilon)
    if positive:
        lo = pl.lit(0.0, dtype=pl.Float64)
        hi = pl.lit(ep, dtype=pl.Float64)
    else:
        half = ep / 2
        lo = pl.lit(-half, dtype=pl.Float64)
        hi = pl.lit(half, dtype=pl.Float64)

    return pl_plugin(
        symbol="pl_perturb",
        args=[str_to_expr(x), lo, hi],
        is_elementwise=True,
    )


def jitter(x: str | pl.Expr, std: float | pl.Expr = 1.0) -> pl.Expr:
    """
    Adds a Gaussian noise of N(0, std) to the column.

    Parameters
    ----------
    x
        Either the name of the column or a Polars expression
    std
        The std of the Gaussian noise.
    """
    if isinstance(std, float):
        if std < 0:
            raise ValueError("Standard deviation must be positive.")
        elif std == 0:
            return str_to_expr(x)

        s = pl.lit(std, dtype=pl.Float64)
    else:
        s = std.cast(pl.Float64)

    return pl_plugin(
        symbol="pl_jitter",
        args=[str_to_expr(x), s],
        is_elementwise=True,
    )


def add_noise(x: str | pl.Expr, noise_type: Noise = "gaussian", **kwargs) -> pl.Expr:
    """
    Adds some noise to the column.

    Parameters
    ----------
    x
        Either the name of the column or a Polars expression
    noise_type
        Either "gaussian" or "uniform"
    kwargs
        If noise_type = "gaussian", this accepts kwargs to "jitter" and if "uniform", this
        accepts kwargs to "perturb".
    """
    if noise_type == "gaussian":
        return jitter(x, **kwargs)
    elif noise_type == "uniform":
        return perturb(x, **kwargs)
    else:
        raise ValueError(f"The noise_type {noise_type} is not currently supported.")


def normal_test(var: str | pl.Expr) -> pl.Expr:
    """
    Perform a normality test which is based on D'Agostino and Pearson's test
    that combines skew and kurtosis to produce an omnibus test of normality.
    Null values, NaN and inf are dropped when running this computation.

    Parameters
    ----------
    var
        Either the name of the column or a Polars expression

    References
    ----------
    D'Agostino, R. B. (1971), "An omnibus test of normality for
        moderate and large sample size", Biometrika, 58, 341-348
    D'Agostino, R. and Pearson, E. S. (1973), "Tests for departure from
        normality", Biometrika, 60, 613-622
    """
    y = str_to_expr(var)
    valid: pl.Expr = y.filter(y.is_finite())
    skew = valid.skew()
    # Pearson Kurtosis, see here: https://en.wikipedia.org/wiki/D%27Agostino%27s_K-squared_test
    kur = valid.kurtosis(fisher=False)
    return pl_plugin(
        symbol="pl_normal_test",
        args=[skew, kur, valid.count().cast(pl.UInt32)],
        returns_scalar=True,
    )


def random(
    lower: Union[pl.Expr, float] = 0.0,
    upper: Union[pl.Expr, float] = 1.0,
    seed: int | None = None,
) -> pl.Expr:
    """
    Generate random numbers in [lower, upper)

    Parameters
    ----------
    lower
        The lower bound
    upper
        The upper bound, exclusive
    seed
        The random seed. None means no seed.
    """
    lo = pl.lit(lower, pl.Float64) if isinstance(lower, float) else lower
    up = pl.lit(upper, pl.Float64) if isinstance(upper, float) else upper
    return pl_plugin(
        symbol="pl_random",
        args=[pl.len(), lo, up, pl.lit(seed, pl.UInt64)],
        is_elementwise=True,
    )


def random_null(var: str | pl.Expr, pct: float, seed: int | None = None) -> pl.Expr:
    """
    Creates random null values in var. If var contains nulls originally, they
    will stay null.

    Parameters
    ----------
    var
        Either the name of the column or a Polars expression
    pct
        Percentage of nulls to randomly generate. This percentage is based on the
        length of the column, so may not be the actual percentage of nulls depending
        on how many values are originally null.
    seed
        A seed to fix the random numbers. If none, use the system's entropy.
    """
    if pct <= 0.0 or pct >= 1.0:
        raise ValueError("Input `pct` must be > 0 and < 1")

    to_null = random(0.0, 1.0, seed=seed) < pct
    return pl.when(to_null).then(None).otherwise(str_to_expr(var))


def random_int(
    lower: Union[int, pl.Expr], upper: Union[int, pl.Expr], seed: int | None = None
) -> pl.Expr:
    """
    Generates random integer between lower and upper.

    Parameters
    ----------
    lower
        The lower bound, inclusive
    upper
        The upper bound, exclusive
    seed
        The random seed. None means no seed.
    """
    if lower == upper:
        raise ValueError("Input `lower` must be smaller than `higher`")

    lo = pl.lit(lower, pl.Int32) if isinstance(lower, int) else lower.cast(pl.Int32)
    hi = pl.lit(upper, pl.Int32) if isinstance(upper, int) else upper.cast(pl.Int32)
    return pl_plugin(
        symbol="pl_rand_int",
        args=[
            pl.len().cast(pl.UInt32),
            lo,
            hi,
            pl.lit(seed, pl.UInt64),
        ],
        is_elementwise=True,
    )


def random_str(min_size: int, max_size: int) -> pl.Expr:
    """
    Generates random strings of length between min_size and max_size.

    Parameters
    ----------
    min_size
        The min size of the string, inclusive
    max_size
        The max size of the string, inclusive
    seed
        The random seed. None means no seed.
    """
    mi, ma = min_size, max_size
    if min_size > max_size:
        mi, ma = max_size, min_size

    return pl_plugin(
        symbol="pl_rand_str",
        args=[
            pl.len().cast(pl.UInt32),
            pl.lit(mi, pl.UInt32),
            pl.lit(ma, pl.UInt32),
            pl.lit(42, pl.UInt64),
        ],
        is_elementwise=True,
    )


def random_binomial(n: int, p: int, seed: int | None = None) -> pl.Expr:
    """
    Generates random integer following a binomial distribution.

    Parameters
    ----------
    n
        The n in a binomial distribution
    p
        The p in a binomial distribution
    seed
        The random seed. None means no seed.
    """
    if n < 1:
        raise ValueError("Input `n` must be > 1.")

    return pl_plugin(
        symbol="pl_rand_binomial",
        args=[
            pl.len().cast(pl.UInt32),
            pl.lit(n, pl.Int32),
            pl.lit(p, pl.Float64),
            pl.lit(seed, pl.UInt64),
        ],
        is_elementwise=True,
    )


def random_exp(lambda_: float, seed: int | None = None) -> pl.Expr:
    """
    Generates random numbers following an exponential distribution.

    Parameters
    ----------
    lambda_
        The lambda in an exponential distribution
    seed
        The random seed. None means no seed.
    """
    return pl_plugin(
        symbol="pl_rand_exp",
        args=[pl.len().cast(pl.UInt32), pl.lit(lambda_, pl.Float64), pl.lit(seed, pl.UInt64)],
        is_elementwise=True,
    )


def random_normal(
    mean: Union[pl.Expr, float], std: Union[pl.Expr, float], seed: int | None = None
) -> pl.Expr:
    """
    Generates random number following a normal distribution.

    Parameters
    ----------
    mean
        The mean in a normal distribution
    std
        The std in a normal distribution
    seed
        The random seed. None means no seed.
    """
    m = pl.lit(mean, pl.Float64) if isinstance(mean, float) else mean
    s = pl.lit(std, pl.Float64) if isinstance(std, float) else std
    return pl_plugin(
        symbol="pl_rand_normal",
        args=[pl.len().cast(pl.UInt32), m, s, pl.lit(seed, pl.UInt64)],
        is_elementwise=True,
    )


def hmean(var: str | pl.Expr) -> pl.Expr:
    """
    Computes the harmonic mean.

    Parameters
    ----------
    var
        The variable
    """
    x = str_to_expr(var)
    return x.count() / (1.0 / x).sum()


def gmean(var: str | pl.Expr) -> pl.Expr:
    """
    Computes the geometric mean.

    Parameters
    ----------
    var
        The variable
    """
    return str_to_expr(var).ln().mean().exp()


def weighted_gmean(
    var: str | pl.Expr, weights: str | pl.Expr, is_normalized: bool = False
) -> pl.Expr:
    """
    Computes the weighted geometric mean.

    Parameters
    ----------
    var
        The variable
    weights
        An expr representing weights. Must be of same length as var.
    is_normalized
        If true, the weights are assumed to sum to 1. If false, will divide by sum of the weights
    """
    x, w = str_to_expr(var), str_to_expr(weights)
    if is_normalized:
        return (x.ln().dot(w)).exp()
    else:
        return (x.ln().dot(w) / (w.sum())).exp()


def weighted_mean(
    var: str | pl.Expr, weights: str | pl.Expr, is_normalized: bool = False
) -> pl.Expr:
    """
    Computes the weighted mean, where weights is an expr represeting
    a weight column. The weights column must have the same length as var.

    All weights are assumed to be > 0. This will not check if weights are valid.

    Parameters
    ----------
    var
        The variable
    weights
        An expr representing weights. Must be of same length as var.
    is_normalized
        If true, the weights are assumed to sum to 1. If false, will divide by sum of the weights
    """
    x, w = str_to_expr(var), str_to_expr(weights)
    out = x.dot(w)
    if is_normalized:
        return out
    return out / w.sum()


def weighted_var(var: str | pl.Expr, weights: str | pl.Expr, freq_weights: bool = False) -> pl.Expr:
    """
    Computes the weighted variance. The weights column must have the same length as var.

    All weights are assumed to be > 0. This will not check if weights are valid.

    Parameters
    ----------
    var
        The variable
    weights
        An expr representing weights. Must be of same length as var.
    freq_weights
        Whether to follow the formula for frequency weights or other types of weights. See reference
        for detail. If true, this assumes frequency weights are NOT normalized. If false, the
        weighted sample variance is biased. See reference for more info.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance
    """
    x, w = str_to_expr(var), str_to_expr(weights)
    wm = weighted_mean(x, w, False)
    summand = w.dot((x - wm).pow(2))
    if freq_weights:
        return summand / (w.sum() - 1)
    return summand / w.sum()


def weighted_cov(x: str | pl.Expr, y: str | pl.Expr, weights: Union[pl.Expr, float]) -> pl.Expr:
    """
    Computes the weighted covariance between x and y. The weights column must have the same
    length as both x an y.

    All weights are assumed to be > 0. This will not check if weights are valid.

    Parameters
    ----------
    x
        The first variable
    y
        The second variable
    weights
        An expr representing weights. Must be of same length as var.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Weighted_correlation_coefficient
    """
    xx, yy, w = str_to_expr(x), str_to_expr(y), str_to_expr(weights)
    wx, wy = weighted_mean(xx, w, False), weighted_mean(yy, w, False)
    return w.dot((xx - wx) * (yy - wy)) / w.sum()


def weighted_corr(x: str | pl.Expr, y: str | pl.Expr, weights: str | pl.Expr) -> pl.Expr:
    """
    Computes the weighted correlation between x and y. The weights column must have the same
    length as both x an y.

    All weights are assumed to be > 0. This will not check if weights are valid.

    Parameters
    ----------
    x
        The first variable
    y
        The second variable
    weights
        An expr representing weights. Must be of same length as var.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Weighted_correlation_coefficient
    """
    xx, yy = str_to_expr(x), str_to_expr(y)
    w = str_to_expr(weights)
    numerator = weighted_cov(xx, yy, w)
    sxx = w.dot((xx - weighted_mean(xx, w, False)).pow(2))
    syy = w.dot((xx - weighted_mean(yy, w, False)).pow(2))
    return numerator * w.sum() / (sxx * syy).sqrt()


def cosine_sim(x: str | pl.Expr, y: str | pl.Expr) -> pl.Expr:
    """
    Column-and-column cosine similarity

    Parameters
    ----------
    x
        The first variable
    y
        The second variable
    """
    xx, yy = str_to_expr(x), str_to_expr(y)
    x2 = xx.dot(xx).sqrt()
    y2 = yy.dot(yy).sqrt()
    return xx.dot(yy) / (x2 * y2).sqrt()


def weighted_cosine_sim(x: str | pl.Expr, y: str | pl.Expr, weights: str | pl.Expr) -> pl.Expr:
    """
    Computes the weighted cosine similarity between x and y (column-wise). The weights column
    must have the same length as both x an y.

    All weights are assumed to be > 0. This will not check if weights are valid.

    Parameters
    ----------
    x
        The first variable
    y
        The second variable
    weights
        An expr representing weights. Must be of same length as var.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Weighted_correlation_coefficient
    """
    xx, yy = str_to_expr(x), str_to_expr(y)
    w = str_to_expr(weights)
    wx2 = xx.pow(2).dot(w)
    wy2 = yy.pow(2).dot(w)
    return (w * xx).dot(yy) / (wx2 * wy2).sqrt()


def kendall_tau(x: str | pl.Expr, y: str | pl.Expr) -> pl.Expr:
    """
    Computes Kendall's Tau (b) correlation between x and y. This automatically drops rows with null.

    Note: this will map NaN to null and drop all rows with null. Inf will be kept and cosidered as
    the largest value and multiple Infs will be equal. -Inf will be the smallest if it exists in the
    data. A value of NaN will be returned if the data has < 2 rows after nulls are dropped.

    Parameters
    ----------
    x
        The first variable
    y
        The second variable
    """
    xx, yy = str_to_expr(x).fill_nan(None), str_to_expr(y).fill_nan(None)
    return pl_plugin(
        symbol="pl_kendall_tau",
        args=[xx.rank(method="min"), yy.rank(method="min")],
        returns_scalar=True,
    )


def bicor(x: str | pl.Expr, y: str | pl.Expr, c: float = 9.0) -> pl.Expr:
    """
    Computes the Biweight Midcorrelation between x and y. This is commonly referred to as bicor.

    Performance hint: this expression benefits from .lazy() a lot.

    Parameters
    ----------
    x
        The first variable
    y
        The second variable
    c
        Biweight tuning constant which is typically 9

    Reference
    ---------
    https://en.wikipedia.org/wiki/Biweight_midcorrelation
    """
    a, b = str_to_expr(x), str_to_expr(y)
    med_a = a.median()
    med_b = b.median()

    diff_a = a - med_a
    diff_b = b - med_b

    ua = diff_a / (c * diff_a.abs().median())
    ub = diff_b / (c * diff_b.abs().median())

    w_a = (1 - ua.pow(2)).pow(2) * ((1 - ua.abs()) > 0).cast(pl.Float64)
    w_b = (1 - ub.pow(2)).pow(2) * ((1 - ub.abs()) > 0).cast(pl.Float64)

    aa = diff_a * w_a
    bb = diff_b * w_b

    return aa.dot(bb) / (aa.dot(aa) * (bb.dot(bb))).sqrt()


def xi_corr(
    x: str | pl.Expr, y: str | pl.Expr, seed: int | None = None, return_p: bool = False
) -> pl.Expr:
    """
    Computes the ξ(xi) correlation developed by SOURAV CHATTERJEE in the paper in the reference.
    This will return both the correlation (the statistic) and the p-value. Note that if sample size
    is smaller than 30, p-value will always be NaN. The ξ correlation is not symmetric, as it only
    tries to explain whether y is a function of x.

    Parameters
    ----------
    x
        The first variable
    y
        The second variable
    seed
        Whether to have a seed when we break ties at random
    return_p
        Whether to return a two-sided p value for the statistic

    Reference
    ---------
    https://arxiv.org/pdf/1909.10140.pdf
    """
    xx, yy = str_to_expr(x), str_to_expr(y)
    args = [
        xx.rank(method="random", seed=seed),
        yy.rank(method="max").cast(pl.Float64),
        (-yy).rank(method="max").cast(pl.Float64),
    ]
    if return_p:
        return pl_plugin(
            symbol="pl_xi_corr_w_p",
            args=args,
            returns_scalar=True,
        )
    else:
        return pl_plugin(
            symbol="pl_xi_corr",
            args=args,
            returns_scalar=True,
        )


def corr(x: str | pl.Expr, y: str | pl.Expr, method: CorrMethod = "pearson") -> pl.Expr:
    """
    A convenience function for calling different types of correlations. Pearson and Spearman correlation
    runs on Polar's native expression, while Kendall and Xi correlation runs on code in this package.

    Paramters
    ---------
    x
        The first variable
    y
        The second variable
    method
        One of ["pearson", "spearman", "xi", "kendall", "bicor"]
    """
    if method in ["pearson", "spearman"]:
        return pl.corr(x, y, method=method)
    elif method == "xi":
        return xi_corr(x, y)
    elif method == "kendall":
        return kendall_tau(x, y)
    elif method == "bicor":
        return bicor(x, y)
    else:
        raise ValueError(f"Unknown correlation method: {method}.")
