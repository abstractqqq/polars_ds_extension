from __future__ import annotations
import math
import polars as pl
from typing import Union, List, Iterable
from .type_alias import (
    DetrendMethod,
    ConvMode,
    ConvMethod,
    str_to_expr,
)
from ._utils import pl_plugin

__all__ = [
    "query_singular_values",
    "query_principal_components",
    "softmax",
    "query_gcd",
    "query_lcm",
    "haversine",
    "query_pca",
    "query_jaccard_row",
    "query_jaccard_col",
    "query_psi",
    "query_psi_w_breakpoints",
    "query_psi_discrete",
    "query_woe",
    "query_woe_discrete",
    "query_iv",
    "query_iv_discrete",
    "integrate_trapz",
    "convolve",
    "list_amax",
    "gamma",
    "expit",
    "exp2",
    "expit",
    "logit",
    "trunc",
    "detrend",
    "rfft",
    "fract",
]


def softmax(x: str | pl.Expr) -> pl.Expr:
    """
    Applies the softmax function to the column, which turns any real valued column into valid probability
    values. This is simply a shorthand for x.exp() / x.exp().sum() for expressions x.

    Paramters
    ---------
    x
        Either a str represeting a column name or a Polars expression
    """
    xx = str_to_expr(x)
    return xx.exp() / (xx.exp().sum())


def query_gcd(x: str | pl.Expr, y: int | str | pl.Expr) -> pl.Expr:
    """
    Computes GCD of two integer columns. This will try to cast everything to int32.

    Parameters
    ----------
    x
        An integer column
    y
        Either an int, or another integer column
    """
    if isinstance(y, int):
        yy = pl.lit(y, dtype=pl.Int32)
    else:
        yy = str_to_expr(y).cast(pl.Int32)

    return pl_plugin(
        symbol="pl_gcd",
        args=[str_to_expr(x).cast(pl.Int32), yy],
        is_elementwise=True,
    )


def query_lcm(x: str | pl.Expr, y: Union[int, str, pl.Expr]) -> pl.Expr:
    """
    Computes LCM of two integer columns. This will try to cast everything to int32.

    Parameters
    ----------
    x
        An integer column
    y
        Either an int, or another integer column
    """
    if isinstance(y, int):
        yy = pl.lit(y, dtype=pl.Int32)
    else:
        yy = str_to_expr(y).cast(pl.Int32)

    return pl_plugin(
        symbol="pl_lcm",
        args=[str_to_expr(x).cast(pl.Int32), yy],
        is_elementwise=True,
    )


def haversine(
    x_lat: str | pl.Expr,
    x_long: str | pl.Expr,
    y_lat: Union[float, str, pl.Expr],
    y_long: Union[float, str, pl.Expr],
) -> pl.Expr:
    """
    Computes haversine distance using the naive method. The output unit is km.

    Parameters
    ----------
    x_lat
        Column representing latitude in x
    x_long
        Column representing longitude in x
    y_lat
        Column representing latitude in y
    y_long
        Column representing longitude in y
    """
    xlat = str_to_expr(x_lat)
    xlong = str_to_expr(x_long)
    ylat = pl.lit(y_lat) if isinstance(y_lat, float) else str_to_expr(y_lat)
    ylong = pl.lit(y_long) if isinstance(y_long, float) else str_to_expr(y_long)
    return pl_plugin(
        symbol="pl_haversine",
        args=[xlat, xlong, ylat, ylong],
        is_elementwise=True,
        cast_to_supertype=True,
    )


def query_singular_values(
    *features: str | pl.Expr,
    center: bool = True,
    as_explained_var: bool = False,
    as_ratio: bool = False,
) -> pl.Expr:
    """
    Finds all principal values (singular values) for the data matrix formed by the given features
    and returns them in descending order.

    Note: if a row has null values, it will be dropped.

    Paramters
    ---------
    features
        Feature columns
    center
        Whether to center the data or not. If you want to standard-normalize, set this to False,
        and do it for input features by hand.
    as_explained_var
        If true, return the explained variance, which is singular_value ^ 2 / (n_samples - 1)
    as_ratio
        If true, normalize output to between 0 and 1.
    """
    feats = [str_to_expr(f) for f in features]
    if center:
        actual_inputs = [f - f.mean() for f in feats]
    else:
        actual_inputs = feats

    out = pl_plugin(symbol="pl_singular_values", args=actual_inputs, returns_scalar=True)
    if as_explained_var:
        out = out.list.eval(pl.element().pow(2) / (pl.count() - 1))
    if as_ratio:
        out = out.list.eval(pl.element() / pl.element().sum())

    return out


def query_pca(
    *features: str | pl.Expr,
    center: bool = True,
) -> pl.Expr:
    """
    Finds all singular values as well as the principle vectors.

    Paramters
    ---------
    features
        Feature columns
    center
        Whether to center the data or not. If you want to standard normalize, set this to False,
        and do it for input features by hand.
    """
    feats = [str_to_expr(f) for f in features]
    if center:
        actual_inputs = [f - f.mean() for f in feats]
    else:
        actual_inputs = feats

    return pl_plugin(symbol="pl_pca", args=actual_inputs, changes_length=True)


def query_principal_components(
    *features: str | pl.Expr,
    k: int = 2,
    center: bool = True,
) -> pl.Expr:
    """
    Transforms the features to get the first k principal components.

    Paramters
    ---------
    features
        Feature columns
    center
        Whether to center the data or not. If you want to standard normalize, set this to False,
        and do it for input features by hand.
    """
    feats = [str_to_expr(f) for f in features]
    if k > len(feats) or k <= 0:
        raise ValueError("Input `k` should be between 1 and the number of features inclusive.")

    actual_inputs = [pl.lit(k, dtype=pl.UInt32).alias("principal_components")]
    if center:
        actual_inputs.extend(f - f.mean() for f in feats)
    else:
        actual_inputs.extend(feats)

    return pl_plugin(symbol="pl_principal_components", args=actual_inputs)


def query_jaccard_row(first: str | pl.Expr, second: str | pl.Expr) -> pl.Expr:
    """
    Computes jaccard similarity pairwise between this and the other column. The type of
    each column must be list and the lists must have the same inner type. The inner type
    must either be integer or string.

    Parameters
    ----------
    first
        A list column with a hashable inner type
    second
        A list column with a hashable inner type
    """
    return pl_plugin(
        symbol="pl_list_jaccard",
        args=[str_to_expr(first), str_to_expr(second)],
        is_elementwise=True,
    )


def query_jaccard_col(
    first: str | pl.Expr, second: str | pl.Expr, count_null: bool = False
) -> pl.Expr:
    """
    Computes jaccard similarity column-wise. This will hash entire columns and compares the two
    hashsets. Note: only integer/str columns can be compared.

    Parameters
    ----------
    first
        A column with a hashable type
    second
        A column with a hashable type
    count_null
        Whether to count null as a distinct element.
    """
    return pl_plugin(
        symbol="pl_jaccard",
        args=[str_to_expr(first), str_to_expr(second), pl.lit(count_null, dtype=pl.Boolean)],
        returns_scalar=True,
    )


def query_psi(
    new: str | pl.expr | Iterable[float],
    baseline: str | pl.expr | Iterable[float],
    n_bins: int = 10,
    return_report: bool = False,
) -> pl.Expr:
    """
    Compute the Population Stability Index between x and the reference column (usually x's historical values).
    The reference column will be divided into n_bins quantile bins which will be used as basis of comparison.

    Note this assumes values in self and ref are continuous. This will also remove all infinite, null, NA.
    values.

    Also note that it will try to create `n_bins` many unique breakpoints. If input data has < n_bins
    unique breakpoints, the repeated breakpoints will be grouped together, and the computation will be done
    with < `n_bins` many bins. This happens when a single value appears too many times in data. This also
    differs from the reference implementation by treating breakpoints as right-closed intervals with -inf
    and inf being the first and last values of the intervals. This is because we need to accommodate all data
    in the case when actual data's min and the reference data's min are not the same, which is common in reality.

    Parameters
    ----------
    new
        An expression or any iterable that can be turned into a Polars series that represents newly
        arrived feature values
    baseline
        An expression or any iterable that can be turned into a Polars series. Usually this should
        be the feature's historical values
    n_bins : int, > 1
        The number of quantile bins to use
    return_report
        Whether to return a PSI report or not.

    Reference
    ---------
    https://github.com/mwburke/population-stability-index/blob/master/psi.py
    https://www.listendata.com/2015/05/population-stability-index.html
    """
    if n_bins <= 1:
        raise ValueError("Input `n_bins` must be >= 2.")

    if isinstance(new, (str, pl.Expr)):
        new_ = str_to_expr(new)
        valid_new: Union[pl.Series, pl.Expr] = new_.filter(new_.is_finite()).cast(pl.Float64)
    else:
        temp = pl.Series(values=new, dtype=pl.Float64)
        valid_new: Union[pl.Series, pl.Expr] = temp.filter(temp.is_finite())

    if isinstance(baseline, (str, pl.Expr)):
        base = str_to_expr(baseline)
        valid_ref: Union[pl.Series, pl.Expr] = base.filter(base.is_finite()).cast(pl.Float64)
    else:
        temp = pl.Series(values=baseline, dtype=pl.Float64)
        valid_ref: Union[pl.Series, pl.Expr] = temp.filter(temp.is_finite())

    vc = (
        valid_ref.qcut(n_bins, left_closed=False, allow_duplicates=True, include_breaks=True)
        .struct.rename_fields(
            ["brk", "category"]
        )  # Use "breakpoints" in the future. Skip this rename. After polars v1
        .struct.field("brk")
        .value_counts()
        .sort()
    )
    # breakpoints learned from ref
    brk = vc.struct.field("brk")  # .cast(pl.Float64)
    # counts of points in the buckets
    cnt_ref = vc.struct.field("count")  # .cast(pl.UInt32)
    psi_report = pl_plugin(
        symbol="pl_psi_report",
        args=[valid_new, brk, cnt_ref],
        changes_length=True,
    ).alias("psi_report")
    if return_report:
        return psi_report

    return psi_report.struct.field("psi_bin").sum()


def query_psi_discrete(
    new: str | pl.expr | Iterable[float],
    baseline: str | pl.expr | Iterable[float],
    return_report: bool = False,
) -> pl.Expr:
    """
    Compute the Population Stability Index between self (actual) and the reference column. The baseline
    column will be used as categories which are the basis of comparison.

    Note this assumes values in new and ref baseline discrete columns (e.g. str categories). This will
    treat each value as a distinct category and null will be treated as a category by itself. If a category
    exists in new but not in baseline, the percentage will be imputed by 0.0001. If you do not wish to include
    new distinct values in PSI calculation, you can still compute the PSI by generating the report and filtering.

    Also note that discrete columns must have the same type in order to be considered the same.

    Parameters
    ----------
    x
        The feature
    baseline
        An expression, or any iterable that can be turned into a Polars series. Usually this should
        be x's historical values
    return_report
        Whether to return a PSI report or not.

    Reference
    ---------
    https://www.listendata.com/2015/05/population-stability-index.html
    """
    if isinstance(new, (str, pl.Expr)):
        new_ = str_to_expr(new)
        temp = new_.value_counts().struct.rename_fields(["", "count"])
        new_cnt: Union[pl.Series, pl.Expr] = temp.struct.field("count")
        new_cat: Union[pl.Series, pl.Expr] = temp.struct.field("")
    else:
        temp = pl.Series(values=new)
        temp: pl.DataFrame = temp.value_counts()  # This is a df in this case
        ref_cnt: Union[pl.Series, pl.Expr] = temp.drop_in_place("count")
        ref_cat: Union[pl.Series, pl.Expr] = temp[temp.columns[0]]

    if isinstance(baseline, (str, pl.Expr)):
        base = str_to_expr(baseline)
        temp = base.value_counts().struct.rename_fields(["", "count"])
        ref_cnt: Union[pl.Series, pl.Expr] = temp.struct.field("count")
        ref_cat: Union[pl.Series, pl.Expr] = temp.struct.field("")
    else:
        temp = pl.Series(values=baseline)
        temp: pl.DataFrame = temp.value_counts()  # This is a df in this case
        ref_cnt: Union[pl.Series, pl.Expr] = temp.drop_in_place("count")
        ref_cat: Union[pl.Series, pl.Expr] = temp[temp.columns[0]]

    psi_report = pl_plugin(
        symbol="pl_psi_discrete_report",
        args=[new_cat, new_cnt, ref_cat, ref_cnt],
        changes_length=True,
    )
    if return_report:
        return psi_report

    return psi_report.struct.field("psi_bin").sum()


def query_psi_w_breakpoints(
    new: str | pl.expr | Iterable[float],
    baseline: str | pl.expr | Iterable[float],
    breakpoints: List[float],
) -> pl.Expr:
    """
    Creates a PSI report using the custom breakpoints.

    Parameters
    ----------
    baseline
        The data representing the baseline data. Any sequence of numerical values that
        can be turned into a Polars'series, or an expression representing a column will work
    actual
        The data representing the actual, observed data. Any sequence of numerical values that
        can be turned into a Polars'series, or an expression representing a column will work
    breakpoints
        The data that represents breakpoints. Input must be sorted, distinct, finite numeric values.
        This function will not cleanse the breakpoints for the user. E.g. [0.1, 0.5, 0.9] will create
        four bins: (-inf. 0.1], (0.1, 0.5], (0.5, 0.9] and (0.9, inf).
    """
    if isinstance(baseline, (str, pl.Expr)):
        x: pl.Expr = str_to_expr(baseline)
        x = x.filter(x.is_finite())
    else:
        temp = pl.Series(values=baseline)
        x: pl.Expr = pl.lit(temp.filter(temp.is_finite()))

    if isinstance(new, (str, pl.Expr)):
        y: pl.Expr = str_to_expr(new)
        y = y.filter(y.is_finite())
    else:
        temp = pl.Series(values=new)
        y: pl.Expr = pl.lit(temp.filter(temp.is_finite()))

    if len(breakpoints) == 0:
        raise ValueError("Breakpoints is empty.")

    bp = breakpoints + [float("inf")]
    return pl_plugin(
        symbol="pl_psi_w_bps",
        args=[x.rechunk(), y.rechunk(), pl.Series(values=bp)],
        changes_length=True,
    ).alias("psi_report")


def query_woe(
    x: str | pl.Expr, target: str | pl.expr | Iterable[float], n_bins: int = 10
) -> pl.Expr:
    """
    Compute the Weight of Evidence for x with respect to target. This assumes x
    is continuous. A value of 1 is added to all events/non-events
    (goods/bads) to smooth the computation.

    Currently only quantile binning strategy is implemented.

    Parameters
    ----------
    x
        The feature
    target
        The target variable. Should be 0s and 1s.
    n_bins
        The number of bins to bin the variable.

    Reference
    ---------
    https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html
    """
    if isinstance(target, (str, pl.Expr)):
        t = str_to_expr(target)
    else:
        t = pl.Series(values=target)
    xx = str_to_expr(x)
    valid = xx.filter(xx.is_finite())
    brk = valid.qcut(n_bins, left_closed=False, allow_duplicates=True).cast(pl.String)
    return pl_plugin(symbol="pl_woe_discrete", args=[brk, t], changes_length=True)


def query_woe_discrete(
    x: str | pl.Expr,
    target: Union[str | pl.Expr, Iterable[int]],
) -> pl.Expr:
    """
    Compute the Weight of Evidence for x with respect to target. This assumes x
    is discrete and castable to String. A value of 1 is added to all events/non-events
    (goods/bads) to smooth the computation.

    Parameters
    ----------
    x
        The feature
    target
        The target variable. Should be 0s and 1s.

    Reference
    ---------
    https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html
    """
    if isinstance(target, (str, pl.Expr)):
        t = str_to_expr(target)
    else:
        t = pl.Series(values=target)
    return pl_plugin(
        symbol="pl_woe_discrete",
        args=[str_to_expr(x).cast(pl.String), t],
        changes_length=True,
    )


def query_iv(
    x: str | pl.Expr,
    target: str | pl.expr | Iterable[float],
    n_bins: int = 10,
    return_sum: bool = True,
) -> pl.Expr:
    """
    Compute Information Value for x with respect to target. This assumes the variable x
    is continuous. A value of 1 is added to all events/non-events
    (goods/bads) to smooth the computation.

    Currently only quantile binning strategy is implemented.

    Parameters
    ----------
    x
        The feature. Must be numeric.
    target
        The target column. Should be 0s and 1s.
    n_bins
        The number of bins to bin x.
    return_sum
        If false, the output is a struct containing the ranges and the corresponding IVs. If true,
        it is the sum of the individual information values.

    Reference
    ---------
    https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html
    """
    if isinstance(target, (str, pl.Expr)):
        t = str_to_expr(target)
    else:
        t = pl.Series(values=target)
    xx = str_to_expr(x)
    valid = xx.filter(xx.is_finite())
    brk = valid.qcut(n_bins, left_closed=False, allow_duplicates=True).cast(pl.String)
    out = pl_plugin(symbol="pl_iv", args=[brk, t], changes_length=True)
    return out.struct.field("iv").sum() if return_sum else out


def query_iv_discrete(
    x: str | pl.Expr, target: str | pl.Expr | Iterable[int], return_sum: bool = True
) -> pl.Expr:
    """
    Compute the Information Value for x with respect to target. This assumes x
    is discrete and castable to String. A value of 1 is added to all events/non-events
    (goods/bads) to smooth the computation.

    Parameters
    ----------
    x
        The feature. The column must be castable to String
    target
        The target variable. Should be 0s and 1s.
    return_sum
        If false, the output is a struct containing the categories and the corresponding IVs. If true,
        it is the sum of the individual information values.

    Reference
    ---------
    https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html
    """
    if isinstance(target, (str, pl.Expr)):
        t = str_to_expr(target)
    else:
        t = pl.Series(values=target)
    out = pl_plugin(symbol="pl_iv", args=[str_to_expr(x).cast(pl.String), t], changes_length=True)
    return out.struct.field("iv").sum() if return_sum else out


def integrate_trapz(y: str | pl.Expr, x: float | pl.Expr) -> pl.Expr:
    """
    Integrate y along x using the trapezoidal rule. If x is not a single
    value, then x should be sorted.

    Parameters
    ----------
    y
        A column of numbers
    x
        If it is a single float, it must be positive and it will represent a uniform
        distance between points. If it is an expression, it must be sorted, does not contain
        null, and have the same length as self.
    """
    yy = str_to_expr(y).cast(pl.Float64)
    if isinstance(x, float):
        xx = pl.lit(abs(x), pl.Float64)
    else:
        xx = str_to_expr(x).cast(pl.Float64)

    return pl_plugin(
        symbol="pl_trapz",
        args=[yy, xx],
        returns_scalar=True,
    )


def convolve(
    x: str | pl.Expr,
    kernel: List[float] | "np.ndarray" | pl.Series | pl.Expr,  # noqa: F821
    fill_value: Union[float, pl.Expr] = 0.0,
    method: ConvMethod = "direct",
    mode: ConvMode = "full",
    parallel: bool = False,
) -> pl.Expr:
    """
    Performs a convolution with the given kernel(filter). The current implementation's performance is worse
    than SciPy but offers parallelization within Polars.

    For large kernels (usually kernel length > 120), convolving with FFT is faster, but for smaller kernels,
    convolving with direct method is faster.

    parameters
    ----------
    x
        A column of numbers
    kernel
        The filter for the convolution. Anything that can be turned into a Polars Series will work. All non-finite
        values will be filtered out before the convolution.
    fill_value
        Fill null values in `x` with this value. Either a float or a polars's expression representing 1 element
    method
        Either `fft` or `direct`.
    mode
        Please check the reference. One of `same`, `left` (left-aligned same), `right` (right-aligned same),
        `valid` or `full`.
    parallel
        Only applies when method is `direct`. Whether to compute the convulotion in parallel. Note that this may not
        have the expected performance when you are in group_by or other parallel context already. It is recommended
        to use this in select/with_columns context, when few expressions are being run at the same time.

    Reference
    ---------
    https://brianmcfee.net/dstbook-site/content/ch03-convolution/Modes.html
    https://en.wikipedia.org/wiki/Convolution
    """
    xx = str_to_expr(x).fill_null(fill_value).cast(pl.Float64).rechunk()  # One cont slice
    f: Union[pl.Expr, pl.Series]
    if isinstance(kernel, pl.Expr):
        f = kernel.filter(kernel.is_finite()).rechunk()  # One cont slice
    else:
        f = pl.Series(values=kernel, dtype=pl.Float64)
        f = f.filter(f.is_finite()).rechunk()  # One cont slice

    if method == "direct":
        f = f.reverse()

    return pl_plugin(
        symbol="pl_convolve",
        args=[xx, f],
        kwargs={"mode": mode, "method": method, "parallel": parallel},
        changes_length=True,
    )


def list_amax(list_col: str | pl.Expr) -> pl.Expr:
    """
    Finds the argmax of the list in this column. This is useful for

    (1) Turning sparse multiclass target into dense target.
    (2) Finding the max probability class of a multiclass classification output.
    (3) As a shortcut for expr.list.eval(pl.element().arg_max()).
    """
    return str_to_expr(list_col).list.eval(pl.element().arg_max())


def gamma(x: str | pl.Expr) -> pl.Expr:
    """
    Applies the gamma function to self. Note, this will return NaN for negative values and inf when x = 0,
    whereas SciPy's gamma function will return inf for all x <= 0.
    """
    return pl_plugin(
        args=[str_to_expr(x)],
        symbol="pl_gamma",
        is_elementwise=True,
    )


def expit(x: str | pl.Expr) -> pl.Expr:
    """
    Applies the Expit function to self. Expit(x) = 1 / (1 + e^(-x))
    """
    return pl_plugin(
        args=[str_to_expr(x)],
        symbol="pl_expit",
        is_elementwise=True,
    )


def logit(x: str | pl.Expr) -> pl.Expr:
    """
    Applies the logit function to self. Logit(x) = ln(x/(1-x)).
    Note that logit(0) = -inf, logit(1) = inf, and logit(p) for p < 0 or p > 1 yields nan.
    """
    return pl_plugin(
        args=[str_to_expr(x)],
        symbol="pl_logit",
        is_elementwise=True,
    )


def exp2(x: str | pl.Expr) -> pl.Expr:
    """
    Returns 2^x.
    """
    return pl_plugin(
        args=[str_to_expr(x)],
        symbol="pl_exp2",
        is_elementwise=True,
    )


def fract(x: str | pl.Expr) -> pl.Expr:
    """
    Returns the fractional part of the input values. E.g. fractional part of 1.1 is 0.1
    """
    return pl_plugin(
        args=[str_to_expr(x)],
        symbol="pl_fract",
        is_elementwise=True,
    )


def trunc(x: str | pl.Expr) -> pl.Expr:
    """
    Returns the integer part of the input values. E.g. integer part of 1.1 is 1.0
    """
    return pl_plugin(
        args=[str_to_expr(x)],
        symbol="pl_trunc",
        is_elementwise=True,
    )


def sinc(x: str | pl.Expr) -> pl.Expr:
    """
    Computes the sinc function normalized by pi.
    """
    xx = str_to_expr(x)
    y = math.pi * pl.when(xx == 0).then(1e-20).otherwise(xx)
    return y.sin() / y


def detrend(x: str | pl.Expr, method: DetrendMethod = "linear") -> pl.Expr:
    """
    Detrends self using either linear/mean method. This does not persist.

    Parameters
    ----------
    method
        Either `linear` or `mean`
    """
    ts = str_to_expr(x)
    if method == "linear":
        N = ts.count()
        x = pl.int_range(0, N, eager=False)
        coeff = pl.cov(ts, x) / x.var()
        const = ts.mean() - coeff * (N - 1) / 2
        return ts - x * coeff - const
    elif method == "mean":
        return ts - ts.mean()
    else:
        raise ValueError(f"Unknown detrend method: {method}")


def rfft(series: str | pl.Expr, n: int | None = None, return_full: bool = False) -> pl.Expr:
    """
    Computes the DFT transform of a real-valued input series using FFT Algorithm. Note that
    by default a series of length (length // 2 + 1) will be returned.

    Parameters
    ----------
    series
        Input real series
    n
        The number of points to use. If n is smaller than the length of the input,
        the input is cropped. If it is larger, the input is padded with zeros.
        If n is not given, the length of the input is used.
    return_full
        If true, output will have the same length as determined by n.
    """
    if n is not None and n <= 1:
        raise ValueError("Input `n` should be > 1.")

    full = pl.lit(return_full, pl.Boolean)
    nn = pl.lit(n, pl.UInt32)
    x: pl.Expr = str_to_expr(series).cast(pl.Float64)
    return pl_plugin(symbol="pl_rfft", args=[x, nn, full], changes_length=True)


def target_encode(
    s: str | pl.Expr,
    target: str | pl.Expr | Iterable[int],
    min_samples_leaf: int = 20,
    smoothing: float = 10.0,
) -> pl.Expr:
    """
    Compute information necessary to target encode a string column.

    Note: nulls will be encoded as well.

    Parameters
    ----------
    s
        The string column to encode
    target
        The target column. Should be 0s and 1s.
    min_samples_leaf
        A regularization factor
    smoothing
        Smoothing effect to balance categorical average vs prior

    Reference
    ---------
    https://contrib.scikit-learn.org/category_encoders/targetencoder.html
    """
    if isinstance(target, (str, pl.Expr)):
        t = str_to_expr(target)
    else:
        t = pl.Series(values=target)
    return pl_plugin(
        symbol="pl_target_encode",
        args=[str_to_expr(s), t, t.mean()],
        kwargs={"min_samples_leaf": float(min_samples_leaf), "smoothing": smoothing},
        changes_length=True,
    )
