"""Miscallaneous Numerical Functions and Transforms."""

from __future__ import annotations
import math
import polars as pl
from typing import List, Iterable, TYPE_CHECKING

# Internal dependencies
from polars_ds.typing import (
    DetrendMethod,
    ConvMode,
    ConvMethod,
)
from polars_ds._utils import pl_plugin, str_to_expr

if TYPE_CHECKING:
    from numpy import ndarray

__all__ = [
    "singular_values",
    "principal_components",
    "pca",
    "softmax",
    "gcd",
    "lcm",
    "haversine",
    "jaccard_row",
    "jaccard_col",
    "psi",
    "psi_w_breakpoints",
    "psi_discrete",
    "woe",
    "woe_discrete",
    "info_value",
    "info_value_discrete",
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
    "center",
    "z_normalize",
    "isotonic_regression",
    "is_increasing",
    "is_decreasing",
    "next_up",
    "next_down",
    "digamma",
    "xlogy",
    "l_inf_horizontal",
    "l1_horizontal",
    "l2_sq_horizontal",
    # "mutual_info_disc",
]


def l_inf_horizontal(*v: str | pl.Expr, normalize: bool = False) -> pl.Expr:
    """
    Horizontally computes L inf norm. Shorthand for pl.max_horizontal(pl.col(x).abs() for x in exprs).

    Parameters
    ----------
    *v
        Expressions to compute horizontal L infinity.
    normalize
        Whether to divide by the dimension
    """
    if normalize:
        exprs = list(v)
        return pl.max_horizontal(str_to_expr(x).abs() for x in exprs) / len(exprs)
    else:
        return pl.max_horizontal(str_to_expr(x).abs() for x in v)


def l2_sq_horizontal(*v: str | pl.Expr, normalize: bool = False) -> pl.Expr:
    """
    Horizontally computes L2 norm squared. Shorthand for pl.sum_horizontal(pl.col(x).pow(2) for x in exprs).

    Parameters
    ----------
    *v
        Expressions to compute horizontal L2.
    normalize
        Whether to divide by the dimension
    """
    if normalize:
        exprs = list(v)
        return pl.sum_horizontal(str_to_expr(x).pow(2) for x in exprs) / len(exprs)
    else:
        return pl.sum_horizontal(str_to_expr(x).pow(2) for x in v)


def l1_horizontal(*v: str | pl.Expr, normalize: bool = False) -> pl.Expr:
    """
    Horizontally computes L1 norm. Shorthand for pl.sum_horizontal(pl.col(x).abs() for x in exprs).

    Parameters
    ----------
    *v
        Expressions to compute horizontal L1.
    normalize
        Whether to divide by the dimension
    """
    if normalize:
        exprs = list(v)
        return pl.sum_horizontal(str_to_expr(x).abs() for x in exprs) / len(exprs)
    else:
        return pl.sum_horizontal(str_to_expr(x).abs() for x in v)


def is_increasing(x: str | pl.Expr, strict: bool = False) -> pl.Expr:
    """
    Checks whether the column is monotonically increasing.

    Parameters
    ----------
    x
        A numerical column
    strict
        Whether the check should be strict
    """
    if strict:
        return (str_to_expr(x).diff() > 0.0).all()
    else:
        return (str_to_expr(x).diff() >= 0.0).all()


def is_decreasing(x: str | pl.Expr, strict: bool = False) -> pl.Expr:
    """
    Checks whether the column is monotonically decreasing.

    Parameters
    ----------
    x
        A numerical column
    strict
        Whether the check should be strict
    """
    xx = str_to_expr(x)
    if strict:
        return (xx.diff() < 0.0).all()
    else:
        return (xx.diff() <= 0.0).all()


def center(x: str | pl.Expr) -> pl.Expr:
    """
    Centers the column.

    This is only a short cut for a standard feature transform, and is not recommended
    to be used in settings where the means need to be persisted.
    """
    xx = str_to_expr(x)
    return xx - xx.mean()


def z_normalize(x: str | pl.Expr) -> pl.Expr:
    """
    Z-normalizes the column.

    This is only a short cut for a standard feature transform, and is not recommended
    to be used in settings where the means/stds need to be persisted.
    """
    xx = str_to_expr(x)
    mean = xx.mean()
    std = xx.std()
    return (xx - mean) / std


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


def gcd(x: str | pl.Expr, y: int | str | pl.Expr) -> pl.Expr:
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


def lcm(x: str | pl.Expr, y: int | str | pl.Expr) -> pl.Expr:
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
    y_lat: float | str | pl.Expr,
    y_long: float | str | pl.Expr,
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


def singular_values(
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


def pca(
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

    return pl_plugin(
        symbol="pl_pca", args=actual_inputs, changes_length=True, pass_name_to_apply=True
    )


def principal_components(
    *features: str | pl.Expr,
    k: int = 2,
    center: bool = True,
) -> pl.Expr:
    """
    Transforms the features to get the first k principal components. This returns NaN if the number
    of rows is less than `k`.

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

    actual_inputs = [pl.lit(k, dtype=pl.UInt32)]
    if center:
        actual_inputs.extend(f - f.mean() for f in feats)
    else:
        actual_inputs.extend(feats)

    return pl_plugin(symbol="pl_principal_components", args=actual_inputs, pass_name_to_apply=True)


def jaccard_row(first: str | pl.Expr, second: str | pl.Expr) -> pl.Expr:
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


def jaccard_col(first: str | pl.Expr, second: str | pl.Expr, count_null: bool = False) -> pl.Expr:
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


def psi(
    new: str | pl.Expr | Iterable[float],
    baseline: str | pl.Expr | Iterable[float],
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
        valid_new = new_.filter(new_.is_finite()).cast(pl.Float64)
    else:
        temp = pl.Series(values=new, dtype=pl.Float64)
        valid_new = pl.lit(temp.filter(temp.is_finite()))

    if isinstance(baseline, (str, pl.Expr)):
        base = str_to_expr(baseline)
        valid_ref = base.filter(base.is_finite()).cast(pl.Float64)
    else:
        temp = pl.lit(pl.Series(values=baseline, dtype=pl.Float64))
        valid_ref = temp.filter(temp.is_finite())

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


def psi_discrete(
    new: str | pl.Expr | Iterable[float],
    baseline: str | pl.Expr | Iterable[float],
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
    new
        The feature
    baseline
        An expression, or any iterable that can be turned into a Polars series. Usually this should
        be the historical values
    return_report
        Whether to return a PSI report or not.

    Reference
    ---------
    https://www.listendata.com/2015/05/population-stability-index.html
    """
    if isinstance(new, (str, pl.Expr)):
        new_ = str_to_expr(new)
        temp = new_.value_counts().struct.rename_fields(["", "count"])
        new_cnt = temp.struct.field("count")
        new_cat = temp.struct.field("")
    else:
        temp = pl.Series(values=new)
        temp = temp.value_counts()  # This is a df in this case
        new_cnt = pl.lit(temp.drop_in_place("count"))
        new_cat = pl.lit(temp[temp.columns[0]])

    if isinstance(baseline, (str, pl.Expr)):
        base = str_to_expr(baseline)
        temp = base.value_counts().struct.rename_fields(["", "count"])
        ref_cnt = temp.struct.field("count")
        ref_cat = temp.struct.field("")
    else:
        temp = pl.Series(values=baseline)
        temp = temp.value_counts()  # This is a df in this case
        ref_cnt = pl.lit(temp.drop_in_place("count"))
        ref_cat = pl.lit(temp[temp.columns[0]])

    psi_report = pl_plugin(
        symbol="pl_psi_discrete_report",
        args=[new_cat, new_cnt, ref_cat, ref_cnt],
        changes_length=True,
    )
    if return_report:
        return psi_report

    return psi_report.struct.field("psi_bin").sum()


def psi_w_breakpoints(
    new: str | pl.expr | Iterable[float],
    baseline: str | pl.expr | Iterable[float],
    breakpoints: List[float],
) -> pl.Expr:
    """
    Creates a PSI report using the custom breakpoints.

    Parameters
    ----------
    new
        The data representing the new observed data. Any sequence of numerical values that
        can be turned into a Polars'series, or an expression representing a column will work
    baseline
        The data representing the baseline data. Any sequence of numerical values that
        can be turned into a Polars'series, or an expression representing a column will work
    breakpoints
        The data that represents breakpoints. Input must be sorted, distinct, finite numeric values.
        This function will not cleanse the breakpoints for the user. E.g. [0.1, 0.5, 0.9] will create
        four bins: (-inf. 0.1], (0.1, 0.5], (0.5, 0.9] and (0.9, inf). Please do not pass inf or NaN values
        as breakpoints.
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


def woe(x: str | pl.Expr, target: str | pl.expr | Iterable[float], n_bins: int = 10) -> pl.Expr:
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


def woe_discrete(
    x: str | pl.Expr,
    target: str | pl.Expr | Iterable[int],
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


def info_value(
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


def info_value_discrete(
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
    yy = str_to_expr(y).cast(pl.Float64).rechunk()
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
    kernel: List[float] | ndarray | pl.Series | pl.Expr,  # noqa: F821
    fill_value: float | pl.Expr = 0.0,
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
    f: pl.Expr | pl.Series
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


def xlogy(x: str | pl.Expr, y: str | pl.Expr) -> pl.Expr:
    """
    Computes x * log(y) so that if x = 0, the product is 0.

    Parameters
    ----------
    x
        A numerical column
    y
        A numerical column
    """
    return pl_plugin(
        args=[str_to_expr(x).cast(pl.Float64), str_to_expr(y).cast(pl.Float64)],
        symbol="pl_xlogy",
        is_elementwise=True,
    )


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
        t = pl.lit(pl.Series(values=target))
    return pl_plugin(
        symbol="pl_target_encode",
        args=[str_to_expr(s), t, t.mean()],
        kwargs={"min_samples_leaf": float(min_samples_leaf), "smoothing": smoothing},
        changes_length=True,
    )


def isotonic_regression(
    y: str | pl.Expr, weights: str | pl.Expr | None = None, increasing: bool = True
) -> pl.Expr:
    """
    Performs isotonic regression on the data. This is the same as scipy.optimize.isotonic_regression.

    Parameters
    ----------
    y
        The response variable
    weights
        The weights for the response
    increasing
        If true, output will be monotonically inreasing. If false, it will be monotonically
        decreasing.
    """

    yy = str_to_expr(y).cast(pl.Float64)
    args = [yy]
    has_weights = weights is not None
    if has_weights:
        args.append(str_to_expr(weights).cast(pl.Float64))

    return pl_plugin(
        symbol="pl_isotonic_regression",
        args=args,
        kwargs={
            "has_weights": has_weights,
            "increasing": increasing,
        },
    )


def next_up(x: str | pl.Expr) -> pl.Expr:
    """
    For any float, return the least number greater than itself (within the precision).
    Intergers will be treated as f32. E.g. The next value up for 0.1 is 0.10000000000000002
    because of precision issues. This is useful when you need to make extremely small changes
    to certain values and you don't want to add random noise.
    """
    return pl_plugin(
        symbol="pl_next_up",
        args=[str_to_expr(x)],
        is_elementwise=True,
    )


def next_down(x: str | pl.Expr) -> pl.Expr:
    """
    For any float, return the greatest number smaller than itself (within the precision).
    Intergers will be treated as f32. E.g. The next value down for 0.1 is 0.09999999999999999.
    This is useful when you need to make extremely small changes to certain values and you don't
    want to add random noise.
    """
    return pl_plugin(
        symbol="pl_next_down",
        args=[str_to_expr(x)],
        is_elementwise=True,
    )


def digamma(x: str | pl.Expr) -> pl.Expr:
    """
    The diagamma function
    """
    return pl_plugin(
        symbol="pl_diagamma",
        args=[str_to_expr(x)],
        is_elementwise=True,
    )


# Not sure what is wrong here

# def mutual_info_disc(
#     x: str | pl.Expr,
#     target: str | pl.Expr,
#     n_neighbors: int = 3,
#     seed: int | None = None,
# ) -> float:
#     """
#     Computes the mutual infomation between a continuous variable x and a discrete
#     target varaible. Note: (1) This always assume `x` is continuous. (2) Unlike Scikit-learn,
#     if a target category has <= `n_neighbors` records, then the result here may not make any sense.
#     This is because (a) `n_neighbors` is typically a small number and if your target category has so
#     few records, then maybe you need to rethink the target definition, and (b) if we assume target category
#     always has > `n_neighbors` records, then we can speed up the algorithm by quite a bit.


#     Reference
#     ---------
#     https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0087357
#     """

#     if n_neighbors <= 0:
#         raise ValueError("Input `n_neighbors` must be > 0.")

#     feat = str_to_expr(x).cast(pl.Float64)
#     feat = feat / feat.std(ddof=0)
#     t = str_to_expr(target)

#     scale_factor = pl.max_horizontal(pl.lit(1.0, dtype=pl.Float64), feat.abs().mean())

#     c = pl_plugin(
#         symbol="pl_jitter",
#         args=[feat, 1e-10 * scale_factor, pl.lit(seed, dtype=pl.UInt64)],
#         is_elementwise=True,
#     )

#     kwargs = {
#         "k": n_neighbors,
#         "metric": "l1",
#         "parallel": False,
#         "skip_eval": False,
#         "max_bound": 99999.0,
#         "epsilon": 0.
#     }

#     # This is not really exposed to the user. It does `dist_from_kth_nb` and a `next_down` in one go.
#     r = pl_plugin(
#         symbol="pl_dist_from_kth_nb",
#         args=[c],
#         kwargs=kwargs,
#     ).over(t)

#     label_counts = c.len().over(t)

#     # NB Cnt contains the point itself. This is what we want
#     # This corresponds to m_i in the paper.
#     nb_cnt = pl_plugin(
#         symbol="pl_nb_cnt",
#         args=[r, c],
#         kwargs={
#             "k": 0,
#             "metric": "l1",  # Data is 1d, l2 metric is equal to l1, but l2 does more instructions. So use l1
#             "parallel": False,
#             "skip_eval": False,
#             "skip_data": False,
#         },
#     )


#     # psi in SciPy is the diagamma function
#     psi_label_counts = pl_plugin(
#         symbol="pl_diagamma",
#         args=[label_counts],
#         is_elementwise=True,
#     )

#     psi_nb_cnt = pl_plugin(
#         symbol="pl_diagamma",
#         args=[nb_cnt],
#         is_elementwise=True,
#     )

#     psi_n_samples = pl_plugin(
#         symbol="pl_diagamma",
#         args=[pl.len()],
#         is_elementwise=True,
#     )

#     psi_n_neighbors = pl_plugin(
#         symbol="pl_diagamma",
#         args=[pl.lit(n_neighbors)],
#         is_elementwise=True,
#     )

#     return psi_n_samples + psi_n_neighbors - (psi_label_counts + psi_nb_cnt).mean()

#     return pl.max_horizontal(
#         pl.lit(0.0, dtype=pl.Float64),
#         psi_n_samples + psi_n_neighbors - psi_label_counts.mean() - psi_nb_cnt.mean(),
#     )
