import polars as pl
from typing import Union, Optional
from .type_alias import DetrendMethod
from polars.utils.udfs import _get_shared_lib_location

_lib = _get_shared_lib_location(__file__)


@pl.api.register_expr_namespace("num_ext")
class NumExt:

    """
    This class contains tools for dealing with well-known numerical operations and other metrics inside Polars DataFrame.

    Polars Namespace: num_ext

    Example: pl.col("a").num_ext.range_over_mean()

    It currently contains some time series stuff such as detrend, rfft, and time series metrics like SMAPE.
    """

    def __init__(self, expr: pl.Expr):
        self._expr: pl.Expr = expr

    def binarize(self, cond: Optional[pl.Expr]) -> pl.Expr:
        """
        Binarize the column by a boolean condition.

        Parameters
        ----------
        cond : Optional[pl.Expr]
            If cond is none, this is equivalent to self._expr == self._expr.max(). If provided,
            this will binarize by (self._expr >= cond).
        """
        if cond is None:
            return (self._expr.eq(self._expr.max())).cast(pl.UInt8)
        return (self._expr.ge(cond)).cast(pl.UInt8)

    def std_err(self, ddof: int = 1) -> pl.Expr:
        """
        Estimates the standard error for the mean of the expression.
        """
        return self._expr.std(ddof=ddof) / self._expr.count().sqrt()

    def std_over_range(self, ddof: int = 1) -> pl.Expr:
        """
        Computes the standard deviation over the range.
        """
        return self._expr.std(ddof=ddof) / (self._expr.max() - self._expr.min())

    def rms(self) -> pl.Expr:
        """
        Returns root mean square of the expression
        """
        return (self._expr.dot(self._expr) / self._expr.count()).sqrt()

    def harmonic_mean(self) -> pl.Expr:
        """
        Returns the harmonic mean of the expression
        """
        return self._expr.count() / (1.0 / self._expr).sum()

    def geometric_mean(self) -> pl.Expr:
        """
        Returns the geometric mean of the expression
        """
        return self._expr.product().pow(1.0 / self._expr.count())

    def c_o_v(self, ddof: int = 1) -> pl.Expr:
        """
        Returns the coefficient of variation of the expression
        """
        return self._expr.std(ddof=ddof) / self._expr.mean()

    def range_over_mean(self) -> pl.Expr:
        """
        Returns (max - min) / mean
        """
        return (self._expr.max() - self._expr.min()) / self._expr.mean()

    def z_normalize(self, ddof: int = 1) -> pl.Expr:
        """
        z_normalize the given expression: remove the mean and scales by the std
        """
        return (self._expr - self._expr.mean()) / self._expr.std(ddof=ddof)

    def min_max_normalize(self) -> pl.Expr:
        """
        Min max normalize the given expression.
        """
        return (self._expr - self._expr.min()) / (self._expr.max() - self._expr.min())

    def frac(self) -> pl.Expr:
        """
        Returns the fractional part of the input values. E.g. fractional part of 1.1 is 0.1
        """
        return self._expr.mod(1.0)

    def max_abs(self) -> pl.Expr:
        """
        Returns the maximum of absolute values of self.
        """
        return pl.max_horizontal(self._expr.max().abs(), self._expr.min().abs())

    def n_bins(self, n: int) -> pl.Expr:
        """
        Maps values in this series into n bins, with each bin having equal size. This ensures that
        the bins' ranges are the same, unlike quantiles. This may have tiny numerical errors but
        should be tolerable.

        Parameters
        ----------
        n
            Any positive integer
        """
        if n <= 0:
            raise ValueError("Input `n` must be positive.")

        x = self._expr
        return (
            (x - x.min()).floordiv(pl.lit(1e-12) + (x.max() - x.min()) / pl.lit(n)).cast(pl.UInt32)
        )

    def count_max(self) -> pl.Expr:
        """
        Count the number of occurrences of max.
        """
        return (self._expr == self._expr.max()).sum()

    def count_min(self) -> pl.Expr:
        """
        Count the number of occurrences of min.
        """
        return (self._expr == self._expr.min()).sum()

    def list_amax(self) -> pl.Expr:
        """
        Finds the argmax of the list in this column. This is useful for

        (1) Turning sparse multiclass target into dense target.
        (2) Finding the max probability class of a multiclass classification output.
        (3) Just a shortcut for expr.list.eval(pl.element().arg_max()).
        """
        return self._expr.list.eval(pl.element().arg_max())

    def gcd(self, other: Union[int, pl.Expr]) -> pl.Expr:
        """
        Computes GCD of two integer columns. This will try to cast everything to int64.

        Parameters
        ----------
        other
            Either an int or a Polars expression
        """
        if isinstance(other, int):
            other_ = pl.lit(other, dtype=pl.Int64)
        else:
            other_ = other.cast(pl.Int64)

        return self._expr.cast(pl.Int64).register_plugin(
            lib=_lib,
            symbol="pl_gcd",
            args=[other_],
            is_elementwise=True,
        )

    def lcm(self, other: Union[int, pl.Expr]) -> pl.Expr:
        """
        Computes LCM of two integer columns. This will try to cast everything to int64.

        Parameters
        ----------
        other
            Either an int or a Polars expression
        """
        if isinstance(other, int):
            other_ = pl.lit(other, dtype=pl.Int64)
        else:
            other_ = other.cast(pl.Int64)

        return self._expr.cast(pl.Int64).register_plugin(
            lib=_lib,
            symbol="pl_lcm",
            args=[other_],
            is_elementwise=True,
        )

    def is_equidistant(self) -> pl.Expr:
        """
        Checks if a column has equal distance between consecutive values.
        """
        return self._expr.diff(null_behavior="drop").unique_counts().count().eq(1)

    def hubor_loss(self, pred: pl.Expr, delta: float) -> pl.Expr:
        """
        Computes huber loss between this and the other expression. This assumes
        this expression is actual, and the input is predicted, although the order
        does not matter in this case.

        Parameters
        ----------
        pred
            An expression represeting the column with predicted probability.
        """
        temp = (self._expr - pred).abs()
        return (
            pl.when(temp <= delta).then(0.5 * temp.pow(2)).otherwise(delta * (temp - 0.5 * delta))
            / self._expr.count()
        )

    def mad(self, pred: pl.Expr) -> pl.Expr:
        """Computes mean absolute deivation between this and the other `pred` expression."""
        return (self._expr - pred).abs().mean()

    def l1_loss(self, pred: pl.Expr, normalize: bool = True) -> pl.Expr:
        """
        Computes L1 loss (absolute difference) between this and the other `pred` expression.

        Parameters
        ----------
        pred
            An expression represeting the column with predicted probability.
        normalize
            If true, divide the result by length of the series
        """
        temp = (self._expr - pred).abs().sum()
        if normalize:
            return temp / self._expr.count()
        return temp

    def l2_loss(self, pred: pl.Expr, normalize: bool = True) -> pl.Expr:
        """
        Computes L2 loss (normalized L2 distance) between this and the other `pred` expression. This
        is the norm without 1/p power.

        Parameters
        ----------
        pred
            An expression represeting the column with predicted probability.
        normalize
            If true, divide the result by length of the series
        """
        temp = self._expr - pred
        temp = temp.dot(temp)
        if normalize:
            return temp / self._expr.count()
        return temp

    def msle(self, pred: pl.Expr, normalize: bool = True) -> pl.Expr:
        """
        Computes the mean square log error between this and the other `pred` expression.

        Parameters
        ----------
        pred
            An expression represeting the column with predicted probability.
        normalize
            If true, divide the result by length of the series
        """
        diff = self._expr.log1p() - pred.log1p()
        out = diff.dot(diff)
        if normalize:
            return out / self._expr.count()
        return out

    def chebyshev_loss(self, pred: pl.Expr, normalize: bool = True) -> pl.Expr:
        """
        Alias for l_inf_loss.
        """
        return self.l_inf_dist(pred, normalize)

    def l_inf_loss(self, pred: pl.Expr, normalize: bool = True) -> pl.Expr:
        """
        Computes L^infinity loss between this and the other `pred` expression

        Parameters
        ----------
        pred
            An expression represeting the column with predicted probability.
        normalize
            If true, divide the result by length of the series
        """
        temp = self._expr - pred
        out = pl.max_horizontal(temp.min().abs(), temp.max().abs())
        if normalize:
            return out / self._expr.count()
        return out

    def mape(self, pred: pl.Expr, weighted: bool = False) -> pl.Expr:
        """
        Computes mean absolute percentage error between self and the other `pred` expression.
        If weighted, it will compute the weighted version as defined here:

        https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        Parameters
        ----------
        pred
            An expression represeting the column with predicted probability.
        weighted
            If true, computes wMAPE in the wikipedia article
        """
        if weighted:
            return (self._expr - pred).abs().sum() / self._expr.abs().sum()
        else:
            return (1 - pred / self._expr).abs().mean()

    def smape(self, pred: pl.Expr) -> pl.Expr:
        """
        Computes symmetric mean absolute percentage error between self and other `pred` expression.
        The value is always between 0 and 1. This is the third version in the wikipedia without
        the 100 factor.

        https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error

        Parameters
        ----------
        pred
            A Polars expression representing predictions
        """
        numerator = (self._expr - pred).abs()
        denominator = 1.0 / (self._expr.abs() + pred.abs())
        return (1.0 / self._expr.count()) * numerator.dot(denominator)

    def logloss(self, pred: pl.Expr, normalize: bool = True) -> pl.Expr:
        """
        Computes log loss, aka binary cross entropy loss, between self and other `pred` expression.

        Parameters
        ----------
        pred
            An expression represeting the column with predicted probability.
        normalize
            Whether to divide by N.
        """
        out = self._expr.dot(pred.log()) + (1 - self._expr).dot((1 - pred).log())
        if normalize:
            return -(out / self._expr.count())
        return -out

    def bce(self, pred: pl.Expr, normalize: bool = True) -> pl.Expr:
        """
        Binary cross entropy. Alias for logloss.
        """
        return self.logloss(pred, normalize)

    def roc_auc(self, pred: pl.Expr) -> pl.Expr:
        """
        Computes ROC AUC using self as actual and pred as predictions.

        Self must be binary and castable to type UInt32. If self is not all 0s and 1s or not binary,
        the result will not make sense, or some error may occur.

        Parameters
        ----------
        pred
            An expression represeting the column with predicted probability.
        """
        y = self._expr.cast(pl.UInt32)
        return y.register_plugin(
            lib=_lib,
            symbol="pl_roc_auc",
            args=[pred],
            is_elementwise=False,
            returns_scalar=True,
        )

    def binary_metrics_combo(self, pred: pl.Expr, threshold: float = 0.5) -> pl.Expr:
        """
        Computes the following binary classificaition metrics using self as actual and pred as predictions:
        precision, recall, f, average_precision and roc_auc. The return will be a struct with values
        having the names as given here.

        Self must be binary and castable to type UInt32. If self is not all 0s and 1s,
        the result will not make sense, or some error may occur.

        Average precision is computed using Sum (R_n - R_n-1)*P_n-1, which is not the textbook definition,
        but is consistent with Scikit-learn. For more information, see
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html

        Parameters
        ----------
        pred
            An expression represeting the column with predicted probability.
        threshold
            The threshold used to compute precision, recall and f (f score).
        """
        y = self._expr.cast(pl.UInt32)
        return y.register_plugin(
            lib=_lib,
            symbol="pl_combo_b",
            args=[pred, pl.lit(threshold, dtype=pl.Float64)],
            is_elementwise=False,
            returns_scalar=True,
        )

    def trapz(self, x: Union[float, pl.Expr]) -> pl.Expr:
        """
        Treats self as y axis, integrates along x using the trapezoidal rule. If x is not a single
        value, then x should be sorted.

        Parameters
        ----------
        x
            If it is a single float, it must be positive and it will represent a uniform
            distance between points. If it is an expression, it must be sorted, does not contain
            null, and have the same length as self.
        """
        y = self._expr.cast(pl.Float64)
        if isinstance(x, float):
            x_ = pl.lit(abs(x), pl.Float64)
        else:
            x_ = x.cast(pl.Float64)

        return y.register_plugin(
            lib=_lib,
            symbol="pl_trapz",
            args=[x_],
            is_elementwise=False,
            returns_scalar=True,
        )

    def r2(self, pred: pl.Expr) -> pl.Expr:
        """
        Returns the coefficient of determineation for a regression model.

        Parameters
        ----------
        pred
            A Polars expression representing predictions
        """
        diff = self._expr - pred
        ss_res = diff.dot(diff)
        diff2 = self._expr - self._expr.mean()
        ss_tot = diff2.dot(diff2)
        return 1.0 - ss_res / ss_tot

    def adjusted_r2(self, pred: pl.Expr, p: int) -> pl.Expr:
        """
        Returns the adjusted r2 for a regression model.

        Parameters
        ----------
        pred
            A Polars expression representing predictions
        p
            The total number of explanatory variables in the model
        """
        diff = self._expr - pred
        ss_res = diff.dot(diff)
        diff2 = self._expr - self._expr.mean()
        ss_tot = diff2.dot(diff2)
        df_res = self._expr.count() - p
        df_tot = self._expr.count() - 1
        return 1.0 - (ss_res / df_res) / (ss_tot / df_tot)

    # Not recommended to use
    def powi(self, n: Union[int, pl.Expr]) -> pl.Expr:
        """
        Computes positive integer power using the fast exponentiation algorithm. This is the
        fastest when n is an integer input (Faster than Polars's builtin when n >= 16). When n
        is an expression, it would depend on values in the expression (Still researching...)

        Parameters
        ----------
        n
            A single positive int or an expression representing a column of type i32. If type is
            not i32, an error will occur.
        """

        if isinstance(n, int):
            n_ = pl.lit(n, pl.Int32)
        else:
            n_ = n

        return self._expr.register_plugin(
            lib=_lib, symbol="pl_fast_exp", args=[n_], is_elementwise=True, returns_scalar=False
        )

    def jaccard(self, other: pl.Expr, include_null: bool = False) -> pl.Expr:
        """
        Computes jaccard similarity between this column and the other. This will hash entire
        columns and compares the two hashsets. Note: only integer/str columns can be compared.
        Input expressions must represent columns of the same dtype.

        Parameters
        ----------
        other
            Either an int or a Polars expression
        include_null
            Whether to include null as a distinct element.
        """
        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_jaccard",
            args=[other, pl.lit(include_null, dtype=pl.Boolean)],
            is_elementwise=False,
            returns_scalar=True,
        )

    def list_jaccard(self, other: pl.Expr) -> pl.Expr:
        """
        Computes jaccard similarity pairwise between this and the other column. The type of
        each column must be list and the lists must have the same inner type. The inner type
        must either be integer or string.

        Parameters
        ----------
        other
            Either an int or a Polars expression
        include_null : to be added
            Currently there are some technical issue with adding this parameter.
        """
        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_list_jaccard",
            args=[other],
            is_elementwise=True,
        )

    def cond_entropy(self, other: pl.Expr) -> pl.Expr:
        """
        Computes the conditional entropy of self(y) given other, aka. H(y|other).

        Parameters
        ----------
        other
            A Polars expression
        """

        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_conditional_entropy",
            args=[other],
            is_elementwise=False,
            returns_scalar=True,
        )

    def lstsq(self, *others: pl.Expr, add_bias: bool = False) -> pl.Expr:
        """
        Computes least squares solution to the equation Ax = y. If columns are
        not linearly independent, some numerical issue may occur. E.g you may see
        unrealistic coefficient in the output. It is possible to have `silent` numerical
        issue during computation.

        All positional arguments should be expressions representing predictive variables. This
        does not support composite expressions like pl.col(["a", "b"]), pl.all(), etc.

        If add_bias is true, it will be the last coefficient in the output
        and output will have len(others) + 1

        Parameters
        ----------
        others
            Polars expressions.
        add_bias
            Whether to add a bias term
        """
        y = self._expr.cast(pl.Float64)
        return y.register_plugin(
            lib=_lib,
            symbol="pl_lstsq",
            args=[pl.lit(add_bias, dtype=pl.Boolean)] + list(others),
            is_elementwise=False,
            returns_scalar=True,
        )

    def detrend(self, method: DetrendMethod = "linear") -> pl.Expr:
        """
        Detrends self using either linear/mean method.

        Parameters
        ----------
        method
            Either `linear` or `mean`
        """

        if method == "linear":
            N = self._expr.count()
            x = pl.int_range(0, N, dtype=pl.Float64, eager=False)
            coeff = pl.cov(self._expr, x) / x.var()
            const = self._expr.mean() - coeff * (N - 1) / 2
            return self._expr - x * coeff - const
        elif method == "mean":
            return self._expr - self._expr.mean()
        else:
            raise ValueError(f"Unknown detrend method: {method}")

    def rfft(self, length: Optional[int] = None) -> pl.Expr:
        """
        Computes the DFT transform of a real-valued input series using FFT Algorithm. Note that
        a series of length (length // 2 + 1) will be returned.

        Parameters
        ----------
        length
            A positive integer
        """
        # Add a k step argument?
        if length is not None and length <= 1:
            raise ValueError("Input `length` should be > 1.")

        le = pl.lit(length, dtype=pl.UInt32)
        x: pl.Expr = self._expr.cast(pl.Float64)
        return x.register_plugin(
            lib=_lib, symbol="pl_rfft", args=[le], is_elementwise=False, changes_length=True
        )
