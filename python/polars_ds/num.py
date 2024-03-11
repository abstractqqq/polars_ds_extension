from __future__ import annotations
import math
import polars as pl
from typing import Union, Optional, List, Iterable
from .type_alias import DetrendMethod, Distance, ConvMode, str_to_expr
from polars.utils.udfs import _get_shared_lib_location

_lib = _get_shared_lib_location(__file__)


@pl.api.register_expr_namespace("num")
class NumExt:
    """
    This class contains tools for dealing with well-known numerical operations and other metrics inside Polars DataFrame.
    All the metrics/losses provided here is meant for use in cases like evaluating models outside training,
    not for actual use in ML models.

    Polars Namespace: num

    Example: pl.col("a").num.range_over_mean()

    It currently contains some time series stuff such as detrend, rfft, and entropies, and other common numerical quantities.
    """

    def __init__(self, expr: pl.Expr):
        self._expr: pl.Expr = expr

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

    def hmean(self) -> pl.Expr:
        """
        Returns the harmonic mean of the expression
        """
        return self._expr.count() / (1.0 / self._expr).sum()

    def gmean(self) -> pl.Expr:
        """
        Returns the geometric mean of the expression
        """
        return self._expr.ln().mean().exp()

    def cv(self, ddof: int = 1) -> pl.Expr:
        """
        Returns the coefficient of variation of the expression
        """
        return self._expr.std(ddof=ddof) / self._expr.mean()

    def range_over_mean(self) -> pl.Expr:
        """
        Returns (max - min) / mean
        """
        return (self._expr.max() - self._expr.min()) / self._expr.mean()

    def z_scale(self, ddof: int = 1) -> pl.Expr:
        """
        z_normalize the given expression: remove the mean and scales by the std
        """
        return (self._expr - self._expr.mean()) / self._expr.std(ddof=ddof)

    def min_max_scale(self) -> pl.Expr:
        """
        Min max normalize the given expression.
        """
        return (self._expr - self._expr.min()) / (self._expr.max() - self._expr.min())

    def yeo_johnson(self, lam: float) -> pl.Expr:
        """
        Performs the Yeo Johnson transform with parameters lambda.

        Unfortunately, the package does not provide estimate for lambda as of now.

        Parameters
        ----------
        lam
            The lambda in Yeo Johnson transform

        Reference
        ---------
        https://en.wikipedia.org/wiki/Power_transform
        """
        x = self._expr

        if lam == 0:  # log(x + 1)
            x_ge = x.log1p()
        else:  # ((x + 1)**lmbda - 1) / lmbda
            x_ge = ((1 + x).pow(lam) - 1) / lam

        if lam == 2:  # -log(-x + 1)
            x_lt = pl.lit(-1) * (-x).log1p()
        else:  #  -((-x + 1)**(2 - lmbda) - 1) / (2 - lmbda)
            t = 2 - lam
            x_lt = -((1 - x).pow(t) - 1) / t

        return pl.when(x >= 0.0).then(x_ge).otherwise(x_lt)

    def box_cox(self, lam: float, lam2: float = 0.0) -> pl.Expr:
        """
        Performs the two-parameter Box Cox transform with parameters lambda. This
        transform is only valid for values >= -lam2. Every other value will be mapped to None.

        Unfortunately, the package does not provide estimate for lambda as of now.

        Parameters
        ----------
        lam
            The first lambda in Box Cox transform
        lam2
            The second lambda in Box Cox transform

        Reference
        ---------
        https://en.wikipedia.org/wiki/Power_transform
        """
        if lam2 == 0.0:
            x = self._expr
            cond = self._expr > 0
        else:
            x = self._expr + lam2
            cond = self._expr > -lam2

        if lam == 0.0:
            return pl.when(cond).then(x.log()).otherwise(None)
        else:
            return pl.when(cond).then((x.pow(lam) - 1) / lam).otherwise(None)

    def exp2(self) -> pl.Expr:
        """
        Returns 2^x.
        """
        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_exp2",
            is_elementwise=True,
        )

    def fract(self) -> pl.Expr:
        """
        Returns the fractional part of the input values. E.g. fractional part of 1.1 is 0.1
        """
        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_fract",
            is_elementwise=True,
        )

    def trunc(self) -> pl.Expr:
        """
        Returns the integer part of the input values. E.g. integer part of 1.1 is 1.0
        """
        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_trunc",
            is_elementwise=True,
        )

    def signum(self) -> pl.Expr:
        """
        Returns sign of the input values. Note: NaN is returned for NaN. This is faster
        and more accurate than doing pl.when(..).then().otherwise().
        """
        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_signum",
            is_elementwise=True,
        )

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

    def is_equidistant(self, tol: float = 1e-6) -> pl.Expr:
        """
        Checks if a column has equal distance between consecutive values.

        Parameters
        ----------
        tol
            Tolerance. If difference is all smaller (<=) than this, then true.
        """
        return (self._expr.diff(null_behavior="drop").abs() <= tol).all()

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

    def jaccard(self, other: pl.Expr, count_null: bool = False) -> pl.Expr:
        """
        Computes jaccard similarity between this column and the other. This will hash entire
        columns and compares the two hashsets. Note: only integer/str columns can be compared.

        Parameters
        ----------
        other
            Either an int or a Polars expression
        count_null
            Whether to count null as a distinct element.
        """
        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_jaccard",
            args=[other, pl.lit(count_null, dtype=pl.Boolean)],
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

    def lempel_ziv_complexity(self, as_ratio: bool = True) -> pl.Expr:
        """
        Computes Lempel Ziv complexity on a boolean column. Null will be mapped to False.

        Parameters
        ----------
        as_ratio : bool
            If true, return complexity / length.
        """
        out = self._expr.register_plugin(
            lib=_lib,
            symbol="pl_lempel_ziv_complexity",
            is_elementwise=False,
            returns_scalar=True,
        )
        if as_ratio:
            return out / self._expr.count()
        return out

    def cond_entropy(self, other: pl.Expr) -> pl.Expr:
        """
        See query_cond_entropy
        """
        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_conditional_entropy",
            args=[other],
            is_elementwise=False,
            returns_scalar=True,
        )

    def rel_entropy(self, other: pl.Expr) -> pl.Expr:
        """
        Computes relative entropy between self and other. (self = x, other = y).

        Parameters
        ----------
        other
            A Polars expression

        Reference
        ---------
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.rel_entr.html
        """
        return (
            pl.when((self._expr > 0) & (other > 0))
            .then(self._expr * (self._expr / other).log())
            .when((self._expr == 0) & (other >= 0))
            .then(pl.lit(0.0, dtype=pl.Float64))
            .otherwise(pl.lit(float("inf"), dtype=pl.Float64))
        )

    def kl_div(self, other: pl.Expr) -> pl.Expr:
        """
        Computes Kullback-Leibler divergence between self and other. (self = x, other = y).

        Parameters
        ----------
        other
            A Polars expression

        Reference
        ---------
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kl_div.html
        """
        return (
            pl.when((self._expr > 0) & (other > 0))
            .then(self._expr * (self._expr / other).log() - self._expr + other)
            .when((self._expr == 0) & (other >= 0))
            .then(other)
            .otherwise(pl.lit(float("inf"), dtype=pl.Float64))
        )

    def sinc(self) -> pl.Expr:
        """
        Computes the sinc function normalized by pi.
        """
        y = math.pi * pl.when(self._expr == 0).then(1e-20).otherwise(self._expr)
        return y.sin() / y

    def gamma(self) -> pl.Expr:
        """
        Applies the gamma function to self. Note, this will return NaN for negative values and inf when x = 0,
        whereas SciPy's gamma function will return inf for all x <= 0.
        """
        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_gamma",
            is_elementwise=True,
        )

    def expit(self) -> pl.Expr:
        """
        Applies the Expit function to self. Expit(x) = 1 / (1 + e^(-x))
        """
        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_expit",
            is_elementwise=True,
        )

    def logit(self) -> pl.Expr:
        """
        Applies the logit function to self. Logit(x) = ln(x/(1-x)).
        Note that logit(0) = -inf, logit(1) = inf, and logit(p) for p < 0 or p > 1 yields nan.
        """
        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_logit",
            is_elementwise=True,
        )

    def lstsq(
        self,
        *variables: pl.Expr,
        add_bias: bool = False,
        skip_null: bool = False,
        return_pred: bool = False,
    ) -> pl.Expr:
        """
        See query_lstsq
        """
        y = self._expr.cast(pl.Float64)
        if return_pred:
            return y.register_plugin(
                lib=_lib,
                symbol="pl_lstsq_pred",
                args=list(variables),
                kwargs={"bias": add_bias, "skip_null": skip_null},
                is_elementwise=True,
            )
        else:
            return y.register_plugin(
                lib=_lib,
                symbol="pl_lstsq",
                args=list(variables),
                kwargs={"bias": add_bias, "skip_null": skip_null},
                returns_scalar=True,
            )

    def lstsq_report(
        self, *variables: pl.Expr, add_bias: bool = False, skip_null: bool = False
    ) -> pl.Expr:
        """
        See query_lstsq_report
        """
        y = self._expr.cast(pl.Float64)
        return y.register_plugin(
            lib=_lib,
            symbol="pl_lstsq_report",
            args=list(variables),
            kwargs={"bias": add_bias, "skip_null": skip_null},
            changes_length=True,
        )

    def detrend(self, method: DetrendMethod = "linear") -> pl.Expr:
        """
        Detrends self using either linear/mean method. This does not persist.

        Parameters
        ----------
        method
            Either `linear` or `mean`
        """
        if method == "linear":
            N = self._expr.count()
            x = pl.int_range(0, N, eager=False)
            coeff = pl.cov(self._expr, x) / x.var()
            const = self._expr.mean() - coeff * (N - 1) / 2
            return self._expr - x * coeff - const
        elif method == "mean":
            return self._expr - self._expr.mean()
        else:
            raise ValueError(f"Unknown detrend method: {method}")

    def rfft(self, n: Optional[int] = None, return_full: bool = False) -> pl.Expr:
        """
        Computes the DFT transform of a real-valued input series using FFT Algorithm. Note that
        by default a series of length (length // 2 + 1) will be returned.

        Parameters
        ----------
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
        x: pl.Expr = self._expr.cast(pl.Float64)
        return x.register_plugin(
            lib=_lib, symbol="pl_rfft", args=[nn, full], is_elementwise=False, changes_length=True
        )

    def _knn_ptwise(
        self,
        *others: pl.Expr,
        k: int = 5,
        leaf_size: int = 32,
        dist: Distance = "l2",
        parallel: bool = False,
        return_dist: bool = False,
    ) -> pl.Expr:
        """
        See query_knn_ptwise.
        """
        if k < 1:
            raise ValueError("Input `k` must be >= 1.")

        metric = str(dist).lower()
        index: pl.Expr = self._expr.cast(pl.UInt32)
        if return_dist:
            return index.register_plugin(
                lib=_lib,
                symbol="pl_knn_ptwise_w_dist",
                args=list(others),
                kwargs={"k": k, "leaf_size": leaf_size, "metric": metric, "parallel": parallel},
                is_elementwise=True,
            )
        else:
            return index.register_plugin(
                lib=_lib,
                symbol="pl_knn_ptwise",
                args=list(others),
                kwargs={"k": k, "leaf_size": leaf_size, "metric": metric, "parallel": parallel},
                is_elementwise=True,
            )

    def _radius_ptwise(
        self,
        *others: pl.Expr,
        r: float,
        leaf_size: int = 32,
        dist: Distance = "l2",
        parallel: bool = False,
    ) -> pl.Expr:
        """
        See query_radius_ptwise.
        """
        if r <= 0.0:
            raise ValueError("Input `r` must be > 0.")
        elif isinstance(r, pl.Expr):
            raise ValueError("Input `r` must be a scalar now. Expression input is not implemented.")

        metric = str(dist).lower()
        index: pl.Expr = self._expr.cast(pl.UInt32)
        return index.register_plugin(
            lib=_lib,
            symbol="pl_query_radius_ptwise",
            args=list(others),
            kwargs={"r": r, "leaf_size": leaf_size, "metric": metric, "parallel": parallel},
            is_elementwise=True,
        )

    def _knn_pt(
        self,
        *others: pl.Expr,
        k: int = 5,
        leaf_size: int = 32,
        dist: Distance = "l2",
    ) -> pl.Expr:
        """
        See query_knn_at_pt
        """
        if k < 1:
            raise ValueError("Input `k` must be >= 1.")

        metric = str(dist).lower()
        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_knn_pt",
            args=list(others),
            kwargs={"k": k, "leaf_size": leaf_size, "metric": metric, "parallel": False},
            is_elementwise=True,
        )

    def _nb_cnt(
        self,
        *others: pl.Expr,
        leaf_size: int = 32,
        dist: Distance = "l2",
        parallel: bool = False,
    ) -> pl.Expr:
        """
        See query_nb_cnt
        """
        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_nb_cnt",
            args=list(others),
            kwargs={"k": 0, "leaf_size": leaf_size, "metric": dist, "parallel": parallel},
            is_elementwise=True,
        )

    # Rewrite of Functime's approximate entropy
    def approximate_entropy(
        self, m: int, filtering_level: float, scale_by_std: bool = True, parallel: bool = True
    ) -> pl.Expr:
        """
        Approximate sample entropies of a time series given the filtering level. It is highly
        recommended that the user impute nulls before calling this.

        If NaN/some error is returned/thrown, it is likely that:
        (1) Too little data, e.g. m + 1 > length
        (2) filtering_level or (filtering_level * std) is too close to 0 or std is null/NaN.

        Parameters
        ----------
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
        if filtering_level <= 0:
            raise ValueError("Filter level must be positive.")

        if scale_by_std:
            r: pl.Expr = filtering_level * self._expr.std()
        else:
            r: pl.Expr = pl.lit(filtering_level, dtype=pl.Float64)

        rows = self._expr.count() - m + 1
        data = [self._expr.slice(0, length=rows)]
        # See rust code for more comment on why I put m + 1 here.
        data.extend(
            self._expr.shift(-i).slice(0, length=rows).alias(f"{i}") for i in range(1, m + 1)
        )
        # More errors are handled in Rust
        return r.register_plugin(
            lib=_lib,
            symbol="pl_approximate_entropy",
            args=data,
            kwargs={"k": 0, "leaf_size": 32, "metric": "inf", "parallel": parallel},
            is_elementwise=False,
            returns_scalar=True,
        )

    # Rewrite of Functime's sample_entropy
    def sample_entropy(self, ratio: float = 0.2, m: int = 2, parallel: bool = False) -> pl.Expr:
        """
        Calculate the sample entropy of this column. It is highly
        recommended that the user impute nulls before calling this.

        If NaN/some error is returned/thrown, it is likely that:
        (1) Too little data, e.g. m + 1 > length
        (2) ratio or (ratio * std) is too close to 0 or std is null/NaN.

        Parameters
        ----------
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
        r = ratio * self._expr.std(ddof=0)
        rows = self._expr.count() - m + 1
        data = [self._expr.slice(0, length=rows)]
        # See rust code for more comment on why I put m + 1 here.
        data.extend(
            self._expr.shift(-i).slice(0, length=rows).alias(f"{i}") for i in range(1, m + 1)
        )
        # More errors are handled in Rust
        return r.register_plugin(
            lib=_lib,
            symbol="pl_sample_entropy",
            args=data,
            kwargs={"k": 0, "leaf_size": 32, "metric": "inf", "parallel": parallel},
            is_elementwise=False,
            returns_scalar=True,
        )

    def permutation_entropy(
        self,
        tau: int = 1,
        n_dims: int = 3,
        base: float = math.e,
    ) -> pl.Expr:
        """
        Computes permutation entropy.

        Parameters
        ----------
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

        if tau == 1:  # Fast track the most common use case
            return (
                pl.concat_list(self._expr, *(self._expr.shift(-i) for i in range(1, n_dims)))
                .head(self._expr.count() - n_dims + 1)
                .list.eval(pl.element().arg_sort())
                .value_counts()  # groupby and count, but returns a struct
                .struct.field("count")  # extract the field named "counts"
                .entropy(base=base, normalize=True)
            )
        else:
            return (
                pl.concat_list(
                    self._expr.gather_every(tau),
                    *(self._expr.shift(-i).gather_every(tau) for i in range(1, n_dims)),
                )
                .slice(0, length=(self._expr.count() // tau) + 1 - (n_dims // tau))
                .list.eval(pl.element().arg_sort())
                .value_counts()
                .struct.field("count")
                .entropy(base=base, normalize=True)
            )

    def woe(self, target: pl.Expr, n_bins: int = 10) -> pl.Expr:
        """
        Compute the Weight of Evidence for self with respect to target. This assumes self
        is continuous. A value of 1 is added to all events/non-events
        (goods/bads) to smooth the computation.

        Currently only quantile binning strategy is implemented.

        Parameters
        ----------
        target
            The target variable. Should be 0s and 1s.
        n_bins
            The number of bins to bin the variable.

        Reference
        ---------
        https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html
        """
        valid = self._expr.filter(self._expr.is_finite()).cast(pl.Float64)
        brk = valid.qcut(n_bins, left_closed=False, allow_duplicates=True)
        return brk.register_plugin(
            lib=_lib, symbol="pl_woe_discrete", args=[target], changes_length=True
        )

    def woe_discrete(
        self,
        target: pl.Expr,
    ) -> pl.Expr:
        """
        Compute the Weight of Evidence for self with respect to target. This assumes self
        is discrete and castable to String. A value of 1 is added to all events/non-events
        (goods/bads) to smooth the computation.

        Parameters
        ----------
        target
            The target variable. Should be 0s and 1s.

        Reference
        ---------
        https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html
        """
        return self._expr.register_plugin(
            lib=_lib, symbol="pl_woe_discrete", args=[target], changes_length=True
        )

    def iv(self, target: pl.Expr, n_bins: int = 10, return_sum: bool = True) -> pl.Expr:
        """
        Compute the Information Value for self with respect to target. This assumes the variable
        is continuous. A value of 1 is added to all events/non-events
        (goods/bads) to smooth the computation.

        Currently only quantile binning strategy is implemented.

        Parameters
        ----------
        target
            The target variable. Should be 0s and 1s.
        n_bins
            The number of bins to bin the variable.
        return_sum
            If false, the output is a struct containing the ranges and the corresponding IVs. If true,
            it is the sum of the individual information values.

        Reference
        ---------
        https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html
        """
        valid = self._expr.filter(self._expr.is_finite()).cast(pl.Float64)
        brk = valid.qcut(n_bins, left_closed=False, allow_duplicates=True)

        out = brk.register_plugin(lib=_lib, symbol="pl_iv", args=[target], changes_length=True)
        if return_sum:
            return out.struct.field("iv").sum()
        else:
            return out

    def iv_discrete(self, target: pl.Expr, return_sum: bool = True) -> pl.Expr:
        """
        Compute the Information Value for self with respect to target. This assumes self
        is discrete and castable to String. A value of 1 is added to all events/non-events
        (goods/bads) to smooth the computation.

        Parameters
        ----------
        target
            The target variable. Should be 0s and 1s.
        return_sum
            If false, the output is a struct containing the categories and the corresponding IVs. If true,
            it is the sum of the individual information values.

        Reference
        ---------
        https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html
        """
        out = self._expr.register_plugin(
            lib=_lib, symbol="pl_iv", args=[target], changes_length=True
        )
        if return_sum:
            return out.struct.field("iv").sum()
        else:
            return out

    def target_encode(
        self, target: pl.Expr, min_samples_leaf: int = 20, smoothing: float = 10.0
    ) -> pl.Expr:
        """
        Compute information necessary to target encode a string column. Target must be binary
        and be 0s and 1s.

        Parameters
        ----------
        target
            The target variable. Should be 0s and 1s.
        """
        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_target_encode",
            args=[target, target.mean()],
            kwargs={"min_samples_leaf": float(min_samples_leaf), "smoothing": smoothing},
            changes_length=True,
        )

    def psi(
        self,
        ref: Union[pl.Expr, List[float], "np.ndarray", pl.Series],  # noqa: F821
        n_bins: int = 10,
    ) -> pl.Expr:
        """
        Compute the Population Stability Index between self (actual) and the reference column. The reference
        column will be divided into n_bins quantile bins which will be used as basis of comparison.

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
        ref
            An expression, or any iterable that can be turned into a Polars series
        n_bins : int, > 1
            The number of quantile bins to use

        Reference
        ---------
        https://github.com/mwburke/population-stability-index/blob/master/psi.py
        https://www.listendata.com/2015/05/population-stability-index.html
        """
        if n_bins <= 1:
            raise ValueError("Input `n_bins` must be >= 2.")

        valid_self = self._expr.filter(self._expr.is_finite()).cast(pl.Float64)
        if isinstance(ref, pl.Expr):
            valid_ref = ref.filter(ref.is_finite()).cast(pl.Float64)
        else:
            temp = pl.Series(values=ref, dtype=pl.Float64)
            temp = temp.filter(temp.is_finite())
            valid_ref = pl.lit(temp)

        vc = (
            valid_ref.qcut(n_bins, left_closed=False, allow_duplicates=True, include_breaks=True)
            .struct.field("brk")
            .value_counts()
            .sort()
        )
        brk = vc.struct.field("brk")  # .cast(pl.Float64)
        cnt_ref = vc.struct.field("count")  # .cast(pl.UInt32)

        return valid_self.register_plugin(
            lib=_lib,
            symbol="pl_psi",
            args=[brk, cnt_ref],
            is_elementwise=False,
            returns_scalar=True,
        )

    def psi_discrete(
        self,
        ref: Union[pl.Expr, List[float], "np.ndarray", pl.Series],  # noqa: F821
    ) -> pl.Expr:
        """
        Compute the Population Stability Index between self (actual) and the reference column. The reference
        column will be used as bins which are the basis of comparison.

        Note this assumes values in self and ref are discrete columns. This will treat each value as a discrete
        category, e.g. null will be treated as a category by itself. If a category exists in actual but not in
        ref, then 0 is imputed, and 0.0001 is used to avoid numerical issue when computing psi. It is recommended
        to use for str and str column PSI comparison, or discrete numerical column PSI comparison.

        Also note that discrete columns must have the same type in order to be considered the same.

        Parameters
        ----------
        ref
            An expression, or any iterable that can be turned into a Polars series

        Reference
        ---------
        https://www.listendata.com/2015/05/population-stability-index.html
        """
        if isinstance(ref, pl.Expr):
            temp = ref.alias("__ref").value_counts()
            ref_cnt = temp.struct.field("count")
            ref_cats = temp.struct.field("__ref")
        else:
            temp = pl.Series(values=ref, dtype=pl.Float64)
            temp = temp.value_counts()  # This is a df in this case
            ref_cnt = temp.drop_in_place("count")
            ref_cats = temp[temp.columns[0]]

        vc = self._expr.alias("_self").value_counts()
        data_cats = vc.struct.field("_self")
        data_cnt = vc.struct.field("count")

        return data_cats.register_plugin(
            lib=_lib,
            symbol="pl_psi_discrete",
            args=[data_cnt, ref_cats, ref_cnt],
            is_elementwise=False,
            returns_scalar=True,
        )

    def convolve(
        self,
        other: Union[List[float], "np.ndarray", pl.Series],  # noqa: F821
        mode: ConvMode = "full",
    ) -> pl.Expr:
        """
        Performs a convolution with the filter via FFT. The current implementation's performance is worse
        than SciPy but offers parallelization within Polars Context.

        parameters
        ----------
        other
            The filter for the convolution. Anything that can be turned into a Polars Series will work.
        mode
            Please check the reference. One of `same`, `left` (left-aligned same), `right` (right-aligned same),
            `valid` or `full`.

        Reference
        ---------
        https://brianmcfee.net/dstbook-site/content/ch03-convolution/Modes.html
        """

        filter_ = pl.Series(values=other, dtype=pl.Float64)
        return self._expr.cast(pl.Float64).register_plugin(
            lib=_lib,
            symbol="pl_fft_convolve",
            args=[filter_, pl.lit(mode, dtype=pl.String)],
            changes_length=True,
        )

    def _haversine(
        self,
        x_long: pl.Expr,
        y_lat: pl.Expr,
        y_long: pl.Expr,
    ) -> pl.Expr:
        """
        Treats self as x_lat and computes haversine distance naively.
        """
        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_haversine",
            args=[x_long, y_lat, y_long],
            is_elementwise=True,
            cast_to_supertypes=True,
        )


# ----------------------------------------------------------------------------------


def haversine(
    x_lat: Union[str, pl.Expr],
    x_long: Union[str, pl.Expr],
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
    ylat = pl.lit(y_lat) if isinstance(y_lat, float) else str_to_expr(y_lat)
    ylong = pl.lit(y_long) if isinstance(y_long, float) else str_to_expr(y_long)
    return str_to_expr(x_lat).num._haversine(str_to_expr(x_long), ylat, ylong)


def query_knn_ptwise(
    *others: Union[str, pl.Expr],
    index: Union[str, pl.Expr],
    k: int = 5,
    leaf_size: int = 32,
    dist: Distance = "l2",
    parallel: bool = False,
    return_dist: bool = False,
) -> pl.Expr:
    """
    Takes the index column, and uses other columns to determine the k nearest neighbors
    to every id in the index columns. By default, this will return self, and k more neighbors.
    So the output size is actually k + 1. This will throw an error if any null value is found.

    Note that the index column must be convertible to u32. If you do not have a u32 column,
    you can generate one using pl.int_range(..), which should be a step before this. The index column
    must not contain nulls.

    Also note that this internally builds a kd-tree for fast querying and deallocates it once we
    are done. If you need to repeatedly run the same query on the same data, then it is not
    ideal to use this. A specialized external kd-tree structure would be better in that case.

    Parameters
    ----------
    *others : str | pl.Expr
        Other columns used as features
    index : str | pl.Expr
        The column used as index, must be castable to u32
    k : int
        Number of neighbors to query
    leaf_size : int
        Leaf size for the kd-tree. Tuning this might improve runtime performance.
    dist : Literal[`l1`, `l2`, `inf`, `h`, `cosine`]
        Note `l2` is actually squared `l2` for computational efficiency.
    parallel : bool
        Whether to run the k-nearest neighbor query in parallel. This is recommended when you
        are running only this expression, and not in group_by context.
    return_dist
        If true, return a struct with indices and distances.
    """
    idx = str_to_expr(index)
    return idx.num._knn_ptwise(
        *[str_to_expr(x) for x in others],
        k=k,
        leaf_size=leaf_size,
        dist=dist,
        parallel=parallel,
        return_dist=return_dist,
    )


def query_radius_at_pt(
    *others: Union[str, pl.Expr],
    pt: Iterable[float],
    r: Union[float, pl.Expr],
    dist: Distance = "l2",
) -> pl.Expr:
    """
    Returns an expression that queries the neighbors within (<=) radius from x. Note that
    this only queries around a single point x and returns a boolean column.

    Parameters
    ----------
    *others : str | pl.Expr
        Other columns used as features
    pt : Iterable[float]
        The point, at which we filter using the radius.
    r : either a float or an expression
        The radius to query with. If this is an expression, the radius will be applied row-wise.
    dist : Literal[`l1`, `l2`, `inf`, `h`, `cosine`]
        Note `l2` is actually squared `l2` for computational efficiency.
    """
    # For a single point, it is faster to just do it in native polars
    oth = [str_to_expr(x) for x in others]
    if len(pt) != len(oth):
        raise ValueError("Dimension does not match.")

    if dist == "l1":
        return (
            pl.sum_horizontal((e - pl.lit(xi, dtype=pl.Float64)).abs() for xi, e in zip(pt, oth))
            <= r
        )
    elif dist == "inf":
        return (
            pl.max_horizontal((e - pl.lit(xi, dtype=pl.Float64)).abs() for xi, e in zip(pt, oth))
            <= r
        )
    elif dist == "cosine":
        x_list = list(pt)
        x_norm = sum(z * z for z in x_list)
        oth_norm = pl.sum_horizontal([e * e for e in oth])
        dist = (
            1.0
            - pl.sum_horizontal(xi * e for xi, e in zip(x_list, oth)) / (x_norm * oth_norm).sqrt()
        )
        return dist <= r
    elif dist in ("h", "haversine"):
        x_list = list(pt)
        if (len(x_list) != 2) or (len(oth) < 2):
            raise ValueError(
                "For Haversine distance, input x must have dimension 2 and 2 other columns"
                " must be provided as lat and long."
            )

        y_lat = pl.Series(values=[x_list[0]], dtype=pl.Float64)
        y_long = pl.Series(values=[x_list[1]], dtype=pl.Float64)
        dist = oth[0].num._haversine(oth[1], y_lat, y_long)
        return dist <= r
    else:  # defaults to l2, actually squared l2
        return (
            pl.sum_horizontal((e - pl.lit(xi, dtype=pl.Float64)).pow(2) for xi, e in zip(pt, oth))
            <= r
        )


def query_radius_ptwise(
    *others: Union[str, pl.Expr],
    index: Union[str, pl.Expr],
    r: float,
    dist: Distance = "l2",
    parallel: bool = False,
) -> pl.Expr:
    """
    Takes the index column, and uses other columns to determine distance, and query all neighbors
    within distance r from each id in the index column.

    Note that the index column must be convertible to u32. If you do not have a u32 ID column,
    you can generate one using pl.int_range(..), which should be a step before this.

    Also note that this internally builds a kd-tree for fast querying and deallocates it once we
    are done. If you need to repeatedly run the same query on the same data, then it is not
    ideal to use this. A specialized external kd-tree structure would be better in that case.

    Parameters
    ----------
    *others : str | pl.Expr
        Other columns used as features
    index : str | pl.Expr
        The column used as index, must be castable to u32
    r : float
        The radius. Must be a scalar value now.
    dist : Literal[`l1`, `l2`, `inf`, `h`, `cosine`]
        Note `l2` is actually squared `l2` for computational efficiency.
    parallel : bool
        Whether to run the k-nearest neighbor query in parallel. This is recommended when you
        are running only this expression, and not in group_by context.
    """
    idx = str_to_expr(index)
    return idx.num._radius_ptwise(
        *[str_to_expr(x) for x in others], r=r, dist=dist, parallel=parallel
    )


def query_nb_cnt(
    r: Union[float, str, pl.Expr, List[float], "np.ndarray", pl.Series],  # noqa: F821
    *others: Union[str, pl.Expr],
    leaf_size: int = 32,
    dist: Distance = "l2",
    parallel: bool = False,
) -> pl.Expr:
    """
    Return the number of neighbors within (<=) radius r for each row under the given distance
    metric. The point itself is always a neighbor of itself.

    Parameters
    ----------
    r : float | Iterable[float] | pl.Expr | str
        If this is a scalar, then it will run the query with fixed radius for all rows. If
        this is a list, then it must have the same height as the dataframe. If
        this is an expression, it must be an expression representing radius. If this is a str,
        it must be the name of a column
    *others : str | pl.Expr
        Other columns used as features
    leaf_size : int, > 0
        Leaf size for the kd-tree. Tuning this might improve performance.
    dist : Literal[`l1`, `l2`, `inf`, `h`, `cosine`]
        Note `l2` is actually squared `l2` for computational efficiency.
    parallel : bool
        Whether to run the distance query in parallel. This is recommended when you
        are running only this expression, and not in group_by context.
    """
    if isinstance(r, (float, int)):
        rad = pl.lit(pl.Series(values=[r], dtype=pl.Float64))
    elif isinstance(r, pl.Expr):
        rad = r
    elif isinstance(r, str):
        rad = pl.col(r)
    else:
        rad = pl.lit(pl.Series(values=r, dtype=pl.Float64))

    return rad.num._nb_cnt(
        *[str_to_expr(x) for x in others], leaf_size=leaf_size, dist=dist, parallel=parallel
    )


def query_knn_at_pt(
    *others: Union[str, pl.Expr],
    pt: Union[List[float], "np.ndarray", pl.Series],  # noqa: F821
    k: int = 5,
    leaf_size: int = 32,
    dist: Distance = "l2",
) -> pl.Expr:
    """
    Returns an expression that queries the k nearest neighbors to a single point x.

    Note that this internally builds a kd-tree for fast querying and deallocates it once we
    are done. If you need to repeatedly run the same query on the same data, then it is not
    ideal to use this. A specialized external kd-tree structure would be better in that case.

    Parameters
    ----------
    *others : str | pl.Expr
        Other columns used as features
    pt : Iterable[float]
        The point. It must be of the same length as the number of columns in `others`.
    k : int, > 0
        Number of neighbors to query
    leaf_size : int, > 0
        Leaf size for the kd-tree. Tuning this might improve performance.
    dist : Literal[`l1`, `l2`, `inf`, `h`, `cosine`]
        Note `l2` is actually squared `l2` for computational efficiency.
    """
    if k <= 0:
        raise ValueError("Input `k` should be strictly positive.")

    pt = pl.Series(values=pt, dtype=pl.Float64)
    return pl.lit(pt).num._knn_pt(
        *[str_to_expr(x) for x in others],
        k=k,
        leaf_size=leaf_size,
        dist=dist,
    )


def query_lstsq(
    *variables: Union[str, pl.Expr],
    target: Union[str, pl.Expr],
    add_bias: bool = False,
    skip_null: bool = False,
    return_pred: bool = False,
) -> pl.Expr:
    """
    Computes least squares solution to the equation Ax = y where y is the target.

    All positional arguments should be expressions representing predictive variables. This
    does not support composite expressions like pl.col(["a", "b"]), pl.all(), etc.

    If add_bias is true, it will be the last coefficient in the output
    and output will have len(variables) + 1.

    Note: if columns are not linearly independent, some numerical issue may occur. E.g
    you may see unrealistic coefficients in the output. It is possible to have
    `silent` numerical issue during computation. Also note that if any input column contains null,
    NaNs will be returned.

    Parameters
    ----------
    variables : str | pl.Expr
        The variables used to predict target (self).
    target : str | pl.Expr
        The target variable
    add_bias
        Whether to add a bias term
    skip_null
        Whether to skip a row if there is a null value in row
    return_pred
        If true, return prediction and residue. If false, return coefficients. Note that
        for coefficients, it reduces to one output (like max/min), but for predictions and
        residue, it will return the same number of rows as in input.
    """
    t = str_to_expr(target)
    return t.num.lstsq(
        *[str_to_expr(x) for x in variables],
        add_bias=add_bias,
        skip_null=skip_null,
        return_pred=return_pred,
    )


def query_lstsq_report(
    *variables: Union[str, pl.Expr],
    target: Union[str, pl.Expr],
    add_bias: bool = False,
    skip_null: bool = False,
) -> pl.Expr:
    """
    Creates a least square report with more stats about each coefficient.

    Note: if columns are not linearly independent, some numerical issue may occur. E.g
    you may see unrealistic coefficients in the output. It is possible to have
    `silent` numerical issue during computation. For this report, input must not
    contain nulls and there must be > # features number of records. This uses the closed
    form solution to compute the least square report.

    This functions returns a struct with the same length as the number of features used
    in the linear regression, and +1 if add_bias is true.

    Parameters
    ----------
    variables : str | pl.Expr
        The variables used to predict target (self).
    target : str | pl.Expr
        The target variable
    add_bias
        Whether to add a bias term. If bias is added, it is always the last feature.
    skip_null
        Whether to skip a row if there is a null value in row
    """
    t = str_to_expr(target)
    return t.num.lstsq_report(
        *[str_to_expr(x) for x in variables], add_bias=add_bias, skip_null=skip_null
    )


def query_cond_entropy(x: Union[str, pl.Expr], y: Union[str, pl.Expr]) -> pl.Expr:
    """
    Queries the conditional entropy of x on y, aka. H(x|y).

    Parameters
    ----------
    other : str | pl.Expr
        Either a str represeting a column name or a Polars expression
    """
    return str_to_expr(x).num.cond_entropy(str_to_expr(y))
