from __future__ import annotations
import math
import polars as pl
from typing import Union, Optional, List
from .type_alias import DetrendMethod, Distance
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
        Computes Lempel Ziv complexity on a boolean column. None will be mapped to False.

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
        self, *variables: pl.Expr, add_bias: bool = False, return_pred: bool = False
    ) -> pl.Expr:
        """
        Computes least squares solution to the equation Ax = y by treating self as y.

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
        variables
            The variables used to predict target (self).
        add_bias
            Whether to add a bias term
        return_pred
            If true, return prediction and residue. If false, return coefficients. Note that
            for coefficients, it reduces to one output (like max/min), but for predictions and
            residue, it will return the same number of rows as in input.
        """
        y = self._expr.cast(pl.Float64)
        if return_pred:
            return y.register_plugin(
                lib=_lib,
                symbol="pl_lstsq_pred",
                args=list(variables) + [pl.lit(add_bias, dtype=pl.Boolean)],
                is_elementwise=True,
            )
        else:
            return y.register_plugin(
                lib=_lib,
                symbol="pl_lstsq",
                args=list(variables) + [pl.lit(add_bias, dtype=pl.Boolean)],
                is_elementwise=False,
                returns_scalar=True,
            )

    def lstsq_report(self, *variables: pl.Expr, add_bias: bool = False) -> pl.Expr:
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
        variables
            The variables used to predict target (self).
        add_bias
            Whether to add a bias term. If bias is added, it is always the last feature.
        """
        y = self._expr.cast(pl.Float64)
        return y.register_plugin(
            lib=_lib,
            symbol="pl_lstsq_report",
            args=list(variables) + [pl.lit(add_bias, dtype=pl.Boolean)],
            is_elementwise=False,
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

    def knn_ptwise(
        self,
        *others: pl.Expr,
        k: int = 5,
        leaf_size: int = 40,
        dist: Distance = "l2",
        parallel: bool = False,
        return_dist: bool = False,
    ) -> pl.Expr:
        """
        Treats self as an index column, and uses other columns to determine the k nearest neighbors
        to every id. By default, this will return self, and k more neighbors. So the output size
        is actually k + 1. This will throw an error if any null value is found.

        Note that the node index column must be convertible to u64. If you do not have a u64 ID column,
        you can generate one using pl.int_range(..), which should be a step before this. The index column
        must not contain nulls.

        Also note that this internally builds a kd-tree for fast querying and deallocates it once we
        are done. If you need to repeatedly run the same query on the same data, then it is not
        ideal to use this. A specialized external kd-tree structure would be better in that case.

        Parameters
        ----------
        others
            Other columns used as features
        k
            Number of neighbors to query
        leaf_size
            Leaf size for the kd-tree. Tuning this might improve runtime performance.
        dist
            One of `l1`, `l2`, `inf`, `cosine` or `h` or `haversine`, where h stands for haversine. Note
            `l2` is actually squared `l2` for computational efficiency. It defaults to `l2`.
        parallel
            Whether to run the k-nearest neighbor query in parallel. This is recommended when you
            are running only this expression, and not in group_by context.
        return_dist
            If true, return a struct with indices and distances.
        """
        if k < 1:
            raise ValueError("Input `k` must be >= 1.")

        metric = str(dist).lower()
        index: pl.Expr = self._expr.cast(pl.UInt64)
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

    def query_radius_ptwise(
        self,
        *others: pl.Expr,
        r: float,
        leaf_size: int = 40,
        dist: Distance = "l2",
        parallel: bool = False,
    ) -> pl.Expr:
        """
        Treats self as an index column, and uses other columns to determine distance, and query all neighbors
        within distance r from each node in ID.

        Note that the index column must be convertible to u64. If you do not have a u64 ID column,
        you can generate one using pl.int_range(..), which should be a step before this.

        Also note that this internally builds a kd-tree for fast querying and deallocates it once we
        are done. If you need to repeatedly run the same query on the same data, then it is not
        ideal to use this. A specialized external kd-tree structure would be better in that case.

        Parameters
        ----------
        others
            Other columns used as features
        r
            The radius. Must be a scalar value now.
        leaf_size
            Leaf size for the kd-tree. Tuning this might improve runtime performance.
        dist
            One of `l1`, `l2`, `inf`, `cosine` or `h` or `haversine`, where h stands for haversine. Note
            `l2` is actually squared `l2` for computational efficiency. It defaults to `l2`.
        parallel
            Whether to run the k-nearest neighbor query in parallel. This is recommended when you
            are running only this expression, and not in group_by context.
        """
        if r <= 0.0:
            raise ValueError("Input `r` must be > 0.")
        elif isinstance(r, pl.Expr):
            raise ValueError("Input `r` must be a scalar now. Expression input is not implemented.")

        metric = str(dist).lower()
        index: pl.Expr = self._expr.cast(pl.UInt64)
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
        leaf_size: int = 40,
        dist: Distance = "l2",
    ) -> pl.Expr:
        """
        Treats self as a point, and uses other columns to filter to the k nearest neighbors
        to self. The recommendation is to use the knn function in polars_ds.

        Note that this internally builds a kd-tree for fast querying and deallocates it once we
        are done. If you need to repeatedly run the same query on the same data, then it is not
        ideal to use this. A specialized external kd-tree structure would be better in that case.

        Parameters
        ----------
        others
            Other columns used as features
        k
            Number of neighbors to query
        leaf_size
            Leaf size for the kd-tree. Tuning this might improve performance.
        dist
            One of `l1`, `l2`, `inf`, `cosine` or `h` or `haversine`, where h stands for haversine. Note
            `l2` is actually squared `l2` for computational efficiency. It defaults to `l2`.
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
        leaf_size: int = 40,
        dist: Distance = "l2",
        parallel: bool = False,
    ) -> pl.Expr:
        """
        Treats self as radius, which can be a scalar, or a column with the same length as the columns
        in `others`. This will return the number of neighbors within (<=) distance for each row.
        The recommendation is to use the nb_cnt function in polars_ds.

        Parameters
        ----------
        others
            Other columns used as features
        leaf_size
            Leaf size for the kd-tree. Tuning this might improve performance.
        dist
            One of `l1`, `l2`, `inf`, `cosine` or `h` or `haversine`, where h stands for haversine. Note
            `l2` is actually squared `l2` for computational efficiency. It defaults to `l2`.
        parallel
            Whether to run the k-nearest neighbor query in parallel. This is recommended when you
            are running only this expression, and not in group_by context.
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
            kwargs={"k": 0, "leaf_size": 50, "metric": "inf", "parallel": parallel},
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
            kwargs={"k": 0, "leaf_size": 50, "metric": "inf", "parallel": parallel},
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

    def woe(self, variable: pl.Expr, n_bins: int = 10) -> pl.Expr:
        """
        Compute the Weight Of Evidence for the variable by treating self as the binary target of 0s
        and 1s. This assumes the variable is continuous. The output is a struct containing the ranges
        and the corresponding WOEs. A value of 1 is added to all events/non-events (goods/bads)
        to smooth the computation.

        Currently only quantile binning strategy is implemented.

        Parameters
        ----------
        variable
            The variable whose WOE you want to compute
        n_bins
            The number of bins to bin the variable.

        Reference
        ---------
        https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html
        """
        valid = variable.filter(variable.is_finite()).cast(pl.Float64)
        brk = valid.qcut(n_bins, left_closed=False, allow_duplicates=True)
        return self._expr.register_plugin(
            lib=_lib, symbol="pl_woe_discrete", args=[brk], changes_length=True
        )

    def woe_discrete(
        self,
        discrete_var: pl.Expr,
    ) -> pl.Expr:
        """
        Compute the Weight Of Evidence for the variable by treating self as the binary target of 0s
        and 1s. This assumes the variable is discrete. The output is a struct containing the categories
        and the corresponding WOEs. A value of 1 is added to all events/non-events (goods/bads)
        to smooth the computation.

        Parameters
        ----------
        discrete_var
            The variable whose WOE you want to compute

        Reference
        ---------
        https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html
        """
        return self._expr.register_plugin(
            lib=_lib, symbol="pl_woe_discrete", args=[discrete_var], changes_length=True
        )

    def iv(self, variable: pl.Expr, n_bins: int = 10, return_sum: bool = True) -> pl.Expr:
        """
        Compute the Information Value for the variable by treating self as the binary target of 0s
        and 1s. This assumes the variable is continuous. A value of 1 is added to all events/non-events
        (goods/bads) to smooth the computation.

        Currently only quantile binning strategy is implemented.

        Parameters
        ----------
        variable
            The variable whose IV you want to compute
        n_bins
            The number of bins to bin the variable.
        return_sum
            If false, the output is a struct containing the ranges and the corresponding IVs. If true,
            it is the sum of the individual information values.

        Reference
        ---------
        https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html
        """
        valid = variable.filter(variable.is_finite()).cast(pl.Float64)
        brk = valid.qcut(n_bins, left_closed=False, allow_duplicates=True)

        out = self._expr.register_plugin(lib=_lib, symbol="pl_iv", args=[brk], changes_length=True)
        if return_sum:
            return out.struct.field("iv").sum()
        else:
            return out

    def iv_discrete(self, discrete_var: pl.Expr, return_sum: bool = True) -> pl.Expr:
        """
        Compute the Information Value for the variable by treating self as the binary target of 0s
        and 1s. This assumes the variable is discrete. A value of 1 is added to all events/non-events
        (goods/bads) to smooth the computation.

        Parameters
        ----------
        discrete_var
            The variable whose IV you want to compute
        return_sum
            If false, the output is a struct containing the categories and the corresponding IVs. If true,
            it is the sum of the individual information values.

        Reference
        ---------
        https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html
        """
        out = self._expr.register_plugin(
            lib=_lib, symbol="pl_iv", args=[discrete_var], changes_length=True
        )
        if return_sum:
            return out.struct.field("iv").sum()
        else:
            return out

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
