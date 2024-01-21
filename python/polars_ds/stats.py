import polars as pl
from .type_alias import Alternative
from typing import Optional, Union
from polars.utils.udfs import _get_shared_lib_location
# from polars.type_aliases import IntoExpr

_lib = _get_shared_lib_location(__file__)


@pl.api.register_expr_namespace("stats")
class StatsExt:

    """
    This class contains tools for dealing with well-known statistical tests and random sampling inside Polars DataFrame.

    Polars Namespace: stats

    Example: pl.col("a").stats.ttest_ind(pl.col("b"), equal_var = True)
    """

    def __init__(self, expr: pl.Expr):
        self._expr: pl.Expr = expr

    def ttest_ind(
        self, other: pl.Expr, alternative: Alternative = "two-sided", equal_var: bool = False
    ) -> pl.Expr:
        """
        Performs 2 sample student's t test or Welch's t test. Functionality-wise this is desgined
        to be equivalent to SciPy's ttest_ind, with fewer options. The result is not exact but
        within 1e-10 precision from SciPy's.

        In the case of student's t test, the data is assumed to have no nulls, and n = self._expr.count()
        is used. Note self._expr.count() only counts non-null elements after polars 0.20.
        The degree of freedom will be 2n - 2. As a result, nulls might cause problems.

        In the case of Welch's t test, data will be sanitized (nulls, NaNs, Infs will be dropped
        before the test), and df will be counted based on the length of sanitized data.

        Parameters
        ----------
        other
            The other expression
        alternative : {"two-sided", "less", "greater"}
            Alternative of the hypothesis test
        equal_var
            If true, perform standard student t 2 sample test. Otherwise, perform Welch's
            t test.
        """
        if equal_var:
            m1 = self._expr.mean()
            m2 = other.mean()
            v1 = self._expr.var()
            v2 = other.var()
            cnt = self._expr.count().cast(pl.UInt64)
            return m1.register_plugin(
                lib=_lib,
                symbol="pl_ttest_2samp",
                args=[m2, v1, v2, cnt, pl.lit(alternative, dtype=pl.String)],
                is_elementwise=False,
                returns_scalar=True,
            )
        else:
            s1 = self._expr.filter(self._expr.is_finite())
            s2 = other.filter(other.is_finite())
            m1 = s1.mean()
            m2 = s2.mean()
            v1 = s1.var()
            v2 = s2.var()
            n1 = s1.count().cast(pl.UInt64)
            n2 = s2.count().cast(pl.UInt64)
            return m1.register_plugin(
                lib=_lib,
                symbol="pl_welch_t",
                args=[m2, v1, v2, n1, n2, pl.lit(alternative, dtype=pl.String)],
                is_elementwise=False,
                returns_scalar=True,
            )

    def ttest_1samp(self, pop_mean: float, alternative: Alternative = "two-sided") -> pl.Expr:
        """
        Performs a standard 1 sample t test using reference column and expected mean. This function
        sanitizes the self column first. The df is the count of valid values.

        Parameters
        ----------
        pop_mean
            The expected population mean in the hypothesis test
        alternative : {"two-sided", "less", "greater"}
            Alternative of the hypothesis test
        """
        s1 = self._expr.filter(self._expr.is_finite())
        sm = s1.mean()
        pm = pl.lit(pop_mean, dtype=pl.Float64)
        var = s1.var()
        cnt = s1.count().cast(pl.UInt64)
        alt = pl.lit(alternative, dtype=pl.String)
        return sm.register_plugin(
            lib=_lib,
            symbol="pl_ttest_1samp",
            args=[pm, var, cnt, alt],
            is_elementwise=False,
            returns_scalar=True,
        )

    def f_stats(self, *variables: pl.Expr) -> pl.Expr:
        """
        Computes multiple F statistics at once by using self as the grouping (class) column. This
        does not output p values. If the p value is desired, use `f_test`. This will return
        all the stats as a scalar list in order.

        Parameters
        ----------
        *variables
            The variables (Polars df columns) to compute the F statistics
        """
        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_f_stats",
            args=list(variables),
            is_elementwise=False,
            returns_scalar=True,
        )

    def f_test(self, var: pl.Expr) -> pl.Expr:
        """
        Performs the ANOVA F-test using self as the grouping column.

        Parameters
        ----------
        var
            The column to run ANOVA F-test on
        """
        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_f_test",
            args=[var],
            is_elementwise=False,
            returns_scalar=True,
        )

    def normal_test(self) -> pl.Expr:
        """
        Perform a normality test which is based on D'Agostino and Pearson's test
        that combines skew and kurtosis to produce an omnibus test of normality.
        Null values, NaN and inf are dropped when running this computation.

        References
        ----------
        D'Agostino, R. B. (1971), "An omnibus test of normality for
            moderate and large sample size", Biometrika, 58, 341-348
        D'Agostino, R. and Pearson, E. S. (1973), "Tests for departure from
            normality", Biometrika, 60, 613-622
        """
        valid: pl.Expr = self._expr.filter(self._expr.is_finite())
        skew = valid.skew()
        # Pearson Kurtosis, see here: https://en.wikipedia.org/wiki/D%27Agostino%27s_K-squared_test
        kur = valid.kurtosis(fisher=False)
        return skew.register_plugin(
            lib=_lib,
            symbol="pl_normal_test",
            args=[kur, valid.count().cast(pl.UInt64)],
            is_elementwise=False,
            returns_scalar=True,
        )

    def ks_stats(self, var: pl.Expr) -> pl.Expr:
        """
        Computes two-sided KS statistics with other. Currently it only returns the statistics. This will
        sanitize data (only non-null finite values are used) before doing the computation.

        Parameters
        ----------
        var
            The column to run KS statistic on
        """
        y = self._expr.filter(self._expr.is_finite()).cast(pl.Float64)
        other_ = var.filter(var.is_finite()).cast(pl.Float64)
        return y.register_plugin(
            lib=_lib,
            symbol="pl_ks_2samp",
            args=[other_, pl.lit(True, dtype=pl.Boolean)],
            is_elementwise=False,
            returns_scalar=True,
        )

    def ks_binary_classif(self, target: pl.Expr) -> pl.Expr:
        """
        Given a binary target, compute the ks statistics by comparing the feature where target = 1
        with the same feature where target != 1.

        Parameters
        ----------
        target
            A Polars Expression representing the binary target
        """
        y = self._expr.cast(pl.Float64)
        y1 = y.filter(target == target.max())
        y2 = y.filter((target == target.max()).not_())
        return y1.register_plugin(
            lib=_lib,
            symbol="pl_ks_2samp",
            args=[y2, pl.lit(True, dtype=pl.Boolean)],
            is_elementwise=False,
            returns_scalar=True,
        )

    def chi2(self, var: pl.Expr) -> pl.Expr:
        """
        Computes the Chi Squared statistic and p value between two categorical values by treating
        self as the first column and var as the second.

        Note that it is up to the user to make sure that the two columns contain categorical
        values. This method is equivalent to SciPy's chi2_contingency, except that it also
        computes the contingency table internally for the user.

        Parameters
        ----------
        var
            The second column to run chi squared test on
        """

        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_chi2",
            args=[var],
            is_elementwise=False,
            returns_scalar=True,
        )

    def rand_int(
        self,
        low: Union[int, pl.Expr] = 0,
        high: Optional[Union[int, pl.Expr]] = 10,
        respect_null: bool = False,
        use_ref: bool = False,
    ) -> pl.Expr:
        """
        Generates random integers uniformly from the range [low, high). Throws an error if low == high
        or if low is None and high is None.

        This treats self as the reference column.

        Parameters
        ----------
        low
            Lower end of random sample. If high is none, low will be set to 0.
        high
            Higher end of random sample. If this is None, then it will be replaced n_unique of reference.
        respect_null
            If true, null in reference column will be null in the new column
        """

        if high is None:
            lo = pl.lit(0, dtype=pl.Int32)
            hi = self._expr.n_unique.cast(pl.UInt32)
        else:
            if isinstance(low, pl.Expr):
                lo = low
            elif isinstance(low, int):
                lo = pl.lit(low, dtype=pl.Int32)
            else:
                raise ValueError("Input `low` must be expression or int.")

            if isinstance(high, pl.Expr):
                hi = high
            elif isinstance(high, int):
                hi = pl.lit(high, dtype=pl.Int32)
            else:
                raise ValueError("Input `high` must be expression or int.")

        resp = pl.lit(respect_null, dtype=pl.Boolean)
        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_rand_int",
            args=[lo, hi, resp],
            is_elementwise=True,
            returns_scalar=False,
        )

    def sample_uniform(
        self,
        low: Optional[Union[float, pl.Expr]] = None,
        high: Optional[Union[float, pl.Expr]] = None,
        respect_null: bool = False,
    ) -> pl.Expr:
        """
        Creates self.len() many random points sampled from a uniform distribution within [low, high).
        This will throw an error if low == high.

        This treats self as the reference column.

        Parameters
        ----------
        low
            Lower end of random sample. If none, use reference col's min.
        high
            Higher end of random sample. If none, use reference col's max.
        respect_null
            If true, null in reference column will be null in the new column
        """

        if isinstance(low, pl.Expr):
            lo = low
        elif isinstance(low, float):
            lo = pl.lit(low, dtype=pl.Float64)
        else:
            lo = self._expr.min()

        if isinstance(high, pl.Expr):
            hi = high
        elif isinstance(high, float):
            hi = pl.lit(high, dtype=pl.Float64)
        else:
            hi = self._expr.max()

        resp = pl.lit(respect_null, dtype=pl.Boolean)
        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_sample_uniform",
            args=[lo, hi, resp],
            is_elementwise=True,
            returns_scalar=False,
        )

    def sample_binomial(self, n: int, p: float, respect_null: bool = False) -> pl.Expr:
        """
        Creates self.len() many random points sampled from a binomial distribution with n and p.

        This treats self as the reference column.

        Parameters
        ----------
        n
            n in a binomial distribution
        p
            p in a binomial distribution
        respect_null
            If true, null in reference column will be null in the new column
        """

        nn = pl.lit(n, dtype=pl.UInt64)
        pp = pl.lit(p, dtype=pl.Float64)
        resp = pl.lit(respect_null, dtype=pl.Boolean)
        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_sample_binomial",
            args=[nn, pp, resp],
            is_elementwise=True,
            returns_scalar=False,
        )

    def sample_exp(
        self, lambda_: Optional[Union[float, pl.Expr]] = None, respect_null: bool = False
    ) -> pl.Expr:
        """
        Creates self.len() many random points sampled from a exponential distribution with n and p.

        This treats self as the reference column.

        Parameters
        ----------
        lambda_
            lambda in a exponential distribution. If none, it will be 1/reference col's mean. Note that if
            lambda < 0 will throw an error and lambda = 0 will only return infinity.
        respect_null
            If true, null in reference column will be null in the new column
        """
        if isinstance(lambda_, pl.Expr):
            la = lambda_
        elif isinstance(lambda_, float):
            la = pl.lit(lambda_, dtype=pl.Float64)
        else:
            la = 1.0 / self._expr.mean()

        resp = pl.lit(respect_null, dtype=pl.Boolean)
        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_sample_exp",
            args=[la, resp],
            is_elementwise=True,
            returns_scalar=False,
        )

    def sample_normal(
        self,
        mean: Optional[Union[float, pl.Expr]] = None,
        std: Optional[Union[float, pl.Expr]] = None,
        respect_null: bool = False,
    ) -> pl.Expr:
        """
        Creates self.len() many random points sampled from a normal distribution with the given
        mean and std.

        This treats self as the reference column.

        Parameters
        ----------
        mean
            Mean of the normal distribution. If none, use reference col's mean.
        std
            Std of the normal distribution. If none, use reference col's std.
        respect_null
            If true, null in reference column will be null in the new column
        """
        if isinstance(mean, pl.Expr):
            me = mean
        elif isinstance(mean, (float, int)):
            me = pl.lit(mean, dtype=pl.Float64)
        else:
            me = self._expr.mean()

        if isinstance(std, pl.Expr):
            st = std
        elif isinstance(std, (float, int)):
            st = pl.lit(std, dtype=pl.Float64)
        else:
            st = self._expr.std()

        resp = pl.lit(respect_null, dtype=pl.Boolean)
        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_sample_normal",
            args=[me, st, resp],
            is_elementwise=True,
            returns_scalar=False,
        )

    def rand_str(
        self, min_size: int = 1, max_size: int = 10, respect_null: bool = False
    ) -> pl.Expr:
        """
        Creates self.len() many random strings with alpha-numerical values. Unfortunately that
        means this currently only generates strings satisfying [0-9a-zA-Z]. The string's
        length will also be uniformly.

        This treats self as the reference column.

        Parameters
        ----------
        min_size
            The minimum length of the string to be generated. The length of the string will be
            uniformly generated in [min_size, max_size), except when min_size = max_size, in
            which case only fixed length strings will be generated.
        max_size
            The maximum length of the string to be generated.
        respect_null
            If true, null in reference column will be null in the new column
        """
        if min_size <= 0 or (max_size < min_size):
            raise ValueError("String size must be positive and max_size must be >= min_size.")

        min_s = pl.lit(min_size, dtype=pl.UInt32)
        max_s = pl.lit(max_size, dtype=pl.UInt32)
        resp = pl.lit(respect_null, dtype=pl.Boolean)
        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_sample_alphanumeric",
            args=[min_s, max_s, resp],
            is_elementwise=True,
            returns_scalar=False,
        )
