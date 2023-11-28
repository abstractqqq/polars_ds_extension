import polars as pl
from .type_alias import Alternative
from typing import Optional
from polars.utils.udfs import _get_shared_lib_location
# from polars.type_aliases import IntoExpr

lib = _get_shared_lib_location(__file__)


@pl.api.register_expr_namespace("stats_ext")
class StatsExt:
    def __init__(self, expr: pl.Expr):
        self._expr: pl.Expr = expr

    def ttest_ind(
        self, other: pl.Expr, alternative: Alternative = "two-sided", equal_var: bool = False
    ) -> pl.Expr:
        """
        Performs 2 sample student's t test or Welch's t test. Functionality-wise this is
        equivalent to SciPy's ttest_ind, with fewer options. The result is not exact but
        within 1e-10 precision from SciPy's result.

        In the case of student's t test, the user is responsible for data to have equal length,
        and nulls will be ignored when computing mean and variance. The df will be 2n - 2. As a
        result, nulls might cause problems. In the case of Welch's t test, data
        will be sanitized (nulls, NaNs, Infs will be dropped before the test), and df will be
        counted based on the length of sanitized data.

        Parameters
        ----------
        other
            The other expression
        alternative
            One of "two-sided", "less" or "greater"
        equal_var
            If true, perform standard student t 2 sample test. Otherwise, perform Welch's
            t test.
        """
        if equal_var:
            m1 = self._expr.mean()
            m2 = other.mean()
            v1 = self._expr.var()
            v2 = other.var()
            # Note here that nulls are not filtered to ensure the same length
            cnt = self._expr.count().cast(pl.UInt64)
            return m1.register_plugin(
                lib=lib,
                symbol="pl_ttest_2samp",
                args=[m2, v1, v2, cnt, pl.lit(alternative, dtype=pl.Utf8)],
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
                lib=lib,
                symbol="pl_welch_t",
                args=[m2, v1, v2, n1, n2, pl.lit(alternative, dtype=pl.Utf8)],
                is_elementwise=False,
                returns_scalar=True,
            )

    def ttest_1samp(self, pop_mean: float, alternative: Alternative = "two-sided") -> pl.Expr:
        """
        Performs a standard 1 sample t test using reference column and expected mean. This function
        sanitizes the self column first. The df is the count of valid (non-null, finite) values.

        Parameters
        ----------
        pop_mean
            The expected population mean in the hypothesis test
        alternative
            One of "two-sided", "less" or "greater"
        """
        s1 = self._expr.filter(self._expr.is_finite())
        sm = s1.mean()
        pm = pl.lit(pop_mean, dtype=pl.Float64)
        var = s1.var()
        cnt = s1.count().cast(pl.UInt64)
        alt = pl.lit(alternative, dtype=pl.Utf8)
        return sm.register_plugin(
            lib=lib,
            symbol="pl_ttest_1samp",
            args=[pm, var, cnt, alt],
            is_elementwise=False,
            returns_scalar=True,
        )

    def normal_test(self) -> pl.Expr:
        """
        Perform a normality test which is based on D'Agostino and Pearson's test
        [1], [2] that combines skew and kurtosis to produce an omnibus test of normality.
        Null values, NaN and inf are dropped when running this computation.

        References
        ----------
        .. [1] D'Agostino, R. B. (1971), "An omnibus test of normality for
            moderate and large sample size", Biometrika, 58, 341-348
        .. [2] D'Agostino, R. and Pearson, E. S. (1973), "Tests for departure from
            normality", Biometrika, 60, 613-622
        """
        valid: pl.Expr = self._expr.filter(self._expr.is_finite())
        skew = valid.skew()
        # Pearson Kurtosis, see here: https://en.wikipedia.org/wiki/D%27Agostino%27s_K-squared_test
        kur = valid.kurtosis(fisher=False)
        return skew.register_plugin(
            lib=lib,
            symbol="pl_normal_test",
            args=[kur, valid.count().cast(pl.UInt64)],
            is_elementwise=False,
            returns_scalar=True,
        )

    def ks_stats(self, other: pl.Expr) -> pl.Expr:
        """
        Computes two-sided KS statistics with other. Currently it only returns the statistics.

        Parameters
        ----------
        other
            A Polars Expression
        """
        y = self._expr.cast(pl.Float64)
        other_ = other.cast(pl.Float64)
        return y.register_plugin(
            lib=lib,
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
        other
            A Polars Expression
        """
        y = self._expr.cast(pl.Float64)
        y1 = y.filter(target == target.max())
        y2 = y.filter((target == target.max()).not_())
        return y1.register_plugin(
            lib=lib,
            symbol="pl_ks_2samp",
            args=[y2, pl.lit(True, dtype=pl.Boolean)],
            is_elementwise=False,
            returns_scalar=True,
        )

    def rand_int(
        self,
        low: Optional[int] = 0,
        high: Optional[int] = 10,
        respect_null: bool = False,
        use_ref: bool = False,
    ) -> pl.Expr:
        """
        Generates random integers uniformly from the range [low, high). Throws an error if low == high
        or if low is None and high is None and use_ref_nunique == False.

        This treats self as the reference column.

        Parameters
        ----------
        low
            Lower end of random sample. None will be replaced by 0.
        high
            Higher end of random sample. None will be replaced by 10.
        respect_null
            If true, null in reference column will be null in the new column
        use_ref
            If true, will overried low to be 0 and high = nunique of the reference column.
            If reference column has 0 unique values, this will be set to 10.
        """
        if (low is None) & (high is None) & (not use_ref):
            raise ValueError("Either set valid low and high values or set use_ref = True")

        lo = pl.lit(low, dtype=pl.Int32)
        hi = pl.lit(high, dtype=pl.Int32)
        resp = pl.lit(respect_null, dtype=pl.Boolean)
        use_r = pl.lit(use_ref, dtype=pl.Boolean)
        return self._expr.register_plugin(
            lib=lib,
            symbol="pl_rand_int",
            args=[lo, hi, resp, use_r],
            is_elementwise=True,
            returns_scalar=False,
        )

    def sample_uniform(
        self, low: float = 0.0, high: float = 1.0, respect_null: bool = False
    ) -> pl.Expr:
        """
        Creates self.len() many random points sampled from a uniform distribution within [low, high).
        This will throw an error if low == high.

        This treats self as the reference column.

        Parameters
        ----------
        low
            Lower end of random sample.
        high
            Higher end of random sample.
        respect_null
            If true, null in reference column will be null in the new column
        """

        lo = pl.lit(low, dtype=pl.Float64)
        hi = pl.lit(high, dtype=pl.Float64)
        resp = pl.lit(respect_null, dtype=pl.Boolean)
        return self._expr.register_plugin(
            lib=lib,
            symbol="pl_sample_uniform",
            args=[lo, hi, resp],
            is_elementwise=True,
            returns_scalar=False,
        )

    def sample_normal(
        self, mean: float = 0.0, std: float = 1.0, respect_null: bool = False
    ) -> pl.Expr:
        """
        Creates self.len() many random points sampled from a normal distribution with the given
        mean and std.

        This treats self as the reference column.

        Parameters
        ----------
        mean
            Mean of the normal distribution
        std
            Std of the normal distribution
        respect_null
            If true, null in reference column will be null in the new column
        """
        if std <= 0:
            raise ValueError("Input `std` must be positive.")

        me = pl.lit(mean, dtype=pl.Float64)
        st = pl.lit(std, dtype=pl.Float64)
        resp = pl.lit(respect_null, dtype=pl.Boolean)
        return self._expr.register_plugin(
            lib=lib,
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
        if max_size <= 0:
            raise ValueError("Input `max_size` must be positive.")

        min_s = pl.lit(min_size, dtype=pl.UInt32)
        max_s = pl.lit(max_size, dtype=pl.UInt32)
        resp = pl.lit(respect_null, dtype=pl.Boolean)
        return self._expr.register_plugin(
            lib=lib,
            symbol="pl_sample_alphanumeric",
            args=[min_s, max_s, resp],
            is_elementwise=True,
            returns_scalar=False,
        )
