import polars as pl
from .type_alias import Alternative

# from typing import Union, Optional
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
        and nulls will be ignored when computing mean and variance. As a result, nulls might
        cause problems for student's t test. In the case of Welch's t test, data
        will be sanitized (nulls, NaNs, Infs will be dropped before the test).

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
            # Note here that nulls are not filtered
            cnt = self._expr.count().cast(pl.UInt64)
            return m1.register_plugin(
                lib=lib,
                symbol="pl_student_t_2samp",
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
