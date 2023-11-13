import polars as pl
from typing import Union, Optional
from polars.utils.udfs import _get_shared_lib_location
# from polars.type_aliases import IntoExpr

lib = _get_shared_lib_location(__file__)


@pl.api.register_expr_namespace("num_ext")
class NumExt:
    def __init__(self, expr: pl.Expr):
        self._expr: pl.Expr = expr

    def binarize(self, cond: Optional[pl.Expr]) -> pl.Expr:
        """
        Binarize the column.

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
        Returns the maximum of |x|.
        """
        return pl.max_horizontal(self._expr.max().abs(), self._expr.min().abs())

    def gcd(self, other: Union[int, pl.Expr]) -> pl.Expr:
        """
        Computes GCD of two integer columns. This will try to cast everything to int64 and may
        fail.

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
            lib=lib,
            symbol="pl_gcd",
            args=[other_],
            is_elementwise=True,
        )

    def lcm(self, other: Union[int, pl.Expr]) -> pl.Expr:
        """
        Computes LCM of two integer columns. This will try to cast everything to int64 and may
        fail.

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
            lib=lib,
            symbol="pl_lcm",
            args=[other_],
            is_elementwise=True,
        )

    def hubor_loss(self, pred: pl.Expr, delta: float) -> pl.Expr:
        """
        Computes huber loss between this and the other expression. This assumes
        this expression is actual, and the input is predicted, although the order
        does not matter in this case.

        Parameters
        ----------
        pred
            A Polars expression representing predictions
        """
        temp = (self._expr - pred).abs()
        return (
            pl.when(temp <= delta).then(0.5 * temp.pow(2)).otherwise(delta * (temp - 0.5 * delta))
            / self._expr.count()
        )

    def l1_loss(self, pred: pl.Expr, normalize: bool = True) -> pl.Expr:
        """
        Computes L1 loss (absolute difference) between this and the other expression.

        Parameters
        ----------
        pred
            A Polars expression representing predictions
        normalize
            If true, divide the result by length of the series
        """
        temp = (self._expr - pred).abs().sum()
        if normalize:
            return temp / self._expr.count()
        return temp

    def l2_loss(self, pred: pl.Expr, normalize: bool = True) -> pl.Expr:
        """
        Computes L2 loss (normalized L2 distance) between this and the other expression. This
        is the norm without 1/p power.

        Parameters
        ----------
        pred
            A Polars expression representing predictions
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
        Computes the mean square log error.

        Parameters
        ----------
        pred
            A Polars expression representing predictions
        normalize
            If true, divide the result by length of the series
        """
        diff = self._expr.log1p() - pred.log1p()
        out = diff.dot(diff)
        if normalize:
            return out / self._expr.count()
        return out

    # def lp_loss(self, other: pl.Expr, p: float, normalize: bool = True) -> pl.Expr:
    #     """
    #     Computes LP loss (normalized LP distance) between this and the other expression. This
    #     is the norm without 1/p power.

    #     for p finite.

    #     Parameters
    #     ----------
    #     other
    #         Either an int or a Polars expression
    #     normalize
    #         If true, divide the result by length of the series
    #     """
    #     if p <= 0:
    #         raise ValueError(f"Input `p` must be > 0, not {p}")

    #     temp = (self._expr - other).abs().pow(p).sum()
    #     if normalize:
    #         return (temp / self._expr.count())
    #     return temp

    def chebyshev_loss(self, pred: pl.Expr, normalize: bool = True) -> pl.Expr:
        """
        Alias for l_inf_loss.
        """
        return self.l_inf_dist(pred, normalize)

    def l_inf_loss(self, pred: pl.Expr, normalize: bool = True) -> pl.Expr:
        """
        Computes L^infinity loss between this and the other expression

        Parameters
        ----------
        pred
            A Polars expression representing predictions
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
        Computes mean absolute percentage error between self and other. Self is actual.
        If weighted, it will compute the weighted version as defined here:

        https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        Parameters
        ----------
        pred
            A Polars expression representing predictions
        weighted
            If true, computes wMAPE in the wikipedia article
        """
        if weighted:
            return (self._expr - pred).abs().sum() / self._expr.abs().sum()
        else:
            return (1 - pred / self._expr).abs().mean()

    def smape(self, pred: pl.Expr) -> pl.Expr:
        """
        Computes symmetric mean absolute percentage error between self and other. Self is actual.
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

    def bce(self, pred: pl.Expr, normalize: bool = True) -> pl.Expr:
        """
        Computes Binary Cross Entropy loss.

        Parameters
        ----------
        pred
            The predicted probability.
        normalize
            Whether to divide by N.
        """
        out = self._expr.dot(pred.log()) + (1 - self._expr).dot((1 - pred).log())
        if normalize:
            return -(out / self._expr.count())
        return -out

    def roc_auc(self, pred: pl.Expr) -> pl.Expr:
        """
        Computes ROC AUC with self (actual) and the predictions. Self must be binary and castable to
        type UInt32. If self is not all 0s and 1s, the result will not make sense, or some error
        may occur.

        Parameters
        ----------
        pred
            The predicted probability.
        """
        y = self._expr.cast(pl.UInt32)
        return y.register_plugin(
            lib=lib,
            symbol="pl_roc_auc",
            args=[pred],
            is_elementwise=False,
            returns_scalar=True,
        )

    def trapz(self, x: Union[float, pl.Expr]) -> pl.Expr:
        """
        Treats self as y axis, integrates along x using the trapezoidal rule.

        Parameters
        ----------
        x
            If it is a single float, it must be positive and it will represent a uniform
            distance between points. If it is an expression, it must be sorted and have the
            same length as self.
        """
        y = self._expr.cast(pl.Float64)
        if isinstance(x, float):
            x_ = pl.lit(abs(x), pl.Float64)
        else:
            x_ = x.cast(pl.Float64)

        return y.register_plugin(
            lib=lib,
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
            lib=lib, symbol="pl_fast_exp", args=[n_], is_elementwise=True, returns_scalar=False
        )

    def t_2samp(self, other: pl.Expr) -> pl.Expr:
        """
        Computes the t statistics for an Independent two-sample t-test. It is highly recommended
        that nulls be imputed before calling this.

        Parameters
        ----------
        other
            Either an int or a Polars expression
        """
        numerator = self._expr.mean() - other.mean()
        denom = ((self._expr.var() + other.var()) / self._expr.count()).sqrt()
        return numerator / denom

    def welch_t(self, other: pl.Expr, return_df: bool = True) -> pl.Expr:
        """
        Computes the statistics for Welch's t-test. Welch's t-test is often used when
        the two series do not have the same length. Two series in a dataframe will always
        have the same length. Here, only non-null values are counted.

        Parameters
        ----------
        other
            Either an int or a Polars expression
        return_df
            Whether to return the degree of freedom or not.
        """
        e1 = self._expr.drop_nulls()
        e2 = other.drop_nulls()
        numerator = e1.mean() - e2.mean()
        s1: pl.Expr = e1.var() / e1.count()
        s2: pl.Expr = e2.var() / e2.count()
        denom = (s1 + s2).sqrt()
        if return_df:
            df_num = (s1 + s2).pow(2)
            df_denom = s1.pow(2) / (e1.count() - 1) + s2.pow(2) / (e2.count() - 1)
            return pl.concat_list(numerator / denom, df_num / df_denom)
        else:
            return numerator / denom

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
            lib=lib,
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
            lib=lib,
            symbol="pl_list_jaccard",
            args=[other],
            is_elementwise=True,
        )

    def cond_entropy(self, other: pl.Expr) -> pl.Expr:
        """
        Computes the conditional entropy of self(y) given other. H(y|other).

        Parameters
        ----------
        other
            A Polars expression
        """

        return self._expr.register_plugin(
            lib=lib,
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
        and output will have length |other| + 1

        Parameters
        ----------
        other
            Polars expressions.
        add_bias
            Whether to add a bias term
        """
        y = self._expr.cast(pl.Float64)
        return y.register_plugin(
            lib=lib,
            symbol="pl_lstsq",
            args=[pl.lit(add_bias, dtype=pl.Boolean)] + list(others),
            is_elementwise=False,
            returns_scalar=True,
        )

    def fft(self, forward: bool = True) -> pl.Expr:
        """
        Computes the DFT transform of input series using FFT Algorithm. A series of equal length will
        be returned, with elements being the real and complex part of the transformed values.

        Parameters
        ----------
        forward
            If true, compute DFT. If false, compute inverse DFT.
        """
        return self._expr.register_plugin(
            lib=lib,
            symbol="pl_fft",
            args=[pl.lit(forward, dtype=pl.Boolean)],
            is_elementwise=True,
        )


@pl.api.register_expr_namespace("str_ext")
class StrExt:
    def __init__(self, expr: pl.Expr):
        self._expr: pl.Expr = expr

    def extract_numbers(
        self, ignore_comma: bool = False, join_by: str = "", dtype: pl.DataType = pl.Utf8
    ) -> pl.Expr:
        """
        Extracts numbers from the string column, and stores them in a list.

        Parameters
        ----------
        ignore_comma
            Whether to remove all comma before matching for numbers
        join_by
            If dtype is pl.Utf8, join the list of strings using the value given here
        dtype
            The desired inner dtype for the extracted data. Should either be one of
            pl.NUMERIC_DTYPES or pl.Utf8

        Examples
        --------
        >>> df = pl.DataFrame({
        ...     "survey":["0% of my time", "1% to 25% of my time", "75% to 99% of my time",
        ...            "50% to 74% of my time", "75% to 99% of my time",
        ...            "50% to 74% of my time"]
        ... })
        >>> df.select(pl.col("survey").str_ext.extract_numbers(dtype=pl.UInt32))
        shape: (6, 1)
        ┌───────────┐
        │ survey    │
        │ ---       │
        │ list[u32] │
        ╞═══════════╡
        │ [0]       │
        │ [1, 25]   │
        │ [75, 99]  │
        │ [50, 74]  │
        │ [75, 99]  │
        │ [50, 74]  │
        └───────────┘
        >>> df.select(pl.col("survey").str_ext.extract_numbers(join_by="-", dtype=pl.Utf8))
        shape: (6, 1)
        ┌────────┐
        │ survey │
        │ ---    │
        │ str    │
        ╞════════╡
        │ 0      │
        │ 1-25   │
        │ 75-99  │
        │ 50-74  │
        │ 75-99  │
        │ 50-74  │
        └────────┘
        """
        expr = self._expr
        if ignore_comma:
            expr = expr.str.replace_all(",", "")

        # Find all numbers
        expr = expr.str.extract_all("(\d*\.?\d+)")
        if dtype in pl.NUMERIC_DTYPES:
            expr = expr.list.eval(pl.element().cast(dtype))
        elif dtype == pl.Utf8:  # As a list of strings
            if join_by != "":
                expr = expr.list.join(join_by)

        return expr

    def infer_infreq(
        self,
        *,
        min_count: Optional[int] = None,
        min_frac: Optional[float] = None,
        parallel: bool = False,
    ) -> pl.Expr:
        """
        Infers infrequent categories (strings) by min_count or min_frac and return a list as output.

        Parameters
        ----------
        min_count
            If set, an infrequency category will be defined as a category with count < this.
        min_frac
            If set, an infrequency category will be defined as a category with pct < this. min_count
            takes priority over this.
        parallel
            Whether to run value_counts in parallel. This may not provide much speed up and is not
            recommended in a group_by context.
        """
        name = self._expr.meta.root_names()[0]
        vc = self._expr.value_counts(parallel=parallel, sort=True)
        if min_count is None and min_frac is None:
            raise ValueError("Either min_count or min_frac must be provided.")
        elif min_count is not None:
            infreq: pl.Expr = vc.filter(vc.struct.field("counts") < min_count).struct.field(name)
        elif min_frac is not None:
            infreq: pl.Expr = vc.filter(
                vc.struct.field("counts") / vc.struct.field("counts").sum() < min_frac
            ).struct.field(name)

        return infreq.implode()

    def merge_infreq(
        self,
        *,
        min_count: Optional[int] = None,
        min_frac: Optional[float] = None,
        separator: str = "|",
        parallel: bool = False,
    ) -> pl.Expr:
        """
        Merge infrequent categories (strings) in the column into one category (string) separated by a
        separator. This is useful when you want to do one-hot-encoding but do not want too many distinct
        values because of low count values. However, this does not mean that the categories are similar
        with respect to the your modelling problem.

        Parameters
        ----------
        min_count
            If set, an infrequency category will be defined as a category with count < this.
        min_frac
            If set, an infrequency category will be defined as a category with pct < this. min_count
            takes priority over this.
        separator
            What separator to use when joining the categories. E.g if "a" and "b" are rare categories,
            and separator = "|", they will be mapped to "a|b"
        parallel
            Whether to run value_counts in parallel. This may not provide much speed up and is not
            recommended in a group_by context.
        """

        # Will be fixed soon and sort will not be needed
        name = self._expr.meta.root_names()[0]
        vc = self._expr.value_counts(parallel=parallel, sort=True)
        if min_count is None and min_frac is None:
            raise ValueError("Either min_count or min_frac must be provided.")
        elif min_count is not None:
            to_merge: pl.Expr = vc.filter(vc.struct.field("counts") < min_count).struct.field(name)
        elif min_frac is not None:
            to_merge: pl.Expr = vc.filter(
                vc.struct.field("counts") / vc.struct.field("counts").sum() < min_frac
            ).struct.field(name)

        return (
            pl.when(self._expr.is_in(to_merge))
            .then(to_merge.cast(pl.Utf8).fill_null("null").implode().first().list.join(separator))
            .otherwise(self._expr)
        )

    def str_jaccard(
        self, other: Union[str, pl.Expr], substr_size: int = 2, parallel: bool = False
    ) -> pl.Expr:
        """
        Treats substrings of size `substr_size` as a set. And computes the jaccard similarity between
        this word and the other.

        Parameters
        ----------
        other
            If this is a string, then the entire column will be compared with this string. If this
            is an expression, then perform element-wise jaccard similarity computation between this column
            and the other (given by the expression).
        substr_size
            The substring size for Jaccard similarity. E.g. if substr_size = 2, "apple" will be decomposed into
            the set ('ap', 'pp', 'pl', 'le') before being compared.
        parallel
            Whether to run the comparisons in parallel. Note that this is not always faster, especially
            when used with other expressions or in group_by/over context.
        """
        if isinstance(other, str):
            other_ = pl.lit(other, dtype=pl.Utf8)
        else:
            other_ = other

        return self._expr.register_plugin(
            lib=lib,
            symbol="pl_str_jaccard",
            args=[other_, pl.lit(substr_size, dtype=pl.UInt32), pl.lit(parallel, dtype=pl.Boolean)],
            is_elementwise=True,
        )

    def levenshtein_dist(self, other: Union[str, pl.Expr], parallel: bool = False) -> pl.Expr:
        """
        Computes the levenshtein distance between this each value in the column with the str other.

        Parameters
        ----------
        other
            If this is a string, then the entire column will be compared with this string. If this
            is an expression, then an element-wise Levenshtein distance computation between this column
            and the other (given by the expression) will be performed.
        parallel
            Whether to run the comparisons in parallel. Note that this is not always faster, especially
            when used with other expressions or in group_by/over context.
        """
        if isinstance(other, str):
            other_ = pl.lit(other, dtype=pl.Utf8)
        else:
            other_ = other

        return self._expr.register_plugin(
            lib=lib,
            symbol="pl_levenshtein_dist",
            args=[other_, pl.lit(parallel, dtype=pl.Boolean)],
            is_elementwise=True,
        )

    def hamming_dist(self, other: Union[str, pl.Expr], parallel: bool = False) -> pl.Expr:
        """
        Computes the hamming distance between two strings. If they do not have the same length, null will
        be returned.

        Parameters
        ----------
        other
            If this is a string, then the entire column will be compared with this string. If this
            is an expression, then an element-wise hamming distance computation between this column
            and the other (given by the expression) will be performed.
        parallel
            Whether to run the comparisons in parallel. Note that this is not always faster, especially
            when used with other expressions or in group_by/over context.
        """
        if isinstance(other, str):
            other_ = pl.lit(other)
        else:
            other_ = other

        return self._expr.register_plugin(
            lib=lib,
            symbol="pl_hamming_dist",
            args=[other_, pl.lit(parallel, dtype=pl.Boolean)],
            is_elementwise=True,
        )

    def tokenize(self, pattern: str = r"(?u)\b\w\w+\b", stem: bool = False) -> pl.Expr:
        """
        Tokenize the string according to the pattern. This will only extract the words
        satisfying the pattern.

        Parameters
        ----------
        pattern
            The word pattern to extract
        stem
            If true, then this will stem the words and keep only the unique ones. Stop words
            will be removed. (Common words like `he`, `she`, etc., will be removed.)
        """
        out = self._expr.str.extract_all(pattern)
        if stem:
            out = out.list.eval(
                pl.element()
                .register_plugin(
                    lib=lib,
                    symbol="pl_snowball_stem",
                    args=[pl.lit(True, dtype=pl.Boolean), pl.lit(False, dtype=pl.Boolean)],
                    is_elementwise=True,
                )  # True to no stop word, False to Parallel
                .drop_nulls()
            )
        return out

    def freq_removal(
        self, lower: float = 0.05, upper: float = 0.95, parallel: bool = True
    ) -> pl.Expr:
        """
        Removes from each documents words that are too frequent (in the entire dataset). This assumes
        that the input expression represents lists of strings. E.g. output of tokenize.

        Parameters
        ----------
        lower
            Lower percentile. If a word's frequency is < than this, it will be removed.
        upper
            Upper percentile. If a word's frequency is > than this, it will be removed.
        parallel
            Whether to run word count in parallel. It is not recommended when you are in a group_by
            context.
        """

        name = self._expr.meta.root_names()[0]
        vc = self._expr.list.explode().value_counts(parallel=parallel).sort()
        lo = vc.struct.field("counts").quantile(lower)
        u = vc.struct.field("counts").quantile(upper)
        remove = (
            vc.filter((vc.struct.field("counts") < lo) | (vc.struct.field("counts") > u))
            .struct.field(name)
            .implode()
        )

        return self._expr.list.set_difference(remove)

    def snowball(self, no_stopwords: bool = True, parallel: bool = False) -> pl.Expr:
        """
        Applies the snowball stemmer for the column. The column is supposed to be a column of single words.

        Parameters
        ----------
        no_stopwords
            If true, stopwords will be mapped to None. If false, stopwords will be stemmed.
        parallel
            Whether to run the comparisons in parallel. Note that this is not always faster, especially
            when used with other expressions or in group_by/over context.
        """
        return self._expr.register_plugin(
            lib=lib,
            symbol="pl_snowball_stem",
            args=[pl.lit(no_stopwords, dtype=pl.Boolean), pl.lit(parallel, dtype=pl.Boolean)],
            is_elementwise=True,
        )


class LintExtExpr(pl.Expr):
    @property
    def num_ext(self) -> NumExt:
        return NumExt(self)

    @property
    def str_ext(self) -> StrExt:
        return StrExt(self)
