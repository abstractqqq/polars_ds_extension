import polars as pl
from typing import Union
from polars.utils.udfs import _get_shared_lib_location

lib = _get_shared_lib_location(__file__)


@pl.api.register_expr_namespace("num_ext")
class NumExt:
    def __init__(self, expr: pl.Expr):
        self._expr: pl.Expr = expr

    def std_err(self, ddof: int = 1) -> pl.Expr:
        """
        Returns the standard error of the variable.
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

    def hubor_loss(self, other: pl.Expr, delta: float) -> pl.Expr:
        """
        Computes huber loss between this and the other expression

        Parameters
        ----------
        other
            Either an int or a Polars expression
        """
        temp = (self._expr - other).abs()
        return (
            pl.when(temp <= delta)
            .then(0.5 * temp.pow(2))
            .otherwise(delta * (temp - 0.5 * delta))
            / self._expr.count()
        )

    def l1_loss(self, other: pl.Expr, normalize: bool = True) -> pl.Expr:
        """
        Computes L1 loss (normalized L1 distance) between this and the other expression. This
        is the norm without 1/p power.

        Parameters
        ----------
        other
            Either an int or a Polars expression
        normalize
            If true, divide the result by length of the series
        """
        temp = (self._expr - other).abs().sum()
        if normalize:
            return temp / self._expr.count()
        return temp

    def l2_loss(self, other: pl.Expr, normalize: bool = True) -> pl.Expr:
        """
        Computes L2 loss (normalized L2 distance) between this and the other expression. This
        is the norm without 1/p power.


        Parameters
        ----------
        other
            Either an int or a Polars expression
        normalize
            If true, divide the result by length of the series
        """
        temp = self._expr - other
        temp = temp.dot(temp)
        if normalize:
            return temp / self._expr.count()
        return temp

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

    def chebyshev_loss(self, other: pl.Expr, normalize: bool = True) -> pl.Expr:
        """
        Alias for l_inf_loss.
        """
        return self.l_inf_dist(other, normalize)

    def l_inf_loss(self, other: pl.Expr, normalize: bool = True) -> pl.Expr:
        """
        Computes L^infinity loss between this and the other expression

        Parameters
        ---------- 
        other
            Either an int or a Polars expression
        normalize
            If true, divide the result by length of the series
        """
        temp = self._expr - other
        out = pl.max_horizontal(temp.min().abs(), temp.max().abs())
        if normalize:
            return out / self._expr.count()
        return out

    def mape(self, other: pl.Expr, weighted: bool = False) -> pl.Expr:
        """
        Computes mean absolute percentage error between self and other. Self is actual.
        If weighted, it will compute the weighted version as defined here:

        https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        Parameters
        ----------
        other
            Either an int or a Polars expression
        weighted
            If true, computes wMAPE in the wikipedia article
        """
        if weighted:
            return (self._expr - other).abs().sum() / self._expr.abs().sum()
        else:
            return (1 - other / self._expr).abs().mean()

    def smape(self, other: pl.Expr) -> pl.Expr:
        """
        Computes symmetric mean absolute percentage error between self and other. Self is actual.
        The value is always between 0 and 1. This is the third version in the wikipedia without
        the 100 factor.

        https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error

        Parameters
        ----------
        other
            Either an int or a Polars expression
        """
        numerator = (self._expr - other).abs()
        denominator = 1.0 / (self._expr.abs() + other.abs())
        return (1.0 / self._expr.count()) * numerator.dot(denominator)

    def bce(self, actual: pl.Expr, normalize:bool=True) -> pl.Expr:
        """
        Treats self as the prediction. and computes Binary Cross Entropy loss.

        Parameters
        ----------
        actual
            The actual binary lable. Note: if this column is not binary, then the result
            will be nonsense.
        normalize
            Whether to divide by N.
        """
        out = actual.dot(self._expr.log()) + (1 - actual).dot((1 - self._expr).log())
        if normalize:
            return -(out / self._expr.count())
        return -out

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
            returns_scalar=True
        )

    def lstsq(self, *others: pl.Expr, add_bias:bool=False) -> pl.Expr:
        """
        Computes least squares solution to a linear matrix equation. If columns are
        not linearly independent, some numerical issue may occur. E.g you may see
        unrealistic coefficient in the output. This is a `silent` numerical issue during the 
        computation.

        All positional arguments should be expressions representing individual columns. This does
        not support composite expressions like pl.col(["a", "b"]), pl.all(), etc.
        
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
            returns_scalar=True
        )

    def fft(self, forward:bool=True) -> pl.Expr:
        """
        Computes the DST transform of input series using FFT Algorithm. A series of equal length will
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

    def str_jaccard(
        self
        , other: Union[str, pl.Expr]
        , substr_size: int = 2
        , parallel: bool = False
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

    def levenshtein_dist(
        self
        , other: Union[str, pl.Expr]
        , parallel: bool = False
    ) -> pl.Expr:
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

    def hamming_dist(
        self
        , other: Union[str, pl.Expr]
        , parallel: bool = False
    ) -> pl.Expr:
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
                    is_elementwise=True,
                )
                .drop_nulls()
            ).list.unique()
        return out

    def snowball(
        self
        , no_stopwords:bool=True
        , parallel:bool=False
    ) -> pl.Expr:
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
