import polars as pl
from typing import Union
from polars.utils.udfs import _get_shared_lib_location

lib = _get_shared_lib_location(__file__)

@pl.api.register_expr_namespace("num_ext")
class NumExt:
    def __init__(self, expr: pl.Expr):
        self._expr:pl.Expr = expr

    def std_err(self, ddof:int = 1) -> pl.Expr:
        '''
        Returns the standard error of the variable.
        '''
        return self._expr.std(ddof=ddof) / self._expr.count().sqrt()
    
    def rms(self) -> pl.Expr: 
        '''
        Returns root mean square of the expression
        '''
        return (self._expr.dot(self._expr)/self._expr.count()).sqrt()

    def harmonic_mean(self) -> pl.Expr:
        '''
        Returns the harmonic mean of the expression
        '''
        return (
            self._expr.count() / (1.0 / self._expr).sum()
        )
    
    def geometric_mean(self) -> pl.Expr:
        '''
        Returns the geometric mean of the expression
        '''
        return self._expr.product().pow(1.0 / self._expr.count())
    
    def c_o_v(self, ddof:int = 1) -> pl.Expr:
        '''
        Returns the coefficient of variation of the expression
        '''
        return self._expr.std(ddof=ddof) / self._expr.mean()
    
    def range_over_mean(self) -> pl.Expr:
        '''
        Returns (max - min) / mean
        '''
        return (self._expr.max() - self._expr.min()) / self._expr.mean()

    def z_normalize(self, ddof:int=1) -> pl.Expr:
        '''
        z_normalize the given expression: remove the mean and scales by the std
        '''
        return (self._expr - self._expr.mean()) / self._expr.std(ddof=ddof)

    def min_max_normalize(self) -> pl.Expr:
        ''' 
        Min max normalize the given expression.
        '''
        return (self._expr - self._expr.min()) / (self._expr.max() - self._expr.min())
    
    def frac(self) -> pl.Expr:
        '''
        Returns the fractional part of the input values. E.g. fractional part of 1.1 is 0.1
        '''
        return self._expr.mod(1.0)

    def max_abs(self) -> pl.Expr:
        '''
        Returns the maximum of |x|.
        '''
        return pl.max_horizontal(self._expr.max().abs(), self._expr.min().abs())
    
    def gcd(self, other:Union[int, pl.Expr]) -> pl.Expr:
        '''
        Computes GCD of two integer columns. This will try to cast everything to int64 and may
        fail.

        Parameters
        ----------
        other
            Either an int or a Polars expression
        '''
        if isinstance(other, int):
            other_ = pl.lit(other, dtype=pl.Int64)
        else:
            other_ = other.cast(pl.Int64)
        
        return self._expr.cast(pl.Int64).register_plugin(
            lib=lib,
            symbol="pl_gcd",
            args = [other_],
            is_elementwise=True,
        )
    
    def lcm(self, other:Union[int, pl.Expr]) -> pl.Expr:
        '''
        Computes LCM of two integer columns. This will try to cast everything to int64 and may
        fail.

        Parameters
        ----------
        other
            Either an int or a Polars expression
        '''
        if isinstance(other, int):
            other_ = pl.lit(other, dtype=pl.Int64)
        else:
            other_ = other.cast(pl.Int64)
        
        return self._expr.cast(pl.Int64).register_plugin(
            lib=lib,
            symbol="pl_lcm",
            args = [other_],
            is_elementwise=True,
        )

    
@pl.api.register_expr_namespace("str_ext")
class StrExt:
    def __init__(self, expr: pl.Expr):
        self._expr:pl.Expr = expr

    def str_jaccard(
        self,
        other: Union[str, pl.Expr],
        substr_size: int = 2
    ) -> pl.Expr:
        '''
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
        '''
        if isinstance(other, str):
            other_ = pl.lit(other)
        else:
            other_ = other
        
        return self._expr.register_plugin(
            lib=lib,
            symbol="pl_str_jaccard",
            args = [other_, pl.lit(substr_size, dtype=pl.UInt32)],
            is_elementwise=True,
        )
    
    def levenshtein_dist(
        self, 
        other:Union[str, pl.Expr], 
    ) -> pl.Expr:
        '''
        Computes the levenshtein distance between this each value in the column with the str other.

        Parameters
        ----------
        other
            If this is a string, then the entire column will be compared with this string. If this 
            is an expression, then an element-wise Levenshtein distance computation between this column 
            and the other (given by the expression) will be performed.
        '''
        if isinstance(other, str):
            other_ = pl.lit(other)
        else:
            other_ = other

        return self._expr.register_plugin(
            lib=lib,
            symbol="pl_levenshtein_dist",
            args = [other_],
            is_elementwise=True,
        )
    
    def hamming_dist(
        self, 
        other:Union[str, pl.Expr], 
    ) -> pl.Expr:
        '''
        Computes the hamming distance between two strings. If they do not have the same length, null will
        be returned.

        Parameters
        ----------
        other
            If this is a string, then the entire column will be compared with this string. If this 
            is an expression, then an element-wise hamming distance computation between this column 
            and the other (given by the expression) will be performed.
        '''
        if isinstance(other, str):
            other_ = pl.lit(other)
        else:
            other_ = other

        return self._expr.register_plugin(
            lib=lib,
            symbol="pl_hamming_dist",
            args = [other_],
            is_elementwise=True,
        )

    def tokenize(self, pattern:str = r"(?u)\b\w\w+\b", stem:bool=False) -> pl.Expr:
        '''
        Tokenize the string according to the pattern. This will only extract the words
        satisfying the pattern.

        Parameters
        ----------
        pattern
            The word pattern to extract
        stem
            If true, then this will stem the words and keep only the unique ones. Stop words 
            will be removed. (Common words like `he`, `she`, etc., will be removed.)
        '''
        out = self._expr.str.extract_all(pattern)
        if stem:
            out = out.list.eval(pl.element()
                .register_plugin(
                    lib=lib,
                    symbol="pl_snowball_stem",
                    is_elementwise=True,
                ).drop_nulls()).list.unique()
        return out
    
    def snowball(self) -> pl.Expr:
        '''
        Applies the snowball stemmer for the column. The column is supposed to be a column of single words.
        '''
        return self._expr.register_plugin(
            lib=lib,
            symbol="pl_snowball_stem",
            is_elementwise=True,
        )

