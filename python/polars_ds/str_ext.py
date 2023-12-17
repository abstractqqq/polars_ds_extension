import polars as pl
from typing import Union, Optional
from polars.utils.udfs import _get_shared_lib_location
from .type_alias import AhoCorasickMatchKind
import warnings

_lib = _get_shared_lib_location(__file__)


@pl.api.register_expr_namespace("str_ext")
class StrExt:

    """
    This class contains tools for dealing with string similarity, common string operations like tokenize,
    extract numbers, etc., inside Polars DataFrame.

    Polars Namespace: str_ext

    Example: pl.col("a").str_ext.levenshtein(pl.col("b"), return_sim=True)
    """

    def __init__(self, expr: pl.Expr):
        self._expr: pl.Expr = expr

    def is_stopword(self) -> pl.Expr:
        """
        Checks whether the string is a stopword or not.
        """
        self._expr.register_plugin(
            lib=_lib,
            symbol="pl_is_stopword",
            args=[],
            is_elementwise=True,
        )

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

    def line_count(self) -> pl.Expr:
        """
        Return the line count of the string column.
        """
        return self._expr.str.count_matches(pattern="\n")

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
            infreq: pl.Expr = vc.filter(vc.struct.field("count") < min_count).struct.field(name)
        elif min_frac is not None:
            infreq: pl.Expr = vc.filter(
                vc.struct.field("count") / vc.struct.field("count").sum() < min_frac
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
            to_merge: pl.Expr = vc.filter(vc.struct.field("count") < min_count).struct.field(name)
        elif min_frac is not None:
            to_merge: pl.Expr = vc.filter(
                vc.struct.field("count") / vc.struct.field("count").sum() < min_frac
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
        this word and the other. This is not the same as comparing bigrams.

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
            lib=_lib,
            symbol="pl_str_jaccard",
            args=[other_, pl.lit(substr_size, pl.UInt32), pl.lit(parallel, pl.Boolean)],
            is_elementwise=True,
        )

    def sorensen_dice(
        self, other: Union[str, pl.Expr], substr_size: int = 2, parallel: bool = False
    ) -> pl.Expr:
        """
        Treats substrings of size `substr_size` as a set. And computes the Sorensen-Dice similarity between
        this word and the other. This is not the same as comparing bigrams.

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
            lib=_lib,
            symbol="pl_sorensen_dice",
            args=[other_, pl.lit(substr_size, pl.UInt32), pl.lit(parallel, pl.Boolean)],
            is_elementwise=True,
        )

    def overlap_coeff(
        self, other: Union[str, pl.Expr], substr_size: int = 2, parallel: bool = False
    ) -> pl.Expr:
        """
        Treats substrings of size `substr_size` as a set. And computes the overlap coefficient as
        similarity between this word and the other. This is not the same as comparing bigrams.

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
            other_ = pl.lit(other, pl.Utf8)
        else:
            other_ = other

        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_overlap_coeff",
            args=[other_, pl.lit(substr_size, pl.UInt32), pl.lit(parallel, pl.Boolean)],
            is_elementwise=True,
        )

    def levenshtein(
        self, other: Union[str, pl.Expr], parallel: bool = False, return_sim: bool = False
    ) -> pl.Expr:
        """
        Computes the Levenshtein distance between this and the other str.

        Parameters
        ----------
        other
            If this is a string, then the entire column will be compared with this string. If this
            is an expression, then an element-wise Levenshtein distance computation between this column
            and the other (given by the expression) will be performed.
        parallel
            Whether to run the comparisons in parallel. Note that this is not always faster, especially
            when used with other expressions or in group_by/over context.
        return_sim
            If true, return normalized Levenshtein.
        """
        if isinstance(other, str):
            other_ = pl.lit(other, dtype=pl.Utf8)
        else:
            other_ = other

        if return_sim:
            return self._expr.register_plugin(
                lib=_lib,
                symbol="pl_levenshtein_sim",
                args=[other_, pl.lit(parallel, pl.Boolean)],
                is_elementwise=True,
            )
        else:
            return self._expr.register_plugin(
                lib=_lib,
                symbol="pl_levenshtein",
                args=[other_, pl.lit(parallel, pl.Boolean)],
                is_elementwise=True,
            )

    def levenshtein_within(
        self,
        other: Union[str, pl.Expr],
        bound: int,
        parallel: bool = False,
    ) -> pl.Expr:
        """
        Returns whether the Levenshtein distance between self and other is <= bound. This is much
        faster than computing levenshtein distance and then doing <= bound.

        Parameters
        ----------
        other
            If this is a string, then the entire column will be compared with this string. If this
            is an expression, then an element-wise Levenshtein distance computation between this column
            and the other (given by the expression) will be performed.
        bound
            Closed upper bound. If levenshtein distance <= bound, return true and false otherwise.
        parallel
            Whether to run the comparisons in parallel. Note that this is not always faster, especially
            when used with other expressions or in group_by/over context.
        """
        if isinstance(other, str):
            other_ = pl.lit(other, pl.Utf8)
        else:
            other_ = other

        bound = pl.lit(abs(bound), pl.UInt32)
        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_levenshtein_within",
            args=[other_, bound, pl.lit(parallel, pl.Boolean)],
            is_elementwise=True,
        )

    def d_levenshtein(
        self, other: Union[str, pl.Expr], parallel: bool = False, return_sim: bool = False
    ) -> pl.Expr:
        """
        Computes the Damerau-Levenshtein distance between this and the other str.

        Parameters
        ----------
        other
            If this is a string, then the entire column will be compared with this string. If this
            is an expression, then an element-wise Levenshtein distance computation between this column
            and the other (given by the expression) will be performed.
        parallel
            Whether to run the comparisons in parallel. Note that this is not always faster, especially
            when used with other expressions or in group_by/over context.
        return_sim
            If true, return normalized Damerau-Levenshtein.
        """
        if isinstance(other, str):
            other_ = pl.lit(other, dtype=pl.Utf8)
        else:
            other_ = other

        if return_sim:
            return self._expr.register_plugin(
                lib=_lib,
                symbol="pl_d_levenshtein_sim",
                args=[other_, pl.lit(parallel, pl.Boolean)],
                is_elementwise=True,
            )
        else:
            return self._expr.register_plugin(
                lib=_lib,
                symbol="pl_d_levenshtein",
                args=[other_, pl.lit(parallel, pl.Boolean)],
                is_elementwise=True,
            )

    def osa(
        self, other: Union[str, pl.Expr], parallel: bool = False, return_sim: bool = False
    ) -> pl.Expr:
        """
        Computes the Optimal String Alignment distance between this and the other str.

        Parameters
        ----------
        other
            If this is a string, then the entire column will be compared with this string. If this
            is an expression, then an element-wise OSA distance computation between this column
            and the other (given by the expression) will be performed.
        parallel
            Whether to run the comparisons in parallel. Note that this is not always faster, especially
            when used with other expressions or in group_by/over context.
        return_sim
            If true, return normalized OSA similarity.
        """
        if isinstance(other, str):
            other_ = pl.lit(other, dtype=pl.Utf8)
        else:
            other_ = other

        if return_sim:
            return self._expr.register_plugin(
                lib=_lib,
                symbol="pl_osa_sim",
                args=[other_, pl.lit(parallel, pl.Boolean)],
                is_elementwise=True,
            )
        else:
            return self._expr.register_plugin(
                lib=_lib,
                symbol="pl_osa",
                args=[other_, pl.lit(parallel, pl.Boolean)],
                is_elementwise=True,
            )

    def jaro(self, other: Union[str, pl.Expr], parallel: bool = False) -> pl.Expr:
        """
        Computes the Jaro similarity between this and the other str. Jaro distance = 1 - Jaro sim.

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
            lib=_lib,
            symbol="pl_jaro",
            args=[other_, pl.lit(parallel, pl.Boolean)],
            is_elementwise=True,
        )

    def jw(
        self, other: Union[str, pl.Expr], weight: float = 0.1, parallel: bool = False
    ) -> pl.Expr:
        """
        Computes the Jaro-Winker similarity between this and the other str.
        Jaro-Winkler distance = 1 - Jaro-Winkler sim.

        Parameters
        ----------
        other
            If this is a string, then the entire column will be compared with this string. If this
            is an expression, then an element-wise Levenshtein distance computation between this column
            and the other (given by the expression) will be performed.
        weight
            Weight for prefix. A typical value is 0.1.
        parallel
            Whether to run the comparisons in parallel. Note that this is not always faster, especially
            when used with other expressions or in group_by/over context.
        """
        if isinstance(other, str):
            other_ = pl.lit(other, pl.Utf8)
        else:
            other_ = other

        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_jw",
            args=[other_, pl.lit(weight, pl.Float64), pl.lit(parallel, pl.Boolean)],
            is_elementwise=True,
        )

    def hamming(
        self, other: Union[str, pl.Expr], pad: bool = False, parallel: bool = False
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
        pad
            Whether to pad the string when lengths are not equal.
        parallel
            Whether to run the comparisons in parallel. Note that this is not always faster, especially
            when used with other expressions or in group_by/over context.
        """
        if isinstance(other, str):
            other_ = pl.lit(other)
        else:
            other_ = other

        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_hamming",
            args=[other_, pl.lit(pad, pl.Boolean), pl.lit(parallel, pl.Boolean)],
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
                    lib=_lib,
                    symbol="pl_snowball_stem",
                    args=[pl.lit(True, pl.Boolean), pl.lit(False, pl.Boolean)],
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
        lo = vc.struct.field("count").quantile(lower)
        u = vc.struct.field("count").quantile(upper)
        remove = (
            vc.filter((vc.struct.field("count") < lo) | (vc.struct.field("count") > u))
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
            lib=_lib,
            symbol="pl_snowball_stem",
            args=[pl.lit(no_stopwords, pl.Boolean), pl.lit(parallel, pl.Boolean)],
            is_elementwise=True,
        )

    def ac_match(
        self,
        patterns: list[str],
        case_sensitive: bool = False,
        match_kind: AhoCorasickMatchKind = "standard",
        return_str: bool = False,
    ) -> pl.Expr:
        """
        Try to match the patterns using the Aho-Corasick algorithm. The matched pattern's indices will be
        returned. E.g. If for string1, pattern 2, 1, 3 are matched in this order, then [1, 0, 2] are
        returned. (Indices in pattern list)

        Polars >= 0.20 now has native aho-corasick support. The backend package is the same, though the function
        api is different. See polars's str.contains_any and str.replace_many.

        Parameters
        ----------
        patterns
            A list of strs, which are patterns to be matched
        case_sensitive
            Should this match be case sensitive? Default is false. Not working now.
        match_kind
            One of `standard`, `left_most_first`, or `left_most_longest`. For more information, see
            https://docs.rs/aho-corasick/latest/aho_corasick/enum.MatchKind.html. Any other input will
            be treated as standard.
        """

        # Currently value_capacity for each list is hard-coded to 20. If there are more than 20 matches,
        # then this will be slow (doubling vec capacity)
        warnings.warn("Argument `case_sensitive` does not seem to work right now.")
        warnings.warn(
            "This function is unstable and is subject to change and may not perform well if there are more than "
            "20 matches. Read the source code or contact the author for more information. The most difficulty part "
            "is to design an output API that works well with Polars, which is harder than one might think."
        )

        pat = pl.Series(patterns, dtype=pl.Utf8)
        cs = pl.lit(case_sensitive, pl.Boolean)
        mk = pl.lit(match_kind, pl.Utf8)
        if return_str:
            return self._expr.register_plugin(
                lib=_lib,
                symbol="pl_ac_match_str",
                args=[pat, cs, mk],
                is_elementwise=True,
            )
        else:
            return self._expr.register_plugin(
                lib=_lib,
                symbol="pl_ac_match",
                args=[pat, cs, mk],
                is_elementwise=True,
            )

    def ac_replace(
        self, patterns: list[str], replacements: list[str], parallel: bool = False
    ) -> pl.Expr:
        """
        Try to replace the patterns using the Aho-Corasick algorithm. The length of patterns should match
        the length of replacements. If not, both sequences will be capped at the shorter length. If an error
        happens during replacement, None will be returned.

        Polars >= 0.20 now has native aho-corasick support. The backend package is the same, though the function
        api is different. See polars's str.contains_any and str.replace_many.

        Parameters
        ----------
        patterns
            A list of strs, which are patterns to be matched
        replacements
            A list of strs to replace the patterns with
        parallel
            Whether to run the comparisons in parallel. Note that this is not always faster, especially
            when used with other expressions or in group_by/over context.
        """
        if (len(replacements) == 0) or (len(patterns) == 0):
            return self._expr

        mlen = min(len(patterns), len(replacements))
        pat = pl.Series(patterns[:mlen], dtype=pl.Utf8)
        rpl = pl.Series(replacements[:mlen], dtype=pl.Utf8)
        par = pl.lit(parallel, pl.Boolean)
        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_ac_replace",
            args=[pat, rpl, par],
            is_elementwise=True,
        )
