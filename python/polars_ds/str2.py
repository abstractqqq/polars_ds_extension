from __future__ import annotations
import polars as pl
from typing import Union, Optional, Literal, List
from polars.utils.udfs import _get_shared_lib_location

_lib = _get_shared_lib_location(__file__)


@pl.api.register_expr_namespace("str2")
class StrExt:

    """
    This class contains tools for dealing with string similarity, common string operations like tokenize,
    extract numbers, etc., inside Polars DataFrame.

    Polars Namespace: str2

    Example: pl.col("a").str2.levenshtein(pl.col("b"), return_sim=True)
    """

    def __init__(self, expr: pl.Expr):
        self._expr: pl.Expr = expr

    def is_stopword(self) -> pl.Expr:
        """
        Checks whether the string is a stopword in English or not.
        """
        self._expr.register_plugin(
            lib=_lib,
            symbol="pl_is_stopword",
            is_elementwise=True,
        )

    def extract_numbers(
        self, ignore_comma: bool = False, join_by: str = "", dtype: pl.DataType = pl.String
    ) -> pl.Expr:
        """
        Extracts numbers from the string column, and stores them in a list.

        Parameters
        ----------
        ignore_comma
            Whether to remove all comma before matching for numbers
        join_by
            If dtype is pl.String, join the list of strings using the value given here
        dtype
            The desired inner dtype for the extracted data. Should either be one of
            one of Polars' numerical types or pl.String

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
        >>> df.select(pl.col("survey").str_ext.extract_numbers(join_by="-", dtype=pl.String))
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
        elif dtype == pl.String:  # As a list of strings
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
        separator. This is useful when you want to do one-hot-encoding but not too many distinct
        values because of low value counts. This does not mean that infreq categories are similar, however.

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
            .then(to_merge.cast(pl.String).fill_null("null").implode().first().list.join(separator))
            .otherwise(self._expr)
        )

    def fuzz(self, other: Union[str, pl.Expr], parallel: bool = False) -> pl.Expr:
        """
        A string similarity based on Longest Common Subsequence.

        Parameters
        ----------
        other
            If this is a string, then the entire column will be compared with this string. If this
            is an expression, then perform element-wise fuzz computation between this column
            and the other (given by the expression).
        parallel
            Whether to run the comparisons in parallel. Note that this is not always faster, especially
            when used with other expressions or in group_by/over context.
        """
        if isinstance(other, str):
            other_ = pl.lit(other, dtype=pl.String)
        else:
            other_ = other

        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_fuzz",
            args=[other_, pl.lit(parallel, pl.Boolean)],
            is_elementwise=True,
        )

    def str_jaccard(
        self, other: Union[str, pl.Expr], substr_size: int = 2, parallel: bool = False
    ) -> pl.Expr:
        """
        Treats substrings of size `substr_size` as a set. And computes the jaccard similarity between
        this word and the other.

        Note this treats substrings at the byte level under the hood, not at the char level. So non-ASCII
        characters may have problems.

        Parameters
        ----------
        other
            If this is a string, then the entire column will be compared with this string. If this
            is an expression, then perform row wise jaccard similarity
        substr_size
            The substring size for Jaccard similarity. E.g. if substr_size = 2, "apple" will be decomposed into
            the set ('ap', 'pp', 'pl', 'le') before being compared.
        parallel
            Whether to run the comparisons in parallel. Note that this is not always faster, especially
            when used with other expressions or in group_by/over context.
        """
        if isinstance(other, str):
            other_ = pl.lit(other, dtype=pl.String)
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
        this word and the other.

        Note this treats substrings at the byte level under the hood, not at the char level. So non-ASCII
        characters may have problems.

        Parameters
        ----------
        other
            If this is a string, then the entire column will be compared with this string. If this
            is an expression, then perform row wise sorensen_dice similarity
        substr_size
            The substring size for Jaccard similarity. E.g. if substr_size = 2, "apple" will be decomposed into
            the set ('ap', 'pp', 'pl', 'le') before being compared.
        parallel
            Whether to run the comparisons in parallel. Note that this is not always faster, especially
            when used with other expressions or in group_by/over context.
        """
        if isinstance(other, str):
            other_ = pl.lit(other, dtype=pl.String)
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
        similarity between this word and the other.

        Note this treats substrings at the byte level under the hood, not at the char level. So non-ASCII
        characters may have problems.

        Parameters
        ----------
        other
            If this is a string, then the entire column will be compared with this string. If this
            is an expression, then perform row wise overlap_coefficient.
        substr_size
            The substring size for Jaccard similarity. E.g. if substr_size = 2, "apple" will be decomposed into
            the set ('ap', 'pp', 'pl', 'le') before being compared.
        parallel
            Whether to run the comparisons in parallel. Note that this is not always faster, especially
            when used with other expressions or in group_by/over context.
        """
        if isinstance(other, str):
            other_ = pl.lit(other, pl.String)
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
            other_ = pl.lit(other, dtype=pl.String)
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

    def levenshtein_filter(
        self,
        other: Union[str, pl.Expr],
        bound: int,
        parallel: bool = False,
    ) -> pl.Expr:
        """
        Returns whether the Levenshtein distance between self and other is <= bound. This is much
        faster than computing levenshtein distance and then doing a filter.

        Parameters
        ----------
        other
            If this is a string, then the entire column will be compared with this string. If this
            is an expression, then an element-wise Levenshtein distance computation between this column
            and the other (given by the expression) will be performed.
        bound
            Closed upper bound. If distance <= bound, return true and false otherwise.
        parallel
            Whether to run the comparisons in parallel. Note that this is not always faster, especially
            when used with other expressions or in group_by/over context.
        """
        if isinstance(other, str):
            other_ = pl.lit(other, pl.String)
        else:
            other_ = other

        bound = pl.lit(abs(bound), pl.UInt32)
        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_levenshtein_filter",
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
            other_ = pl.lit(other, dtype=pl.String)
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
            other_ = pl.lit(other, dtype=pl.String)
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
            other_ = pl.lit(other, dtype=pl.String)
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
        Computes the Jaro-Winkler similarity between this and the other str.
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
            other_ = pl.lit(other, pl.String)
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

        if pad:
            return self._expr.register_plugin(
                lib=_lib,
                symbol="pl_hamming_padded",
                args=[other_, pl.lit(parallel, pl.Boolean)],
                is_elementwise=True,
            )
        else:
            return self._expr.register_plugin(
                lib=_lib,
                symbol="pl_hamming",
                args=[other_, pl.lit(parallel, pl.Boolean)],
                is_elementwise=True,
            )

    def hamming_filter(
        self, other: Union[str, pl.Expr], bound: int, pad: bool = False, parallel: bool = False
    ) -> pl.Expr:
        """
        Returns whether the hamming distance between self and other is <= bound. This is
        faster than computing hamming distance and then doing a filter. Note this does not pad
        the strings. If the lengths of the two strings do not match, they will be filtered out.

        Parameters
        ----------
        other
            If this is a string, then the entire column will be compared with this string. If this
            is an expression, then an element-wise hamming distance computation between this column
            and the other (given by the expression) will be performed.
        bound
            Closed upper bound. If distance <= bound, return true and false otherwise.
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
            symbol="pl_hamming_filter",
            args=[
                other_,
                pl.lit(bound, dtype=pl.UInt32),
                pl.lit(parallel, pl.Boolean),
            ],
            is_elementwise=True,
        )

    def similar_to_vocab(
        self,
        vocab: List[str],
        threshold: float,
        metric: Literal["lv", "dlv", "jw", "osa"] = "lv",
        strategy: Literal["avg", "all", "any"] = "avg",
    ) -> pl.Expr:
        """
        Compare each word in the vocab with each word in self. Filters self to the words
        that are most similar to the words in the vocab.

        Parameters
        ----------
        vocab
            Any iterable collection of strings
        threshold
            A entry is considered similar to the words in the vocabulary if the similarity
            is above (>=) the threshold
        metric
            Which similarity metric to use. One of `lv`, `dlv`, `jw`, `osa`
        strategy
            If `avg`, then will return true if the average similarity is above the threshold.
            If `all`, then will return true if the similarity to all words in the vocab is above
            the threshold.
            If `any`, then will return true if the similarity to any words in the vocab is above
            the threshold.
        """
        if metric == "lv":
            sims = [self.levenshtein(w, return_sim=True) for w in vocab]
        elif metric == "dlv":
            sims = [self.d_levenshtein(w, return_sim=True) for w in vocab]
        elif metric == "osa":
            sims = [self.osa(w, return_sim=True) for w in vocab]
        elif metric == "jw":
            sims = [self.jw(w, return_sim=True) for w in vocab]
        else:
            raise ValueError(f"Unknown metric for find_similar: {metric}")

        if strategy == "all":
            return pl.all_horizontal(s >= threshold for s in sims)
        elif strategy == "any":
            return pl.any_horizontal(s >= threshold for s in sims)
        elif strategy == "avg":
            return (pl.sum_horizontal(sims) / len(vocab)) >= threshold
        else:
            raise ValueError(f"Unknown strategy for similar_to_vocab: {strategy}")

    def similar_words(
        self,
        vocab: Union[pl.Expr, List[str]],
        k: int = 1,
        threshold: int = 100,
        metric: Literal["lv", "hamming"] = "lv",
        parallel: bool = False,
    ) -> pl.Expr:
        """
        Finds the k most similar words in vocab to each word in self. This works by computing
        Hamming/Levenshtein distances, instead of similarity, and then taking the smallest distance
        words as similar words. The result is correct because of the relationship between distance
        and similarity. This will deduplicate words in vocab. In case of a tie, any one may be chosen.

        Comment: This can be very slow due to underlying data structure problems. Setting a threshold
        may speed up the process.

        Parameters
        ----------
        vocab
            Any iterable collection of strings that can be turned into a polars Series, or an expression
        k : int, >0
            k most similar words will be found
        threshold : int, >0
            A word in the vocab has to be within threshold distance from words in self to be considered.
            This is a positive integer because all the distances output integers.
        metric
            Which similarity metric to use. One of `lv`, `hamming`
        parallel
            Whether to run the comparisons in parallel. Note that this is not always faster, especially
            when used with other expressions or in group_by/over context.
        """
        if metric not in ("lv", "hamming"):
            raise ValueError(f"Unknown metric for similar_words: {metric}")

        if isinstance(vocab, pl.Expr):
            vb = vocab.unique().drop_nulls()
        else:
            vb = pl.Series(values=vocab, dtype=pl.String).unique().drop_nulls()

        if k == 1:  # k = 1, this is a fast path (no heap)
            return self._expr.register_plugin(
                lib=_lib,
                symbol="pl_nearest_str",
                args=[vb],
                kwargs={
                    "k": k,
                    "metric": str(metric).lower(),
                    "threshold": threshold,
                    "parallel": parallel,
                },
                is_elementwise=True,
            )
        elif k > 1:
            return self._expr.register_plugin(
                lib=_lib,
                symbol="pl_knn_str",
                args=[vb],
                kwargs={
                    "k": k,
                    "metric": str(metric).lower(),
                    "threshold": threshold,
                    "parallel": parallel,
                },
                is_elementwise=True,
            )
        else:
            raise ValueError("Input `k` must be >= 1.")

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
                    args=[pl.lit(True, pl.Boolean)],
                    is_elementwise=True,
                )  # True to no stop word
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

    def snowball(self, no_stopwords: bool = True) -> pl.Expr:
        """
        Applies the snowball stemmer for the column. The column is supposed to be a column of single words.
        Numbers will be stemmed to the empty string.

        Parameters
        ----------
        no_stopwords
            If true, stopwords will be mapped to the empty string. If false, stopwords will remain. Removing
            stopwords may impact performance.
        """
        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_snowball_stem",
            args=[pl.lit(no_stopwords, pl.Boolean)],
            is_elementwise=True,
        )

    def to_camel_case(self) -> pl.Expr:
        """Turns itself into camel case. E.g. helloWorld"""
        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_to_camel",
            is_elementwise=True,
        )

    def to_snake_case(self) -> pl.Expr:
        """Turns itself into snake case. E.g. hello_world"""
        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_to_snake",
            is_elementwise=True,
        )

    def to_pascal_case(self) -> pl.Expr:
        """Turns itself into Pascal case. E.g. HelloWorld"""
        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_to_pascal",
            is_elementwise=True,
        )

    def to_constant_case(self) -> pl.Expr:
        """Turns itself into constant case. E.g. Hello_World"""
        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_to_constant",
            is_elementwise=True,
        )
