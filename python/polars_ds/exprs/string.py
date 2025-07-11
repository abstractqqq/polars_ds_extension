"""String-related utils, including edit distances, simple cleaning, etc."""

from __future__ import annotations

import warnings
import polars as pl
from typing import List, Literal, Dict

# Internal dependencies
from polars_ds._utils import pl_plugin, to_expr

__all__ = [
    "filter_by_levenshtein",
    "filter_by_hamming",
    "to_camel_case",
    "to_snake_case",
    "to_pascal_case",
    "to_constant_case",
    "str_nearest",
    "str_jaccard",
    "str_hamming",
    "str_sorensen_dice",
    "str_tversky_sim",
    "str_jw",
    "str_jaro",
    "str_d_leven",
    "str_leven",
    "str_osa",
    "str_lcs_seq", # deprecated
    "str_lcs_subseq",
    "str_lcs_subseq_dist",
    "str_lcs_substr",
    "str_fuzz",
    "similar_to_vocab",
    "extract_numbers",
    "replace_non_ascii",
    "remove_diacritics",
    "normalize_string",
    "map_words",
    "normalize_whitespace",
]


def filter_by_levenshtein(
    c: str | pl.Expr,
    other: str | pl.Expr,
    bound: int,
    parallel: bool = False,
) -> pl.Expr:
    """
    Returns whether the Levenshtein distance between self and other is <= bound. This is
    faster than computing levenshtein distance and then doing a filter.

    Parameters
    ----------
    c
        Either the name of the column or a Polars expression
    other
        Either the name of the column or a Polars expression. If you want to compare a single
        string with all of column c, use pl.lit(your_str)
    bound
        Closed upper bound. If distance <= bound, return true and false otherwise.
    parallel
        Whether to run the comparisons in parallel. Note that this is only recommended when this query
        is the only one in the context and we are not in any aggregation context.
    """

    return pl_plugin(
        symbol="pl_levenshtein_filter",
        args=[
            to_expr(c),
            to_expr(other),
            pl.lit(abs(bound), pl.UInt32),
            pl.lit(parallel, pl.Boolean),
        ],
    )


def filter_by_hamming(
    c: str | pl.Expr,
    other: str | pl.Expr,
    bound: int,
    pad: bool = False,
    parallel: bool = False,
) -> pl.Expr:
    """
    Returns whether the hamming distance between self and other is <= bound. This is
    faster than computing hamming distance and then doing a filter. Note this does not pad
    the strings. If the lengths of the two strings do not match, they will be filtered out.

    Parameters
    ----------
    c
        Either the name of the column or a Polars expression
    other
        Either the name of the column or a Polars expression. If you want to compare a single
        string with all of column c, use pl.lit(your_str)
    bound
        Closed upper bound. If distance <= bound, return true and false otherwise.
    parallel
        Whether to run the comparisons in parallel. Note that this is not always faster, especially
        when used with other expressions or in group_by/over context.
    """

    return pl_plugin(
        symbol="pl_hamming_filter",
        args=[
            to_expr(c),
            to_expr(other),
            pl.lit(bound, dtype=pl.UInt32),
            pl.lit(parallel, pl.Boolean),
        ],
    )


def str_hamming(
    c: str | pl.Expr, other: str | pl.Expr, pad: bool = False, parallel: bool = False
) -> pl.Expr:
    """
    Computes the hamming distance between two strings. If they do not have the same length, null will
    be returned.

    Parameters
    ----------
    c
        Either the name of the column or a Polars expression
    other
        Either the name of the column or a Polars expression. If you want to compare a single
        string with all of column c, use pl.lit(your_str)
    pad
        Whether to pad the string when lengths are not equal.
    parallel
        Whether to run the comparisons in parallel. Note that this is not always faster, especially
        when used with other expressions or in group_by/over context.
    """

    if pad:
        return pl_plugin(
            symbol="pl_hamming_padded",
            args=[to_expr(c), to_expr(other), pl.lit(parallel, pl.Boolean)],
        )
    else:
        return pl_plugin(
            symbol="pl_hamming",
            args=[to_expr(c), to_expr(other), pl.lit(parallel, pl.Boolean)],
        )


def to_camel_case(c: str | pl.Expr) -> pl.Expr:
    """Turns itself into camel case. E.g. helloWorld"""
    return pl_plugin(
        symbol="pl_to_camel",
        args=[to_expr(c)],
        is_elementwise=True,
    )


def to_snake_case(c: str | pl.Expr) -> pl.Expr:
    """Turns itself into snake case. E.g. hello_world"""
    return pl_plugin(
        symbol="pl_to_snake",
        args=[to_expr(c)],
        is_elementwise=True,
    )


def to_pascal_case(c: str | pl.Expr) -> pl.Expr:
    """Turns itself into Pascal case. E.g. HelloWorld"""
    return pl_plugin(
        symbol="pl_to_pascal",
        args=[to_expr(c)],
        is_elementwise=True,
    )


def to_constant_case(c: str | pl.Expr) -> pl.Expr:
    """Turns itself into constant case. E.g. Hello_World"""
    return pl_plugin(
        symbol="pl_to_constant",
        args=[to_expr(c)],
        is_elementwise=True,
    )


def str_nearest(
    c: str | pl.Expr,
    word: str,
    threshold: int = 100,
    metric: Literal["lv", "hamming"] = "lv",
) -> pl.Expr:
    """
    Finds the string in the column that is nearest to the given word in the given metric. This algorithm is 
    very slow.

    Note: Nearest-k strings search functionality is temporarily dropped.

    Parameters
    ----------
    c
        The string column or its name
    word
        Any iterable collection of strings that can be turned into a polars Series, or an expression
    threshold : int
        Only considers strings to be near if they are within distance threshold. This is a positive integer
        because all the distances output integers.
    metric
        Which similarity metric to use. One of `lv`, `hamming`
    """
    if metric not in ("lv", "hamming"):
        raise ValueError(f"Unknown metric for similar_words: {metric}")

    if threshold <= 0:
        raise ValueError("Distance threshold must be > 0.")

    return pl_plugin(
        symbol="pl_nearest_str",
        args=[to_expr(c)],
        kwargs={
            "word": word,
            "metric": str(metric).lower(),
            "threshold": threshold,
        },
        returns_scalar=True,
    )


def str_jaccard(
    c: str | pl.Expr,
    other: str | pl.Expr,
    substr_size: int = 2,
    parallel: bool = False,
) -> pl.Expr:
    """
    Treats substrings of size `substr_size` as a set. And computes the jaccard similarity between
    this word and the other.

    Note this treats substrings at the byte level under the hood, not at the char level. So non-ASCII
    characters may have problems.

    Parameters
    ----------
    c
        The string column
    other
        Either the name of the column or a Polars expression. If you want to compare a single
        string with all of column c, use pl.lit(your_str)
    substr_size
        The substring size for Jaccard similarity. E.g. if substr_size = 2, "apple" will be decomposed into
        the set ('ap', 'pp', 'pl', 'le') before being compared.
    parallel
        Whether to run the comparisons in parallel. Note that this is only recommended when this query
        is the only one in the context and we are not in any aggregation context.
    """
    return pl_plugin(
        symbol="pl_str_jaccard",
        args=[
            to_expr(c),
            to_expr(other),
            pl.lit(substr_size, pl.UInt32),
            pl.lit(parallel, pl.Boolean),
        ],
    )


def str_overlap_coeff(
    c: str | pl.Expr,
    other: str | pl.Expr,
    substr_size: int = 2,
    parallel: bool = False,
) -> pl.Expr:
    """
    Treats substrings of size `substr_size` as a set. And computes the overlap coefficient as
    similarity between this word and the other.

    Note this treats substrings at the byte level under the hood, not at the char level. So non-ASCII
    characters may have problems.

    Parameters
    ----------
    c
        The string column
    other
        Either the name of the column or a Polars expression. If you want to compare a single
        string with all of column c, use pl.lit(your_str)
    substr_size
        The substring size for Jaccard similarity. E.g. if substr_size = 2, "apple" will be decomposed into
        the set ('ap', 'pp', 'pl', 'le') before being compared.
    parallel
        Whether to run the comparisons in parallel. Note that this is only recommended when this query
        is the only one in the context and we are not in any aggregation context.
    """
    return pl_plugin(
        symbol="pl_overlap_coeff",
        args=[
            to_expr(c),
            to_expr(other),
            pl.lit(substr_size, pl.UInt32),
            pl.lit(parallel, pl.Boolean),
        ],
    )


def str_sorensen_dice(
    c: str | pl.Expr,
    other: str | pl.Expr,
    substr_size: int = 2,
    parallel: bool = False,
) -> pl.Expr:
    """
    Treats substrings of size `substr_size` as a set. And computes the Sorensen-Dice similarity between
    this word and the other.

    Note this treats substrings at the byte level under the hood, not at the char level. So non-ASCII
    characters may have problems.

    Parameters
    ----------
    c
        The string column
    other
        Either the name of the column or a Polars expression. If you want to compare a single
        string with all of column c, use pl.lit(your_str)
    substr_size
        The substring size for Jaccard similarity. E.g. if substr_size = 2, "apple" will be decomposed into
        the set ('ap', 'pp', 'pl', 'le') before being compared.
    parallel
        Whether to run the comparisons in parallel. Note that this is only recommended when this query
        is the only one in the context and we are not in any aggregation context.
    """
    return pl_plugin(
        symbol="pl_sorensen_dice",
        args=[
            to_expr(c),
            to_expr(other),
            pl.lit(substr_size, pl.UInt32),
            pl.lit(parallel, pl.Boolean),
        ],
    )


def str_tversky_sim(
    c: str | pl.Expr,
    other: str | pl.Expr,
    alpha: float,
    beta: float,
    substr_size: int = 2,
    parallel: bool = False,
) -> pl.Expr:
    """
    Treats substrings of size `substr_size` as a set. And computes the tversky_sim similarity between
    this word and the other. See the reference for information on how Tversky similarity is related
    the other ngram based similarity.

    Note this treats substrings at the byte level under the hood, not at the char level. So non-ASCII
    characters may have problems. Also note that alpha and beta are supposed to be weighting factors,
    but this doesn't check whether they satisfy the definition of weights and has to be chosen at the
    discretion of the user.

    Parameters
    ----------
    c
        The string column
    other
        Either the name of the column or a Polars expression. If you want to compare a single
        string with all of column c, use pl.lit(your_str)
    alpha
        The first weighting factor. See reference
    beta
        The second weighting factor. See reference
    substr_size
        The substring size for Jaccard similarity. E.g. if substr_size = 2, "apple" will be decomposed into
        the set ('ap', 'pp', 'pl', 'le') before being compared.
    parallel
        Whether to run the comparisons in parallel. Note that this is only recommended when this query
        is the only one in the context and we are not in any aggregation context.

    Reference
    ---------
    https://yassineelkhal.medium.com/the-complete-guide-to-string-similarity-algorithms-1290ad07c6b7
    """
    if alpha < 0 or beta < 0:
        raise ValueError("Input `alpha` and `beta` must be >= 0.")

    return pl_plugin(
        symbol="pl_tversky_sim",
        args=[
            to_expr(c),
            to_expr(other),
            pl.lit(substr_size, pl.UInt32),
            pl.lit(alpha, pl.Float64),
            pl.lit(beta, pl.Float64),
            pl.lit(parallel, pl.Boolean),
        ],
    )


def str_jw(
    c: str | pl.Expr,
    other: str | pl.Expr,
    weight: float = 0.1,
    parallel: bool = False,
) -> pl.Expr:
    """
    Computes the Jaro-Winkler similarity between this and the other str.
    Jaro-Winkler distance = 1 - Jaro-Winkler sim.

    Parameters
    ----------
    c
        The string column
    other
        Either the name of the column or a Polars expression. If you want to compare a single
        string with all of column c, use pl.lit(your_str)
    weight
        Weight for prefix. A typical value is 0.1.
    parallel
        Whether to run the comparisons in parallel. Note that this is only recommended when this query
        is the only one in the context and we are not in any aggregation context.
    """
    return pl_plugin(
        symbol="pl_jw",
        args=[
            to_expr(c),
            to_expr(other),
            pl.lit(weight, pl.Float64),
            pl.lit(parallel, pl.Boolean),
        ],
    )


def str_jaro(c: str | pl.Expr, other: str | pl.Expr, parallel: bool = False) -> pl.Expr:
    """
    Computes the Jaro similarity between this and the other str. Jaro distance = 1 - Jaro sim.

    Parameters
    ----------
    c
        The string column
    other
        Either the name of the column or a Polars expression. If you want to compare a single
        string with all of column c, use pl.lit(your_str)
    parallel
        Whether to run the comparisons in parallel. Note that this is only recommended when this query
        is the only one in the context and we are not in any aggregation context.
    """
    return pl_plugin(
        symbol="pl_jaro",
        args=[to_expr(c), to_expr(other), pl.lit(parallel, pl.Boolean)],
        is_elementwise=True,
    )


def str_d_leven(
    c: str | pl.Expr,
    other: str | pl.Expr,
    parallel: bool = False,
    return_sim: bool = False,
) -> pl.Expr:
    """
    Computes the Damerau-Levenshtein distance between this and the other str.

    Parameters
    ----------
    c
        The string column
    other
        Either the name of the column or a Polars expression. If you want to compare a single
        string with all of column c, use pl.lit(your_str)
    parallel
        Whether to run the comparisons in parallel. Note that this is only recommended when this query
        is the only one in the context and we are not in any aggregation context.
    return_sim
        If true, return normalized Damerau-Levenshtein.
    """
    if return_sim:
        return pl_plugin(
            symbol="pl_d_levenshtein_sim",
            args=[to_expr(c), to_expr(other), pl.lit(parallel, pl.Boolean)],
        )
    else:
        return pl_plugin(
            symbol="pl_d_levenshtein",
            args=[to_expr(c), to_expr(other), pl.lit(parallel, pl.Boolean)],
        )


def str_leven(
    c: str | pl.Expr,
    other: str | pl.Expr,
    parallel: bool = False,
    return_sim: bool = False,
) -> pl.Expr:
    """
    Computes the Levenshtein distance between this and the other str.

    Parameters
    ----------
    c
        The string column
    other
        Either the name of the column or a Polars expression. If you want to compare a single
        string with all of column c, use pl.lit(your_str)
    parallel
        Whether to run the comparisons in parallel. Note that this is only recommended when this query
        is the only one in the context and we are not in any aggregation context.
    return_sim
        If true, return normalized Levenshtein.
    """
    if return_sim:
        return pl_plugin(
            symbol="pl_levenshtein_sim",
            args=[to_expr(c), to_expr(other), pl.lit(parallel, pl.Boolean)],
        )
    else:
        return pl_plugin(
            symbol="pl_levenshtein",
            args=[to_expr(c), to_expr(other), pl.lit(parallel, pl.Boolean)],
        )
    
def str_lcs_substr(
    c: str | pl.Expr,
    other: str | pl.Expr,
    parallel: bool = False,
) -> pl.Expr:
    """
    Extracts the longest common substring from the string between this and the other string.

    Note: this is not the same as the longest common subsequence.

    Parameters
    ----------
    c
        The string column
    other
        Either the name of the column or a Polars expression. If you want to compare a single
        string with all of column c, use pl.lit(your_str)
    parallel
        Whether to run the comparisons in parallel. Note that this is only recommended when this query
        is the only one in the context and we are not in any aggregation context.
    """
    return pl_plugin(
        symbol="pl_lcs_substr",
        args=[to_expr(c), to_expr(other), pl.lit(parallel, pl.Boolean)],
    )

def str_lcs_subseq(
    c: str | pl.Expr,
    other: str | pl.Expr,
    parallel: bool = False,
) -> pl.Expr:
    """
    Extracts the longest common subsequence from the string between this and the other string.

    Note: this is not the same as the longest common substring.

    Parameters
    ----------
    c
        The string column
    other
        Either the name of the column or a Polars expression. If you want to compare a single
        string with all of column c, use pl.lit(your_str)
    parallel
        Whether to run the comparisons in parallel. Note that this is only recommended when this query
        is the only one in the context and we are not in any aggregation context.
    """
    return pl_plugin(
        symbol="pl_lcs_subseq",
        args=[to_expr(c), to_expr(other), pl.lit(parallel, pl.Boolean)],
    )

def str_lcs_seq(
    c: str | pl.Expr,
    other: str | pl.Expr,
    parallel: bool = False,
    return_sim: bool = True,   
) -> pl.Expr:
    warnings.warn(
        "`str_lcs_seq` is deprecated and users should use `str_lcs_subseq_dist`."
        , DeprecationWarning
        , stacklevel=2
    )
    return str_lcs_subseq_dist(c, other, parallel, return_sim)


def str_lcs_subseq_dist(
    c: str | pl.Expr,
    other: str | pl.Expr,
    parallel: bool = False,
    return_sim: bool = True,
) -> pl.Expr:
    """
    Computes the Longest Common Subsequence distance/similarity between this and the other str.
    The distance is calculated as max(len1, len2) - similarity, where the similarity is the
    the length of the longest common subsequence. Subsequences may not occupy consecutive positions.

    The subsequence does not need to be consecutive.

    Parameters
    ----------
    c
        The string column
    other
        Either the name of the column or a Polars expression. If you want to compare a single
        string with all of column c, use pl.lit(your_str)
    parallel
        Whether to run the comparisons in parallel. Note that this is only recommended when this query
        is the only one in the context and we are not in any aggregation context.
    return_sim
        If true, return normalized similarity.
    """
    if return_sim:
        return pl_plugin(
            symbol="pl_lcs_subseq_sim",
            args=[to_expr(c), to_expr(other), pl.lit(parallel, pl.Boolean)],
        )
    else:
        return pl_plugin(
            symbol="pl_lcs_subseq_dist",
            args=[to_expr(c), to_expr(other), pl.lit(parallel, pl.Boolean)],
        )


def str_osa(
    c: str | pl.Expr,
    other: str | pl.Expr,
    parallel: bool = False,
    return_sim: bool = False,
) -> pl.Expr:
    """
    Computes the Optimal String Alignment distance between this and the other str.

    Parameters
    ----------
    c
        The string column
    other
        Either the name of the column or a Polars expression. If you want to compare a single
        string with all of column c, use pl.lit(your_str)
    parallel
        Whether to run the comparisons in parallel. Note that this is only recommended when this query
        is the only one in the context and we are not in any aggregation context.
    return_sim
        If true, return normalized OSA similarity.
    """
    if return_sim:
        return pl_plugin(
            symbol="pl_osa_sim",
            args=[to_expr(c), to_expr(other), pl.lit(parallel, pl.Boolean)],
        )
    else:
        return pl_plugin(
            symbol="pl_osa",
            args=[to_expr(c), to_expr(other), pl.lit(parallel, pl.Boolean)],
        )


def str_fuzz(c: str | pl.Expr, other: str | pl.Expr, parallel: bool = False) -> pl.Expr:
    """
    Calculates the normalized Indel similarity. (See the package rapidfuzz, fuzz.ratio for more
    information.)

    Parameters
    ----------
    c
        The string column
    other
        Either the name of the column or a Polars expression. If you want to compare a single
        string with all of column c, use pl.lit(your_str)
    parallel
        Whether to run the comparisons in parallel. Note that this is only recommended when this query
        is the only one in the context and we are not in any aggregation context.
    """
    return pl_plugin(
        symbol="pl_fuzz",
        args=[to_expr(c), to_expr(other), pl.lit(parallel, pl.Boolean)],
    )


def similar_to_vocab(
    c: str | pl.Expr,
    vocab: List[str],
    threshold: float,
    metric: Literal["lv", "dlv", "jw", "osa"] = "lv",
    strategy: Literal["avg", "all", "any"] = "avg",
) -> pl.Expr:
    """
    Compare each word in the vocab with each word in the column c. Returns a boolean
    that indicates whether there exist words in c that are similar to words in vocab.

    Parameters
    ----------
    c
        The string column
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
        sims = [str_leven(c, pl.lit(w, dtype=pl.String), return_sim=True) for w in vocab]
    elif metric == "dlv":
        sims = [str_d_leven(c, pl.lit(w, dtype=pl.String), return_sim=True) for w in vocab]
    elif metric == "osa":
        sims = [str_osa(c, pl.lit(w, dtype=pl.String), return_sim=True) for w in vocab]
    elif metric == "jw":
        sims = [str_jw(c, pl.lit(w, dtype=pl.String), return_sim=True) for w in vocab]
    else:
        raise ValueError(f"Unknown metric: {metric}")

    if strategy == "all":
        return pl.all_horizontal(s >= threshold for s in sims)
    elif strategy == "any":
        return pl.any_horizontal(s >= threshold for s in sims)
    elif strategy == "avg":
        return (pl.sum_horizontal(sims) / len(vocab)) >= threshold
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def extract_numbers(
    c: str | pl.Expr,
    ignore_comma: bool = False,
    join_by: str = "",
    dtype: pl.DataType = pl.String,
) -> pl.Expr:
    """
    Extracts numbers from the string column, and stores them in a list.

    Parameters
    ----------
    c
        The string column
    ignore_comma
        Whether to remove all comma before matching for numbers
    join_by
        If dtype is pl.String, join the list of strings using the value given here
    dtype
        The desired inner dtype for the extracted data. Should either be one of
        one of Polars' numerical types or pl.String

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "survey": [
    ...             "0% of my time",
    ...             "1% to 25% of my time",
    ...             "75% to 99% of my time",
    ...             "50% to 74% of my time",
    ...             "75% to 99% of my time",
    ...             "50% to 74% of my time",
    ...         ]
    ...     }
    ... )
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
    expr = to_expr(c)
    if ignore_comma:
        expr = expr.str.replace_all(",", "")

    # Find all numbers
    expr = expr.str.extract_all(r"(\d*\.?\d+)")

    if dtype in [
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.Float32,
        pl.Float64,
    ]:
        expr = expr.list.eval(pl.element().cast(dtype))
    elif dtype == pl.String:  # As a list of strings
        if join_by != "":
            expr = expr.list.join(join_by)

    return expr


def replace_non_ascii(c: str | pl.Expr, value: str = "") -> pl.Expr:
    """Replaces non-Ascii values with the specified value.

    Parameters
    ----------
    c : str | pl.Expr
        The column name or expression
    value : str
        The value to replace non-Ascii values with, by default ""

    Returns
    -------
    pl.Expr

    Examples
    --------
    >>> df = pl.DataFrame({"x": ["mercy", "xbĤ", "ĤŇƏ"]})
    >>> df.select(pds.replace_non_ascii("x"))
    shape: (3, 1)
    ┌───────┐
    │ x     │
    │ ---   │
    │ str   │
    ╞═══════╡
    │ mercy │
    │ xb    │
    │       │
    └───────┘
    """
    expr = to_expr(c)

    if value == "":
        return pl_plugin(
            symbol="remove_non_ascii",
            args=[expr],
            is_elementwise=True,
        )

    return expr.str.replace_all(r"[^\p{Ascii}]", value)


def remove_diacritics(c: str | pl.Expr) -> pl.Expr:
    """Remove diacritics (e.g. è -> e) by converting the string to its NFD normalized
    form and removing the resulting non-ASCII components.

    Parameters
    ----------
    c : str | pl.Expr

    Returns
    -------
    pl.Expr

    Examples
    --------
    >>> df = pl.DataFrame({"x": ["mercy", "mèrcy"]})
    >>> df.select(pds.replace_non_ascii("x"))
    shape: (2, 1)
    ┌───────┐
    │ x     │
    │ ---   │
    │ str   │
    ╞═══════╡
    │ mercy │
    │ mercy │
    └───────┘
    """
    return pl_plugin(
        symbol="remove_diacritics",
        args=[to_expr(c)],
        is_elementwise=True,
    )


def normalize_string(c: str | pl.Expr, form: Literal["NFC", "NFKC", "NFD", "NFKD"]) -> pl.Expr:
    """
    Normalize Unicode string using one of 'NFC', 'NFKC', 'NFD', or 'NFKD'
    normalization.

    See https://en.wikipedia.org/wiki/Unicode_equivalence for more information.

    Parameters
    ----------
    c : str | pl.Expr
        The string column
    form: Literal["NFC", "NFKC", "NFD", "NFKD"]
        The Unicode normalization form to use

    Returns
    -------
    pl.Expr

    Examples
    --------
    >>> df = pl.DataFrame({"x": ["\u0043\u0327"], "y": ["\u00c7"]})
    >>> df.with_columns(
    >>>     pl.col("x").eq(pl.col("y")).alias("is_equal"),
    >>>     pds.normalize_string("x", "NFC")
    >>>     .eq(pds.normalize_string("y", "NFC"))
    >>>     .alias("normalized_is_equal"),
    >>> )
    shape: (1, 4)
    ┌─────┬─────┬──────────┬─────────────────────┐
    │ x   ┆ y   ┆ is_equal ┆ normalized_is_equal │
    │ --- ┆ --- ┆ ---      ┆ ---                 │
    │ str ┆ str ┆ bool     ┆ bool                │
    ╞═════╪═════╪══════════╪═════════════════════╡
    │ Ç   ┆ Ç   ┆ false    ┆ true                │
    └─────┴─────┴──────────┴─────────────────────┘
    """

    warnings.warn(
        "This function is deprecated because Polars>=1.20 has a native str.normalize() method. "
        + "This will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2,
    )

    if form not in ("NFC", "NFKC", "NFD", "NFKD"):
        raise ValueError(
            f"{form} is not a valid Unicode normalization form.",
            " Please specify one of `NFC, NFKC, NFD, NFKD`",
        )

    return pl_plugin(
        symbol="normalize_string",
        args=[to_expr(c)],
        kwargs={"form": form},
        is_elementwise=True,
    )


def map_words(c: str | pl.Expr, mapping: Dict[str, str]) -> pl.Expr:
    """
    Replace words based on the specified mapping.

    Parameters
    ----------
    c : str | pl.Expr
        The string column
    mapping : dict[str, str]
        A dictionary of {word: the replacement}

    Returns
    -------
    pl.Expr

    Examples
    --------
    >>> df = pl.DataFrame({"x": ["one two three"]})
    >>> df.select(pds.map_words("x", {"two": "2"}))
    shape: (1, 1)
    ┌─────────────┐
    │ x           │
    │ ---         │
    │ str         │
    ╞═════════════╡
    │ one 2 three │
    └─────────────┘
    """
    return pl_plugin(
        symbol="map_words",
        args=[to_expr(c)],
        kwargs={"mapping": mapping},
        is_elementwise=True,
    )


def normalize_whitespace(c: str | pl.Expr, only_spaces: bool = False) -> pl.Expr:
    """
    Normalize whitespace to one, e.g. 'a   b' -> 'a b'.

    Parameters
    ----------
    c : str | pl.Expr
        The string column
    only_spaces: bool
        If True, only split on the space character ' ' instead of any whitespace
        character such as '\t' and '\n', by default False

    Returns
    -------
    pl.Expr

    Examples
    --------
    shape: (2, 3)
    ┌─────────┬─────┬────────┐
    │ x       ┆ y   ┆ z      │
    │ ---     ┆ --- ┆ ---    │
    │ str     ┆ str ┆ str    │
    ╞═════════╪═════╪════════╡
    │ a     b ┆ a b ┆ a b    │
    │ a	    b ┆ a b ┆ a	    b│
    └─────────┴─────┴────────┘
    """
    expr = to_expr(c)

    if only_spaces:
        return expr.str.replace_all(" +", " ")

    return pl_plugin(
        symbol="normalize_whitespace",
        args=[expr],
        is_elementwise=True,
    )
