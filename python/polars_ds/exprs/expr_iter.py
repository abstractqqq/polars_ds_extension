"""Iteration related helper expressions"""

from __future__ import annotations

import polars as pl

# Internal dependencies
from polars_ds._utils import pl_plugin, str_to_expr

__all__ = ["combinations", "product"]


def product(s1: str | pl.Expr, s2: str | pl.Expr) -> pl.Expr:
    """
    Get the cartesian product of two series. Only non-nulls values will be used.

    Parameters
    ----------
    s1
        The first column / series
    s2
        The second column / series

    Examples
    --------
    >>> df = pl.DataFrame({
    >>> "a": [1, 2]
    >>> , "b": [4, 5]
    >>> })
    >>> df.select(
    >>>     pds.product("a", "b")
    >>> )
    shape: (4, 1)
    ┌───────────┐
    │ a         │
    │ ---       │
    │ list[i64] │
    ╞═══════════╡
    │ [1, 4]    │
    │ [1, 5]    │
    │ [2, 4]    │
    │ [2, 5]    │
    └───────────┘

    >>> df = pl.DataFrame({
    >>>     "a": [[1,2], [3,4]]
    >>>     , "b": [[3], [1, 2]]
    >>> }).with_row_index()
    >>> df
    shape: (2, 3)
    ┌───────┬───────────┬───────────┐
    │ index ┆ a         ┆ b         │
    │ ---   ┆ ---       ┆ ---       │
    │ u32   ┆ list[i64] ┆ list[i64] │
    ╞═══════╪═══════════╪═══════════╡
    │ 0     ┆ [1, 2]    ┆ [3]       │
    │ 1     ┆ [3, 4]    ┆ [1, 2]    │
    └───────┴───────────┴───────────┘

    >>> df.group_by(
    >>>     "index"
    >>> ).agg(
    >>>     pds.product(
    >>>         pl.col("a").list.explode()
    >>>         , pl.col("b").list.explode()
    >>>     ).alias("product")
    >>> )
    shape: (2, 2)
    ┌───────┬────────────────────────────┐
    │ index ┆ product                    │
    │ ---   ┆ ---                        │
    │ u32   ┆ list[list[i64]]            │
    ╞═══════╪════════════════════════════╡
    │ 0     ┆ [[1, 3], [2, 3]]           │
    │ 1     ┆ [[3, 1], [3, 2], … [4, 2]] │
    └───────┴────────────────────────────┘
    """
    return pl_plugin(
        symbol="pl_product",
        args=[str_to_expr(s1).drop_nulls(), str_to_expr(s2).drop_nulls()],
        changes_length=True,
    )


def combinations(source: str | pl.Expr, k: int, unique: bool = False) -> pl.Expr:
    """
    Get all k-combinations of non-null values in source. This is an expensive operation, as
    n choose k can grow very fast.

    Parameters
    ----------
    source
        Input source column, must have numeric or string type
    k
        The k in N choose k
    unique
        Whether to run .unique() on the source column

    Examples
    --------
    >>> df = pl.DataFrame({
    >>>     "category": ["a", "a", "a", "b", "b"],
    >>>     "values": [1, 2, 3, 4, 5]
    >>> })
    >>> df.select(
    >>>     pds.combinations("values", 3)
    >>> )
    shape: (10, 1)
    ┌───────────┐
    │ values    │
    │ ---       │
    │ list[i64] │
    ╞═══════════╡
    │ [1, 2, 3] │
    │ [1, 2, 4] │
    │ [1, 2, 5] │
    │ [1, 3, 4] │
    │ [1, 3, 5] │
    │ [1, 4, 5] │
    │ [2, 3, 4] │
    │ [2, 3, 5] │
    │ [2, 4, 5] │
    │ [3, 4, 5] │
    └───────────┘
    >>> df.group_by("category").agg(
    >>>     pds.combinations("values", 2)
    >>> )
    shape: (2, 2)
    ┌──────────┬──────────────────────────┐
    │ category ┆ values                   │
    │ ---      ┆ ---                      │
    │ str      ┆ list[list[i64]]          │
    ╞══════════╪══════════════════════════╡
    │ a        ┆ [[1, 2], [1, 3], [2, 3]] │
    │ b        ┆ [[4, 5]]                 │
    └──────────┴──────────────────────────┘
    """
    s = (
        str_to_expr(source).unique().drop_nulls().sort()
        if unique
        else str_to_expr(source).drop_nulls()
    )
    return pl_plugin(
        symbol="pl_combinations",
        args=[s],
        changes_length=True,
        kwargs={
            "k": k,
        },
    )
