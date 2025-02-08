"""Iteration related helper expressions"""

from __future__ import annotations

import polars as pl

# Internal dependencies
from polars_ds._utils import pl_plugin, str_to_expr

__all__ = [
    "combinations",
]


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
    s = str_to_expr(source).unique().sort() if unique else str_to_expr(source)
    return pl_plugin(
        symbol="pl_combinations",
        args=[s],
        changes_length=True,
        kwargs={
            "k": k,
        },
    )
