from __future__ import annotations
from typing import Callable
import polars as pl


def E(
    cols: str | pl.Expr | list[str] | list[pl.Expr],
    mappings: str | list[str],
    *,
    separator: str = "_",
    len_alias: str = "__len__",
    customizer: dict[str, Callable[[pl.Expr], pl.Expr]] | None = None,
) -> list[pl.Expr]:
    """
    Automatically expands Polars expressions so that the syntax can be more concise
    in most cases.

    Parameters
    ----------
    cols
        Either the column name list of column names. Due to limitations in Polars,
        this does not support expression inputs.
    mappings
        String names for any polars expression methods. E.g. ['max'] will map to `pl.col(..).max()`.
        There are some special strings that would represent some more complicated composite expressions,
        and here is a the list of such special mappings as of now:
        'null_rate' -> pl.col('').null_count() / pl.len()
    separator
        The separator between the name of the mapping and the original column name.
    len_alias
        Column alias to `len`.
    customizer
        An additional mapping to add your own custom special mappings. The keys are the name
        of additional mappings, the values are functions that maps pl.Expr to another pl.Expr.

    Examples
    --------
    >>> import polars_ds as pds
    >>> df = pl.DataFrame({
    >>>     "group": ['A', 'A', 'B', 'B', 'A']
    >>>     , "a": [1, 2, 3, 4, 5]
    >>>     , "b": [4, 1, 99, 12, 33]
    >>> })
    >>> df.group_by("group").agg(
    >>>     *pds.E(['a', 'b'], ["min", "max", "n_unique", "len"])
    >>> )
    shape: (2, 8)
    ┌───────┬───────┬───────┬───────┬───────┬────────────┬────────────┬─────────┐
    │ group ┆ a_min ┆ b_min ┆ a_max ┆ b_max ┆ a_n_unique ┆ b_n_unique ┆ __len__ │
    │ ---   ┆ ---   ┆ ---   ┆ ---   ┆ ---   ┆ ---        ┆ ---        ┆ ---     │
    │ str   ┆ i64   ┆ i64   ┆ i64   ┆ i64   ┆ u32        ┆ u32        ┆ u32     │
    ╞═══════╪═══════╪═══════╪═══════╪═══════╪════════════╪════════════╪═════════╡
    │ A     ┆ 1     ┆ 1     ┆ 5     ┆ 33    ┆ 3          ┆ 3          ┆ 3       │
    │ B     ┆ 3     ┆ 12    ┆ 4     ┆ 99    ┆ 2          ┆ 2          ┆ 2       │
    └───────┴───────┴───────┴───────┴───────┴────────────┴────────────┴─────────┘

    >>> customizer = {
    >>>     "quantile_90": lambda x: x.quantile(0.9).name.suffix("_q90")
    >>> }
    >>> df.group_by("group").agg(
    >>>     *pds.E(['a', 'b'], ["min", "max", "null_rate", "quantile_90"], customizer=customizer)
    >>> )
    shape: (2, 9)
    ┌───────┬───────┬───────┬───────┬───┬─────────────┬─────────────┬───────┬───────┐
    │ group ┆ a_min ┆ b_min ┆ a_max ┆ … ┆ a_null_rate ┆ b_null_rate ┆ a_q90 ┆ b_q90 │
    │ ---   ┆ ---   ┆ ---   ┆ ---   ┆   ┆ ---         ┆ ---         ┆ ---   ┆ ---   │
    │ str   ┆ i64   ┆ i64   ┆ i64   ┆   ┆ f64         ┆ f64         ┆ f64   ┆ f64   │
    ╞═══════╪═══════╪═══════╪═══════╪═══╪═════════════╪═════════════╪═══════╪═══════╡
    │ A     ┆ 1     ┆ 1     ┆ 5     ┆ … ┆ 0.0         ┆ 0.0         ┆ 5.0   ┆ 33.0  │
    │ B     ┆ 3     ┆ 12    ┆ 4     ┆ … ┆ 0.0         ┆ 0.0         ┆ 4.0   ┆ 99.0  │
    └───────┴───────┴───────┴───────┴───┴─────────────┴─────────────┴───────┴───────┘
    """

    if isinstance(cols, str):
        columns = [cols]
    elif isinstance(cols, list):
        if any(not isinstance(c, str) for c in cols):
            raise TypeError(
                "Input `cols` must either be a single str/pl.Expr or a list of str/pl.expr."
            )
        columns = list(cols)
    else:
        raise TypeError(
            "Input `cols` must either be a single str/pl.Expr or a list of str/pl.expr."
        )

    in_expr = pl.col(columns)

    if isinstance(mappings, str):
        mappings_ = [mappings]
    else:
        mappings_ = list(mappings)

    SPECIAL_MAPPINGS = {
        "len": pl.len().alias(len_alias),
        "count": pl.len().alias(len_alias),
        "null_rate": (in_expr.null_count() / pl.len()).name.suffix(f"{separator}null_rate"),
    }

    if customizer is not None and isinstance(customizer, dict):
        for k in customizer.keys():
            SPECIAL_MAPPINGS[k] = customizer[k](in_expr)

    bad = [m for m in mappings_ if not (hasattr(in_expr, m) or m in SPECIAL_MAPPINGS)]
    if len(bad) > 0:
        raise ValueError(
            f"Polars expressions does not have `{bad}` method(s) and they do not belong "
            "in the special mappings."
        )

    return [
        SPECIAL_MAPPINGS[m]
        if m in SPECIAL_MAPPINGS
        else getattr(in_expr, m)().name.suffix(f"{separator}{m}")
        for m in mappings_
    ]
