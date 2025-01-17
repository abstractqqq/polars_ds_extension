from __future__ import annotations
import polars as pl
from enum import Enum
from typing import List, Sequence
import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:  # 3.10, 3.9, 3.8
    from typing_extensions import Self


class PLContext(Enum):
    SELECT = 0
    WITH_COLUMNS = 1
    SQL = 2
    FILTER = 3

    @staticmethod
    def from_str(s: str) -> "PLContext":
        ss = s.lower()
        if ss == "select":
            return PLContext.SELECT
        elif ss in ("with_columns", "withcolumns"):
            return PLContext.WITH_COLUMNS
        elif ss == "filter":
            return PLContext.FILTER
        elif ss == "sql":
            return PLContext.SQL
        else:
            raise ValueError(f"Cannot turn string into a PLContext. String value: {s}")


class PLStep:
    def __init__(self, data: str | pl.Expr | Sequence[pl.Expr], context: PLContext):
        if isinstance(data, pl.Expr):
            self.exprs = [data]
        elif isinstance(data, str):
            self.exprs = [pl.col(data)]
        elif hasattr(data, "__iter__"):
            self.exprs = list(data)
        else:
            raise ValueError(
                "A pipeline step must be either a str or expression or a list of str and expressions."
            )

        self.context = context

    def __iter__(self):
        return self.exprs.__iter__()

    def apply_df(self, df: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:
        if self.context == PLContext.SELECT:
            return df.select(self.exprs)
        elif self.context == PLContext.WITH_COLUMNS:
            return df.with_columns(self.exprs)
        elif self.context == PLContext.SQL:
            pl.SQLContext(df=df, eager=False).execute(self.exprs[0])
        elif self.context == PLContext.FILTER:
            return df.filter(self.exprs[0])
