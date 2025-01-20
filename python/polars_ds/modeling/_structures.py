from __future__ import annotations

import polars as pl
from io import StringIO
from enum import Enum
from dataclasses import dataclass
from polars_ds.typing import FitTransformFunc, ExprTransform
from typing import List, Sequence, Dict
from polars._typing import IntoExprColumn


@dataclass
class FitStep:  # Not a FittedStep
    func: FitTransformFunc
    cols: IntoExprColumn | None
    exclude: List[str]

    # Here we allow IntoExprColumn as input so that users can use selectors, or other polars expressions
    # to specify input columns, which adds flexibility.
    # We still need real column names so that the functions in transforms.py will work.
    def fit(self, df: pl.DataFrame | pl.LazyFrame) -> ExprTransform:
        if self.cols is None:
            return self.func(df)
        else:
            real_cols: List[str] = [
                x
                for x in df.lazy().select(self.cols).collect_schema().names()
                if x not in self.exclude
            ]
            return self.func(df, real_cols)


class PLContext(Enum):
    SELECT = "select"
    WITH_COLUMNS = "with_columns"
    SQL = "sql"
    FILTER = "filter"


class PipelineStep:
    def __init__(self, action: str | pl.Expr | Sequence[pl.Expr], context: str | PLContext):
        self.context = context if isinstance(context, PLContext) else PLContext(context)
        if isinstance(action, pl.Expr):
            self.exprs = [action]
        elif isinstance(action, str) and self.context == PLContext.SQL:
            self.exprs = [action]
        elif hasattr(action, "__iter__"):
            self.exprs = list(action)
            if any(not isinstance(e, pl.Expr) for e in self.exprs):
                raise ValueError(
                    "When input is a list, all elements in the list must be polars expressions."
                )
        else:
            raise ValueError(
                "A pipeline step must be either an expression or a list of expressions, or a str in SQL Context."
            )

    @staticmethod
    def from_json(step_dict: dict) -> "PipelineStep":
        context, json_actions = step_dict["context"], step_dict["action"]
        step_context = PLContext(context)
        if step_context in (PLContext.SELECT, PLContext.WITH_COLUMNS):
            actions = [pl.Expr.deserialize(StringIO(e), format="json") for e in json_actions]
        elif step_context == PLContext.SQL:
            actions = str(json_actions)  # SQL only needs the string
        elif step_context == PLContext.FILTER:
            actions = [pl.Expr.deserialize(StringIO(json_actions), format="json")]
        else:
            raise ValueError("Input is not a valid PDS pipeline.")

        return PipelineStep(action=actions, context=step_context)

    def __iter__(self):
        return self.exprs.__iter__()

    def to_json(self) -> Dict:
        if self.context == PLContext.SELECT:
            return {
                "context": "select",
                "action": [e.meta.serialize(format="json") for e in self.exprs],
            }
        elif self.context == PLContext.WITH_COLUMNS:
            return {
                "context": "with_columns",
                "action": [e.meta.serialize(format="json") for e in self.exprs],
            }
        elif self.context == PLContext.SQL:
            return {"context": "sql", "action": self.exprs[0]}
        elif self.context == PLContext.FILTER:
            return {"context": "filter", "action": self.exprs[0].meta.serialize(format="json")}
        else:
            raise ValueError(f"Unknown context: {self.context}")

    def apply_df(self, df: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:
        if self.context == PLContext.SELECT:
            return df.select(self.exprs)
        elif self.context == PLContext.WITH_COLUMNS:
            return df.with_columns(self.exprs)
        elif self.context == PLContext.SQL:
            return pl.SQLContext(df=df, eager=False).execute(self.exprs[0])
        elif self.context == PLContext.FILTER:
            return df.filter(self.exprs[0])
