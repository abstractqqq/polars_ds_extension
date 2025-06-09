from __future__ import annotations

import polars as pl
import json
from io import StringIO
from enum import Enum
from dataclasses import dataclass
from polars_ds._utils import to_expr
from polars_ds.typing import FitTransformFunc, ExprTransform
from typing import List, Sequence, Dict, Protocol, ClassVar
from polars._typing import IntoExprColumn

class PLContext(Enum):
    # Regular ExpreStep
    SELECT = "select"
    WITH_COLUMNS = "with_columns"
    SQL = "sql"
    FILTER = "filter"
    # SortStep
    SORT = "sort"
    # 

    def get_context_dict(self) -> Dict:
        return {"step_classif": StepClassif.from_context(self).value, "context": self.value}
    
class StepClassif(Enum):
    EXPR = "expr"
    SORT = "sort"

    def __str__(self) -> str:
        return self.value
        
    def build_from_json(self, data: str | bytes) -> PipelineStep:
        # Build a step from json str
        if self == StepClassif.EXPR:
            return ExprStep.from_json(data)
        elif self == StepClassif.SORT:
            return SortStep.from_json(data)

    @staticmethod
    def from_context(ctx: PLContext) -> "StepClassif":
        if ctx in (PLContext.SELECT, PLContext.WITH_COLUMNS, PLContext.FILTER, PLContext.SQL):
            return StepClassif.EXPR
        elif ctx == PLContext.SORT:
            return StepClassif.SORT
        

class PipelineStep(Protocol):
    classif: ClassVar[StepClassif]
    context: PLContext

    def apply_df(self, df: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame: ...
    def to_json(self) -> str: ...

    @staticmethod
    def from_json(data: str) -> "PipelineStep": ...


class MakeStep:

    @staticmethod
    def make(data: str | bytes) -> PipelineStep:
        data_dict: dict = json.loads(data)
        step = data_dict.get("step_classif", None)
        if step is None: 
            raise ValueError("There is no `step_classif` in the dictionary. Input is not a conformant dict.")
        else:
            return StepClassif(step).build_from_json(data)
    

@dataclass
class SortStep(PipelineStep):
    classif: ClassVar[StepClassif] = StepClassif.SORT
    # ------------------------------------------------
    by_cols: List[pl.Expr]
    descending: List[bool]
    context: PLContext = PLContext.SORT

    @staticmethod
    def from_json(data: str | bytes) -> "SortStep":
        try:
            data_dict = json.loads(data)
            by_cols = [pl.Expr.deserialize(StringIO(e), format="json") for e in data_dict["by_cols"]]
            descending = [bool(e) for e in data_dict["descending"]]
            return SortStep(by_cols = by_cols, descending = descending)
        except Exception as e:
            raise ValueError(f"Input is not a valid `SortStep`. Must have `by_cols` and `descending` keys with valid values.\nOriginal error: {e}")

    def __init__(self, by_cols: str | pl.Expr | Sequence[str] | Sequence[pl.Expr], descending: bool | Sequence[bool]):

        if isinstance(by_cols, (str, pl.Expr)):
            self.by_cols = [to_expr(by_cols)]
        elif isinstance(by_cols, list):
            self.by_cols = [to_expr(e) for e in by_cols]
        else:
            raise ValueError("Input `by_cols` can only be scalar str/pl.Expr or a list of them.")
        
        if isinstance(descending, bool):
            self.descending = [descending]
        elif isinstance(descending, list):
            self.descending = [bool(b) for b in descending]
        else:
            raise ValueError("Input `descending` must be a scalar bool or a list of bools.")
        
        if len(self.by_cols) != len(self.descending):
            raise ValueError(f"Input `by_cols` (len {len(by_cols)}) doesn't match the length of `descending` (len {len(self.descending)}).")

    def apply_df(self, df: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:
        return df.sort(by=self.by_cols, descending=self.descending)
    
    def to_json(self) -> str:
        d = self.context.get_context_dict() | {
            "by_cols": [e.meta.serialize(format="json") for e in self.by_cols],
            "descending": self.descending
        }
        return json.dumps(d)


@dataclass
class FitStep:  
    # This doesn't satisfy the 'PipelineStep' protocol
    # This is turned into an ExprStep at the time of fit(materialization) in a blueprint.

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

@dataclass
class ExprStep(PipelineStep):
    classif: ClassVar[StepClassif] = StepClassif.EXPR
    # ------------------------------------------------
    exprs: list[pl.Expr]
    context: PLContext

    def __init__(
        self, 
        exprs: str | pl.Expr | Sequence[pl.Expr], 
        context: str | PLContext
    ):
        self.context = context if isinstance(context, PLContext) else PLContext(context)
        if isinstance(exprs, pl.Expr):
            self.exprs = [exprs]
        elif isinstance(exprs, str) and self.context == PLContext.SQL:
            self.exprs = [exprs]
        elif isinstance(exprs, str) and self.context == PLContext.FILTER:
            # This case only happens when we deserialize a filter expression from json
            self.exprs = [pl.Expr.deserialize(StringIO(exprs), format="json")]
        elif isinstance(exprs, list):
            self.exprs = list(exprs)
            if any(not isinstance(e, pl.Expr) for e in self.exprs):
                raise ValueError(
                    "When input is a list, all elements in the list must be polars expressions."
                )
        else:
            raise ValueError(
                "A pipeline step must be either an expression or a list of expressions, or a str in SQL Context."
            )

    @staticmethod
    def from_json(data: str | bytes) -> "ExprStep":
        data_dict = json.loads(data)
        context, json_exprs = data_dict["context"], data_dict["exprs"]
        step_context = PLContext(context)
        if step_context in (PLContext.SELECT, PLContext.WITH_COLUMNS):
            exprs = [pl.Expr.deserialize(StringIO(e), format="json") for e in json_exprs]
        elif step_context == PLContext.FILTER:
            exprs = [pl.Expr.deserialize(StringIO(json_exprs), format="json")]
        elif step_context == PLContext.SQL:
            exprs = str(json_exprs) # SQL only needs the string
        else:
            raise ValueError("Input is not a valid PDS pipeline.")

        return ExprStep(exprs=exprs, context=step_context)

    def __iter__(self):
        return self.exprs.__iter__()

    def to_json(self) -> str:
        ctx = self.context
        d = ctx.get_context_dict()
        if ctx in (PLContext.SELECT, PLContext.WITH_COLUMNS):
            d["exprs"] = [e.meta.serialize(format="json") for e in self.exprs]
        elif self.context == PLContext.SQL:
            d["exprs"] = self.exprs[0]
        elif self.context == PLContext.FILTER:
            d["exprs"] = self.exprs[0].meta.serialize(format="json")
        else:
            raise ValueError(f"Unknown context: {self.context}")
        
        return json.dumps(d)

    def apply_df(self, df: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:
        if self.context == PLContext.SELECT:
            return df.select(self.exprs)
        elif self.context == PLContext.WITH_COLUMNS:
            return df.with_columns(self.exprs)
        elif self.context == PLContext.SQL:
            return pl.SQLContext(df=df, eager=False).execute(self.exprs[0])
        elif self.context == PLContext.FILTER:
            return df.filter(self.exprs[0])
