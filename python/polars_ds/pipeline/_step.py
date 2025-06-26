from __future__ import annotations

import polars as pl
import json
from io import StringIO
from enum import Enum
from polars_ds._utils import to_expr
from polars_ds.typing import FitTransformFunc, ExprTransform
from typing import List, Sequence, Dict, Protocol, Literal
from polars._typing import IntoExprColumn


class PLContext(Enum):
    # Regular ExpreStep (purely expressed-based)
    SELECT = "select"
    WITH_COLUMNS = "with_columns"
    FILTER = "filter"
    EXPLODE = "explode"
    # SQLStep
    SQL = "sql"
    # SortStep
    SORT = "sort"
    # GROUP_BY_AGG
    GROUP_BY_AGG = "group_by_agg"
    # GROUP_BY_DYNAMIC_AGG
    GROUP_BY_DYN_AGG = "group_by_dyn_agg"

    def get_context_dict(self) -> Dict:
        return {"context": self.value}

    def build_step(self, deser_dict: Dict) -> PipelineStep:
        """
        Build a step from an already deserialized dict. The dict may contain values of type list[str],
        which requires further deserialization by polars from str to pl.Expr.
        """
        if self in (PLContext.SELECT, PLContext.WITH_COLUMNS, PLContext.FILTER, PLContext.EXPLODE):
            return ExprStep.from_partial_dict(deser_dict)
        elif self == PLContext.SORT:
            return SortStep.from_partial_dict(deser_dict)
        elif self == PLContext.GROUP_BY_AGG:
            return GroupByAggStep.from_partial_dict(deser_dict)
        elif self == PLContext.GROUP_BY_DYN_AGG:
            return GroupByDynAggStep.from_partial_dict(deser_dict)
        elif self == PLContext.SQL:
            return SQLStep.from_partial_dict(deser_dict)
        else:
            raise ValueError(f"Unknown PLContext: {self}")


class PipelineStep(Protocol):
    context: PLContext

    def apply_df(self, df: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame: ...
    def to_json(self) -> str: ...

    @staticmethod
    def from_partial_dict(data: Dict) -> "PipelineStep":
        """
        See MakeStep.make's docstrings.
        """
        ...


class MakeStep:
    __slots__ = ()

    @staticmethod
    def make(deser_dict: Dict) -> PipelineStep:
        """
        Constructs a PipelineStep from a partially deserialized dictionary. The dictionary may be partially
        deserialized because some fields may actually be list[str], but it should be deserialized to list[pl.Expr]
        where the string elements will be further deserialized to Polars Expressions based on context.
        """
        ctx = deser_dict.get("context", None)
        if ctx is None:
            raise ValueError(
                "There is no 'context' key in the dictionary. Input is not conformant."
            )
        else:
            return PLContext(ctx).build_step(deser_dict)


class SQLStep:
    """
    Container for SQL transformation on a Polars dataframe.
    """

    __slots__ = ("sql_str", "context")

    def __init__(self, sql_str: str):
        self.sql_str: str = sql_str
        self.context: PLContext = PLContext.SQL

    @staticmethod
    def from_partial_dict(d: Dict) -> "SQLStep":
        return SQLStep(sql_str=d["sql_str"])

    def apply_df(self, df: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:
        return pl.SQLContext(df=df).execute(self.sql_str)

    def to_json(self) -> str:
        d = self.context.get_context_dict() | {"sql_str": self.sql_str}
        return json.dumps(d)


class SortStep:
    """
    Container for a polars df.sort() transform.
    """

    __slots__ = ("by", "descending", "context")

    def __init__(
        self,
        by: str | pl.Expr | Sequence[str] | Sequence[pl.Expr],
        descending: bool | Sequence[bool],
    ):
        if isinstance(by, (str, pl.Expr)):
            self.by: List[pl.Expr] = [to_expr(by)]
        elif isinstance(by, list):
            self.by: List[pl.Expr] = [to_expr(e) for e in by]
        else:
            raise ValueError("Input `by` can only be scalar str/pl.Expr or a list of them.")

        if isinstance(descending, bool):
            self.descending: List[bool] = [descending]
        elif isinstance(descending, list):
            self.descending: List[bool] = [bool(b) for b in descending]
        else:
            raise ValueError("Input `descending` must be a scalar bool or a list of bools.")

        if len(self.by) != len(self.descending):
            raise ValueError(
                f"Input `by` (len {len(by)}) doesn't match the length of `descending` (len {len(self.descending)})."
            )

        self.context: PLContext = PLContext.SORT

    @staticmethod
    def from_partial_dict(d: Dict) -> "SortStep":
        by = [pl.Expr.deserialize(StringIO(e), format="json") for e in d["by"]]
        descending = [bool(e) for e in d["descending"]]
        return SortStep(by=by, descending=descending)

    def apply_df(self, df: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:
        return df.sort(by=self.by, descending=self.descending)

    def to_json(self) -> str:
        d = self.context.get_context_dict() | {
            "by": [e.meta.serialize(format="json") for e in self.by],
            "descending": self.descending,
        }
        return json.dumps(d)


class GroupByAggStep:
    """
    Container for a polars df.group_by(...).agg(...) transform.
    """

    __slots__ = ("by", "agg", "maintain_order", "context")

    def __init__(
        self,
        by: str | pl.Expr | Sequence[str] | Sequence[pl.Expr],
        agg: List[pl.Expr],
        maintain_order: bool = False,
    ):
        if isinstance(by, (str, pl.Expr)):
            self.by: List[pl.Expr] = [to_expr(by)]
        elif isinstance(by, list):
            self.by: List[pl.Expr] = [to_expr(e) for e in by]
        else:
            raise ValueError("Input `by` can only be scalar str/pl.Expr or a list of them.")

        self.agg: List[pl.Expr] = agg
        self.context: PLContext = PLContext.GROUP_BY_AGG
        self.maintain_order = maintain_order

    @staticmethod
    def from_partial_dict(d: dict) -> "GroupByAggStep":
        by = [pl.Expr.deserialize(StringIO(e), format="json") for e in d["by"]]
        agg = [pl.Expr.deserialize(StringIO(e), format="json") for e in d["agg"]]
        return GroupByAggStep(by=by, agg=agg, maintain_order=d["maintain_order"])

    def apply_df(self, df: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:
        return df.group_by(self.by, maintain_order=self.maintain_order).agg(self.agg)

    def to_json(self) -> str:
        d = self.context.get_context_dict() | {
            "by": [e.meta.serialize(format="json") for e in self.by],
            "agg": [e.meta.serialize(format="json") for e in self.agg],
            "maintain_order": self.maintain_order,
        }
        return json.dumps(d)


class GroupByDynAggStep:
    """
    Container for a polars df.group_by_dynamic(...).agg(...) transform.
    """

    __slots__ = (
        "index_column",
        "every",
        "agg",
        "group_by",
        "period",
        "offset",
        "include_boundaries",
        "closed",
        "label",
        "start_by",
        "context",
    )

    def __init__(
        self,
        index_column: str,
        every: str,
        agg: List[pl.Expr],
        period: str | None = None,
        offset: str | None = None,
        include_boundaries: bool = False,
        closed: Literal["left", "right", "both", "none"] = "left",
        label: Literal["left", "right", "datapoint"] = "left",
        group_by: str | pl.Expr | Sequence[str] | Sequence[pl.Expr] | None = None,
        start_by: Literal[
            "window",
            "datapoint",
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ] = "window",
    ):
        if closed not in ["left", "right", "both", "none"]:
            raise ValueError(
                "Input `closed` must be one of ['left', 'right', 'both', 'none']. See polars's group_by_dynamic for more info."
            )

        if label not in ["left", "right", "datapoint"]:
            raise ValueError(
                "Input `closed` must be one of ['left', 'right', 'datapoint']. See polars's group_by_dynamic for more info."
            )

        if start_by not in [
            "window",
            "datapoint",
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ]:
            raise ValueError(
                "Input `closed` must be one of ['window', 'datapoint', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']. See polars's group_by_dynamic for more info."
            )

        # Because of a strange bug.
        if isinstance(index_column, pl.Expr):
            raise ValueError(
                "Input `index_column` must be a string (name of a column) and not a Polars expression."
            )

        self.index_column: str = index_column
        self.every: str = every
        self.period: str | None = period
        self.offset: str | None = offset
        self.include_boundaries: bool = include_boundaries
        self.closed: Literal["left", "right", "both", "none"] = closed
        self.label: Literal["left", "right", "datapoint"] = label
        self.start_by: Literal[
            "window",
            "datapoint",
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ] = start_by

        if isinstance(group_by, (str, pl.Expr)):
            self.group_by: List[pl.Expr] | None = [to_expr(group_by)]
        elif isinstance(group_by, list):
            self.group_by: List[pl.Expr] | None = [to_expr(e) for e in group_by]
        elif group_by is None:
            self.group_by: List[pl.Expr] | None = None
        else:
            raise ValueError(
                "Input `by` can only be scalar str / pl.Expr or a list of them or None."
            )

        self.agg: List[pl.Expr] = agg
        self.context: PLContext = PLContext.GROUP_BY_DYN_AGG

    @staticmethod
    def from_partial_dict(d: Dict) -> "GroupByDynAggStep":
        return GroupByDynAggStep(
            index_column=d["index_column"],
            every=d["every"],
            period=d["period"],
            offset=d["offset"],
            include_boundaries=d["include_boundaries"],
            label=d["label"],
            start_by=d["start_by"],
            group_by=None
            if d["group_by"] is None
            else [pl.Expr.deserialize(StringIO(e), format="json") for e in d["group_by"]],
            agg=[pl.Expr.deserialize(StringIO(e), format="json") for e in d["agg"]],
        )

    def apply_df(self, df: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:
        return df.group_by_dynamic(
            index_column=self.index_column,
            every=self.every,
            period=self.period,
            offset=self.offset,
            include_boundaries=self.include_boundaries,
            label=self.label,
            group_by=self.group_by,
            start_by=self.start_by,
        ).agg(self.agg)

    def to_json(self) -> str:
        d = self.context.get_context_dict() | {
            "group_by": None
            if self.group_by is None
            else [e.meta.serialize(format="json") for e in self.group_by],
            "agg": [e.meta.serialize(format="json") for e in self.agg],
            "index_column": self.index_column,
            "every": self.every,
            "period": self.period,
            "offset": self.offset,
            "include_boundaries": self.include_boundaries,
            "closed": self.closed,
            "label": self.label,
            "start_by": self.start_by,
        }
        return json.dumps(d)


class ExprStep:
    """
    Container for either one of these polars transforms:

    1. df.select(...)
    2. df.with_columns(...)
    3. df.filter(...)
    """

    __slots__ = ("exprs", "context")

    def __init__(self, exprs: pl.Expr | Sequence[pl.Expr], context: str | PLContext):
        self.context: PLContext = context if isinstance(context, PLContext) else PLContext(context)
        if isinstance(exprs, pl.Expr):
            self.exprs = [exprs]
        elif isinstance(exprs, list):
            self.exprs = list(exprs)
            if any(not isinstance(e, pl.Expr) for e in self.exprs):
                raise ValueError(
                    "When input is a list, all elements in the list must be polars expressions."
                )
        else:
            raise ValueError(
                "A pipeline step must be either an expression or a list of expressions."
            )

    @staticmethod
    def from_partial_dict(d: dict) -> "ExprStep":
        context, json_exprs = d["context"], d["exprs"]
        step_context = PLContext(context)
        if step_context in (PLContext.SELECT, PLContext.WITH_COLUMNS, PLContext.EXPLODE):
            exprs = [pl.Expr.deserialize(StringIO(e), format="json") for e in json_exprs]
        elif step_context == PLContext.FILTER:
            exprs = [pl.Expr.deserialize(StringIO(json_exprs), format="json")]
        elif step_context == PLContext.SQL:
            exprs = str(json_exprs)  # SQL context json_exprs is just a str
        else:
            raise ValueError("Input is not a valid PDS pipeline.")

        return ExprStep(exprs=exprs, context=step_context)

    def to_json(self) -> str:
        ctx = self.context
        d = ctx.get_context_dict()
        if ctx in (PLContext.SELECT, PLContext.WITH_COLUMNS, PLContext.EXPLODE):
            d["exprs"] = [e.meta.serialize(format="json") for e in self.exprs]
        elif self.context == PLContext.SQL:
            d["exprs"] = self.exprs[0]
        elif self.context == PLContext.FILTER:
            d["exprs"] = self.exprs[0].meta.serialize(format="json")
        else:  # Should never reach here
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
        elif self.context == PLContext.EXPLODE:
            return df.explode(self.exprs)


class FitStep:
    # This doesn't satisfy the 'PipelineStep' protocol
    # This is turned into an ExprStep at the time of fit(materialization) in a blueprint.

    __slots__ = ("func", "cols", "exclude")

    def __init__(self, func: FitTransformFunc, cols: IntoExprColumn | None, exclude: List[str]):
        self.func: FitTransformFunc = func
        self.cols: IntoExprColumn | None = cols
        self.exclude: List[str] = exclude

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


# ---------------------------------------------------------------------------------------
# Steps that don't need fitting, can be directly serialized.
# FitStep becomes ExprStep after fitting
_SERIALIZABLE_STEPS = [ExprStep, SortStep, SQLStep, GroupByAggStep, GroupByDynAggStep]
