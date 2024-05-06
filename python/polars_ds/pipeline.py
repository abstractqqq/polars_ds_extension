import polars as pl
import transforms as t  # local transforms module
import sys
from functools import partial
from dataclasses import dataclass
from .type_alias import (
    List,
    Union,
    TypeAlias,
    PolarsFrame,
    ExprTransform,
    FitTransformFunc,
    SimpleImputeMethod,
    SimpleScaleMethod,
    PipeComponentType,
)

if sys.version_info >= (3, 11):
    from typing import Self
else:  # 3.10, 3.9, 3.8
    from typing_extensions import Self


@dataclass
class SelectComponent:
    exprs: ExprTransform


@dataclass
class WithColumnsComponent:
    exprs: ExprTransform


@dataclass
class FitComponent:
    func: FitTransformFunc

    def fit(self, df: PolarsFrame) -> ExprTransform:
        return self.func(df)


PipeComponent: TypeAlias = Union[FitComponent, SelectComponent, WithColumnsComponent]
FittedPipeComponent: TypeAlias = Union[SelectComponent, WithColumnsComponent]


class Pipeline:
    """

    If the input df is lazy, the pipeline will collect once at the time of fit.
    """

    def __init__(self, df: PolarsFrame):
        self._df: pl.LazyFrame = df.lazy()
        self._steps: List[PipeComponent] = []
        self._transform_queue: List[FittedPipeComponent] = []

    def impute(self, cols: List[str], method: SimpleImputeMethod = "mean") -> Self:
        self._steps.append(FitComponent(partial(t.impute, self._df, cols=cols, method=method)))
        return self

    def scale(self, cols: List[str], method: SimpleScaleMethod = "standard") -> Self:
        self._steps.append(FitComponent(partial(t.scale, self._df, cols=cols, method=method)))
        return self

    def robust_scale(self, cols: List[str], q1: float, q2: float) -> Self:
        self._steps.append(FitComponent(partial(t.robust_scale, self._df, cols=cols, q1=q1, q2=q2)))
        return self

    def center(self, cols: List[str]) -> Self:
        self._steps.append(FitComponent(partial(t.scale, self._df, cols=cols)))
        return self

    def append_expr(self, exprs: ExprTransform, ct: PipeComponentType = "with_columns") -> Self:
        if ct == "with_columns":
            self._steps.append(WithColumnsComponent(exprs))
        elif ct == "select":
            self._steps.append(SelectComponent(exprs))
        else:
            raise ValueError("This method only works for appending expressions.")

    def fit(self) -> Self:
        df: pl.DataFrame = self._df.collect()  # Add config here if streaming is needed
        self._transform_queue.clear()
        for comp in self._steps:
            if comp is FitComponent:
                self._transform_queue.append(WithColumnsComponent(comp.fit(df)))
            elif comp is WithColumnsComponent or comp is SelectComponent:
                self._transform_queue.append(comp)
            else:
                raise ValueError("Not a valid pipe component.")
        return self
