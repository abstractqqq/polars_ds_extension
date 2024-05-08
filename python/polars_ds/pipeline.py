import polars as pl
from . import transforms as t
import sys
from functools import partial
from dataclasses import dataclass
from polars.type_aliases import IntoExprColumn
from .type_alias import (
    List,
    Optional,
    Union,
    Dict,
    TypeAlias,
    PolarsFrame,
    ExprTransform,
    FitTransformFunc,
    SimpleImputeMethod,
    SimpleScaleMethod,
)

if sys.version_info >= (3, 11):
    from typing import Self
else:  # 3.10, 3.9, 3.8
    from typing_extensions import Self

__all__ = ["Pipeline"]


@dataclass
class SelectComponent:
    exprs: ExprTransform


@dataclass
class WithColumnsComponent:
    exprs: ExprTransform


@dataclass
class FitComponent:
    func: FitTransformFunc
    cols: IntoExprColumn

    # Here we allow cols as input so that users can use selectors, or other polars expressions
    # to specify input columns, which adds flexibility.
    # We still need real column names so that the functions in transforms.py will work.
    def fit(self, df: PolarsFrame) -> ExprTransform:
        real_cols = df.select(self.cols).columns
        return self.func(df, real_cols)


PipeComponent: TypeAlias = Union[FitComponent, SelectComponent, WithColumnsComponent]
FittedPipeComponent: TypeAlias = Union[SelectComponent, WithColumnsComponent]


class Pipeline:
    """

    If the input df is lazy, the pipeline will collect at the time of fit.
    """

    def __init__(self, df: PolarsFrame, lowercase: bool = False):
        """ """
        if lowercase:
            self._df: pl.LazyFrame = df.lazy().select(
                pl.col(c).alias(c.lower()) for c in df.columns
            )
        else:
            self._df: pl.LazyFrame = df.lazy()

        self._steps: List[PipeComponent] = []
        self._transform_queue: List[FittedPipeComponent] = []

    def impute(self, cols: IntoExprColumn, method: SimpleImputeMethod = "mean") -> Self:
        """
        Impute null values in the given columns.

        Parameters
        ----------
        cols
            Any Polars expression that can be understood as columns.
        method
            One of `mean`, `median`, `mode`. If `mode`, a random value will be chosen if there is
            a tie.
        """
        self._steps.append(FitComponent(partial(t.impute, method=method), cols))
        return self

    def scale(self, cols: IntoExprColumn, method: SimpleScaleMethod = "standard") -> Self:
        """
        Scales values in the given columns.

        Parameters
        ----------
        cols
            Any Polars expression that can be understood as columns.
        method
            One of `standard`, `min_max`, `abs_max`
        """
        self._steps.append(FitComponent(partial(t.scale, method=method), cols))
        return self

    def robust_scale(self, cols: IntoExprColumn, q1: float, q2: float) -> Self:
        """
        Performs robust scaling on the given columns

        Paramters
        ---------
        cols
            Any Polars expression that can be understood as columns.
        q1
            The lower quantile value
        q2
            The higher quantile value
        """
        self._steps.append(FitComponent(partial(t.robust_scale, q1=q1, q2=q2), cols))
        return self

    def center(self, cols: IntoExprColumn) -> Self:
        """
        Centers the columns by subtracting each with its mean.

        Paramters
        ---------
        cols
            Any Polars expression that can be understood as columns.
        """
        self._steps.append(FitComponent(partial(t.center), cols))
        return self

    def select(self, cols: IntoExprColumn) -> Self:
        """
        Selects the columns from the dataset.

        Paramters
        ---------
        cols
            Any Polars expression that can be understood as columns.
        """
        self._steps.append(SelectComponent(cols))
        return self

    def remove(self, cols: IntoExprColumn) -> Self:
        """
        Removes the columns from the dataset.

        Paramters
        ---------
        cols
            Any Polars expression that can be understood as columns.
        """
        self._steps.append(SelectComponent(pl.all().exclude(cols)))
        return self

    def rename(self, rename_dict: Dict[str, str]) -> pl.Expr:
        """
        Renames the columns by the mapping.

        Paramters
        ---------
        rename_dict
            The name mapping
        """
        old = list(rename_dict.keys())
        self._steps.append(
            WithColumnsComponent([pl.col(k).alias(v) for k, v in rename_dict.items()])
        )
        return self.remove(old)

    def lowercase(self) -> Self:
        """
        Lowercases all column names.
        """
        self._steps.append(SelectComponent([pl.col(c).alias(c.lower()) for c in self._df.columns]))
        return self

    def one_hot_encode(
        self, cols: IntoExprColumn, separator: str = "_", drop_first: bool = False
    ) -> Self:
        """ """
        self._steps.append(
            FitComponent(
                partial(t.one_hot_encode, separator=separator, drop_first=drop_first), cols
            )
        )
        return self

    def append_expr(self, exprs: ExprTransform, is_select: bool = False) -> Self:
        """
        Appends the expressions to the pipeline.

        Paramters
        ---------
        exprs
            Either a single expression or a list of expressions.
        is_select
            If true, the expression will be executed in a .select(..) context. If false, they
            will be executed in a .with_columns(..) context.
        """
        if is_select:
            self._steps.append(SelectComponent(exprs))
        else:
            self._steps.append(WithColumnsComponent(exprs))
        return self

    def append_fit_func(self, *args, **kwargs) -> Self:
        return NotImplemented

    def finish(self) -> Self:
        """
        Finish the pipeline preparation, fit and learn all the paramters needed.
        """
        self._transform_queue.clear()
        df: pl.DataFrame = self._df.collect()  # Add config here if streaming is needed
        # Let this lazy plan go through the fit process. The frame will be collected temporarily but
        # the collect should be and optimized.
        df_lazy: pl.LazyFrame = df.lazy()
        for comp in self._steps:
            if isinstance(comp, FitComponent):
                exprs = comp.fit(df_lazy)
                self._transform_queue.append(WithColumnsComponent(exprs))
                df_lazy = df_lazy.with_columns(exprs)
            elif isinstance(comp, WithColumnsComponent):
                self._transform_queue.append(comp)
                df_lazy = df_lazy.with_columns(comp.exprs)
            elif isinstance(comp, SelectComponent):
                self._transform_queue.append(comp)
                df_lazy = df_lazy.select(comp.exprs)
            else:
                raise ValueError("Not a valid PipeComponent.")
        return self

    def fit(self) -> Self:
        """
        Alias for self.finish()
        """
        return self.finish()

    def transform(self, df: Optional[PolarsFrame] = None) -> pl.DataFrame:
        """
        Transforms the df using the learned expressions.

        Paramters
        ---------
        df
            If none, transform the df that the pipe is initialized with. Otherwise, perform
            the learned transformations on the incoming df.
        """
        if df is None:
            df_out = self._df.lazy()
        else:
            df_out = df.lazy()

        for comp in self._transform_queue:
            if isinstance(comp, WithColumnsComponent):
                df_out = df_out.with_columns(comp.exprs)
            elif isinstance(comp, SelectComponent):
                df_out = df_out.select(comp.exprs)
            else:
                raise ValueError("Not a valid FittedPipeComponent.")

        return df_out.collect()  # Add config here if streaming is needed

    def show_graph(self, optimized: bool = True) -> Optional[str]:
        """
        Shows the execution graph.

        Parameters
        ----------
        optimized
            Whether this will show the optimized plan or not.
        """
        df_out = self._df.lazy()
        for comp in self._transform_queue:
            if isinstance(comp, WithColumnsComponent):
                df_out = df_out.with_columns(comp.exprs)
            elif isinstance(comp, SelectComponent):
                df_out = df_out.select(comp.exprs)

        return df_out.show_graph(optimized=optimized)
