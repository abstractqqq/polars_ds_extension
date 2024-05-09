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
        real_cols: List[str] = df.select(self.cols).columns
        return self.func(df, real_cols)


PipeComponent: TypeAlias = Union[FitComponent, SelectComponent, WithColumnsComponent]
FittedPipeComponent: TypeAlias = Union[SelectComponent, WithColumnsComponent]


class Pipeline:
    """

    If the input df is lazy, the pipeline will collect at the time of fit.
    """

    def __init__(
        self,
        df: PolarsFrame,
        name: str = "test",
        lowercase: bool = False,
        target: Optional[str] = None,
    ):
        """ """
        if lowercase:
            self._df: pl.LazyFrame = df.lazy().select(
                pl.col(c).alias(c.lower()) for c in df.columns
            )
        else:
            self._df: pl.LazyFrame = df.lazy()

        self.name: str = name
        self.target: Optional[str] = target
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
        """
        Find the unique values in the string/categorical columns and one-hot encode them. This will NOT
        consider nulls as one of the unique values. Append pl.col(c).is_null().cast(pl.UInt8)
        expression to the pipeline if you want null indicators.

        Parameters
        ----------
        cols
            Any Polars expression that can be understood as columns.
        separator
            E.g. if column name is `col` and `a` is an elemenet in it, then the one-hot encoded column will be called
            `col_a` where the separator `_` is used.
        drop_first
            Whether to drop the first distinct value (in terms of str/categorical order). This helps with reducing
            dimension and prevents some issues from linear dependency.
        """
        self._steps.append(
            FitComponent(
                partial(t.one_hot_encode, separator=separator, drop_first=drop_first), cols
            )
        )
        return self.remove(cols)

    def target_encode(
        self,
        cols: IntoExprColumn,
        /,
        target: Union[str, pl.Expr],
        min_samples_leaf: int = 20,
        smoothing: float = 10.0,
        default: Optional[float] = None,
    ) -> Self:
        """
        Target encode the given variables.

        Note: nulls will be encoded as well.

        Parameters
        ----------
        cols
            Any Polars expression that can be understood as columns. Columns of type != string/categorical
            will not produce any expression.
        target
            The target column
        min_samples_leaf
            A regularization factor
        smoothing
            Smoothing effect to balance categorical average vs prior
        default
            If new value is encountered during transform, it will be mapped to default

        Reference
        ---------
        https://contrib.scikit-learn.org/category_encoders/targetencoder.html
        """
        self._steps.append(
            FitComponent(
                partial(
                    t.target_encode,
                    target=target,
                    min_samples_leaf=min_samples_leaf,
                    smoothing=smoothing,
                    default=default,
                ),
                cols,
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
        for component in self._steps:
            if isinstance(component, FitComponent):
                exprs = component.fit(df_lazy)
                self._transform_queue.append(WithColumnsComponent(exprs))
                df_lazy = df_lazy.with_columns(exprs)
            elif isinstance(component, WithColumnsComponent):
                self._transform_queue.append(component)
                df_lazy = df_lazy.with_columns(component.exprs)
            elif isinstance(component, SelectComponent):
                self._transform_queue.append(component)
                df_lazy = df_lazy.select(component.exprs)
            else:
                raise ValueError("Not a valid PipeComponent.")
        return self

    def fit(self) -> Self:
        """
        Alias for self.finish()
        """
        return self.finish()

    def _generate_lazy_plan(self, df: Optional[PolarsFrame] = None) -> pl.LazyFrame:
        """
        Generates a lazy plan for the incoming df

        Paramters
        ---------
        df
            If none, create the plan for the df that the pipe is initialized with. Otherwise,
            create the plan for the incoming df.
        """
        plan = self._df if df is None else df.lazy()
        for comp in self._transform_queue:
            if isinstance(comp, WithColumnsComponent):
                plan = plan.with_columns(comp.exprs)
            elif isinstance(comp, SelectComponent):
                plan = plan.select(comp.exprs)
            else:
                raise ValueError("Not a valid FittedPipeComponent.")

        return plan

    def transform(self, df: Optional[PolarsFrame] = None, return_lazy: bool = False) -> PolarsFrame:
        """
        Transforms the df using the learned expressions.

        Paramters
        ---------
        df
            If none, transform the df that the pipe is initialized with. Otherwise, perform
            the learned transformations on the incoming df.
        return_lazy
            If true, return the lazy plan for the transformations
        """
        plan = self._generate_lazy_plan(df)
        return plan if return_lazy else plan.collect()  # Add config here if streaming is needed

    def show_graph(self, optimized: bool = True) -> Optional[str]:
        """
        Shows the execution graph.

        Parameters
        ----------
        optimized
            Whether this will show the optimized plan or not.
        """
        return self._generate_lazy_plan().show_graph(optimized=optimized)
