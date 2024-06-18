import polars as pl
import json
import sys
from . import transforms as t
from functools import partial
from dataclasses import dataclass
from polars.type_aliases import IntoExprColumn
from typing import List, Optional, Union, Dict, Any
from .type_alias import (
    TypeAlias,
    PolarsFrame,
    ExprTransform,
    FitTransformFunc,
    SimpleImputeMethod,
    SimpleScaleMethod,
    StrOrExpr,
    RollingInterpolationMethod,
)

from ._utils import _POLARS_LEGACY_SUPPORT

if sys.version_info >= (3, 11):
    from typing import Self
else:  # 3.10, 3.9, 3.8
    from typing_extensions import Self

__all__ = ["Pipeline", "Blueprint"]


@dataclass
class SelectStep:
    exprs: ExprTransform

    def __iter__(self):
        return [self.exprs].__iter__() if isinstance(self.exprs, pl.Expr) else self.exprs.__iter__()


@dataclass
class WithColumnsStep:
    exprs: ExprTransform

    def __iter__(self):
        return [self.exprs].__iter__() if isinstance(self.exprs, pl.Expr) else self.exprs.__iter__()


@dataclass
class FitStep:
    func: FitTransformFunc
    cols: IntoExprColumn
    exclude: List[str]

    # Here we allow IntoExprColumn as input so that users can use selectors, or other polars expressions
    # to specify input columns, which adds flexibility.
    # We still need real column names so that the functions in transforms.py will work.
    def fit(self, df: PolarsFrame) -> ExprTransform:
        real_cols: List[str] = [x for x in df.select(self.cols).columns if x not in self.exclude]
        return self.func(df, real_cols)


Step: TypeAlias = Union[FitStep, SelectStep, WithColumnsStep]
FittedStep: TypeAlias = Union[SelectStep, WithColumnsStep]


def _to_json_dict(step: FittedStep) -> Dict:
    """
    Turns a fitted step into a JSON dict.
    """
    exprs = []
    if isinstance(step.exprs, pl.Expr):
        exprs.append(step.exprs.meta.serialize())
    elif isinstance(step.exprs, list):
        for e in step.exprs:
            if isinstance(e, pl.Expr):
                exprs.append(e.meta.serialize())
            else:
                raise ValueError("Non-expression detected. This object is ill-defined.")

    if isinstance(step, SelectStep):
        return {"SelectStep": exprs}
    else:
        return {"WithColumnsStep": exprs}


@dataclass
class Pipeline:
    """
    A ML/data transform pipeline. Pipelines should always come from the materialize call from a
    blueprint. In other words, a pipeline is a fitted blueprint.
    """

    name: str
    target: Optional[str]
    feature_names_in_: List[str]
    feature_names_out_: List[str]
    transforms: List[FittedStep]
    ensure_features_in: bool = False
    ensure_features_out: bool = True

    def __str__(self) -> str:
        return self.transforms.__str__()

    def __repr__(self) -> str:
        text: str = "Naive Query Steps: \n\n"
        for i, step in enumerate(self.transforms):
            text += f"Step {i+1}:\n"
            text += ",\n".join(str(e) for e in step)  # SelectStep, WithColumnsStep are iterable
            text += "\n\n"

        return text

    def _generate_lazy_plan(self, df: PolarsFrame) -> pl.LazyFrame:
        """
        Generates a lazy plan for the incoming df

        Paramters
        ---------
        df
            If none, create the plan for the df that the pipe is initialized with. Otherwise,
            create the plan for the incoming df.
        """
        plan = df.lazy()
        for step in self.transforms:
            if isinstance(step, WithColumnsStep):
                plan = plan.with_columns(step.exprs)
            elif isinstance(step, SelectStep):
                plan = plan.select(step.exprs)
            else:
                raise ValueError(f"Transform is not a valid FittedStep: {str(step)}")

        return plan

    def to_dict(self) -> Dict:
        """
        Converts self to a dict, with all expressions turned into JSON strings.
        """
        if _POLARS_LEGACY_SUPPORT:
            raise ValueError("This functionality doesn't support polars < 0.20.16.")

        return {
            "name": str(self.name),
            "target": self.target,
            "feature_names_in_": list(self.feature_names_in_),
            "feature_names_out_": list(self.feature_names_out_),
            "transforms": [_to_json_dict(step) for step in self.transforms],
            "ensure_features_in": self.ensure_features_in,
            "ensure_features_out": self.ensure_features_out,
        }

    def to_json(self, path: Optional[str] = None, **kwargs) -> Optional[str]:
        """
        Turns self into a JSON string.

        Parameters
        ----------
        path
            If none, will return a json string. If given, this will be used as the path
            to save the pipeline and None will be returned.
        kwargs
            Keyword arguments to Python's default json
        """
        if _POLARS_LEGACY_SUPPORT:
            raise ValueError("This functionality doesn't support polars < 0.20.16.")

        # Maybe support other json package?
        if path is None:
            return json.dumps(self.to_dict(), **kwargs)
        else:
            with open(path, "w") as f:
                json.dump(self.to_dict(), f)

            return None

    def from_dict(pipeline_dict: Dict[str, Any]) -> Self:
        """
        Recreates a pipeline from a dictionary created by the `to_dict` call.
        """
        if _POLARS_LEGACY_SUPPORT:
            raise ValueError("This functionality doesn't support polars < 0.20.16.")

        from io import StringIO

        transforms = pipeline_dict["transforms"]
        transform_steps = []
        step: Dict[str, List[str]]
        for (
            step
        ) in transforms:  # each step is a dict like {'SelectStep': [jsonified str expressions..]}
            if "SelectStep" in step:
                json_exprs = step.pop("SelectStep")
                actual_exprs = [pl.Expr.deserialize(StringIO(e)) for e in json_exprs]
                transform_steps.append(SelectStep(actual_exprs))
            elif "WithColumnsStep" in step:
                json_exprs = step.pop("WithColumnsStep")
                actual_exprs = [pl.Expr.deserialize(StringIO(e)) for e in json_exprs]
                transform_steps.append(WithColumnsStep(actual_exprs))
            else:
                raise ValueError(f"Invalid step {step}")

        name = pipeline_dict["name"]
        target = pipeline_dict["target"]
        feature_names_in_ = pipeline_dict["feature_names_in_"]
        feature_names_out_ = pipeline_dict["feature_names_out_"]
        ensure_features_in = pipeline_dict["ensure_features_in"]
        ensure_features_out = pipeline_dict["ensure_features_out"]

        return Pipeline(
            name=name,
            target=target,
            feature_names_in_=feature_names_in_,
            feature_names_out_=feature_names_out_,
            transforms=transform_steps,
            ensure_features_in=ensure_features_in,
            ensure_features_out=ensure_features_out,
        )

    def from_json_str(json_str: str) -> Self:
        """
        Creates the Pipeline from the JSON string.
        """
        if _POLARS_LEGACY_SUPPORT:
            raise ValueError("This functionality doesn't support polars < 0.20.16.")
        return Pipeline.from_dict(json.loads(json_str))

    def from_json(path: str) -> Self:
        """
        Creates the Pipeline by loading a local JSON file at path
        """
        if _POLARS_LEGACY_SUPPORT:
            raise ValueError("This functionality doesn't support polars < 0.20.16.")
        with open(path, "r") as f:
            pipe_dict = json.load(f)
        return Pipeline.from_dict(pipe_dict)

    def ensure_features_io(self, ensure_in: bool = True, ensure_out: bool = True) -> Self:
        """
        Whether or not this pipeline should check the features coming in and out during transform.

        Parameters
        ----------
        ensure_in
            If true, the input df (during transform) must have the exact same features
            as when this pipeline was fitted (blueprint.materialize()). Setting this false means the input may
            have additional columns, and so this adds flexibility in the pipeline.
        ensure_out
            If true, only the output features during blueprint.materialize() will be kept at the end of the
            pipeline.
        """
        self.ensure_features_in = ensure_in
        self.ensure_features_out = ensure_out
        return self

    def transform(self, df: PolarsFrame, return_lazy: bool = False) -> PolarsFrame:
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
        if self.ensure_features_in:
            extras = [c for c in df.columns if c not in self.feature_names_in_]
            missing = [c for c in self.feature_names_in_ if c not in df.columns]
            if len(extras) > 0 or len(missing):
                raise ValueError(
                    f"Input df doesn't have the features expected. Extra columns: {extras}. Missing columns: {missing}"
                )

        plan = self._generate_lazy_plan(df)
        if self.ensure_features_out:
            plan = plan.select(self.feature_names_out_)
        return plan if return_lazy else plan.collect()  # Add config here if streaming is needed


class Blueprint:
    """
    Blueprints for a ML/data transformation pipeline. In other words, this is a description of
    what a pipeline will be. No learning/fitting is done until self.materialize() is called.

    If the input df is lazy, the pipeline will collect at the time of fit.
    """

    def __init__(
        self,
        df: PolarsFrame,
        name: str = "test",
        target: Optional[str] = None,
        exclude: Optional[List[str]] = None,
    ):
        """
        Creates a blueprint object.

        Parameters
        ----------
        df
            Either a lazy or an eager Polars dataframe
        name
            Name of the blueprint.
        target
            Optionally indicate the target column in the ML pipeline. This will automatically prevent any transformation
            from changing the target column. (To be implemented: this should also automatically fill any transformation
            that requires a target name)
        exclude
            Any other column to exclude from global transformation. Note: this is only needed if you are not specifiying
            the exact columns to transform. E.g. when you are using a selector like cs.numeric() for all numeric columns.
            If this is the case and target is not set nor excluded, then the transformation may be applied to the target
            as well, which is not desired in most cases. Therefore, it is highly recommended you initialize with target name.
        """

        self._df: pl.LazyFrame = df.lazy()
        self.name: str = str(name)
        self.target = target
        self.feature_names_in_: list[str] = list(df.columns)
        self._steps: List[Step] = []
        self.exclude: List[str] = [] if target is None else [target]
        if exclude is not None:  # dedup in case user accidentally puts the same column name twice
            self.exclude = list(set(self.exclude + exclude))

    def __str__(self) -> str:
        out: str = ""
        out += f"Blueprint name: {self.name}\n"
        out += f"Blueprint current steps: {len(self._steps)}\n"
        out += f"Features Expected: {self.feature_names_in_}\n"
        return out

    def _get_target(self, target: Optional[StrOrExpr] = None) -> StrOrExpr:
        if target is None:
            if self.target is None:
                raise ValueError(
                    "Target is not given and blueprint is not initialized with a target."
                )
            return self.target
        else:
            return target

    def reset_df(self, df: PolarsFrame) -> Self:
        """
        Resets the underlying dataset to learn from. This will keep all the existing
        steps in the blueprint.

        Parameters
        ----------
        df
            The new dataframe to use when materializing.
        """
        from copy import deepcopy

        self._df = df.lazy()
        self.name = str(self.name)
        self.feature_names_in_ = list(self._df.columns)
        self._steps = [deepcopy(s) for s in self._steps]
        return self

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
        self._steps.append(FitStep(partial(t.impute, method=method), cols, self.exclude))
        return self

    def linear_impute(
        self, features: IntoExprColumn, target: Optional[StrOrExpr] = None, add_bias: bool = False
    ) -> Self:
        """ """
        self._steps.append(
            FitStep(
                partial(t.linear_impute, target=self._get_target(target), add_bias=add_bias),
                features,
                self.exclude,
            )
        )
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
        self._steps.append(FitStep(partial(t.scale, method=method), cols, self.exclude))
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
        self._steps.append(FitStep(partial(t.robust_scale, q1=q1, q2=q2), cols, self.exclude))
        return self

    def center(self, cols: IntoExprColumn) -> Self:
        """
        Centers the columns by subtracting each with its mean.

        Paramters
        ---------
        cols
            Any Polars expression that can be understood as columns.
        """
        self._steps.append(FitStep(partial(t.center), cols, self.exclude))
        return self

    def select(self, cols: IntoExprColumn) -> Self:
        """
        Selects the columns from the dataset.

        Paramters
        ---------
        cols
            Any Polars expression that can be understood as columns.
        """
        self._steps.append(SelectStep(cols))
        return self

    def winsorize(
        self,
        cols: IntoExprColumn,
        lower: float = 0.05,
        upper: float = 0.95,
        method: RollingInterpolationMethod = "nearest",
    ) -> Self:
        """
        Learns the lower and upper percentile from the columns, then clip each end at those values.
        If you wish to clip by constant values, you may append expression like pl.col(c).clip(c1, c2),
        where c1 and c2 are constants decided by the user.

        Parameters
        ----------
        cols
            Any Polars expression that can be understood as columns. Columns must be numerical.
        lower
            The lower quantile value
        upper
            The higher quantile value
        method
            Method to compute quantile. One of `nearest`, `higher`, `lower`, `midpoint`, `linear`.
        """
        self._steps.append(
            FitStep(
                partial(t.winsorize, lower=lower, upper=upper, method=method),
                cols,
                self.exclude,
            )
        )
        return self

    def drop(self, cols: IntoExprColumn) -> Self:
        """
        Drops the columns from the dataset.

        Paramters
        ---------
        cols
            Any Polars expression that can be understood as columns.
        """
        self._steps.append(SelectStep(pl.all().exclude(cols)))
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
        self._steps.append(WithColumnsStep([pl.col(k).alias(v) for k, v in rename_dict.items()]))
        return self.drop(old)

    def lowercase(self) -> Self:
        """
        Lowercases all column names.
        """
        self._steps.append(SelectStep([pl.col(c).alias(c.lower()) for c in self._df.columns]))
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
            FitStep(
                partial(t.one_hot_encode, separator=separator, drop_first=drop_first),
                cols,
                self.exclude,
            )
        )
        return self.drop(cols)

    def target_encode(
        self,
        cols: IntoExprColumn,
        /,
        target: Optional[StrOrExpr] = None,
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
            FitStep(
                partial(
                    t.target_encode,
                    target=self._get_target(target),
                    min_samples_leaf=min_samples_leaf,
                    smoothing=smoothing,
                    default=default,
                ),
                cols,
                self.exclude,
            )
        )
        return self

    def woe_encode(
        self,
        cols: IntoExprColumn,
        /,
        target: Optional[StrOrExpr] = None,
        default: Optional[float] = None,
    ) -> Self:
        """
        Use Weight of Evidence to encode a discrete variable x with respect to target. This assumes x
        is discrete and castable to String. A value of 1 is added to all events/non-events
        (goods/bads) to smooth the computation. This is -1 * output of the package category_encoder's WOEEncoder.

        Note: nulls will be encoded as well.

        Parameters
        ----------
        cols
            Any Polars expression that can be understood as columns. Columns of type != string/categorical
            will not produce any expression.
        target
            The target column
        default
            If new value is encountered during transform, it will be mapped to default

        Reference
        ---------
        https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html
        """
        self._steps.append(
            FitStep(
                partial(
                    t.woe_encode,
                    target=self._get_target(target),
                    default=default,
                ),
                cols,
                self.exclude,
            )
        )
        return self

    def iv_encode(
        self,
        cols: IntoExprColumn,
        /,
        target: Optional[StrOrExpr] = None,
        default: Optional[float] = None,
    ) -> Self:
        """
        Use Information Value to encode a discrete variable x with respect to target. This assumes x
        is discrete and castable to String. A value of 1 is added to all events/non-events
        (goods/bads) to smooth the computation.

        Note: nulls will be encoded as well.

        Parameters
        ----------
        cols
            Any Polars expression that can be understood as columns. Columns of type != string/categorical
            will not produce any expression.
        target
            The target column
        default
            If new value is encountered during transform, it will be mapped to default

        Reference
        ---------
        https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html
        """
        self._steps.append(
            FitStep(
                partial(
                    t.iv_encode,
                    target=self._get_target(target),
                    default=default,
                ),
                cols,
                self.exclude,
            )
        )
        return self

    def append_expr(self, *exprs: ExprTransform, is_select: bool = False) -> Self:
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
            self._steps.append(SelectStep(list(exprs)))
        else:
            self._steps.append(WithColumnsStep(list(exprs)))
        return self

    # How to type this?
    def append_fit_func(self, func, cols: IntoExprColumn, **kwargs) -> Self:
        """
        Adds a custom transform that requires a fit step in the blueprint.

        Any custom function must satistfy the following function signature:
        my_func(df:Union[pl.DataFrame, pl.LazyFrame], cols: List[str], ...) -> List[pl.Expr]
        where ... means kwargs. The fit step "learns" the parameters needed to translate
        the transform into concrete expressions.

        Parameters
        ----------
        func
            A callable with signature (pl.DataFrame, pl.LazyFrame], cols: List[str], ...) -> ExprTransform,
        cols
            The columns to be fed into the func. Note that in func's signature, a list of strings
            should be expected. But here, cols can be any polars selector expression. The reason is that
            during "fit", cols is turned into concrete column names.
        **kwargs
            Any other arguments to func must be passed as kwargs
        """
        import inspect

        keywords = kwargs.copy()
        if "target" in inspect.signature(func).parameters:  # func has "target" as input
            if "target" not in kwargs:  # if target is not explicitly given
                keywords["target"] = self._get_target()

        self._steps.append(
            FitStep(
                partial(func, **keywords),
                cols,
                self.exclude,
            )
        )
        return self

    def materialize(self) -> Pipeline:
        """
        Materialize the blueprint, which means that it will fit and learn all the paramters needed.
        """
        transforms: List[FittedStep] = []
        df: pl.DataFrame = self._df.collect()  # Add config here if streaming is needed
        # Let this lazy plan go through the fit process. The frame will be collected temporarily but
        # the collect should be and optimized.
        df_lazy: pl.LazyFrame = df.lazy()
        for step in self._steps:
            if isinstance(step, FitStep):
                exprs = step.fit(df_lazy)
                transforms.append(WithColumnsStep(exprs))
                df_lazy = df_lazy.with_columns(exprs)
            elif isinstance(step, WithColumnsStep):
                transforms.append(step)
                df_lazy = df_lazy.with_columns(step.exprs)
            elif isinstance(step, SelectStep):
                transforms.append(step)
                df_lazy = df_lazy.select(step.exprs)
            else:
                raise ValueError("Not a valid step.")

        return Pipeline(
            name=self.name,
            target=self.target,
            feature_names_in_=list(self.feature_names_in_),
            feature_names_out_=list(df_lazy.columns),
            transforms=transforms,
        )

    def fit(self, X=None, y=None) -> Pipeline:
        """
        Alias for self.materialize()
        """
        return self.materialize()

    def transform(self, df: PolarsFrame) -> pl.DataFrame:
        return self.materialize().transform(df)

    def finish(self) -> Pipeline:
        """
        Alias for self.materialize()
        """
        return self.materialize()
