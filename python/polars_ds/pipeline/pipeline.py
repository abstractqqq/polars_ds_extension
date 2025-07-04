"""Machine Learning / Time series Pipelines with native Polars support."""

from __future__ import annotations

import polars as pl
import json
import sys
import polars.selectors as cs
from polars._typing import IntoExprColumn
from functools import partial
from typing import List, Dict, Any, Literal
from dataclasses import dataclass
from polars_ds._utils import to_expr

if sys.version_info >= (3, 11):
    from typing import Self
else:  # 3.10, 3.9, 3.8
    from typing_extensions import Self

# Internal Depenedncies
from . import transforms as t
from ._step import (
    PLContext,
    ExprStep,
    SortStep,
    SQLStep,
    GroupByAggStep,
    GroupByDynAggStep,
    FitStep,
    PipelineStep,
    MakeStep,
    _SERIALIZABLE_STEPS,
)
from polars_ds.typing import (
    PolarsFrame,
    SimpleImputeMethod,
    SimpleScaleMethod,
    QuantileMethod,
    EncoderDefaultStrategy,
)

__all__ = ["Pipeline", "Blueprint", "PLContext", "ExprStep"]


@dataclass
class StepRepr:
    """
    A representation of a step
    """

    name: str
    args: List[Any]
    kwargs: Dict[str, Any]

    @staticmethod
    def from_dict(dictionary: Dict[str, Any]) -> "StepRepr":
        try:
            name: str = dictionary["name"]
            args: List[Any] = dictionary.get("args", [])
            kwargs: Dict[str, Any] = dictionary.get("kwargs", {})
            if not isinstance(name, str):
                raise ValueError("Value of `name` must be a string.")
            if not isinstance(args, list):
                raise ValueError("Value of `args` must be a list.")
            if not isinstance(kwargs, dict):
                raise ValueError("Value of `kwargs` must be a dict.")
            if not all(isinstance(s, str) for s in kwargs.keys()):
                raise ValueError("All keys in `kwargs` must be strings.")
            return StepRepr(name=name, args=args, kwargs=kwargs)
        except Exception as e:
            raise ValueError(f"Keys missing or data type is not expected. Original error: \n{e}")


@dataclass
class Pipeline:
    """
    A ML/data transform pipeline. Pipelines should always come from the materialize call from a
    blueprint. In other words, a pipeline is a fitted blueprint.
    """

    name: str
    feature_names_in_: List[str]
    feature_names_out_: List[str]
    transforms: List[PipelineStep]
    ensure_features_in: bool = False
    ensure_features_out: bool = True
    lowercase: bool = False
    uppercase: bool = False

    def __str__(self) -> str:
        return self.transforms.__str__()

    def _get_init_plan(self, df: PolarsFrame) -> pl.LazyFrame:
        """
        Get an initial plan without any pipeline transforms.
        """
        if self.lowercase:
            plan = df.lazy().select(pl.all().name.to_lowercase())
        else:
            if self.uppercase:
                plan = df.lazy().select(pl.all().name.to_uppercase())
            else:
                plan = df.lazy()

        return plan

    def _generate_lazy_plan(self, df: PolarsFrame) -> pl.LazyFrame:
        """
        Generates a lazy plan for the incoming df

        Paramters
        ---------
        df
            If none, create the plan for the df that the pipe is initialized with. Otherwise,
            create the plan for the incoming df.
        """
        plan = self._get_init_plan(df)
        for step in self.transforms:
            plan = step.apply_df(plan)
        return plan

    def with_features_out(self, features: List[str], ensure_features_out: bool = True) -> Self:
        self.feature_names_out_ = list(features)
        self.ensure_features_out = ensure_features_out

    def to_json(self, path: str | None = None, **kwargs) -> str | None:
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
        # Maybe support other json package?
        d = {
            "name": str(self.name),
            "feature_names_in_": list(self.feature_names_in_),
            "feature_names_out_": list(self.feature_names_out_),
            "transforms": [step.to_json() for step in self.transforms],
            "ensure_features_in": self.ensure_features_in,
            "ensure_features_out": self.ensure_features_out,
            "lowercase": self.lowercase,
            "uppercase": self.uppercase,
        }
        if path is None:
            return json.dumps(d, **kwargs)
        else:
            with open(path, "w") as f:
                json.dump(d, f)

            return None

    @staticmethod
    def from_json(json_str: str | bytes) -> "Pipeline":
        """
        Recreates a pipeline from a dictionary created by the `to_json` call.
        """
        pipeline_dict: Dict = json.loads(json_str)
        try:
            name: str = pipeline_dict["name"]
            transforms: List[str] = pipeline_dict["transforms"]
            feature_names_in_: List[str] = pipeline_dict["feature_names_in_"]
            feature_names_out_: List[str] = pipeline_dict["feature_names_out_"]
            ensure_features_in: bool = pipeline_dict["ensure_features_in"]
            ensure_features_out: bool = pipeline_dict["ensure_features_out"]
            lowercase: bool = pipeline_dict.get("lowercase", False)
            uppercase: bool = pipeline_dict.get("uppercase", False)
        except Exception as e:
            raise ValueError(f"Input dictionary is missing keywords. Original error: \n{e}")

        return Pipeline(
            name=name,
            feature_names_in_=feature_names_in_,
            feature_names_out_=feature_names_out_,
            transforms=[MakeStep.make(json.loads(step_str)) for step_str in transforms],
            ensure_features_in=ensure_features_in,
            ensure_features_out=ensure_features_out,
            lowercase=lowercase,
            uppercase=uppercase,
        )

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

    def transform(
        self, df: PolarsFrame, return_lazy: bool = False, set_features_out: bool = False
    ) -> PolarsFrame:
        """
        Transforms the df using the learned expressions.

        Paramters
        ---------
        df
            If none, transform the df that the pipe is initialized with. Otherwise, perform
            the learned transformations on the incoming df.
        return_lazy
            If true, return the lazy plan for the transformations
        set_features_out
            If true, set `self.feature_names_out_` to the output features from this transform run.
        """
        if self.ensure_features_in:
            if self.lowercase:
                columns = [c.lower() for c in df.lazy().collect_schema().names()]
            elif self.uppercase:
                columns = [c.upper() for c in df.lazy().collect_schema().names()]
            else:
                columns = df.lazy().collect_schema().names()
            extras = [c for c in columns if c not in self.feature_names_in_]
            missing = [c for c in self.feature_names_in_ if c not in columns]
            if len(extras) > 0 or len(missing) > 0:
                raise ValueError(
                    f"Input df doesn't have the features expected. Extra columns: {extras}. Missing columns: {missing}"
                )

        plan = self._generate_lazy_plan(df)
        if self.ensure_features_out:
            plan = plan.select(self.feature_names_out_)

        if set_features_out:
            self.feature_names_out_ = plan.collect_schema().names()

        # Add config here if streaming is needed
        return plan if return_lazy else plan.collect()


class Blueprint:
    """
    Blueprints for a ML/data transformation pipeline. In other words, this is a description of
    what a pipeline will be. No learning/fitting is done until self.materialize() is called.

    If the input df is lazy, the pipeline will collect at the time of fit.

    Note: although polars selectors work for most transformations and in most cases, it is still
    recommended that the user should use explicit expressions instead of selectors for most transformations.
    """

    def __init__(
        self,
        df: PolarsFrame,
        name: str = "test",
        target: str | None = None,
        exclude: List[str] | None = None,
        lowercase: bool = False,
        uppercase: bool = False,
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
        lowercase
            Whether to insert a lowercase column name step before all other transformations.
            This takes precedence over uppercase.
        uppercase
            Whether to insert a uppercase column name step before all other transformations.
            This only happens if lowercase is False
        """

        self._df: pl.LazyFrame = df.lazy()
        if lowercase:
            self._df = self._df.select(pl.all().name.to_lowercase())
        else:
            if uppercase:
                self._df = self._df.select(pl.all().name.to_uppercase())

        self.name: str = str(name)
        self.target = target
        self.feature_names_in_: list[str] = self._df.collect_schema().names()

        self._steps: List[ExprStep | FitStep] = []
        self.exclude: List[str] = [] if target is None else [target]
        if exclude is not None:  # dedup in case user accidentally puts the same column name twice
            self.exclude = list(set(self.exclude + exclude))

        self.lowercase = lowercase
        self.uppercase = uppercase

    def __str__(self) -> str:
        out: str = ""
        out += f"Blueprint name: {self.name}\n"
        if self.lowercase:
            out += "Column names: Lowercase all incoming columns.\n"
        elif self.uppercase:
            out += "Column names: Uppercase all incoming columns.\n"

        out += f"Blueprint current steps: {len(self._steps)}\n"
        out += f"Features Expected: {self.feature_names_in_}\n"
        return out

    def _get_target(self, target: str | pl.Expr | None = None) -> str | pl.Expr:
        if target is None:
            if self.target is None:
                raise ValueError(
                    "Target is not given and blueprint is not initialized with a target."
                )
            return self.target
        else:
            return target

    def filter(self, by: str | pl.Expr) -> Self:
        """
        Filters on the dataframe using native polars expressions or SQL boolean expressions.

        Parameters
        ----------
        by
            Native polars boolean expression or SQL strings
        """
        self._steps.append(
            ExprStep(by if isinstance(by, pl.Expr) else pl.sql_expr(by), PLContext.FILTER)
        )
        return self

    def sql_transform(self, sql: str) -> Self:
        """
        Runs the SQL on the dataframe when it reaches this step. The user must ensure that
        the SQL is valid Polars SQL and all columns referred in the SQL exist at this point.
        The name "df" should be used to refer to the current state of the dataframe in the SQL.
        E.g. select * from df where A is True.

        Parameters
        ----------
        sql
            The SQL to run on the dataframe. Note: this step doesn't immedinately check the validity of
            the SQL statement.
        """
        self._steps.append(SQLStep(sql_str=sql))
        return self

    def cast_bools(self, dtype: pl.DataType = pl.UInt8) -> Self:
        """
        Cast all boolean columns in the dataframe to the given type.
        """
        self._steps.append(ExprStep(cs.boolean().cast(dtype), PLContext.WITH_COLUMNS))
        return self

    def impute(self, cols: IntoExprColumn, method: SimpleImputeMethod = "mean") -> Self:
        """
        Imputes null values in the given columns. Note: this doesn't fill NaN. If filling for NaN is needed,
        please manually do that.

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

    def conditional_impute(
        self, rules_dict: Dict[str, str | pl.Expr], method: SimpleImputeMethod = "mean"
    ) -> Self:
        """
        Conditionally imputes values in the given columns. This transform will collect if input is lazy.

        Parameters
        ----------
        rules_dict
            Dictionary where keys are column names (must be string), and values are SQL/Polars Conditions
            that when true, those values in the column will be imputed,
            and the value to impute will be learned on the data where the condition is false.
        method
            One of `mean`, `median`, `mode`. If `mode`, a random value will be chosen if there is
            a tie.
        """
        self._steps.append(
            FitStep(
                partial(t.conditional_impute, rules_dict=rules_dict, method=method),
                None,
                self.exclude,
            )
        )
        return self

    def nan_to_null(self) -> Self:
        """
        Maps NaN values in all columns to null.
        """
        self._steps.append(ExprStep(cs.float().fill_nan(None), PLContext.WITH_COLUMNS))
        return self

    def int_to_float(self, f32: bool = True) -> Self:
        """
        Maps all integer columns to float.

        Parameters
        ----------
        f32
            If true, map all integer columns to f32 columns. Otherwise they will be
            casted to f64 columns.
        """
        if f32:
            self._steps.append(ExprStep(cs.integer().cast(pl.Float32), PLContext.WITH_COLUMNS))
        else:
            self._steps.append(ExprStep(cs.integer().cast(pl.Float64), PLContext.WITH_COLUMNS))
        return self

    def linear_impute(
        self, features: IntoExprColumn, target: str | pl.Expr | None = None, add_bias: bool = False
    ) -> Self:
        """
        Imputes the target column by training a simple linear regression using the other features. This will
        cast the target column to f64.

        Note: The linear regression will skip nulls whenever there is a null in the features or in the target.
        Additionally, if NaN or Inf exists in data, the linear regression result may be invalid or an error
        will be thrown. It is recommended to use this only after imputing and dealing with NaN and Infs for
        all feature columns first.

        Parameters
        ----------
        features
            Any Polars expression that can be understood as numerical columns which will be used as features
        target
            The target column
        add_bias
            Whether to add a bias term to the linear regression
        """
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

    def robust_scale(self, cols: IntoExprColumn, q_low: float, q_high: float) -> Self:
        """
        Performs robust scaling on the given columns

        Paramters
        ---------
        cols
            Any Polars expression that can be understood as columns.
        q_low
            The lower quantile value
        q_high
            The higher quantile value
        """
        self._steps.append(
            FitStep(partial(t.robust_scale, q_low=q_low, q_high=q_high), cols, self.exclude)
        )
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

    def select(self, *cols: pl.Expr) -> Self:
        """
        Selects the columns from the dataset.

        Paramters
        ---------
        cols
            Any Polars expression that can be understood as columns.
        """
        self._steps.append(ExprStep(list(cols), PLContext.SELECT))
        return self

    def polynomial_features(
        self, cols: List[str], degree: int, interaction_only: bool = True
    ) -> Self:
        """
        Generates polynomial combinations out of the features given, at the given degree.

        Parameters
        ----------
        cols
            A list of strings representing column names. Input to this function cannot be Polars expressions.
        degree
            The degree of the polynomial combination
        interaction_only
            It true, only combinations that involve 2 or more variables will be used.
        """
        if not all(isinstance(s, str) for s in cols):
            raise ValueError(
                "Input columns to `polynomial_features` must all be strings represeting column names."
            )

        self._steps.append(
            ExprStep(
                t.polynomial_features(cols, degree=degree, interaction_only=interaction_only),
                PLContext.WITH_COLUMNS,
            )
        )
        return self

    def winsorize(
        self,
        cols: IntoExprColumn,
        q_low: float = 0.05,
        q_high: float = 0.95,
        method: QuantileMethod = "nearest",
    ) -> Self:
        """
        Learns the lower and upper percentile from the columns, then clip each end at those values.
        If you wish to clip by constant values, you may append expression like pl.col(c).clip(c1, c2),
        where c1 and c2 are constants decided by the user.

        Parameters
        ----------
        cols
            Any Polars expression that can be understood as columns. Columns must be numerical.
        q_low
            The lower quantile value
        q_high
            The higher quantile value
        method
            Method to compute quantile. One of `nearest`, `higher`, `lower`, `midpoint`, `linear`.
        """
        self._steps.append(
            FitStep(
                partial(t.winsorize, q_low=q_low, q_high=q_high, method=method),
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
        self._steps.append(ExprStep(pl.exclude(cols), PLContext.SELECT))
        return self

    def rename(self, rename_dict: Dict[str, str]) -> Self:
        """
        Renames the columns by the mapping.

        Paramters
        ---------
        rename_dict
            The name mapping
        """
        old = list(rename_dict.keys())
        self._steps.append(
            ExprStep([pl.col(k).alias(v) for k, v in rename_dict.items()], PLContext.WITH_COLUMNS)
        )

        return self.drop([o for o in old if o not in set(rename_dict.values())])

    def one_hot_encode(
        self,
        cols: IntoExprColumn | None = None,
        separator: str = "_",
        drop_first: bool = False,
        drop_cols: bool = True,
    ) -> Self:
        """
        Find the unique values in the string/categorical columns and one-hot encode them. This will NOT
        consider nulls as one of the unique values. Append pl.col(c).is_null().cast(pl.UInt8)
        expression to the pipeline if you want null indicators.

        Parameters
        ----------
        cols
            Any Polars expression that can be understood as columns. If None, all string/categorical columns will be encoded.
        separator
            E.g. if column name is `col` and `a` is an elemenet in it, then the one-hot encoded column will be called
            `col_a` where the separator `_` is used.
        drop_first
            Whether to drop the first distinct value (in terms of str/categorical order). This helps with reducing
            dimension and prevents some issues from linear dependency.
        drop_cols
            Whether to drop the original columns after the transform
        """
        # First append new columns
        self._steps.append(
            FitStep(
                partial(t.one_hot_encode, separator=separator, drop_first=drop_first),
                cols if cols is not None else cs.string() | cs.categorical(),
                self.exclude,
            )
        )
        # Whether to drop the og str columns
        if drop_cols:
            if cols is None:
                return self.drop([pl.String, pl.Categorical])
            return self.drop(cols)
        return self

    def rank_hot_encode(
        self, col: str | pl.Expr, ranking: List[str], drop_cols: bool = True
    ) -> Self:
        """
        Given a ranking, e.g. ["bad", "neutral", "good"], where "bad", "neutral" and "good" are values coming
        from the column `col`, this will create 2 additional columns, where a row of [0, 0] will represent
        "bad", and a row of [1, 0] will represent "neutral", and a row of [1,1] will represent "good". The meaning
        of each rows is that the value is at least this rank. This currently only works on string columns.

        Values not in the provided ranking will have -1 in all the new columns.

        Parameters
        ----------
        col
            The name of a single column
        ranking
            A list of string representing the ranking of the values
        drop_cols
            Whether to drop the original column after the transform
        """
        self._steps.append(
            ExprStep(t.rank_hot_encode(col=col, ranking=ranking), PLContext.WITH_COLUMNS)
        )
        if drop_cols:
            return self.drop(cols=[col])
        return self

    def target_encode(
        self,
        cols: IntoExprColumn | None = None,
        target: str | pl.Expr | None = None,
        min_samples_leaf: int = 20,
        smoothing: float = 10.0,
        default: EncoderDefaultStrategy | float | None = None,
    ) -> Self:
        """
        Target encode the given variables.

        Note: nulls will be encoded as well.

        Parameters
        ----------
        cols
            Any Polars expression that can be understood as columns. Columns of type != string/categorical
            will not produce any expression. If None, all string/categorical columns will be used.
        target
            The target column
        min_samples_leaf
            A regularization factor
        smoothing
            Smoothing effect to balance categorical average vs prior
        default
            If a new value is encountered during transform (unseen in training dataset), it will be mapped to default.
            If this is a string, it can be `null`, `zero`, or `mean`, where `mean` means map them to the mean of the target.

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
                cols if cols is not None else cs.string() | cs.categorical(),
                self.exclude,
            )
        )
        return self

    def woe_encode(
        self,
        cols: IntoExprColumn | None = None,
        target: str | pl.Expr | None = None,
        default: EncoderDefaultStrategy | float | None = None,
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
            will not produce any expression. If None, all string/categorical columns will be used.
        target
            The target column
        default
            If a new value is encountered during transform (unseen in training dataset), it will be mapped to default.
            If this is a string, it can be `null`, `zero`, or `mean`, where `mean` means map them to the mean of the target.

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
                cols if cols is not None else cs.string() | cs.categorical(),
                self.exclude,
            )
        )
        return self

    def iv_encode(
        self,
        cols: IntoExprColumn | None = None,
        target: str | pl.Expr | None = None,
        default: EncoderDefaultStrategy | float | None = None,
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
            will not produce any expression. If None, all string/categorical columns will be used.
        target
            The target column
        default
            If a new value is encountered during transform (unseen in training dataset), it will be mapped to default.
            If this is a string, it can be `null`, `zero`, or `mean`, where `mean` means map them to the mean of the target.

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
                cols if cols is not None else cs.string() | cs.categorical(),
                self.exclude,
            )
        )
        return self

    def with_columns(self, *exprs: pl.Expr) -> Self:
        """
        Run Polars with_columns for the expressions.
        """
        self._steps.append(ExprStep(list(exprs), PLContext.WITH_COLUMNS))
        return self

    def sort(self, by: IntoExprColumn, descending: bool | List[bool]) -> Self:
        """Sorts the dataframe by the columns.

        Parameters
        ----------
        by
            The columns to sort by
        descending
            Whether the sort should be descending for the corresponding sort column
        """
        self._steps.append(SortStep(by=by, descending=descending))
        return self

    def explode(self, columns: str | pl.Expr | List[str] | List[pl.Expr]) -> Self:
        """Transform that represents `df.explode(columns)`"""
        if isinstance(columns, (str, pl.Expr)):
            exprs = [to_expr(columns)]
        elif isinstance(columns, list):
            exprs = [to_expr(c) for c in columns]
        else:
            raise ValueError(
                "Input `columns` must be a string, or a pl.Expr or a list of str or pl.Expr."
            )

        self._steps.append(ExprStep(exprs, PLContext.EXPLODE))
        return self

    def group_by_agg(
        self, by: IntoExprColumn, agg: List[pl.Expr], maintain_order: bool = False
    ) -> Self:
        """
        Performs a group by and agg on the data.

        Parameters
        ----------
        by
            The columns to group by
        agg
            The aggregation functions to run
        maintain_order
            Whether to maintain the group by order
        """
        self._steps.append(
            GroupByAggStep(by=by, agg=[to_expr(a) for a in agg], maintain_order=maintain_order)
        )
        return self

    def group_by_dynamic_agg(
        self,
        index_column: str,
        agg: List[pl.Expr],
        every: str,
        period: str | None = None,
        offset: str | None = None,
        include_boundaries: bool = False,
        closed: Literal["left", "right", "both", "none"] = "left",
        label: Literal["left", "right", "datapoint"] = "left",
        group_by: IntoExprColumn | None = None,
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
    ) -> Self:
        """
        See polars group_by_dynamic documentation for an explanation on the input arguments.

        https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.group_by_dynamic.html#polars.DataFrame.group_by_dynamic
        """
        self._steps.append(
            GroupByDynAggStep(
                index_column=index_column,
                agg=[to_expr(a) for a in agg],
                every=every,
                group_by=group_by,
                period=period,
                offset=offset,
                include_boundaries=include_boundaries,
                closed=closed,
                label=label,
                start_by=start_by,
            )
        )
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
            A callable with signature (pl.DataFrame | pl.LazyFrame, cols: List[str], ...) -> ExprTransform,
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
                if keywords["target"] is None:
                    raise ValueError(
                        "Target is not explicitly given and is required by the custom function."
                    )

        self._steps.append(
            FitStep(
                partial(func, **keywords),
                cols,
                self.exclude,
            )
        )
        return self

    def append_step_from_dict(self, dictionary: Dict[str, Any]) -> Self:
        """
        Append a step to the blueprint by taking in a dictionary with keys `name`, `args`, and `kwargs`, where
        the value of args must be a List[Any] and the value of kwargs must be a dict[str, Any].
        """
        step_repr: StepRepr = StepRepr.from_dict(dictionary)
        func = getattr(self, step_repr.name, None)  # Default is None
        if func is None or step_repr.name.startswith("_"):
            raise ValueError("Unknown / invalid method name.")

        return func(*step_repr.args, **step_repr.kwargs)

    def materialize(self) -> Pipeline:
        """
        Materialize the blueprint, which means that it will fit and learn all the paramters needed.
        """
        transforms: List[PipelineStep] = []
        df: pl.DataFrame = self._df.collect()  # Add config here if streaming is needed
        # Let this lazy plan go through the fit process. The frame will be collected temporarily but
        # the collect should be and optimized.
        df_lazy: pl.LazyFrame = df.lazy()
        for step in self._steps:
            if isinstance(step, FitStep):  # Need fitting, which is done here
                df_temp = df_lazy.collect()
                exprs = step.fit(df_temp)
                transforms.append(ExprStep(exprs, PLContext.WITH_COLUMNS))
                df_lazy = df_temp.lazy().with_columns(exprs)
            elif isinstance(step, tuple(_SERIALIZABLE_STEPS)):
                transforms.append(step)
                df_lazy = step.apply_df(df_lazy)
            else:
                raise ValueError(f"Not a valid step: {step.__class__}")

        return Pipeline(
            name=self.name,
            feature_names_in_=list(self.feature_names_in_),
            feature_names_out_=df_lazy.collect_schema().names(),
            transforms=transforms,
            lowercase=self.lowercase,
            uppercase=self.uppercase,
        )

    def fit(self, X=None, y=None) -> Pipeline:
        """
        Alias for self.materialize()
        """
        return self.materialize()

    def transform(self, df: PolarsFrame) -> pl.DataFrame:
        """
        Fits the blueprint with the dataframe that it is initialized with, and
        transforms the input dataframe.

        Parameters
        ----------
        df
            Any Polars dataframe
        """
        return self.materialize().transform(df)
