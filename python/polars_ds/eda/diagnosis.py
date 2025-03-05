"""
Data Inspection Assistant and Visualizations for Polars Dataframe.

Currently, the plot backend is Altair but this is subject to change, and will be decided base on
which plotting backend supports Polars more natively.
"""

from __future__ import annotations

import altair as alt
# import plotly.express as px
# import plotly.graph_objs as go

import polars.selectors as cs
from polars._typing import IntoExpr
from typing import Tuple, List
import polars as pl
import graphviz
import warnings

from typing import Iterable, Dict
from functools import lru_cache
from itertools import combinations
from great_tables import GT

# Internal dependencies
from polars_ds.exprs.ts_features import query_cond_entropy
from polars_ds.exprs.stats import corr
from polars_ds.typing import CorrMethod, PolarsFrame

from .plots import plot_feature_distr

alt.data_transformers.enable("vegafusion")

__all__ = ["DIA"]


# DIA = Data Inspection Assistant / DIAgonsis
class DIA:
    """
    Data Inspection Assistant. Most plots are powered by Altair/great_tables. Altair may require
    additional package downloads.

    If you cannot import this module, please try: pip install "polars_ds[plot]"

    Note: most plots are sampled by default because (1) typically plots don't look good when there
    are too many points, and (2) because of interactivity, if we don't sample, the plots will be too
    large and won't get rendered in a reasonable amount of time. If speed of rendering is crucial and
    you don't need interactivity, use matplotlib.
    """

    # --- Static / Class Methods ---

    # --- Methods ---

    def __init__(self, df: PolarsFrame):
        self._frame: pl.LazyFrame = df.lazy()
        self.numerics: List[str] = df.select(cs.numeric()).collect_schema().names()
        self.ints: List[str] = df.select(cs.integer()).collect_schema().names()
        self.floats: List[str] = df.select(cs.float()).collect_schema().names()
        self.strs: List[str] = df.select(cs.string()).collect_schema().names()
        self.bools: List[str] = df.select(cs.boolean()).collect_schema().names()
        self.cats: List[str] = df.select(cs.categorical()).collect_schema().names()

        schema_dict = df.collect_schema()
        columns = schema_dict.names()

        self.list_floats: List[str] = [
            c
            for c, t in schema_dict.items()
            if (t.is_(pl.List(pl.Float32)) or (t.is_(pl.List(pl.Float64))))
        ]
        self.list_bool: List[str] = [
            c for c, t in schema_dict.items() if t.is_(pl.List(pl.Boolean))
        ]
        self.list_str: List[str] = [c for c, t in schema_dict.items() if t.is_(pl.List(pl.String))]
        self.list_ints: List[str] = [
            c
            for c, t in schema_dict.items()
            if t.is_(pl.List(pl.UInt8))
            or t.is_(pl.List(pl.UInt16))
            or t.is_(pl.List(pl.UInt32))
            or t.is_(pl.List(pl.UInt64))
            or t.is_(pl.List(pl.Int8))
            or t.is_(pl.List(pl.Int16))
            or t.is_(pl.List(pl.Int32))
            or t.is_(pl.List(pl.Int64))
        ]

        self.simple_types: List[str] = (
            self.numerics
            + self.strs
            + self.bools
            + self.cats
            + self.list_floats
            + self.list_ints
            + self.list_bool
            + self.list_str
        )
        self.other_types: List[str] = [c for c in columns if c not in self.simple_types]

    def special_values_report(self) -> pl.DataFrame:
        """
        Checks null, NaN, and non-finite values for float columns. Note that for integers, only null_count
        can possibly be non-zero.
        """
        to_check = self.numerics
        frames = [
            self._frame.select(
                pl.lit(c, dtype=pl.String).alias("column"),
                pl.col(c).null_count().alias("null_count"),
                (pl.col(c).null_count() / pl.len()).alias("null%"),
                pl.col(c).is_nan().sum().alias("NaN_count"),
                (pl.col(c).is_nan().sum() / pl.len()).alias("NaN%"),
                pl.col(c).is_infinite().sum().alias("inf_count"),
                (pl.col(c).is_infinite().sum() / pl.len()).alias("Inf%"),
            )
            for c in to_check
        ]
        return pl.concat(pl.collect_all(frames))

    def numeric_profile(
        self, n_bins: int = 20, iqr_multiplier: float = 1.5, histogram: bool = True, gt: bool = True
    ) -> GT | pl.DataFrame:
        """
        Creates a numerical profile with a histogram plot. Notice that the histograms may have
        completely different scales on the x-axis.

        Parameters
        ----------
        n_bins
            Bins in the histogram
        iqr_multiplier
            Inter Quartile Ranger multiplier. Inter quantile range is the range between
            Q1 and Q3, and this multiplier will enlarge the range by a certain amount and
            use this to count outliers.
        histogram
            Whether to show a histogram or not
        gt
            Whether to show the table as a formatted Great Table or not
        """
        to_check = self.numerics

        cuts = [i / n_bins for i in range(n_bins)]
        cuts[0] -= 1e-5
        cuts[-1] += 1e-5

        if histogram:
            columns_needed = [
                [
                    pl.lit(c, dtype=pl.String).alias("column"),
                    pl.col(c).count().alias("non_null_cnt"),
                    (pl.col(c).null_count() / pl.len()).alias("null%"),
                    pl.col(c).mean().cast(pl.Float64).alias("mean"),
                    pl.col(c).std().cast(pl.Float64).alias("std"),
                    pl.col(c).min().cast(pl.Float64).cast(pl.Float64).alias("min"),
                    pl.col(c).quantile(0.25).cast(pl.Float64).alias("q1"),
                    pl.col(c).median().cast(pl.Float64).round(2).alias("median"),
                    pl.col(c).quantile(0.75).cast(pl.Float64).alias("q3"),
                    pl.col(c).max().cast(pl.Float64).alias("max"),
                    (pl.col(c).quantile(0.75) - pl.col(c).quantile(0.25))
                    .cast(pl.Float64)
                    .alias("IQR"),
                    pl.any_horizontal(
                        pl.col(c)
                        < pl.col(c).quantile(0.25)
                        - iqr_multiplier * (pl.col(c).quantile(0.75) - pl.col(c).quantile(0.25)),
                        pl.col(c)
                        > pl.col(c).quantile(0.75)
                        + iqr_multiplier * (pl.col(c).quantile(0.75) - pl.col(c).quantile(0.25)),
                    )
                    .sum()
                    .alias("outlier_cnt"),
                    pl.struct(
                        ((pl.col(c) - pl.col(c).min()) / (pl.col(c).max() - pl.col(c).min()))
                        .filter(pl.col(c).is_finite())
                        .cut(breaks=cuts, left_closed=True, include_breaks=True)
                        .struct.rename_fields(["brk", "category"])
                        .struct.field("brk")
                        .value_counts()
                        .sort()
                        .struct.field("count")
                        .implode()
                    ).alias("histogram"),
                ]
                for c in to_check
            ]
        else:
            columns_needed = [
                [
                    pl.lit(c, dtype=pl.String).alias("column"),
                    pl.col(c).count().alias("non_null_cnt"),
                    (pl.col(c).null_count() / pl.len()).alias("null%"),
                    pl.col(c).mean().cast(pl.Float64).alias("mean"),
                    pl.col(c).std().cast(pl.Float64).alias("std"),
                    pl.col(c).min().cast(pl.Float64).cast(pl.Float64).alias("min"),
                    pl.col(c).quantile(0.25).cast(pl.Float64).alias("q1"),
                    pl.col(c).median().cast(pl.Float64).round(2).alias("median"),
                    pl.col(c).quantile(0.75).cast(pl.Float64).alias("q3"),
                    pl.col(c).max().cast(pl.Float64).alias("max"),
                    (pl.col(c).quantile(0.75) - pl.col(c).quantile(0.25))
                    .cast(pl.Float64)
                    .alias("IQR"),
                    pl.any_horizontal(
                        pl.col(c)
                        < pl.col(c).quantile(0.25)
                        - iqr_multiplier * (pl.col(c).quantile(0.75) - pl.col(c).quantile(0.25)),
                        pl.col(c)
                        > pl.col(c).quantile(0.75)
                        + iqr_multiplier * (pl.col(c).quantile(0.75) - pl.col(c).quantile(0.25)),
                    )
                    .sum()
                    .alias("outlier_cnt"),
                ]
                for c in to_check
            ]

        frames = [self._frame.select(*cols) for cols in columns_needed]
        df_final = pl.concat(pl.collect_all(frames))

        if gt:
            gt_out = (
                GT(df_final, rowname_col="column")
                .tab_stubhead("column")
                .fmt_percent(columns="null%")
                .fmt_number(
                    columns=["mean", "std", "min", "q1", "median", "q3", "max", "IQR"], decimals=3
                )
            )
            if histogram:
                return gt_out.fmt_nanoplot(columns="histogram", plot_type="bar")
            return gt_out
        else:
            return df_final

    def col_validation(
        self,
        *rules: Tuple[pl.Expr, str],
    ) -> pl.DataFrame:
        """
        Generates a validation report based on rules (pl.Expr) which evaluates to a single
        boolean per column.

        Parameters
        ----------
        rules
            A tuple of (pl.Expr, str), where the pl.Expr should evaluate to a single boolean value.
            If the boolean is False, then the entire column is considered to be violiating the rule.
        """
        rules_to_check = list(rules)
        rules_exprs = [r.name.keep() for r, _ in rules_to_check]
        violation_messages = [msg for _, msg in rules_to_check]

        df_temp = self._frame.select(*rules_exprs).collect()

        if len(df_temp) > 1:
            raise ValueError(
                "Column rules must evaluate to a single boolean expression for each rule. "
                f"But a dataframe of shape {df_temp.shape} is produced."
            )

        df_violation = df_temp.transpose(include_header=True, column_names=["pass"]).with_columns(
            pl.Series(name="__reason__", values=violation_messages)
        )

        return df_violation.filter(pl.col("pass").not_()).select("column", "__reason__")

    def row_validation(
        self,
        *rules: Tuple[pl.Expr, str],
        id_col: str | None = None,
        columns_to_keep: List[str] | None = None,
        all_reasons: bool = False,
    ) -> pl.DataFrame:
        """
        Generates a validation report based on rules (pl.Expr) which evaluates to booleans
        per row.

        Parameters
        ----------
        rules
            A tuple of (pl.Expr, str), where the pl.Expr should evaluate to a boolean value
            per row. If the boolean is False, then the row is considered a violation. The string
            should be an explanation of the violation.
        id_col
            If None, an "__index__" column will be generated which is the row number.
        columns_to_keep
            Other columns you wish to keep in the final report.
        all_reasons
            If true, all reasons for violations will be returned. If false, only 1 will be returned.
        """

        if id_col is None:
            df = self._frame.with_row_index(name="__index__")
            to_keep = ["__index__"]
        else:
            df = self._frame
            to_keep = [id_col]

        rules_to_check = list(rules)
        rules_exprs = [r.alias(n) for r, n in rules_to_check]
        all_rule_names = [n for _, n in rules_to_check]
        # Do not allow duplicate rule names
        existing_names = set()
        for name in all_rule_names:
            if name not in existing_names:
                existing_names.add(name)
            else:
                raise ValueError(f"Rule name {name} is duplicate. Please rename it.")

        # We cannot use list(set(..)) because that might change the order of all_rule_names

        if columns_to_keep is not None:
            to_keep += columns_to_keep

        df_temp = df.select(*to_keep, *rules_exprs).filter(
            # Filter to the violators.
            # pl.all_horizontal(*all_rule_names) = people who pass all rules
            # pl.all_horizontal(*all_rule_names).not_() = people who failed any one of the rules
            pl.all_horizontal(*all_rule_names).not_()
        )

        if all_reasons:
            reasons = [
                pl.when(pl.col(c)).then(None).otherwise(pl.lit(c, dtype=pl.String))
                for c in all_rule_names
            ]  # When true, return None. When false, return reason

            return df_temp.select(
                *to_keep, pl.concat_list(reasons).list.drop_nulls().list.sort().alias("__reason__")
            ).collect()
        else:
            # df_temp = all people who failed any one of the rules. So there must be at least one 0 in concat-ed list.
            return df_temp.select(
                *to_keep,
                pl.concat_list(all_rule_names)
                .list.arg_min()
                .replace_strict(old=list(range(len(all_rule_names))), new=all_rule_names)
                .alias("__reason__"),
            ).collect()

    def null_corr(
        self,
        subset: IntoExpr | Iterable[IntoExpr] = pl.all(),
        filter_by: pl.Expr | None = None,
    ) -> pl.DataFrame:
        """
        Computes the correlation between A is null and B is null for all (A, B) combinations
        in the given subset of columns.

        If either A or B is all null or all non-null, the null correlation will not be
        computed, since the value is not going to be meaningful.

        Parameters
        ----------
        subset
            Anything that can be put into a Polars .select statement. Defaults to pl.all()
        filter_by
            A boolean expression
        """

        cols = self._frame.select(subset).collect_schema().names()

        if filter_by is None:
            frame = self._frame.select(pl.col(cols).is_null()).collect()
        else:
            frame = self._frame.filter(filter_by).select(pl.col(cols).is_null()).collect()

        df_null_cnt = frame.sum()
        n = frame.shape[0]

        invalid = set(
            c for c, cnt in zip(df_null_cnt.columns, df_null_cnt.row(0)) if (cnt == 0 or cnt == n)
        )

        xx = []
        yy = []
        for x, y in combinations(cols, 2):
            if not (x in invalid or y in invalid):
                xx.append(x)
                yy.append(y)

        if len(xx) == 0:
            return pl.DataFrame(
                {"column_1": [], "column_2": [], "null_corr": []},
                schema={
                    "column_1": pl.String,
                    "column_2": pl.String,
                    "null_corr": pl.Float64,
                },
            )
        else:
            corrs = frame.select(
                pl.corr(x, y).alias(str(i)) for i, (x, y) in enumerate(zip(xx, yy))
            ).row(0)
            return pl.DataFrame({"column_1": xx, "column_2": yy, "null_corr": corrs}).sort(
                pl.col("null_corr").abs(), descending=True
            )

    def meta(self) -> Dict:
        """
        Returns internal data in this class as a dictionary.
        """
        out = self.__dict__.copy()
        out.pop("_frame")
        return out

    def str_stats(self) -> pl.DataFrame:
        """
        Returns basic statistics about the string columns.
        """
        to_check = self.strs
        frames = [
            self._frame.select(
                pl.lit(c).alias("column"),
                pl.col(c).null_count().alias("null_count"),
                pl.col(c).n_unique().alias("n_unique"),
                pl.col(c).value_counts(sort=True).first().struct.field(c).alias("most_freq"),
                pl.col(c)
                .value_counts(sort=True)
                .first()
                .struct.field("count")
                .alias("most_freq_cnt"),
                pl.col(c).str.len_bytes().min().alias("min_byte_len"),
                pl.col(c).str.len_chars().min().alias("min_char_len"),
                pl.col(c).str.len_bytes().mean().alias("avg_byte_len"),
                pl.col(c).str.len_chars().mean().alias("avg_char_len"),
                pl.col(c).str.len_bytes().max().alias("max_byte_len"),
                pl.col(c).str.len_chars().max().alias("max_char_len"),
                pl.col(c).str.len_bytes().quantile(0.05).alias("5p_byte_len"),
                pl.col(c).str.len_bytes().quantile(0.95).alias("95p_byte_len"),
            )
            for c in to_check
        ]
        return pl.concat(pl.collect_all(frames))

    def corr(
        self, subset: IntoExpr | Iterable[IntoExpr], method: CorrMethod = "pearson"
    ) -> pl.DataFrame:
        """
        Returns a dataframe containing correlation information between the subset and all numeric columns.
        Only numerical columns will be checked.

        Parameters
        ----------
        subset
            Anything that can be put into a Polars .select statement.
        method
            One of ["pearson", "spearman", "xi", "kendall", "bicor"]
        """

        to_check = self._frame.select(subset).collect_schema().names()

        corrs = [
            self._frame.select(
                # This calls corr from .stats
                pl.lit(x).alias("column"),
                *(corr(x, y, method=method).alias(y) for y in self.numerics),
            )
            for x in to_check
        ]

        return pl.concat(pl.collect_all(corrs))

    def plot_corr(
        self, subset: IntoExpr | Iterable[IntoExpr], method: CorrMethod = "pearson"
    ) -> GT:
        """
        Plots the correlations using classic heat maps.

        Parameters
        ----------
        subset
            Anything that can be put into a Polars .select statement.
        method
            One of ["pearson", "spearman", "xi", "kendall", "bicor"]
        """
        corr_values = self.corr(subset, method)
        cols = [c for c in corr_values.columns if c != "column"]
        return (
            GT(corr_values)
            .fmt_number(columns=cols, decimals=3)
            .data_color(
                columns=cols,
                palette=["#0202bd", "#bd0237"],
                domain=[-1, 1],
                alpha=0.5,
                na_color="#000000",
            )
        )

    def infer_prob(self) -> List[str]:
        """
        Infers columns that can potentially be probabilities. For f32/f64 columns, this checks if all values are
        between 0 and 1. For List[f32] or List[f64] columns, this checks whether the column can potentially be
        multi-class probabilities.
        """
        is_ok = (
            self._frame.select(
                *((pl.col(c).is_between(0.0, 1.0).all()).alias(c) for c in self.floats),
                *(
                    (
                        (
                            pl.col(c).list.eval((pl.element() >= 0.0).all()).list.first()
                        )  # every number must be positive
                        & ((pl.col(c).list.sum() - 1.0).abs() < 1e-6)  # class prob must sum to 1
                        & (
                            pl.col(c).list.len().min() == pl.col(c).list.len().max()
                        )  # class prob column must have the same length
                    ).alias(c)
                    for c in self.list_floats
                ),
            )
            .collect()
            .row(0)
        )

        return [c for c, ok in zip(self.floats + self.list_floats, is_ok) if ok is True]

    @lru_cache
    def infer_high_null(self, threshold: float = 0.75) -> List[str]:
        """
        Infers columns with more than threshold percentage nulls.

        Parameters
        ----------
        threshold
            The threshold above which a column will be considered high null
        """
        is_ok = (
            self._frame.select(
                (pl.col(c).null_count() >= pl.len() * threshold).alias(c)
                for c in self._frame.columns
            )
            .collect()
            .row(0)
        )

        return [c for c, ok in zip(self._frame.columns, is_ok) if ok is True]

    @lru_cache
    def infer_discrete(self, threshold: float = 0.1, max_val_cnt: int = 100) -> List[str]:
        """
        Infers discrete columns based on unique percentage and max_val_count.

        Parameters
        ----------
        threshold
            Columns with unique percentage lower than threshold will be considered
            discrete
        max_val_cnt
            Max number of unique values the column can have in order for it to be considered
            discrete
        """
        out: List[str] = self.bools + self.cats
        to_check = [c for c in self._frame.columns if c not in out]
        is_ok = (
            self._frame.select(
                (
                    (pl.col(c).n_unique() < max_val_cnt)
                    | (pl.col(c).n_unique() < threshold * pl.len())
                ).alias(c)
                for c in to_check
            )
            .collect()
            .row(0)
        )

        return [c for c, ok in zip(to_check, is_ok) if ok is True]

    @lru_cache
    def infer_const(self, include_null: bool = False) -> List[str]:
        """
        Infers whether the column is constant.

        Parameters
        ----------
        include_null
            If true, a constant column with null values will also be included.
        """
        if include_null:
            is_ok = (
                self._frame.select(
                    (
                        (pl.col(c).n_unique() == 1)
                        | ((pl.col(c).null_count() > 0) & (pl.col(c).n_unique() == 2))
                    ).alias(c)
                    for c in self._frame.columns
                )
                .collect()
                .row(0)
            )
        else:
            is_ok = (
                self._frame.select(
                    (pl.col(c).n_unique() == 1).alias(c) for c in self._frame.columns
                )
                .collect()
                .row(0)
            )

        return [c for c, ok in zip(self._frame.columns, is_ok) if ok is True]

    @lru_cache
    def infer_binary(self, include_null: bool = False) -> List[str]:
        """
        Infers whether the column is binary.

        Parameters
        ----------
        include_null
            If true, a binary column with 2 non-null distinct values and null will also be included.
        """
        if include_null:
            is_ok = (
                self._frame.select(
                    (
                        (pl.col(c).n_unique() == 2)
                        | ((pl.col(c).null_count() > 0) & (pl.col(c).n_unique() == 3))
                    ).alias(c)
                    for c in self._frame.columns
                )
                .collect()
                .row(0)
            )
        else:
            is_ok = (
                self._frame.select(
                    (pl.col(c).n_unique() == 2).alias(c) for c in self._frame.columns
                )
                .collect()
                .row(0)
            )

        return [c for c, ok in zip(self._frame.columns, is_ok) if ok is True]

    @lru_cache
    def infer_k_distinct(self, k: int, include_null: bool = False) -> List[str]:
        """
        Infers whether the column has k distinct values.

        Parameters
        ----------
        k
            Any positive integer.
        include_null
            If true, a binary column with k non-null distinct values and null will also be included.
        """
        if k < 1:
            raise ValueError("Input `k` must be >= 1.")

        if include_null:
            is_ok = (
                self._frame.select(
                    (
                        (pl.col(c).n_unique() == k)
                        | ((pl.col(c).null_count() > 0) & (pl.col(c).n_unique() == (k + 1)))
                    ).alias(c)
                    for c in self._frame.columns
                )
                .collect()
                .row(0)
            )
        else:
            is_ok = (
                self._frame.select(
                    (pl.col(c).n_unique() == k).alias(c) for c in self._frame.columns
                )
                .collect()
                .row(0)
            )

        return [c for c, ok in zip(self._frame.columns, is_ok) if ok is True]

    def infer_corr(self, method: CorrMethod = "pearson") -> pl.DataFrame:
        """
        Trying to infer highly correlated columns by computing correlation between
        all numerical (including boolean) columns.

        Parameters
        ----------
        method
            One of ["pearson", "spearman", "xi", "kendall"]
        """
        to_check = self.numerics + self.bools

        xx = []
        yy = []
        for x, y in combinations(to_check, 2):
            xx.append(x)
            yy.append(y)

        corrs = (
            self._frame.with_columns(pl.col(c).cast(pl.UInt8) for c in self.bools)
            .select(corr(x, y, method=method).alias(f"{i}") for i, (x, y) in enumerate(zip(xx, yy)))
            .collect()
            .row(0)
        )

        return pl.DataFrame({"x": xx, "y": yy, "corr": corrs}).sort(
            pl.col("corr").abs(), descending=True
        )

    def infer_dependency(self, subset: IntoExpr | Iterable[IntoExpr] = pl.all()) -> pl.DataFrame:
        """
        Infers (functional) dependency using the method of conditional entropy. This only evaluates
        potential qualifying columns. Potential qualifying columns are columns of type:
        int, str, categorical, or booleans.

        If returned conditional entropy is very low, that means knowning the column in
        `by` is enough to to infer the column in `column`, or the column in `column` can
        be determined by the column in `by`.

        Parameters
        ----------
        subset
            A subset of columns to try running the dependency check. The subset input can be
            anything that can be turned into a Polars selector. The df or the column subset of the df
            may contain columns that cannot be used for dependency detection, e.g. column of list of values.
            Only valid columns will be checked.
        """

        # Infer valid columns to run this detection
        valid = self.ints + self.strs + self.cats + self.bools
        check_frame = self._frame.select(subset)
        all_names = check_frame.collect_schema().names()
        to_check = [x for x in all_names if x in valid]
        n_uniques = check_frame.select(pl.col(c).n_unique() for c in to_check).collect().row(0)

        frame = (
            pl.DataFrame({"column": to_check, "n_unique": n_uniques})
            .filter(pl.col("n_unique") > 1)
            .sort("n_unique")
        )

        check = list(frame["column"])
        if len(check) <= 1:
            warnings.warn(
                f"Not enough valid columns to detect dependency on. Valid column count: {len(check)}. Empty dataframe returned.",
                stacklevel=2,
            )
            return pl.DataFrame(
                {"column": [], "by": [], "cond_entropy": []},
                schema={"column": pl.String, "by": pl.String, "cond_entropy": pl.Float64},
            )

        if len(check) != len(all_names):
            warnings.warn(
                f"The following columns are dropped because they cannot be used in dependency detection: {[f for f in all_names if f not in check]}",
                stacklevel=2,
            )

        # Construct output
        column = []
        by = []
        for x, y in combinations(check, 2):
            column.append(x)
            by.append(y)

        ce = (
            self._frame.select(
                query_cond_entropy(x, y).abs().alias(f"{i}")
                for i, (x, y) in enumerate(zip(column, by))
            )
            .collect()
            .row(0)
        )

        out = pl.DataFrame({"column": column, "by": by, "cond_entropy": ce}).sort("cond_entropy")

        return out

    def plot_dependency(
        self, threshold: float = 0.01, subset: IntoExpr | Iterable[IntoExpr] = pl.all()
    ) -> graphviz.Digraph:
        """
        Plot dependency using the result of self.infer_dependency and positively dtermines
        dependency by the threshold.

        Parameters
        ----------
        threshold
            If conditional entropy is < threshold, we draw a line indicating dependency.
        subset
            A subset of columns to try running the dependency check. The subset input can be
            anything that can be turned into a Polars selector
        """

        dep_frame = self.infer_dependency(subset=subset)

        df_local = dep_frame.filter((pl.col("cond_entropy") < threshold)).select(
            pl.col("column").alias("child"),  # c for child
            pl.col("by").alias("parent"),  # p for parent
        )
        cp = df_local.group_by("child").agg(pl.col("parent"))
        pc = df_local.group_by("parent").agg(pl.col("child"))
        child_parent: dict[str, pl.Series] = dict(zip(cp["child"], cp["parent"]))
        parent_child: dict[str, pl.Series] = dict(zip(pc["parent"], pc["child"]))

        dot = graphviz.Digraph(
            "Dependency Plot", comment=f"Conditional Entropy < {threshold:.2f}", format="png"
        )
        for c, par in child_parent.items():
            parents_of_c = set(par)
            for p in par:
                # Does parent p have a child that is also a parent of c? If so, remove p.
                children_of_p = parent_child.get(p, None)
                if children_of_p is not None:
                    if len(parents_of_c.intersection(children_of_p)) > 0:
                        parents_of_c.remove(p)

            dot.node(c)
            for p in parents_of_c:
                dot.node(p)
                dot.edge(p, c)

        return dot

    def plot_feature_distr(
        self,
        feature: str,
        n_bins: int | None = None,
        density: bool = False,
        show_bad_values: bool = True,
        min_: float | pl.Expr | None = None,
        max_: float | pl.Expr | None = None,
        over: str | None = None,
        filter_by: pl.Expr | None = None,
    ) -> alt.Chart:
        """
        Plot distribution of the feature with a few statistical details.

        Parameters
        ----------
        feature
            A string representing a column name
        n_bins
            The number of bins used for histograms. Not used when the feature column is categorical.
        density
            Whether to plot a probability density or not
        show_bad_values
            Whether to show % of bad (null or non-finite) values
        min_
            Whether to ignore values strictly lower than min_
        max_
            Whether to ignore values strictly higher than max_
        over
            Whether to look at the distribution over another categorical column
        filter_by
            An extra condition you may want to impose on the underlying dataset
        """
        if feature not in self.numerics:
            raise ValueError("Input feature must be numeric.")

        conditions = []
        if filter_by is not None:
            conditions.append(filter_by)
        if min_ is not None:
            conditions.append(pl.col(feature) >= min_)
        if max_ is not None:
            conditions.append(pl.col(feature) <= max_)

        if len(conditions) > 0:
            df = self._frame.filter(pl.all_horizontal(*conditions)).select(feature, over).collect()
        else:
            df = self._frame.select(feature, over).collect()

        return plot_feature_distr(
            df=df,
            feature=feature,
            n_bins=n_bins,
            density=density,
            show_bad_values=show_bad_values,
            over=over,
        )
