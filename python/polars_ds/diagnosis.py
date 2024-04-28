import polars.selectors as cs
import polars as pl
import logging
import plotly.express as px
import plotly.graph_objects as go
from typing import Union, List, Optional, Iterable, Tuple
from functools import lru_cache
from .num import query_cond_entropy, query_principal_components, query_lstsq_report
from itertools import combinations
from .type_alias import CorrMethod
from .stats import corr
from .sample import sample
import graphviz
from great_tables import GT, nanoplot_options
from polars.type_aliases import IntoExpr

logger = logging.getLogger(__name__)


# DIA = Data Inspection Assistant / DIAgonsis
class DIA:

    """
    Data Inspection Assistant. Most plots are powered by plotly/great_tables. Plotly may require
    additional package downloads.

    If you cannot import this module, please try: pip install "polars_ds[plot]"
    """

    def __init__(self, df: Union[pl.DataFrame, pl.LazyFrame]):
        self._frame: pl.LazyFrame = df.lazy()
        self.numerics: List[str] = df.select(cs.numeric()).columns
        self.ints: List[str] = df.select(cs.integer()).columns
        self.floats: List[str] = df.select(cs.float()).columns
        self.strs: List[str] = df.select(cs.string()).columns
        self.bools: List[str] = df.select(cs.boolean()).columns
        self.cats: List[str] = df.select(cs.categorical()).columns
        self.list_floats: List[str] = [
            c
            for c, t in df.schema.items()
            if (t.is_(pl.List(pl.Float32)) or (t.is_(pl.List(pl.Float64))))
        ]
        self.list_ints: List[str] = [
            c
            for c, t in df.schema.items()
            if t.is_(pl.List(pl.UInt8))
            or t.is_(pl.List(pl.UInt16))
            or t.is_(pl.List(pl.UInt32))
            or t.is_(pl.List(pl.UInt64))
            or t.is_(pl.List(pl.Int8))
            or t.is_(pl.List(pl.Int16))
            or t.is_(pl.List(pl.Int32))
            or t.is_(pl.List(pl.Int64))
        ]
        self.list_bool: List[str] = [c for c, t in df.schema.items() if t.is_(pl.List(pl.Boolean))]
        self.list_str: List[str] = [c for c, t in df.schema.items() if t.is_(pl.List(pl.String))]

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
        self.other_types: List[str] = [c for c in self._frame.columns if c not in self.simple_types]

    def numeric_profile(self, n_bins: int = 20, iqr_multiplier: float = 1.5):
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
        """
        to_check = self.numerics

        cuts = [i / n_bins for i in range(n_bins)]
        cuts[0] -= 1e-5
        cuts[-1] += 1e-5
        frames = []
        for c in to_check:
            temp = self._frame.select(
                pl.lit(c).alias("column"),
                pl.col(c).count().alias("non_null_cnt"),
                (pl.col(c).null_count() / pl.len()).alias("null%"),
                pl.col(c).mean().alias("mean"),
                pl.col(c).std().alias("std"),
                pl.col(c).min().cast(pl.Float64).alias("min"),
                pl.col(c).quantile(0.25).cast(pl.Float64).alias("q1"),
                pl.col(c).median().cast(pl.Float64).round(2).alias("median"),
                pl.col(c).quantile(0.75).cast(pl.Float64).alias("q3"),
                pl.col(c).max().cast(pl.Float64).alias("max"),
                (pl.col(c).quantile(0.75) - pl.col(c).quantile(0.25)).cast(pl.Float64).alias("IQR"),
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
                    .struct.field("brk")
                    .value_counts()
                    .sort()
                    .struct.field("count")
                    .implode()
                ).alias("histogram"),
            )
            frames.append(temp)

        df_final = pl.concat(pl.collect_all(frames))
        return (
            GT(df_final, rowname_col="column")
            .tab_stubhead("column")
            .fmt_percent(columns="null%")
            .fmt_number(
                columns=["mean", "std", "min", "q1", "median", "q3", "max", "IQR"], decimals=3
            )
            .fmt_nanoplot(columns="histogram", plot_type="bar")
        )

    def plot_null_distribution(
        self, subset: Union[IntoExpr, Iterable[IntoExpr]] = pl.all(), n_bins: int = 50
    ):
        """
        Checks the null percentages per row group. Row groups are consecutive rows grouped by row number,
        with each group having len//n_bins number of elements. The height of each bin is the percentage
        of nulls in the row group.

        This plot shows whether nulls in one feature is correlated with nulls in other features.

        Parameters
        ----------
        subset
            Anything that can be put into a Polars .select statement. Defaults to pl.all()
        n_bins
            The number
        """
        cols = self._frame.select(subset).columns
        temp = (
            self._frame.with_row_index(name="row_group")
            .group_by((pl.col("row_group") // (pl.len() // n_bins)).alias("row_group"))
            .agg(pl.col(cols).null_count() / pl.len())
            .sort("row_group")
            .select(
                pl.col(cols).exclude(["row_group"]).implode(),
            )
            .collect()
        )
        # Values for plot. The first n are list[f64] used in nanoplot. The rest are overall null rates
        percentages = temp.row(0)

        temp2 = self._frame.select(pl.col(cols).null_count() / pl.len()).collect()
        null_rates = temp2.row(0)

        null_table = pl.DataFrame(
            {
                "column": cols,
                "percentages in row groups": [{"val": values} for values in percentages],
                "null%": null_rates,
            }
        )

        return (
            GT(null_table, rowname_col="column")
            .tab_header(title="Null Distribution")
            .tab_stubhead("column")
            .fmt_number(columns=["null%"], decimals=5)
            .fmt_percent(columns="null%")
            .fmt_nanoplot(
                columns="percentages in row groups",
                plot_type="bar",
                options=nanoplot_options(data_bar_fill_color="red"),
            )
        )

    def meta(self):
        """
        Returns internal data in this class as a dictionary.
        """
        out = self.__dict__.copy()
        out.pop("_frame")
        return out

    def str_stats(self):
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
        self, subset: Union[IntoExpr, Iterable[IntoExpr]], method: CorrMethod = "pearson"
    ) -> pl.DataFrame:
        """
        Returns a dataframe containing correlation information between the subset and all numeric columns.

        Parameters
        ----------
        subset
            Anything that can be put into a Polars .select statement.
        method
            One of ["pearson", "spearman", "xi", "kendall"]
        """
        temp = self._frame.select(subset).columns
        to_check = [c for c in temp if c in self.numerics]
        if len(to_check) != len(temp):
            removed = list(set(temp).difference(to_check))
            logger.info(
                f"The following columns are not numeric/not in the dataframe, skipped: \n{removed}"
            )

        corrs = [
            self._frame.select(
                pl.lit(x).alias("column"), *(corr(x, y).alias(y) for y in self.numerics)
            )
            for x in to_check
        ]

        return pl.concat(pl.collect_all(corrs))

    def plot_corr(
        self, subset: Union[IntoExpr, Iterable[IntoExpr]], method: CorrMethod = "pearson"
    ):
        """
        Plots the correlations using classic heat maps.

        Parameters
        ----------
        subset
            Anything that can be put into a Polars .select statement.
        method
            One of ["pearson", "spearman", "xi", "kendall"]
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
                        & (pl.col(c).list.sum() == 1.0)  # class prob must sum to 1
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

    @lru_cache
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
        correlation = (
            self._frame.with_columns(pl.col(c).cast(pl.UInt8) for c in self.bools)
            .select(corr(x, y).alias(f"{i}") for i, (x, y) in enumerate(combinations(to_check, 2)))
            .collect()
            .row(0)
        )

        xx = []
        yy = []
        for x, y in combinations(to_check, 2):
            xx.append(x)
            yy.append(y)

        return pl.DataFrame({"x": xx, "y": yy, "corr": correlation}).sort(
            pl.col("corr").abs(), descending=True
        )

    def infer_dependency(
        self, subset: Union[IntoExpr, Iterable[IntoExpr]] = pl.all()
    ) -> pl.DataFrame:
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
            anything that can be turned into a Polars selector. Only valid columns will be checked,
            however.
        """

        # Infer valid columns to run this detection
        valid = self.ints + self.strs + self.cats + self.bools
        to_check = [x for x in self._frame.select(subset).columns if x in valid]

        n_uniques = self._frame.select(pl.col(c).n_unique() for c in to_check).collect().row(0)

        frame = (
            pl.DataFrame({"column": to_check, "n_unique": n_uniques})
            .filter(pl.col("n_unique") > 1)
            .sort("n_unique")
        )

        check = list(frame["column"])
        if len(check) <= 1:
            logger.info(
                f"Not enough valid columns to detect dependency on. Valid column count: {len(check)}."
            )
            return pl.DataFrame(
                {"column": [], "by": [], "cond_entropy": []},
                schema={"column": pl.String, "by": pl.String, "cond_entropy": pl.Float64},
            )

        # Compute conditional entropy on the rest of the columns
        logger.info(f"Running dependency for columns: {check}.")

        ce = (
            self._frame.select(
                query_cond_entropy(x, y).abs().alias(f"{i}")
                for i, (x, y) in enumerate(combinations(check, 2))
            )
            .collect()
            .row(0)
        )

        # Construct output
        column = []
        by = []
        for x, y in combinations(check, 2):
            column.append(x)
            by.append(y)

        out = pl.DataFrame({"column": column, "by": by, "cond_entropy": ce}).sort("cond_entropy")

        return out

    def plot_dependency(
        self, threshold: float = 0.01, subset: Union[IntoExpr, Iterable[IntoExpr]] = pl.all()
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
        child_parent: dict[str, pl.Series] = dict(
            zip(cp.drop_in_place("child"), cp.drop_in_place("parent"))
        )
        parent_child: dict[str, pl.Series] = dict(
            zip(pc.drop_in_place("parent"), pc.drop_in_place("child"))
        )

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

    def plot_lstsq(
        self,
        x: Union[IntoExpr, Iterable[IntoExpr]],
        target: Union[IntoExpr, Iterable[IntoExpr]],
        add_bias: bool = False,
        condition: Optional[pl.Expr] = None,
        max_points: int = 20_000,
        **kwargs,
    ) -> Tuple[pl.DataFrame, go.Figure]:
        """
        Plots the least squares between x and target.

        Paramters
        ---------
        x
            The preditive variable
        target
            The target variable
        add_bias
            Whether to add bias in the linear regression
        condition
            An additional filter condition you want to apply before runing lstsq on the data. This must
            be a boolean expression. If none, run this on the entire dataset. (The frame used to initialize DIA.)
        max_points
            The max number of points to be displayed. If data > this limit, the data will be sampled
        kwargs
            Kwargs to be passed to Plotly's Figure object
        """
        if condition is None:
            temp = self._frame.select(x, target)
            condition_str = ""
        else:
            temp = self._frame.filter(condition).select(x, target)
            condition_str = "\nCondition: " + str(condition)

        x_name, y_name = temp.columns
        coeffs = (
            temp.select(
                query_lstsq_report(x_name, target=y_name, add_bias=add_bias).alias("report")
            )
            .unnest("report")
            .select(
                pl.all().exclude("idx")  # All but the idx column in lstsq_report
            )
            .collect()
        )
        if add_bias:
            b1, alpha = coeffs["coeff"]
        else:
            b1, alpha = coeffs["coeff"][0], 0
        # Get the data necessary for plotting
        temp = self._frame.select(
            pl.col(x_name).alias("x"),
            pl.col(y_name).alias("y"),
            (pl.col(x_name) * b1 + alpha).alias("y_pred"),
        )  # Sample down. If len(temp) < max_points, all temp will be selected. This sample supports lazy.
        df = sample(temp, value=max_points)

        fig = go.Figure(**kwargs)
        fig.update_layout(
            title=f"y={y_name}, x={x_name}, y_pred = ({x_name}) * {b1:.5f} + {alpha:.5f}<br><sup>{condition_str}</sup>",
            xaxis_title=x_name,
            yaxis_title=y_name,
        )

        fig.add_trace(go.Scatter(x=df["x"], y=df["y"], mode="markers", name="data scatter"))
        fig.add_trace(go.Scatter(x=df["x"], y=df["y_pred"], mode="lines", name="Least Squares"))
        print(coeffs)
        return fig

    def plot_pca(
        self,
        *features: Union[IntoExpr, Iterable[IntoExpr]],
        by: Union[IntoExpr, Iterable[IntoExpr]],
        center: bool = True,
        dim: int = 2,
        max_points: int = 20_000,
        **kwargs,
    ) -> go.Figure:
        """
        Creates a scatter plot based on the reduced dimensions via PCA, and color it by `by`.

        Paramters
        ---------
        features
            Any selection expression for Polars
        by
            Color the 2-D PCA plot by the values in the column
        center
            Whether to automatically center the features
        dim
            Either 2 or 3. Plot either a 2d principal component plot or a 3d one.
        max_points
            The max number of points to be displayed. If data > this limit, the data will be sampled.
        kwargs
            Anything else that will be passed to plotly's scatter function
        """
        feats = self._frame.select(features).columns
        if len(feats) < 2:
            raise ValueError("You must pass >= 2 features.")
        if dim < 2 or dim > 3:
            raise ValueError("Input `dim` must either be 2 or 3.")

        temp = self._frame.select(
            query_principal_components(*feats, center=center, k=dim).alias("pc"), by
        )
        df = sample(temp, value=max_points).unnest("pc")

        if dim == 2:
            fig = px.scatter(
                x=df["pc1"],
                y=df["pc2"],
                color=df[by],
                labels={"x": "pc1", "y": "pc2"},
                title="2 Principal Components",
                **kwargs,
            )
        else:
            fig = px.scatter_3d(
                x=df["pc1"],
                y=df["pc2"],
                z=df["pc3"],
                color=df[by],
                labels={"x": "pc1", "y": "pc2", "z": "pc3"},
                title="3 Principal Components",
                **kwargs,
            )
        return fig
