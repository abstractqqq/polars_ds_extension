from __future__ import annotations

import polars as pl
import altair as alt
from typing import Iterable, List, Tuple
from polars._typing import IntoExpr
# Internal dependencies
import polars_ds.sample_and_split as sa
from polars_ds import query_r2, principal_components, query_tpr_fpr, integrate_trapz

alt.data_transformers.enable("vegafusion")

# Plots should never have a title. Title must be editable by the end user
# Interactivity should only be enabled by the end user

def plot_feature(
    *, 
    feature: str | pl.Expr | Iterable[float],
    n_bins: int | None = None,
    density: bool = False,
    show_bad_values: bool = True,
    df: pl.DataFrame | pl.LazyFrame | None = None,
) -> Tuple[pl.DataFrame, alt.Chart]:
    """
    Plot distribution of the feature with a few statistical details.

    Parameters
    ----------
    df
        Either an eager or lazy Polars Dataframe
    feature
        A string representing a column name
    n_bins
        The number of bins used for histograms. Not used when the feature column is categorical.
    density
        Whether to plot a probability density or not
    show_bad_values
        Whether to show % of bad (null or inf or nan) values
    """
    # include_null
    #     When by is not null, whether to consider null a segment or not. If true, null values will be
    #     mapped to the name "__null__". The string "__null__" should not exist originally in the column.
    #     This is a workaround to get plotly to recognize null values.
    
    if n_bins <= 2:
        raise ValueError("Input `n_bins` must be > 2.")

    if isinstance(feature, str):
        if df is None:
            raise ValueError("If `feature` is str, then df cannot be none.")
        feat = feature
        data = df.lazy()
    elif isinstance(feature, pl.Expr):
        if df is None:
            raise ValueError("If `feature` is pl.expr, then df cannot be none.")
        data = df.lazy()
        feat = data.select(feature).collect_schema().names()[0]
    else:
        feat = "feature"
        data = pl.Series(name = "feature", values = feature).to_frame().lazy()

    frame = data.filter(
        pl.all_horizontal(pl.col(feat).is_finite(), pl.col(feat).is_not_null())
    ).collect()

    p5, median, mean, p95, min_, max_ = frame.select(
        p5=pl.col(feat).quantile(0.05),
        median=pl.col(feat).median(),
        mean=pl.col(feat).mean(),
        p95=pl.col(feat).quantile(0.95),
        min=pl.col(feat).min(),
        max=pl.col(feat).max(),
    ).row(0)

    # bin computation
    range_ = max_ - min_
    recip = 1 / n_bins
    cuts = [recip * (i + 0.5) for i in range(1, n_bins + 1)]
    df_plot = (
        frame.select(
            ((pl.col(feat) - min_) / range_)
            .cut(breaks=cuts, include_breaks=True)
            .struct.rename_fields(["brk", "category"])
            .struct.field("brk")
            .value_counts(parallel=True)
            .sort()
            .alias("bins")
        )
        .unnest("bins")
        .select(counts=pl.col("count"), cuts=pl.col("brk") * range_ + min_)
    )
    # histgram plot
    # df_plot = pl.DataFrame({"counts": cnt, "cuts": values})
    density_str = "density" if density else "counts"
    alt_y = alt.Y(f"{density_str}:Q", scale=alt.Scale(domainMin=0)).title(density_str)
    if density:
        df_plot = df_plot.with_columns(density=pl.col("counts") / pl.col("counts").sum())

    base = alt.Chart(df_plot)
    dist_chart = base.mark_bar(size=15).encode(
        alt.X("cuts:Q", axis=alt.Axis(tickCount=n_bins // 2, grid=False)),
        alt_y,
        tooltip=[
            alt.Tooltip("cuts:Q", title="CutValue"),
            alt.Tooltip(f"{density_str}:Q", title=density_str),
        ],
    )
    # stats overlay
    df_stats = pl.DataFrame(
        {"names": ["p5", "p50", "avg", "p95"], "stats": [p5, median, mean, p95]}
    )

    stats_base = alt.Chart(df_stats)
    stats_chart = stats_base.mark_rule(color="#f086ab").encode(
        x=alt.X("stats").title(""),
        tooltip=[
            alt.Tooltip("names:N", title="Stats"),
            alt.Tooltip("stats:Q", title="Value"),
        ],
    )
    # null, inf, nan percentages bar
    if show_bad_values:
        bad_pct = (
            data.select(
                pl.any_horizontal(pl.col(feat).is_null(), ~pl.col(feat).is_finite()).sum()
                / pl.len()
            )
            .collect()
            .item(0, 0)
        )

        df_bad = pl.DataFrame({"(Null/NaN/Inf)%": [bad_pct]})
        bad_chart = (
            alt.Chart(df_bad)
            .mark_bar(opacity=0.5)
            .encode(
                alt.X("(Null/NaN/Inf)%:Q", scale=alt.Scale(domain=[0, 1])),
                tooltip=[
                    alt.Tooltip("(Null/NaN/Inf)%:Q", title="(Null/NaN/Inf)%"),
                ],
            )
        )
        chart = alt.vconcat(dist_chart + stats_chart, bad_chart)
    else:
        chart = dist_chart + stats_chart

    return df_plot, chart

def plot_feature_over(
    *,
    df: pl.DataFrame | pl.LazyFrame,
    feature: str,
    segment: str,
    n_bins: int = 30,
    density: bool = True,
    show_bad_values: bool = True,
    include_null_segment: bool = False,
    # segment_null_replacer
) -> alt.Chart:
    """
    Compare the distribution of a feature over a segment.

    Parameters
    ----------
    df
        Either an eager or lazy Polars Dataframe
    feature
        A string representing a column name
    segment
        The segment.
    n_bins
        The max number of bins for the plot.
    density
        Whether to show a histogram or a density plot
    show_bad_values
        Whether to show % of bad (null or inf or nan) values
    include_null_segment
        Whether to treat null values in the segment column as a segment.
    """
    if n_bins <= 2:
        raise ValueError("Input `n_bins` must be > 2.")

    if not isinstance(segment, str):
        raise ValueError("Input `segment` must be a string.")

    if isinstance(feature, str):
        feat = feature
        data = df.lazy() 
    elif isinstance(feature, pl.Expr):
        data = df.lazy()
        feat = data.select(feature).collect_schema().names()[0]
    else:
        feat = "feature"
        data = pl.Series(name = "feature", values = feature).to_frame().lazy()
    
    if not include_null_segment:
        data = data.filter(pl.col(segment).is_not_null())

    feat, segment = data.select(feature, segment).collect_schema().names()
    frame = (
        data.filter(
            pl.all_horizontal(pl.col(feat).is_not_null(), pl.col(feat).is_finite())
        )
        .select(feat, pl.col(segment))
        .collect()
    )

    selection = alt.selection_point(fields=[segment], bind="legend")
    # Null will be a group in Altair's chart, but it breaks the predicate evaluation, making
    # toggling the null group impossible. (This is likely a Altair bug). We can
    # map nulls to a special string '__null__' to avoid that issue
    # frame = frame.with_columns(pl.col(segment).cast(pl.String).fill_null(pl.lit("__null__")))
    base = alt.Chart(frame)
    if density:
        dist_chart = (
            base.transform_density(
                feat,
                groupby=[segment],
                as_=[feat, "density"],
            )
            .mark_bar(opacity=0.5, binSpacing=0)
            .encode(
                alt.X(f"{feat}:Q"),
                alt.Y("density:Q", scale=alt.Scale(domainMin=0)).stack(None),
                color=alt.Color(f"{segment}:N"), # legend=alt.Legend(columns=8)
                opacity=alt.condition(selection, alt.value(0.5), alt.value(0.0)),
            )
            .add_selection(selection)
        )
    else:
        dist_chart = (
            base.mark_bar(opacity=0.5, binSpacing=0)
            .encode(
                alt.X(f"{feat}:Q"),
                alt.Y("count()", scale=alt.Scale(domainMin=0)).stack(None),
                color=f"{segment}:N",
                opacity=alt.condition(selection, alt.value(0.5), alt.value(0.0)),
            )
            .add_selection(selection)
        )

    if show_bad_values:
        df_bad = (
            data.group_by(segment)
            .agg(bad_rate=(pl.col(feat).is_null() | (~pl.col(feat).is_finite())).sum() / pl.len())
            .collect()
            # .with_columns(pl.col(segment).fill_null(pl.lit("__null__")))
        )
        bad_chart = (
            alt.Chart(df_bad)
            .mark_bar(opacity=0.5)
            .encode(
                alt.X("bad_rate:Q", scale=alt.Scale(domain=[0, 1])).title("(Null/NaN/Inf)%"),
                alt.Y(f"{segment}:N"),
                color=f"{segment}:N",
                tooltip=[
                    alt.Tooltip("bad_rate:Q", title="(Null/NaN/Inf)%"),
                ],
            )
        )
        return alt.vconcat(dist_chart, bad_chart)
    else:
        return dist_chart

def plot_lin_reg(
    df: pl.DataFrame | pl.LazyFrame,
    x: str,
    target: str,
    add_bias: bool = False,
    weights: str | None = None,
    max_points: int = 20_000,
    show_lin_reg_eq: bool = True,
) -> alt.Chart:
    """
    Plots the linear regression line between x and target.

    Paramters
    ---------
    df
        Either an eager or lazy Polars Dataframe
    x
        The preditive variable
    target
        The target variable
    add_bias
        Whether to add bias in the linear regression
    weights
        Weights for the linear regression
    max_points
        The max number of points to be displayed. Notice that this only affects the number of points
        on the plot. The linear regression will still be fit on the entire dataset.
    show_lin_reg_eq
        Whether to show the linear regression equation at the bottom or not
    """

    to_select = [x, target] if weights is None else [x, target, weights]
    temp = df.lazy().select(*to_select)

    xx = pl.col(x)
    yy = pl.col(target)
    # Although using simple_lin_reg might seem to be able to reduce some code here,
    # it adds complexity because of output type and the r2 query.
    # A little bit of code dup is reasonable.
    if add_bias:
        if weights is None:
            x_mean = xx.mean()
            y_mean = yy.mean()
            beta = (xx - x_mean).dot(yy - y_mean) / (xx - x_mean).dot(xx - x_mean)
            alpha = y_mean - beta * x_mean
        else:
            w = pl.col(weights)
            w_sum = w.sum()
            x_wmean = w.dot(xx) / w_sum
            y_wmean = w.dot(yy) / w_sum
            beta = w.dot((xx - x_wmean) * (yy - y_wmean)) / (w.dot((xx - x_wmean).pow(2)))
            alpha = y_wmean - beta * x_wmean
    else:
        if weights is None:
            beta = xx.dot(yy) / xx.dot(xx)
        else:
            w = pl.col(weights)
            beta = w.dot(xx * yy) / w.dot(xx.pow(2))

        alpha = pl.lit(0, dtype=pl.Float64)

    beta, alpha, r2, length = (
        temp.select(
            beta.alias("beta"),
            alpha.alias("alpha"),
            query_r2(yy, xx * beta + alpha).alias("r2"),
            pl.len()
        )
        .collect()
        .row(0)
    )

    df_need = temp.select(
        xx,
        yy,
        (xx * beta + alpha).alias("y_pred"),
    )
    # Sample down if len(temp) > max_points
    df_sampled = sa.sample(df_need, value=max_points) if length > max_points else df_need.collect()

    x_title = [x]
    if show_lin_reg_eq:
        if add_bias and alpha > 0:
            reg_info = f"y = {beta:.4f} * x + {round(alpha, 4) if add_bias else ''}, r2 = {r2:.4f}"
        elif add_bias and alpha < 0:
            reg_info = (
                f"y = {beta:.4f} * x - {abs(round(alpha, 4)) if add_bias else ''}, r2 = {r2:.4f}"
            )
        else:
            reg_info = f"y = {beta:.4f} * x, r2 = {r2:.4f}"

        x_title.append(reg_info)

    chart = (
        alt.Chart(df_sampled)
        .mark_point()
        .encode(alt.X(x).scale(zero=False), alt.Y(target))
    )
    return (
        chart 
        + chart.mark_line().encode(
            alt.X(x, title = x_title).scale(zero=False), 
            alt.Y("y_pred"),
        )
    )


def plot_pca(
    df: pl.DataFrame | pl.LazyFrame,
    features: List[str],
    by: IntoExpr,
    center: bool = True,
    dim: int = 2,
    filter_by: pl.Expr | None = None,
    max_points: int = 10_000,
    **kwargs,
) -> alt.Chart:
    """
    Creates a scatter plot based on the reduced dimensions via PCA, and color it by `by`.

    Paramters
    ---------
    df
        Either an eager or lazy Polars Dataframe
    features
        List of feature names
    by
        Color the 2-D PCA plot by the values in the column
    center
        Whether to automatically center the features
    dim
        Only 2 principal components plot can be done at this moment.
    filter_by
        A boolean expression
    max_points
        The max number of points to be displayed. If data > this limit, the data will be sampled.
    kwargs
        Anything else that will be passed to Altair encode function
    """
    if len(features) < 2:
        raise ValueError("You must pass >= 2 features.")
    if dim not in (2, 3):
        raise ValueError("Dim must be 2 or 3.")

    frame = df if filter_by is None else df.filter(filter_by)

    temp = frame.select(principal_components(*features, center=center, k=dim).alias("pc"), by)
    df_plot = sa.sample(temp, value=max_points).unnest("pc")

    if dim == 2:
        return alt.Chart(df_plot).mark_circle(size=60).encode(
            x='pc1',
            y='pc2',
            color=by,
            **kwargs
        ) # .interactive()
    else: # 3d
        raise NotImplementedError

def plot_roc_auc(
    *,
    actual: Iterable[int] | str | pl.Expr,
    pred: Iterable[float] | str | pl.Expr,
    df: pl.DataFrame | pl.LazyFrame | None = None,
    show_auc: bool = True,
    estimator_name: str = "",
    line_color: str = "#92e884",
    round_to: int = 4
) -> alt.Chart:
    """
    Plots ROC AUC curve.

    Paramters
    ---------
    df
        Either an eager or lazy Polars Dataframe
    actual
        A column which has the actual binary target information
    pred
        The prediction
    show_auc
        Whether to show the AUC value or not
    estimator_name
        Name for the estiamtor. Only shown if show_auc is True
    line_color
        HTML color code
    round_to
        Round to n-th decimal digit if show_auc is True
    """
    # expr_based = isinstance(actual, (str, pl.Expr)) and isinstance(pred, (str, pl.Expr)) and isinstance(df, (pl.DataFrame, pl.LazyFrame))
    if isinstance(actual, (str, pl.Expr)) and isinstance(pred, (str, pl.Expr)) and isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        zero = pl.DataFrame({
            "tpr": [0.],
            "fpr": [0.],
        }, schema = {
            "tpr": pl.Float64,
            "fpr": pl.Float64,
        })
        
        tpr_fpr = df.lazy().select(
            tpr_fpr = query_tpr_fpr(actual, pred).reverse()
        ).unnest("tpr_fpr").select(
            "tpr",
            "fpr",
        ).collect()
        df_plot = pl.concat([zero, tpr_fpr])

        chart = alt.Chart(df_plot).mark_line(interpolate="step", color = line_color).encode(
            x=alt.X('fpr', title = "False Positive Rate"),
            y=alt.Y('tpr', title = "True Positive Rate"),
        )
        if show_auc:
            auc = tpr_fpr.select(
                integrate_trapz("tpr", "fpr")
            ).item(0, 0)
            df_text = pl.DataFrame({
                "x": [1.0]
                , "y": [0.]
            })
            estimator = estimator_name.strip()
            auc_text = f"AUC = {round(auc, round_to)}" if estimator == "" else f"{estimator} (AUC = {round(auc, round_to)})"
            text = alt.Chart(df_text).mark_point(opacity=0.0).encode(
                x = alt.X("x"),
                y = alt.Y("y"),
            ).mark_text(
                dx = -1,
                dy = -5,
                fontWeight="bold",
                text = auc_text,
                align="right"
            )
            return chart + text
        else:
            return chart
    else: # May fail. User should catch
        s1 = pl.Series("actual", values=actual, dtype=pl.UInt32)
        s2 = pl.Series("pred", values=pred)
        df_temp = pl.DataFrame({
            "actual": s1,
            "pred": s2, 
        })
        return plot_roc_auc(df = df_temp, actual = "actual", pred = "pred", show_auc=show_auc, estimator_name = estimator_name, line_color=line_color, round_to=round_to)

