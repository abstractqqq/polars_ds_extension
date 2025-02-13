from __future__ import annotations

import polars as pl
import altair as alt
from typing import Iterable, List
from polars._typing import IntoExpr

# Internal dependencies
import polars_ds.sample_and_split as sa
from polars_ds import query_r2, principal_components, query_tpr_fpr, integrate_trapz

alt.data_transformers.enable("vegafusion")

# Plots should never have a title. Title must be editable by the end user
# Interactivity should only be enabled by the end user


def plot_feature_distr(
    *,
    feature: str | Iterable[float],
    n_bins: int = 10,
    density: bool = False,
    show_bad_values: bool = True,
    over: str | None = None,
    df: pl.DataFrame | pl.LazyFrame | None = None,
) -> alt.Chart:
    """
    Plot distribution of the feature with a few statistical details.

    Parameters
    ----------
    df
        Either an eager or lazy Polars Dataframe
    feature
        A string representing a column name
    n_bins
        The max number of bins used for histograms.
    density
        Whether to plot a probability density or not
    show_bad_values
        Whether to show % of bad (null or non-finite) values
    over
        Whether to look at the distribution over another categorical column
    """

    if n_bins <= 2:
        raise ValueError("Input `n_bins` must be > 2.")

    if isinstance(feature, str):
        if df is None:
            raise ValueError("If `feature` is str, then df cannot be none.")
        data = df.lazy().collect()
        feat = feature
    else:
        data = pl.Series(name="feature", values=feature).to_frame()
        feat = "feature"

    # selection = alt.selection_point(fields=['species'], bind='legend')
    if density:
        if over is None:
            chart = (
                alt.Chart(data)
                .transform_density(
                    feat,
                    as_=[feat, "density"],
                )
                .mark_area()
                .encode(
                    x=f"{feat}:Q",
                    y=alt.Y("density:Q").stack(None),
                )
            )
        else:
            selection = alt.selection_multi(fields=[over], bind="legend")
            chart = (
                alt.Chart(data)
                .transform_density(feat, as_=[feat, "density"], groupby=[over])
                .mark_area()
                .encode(
                    x=f"{feat}:Q",
                    y=alt.Y("density:Q").stack(None),
                    color=over,
                    opacity=alt.condition(selection, alt.value(0.8), alt.value(0.2)),
                )
                .add_params(selection)
            )
    else:
        if over is None:
            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    alt.X(f"{feat}:Q").bin(maxbins=n_bins).title(feat),
                    y=alt.Y("count()").stack(None),
                )
            )
        else:
            selection = alt.selection_point(fields=[over], bind="legend")
            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    alt.X(f"{feat}:Q").bin(maxbins=n_bins).title(feat),
                    y=alt.Y("count()").stack(None),
                    color=over,
                    opacity=alt.condition(selection, alt.value(0.8), alt.value(0.2)),
                )
                .add_params(selection)
            )

    if over is None:
        p5, median, mean, p95, min_, max_, cnt, null_cnt, not_finite = data.select(
            p5=pl.col(feat).quantile(0.05),
            median=pl.col(feat).median(),
            mean=pl.col(feat).mean(),
            p95=pl.col(feat).quantile(0.95),
            min=pl.col(feat).min(),
            max=pl.col(feat).max(),
            cnt=pl.len(),
            null_cnt=pl.col(feat).null_count(),
            not_finite=pl.col(feat).is_finite().not_().sum(),
        ).row(0)

        # stats overlay
        df_stats = pl.DataFrame(
            {"names": ["p5", "median", "avg", "p95"], "stats": [p5, median, mean, p95]}
        )

        stats_base = alt.Chart(df_stats)
        stats_chart = stats_base.mark_rule(color="#f086ab").encode(
            x=alt.X("stats").title(""),
            tooltip=[
                alt.Tooltip("names:N", title="Stats"),
                alt.Tooltip("stats:Q", title="Value"),
            ],
        )

        chart = chart + stats_chart
        if show_bad_values:
            df_bad_values = pl.DataFrame(
                {
                    "names": [""],
                    "pcts": [(null_cnt + not_finite) / cnt],
                }
            )

            bad_values_chart = (
                alt.Chart(df_bad_values)
                .mark_bar(opacity=0.7)
                .encode(
                    x=alt.X("pcts:Q", scale=alt.Scale(domain=[0, 1]))
                    .axis(format=".0%")
                    .title("Null or Non-Finite %"),
                    y=alt.Y("names:N").title(""),
                    tooltip=[
                        alt.Tooltip("pcts:Q", title="Null or Non-Finite %"),
                    ],
                )
            )

            return alt.vconcat(chart, bad_values_chart)
        else:
            return chart

    else:  # over is not None
        if show_bad_values:
            df_bad = data.group_by(over).agg(
                pcts=(pl.col(feat).null_count() + pl.col(feat).is_finite().not_().sum()) / pl.len()
            )
            bad_values_chart = (
                alt.Chart(df_bad)
                .mark_bar(opacity=0.7)
                .encode(
                    x=alt.X("pcts:Q", scale=alt.Scale(domain=[0, 1]))
                    .axis(format=".0%")
                    .title("Null or Non-Finite %"),
                    y=alt.Y(f"{over}:N"),
                    tooltip=[
                        alt.Tooltip("pcts:Q", title="Null or Non-Finite %"),
                    ],
                )
            )
            return alt.vconcat(chart, bad_values_chart)
        else:
            return chart


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
            pl.len(),
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

    chart = alt.Chart(df_sampled).mark_point().encode(alt.X(x).scale(zero=False), alt.Y(target))
    return chart + chart.mark_line().encode(
        alt.X(x, title=x_title).scale(zero=False),
        alt.Y("y_pred"),
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
        return (
            alt.Chart(df_plot).mark_circle(size=60).encode(x="pc1", y="pc2", color=by, **kwargs)
        )  # .interactive()
    else:  # 3d
        raise NotImplementedError


def plot_roc_auc(
    *,
    actual: Iterable[int] | str | pl.Expr,
    pred: Iterable[float] | str | pl.Expr,
    df: pl.DataFrame | pl.LazyFrame | None = None,
    show_auc: bool = True,
    estimator_name: str = "",
    n_decimals: int = 4,
    auc_y_offset: int = 0,
    text_color: str = "black",
    **kwargs,
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
    n_decimals
        Round to n-th decimal digit if show_auc is True
    auc_y_offset
        Y offset for the roc auc value if show_auc is True. The more negative, the higher it gets.
    text_color
        Color for the model AUC text
    kwargs
        Other keyword arguments to Altair's mark_line
    """
    # expr_based = isinstance(actual, (str, pl.Expr)) and isinstance(pred, (str, pl.Expr)) and isinstance(df, (pl.DataFrame, pl.LazyFrame))
    if (
        isinstance(actual, (str, pl.Expr))
        and isinstance(pred, (str, pl.Expr))
        and isinstance(df, (pl.DataFrame, pl.LazyFrame))
    ):
        zero = pl.DataFrame(
            {
                "tpr": [0.0],
                "fpr": [0.0],
            },
            schema={
                "tpr": pl.Float64,
                "fpr": pl.Float64,
            },
        )

        tpr_fpr = (
            df.lazy()
            .select(tpr_fpr=query_tpr_fpr(actual, pred).reverse())
            .unnest("tpr_fpr")
            .select(
                "tpr",
                "fpr",
            )
            .collect()
        )

        df_plot = pl.concat([zero, tpr_fpr])

        base = alt.Chart(df_plot)
        chart = base.mark_rule(strokeDash=[4, 4]).encode(
            x="min(fpr)",
            x2="max(fpr)",
            y="min(tpr)",
            y2="max(tpr)",
        ) + base.mark_line(interpolate="step", **kwargs).encode(
            x=alt.X("fpr", title="False Positive Rate"),
            y=alt.Y("tpr", title="True Positive Rate"),
        )

        if show_auc:
            auc = tpr_fpr.select(integrate_trapz("tpr", "fpr")).item(0, 0)
            df_text = pl.DataFrame({"x": [0.97], "y": [0.03]})
            estimator = estimator_name.strip()
            auc_text = (
                f"AUC = {round(auc, n_decimals)}"
                if estimator == ""
                else f"{estimator} (AUC = {round(auc, n_decimals)})"
            )
            base_text = alt.Chart(df_text)
            text = base_text.mark_text(
                dy=auc_y_offset, color=text_color, fontWeight="bold", text=auc_text, align="right"
            ).encode(
                x=alt.X("x"),
                y=alt.Y("y"),
            )
            return chart + text
        else:
            return chart
    else:  # May fail. User should catch
        s1 = pl.Series("actual", values=actual, dtype=pl.UInt32)
        s2 = pl.Series("pred", values=pred)
        df_temp = pl.DataFrame(
            {
                "actual": s1,
                "pred": s2,
            }
        )
        return plot_roc_auc(
            df=df_temp,
            actual="actual",
            pred="pred",
            show_auc=show_auc,
            estimator_name=estimator_name,
            auc_y_offset=auc_y_offset,
            n_decimals=n_decimals,
        )
