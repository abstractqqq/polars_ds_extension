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

    if over is not None and df is None:
        raise ValueError("Input `over` can only be used when df is not None.")

    if isinstance(feature, str):
        if df is None:
            raise ValueError("If `feature` is str, then df cannot be none.")
        feat = feature
        if over is None:
            data = df.lazy().select(pl.col(feat).cast(pl.Float64)).collect()
        else:
            data = df.lazy().select(pl.col(feat).cast(pl.Float64), over).collect()
    else:
        if over is None:
            data = pl.Series(name="feature", values=feature, dtype=pl.Float64).to_frame()
            feat = "feature"
        else:
            raise ValueError("If input `feature` is a Series, then `over` cannot be used.")

    # selection = alt.selection_point(fields=['species'], bind='legend')
    # .filter(pl.col(feat).is_not_null())
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


def plot_prob_calibration(
    *,
    target: Iterable[int],
    score: pl.Series | None = None,
    name: str | None = None,
    scores: List[pl.Series] | None = None,
    names: List[str] | None = None,
    n_bins: int = 10,
) -> alt.Chart:
    """
    Plots probability calibration of score(s) with respect to the binary target.

    Parameters
    ----------
    target
        The target binary varialbe
    score
        The probability score values
    name
        The name of the probability score values
    scores
        If score is None, and scores is a list of probability scores, this will
        generate a plot with all probability calibrations.
    names
        If scores is population, this must be a list of corresponding score names.
    n_bins
        N quantile bins for the score(s).
    """

    if score is not None:
        if name is None:
            raise ValueError("If `score` is not None, then `name` must not be none.")
        else:
            new_dict = {name: pl.Series(values=score)}

    else:  # score is None
        if (scores is None) or (names is None):
            raise ValueError("If `score` is None, then `scores` and `names` must be populated.")

        if hasattr(scores, "__len__") and (hasattr(names, "__len__")):
            if len(scores) != len(names):
                raise ValueError("Input `scores` and `names` must have the same length.")

            new_dict = {n: pl.Series(values=s) for n, s in zip(names, scores)}
        else:
            raise ValueError("Input `scores` and `names` must be iterables with a length.")

    target_series = pl.Series(name="__actual__", values=target)

    if any(len(s) != len(target_series) for s in new_dict.values()):
        raise ValueError("All input `score(s)` and `target` must have the same length.")

    new_dict["__actual__"] = target_series

    df = pl.from_dict(new_dict)
    perfect_line = pl.int_range(1, 100, step=5, eager=True) / 100
    df_line = pl.DataFrame(
        {"mean_predicted_prob": perfect_line, "fraction_of_positives": perfect_line}
    ).with_columns(score=pl.lit(" y=x", dtype=pl.String), __point__=pl.lit(False, dtype=pl.Boolean))

    df_socres = [
        df.select(s, "__actual__")
        .with_columns(
            pl.col(s).qcut(n_bins, labels=[str(i) for i in range(n_bins)]).alias("__qcuts__")
        )
        .group_by("__qcuts__")
        .agg(
            mean_predicted_prob=pl.col(s).mean().cast(pl.Float64),
            fraction_of_positives=pl.col("__actual__").mean().cast(pl.Float64),
        )
        .sort("__qcuts__")
        .select(
            "mean_predicted_prob",
            "fraction_of_positives",
            score=pl.lit(s, dtype=pl.String),
            __point__=pl.lit(False, dtype=pl.Boolean),
        )
        for s in new_dict.keys()
        if s != "__actual__"
    ]

    chart1 = (
        alt.Chart(df_line)
        .mark_line(point=False)
        .encode(
            x="mean_predicted_prob:Q",
            y="fraction_of_positives:Q",
            color="score:N",
            strokeDash=alt.value([5, 5]),
        )
    )
    selection = alt.selection_multi(fields=["score"], bind="legend")
    chart2 = (
        alt.Chart(pl.concat(df_socres))
        .mark_line(point=True)
        .encode(
            x="mean_predicted_prob:Q",
            y="fraction_of_positives:Q",
            color="score:N",
            tooltip=["mean_predicted_prob", "fraction_of_positives"],
            opacity=alt.condition(selection, alt.value(0.8), alt.value(0.1)),
        )
        .add_params(selection)
    )

    return chart1 + chart2


def plot_roc_auc(
    *,
    target: Iterable[int],
    pred: pl.Series | None = None,
    name: str | None = None,
    preds: List[pl.Series] | None = None,
    names: List[str] | None = None,
    show_auc: bool = True,
    estimator_name: str = "",
    n_decimals: int = 4,
    **kwargs,
) -> alt.Chart:
    """
    Paramters
    ---------
    target
        A column which has the actual binary target information
    pred
        The prediction probability variable
    name
        The name for the prediction
    preds
        The prediction probability variables
    names
        The names for the predictions
    show_auc
        Whether to show the AUC value or not
    n_decimals
        Round to n-th decimal digit if show_auc is True
    kwargs
        Other keyword arguments to Altair's mark_line
    """

    if pred is not None:
        if name is None:
            raise ValueError("If `pred` is not None, then `name` must not be none.")
        else:
            new_dict = {name: pl.Series(values=pred)}

    else:  # pred is None
        if (preds is None) or (names is None):
            raise ValueError("If `pred` is None, then `preds` and `names` must be populated.")

        if hasattr(preds, "__len__") and (hasattr(names, "__len__")):
            if len(preds) != len(names):
                raise ValueError("Input `preds` and `names` must have the same length.")

            new_dict = {n: pl.Series(values=s) for n, s in zip(names, preds)}
        else:
            raise ValueError("Input `preds` and `names` must be iterables with a length.")

    target_series = pl.Series(name="__actual__", values=target)

    if any(len(s) != len(target_series) for s in new_dict.values()):
        raise ValueError("All input `score(s)` and `target` must have the same length.")

    pred_names = list(new_dict.keys())
    new_dict["__actual__"] = target_series
    df_tmp = pl.from_dict(new_dict)
    dfs = []

    for p in pred_names:
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
            df_tmp.select(tpr_fpr=query_tpr_fpr("__actual__", p).reverse())
            .unnest("tpr_fpr")
            .select("tpr", "fpr")
        )

        text = p
        if show_auc:
            auc = tpr_fpr.select(integrate_trapz("tpr", "fpr")).item(0, 0)
            text += f" (AUC = {round(auc, n_decimals)})"

        dfs.append(pl.concat([zero, tpr_fpr]).with_columns(name=pl.lit(text, dtype=pl.String)))

    perfect_line = pl.int_range(1, 100, step=5, eager=True) / 100
    df_line = pl.DataFrame({"fpr": perfect_line, "tpr": perfect_line}).with_columns(
        name=pl.lit(" y=x", dtype=pl.String),
    )
    chart1 = (
        alt.Chart(df_line)
        .mark_line()
        .encode(
            x=alt.X("fpr", title="False Positive Rate"),
            y=alt.Y("tpr", title="True Positive Rate"),
            color="name:N",
            strokeDash=alt.value([5, 5]),
        )
    )
    selection = alt.selection_multi(fields=["name"], bind="legend")
    chart2 = (
        alt.Chart(pl.concat(dfs))
        .mark_line(interpolate="step", **kwargs)
        .encode(
            x=alt.X("fpr", title="False Positive Rate"),
            y=alt.Y("tpr", title="True Positive Rate"),
            color="name:N",
            opacity=alt.condition(selection, alt.value(0.8), alt.value(0.1)),
        )
        .add_params(selection)
    )

    return chart1 + chart2
