"""Machine Learning / Time series Loss and Evaluation Metrics"""

from __future__ import annotations

import polars as pl

# Internal dependencies
from polars_ds._utils import pl_plugin, str_to_expr
from polars_ds.typing import MultiAUCStrategy

__all__ = [
    "query_r2",
    "query_adj_r2",
    "query_log_cosh",
    "query_hubor_loss",
    "query_l1",
    "query_l2",
    "query_l_inf",
    "query_log_loss",
    "query_mape",
    "query_smape",
    "query_mase",
    "query_msle",
    "query_mad",
    "query_roc_auc",
    "query_tpr_fpr",
    "query_binary_metrics",
    "query_multi_roc_auc",
    "query_cat_cross_entropy",
    "query_confusion_matrix",
    "query_fairness",
    "query_p_pct_score",
    "query_mcc",
]

# Confusion matrix based metrics should all be covered. If there is no
# specific function for it, we can find it in query_confusion_matrix.


def query_mad(x: str | pl.Expr, use_mean: bool = True) -> pl.Expr:
    """
    Computes the Mean/median Absolute Deviation.

    Parameters
    ----------
    x
        An expression represeting the actual
    use_mean
        If true, computes mean absolute deviation. If false, use median instead of mean.
    """
    xx = str_to_expr(x)
    if use_mean:
        return (xx - xx.mean()).abs().mean()
    else:
        return (xx - xx.median()).abs().median()


def query_r2(actual: str | pl.Expr, pred: str | pl.Expr) -> pl.Expr:
    """
    Returns the coefficient of determineation for a regression model.

    Parameters
    ----------
    actual
        An expression represeting the actual
    pred
        A Polars expression representing predictions
    """
    a = str_to_expr(actual)
    p = str_to_expr(pred)
    diff = a - p
    ss_res = diff.dot(diff)
    diff2 = a - a.mean()
    ss_tot = diff2.dot(diff2)
    return 1.0 - ss_res / ss_tot


def query_adj_r2(actual: str | pl.Expr, pred: str | pl.Expr, p: int) -> pl.Expr:
    """
    Returns the adjusted coefficient of determineation for a regression model.

    Parameters
    ----------
    actual
        An expression represeting the actual
    pred
        A Polars expression representing predictions
    p
        The number of explanatory variables
    """
    actual_expr = str_to_expr(actual)
    pred_expr = str_to_expr(pred)
    diff = actual_expr - pred_expr
    ss_res = diff.dot(diff)
    diff2 = actual_expr - actual_expr.mean()
    ss_tot = diff2.dot(diff2)
    df_res = actual_expr.len() - p
    df_tot = actual_expr.len() - 1
    return 1.0 - (ss_res / df_res) / (ss_tot / df_tot)


def query_log_cosh(actual: str | pl.Expr, pred: str | pl.Expr, normalize: bool = True) -> pl.Expr:
    """
    Computes log cosh of the the prediction error, which is a smooth variation of MAE (L1 loss).
    """
    a, p = str_to_expr(actual), str_to_expr(pred)
    if normalize:
        return (p - a).cosh().log().sum() / a.count()
    return (p - a).cosh().log().sum()


def query_hubor_loss(actual: str | pl.Expr, pred: str | pl.Expr, delta: float) -> pl.Expr:
    """
    Computes huber loss between this and the other expression. This assumes
    this expression is actual, and the input is predicted, although the order
    does not matter in this case.

    Parameters
    ----------
    pred
        An expression represeting the column with predicted probability.
    """
    a, p = str_to_expr(actual), str_to_expr(pred)
    temp = (a - p).abs()
    return (
        pl.when(temp <= delta).then(0.5 * temp.pow(2)).otherwise(delta * (temp - 0.5 * delta)).sum()
        / a.count()
    )


def query_l2(actual: str | pl.Expr, pred: str | pl.Expr, normalize: bool = True) -> pl.Expr:
    """
    Returns squared L2 loss, aka. mean squared error.

    Parameters
    ----------
    actual
        An expression represeting the actual
    pred
        A Polars expression representing predictions
    normalize
        Whether to divide by N.
    """
    a = str_to_expr(actual)
    p = str_to_expr(pred)
    diff = a - p
    if normalize:
        return diff.dot(diff) / a.count()
    return diff.dot(diff)


def query_l1(actual: str | pl.Expr, pred: str | pl.Expr, normalize: bool = True) -> pl.Expr:
    """
    Returns L1 loss, aka. mean absolute error.

    Parameters
    ----------
    actual
        An expression represeting the actual
    pred
        A Polars expression representing predictions
    normalize
        Whether to divide by N. Nulls won't be counted in N.
    """
    a = str_to_expr(actual)
    p = str_to_expr(pred)
    if normalize:
        return (a - p).abs().sum() / a.count()
    return (a - p).abs().sum()


def query_l_inf(actual: str | pl.Expr, pred: str | pl.Expr) -> pl.Expr:
    """
    Returns L Inf loss.

    Parameters
    ----------
    actual
        An expression represeting the actual
    pred
        A Polars expression representing predictions
    """
    a = str_to_expr(actual)
    p = str_to_expr(pred)
    return (a - p).abs().max()


def query_log_loss(actual: str | pl.Expr, pred: str | pl.Expr, normalize: bool = True) -> pl.Expr:
    """
    Computes log loss, aka binary cross entropy loss, between self and other `pred` expression.

    Parameters
    ----------
    actual
        An expression represeting the actual
    pred
        An expression represeting the column with predicted probability.
    normalize
        Whether to divide by N.
    """
    a = str_to_expr(actual).cast(pl.Float64)
    p = str_to_expr(pred).cast(pl.Float64)
    first = pl_plugin(
        args=[a, p],
        symbol="pl_xlogy",
        is_elementwise=True,
    )
    second = pl_plugin(
        args=[pl.lit(1.0, dtype=pl.Float64) - a, pl.lit(1.0, dtype=pl.Float64) - p],
        symbol="pl_xlogy",
        is_elementwise=True,
    )

    if normalize:
        return -(first + second).mean()
    return -(first + second).sum()


def query_mape(actual: str | pl.Expr, pred: str | pl.Expr, weighted: bool = False) -> pl.Expr:
    """
    Computes mean absolute percentage error between self and the other `pred` expression.
    If weighted, it will compute the weighted version as defined here:

    https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

    Parameters
    ----------
    actual
        An expression represeting the actual
    pred
        An expression represeting the column with predicted probability.
    weighted
        If true, computes wMAPE in the wikipedia article
    """
    a = str_to_expr(actual)
    p = str_to_expr(pred)
    if weighted:
        return (a - p).abs().sum() / a.abs().sum()
    else:
        return (1 - p / a).abs().mean()


def query_smape(actual: str | pl.Expr, pred: str | pl.Expr) -> pl.Expr:
    """
    Computes symmetric mean absolute percentage error between self and other `pred` expression.
    The value is always between 0 and 1. This is the third version in the wikipedia without
    the 100 factor.

    https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error

    Parameters
    ----------
    actual
        An expression represeting the actual
    pred
        A Polars expression representing predictions
    """
    a = str_to_expr(actual)
    p = str_to_expr(pred)
    numerator = (a - p).abs()
    denominator = a.abs() + p.abs()
    return (numerator / denominator).sum() / a.count()


def query_mase(
    actual: str | pl.Expr,
    pred: str | pl.Expr,
    train: str | pl.Expr | float,
    freq: int = 1,
    use_mean: bool = True,
) -> pl.Expr:
    """
    Computes the Mean/Median Absolute Scaled Error. This is the time series version in the reference article.

    Note: typically, train = pl.col('y').filter(pl.col('time') < T), and
    pred = pl.col('y_pred').filter(pl.col('time') >= T) and actual = pl.col('y').filter(pl.col('time') >= T)

    Parameters
    ----------
    actual
        An expression represeting the actual
    pred
        A Polars expression representing predictions
    train
        A polars exression representing training data values. If train is a float, it is treated
        as the precomputed naive one-step forecast loss on training as in the definition.
    freq
        Defaults to 1 which applies to non-seasonal data, and you may set it to m (>0)
        which indicates the length of the season. How frequent does the period repeat itself? Every `freq`
        records.
    use_mean
        If true, this will compute Mean Absolute Scaled Error. If false, this uses median instead of mean.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Mean_absolute_scaled_error
    """
    if freq < 1:
        raise ValueError("Input `freq` must be >= 1.")

    a: pl.Expr = str_to_expr(actual)
    p: pl.Expr = str_to_expr(pred)

    if isinstance(train, float):
        if use_mean:
            numerator = (a - p).abs().mean()
        else:
            numerator = (a - p).abs().median()

        return numerator / pl.lit(train)

    else:
        train_expr = str_to_expr(train)
        if use_mean:
            numerator = (a - p).abs().mean()
            denom = train_expr.diff(n=freq).abs().mean()
        else:
            numerator = (a - p).abs().median()
            denom = train_expr.diff(n=freq).abs().median()

        return numerator / denom


def query_msle(actual: str | pl.Expr, pred: str | pl.Expr, normalize: bool = True) -> pl.Expr:
    """
    Computes the mean square log error between this and the other `pred` expression.

    Parameters
    ----------
    pred
        An expression represeting the column with predicted probability.
    normalize
        If true, divide the result by length of the series
    """
    a = str_to_expr(actual)
    p = str_to_expr(pred)
    diff = a.log1p() - p.log1p()
    out = diff.dot(diff)
    if normalize:
        return out / a.count()
    return out


def query_roc_auc(
    actual: str | pl.Expr,
    pred: str | pl.Expr,
) -> pl.Expr:
    """
    Computes ROC AUC using self as actual and pred as predictions.

    Self must be binary and castable to type UInt32. If self is not all 0s and 1s or not binary,
    the result will not make sense, or some error may occur. If no positive class exist in data,
    NaN will be returned.

    Parameters
    ----------
    actual
        An expression represeting the actual. Must be castable to UInt32.
    pred
        An expression represeting the column with predicted probability.
    """
    return pl_plugin(
        symbol="pl_roc_auc",
        args=[str_to_expr(actual).cast(pl.UInt32), str_to_expr(pred)],
        returns_scalar=True,
    )


def query_tpr_fpr(
    actual: str | pl.Expr,
    pred: str | pl.Expr,
) -> pl.Expr:
    """
    Returns the TPR and FPR for all thresholds. This is useful when you want to study the thresholds
    or when you want to plot roc auc curve.

    Parameters
    ----------
    actual
        An expression represeting the actual. Must be castable to UInt32.
    pred
        An expression represeting the column with predicted probability.
    """
    return pl_plugin(
        symbol="pl_tpr_fpr",
        args=[str_to_expr(actual).cast(pl.UInt32), str_to_expr(pred)],
    )


def query_gini(actual: str | pl.Expr, pred: str | pl.Expr) -> pl.Expr:
    """
    Computes the Gini coefficient. This is 2 * AUC - 1.

    Self must be binary and castable to type UInt32. If self is not all 0s and 1s or not binary,
    the result will not make sense, or some error may occur. If no positive class exist in data,
    NaN will be returned.

    Parameters
    ----------
    actual
        An expression represeting the actual
    pred
        An expression represeting the column with predicted probability.
    """
    return query_roc_auc(actual, pred) * 2.0 - 1.0


def query_confusion_matrix(
    actual: str | pl.Expr,
    pred: str | pl.Expr,
    threshold: float = 0.5,
    all_metrics: bool = False,
) -> pl.Expr:
    """
    Computes the binary confusion matrix given the true labels (`actual`) and
    the predicted labels (computed from `pred`, a column of predicted scores and
    `threshold`). When a divide by zero is encountered, NaN is returned.

    Parameters
    ----------
    actual : str | pl.Expr
        An expression representing the actual labels. Must be castable to boolean
    pred : str | pl.Expr
        An expression representing the column with predicted probability
    threshold : float, optional
        The threshold used to compute the predicted labels, by default 0.5
    all_metrics : bool, optional
        If True, compute all 25 possible confusion matrix statistics instead of
        just True Positive, False Positive, True Negative, False Negative,
        by default False

    Returns
    -------
    pl.Expr
        A struct of confusion matrix metrics

    Examples
    --------
    Limited to just the basic confusion matrix

    >>> df = pl.DataFrame({"actual": [1, 0, 1], "pred": [0.4, 0.6, 0.9]})
    >>> df.select(pds.query_confusion_matrix("actual", "pred").alias("metrics")).unnest("metrics")
    shape: (1, 4)
    ┌─────┬─────┬─────┬─────┐
    │ tn  ┆ fp  ┆ fn  ┆ tp  │
    │ --- ┆ --- ┆ --- ┆ --- │
    │ u32 ┆ u32 ┆ u32 ┆ u32 │
    ╞═════╪═════╪═════╪═════╡
    │ 0   ┆ 1   ┆ 1   ┆ 1   │
    └─────┴─────┴─────┴─────┘

    With `all_metrics` set to True

    >>> df.select(
    ...     pds.query_confusion_matrix("actual", "pred", all_metrics=True).alias("metrics")
    ... ).unnest("metrics")
    shape: (1, 25)
    ┌─────┬─────┬─────┬─────┬───┬────────────┬─────┬─────┬─────┐
    │ tn  ┆ fp  ┆ fn  ┆ tp  ┆ … ┆ markedness ┆ fdr ┆ npv ┆ dor │
    │ --- ┆ --- ┆ --- ┆ --- ┆   ┆ ---        ┆ --- ┆ --- ┆ --- │
    │ u32 ┆ u32 ┆ u32 ┆ u32 ┆   ┆ f64        ┆ f64 ┆ f64 ┆ f64 │
    ╞═════╪═════╪═════╪═════╪═══╪════════════╪═════╪═════╪═════╡
    │ 0   ┆ 1   ┆ 1   ┆ 1   ┆ … ┆ -0.5       ┆ 0.5 ┆ 0.0 ┆ NaN │
    └─────┴─────┴─────┴─────┴───┴────────────┴─────┴─────┴─────┘
    """
    # Cast to bool first to check the label is in correct format. Then back to u32.
    act = str_to_expr(actual).cast(pl.Boolean).cast(pl.UInt32)
    p = str_to_expr(pred).gt(threshold).cast(pl.UInt32)
    res = pl_plugin(
        symbol="pl_binary_confusion_matrix",
        args=[(2 * act) + p],  # See Rust code for bincount trick
        returns_scalar=True,
    )
    if all_metrics:
        return res
    else:
        return pl.struct(
            res.struct.field("tn"),
            res.struct.field("fp"),
            res.struct.field("fn"),
            res.struct.field("tp"),
        )


def query_binary_metrics(
    actual: str | pl.Expr, pred: str | pl.Expr, threshold: float = 0.5
) -> pl.Expr:
    """
    Computes the following binary classificaition metrics using self as actual and pred as predictions:
    precision, recall, f, average_precision and roc_auc. The return will be a struct with values
    having the names as given here.

    Self must be binary and castable to type UInt32. If self is not all 0s and 1s,
    the result will not make sense, or some error may occur. If there is no positive class in data,
    NaN or other numerical error may occur.

    Average precision is computed using Sum (R_n - R_n-1)*P_n-1, which is not the textbook definition,
    but is consistent with Scikit-learn. For more information, see
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html

    Parameters
    ----------
    actual
        An expression represeting the actual
    pred
        An expression represeting the column with predicted probability.
    threshold
        The threshold used to compute precision, recall and f (f score).
    """
    return pl_plugin(
        symbol="pl_combo_b",
        args=[
            str_to_expr(actual).cast(pl.UInt32),
            str_to_expr(pred),
            pl.lit(threshold, dtype=pl.Float64),
        ],
        returns_scalar=True,
    )


def query_multi_roc_auc(
    actual: str | pl.Expr,
    pred: str | pl.Expr,
    n_classes: int,
    strategy: MultiAUCStrategy = "weighted",
) -> pl.Expr:
    """
    Computes multiclass ROC AUC. Self (actuals) must be labels represented by integer values
    ranging in the range [0, n_classes), and pred must be a column of list[f64] with size `n_classes`.

    Parameters
    ----------
    actual
        An expression represeting the actual
    pred
        The multilabel prediction column
    n_classes
        The number of classes
    strategy
        Either `macro` or `weighted`, which are defined the same as in Scikit-learn.
    """
    a = str_to_expr(actual)
    p = str_to_expr(pred)
    if strategy == "macro":
        actuals = [a == i for i in range(n_classes)]
        preds = [p.list.get(i) for i in range(n_classes)]
        return pl.sum_horizontal(query_roc_auc(a, p) for a, p in zip(actuals, preds)) / n_classes
    elif strategy == "weighted":
        actuals = [a == i for i in range(n_classes)]
        preds = [p.list.get(i) for i in range(n_classes)]
        return (
            pl.sum_horizontal(a.sum() * query_roc_auc(a, p) for a, p in zip(actuals, preds))
            / pl.len()
        )
    else:
        raise NotImplementedError


def query_cat_cross_entropy(
    actual: str | pl.Expr, pred: str | pl.Expr, normalize: bool = True, dense: bool = True
) -> pl.Expr:
    """
    Returns the categorical cross entropy. If you want to avoid numerical error due to log, please
    set pred = pred + epsilon.

    Parameters
    ----------
    actual
        An expression represeting the actual
    pred
        An expression represeting the predicted probabilities for the classes. Must of be List/arr[f64] type.
    normalize
        Whether to divide by N.
    dense
        If true, actual has to be a dense vector (a single number for each row, starting from 0). If false, it has
        to be a column of lists/arrs with only one 1 and 0s otherwise.
    """
    a = str_to_expr(actual)
    p = str_to_expr(pred)
    if dense:
        y_prob = p.list.get(a)
    else:
        y_prob = p.list.get(a.list.arg_max())
    if normalize:
        return -y_prob.log().sum() / a.count()
    return -y_prob.log().sum()


def query_mcc(y_true: str | pl.Expr, y_pred: str | pl.Expr) -> pl.Expr:
    """
    Returns the Matthews correlation coefficient (phi coefficient). The inputs must be 0s and 1s
    and castable to u32. If not, the result may not be correct. See query_confusion_matrix for querying
    all the confusion metrics at the same time.

    Parameters
    ----------
    y_true
        The true labels. Must be 0s and 1s.
    y_pred
        The predicted labels. Must be 0s and 1s. E.g. This could be say (y_prob > 0.5).cast(pl.UInt32)

    Reference
    ---------
    https://en.wikipedia.org/wiki/Phi_coefficient
    """

    y = str_to_expr(y_true)
    x = str_to_expr(y_pred)
    combined = (2 * y + x).cast(pl.UInt32)

    return pl_plugin(
        symbol="pl_mcc",
        args=[combined],
        returns_scalar=True,
    )


def query_fairness(pred: str | pl.Expr, sensitive_cond: pl.Expr) -> pl.Expr:
    """
    A simple fairness metric for regression output. Computes the absolute difference between
    the average of the `pred` values on when the `sensitive_cond` is true vs the
    avg of the values when `sensitive_cond` is false.

    The lower this value is, the more fair is the model on the sensitive condition.

    Parameters
    ----------
    pred
        The predictions
    sensitive_cond
        A boolean expression representing the sensitive condition
    """
    p = str_to_expr(pred)
    return (p.filter(sensitive_cond).mean() - p.filter(~sensitive_cond).mean()).abs()


def query_p_pct_score(pred: str | pl.Expr, sensitive_cond: pl.Expr) -> pl.Expr:
    """
    Computes the 'p-percent score', which measures the fairness of a classification
    model on a sensitive_cond. Let z = the sensitive_cond, then:

    p-percent score = min(P(y = 1 | z = 1) / P(y = 1 | z = 0), P(y = 1 | z = 0) / P(y = 1 | z = 1))

    Parameters
    ----------
    pred
        The predictions. Must be 0s and 1s.
    sensitive_cond
        A boolean expression representing the sensitive condition
    """
    p = str_to_expr(pred)
    p_y1_z1 = p.filter(
        sensitive_cond
    ).mean()  # since p is 0s and 1s, this is equal to P(pred = 1 | sensitive_cond)
    p_y1_z0 = p.filter(~sensitive_cond).mean()
    ratio = p_y1_z1 / p_y1_z0
    return pl.min_horizontal(ratio, 1 / ratio)
