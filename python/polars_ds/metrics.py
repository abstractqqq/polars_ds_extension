from __future__ import annotations
import polars as pl
from .type_alias import ROCAUCStrategy, str_to_expr, StrOrExpr
from polars.utils.udfs import _get_shared_lib_location
from ._utils import pl_plugin

_lib = _get_shared_lib_location(__file__)


# @pl.api.register_expr_namespace("metric")
# class MetricExt:

#     """
#     All the metrics/losses provided here is meant for model evaluation outside training,
#     e.g. for report generation, model performance monitoring, etc., not for actual use in ML models.
#     All metrics follow the convention by treating self as the actual column, and pred as the column
#     of predictions.

#     Polars Namespace: metric

#     Example: pl.col("a").metric.hubor_loss(pl.col("pred"), delta = 0.5)
#     """

#     def __init__(self, expr: pl.Expr):
#         self._expr: pl.Expr = expr

#     def max_error(self, pred: pl.Expr) -> pl.Expr:
#         """
#         Computes the max absolute error between actual and pred.
#         """
#         x = self._expr - pred
#         return pl.max_horizontal(x.max(), -x.min())

#     # def mean_gamma_deviance(self, pred: pl.Expr) -> pl.Expr:
#     #     """
#     #     Computes the mean gamma deviance between actual and pred.

#     #     Note that this will return NaNs when any value is < 0. This only makes sense when y_true
#     #     and y_pred as strictly positive.
#     #     """
#     #     x = self._expr / pred
#     #     return 2.0 * (x.log() + x - 1).mean()

#     def pinball_loss(self, pred: pl.Expr, tau: float = 0.5) -> pl.Expr:
#         """
#         This loss yields an estimator of the tau conditional quantile in quantile regression models.
#         This will treat self as y_true.

#         Parameters
#         ----------
#         pred
#             An expression represeting the column which is the prediction.
#         tau
#             A float in [0,1] represeting the conditional quantile level
#         """
#         return pl.max_horizontal(tau * (self._expr - pred), (tau - 1) * (self._expr - pred))


#     def kl_divergence(self, pred: pl.Expr) -> pl.Expr:
#         """
#         Computes the discrete KL Divergence.

#         Parameters
#         ----------
#         pred
#             An expression represeting the predicted probabilities for the classes

#         Reference
#         ---------
#         https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
#         """
#         return self._expr * (self._expr / pred).log()

# ----------------------------------------------------------------------------------


def query_r2(actual: StrOrExpr, pred: StrOrExpr) -> pl.Expr:
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


def query_adj_r2(actual: StrOrExpr, pred: StrOrExpr, p: int) -> pl.Expr:
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
    a = str_to_expr(actual)
    p = str_to_expr(pred)
    diff = a - p
    ss_res = diff.dot(diff)
    diff2 = a - a.mean()
    ss_tot = diff2.dot(diff2)
    df_res = a.count() - p
    df_tot = a.count() - 1
    return 1.0 - (ss_res / df_res) / (ss_tot / df_tot)


def query_log_cosh(actual: StrOrExpr, pred: StrOrExpr, normalize: bool = True) -> pl.Expr:
    """
    Computes log cosh of the the prediction error, which is a smooth variation of MAE (L1 loss).
    """
    a, p = str_to_expr(actual), str_to_expr(pred)
    if normalize:
        return (p - a).cosh().log().sum() / a.count()
    return (p - a).cosh().log().sum()


def query_hubor_loss(actual: StrOrExpr, pred: StrOrExpr, delta: float) -> pl.Expr:
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


def query_l2(actual: StrOrExpr, pred: StrOrExpr, normalize: bool = True) -> pl.Expr:
    """
    Returns squared L2 loss.

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


def query_l1(actual: StrOrExpr, pred: StrOrExpr, normalize: bool = True) -> pl.Expr:
    """
    Returns L1 loss.

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


def query_l_inf(actual: StrOrExpr, pred: StrOrExpr) -> pl.Expr:
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


def query_log_loss(actual: StrOrExpr, pred: StrOrExpr, normalize: bool = True) -> pl.Expr:
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
    a = str_to_expr(actual)
    p = str_to_expr(pred)
    out = a.dot(p.log()) + (1 - a).dot((1 - p).log())
    if normalize:
        return -(out / a.count())
    return -out


def query_mape(actual: StrOrExpr, pred: StrOrExpr, weighted: bool = False) -> pl.Expr:
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


def query_smape(actual: StrOrExpr, pred: StrOrExpr) -> pl.Expr:
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


def query_msle(actual: StrOrExpr, pred: StrOrExpr, normalize: bool = True) -> pl.Expr:
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
    actual: StrOrExpr,
    pred: StrOrExpr,
) -> pl.Expr:
    """
    Computes ROC AUC using self as actual and pred as predictions.

    Self must be binary and castable to type UInt32. If self is not all 0s and 1s or not binary,
    the result will not make sense, or some error may occur.

    Parameters
    ----------
    actual
        An expression represeting the actual
    pred
        An expression represeting the column with predicted probability.
    """
    return pl_plugin(
        lib=_lib,
        symbol="pl_roc_auc",
        args=[str_to_expr(actual).cast(pl.UInt32), str_to_expr(pred)],
        returns_scalar=True,
    )


def query_gini(actual: StrOrExpr, pred: StrOrExpr) -> pl.Expr:
    """
    Computes the Gini coefficient. This is 2 * AUC - 1.

    Self must be binary and castable to type UInt32. If self is not all 0s and 1s or not binary,
    the result will not make sense, or some error may occur.

    Parameters
    ----------
    actual
        An expression represeting the actual
    pred
        An expression represeting the column with predicted probability.
    """
    return query_roc_auc(actual, pred) * 2.0 - 1.0


def query_binary_metrics(actual: StrOrExpr, pred: StrOrExpr, threshold: float = 0.5) -> pl.Expr:
    """
    Computes the following binary classificaition metrics using self as actual and pred as predictions:
    precision, recall, f, average_precision and roc_auc. The return will be a struct with values
    having the names as given here.

    Self must be binary and castable to type UInt32. If self is not all 0s and 1s,
    the result will not make sense, or some error may occur.

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
        lib=_lib,
        symbol="pl_combo_b",
        args=[
            str_to_expr(actual).cast(pl.UInt32),
            str_to_expr(pred),
            pl.lit(threshold, dtype=pl.Float64),
        ],
        returns_scalar=True,
    )


def query_multi_roc_auc(
    actual: StrOrExpr,
    pred: StrOrExpr,
    n_classes: int,
    strategy: ROCAUCStrategy = "weighted",
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
    actual: StrOrExpr, pred: StrOrExpr, normalize: bool = True, dense: bool = True
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
