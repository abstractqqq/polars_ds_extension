from __future__ import annotations
import polars as pl
from .type_alias import ROCAUCStrategy
from polars.utils.udfs import _get_shared_lib_location

_lib = _get_shared_lib_location(__file__)


@pl.api.register_expr_namespace("metric")
class MetricExt:

    """
    All the metrics/losses provided here is meant for model evaluation outside training,
    e.g. for report generation, model performance monitoring, etc., not for actual use in ML models.
    All metrics follow the convention by treating self as the actual column, and pred as the column
    of predictions.

    Polars Namespace: metric

    Example: pl.col("a").metric.hubor_loss(pl.col("pred"), delta = 0.5)
    """

    def __init__(self, expr: pl.Expr):
        self._expr: pl.Expr = expr

    def max_error(self, pred: pl.Expr) -> pl.Expr:
        """
        Computes the max absolute error between actual and pred.
        """
        x = self._expr - pred
        return pl.max_horizontal(x.max(), -x.min())

    def mean_gamma_deviance(self, pred: pl.Expr) -> pl.Expr:
        """
        Computes the mean gamma deviance between actual and pred.

        Note that this will return NaNs when any value is < 0. This only makes sense when y_true
        and y_pred as strictly positive.
        """
        x = self._expr / pred
        return 2.0 * (x.log() + x - 1).mean()

    def hubor_loss(self, pred: pl.Expr, delta: float) -> pl.Expr:
        """
        Computes huber loss between this and the other expression. This assumes
        this expression is actual, and the input is predicted, although the order
        does not matter in this case.

        Parameters
        ----------
        pred
            An expression represeting the column with predicted probability.
        """
        temp = (self._expr - pred).abs()
        return (
            pl.when(temp <= delta).then(0.5 * temp.pow(2)).otherwise(delta * (temp - 0.5 * delta))
            / self._expr.count()
        )

    def l1_loss(self, pred: pl.Expr, normalize: bool = True) -> pl.Expr:
        """
        Computes L1 loss (absolute difference) between this and the other `pred` expression.

        Parameters
        ----------
        pred
            An expression represeting the column with predicted probability.
        normalize
            If true, divide the result by length of the series
        """
        temp = (self._expr - pred).abs().sum()
        if normalize:
            return temp / self._expr.count()
        return temp

    def l2_loss(self, pred: pl.Expr, normalize: bool = True) -> pl.Expr:
        """
        Computes L2 loss (normalized L2 distance) between this and the other `pred` expression. This
        is the norm without 1/p power.

        Parameters
        ----------
        pred
            An expression represeting the column with predicted probability.
        normalize
            If true, divide the result by length of the series
        """
        temp = self._expr - pred
        temp = temp.dot(temp)
        if normalize:
            return temp / self._expr.count()
        return temp

    def msle(self, pred: pl.Expr, normalize: bool = True) -> pl.Expr:
        """
        Computes the mean square log error between this and the other `pred` expression.

        Parameters
        ----------
        pred
            An expression represeting the column with predicted probability.
        normalize
            If true, divide the result by length of the series
        """
        diff = self._expr.log1p() - pred.log1p()
        out = diff.dot(diff)
        if normalize:
            return out / self._expr.count()
        return out

    def chebyshev_loss(self, pred: pl.Expr, normalize: bool = True) -> pl.Expr:
        """
        Alias for l_inf_loss.
        """
        return self.l_inf_dist(pred, normalize)

    def r2(self, pred: pl.Expr) -> pl.Expr:
        """
        Returns the coefficient of determineation for a regression model.

        Parameters
        ----------
        pred
            A Polars expression representing predictions
        """
        diff = self._expr - pred
        ss_res = diff.dot(diff)
        diff2 = self._expr - self._expr.mean()
        ss_tot = diff2.dot(diff2)
        return 1.0 - ss_res / ss_tot

    def adjusted_r2(self, pred: pl.Expr, p: int) -> pl.Expr:
        """
        Returns the adjusted r2 for a regression model.

        Parameters
        ----------
        pred
            A Polars expression representing predictions
        p
            The total number of explanatory variables in the model
        """
        diff = self._expr - pred
        ss_res = diff.dot(diff)
        diff2 = self._expr - self._expr.mean()
        ss_tot = diff2.dot(diff2)
        df_res = self._expr.count() - p
        df_tot = self._expr.count() - 1
        return 1.0 - (ss_res / df_res) / (ss_tot / df_tot)

    def l_inf_loss(self, pred: pl.Expr, normalize: bool = True) -> pl.Expr:
        """
        Computes L^infinity loss between this and the other `pred` expression

        Parameters
        ----------
        pred
            An expression represeting the column with predicted probability.
        normalize
            If true, divide the result by length of the series
        """
        temp = self._expr - pred
        out = pl.max_horizontal(temp.min().abs(), temp.max().abs())
        if normalize:
            return out / self._expr.count()
        return out

    def mape(self, pred: pl.Expr, weighted: bool = False) -> pl.Expr:
        """
        Computes mean absolute percentage error between self and the other `pred` expression.
        If weighted, it will compute the weighted version as defined here:

        https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        Parameters
        ----------
        pred
            An expression represeting the column with predicted probability.
        weighted
            If true, computes wMAPE in the wikipedia article
        """
        if weighted:
            return (self._expr - pred).abs().sum() / self._expr.abs().sum()
        else:
            return (1 - pred / self._expr).abs().mean()

    def smape(self, pred: pl.Expr) -> pl.Expr:
        """
        Computes symmetric mean absolute percentage error between self and other `pred` expression.
        The value is always between 0 and 1. This is the third version in the wikipedia without
        the 100 factor.

        https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error

        Parameters
        ----------
        pred
            A Polars expression representing predictions
        """
        numerator = (self._expr - pred).abs()
        denominator = 1.0 / (self._expr.abs() + pred.abs())
        return (1.0 / self._expr.count()) * numerator.dot(denominator)

    def log_loss(self, pred: pl.Expr, normalize: bool = True) -> pl.Expr:
        """
        Computes log loss, aka binary cross entropy loss, between self and other `pred` expression.

        Parameters
        ----------
        pred
            An expression represeting the column with predicted probability.
        normalize
            Whether to divide by N.
        """
        out = self._expr.dot(pred.log()) + (1 - self._expr).dot((1 - pred).log())
        if normalize:
            return -(out / self._expr.count())
        return -out

    def pinball_loss(self, pred: pl.Expr, tau: float = 0.5) -> pl.Expr:
        """
        This loss yields an estimator of the tau conditional quantile in quantile regression models.
        This will treat self as y_true.

        Parameters
        ----------
        pred
            An expression represeting the column which is the prediction.
        tau
            A float in [0,1] represeting the conditional quantile level
        """
        return pl.max_horizontal(tau * (self._expr - pred), (tau - 1) * (self._expr - pred))

    def bce(self, pred: pl.Expr, normalize: bool = True) -> pl.Expr:
        """
        Binary cross entropy. Alias for log_loss.
        """
        return self.log_loss(pred, normalize)

    def categorical_cross_entropy(
        self, pred: pl.Expr, normalize: bool = True, dense: bool = True
    ) -> pl.Expr:
        """
        Returns the categorical cross entropy. If you want to avoid numerical error due to log, please
        set pred = pred + epsilon.

        Parameters
        ----------
        pred
            An expression represeting the predicted probabilities for the classes
        normalize
            Whether to divide by N.
        dense
            If true, self has to be a dense vector (a single number for each row). If false, self has to be
            a column of lists with only one 1 and 0s otherwise.
        """
        if dense:
            y_prob = pred.list.get(self._expr)
        else:
            y_prob = pred.list.get(self._expr.list.arg_max())
        if normalize:
            return -y_prob.log().sum() / self._expr.count()
        return -y_prob.log().sum()

    def kl_divergence(self, pred: pl.Expr) -> pl.Expr:
        """
        Computes the discrete KL Divergence.

        Parameters
        ----------
        pred
            An expression represeting the predicted probabilities for the classes

        Reference
        ---------
        https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
        """
        return self._expr * (self._expr / pred).log()

    def log_cosh(self, pred: pl.Expr) -> pl.Expr:
        """
        Computes log cosh of the the prediction error (pred - self (y_true))
        """
        return (pred - self._expr).cosh().log()

    def roc_auc(self, pred: pl.Expr) -> pl.Expr:
        """
        Computes ROC AUC using self as actual and pred as predictions.

        Self must be binary and castable to type UInt32. If self is not all 0s and 1s or not binary,
        the result will not make sense, or some error may occur.

        Parameters
        ----------
        pred
            An expression represeting the column with predicted probability.
        """
        y = self._expr.cast(pl.UInt32)
        return y.register_plugin(
            lib=_lib,
            symbol="pl_roc_auc",
            args=[pred],
            is_elementwise=False,
            returns_scalar=True,
        )

    def multiclass_roc_auc(
        self, pred: pl.Expr, n_classes: int, strategy: ROCAUCStrategy = "weighted"
    ) -> pl.Expr:
        """
        Computes multiclass ROC AUC. Self (actuals) must be labels represented by integer values
        ranging in the range [0, n_classes), and pred must be a column of list[f64] with size `n_classes`.

        Parameters
        ----------
        pred
            The multilabel prediction column
        n_classes
            The number of classes
        strategy
            Either `macro` or `weighted`, which are defined the same as in Scikit-learn.
        """
        if strategy == "macro":
            actuals = [self._expr == i for i in range(n_classes)]
            preds = [pred.list.get(i) for i in range(n_classes)]
            return (
                pl.sum_horizontal(a.metric.roc_auc(p) for a, p in zip(actuals, preds)) / n_classes
            )
        elif strategy == "weighted":
            actuals = [self._expr == i for i in range(n_classes)]
            preds = [pred.list.get(i) for i in range(n_classes)]
            return (
                pl.sum_horizontal(a.sum() * a.metric.roc_auc(p) for a, p in zip(actuals, preds))
                / pl.len()
            )
        else:
            raise NotImplementedError

    def gini(self, pred: pl.Expr) -> pl.Expr:
        """
        Computes the Gini coefficient. This is 2 * AUC - 1.

        Self must be binary and castable to type UInt32. If self is not all 0s and 1s or not binary,
        the result will not make sense, or some error may occur.

        Parameters
        ----------
        pred
            An expression represeting the column with predicted probability.
        """
        return self.roc_auc(pred) * 2 - 1

    def binary_metrics_combo(self, pred: pl.Expr, threshold: float = 0.5) -> pl.Expr:
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
        pred
            An expression represeting the column with predicted probability.
        threshold
            The threshold used to compute precision, recall and f (f score).
        """
        y = self._expr.cast(pl.UInt32)
        return y.register_plugin(
            lib=_lib,
            symbol="pl_combo_b",
            args=[pred, pl.lit(threshold, dtype=pl.Float64)],
            is_elementwise=False,
            returns_scalar=True,
        )
