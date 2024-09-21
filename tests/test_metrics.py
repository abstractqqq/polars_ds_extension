from __future__ import annotations

import numpy as np
import polars as pl
import polars_ds as pds
import pytest
import sklearn.metrics

SEED = 208
N_ROWS = 2_000

np.random.seed(SEED)


Y_TRUE = np.random.choice([True, False], N_ROWS)
Y_SCORE = np.random.rand(N_ROWS)
Y_PRED = Y_SCORE > 0.5
Y_TRUE_SC = np.full(N_ROWS, True)
Y_SCORE_SC = np.ones(N_ROWS)
Y_PRED_SC = Y_SCORE_SC > 0.5

# The point of the combos is to test our code under a matrix of different scenarios:
# any combination of all classes present in y_true, only one class present in y_true,
# a range of scores present in y_score, only one score present in y_score, all classes
# present in y_pred, and only one class present in y_pred.

TRUE_PRED_COMBOS = [
    (Y_TRUE, Y_PRED),
    (Y_TRUE_SC, Y_PRED),
    (Y_TRUE, Y_PRED_SC),
    (Y_TRUE_SC, Y_PRED_SC),
]

TRUE_SCORE_COMBOS = [
    (Y_TRUE, Y_SCORE),
    (Y_TRUE, Y_SCORE_SC),
    (Y_TRUE_SC, Y_SCORE),
    (Y_TRUE_SC, Y_SCORE_SC),
]


def nandiv(a: np.float64, b: np.float64) -> np.float64:
    if b == 0:
        return np.nan

    return a / b


def reference_confusion_matrix(y_true, y_pred):
    tn, fp, fn_, tp = sklearn.metrics.confusion_matrix(
        y_true, y_pred, labels=[False, True]
    ).ravel()  # Setting labels will ensure that the matrix is zero-filled

    # Where possible, test against sklearn. Otherwise implement it by hand.
    p = tp + fn_
    n = fp + tn
    tpr = nandiv(tp, p)
    fnr = 1.0 - tpr
    fpr = nandiv(fp, n)
    tnr = 1.0 - fpr
    precision = sklearn.metrics.precision_score(
        y_true, y_pred, labels=[False, True], zero_division=np.nan
    )
    false_omission_rate = nandiv(fn_, fn_ + tn)
    plr = nandiv(tpr, fpr)
    nlr = nandiv(fnr, tnr)
    npv = 1.0 - false_omission_rate
    fdr = 1.0 - precision
    prevalence = nandiv(p, p + n)
    informedness = tpr + tnr - 1.0
    prevalence_threshold = nandiv(np.sqrt(tpr * fpr) - fpr, tpr - fpr)
    markedness = precision - false_omission_rate
    dor = nandiv(plr, nlr)
    balanced_accuracy = sklearn.metrics.balanced_accuracy_score(y_true, y_pred)
    f1 = sklearn.metrics.f1_score(y_true, y_pred, zero_division=np.nan)
    folkes_mallows_index = sklearn.metrics.fowlkes_mallows_score(y_true, y_pred)
    mcc = sklearn.metrics.matthews_corrcoef(y_true, y_pred)
    acc = nandiv(tp + tn, p + n)
    threat_score = nandiv(tp, tp + fn_ + fp)

    return (
        tn,
        fp,
        fn_,
        tp,
        tpr,
        fpr,
        fnr,
        tnr,
        prevalence,
        prevalence_threshold,
        informedness,
        precision,
        false_omission_rate,
        plr,
        nlr,
        acc,
        balanced_accuracy,
        f1,
        folkes_mallows_index,
        mcc,
        threat_score,
        markedness,
        fdr,
        npv,
        dor,
    )


@pytest.mark.parametrize("y_true,y_score", TRUE_PRED_COMBOS)
def test_confusion_matrix(y_true, y_score):
    ref = reference_confusion_matrix(y_true, y_score > 0.5)
    res = [
        x.to_list()[0]
        for x in pl.DataFrame({"y_true": y_true, "y_score": y_score})
        .select(pds.query_confusion_matrix("y_true", "y_score", all_metrics=True).alias("metrics"))
        .unnest("metrics")
        .iter_columns()
    ]

    pytest.approx(res) == ref


def test_roc_auc():
    from sklearn.metrics import roc_auc_score

    df = pds.frame(size=2000).select(
        pds.random(0.0, 1.0).alias("predictions"),
        pds.random(0.0, 1.0).round().cast(pl.Int32).alias("target"),
        pl.lit(0).alias("zero_target"),
    )

    roc_auc = df.select(pds.query_roc_auc("target", "predictions")).item(0, 0)

    answer = roc_auc_score(df["target"].to_numpy(), df["predictions"].to_numpy())

    assert np.isclose(roc_auc, answer)

    # When all classes are 0, roc_auc returns NaN
    nan_roc = df.select(pds.query_roc_auc("zero_target", "predictions")).item(0, 0)

    assert np.isnan(nan_roc)


def test_multiclass_roc_auc():
    from sklearn.metrics import roc_auc_score

    def roc_auc_random_data(size: int = N_ROWS) -> pl.DataFrame:
        df = pds.frame(size=N_ROWS, index_name="id").with_columns(
            pl.col("id").cast(pl.UInt64),
            pds.random().alias("val1"),
            pds.random().alias("val2"),
            pds.random().alias("val3"),
            pl.col("id").mod(3).alias("actuals"),
        )
        # Need to normalize to make sure this is valid ROC AUC data
        return (
            df.lazy()
            .with_columns(
                pl.concat_list(
                    pl.col("val1")
                    / pl.sum_horizontal(pl.col("val1"), pl.col("val2"), pl.col("val3")),
                    pl.col("val2")
                    / pl.sum_horizontal(pl.col("val1"), pl.col("val2"), pl.col("val3")),
                    pl.col("val3")
                    / pl.sum_horizontal(pl.col("val1"), pl.col("val2"), pl.col("val3")),
                ).alias("pred")
            )
            .select(
                pl.col("actuals"),
                pl.col("pred"),
            )
            .collect()
        )

    df = roc_auc_random_data()
    y_pred = np.stack(df["pred"].to_numpy())
    y_true = df["actuals"]

    macro = df.select(pds.query_multi_roc_auc("actuals", "pred", 3, "macro")).item(0, 0)
    macro_sklearn = roc_auc_score(y_true, y_pred, average="macro", multi_class="ovr")

    weighted = df.select(pds.query_multi_roc_auc("actuals", "pred", 3, "weighted")).item(0, 0)
    weighted_sklearn = roc_auc_score(y_true, y_pred, average="weighted", multi_class="ovr")

    assert np.isclose(macro, macro_sklearn, rtol=1e-10, atol=1e-12)
    assert np.isclose(weighted, weighted_sklearn, rtol=1e-10, atol=1e-10)


def test_precision_recall_roc_auc():
    import numpy as np
    from sklearn.metrics import roc_auc_score

    df = pl.DataFrame(
        {
            "a": np.random.random(size=N_ROWS),
            "b": np.random.random(size=N_ROWS),
            "y": np.round(np.random.random(size=N_ROWS)).astype(int),
        }
    )
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        res = df.select(
            pds.query_binary_metrics("y", "a", threshold=threshold).alias("metrics")
        ).unnest("metrics")
        precision_res = res.get_column("precision")[0]
        recall_res = res.get_column("recall")[0]
        roc_auc_res = res.get_column("roc_auc")[0]

        # precision, recall by hand
        predicted_prob = np.array(df["a"])
        predicted = predicted_prob >= threshold  # boolean
        actual = np.array(df["y"])  # .to_numpy()
        precision = actual[predicted].sum() / np.sum(predicted)
        recall = ((actual == 1) & (predicted == 1)).sum() / (actual.sum())

        assert np.isclose(precision, precision_res)
        assert np.isclose(recall, recall_res)
        assert np.isclose(roc_auc_score(actual, predicted_prob), roc_auc_res)
