from __future__ import annotations

import numpy as np
import polars as pl
import polars_ds as pds
import pytest
import sklearn.metrics

SEED = 208
N_ROWS = 1_000

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
    p = tn + fn_
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
