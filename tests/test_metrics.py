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

    # This is an edge case where we have only one value for predicted prob.
    # This is just guessing at random so 0.5 is the right output.
    # Technical reason:
    # (TPR, FPR becomes 1 value, which messes up the trapz calculation, and we need to fill
    # a 0 at the beginning to make sure the trapz calculation is always valid)
    df = pl.DataFrame({"a": [0, 1], "b": [0.5, 0.5]})
    result = df.select(pds.query_roc_auc(pl.col("a"), pl.col("b"))).item(0, 0)
    assert result == 0.5


def test_roc_auc_2():
    from sklearn.metrics import roc_auc_score

    # A test submitted by a user. PDS didn't have 0 padding for TPR and FPR before the user
    # submitted the issue and this test is added to check that. This behavior is consistent
    # with scipy's trapz calculation and therefore with sklearn's roc auc.
    df = pl.from_dict(
        {
            "ytrue": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "ypred": [1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1],
        }
    )

    roc_auc_pds = df.select(pds.query_roc_auc("ytrue", "ypred")).item(0, 0)

    roc_auc_sklearn = roc_auc_score(df["ytrue"].to_numpy(), df["ypred"].to_numpy())

    assert np.isclose(roc_auc_pds, roc_auc_sklearn)


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


def test_log_loss():
    import numpy as np
    from sklearn.metrics import log_loss

    df = pl.DataFrame(
        {
            "x": np.random.random(size=N_ROWS),
            "y": np.round(np.random.random(size=N_ROWS)).astype(int),
        }
    )

    res = df.select(pds.query_log_loss("y", "x")).item(0, 0)
    ans = log_loss(df["y"].to_numpy(), df["x"].to_numpy())
    assert np.isclose(res, ans, rtol=1e-10)


def test_dcg_score():
    from sklearn.metrics import dcg_score

    df = pds.frame(size=1000, index_name="id").with_columns(
        pl.col("id").mod(3).alias("example_id"),
        pds.random_int(0, 10).alias("relevance"),
        pds.random().alias("score"),
    )

    for tie in [True, False]:
        df_agg = (
            df.group_by("example_id")
            .agg(dcg_score=pds.query_dcg_score("relevance", "score", ignore_ties=tie))
            .sort("example_id")
        )

        sklearn_results = []
        for by, subdf in df.sort("example_id").partition_by("example_id", as_dict=True).items():
            rel = subdf["relevance"].to_numpy().reshape((1, -1))
            score = subdf["score"].to_numpy().reshape((1, -1))
            sklearn_results.append(dcg_score(rel, score, ignore_ties=tie))

        sklearn_results = np.array(sklearn_results)

        np.testing.assert_allclose(df_agg["dcg_score"].to_numpy(), sklearn_results, rtol=1e-5)

    for k in range(1, 5):
        df_agg = (
            df.group_by("example_id")
            .agg(dcg_score=pds.query_dcg_score("relevance", "score", k=k, ignore_ties=False))
            .sort("example_id")
        )

        sklearn_results = []
        for by, subdf in df.sort("example_id").partition_by("example_id", as_dict=True).items():
            rel = subdf["relevance"].to_numpy().reshape((1, -1))
            score = subdf["score"].to_numpy().reshape((1, -1))
            sklearn_results.append(dcg_score(rel, score, k=k, ignore_ties=False))

        sklearn_results = np.array(sklearn_results)

        np.testing.assert_allclose(df_agg["dcg_score"].to_numpy(), sklearn_results, rtol=1e-5)


def test_ndcg_score():
    from sklearn.metrics import ndcg_score

    df = pds.frame(size=1000, index_name="id").with_columns(
        pl.col("id").mod(3).alias("example_id"),
        pds.random_int(0, 10).alias("relevance"),
        pds.random().alias("score"),
    )

    for tie in [True, False]:
        df_agg = (
            df.group_by("example_id")
            .agg(ndcg_score=pds.query_ndcg_score("relevance", "score", ignore_ties=tie))
            .sort("example_id")
        )

        sklearn_results = []
        for by, subdf in df.sort("example_id").partition_by("example_id", as_dict=True).items():
            rel = subdf["relevance"].to_numpy().reshape((1, -1))
            score = subdf["score"].to_numpy().reshape((1, -1))
            sklearn_results.append(ndcg_score(rel, score, ignore_ties=tie))

        sklearn_results = np.array(sklearn_results)
        np.testing.assert_allclose(df_agg["ndcg_score"].to_numpy(), sklearn_results, rtol=1e-5)

    for k in range(1, 5):
        df_agg = (
            df.group_by("example_id")
            .agg(ndcg_score=pds.query_ndcg_score("relevance", "score", k=k, ignore_ties=False))
            .sort("example_id")
        )

        sklearn_results = []
        for by, subdf in df.sort("example_id").partition_by("example_id", as_dict=True).items():
            rel = subdf["relevance"].to_numpy().reshape((1, -1))
            score = subdf["score"].to_numpy().reshape((1, -1))
            sklearn_results.append(ndcg_score(rel, score, k=k, ignore_ties=False))

        sklearn_results = np.array(sklearn_results)
        np.testing.assert_allclose(df_agg["ndcg_score"].to_numpy(), sklearn_results, rtol=1e-5)


# ---------------------------------------------------------------------------
# PSI tests
# ---------------------------------------------------------------------------
# Wrapper discovery note:
#   pl_psi_w_bps is exposed as pds.psi_w_breakpoints(new, baseline, breakpoints).
#   It always returns a struct report (no scalar-only mode); the caller sums
#   psi_bin to get the scalar.  Struct fields (in order): "<=", "baseline_pct",
#   "actual_pct", "psi_bin".
#
# Smoothing / clipping:
#   Both baseline_pct and actual_pct are clipped to a minimum of 0.0001 before
#   the PSI per-bin computation.  This means identical distributions yield PSI
#   close to 0 but not exactly 0 unless all bins have equal counts.


def _reference_psi(new_arr, baseline_arr, breakpoints):
    """
    Independent Python PSI calculation that mirrors the Rust implementation.
    breakpoints: sorted list of finite floats (NOT including inf).
    Returns (per_bin_psi, scalar_psi).
    """
    bp = list(breakpoints) + [float("inf")]

    def bucket_counts(arr, bp):
        counts = np.zeros(len(bp), dtype=np.float64)
        for v in arr:
            idx = np.searchsorted(bp, v, side="left")
            # binary_search in Rust: Ok(j)->c[j], Err(k)->c[k].
            # searchsorted 'left' returns the insertion point k such that
            # bp[k-1] < v <= bp[k], i.e. the same as Rust's Err(k)/Ok(j).
            counts[idx] += 1
        return counts

    baseline_counts = bucket_counts(baseline_arr, bp)
    new_counts = bucket_counts(new_arr, bp)

    baseline_pct = np.maximum(baseline_counts / baseline_counts.sum(), 0.0001)
    actual_pct = np.maximum(new_counts / new_counts.sum(), 0.0001)
    per_bin = (baseline_pct - actual_pct) * np.log(baseline_pct / actual_pct)
    return per_bin, per_bin.sum()


def test_psi_w_breakpoints():
    """psi_w_breakpoints scalar matches independent Python reference."""
    rng = np.random.default_rng(42)
    baseline = rng.normal(loc=0.0, scale=1.0, size=1000)
    new = rng.normal(loc=0.5, scale=1.2, size=1000)

    # Use quartiles of the baseline as breakpoints (excluding inf).
    bps = list(np.quantile(baseline, [0.25, 0.50, 0.75]))

    df = pl.DataFrame({"new": new, "baseline": baseline})
    report = df.select(pds.psi_w_breakpoints("new", "baseline", breakpoints=bps)).unnest(
        "psi_report"
    )

    # Scalar PSI from report
    psi_scalar = report["psi_bin"].sum()

    # Must be finite and positive (different distributions)
    assert np.isfinite(psi_scalar), "PSI should be finite"
    assert psi_scalar > 0, "PSI should be > 0 for different distributions"

    # Compare against independent Python reference
    _, ref_psi = _reference_psi(new, baseline, bps)
    assert np.isclose(psi_scalar, ref_psi, atol=1e-10), (
        f"PSI mismatch: got {psi_scalar}, expected {ref_psi}"
    )


def test_psi_w_breakpoints_identical_distributions():
    """
    When new == baseline (exactly the same values), PSI should be very close
    to 0.  It may not be exactly 0 because both baseline_pct and actual_pct
    are clipped to 0.0001, but for large n all bins will have equal non-zero
    counts and the clip won't fire, so the result should be 0.0.
    """
    rng = np.random.default_rng(7)
    data = rng.normal(size=1000)
    bps = list(np.quantile(data, [0.25, 0.50, 0.75]))

    df = pl.DataFrame({"a": data, "b": data})
    report = df.select(pds.psi_w_breakpoints("a", "b", breakpoints=bps)).unnest("psi_report")
    psi_scalar = report["psi_bin"].sum()

    assert np.isclose(psi_scalar, 0.0, atol=1e-12), (
        f"PSI of identical distributions should be ~0, got {psi_scalar}"
    )


def test_psi_report_struct():
    """
    psi(..., return_report=True) returns a struct whose per-bin PSI sums to
    the scalar returned by psi(..., return_report=False).
    Also verifies that all documented field names are present.
    """
    rng = np.random.default_rng(99)
    baseline = rng.normal(loc=0.0, scale=1.0, size=1000)
    new = rng.normal(loc=0.3, scale=1.1, size=1000)

    df = pl.DataFrame({"new": new, "baseline": baseline})

    # Struct output path (return_report=True)
    report = df.select(pds.psi("new", "baseline", n_bins=10, return_report=True)).unnest(
        "psi_report"
    )

    # Verify expected field names exist (order-independent)
    expected_fields = {"<=", "baseline_pct", "actual_pct", "psi_bin"}
    actual_fields = set(report.columns)
    assert expected_fields <= actual_fields, (
        f"Missing fields: {expected_fields - actual_fields}"
    )

    # Struct should have n_bins rows (one per bin)
    assert report.shape[0] == 10, f"Expected 10 rows, got {report.shape[0]}"

    # Sum of per-bin PSI must match the scalar path (atol=1e-10)
    struct_psi = report["psi_bin"].sum()
    scalar_psi = df.select(pds.psi("new", "baseline", n_bins=10)).item(0, 0)
    assert np.isclose(struct_psi, scalar_psi, atol=1e-10), (
        f"Struct sum {struct_psi} != scalar {scalar_psi}"
    )
