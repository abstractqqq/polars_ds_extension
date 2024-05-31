/// All things true positive, false positive related.
/// ROC AUC, Average Precision, precision, recall, etc. m
use ndarray::ArrayView1;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

const BINARY_CM_VALUES: [i32; 4] = [0, 1, 2, 3];

fn combo_output(_: &[Field]) -> PolarsResult<Field> {
    let roc_auc = Field::new("roc_auc", DataType::Float64);
    let precision = Field::new("precision", DataType::Float64);
    let recall = Field::new("recall", DataType::Float64);
    let f = Field::new("f", DataType::Float64);
    let avg_precision = Field::new("avg_precision", DataType::Float64);
    let v: Vec<Field> = vec![precision, recall, f, avg_precision, roc_auc];
    Ok(Field::new("", DataType::Struct(v)))
}

fn tp_fp_frame(predicted: &Series, actual: &Series, as_ratio: bool) -> PolarsResult<LazyFrame> {
    // Checking for data quality issues
    if (actual.len() != predicted.len())
        || actual.is_empty()
        || predicted.is_empty()
        || !predicted.dtype().is_numeric()
        || ((actual.null_count() + predicted.null_count()) > 0)
    {
        return Err(PolarsError::ComputeError(
            "ROC AUC: Input columns must be the same length, non-empty, numeric, and shouldn't contain nulls."
            .into(),
        ));
    }

    let positive_counts = actual.sum::<u32>().unwrap_or(0);
    if positive_counts == 0 {
        return Err(PolarsError::ComputeError(
            "No positives in actual, or actual cannot be turned into integers.".into(),
        ));
    }

    // Start computing
    let n = predicted.len() as u32;
    let df = df!(
        "threshold" => predicted,
        "actual" => actual
    )?;

    let temp = df
        .lazy()
        .group_by([col("threshold")])
        .agg([
            len().alias("cnt"),
            col("actual").sum().alias("pos_cnt_at_threshold"),
        ])
        .sort(["threshold"], Default::default())
        .with_columns([
            (lit(n) - col("cnt").cum_sum(false) + col("cnt")).alias("predicted_positive"),
            (lit(positive_counts) - col("pos_cnt_at_threshold").cum_sum(false))
                .shift_and_fill(1, positive_counts)
                .alias("tp"),
        ])
        .select([
            col("threshold"),
            col("tp"),
            (col("predicted_positive") - col("tp")).alias("fp"),
            (col("tp").cast(DataType::Float64) / col("predicted_positive").cast(DataType::Float64))
                .alias("precision"),
        ]);
    // col("cnt"),
    // col("predicted_positive")
    // col("pos_cnt_at_threshold"),
    if as_ratio {
        Ok(temp.select([
            col("threshold"),
            (col("tp").cast(DataType::Float64) / col("tp").first().cast(DataType::Float64))
                .alias("tpr"),
            (col("fp").cast(DataType::Float64) / col("fp").first().cast(DataType::Float64))
                .alias("fpr"),
            col("precision"),
        ]))
    } else {
        Ok(temp)
    }
}

#[polars_expr(output_type_func=combo_output)]
fn pl_combo_b(inputs: &[Series]) -> PolarsResult<Series> {
    // actual, when passed in, is always u32 (done in Python extension side)
    let actual = &inputs[0];
    let predicted = &inputs[1];
    // Threshold for precision and recall
    let threshold = inputs[2].f64()?;
    let threshold = threshold.get(0).unwrap_or(0.5);

    let mut frame = tp_fp_frame(predicted, actual, true)?
        .collect()?
        .agg_chunks();

    let tpr = frame.drop_in_place("tpr").unwrap();
    let fpr = frame.drop_in_place("fpr").unwrap();

    let precision = frame.drop_in_place("precision").unwrap();
    let precision = precision.f64()?;
    let precision = precision.cont_slice().unwrap();

    let probs = frame.drop_in_place("threshold").unwrap();
    let probs = probs.f64()?;
    let probs = probs.cont_slice().unwrap();
    let index = match probs.binary_search_by(|x| x.partial_cmp(&threshold).unwrap()) {
        Ok(i) => i,
        Err(i) => i,
    };

    // ROC AUC
    let y = tpr.f64().unwrap();
    let x = fpr.f64().unwrap();

    let y: ArrayView1<f64> = y.to_ndarray()?; // Zero copy
    let x: ArrayView1<f64> = x.to_ndarray()?; // Zero copy

    let roc_auc: f64 = -super::trapz::trapz(y, x);
    let roc_auc: Series = Series::from_vec("roc_auc", vec![roc_auc]);

    // Average Precision
    let ap: f64 = -1.0
        * y.iter()
            .zip(y.iter().skip(1))
            .zip(precision)
            .fold(0., |acc, ((y, y_next), p)| (y_next - y).mul_add(*p, acc));
    let ap: Series = Series::from_vec("average_precision", vec![ap]);

    // Precision & Recall & F
    let recall = y[index];
    let precision = precision[index];
    let f: f64 = 2.0 * (precision * recall) / (precision + recall);
    let recall: Series = Series::from_vec("recall", vec![recall]);
    let precision: Series = Series::from_vec("precision", vec![precision]);
    let f: Series = Series::from_vec("f", vec![f]);

    let out = StructChunked::new("metrics", &[precision, recall, f, ap, roc_auc])?;
    Ok(out.into_series())
}

fn bcm_output(_: &[Field]) -> PolarsResult<Field> {
    let tp = Field::new("tp", DataType::UInt32);
    let fp = Field::new("fp", DataType::UInt32);
    let tn = Field::new("tn", DataType::UInt32);
    let fn_ = Field::new("fn", DataType::UInt32);

    Ok(Field::new("", DataType::Struct(vec![tp, fp, tn, fn_])))
}

fn bcm(df: DataFrame) -> PolarsResult<DataFrame> {
    // bincount trick as described here:
    // https://stackoverflow.com/questions/59080843/faster-method-of-computing-confusion-matrix
    // 0 is tn, 1 is fp, 2 is fn, 3 is tp
    let df = df
        .lazy()
        .select([(lit(2) * col("y_true") + col("y_pred")).alias("y")])
        .collect()?;

    let value_counts = df["y"].value_counts(false, false)?;

    // It's possible that there are no True Positive, True Negatives, etc.
    // If our confusion matrix isn't complete, concat to the value counts dataframe
    // the missing ones with the count set to 0
    let value_counts = if value_counts.height() < 4 {
        let seen: Vec<i32> = value_counts["y"]
            .i32()?
            .iter()
            .map(|x| x.unwrap())
            .collect();
        let not_seen: Vec<i32> = BINARY_CM_VALUES
            .into_iter()
            .filter(|x| !seen.contains(x))
            .collect();

        let zeros = vec![0u32; not_seen.len()];

        value_counts
            .vstack(&df!("y" => not_seen, "count" => zeros)?)
            .unwrap()
    } else {
        value_counts
    };

    value_counts.sort(["y"], Default::default())
}

#[polars_expr(output_type_func=bcm_output)]
fn pl_binary_confusion_matrix(inputs: &[Series]) -> PolarsResult<Series> {
    let value_counts = bcm(df!("y_true" => &inputs[0], "y_pred" => &inputs[1])?)?;

    let s: ArrayView1<u32> = value_counts["count"].u32()?.to_ndarray()?;
    let tn = s[0];
    let fp = s[1];
    let fn_ = s[2];
    let tp = s[3];

    let out = StructChunked::new(
        "confusion_matrix",
        &[
            Series::from_vec("tn", vec![tn]),
            Series::from_vec("fp", vec![fp]),
            Series::from_vec("fn", vec![fn_]),
            Series::from_vec("tp", vec![tp]),
        ],
    )?;

    Ok(out.into_series())
}

fn nandiv(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        return f64::NAN;
    }

    a / b
}

fn bcm_output_full(_: &[Field]) -> PolarsResult<Field> {
    let tp = Field::new("tp", DataType::UInt32);
    let fp = Field::new("fp", DataType::UInt32);
    let tn = Field::new("tn", DataType::UInt32);
    let fn_ = Field::new("fn", DataType::UInt32);

    let fpr = Field::new("fpr", DataType::Float64);
    let fnr = Field::new("fnr", DataType::Float64);
    let tnr = Field::new("tnr", DataType::Float64);
    let prevalence = Field::new("prevalence", DataType::Float64);
    let prevalence_threshold = Field::new("prevalence_threshold", DataType::Float64);
    let tpr = Field::new("tpr", DataType::Float64);
    let informedness = Field::new("informedness", DataType::Float64);
    let precision = Field::new("precision", DataType::Float64);
    let false_omission_rate = Field::new("false_omission_rate", DataType::Float64);
    let plr = Field::new("plr", DataType::Float64);
    let nlr = Field::new("nlr", DataType::Float64);
    let acc = Field::new("acc", DataType::Float64);
    let balanced_accuracy = Field::new("balanced_accuracy", DataType::Float64);
    let f1 = Field::new("f1", DataType::Float64);
    let folkes_mallows_index = Field::new("folkes_mallows_index", DataType::Float64);
    let mcc = Field::new("mcc", DataType::Float64);
    let threat_score = Field::new("threat_score", DataType::Float64);
    let markedness = Field::new("markedness", DataType::Float64);
    let fdr = Field::new("fdr", DataType::Float64);
    let npv = Field::new("npv", DataType::Float64);
    let dor = Field::new("dor", DataType::Float64);

    Ok(Field::new(
        "",
        DataType::Struct(vec![
            tp,
            fp,
            tn,
            fn_,
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
        ]),
    ))
}

#[polars_expr(output_type_func=bcm_output_full)]
fn pl_binary_confusion_matrix_full(inputs: &[Series]) -> PolarsResult<Series> {
    let value_counts = bcm(df!("y_true" => &inputs[0], "y_pred" => &inputs[1])?)?;

    let s: ArrayView1<u32> = value_counts["count"].u32()?.to_ndarray()?;
    let tn = s[0] as f64;
    let fp = s[1] as f64;
    let fn_ = s[2] as f64;
    let tp = s[3] as f64;

    let p = tn + fn_;
    let n = fp + tn;
    let tpr = nandiv(tp, p);
    let fnr = 1.0 - tpr;
    let fpr = nandiv(fp, n);
    let tnr = 1.0 - fpr;
    let precision = nandiv(tp, tp + fp);
    let false_omission_rate = nandiv(fn_, fn_ + tn);
    let plr = nandiv(tpr, fpr);
    let nlr = nandiv(fnr, tnr);
    let npv = 1.0 - false_omission_rate;
    let fdr = 1.0 - precision;
    let prevalence = nandiv(p, p + n);
    let informedness = tpr + tnr - 1.0;
    let prevalence_threshold = nandiv((tpr * fpr).sqrt() - fpr, tpr - fpr);
    let markedness = precision - false_omission_rate;
    let dor = nandiv(plr, nlr);
    let balanced_accuracy = (tpr + tnr) / 2.0;
    let f1 = nandiv(2.0 * precision * tpr, precision + tpr);
    let folkes_mallows_index = (precision * tpr).sqrt();
    let mcc = (tpr * tnr * precision * npv).sqrt() - (fnr * fpr * false_omission_rate * fdr).sqrt();
    let acc = nandiv(tp + tn, p + n);
    let threat_score = nandiv(tp, tp + fn_ + fp);

    let out = StructChunked::new(
        "confusion_matrix",
        &[
            Series::from_vec("tn", vec![s[0]]),
            Series::from_vec("fp", vec![s[1]]),
            Series::from_vec("fn", vec![s[2]]),
            Series::from_vec("tp", vec![s[3]]),
            Series::from_vec("tpr", vec![tpr]),
            Series::from_vec("fpr", vec![fpr]),
            Series::from_vec("fnr", vec![fnr]),
            Series::from_vec("tnr", vec![tnr]),
            Series::from_vec("prevalence", vec![prevalence]),
            Series::from_vec("prevalence_threshold", vec![prevalence_threshold]),
            Series::from_vec("informedness", vec![informedness]),
            Series::from_vec("precision", vec![precision]),
            Series::from_vec("false_omission_rate", vec![false_omission_rate]),
            Series::from_vec("plr", vec![plr]),
            Series::from_vec("nlr", vec![nlr]),
            Series::from_vec("acc", vec![acc]),
            Series::from_vec("balanced_accuracy", vec![balanced_accuracy]),
            Series::from_vec("f1", vec![f1]),
            Series::from_vec("folkes_mallows_index", vec![folkes_mallows_index]),
            Series::from_vec("mcc", vec![mcc]),
            Series::from_vec("threat_score", vec![threat_score]),
            Series::from_vec("markedness", vec![markedness]),
            Series::from_vec("fdr", vec![fdr]),
            Series::from_vec("npv", vec![npv]),
            Series::from_vec("dor", vec![dor]),
        ],
    )?;

    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn pl_roc_auc(inputs: &[Series]) -> PolarsResult<Series> {
    // actual, when passed in, is always u32 (done in Python extension side)
    let actual = &inputs[0];
    let predicted = &inputs[1];

    let mut frame = tp_fp_frame(predicted, actual, true)?
        .select([col("tpr"), col("fpr")])
        .collect()?
        .agg_chunks();

    let tpr = frame.drop_in_place("tpr").unwrap();
    let fpr = frame.drop_in_place("fpr").unwrap();

    // Should be contiguous. No need to rechunk
    let y = tpr.f64().unwrap();
    let x = fpr.f64().unwrap();

    let y: ArrayView1<f64> = y.to_ndarray()?;
    let x: ArrayView1<f64> = x.to_ndarray()?;

    let out: f64 = -super::trapz::trapz(y, x);

    Ok(Series::from_iter([out]))
}
