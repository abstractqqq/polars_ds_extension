use ndarray::ArrayView1;
use polars::{prelude::*, lazy::dsl::count};
use pyo3_polars::derive::polars_expr;

fn tp_fp_frame(predicted: Series, actual:Series, as_ratio:bool) -> PolarsResult<LazyFrame> {


    let n = predicted.len() as u32;
    let df = df!(
        "threshold" => predicted.clone(),
        "actual" => actual.clone()
    )?;

    let positive_counts = actual.sum::<u32>().unwrap_or(0);
    if positive_counts == 0 {
        return Err(PolarsError::ComputeError(
            "No positives in actual.".into(),
        ))
    }

    let temp = df.lazy().group_by([col("threshold")]).agg([
        count().alias("cnt")
        , col("actual").sum().alias("pos_cnt_at_threshold")
    ])
    .sort("threshold", Default::default())
    .with_columns([
        (
            lit(n) - col("cnt").cumsum(false) + col("cnt")
        ).alias("predicted_positive")
      , (
            lit(positive_counts) - col("pos_cnt_at_threshold").cumsum(false)
        ).shift_and_fill(1, lit(positive_counts)).alias("tp")
    ]).select([
        col("threshold"),
        col("cnt"),
        // col("predicted_positive")
        col("pos_cnt_at_threshold"),
        col("tp"),
        (col("predicted_positive") - col("tp")).alias("fp"),
        (col("tp").cast(DataType::Float64)/col("predicted_positive").cast(DataType::Float64)).alias("precision")
    ]);

    if as_ratio {
        Ok(
            temp.select([
                col("threshold"),
                col("cnt"),
                // col("predicted_positive")
                col("pos_cnt_at_threshold"),
                (col("tp").cast(DataType::Float64)/col("tp").first().cast(DataType::Float64)).alias("tpr"),
                (col("fp").cast(DataType::Float64)/col("fp").first().cast(DataType::Float64)).alias("fpr"),
                col("precision")
            ])
        )
    } else {
        Ok(temp)
    }

}

#[polars_expr(output_type=Float64)]
fn pl_roc_auc(inputs: &[Series]) -> PolarsResult<Series> {

    // actual, when passed in, is always u32 (done in Python extension side)
    let actual = inputs[0].rechunk();
    let predicted = inputs[1].rechunk();
    
    if (actual.len() != predicted.len()) 
        | actual.is_empty() 
        | predicted.is_empty() 
        | (actual.null_count() + predicted.null_count() > 0)  
    {
        return Err(PolarsError::ComputeError(
            "Input columns must be the same length, cannot be empty, and shouldn't contain nulls.".into(),
        ))
    }

    if actual.n_unique().unwrap_or(0) != 2 {
        return Err(PolarsError::ComputeError(
            "Actual column must be binary without any nulls.".into(),
        ))
    }

    let mut frame = tp_fp_frame(predicted, actual, true)?
        .select([col("tpr"), col("fpr")])
        .collect()?;
    
    let tpr = frame.drop_in_place("tpr")?;
    let fpr = frame.drop_in_place("fpr")?;
    
    let y = tpr.f64()?.rechunk();
    let x = fpr.f64()?.rechunk();
    
    let y:ArrayView1<f64> = y.to_ndarray()?;
    let x:ArrayView1<f64> = x.to_ndarray()?;
    
    let out:f64 = -super::trapz::trapz(y,x);
    
    Ok(Series::from_iter([out]))

}