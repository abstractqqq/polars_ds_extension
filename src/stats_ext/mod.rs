mod ks;
mod normal_test;

use polars::prelude::*;

fn simple_stats_output(_: &[Field]) -> PolarsResult<Field> {
    let s = Field::new("statistic", DataType::Float64);
    let p = Field::new("pvalue", DataType::Float64);
    let v: Vec<Field> = vec![s, p];
    Ok(Field::new("", DataType::Struct(v)))
}

pub struct StatsResult {
    pub stats: f64,
    pub p: Option<f64>,
}

impl StatsResult {
    pub fn new(s: f64, p: Option<f64>) -> StatsResult {
        StatsResult { stats: s, p: p }
    }

    pub fn from_stats(s: f64) -> StatsResult {
        StatsResult { stats: s, p: None }
    }
}

pub enum Alternative {
    TwoSided,
    Less,
    Greater,
}

impl From<&str> for Alternative {
    fn from(s: &str) -> Alternative {
        match s {
            "two-sided" | "two" => Alternative::TwoSided,
            "less" => Alternative::Less,
            "greater" => Alternative::Greater,
            _ => Alternative::TwoSided,
        }
    }
}
