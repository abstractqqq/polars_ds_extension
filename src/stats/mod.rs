mod chi2;
mod fstats;
mod ks;
mod normal_test;
mod sample;
mod t_test;

use polars::prelude::*;

pub fn simple_stats_output(_: &[Field]) -> PolarsResult<Field> {
    let s = Field::new("statistic", DataType::Float64);
    let p = Field::new("pvalue", DataType::Float64);
    let v: Vec<Field> = vec![s, p];
    Ok(Field::new("", DataType::Struct(v)))
}

struct StatsResult {
    pub statistic: f64,
    pub p: Option<f64>,
}

impl StatsResult {
    pub fn new(s: f64, p: f64) -> StatsResult {
        StatsResult {
            statistic: s,
            p: Some(p),
        }
    }

    pub fn from_stats(s: f64) -> StatsResult {
        StatsResult {
            statistic: s,
            p: None,
        }
    }

    // pub fn unwrap_p_or(&self, default: f64) -> f64 {
    //     self.p.unwrap_or(default)
    // }
}

pub enum Alternative {
    TwoSided,
    Less,
    Greater,
}

impl From<&str> for Alternative {
    fn from(s: &str) -> Alternative {
        match s.to_lowercase().as_str() {
            "two-sided" | "two" => Alternative::TwoSided,
            "less" => Alternative::Less,
            "greater" => Alternative::Greater,
            _ => Alternative::TwoSided,
        }
    }
}
