#![allow(non_snake_case)]

use num::Float;
use std::str::FromStr;
pub mod glm;
pub mod logistic;
pub mod lr;
pub mod mixed;
pub mod online_lr;

pub enum LinalgErrors {
    DimensionMismatch,
    NotContiguousArray,
    NotEnoughData,
    MatNotLearnedYet,
    NotContiguousOrEmpty,
    Other(String),
}

impl LinalgErrors {
    pub fn to_string(self) -> String {
        match self {
            Self::DimensionMismatch => "inputs have mismatched dimensions".to_string(),
            Self::NotContiguousArray => "input array is not contiguous".to_string(),
            Self::MatNotLearnedYet => "matrix is not learned yet".to_string(),
            Self::NotEnoughData => "number of samples must be greater than or equal to number of features".to_string(),
            Self::NotContiguousOrEmpty => "input is not contiguous or is empty".to_string(),
            LinalgErrors::Other(s) => s,
        }
    }
}

#[derive(PartialEq, Clone, Copy)]
pub enum NullPolicy<T: Float + FromStr> {
    RAISE,
    SKIP,
    SKIP_WINDOW, // `SKIP` in rolling. Skip, but doesn't drop data. A specialized algorithm will handle this.
    IGNORE,
    FILL(T),        // Fill all null data with T. If target is null, drop the rows.
    FILL_WINDOW(T), // `FILL`` rolling. Doesn't drop data. A specialized algorithm will handle this.
}

impl<T: Float + FromStr> TryFrom<&str> for NullPolicy<T> {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        if value.eq_ignore_ascii_case("raise") {
            Ok(Self::RAISE)
        } else if value.eq_ignore_ascii_case("skip") {
            Ok(Self::SKIP)
        } else if value.eq_ignore_ascii_case("zero") {
            Ok(Self::FILL(T::zero()))
        } else if value.eq_ignore_ascii_case("one") {
            Ok(Self::FILL(T::one()))
        } else if value.eq_ignore_ascii_case("ignore") {
            Ok(Self::IGNORE)
        } else if value.eq_ignore_ascii_case("skip_window") {
            Ok(Self::SKIP_WINDOW)
        } else {
            match value.parse::<T>() {
                Ok(x) => Ok(Self::FILL(x)),
                Err(_) => Err("invalid null_policy: expected 'raise', 'skip', 'zero', 'one', 'ignore', 'skip_window', or a numeric fill value".into()),
            }
        }
    }
}

impl<T: Float + FromStr> TryFrom<String> for NullPolicy<T> {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::try_from(value.as_str())
    }
}

impl<T: Float + FromStr> TryFrom<Box<str>> for NullPolicy<T> {
    type Error = String;

    fn try_from(value: Box<str>) -> Result<Self, Self::Error> {
        Self::try_from(&*value)
    }
}
