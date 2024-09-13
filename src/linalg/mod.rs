pub mod lstsq;

pub enum LinalgErrors {
    DimensionMismatch,
    NotContiguousArray,
    NotEnoughRows,
    MatNotLearnedYet,
    NotContiguousOrEmpty,
    Other(String),
}

impl LinalgErrors {
    pub fn to_string(self) -> String {
        match self {
            Self::DimensionMismatch => "Dimension mismatch.".to_string(),
            Self::NotContiguousArray => "Input array is not contiguous.".to_string(),
            Self::MatNotLearnedYet => "Matrix is not learned yet.".to_string(),
            Self::NotEnoughRows => "Not enough rows.".to_string(),
            Self::NotContiguousOrEmpty => "Input is not contiguous or is empty".to_string(),
            LinalgErrors::Other(s) => s,
        }
    }
}
