pub mod lstsq;

pub enum LinalgErrors {
    DimensionMismatch,
    NotContiguousArray,
    NotEnoughRows,
    MatNotLearnedYet,
    Other(String),
}

impl LinalgErrors {
    pub fn to_string(self) -> String {
        match self {
            LinalgErrors::DimensionMismatch => "Dimension mismatch.".to_string(),
            LinalgErrors::NotContiguousArray => "Input array is not contiguous.".to_string(),
            LinalgErrors::MatNotLearnedYet => "Matrix is not learned yet.".to_string(),
            LinalgErrors::NotEnoughRows => "Not enough rows.".to_string(),
            LinalgErrors::Other(s) => s,
        }
    }
}
