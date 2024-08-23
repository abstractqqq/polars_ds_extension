pub mod lstsq;

pub enum LinalgErrors {
    DimensionMismatch,
    NotContiguousArray,
    Other(String)
}