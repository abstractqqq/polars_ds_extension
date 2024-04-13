use crate::utils::rechunk_to_frame;
use faer::dyn_stack::{GlobalPodBuffer, PodStack};
use faer::prelude::*;
use faer_ext::IntoFaer;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

pub fn singular_values_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "singular_values",
        DataType::List(Box::new(DataType::Float64)),
    ))
}

pub fn pca_output(_: &[Field]) -> PolarsResult<Field> {
    let singular_value = Field::new("singular_value", DataType::Float64);
    let principal_vector = Field::new(
        "principal_values",
        DataType::List(Box::new(DataType::Float64)),
    );
    Ok(Field::new(
        "pca",
        DataType::Struct(vec![singular_value, principal_vector]),
    ))
}

#[polars_expr(output_type_func=singular_values_output)]
fn pl_principal_components(inputs: &[Series]) -> PolarsResult<Series> {
    let df = rechunk_to_frame(inputs)?.drop_nulls::<String>(None)?;
    let mat = df.to_ndarray::<Float64Type>(IndexOrder::Fortran)?;
    let mat = mat.view().into_faer();

    let dim = Ord::min(mat.nrows(), mat.ncols());
    let mut s = Mat::<f64>::zeros(dim, 1);
    let parallelism = faer::Parallelism::Rayon(0); // use current num threads
    let params = Default::default();
    faer::linalg::svd::compute_svd(
        mat.canonicalize().0,
        s.as_mut(),
        None,
        None,
        parallelism,
        PodStack::new(&mut GlobalPodBuffer::new(
            faer::linalg::svd::compute_svd_req::<f64>(
                mat.nrows(),
                mat.ncols(),
                faer::linalg::svd::ComputeVectors::No,
                faer::linalg::svd::ComputeVectors::No,
                parallelism,
                params,
            )
            .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?,
        )),
        params,
    );

    let mut list_builder: ListPrimitiveChunkedBuilder<Float64Type> =
        ListPrimitiveChunkedBuilder::new("singular_values", 1, dim, DataType::Float64);

    list_builder.append_slice(s.col_as_slice(0));
    let out = list_builder.finish();
    Ok(out.into_series())
}

#[polars_expr(output_type_func=pca_output)]
fn pl_pca(inputs: &[Series]) -> PolarsResult<Series> {
    let df = rechunk_to_frame(inputs)?.drop_nulls::<String>(None)?;
    let mat = df.to_ndarray::<Float64Type>(IndexOrder::Fortran)?;
    let mat = mat.view().into_faer();
    let dim = Ord::min(mat.nrows(), mat.ncols());
    let mut s = Mat::<f64>::zeros(dim, 1);
    let mut v = Mat::<f64>::zeros(dim, dim);
    let parallelism = faer::Parallelism::Rayon(0); // use current num threads
    let params = Default::default();
    faer::linalg::svd::compute_svd(
        mat.canonicalize().0,
        s.as_mut(),
        None,
        Some(v.as_mut()),
        parallelism,
        PodStack::new(&mut GlobalPodBuffer::new(
            faer::linalg::svd::compute_svd_req::<f64>(
                mat.nrows(),
                mat.ncols(),
                faer::linalg::svd::ComputeVectors::No,
                faer::linalg::svd::ComputeVectors::Thin,
                parallelism,
                params,
            )
            .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?,
        )),
        params,
    );

    let mut builder: PrimitiveChunkedBuilder<Float64Type> =
        PrimitiveChunkedBuilder::new("singular_value", dim);
    let mut list_builder: ListPrimitiveChunkedBuilder<Float64Type> =
        ListPrimitiveChunkedBuilder::new("principal_vectors", dim, dim, DataType::Float64);

    for i in 0..v.nrows() {
        builder.append_value(s.read(i, 0));
        list_builder.append_slice(v.col_as_slice(i));
    }

    let out1 = builder.finish();
    let out2 = list_builder.finish();
    let out = StructChunked::new("pca", &[out1.into_series(), out2.into_series()])?;
    Ok(out.into_series())
}
