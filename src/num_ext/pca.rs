use crate::utils::{to_f64_matrix_fail_on_nulls, to_f64_matrix_without_nulls};
use faer::dyn_stack::{GlobalPodBuffer, PodStack};
use faer::prelude::*;
use faer_ext::IntoFaer;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

pub fn singular_values_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "singular_values".into(),
        DataType::List(Box::new(DataType::Float64)),
    ))
}

pub fn pca_output(_: &[Field]) -> PolarsResult<Field> {
    let singular_value = Field::new("singular_value".into(), DataType::Float64);
    let weights = Field::new(
        "weight_vector".into(),
        DataType::List(Box::new(DataType::Float64)),
    );
    Ok(Field::new(
        "pca".into(),
        DataType::Struct(vec![singular_value, weights]),
    ))
}

pub fn principal_components_output(fields: &[Field]) -> PolarsResult<Field> {
    let components: Vec<_> = (0..fields.len())
        .map(|i| Field::new(format!("pc{}", i + 1).into(), DataType::Float64))
        .collect();
    Ok(Field::new(
        "principal_components".into(),
        DataType::Struct(components),
    ))
}

#[polars_expr(output_type_func=singular_values_output)]
fn pl_singular_values(inputs: &[Series]) -> PolarsResult<Series> {
    let mat = to_f64_matrix_without_nulls(inputs, IndexOrder::Fortran)?;
    let mat = mat.view().into_faer();

    let dim = Ord::min(mat.nrows(), mat.ncols());
    let mut s = Col::zeros(dim);
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
        ListPrimitiveChunkedBuilder::new("singular_values".into(), 1, dim, DataType::Float64);

    list_builder.append_slice(s.as_slice());
    let out = list_builder.finish();
    Ok(out.into_series())
}

#[polars_expr(output_type_func=principal_components_output)]
fn pl_principal_components(inputs: &[Series]) -> PolarsResult<Series> {
    let k = inputs[0].u32()?;
    let k = k.get(0).unwrap() as usize;

    let mat = to_f64_matrix_fail_on_nulls(&inputs[1..], IndexOrder::Fortran)?;
    let mat = mat.view().into_faer();
    let n = mat.nrows();

    let columns = if mat.nrows() < k {
        (0..k)
            .map(|i| Series::from_vec(format!("pc{}", i + 1).into(), vec![f64::NAN]).into_column())
            .collect::<Vec<Column>>()
    } else {
        let dim = Ord::min(mat.nrows(), mat.ncols());
        let mut s = Col::zeros(dim);
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

        let components = mat * v;

        (0..k)
            .map(|i| {
                let name = format!("pc{}", i + 1);
                let s = Float64Chunked::from_slice(name.into(), components.col_as_slice(i));
                s.into_series().into_column()
            })
            .collect::<Vec<Column>>()
    };

    let ca = StructChunked::from_columns("principal_components".into(), n, &columns)?;
    Ok(ca.into_series())
}

#[polars_expr(output_type_func=pca_output)]
fn pl_pca(inputs: &[Series]) -> PolarsResult<Series> {
    let mat = to_f64_matrix_without_nulls(inputs, IndexOrder::Fortran)?;
    let mat = mat.view().into_faer();
    let dim = Ord::min(mat.nrows(), mat.ncols());
    let mut s = Col::zeros(dim);
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
        PrimitiveChunkedBuilder::new("singular_value".into(), dim);
    let mut list_builder: ListPrimitiveChunkedBuilder<Float64Type> =
        ListPrimitiveChunkedBuilder::new("weight_vector".into(), dim, dim, DataType::Float64);

    for i in 0..v.nrows() {
        builder.append_value(s.read(i));
        list_builder.append_slice(v.col_as_slice(i));
    }

    let out1 = builder.finish();
    let out2 = list_builder.finish();
    let out = StructChunked::from_series(
        "pca".into(),
        out1.len(),
        [&out1.into_series(), &out2.into_series()].into_iter(),
    )?;
    Ok(out.into_series())
}
