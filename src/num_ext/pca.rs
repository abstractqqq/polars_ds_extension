use crate::utils::{to_f64_vec_fail_on_nulls, to_f64_vec_without_nulls, IndexOrder};
use faer::{
    dyn_stack::{MemBuffer, MemStack},
    linalg::svd::ComputeSvdVectors,
    prelude::*,
};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Deserialize)]
struct SvdKwargs {
    as_explained_var: bool,
    as_ratio: bool,
}

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

#[derive(Deserialize)]
pub struct PrincipalComponentsKwargs {
    k: usize,
}

pub fn principal_components_output(
    _: &[Field],
    kwargs: PrincipalComponentsKwargs,
) -> PolarsResult<Field> {
    let components: Vec<_> = (0..kwargs.k)
        .map(|i| Field::new(format!("pc{}", i + 1).into(), DataType::Float64))
        .collect();
    Ok(Field::new(
        "principal_components".into(),
        DataType::Struct(components),
    ))
}

fn named_features(inputs: &[Series]) -> Vec<Series> {
    inputs
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let mut s = s.clone();
            s.rename(format!("feature_{i}").into());
            s
        })
        .collect()
}

#[polars_expr(output_type_func=singular_values_output)]
fn pl_singular_values(inputs: &[Series], kwargs: SvdKwargs) -> PolarsResult<Series> {
    polars_ensure!(!inputs.is_empty(), ComputeError: "at least one feature is required");
    // Grouped plugin inputs can have empty or duplicate names on older Polars.
    let named_inputs = named_features(inputs);
    let mat_slice = to_f64_vec_without_nulls(&named_inputs, IndexOrder::Fortran)?;
    let ncols = inputs.len();
    let nrows = mat_slice.len() / ncols;
    if kwargs.as_explained_var && nrows < 2 {
        return Ok(Series::full_null(
            "singular_values".into(),
            1,
            &DataType::List(Box::new(DataType::Float64)),
        ));
    }
    if nrows == 0 {
        let mut builder = ListPrimitiveChunkedBuilder::<Float64Type>::new(
            "singular_values".into(),
            1,
            0,
            DataType::Float64,
        );
        builder.append_slice(&[]);
        return Ok(builder.finish().into_series());
    }
    let mat = MatRef::from_column_major_slice(&mat_slice, nrows, ncols);

    let (m, n) = mat.shape();
    let compute = ComputeSvdVectors::Thin;

    let dim = Ord::min(mat.nrows(), mat.ncols());

    let mut s = vec![0f64; dim];
    let cs = ColMut::from_slice_mut(&mut s);

    let par = Par::rayon(0);

    faer::linalg::svd::svd(
        mat,
        cs.as_diagonal_mut(),
        None,
        None,
        par,
        MemStack::new(&mut MemBuffer::new(faer::linalg::svd::svd_scratch::<f64>(
            m,
            n,
            compute,
            compute,
            par,
            default(),
        ))),
        Default::default(),
    )
    .map_err(|_| PolarsError::ComputeError("svd algorithm did not converge".into()))?;

    if kwargs.as_explained_var {
        let denominator = (nrows - 1) as f64;
        s.iter_mut().for_each(|v| *v = v.powi(2) / denominator);
    }
    if kwargs.as_ratio {
        let total: f64 = s.iter().sum();
        s.iter_mut().for_each(|v| *v /= total);
    }

    let mut list_builder: ListPrimitiveChunkedBuilder<Float64Type> =
        ListPrimitiveChunkedBuilder::new("singular_values".into(), 1, dim, DataType::Float64);

    list_builder.append_slice(&s);
    let out = list_builder.finish();
    Ok(out.into_series())
}

#[polars_expr(output_type_func_with_kwargs=principal_components_output)]
fn pl_principal_components(
    inputs: &[Series],
    kwargs: PrincipalComponentsKwargs,
) -> PolarsResult<Series> {
    let k = kwargs.k;
    polars_ensure!(k > 0 && k <= inputs.len(), ComputeError: "invalid number of principal components");
    let named_inputs = named_features(inputs);
    let nrows = inputs[0].len();
    let mat_slice = to_f64_vec_fail_on_nulls(&named_inputs, IndexOrder::Fortran)?;

    let columns = if nrows < k {
        (0..k)
            .map(|i| {
                Series::from_vec(format!("pc{}", i + 1).into(), vec![f64::NAN; nrows]).into_column()
            })
            .collect::<Vec<Column>>()
    } else {
        let mat = MatRef::from_column_major_slice(&mat_slice, nrows, inputs.len());
        let (m, n) = mat.shape();
        let compute = ComputeSvdVectors::Thin;
        let dim = Ord::min(mat.nrows(), mat.ncols());
        let mut s = Col::<f64>::zeros(dim);
        let mut v = Mat::<f64>::zeros(inputs.len(), dim);
        let par = Par::rayon(0);
        faer::linalg::svd::svd(
            mat,
            s.as_diagonal_mut(),
            None,
            Some(v.as_mut()),
            par,
            MemStack::new(&mut MemBuffer::new(faer::linalg::svd::svd_scratch::<f64>(
                m,
                n,
                compute,
                compute,
                par,
                default(),
            ))),
            Default::default(),
        )
        .map_err(|_| PolarsError::ComputeError("svd algorithm did not converge".into()))?;

        let components = mat * v;

        (0..k)
            .map(|i| {
                let name = format!("pc{}", i + 1);
                let s = Float64Chunked::from_slice(name.into(), components.col_as_slice(i));
                s.into_series().into_column()
            })
            .collect::<Vec<Column>>()
    };

    let ca = StructChunked::from_columns("principal_components".into(), nrows, &columns)?;
    Ok(ca.into_series())
}

#[polars_expr(output_type_func=pca_output)]
fn pl_pca(inputs: &[Series]) -> PolarsResult<Series> {
    polars_ensure!(!inputs.is_empty(), ComputeError: "at least one feature is required");
    let named_inputs = named_features(inputs);
    let mat_slice = to_f64_vec_without_nulls(&named_inputs, IndexOrder::Fortran)?;
    let nrows = mat_slice.len() / inputs.len();
    if nrows == 0 {
        return Ok(Series::new_empty("pca".into(), pca_output(&[])?.dtype()));
    }
    let mat = MatRef::from_column_major_slice(&mat_slice, nrows, inputs.len());

    let (m, n) = mat.shape();
    let dim = Ord::min(mat.nrows(), mat.ncols());
    let mut s = vec![0f64; dim];
    let cs = ColMut::from_slice_mut(&mut s);
    let mut v = Mat::<f64>::zeros(inputs.len(), dim);
    let par = Par::rayon(0);
    let compute = ComputeSvdVectors::Thin;

    faer::linalg::svd::svd(
        mat,
        cs.as_diagonal_mut(),
        None,
        Some(v.as_mut()),
        par,
        MemStack::new(&mut MemBuffer::new(faer::linalg::svd::svd_scratch::<f64>(
            m,
            n,
            compute,
            compute,
            par,
            default(),
        ))),
        Default::default(),
    )
    .map_err(|_| PolarsError::ComputeError("svd algorithm did not converge".into()))?;

    let mut builder: PrimitiveChunkedBuilder<Float64Type> =
        PrimitiveChunkedBuilder::new("singular_value".into(), dim);
    let mut list_builder: ListPrimitiveChunkedBuilder<Float64Type> =
        ListPrimitiveChunkedBuilder::new("weight_vector".into(), dim, dim, DataType::Float64);

    for i in 0..dim {
        builder.append_value(s[i]);
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
