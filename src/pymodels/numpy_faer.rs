use pyo3::prelude::*;
use pyo3::types::PyDict;
use faer::{Mat, MatRef};

pub fn numpy_mat_to_faer<'py>(array_obj: Bound<'py, PyAny>) -> PyResult<MatRef<'py, f64>> {
    // 1. Extract the __array_interface__ dictionary
    let interface = array_obj
        .getattr("__array_interface__")?
        .downcast_into::<PyDict>()?;

    let typestr: String = interface.get_item("typestr")?.unwrap().extract()?;
    if typestr != "<f8" {
        return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Only little-endian f64 dtype is expected. Found {}", typestr)
            )
        );
    }

    // If we have __array_interface__, we should have shape. However, to prevent bad
    // implementations, we will still do the safety checks.
    // 2. Extract Shape
    let shape: Vec<usize> = interface.get_item("shape")?.unwrap().extract()?;
    if shape.len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err("Expected a 2D array"));
    }

    let nrows = shape[0];
    let ncols = shape[1];

    // 3. Extract the Data Pointer
    // "data" returns a tuple: (integer_pointer, read_only_boolean)
    let data_tuple = interface.get_item("data")?.unwrap();
    let ptr_int: usize = data_tuple.get_item(0)?.extract()?;
    let ptr = ptr_int as *const f64;

    // 4. Extract Strides (or calculate them if C-contiguous)
    let elem_size = std::mem::size_of::<f64>();
    let (row_stride, col_stride) = match interface.get_item("strides")? {
        // If strides is None, the array is C-contiguous
        Some(strides_obj) if !strides_obj.is_none() => {
            let strides: Vec<isize> = strides_obj.extract()?;
            (
                (strides[0] / elem_size as isize) as usize,
                (strides[1] / elem_size as isize) as usize,
            )
        },
        _ => (ncols, 1), 
    };

    // 5. Construct the Faer MatRef safely
    let mat: MatRef<'py, f64> = unsafe {
        MatRef::from_raw_parts(ptr, nrows, ncols, row_stride as isize, col_stride as isize)
    };
    Ok(mat)
}

pub fn numpy_1d_to_slice<'py>(array_obj: Bound<'py, PyAny>) -> PyResult<&'py [f64]> {
    // 1. Extract the __array_interface__ dictionary
    let interface = array_obj.getattr("__array_interface__")?.downcast_into::<PyDict>()?;

    // 2. Validate Data Type (ensure it is a 64-bit float)
    let typestr: String = interface.get_item("typestr")?.unwrap().extract()?;
    if typestr != "<f8" {
        return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Only little-endian f64 dtype is expected. Found {}", typestr)
            )
        );
    }
    // 3. Extract Shape and validate it is exactly 1D
    let shape: Vec<usize> = interface.get_item("shape")?.unwrap().extract()?;
    if shape.len() != 1 {
        return Err(pyo3::exceptions::PyValueError::new_err("Expected a 1D array"));
    }
    let len = shape[0];

    // 4. Extract Pointer
    let data_tuple = interface.get_item("data")?.unwrap();
    let ptr_int: usize = data_tuple.get_item(0)?.extract()?;
    let ptr = ptr_int as *const f64;

    // 5. Validate Contiguity
    // Check contiguity based on strides
    let elem_size = std::mem::size_of::<f64>() as isize;
    let is_contiguous = match interface.get_item("strides")? {
        // If "strides" exists and is NOT Python's None
        Some(strides_obj) if !strides_obj.is_none() => {
            let strides: Vec<isize> = strides_obj.extract()?;
            // For a 1D array, it is contiguous if the distance between elements 
            // exactly equals the byte size of a single element (8 bytes for f64).
            strides[0] == elem_size
        },
        // If "strides" is None or completely missing, NumPy guarantees it is contiguous.
        _ => true, 
    };
    if !is_contiguous {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Array is not contiguous in memory. Please call np.ascontiguousarray() in Python before passing."
        ));
    }

    // 6. Safely construct the standard Rust slice
    // Safety: `ptr` is valid as long as `array_obj` is alive. We verified length and contiguity.
    let slice: &'py [f64] = unsafe { std::slice::from_raw_parts(ptr, len) };
    Ok(slice)
}

/// --------------------------------------------------------------------------------------------
/// A Python-visible wrapper around an owned Faer matrix
#[pyclass]
pub struct PyFaerMat {pub mat: Mat<f64>}

#[pymethods]
impl PyFaerMat {
    /// Expose the memory to Python/NumPy via the Array Interface
    #[getter]
    fn __array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);

        let nrows = self.mat.nrows();
        let ncols = self.mat.ncols();
        let elem_size = std::mem::size_of::<f64>() as isize;
        // 1. Shape
        dict.set_item("shape", (nrows, ncols))?;
        // 2. Data Type (typestr)
        // always?
        dict.set_item("typestr", "<f8")?;

        // 3. Strides (Must be converted from elements to BYTES for Python)
        let row_stride_bytes = self.mat.row_stride() * elem_size;
        let col_stride_bytes = self.mat.col_stride() * elem_size;
        dict.set_item("strides", (row_stride_bytes, col_stride_bytes))?;

        // 4. Data Pointer & Mutability (False = writable, True = read-only)
        let ptr = self.mat.as_ptr() as usize;
        dict.set_item("data", (ptr, false))?;
        // 5. Version (Standard protocol version)
        dict.set_item("version", 3)?;
        Ok(dict)
    }
}

#[pyclass]
pub struct PyVec {pub data: Vec<f64>}

#[pymethods]
impl PyVec {
    /// Expose the memory to Python/NumPy via the Array Interface
    #[getter]
    fn __array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);

        // 1. Shape
        // CRITICAL: A 1D shape must be a tuple with one element. 
        // In Rust, `(len,)` creates a 1-element tuple.
        dict.set_item("shape", (self.data.len(),))?;

        // 2. Data Type
        dict.set_item("typestr", "<f8")?;

        // 3. Strides. 1D
        dict.set_item("strides", py.None())?;

        // 4. Data Pointer & Mutability (false = writable)
        let ptr = self.data.as_ptr() as usize;
        dict.set_item("data", (ptr, false))?;

        // 5. Protocol Version
        dict.set_item("version", 3)?;

        Ok(dict)
    }
}