/// Conversions from external library matrix views into `faer` types.
pub trait IntoFaer {
    type Faer;
    fn into_faer(self) -> Self::Faer;
}

/// Conversions from external library matrix views into `ndarray` types.
pub trait IntoNdarray {
    type Ndarray;
    fn into_ndarray(self) -> Self::Ndarray;
}

const _: () = {
    use faer::prelude::*;
    use ndarray::{ArrayView, ArrayViewMut, Ix2, ShapeBuilder};

    impl<'a, T> IntoFaer for ArrayView<'a, T, Ix2> {
        type Faer = MatRef<'a, T>;

        #[track_caller]
        fn into_faer(self) -> Self::Faer {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let strides: [isize; 2] = self.strides().try_into().unwrap();
            let ptr = self.as_ptr();
            unsafe { faer::MatRef::from_raw_parts(ptr, nrows, ncols, strides[0], strides[1]) }
        }
    }

    impl<'a, T> IntoFaer for ArrayViewMut<'a, T, Ix2> {
        type Faer = MatMut<'a, T>;

        #[track_caller]
        fn into_faer(self) -> Self::Faer {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let strides: [isize; 2] = self.strides().try_into().unwrap();
            let ptr = { self }.as_mut_ptr();
            unsafe { faer::MatMut::from_raw_parts_mut(ptr, nrows, ncols, strides[0], strides[1]) }
        }
    }

    impl<'a, T> IntoNdarray for MatRef<'a, T> {
        type Ndarray = ArrayView<'a, T, Ix2>;

        #[track_caller]
        fn into_ndarray(self) -> Self::Ndarray {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let row_stride: usize = self.row_stride().try_into().unwrap();
            let col_stride: usize = self.col_stride().try_into().unwrap();
            let ptr = self.as_ptr();
            unsafe {
                ArrayView::<'_, T, Ix2>::from_shape_ptr(
                    (nrows, ncols).strides((row_stride, col_stride)),
                    ptr,
                )
            }
        }
    }

    impl<'a, T> IntoNdarray for MatMut<'a, T> {
        type Ndarray = ArrayViewMut<'a, T, Ix2>;

        #[track_caller]
        fn into_ndarray(self) -> Self::Ndarray {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let row_stride: usize = self.row_stride().try_into().unwrap();
            let col_stride: usize = self.col_stride().try_into().unwrap();
            let ptr = self.as_ptr_mut();
            unsafe {
                ArrayViewMut::<'_, T, Ix2>::from_shape_ptr(
                    (nrows, ncols).strides((row_stride, col_stride)),
                    ptr,
                )
            }
        }
    }
};

const _: () = {
    use faer::prelude::*;
    use numpy::Element;
    use numpy::{PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};

    impl<'a, T: Element + 'a> IntoFaer for PyReadonlyArray2<'a, T> {
        type Faer = MatRef<'a, T>;

        #[track_caller]
        fn into_faer(self) -> Self::Faer {
            let raw_arr = self.as_raw_array();
            let nrows = raw_arr.nrows();
            let ncols = raw_arr.ncols();
            let strides: [isize; 2] = raw_arr.strides().try_into().unwrap();
            let ptr = raw_arr.as_ptr();
            unsafe { MatRef::from_raw_parts(ptr, nrows, ncols, strides[0], strides[1]) }
        }
    }

    impl<'a, T: Element + 'a> IntoFaer for PyReadonlyArray1<'a, T> {
        type Faer = ColRef<'a, T>;

        #[track_caller]
        fn into_faer(self) -> Self::Faer {
            let raw_arr = self.as_raw_array();
            let nrows = raw_arr.len();
            let strides: [isize; 1] = raw_arr.strides().try_into().unwrap();
            let ptr = raw_arr.as_ptr();
            unsafe { ColRef::from_raw_parts(ptr, nrows, strides[0]) }
        }
    }
};
