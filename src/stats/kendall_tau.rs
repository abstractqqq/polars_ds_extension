/// O(nlogn) implementation of Kendall's Tau correlation
/// Implemented by translating the Java code:
/// https://www.hipparchus.org/xref/org/hipparchus/stat/correlation/KendallsCorrelation.html
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

#[polars_expr(output_type=Float64)]
pub fn pl_kendall_tau(inputs: &[Series]) -> PolarsResult<Series> {
    let name = inputs[0].name();
    let mut binding = df!("x" => &inputs[0], "y" => &inputs[1])?
        .lazy()
        .filter(col("x").is_not_null().and(col("y").is_not_null()))
        .sort(["x", "y"], Default::default())
        .collect()?;
    let df = binding.align_chunks();

    let n = df.height();
    if n <= 1 {
        return Ok(Series::from_vec(name, vec![f64::NAN]));
    }

    let n_pairs = ((n * (n - 1)) >> 1) as i64;

    let x = df.drop_in_place("x").unwrap();
    let y = df.drop_in_place("y").unwrap();
    let x = x.u32().unwrap();
    let y = y.u32().unwrap();
    let x = x.cont_slice().unwrap();
    let y = y.cont_slice().unwrap();

    let mut tied_x: i64 = 0;
    let mut tied_xy: i64 = 0;
    let mut consecutive_x_ties: i64 = 1;
    let mut consecutive_xy_ties: i64 = 1;
    let (mut xj, mut yj) = (x[0], y[0]);
    for i in 1..x.len() {
        // i current, j prev
        let (xi, yi) = (x[i], y[i]);
        if xi == xj {
            consecutive_x_ties += 1;
            if yi == yj {
                consecutive_xy_ties += 1;
            } else {
                tied_xy += (consecutive_xy_ties * (consecutive_xy_ties - 1)) >> 1;
                consecutive_xy_ties = 1;
            }
        } else {
            tied_x += (consecutive_x_ties * (consecutive_x_ties - 1)) >> 1;
            consecutive_x_ties = 1;
            tied_xy += (consecutive_xy_ties * (consecutive_xy_ties - 1)) >> 1;
            consecutive_xy_ties = 1;
        }
        xj = xi;
        yj = yi;
    }

    tied_x += (consecutive_x_ties * (consecutive_x_ties - 1)) >> 1;
    tied_xy += (consecutive_xy_ties * (consecutive_xy_ties - 1)) >> 1;

    let mut swaps: usize = 0;
    let mut xx = x.to_vec();
    let mut yy = y.to_vec();
    let mut x_copy = x.to_vec();
    let mut y_copy = y.to_vec();
    let mut seg_size: usize = 1;
    while seg_size < n {
        let mut offset: usize = 0;
        while offset < n {
            let mut i = offset;
            let i_end = (i + seg_size).min(n);
            let mut j = i_end;
            let j_end = (j + seg_size).min(n);

            let mut copy_loc = offset;
            while (i < i_end) || (j < j_end) {
                if i < i_end {
                    if j < j_end {
                        if yy[i] <= yy[j] {
                            x_copy[copy_loc] = xx[i];
                            y_copy[copy_loc] = yy[i];
                            i += 1;
                        } else {
                            x_copy[copy_loc] = xx[j];
                            y_copy[copy_loc] = yy[j];
                            j += 1;
                            swaps += i_end - i;
                        }
                    } else {
                        x_copy[copy_loc] = xx[i];
                        y_copy[copy_loc] = yy[i];
                        i += 1;
                    }
                } else {
                    x_copy[copy_loc] = xx[j];
                    y_copy[copy_loc] = yy[j];
                    j += 1;
                }
                copy_loc += 1;
            }
            offset += seg_size << 1; // multiply by 2
        }
        std::mem::swap(&mut xx, &mut x_copy);
        std::mem::swap(&mut yy, &mut y_copy);
        seg_size <<= 1;
    }

    let mut tied_y: i64 = 0;
    let mut consecutive_y_ties: i64 = 1;
    let mut prev = yy[0];
    for i in 1..n {
        if yy[i] == prev {
            consecutive_y_ties += 1;
        } else {
            tied_y += (consecutive_y_ties * (consecutive_y_ties - 1)) >> 1;
            consecutive_y_ties = 1;
        }
        prev = yy[i];
    }
    tied_y += (consecutive_y_ties * (consecutive_y_ties - 1)) >> 1;

    let nc_m_nd = n_pairs - tied_x - tied_y + tied_xy - ((swaps << 1) as i64);
    // Prevent overflow
    let denom = (((n_pairs - tied_x) as f64) * ((n_pairs - tied_y) as f64)).sqrt();
    let out = nc_m_nd as f64 / denom;
    Ok(Series::from_vec(name, vec![out]))
}
