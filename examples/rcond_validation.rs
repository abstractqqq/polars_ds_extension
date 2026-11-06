//! Run numerical checks with:
//! cargo test --release --example rcond_validation

#[path = "../src/linear/lr/rcond.rs"]
mod rcond;

fn main() {
    let x = faer::Mat::from_fn(3, 1, |i, _| (i + 1) as f64);
    let y = faer::Mat::from_fn(3, 1, |i, _| 2.0 * (i + 1) as f64);
    let (beta, _) =
        rcond::faer_solve_lr_rcond_stable(x.as_ref(), y.as_ref(), 0.0, false, 1e-12).unwrap();
    println!(
        "Example slope: {}. See source comments for validation commands.",
        beta[(0, 0)]
    );
}
