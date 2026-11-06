use super::*;
use std::time::{Duration, Instant};

fn options() -> Vec<RcondOptions> {
    [128, 512, 1024, 2048, 4096]
        .into_iter()
        .flat_map(|block_rows| {
            [(None, false), (Some(256), true), (Some(256), false)].map(
                |(qr_target_exponent, rescale_each_block)| RcondOptions {
                    block_rows,
                    qr_target_exponent,
                    rescale_each_block,
                },
            )
        })
        .collect()
}

fn problem<T: RealField + Float>(rows: usize, features: usize, weak: f64) -> (Mat<T>, Mat<T>) {
    // Deterministic, nontrivial dense data; no RNG or external test dependency.
    let x = Mat::from_fn(rows, features, |i, j| {
        let mut bits = (i as u64).wrapping_add((j as u64).wrapping_mul(0x9e3779b97f4a7c15));
        bits = (bits ^ (bits >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        bits = (bits ^ (bits >> 27)).wrapping_mul(0x94d049bb133111eb);
        bits ^= bits >> 31;
        let value = ((bits >> 11) as f64 / (1u64 << 53) as f64 - 0.5)
            * if j == features - 1 { weak } else { 1.0 };
        T::from(value).unwrap()
    });
    let y = Mat::from_fn(rows, 1, |i, _| {
        T::from(
            (0..features)
                .map(|j| x[(i, j)].to_f64().unwrap() * (j + 1) as f64 / features as f64)
                .sum::<f64>()
                + 0.03 * ((i * 13 + 2) as f64).sin(),
        )
        .unwrap()
    });
    (x, y)
}

fn direct_svd_reference<T: RealField + Float>(
    x: MatRef<'_, T>,
    y: MatRef<'_, T>,
    lambda: f64,
    cutoff: f64,
) -> (Mat<f64>, Vec<f64>) {
    // Direct SVD of the full augmented design: no QR reduction or carried
    // target. Python tests independently validate against NumPy/LAPACK.
    let design = Mat::<f64>::from_fn(x.nrows() + x.ncols(), x.ncols(), |i, j| {
        if i < x.nrows() {
            x[(i, j)].to_f64().unwrap()
        } else if i - x.nrows() == j {
            lambda.sqrt()
        } else {
            0.0
        }
    });
    let target = Mat::<f64>::from_fn(design.nrows(), 1, |i, _| {
        if i < y.nrows() {
            y[(i, 0)].to_f64().unwrap()
        } else {
            0.0
        }
    });
    let svd = design.thin_svd().unwrap();
    let spectrum: Vec<_> = svd.S().column_vector().iter().copied().collect();
    let mut projected = svd.U().transpose() * target;
    for i in 0..x.ncols() {
        projected[(i, 0)] = if spectrum[i] > cutoff * spectrum[0] {
            projected[(i, 0)] / spectrum[i]
        } else {
            0.0
        };
    }
    (svd.V() * projected, spectrum)
}

fn validate_options<T: RealField + Float>(weak: f64, tolerance: f64) {
    for rows in [513, 4097] {
        let (x, y) = problem::<T>(rows, 5, weak);
        for lambda in [0.0, 0.2] {
            let cutoff = T::epsilon().to_f64().unwrap() * rows as f64;
            let (expected, spectrum) = direct_svd_reference(x.as_ref(), y.as_ref(), lambda, cutoff);
            let x64 = Mat::from_fn(rows, 5, |i, j| x[(i, j)].to_f64().unwrap());
            let y64 = Mat::from_fn(rows, 1, |i, _| y[(i, 0)].to_f64().unwrap());
            let reference_prediction = &x64 * &expected;
            let reference_residual = &reference_prediction - &y64;
            let reference_objective =
                reference_residual.squared_norm_l2() + lambda * expected.squared_norm_l2();
            for config in options() {
                let (actual, actual_spectrum) = solve_with_options(
                    x.as_ref(),
                    y.as_ref(),
                    T::from(lambda).unwrap(),
                    false,
                    T::from(cutoff).unwrap(),
                    config,
                )
                .unwrap();
                let actual = Mat::from_fn(5, 1, |i, _| actual[(i, 0)].to_f64().unwrap());
                let prediction = &x64 * &actual;
                let residual = &prediction - &y64;
                let errors = [
                    (&actual - &expected).norm_l2() / expected.norm_l2(),
                    (&prediction - &reference_prediction).norm_l2()
                        / reference_prediction.norm_l2(),
                    ((residual.squared_norm_l2() + lambda * actual.squared_norm_l2())
                        - reference_objective)
                        .abs()
                        / reference_objective,
                ];
                assert!(errors.iter().all(|&e| e < tolerance),
                    "rows={rows}, lambda={lambda}, block={}, scale={:?}, coefficient/prediction/objective errors={errors:?}",
                    config.block_rows, config.qr_target_exponent);
                for (a, b) in actual_spectrum.iter().zip(&spectrum) {
                    assert!((a.to_f64().unwrap() - b).abs() / spectrum[0] < tolerance);
                }
            }
        }
    }
}

#[test]
fn block_sizes_and_scaling_preserve_float64_solution() {
    validate_options::<f64>(1e-7, 2e-7);
}

#[test]
fn block_sizes_and_scaling_preserve_float32_solution() {
    validate_options::<f32>(0.02, 2e-5);
}

#[test]
fn block_sizes_preserve_rank_deficiency_and_cutoff_boundary() {
    for config in options() {
        let mut x = Mat::<f64>::zeros(4097, 4);
        x[(0, 0)] = 1.0;
        x[(512, 1)] = 0.125;
        x[(2048, 2)] = 0.015625;
        // An exactly zero first block for the last column must not lead to
        // an extra rank decision before the final SVD.
        x[(4096, 3)] = 0.0;
        let y = &x * Mat::from_fn(4, 1, |i, _| (i + 1) as f64);
        let (beta, spectrum) =
            solve_with_options(x.as_ref(), y.as_ref(), 0.0, false, 0.125, config).unwrap();
        assert_eq!(spectrum, vec![1.0, 0.125, 0.015625, 0.0]);
        for i in 0..4 {
            assert!((beta[(i, 0)] - if i == 0 { 1.0 } else { 0.0 }).abs() < 1e-14);
        }
    }
}

#[test]
fn block_sizes_preserve_extreme_scales_and_late_rank() {
    for scale in [1e-280, 1.0, 1e280] {
        for prefix in [0.0, 1e-280] {
            let (mut x, mut y) = problem::<f64>(4097, 3, 1.0);
            for j in 0..3 {
                for i in 0..4097 {
                    x[(i, j)] *= scale * if i < 1024 { prefix } else { 1.0 };
                }
            }
            for i in 0..4097 {
                y[(i, 0)] = (0..3).map(|j| x[(i, j)] * (j + 1) as f64).sum();
            }
            for config in options() {
                let (beta, _) =
                    solve_with_options(x.as_ref(), y.as_ref(), 0.0, false, 1e-12, config).unwrap();
                for j in 0..3 {
                    assert!((beta[(j, 0)] - (j + 1) as f64).abs() < 1e-11);
                }
            }
        }
    }
}

#[test]
fn block_sizes_preserve_wide_designs() {
    for features in [65, 129] {
        let (x, y) = problem::<f64>(3 * features + 1, features, 1.0);
        let (expected, spectrum) = direct_svd_reference(x.as_ref(), y.as_ref(), 0.2, 1e-12);
        for config in options() {
            let (beta, actual_spectrum) =
                solve_with_options(x.as_ref(), y.as_ref(), 0.2, false, 1e-12, config).unwrap();
            assert!((&beta - &expected).norm_l2() / expected.norm_l2() < 1e-11);
            for (a, b) in actual_spectrum.iter().zip(&spectrum) {
                assert!((a - b).abs() / spectrum[0] < 1e-12);
            }
        }
    }
}

#[test]
fn qr_scaling_does_not_underflow_a_representable_coefficient() {
    let x = Mat::from_fn(2, 1, |i, _| if i == 0 { 2.0f64.powi(-600) } else { 0.0 });
    for gap in [600, 900, 1000] {
        let y = Mat::from_fn(2, 1, |i, _| {
            2.0f64.powi(if i == 0 { 1000 - gap } else { 1000 })
        });
        for config in options() {
            let (beta, _) =
                solve_with_options(x.as_ref(), y.as_ref(), 0.0, false, 1e-12, config).unwrap();
            assert!(
                (beta[(0, 0)] / 2.0f64.powi(1600 - gap) - 1.0).abs() < 1e-12,
                "gap={gap}, per_block={}",
                config.rescale_each_block
            );
        }
    }
}

#[test]
#[ignore = "manual release benchmark; run without concurrent builds or tests"]
fn benchmark_rcond() {
    // Keep the first JSON record separate from the test harness's status text.
    println!();
    fn run<T: RealField + Float>(precision: &str) {
        for (rows, features, batch) in [
            (64, 3, 200),
            (1000, 20, 10),
            (100000, 3, 1),
            (100000, 20, 1),
            (1000000, 3, 1),
        ] {
            let (x, y) = problem::<T>(rows, features, 1.0);
            let configurations: Vec<_> = options()
                .into_iter()
                .filter(|c| {
                    c.block_rows >= 512 && (c.qr_target_exponent.is_some() || c.block_rows == 512)
                })
                .collect();
            let mut timings = vec![Vec::new(); configurations.len()];
            let cutoff = T::epsilon() * T::from(rows).unwrap();
            let (reference, spectrum) =
                faer_solve_lr_rcond_stable(x.as_ref(), y.as_ref(), T::zero(), false, cutoff)
                    .unwrap();
            let tolerance = if precision == "f64" { 1e-8 } else { 2e-5 };
            for config in &configurations {
                let (beta, actual_spectrum) =
                    solve_with_options(x.as_ref(), y.as_ref(), T::zero(), false, cutoff, *config)
                        .unwrap();
                for i in 0..features {
                    assert!(
                        (beta[(i, 0)].to_f64().unwrap() - reference[(i, 0)].to_f64().unwrap())
                            .abs()
                            < tolerance
                    );
                    assert!(
                        (actual_spectrum[i].to_f64().unwrap() - spectrum[i].to_f64().unwrap())
                            .abs()
                            / spectrum[0].to_f64().unwrap()
                            < tolerance
                    );
                    assert!(
                        actual_spectrum[i] > cutoff * actual_spectrum[0],
                        "benchmark must retain full rank"
                    );
                }
            }
            for repeat in 0..5 {
                let mut order: Vec<_> = (0..configurations.len()).collect();
                if repeat % 2 == 1 {
                    order.reverse();
                }
                for index in order {
                    let start = Instant::now();
                    for _ in 0..batch {
                        std::hint::black_box(
                            solve_with_options(
                                x.as_ref(),
                                y.as_ref(),
                                T::zero(),
                                false,
                                T::epsilon() * T::from(rows).unwrap(),
                                configurations[index],
                            )
                            .unwrap(),
                        );
                    }
                    timings[index].push(start.elapsed() / batch);
                }
            }
            for (config, mut times) in configurations.into_iter().zip(timings) {
                times.sort();
                let median: Duration = times[2];
                println!("{{\"precision\":\"{precision}\",\"rows\":{rows},\"features\":{features},\"block_rows\":{},\"qr_exponent\":{:?},\"per_block\":{},\"median_ms\":{},\"samples_ms\":{:?}}}",
                    config.block_rows, config.qr_target_exponent.unwrap_or(0), config.rescale_each_block, median.as_secs_f64() * 1000.0,
                    times.iter().map(|d| d.as_secs_f64() * 1000.0).collect::<Vec<_>>());
            }
        }
    }
    run::<f64>("f64");
    run::<f32>("f32");
}
