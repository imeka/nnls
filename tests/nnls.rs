use approx::assert_relative_eq;
use ndarray::{arr1, arr2, Array1, Array2};

use nnls::nnls;

#[test]
fn test_nnls_1() {
    let a = arr2(&[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
    let b = arr1(&[2.0, 1.0, 1.0]);
    let (x, r_norm) = nnls(a.view(), b.view());
    assert_relative_eq!(x, arr1(&[1.5, 1.0]), epsilon = 1e-9);
    assert_relative_eq!(r_norm, 0.7071067811865475, epsilon = 1e-9);
}

#[test]
fn test_nnls_2() {
    let a = arr2(&[[1.0, 0.5, 0.0], [1.0, 0.6, 0.0], [0.9, 0.5, 0.1], [0.0, 0.0, 1.0]]);
    let b = arr1(&[2.0, 1.0, 1.0, 0.5]);
    let (x, r_norm) = nnls(a.view(), b.view());
    assert_relative_eq!(x, arr1(&[1.37279152, 0.0, 0.47173145]), epsilon = 1e-5);
    assert_relative_eq!(r_norm, 0.7829905522718629, epsilon = 1e-5);
}

#[test]
fn test_mixture_of_exponentials() {
    let mut t0 = Array1::zeros(19);
    t0[0] = 2.0;
    t0[1] = t0[0] * 2.0f64.sqrt();
    for i in 2..t0.len() {
        t0[i] = 2.0 * t0[i - 2];
    }

    let arg_limit = (2.0 * 2.2250738585072014e-308f64).ln();
    let a = Array2::from_shape_fn((100, 19), |(i, j)| {
        let i = (i + 1) as f64;
        let arg = -10.0 * i / t0[j];
        if arg > arg_limit {
            arg.exp()
        } else {
            0.0
        }
    });

    let b = Array1::from_shape_fn(100, |i| {
        let t = 10.0 * (i + 1) as f64;
        100.0 * ((-t / 5.0).exp() + (-t / 50.0).exp() + (-t / 500.0).exp()) - 0.5
    });

    let (x, r_norm) = nnls(a.view(), b.view());

    let mut x_gt = Array1::zeros(19);
    x_gt[2] = 146.18289932114504;
    x_gt[9] = 82.906587033240356;
    x_gt[10] = 18.699400056374593;
    x_gt[15] = 13.315689103364351;
    x_gt[16] = 87.197680367574506;
    assert_relative_eq!(x, x_gt, epsilon = 1e-5);
    assert_relative_eq!(r_norm, 0.88433322916388879, epsilon = 1e-5);
}
