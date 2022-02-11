use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::{Array1, Array2};

use nnls::nnls;

fn bench_nnls(c: &mut Criterion) {
    let m = 100;
    let n = 144;

    let mut t0 = Array1::zeros(n);
    t0[0] = 2.0;
    t0[1] = t0[0] * 2.0f64.sqrt();
    for i in 2..t0.len() {
        t0[i] = 2.0 * t0[i - 2];
    }

    let arg_limit = (2.0 * 2.2250738585072014e-308f64).ln();
    let a = Array2::from_shape_fn((m, n), |(i, j)| {
        let i = (i + 1) as f64;
        let arg = -10.0 * i / t0[j];
        if arg > arg_limit {
            arg.exp()
        } else {
            0.0
        }
    });

    let b = Array1::from_shape_fn(m, |i| {
        let t = 10.0 * (i + 1) as f64;
        100.0 * ((-t / 5.0).exp() + (-t / 50.0).exp() + (-t / 500.0).exp()) - 0.5
    });

    c.bench_function("nnls", |bencher| {
        bencher.iter(|| black_box(nnls(&a, &b)));
    });
}

criterion_group!(benches, bench_nnls);
criterion_main!(benches);
