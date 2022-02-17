#![allow(clippy::many_single_char_names, clippy::too_many_arguments)]
#![deny(unsafe_code)]

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, Zip};
use ndarray_stats::QuantileExt;

const FACTOR: f64 = 0.01;

/// Non-negative least squares solver.
///
/// NNLS solves the following matrix problem:
///   Minimize || E x - f || subject to G x >= h.
///
///   where
///    E is an m2 x n matrix and f is the m2 element data vector.
///    In general, G is an m x n constraint matrix, but for his case,
///    G is the identity matrix and h is the zero vector.
pub fn nnls(a: ArrayView2<f64>, b: ArrayView1<f64>) -> (Array1<f64>, f64) {
    let (m, n) = a.dim();
    if m == 0 || n == 0 {
        panic!("The dimensions of the problem are bad. Either `a` has zero row or column.");
    }
    let mut nnls = Nnls::new(a, b);
    nnls.run()
}

struct Nnls {
    a: Array2<f64>,
    b: Array1<f64>,
    zz: Array1<f64>,
    w: Vec<f64>,
    index: Vec<usize>,
}

impl Nnls {
    pub fn new(a: ArrayView2<f64>, b: ArrayView1<f64>) -> Nnls {
        let (m, n) = a.dim();
        let zz = Array1::zeros(m);
        let w = vec![0.0; n];
        let index: Vec<_> = (0..n).collect();
        Nnls { a: a.to_owned(), b: b.to_owned(), zz, w, index }
    }

    pub fn run(&mut self) -> (Array1<f64>, f64) {
        let (m, n) = self.a.dim();
        let mut x = Array1::<f64>::zeros(n);

        let max_iter = 3 * n;
        let mut iz1 = 0;
        let mut nsetp = 0;
        let mut npp1 = 0;
        let mut up = 0.0;

        loop {
            // Compute components of the dual (negative gradient) vector w
            for iz in iz1..n {
                let j = self.index[iz];
                self.w[j] = self.a.slice(s![npp1.., j]).dot(&self.b.slice(s![npp1..]));
            }

            // Find largest positive w
            let (iz, w_max) = self.arg_max_w(iz1);
            if w_max <= 0.0 {
                break;
            }

            let j = self.index[iz];

            // The sign of w[j] is ok for j to be moved to set P. Begin the transformation and check
            // new diagonal element to avoid near linear dependance

            //let a_save = self.a[(npp1, j)];
            h12(1, npp1, npp1 + 1, m, self.a.column_mut(j), &mut up, self.zz.view_mut(), 1, 0, 0);

            // In the original implementation `nsetp` is used here instead of `npp1`, but `npp1` is
            // a better choise in 0-indexed langages.
            let u_norm = if npp1 == 0 {
                0.0
            } else {
                self.a.slice(s![..nsetp, j]).fold(0.0, |acc, &a| acc + a.powi(2)).sqrt()
            };
            if u_norm + self.a[(npp1, j)].abs() * FACTOR - u_norm <= 0.0 {
                // Project j as a candidate to be moved from set z to set p.
                //self.a[(npp1, j)] = a_save;
                //self.w[j] = 0.0;
                panic!("GoTo iz_max");
            }

            // Column j is sufficiently independant. Copy b into zz, update zz and solve for
            // ztest ( = proposed new value for x[j]) )
            self.zz.assign(&self.b);
            h12(2, npp1, npp1 + 1, m, self.a.column_mut(j), &mut up, self.zz.view_mut(), 1, 0, 1);
            let ztest = self.zz[npp1] / self.a[(npp1, j)];
            if ztest <= 0.0 {
                // Project j as a candidate to be moved from set z to set p.
                //self.a[(npp1, j)] = a_save;
                //self.w[j] = 0.0;
                panic!("GoTo iz_max");
            }

            self.b.assign(&self.zz);
            self.index[iz] = self.index[iz1];
            self.index[iz1] = j;
            iz1 += 1;
            nsetp = npp1;
            npp1 += 1;

            if iz1 < n {
                for &jj in &self.index[iz1..] {
                    let (u, c) = self.a.multi_slice_mut((s![.., j], s![.., jj]));
                    h12(2, nsetp, npp1, m, u, &mut up, c, 1, m, 1);
                }
            }

            // This line is not present in the original implementation. We need it here because of
            // the difference in indexing (0 or 1) between Fortran and Rust.
            nsetp += 1;

            if nsetp != m {
                self.a.slice_mut(s![npp1.., j]).fill(0.0);
            }

            self.w[j] = 0.0;

            // Solve the triangular system.  Store the solution temporarily in zz.
            self.solve_triangular(nsetp, j);

            // Secondary loop begins here
            for _ in 0..max_iter {
                // If the new constained coefficients are all feasible, then alpha will still be
                // equal to 2.0. If so, exit secondary loop.
                let (mut jj, alpha) = self.find_alpha(&x, nsetp);
                if alpha == 2.0 {
                    Zip::from(&self.index[0..nsetp])
                        .and(self.zz.slice(s![0..nsetp]))
                        .for_each(|&i, &zz| x[i] = zz);
                    break; // to main loop
                }

                // Otherwise, use alpha [0.0, 1.0] to interpolate between the old x and the new zz
                Zip::from(&self.index[0..nsetp])
                    .and(self.zz.slice(s![0..nsetp]))
                    .for_each(|&i, &zz| x[i] += alpha * (zz - x[i]));

                // Modify a, b and the index arrays to move coefficient I from set P to set Z
                let i = self.index[jj];
                x[i] = 0.0;

                if jj != nsetp {
                    jj += 1;
                    for j in jj..nsetp {
                        let ii = self.index[j];
                        self.index[j - 1] = ii;
                        let (cc, ss, sig) = g1(self.a[(j - 1, ii)], self.a[(j, ii)]);
                        self.a[(j - 1, ii)] = sig;
                        self.a[(j, ii)] = 0.0;
                        for l in 0..n {
                            if l != ii {
                                // Apply procedure G2 (CC, SS, A(J-1,L), A(J,L))
                                let v = self.a[(j - 1, l)];
                                self.a[(j - 1, l)] = cc * v + ss * self.a[(j, l)];
                                self.a[(j, l)] = -ss * v + cc * self.a[(j, l)];
                            }
                        }

                        // Apply procedure G2 (CC, SS, B(J-1), B(J))
                        let v = self.b[j - 1];
                        self.b[j - 1] = cc * v + ss * self.b[j];
                        self.b[j] = -ss * v + cc * self.b[j];
                    }
                }

                npp1 = nsetp - 1;
                nsetp -= 1;
                iz1 -= 1;
                self.index[iz1] = i;

                // See of the remaining coefficients are feasible. They should be because of the way
                // `alpha` was determined. If any are infeasible, it's due to round-off error. Any
                // that are non-positive will be set to 0.0 and moved from set P to set Z.

                if self.index[0..nsetp].iter().any(|&i| x[i] <= 0.0) {
                    panic!("GOTO x[i] = 0.0;");
                }

                // Copy b into zz. Then solve again and loop back
                self.zz.assign(&self.b);
                self.solve_triangular(nsetp, j);
            }

            if iz1 >= n || nsetp >= m {
                break;
            }
        }

        let mut sm = 0.0;
        if npp1 <= m {
            sm = self.b.slice(s![npp1..]).fold(0.0, |acc, &a| acc + a.powi(2));
        }

        (x, sm.sqrt())
    }

    fn arg_max_w(&self, start: usize) -> (usize, f64) {
        let mut max_idx = 0;
        let mut max_val = f64::MIN;
        for (i, &index) in self.index[start..].iter().enumerate() {
            let w = self.w[index];
            if w > max_val {
                max_val = w;
                max_idx = i + start;
            }
        }
        (max_idx, max_val)
    }

    fn solve_triangular(&mut self, nsetp: usize, mut j: usize) {
        for l in 0..nsetp {
            let ip = nsetp - 1 - l;
            if l != 0 {
                let zz_ip = self.zz[ip + 1];
                Zip::from(self.zz.slice_mut(s![..=ip]))
                    .and(self.a.slice(s![..=ip, j]))
                    .for_each(|zz, &a| *zz -= a * zz_ip);
            }
            j = self.index[ip];
            self.zz[ip] /= self.a[(ip, j)];
        }
    }

    fn find_alpha(&self, x: &Array1<f64>, nsetp: usize) -> (usize, f64) {
        let mut jj = 0;
        let mut alpha = 2.0;
        Zip::indexed(&self.index[0..nsetp]).and(self.zz.slice(s![0..nsetp])).for_each(
            |ip, &l, &zz| {
                if zz <= 0.0 {
                    let t = -x[l] / (zz - x[l]);
                    if alpha > t {
                        alpha = t;
                        jj = ip;
                    }
                }
            },
        );
        (jj, alpha)
    }
}

/// Construction and/or application of a single householder transformtion.
fn h12(
    mode: usize,
    pivot: usize,
    l1: usize,
    m: usize,
    mut u: ArrayViewMut1<f64>,
    up: &mut f64,
    mut c: ArrayViewMut1<f64>,
    ice: usize,
    icv: usize,
    ncv: usize,
) {
    if pivot >= l1 || l1 > m {
        return;
    }

    let mut cl = u[pivot].abs();
    if mode == 1 {
        let u_ = u.slice(s![l1..]);
        cl = f64::max(cl, *u_.max_skipnan());
        if cl <= 0.0 {
            return;
        }

        let clinv = 1.0 / cl;
        let sm = (u[pivot] * clinv).powi(2) + u_.fold(0.0, |acc, &u| acc + (u * clinv).powi(2));
        cl *= sm.sqrt();
        if u[pivot] > 0.0 {
            cl = -cl;
        }
        *up = u[pivot] - cl;
        u[pivot] = cl;
    } else if cl <= 0.0 {
        return;
    }

    if ncv == 0 {
        return;
    }

    let b = *up * u[pivot];
    if b < 0.0 {
        let b = 1.0 / b;
        let mut i2 = ice * pivot;
        let incr = ice * (l1 - pivot);
        for _ in 0..ncv {
            let mut i3 = i2 + incr;
            let mut i4 = i3;
            let mut sm = c[i2] * *up;
            for i in l1..m {
                sm += c[i3] * u[i];
                i3 += ice;
            }
            if sm != 0.0 {
                sm *= b;
                c[i2] += sm * *up;
                for i in l1..m {
                    c[i4] += sm * u[i];
                    i4 += ice;
                }
            }
            i2 += icv;
        }
    }
}

fn g1(a: f64, b: f64) -> (f64, f64, f64) {
    if a.abs() > b.abs() {
        let xr = b / a;
        let yr = (1.0 + xr.powi(2)).sqrt();
        let cterm = (1.0 / yr) * a.signum();
        (cterm, cterm * xr, a.abs() * yr)
    } else if b != 0.0 {
        let xr = a / b;
        let yr = (1.0 + xr.powi(2)).sqrt();
        let sterm = (1.0 / yr) * b.signum();
        let cterm = sterm * xr;
        (cterm, sterm, b.abs() * yr)
    } else {
        (0.0, 1.0, a)
    }
}
