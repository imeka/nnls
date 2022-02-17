# nnls

This is a Rust version of nnls (Non-Negative Least Squares). It's a port from a [Fortran90 script](https://hesperia.gsfc.nasa.gov/~schmahl/nnls/) used by [scipy.optimize.nnls](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html) and probably several others.

## Why?

I decided to port nnls to Rust because:

- The C version of nnls is not thread-safe! There are several `static` variables throughout the code which make it unusable in parallel. If you plan to use nnls only in single-thread, you might prefer linking to that version: it has been heavily tested and has no known problem.
- Both Fortran versions (77 and 90) require some knowledge of Fortran memory model and some compiler tricks to [successfuly link against](https://users.rust-lang.org/t/c-fortran-ffi-memory-error/71298/10?u=nil). I wasn't able to do it and I didn't want to waste more hours into this adventure, so I simply ported the script to Rust.

## Advantages

- It has been used several millions of times and it will be used billions of times soon. I work with 3D images and each image I see contains around 1 million voxels. For a specific algorithm, I need to call `nnls` 2 times for each voxel. Several of those images has been tested and compared with the original Fortran version.
- There's no `unsafe` and it is forbidden to use any in this crate.

## Problems

- This is not idiomatic Rust. I tried cleaning and enhancing the code but it's not always possible.
- I use [ndarray](https://github.com/rust-ndarray/ndarray). You might prefer something else. I do not plan to change this but you're welcome to discuss it.
- There are currently 3 `panic!` in the code because there are 3 code paths in the original code which use `goto` that I couldn't translate properly to Rust. I planned to handle those cases but it turns out that they are never called. As I wrote, `nnls` has been extensively used and I couldn't find any dataset that triggers those conditions. If my programs ever panic because of this, it will be repaired quickly.
