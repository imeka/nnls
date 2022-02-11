# nnls

This is a Rust version of nnls (Non-Negative Least Square). It's a port from a Fortran90 script used by [scipy.optimize.nnls](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html) and probablys everal others.

## Why?

I decided to port nnls to Rust because:

- The C version of nnls is not thread-safe! There are several `static` variables throughout the code which it unusable in parallel. If you plan to use nnls only in single-thread, you might prefer linkking to that version: it has been heavily tested and has no known problem.
- Both Fortran versions (77 and 90) require some knowledge of Fortran memory model and some compiler tricks to [successfuly link against](https://users.rust-lang.org/t/c-fortran-ffi-memory-error/71298/10?u=nil). I wasn't able to do it and I didn't want to waste more hours into this adventure, so I simply ported the script to Rust.

## Advantages

- It has been used several millions of times and it will be used billions of times quite soon. I work with 3D images and each image I see contains around 1 million voxels. For a specific algorithm, I just call `nnls` 2 times for each voxel.
- There's no `unsafe`

## Problems

- I use [ndarray](https://github.com/rust-ndarray/ndarray). You might prefer something else. I do not plan to change this but you're welcome to discuss it.
- I currently receive reference to owned data instead of views. Just open an issue or a PR if you need views.
- There are currently 3 `panic!` in the code because there are 3 code paths in the original code which use `goto` that I couldn't translate properly to Rust. I planned to handle those cases but it
turns out that ther are never called. As I wrote, `nnls` has been extensively used and I couldn't find any dataset that triggers those conditions. If my programs ever panic because of this, I'll found a solution!