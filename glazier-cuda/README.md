# glazier-cuda

The checkerboard cellular Potts sweep on the GPU, behind
[`glazier`](https://crates.io/crates/glazier).

[![crates.io](https://img.shields.io/crates/v/glazier-cuda.svg)](https://crates.io/crates/glazier-cuda)
[![docs.rs](https://docs.rs/glazier-cuda/badge.svg)](https://docs.rs/glazier-cuda)

Copy attempts run in parallel by lattice colour, four in a plane and eight in a
volume, so same-colour targets stay two apart on every axis and no two attempts
read a site the other is writing. The field solver, chemotaxis, division, death
and the length term all run on the device as well.

cudarc dlopens the driver at run time, so this crate builds with no CUDA
headers and no linking. A device is needed only to run it.
