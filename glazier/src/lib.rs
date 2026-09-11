//! Cellular Potts tissue simulation, with a serial CPU reference and a GPU
//! sweep that has to agree with it.
//!
//! Named after James Glazier, whose work with Graner and Hogeweg applied the
//! Potts lattice to tissue.
//!
//! The lattice, the energies and the serial sweep are `glazier-core`. The
//! device sweep is `glazier-cuda`, behind the `cuda` feature. This crate is
//! the facade over both, and the binary that runs a Blueprint from JSON.

pub use glazier_core::{
    Blueprint, CellType, Lattice, Model, Simulation, blueprint, cpu, lattice, model, npy, rng,
};

#[cfg(feature = "cuda")]
pub use glazier_cuda as cuda;
