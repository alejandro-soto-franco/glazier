# glazier

Cellular Potts tissue simulation on the CPU and the GPU, in Rust.

[![crates.io](https://img.shields.io/crates/v/glazier.svg)](https://crates.io/crates/glazier)
[![docs.rs](https://docs.rs/glazier/badge.svg)](https://docs.rs/glazier)

This is the facade crate: it re-exports `glazier-core`, adds `glazier-cuda`
behind the `cuda` feature, and ships the `glazier` binary that runs a model
description from JSON.

```bash
cargo install glazier --features cuda
glazier --model monolayer.json --out runs/gpu --engine gpu
```

The engine keeps a sheet or a block of cells on a periodic lattice under
contact, volume, surface and length energies, with diffusing chemical fields,
chemotaxis, persistent motility, division, death and a connectivity veto. The
device runs every term the serial reference does, 300 times faster on a 4096 by
4096 lattice.

Full documentation, figures and the exchange-format work are in the
[repository](https://github.com/alejandro-soto-franco/glazier).

Named after James Glazier, whose work with Graner and Hogeweg made the Potts
lattice a model of tissue.
