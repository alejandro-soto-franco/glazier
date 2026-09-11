# glazier-core

The lattice, the energies and the serial sweep behind
[`glazier`](https://crates.io/crates/glazier).

[![crates.io](https://img.shields.io/crates/v/glazier-core.svg)](https://crates.io/crates/glazier-core)
[![docs.rs](https://docs.rs/glazier-core/badge.svg)](https://docs.rs/glazier-core)

Contact, volume, surface and length energies on a periodic lattice in two or
three dimensions, diffusing species with secretion and uptake, chemotaxis,
persistent motility, division, death, and a local connectivity test. The
Blueprint reader is here too, for the JSON description every engine in the
workspace reads.

The serial sweep is the reference. Every other engine is compared against it.
