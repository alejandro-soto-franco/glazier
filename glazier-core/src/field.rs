//! Diffusing chemical fields on the same lattice as the cells.
//!
//! One value per site per species. A step diffuses and decays the field by
//! explicit forward Euler, then lets every cell secrete into the sites it owns
//! and take up from them, which is the order CompuCell3D's solver uses.
//!
//! Explicit diffusion is stable while `D dt / dx^2` stays at or below
//! `1 / (2 n)` for `n` dimensions, so a stated diffusion constant sets the
//! number of sub-steps rather than the other way round: a model states physics
//! and the engine finds a schedule that survives it.

use crate::lattice::Lattice;

/// One diffusing species.
#[derive(Clone, Debug, PartialEq)]
pub struct Species {
    /// The name a description and an engine agree on.
    pub name: String,
    /// Diffusion constant in sites squared per Monte Carlo step.
    pub diffusion: f64,
    /// Fractional decay per Monte Carlo step.
    pub decay: f64,
    /// Uniform value at the start.
    pub initial: f64,
}

/// Secretion and uptake for one cell type, in the order of `Fields::species`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Exchange {
    /// Amount added to each site the cell owns, per step.
    pub secretion: Vec<f64>,
    /// Fraction removed from each site the cell owns, per step.
    pub uptake: Vec<f64>,
}

/// Every species, and the concentrations on the lattice.
#[derive(Clone, Debug)]
pub struct Fields {
    /// The species, in the order a description lists them.
    pub species: Vec<Species>,
    /// One concentration grid per species, indexed as the lattice is.
    pub values: Vec<Vec<f64>>,
    /// Diffusion sub-steps per Monte Carlo step, one per species.
    pub substeps: Vec<usize>,
    scratch: Vec<f64>,
}

/// The largest `D dt / dx^2` an explicit Laplacian survives, by dimension.
#[must_use]
pub fn stability_limit(dimensions: usize) -> f64 {
    1.0 / (2.0 * dimensions.max(1) as f64)
}

impl Fields {
    /// Lay out the fields for a lattice.
    #[must_use]
    pub fn new(species: Vec<Species>, sites: usize, dimensions: usize) -> Self {
        let values = species
            .iter()
            .map(|s| vec![s.initial; sites])
            .collect::<Vec<_>>();
        let substeps = species
            .iter()
            .map(|s| substeps_for(s.diffusion, dimensions))
            .collect();
        Self {
            species,
            values,
            substeps,
            scratch: vec![0.0; sites],
        }
    }

    /// Number of species.
    #[must_use]
    pub fn len(&self) -> usize {
        self.species.len()
    }

    /// Whether the model states no field at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.species.is_empty()
    }

    /// Index of a species by name.
    #[must_use]
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.species.iter().position(|s| s.name == name)
    }

    /// Total amount of a species over the lattice.
    #[must_use]
    pub fn total(&self, index: usize) -> f64 {
        self.values[index].iter().sum()
    }

    /// Diffuse and decay every species by one Monte Carlo step.
    pub fn diffuse(&mut self, lattice: &Lattice) {
        for index in 0..self.species.len() {
            let steps = self.substeps[index];
            let d = self.species[index].diffusion / steps as f64;
            let decay = self.species[index].decay / steps as f64;
            for _ in 0..steps {
                self.one_substep(lattice, index, d, decay);
            }
        }
    }

    fn one_substep(&mut self, lattice: &Lattice, index: usize, d: f64, decay: f64) {
        // The face neighbours alone, which is the Laplacian this scheme is
        // stable for; a plane simply has no pair along the third axis.
        let faces: &[(i64, i64, i64)] = if lattice.depth > 1 {
            &crate::lattice::ORDER1
        } else {
            &crate::lattice::ORDER1[..4]
        };
        {
            let values = &self.values[index];
            for here in 0..values.len() {
                let (x, y, z) = lattice.coords(here);
                let mut laplacian = -(faces.len() as f64) * values[here];
                for &(dx, dy, dz) in faces {
                    laplacian += values[lattice.index(x + dx, y + dy, z + dz)];
                }
                self.scratch[here] = values[here] + d * laplacian - decay * values[here];
            }
        }
        self.values[index].copy_from_slice(&self.scratch);
    }

    /// Secretion and uptake by whichever cell owns each site.
    pub fn exchange(&mut self, lattice: &Lattice, cell_type: &[u8], exchange: &[Exchange]) {
        for (index, values) in self.values.iter_mut().enumerate() {
            for (site, &label) in lattice.labels.iter().enumerate() {
                if label == 0 {
                    continue;
                }
                let spec = &exchange[cell_type[label as usize] as usize];
                let secreted = spec.secretion.get(index).copied().unwrap_or(0.0);
                let taken = spec.uptake.get(index).copied().unwrap_or(0.0);
                values[site] += secreted;
                values[site] -= taken * values[site];
                if values[site] < 0.0 {
                    values[site] = 0.0;
                }
            }
        }
    }
}

/// Sub-steps needed to diffuse at `diffusion` sites squared per step and stay
/// inside the stability limit for the stated dimension count.
#[must_use]
pub fn substeps_for(diffusion: f64, dimensions: usize) -> usize {
    if diffusion <= 0.0 {
        return 1;
    }
    (diffusion / stability_limit(dimensions)).ceil().max(1.0) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(species: Vec<Species>, w: usize, h: usize) -> (Fields, Lattice) {
        (Fields::new(species, w * h, 2), Lattice::medium(w, h, 1, 2))
    }

    fn cube(species: Vec<Species>, side: usize) -> (Fields, Lattice) {
        (
            Fields::new(species, side * side * side, 3),
            Lattice::medium(side, side, side, 2),
        )
    }

    #[test]
    fn a_uniform_field_stays_uniform_under_diffusion() {
        let (mut fields, lattice) = flat(
            vec![Species {
                name: "a".into(),
                diffusion: 0.2,
                decay: 0.0,
                initial: 3.0,
            }],
            16,
            16,
        );
        fields.diffuse(&lattice);
        assert!(fields.values[0].iter().all(|v| (v - 3.0).abs() < 1e-12));
    }

    #[test]
    fn diffusion_conserves_the_total_without_decay() {
        let (mut fields, lattice) = flat(
            vec![Species {
                name: "a".into(),
                diffusion: 0.2,
                decay: 0.0,
                initial: 0.0,
            }],
            16,
            16,
        );
        fields.values[0][8 * 16 + 8] = 100.0;
        let before = fields.total(0);
        for _ in 0..50 {
            fields.diffuse(&lattice);
        }
        assert!(
            (fields.total(0) - before).abs() < 1e-9,
            "{}",
            fields.total(0)
        );
    }

    #[test]
    fn a_point_source_spreads_and_stays_positive() {
        let (mut fields, lattice) = flat(
            vec![Species {
                name: "a".into(),
                diffusion: 0.2,
                decay: 0.0,
                initial: 0.0,
            }],
            32,
            32,
        );
        let centre = 16 * 32 + 16;
        fields.values[0][centre] = 100.0;
        for _ in 0..30 {
            fields.diffuse(&lattice);
        }
        assert!(fields.values[0].iter().all(|&v| v >= 0.0));
        assert!(fields.values[0][centre] < 100.0);
        assert!(fields.values[0][centre + 3] > 0.0);
    }

    #[test]
    fn decay_removes_a_stated_fraction() {
        let (mut fields, lattice) = flat(
            vec![Species {
                name: "a".into(),
                diffusion: 0.0,
                decay: 0.1,
                initial: 1.0,
            }],
            8,
            8,
        );
        fields.diffuse(&lattice);
        assert!((fields.values[0][0] - 0.9).abs() < 1e-12);
    }

    #[test]
    fn a_large_diffusion_constant_takes_more_substeps() {
        assert_eq!(substeps_for(0.0, 2), 1);
        assert_eq!(substeps_for(0.25, 2), 1);
        assert_eq!(substeps_for(0.26, 2), 2);
        assert_eq!(substeps_for(2.0, 2), 8);
    }

    #[test]
    fn a_third_dimension_tightens_the_limit() {
        assert!((stability_limit(2) - 0.25).abs() < 1e-12);
        assert!((stability_limit(3) - 1.0 / 6.0).abs() < 1e-12);
        // The same constant needs half again as many sub-steps in a volume.
        assert_eq!(substeps_for(1.0, 2), 4);
        assert_eq!(substeps_for(1.0, 3), 6);
    }

    #[test]
    fn diffusion_conserves_the_total_in_a_volume() {
        let (mut fields, lattice) = cube(
            vec![Species {
                name: "a".into(),
                diffusion: 0.15,
                decay: 0.0,
                initial: 0.0,
            }],
            12,
        );
        fields.values[0][lattice.index(6, 6, 6)] = 100.0;
        let before = fields.total(0);
        for _ in 0..40 {
            fields.diffuse(&lattice);
        }
        assert!((fields.total(0) - before).abs() < 1e-9);
        assert!(
            fields.values[0][lattice.index(6, 6, 7)] > 0.0,
            "nothing moved in z"
        );
    }

    #[test]
    fn a_field_stays_stable_at_a_diffusion_constant_far_past_the_limit() {
        let (mut fields, lattice) = flat(
            vec![Species {
                name: "a".into(),
                diffusion: 10.0,
                decay: 0.0,
                initial: 0.0,
            }],
            32,
            32,
        );
        fields.values[0][16 * 32 + 16] = 100.0;
        for _ in 0..20 {
            fields.diffuse(&lattice);
        }
        assert!(
            fields.values[0]
                .iter()
                .all(|v| v.is_finite() && *v >= -1e-12),
            "an explicit solver past its limit blows up rather than merely losing accuracy"
        );
    }

    #[test]
    fn secretion_and_uptake_follow_the_owning_cell() {
        let mut lattice = Lattice::medium(8, 8, 1, 2);
        lattice.labels[0] = 1;
        let mut fields = Fields::new(
            vec![Species {
                name: "a".into(),
                diffusion: 0.0,
                decay: 0.0,
                initial: 1.0,
            }],
            64,
            2,
        );
        let exchange = vec![
            Exchange::default(),
            Exchange {
                secretion: vec![2.0],
                uptake: vec![0.5],
            },
        ];
        fields.exchange(&lattice, &[0, 1], &exchange);
        // The owned site takes 1 + 2 = 3, then loses half of it.
        assert!((fields.values[0][0] - 1.5).abs() < 1e-12);
        // Every other site is medium and untouched.
        assert!((fields.values[0][1] - 1.0).abs() < 1e-12);
    }
}
