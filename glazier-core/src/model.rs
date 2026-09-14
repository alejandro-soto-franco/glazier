//! Model description: cell types, the contact matrix and the constraints.

/// A cell type's own parameters. Type 0 is the medium and takes no volume
/// constraint.
///
/// Every term is off at zero, so `CellType::default()` is the medium and a
/// type states only what it uses. New terms therefore leave existing
/// descriptions and existing callers alone.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CellType {
    /// Target volume in lattice sites.
    pub target_volume: f64,
    /// Volume constraint strength.
    pub lambda_volume: f64,
    /// Target surface in boundary bonds.
    pub target_surface: f64,
    /// Surface constraint strength.
    pub lambda_surface: f64,
    /// Volume at which a cell divides. Zero never divides.
    pub division_volume: f64,
    /// Probability per step that a cell of this type dies. Zero never dies.
    pub death_rate: f64,
    /// Target major axis in sites. Zero leaves the length unconstrained.
    pub target_length: f64,
    /// Length constraint strength.
    pub lambda_length: f64,
    /// Whether a copy that would locally pinch this cell is refused.
    pub connected: bool,
    /// Steps of memory a site keeps after this cell takes it. Zero leaves the
    /// cell with no persistence.
    pub max_activity: f64,
    /// Strength of the persistence.
    pub lambda_activity: f64,
    /// A constant drift, as a force per axis. The work is the displacement
    /// along it, so a cell with one travels that way whatever its neighbours
    /// do.
    pub external: [f64; 3],
    /// Strength of the coupling between a cell's elongation and the nematic
    /// field it sits in. Zero leaves the cell blind to the field.
    pub lambda_nematic: f64,
}

/// The whole model: what a Blueprint would state, in one struct.
#[derive(Clone, Debug)]
pub struct Model {
    /// Diffusing species, in the order a description lists them.
    pub species: Vec<crate::field::Species>,
    /// Secretion and uptake per cell type, index 0 the medium.
    pub exchange: Vec<crate::field::Exchange>,
    /// Chemotactic sensitivity per cell type and species, index 0 the medium.
    /// Positive climbs a gradient.
    pub chemotaxis: Vec<Vec<f64>>,
    /// Lattice width in sites.
    pub width: usize,
    /// Lattice height in sites.
    pub height: usize,
    /// Lattice depth in sites. One is a plane, and every neighbourhood then
    /// drops its out-of-plane offsets.
    pub depth: usize,
    /// Contact energies, row-major over `(type_a, type_b)`, symmetric.
    pub contact: Vec<f64>,
    /// One entry per type, index 0 the medium.
    pub types: Vec<CellType>,
    /// Metropolis fluctuation amplitude, the Potts temperature.
    pub temperature: f64,
    /// 1 for the four von Neumann neighbours, 2 for the eight Moore ones.
    pub neighbour_order: u8,
    /// Random seed.
    pub seed: u64,
    /// A nematic field given per site as `(Q_xx, Q_xy)`, the independent
    /// components of a traceless symmetric tensor in the plane. Empty when the
    /// model has none. It is an input the run reads and never changes; a
    /// caller that evolves it, such as a continuum nematic solver, updates it
    /// between runs.
    pub nematic: Vec<[f64; 2]>,
}

impl Model {
    /// Number of cell types, counting the medium.
    #[must_use]
    pub fn n_types(&self) -> usize {
        self.types.len()
    }

    /// How many dimensions the lattice spans.
    #[must_use]
    pub fn dimensions(&self) -> usize {
        if self.depth > 1 { 3 } else { 2 }
    }

    /// Whether any type keeps a memory of where it has been.
    #[must_use]
    pub fn has_motility(&self) -> bool {
        self.types
            .iter()
            .any(|t| t.lambda_activity != 0.0 && t.max_activity > 0.0)
    }

    /// Whether any type drifts.
    #[must_use]
    pub fn has_external_potential(&self) -> bool {
        self.types
            .iter()
            .any(|t| t.external.iter().any(|&v| v != 0.0))
    }

    /// Whether any type refuses a copy that would pinch it.
    #[must_use]
    pub fn has_connectivity(&self) -> bool {
        self.types.iter().any(|t| t.connected)
    }

    /// Whether any type constrains its length, so a run without one skips the
    /// moment bookkeeping.
    #[must_use]
    pub fn has_length_constraint(&self) -> bool {
        self.types.iter().any(|t| t.lambda_length != 0.0)
    }

    /// Whether a nematic field is present and some type couples to it.
    #[must_use]
    pub fn has_nematic(&self) -> bool {
        !self.nematic.is_empty() && self.types.iter().any(|t| t.lambda_nematic != 0.0)
    }

    /// Whether the run keeps per-cell second moments, which the length term
    /// and the nematic term both read.
    #[must_use]
    pub fn tracks_moments(&self) -> bool {
        self.has_length_constraint() || self.has_nematic()
    }

    /// Whether any type follows a gradient.
    #[must_use]
    pub fn has_chemotaxis(&self) -> bool {
        self.chemotaxis
            .iter()
            .any(|row| row.iter().any(|&l| l != 0.0))
    }

    /// Whether any type divides or dies, so a run without either skips the
    /// per-step scan over labels.
    #[must_use]
    pub fn has_population_events(&self) -> bool {
        self.types
            .iter()
            .any(|t| t.division_volume > 0.0 || t.death_rate > 0.0)
    }

    /// Contact energy between two types.
    #[must_use]
    pub fn contact_energy(&self, a: u8, b: u8) -> f64 {
        self.contact[a as usize * self.n_types() + b as usize]
    }

    /// Check the shapes agree before a run starts.
    ///
    /// # Errors
    /// A message naming the first inconsistency found.
    pub fn validate(&self) -> Result<(), String> {
        let n = self.n_types();
        if n == 0 {
            return Err("a model needs at least the medium type".into());
        }
        if self.contact.len() != n * n {
            return Err(format!(
                "contact matrix is {} entries for {n} types, want {}",
                self.contact.len(),
                n * n
            ));
        }
        for a in 0..n {
            for b in 0..n {
                let (ab, ba) = (self.contact[a * n + b], self.contact[b * n + a]);
                if (ab - ba).abs() > 1e-12 {
                    return Err(format!("contact matrix is asymmetric at ({a}, {b})"));
                }
            }
        }
        if self.width == 0 || self.height == 0 || self.depth == 0 {
            return Err("the lattice has a zero dimension".into());
        }
        if !matches!(self.neighbour_order, 1..=3) {
            return Err(format!(
                "neighbour order {} is outside the range one to three",
                self.neighbour_order
            ));
        }
        if self.temperature <= 0.0 {
            return Err("the temperature must be positive".into());
        }
        let sites = self.width * self.height * self.depth;
        if !self.nematic.is_empty() && self.nematic.len() != sites {
            return Err(format!(
                "the nematic field has {} sites for a lattice of {sites}",
                self.nematic.len()
            ));
        }
        if !self.exchange.is_empty() && self.exchange.len() != n {
            return Err(format!(
                "exchange has {} entries for {n} types",
                self.exchange.len()
            ));
        }
        if !self.chemotaxis.is_empty() && self.chemotaxis.len() != n {
            return Err(format!(
                "chemotaxis has {} entries for {n} types",
                self.chemotaxis.len()
            ));
        }
        for (index, row) in self.chemotaxis.iter().enumerate() {
            if !row.is_empty() && row.len() != self.species.len() {
                return Err(format!(
                    "type {index} states {} chemotactic sensitivities for {} species",
                    row.len(),
                    self.species.len()
                ));
            }
        }
        for (index, spec) in self.exchange.iter().enumerate() {
            for (what, list) in [("secretion", &spec.secretion), ("uptake", &spec.uptake)] {
                if !list.is_empty() && list.len() != self.species.len() {
                    return Err(format!(
                        "type {index} states {} {what} rates for {} species",
                        list.len(),
                        self.species.len()
                    ));
                }
            }
        }
        Ok(())
    }
}

impl Default for Model {
    /// A one-site medium-only lattice at a temperature that runs. It exists so
    /// a caller states the terms it cares about and leaves the rest, which is
    /// what keeps a new energy term from touching every call site.
    fn default() -> Self {
        Self {
            species: Vec::new(),
            exchange: Vec::new(),
            chemotaxis: Vec::new(),
            width: 1,
            height: 1,
            depth: 1,
            contact: vec![0.0],
            types: vec![CellType::default()],
            temperature: 10.0,
            neighbour_order: 2,
            seed: 0,
            nematic: Vec::new(),
        }
    }
}
