//! The model description both engines read.
//!
//! One file states the lattice, the cell types, the contact matrix, the
//! initial condition, the schedule and the units. Anything an engine needs and
//! this does not state is a gap in the format rather than a detail of the
//! engine.

use crate::field::{Exchange, Species};
use crate::model::{CellType, Model};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// What one lattice site and one Monte Carlo step stand for.
///
/// A Potts lattice has no physical scale of its own, so a description that
/// omits this cannot be compared with a model in microns and minutes.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Units {
    /// Physical length of one lattice site.
    pub micron_per_site: f64,
    /// Physical duration of one Monte Carlo step.
    pub minute_per_step: f64,
}

/// The depth a description that says nothing about it means.
fn one() -> usize {
    1
}

/// Adhesion molecules and how strongly each pair of them binds.
///
/// A cell type states how much of each molecule it presents, and the binding
/// matrix says what a pair of them is worth. The contact energy between two
/// types is then the stated one less what their molecules bind, which is a
/// function of the two types alone and so collapses into the contact matrix
/// before an engine ever sees it.
///
/// Per-cell adhesion, where two cells of one type present different amounts
/// and the amounts change as the cell runs, is a different thing and is not
/// this.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Adhesion {
    /// The molecules, in the order the binding matrix indexes them.
    pub molecules: Vec<String>,
    /// Binding energies, row-major over the molecules and symmetric. A
    /// positive entry binds, which lowers the contact energy.
    pub binding: Vec<f64>,
}

/// A diffusing species as the description states it.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SpeciesSpec {
    /// The name every engine refers to it by.
    pub name: String,
    /// Diffusion constant in sites squared per Monte Carlo step.
    pub diffusion: f64,
    /// Fractional decay per step.
    #[serde(default)]
    pub decay: f64,
    /// Uniform concentration at the start.
    #[serde(default)]
    pub initial: f64,
}

/// A cell type as the description states it.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TypeSpec {
    /// The name an engine shows.
    pub name: String,
    /// Target volume in lattice sites.
    pub target_volume: f64,
    /// Volume constraint strength.
    pub lambda_volume: f64,
    /// Amount of each named species added to every site the cell owns, per
    /// step. A species the map leaves out is not secreted.
    #[serde(default)]
    pub secretion: BTreeMap<String, f64>,
    /// Fraction of each named species removed from every site the cell owns,
    /// per step.
    #[serde(default)]
    pub uptake: BTreeMap<String, f64>,
    /// Sensitivity to each named species' gradient. Positive climbs it.
    #[serde(default)]
    pub chemotaxis: BTreeMap<String, f64>,
    /// How much of each adhesion molecule this type presents.
    #[serde(default)]
    pub presents: BTreeMap<String, f64>,
    /// Target surface in boundary bonds. Zero with `lambda_surface` zero
    /// leaves the constraint out, which is what a description that says
    /// nothing about surface means.
    #[serde(default)]
    pub target_surface: f64,
    /// Surface constraint strength.
    #[serde(default)]
    pub lambda_surface: f64,
    /// Volume at which a cell divides, in sites. Absent never divides.
    #[serde(default)]
    pub division_volume: f64,
    /// Probability per step that a cell of this type dies. Absent never dies.
    #[serde(default)]
    pub death_rate: f64,
    /// Target major axis in sites. Absent leaves the length unconstrained.
    #[serde(default)]
    pub target_length: f64,
    /// Length constraint strength.
    #[serde(default)]
    pub lambda_length: f64,
    /// Refuse a copy that would locally pinch a cell of this type in two.
    #[serde(default)]
    pub connected: bool,
    /// Steps of memory a site keeps after this cell takes it.
    #[serde(default)]
    pub max_activity: f64,
    /// Strength of the persistence that memory buys.
    #[serde(default)]
    pub lambda_activity: f64,
    /// A constant drift, stated per axis.
    #[serde(default)]
    pub external: [f64; 3],
}

/// The initial condition. Squares laid down from the origin, all of the first
/// type unless `fractions` says otherwise.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Initial {
    /// Side of each square in sites.
    pub side: usize,
    /// Squares across.
    pub nx: usize,
    /// Squares down.
    pub ny: usize,
    /// Cubes through the depth. Absent is one layer.
    #[serde(default = "one")]
    pub nz: usize,
    /// Fraction of the cells to start as each named type. What the map leaves
    /// unassigned starts as the first type the description lists, so an
    /// infection seeded at two percent states one number.
    #[serde(default)]
    pub fractions: BTreeMap<String, f64>,
}

/// The whole description.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Blueprint {
    /// A name for the model.
    pub name: String,
    /// Lattice width in sites.
    pub width: usize,
    /// Lattice height in sites.
    pub height: usize,
    /// Lattice depth in sites. Absent is a plane.
    #[serde(default = "one")]
    pub depth: usize,
    /// Cell types beside the medium, in order.
    pub types: Vec<TypeSpec>,
    /// Diffusing species, in order.
    #[serde(default)]
    pub fields: Vec<SpeciesSpec>,
    /// Adhesion molecules and their binding matrix, absent when the contact
    /// matrix says everything.
    #[serde(default)]
    pub adhesion: Option<Adhesion>,
    /// Contact energies over `["Medium", types...]`, row-major and symmetric.
    pub contact: Vec<f64>,
    /// Metropolis fluctuation amplitude.
    pub temperature: f64,
    /// 1 for four neighbours, 2 for eight.
    pub neighbour_order: u8,
    /// Random seed.
    pub seed: u64,
    /// Monte Carlo steps to run.
    pub steps: u64,
    /// Steps between label-field dumps.
    pub dump_every: u64,
    /// The initial condition.
    pub initial: Initial,
    /// The physical scale.
    pub units: Units,
}

impl Blueprint {
    /// The contact matrix with the adhesion molecules folded in.
    ///
    /// Binding lowers the energy of a bond, so what two types' molecules bind
    /// comes off the contact energy the description states between them.
    #[must_use]
    pub fn effective_contact(&self) -> Vec<f64> {
        let mut contact = self.contact.clone();
        let Some(adhesion) = &self.adhesion else {
            return contact;
        };

        let n_types = self.types.len() + 1;
        let presented = |index: usize| -> Vec<f64> {
            if index == 0 {
                return vec![0.0; adhesion.molecules.len()];
            }
            adhesion
                .molecules
                .iter()
                .map(|name| {
                    self.types[index - 1]
                        .presents
                        .get(name)
                        .copied()
                        .unwrap_or(0.0)
                })
                .collect()
        };

        let molecules: Vec<Vec<f64>> = (0..n_types).map(presented).collect();
        let n = adhesion.molecules.len();
        for a in 0..n_types {
            for b in 0..n_types {
                let mut bound = 0.0;
                for i in 0..n {
                    for j in 0..n {
                        bound += adhesion.binding[i * n + j] * molecules[a][i] * molecules[b][j];
                    }
                }
                contact[a * n_types + b] -= bound;
            }
        }
        contact
    }

    /// Type index of each cell at the start, by the fractions stated.
    ///
    /// Assignment walks the cells in order and hands out each type its share,
    /// so the same description and the same cell count always start the same
    /// way whatever an engine's own random stream does.
    #[must_use]
    pub fn initial_types(&self, cells: usize) -> Vec<u8> {
        let mut types = vec![1u8; cells];
        let mut next = 0usize;
        for (index, spec) in self.types.iter().enumerate() {
            let Some(&fraction) = self.initial.fractions.get(&spec.name) else {
                continue;
            };
            let share = ((fraction * cells as f64).round() as usize).min(cells - next);
            for slot in types.iter_mut().skip(next).take(share) {
                *slot = index as u8 + 1;
            }
            next += share;
        }
        types
    }

    /// The engine-facing model this description states.
    #[must_use]
    pub fn model(&self) -> Model {
        let mut types = vec![CellType::default()];
        types.extend(self.types.iter().map(|t| CellType {
            target_volume: t.target_volume,
            lambda_volume: t.lambda_volume,
            target_surface: t.target_surface,
            lambda_surface: t.lambda_surface,
            division_volume: t.division_volume,
            death_rate: t.death_rate,
            target_length: t.target_length,
            lambda_length: t.lambda_length,
            connected: t.connected,
            max_activity: t.max_activity,
            lambda_activity: t.lambda_activity,
            external: t.external,
        }));
        let species: Vec<Species> = self
            .fields
            .iter()
            .map(|f| Species {
                name: f.name.clone(),
                diffusion: f.diffusion,
                decay: f.decay,
                initial: f.initial,
            })
            .collect();

        let rates = |map: &BTreeMap<String, f64>| -> Vec<f64> {
            species
                .iter()
                .map(|s| map.get(&s.name).copied().unwrap_or(0.0))
                .collect()
        };
        let mut exchange = vec![Exchange {
            secretion: vec![0.0; species.len()],
            uptake: vec![0.0; species.len()],
        }];
        exchange.extend(self.types.iter().map(|t| Exchange {
            secretion: rates(&t.secretion),
            uptake: rates(&t.uptake),
        }));

        let mut chemotaxis = vec![vec![0.0; species.len()]];
        chemotaxis.extend(self.types.iter().map(|t| rates(&t.chemotaxis)));

        Model {
            species,
            exchange,
            chemotaxis,
            contact: self.effective_contact(),
            width: self.width,
            height: self.height,
            depth: self.depth,
            types,
            temperature: self.temperature,
            neighbour_order: self.neighbour_order,
            seed: self.seed,
        }
    }

    /// Read a description from JSON.
    ///
    /// # Errors
    /// A message naming the parse failure or the inconsistency found. A
    /// secretion or uptake entry naming a species the description does not
    /// state is an error rather than a silent zero, since a typo there would
    /// otherwise read as a cell that secretes nothing.
    pub fn from_json(text: &str) -> Result<Self, String> {
        let bp: Self = serde_json::from_str(text).map_err(|e| e.to_string())?;
        let names: Vec<&str> = bp.types.iter().map(|t| t.name.as_str()).collect();
        for name in bp.initial.fractions.keys() {
            if !names.contains(&name.as_str()) {
                return Err(format!(
                    "the initial condition names the type {name}, which the description does \
                     not state"
                ));
            }
        }
        if let Some(adhesion) = &bp.adhesion {
            let n = adhesion.molecules.len();
            if adhesion.binding.len() != n * n {
                return Err(format!(
                    "the binding matrix is {} entries for {n} molecules, want {}",
                    adhesion.binding.len(),
                    n * n
                ));
            }
            for spec in &bp.types {
                for name in spec.presents.keys() {
                    if !adhesion.molecules.contains(name) {
                        return Err(format!(
                            "type {} presents {name}, which is not a molecule this description \
                             declares",
                            spec.name
                        ));
                    }
                }
            }
        } else {
            for spec in &bp.types {
                if !spec.presents.is_empty() {
                    return Err(format!(
                        "type {} presents adhesion molecules, and the description declares none",
                        spec.name
                    ));
                }
            }
        }

        let known: Vec<&str> = bp.fields.iter().map(|f| f.name.as_str()).collect();
        for spec in &bp.types {
            for (what, map) in [
                ("secretion", &spec.secretion),
                ("uptake", &spec.uptake),
                ("chemotaxis", &spec.chemotaxis),
            ] {
                for name in map.keys() {
                    if !known.contains(&name.as_str()) {
                        return Err(format!(
                            "type {} states {what} of {name}, which is not a field this \
                             description declares",
                            spec.name
                        ));
                    }
                }
            }
        }
        bp.model().validate()?;
        Ok(bp)
    }
}
