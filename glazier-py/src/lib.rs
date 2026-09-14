//! Python bindings: read a description, run it, take the lattice back.
//!
//! The Python side reads the same Blueprint through `glazier.blueprint`, so
//! `parse` exists to be compared against it rather than to be convenient. A
//! field that drifts between the two readers is the failure the exchange
//! format exists to prevent.

use glazier_core::blueprint::Blueprint;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn parse_blueprint(text: &str) -> PyResult<Blueprint> {
    Blueprint::from_json(text).map_err(PyValueError::new_err)
}

/// What this engine read from a description.
#[pyfunction]
fn parse(py: Python<'_>, text: &str) -> PyResult<Py<PyDict>> {
    let bp = parse_blueprint(text)?;
    let model = bp.model();
    let out = PyDict::new(py);
    out.set_item("name", bp.name)?;
    out.set_item("width", model.width)?;
    out.set_item("height", model.height)?;
    out.set_item("depth", model.depth)?;
    out.set_item("n_types", model.n_types())?;
    out.set_item("contact", model.contact.clone())?;
    out.set_item(
        "target_volume",
        model
            .types
            .iter()
            .map(|t| t.target_volume)
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "lambda_volume",
        model
            .types
            .iter()
            .map(|t| t.lambda_volume)
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "target_surface",
        model
            .types
            .iter()
            .map(|t| t.target_surface)
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "lambda_surface",
        model
            .types
            .iter()
            .map(|t| t.lambda_surface)
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "target_length",
        model
            .types
            .iter()
            .map(|t| t.target_length)
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "lambda_length",
        model
            .types
            .iter()
            .map(|t| t.lambda_length)
            .collect::<Vec<_>>(),
    )?;
    out.set_item("chemotaxis", model.chemotaxis.clone())?;
    out.set_item(
        "division_volume",
        model
            .types
            .iter()
            .map(|t| t.division_volume)
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "death_rate",
        model.types.iter().map(|t| t.death_rate).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "species",
        model
            .species
            .iter()
            .map(|s| s.name.clone())
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "diffusion",
        model
            .species
            .iter()
            .map(|s| s.diffusion)
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "decay",
        model.species.iter().map(|s| s.decay).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "secretion",
        model
            .exchange
            .iter()
            .map(|e| e.secretion.clone())
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "uptake",
        model
            .exchange
            .iter()
            .map(|e| e.uptake.clone())
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "lambda_nematic",
        model
            .types
            .iter()
            .map(|t| t.lambda_nematic)
            .collect::<Vec<_>>(),
    )?;
    out.set_item("nematic_field", bp.nematic_field.clone())?;
    out.set_item("temperature", model.temperature)?;
    out.set_item("neighbour_order", model.neighbour_order)?;
    out.set_item("seed", model.seed)?;
    out.set_item("steps", bp.steps)?;
    out.set_item(
        "initial",
        vec![bp.initial.side, bp.initial.nx, bp.initial.ny, bp.initial.nz],
    )?;
    out.set_item("micron_per_site", bp.units.micron_per_site)?;
    out.set_item("minute_per_step", bp.units.minute_per_step)?;
    Ok(out.into())
}

/// Run a description on the serial engine and return the lattice.
///
/// The labels come back flat with `x` fastest and `z` slowest, so the caller
/// reshapes to `(depth, height, width)`.
#[pyfunction]
#[pyo3(signature = (text, steps = None))]
fn run(text: &str, steps: Option<u64>) -> PyResult<(Vec<u32>, usize, usize, usize)> {
    let bp = parse_blueprint(text)?;
    let mut sim = bp
        .simulation(std::path::Path::new("."))
        .map_err(PyValueError::new_err)?;
    for _ in 0..steps.unwrap_or(bp.steps) {
        sim.step();
    }
    Ok((sim.lattice.labels.clone(), bp.width, bp.height, bp.depth))
}

/// Per-cell volumes after a run, index 0 the medium.
#[pyfunction]
#[pyo3(signature = (text, steps = None))]
fn run_volumes(text: &str, steps: Option<u64>) -> PyResult<Vec<u32>> {
    let bp = parse_blueprint(text)?;
    let mut sim = bp
        .simulation(std::path::Path::new("."))
        .map_err(PyValueError::new_err)?;
    for _ in 0..steps.unwrap_or(bp.steps) {
        sim.step();
    }
    Ok(sim.volume)
}

/// A run kept on the device between calls, for coupling to another solver.
///
/// The kernel compiles once and the lattice stays on the GPU, where the
/// command-line engine starts a process, compiles and uploads for every run.
/// A caller alternates `step`, `labels` and `set_nematic_field`.
#[cfg(feature = "cuda")]
#[pyclass(unsendable)]
struct GpuSession {
    sim: glazier::cuda::GpuSimulation,
    sites: usize,
}

#[cfg(feature = "cuda")]
#[pymethods]
impl GpuSession {
    /// Build from a description; relative `.npy` paths resolve against `base`.
    #[new]
    #[pyo3(signature = (text, base = "."))]
    fn new(text: &str, base: &str) -> PyResult<Self> {
        let bp = parse_blueprint(text)?;
        let sim = bp
            .simulation(std::path::Path::new(base))
            .map_err(PyValueError::new_err)?;
        let sites = bp.width * bp.height * bp.depth;
        let gpu = glazier::cuda::GpuSimulation::from_cpu(&sim)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { sim: gpu, sites })
    }

    /// Run `steps` Monte Carlo steps.
    fn step(&mut self, py: Python<'_>, steps: u64) -> PyResult<()> {
        py.detach(|| self.sim.step(steps))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// The lattice, flat with `x` fastest.
    fn labels(&self) -> PyResult<Vec<u32>> {
        self.sim
            .labels()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Replace the field from a flat list of `2 * sites` numbers, (Q_xx, Q_xy) per site.
    fn set_nematic_field(&mut self, flat: Vec<f64>) -> PyResult<()> {
        if flat.len() != 2 * self.sites {
            return Err(PyValueError::new_err(format!(
                "expected {} numbers, got {}",
                2 * self.sites,
                flat.len()
            )));
        }
        let pairs: Vec<[f64; 2]> = flat.chunks_exact(2).map(|c| [c[0], c[1]]).collect();
        self.sim
            .set_nematic_field(&pairs)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Monte Carlo steps completed.
    #[getter]
    fn mcs(&self) -> u64 {
        self.sim.mcs
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse, m)?)?;
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(run_volumes, m)?)?;
    #[cfg(feature = "cuda")]
    m.add_class::<GpuSession>()?;
    Ok(())
}
