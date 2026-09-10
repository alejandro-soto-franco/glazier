//! A cell that keeps moving, and a cell that drifts.

use glazier::Simulation;
use glazier_core::{CellType, Model};

/// `connected` is stated per test rather than fixed, since the two terms want
/// different regimes: a drifting cell has to stay whole to be dragged, and a
/// cell driven hard by its own memory can wrap medium into a ring that the
/// connectivity veto then locks.
fn model(max_activity: f64, lambda_activity: f64, external: [f64; 3], connected: bool) -> Model {
    Model {
        width: 96,
        height: 96,
        contact: vec![0.0, 16.0, 16.0, 8.0],
        types: vec![
            CellType::default(),
            CellType {
                target_volume: 64.0,
                lambda_volume: 2.0,
                connected,
                max_activity,
                lambda_activity,
                external,
                ..Default::default()
            },
        ],
        temperature: 10.0,
        neighbour_order: 2,
        seed: 29,
        ..Default::default()
    }
}

/// One cell in the middle of an empty lattice, so nothing but the terms under
/// test decides where it goes.
fn lone_cell(model: Model) -> Simulation {
    let mut sim = Simulation::tiled_grid(model, 8, 1, 1, 1).unwrap();
    for site in 0..sim.lattice.len() {
        let (x, y, _) = sim.lattice.coords(site);
        sim.lattice.labels[site] = u32::from((44..52).contains(&x) && (44..52).contains(&y));
    }
    sim.volume = sim.recounted_volumes();
    sim.surface = sim.recounted_surfaces();
    sim
}

fn centroid(sim: &Simulation) -> (f64, f64) {
    let sites: Vec<usize> = sim
        .lattice
        .labels
        .iter()
        .enumerate()
        .filter(|&(_, &l)| l == 1)
        .map(|(site, _)| site)
        .collect();
    let n = sites.len() as f64;
    (
        sites
            .iter()
            .map(|&s| sim.lattice.coords(s).0 as f64)
            .sum::<f64>()
            / n,
        sites
            .iter()
            .map(|&s| sim.lattice.coords(s).1 as f64)
            .sum::<f64>()
            / n,
    )
}

/// How far the cell travels from where it started, over `steps`.
fn displacement(model: Model, steps: usize) -> f64 {
    let mut sim = lone_cell(model);
    let (x0, y0) = centroid(&sim);
    for _ in 0..steps {
        sim.step();
    }
    let (x1, y1) = centroid(&sim);
    ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt()
}

#[test]
fn memory_makes_a_cell_travel_further_than_it_jiggles() {
    let steps = 400;
    let still = displacement(model(0.0, 0.0, [0.0; 3], false), steps);
    let motile = displacement(model(20.0, 60.0, [0.0; 3], false), steps);
    assert!(
        motile > 3.0 * still,
        "a motile cell went {motile} against {still} for a plain one"
    );
}

#[test]
fn a_motile_cell_holds_its_volume() {
    let mut sim = lone_cell(model(20.0, 60.0, [0.0; 3], false));
    for _ in 0..400 {
        sim.step();
    }
    let volume = f64::from(sim.volume[1]);
    assert!(
        (volume - 64.0).abs() < 16.0,
        "the cell reached {volume} against a target of 64"
    );
}

#[test]
fn a_drift_takes_a_cell_the_way_it_points() {
    let steps = 300;
    let mut sim = lone_cell(model(0.0, 0.0, [3.0, 0.0, 0.0], true));
    let (x0, y0) = centroid(&sim);
    for _ in 0..steps {
        sim.step();
    }
    let (x1, y1) = centroid(&sim);
    assert!(x1 - x0 > 5.0, "the cell moved {} along x", x1 - x0);
    assert!(
        (y1 - y0).abs() < 4.0,
        "the cell wandered {} across the drift",
        y1 - y0
    );
}

#[test]
fn a_reversed_drift_reverses_the_travel() {
    let steps = 300;
    let mut sim = lone_cell(model(0.0, 0.0, [-3.0, 0.0, 0.0], true));
    let (x0, _) = centroid(&sim);
    for _ in 0..steps {
        sim.step();
    }
    let (x1, _) = centroid(&sim);
    assert!(x1 - x0 < -5.0, "the cell moved {} along x", x1 - x0);
}

#[test]
fn memory_decays_to_nothing_when_a_cell_stops_being_taken() {
    let mut sim = lone_cell(model(5.0, 60.0, [0.0; 3], false));
    for _ in 0..20 {
        sim.step();
    }
    assert!(!sim.activity.is_quiet(), "a moving cell should remember");

    // Freeze the sheet by taking the memory away and letting it run down.
    sim.model.types[1].lambda_activity = 0.0;
    for _ in 0..10 {
        sim.activity.decay();
    }
    assert!(sim.activity.is_quiet(), "memory should have run out");
}

#[test]
fn a_run_with_neither_term_leaves_the_memory_empty() {
    let mut sim = lone_cell(model(0.0, 0.0, [0.0; 3], false));
    for _ in 0..50 {
        sim.step();
    }
    assert!(sim.activity.is_quiet());
}

#[cfg(feature = "cuda")]
mod device {
    use super::*;
    use glazier::cuda::GpuSimulation;

    fn travelled(model: Model, steps: u64) -> Option<f64> {
        let seeded = lone_cell(model);
        let (x0, y0) = centroid(&seeded);
        let mut gpu = match GpuSimulation::from_cpu(&seeded) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("no device, skipping: {e}");
                return None;
            }
        };
        gpu.step(steps).unwrap();

        let mut after = seeded.clone();
        after.lattice.labels = gpu.labels().unwrap();
        let (x1, y1) = centroid(&after);
        Some(((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt())
    }

    #[test]
    fn the_device_makes_a_cell_travel_on_its_memory() {
        let steps = 400;
        let (Some(still), Some(motile)) = (
            travelled(model(0.0, 0.0, [0.0; 3], false), steps),
            travelled(model(20.0, 60.0, [0.0; 3], false), steps),
        ) else {
            return;
        };
        assert!(
            motile > 3.0 * still,
            "a motile cell went {motile} on the device against {still} for a plain one"
        );
    }

    #[test]
    fn the_device_drifts_a_cell_the_way_it_points() {
        let steps = 300;
        let seeded = lone_cell(model(0.0, 0.0, [3.0, 0.0, 0.0], true));
        let (x0, _) = centroid(&seeded);
        let Ok(mut gpu) = GpuSimulation::from_cpu(&seeded) else {
            eprintln!("no device, skipping");
            return;
        };
        gpu.step(steps).unwrap();

        let mut after = seeded.clone();
        after.lattice.labels = gpu.labels().unwrap();
        let (x1, _) = centroid(&after);
        assert!(x1 - x0 > 5.0, "the cell moved {} along x", x1 - x0);
    }
}
