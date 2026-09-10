//! A cell that follows a gradient.

use glazier::Simulation;
use glazier_core::field::{Exchange, Species};
use glazier_core::{Blueprint, CellType, Model};

fn model(sensitivity: f64) -> Model {
    Model {
        species: vec![Species {
            name: "attractant".into(),
            // The gradient is imposed and held, so the species neither
            // diffuses nor decays and the test measures the cell alone.
            diffusion: 0.0,
            decay: 0.0,
            initial: 0.0,
        }],
        exchange: vec![Exchange::default(), Exchange::default()],
        chemotaxis: vec![vec![0.0], vec![sensitivity]],
        width: 64,
        height: 64,
        contact: vec![0.0, 16.0, 16.0, 8.0],
        types: vec![
            CellType::default(),
            CellType {
                target_volume: 64.0,
                lambda_volume: 2.0,
                ..Default::default()
            },
        ],
        temperature: 10.0,
        neighbour_order: 2,
        seed: 4,
        ..Default::default()
    }
}

/// A linear ramp in x, steady through the run.
fn ramp(sim: &mut Simulation, slope: f64) {
    for site in 0..sim.lattice.labels.len() {
        let (x, _, _) = sim.lattice.coords(site);
        sim.fields.values[0][site] = slope * x as f64;
    }
}

fn centroid_x(sim: &Simulation, label: u32) -> f64 {
    let sites: Vec<usize> = sim
        .lattice
        .labels
        .iter()
        .enumerate()
        .filter(|&(_, &l)| l == label)
        .map(|(site, _)| site)
        .collect();
    sites
        .iter()
        .map(|&site| sim.lattice.coords(site).0 as f64)
        .sum::<f64>()
        / sites.len() as f64
}

fn drift(sensitivity: f64, steps: usize) -> f64 {
    // One cell in the middle of an otherwise empty lattice, so nothing but the
    // gradient decides where it goes.
    let mut sim = Simulation::tiled_grid(model(sensitivity), 8, 1, 1, 1).unwrap();
    for site in 0..sim.lattice.labels.len() {
        let (x, y, _) = sim.lattice.coords(site);
        if (24..32).contains(&x) && (24..32).contains(&y) {
            sim.lattice.labels[site] = 1;
        } else {
            sim.lattice.labels[site] = 0;
        }
    }
    sim.volume = sim.recounted_volumes();
    sim.surface = sim.recounted_surfaces();

    let start = centroid_x(&sim, 1);
    for _ in 0..steps {
        ramp(&mut sim, 1.0);
        sim.step();
    }
    centroid_x(&sim, 1) - start
}

#[test]
fn a_sensitive_cell_climbs_the_gradient() {
    let moved = drift(4.0, 300);
    assert!(
        moved > 3.0,
        "the cell drifted {moved} sites up a rising ramp"
    );
}

#[test]
fn a_negative_sensitivity_runs_the_other_way() {
    let moved = drift(-4.0, 300);
    assert!(moved < -3.0, "the cell drifted {moved} sites");
}

#[test]
fn without_a_sensitivity_the_cell_stays_put() {
    let moved = drift(0.0, 300).abs();
    assert!(
        moved < 3.0,
        "the cell drifted {moved} sites with no sensitivity"
    );
}

#[test]
fn the_work_reads_the_two_sites_the_copy_runs_between() {
    let mut sim = Simulation::tiled_grid(model(2.0), 8, 2, 2, 1).unwrap();
    sim.fields.values[0][10] = 5.0;
    sim.fields.values[0][11] = 1.0;

    // Moving into the richer site is cheap, and into the poorer one costly.
    assert!((sim.chemotaxis_work(1, 10, 11) + 8.0).abs() < 1e-12);
    assert!((sim.chemotaxis_work(1, 11, 10) - 8.0).abs() < 1e-12);
    // The medium follows nothing.
    assert_eq!(sim.chemotaxis_work(0, 10, 11), 0.0);
}

#[test]
fn a_description_states_its_sensitivities() {
    let text = r#"{
        "name": "chemotaxis",
        "width": 32, "height": 32,
        "types": [{ "name": "Cell", "target_volume": 64.0, "lambda_volume": 2.0,
                    "chemotaxis": {"attractant": 300.0} }],
        "fields": [{ "name": "attractant", "diffusion": 0.2 }],
        "contact": [0.0, 16.0, 16.0, 8.0],
        "temperature": 10.0, "neighbour_order": 2, "seed": 1,
        "steps": 10, "dump_every": 0,
        "initial": { "side": 8, "nx": 2, "ny": 2 },
        "units": { "micron_per_site": 1.0, "minute_per_step": 1.0 }
    }"#;
    let model = Blueprint::from_json(text).unwrap().model();
    assert_eq!(model.chemotaxis[0], vec![0.0]);
    assert_eq!(model.chemotaxis[1], vec![300.0]);
    assert!(model.has_chemotaxis());
}
