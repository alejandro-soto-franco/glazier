//! Division and death on the device, against the serial engine.

#![cfg(feature = "cuda")]

use glazier::Simulation;
use glazier::cuda::GpuSimulation;
use glazier_core::{CellType, Model};

fn model(division_volume: f64, death_rate: f64, target_volume: f64, seed: u64) -> Model {
    Model {
        width: 64,
        height: 64,
        contact: vec![0.0, 16.0, 16.0, 8.0],
        types: vec![
            CellType::default(),
            CellType {
                target_volume,
                lambda_volume: 2.0,
                division_volume,
                death_rate,
                ..Default::default()
            },
        ],
        temperature: 10.0,
        neighbour_order: 2,
        seed,
        ..Default::default()
    }
}

fn device(sim: &Simulation) -> Option<GpuSimulation> {
    match GpuSimulation::from_cpu(sim) {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("no device, skipping: {e}");
            None
        }
    }
}

fn live(volumes: &[u32]) -> usize {
    volumes[1..].iter().filter(|&&v| v > 0).count()
}

#[test]
fn the_device_kills_cells_at_the_stated_rate() {
    let seeded = Simulation::tiled_grid(model(0.0, 0.02, 64.0, 5), 8, 6, 6, 1).unwrap();
    let started = seeded.n_cells() as f64;
    let Some(mut gpu) = device(&seeded) else {
        return;
    };
    gpu.step(50).unwrap();

    let left = live(&gpu.volumes().unwrap()) as f64;
    let expected = started * 0.98_f64.powi(50);
    assert!(
        (left - expected).abs() < 0.35 * started,
        "left {left} against an expected {expected} of {started}"
    );
}

#[test]
fn a_dead_cell_leaves_medium_on_the_device() {
    let seeded = Simulation::tiled_grid(model(0.0, 1.0, 64.0, 6), 8, 2, 2, 1).unwrap();
    let Some(mut gpu) = device(&seeded) else {
        return;
    };
    gpu.step(1).unwrap();

    assert_eq!(
        live(&gpu.volumes().unwrap()),
        0,
        "a death rate of one takes every cell"
    );
    assert!(
        gpu.labels().unwrap().iter().all(|&l| l == 0),
        "a dead cell left sites behind"
    );
}

#[test]
fn the_device_divides_and_conserves_the_sites() {
    let seeded = Simulation::tiled_grid(model(40.0, 0.0, 100.0, 7), 8, 4, 4, 1).unwrap();
    let started = seeded.n_cells();
    let Some(mut gpu) = device(&seeded) else {
        return;
    };
    gpu.step(1).unwrap();

    let volumes = gpu.volumes().unwrap();
    assert!(
        live(&volumes) > started,
        "no cell divided, {} of {started}",
        live(&volumes)
    );
    let occupied: u32 = volumes[1..].iter().sum();
    let counted = gpu.labels().unwrap().iter().filter(|&&l| l != 0).count() as u32;
    assert_eq!(
        occupied, counted,
        "division moved sites rather than making them"
    );
}

#[test]
fn the_device_counters_match_the_lattice_they_wrote() {
    let seeded = Simulation::tiled_grid(model(72.0, 0.005, 100.0, 8), 6, 6, 6, 1).unwrap();
    let Some(mut gpu) = device(&seeded) else {
        return;
    };
    gpu.step(120).unwrap();

    let labels = gpu.labels().unwrap();
    let volumes = gpu.volumes().unwrap();
    let mut counted = vec![0u32; volumes.len()];
    for &label in &labels {
        counted[label as usize] += 1;
    }
    assert_eq!(&volumes[1..], &counted[1..volumes.len()]);
}

#[test]
fn both_engines_lose_cells_at_the_rate_the_description_states() {
    // Two independent draws of the same rate, so the test asks each of them to
    // land near the expected survival rather than near the other. At a hundred
    // cells the binomial spread is about five, and the band is three of those.
    let steps = 80u64;
    let rate = 0.01;
    let mut cpu = Simulation::tiled_grid(model(0.0, rate, 64.0, 9), 8, 10, 10, 1).unwrap();
    let seeded = Simulation::tiled_grid(model(0.0, rate, 64.0, 9), 8, 10, 10, 1).unwrap();
    let started = seeded.n_cells() as f64;
    let Some(mut gpu) = device(&seeded) else {
        return;
    };

    for _ in 0..steps {
        cpu.step();
    }
    gpu.step(steps).unwrap();

    let survival = (1.0 - rate).powi(steps as i32);
    let expected = started * survival;
    let spread = (started * survival * (1.0 - survival)).sqrt();

    for (name, left) in [
        ("serial", cpu.live_cells() as f64),
        ("device", live(&gpu.volumes().unwrap()) as f64),
    ] {
        assert!(
            (left - expected).abs() < 3.0 * spread,
            "{name} left {left} against an expected {expected} give or take {spread}"
        );
    }
}
