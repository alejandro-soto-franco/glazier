//! Cells that divide and cells that die.

use glazier::Simulation;
use glazier_core::field::{Exchange, Species};
use glazier_core::{CellType, Model};

fn model(division_volume: f64, death_rate: f64, target_volume: f64) -> Model {
    Model {
        species: Vec::<Species>::new(),
        exchange: vec![Exchange::default(), Exchange::default()],
        width: 64,
        height: 64,
        contact: vec![0.0, 16.0, 16.0, 8.0],
        types: vec![
            CellType {
                target_volume: 0.0,
                lambda_volume: 0.0,
                target_surface: 0.0,
                lambda_surface: 0.0,
                division_volume: 0.0,
                death_rate: 0.0,
                ..Default::default()
            },
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
        seed: 7,
        ..Default::default()
    }
}

#[test]
fn a_growing_cell_divides_once_it_reaches_its_volume() {
    // Cells start at 36 sites, grow towards 100, and split at 72.
    let mut sim = Simulation::tiled_grid(model(72.0, 0.0, 100.0), 6, 4, 4, 1).unwrap();
    let started = sim.live_cells();
    assert_eq!(started, 16);

    for _ in 0..400 {
        sim.step();
    }
    assert!(
        sim.live_cells() > started,
        "no cell divided in 400 steps, {} cells",
        sim.live_cells()
    );
    assert_eq!(sim.volume, sim.recounted_volumes());
    assert_eq!(sim.surface, sim.recounted_surfaces());
}

#[test]
fn a_division_splits_the_volume_between_two_labels() {
    let mut sim = Simulation::tiled_grid(model(40.0, 0.0, 100.0), 8, 2, 2, 1).unwrap();
    let before: u32 = sim.volume[1..].iter().sum();
    let (divided, died) = sim.divide_and_die();

    assert!(
        divided > 0,
        "a 64-site cell past a 40-site threshold should split"
    );
    assert_eq!(died, 0);
    let after: u32 = sim.volume[1..].iter().sum();
    assert_eq!(
        before, after,
        "division moved sites rather than making them"
    );
    assert_eq!(sim.volume, sim.recounted_volumes());
}

#[test]
fn a_dying_cell_leaves_medium_behind() {
    let mut sim = Simulation::tiled_grid(model(0.0, 1.0, 64.0), 8, 2, 2, 1).unwrap();
    assert_eq!(sim.live_cells(), 4);
    let (_, died) = sim.divide_and_die();

    assert_eq!(died, 4, "a death rate of one takes every cell");
    assert_eq!(sim.live_cells(), 0);
    assert!(
        sim.lattice.labels.iter().all(|&l| l == 0),
        "a dead cell left sites behind"
    );
    assert_eq!(sim.volume, sim.recounted_volumes());
}

#[test]
fn a_run_without_either_leaves_the_population_alone() {
    let mut sim = Simulation::tiled(model(0.0, 0.0, 64.0), 8).unwrap();
    let started = sim.live_cells();
    for _ in 0..100 {
        sim.step();
    }
    assert_eq!(sim.live_cells(), started);
}

#[test]
fn death_takes_the_stated_fraction_over_many_steps() {
    let mut sim = Simulation::tiled_grid(model(0.0, 0.02, 64.0), 8, 6, 6, 1).unwrap();
    let started = sim.live_cells() as f64;
    for _ in 0..50 {
        sim.step();
    }
    let left = sim.live_cells() as f64;
    // Survival after fifty steps at two percent a step is about 0.36.
    let expected = started * 0.98_f64.powi(50);
    assert!(
        (left - expected).abs() < 0.35 * started,
        "left {left} against an expected {expected} of {started}"
    );
}
