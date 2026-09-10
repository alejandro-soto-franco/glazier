//! What the serial engine has to satisfy before a parallel one is worth writing.

use glazier::Simulation;
use glazier::rng::Xoshiro;
use glazier::{CellType, Model};

fn model(width: usize, height: usize) -> Model {
    Model {
        width,
        height,
        // Medium against cell 16, cell against cell 8: unlike cells stick to
        // each other more than either sticks to the medium, so a tiled sheet
        // stays confluent.
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
        seed: 42,
        ..Default::default()
    }
}

#[test]
fn the_incremental_delta_matches_a_full_recomputation() {
    let mut sim = Simulation::tiled(model(32, 32), 8).unwrap();
    let mut rng = Xoshiro::seed(5);

    for _ in 0..40 {
        sim.step();
        for _ in 0..50 {
            let target = rng.below(sim.lattice.labels.len() as u64) as usize;
            let source = rng.below(sim.lattice.labels.len() as u64) as usize;
            let new = sim.lattice.labels[source];
            let old = sim.lattice.labels[target];
            if old == new || (old != 0 && sim.volume[old as usize] <= 1) {
                continue;
            }

            let predicted = sim.delta_energy(target, new);

            let before = sim.energy();
            let mut after_sim = sim.clone();
            after_sim.lattice.labels[target] = new;
            if old != 0 {
                after_sim.volume[old as usize] -= 1;
            }
            if new != 0 {
                after_sim.volume[new as usize] += 1;
            }
            let measured = after_sim.energy() - before;

            assert!(
                (predicted - measured).abs() < 1e-9,
                "delta {predicted} against {measured} at site {target}"
            );
        }
    }
}

fn model_with_surface(width: usize, height: usize) -> Model {
    let mut m = model(width, height);
    // A square of 64 sites has 32 boundary bonds at neighbour order 2; asking
    // for fewer pulls the cell towards a rounder outline.
    m.types[1].target_surface = 28.0;
    m.types[1].lambda_surface = 1.0;
    m
}

#[test]
fn the_incremental_delta_matches_a_full_recomputation_with_a_surface_term() {
    let mut sim = Simulation::tiled(model_with_surface(32, 32), 8).unwrap();
    let mut rng = Xoshiro::seed(9);

    for _ in 0..30 {
        sim.step();
        for _ in 0..40 {
            let target = rng.below(sim.lattice.labels.len() as u64) as usize;
            let source = rng.below(sim.lattice.labels.len() as u64) as usize;
            let new = sim.lattice.labels[source];
            let old = sim.lattice.labels[target];
            if old == new || (old != 0 && sim.volume[old as usize] <= 1) {
                continue;
            }

            let predicted = sim.delta_energy(target, new);
            let before = sim.energy();
            let mut after_sim = sim.clone();
            after_sim.lattice.labels[target] = new;
            if old != 0 {
                after_sim.volume[old as usize] -= 1;
            }
            if new != 0 {
                after_sim.volume[new as usize] += 1;
            }
            let measured = after_sim.energy() - before;

            assert!(
                (predicted - measured).abs() < 1e-9,
                "delta {predicted} against {measured} at site {target}"
            );
        }
    }
}

#[test]
fn surface_bookkeeping_survives_a_run() {
    let mut sim = Simulation::tiled(model_with_surface(48, 48), 8).unwrap();
    for _ in 0..100 {
        sim.step();
    }
    assert_eq!(sim.surface, sim.recounted_surfaces());
}

#[test]
fn the_surface_constraint_pulls_the_outline_in() {
    let mut loose = Simulation::tiled(model(48, 48), 8).unwrap();
    let mut tight = Simulation::tiled(model_with_surface(48, 48), 8).unwrap();
    for _ in 0..200 {
        loose.step();
        tight.step();
    }
    let mean = |s: &Simulation| {
        s.recounted_surfaces()[1..].iter().sum::<i64>() as f64 / s.n_cells() as f64
    };
    let (a, b) = (mean(&loose), mean(&tight));
    assert!(
        b < a,
        "surface went from {a} unconstrained to {b} constrained"
    );
}

#[test]
fn volume_bookkeeping_survives_a_run() {
    let mut sim = Simulation::tiled(model(48, 48), 8).unwrap();
    for _ in 0..100 {
        sim.step();
    }
    assert_eq!(sim.volume, sim.recounted_volumes());
}

#[test]
fn no_cell_is_lost() {
    let mut sim = Simulation::tiled(model(48, 48), 8).unwrap();
    let n = sim.n_cells();
    for _ in 0..200 {
        sim.step();
    }
    let counts = sim.recounted_volumes();
    assert_eq!(sim.n_cells(), n);
    assert!(
        counts[1..].iter().all(|&v| v >= 1),
        "a cell reached zero volume"
    );
}

#[test]
fn a_seed_reproduces_a_run() {
    let mut a = Simulation::tiled(model(32, 32), 8).unwrap();
    let mut b = Simulation::tiled(model(32, 32), 8).unwrap();
    for _ in 0..25 {
        a.step();
        b.step();
    }
    assert_eq!(a.lattice.labels, b.lattice.labels);
}

#[test]
fn the_volume_constraint_pulls_cells_to_target() {
    // Start from cells of volume 36 against a target of 64, with medium round
    // them to grow into.
    let mut sim = Simulation::tiled_grid(model(64, 64), 6, 6, 6, 1).unwrap();
    let start: f64 =
        sim.volume[1..].iter().map(|&v| f64::from(v)).sum::<f64>() / sim.n_cells() as f64;
    for _ in 0..300 {
        sim.step();
    }
    let end: f64 =
        sim.volume[1..].iter().map(|&v| f64::from(v)).sum::<f64>() / sim.n_cells() as f64;
    assert!((start - 36.0).abs() < 1e-9, "start {start}");
    assert!(
        end > start && end < 64.0 + 8.0,
        "mean volume went {start} to {end}, target 64"
    );
}
