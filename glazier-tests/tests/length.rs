//! The length constraint, and the bookkeeping under it.

use glazier::Simulation;
use glazier_core::rng::Xoshiro;
use glazier_core::{CellType, Model};

fn model(target_length: f64, lambda_length: f64) -> Model {
    Model {
        width: 48,
        height: 48,
        contact: vec![0.0, 16.0, 16.0, 8.0],
        types: vec![
            CellType::default(),
            CellType {
                target_volume: 64.0,
                lambda_volume: 2.0,
                target_length,
                lambda_length,
                ..Default::default()
            },
        ],
        temperature: 10.0,
        neighbour_order: 2,
        seed: 3,
        ..Default::default()
    }
}

#[test]
fn the_incremental_delta_matches_a_full_recomputation() {
    let mut sim = Simulation::tiled(model(20.0, 1.0), 8).unwrap();
    let mut rng = Xoshiro::seed(2);

    for _ in 0..25 {
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

            let mut after = sim.clone();
            after.lattice.labels[target] = new;
            after.volume[old as usize] -= 1;
            after.volume[new as usize] += 1;
            let (x, y, z) = after.lattice.coords(target);
            let extent = (
                after.lattice.width as f64,
                after.lattice.height as f64,
                after.lattice.depth as f64,
            );
            let site = (x as f64, y as f64, z as f64);
            let leaving = after.moments[old as usize].unwrap(site.0, site.1, site.2, extent);
            after.moments[old as usize].remove(leaving);
            let joining = after.moments[new as usize].unwrap(site.0, site.1, site.2, extent);
            after.moments[new as usize].add(joining);

            let measured = after.energy() - before;
            assert!(
                (predicted - measured).abs() < 1e-9,
                "delta {predicted} against {measured} at site {target}"
            );
        }
    }
}

#[test]
fn moment_bookkeeping_survives_a_run() {
    let mut sim = Simulation::tiled(model(20.0, 1.0), 8).unwrap();
    for _ in 0..80 {
        sim.step();
    }
    for (label, counted) in sim.recounted_moments().iter().enumerate().skip(1) {
        let kept = sim.moments[label];
        assert!(
            (kept.length() - counted.length()).abs() < 1e-6,
            "cell {label} kept {} against a recounted {}",
            kept.length(),
            counted.length()
        );
    }
}

#[test]
fn a_length_target_stretches_the_cells() {
    let mut round = Simulation::tiled(model(0.0, 0.0), 8).unwrap();
    let mut long = Simulation::tiled(model(24.0, 1.0), 8).unwrap();
    for _ in 0..300 {
        round.step();
        long.step();
    }
    let mean = |s: &Simulation| {
        let m = s.recounted_moments();
        m[1..].iter().map(|c| c.length()).sum::<f64>() / s.n_cells() as f64
    };
    let (a, b) = (mean(&round), mean(&long));
    assert!(
        b > a,
        "length went from {a} unconstrained to {b} at a target of 24"
    );
}
