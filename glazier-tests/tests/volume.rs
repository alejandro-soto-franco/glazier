//! The same engine on a three-dimensional lattice.

use glazier::Simulation;
use glazier_core::field::{Exchange, Species};
use glazier_core::rng::Xoshiro;
use glazier_core::{CellType, Model};

fn model(order: u8) -> Model {
    Model {
        width: 32,
        height: 32,
        depth: 32,
        contact: vec![0.0, 16.0, 16.0, 8.0],
        types: vec![
            CellType::default(),
            CellType {
                target_volume: 512.0,
                lambda_volume: 2.0,
                ..Default::default()
            },
        ],
        temperature: 10.0,
        neighbour_order: order,
        seed: 13,
        ..Default::default()
    }
}

#[test]
fn a_volume_tiles_into_cubes() {
    let sim = Simulation::tiled(model(2), 8).unwrap();
    assert_eq!(sim.n_cells(), 64, "four cubes on a side");
    assert!(
        sim.volume[1..].iter().all(|&v| v == 512),
        "a cube of side eight is 512 sites"
    );
}

#[test]
fn the_incremental_delta_matches_a_full_recomputation() {
    let mut sim = Simulation::tiled(model(2), 8).unwrap();
    let mut rng = Xoshiro::seed(17);

    for _ in 0..5 {
        sim.step();
        for _ in 0..15 {
            let target = rng.below(sim.lattice.len() as u64) as usize;
            let source = rng.below(sim.lattice.len() as u64) as usize;
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
            let measured = after.energy() - before;

            assert!(
                (predicted - measured).abs() < 1e-9,
                "delta {predicted} against {measured} at site {target}"
            );
        }
    }
}

#[test]
fn the_bookkeeping_survives_a_run_in_a_volume() {
    let mut sim = Simulation::tiled(model(2), 8).unwrap();
    for _ in 0..20 {
        sim.step();
    }
    assert_eq!(sim.volume, sim.recounted_volumes());
    assert_eq!(sim.surface, sim.recounted_surfaces());
}

#[test]
fn a_cell_reaches_across_the_third_axis() {
    // With eighteen neighbours a copy can travel in z, so a sheet started one
    // layer deep spreads through the depth.
    let mut sim = Simulation::tiled_grid(model(2), 8, 4, 4, 1).unwrap();
    let start_depth = |s: &Simulation| {
        s.lattice
            .labels
            .iter()
            .enumerate()
            .filter(|&(_, &l)| l != 0)
            .map(|(site, _)| s.lattice.coords(site).2)
            .max()
            .unwrap_or(0)
    };
    assert_eq!(start_depth(&sim), 7, "the tiling starts eight layers deep");
    for _ in 0..30 {
        sim.step();
    }
    assert!(
        start_depth(&sim) > 7,
        "nothing moved out of the starting slab"
    );
}

#[test]
fn a_field_diffuses_through_the_depth() {
    let mut m = model(2);
    m.species = vec![Species {
        name: "signal".into(),
        diffusion: 0.3,
        decay: 0.0,
        initial: 0.0,
    }];
    m.exchange = vec![Exchange::default(); 2];

    let mut sim = Simulation::tiled(m, 8).unwrap();
    let centre = sim.lattice.index(16, 16, 16);
    sim.fields.values[0][centre] = 1000.0;
    let before = sim.fields.total(0);
    for _ in 0..20 {
        sim.step();
    }

    assert!(
        (sim.fields.total(0) - before).abs() < 1e-6,
        "diffusion lost mass"
    );
    let above = sim.lattice.index(16, 16, 19);
    assert!(sim.fields.values[0][above] > 0.0, "nothing reached along z");
}

#[test]
fn a_higher_neighbour_order_gives_a_cell_more_boundary() {
    let faces = Simulation::tiled(model(1), 8).unwrap();
    let corners = Simulation::tiled(model(3), 8).unwrap();
    let mean = |s: &Simulation| s.surface[1..].iter().sum::<i64>() as f64 / s.n_cells() as f64;
    assert!(
        mean(&corners) > mean(&faces),
        "twenty-six neighbours gave {} against six giving {}",
        mean(&corners),
        mean(&faces)
    );
}
