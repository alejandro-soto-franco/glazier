//! Cells that stay in one piece.

use glazier::Simulation;
use glazier_core::{CellType, Model};

/// A sheet at a temperature high enough to tear cells apart when nothing stops
/// it, so the test has something to measure.
fn model(connected: bool, temperature: f64) -> Model {
    Model {
        width: 64,
        height: 64,
        contact: vec![0.0, 4.0, 4.0, 4.0],
        types: vec![
            CellType::default(),
            CellType {
                target_volume: 64.0,
                lambda_volume: 1.0,
                connected,
                ..Default::default()
            },
        ],
        temperature,
        neighbour_order: 2,
        seed: 23,
        ..Default::default()
    }
}

/// Pieces each label falls into under the eight-neighbour rule.
fn pieces(sim: &Simulation) -> usize {
    let mut seen = vec![false; sim.lattice.len()];
    let mut total = 0usize;
    for start in 0..sim.lattice.len() {
        let label = sim.lattice.labels[start];
        if label == 0 || seen[start] {
            continue;
        }
        total += 1;
        let mut stack = vec![start];
        seen[start] = true;
        while let Some(site) = stack.pop() {
            let (x, y, z) = sim.lattice.coords(site);
            for &(dx, dy, dz) in &glazier_core::connectivity::ring(sim.lattice.depth) {
                let n = sim.lattice.index(x + dx, y + dy, z + dz);
                if !seen[n] && sim.lattice.labels[n] == label {
                    seen[n] = true;
                    stack.push(n);
                }
            }
        }
    }
    total
}

#[test]
fn a_tiled_sheet_starts_in_one_piece_per_cell() {
    let sim = Simulation::tiled(model(true, 10.0), 8).unwrap();
    assert_eq!(pieces(&sim), sim.n_cells());
}

#[test]
fn the_constraint_keeps_cells_whole_where_nothing_else_does() {
    let steps = 200;
    let mut loose = Simulation::tiled(model(false, 40.0), 8).unwrap();
    let mut whole = Simulation::tiled(model(true, 40.0), 8).unwrap();
    for _ in 0..steps {
        loose.step();
        whole.step();
    }

    let cells = loose.n_cells();
    let torn = pieces(&loose);
    let kept = pieces(&whole);

    assert!(
        torn > cells,
        "the unconstrained sheet stayed whole at {torn} pieces for {cells} cells, so the \
         test measures nothing"
    );
    assert_eq!(
        kept, cells,
        "a constrained cell broke: {kept} pieces for {cells} cells"
    );
}

#[test]
fn the_constraint_leaves_the_population_and_the_volume_alone() {
    let mut sim = Simulation::tiled(model(true, 20.0), 8).unwrap();
    let started = sim.n_cells();
    for _ in 0..100 {
        sim.step();
    }
    assert_eq!(sim.n_cells(), started);
    assert_eq!(sim.volume, sim.recounted_volumes());
    assert_eq!(sim.surface, sim.recounted_surfaces());
}

#[test]
fn a_volume_keeps_its_cells_whole_too() {
    let mut m = model(true, 40.0);
    m.width = 32;
    m.height = 32;
    m.depth = 32;
    m.types[1].target_volume = 512.0;

    let mut sim = Simulation::tiled(m, 8).unwrap();
    let cells = sim.n_cells();
    for _ in 0..40 {
        sim.step();
    }
    assert_eq!(pieces(&sim), cells, "a cell broke in three dimensions");
}

#[cfg(feature = "cuda")]
mod device {
    use super::*;
    use glazier::cuda::GpuSimulation;

    #[test]
    fn the_device_keeps_cells_whole_where_nothing_else_does() {
        let steps = 200u64;
        let loose_seed = Simulation::tiled(model(false, 40.0), 8).unwrap();
        let whole_seed = Simulation::tiled(model(true, 40.0), 8).unwrap();
        let cells = whole_seed.n_cells();

        let (Ok(mut loose), Ok(mut whole)) = (
            GpuSimulation::from_cpu(&loose_seed),
            GpuSimulation::from_cpu(&whole_seed),
        ) else {
            eprintln!("no device, skipping");
            return;
        };
        loose.step(steps).unwrap();
        whole.step(steps).unwrap();

        let mut torn_sim = loose_seed.clone();
        torn_sim.lattice.labels = loose.labels().unwrap();
        let mut whole_sim = whole_seed.clone();
        whole_sim.lattice.labels = whole.labels().unwrap();

        let torn = pieces(&torn_sim);
        assert!(
            torn > cells,
            "the unconstrained device sheet stayed whole at {torn} pieces, so the test \
             measures nothing"
        );
        assert_eq!(
            pieces(&whole_sim),
            cells,
            "a constrained cell broke on the device"
        );
    }

    #[test]
    fn the_device_and_the_serial_engine_agree_on_a_constrained_sheet() {
        let steps = 150u64;
        let mut cpu = Simulation::tiled(model(true, 20.0), 8).unwrap();
        let seeded = Simulation::tiled(model(true, 20.0), 8).unwrap();
        let Ok(mut gpu) = GpuSimulation::from_cpu(&seeded) else {
            eprintln!("no device, skipping");
            return;
        };

        for _ in 0..steps {
            cpu.step();
        }
        gpu.step(steps).unwrap();

        let device = gpu.volumes().unwrap();
        assert_eq!(
            device[1..].iter().sum::<u32>(),
            cpu.volume[1..].iter().sum::<u32>(),
            "a confluent sheet conserves its sites on either engine"
        );

        let spread = |v: &[u32]| {
            (v.iter()
                .map(|&x| (f64::from(x) - 64.0).powi(2))
                .sum::<f64>()
                / v.len() as f64)
                .sqrt()
        };
        let (sc, sg) = (spread(&cpu.volume[1..]), spread(&device[1..]));
        assert!(
            (sc - sg).abs() / sc.max(sg) < 0.3,
            "spread about target {sc} against {sg}"
        );
    }
}
