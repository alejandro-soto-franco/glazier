//! Every term the engine gained after the third axis went in, exercised in a
//! volume rather than in a plane.
//!
//! The generalisation was meant to be transparent: a plane keeps its old
//! arithmetic because the out-of-plane offsets drop out. These check the other
//! half of that claim, which is that the terms do something in the third axis
//! rather than merely compiling there.

use glazier::Simulation;
use glazier_core::field::{Exchange, Species};
use glazier_core::{CellType, Model};

fn base(seed: u64) -> Model {
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
        neighbour_order: 2,
        seed,
        ..Default::default()
    }
}

/// One cell in the middle of an empty volume.
fn lone_cell(model: Model) -> Simulation {
    let mut sim = Simulation::tiled_grid(model, 8, 1, 1, 1).unwrap();
    for site in 0..sim.lattice.len() {
        let (x, y, z) = sim.lattice.coords(site);
        sim.lattice.labels[site] =
            u32::from((12..20).contains(&x) && (12..20).contains(&y) && (12..20).contains(&z));
    }
    sim.volume = sim.recounted_volumes();
    sim.surface = sim.recounted_surfaces();
    sim.moments = sim.recounted_moments();
    sim
}

fn centroid(sim: &Simulation) -> (f64, f64, f64) {
    let sites: Vec<usize> = sim
        .lattice
        .labels
        .iter()
        .enumerate()
        .filter(|&(_, &l)| l == 1)
        .map(|(site, _)| site)
        .collect();
    let n = sites.len() as f64;
    let coords = |pick: fn((i64, i64, i64)) -> i64| {
        sites
            .iter()
            .map(|&s| pick(sim.lattice.coords(s)) as f64)
            .sum::<f64>()
            / n
    };
    (coords(|c| c.0), coords(|c| c.1), coords(|c| c.2))
}

#[test]
fn a_drift_along_the_third_axis_takes_a_cell_that_way() {
    // A cell of 512 sites moves its centroid by a five-hundredth of a site per
    // accepted copy, so a volume asks more of a drift than a plane does.
    let mut m = base(51);
    m.types[1].external = [0.0, 0.0, 10.0];
    m.types[1].connected = true;
    m.types[1].target_volume = 512.0;

    let mut sim = lone_cell(m);
    let (_, _, z0) = centroid(&sim);
    for _ in 0..400 {
        sim.step();
    }
    let (x1, y1, z1) = centroid(&sim);
    let (x0, y0, _) = centroid(&lone_cell({
        let mut m = base(51);
        m.types[1].connected = true;
        m
    }));

    assert!(z1 - z0 > 1.5, "the cell moved {} along z", z1 - z0);
    assert!(
        (x1 - x0).abs() < 2.0 && (y1 - y0).abs() < 2.0,
        "the cell wandered {} and {} across the drift",
        x1 - x0,
        y1 - y0
    );
}

/// A site in a volume has eighteen neighbours where one in a plane has eight,
/// and the neighbourhood average is a geometric mean that drops to zero as
/// soon as one of them has forgotten. A volume therefore asks for a longer
/// memory than a plane before a cell polarises at all: at `max_activity` 20,
/// which moves a cell three times as far in a plane, a cell in a volume does
/// not move at all.
#[test]
fn memory_moves_a_cell_further_in_a_volume_too() {
    let travel = |max_activity: f64, lambda_activity: f64| {
        let mut m = base(53);
        m.types[1].max_activity = max_activity;
        m.types[1].lambda_activity = lambda_activity;
        let mut sim = lone_cell(m);
        let (x0, y0, z0) = centroid(&sim);
        for _ in 0..400 {
            sim.step();
        }
        let (x1, y1, z1) = centroid(&sim);
        ((x1 - x0).powi(2) + (y1 - y0).powi(2) + (z1 - z0).powi(2)).sqrt()
    };

    let still = travel(0.0, 0.0);
    let short_memory = travel(20.0, 60.0);
    let motile = travel(100.0, 100.0);

    assert!(
        (short_memory - still).abs() < 0.1,
        "a plane's memory moved a cell in a volume: {short_memory} against {still}"
    );
    assert!(
        motile > 3.0 * still,
        "a motile cell went {motile} against {still} for a plain one"
    );
}

#[test]
fn a_cell_divides_across_its_own_axis_in_a_volume() {
    let mut m = base(57);
    m.types[1].division_volume = 400.0;
    m.types[1].target_volume = 900.0;

    let mut sim = lone_cell(m);
    assert_eq!(sim.live_cells(), 1);
    let (divided, died) = sim.divide_and_die();

    assert_eq!(
        divided, 1,
        "a 512-site cell past a 400-site threshold should split"
    );
    assert_eq!(died, 0);
    assert_eq!(sim.live_cells(), 2);
    assert_eq!(sim.volume, sim.recounted_volumes());
    // The cut moved sites rather than making them.
    assert_eq!(sim.volume[1] + sim.volume[2], 512);
}

#[test]
fn death_empties_a_volume_at_the_stated_rate() {
    let mut m = base(59);
    m.types[1].death_rate = 1.0;
    let mut sim = Simulation::tiled(m, 8).unwrap();
    assert_eq!(sim.live_cells(), 64);

    sim.divide_and_die();
    assert_eq!(sim.live_cells(), 0);
    assert!(sim.lattice.labels.iter().all(|&l| l == 0));
}

#[test]
fn a_length_target_stretches_a_cell_through_the_depth() {
    let mut m = base(61);
    m.types[1].target_length = 30.0;
    m.types[1].lambda_length = 1.0;

    let mut long = lone_cell(m);
    let mut round = lone_cell(base(61));
    for _ in 0..150 {
        long.step();
        round.step();
    }
    let axis = |s: &Simulation| s.recounted_moments()[1].length();
    assert!(
        axis(&long) > axis(&round),
        "length went from {} unconstrained to {} at a target of 30",
        axis(&round),
        axis(&long)
    );
}

#[test]
fn a_secreting_cell_fills_a_volume_around_itself() {
    let mut m = base(63);
    m.species = vec![Species {
        name: "signal".into(),
        diffusion: 0.3,
        decay: 0.0,
        initial: 0.0,
    }];
    m.exchange = vec![
        Exchange::default(),
        Exchange {
            secretion: vec![0.5],
            uptake: vec![0.0],
        },
    ];

    let mut sim = lone_cell(m);
    for _ in 0..40 {
        sim.step();
    }

    let above = sim.lattice.index(16, 16, 26);
    assert!(sim.fields.total(0) > 100.0, "total {}", sim.fields.total(0));
    assert!(
        sim.fields.values[0][above] > 0.0,
        "nothing reached ten sites away along z"
    );
}

#[cfg(feature = "cuda")]
mod device {
    use super::*;
    use glazier::cuda::GpuSimulation;

    #[test]
    fn the_device_runs_every_term_in_a_volume() {
        let mut m = base(67);
        m.types[1].connected = true;
        m.types[1].max_activity = 20.0;
        m.types[1].lambda_activity = 60.0;
        m.types[1].external = [0.0, 0.0, 1.0];
        m.types[1].death_rate = 0.002;
        m.types[1].target_surface = 300.0;
        m.types[1].lambda_surface = 0.05;
        m.species = vec![Species {
            name: "signal".into(),
            diffusion: 0.3,
            decay: 0.01,
            initial: 0.0,
        }];
        m.exchange = vec![
            Exchange::default(),
            Exchange {
                secretion: vec![0.2],
                uptake: vec![0.0],
            },
        ];
        m.chemotaxis = vec![vec![0.0], vec![10.0]];

        let seeded = Simulation::tiled(m, 8).unwrap();
        let started = seeded.n_cells();
        let mut gpu = match GpuSimulation::from_cpu(&seeded) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("no device, skipping: {e}");
                return;
            }
        };
        gpu.step(60).unwrap();

        let volumes = gpu.volumes().unwrap();
        let live = volumes[1..].iter().filter(|&&v| v > 0).count();
        assert!(
            live > 0 && live <= started,
            "{live} of {started} cells left"
        );

        // The counters have to match the lattice they wrote, whatever the
        // terms did to it.
        let labels = gpu.labels().unwrap();
        let mut counted = vec![0u32; volumes.len()];
        for &label in &labels {
            counted[label as usize] += 1;
        }
        assert_eq!(&volumes[1..], &counted[1..]);

        let total: f64 = gpu.field(0).unwrap().iter().sum();
        assert!(total > 0.0, "the field never filled");
    }
}
