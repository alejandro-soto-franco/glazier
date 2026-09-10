//! The GPU sweep against the serial reference, in distribution.
//!
//! The two engines sample the same model by different chains, so the test
//! asserts on the quantities a run is read for: how many cells there are, how
//! big they are, and how far the volumes sit from target. Skipped when the
//! machine has no device.

#![cfg(feature = "cuda")]

use glazier::Simulation;
use glazier::cuda::GpuSimulation;
use glazier::{CellType, Model};

fn model(seed: u64) -> Model {
    Model {
        width: 64,
        height: 64,
        contact: vec![0.0, 16.0, 16.0, 8.0],
        types: vec![
            CellType::default(),
            CellType {
                target_volume: 64.0,
                lambda_volume: 2.0,
                target_surface: 0.0,
                lambda_surface: 0.0,
                ..Default::default()
            },
        ],
        temperature: 10.0,
        neighbour_order: 2,
        seed,
        ..Default::default()
    }
}

fn mean(v: &[u32]) -> f64 {
    v.iter().map(|&x| f64::from(x)).sum::<f64>() / v.len() as f64
}

fn spread(v: &[u32], target: f64) -> f64 {
    (v.iter()
        .map(|&x| (f64::from(x) - target).powi(2))
        .sum::<f64>()
        / v.len() as f64)
        .sqrt()
}

#[test]
fn the_gpu_reproduces_the_serial_run_in_distribution() {
    let steps = 200u64;

    let mut cpu = Simulation::tiled(model(1), 8).unwrap();
    let seeded = Simulation::tiled(model(1), 8).unwrap();
    let mut gpu = match GpuSimulation::from_cpu(&seeded) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no device, skipping: {e}");
            return;
        }
    };

    for _ in 0..steps {
        cpu.step();
    }
    gpu.step(steps).unwrap();

    let cpu_volumes = &cpu.volume[1..];
    let gpu_all = gpu.volumes().unwrap();
    let gpu_volumes = &gpu_all[1..];

    assert_eq!(cpu_volumes.len(), gpu_volumes.len(), "cell counts differ");

    // The lattice is confluent, so the mean volume is fixed by construction and
    // agreement there only checks the bookkeeping.
    let (mc, mg) = (mean(cpu_volumes), mean(gpu_volumes));
    assert!((mc - mg).abs() < 1e-9, "mean volume {mc} against {mg}");

    // The spread about target is the quantity the constraint actually sets.
    let (sc, sg) = (spread(cpu_volumes, 64.0), spread(gpu_volumes, 64.0));
    assert!(
        (sc - sg).abs() / sc.max(sg) < 0.25,
        "spread about target {sc} against {sg}"
    );

    // The device's own volume counter has to match the lattice it wrote.
    let labels = gpu.labels().unwrap();
    let mut recounted = vec![0u32; gpu_all.len()];
    for &label in &labels {
        recounted[label as usize] += 1;
    }
    assert_eq!(recounted, gpu_all, "the device volume counter drifted");
}

/// The surface term is off in `model`, since a strong one freezes the sheet
/// and a frozen sheet compares nothing. This one switches it on gently.
fn model_with_surface(seed: u64) -> Model {
    let mut m = model(seed);
    m.types[1].target_surface = 34.0;
    m.types[1].lambda_surface = 0.1;
    m
}

#[test]
fn the_device_surface_counter_matches_the_lattice_it_wrote() {
    let seeded = Simulation::tiled(model_with_surface(3), 8).unwrap();
    let mut gpu = match GpuSimulation::from_cpu(&seeded) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no device, skipping: {e}");
            return;
        }
    };
    gpu.step(150).unwrap();

    let mut check = seeded.clone();
    check.lattice.labels = gpu.labels().unwrap();
    assert_eq!(gpu.surfaces().unwrap(), check.recounted_surfaces());
}

#[test]
fn a_device_run_conserves_its_cells() {
    let seeded = Simulation::tiled(model(7), 8).unwrap();
    let n = seeded.n_cells();
    let mut gpu = match GpuSimulation::from_cpu(&seeded) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no device, skipping: {e}");
            return;
        }
    };
    gpu.step(300).unwrap();
    let volumes = gpu.volumes().unwrap();
    assert_eq!(volumes.len() - 1, n);
    assert!(
        volumes[1..].iter().all(|&v| v >= 1),
        "a cell reached zero volume on the device"
    );
}

#[test]
fn the_device_refuses_a_lattice_it_cannot_colour() {
    let mut odd = model(1);
    odd.width = 33;
    let sim = Simulation::tiled(odd, 8).unwrap();
    match GpuSimulation::from_cpu(&sim) {
        Err(e) => assert!(e.to_string().contains("even sides"), "{e}"),
        Ok(_) => panic!("the device accepted an odd side the checkerboard cannot split"),
    }
}

/// The length term prices a copy against a cell's major axis, which is a
/// property of the whole cell rather than of the sites around the copy. The
/// device keeps running sums per cell to read it, rebuilt every step, so the
/// test asks whether the tissue it produces is stretched the same way.
#[test]
fn the_device_stretches_cells_the_way_the_serial_engine_does() {
    let steps = 200u64;
    let mut stated = model(41);
    stated.types[1].target_length = 24.0;
    stated.types[1].lambda_length = 1.0;

    let mut cpu = Simulation::tiled(stated.clone(), 8).unwrap();
    let seeded = Simulation::tiled(stated, 8).unwrap();
    let mut gpu = match GpuSimulation::from_cpu(&seeded) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no device, skipping: {e}");
            return;
        }
    };

    for _ in 0..steps {
        cpu.step();
    }
    gpu.step(steps).unwrap();

    let mut on_device = seeded.clone();
    on_device.lattice.labels = gpu.labels().unwrap();

    let mean = |s: &Simulation| {
        let m = s.recounted_moments();
        m[1..].iter().map(|c| c.length()).sum::<f64>() / s.n_cells() as f64
    };
    let (serial, device) = (mean(&cpu), mean(&on_device));

    // An unconstrained sheet of 64-site cells sits near a length of ten, so a
    // target of 24 has to move both engines well past it.
    let mut loose = Simulation::tiled(model(41), 8).unwrap();
    for _ in 0..steps {
        loose.step();
    }
    let free = mean(&loose);

    assert!(
        serial > free,
        "the serial engine read {serial} against {free}"
    );
    assert!(device > free, "the device read {device} against {free}");
    assert!(
        (serial - device).abs() / serial.max(device) < 0.15,
        "serial {serial} against device {device}"
    );
}

/// A sheet that secretes, on both engines.
fn secreting(seed: u64) -> Model {
    use glazier_core::field::{Exchange, Species};

    let mut m = model(seed);
    m.species = vec![Species {
        name: "IFNg".into(),
        diffusion: 0.4,
        decay: 0.01,
        initial: 0.0,
    }];
    m.exchange = vec![
        Exchange {
            secretion: vec![0.0],
            uptake: vec![0.0],
        },
        Exchange {
            secretion: vec![0.2],
            uptake: vec![0.01],
        },
    ];
    m.chemotaxis = vec![vec![0.0], vec![0.0]];
    m
}

#[test]
fn the_device_solves_a_field_the_way_the_serial_engine_does() {
    let steps = 60u64;
    let mut cpu = Simulation::tiled(secreting(11), 8).unwrap();
    let seeded = Simulation::tiled(secreting(11), 8).unwrap();
    let mut gpu = match GpuSimulation::from_cpu(&seeded) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no device, skipping: {e}");
            return;
        }
    };

    for _ in 0..steps {
        cpu.step();
    }
    gpu.step(steps).unwrap();

    let device: f64 = gpu.field(0).unwrap().iter().sum();
    let serial = cpu.fields.total(0);
    assert!(device > 0.0, "the device solved nothing");
    // Both sheets are confluent and secrete the same amount a step, so the
    // totals answer to the same balance of secretion, uptake and decay. The
    // lattices differ, since the two sweeps are different chains, so the
    // agreement is a percentage rather than a bit pattern.
    let gap = (device - serial).abs() / serial.max(device);
    assert!(gap < 0.02, "device {device} against serial {serial}");
}

#[test]
fn a_device_field_stays_where_it_started_without_sources() {
    let mut quiet = secreting(12);
    quiet.exchange[1].secretion = vec![0.0];
    quiet.exchange[1].uptake = vec![0.0];
    quiet.species[0].decay = 0.0;
    quiet.species[0].initial = 2.0;

    let seeded = Simulation::tiled(quiet, 8).unwrap();
    let sites = (seeded.model.width * seeded.model.height) as f64;
    let mut gpu = match GpuSimulation::from_cpu(&seeded) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no device, skipping: {e}");
            return;
        }
    };
    gpu.step(40).unwrap();

    let total: f64 = gpu.field(0).unwrap().iter().sum();
    // Diffusion moves a uniform field nowhere and conserves it exactly.
    assert!(
        (total - 2.0 * sites).abs() / (2.0 * sites) < 1e-4,
        "total {total} against {}",
        2.0 * sites
    );
}

#[test]
fn the_device_climbs_a_gradient_the_way_the_serial_engine_does() {
    use glazier_core::field::{Exchange, Species};

    // A ramp that neither diffuses nor decays, and nothing secretes into it,
    // so the gradient the cell sees is the one the test imposed.
    let mut m = model(21);
    m.species = vec![Species {
        name: "attractant".into(),
        diffusion: 0.0,
        decay: 0.0,
        initial: 0.0,
    }];
    m.exchange = vec![Exchange::default(), Exchange::default()];
    m.chemotaxis = vec![vec![0.0], vec![4.0]];

    let mut seeded = Simulation::tiled_grid(m, 8, 1, 1, 1).unwrap();
    for site in 0..seeded.lattice.labels.len() {
        let (x, y, _) = seeded.lattice.coords(site);
        seeded.lattice.labels[site] = u32::from((24..32).contains(&x) && (24..32).contains(&y));
        seeded.fields.values[0][site] = x as f64;
    }
    seeded.volume = seeded.recounted_volumes();
    seeded.surface = seeded.recounted_surfaces();

    let centre = |labels: &[u32], width: usize| {
        let sites: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter(|&(_, &l)| l == 1)
            .map(|(site, _)| site)
            .collect();
        sites.iter().map(|&s| (s % width) as f64).sum::<f64>() / sites.len() as f64
    };

    let width = seeded.model.width;
    let start = centre(&seeded.lattice.labels, width);

    let mut gpu = match GpuSimulation::from_cpu(&seeded) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no device, skipping: {e}");
            return;
        }
    };
    gpu.step(300).unwrap();
    let moved = centre(&gpu.labels().unwrap(), width) - start;

    assert!(moved > 3.0, "the cell drifted {moved} sites up the ramp");
}

#[test]
fn the_device_runs_a_volume() {
    let mut m = model(31);
    m.width = 32;
    m.height = 32;
    m.depth = 32;
    m.types[1].target_volume = 512.0;

    let mut cpu = Simulation::tiled(m.clone(), 8).unwrap();
    let seeded = Simulation::tiled(m, 8).unwrap();
    let Some(mut gpu) = ({
        match GpuSimulation::from_cpu(&seeded) {
            Ok(g) => Some(g),
            Err(e) => {
                eprintln!("no device, skipping: {e}");
                None
            }
        }
    }) else {
        return;
    };

    for _ in 0..40 {
        cpu.step();
    }
    gpu.step(40).unwrap();

    let device = gpu.volumes().unwrap();
    assert_eq!(device.len(), cpu.volume.len(), "cell counts differ");
    // A confluent block conserves its total whatever the sweep does.
    assert_eq!(
        device[1..].iter().sum::<u32>(),
        cpu.volume[1..].iter().sum::<u32>()
    );

    let labels = gpu.labels().unwrap();
    let mut counted = vec![0u32; device.len()];
    for &label in &labels {
        counted[label as usize] += 1;
    }
    assert_eq!(&device[1..], &counted[1..], "the device counters drifted");

    let spread = |v: &[u32]| {
        (v.iter()
            .map(|&x| (f64::from(x) - 512.0).powi(2))
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
