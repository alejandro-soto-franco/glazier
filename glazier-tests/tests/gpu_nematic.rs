//! The nematic coupling on the device, against the serial reference.
//!
//! The chains differ, so the test reads what the term is for: cells line up
//! along a uniform field, by about as much as the serial engine lines them up,
//! and not at all when the coupling is zero. Skipped when the machine has no
//! device.

#![cfg(feature = "cuda")]

use glazier::Simulation;
use glazier::cuda::GpuSimulation;
use glazier::{CellType, Model};

const SIDE: usize = 64;

fn model(lambda_nematic: f64, angle: f64) -> Model {
    Model {
        width: SIDE,
        height: SIDE,
        contact: vec![0.0, 16.0, 16.0, 8.0],
        types: vec![
            CellType::default(),
            CellType {
                target_volume: 64.0,
                lambda_volume: 2.0,
                lambda_nematic,
                ..Default::default()
            },
        ],
        temperature: 10.0,
        neighbour_order: 2,
        seed: 7,
        nematic: vec![[(2.0 * angle).cos(), (2.0 * angle).sin()]; SIDE * SIDE],
        ..Default::default()
    }
}

/// Nematic order and director of the cells in a label field.
fn order(model: &Model, labels: Vec<u32>) -> (f64, f64) {
    let sim = Simulation::from_labels(model.clone(), labels).unwrap();
    let moments = sim.recounted_moments();
    let (mut c, mut s, mut n) = (0.0, 0.0, 0.0);
    for (volume, m) in sim.volume.iter().zip(&moments).skip(1) {
        if *volume < 8 {
            continue;
        }
        let (a0, a1) = m.anisotropy();
        let norm = (a0 * a0 + a1 * a1).sqrt();
        if norm < 1e-9 {
            continue;
        }
        c += a0 / norm;
        s += a1 / norm;
        n += 1.0;
    }
    ((c * c + s * s).sqrt() / n, 0.5 * s.atan2(c))
}

fn run_gpu(model: &Model, steps: u64) -> Option<Vec<u32>> {
    let seeded = Simulation::tiled(model.clone(), 8).unwrap();
    let mut gpu = match GpuSimulation::from_cpu(&seeded) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no device, skipping: {e}");
            return None;
        }
    };
    gpu.step(steps).unwrap();
    Some(gpu.labels().unwrap())
}

#[test]
fn the_device_aligns_cells_along_the_field_as_the_serial_engine_does() {
    let angle = 0.6;
    let steps = 400;
    let coupled = model(6.0, angle);
    let Some(gpu_labels) = run_gpu(&coupled, steps) else {
        return;
    };
    let (s_gpu, director) = order(&coupled, gpu_labels);

    let mut cpu = Simulation::tiled(coupled.clone(), 8).unwrap();
    for _ in 0..steps {
        cpu.step();
    }
    let (s_cpu, _) = order(&coupled, cpu.lattice.labels.clone());

    let off = (director - angle)
        .abs()
        .min(std::f64::consts::PI - (director - angle).abs());
    assert!(s_gpu > 0.7, "device order {s_gpu}");
    assert!(
        off < 0.15,
        "device director {director} against the field's {angle}"
    );
    assert!(
        (s_gpu - s_cpu).abs() < 0.15,
        "device {s_gpu} against serial {s_cpu}"
    );

    let blind = model(0.0, angle);
    let (s_blind, _) = order(&blind, run_gpu(&blind, steps).unwrap());
    assert!(s_blind < 0.4, "uncoupled device order {s_blind}");
}
