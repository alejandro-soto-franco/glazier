//! Wall-clock for the same model on both engines.

use glazier::Simulation;
use glazier::{CellType, Model};
use std::time::Instant;

fn model(size: usize) -> Model {
    Model {
        width: size,
        height: size,
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
        seed: 1,
        ..Default::default()
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let steps: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(100);
    // Sizes follow the step count, so a sweep widens without a rebuild.
    let sizes: Vec<usize> = {
        let stated: Vec<usize> = args.filter_map(|s| s.parse().ok()).collect();
        if stated.is_empty() {
            vec![128, 256, 512, 1024]
        } else {
            stated
        }
    };

    println!(
        "{:>6}  {:>7}  {:>12}  {:>12}  {:>8}",
        "size", "cells", "cpu (s)", "gpu (s)", "speedup"
    );
    for size in sizes {
        let mut cpu = Simulation::tiled(model(size), 8).unwrap();
        let cells = cpu.n_cells();

        let t = Instant::now();
        for _ in 0..steps {
            cpu.step();
        }
        let cpu_s = t.elapsed().as_secs_f64();

        #[cfg(feature = "cuda")]
        let gpu_s = {
            let seeded = Simulation::tiled(model(size), 8).unwrap();
            match glazier::cuda::GpuSimulation::from_cpu(&seeded) {
                Ok(mut g) => {
                    // One step first, so the compile and the upload stay out of
                    // the measurement.
                    g.step(1).unwrap();
                    let t = Instant::now();
                    g.step(steps).unwrap();
                    t.elapsed().as_secs_f64()
                }
                Err(e) => {
                    eprintln!("no device: {e}");
                    f64::NAN
                }
            }
        };
        #[cfg(not(feature = "cuda"))]
        let gpu_s = f64::NAN;

        println!(
            "{size:>6}  {cells:>7}  {cpu_s:>12.3}  {gpu_s:>12.3}  {:>8.1}",
            cpu_s / gpu_s
        );
    }
    println!("\n{steps} Monte Carlo steps, one attempt per site per step.");
}
