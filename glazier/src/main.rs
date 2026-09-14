//! Run a Blueprint on the CPU or the GPU and write the label fields.

use glazier::Blueprint;
use glazier::Simulation;
use glazier::npy;
use std::path::PathBuf;
use std::process::ExitCode;

fn usage() -> ExitCode {
    eprintln!(
        "usage: glazier --model <blueprint.json> --out <dir> [--engine cpu|gpu]\n\
         \x20      glazier --model <blueprint.json> --check\n\
         \n\
         Writes labels_NNNNN.npy per dump and summary.json at the end.\n\
         --check parses the description and prints what this engine read, so a\n\
         second reader of the same file can be compared against it.\n\
         --trace writes trace.csv, one row per step: live cells, mean volume,\n\
         and the total of every field."
    );
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let mut model_path: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut engine = "cpu".to_string();
    let mut check = false;
    let mut trace = false;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model_path = args.next().map(PathBuf::from),
            "--out" => out = args.next().map(PathBuf::from),
            "--engine" => engine = args.next().unwrap_or_default(),
            "--check" => check = true,
            "--trace" => trace = true,
            _ => return usage(),
        }
    }
    let Some(model_path) = model_path else {
        return usage();
    };

    let text = match std::fs::read_to_string(&model_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("cannot read {}: {e}", model_path.display());
            return ExitCode::FAILURE;
        }
    };
    let bp = match Blueprint::from_json(&text) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{}: {e}", model_path.display());
            return ExitCode::FAILURE;
        }
    };
    if check {
        let model = bp.model();
        let parsed = serde_json::json!({
            "name": bp.name,
            "width": model.width,
            "height": model.height,
            "n_types": model.n_types(),
            "contact": model.contact,
            "target_volume": model.types.iter().map(|t| t.target_volume).collect::<Vec<_>>(),
            "lambda_volume": model.types.iter().map(|t| t.lambda_volume).collect::<Vec<_>>(),
            "target_surface": model.types.iter().map(|t| t.target_surface).collect::<Vec<_>>(),
            "lambda_surface": model.types.iter().map(|t| t.lambda_surface).collect::<Vec<_>>(),
            "target_length": model.types.iter().map(|t| t.target_length).collect::<Vec<_>>(),
            "lambda_length": model.types.iter().map(|t| t.lambda_length).collect::<Vec<_>>(),
            "connected": model.types.iter().map(|t| t.connected).collect::<Vec<_>>(),
            "max_activity": model.types.iter().map(|t| t.max_activity).collect::<Vec<_>>(),
            "lambda_activity": model.types.iter().map(|t| t.lambda_activity).collect::<Vec<_>>(),
            "external": model.types.iter().map(|t| t.external.to_vec()).collect::<Vec<_>>(),
            "lambda_nematic": model.types.iter().map(|t| t.lambda_nematic).collect::<Vec<_>>(),
            "nematic_field": bp.nematic_field,
            "initial_labels": bp.initial.labels,
            "chemotaxis": model.chemotaxis.clone(),
            "division_volume": model.types.iter().map(|t| t.division_volume).collect::<Vec<_>>(),
            "death_rate": model.types.iter().map(|t| t.death_rate).collect::<Vec<_>>(),
            "species": model.species.iter().map(|s| s.name.clone()).collect::<Vec<_>>(),
            "diffusion": model.species.iter().map(|s| s.diffusion).collect::<Vec<_>>(),
            "decay": model.species.iter().map(|s| s.decay).collect::<Vec<_>>(),
            "secretion": model.exchange.iter().map(|e| e.secretion.clone()).collect::<Vec<_>>(),
            "uptake": model.exchange.iter().map(|e| e.uptake.clone()).collect::<Vec<_>>(),
            "temperature": model.temperature,
            "neighbour_order": model.neighbour_order,
            "seed": model.seed,
            "steps": bp.steps,
            "initial": [bp.initial.side, bp.initial.nx, bp.initial.ny, bp.initial.nz],
            "micron_per_site": bp.units.micron_per_site,
            "minute_per_step": bp.units.minute_per_step,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&parsed).unwrap_or_default()
        );
        return ExitCode::SUCCESS;
    }

    let Some(out) = out else {
        return usage();
    };
    if let Err(e) = std::fs::create_dir_all(&out) {
        eprintln!("cannot create {}: {e}", out.display());
        return ExitCode::FAILURE;
    }

    let base = model_path
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_default();
    let sim = match bp.simulation(&base) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    // `seconds` times the steps alone. Compiling the kernel, uploading the
    // lattice and reading the result back are all reported apart from it: at a
    // small lattice the setup is the larger of the two, and at a large one the
    // read-back is.
    let mut setup = 0.0f64;
    let mut stepping = 0.0f64;
    let mut field_totals: Vec<(String, f64)> = Vec::new();
    let started = std::time::Instant::now();
    let (labels, volumes) = match engine.as_str() {
        "cpu" => run_cpu(sim, &bp, &out, &mut field_totals, trace, &mut stepping),
        "gpu" => match run_gpu(
            &sim,
            &bp,
            &out,
            &mut setup,
            &mut field_totals,
            &mut stepping,
        ) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        },
        other => {
            eprintln!("unknown engine {other}");
            return usage();
        }
    };
    let overall = started.elapsed().as_secs_f64();
    let seconds = if stepping > 0.0 {
        stepping
    } else {
        overall - setup
    };

    // A label whose volume reached zero is a cell that died, so the live count
    // and the mean volume are taken over the survivors.
    let live: Vec<u32> = volumes[1..].iter().copied().filter(|&v| v > 0).collect();
    let mean = if live.is_empty() {
        0.0
    } else {
        live.iter().map(|&v| f64::from(v)).sum::<f64>() / live.len() as f64
    };
    let summary = serde_json::json!({
        "name": bp.name,
        "engine": engine,
        "steps": bp.steps,
        "seconds": seconds,
        "setup_seconds": setup,
        "overall_seconds": overall,
        "n_cells": live.len(),
        "n_labels": volumes.len() - 1,
        "fields": field_totals
            .iter()
            .map(|(name, total)| (name.clone(), *total))
            .collect::<std::collections::BTreeMap<_, _>>(),
        "mean_volume": mean,
        "width": bp.width,
        "height": bp.height,
        "micron_per_site": bp.units.micron_per_site,
        "minute_per_step": bp.units.minute_per_step,
    });
    if let Err(e) = std::fs::write(
        out.join("summary.json"),
        serde_json::to_string_pretty(&summary).unwrap_or_default(),
    ) {
        eprintln!("cannot write the summary: {e}");
        return ExitCode::FAILURE;
    }

    if let Err(e) = npy::write_u32(
        &out.join(format!("labels_{:05}.npy", bp.steps)),
        &labels,
        bp.width,
        bp.height,
        bp.depth,
    ) {
        eprintln!("cannot write the final field: {e}");
        return ExitCode::FAILURE;
    }

    println!(
        "{engine}: {} steps in {seconds:.3} s, {} cells, mean volume {mean:.1}",
        bp.steps,
        live.len()
    );
    ExitCode::SUCCESS
}

fn run_cpu(
    mut sim: Simulation,
    bp: &Blueprint,
    out: &std::path::Path,
    fields: &mut Vec<(String, f64)>,
    trace: bool,
    stepping: &mut f64,
) -> (Vec<u32>, Vec<u32>) {
    // One row per step, so a figure can show what the run did rather than
    // where it ended.
    let mut rows = String::new();
    if trace {
        rows.push_str("step,cells,mean_volume");
        for species in &sim.fields.species {
            rows.push(',');
            rows.push_str(&species.name);
        }
        rows.push('\n');
    }

    let began = std::time::Instant::now();
    for step in 1..=bp.steps {
        sim.step();
        if trace {
            let live: Vec<u32> = sim.volume[1..].iter().copied().filter(|&v| v > 0).collect();
            let mean = if live.is_empty() {
                0.0
            } else {
                live.iter().map(|&v| f64::from(v)).sum::<f64>() / live.len() as f64
            };
            rows.push_str(&format!("{step},{},{mean:.3}", live.len()));
            for index in 0..sim.fields.species.len() {
                rows.push_str(&format!(",{:.6}", sim.fields.total(index)));
            }
            rows.push('\n');
        }
        if bp.dump_every > 0 && step % bp.dump_every == 0 && step != bp.steps {
            let _ = npy::write_u32(
                &out.join(format!("labels_{step:05}.npy")),
                &sim.lattice.labels,
                bp.width,
                bp.height,
                bp.depth,
            );
            for (index, species) in sim.fields.species.iter().enumerate() {
                let _ = npy::write_f64(
                    &out.join(format!("field_{}_{step:05}.npy", species.name)),
                    &sim.fields.values[index],
                    bp.width,
                    bp.height,
                    bp.depth,
                );
            }
        }
    }
    *stepping = began.elapsed().as_secs_f64();
    if trace {
        let _ = std::fs::write(out.join("trace.csv"), rows);
    }
    *fields = sim
        .fields
        .species
        .iter()
        .enumerate()
        .map(|(index, s)| (s.name.clone(), sim.fields.total(index)))
        .collect();
    (sim.lattice.labels.clone(), sim.volume.clone())
}

#[cfg(feature = "cuda")]
fn run_gpu(
    sim: &Simulation,
    bp: &Blueprint,
    out: &std::path::Path,
    setup: &mut f64,
    fields: &mut Vec<(String, f64)>,
    stepping: &mut f64,
) -> Result<(Vec<u32>, Vec<u32>), String> {
    use glazier::cuda::GpuSimulation;

    let began = std::time::Instant::now();
    let mut gpu = GpuSimulation::from_cpu(sim).map_err(|e| e.to_string())?;
    *setup = began.elapsed().as_secs_f64();
    let chunk = if bp.dump_every > 0 {
        bp.dump_every
    } else {
        bp.steps
    };
    let mut done = 0u64;
    while done < bp.steps {
        let take = chunk.min(bp.steps - done);
        gpu.step(take).map_err(|e| e.to_string())?;
        done += take;
        if bp.dump_every > 0 && done != bp.steps {
            let labels = gpu.labels().map_err(|e| e.to_string())?;
            let _ = npy::write_u32(
                &out.join(format!("labels_{done:05}.npy")),
                &labels,
                bp.width,
                bp.height,
                bp.depth,
            );
        }
    }
    *stepping = began.elapsed().as_secs_f64();
    *fields = sim
        .fields
        .species
        .iter()
        .enumerate()
        .map(|(index, s)| {
            let total = gpu
                .field(index)
                .map(|values| values.iter().sum())
                .unwrap_or(f64::NAN);
            (s.name.clone(), total)
        })
        .collect();
    Ok((
        gpu.labels().map_err(|e| e.to_string())?,
        gpu.volumes().map_err(|e| e.to_string())?,
    ))
}

#[cfg(not(feature = "cuda"))]
fn run_gpu(
    _sim: &Simulation,
    _bp: &Blueprint,
    _out: &std::path::Path,
    _setup: &mut f64,
    _fields: &mut Vec<(String, f64)>,
    _stepping: &mut f64,
) -> Result<(Vec<u32>, Vec<u32>), String> {
    Err("this build has no cuda feature".into())
}
