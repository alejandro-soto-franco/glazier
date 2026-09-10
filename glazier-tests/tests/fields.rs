//! Fields on a running sheet: what a description states about a chemical, and
//! what the engine does with it.

use glazier::Simulation;
use glazier_core::field::{Exchange, Species};
use glazier_core::{Blueprint, CellType, Model};

fn model_with_field(secretion: f64, uptake: f64, decay: f64) -> Model {
    Model {
        species: vec![Species {
            name: "IFNg".into(),
            diffusion: 0.2,
            decay,
            initial: 0.0,
        }],
        exchange: vec![
            Exchange {
                secretion: vec![0.0],
                uptake: vec![0.0],
            },
            Exchange {
                secretion: vec![secretion],
                uptake: vec![uptake],
            },
        ],
        width: 48,
        height: 48,
        contact: vec![0.0, 16.0, 16.0, 8.0],
        types: vec![
            CellType {
                target_volume: 0.0,
                lambda_volume: 0.0,
                target_surface: 0.0,
                lambda_surface: 0.0,
                ..Default::default()
            },
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
        seed: 1,
        ..Default::default()
    }
}

#[test]
fn a_secreting_sheet_fills_the_field() {
    let mut sim = Simulation::tiled_grid(model_with_field(0.1, 0.0, 0.0), 8, 3, 3, 1).unwrap();
    assert_eq!(sim.fields.total(0), 0.0);
    for _ in 0..20 {
        sim.step();
    }
    let total = sim.fields.total(0);
    // Nine cells of about 64 sites, secreting 0.1 a step for twenty steps.
    assert!(total > 100.0, "total {total}");
    // Diffusion has taken it beyond the cells that made it.
    let outside = sim.fields.values[0]
        .iter()
        .enumerate()
        .filter(|(site, _)| sim.lattice.labels[*site] == 0)
        .map(|(_, v)| *v)
        .sum::<f64>();
    assert!(outside > 0.0, "nothing reached the medium");
}

#[test]
fn decay_bounds_a_secreting_sheet() {
    let mut steady = Simulation::tiled_grid(model_with_field(0.1, 0.0, 0.05), 8, 3, 3, 1).unwrap();
    let mut growing = Simulation::tiled_grid(model_with_field(0.1, 0.0, 0.0), 8, 3, 3, 1).unwrap();
    for _ in 0..80 {
        steady.step();
        growing.step();
    }
    assert!(
        steady.fields.total(0) < growing.fields.total(0),
        "decay left {} against {}",
        steady.fields.total(0),
        growing.fields.total(0)
    );
}

#[test]
fn uptake_removes_what_secretion_adds() {
    let mut sim = Simulation::tiled_grid(model_with_field(0.1, 1.0, 0.0), 8, 3, 3, 1).unwrap();
    for _ in 0..30 {
        sim.step();
    }
    // Uptake of one takes every owned site to zero after each secretion, so
    // whatever survives is the part that diffused into the medium.
    for (site, &label) in sim.lattice.labels.iter().enumerate() {
        if label != 0 {
            assert!(
                sim.fields.values[0][site] < 1e-9,
                "an owned site kept {} under full uptake",
                sim.fields.values[0][site]
            );
        }
    }
}

#[test]
fn a_description_states_its_fields() {
    let text = r#"{
        "name": "one field",
        "width": 32, "height": 32,
        "types": [{ "name": "Cell", "target_volume": 64.0, "lambda_volume": 2.0,
                    "secretion": {"IFNg": 0.5}, "uptake": {"IFNg": 0.01} }],
        "fields": [{ "name": "IFNg", "diffusion": 0.2, "decay": 0.001 }],
        "contact": [0.0, 16.0, 16.0, 8.0],
        "temperature": 10.0, "neighbour_order": 2, "seed": 1,
        "steps": 10, "dump_every": 0,
        "initial": { "side": 8, "nx": 2, "ny": 2 },
        "units": { "micron_per_site": 1.0, "minute_per_step": 1.0 }
    }"#;
    let bp = Blueprint::from_json(text).unwrap();
    let model = bp.model();
    assert_eq!(model.species.len(), 1);
    assert_eq!(model.species[0].name, "IFNg");
    assert_eq!(model.exchange[0].secretion, vec![0.0]);
    assert_eq!(model.exchange[1].secretion, vec![0.5]);
    assert_eq!(model.exchange[1].uptake, vec![0.01]);
}

#[test]
fn secreting_a_field_the_description_never_declared_is_an_error() {
    let text = r#"{
        "name": "typo",
        "width": 32, "height": 32,
        "types": [{ "name": "Cell", "target_volume": 64.0, "lambda_volume": 2.0,
                    "secretion": {"IFNgamma": 0.5} }],
        "fields": [{ "name": "IFNg", "diffusion": 0.2 }],
        "contact": [0.0, 16.0, 16.0, 8.0],
        "temperature": 10.0, "neighbour_order": 2, "seed": 1,
        "steps": 10, "dump_every": 0,
        "initial": { "side": 8, "nx": 2, "ny": 2 },
        "units": { "micron_per_site": 1.0, "minute_per_step": 1.0 }
    }"#;
    let err = Blueprint::from_json(text).unwrap_err();
    assert!(err.contains("IFNgamma"), "{err}");
}
