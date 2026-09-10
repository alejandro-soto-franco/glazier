//! Adhesion stated as molecules rather than as a table of energies.

use glazier::Simulation;
use glazier_core::Blueprint;

fn description(binding: f64, second_presents: f64) -> String {
    format!(
        r#"{{
        "name": "two types with one molecule",
        "width": 64, "height": 64,
        "types": [
            {{ "name": "Sticky", "target_volume": 64.0, "lambda_volume": 2.0,
               "presents": {{ "cadherin": 1.0 }} }},
            {{ "name": "Other", "target_volume": 64.0, "lambda_volume": 2.0,
               "presents": {{ "cadherin": {second_presents} }} }}
        ],
        "adhesion": {{ "molecules": ["cadherin"], "binding": [{binding}] }},
        "contact": [
            0.0, 16.0, 16.0,
            16.0, 10.0, 10.0,
            16.0, 10.0, 10.0
        ],
        "temperature": 10.0, "neighbour_order": 2, "seed": 1,
        "steps": 10, "dump_every": 0,
        "initial": {{ "side": 8, "nx": 4, "ny": 4 }},
        "units": {{ "micron_per_site": 1.0, "minute_per_step": 1.0 }}
    }}"#
    )
}

fn contact(bp: &Blueprint, a: usize, b: usize) -> f64 {
    let n = bp.types.len() + 1;
    bp.effective_contact()[a * n + b]
}

#[test]
fn binding_comes_off_the_contact_energy_between_the_types_that_present_it() {
    let bp = Blueprint::from_json(&description(4.0, 1.0)).unwrap();
    // Both types present one unit, so every cell-cell bond loses the binding.
    assert!((contact(&bp, 1, 1) - 6.0).abs() < 1e-12);
    assert!((contact(&bp, 1, 2) - 6.0).abs() < 1e-12);
    // The medium presents nothing, so its bonds are untouched.
    assert!((contact(&bp, 0, 1) - 16.0).abs() < 1e-12);
}

#[test]
fn a_type_that_presents_nothing_keeps_its_stated_energies() {
    let bp = Blueprint::from_json(&description(4.0, 0.0)).unwrap();
    assert!((contact(&bp, 1, 1) - 6.0).abs() < 1e-12);
    assert!((contact(&bp, 1, 2) - 10.0).abs() < 1e-12);
    assert!((contact(&bp, 2, 2) - 10.0).abs() < 1e-12);
}

#[test]
fn no_binding_leaves_the_matrix_as_the_description_states_it() {
    let bp = Blueprint::from_json(&description(0.0, 1.0)).unwrap();
    assert_eq!(bp.effective_contact(), bp.contact);
}

/// The same contact matrix stated the other way, with no molecules.
fn stated_directly(cell_cell: f64) -> String {
    format!(
        r#"{{
        "name": "two types with the energies written out",
        "width": 64, "height": 64,
        "types": [
            {{ "name": "Sticky", "target_volume": 64.0, "lambda_volume": 2.0 }},
            {{ "name": "Other", "target_volume": 64.0, "lambda_volume": 2.0 }}
        ],
        "contact": [
            0.0, 16.0, 16.0,
            16.0, {cell_cell}, {cell_cell},
            16.0, {cell_cell}, {cell_cell}
        ],
        "temperature": 10.0, "neighbour_order": 2, "seed": 1,
        "steps": 10, "dump_every": 0,
        "initial": {{ "side": 8, "nx": 4, "ny": 4 }},
        "units": {{ "micron_per_site": 1.0, "minute_per_step": 1.0 }}
    }}"#
    )
}

#[test]
fn a_molecule_description_runs_as_the_matrix_it_folds_to() {
    // Folding is meant to be transparent, so the test is exact rather than
    // statistical: the same seed on the same model has to reach the same
    // lattice, whichever way the description said it.
    let molecules = Blueprint::from_json(&description(4.0, 1.0)).unwrap();
    let written_out = Blueprint::from_json(&stated_directly(6.0)).unwrap();
    assert_eq!(
        molecules.effective_contact(),
        written_out.effective_contact()
    );

    let mut folded = Simulation::tiled(molecules.model(), 8).unwrap();
    let mut direct = Simulation::tiled(written_out.model(), 8).unwrap();
    for _ in 0..150 {
        folded.step();
        direct.step();
    }
    assert_eq!(folded.lattice.labels, direct.lattice.labels);
    assert_eq!(folded.volume, direct.volume);
}

// A tissue-scale claim is deliberately absent here. Lowering a cell-cell bond
// raises the tension at a cluster's surface and at the same time makes each
// cell floppier, and the two pull the outer perimeter in opposite directions:
// at a binding of eight the cluster keeps 504 bonds against the medium where an
// unbound one keeps 396. Which effect wins is a question about the model rather
// than about the folding, so the tests above assert the folding and leave it.

#[test]
fn a_molecule_the_description_never_declared_is_an_error() {
    let text = description(4.0, 1.0).replace("\"cadherin\": 1.0", "\"integrin\": 1.0");
    let err = Blueprint::from_json(&text).unwrap_err();
    assert!(err.contains("integrin"), "{err}");
}

#[test]
fn presenting_a_molecule_with_no_adhesion_block_is_an_error() {
    let text = description(4.0, 1.0).replace(
        r#""adhesion": { "molecules": ["cadherin"], "binding": [4] },"#,
        "",
    );
    let err = Blueprint::from_json(&text).unwrap_err();
    assert!(err.contains("declares none"), "{err}");
}

#[test]
fn a_binding_matrix_of_the_wrong_shape_is_an_error() {
    let text = description(4.0, 1.0).replace(r#""binding": [4]"#, r#""binding": [4, 1]"#);
    let err = Blueprint::from_json(&text).unwrap_err();
    assert!(err.contains("binding matrix"), "{err}");
}
