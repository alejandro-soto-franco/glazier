//! The nematic coupling, the labelled start, and the bookkeeping under both.

use glazier::Simulation;
use glazier_core::rng::Xoshiro;
use glazier_core::{CellType, Model};

const SIDE: usize = 48;

fn field_along(angle: f64, amplitude: f64) -> Vec<[f64; 2]> {
    vec![
        [
            amplitude * (2.0 * angle).cos(),
            amplitude * (2.0 * angle).sin()
        ];
        SIDE * SIDE
    ]
}

fn random_field(seed: u64) -> Vec<[f64; 2]> {
    let mut rng = Xoshiro::seed(seed);
    (0..SIDE * SIDE)
        .map(|_| [rng.next_f64() - 0.5, rng.next_f64() - 0.5])
        .collect()
}

fn model(lambda_nematic: f64, nematic: Vec<[f64; 2]>) -> Model {
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
        seed: 5,
        nematic,
        ..Default::default()
    }
}

/// Nematic order of the cells' long axes, |<exp(2 i theta)>|, and the mean
/// director angle. The moments are recounted, since a run with no length or
/// nematic term does not keep them.
fn order(sim: &Simulation) -> (f64, f64) {
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

#[test]
fn the_incremental_delta_matches_a_full_recomputation() {
    let mut sim = Simulation::tiled(model(3.0, random_field(9)), 8).unwrap();
    let mut rng = Xoshiro::seed(4);
    for _ in 0..20 {
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
            let q = after.model.nematic[target];
            after.nematic_sum[old as usize][0] -= q[0];
            after.nematic_sum[old as usize][1] -= q[1];
            after.nematic_sum[new as usize][0] += q[0];
            after.nematic_sum[new as usize][1] += q[1];
            let (x, y, z) = after.lattice.coords(target);
            let extent = (SIDE as f64, SIDE as f64, 1.0);
            let site = (x as f64, y as f64, z as f64);
            let leaving = after.moments[old as usize].unwrap(site.0, site.1, site.2, extent);
            after.moments[old as usize].remove(leaving);
            let joining = after.moments[new as usize].unwrap(site.0, site.1, site.2, extent);
            after.moments[new as usize].add(joining);

            let measured = after.energy() - before;
            assert!(
                (predicted - measured).abs() < 1e-8,
                "delta {predicted} against {measured} at site {target}"
            );
        }
    }
}

#[test]
fn the_nematic_sums_match_a_recount_after_a_run() {
    let mut sim = Simulation::tiled(model(3.0, random_field(1)), 8).unwrap();
    for _ in 0..60 {
        sim.step();
    }
    for (label, counted) in sim.recounted_nematic().iter().enumerate() {
        let kept = sim.nematic_sum[label];
        assert!(
            (kept[0] - counted[0]).abs() < 1e-8 && (kept[1] - counted[1]).abs() < 1e-8,
            "label {label}: kept {kept:?}, counted {counted:?}"
        );
    }
}

#[test]
fn cells_align_with_a_uniform_field_and_not_without_one() {
    let angle = 0.6;
    let mut coupled = Simulation::tiled(model(6.0, field_along(angle, 1.0)), 8).unwrap();
    let mut blind = Simulation::tiled(model(0.0, field_along(angle, 1.0)), 8).unwrap();
    for _ in 0..400 {
        coupled.step();
        blind.step();
    }
    let (s_coupled, director) = order(&coupled);
    let (s_blind, _) = order(&blind);
    let off = (director - angle)
        .abs()
        .min(std::f64::consts::PI - (director - angle).abs());
    assert!(s_coupled > 0.7, "coupled order {s_coupled}");
    assert!(
        off < 0.15,
        "director {director} against the field's {angle}"
    );
    assert!(s_blind < 0.4, "uncoupled order {s_blind}");
}

#[test]
fn a_labelled_start_reproduces_the_state_it_was_taken_from() {
    let mut first = Simulation::tiled(model(2.0, random_field(3)), 8).unwrap();
    for _ in 0..30 {
        first.step();
    }
    let restarted =
        Simulation::from_labels(first.model.clone(), first.lattice.labels.clone()).unwrap();
    assert_eq!(restarted.volume, first.recounted_volumes());
    assert_eq!(restarted.surface, first.recounted_surfaces());
    for label in 1..restarted.volume.len() {
        let (a, b) = (restarted.moments[label], first.moments[label]);
        assert!((a.length() - b.length()).abs() < 1e-9, "label {label}");
        let (p, q) = (restarted.nematic_sum[label], first.nematic_sum[label]);
        assert!((p[0] - q[0]).abs() < 1e-9 && (p[1] - q[1]).abs() < 1e-9);
    }
    assert!((restarted.energy() - first.energy()).abs() < 1e-6);
}

#[test]
fn a_labelled_start_refuses_a_field_of_the_wrong_size() {
    let err = Simulation::from_labels(model(0.0, Vec::new()), vec![1; 10]).unwrap_err();
    assert!(err.contains("sites"), "{err}");
}
