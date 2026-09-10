//! A local test for whether taking a site would break a cell in two.
//!
//! CompuCell3D's global connectivity plugin walks a cell's whole site graph on
//! every copy attempt, which is serial by construction. The local test reads
//! only the neighbourhood of the site being taken: if the sites around it that
//! belong to the losing cell fall into more than one piece, the copy would
//! pinch the cell there and is refused.
//!
//! That is the criterion cellular Potts codes use to keep cells whole in a
//! parallel sweep. It is a sufficient condition: it stops every local pinch,
//! and a cell can still separate through a sequence of moves that is nowhere
//! locally disconnecting. Whichever engine runs it,
//! both refuse the same copies, since both read the same neighbourhood.

use crate::lattice::{Lattice, ORDER3};

/// Neighbour offsets the test reads, always the full Moore or twenty-six
/// neighbourhood whatever the energy uses.
///
/// A cell that is connected under the energy's own neighbourhood is connected
/// under this one, so testing on the wider set refuses fewer copies.
#[must_use]
pub fn ring(depth: usize) -> Vec<(i64, i64, i64)> {
    ORDER3
        .iter()
        .copied()
        .filter(|o| depth > 1 || o.2 == 0)
        .collect()
}

/// Whether two offsets are neighbours of each other.
fn adjacent(a: (i64, i64, i64), b: (i64, i64, i64)) -> bool {
    let (dx, dy, dz) = (a.0 - b.0, a.1 - b.1, a.2 - b.2);
    dx.abs() <= 1 && dy.abs() <= 1 && dz.abs() <= 1
}

/// Whether the sites of `label` around `target` are one piece.
///
/// Returns true when at most one such site exists, since a single neighbour
/// cannot be disconnected from itself and a cell about to lose its last site
/// is refused elsewhere.
#[must_use]
pub fn locally_connected(
    lattice: &Lattice,
    ring: &[(i64, i64, i64)],
    target: usize,
    label: u32,
) -> bool {
    let (x, y, z) = lattice.coords(target);

    let mut members: [(i64, i64, i64); 26] = [(0, 0, 0); 26];
    let mut count = 0usize;
    for &offset in ring {
        let site = lattice.index(x + offset.0, y + offset.1, z + offset.2);
        if site != target && lattice.labels[site] == label {
            members[count] = offset;
            count += 1;
        }
    }
    if count <= 1 {
        return true;
    }

    let mut seen = [false; 26];
    let mut stack = [0usize; 26];
    let mut top = 1usize;
    stack[0] = 0;
    seen[0] = true;
    let mut reached = 1usize;

    while top > 0 {
        top -= 1;
        let here = stack[top];
        for other in 0..count {
            if !seen[other] && adjacent(members[here], members[other]) {
                seen[other] = true;
                stack[top] = other;
                top += 1;
                reached += 1;
            }
        }
    }
    reached == count
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane_with(pattern: &[(i64, i64)]) -> Lattice {
        let mut lattice = Lattice::medium(9, 9, 1, 2);
        for &(x, y) in pattern {
            let index = lattice.index(4 + x, 4 + y, 0);
            lattice.labels[index] = 1;
        }
        lattice
    }

    #[test]
    fn a_solid_neighbourhood_is_one_piece() {
        let lattice = plane_with(&[
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ]);
        let target = lattice.index(4, 4, 0);
        assert!(locally_connected(&lattice, &ring(1), target, 1));
    }

    #[test]
    fn two_opposite_neighbours_are_two_pieces() {
        // The site between them is the only thing joining the cell there, so
        // taking it would pinch the cell in two.
        let lattice = plane_with(&[(-1, 0), (1, 0)]);
        let target = lattice.index(4, 4, 0);
        assert!(!locally_connected(&lattice, &ring(1), target, 1));
    }

    #[test]
    fn one_neighbour_is_never_disconnected() {
        let lattice = plane_with(&[(1, 0)]);
        let target = lattice.index(4, 4, 0);
        assert!(locally_connected(&lattice, &ring(1), target, 1));
    }

    #[test]
    fn a_diagonal_pair_touches_and_stays_one_piece() {
        let lattice = plane_with(&[(1, 0), (1, 1)]);
        let target = lattice.index(4, 4, 0);
        assert!(locally_connected(&lattice, &ring(1), target, 1));
    }

    #[test]
    fn a_bridge_through_the_third_axis_is_read_in_a_volume() {
        let mut lattice = Lattice::medium(9, 9, 9, 2);
        for &(x, y, z) in &[(4i64, 4, 3), (4, 4, 5)] {
            let index = lattice.index(x, y, z);
            lattice.labels[index] = 1;
        }
        let target = lattice.index(4, 4, 4);
        assert!(!locally_connected(&lattice, &ring(9), target, 1));

        // Joined round the side, the same two sites are one piece.
        let side = lattice.index(5, 4, 4);
        lattice.labels[side] = 1;
        assert!(locally_connected(&lattice, &ring(9), target, 1));
    }
}
