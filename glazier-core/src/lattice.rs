//! The site lattice and its periodic neighbourhood.
//!
//! One lattice type covers both dimensions: a depth of one is a plane, and
//! every offset with a nonzero `z` drops out of the neighbourhood, so a
//! two-dimensional run does the same arithmetic it always did.

/// Offsets of the six face neighbours.
pub const ORDER1: [(i64, i64, i64); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

/// Offsets of the eighteen face and edge neighbours.
pub const ORDER2: [(i64, i64, i64); 18] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
    (1, 1, 0),
    (1, -1, 0),
    (-1, 1, 0),
    (-1, -1, 0),
    (1, 0, 1),
    (1, 0, -1),
    (-1, 0, 1),
    (-1, 0, -1),
    (0, 1, 1),
    (0, 1, -1),
    (0, -1, 1),
    (0, -1, -1),
];

/// Offsets of all twenty-six neighbours of a cube.
pub const ORDER3: [(i64, i64, i64); 26] = {
    let mut out = [(0i64, 0i64, 0i64); 26];
    let mut index = 0;
    let mut dz = -1i64;
    while dz <= 1 {
        let mut dy = -1i64;
        while dy <= 1 {
            let mut dx = -1i64;
            while dx <= 1 {
                if dx != 0 || dy != 0 || dz != 0 {
                    out[index] = (dx, dy, dz);
                    index += 1;
                }
                dx += 1;
            }
            dy += 1;
        }
        dz += 1;
    }
    out
};

/// Neighbour offsets for a stated order, filtered to the plane when the
/// lattice is one site deep.
///
/// A plane has no `z` neighbours to reach, and counting them would make every
/// site look like a boundary. Order 2 on a plane is therefore the eight Moore
/// neighbours, which is what a two-dimensional model means by it.
#[must_use]
pub fn offsets_for(order: u8, depth: usize) -> Vec<(i64, i64, i64)> {
    let all: &[(i64, i64, i64)] = match order {
        1 => &ORDER1,
        2 => &ORDER2,
        _ => &ORDER3,
    };
    if depth > 1 {
        return all.to_vec();
    }
    let mut planar: Vec<(i64, i64, i64)> = all.iter().copied().filter(|o| o.2 == 0).collect();
    if order >= 2 {
        // In the plane, orders 2 and 3 are both the eight Moore neighbours.
        planar = ORDER3
            .iter()
            .copied()
            .filter(|o| o.2 == 0)
            .collect::<Vec<_>>();
    }
    planar
}

/// One coordinate folded into `[0, span)`.
///
/// The common case is a coordinate one step outside the range, which two
/// comparisons settle. Anything further away falls back to the remainder.
#[inline]
#[must_use]
fn wrap(v: i64, span: usize) -> usize {
    let span = span as i64;
    if v >= 0 {
        if v < span {
            v as usize
        } else if v < 2 * span {
            (v - span) as usize
        } else {
            (v % span) as usize
        }
    } else if v >= -span {
        (v + span) as usize
    } else {
        v.rem_euclid(span) as usize
    }
}

/// A periodic lattice of cell labels. Label 0 is the medium.
#[derive(Clone, Debug)]
pub struct Lattice {
    /// Width in sites.
    pub width: usize,
    /// Height in sites.
    pub height: usize,
    /// Depth in sites. One is a plane.
    pub depth: usize,
    /// One label per site, `x` fastest and `z` slowest.
    pub labels: Vec<u32>,
    /// Neighbour offsets this lattice uses, fixed at construction.
    pub offsets: Vec<(i64, i64, i64)>,
}

impl Lattice {
    /// An all-medium lattice.
    #[must_use]
    pub fn medium(width: usize, height: usize, depth: usize, order: u8) -> Self {
        Self {
            width,
            height,
            depth,
            labels: vec![0; width * height * depth.max(1)],
            offsets: offsets_for(order, depth),
        }
    }

    /// Sites in the lattice.
    #[must_use]
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// Whether the lattice has no sites.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Fill the lattice with cubes of side `side`, all of type 1.
    pub fn tile(&mut self, side: usize) -> u32 {
        self.tile_grid(
            side,
            self.width / side,
            self.height / side,
            (self.depth / side).max(1),
        )
    }

    /// Lay down `nx` by `ny` by `nz` cubes of side `side` from the origin and
    /// leave the rest of the lattice as medium.
    ///
    /// A sheet that fills the lattice conserves its total volume, so a cell can
    /// only grow at another's expense. Leaving medium is what lets a volume
    /// constraint move the mean.
    pub fn tile_grid(&mut self, side: usize, nx: usize, ny: usize, nz: usize) -> u32 {
        let mut next = 1u32;
        let nx = nx.min(self.width / side);
        let ny = ny.min(self.height / side);
        let nz = if self.depth == 1 {
            1
        } else {
            nz.min(self.depth / side)
        };
        let side_z = if self.depth == 1 { 1 } else { side };

        for bz in 0..nz {
            for by in 0..ny {
                for bx in 0..nx {
                    for z in bz * side_z..(bz + 1) * side_z {
                        for y in by * side..(by + 1) * side {
                            for x in bx * side..(bx + 1) * side {
                                let index = self.index(x as i64, y as i64, z as i64);
                                self.labels[index] = next;
                            }
                        }
                    }
                    next += 1;
                }
            }
        }
        next - 1
    }

    /// Flat index of a site, wrapping every axis.
    ///
    /// The wrap is a comparison, since every caller steps by one neighbour
    /// offset from a coordinate already in range. A remainder is a division,
    /// and this sits inside the innermost loop of the sweep, where that
    /// difference shows up in the wall clock.
    #[must_use]
    pub fn index(&self, x: i64, y: i64, z: i64) -> usize {
        (wrap(z, self.depth) * self.height + wrap(y, self.height)) * self.width
            + wrap(x, self.width)
    }

    /// The site coordinates of a flat index.
    #[must_use]
    pub fn coords(&self, index: usize) -> (i64, i64, i64) {
        let plane = self.width * self.height;
        let z = index / plane;
        let rest = index % plane;
        (
            (rest % self.width) as i64,
            (rest / self.width) as i64,
            z as i64,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrapping_agrees_with_the_remainder_everywhere_it_is_asked() {
        for span in [1usize, 2, 5, 8] {
            for v in -3 * span as i64..3 * span as i64 {
                assert_eq!(
                    wrap(v, span),
                    v.rem_euclid(span as i64) as usize,
                    "wrap({v}, {span})"
                );
            }
        }
    }

    #[test]
    fn indexing_wraps_on_every_axis() {
        let l = Lattice::medium(8, 4, 2, 2);
        assert_eq!(l.index(0, 0, 0), 0);
        assert_eq!(l.index(8, 4, 2), 0);
        assert_eq!(l.index(-1, -1, -1), l.len() - 1);
    }

    #[test]
    fn coordinates_round_trip_through_the_index() {
        let l = Lattice::medium(5, 4, 3, 2);
        for index in 0..l.len() {
            let (x, y, z) = l.coords(index);
            assert_eq!(l.index(x, y, z), index);
        }
    }

    #[test]
    fn a_plane_takes_the_eight_moore_neighbours_at_order_two() {
        assert_eq!(offsets_for(2, 1).len(), 8);
        assert_eq!(offsets_for(1, 1).len(), 4);
        assert_eq!(offsets_for(3, 1).len(), 8);
    }

    #[test]
    fn a_volume_takes_its_own_counts() {
        assert_eq!(offsets_for(1, 4).len(), 6);
        assert_eq!(offsets_for(2, 4).len(), 18);
        assert_eq!(offsets_for(3, 4).len(), 26);
    }

    #[test]
    fn tiling_lays_down_equal_cells_in_the_plane() {
        let mut l = Lattice::medium(12, 8, 1, 2);
        assert_eq!(l.tile(4), 6);
        let mut counts = std::collections::BTreeMap::new();
        for &label in &l.labels {
            *counts.entry(label).or_insert(0usize) += 1;
        }
        assert!(
            !counts.contains_key(&0),
            "the tiling should fill the lattice"
        );
        assert!(counts.values().all(|&v| v == 16), "{counts:?}");
    }

    #[test]
    fn tiling_lays_down_equal_cells_in_a_volume() {
        let mut l = Lattice::medium(8, 8, 8, 2);
        assert_eq!(l.tile(4), 8);
        let mut counts = std::collections::BTreeMap::new();
        for &label in &l.labels {
            *counts.entry(label).or_insert(0usize) += 1;
        }
        assert!(counts.values().all(|&v| v == 64), "{counts:?}");
    }
}
