//! Second moments per cell, kept as running sums so a cell's length is a
//! constant-time read.
//!
//! Coordinates are unwrapped about the first site the cell was given, by
//! minimum image, so a cell straddling the periodic edge still has a centroid
//! and an axis. That holds while a cell stays smaller than half the lattice,
//! which is the same condition the minimum image convention always carries.
//!
//! The sums are three-dimensional throughout. A plane leaves every `z` at the
//! same value, so the third variance is zero and the major axis comes out of
//! the two that are left.

/// A site in the frame a cell's moments are accumulated in.
pub type Site = (f64, f64, f64);

/// Running sums for one cell.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Moments {
    /// Sites in the cell.
    pub n: f64,
    /// Sums of the unwrapped coordinates.
    pub s: [f64; 3],
    /// Sums of the products, in the order xx, yy, zz, xy, xz, yz.
    pub p: [f64; 6],
    /// The point every coordinate is unwrapped about.
    pub anchor: Option<Site>,
}

impl Moments {
    /// Unwrap a site's coordinates about this cell's anchor.
    #[must_use]
    pub fn unwrap(&self, x: f64, y: f64, z: f64, extent: Site) -> Site {
        let Some(anchor) = self.anchor else {
            return (x, y, z);
        };
        let fold = |v: f64, a: f64, span: f64| {
            let mut out = v;
            if span > 1.0 {
                if out - a > span / 2.0 {
                    out -= span;
                } else if a - out > span / 2.0 {
                    out += span;
                }
            }
            out
        };
        (
            fold(x, anchor.0, extent.0),
            fold(y, anchor.1, extent.1),
            fold(z, anchor.2, extent.2),
        )
    }

    /// Add a site at already-unwrapped coordinates.
    pub fn add(&mut self, site: Site) {
        if self.anchor.is_none() {
            self.anchor = Some(site);
        }
        *self = self.with(site, 1.0);
    }

    /// Remove a site at already-unwrapped coordinates.
    pub fn remove(&mut self, site: Site) {
        let anchor = self.anchor;
        *self = self.with(site, -1.0);
        self.anchor = anchor;
        if self.n <= 0.0 {
            *self = Self::default();
        }
    }

    /// The same sums with one site added or removed, without touching self.
    #[must_use]
    pub fn with(&self, site: Site, sign: f64) -> Self {
        let (x, y, z) = site;
        let mut out = *self;
        out.n += sign;
        out.s[0] += sign * x;
        out.s[1] += sign * y;
        out.s[2] += sign * z;
        out.p[0] += sign * x * x;
        out.p[1] += sign * y * y;
        out.p[2] += sign * z * z;
        out.p[3] += sign * x * y;
        out.p[4] += sign * x * z;
        out.p[5] += sign * y * z;
        out
    }

    /// The covariance matrix, row-major over three axes.
    #[must_use]
    pub fn covariance(&self) -> [[f64; 3]; 3] {
        if self.n <= 0.0 {
            return [[0.0; 3]; 3];
        }
        let m = [self.s[0] / self.n, self.s[1] / self.n, self.s[2] / self.n];
        let xx = self.p[0] / self.n - m[0] * m[0];
        let yy = self.p[1] / self.n - m[1] * m[1];
        let zz = self.p[2] / self.n - m[2] * m[2];
        let xy = self.p[3] / self.n - m[0] * m[1];
        let xz = self.p[4] / self.n - m[0] * m[2];
        let yz = self.p[5] / self.n - m[1] * m[2];
        [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
    }

    /// The centroid in the frame the sums were accumulated in.
    #[must_use]
    pub fn centroid(&self) -> Site {
        if self.n <= 0.0 {
            return (0.0, 0.0, 0.0);
        }
        (self.s[0] / self.n, self.s[1] / self.n, self.s[2] / self.n)
    }

    /// The major axis and the variance along it.
    #[must_use]
    pub fn principal(&self) -> (Site, f64) {
        let (v, lambda) = principal_axis(self.covariance());
        ((v[0], v[1], v[2]), lambda)
    }

    /// In-plane anisotropy of the cell, `((c_xx - c_yy), 2 c_xy) / (c_xx + c_yy)`.
    ///
    /// These are the two independent components of the traceless part of the
    /// in-plane covariance, scaled by its trace, so the vector's direction is
    /// twice the major-axis angle and its length runs from zero for a disc to
    /// one for a line. The coupling to a nematic field reads this, so a round
    /// cell feels no field and an elongated one feels it in proportion to how
    /// elongated it is.
    #[must_use]
    pub fn anisotropy(&self) -> (f64, f64) {
        if self.n < 2.0 {
            return (0.0, 0.0);
        }
        let c = self.covariance();
        let trace = c[0][0] + c[1][1];
        if trace <= 1e-12 {
            return (0.0, 0.0);
        }
        ((c[0][0] - c[1][1]) / trace, 2.0 * c[0][1] / trace)
    }

    /// Major axis of the second-moment ellipsoid, in sites.
    ///
    /// A uniformly filled ellipse of semi-axes `a` and `b` has covariance
    /// eigenvalues `a^2/4` and `b^2/4`, so the major axis is four times the
    /// root of the larger one. A cell of fewer than two sites has no axis and
    /// reads zero.
    #[must_use]
    pub fn length(&self) -> f64 {
        if self.n < 2.0 {
            return 0.0;
        }
        4.0 * self.principal().1.sqrt()
    }
}

/// The dominant eigenvector of a symmetric three by three, and its eigenvalue.
///
/// Power iteration converges to the dominant eigenvector of a symmetric
/// matrix, and the Rayleigh quotient there is its eigenvalue. Twenty rounds is
/// far past what a three by three needs, and it takes no special case for a
/// degenerate pair the way a closed form does.
#[must_use]
pub fn principal_axis(c: [[f64; 3]; 3]) -> ([f64; 3], f64) {
    let apply = |v: [f64; 3]| {
        [
            c[0][0] * v[0] + c[0][1] * v[1] + c[0][2] * v[2],
            c[1][0] * v[0] + c[1][1] * v[1] + c[1][2] * v[2],
            c[2][0] * v[0] + c[2][1] * v[1] + c[2][2] * v[2],
        ]
    };

    let mut v = [1.0, 0.7, 0.3];
    for _ in 0..20 {
        let w = apply(v);
        let norm = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
        if norm < 1e-15 {
            return ([1.0, 0.0, 0.0], 0.0);
        }
        v = [w[0] / norm, w[1] / norm, w[2] / norm];
    }
    let w = apply(v);
    let lambda = v[0] * w[0] + v[1] * w[1] + v[2] * w[2];
    (v, lambda.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filled(w: i64, h: i64, d: i64) -> Moments {
        let mut m = Moments::default();
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    m.add((x as f64, y as f64, z as f64));
                }
            }
        }
        m
    }

    #[test]
    fn a_square_reads_the_same_length_along_either_axis() {
        let square = filled(8, 8, 1);
        assert!((square.length() - filled(8, 8, 1).length()).abs() < 1e-12);
        assert!(square.length() > 0.0);
    }

    #[test]
    fn a_longer_rectangle_reads_a_greater_length() {
        assert!(filled(16, 4, 1).length() > filled(8, 8, 1).length());
        assert!((filled(16, 4, 1).length() - filled(4, 16, 1).length()).abs() < 1e-6);
    }

    #[test]
    fn a_line_of_sites_reads_about_its_own_extent() {
        // A 1 by 16 strip has covariance (16^2 - 1)/12 along its length, so the
        // major axis is 4 * sqrt of that, about 18.5.
        let strip = filled(1, 16, 1);
        assert!((strip.length() - 18.47).abs() < 0.05, "{}", strip.length());
    }

    #[test]
    fn a_column_reads_its_extent_in_the_third_axis_too() {
        let column = filled(1, 1, 16);
        assert!((column.length() - filled(1, 16, 1).length()).abs() < 1e-6);
        let axis = column.principal().0;
        assert!(axis.2.abs() > 0.99, "the axis came out {axis:?}");
    }

    #[test]
    fn adding_then_removing_a_site_returns_the_sums() {
        let before = filled(4, 4, 1);
        let mut after = before;
        after.add((9.0, 9.0, 0.0));
        after.remove((9.0, 9.0, 0.0));
        assert!((after.n - before.n).abs() < 1e-12);
        assert!((after.p[0] - before.p[0]).abs() < 1e-9);
        assert!((after.p[3] - before.p[3]).abs() < 1e-9);
    }

    #[test]
    fn with_matches_an_actual_addition() {
        let base = filled(4, 4, 1);
        let mut added = base;
        added.add((7.0, 2.0, 0.0));
        let predicted = base.with((7.0, 2.0, 0.0), 1.0);
        assert!((predicted.length() - added.length()).abs() < 1e-12);
    }

    #[test]
    fn unwrapping_pulls_a_site_across_the_edge_to_its_own_cell() {
        let mut m = Moments::default();
        m.add((1.0, 1.0, 0.0));
        let (x, y, _) = m.unwrap(63.0, 1.0, 0.0, (64.0, 64.0, 1.0));
        assert!((x + 1.0).abs() < 1e-12, "{x}");
        assert!((y - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_flat_lattice_never_unwraps_the_third_axis() {
        let mut m = Moments::default();
        m.add((1.0, 1.0, 0.0));
        let (_, _, z) = m.unwrap(0.0, 0.0, 0.0, (64.0, 64.0, 1.0));
        assert_eq!(z, 0.0);
    }

    #[test]
    fn anisotropy_points_along_the_long_axis_and_vanishes_for_a_square() {
        let (a, b) = filled(8, 8, 1).anisotropy();
        assert!(a.abs() < 1e-12 && b.abs() < 1e-12, "{a} {b}");
        let (a, b) = filled(16, 2, 1).anisotropy();
        assert!(a > 0.9 && b.abs() < 1e-12, "{a} {b}");
        let (a, _) = filled(2, 16, 1).anisotropy();
        assert!(a < -0.9, "{a}");
    }

    #[test]
    fn a_cell_of_one_site_has_no_axis() {
        let mut m = Moments::default();
        m.add((3.0, 3.0, 0.0));
        assert_eq!(m.length(), 0.0);
    }
}
