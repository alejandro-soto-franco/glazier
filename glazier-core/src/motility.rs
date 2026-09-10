//! Persistent motility: a cell that has just moved keeps moving.
//!
//! Every site remembers how recently it was taken, and a copy is priced
//! against the difference in that memory between the two sites it runs
//! between. A cell that extends in one direction therefore leaves a trail of
//! recent sites behind its front, and extending further along the same
//! direction is the cheaper move: the cell polarises and travels, where a
//! plain Potts cell only jiggles.
//!
//! This is the Act model of Niculescu, Textor and de Boer. The neighbourhood
//! average is a geometric mean, which is zero as soon as one neighbour of the
//! cell has forgotten, so the memory acts as a front rather than as a haze.

use crate::lattice::Lattice;

/// One activity value per site.
#[derive(Clone, Debug)]
pub struct Activity {
    /// Steps of memory remaining at each site.
    pub values: Vec<f64>,
}

impl Activity {
    /// A lattice that remembers nothing.
    #[must_use]
    pub fn new(sites: usize) -> Self {
        Self {
            values: vec![0.0; sites],
        }
    }

    /// Whether any site remembers anything.
    #[must_use]
    pub fn is_quiet(&self) -> bool {
        self.values.iter().all(|&v| v <= 0.0)
    }

    /// Take one step off every site's memory.
    pub fn decay(&mut self) {
        for value in &mut self.values {
            if *value > 0.0 {
                *value -= 1.0;
            }
        }
    }

    /// Give a site the full memory a type states.
    pub fn refresh(&mut self, site: usize, max_activity: f64) {
        self.values[site] = max_activity;
    }

    /// Geometric mean of the activity over the sites of `label` around
    /// `site`, counting the site itself.
    ///
    /// Zero when any of them has forgotten, which is what makes the memory a
    /// front. A site whose cell has no other site nearby reads its own value.
    #[must_use]
    pub fn neighbourhood_mean(&self, lattice: &Lattice, site: usize, label: u32) -> f64 {
        let (x, y, z) = lattice.coords(site);
        let mut log_sum = self.values[site].max(0.0).ln();
        let mut count = 1.0f64;
        for &(dx, dy, dz) in &lattice.offsets {
            let n = lattice.index(x + dx, y + dy, z + dz);
            if n == site || lattice.labels[n] != label {
                continue;
            }
            let value = self.values[n];
            if value <= 0.0 {
                return 0.0;
            }
            log_sum += value.ln();
            count += 1.0;
        }
        if !log_sum.is_finite() {
            return 0.0;
        }
        (log_sum / count).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_cell_lattice() -> Lattice {
        let mut lattice = Lattice::medium(8, 8, 1, 2);
        for x in 0..4 {
            for y in 0..8 {
                let index = lattice.index(x, y, 0);
                lattice.labels[index] = 1;
            }
        }
        lattice
    }

    #[test]
    fn decay_takes_one_step_and_stops_at_nothing() {
        let mut activity = Activity::new(4);
        activity.refresh(0, 2.0);
        activity.decay();
        assert_eq!(activity.values[0], 1.0);
        activity.decay();
        activity.decay();
        assert_eq!(activity.values[0], 0.0);
    }

    #[test]
    fn a_forgotten_neighbour_takes_the_mean_to_nothing() {
        let lattice = two_cell_lattice();
        let mut activity = Activity::new(lattice.len());
        let site = lattice.index(1, 4, 0);
        activity.refresh(site, 10.0);
        assert_eq!(activity.neighbourhood_mean(&lattice, site, 1), 0.0);
    }

    #[test]
    fn a_uniformly_remembered_cell_reads_its_own_value() {
        let lattice = two_cell_lattice();
        let mut activity = Activity::new(lattice.len());
        for (site, &label) in lattice.labels.iter().enumerate() {
            if label == 1 {
                activity.refresh(site, 7.0);
            }
        }
        let site = lattice.index(1, 4, 0);
        assert!((activity.neighbourhood_mean(&lattice, site, 1) - 7.0).abs() < 1e-9);
    }

    #[test]
    fn the_mean_sits_between_the_values_it_reads() {
        let lattice = two_cell_lattice();
        let mut activity = Activity::new(lattice.len());
        for (site, &label) in lattice.labels.iter().enumerate() {
            if label == 1 {
                activity.refresh(site, 4.0);
            }
        }
        let site = lattice.index(1, 4, 0);
        activity.refresh(site, 16.0);
        let mean = activity.neighbourhood_mean(&lattice, site, 1);
        assert!(mean > 4.0 && mean < 16.0, "{mean}");
    }

    #[test]
    fn a_quiet_lattice_says_so() {
        let mut activity = Activity::new(4);
        assert!(activity.is_quiet());
        activity.refresh(2, 1.0);
        assert!(!activity.is_quiet());
    }
}
