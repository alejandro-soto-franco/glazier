//! Serial Metropolis on the CPU. This is the correctness reference.

use crate::field::{Exchange, Fields};
use crate::lattice::Lattice;
use crate::model::Model;
use crate::moments::{Moments, Site};
use crate::motility::Activity;
use crate::rng::Xoshiro;

/// Second moments per label, unwrapped about each cell's first site.
fn count_moments(lattice: &Lattice, labels: usize) -> Vec<Moments> {
    let mut moments = vec![Moments::default(); labels];
    let extent = lattice_extent(lattice);
    for index in 0..lattice.labels.len() {
        let label = lattice.labels[index] as usize;
        let (x, y, z) = lattice.coords(index);
        let site = moments[label].unwrap(x as f64, y as f64, z as f64, extent);
        moments[label].add(site);
    }
    moments
}

/// The lattice sides, for the minimum image convention.
fn lattice_extent(lattice: &Lattice) -> (f64, f64, f64) {
    (
        lattice.width as f64,
        lattice.height as f64,
        lattice.depth as f64,
    )
}

/// Bonds from each label's sites to sites of any other label.
fn count_surface(lattice: &Lattice, labels: usize) -> Vec<i64> {
    let mut surface = vec![0i64; labels];
    for index in 0..lattice.labels.len() {
        let (x, y, z) = lattice.coords(index);
        let l = lattice.labels[index];
        for &(dx, dy, dz) in &lattice.offsets {
            let n = lattice.index(x + dx, y + dy, z + dz);
            if n != index && lattice.labels[n] != l {
                surface[l as usize] += 1;
            }
        }
    }
    surface
}

/// A running simulation: the lattice, one volume and one type per cell, and
/// the random stream.
#[derive(Clone, Debug)]
pub struct Simulation {
    /// The model this run realises.
    pub model: Model,
    /// The site labels.
    pub lattice: Lattice,
    /// Volume per label. Index 0 is the medium, counted like any other so the
    /// total over the vector is the lattice, and never constrained.
    pub volume: Vec<u32>,
    /// Surface per label, counted as bonds from a cell's sites to sites of any
    /// other label, over the model's own neighbour order.
    pub surface: Vec<i64>,
    /// Type per label.
    pub cell_type: Vec<u8>,
    /// Second moments per label, for the length constraint.
    pub moments: Vec<Moments>,
    /// How recently each site was taken, for the types that keep a memory.
    pub activity: Activity,
    /// Diffusing species on the same lattice.
    pub fields: Fields,
    /// Secretion and uptake per cell type.
    pub exchange: Vec<Exchange>,
    /// Whether any type constrains its length, read once rather than scanned
    /// on every copy attempt.
    constrains_length: bool,
    /// The neighbourhood the connectivity test reads, empty when no type asks
    /// for one.
    connectivity_ring: Vec<(i64, i64, i64)>,
    rng: Xoshiro,
    /// Monte Carlo steps completed.
    pub mcs: u64,
    /// Copy attempts accepted since the run started.
    pub accepted: u64,
    /// Copy attempts made since the run started.
    pub attempted: u64,
}

impl Simulation {
    /// Start a run from a tiled lattice of square cells, all of type 1.
    ///
    /// # Errors
    /// Whatever [`Model::validate`] reports.
    pub fn tiled(model: Model, side: usize) -> Result<Self, String> {
        let (nx, ny, nz) = (
            model.width / side,
            model.height / side,
            (model.depth / side).max(1),
        );
        Self::tiled_grid(model, side, nx, ny, nz)
    }

    /// Start a run from `nx` by `ny` square cells, leaving the rest medium.
    ///
    /// # Errors
    /// Whatever [`Model::validate`] reports.
    pub fn tiled_grid(
        model: Model,
        side: usize,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> Result<Self, String> {
        model.validate()?;
        if model.n_types() < 2 {
            return Err("a tiled start needs a cell type beside the medium".into());
        }
        let mut lattice = Lattice::medium(
            model.width,
            model.height,
            model.depth,
            model.neighbour_order,
        );
        let n = lattice.tile_grid(side, nx, ny, nz) as usize;
        let mut volume = vec![0u32; n + 1];
        for &label in &lattice.labels {
            volume[label as usize] += 1;
        }
        let surface = count_surface(&lattice, n + 1);
        let moments = count_moments(&lattice, n + 1);
        let sites = lattice.labels.len();
        let species = model.species.clone();
        let dimensions = model.dimensions();
        let constrains_length = model.has_length_constraint();
        let connectivity_ring = if model.has_connectivity() {
            crate::connectivity::ring(model.depth)
        } else {
            Vec::new()
        };
        let exchange = model.exchange.clone();
        let seed = model.seed;
        Ok(Self {
            model,
            lattice,
            volume,
            surface,
            cell_type: vec![1; n + 1],
            moments,
            activity: Activity::new(sites),
            fields: Fields::new(species, sites, dimensions),
            exchange,
            constrains_length,
            connectivity_ring,
            rng: Xoshiro::seed(seed),
            mcs: 0,
            accepted: 0,
            attempted: 0,
        })
    }

    /// Set the type of each cell, in label order, from a list one per cell.
    pub fn set_cell_types(&mut self, types: &[u8]) {
        for (label, &kind) in types.iter().enumerate() {
            self.cell_type[label + 1] = kind;
        }
    }

    /// Number of cells, excluding the medium.
    #[must_use]
    pub fn n_cells(&self) -> usize {
        self.volume.len() - 1
    }

    fn type_of(&self, label: u32) -> u8 {
        self.cell_type[label as usize]
    }

    /// Energy change if site `target` took the label `new`.
    ///
    /// The contact term counts only unlike-label bonds, which is the
    /// `(1 - delta)` factor of the Potts Hamiltonian. The volume term is the
    /// change in `lambda (V - V_target)^2` for the two cells involved.
    #[must_use]
    pub fn delta_energy(&self, target: usize, new: u32) -> f64 {
        let old = self.lattice.labels[target];
        if old == new {
            return 0.0;
        }
        let (tx, ty, tz) = self.lattice.coords(target);
        let (told, tnew) = (self.type_of(old), self.type_of(new));

        let mut delta = 0.0;
        let mut like_old = 0i64;
        let mut like_new = 0i64;
        let mut bonds = 0i64;
        for &(dx, dy, dz) in &self.lattice.offsets {
            let n = self.lattice.index(tx + dx, ty + dy, tz + dz);
            if n == target {
                continue;
            }
            bonds += 1;
            let ln = self.lattice.labels[n];
            let tn = self.type_of(ln);
            if ln != old {
                delta -= self.model.contact_energy(told, tn);
            } else {
                like_old += 1;
            }
            if ln != new {
                delta += self.model.contact_energy(tnew, tn);
            } else {
                like_new += 1;
            }
        }

        delta += self.volume_term(old, -1);
        delta += self.volume_term(new, 1);
        if self.constrains_length {
            let site = (tx as f64, ty as f64, tz as f64);
            delta += self.length_term(old, site, -1.0);
            delta += self.length_term(new, site, 1.0);
        }
        delta += self.surface_term(old, 2 * like_old - bonds);
        delta += self.surface_term(new, bonds - 2 * like_new);
        delta
    }

    /// Change in the length energy of `label` when a site is added or removed.
    fn length_term(&self, label: u32, site: (f64, f64, f64), sign: f64) -> f64 {
        if label == 0 {
            return 0.0;
        }
        let spec = self.model.types[self.type_of(label) as usize];
        if spec.lambda_length == 0.0 {
            return 0.0;
        }
        let moments = self.moments[label as usize];
        let unwrapped = moments.unwrap(site.0, site.1, site.2, lattice_extent(&self.lattice));
        let before = moments.length();
        let after = moments.with(unwrapped, sign).length();
        spec.lambda_length
            * ((after - spec.target_length).powi(2) - (before - spec.target_length).powi(2))
    }

    /// Work the cell gaining the site does against its own memory and its
    /// drift.
    ///
    /// Both read the two sites a copy runs between, so both are properties of
    /// the move rather than of the configuration and neither appears in
    /// [`Simulation::energy`].
    #[must_use]
    pub fn move_work(&self, new: u32, old: u32, target: usize, source: usize) -> f64 {
        let mut work = 0.0;

        if new != 0 {
            let spec = self.model.types[self.type_of(new) as usize];
            if spec.lambda_activity != 0.0 && spec.max_activity > 0.0 {
                let into = self.activity.neighbourhood_mean(&self.lattice, source, new);
                let out_of = self.activity.neighbourhood_mean(&self.lattice, target, old);
                work -= spec.lambda_activity / spec.max_activity * (into - out_of);
            }
            if spec.external.iter().any(|&v| v != 0.0) {
                let (tx, ty, tz) = self.lattice.coords(target);
                let (sx, sy, sz) = self.lattice.coords(source);
                let (w, h, d) = lattice_extent(&self.lattice);
                let step = |a: i64, b: i64, span: f64| {
                    let raw = (a - b) as f64;
                    if raw > span / 2.0 {
                        raw - span
                    } else if raw < -span / 2.0 {
                        raw + span
                    } else {
                        raw
                    }
                };
                work -= spec.external[0] * step(tx, sx, w)
                    + spec.external[1] * step(ty, sy, h)
                    + spec.external[2] * step(tz, sz, d);
            }
        }
        work
    }

    /// Work done against a chemical gradient by the cell gaining the site.
    ///
    /// This is a property of the move rather than of the configuration: it
    /// reads the field at the two sites the copy runs between, so it has no
    /// counterpart in [`Simulation::energy`] and cannot be checked by
    /// recomputing a total. A positive sensitivity makes a move up the
    /// gradient cheaper, which is what makes a cell climb it.
    #[must_use]
    pub fn chemotaxis_work(&self, new: u32, target: usize, source: usize) -> f64 {
        if new == 0 || self.model.chemotaxis.is_empty() {
            return 0.0;
        }
        let row = &self.model.chemotaxis[self.type_of(new) as usize];
        let mut work = 0.0;
        for (index, &lambda) in row.iter().enumerate() {
            if lambda == 0.0 {
                continue;
            }
            let values = &self.fields.values[index];
            work -= lambda * (values[target] - values[source]);
        }
        work
    }

    /// Change in the surface energy of `label` when its surface moves by
    /// `change` bonds.
    fn surface_term(&self, label: u32, change: i64) -> f64 {
        if label == 0 || change == 0 {
            return 0.0;
        }
        let spec = self.model.types[self.type_of(label) as usize];
        if spec.lambda_surface == 0.0 {
            return 0.0;
        }
        let s = self.surface[label as usize] as f64;
        let after = s + change as f64;
        spec.lambda_surface
            * ((after - spec.target_surface).powi(2) - (s - spec.target_surface).powi(2))
    }

    fn volume_term(&self, label: u32, change: i64) -> f64 {
        if label == 0 {
            return 0.0;
        }
        let spec = self.model.types[self.type_of(label) as usize];
        if spec.lambda_volume == 0.0 {
            return 0.0;
        }
        let v = f64::from(self.volume[label as usize]);
        let after = v + change as f64;
        spec.lambda_volume
            * ((after - spec.target_volume).powi(2) - (v - spec.target_volume).powi(2))
    }

    /// One copy attempt: a random target site takes the label of a random
    /// neighbour, accepted by the Metropolis rule.
    ///
    /// A copy that would empty a cell is refused, so a cell never vanishes and
    /// the label set is fixed for the run.
    pub fn attempt(&mut self) -> bool {
        let sites = self.lattice.labels.len() as u64;
        let target = self.rng.below(sites) as usize;
        let pick = self.rng.below(self.lattice.offsets.len() as u64) as usize;
        let (tx, ty, tz) = self.lattice.coords(target);
        let (dx, dy, dz) = self.lattice.offsets[pick];
        let source = self.lattice.index(tx + dx, ty + dy, tz + dz);

        let old = self.lattice.labels[target];
        let new = self.lattice.labels[source];
        self.attempted += 1;
        if old == new {
            return false;
        }
        if old != 0 && self.volume[old as usize] <= 1 {
            return false;
        }
        if old != 0
            && !self.connectivity_ring.is_empty()
            && self.model.types[self.type_of(old) as usize].connected
            && !crate::connectivity::locally_connected(
                &self.lattice,
                &self.connectivity_ring,
                target,
                old,
            )
        {
            return false;
        }

        let delta = self.delta_energy(target, new)
            + self.chemotaxis_work(new, target, source)
            + self.move_work(new, old, target, source);
        let accept = delta <= 0.0 || self.rng.next_f64() < (-delta / self.model.temperature).exp();
        if accept {
            let (tx, ty, tz) = self.lattice.coords(target);
            let mut like_old = 0i64;
            let mut like_new = 0i64;
            let mut bonds = 0i64;
            for &(dx, dy, dz) in &self.lattice.offsets {
                let n = self.lattice.index(tx + dx, ty + dy, tz + dz);
                if n == target {
                    continue;
                }
                bonds += 1;
                let ln = self.lattice.labels[n];
                if ln == old {
                    like_old += 1;
                }
                if ln == new {
                    like_new += 1;
                }
            }
            self.surface[old as usize] += 2 * like_old - bonds;
            self.surface[new as usize] += bonds - 2 * like_new;

            if self.constrains_length {
                let site = (tx as f64, ty as f64, tz as f64);
                let extent = lattice_extent(&self.lattice);
                let leaving = self.moments[old as usize].unwrap(site.0, site.1, site.2, extent);
                self.moments[old as usize].remove(leaving);
                let joining = self.moments[new as usize].unwrap(site.0, site.1, site.2, extent);
                self.moments[new as usize].add(joining);
            }

            if new != 0 {
                let spec = self.model.types[self.type_of(new) as usize];
                if spec.max_activity > 0.0 {
                    self.activity.refresh(target, spec.max_activity);
                }
            }

            self.lattice.labels[target] = new;
            self.volume[old as usize] -= 1;
            self.volume[new as usize] += 1;
            self.accepted += 1;
        }
        accept
    }

    /// One Monte Carlo step: as many copy attempts as there are sites, then
    /// the fields diffuse, decay and exchange with the cells that own the
    /// sites they sit on.
    pub fn step(&mut self) {
        for _ in 0..self.lattice.labels.len() {
            self.attempt();
        }
        if !self.fields.is_empty() {
            self.fields.diffuse(&self.lattice);
            self.fields
                .exchange(&self.lattice, &self.cell_type, &self.exchange);
        }
        if self.model.has_motility() {
            self.activity.decay();
        }
        if self.model.has_population_events() {
            self.divide_and_die();
        }
        self.mcs += 1;
    }

    /// Total energy of the current configuration.
    ///
    /// Each unlike-label bond is counted once, so this is comparable between
    /// neighbour orders and is what the incremental deltas must reproduce.
    #[must_use]
    pub fn energy(&self) -> f64 {
        let mut contact = 0.0;
        for index in 0..self.lattice.labels.len() {
            let (x, y, z) = self.lattice.coords(index);
            let l = self.lattice.labels[index];
            for &(dx, dy, dz) in &self.lattice.offsets {
                let n = self.lattice.index(x + dx, y + dy, z + dz);
                let ln = self.lattice.labels[n];
                if ln != l {
                    contact += self.model.contact_energy(self.type_of(l), self.type_of(ln));
                }
            }
        }
        contact /= 2.0;

        let mut volume = 0.0;
        let mut surface = 0.0;
        let mut length = 0.0;
        let counted = self.recounted_surfaces();
        for (label, &bonds) in counted.iter().enumerate().skip(1) {
            let spec = self.model.types[self.cell_type[label] as usize];
            volume +=
                spec.lambda_volume * (f64::from(self.volume[label]) - spec.target_volume).powi(2);
            if spec.lambda_length != 0.0 {
                length += spec.lambda_length
                    * (self.moments[label].length() - spec.target_length).powi(2);
            }
            surface += spec.lambda_surface * (bonds as f64 - spec.target_surface).powi(2);
        }
        contact + volume + surface + length
    }

    /// Recompute the moments from the lattice, for checking the bookkeeping.
    #[must_use]
    pub fn recounted_moments(&self) -> Vec<Moments> {
        count_moments(&self.lattice, self.moments.len())
    }

    /// Recount the surfaces from the lattice, for checking the bookkeeping.
    #[must_use]
    pub fn recounted_surfaces(&self) -> Vec<i64> {
        count_surface(&self.lattice, self.surface.len())
    }

    /// Recount the volumes from the lattice, for checking the bookkeeping.
    #[must_use]
    pub fn recounted_volumes(&self) -> Vec<u32> {
        let mut v = vec![0u32; self.volume.len()];
        for &label in &self.lattice.labels {
            v[label as usize] += 1;
        }
        v
    }
}

impl Simulation {
    /// Cells that have reached their division volume split, and cells of a
    /// type with a death rate die.
    ///
    /// A dividing cell is cut by the line through its centroid perpendicular
    /// to its major axis, so the halves are the compact ones. A dying cell's
    /// sites become medium. Both change the boundary of every neighbour, so
    /// the surfaces are recounted whenever either happens rather than patched.
    pub fn divide_and_die(&mut self) -> (usize, usize) {
        let mut divided = 0usize;
        let mut died = 0usize;

        let live: Vec<u32> = (1..self.volume.len() as u32)
            .filter(|&label| self.volume[label as usize] > 0)
            .collect();

        for label in live {
            let spec = self.model.types[self.type_of(label) as usize];
            if spec.division_volume > 0.0
                && f64::from(self.volume[label as usize]) >= spec.division_volume
                && self.divide(label)
            {
                divided += 1;
            }
        }

        let live: Vec<u32> = (1..self.volume.len() as u32)
            .filter(|&label| self.volume[label as usize] > 0)
            .collect();
        for label in live {
            let spec = self.model.types[self.type_of(label) as usize];
            if spec.death_rate > 0.0 && self.rng.next_f64() < spec.death_rate {
                self.kill(label);
                died += 1;
            }
        }

        if divided > 0 || died > 0 {
            self.surface = self.recounted_surfaces();
            self.moments = self.recounted_moments();
        }
        (divided, died)
    }

    /// Sites of a label, unwrapped about the first one so a cell straddling
    /// the periodic edge still has a centroid and an axis.
    fn unwrapped_sites(&self, label: u32) -> (Vec<(usize, Site)>, Moments) {
        let extent = lattice_extent(&self.lattice);
        let mut moments = Moments::default();
        let mut out: Vec<(usize, Site)> = Vec::new();
        for (site, &l) in self.lattice.labels.iter().enumerate() {
            if l != label {
                continue;
            }
            let (x, y, z) = self.lattice.coords(site);
            let unwrapped = moments.unwrap(x as f64, y as f64, z as f64, extent);
            moments.add(unwrapped);
            out.push((site, unwrapped));
        }
        (out, moments)
    }

    /// Split one cell in two. Returns whether it happened.
    ///
    /// The cut is the plane through the centroid perpendicular to the major
    /// axis, so the halves are the compact ones whatever the dimension.
    fn divide(&mut self, label: u32) -> bool {
        let (sites, moments) = self.unwrapped_sites(label);
        if sites.len() < 4 {
            return false;
        }

        let centre = moments.centroid();
        let (axis, _) = moments.principal();

        let daughter = self.volume.len() as u32;
        let mut moved = 0u32;
        for &(site, (x, y, z)) in &sites {
            let along = (x - centre.0) * axis.0 + (y - centre.1) * axis.1 + (z - centre.2) * axis.2;
            if along > 0.0 {
                self.lattice.labels[site] = daughter;
                moved += 1;
            }
        }
        if moved == 0 || moved as usize == sites.len() {
            // A cut that takes everything or nothing is no division.
            for &(site, _) in &sites {
                self.lattice.labels[site] = label;
            }
            return false;
        }

        self.volume[label as usize] -= moved;
        self.volume.push(moved);
        self.surface.push(0);
        self.moments.push(Moments::default());
        self.cell_type.push(self.type_of(label));
        true
    }

    /// Turn a cell's sites back into medium.
    fn kill(&mut self, label: u32) {
        for site in 0..self.lattice.labels.len() {
            if self.lattice.labels[site] == label {
                self.lattice.labels[site] = 0;
            }
        }
        let gone = self.volume[label as usize];
        self.volume[label as usize] = 0;
        self.volume[0] += gone;
        self.surface[label as usize] = 0;
    }

    /// Labels with sites on the lattice.
    #[must_use]
    pub fn live_cells(&self) -> usize {
        self.volume[1..].iter().filter(|&&v| v > 0).count()
    }
}
