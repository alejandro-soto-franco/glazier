# Exchange surface of two agent-based tissue engines

Measured 2026-09-10 against CompuCell3D 4.10 and PhysiCell 1.14.2, both running
locally: the CompuCell3D monolayer in `sims/nematic_tissue.py`, and PhysiCell's
own `virus_macrophage` sample project, 720 simulated minutes over a 1000 by 1000
micron domain, 1031 cells and two diffusing substrates.

A shared model description has to state everything an engine needs to reproduce
a run. Both engines split that between a declarative file and compiled or
interpreted code, and the split falls in a different place in each.

## Declarative surface

**CompuCell3D.** CC3DML, or the equivalent `cc3d.core.PyCoreSpecs` objects,
states the lattice and its boundary conditions, the Monte Carlo temperature and
neighbour order, the cell types, and one parameter block per energy
term. The contact-energy matrix sets which types stick to which, the volume and
length constraints set cell size and elongation, and chemical fields take their
own diffusion parameters. Every term is a named plugin with a fixed parameter
set, and the initial condition is one more such plugin.

**PhysiCell.** `PhysiCell_settings.xml` states the domain and mesh, three
separate timesteps (diffusion, mechanics, phenotype), the substrates with their
diffusion and decay coefficients and Dirichlet conditions, and a
`cell_definitions` tree per cell type: cycle model and transition rates, death
models, volume, mechanics, motility, chemotaxis, secretion, live phagocytosis
rates, attack rates, fusion rates, cell transformations, and a `custom_data`
block of named scalars. The initial condition is a separate `cells.csv`. The
random seed is a first-class element.

PhysiCell states more of a model declaratively than CompuCell3D does, and its
phenotype tree is close to a schema already.

## Mechanism in code

**CompuCell3D** puts the mechanism in Python steppables. A steppable is an
arbitrary class with `start` and `step` methods and full access to the cell
inventory, so a rule such as "an infected cell secretes a cytokine and dies
above a threshold" is a loop over `self.cell_list`, with no schema anywhere.

**PhysiCell** puts the mechanism in `custom_modules/custom.cpp`. In the
virus-macrophage sample, the XML supplies `min_virion_count` and
`burst_virion_count` as `custom_data`, and the C++ decides what happens between
them: the response interpolates linearly in internal virion count, and the cell
apoptoses at the burst count. The rule is a compiled function pointer assigned
to the cell definition.

PhysiCell also ships a declarative rule grammar, a `cell_rules` CSV of the form
signal, direction, behaviour, saturation, half-max, Hill power. That grammar is
the closest existing thing to the mechanism half of a Blueprint. It does not
cover the sample's own viral model, which is why that model is C++.

## Gaps a Blueprint closes

Four things neither file states in full.

**Units and their calibration.** PhysiCell is explicit: minutes and microns,
declared per element. A cellular Potts lattice has no physical time or length
until a calibration is supplied, and the Monte Carlo step is not a fixed
physical interval. A shared description that omits the mapping is unportable by
construction.

**Update schedule.** PhysiCell advances three processes on three timesteps.
CompuCell3D advances one Monte Carlo sweep and calls steppables at a stated
frequency. Two engines given the same rates and no schedule produce different
trajectories.

**Stochasticity.** PhysiCell states a `random_seed`; the CompuCell3D
specification objects do not expose one, and the acceptance rule consumes random
numbers at a rate that depends on the lattice. Reproducibility across engines is
a matter of stated distributions rather than of a shared stream.

**Geometry class.** A cellular Potts cell is a set of lattice sites with a
deformable boundary and an energy functional. A PhysiCell agent is an
off-lattice sphere with a radius and a force law. Adhesion in one is a contact
energy per unit boundary, and in the other a spring coefficient per contact.
The two are not translations of each other, so a Blueprint either states the
biology and lets each engine realise the geometry, or it declares a geometry
class and admits that some models reach only one engine.

Measured, 2026-09-10, with one description run on CompuCell3D, on both glazier
engines and on PhysiCell, 256 cells over 200 steps: cell count agrees exactly
on all four, and so does the net disclination charge of zero. Mean cell area
comes out at 64.0 sites on the three lattice engines and 58.5 on PhysiCell,
the loss nearest-centre rasterisation takes on overlapping discs. Mean
eccentricity is 0.40 on the lattice engines and 0.042 on PhysiCell, so shape
transfers not at all. A description that states adhesion as a contact energy
therefore reaches one geometry class; the `physicell` block in
`blueprints/monolayer-physicell.json` states the centre-based strengths
outright, since nothing derives them.

## Measured connectivity

A description can state that a cell stays in one piece. CompuCell3D walks the
cell's whole site graph on every copy attempt; `glazier` reads the
neighbourhood of the site being taken and asks whether the pieces there touch,
the most a parallel sweep can do. Both keep cells whole and they refuse
different sets of copies.

On `monolayer-connected.json`, 256 cells over 200 steps at a temperature that
tears an unconstrained sheet apart, the three lattice engines agree on cell
count, on mean area of 64.0, on a spread about target of 3.50 against 3.39 and
3.45, and on a net disclination charge of zero. Mean eccentricity reads 0.512
under the global rule against 0.404 and 0.405 under the local one.

So a description that says "connected" states an intent rather than a
procedure, and the tissue that comes out depends on which procedure an engine
uses. A Blueprint has to say which, or say that the quantity it is read for
survives either.

## Reading a finished run

Both engines are readable from Python without their own interface. PhysiCell
writes MultiCellDS XML and MAT per save interval, and `pcdl` returns a cell
table of 128 columns and a concentration table per voxel. CompuCell3D exposes
the cell field and the cell inventory inside a steppable, which is where
`sims/nematic_tissue.py` writes its label field. Neither format states the model
that produced it, so provenance is a Blueprint's problem as well.
