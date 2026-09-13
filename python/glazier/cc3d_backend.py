"""CompuCell3D backend for a Blueprint.

The steppable is defined at module level because the simulation service sends
it to its own process by name. It reports back through files, so nothing has to
cross that boundary as an object.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from cc3d.core.PyCoreSpecs import (
    CellTypePlugin,
    ConnectivityGlobalPlugin,
    ContactPlugin,
    PottsCore,
    SurfacePlugin,
    UniformInitializer,
    VolumePlugin,
)
from cc3d.core.PySteppables import SteppableBasePy
from cc3d.core.simservice import service_cc3d

from glazier.blueprint import Blueprint

OUT_DIR = Path("runs/cc3d")
DUMP_EVERY = 0
TOTAL_STEPS = 0


class DumpSteppable(SteppableBasePy):
    """Write the label field on the dump interval and on the last step."""

    def __init__(self, frequency: int = 1):
        super().__init__(frequency)

    def step(self, mcs: int):
        last = mcs == TOTAL_STEPS - 1
        on_interval = DUMP_EVERY > 0 and mcs > 0 and mcs % DUMP_EVERY == 0
        if not (last or on_interval):
            return

        cell_field = self.cell_field
        if cell_field is None:
            raise RuntimeError("the simulator has not attached a cell field")
        field = np.zeros((self.dim.y, self.dim.x), dtype=np.uint32)
        for x in range(self.dim.x):
            for y in range(self.dim.y):
                cell = cell_field[x, y, 0]
                if cell is not None:
                    field[y, x] = cell.id

        if last:
            np.save(OUT_DIR / "labels_final.npy", field)
        else:
            np.save(OUT_DIR / f"labels_{mcs:05d}.npy", field)


def specs(bp: Blueprint) -> list:
    """The CompuCell3D specification objects this description states."""
    names = bp.type_names

    potts = PottsCore(
        dim_x=bp.width,
        dim_y=bp.height,
        dim_z=1,
        steps=10**9,
        neighbor_order=bp.neighbour_order,
        boundary_x="Periodic",
        boundary_y="Periodic",
        fluctuation_amplitude=bp.temperature,
        random_seed=bp.seed,
    )

    cell_type = CellTypePlugin(*names[1:])

    volume = VolumePlugin()
    for index, spec in enumerate(bp.types, start=1):
        volume.param_new(
            names[index],
            target_volume=spec.target_volume,
            lambda_volume=spec.lambda_volume,
        )

    parts_surface: list[SurfacePlugin | ConnectivityGlobalPlugin] = []
    if any(spec.lambda_surface for spec in bp.types):
        surface = SurfacePlugin()
        for index, spec in enumerate(bp.types, start=1):
            surface.param_new(
                names[index],
                target_surface=spec.target_surface,
                lambda_surface=spec.lambda_surface,
            )
        parts_surface.append(surface)

    if any(spec.connected for spec in bp.types):
        # CompuCell3D walks a cell's whole site graph.
        # glazier reads only the neighbourhood of the copy.
        # The two therefore refuse different copies and both keep cells whole.
        connectivity = ConnectivityGlobalPlugin()
        for index, spec in enumerate(bp.types, start=1):
            if spec.connected:
                connectivity.cell_type_append(names[index])
        parts_surface.append(connectivity)

    contact = ContactPlugin(neighbor_order=bp.neighbour_order)
    for a in range(len(names)):
        for b in range(a, len(names)):
            contact.param_new(names[a], names[b], bp.contact_energy(a, b))

    side = bp.initial["side"]
    initializer = UniformInitializer()
    initializer.region_new(
        pt_min=(0, 0, 0),
        pt_max=(side * bp.initial["nx"], side * bp.initial["ny"], 1),
        width=side,
        cell_types=[names[1]],
    )

    return [potts, cell_type, volume, *parts_surface, contact, initializer]


def run(bp: Blueprint, out_dir: str | Path) -> dict:
    """Run the description and write it out the way glazier does."""
    global OUT_DIR, DUMP_EVERY, TOTAL_STEPS

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    OUT_DIR = out
    DUMP_EVERY = bp.dump_every
    # One step past the run captures the field. The dump then falls outside
    # the measurement below.
    TOTAL_STEPS = bp.steps + 1

    sim = service_cc3d()
    sim.register_specs(specs(bp))
    sim.register_steppable(DumpSteppable, frequency=1)
    sim.run()
    sim.init()
    sim.start()

    # Reading the lattice out is a per-site loop in Python. It runs outside the
    # measurement: the timed loop dumps nothing and one further step captures
    # the field. Every setting the steppable reads is fixed before the
    # service starts, since the class runs in its own process and sees this
    # module as it stood then.
    started = time.perf_counter()
    for _ in range(bp.steps):
        sim.step()
    seconds = time.perf_counter() - started

    sim.step()
    sim.finish()

    final = out / "labels_final.npy"
    if not final.exists():
        raise RuntimeError(f"the run wrote no final field to {final}")
    labels = np.load(final)
    np.save(out / f"labels_{bp.steps:05d}.npy", labels)

    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    summary = {
        "name": bp.name,
        "engine": "cc3d",
        "steps": bp.steps,
        "seconds": seconds,
        "n_cells": int(ids.size),
        "mean_volume": float(counts.mean()) if ids.size else 0.0,
        "width": bp.width,
        "height": bp.height,
        "micron_per_site": bp.units["micron_per_site"],
        "minute_per_step": bp.units["minute_per_step"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
