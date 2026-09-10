"""Elongated-cell tissue in a cellular Potts model, run headless through simservice.

The tissue is a monolayer of length-constrained cells at confluence. Elongation
plus contact energy produces a nematic texture with half-integer disclinations,
which is the same order that mermin measures on fluorescence microscopy. Each
dump writes the cell-id label field, so the director is recoverable per cell
from pixel second moments, with no segmentation step between the model and the
measurement.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from cc3d.core.PyCoreSpecs import (
    CellTypePlugin,
    ConnectivityGlobalPlugin,
    ContactPlugin,
    LengthConstraintPlugin,
    MomentOfInertiaPlugin,
    PixelTrackerPlugin,
    PottsCore,
    UniformInitializer,
    VolumePlugin,
)
from cc3d.core.PySteppables import SteppableBasePy
from cc3d.core.simservice import service_cc3d

CELL_TYPE = "Cell"


class DumpSteppable(SteppableBasePy):
    """Write the label field and a per-cell table every `frequency` steps."""

    out_dir = Path("runs/default")

    def __init__(self, frequency: int = 50):
        super().__init__(frequency)

    def start(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def step(self, mcs: int):
        dim_x = self.dim.x
        dim_y = self.dim.y
        labels = np.zeros((dim_y, dim_x), dtype=np.int32)
        for x in range(dim_x):
            for y in range(dim_y):
                cell = self.cell_field[x, y, 0]
                if cell is not None:
                    labels[y, x] = cell.id
        np.save(self.out_dir / f"labels_{mcs:05d}.npy", labels)

        with open(self.out_dir / f"cells_{mcs:05d}.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "type", "volume", "com_x", "com_y"])
            for cell in self.cell_list:
                writer.writerow(
                    [cell.id, cell.type, cell.volume, cell.xCOM, cell.yCOM]
                )


def specs(dim: int, target_volume: float, target_length: float, temperature: float):
    potts = PottsCore(
        dim_x=dim,
        dim_y=dim,
        dim_z=1,
        steps=10**9,
        neighbor_order=2,
        boundary_x="Periodic",
        boundary_y="Periodic",
        fluctuation_amplitude=temperature,
    )
    cell_type = CellTypePlugin(CELL_TYPE)

    volume = VolumePlugin()
    volume.param_new(CELL_TYPE, target_volume=target_volume, lambda_volume=2.0)

    contact = ContactPlugin(neighbor_order=2)
    contact.param_new("Medium", CELL_TYPE, 16.0)
    contact.param_new(CELL_TYPE, CELL_TYPE, 8.0)

    length = LengthConstraintPlugin()
    length.params_new(CELL_TYPE, target_length=target_length, lambda_length=2.0)

    connectivity = ConnectivityGlobalPlugin()
    connectivity.cell_type_append(CELL_TYPE)

    initializer = UniformInitializer()
    initializer.region_new(
        pt_min=(0, 0, 0),
        pt_max=(dim, dim, 1),
        width=int(round(target_volume**0.5)),
        cell_types=[CELL_TYPE],
    )

    return [
        potts,
        cell_type,
        volume,
        contact,
        length,
        connectivity,
        MomentOfInertiaPlugin(),
        PixelTrackerPlugin(),
        initializer,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--dump-every", type=int, default=100)
    parser.add_argument("--target-volume", type=float, default=64.0)
    parser.add_argument("--target-length", type=float, default=16.0)
    parser.add_argument("--temperature", type=float, default=10.0)
    parser.add_argument("--out", type=Path, default=Path("runs/nematic"))
    args = parser.parse_args()

    DumpSteppable.out_dir = args.out

    sim = service_cc3d()
    sim.register_specs(
        specs(args.dim, args.target_volume, args.target_length, args.temperature)
    )
    sim.register_steppable(DumpSteppable, frequency=args.dump_every)
    sim.run()
    sim.init()
    sim.start()
    for _ in range(args.steps):
        sim.step()
    sim.finish()

    print(f"wrote {len(list(args.out.glob('labels_*.npy')))} label fields to {args.out}")


if __name__ == "__main__":
    main()
