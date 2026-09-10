"""One Blueprint, every engine that can read it, compared on what a run is read for.

CompuCell3D, glazier on the CPU, glazier on the GPU and PhysiCell each
instantiate the same JSON file. The engines sample by different chains, and
PhysiCell is a different geometry class altogether, so the comparison is on the
population quantities: how many cells, how big, how far from target, how
aligned, and how many disclinations the texture has.

PhysiCell runs only when the description states a `physicell` block, since a
contact energy has no counterpart in a centre-based model and the adhesion and
repulsion strengths have to be stated rather than derived.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from glazier.blueprint import Blueprint
from glazier.director import (
    cell_shapes,
    coarse_grain,
    defect_charges,
    director_angle,
    global_order,
    q_field,
)

# The engine binary is built into the workspace target directory at the root.
GLAZIER = Path(__file__).resolve().parents[1] / "target" / "release" / "glazier"


def run_glazier(model: Path, out: Path, engine: str) -> dict:
    subprocess.run(
        [str(GLAZIER), "--model", str(model), "--out", str(out), "--engine", engine],
        check=True,
        capture_output=True,
    )
    return json.loads((out / "summary.json").read_text())


def run_physicell(model: Path, out: Path) -> dict:
    script = (
        "from glazier.blueprint import Blueprint\n"
        "from glazier import physicell_backend\n"
        f"bp = Blueprint.load({str(model)!r})\n"
        f"physicell_backend.run(bp, {str(out)!r})\n"
    )
    subprocess.run(["python", "-c", script], check=True, capture_output=True)
    return json.loads((out / "summary.json").read_text())


def run_cc3d(model: Path, out: Path) -> dict:
    script = (
        "from glazier.blueprint import Blueprint\n"
        "from glazier import cc3d_backend\n"
        f"bp = Blueprint.load({str(model)!r})\n"
        f"cc3d_backend.run(bp, {str(out)!r})\n"
    )
    subprocess.run(["python", "-c", script], check=True, capture_output=True)
    return json.loads((out / "summary.json").read_text())


def measure(labels: np.ndarray, target_volume: float, sigma: float) -> dict:
    shapes = cell_shapes(labels)
    areas = np.array([s.area for s in shapes], dtype=float)
    order, _ = global_order(shapes)
    q = coarse_grain(q_field(labels, shapes), sigma)
    defects = defect_charges(director_angle(q))
    return {
        "cells": len(shapes),
        "mean_area": float(areas.mean()) if areas.size else 0.0,
        "area_rms_from_target": float(np.sqrt(((areas - target_volume) ** 2).mean()))
        if areas.size
        else 0.0,
        "mean_eccentricity": float(np.mean([s.eccentricity for s in shapes]))
        if shapes
        else 0.0,
        "nematic_order": order,
        "defects": len(defects),
        "net_charge": float(sum(c for _, _, c in defects)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/three-engines"))
    parser.add_argument("--coarse-grain-sigma", type=float, default=4.0)
    args = parser.parse_args()

    bp = Blueprint.load(args.model)
    target = bp.types[0].target_volume
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = {
        "cc3d": run_cc3d(args.model, args.out_dir / "cc3d"),
        "glazier-cpu": run_glazier(args.model, args.out_dir / "glazier-cpu", "cpu"),
        "glazier-gpu": run_glazier(args.model, args.out_dir / "glazier-gpu", "gpu"),
    }
    if bp.physicell is not None:
        runs["physicell"] = run_physicell(args.model, args.out_dir / "physicell")

    rows = {}
    for engine, summary in runs.items():
        labels = np.load(args.out_dir / engine / f"labels_{bp.steps:05d}.npy")
        row = measure(labels, target, args.coarse_grain_sigma)
        row["seconds"] = summary["seconds"]
        rows[engine] = row

    columns = [
        "cells",
        "mean_area",
        "area_rms_from_target",
        "mean_eccentricity",
        "nematic_order",
        "defects",
        "net_charge",
        "seconds",
    ]
    print(f"{bp.name}: {bp.width} by {bp.height}, {bp.steps} steps, seed {bp.seed}\n")
    print(f"{'quantity':>22s}" + "".join(f"{e:>14s}" for e in rows))
    for column in columns:
        values = "".join(f"{rows[e][column]:>14.3f}" for e in rows)
        print(f"{column:>22s}{values}")

    (args.out_dir / "comparison.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
