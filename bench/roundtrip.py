"""Model to image to measurement, scored against the model's own director.

Takes a cellular Potts label field, renders it as microscopy, measures it with
mermin in mermin's interpreter, and reports what the measurement recovered:
cell count, centroid agreement, per-cell director error, and defect charge.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment

from glazier.director import (
    cell_shapes,
    coarse_grain,
    defect_charges,
    director_angle,
    global_order,
    q_field,
)
from glazier.render import render, upsample, write_tiff

MERMIN_PYTHON = Path.home() / "mermin" / "mermin-py" / ".venv" / "bin" / "python"

# mermin measures the nuclear ellipse angle from the other axis. A
# pixel-covariance angle theta therefore appears there as pi/2 - theta,
# measured at 0.3 degrees median over 79 matched cells; see README.
MERMIN_ANGLE = lambda t: (np.pi / 2.0 - t) % np.pi  # noqa: E731


def nematic_residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = (a - b) % np.pi
    return np.minimum(d, np.pi - d)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--upsample", type=int, default=4)
    parser.add_argument("--pixel-size-um", type=float, default=0.345)
    parser.add_argument("--coarse-grain-sigma", type=float, default=4.0)
    parser.add_argument("--match-ceiling-px", type=float, default=25.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels = np.load(args.labels)

    lattice = cell_shapes(labels)
    q = coarse_grain(q_field(labels, lattice), args.coarse_grain_sigma)
    truth_defects = defect_charges(director_angle(q))
    order, _ = global_order(lattice)

    up = upsample(labels, args.upsample)
    shapes = cell_shapes(up)
    image = render(
        up,
        shapes,
        nuclear_sigma=1.5 * args.upsample,
        fibre_period=6.0 * args.upsample,
    )
    image_path = args.out_dir / "tissue.tif"
    write_tiff(image_path, image, pixel_size_um=args.pixel_size_um)

    subprocess.run(
        [
            str(MERMIN_PYTHON),
            str(Path(__file__).parent / "measure_with_mermin.py"),
            str(image_path),
            "--out-dir",
            str(args.out_dir),
            "--pixel-size-um",
            str(args.pixel_size_um),
        ],
        check=True,
        capture_output=True,
    )

    cells = pl.read_parquet(args.out_dir / "mermin_cells.parquet")
    measured = np.c_[
        cells["centroid_x"].to_numpy() / args.pixel_size_um,
        cells["centroid_y"].to_numpy() / args.pixel_size_um,
    ]
    truth_xy = np.array([s.com for s in shapes])
    truth_theta = np.array([s.theta for s in shapes])

    distance = np.hypot(
        measured[:, None, 0] - truth_xy[None, :, 0],
        measured[:, None, 1] - truth_xy[None, :, 1],
    )
    rows, cols = linear_sum_assignment(distance)
    keep = distance[rows, cols] < args.match_ceiling_px
    rows, cols = rows[keep], cols[keep]

    residual = nematic_residual(
        cells["nuclear_angle"].to_numpy()[rows], MERMIN_ANGLE(truth_theta[cols])
    )
    summary = json.loads((args.out_dir / "mermin_summary.json").read_text())

    report = {
        "lattice_cells": len(lattice),
        "rendered_cells": len(shapes),
        "measured_cells": summary["n_cells"],
        "matched_cells": int(rows.size),
        "median_centroid_error_px": float(np.median(distance[rows, cols])),
        "director_median_error_deg": float(np.degrees(np.median(residual))),
        "director_within_15deg": float((residual < np.radians(15)).mean()),
        "global_nematic_order": order,
        "truth_defects": len(truth_defects),
        "truth_defect_charge": float(sum(c for _, _, c in truth_defects)),
        "measured_defects": summary["n_defects"],
        "measured_defect_charge": summary["defect_charge_sum"],
        "frank_ratio": summary["frank"]["ratio"],
    }
    (args.out_dir / "roundtrip.json").write_text(json.dumps(report, indent=2))
    for key, value in report.items():
        print(f"{key:32s} {value}")


if __name__ == "__main__":
    main()
