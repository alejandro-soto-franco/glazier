"""Measure a rendered tissue with mermin. Runs in mermin's own interpreter.

mermin publishes wheels for the interpreter its build box runs, and cc3d pins
an older one, so the two live in separate environments and meet through files.
That split is the same discipline the measurement itself is testing: one
artefact on disk, two independent programs that agree on how to read it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mermin import analyze


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pixel-size-um", type=float, default=0.345)
    parser.add_argument("--segmentation", default="threshold")
    args = parser.parse_args()

    result = analyze(
        str(args.image),
        channels={"nuclear": 0, "fibre": 1},
        pixel_size_um=args.pixel_size_um,
        segmentation=args.segmentation,
        k_values=[2],
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result.cells.write_parquet(args.out_dir / "mermin_cells.parquet")
    summary = {
        "n_cells": len(result.cells),
        "n_defects": len(result.defects),
        "defect_charge_sum": sum(d.get("charge", 0.0) for d in result.defects),
        "frank": result.frank,
        "ldg_params": result.ldg_params,
        "pixel_size_um": args.pixel_size_um,
    }
    (args.out_dir / "mermin_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
