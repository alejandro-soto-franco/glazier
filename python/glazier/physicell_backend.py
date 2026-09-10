"""PhysiCell backend for a Blueprint, and the rasterisation that makes it
comparable with a lattice engine.

A Potts cell is a set of lattice sites with an energy functional; a PhysiCell
agent is an off-lattice sphere with a force law. What transfers between them is
the biology the description states: how many cells, how big each one is, how
long the run lasts, and the seed. What does not transfer is the contact energy,
which has no counterpart in a centre-based model, so the mapping states the
adhesion and repulsion strengths outright in a `physicell` block rather than
deriving them from anything.

Agent positions are rasterised onto the Blueprint's own lattice so the same
measurements run on both.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from glazier.blueprint import Blueprint

PHYSICELL = Path.home() / "local" / "PhysiCell-template"
TEMPLATE = PHYSICELL / "sample_projects" / "template" / "config" / "PhysiCell_settings.xml"

# Long enough that no cell reaches the next phase inside any run this drives.
NO_DIVISION_MINUTES = 1.0e9


def _set(root: ET.Element, path: str, value) -> None:
    node = root.find(path)
    if node is None:
        raise KeyError(f"the template has no {path}")
    node.text = str(value)


def radius_for_area(area_microns2: float) -> float:
    """Radius of a disc of the stated area."""
    return math.sqrt(area_microns2 / math.pi)


def volume_for_radius(radius: float) -> float:
    """PhysiCell states a three-dimensional volume even in two dimensions, and
    takes the radius from it by the sphere formula, so a target disc area has
    to be sent through that formula backwards."""
    return 4.0 / 3.0 * math.pi * radius**3


def write_config(bp: Blueprint, out_dir: Path) -> Path:
    """Write a settings file and an initial-condition csv for this model."""
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(TEMPLATE)
    root = tree.getroot()

    width_um, height_um = bp.domain_microns
    _set(root, "domain/x_min", -width_um / 2)
    _set(root, "domain/x_max", width_um / 2)
    _set(root, "domain/y_min", -height_um / 2)
    _set(root, "domain/y_max", height_um / 2)
    _set(root, "domain/z_min", -10)
    _set(root, "domain/z_max", 10)
    _set(root, "domain/use_2D", "true")

    _set(root, "overall/max_time", bp.duration_minutes)
    _set(root, "save/folder", str(out_dir))
    _set(root, "save/full_data/interval", bp.duration_minutes)
    root.find("save/SVG/enable").text = "false"
    _set(root, "options/random_seed", bp.seed)

    spec = bp.types[0]
    scale = bp.units["micron_per_site"]
    area = spec.target_volume * scale * scale
    radius = radius_for_area(area)

    definition = root.find("cell_definitions/cell_definition")
    definition.set("name", spec.name)
    _set(definition, "phenotype/volume/total", volume_for_radius(radius))
    _set(definition, "phenotype/volume/nuclear", volume_for_radius(radius) * 0.2)

    # Both division and death are switched off rather than left at the
    # template's tissue-like rates. The comparison needs a fixed cell count.
    for duration in definition.findall("phenotype/cycle/phase_durations/duration"):
        duration.text = str(NO_DIVISION_MINUTES)
        duration.set("fixed_duration", "true")
    for model in definition.findall("phenotype/death/model"):
        model.find("death_rate").text = "0.0"

    physicell = bp.physicell or {}
    _set(
        definition,
        "phenotype/mechanics/cell_cell_adhesion_strength",
        physicell.get("adhesion", 0.4),
    )
    _set(
        definition,
        "phenotype/mechanics/cell_cell_repulsion_strength",
        physicell.get("repulsion", 10.0),
    )
    motility = definition.find("phenotype/motility")
    if motility is not None:
        motility.find("speed").text = str(physicell.get("motility_speed", 0.0))

    cells_csv = out_dir / "cells.csv"
    _write_cells(bp, cells_csv, spec.name)
    positions = root.find("initial_conditions/cell_positions")
    positions.set("enabled", "true")
    positions.find("folder").text = str(out_dir)
    positions.find("filename").text = "cells.csv"

    ruleset = root.find("cell_rules/rulesets/ruleset")
    if ruleset is not None:
        ruleset.set("enabled", "false")

    # The template's own setup places `number_of_cells` agents on top of
    # whatever the csv states, which is five cells the description never asked
    # for.
    extra = root.find("user_parameters/number_of_cells")
    if extra is not None:
        extra.text = "0"

    config = out_dir / "settings.xml"
    tree.write(config)
    return config


def _write_cells(bp: Blueprint, path: Path, type_name: str) -> None:
    """One agent per square of the Potts initial condition, at its centre."""
    scale = bp.units["micron_per_site"]
    width_um, height_um = bp.domain_microns
    side = bp.initial["side"]
    rows = []
    for iy in range(bp.initial["ny"]):
        for ix in range(bp.initial["nx"]):
            x = (ix + 0.5) * side * scale - width_um / 2
            y = (iy + 0.5) * side * scale - height_um / 2
            rows.append(f"{x},{y},0.0,{type_name}")
    path.write_text("x,y,z,type\n" + "\n".join(rows) + "\n")


def rasterise(bp: Blueprint, centres: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Paint agents onto the Blueprint's lattice, nearest centre within radius.

    A site outside every agent stays medium, which is what a lattice engine
    would call it, so the same area and orientation measurements apply to both.
    """
    from scipy.spatial import cKDTree

    scale = bp.units["micron_per_site"]
    width_um, height_um = bp.domain_microns
    xs = (np.arange(bp.width) + 0.5) * scale - width_um / 2
    ys = (np.arange(bp.height) + 0.5) * scale - height_um / 2
    grid = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)

    tree = cKDTree(centres)
    distance, index = tree.query(grid, k=1)
    inside = distance <= radii[index]
    labels = np.where(inside, index + 1, 0).astype(np.uint32)
    return labels.reshape(bp.height, bp.width)


def run(bp: Blueprint, out_dir: str | Path) -> dict:
    """Run the description on PhysiCell and write it out the way glazier does."""
    import pcdl

    out = Path(out_dir).resolve()
    config = write_config(bp, out)

    started = time.perf_counter()
    finished = subprocess.run(
        [str(PHYSICELL / "project"), str(config)],
        capture_output=True,
        cwd=PHYSICELL,
        text=True,
    )
    seconds = time.perf_counter() - started
    if finished.returncode != 0:
        tail = (finished.stdout or "") + (finished.stderr or "")
        raise RuntimeError(f"PhysiCell exited {finished.returncode}:\n{tail[-2000:]}")

    saves = sorted(out.glob("output*.xml"))
    if not saves:
        raise RuntimeError(f"PhysiCell wrote no save into {out}")
    mcds = pcdl.TimeStep(str(saves[-1]), verbose=False)
    cells = mcds.get_cell_df()

    centres = cells[["position_x", "position_y"]].to_numpy()
    volumes = cells["total_volume"].to_numpy()
    radii = (3.0 * volumes / (4.0 * math.pi)) ** (1.0 / 3.0)

    labels = rasterise(bp, centres, radii)
    np.save(out / f"labels_{bp.steps:05d}.npy", labels)

    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    summary = {
        "name": bp.name,
        "engine": "physicell",
        "steps": bp.steps,
        "seconds": seconds,
        "n_cells": int(len(cells)),
        "n_cells_rasterised": int(ids.size),
        "mean_volume": float(counts.mean()) if ids.size else 0.0,
        "mean_radius_micron": float(radii.mean()),
        "width": bp.width,
        "height": bp.height,
        "micron_per_site": bp.units["micron_per_site"],
        "minute_per_step": bp.units["minute_per_step"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
