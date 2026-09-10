"""Both readers of a Blueprint have to agree about what it says.

The description is parsed twice: by `glazier.blueprint` in Python and by the
Rust engine, reached either through the compiled `glazier._native` module or
through `glazier --check` on the binary. A field that drifts between the two
readers is the failure the whole exchange format exists to prevent, so it is
checked rather than assumed.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from glazier.blueprint import Blueprint

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "target" / "release" / "glazier"
BLUEPRINTS = sorted((ROOT / "blueprints").glob("*.json"))


def native_reading(path: Path) -> dict:
    try:
        from glazier import _native
    except ImportError:
        pytest.skip("glazier._native is not built in this environment")
    return _native.parse(path.read_text())


def binary_reading(path: Path) -> dict:
    if not BINARY.exists():
        pytest.skip(f"{BINARY} is not built")
    done = subprocess.run(
        [str(BINARY), "--model", str(path), "--check"],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(done.stdout)


def assert_agrees(bp: Blueprint, rust: dict) -> None:
    assert rust["name"] == bp.name
    assert rust["width"] == bp.width
    assert rust["height"] == bp.height
    if "depth" in rust:
        assert rust["depth"] == bp.depth
    assert rust["n_types"] == len(bp.type_names)
    # The engine reads the matrix with any adhesion molecules folded in.
    # That folded matrix is what the two readers have to agree on.
    assert list(rust["contact"]) == pytest.approx(bp.effective_contact())
    assert rust["temperature"] == bp.temperature
    assert rust["neighbour_order"] == bp.neighbour_order
    assert rust["seed"] == bp.seed
    assert rust["steps"] == bp.steps
    assert list(rust["initial"])[:3] == [
        bp.initial["side"],
        bp.initial["nx"],
        bp.initial["ny"],
    ]
    if len(rust["initial"]) > 3:
        assert rust["initial"][3] == bp.initial.get("nz", 1)
    assert rust["micron_per_site"] == bp.units["micron_per_site"]
    assert rust["minute_per_step"] == bp.units["minute_per_step"]

    # The medium takes no constraint, and the cell types follow it in the order
    # the description lists them.
    assert rust["target_volume"][0] == 0.0
    assert rust["lambda_volume"][0] == 0.0
    for index, spec in enumerate(bp.types, start=1):
        assert rust["target_volume"][index] == spec.target_volume
        assert rust["lambda_volume"][index] == spec.lambda_volume
        if "target_surface" in rust:
            assert rust["target_surface"][index] == spec.target_surface
            assert rust["lambda_surface"][index] == spec.lambda_surface
        if "division_volume" in rust:
            assert rust["division_volume"][index] == spec.division_volume
            assert rust["death_rate"][index] == spec.death_rate
        if "target_length" in rust:
            assert rust["target_length"][index] == spec.target_length
            assert rust["lambda_length"][index] == spec.lambda_length
        if "connected" in rust:
            assert bool(rust["connected"][index]) == spec.connected
        if "max_activity" in rust:
            assert rust["max_activity"][index] == spec.max_activity
            assert rust["lambda_activity"][index] == spec.lambda_activity
            assert list(rust["external"][index]) == list(spec.external)

    if "species" in rust:
        assert list(rust["species"]) == bp.species_names
        for index, species in enumerate(bp.fields):
            assert rust["diffusion"][index] == species.diffusion
            assert rust["decay"][index] == species.decay
        for type_index, spec in enumerate(bp.types, start=1):
            for species_index, name in enumerate(bp.species_names):
                assert rust["secretion"][type_index][species_index] == spec.secretion.get(
                    name, 0.0
                )
                assert rust["uptake"][type_index][species_index] == spec.uptake.get(
                    name, 0.0
                )
                assert rust["chemotaxis"][type_index][
                    species_index
                ] == spec.chemotaxis.get(name, 0.0)


@pytest.mark.parametrize("path", BLUEPRINTS, ids=lambda p: p.stem)
def test_the_compiled_reader_agrees_with_python(path: Path):
    assert_agrees(Blueprint.load(path), native_reading(path))


@pytest.mark.parametrize("path", BLUEPRINTS, ids=lambda p: p.stem)
def test_the_binary_agrees_with_python(path: Path):
    assert_agrees(Blueprint.load(path), binary_reading(path))


def test_every_blueprint_in_the_tree_is_covered():
    assert BLUEPRINTS, "no blueprints found to check"
