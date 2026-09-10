"""The compiled engine, driven in process rather than through a file.

The binary and the module run the same code, so what matters here is that the
bindings hand back a lattice of the stated shape with the stated cell count.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "blueprints" / "monolayer.json"


@pytest.fixture(scope="module")
def native():
    try:
        from glazier import _native
    except ImportError:
        pytest.skip("glazier._native is not built in this environment")
    return _native


def test_a_short_run_returns_a_lattice_of_the_stated_shape(native):
    text = MODEL.read_text()
    labels, width, height, depth = native.run(text, 5)
    field = np.asarray(labels, dtype=np.uint32).reshape(depth, height, width)

    parsed = native.parse(text)
    assert field.shape == (parsed["depth"], parsed["height"], parsed["width"])

    side, nx, ny, nz = parsed["initial"]
    assert np.unique(field[field > 0]).size == nx * ny * nz


def test_volumes_come_back_per_cell(native):
    text = MODEL.read_text()
    volumes = native.run_volumes(text, 5)
    parsed = native.parse(text)
    side, nx, ny, nz = parsed["initial"]

    assert len(volumes) == nx * ny * nz + 1
    # A confluent sheet conserves its total volume whatever the sweep does.
    sites = parsed["width"] * parsed["height"] * parsed["depth"]
    assert sum(volumes[1:]) == sites
