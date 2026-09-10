"""Measuring a volume: the analysis reads a plane, so a volume is sliced.

Every measurement in `glazier.director` is written for a micrograph, which is a
plane. A three-dimensional label field is therefore cut through the middle
rather than projected: a projection would stack cells that never touched.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from glazier import mid_plane
from glazier.blueprint import Blueprint
from glazier.director import cell_shapes, coarse_grain, defect_charges, director_angle, q_field

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "blueprints" / "volume.json"


@pytest.fixture(scope="module")
def field():
    try:
        from glazier import _native
    except ImportError:
        pytest.skip("glazier._native is not built in this environment")

    text = MODEL.read_text()
    labels, width, height, depth = _native.run(text, 5)
    return np.asarray(labels, dtype=np.uint32).reshape(depth, height, width)


def test_a_volume_comes_back_three_dimensional(field):
    bp = Blueprint.load(MODEL)
    assert bp.dimensions == 3
    assert field.shape == (bp.depth, bp.height, bp.width)


def test_a_slice_is_one_layer_of_it(field):
    plane = mid_plane(field)
    assert plane.shape == field.shape[1:]
    assert np.array_equal(plane, field[field.shape[0] // 2])


def test_a_plane_is_returned_unchanged():
    flat = np.arange(12, dtype=np.uint32).reshape(3, 4)
    assert np.array_equal(mid_plane(flat), flat)


def test_the_measurements_run_on_a_sliced_volume(field):
    plane = mid_plane(field)
    shapes = cell_shapes(plane)
    assert shapes, "the slice held no cells"

    q = coarse_grain(q_field(plane, shapes), 4.0)
    defects = defect_charges(director_angle(q))
    charge = sum(c for _, _, c in defects)

    # Slicing a periodic volume leaves a plane periodic in both axes.
    # Its charges cancel exactly whatever texture came out.
    assert abs(charge) < 1e-9, f"net charge {charge} over {len(defects)} defects"
