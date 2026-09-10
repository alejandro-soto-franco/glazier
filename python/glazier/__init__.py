"""The Python side of glazier: model descriptions, engine backends, measurement.

The engine itself is the Rust workspace beside this package. This side reads
the same Blueprint description, drives CompuCell3D and PhysiCell through their
own interfaces, and measures whatever any of them produces on one lattice, so
the engines can be compared on the quantities a run is read for.
"""

from __future__ import annotations

__all__ = [
    "Blueprint",
    "TypeSpec",
    "cell_shapes",
    "coarse_grain",
    "defect_charges",
    "director_angle",
    "global_order",
    "mid_plane",
    "q_field",
    "render",
    "upsample",
    "write_tiff",
]


def __getattr__(name: str):
    """Import on demand, so reading a description needs no simulation stack."""
    if name in ("Blueprint", "TypeSpec"):
        from glazier import blueprint

        return getattr(blueprint, name)
    if name in (
        "cell_shapes",
        "coarse_grain",
        "defect_charges",
        "director_angle",
        "global_order",
        "mid_plane",
        "q_field",
    ):
        from glazier import director

        return getattr(director, name)
    if name in ("render", "upsample", "write_tiff"):
        from glazier import render as render_module

        return getattr(render_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
