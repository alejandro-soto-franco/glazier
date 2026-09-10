"""Match a measured cell table against the model's own cells and score the director.

Matching is nearest centroid with a distance ceiling. The angle residual is
taken mod pi, the nematic convention, and both sign conventions for the image
row axis are scored, since a measurement package and a lattice model need not
agree on which way y points.
"""

from __future__ import annotations

import numpy as np

from glazier.director import CellShape


def angle_residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest separation between two nematic angles, in [0, pi/2]."""
    d = (a - b) % np.pi
    return np.minimum(d, np.pi - d)


def match(
    truth: list[CellShape],
    measured_xy: np.ndarray,
    measured_theta: np.ndarray,
    *,
    max_distance_px: float = 6.0,
) -> dict:
    truth_xy = np.array([s.com for s in truth])
    truth_theta = np.array([s.theta for s in truth])

    pairs = []
    used = set()
    for i, xy in enumerate(measured_xy):
        d = np.hypot(*(truth_xy - xy).T)
        order = np.argsort(d)
        for j in order[:4]:
            if j in used:
                continue
            if d[j] <= max_distance_px:
                pairs.append((i, int(j), float(d[j])))
                used.add(int(j))
            break

    if not pairs:
        return {"matched": 0}

    mi = np.array([p[0] for p in pairs])
    tj = np.array([p[1] for p in pairs])

    scores = {}
    for name, sign in (("same", 1.0), ("flipped", -1.0)):
        res = angle_residual(sign * measured_theta[mi] % np.pi, truth_theta[tj])
        scores[name] = {
            "median_deg": float(np.degrees(np.median(res))),
            "mean_deg": float(np.degrees(res.mean())),
            "within_15deg": float((res < np.radians(15)).mean()),
        }

    best = min(scores, key=lambda k: scores[k]["median_deg"])
    return {
        "matched": len(pairs),
        "truth_cells": len(truth),
        "measured_cells": len(measured_xy),
        "recovery": len(pairs) / len(truth),
        "median_centroid_error_px": float(np.median([p[2] for p in pairs])),
        "convention": best,
        "scores": scores,
    }
