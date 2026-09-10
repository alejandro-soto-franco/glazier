"""Director, nematic order and disclination charge from a cellular Potts label field.

Every quantity here is computed from the model's own pixels, so it is ground
truth for the simulated tissue: no segmentation, no intensity model, no
threshold. It is the reference against which an image-analysis measurement of
the same tissue is checked.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CellShape:
    label: int
    area: int
    com: tuple[float, float]
    theta: float
    eccentricity: float


def mid_plane(labels: np.ndarray) -> np.ndarray:
    """The middle layer of a label field, or the field itself if it is flat.

    Every measurement below reads a plane, since that is what a micrograph is.
    A volume is therefore sliced rather than projected: a projection would
    stack cells that never touched.
    """
    if labels.ndim == 3:
        return labels[labels.shape[0] // 2]
    return labels


def cell_shapes(labels: np.ndarray) -> list[CellShape]:
    """Second-moment ellipse per label. `theta` is the major-axis angle in
    [0, pi), the nematic convention.
    """
    shapes = []
    for label in np.unique(labels):
        if label == 0:
            continue
        ys, xs = np.nonzero(labels == label)
        area = xs.size
        if area < 4:
            continue
        x0, y0 = xs.mean(), ys.mean()
        dx, dy = xs - x0, ys - y0
        cxx = (dx * dx).mean()
        cyy = (dy * dy).mean()
        cxy = (dx * dy).mean()
        theta = 0.5 * np.arctan2(2.0 * cxy, cxx - cyy) % np.pi
        trace = cxx + cyy
        gap = np.hypot(cxx - cyy, 2.0 * cxy)
        ecc = gap / trace if trace > 0 else 0.0
        shapes.append(CellShape(int(label), int(area), (float(x0), float(y0)), float(theta), float(ecc)))
    return shapes


def q_field(labels: np.ndarray, shapes: list[CellShape]) -> np.ndarray:
    """Per-pixel Q tensor, painted from the owning cell's ellipse.

    Q = S (n n^T - I/2) in two dimensions, stored as the two independent
    components (Qxx, Qxy) with S taken as the cell's eccentricity.
    """
    lookup = {s.label: s for s in shapes}
    q = np.zeros(labels.shape + (2,), dtype=np.float64)
    for label, shape in lookup.items():
        mask = labels == label
        q[mask, 0] = 0.5 * shape.eccentricity * np.cos(2.0 * shape.theta)
        q[mask, 1] = 0.5 * shape.eccentricity * np.sin(2.0 * shape.theta)
    return q


def coarse_grain(q: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    out = np.empty_like(q)
    for c in range(q.shape[-1]):
        out[..., c] = gaussian_filter(q[..., c], sigma, mode="wrap")
    return out


def director_angle(q: np.ndarray) -> np.ndarray:
    return 0.5 * np.arctan2(q[..., 1], q[..., 0])


def scalar_order(q: np.ndarray) -> np.ndarray:
    return 2.0 * np.hypot(q[..., 0], q[..., 1])


def global_order(shapes: list[CellShape]) -> tuple[float, float]:
    """Population nematic order |<e^{2i theta}>| and its director angle."""
    if not shapes:
        return 0.0, 0.0
    z = np.mean([np.exp(2j * s.theta) for s in shapes])
    return float(abs(z)), float(0.5 * np.angle(z) % np.pi)


def defect_charges(theta: np.ndarray) -> list[tuple[int, int, float]]:
    """Half-integer disclinations by plaquette winding on the director angle.

    The angle is defined mod pi, so each of the four bond differences is
    wrapped into (-pi/2, pi/2] before summing. The winding is a multiple of
    pi, and the charge is that multiple over two.
    """
    def wrap(d):
        return (d + np.pi / 2.0) % np.pi - np.pi / 2.0

    a = theta
    b = np.roll(theta, -1, axis=1)
    c = np.roll(np.roll(theta, -1, axis=1), -1, axis=0)
    d = np.roll(theta, -1, axis=0)
    winding = wrap(b - a) + wrap(c - b) + wrap(d - c) + wrap(a - d)
    charge = winding / (2.0 * np.pi)

    out = []
    ys, xs = np.nonzero(np.abs(charge) > 0.2)
    for y, x in zip(ys, xs):
        out.append((int(x), int(y), float(charge[y, x])))
    return out
