"""Render a simulated tissue as a two-channel fluorescence image.

The point is to hand an image-analysis tool the same tissue the model made, in
the form the tool expects, so its measurement is checked against a director
that is known exactly. Channel 0 is a nuclear stain, one elongated blob per
cell. Channel 1 is a fibre stain, a stripe texture inside each cell territory
aligned with that cell's major axis.
"""

from __future__ import annotations

import numpy as np

from glazier.director import CellShape


def upsample(labels: np.ndarray, factor: int) -> np.ndarray:
    """Nearest-neighbour upsample of a label field to microscopy resolution.

    A lattice cell a dozen sites across is a cell a hundred pixels across at
    0.345 um per pixel. Rendering at the lattice pitch instead puts several
    cells inside one nucleus-sized blob, and the measurement that follows is
    then a test of the renderer.
    """
    return np.repeat(np.repeat(labels, factor, axis=0), factor, axis=1)


def render(
    labels: np.ndarray,
    shapes: list[CellShape],
    *,
    nuclear_sigma: float = 2.0,
    nuclear_aspect: float = 1.6,
    fibre_period: float = 6.0,
    background: float = 0.02,
    noise: float = 0.01,
    seed: int = 0,
) -> np.ndarray:
    """Return a (2, H, W) float32 image in [0, 1]: nuclear, then fibre."""
    rng = np.random.default_rng(seed)
    height, width = labels.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)

    nuclear = np.zeros((height, width), dtype=np.float64)
    fibre = np.zeros((height, width), dtype=np.float64)

    for shape in shapes:
        x0, y0 = shape.com
        ct, st = np.cos(shape.theta), np.sin(shape.theta)

        # A bounding box around the cell keeps the work proportional to cell
        # area instead of to image area times cell count.
        ys, xs = np.nonzero(labels == shape.label)
        pad = int(np.ceil(4.0 * nuclear_sigma * nuclear_aspect))
        y_lo, y_hi = max(ys.min() - pad, 0), min(ys.max() + pad + 1, height)
        x_lo, x_hi = max(xs.min() - pad, 0), min(xs.max() + pad + 1, width)
        sub_y = yy[y_lo:y_hi, x_lo:x_hi]
        sub_x = xx[y_lo:y_hi, x_lo:x_hi]

        dx, dy = sub_x - x0, sub_y - y0
        along = dx * ct + dy * st
        across = -dx * st + dy * ct

        # Nuclear blob: an anisotropic Gaussian on the cell's own axes.
        sa = nuclear_sigma * nuclear_aspect
        nuclear[y_lo:y_hi, x_lo:x_hi] += np.exp(
            -0.5 * ((along / sa) ** 2 + (across / nuclear_sigma) ** 2)
        )

        # The fibre texture is a stripe pattern normal to the major axis,
        # painted inside the cell territory alone.
        mask = labels[y_lo:y_hi, x_lo:x_hi] == shape.label
        phase = 2.0 * np.pi * across / fibre_period
        stripes = 0.5 * (1.0 + np.cos(phase)) * (0.4 + 0.6 * shape.eccentricity)
        fibre[y_lo:y_hi, x_lo:x_hi][mask] += stripes[mask]

    image = np.stack([nuclear, fibre])
    image = np.clip(image, 0.0, None)
    image /= max(image.max(), 1e-12)
    image += background + noise * rng.standard_normal(image.shape)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def write_tiff(path, image: np.ndarray, pixel_size_um: float = 0.345) -> None:
    """Write a (C, H, W) stack as an ImageJ-style TIFF with a pixel size."""
    import tifffile

    tifffile.imwrite(
        path,
        (image * 65535).astype(np.uint16),
        imagej=True,
        resolution=(1.0 / pixel_size_um, 1.0 / pixel_size_um),
        metadata={"axes": "CYX", "unit": "um"},
    )
