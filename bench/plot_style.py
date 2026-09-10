"""House figure style: Latin Modern through real LaTeX, no grid, black type.

`~/.config/matplotlib/matplotlibrc` already turns `text.usetex` on for this
machine, and this module never turns it off: without it matplotlib falls back
to its own serif, which is the one face these figures may not use.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Reds through to teal, the palette the film scripts on this machine use.
SERIES = ["#d81e05", "#1f4e9c", "#1a9a3a", "#7a3fa0", "#c98a00", "#00868b"]

# A field reads as a wash: white to one hue, so the cells keep the ink and the
# concentration stays legible under them.
VIRUS = LinearSegmentedColormap.from_list("virus", ["#ffffff", "#d81e05"])
SIGNAL = LinearSegmentedColormap.from_list("signal", ["#ffffff", "#1f4e9c"])
GROWTH = LinearSegmentedColormap.from_list("growth", ["#ffffff", "#1a9a3a"])


def apply() -> None:
    """Set the rcParams every figure in this repository shares."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.grid": False,
            "text.color": "#000000",
            "axes.labelcolor": "#000000",
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "axes.edgecolor": "#000000",
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def save(fig, path) -> None:
    """Write a figure as both a page and a picture."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
