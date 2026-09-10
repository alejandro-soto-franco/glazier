"""Figures for what the engines actually do.

Each one is generated from a run made here, so the numbers on the page are the
numbers the repository produces rather than any recollection of them.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from glazier.blueprint import Blueprint  # noqa: E402
from plot_style import GROWTH, SERIES, SIGNAL, VIRUS, apply, save  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GLAZIER = ROOT / "target" / "release" / "glazier"
BLUEPRINTS = ROOT / "blueprints"
RUNS = ROOT / "runs" / "figures"
FIGURES = ROOT / "figures"


def run(name: str, engine: str = "cpu", extra: list[str] | None = None) -> Path:
    """Run a description and return the directory it wrote."""
    out = RUNS / f"{name}-{engine}"
    subprocess.run(
        [str(GLAZIER), "--model", str(BLUEPRINTS / f"{name}.json"), "--out", str(out),
         "--engine", engine, *(extra or [])],
        check=True,
        capture_output=True,
    )
    return out


def labels(directory: Path, step: int | None = None) -> np.ndarray:
    """The label field a run wrote, sliced to a plane if it is a volume."""
    files = sorted(directory.glob("labels_*.npy"))
    path = files[-1] if step is None else directory / f"labels_{step:05d}.npy"
    field = np.load(path)
    return field[field.shape[0] // 2] if field.ndim == 3 else field


def field(directory: Path, name: str, step: int) -> np.ndarray:
    path = directory / f"field_{name}_{step:05d}.npy"
    values = np.load(path)
    return values[values.shape[0] // 2] if values.ndim == 3 else values


def mosaic(ax, field: np.ndarray, seed: int = 0) -> None:
    """Draw a label field as cells with the medium left white.

    Neighbouring labels are close integers, so the colours are shuffled: a
    smooth ramp over label number would draw a gradient across the sheet that
    no cell is aware of.
    """
    rng = np.random.default_rng(seed)
    top = int(field.max()) + 1
    order = rng.permutation(top - 1) + 1
    colours = plt.get_cmap("twilight")(np.linspace(0.08, 0.92, top - 1))
    table = np.zeros((top, 4))
    table[0] = [1.0, 1.0, 1.0, 1.0]
    table[order] = colours
    ax.imshow(field, cmap=ListedColormap(table), vmin=0, vmax=top - 1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def tissue_figure() -> None:
    """One panel per description, at the step it finished on."""
    panels = [
        ("monolayer", "Contact"),
        ("monolayer-surface", "Surface"),
        ("monolayer-connected", "Connectivity"),
        ("adhesion", "Adhesion"),
        ("immune-motile", "Chemotaxis"),
        ("infection", "Infection"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 7.4))
    for index, (ax, (name, title)) in enumerate(zip(axes.ravel(), panels)):
        mosaic(ax, labels(run(name)), seed=index)
        ax.set_title(title, pad=8)
    fig.suptitle("Tissue", fontsize=18, y=0.98)
    save(fig, FIGURES / "tissue")


def infection_figure() -> None:
    """The virus field over the sheet, and what the run did to the population."""
    name = "infection"
    model = json.loads((BLUEPRINTS / f"{name}.json").read_text())
    model["dump_every"] = 40
    path = RUNS / "infection-figure.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model))

    directory = RUNS / "infection-figure"
    subprocess.run(
        [str(GLAZIER), "--model", str(path), "--out", str(directory), "--engine", "cpu",
         "--trace"],
        check=True, capture_output=True)
    steps = [40, 120, 200]

    fig = plt.figure(figsize=(11.4, 7.4))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.5, 1.0], hspace=0.24, wspace=0.16)

    scale = max(field(directory, "virus", step).max() for step in steps[:-1])
    for column, step in enumerate(steps):
        ax = fig.add_subplot(grid[0, column])
        virus = field(directory, "virus", min(step, 160))
        ax.imshow(virus, cmap=VIRUS, interpolation="bilinear", vmin=0.0, vmax=scale)
        edges = labels(directory, step if step != model["steps"] else None)
        ax.contour(edges, levels=np.arange(0.5, edges.max() + 1), colors="#3a3a3a",
                   linewidths=0.22)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(rf"Step {step}", pad=6)

    trace = np.genfromtxt(directory / "trace.csv", delimiter=",", names=True)
    ax = fig.add_subplot(grid[1, 0:2])
    ax.plot(trace["step"], trace["cells"], color=SERIES[1], lw=1.6)
    ax.set_xlabel("Monte Carlo step")
    ax.set_ylabel("Cells alive")

    ax = fig.add_subplot(grid[1, 2])
    ax.plot(trace["step"], trace["virus"], color=SERIES[0], lw=1.6, label="Virus")
    ax.plot(trace["step"], trace["interferon"], color=SERIES[1], lw=1.6, label="Interferon")
    ax.set_xlabel("Monte Carlo step")
    ax.set_ylabel("Total on the lattice")
    ax.legend(loc="upper left")

    fig.suptitle("Infection", fontsize=18, y=0.97)
    save(fig, FIGURES / "infection")


def engines_figure(comparison: Path) -> None:
    """What four engines make of one description."""
    rows = json.loads(comparison.read_text())
    quantities = [
        ("cells", "Cells"),
        ("mean_area", "Mean area"),
        ("mean_eccentricity", "Mean eccentricity"),
        ("defects", "Disclinations"),
        ("net_charge", "Net charge"),
        ("seconds", "Seconds"),
    ]
    engines = list(rows)

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.2))
    for ax, (key, title) in zip(axes.ravel(), quantities):
        values = [rows[engine][key] for engine in engines]
        ax.bar(range(len(engines)), values,
               color=[SERIES[i % len(SERIES)] for i in range(len(engines))],
               edgecolor="#000000", linewidth=0.7, width=0.62)
        ax.set_xticks(range(len(engines)))
        ax.set_xticklabels([e.replace("glazier-", "") for e in engines], rotation=20, ha="right")
        ax.set_title(title, pad=6)
        if key == "seconds":
            ax.set_yscale("log")
    fig.suptitle("One description, four engines", fontsize=18, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, FIGURES / "engines")


SPEED_SIZES = [64, 128, 256, 512, 1024, 2048, 4096]


def speed_figure(steps: int = 100) -> None:
    """Wall clock against lattice size, both engines.

    The numbers come from the `bench` example rather than from the binary,
    because a device pays for loading its kernel on the first launch of a
    process and that cost belongs to a run rather than to a sweep. The example
    steps once before it starts the clock; the binary reports what a whole run
    took, which is the honest number there and the wrong one here.
    """
    # The sweep runs for minutes at the top end, so its table is kept and
    # reused unless the sizes change.
    cache = RUNS / f"bench-{steps}-{'-'.join(str(s) for s in SPEED_SIZES)}.txt"
    if cache.exists():
        table = cache.read_text()
    else:
        done = subprocess.run(
            ["cargo", "run", "--release", "--features", "cuda", "--example", "bench", "--",
             str(steps), *[str(size) for size in SPEED_SIZES]],
            cwd=ROOT, check=True, capture_output=True, text=True)
        table = done.stdout
        cache.write_text(table)

    sizes, cpu, gpu = [], [], []
    for line in table.splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[0].isdigit():
            sizes.append(int(parts[0]))
            cpu.append(float(parts[2]))
            gpu.append(float(parts[3]))

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.6, 4.4))
    left.plot(sizes, cpu, color=SERIES[1], marker="o", lw=1.6, ms=6,
              label="Serial reference")
    left.plot(sizes, gpu, color=SERIES[0], marker="s", lw=1.6, ms=6, label="Device")
    for seconds, offset in ((cpu[-1], (-9, -4)), (gpu[-1], (-9, 6))):
        left.annotate(rf"${seconds:.2f}$ s", (sizes[-1], seconds),
                      textcoords="offset points", xytext=offset, ha="right", fontsize=10)
    left.set_xscale("log", base=2)
    left.set_yscale("log")
    left.set_xlabel("Lattice side, sites")
    left.set_ylabel(rf"Seconds for {steps} steps")
    left.legend(loc="upper left")

    speedup = [c / g for c, g in zip(cpu, gpu)]
    right.plot(sizes, speedup, color=SERIES[2], marker="D", lw=1.6, ms=6)
    right.set_xscale("log", base=2)
    right.set_xlabel("Lattice side, sites")
    right.set_ylabel("Device against the reference")
    for size, value in zip(sizes, speedup):
        right.annotate(rf"${value:.0f}\times$", (size, value), textcoords="offset points",
                       xytext=(0, 9), ha="center", fontsize=11)
    right.set_ylim(0, max(speedup) * 1.22)

    fig.suptitle("Wall clock", fontsize=18, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, FIGURES / "speed")


def motility_figure() -> None:
    """Where a cell goes when it remembers where it has been."""
    from glazier import _native

    base = json.loads((BLUEPRINTS / "monolayer.json").read_text())

    def track(max_activity: float, lambda_activity: float, steps: list[int]):
        model = dict(base)
        model["name"] = "one cell"
        model["width"] = model["height"] = 96
        model["temperature"] = 10.0
        model["types"] = [
            {"name": "Cell", "target_volume": 64.0, "lambda_volume": 2.0,
             "max_activity": max_activity, "lambda_activity": lambda_activity}
        ]
        model["initial"] = {"side": 8, "nx": 1, "ny": 1}
        model["dump_every"] = 0
        path = []
        for step in steps:
            model["steps"] = step
            labels, width, height, depth = _native.run(json.dumps(model), step)
            field = np.asarray(labels, dtype=np.uint32).reshape(depth, height, width)[0]
            ys, xs = np.nonzero(field)
            path.append((xs.mean(), ys.mean()))
        return np.array(path)

    steps = list(range(0, 801, 40))
    still = track(0.0, 0.0, steps)
    motile = track(20.0, 60.0, steps)

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.6, 4.6))
    for track_points, colour, label in ((still, SERIES[1], "No memory"),
                                        (motile, SERIES[0], "Memory")):
        left.plot(track_points[:, 0], track_points[:, 1], color=colour, lw=1.6, label=label)
        left.scatter(*track_points[0], color=colour, s=28, zorder=3)
        left.scatter(*track_points[-1], color=colour, s=44, marker="s", zorder=3,
                     edgecolor="#000000", linewidths=0.6)
    left.set_aspect("equal")
    left.set_xlabel(r"$x$, sites")
    left.set_ylabel(r"$y$, sites")
    left.legend(loc="best")

    for track_points, colour, label in ((still, SERIES[1], "No memory"),
                                        (motile, SERIES[0], "Memory")):
        displacement = np.linalg.norm(track_points - track_points[0], axis=1)
        right.plot(steps, displacement, color=colour, lw=1.6, label=label)
    right.set_xlabel("Monte Carlo step")
    right.set_ylabel("Distance from the start, sites")
    right.legend(loc="upper left")

    fig.suptitle("Persistence", fontsize=18, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, FIGURES / "motility")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path,
                        default=ROOT / "runs" / "final" / "comparison.json")
    args = parser.parse_args()

    apply()
    RUNS.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    tissue_figure()
    infection_figure()
    if args.comparison.exists():
        engines_figure(args.comparison)
    speed_figure()
    motility_figure()
    print(f"figures written to {FIGURES} in {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
