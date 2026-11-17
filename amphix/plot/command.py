"""What `amphix plot` does (cli.py): find a run's file, draw it, save a PNG.

A run is a path to an .npz file, or a name: results/<run>_fields.npz for fields and
results/<run>.npz for series, from the current directory. The figure goes to figures/ there:
<run>_<fields>_t<t>.png (the snapshot's time), <run>_<fields>_avg<t0>-<t1>.png or
<run>_<keys>_series.png, unless an output path is given.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

from amphix.io.fields import load_fields
from amphix.io.results import load_results
from amphix.plot.fields import field
from amphix.plot.labels import PlotError, nearest
from amphix.plot.series import key_unit, series

FIGURES = Path("figures")


def backend(show: bool) -> None:
    """Draw off screen (Agg) unless the figure is to be shown."""
    if not show:
        plt.switch_backend("Agg")


def find(run: str, kind: str) -> Path:
    """Return the file of a run: the path itself, or results/<run>_fields.npz / <run>.npz."""
    if run.endswith(".npz"):
        path = Path(run)
    else:
        path = Path("results") / (f"{run}_fields.npz" if kind == "fields" else f"{run}.npz")
    if not path.is_file():
        raise PlotError(f"{path}: no such file")
    with np.load(path, allow_pickle=False) as f:
        if ("snap_t" if kind == "fields" else "t") not in f.files:
            raise PlotError(f"{path}: not a {'fields' if kind == 'fields' else 'results'} file")
    return path


def run_name(path: Path) -> str:
    """Return the run's name: the file stem, without _fields."""
    return path.stem.removesuffix("_fields")


def save(fig: Any, out: Path, show: bool = False) -> Path:
    """Write the figure as a PNG (and show it); close it."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    if show:
        plt.show()
    return out


def _for(fields: Any, name: str, names: list[str], options: dict[str, Any]) -> dict[str, Any]:
    """Return the options of one panel: no component or arrows for a scalar among vectors.

    If every field drawn is a scalar, a component or arrows are an error (FieldPlot says so).
    """
    if name not in fields.names or fields.space(name).value_shape:
        return options
    if all(n in fields.names and not fields.space(n).value_shape for n in names):
        return options
    return options | {"component": None, "quiver": None}


def plot_fields(
    run: str,
    names: list[str],
    *,
    at: float | None = None,
    average: tuple[float, float] | None = None,
    out: str | Path | None = None,
    show: bool = False,
    **options: Any,
) -> Path:
    """Draw fields of a run side by side (one panel each) at one time or averaged; save a PNG."""
    path = find(run, "fields")
    fl = load_fields(path, MPI.COMM_SELF)
    t = None
    if average is None:
        t = float(fl.snap_t[nearest(fl, at)])  # warns once, not once per panel
        when = f"t{t:g}"
    else:
        when = f"avg{average[0]:g}-{average[1]:g}"
    fig, axes = plt.subplots(
        1, len(names), figsize=(5.5 * len(names), 4.6), squeeze=False, layout="constrained"
    )
    try:
        for ax, name in zip(axes[0], names, strict=True):
            field(fl, name, ax, t=t, average=average, **_for(fl, name, names, options))
        default = FIGURES / f"{run_name(path)}_{'_'.join(names)}_{when}.png"
        return save(fig, Path(out) if out else default, show)
    finally:
        plt.close(fig)


def plot_series(
    run: str,
    keys: list[str],
    *,
    component: str | None = None,
    out: str | Path | None = None,
    show: bool = False,
) -> Path:
    """Draw time series of a run, one panel per unit, against t; save a PNG."""
    path = find(run, "results")
    results = load_results(path)
    groups: dict[str | None, list[str]] = {}
    for key in keys:
        groups.setdefault(key_unit(results, key), []).append(key)
    fig, axes = plt.subplots(
        len(groups),
        1,
        figsize=(7, 3.4 * len(groups)),
        sharex=True,
        squeeze=False,
        layout="constrained",
    )
    try:
        for ax, group in zip(axes[:, 0], groups.values(), strict=True):
            series(results, group, ax, component=component)
        for ax in axes[:-1, 0]:
            ax.set_xlabel("")  # t is shared: labelled once, at the bottom
        default = FIGURES / f"{run_name(path)}_{'_'.join(keys)}_series.png"
        return save(fig, Path(out) if out else default, show)
    finally:
        plt.close(fig)
