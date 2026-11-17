"""Units, axis labels and titles, the plotted component of a value, and the choice of snapshots.

Lengths are drawn in mm, every other quantity in SI. cflux saved at t is the mean velocity over the
step that ends at t, and so is cbflux_<boundary>: their titles and labels say so.
"""

import warnings
from typing import Any

import numpy as np

from amphix.core.mesh import dimension

UNITS = {"d": "m", "w": "m", "p": "Pa", "flux": "m/s", "velocity": "m/s", "cflux": "m/s"}
STEP_MEANS = ("cflux", "cbflux_")  # saved at t: the mean over the step ending at t
AXES = "xyz"


class PlotError(ValueError):
    """A plot that cannot be made from what was asked or saved (a one-line message)."""


def unit(name: str) -> str | None:
    """Return the SI unit of a quantity, or None if unknown."""
    return UNITS.get(name)


def flow_unit(problem: dict | None) -> str:
    """Return the unit of a flow through a boundary, from the dimension of the problem's mesh."""
    try:
        dim = dimension(problem["mesh"])
    except (KeyError, TypeError, ValueError):
        return "m^d/s"
    return "m²/s" if dim == 2 else "m³/s"


def text(name: str, component: str | None = None) -> str:
    """Return how a quantity is written: "p", "|cflux|" (component "mag"), "d_x"."""
    if component == "mag":
        return f"|{name}|"
    return name if component is None else f"{name}_{component}"


def label(name: str, component: str | None = None) -> str:
    """Return the axis label of a quantity: "p [Pa]", "|cflux| [m/s]", "d_x [m]"."""
    u = unit(name)
    return f"{text(name, component)} [{u}]" if u else text(name, component)


def component_name(which: int | str | None) -> str | None:
    """Return the name of what component_index chose: None, "mag", "x", "y" or "z"."""
    return AXES[which] if isinstance(which, int) else which


def when(name: str, t: float | None = None, average: tuple[float, float] | None = None) -> str:
    """Return the time part of a title: "t = 0.5 s", "step ending at t = 0.5 s", "mean over ..."."""
    if average is not None:
        return f"mean over ({average[0]:.4g}, {average[1]:.4g}] s"
    if t is None:
        return ""
    step = name.startswith(STEP_MEANS)
    return f"{'step ending at ' if step else ''}t = {t:.4g} s"


def component_index(
    name: str, shape: tuple[int, ...], component: str | None, gdim: int
) -> int | str | None:
    """Return what to plot of a value: None (a scalar), "mag", or the index of a component."""
    if len(shape) == 0:
        if component not in (None, "mag"):
            raise PlotError(f"{name} is a scalar: it has no component {component}")
        return None
    if component in (None, "mag"):
        return "mag"
    if component not in AXES:
        raise PlotError(f"{name}: unknown component {component} (mag, x, y or z)")
    if len(shape) > 1:
        raise PlotError(f"{name} is a tensor: only its magnitude (mag) is plotted")
    i = AXES.index(component)
    if i >= gdim:
        raise PlotError(f"{name}: no component {component} on a {gdim}D mesh")
    return i


def reduce(values: np.ndarray, which: int | str | None) -> np.ndarray:
    """Return the plotted scalar of point values (npoints, value size): itself, |.| or one entry."""
    values = np.asarray(values, dtype=float).reshape(len(values), -1)
    if which is None:
        return values[:, 0]
    if which == "mag":
        return np.linalg.norm(values, axis=1)
    return values[:, int(which)]


def nearest(fields: Any, t: float | None = None) -> int:
    """Return the index of the snapshot nearest t (default: the last); warn if it is not at t."""
    snap_t = np.asarray(fields.snap_t)
    if len(snap_t) == 0:
        raise PlotError(f"{fields.path}: no snapshot saved")
    if t is None:
        return len(snap_t) - 1
    i = int(np.argmin(np.abs(snap_t - t)))
    if abs(snap_t[i] - t) > 1e-9 * max(1.0, abs(t)):
        warnings.warn(
            f"no snapshot at t = {t:.6g} s: using the nearest one, t = {snap_t[i]:.6g} s",
            stacklevel=2,
        )
    return i


def between(fields: Any, t_from: float | None = None, t_to: float | None = None) -> list[int]:
    """Return the indices of the snapshots with t_from <= t <= t_to (default: all of them)."""
    snap_t = np.asarray(fields.snap_t)
    lo = -np.inf if t_from is None else t_from - 1e-9 * max(1.0, abs(t_from))
    hi = np.inf if t_to is None else t_to + 1e-9 * max(1.0, abs(t_to))
    idx = np.flatnonzero((snap_t >= lo) & (snap_t <= hi)).tolist()
    if not idx:
        raise PlotError(f"{fields.path}: no snapshot in [{t_from}, {t_to}]")
    return idx
