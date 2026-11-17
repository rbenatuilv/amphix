"""Time series of a results file against t.

    series(load_results("results/run.npz"), ["probe_p", "cbflux_pial"])
    series("results/run.npz", "probe_cflux", component="x")

probe_<q> draws one line per probe, labelled with its name; a vector probe draws its magnitude
unless a component is given. bflux_<b> and cbflux_<b> are flows through a boundary (m²/s in 2D,
m³/s in 3D); cbflux, like cflux, is the mean over the step that ends at t. jump is drawn against
jump_t.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from amphix.io.results import load_results
from amphix.plot.labels import (
    STEP_MEANS,
    PlotError,
    component_index,
    component_name,
    flow_unit,
    label,
    reduce,
    text,
    unit,
)


def key_unit(results: dict[str, Any], key: str) -> str | None:
    """Return the unit of a result key (None if unknown)."""
    if key.startswith(("bflux_", "cbflux_")):
        return flow_unit(results.get("problem"))
    if key.startswith("probe_"):
        return unit(key.removeprefix("probe_"))
    return None


def time_series(results: dict[str, Any]) -> list[str]:
    """Return the keys that are time series (one entry per t)."""
    n = len(results["t"])
    skip = {"t", "probe_names", "probes"}
    return sorted(
        k
        for k, v in results.items()
        if k not in skip and isinstance(v, np.ndarray) and v.ndim and len(v) == n
    )


def _which(results: dict[str, Any], key: str, component: str | None) -> int | str | None:
    """Return what to plot of a probe key's values (see labels.component_index)."""
    shape = np.shape(results[key])[2:]  # (times, probes, value shape)
    return component_index(key.removeprefix("probe_"), shape, component, shape[0] if shape else 3)


def _lines(results: dict[str, Any], key: str, component: str | None) -> list[tuple[str, Any]]:
    """Return (label, values) of each line of a key."""
    data = np.asarray(results[key], dtype=float)
    if key.startswith("probe_"):
        which = _which(results, key, component)
        names = [str(n) for n in results.get("probe_names", range(data.shape[1]))]
        name = text(key.removeprefix("probe_"), component_name(which))
        values = [reduce(data[:, j].reshape(len(data), -1), which) for j in range(data.shape[1])]
        return [(f"{name} at {n}", v) for n, v in zip(names, values, strict=True)]
    if data.ndim != 1:
        raise PlotError(f"{key}: not a scalar time series (shape {data.shape})")
    if key.startswith(STEP_MEANS):
        return [(f"{key}, mean over the step ending at t", data)]
    return [(key, data)]


def series(
    results: dict[str, Any] | str | Path,
    keys: str | Sequence[str],
    ax: Any = None,
    *,
    component: str | None = None,
) -> list[Any]:
    """Plot result arrays against t (a results dict or file); return the lines."""
    if not isinstance(results, dict):
        results = load_results(results)
    keys = [keys] if isinstance(keys, str) else list(keys)
    ax = ax or plt.gca()
    lines, units = [], []
    for key in keys:
        t = results.get("jump_t") if key == "jump" else results.get("t")
        if key not in results or t is None or np.ndim(results[key]) == 0:
            raise PlotError(
                f"{key}: not a time series here (series: {', '.join(time_series(results))})"
            )
        if len(results[key]) != len(t):
            raise PlotError(
                f"{key}: not a time series ({len(results[key])} values, {len(t)} times)"
            )
        for name, values in _lines(results, key, component):
            lines += ax.plot(t, values, label=name)
        units.append(key_unit(results, key))
    ax.set_xlabel("t [s]")
    if len(keys) == 1 and keys[0].startswith("probe_"):
        which = component_name(_which(results, keys[0], component))
        ax.set_ylabel(label(keys[0].removeprefix("probe_"), which))
    elif len(set(units)) == 1 and units[0]:
        ax.set_ylabel(f"[{units[0]}]")
    ax.legend()
    return lines
