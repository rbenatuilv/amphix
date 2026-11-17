"""A field over its saved snapshots, as a GIF (or an MP4 with ffmpeg).

    animate(fl, "cflux", "figures/run_cflux.gif", t_from=1.0, t_to=2.0, fps=10, log=True)

The plot is built once (FieldPlot) and only its values change from frame to frame. The frames are
exactly the saved snapshots with t_from <= t <= t_to, nothing interpolated between them, each titled
with its time (cflux: the step ending at t). The colour scale is the same on every frame: without
vmin/vmax, a first pass over the frames finds the smallest and largest value shown (log: of the
positive ones, at most fields.DECADES decades apart).
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import animation

from amphix.plot.fields import DECADES, FieldPlot
from amphix.plot.labels import PlotError, between


def writer(path: Path, fps: int) -> Any:
    """Return the writer of a .gif (Pillow) or an .mp4 (ffmpeg)."""
    if path.suffix == ".gif":
        return animation.PillowWriter(fps=fps)
    if path.suffix == ".mp4":
        if not animation.writers.is_available("ffmpeg"):
            raise PlotError(f"{path}: an .mp4 needs ffmpeg, which is not installed (write a .gif)")
        return animation.FFMpegWriter(fps=fps)
    raise PlotError(f"{path}: an animation is a .gif or an .mp4")


def limits(fp: FieldPlot, frames: list[int]) -> tuple[float, float] | None:
    """Return the colour limits that hold every frame (None if no frame has a value to show)."""
    each = [lim for i in frames if (lim := fp.limits(fp.values(i))) is not None]
    if not each:
        return None
    lo, hi = min(lim[0] for lim in each), max(lim[1] for lim in each)
    return (max(lo, hi * 10.0**-DECADES), hi) if fp.log else (lo, hi)


def animate(
    fields: Any,
    name: str,
    path: str | Path,
    t_from: float | None = None,
    t_to: float | None = None,
    fps: int = 10,
    **field_kwargs: Any,
) -> Path:
    """Write an animation of a field over its snapshots in [t_from, t_to]; return its path.

    field_kwargs: the options of FieldPlot (component, log, vmin, vmax, zoom, cells, mesh,
    quiver).
    """
    path = Path(path)
    if fps <= 0:
        raise PlotError(f"fps must be positive, not {fps}")
    frames = between(fields, t_from, t_to)
    out = writer(path, fps)
    fig, ax = plt.subplots(figsize=(6.4, 5.2), layout="constrained")
    try:
        fp = FieldPlot(fields, name, ax, **field_kwargs)
        lo, hi = fp.fixed
        if lo is None or hi is None:
            auto = limits(fp, frames)
            if auto is not None:
                fp.set_limits(auto[0] if lo is None else lo, auto[1] if hi is None else hi)
        path.parent.mkdir(parents=True, exist_ok=True)
        with out.saving(fig, path, dpi=100):
            for i in frames:
                fp.update(i)
                out.grab_frame()
    finally:
        plt.close(fig)
    return path
