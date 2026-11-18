"""Time-averaged fields over the last two periods: case 1, case 2 and their difference.

Reads results/case1_fields.npz and results/case2_fields.npz (see transmantle.yaml), which hold
p and cflux at every step of (3, 5] s; the mean of cflux over those steps is exactly its time
average over the window.

Figure: figures/averages.png. Rows: p and |cflux| (with direction arrows). Columns: case 1,
case 2 (same colour scale), case 2 - case 1. Set ZOOM to look at a region, e.g. the thin side
of the ventricle wall: ZOOM = (5.4e-3, 5.8e-3, -2e-4, 2e-4).
"""

from pathlib import Path

import matplotlib.pyplot as plt
from mpi4py import MPI

import amphix
from amphix import plot

T0, T1 = 3.0, 5.0  # the last two periods [s]
ZOOM = None  # (x0, x1, y0, y1) [m]
# ZOOM = (5.4e-3, 6.0e-3, -3e-4, 3e-4)  # thin side, ventricle
# ZOOM = (9.4e-3, 1.0e-2, -3e-4, 3e-4)  # thin side, pia
ROWS = {"p": {}, "cflux": {"log": True, "quiver": 20}}


if __name__ == "__main__":
    HERE = Path(__file__).parent
    (HERE / "figures").mkdir(exist_ok=True)
    fl = {
        k: amphix.load_fields(HERE / "results" / f"{k}_fields.npz", MPI.COMM_SELF)
        for k in ("case1", "case2")
    }

    fig, axes = plt.subplots(len(ROWS), 3, figsize=(16, 4.6 * len(ROWS)), layout="constrained")
    for row, (name, options) in zip(axes, ROWS.items(), strict=True):
        mean = {k: fl[k].average(name, T0, T1) for k in fl}
        fps = []
        for ax, k in zip(row[:2], fl, strict=True):
            fp = plot.FieldPlot(fl[k], name, ax, zoom=ZOOM, **options)
            fp.update(mean[k], f"{name}, {k}, mean over {T0:g}-{T1:g} s")
            fps.append(fp)
        lo = min(fp.image.get_clim()[0] for fp in fps)  # one colour scale for both cases
        hi = max(fp.image.get_clim()[1] for fp in fps)
        for fp in fps:
            fp.set_limits(lo, hi)

        diff = mean["case2"].copy()
        diff.x.array[:] -= mean["case1"].x.array
        fp = plot.FieldPlot(fl["case2"], name, row[2], zoom=ZOOM, **options)
        fp.update(diff, f"{name}, case 2 - case 1")
    fig.savefig(HERE / "figures" / "averages.png", dpi=150)
