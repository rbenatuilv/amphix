"""Inspect a run: the mesh, and p and |q| at the pulse peak, whole annulus and zoomed at a wall.

    python inspect_run.py smoke_4e-5        reads results/smoke_4e-5_fields.npz

The run must have saved its fields: add to `amphix run`
    --set 'output.fields={quantities: [p, cflux], at: {every: 10}}'

The fields are drawn on their DG lattice (amphix.plot); the zoom keeps the colour scale of the whole
annulus.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from amphix import plot
from amphix.io.fields import load_fields

ZOOM = (5.2e-3, 6.2e-3, -0.5e-3, 0.5e-3)  # x0, x1, y0, y1 [m]: the ventricle wall, thin side

if __name__ == "__main__":
    HERE = Path(__file__).parent
    name = sys.argv[1] if len(sys.argv) > 1 else "smoke_4e-5"
    (HERE / "figures").mkdir(exist_ok=True)

    fl = load_fields(HERE / "results" / f"{name}_fields.npz")
    i = int(np.argmax(np.sin(2 * np.pi * fl.snap_t)))  # snapshot closest to the pulse peak
    t = fl.snap_t[i]
    ncells = fl.domain.mesh.topology.index_map(2).size_local
    print(f"{name}: {ncells} cells, snapshot t = {t:.3f} s")

    panels = [("mesh", None, {}), ("p", "p", {}), ("q", "cflux", {"log": True})]
    for stem, quantity, options in panels:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
        if quantity is None:
            plot.mesh(fl, a1)
            plot.mesh(fl, a2, zoom=ZOOM)
            title = "mesh"
        else:
            whole = plot.field(fl, quantity, a1, t=t, **options)
            lo, hi = whole.image.norm.vmin, whole.image.norm.vmax
            plot.field(fl, quantity, a2, t=t, zoom=ZOOM, vmin=lo, vmax=hi, **options)
            v = whole.values(i)
            title = f"{plot.label(quantity, 'mag' if quantity == 'cflux' else None)}"
            print(f"{title}: {v.min():.3g} to {v.max():.3g}")
            title += f", {plot.when(quantity, t=t)}"
        fig.suptitle(title)
        a1.set_title("whole annulus")
        a2.set_title("zoom at the ventricle wall (thin side)")
        fig.savefig(HERE / "figures" / f"inspect_{name}_{stem}.png", dpi=150)
        plt.close(fig)
    print("saved figures/inspect_*.png")
