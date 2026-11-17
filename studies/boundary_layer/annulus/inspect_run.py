"""Inspect a run: the mesh, and p and |q| at the pulse peak, whole annulus and zoomed at a wall.

    python inspect_run.py smoke_4e-5        reads results/smoke_4e-5_fields.npz

The run must have saved its fields: add to `amphix run`
    --set 'output.fields={quantities: [p, cflux], at: {every: 10}}'

Values are taken at the cell centres (one colour per cell).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from amphix.io.fields import load_fields

ZOOM = (5.2e-3, 6.2e-3, -0.5e-3, 0.5e-3)  # x0, x1, y0, y1 [m]: the ventricle wall, thin side

if __name__ == "__main__":
    HERE = Path(__file__).parent
    name = sys.argv[1] if len(sys.argv) > 1 else "smoke_4e-5"
    (HERE / "figures").mkdir(exist_ok=True)

    fl = load_fields(HERE / "results" / f"{name}_fields.npz")
    mesh = fl.domain.mesh  # in m
    X = mesh.geometry.x[:, :2]
    tri = np.asarray(mesh.geometry.dofmap)  # (cells, 3)
    centres = np.zeros((len(tri), 3))
    centres[:, :2] = X[tri].mean(axis=1)
    cells = np.arange(len(tri), dtype=np.int32)

    i = int(np.argmax(np.sin(2 * np.pi * fl.snap_t)))  # snapshot closest to the pulse peak
    t = fl.snap_t[i]
    p = fl.function("p", i).eval(centres, cells)[:, 0]
    q = np.linalg.norm(fl.function("cflux", i).eval(centres, cells), axis=1)
    print(f"{name}: {len(tri)} cells, snapshot t = {t:.3f} s")
    print(f"p: {p.min():.3f} to {p.max():.3f} Pa    |q|: {q.min():.2e} to {q.max():.2e} m/s")

    def show(ax, values=None, **kw):
        """Draw the mesh, or the cell values, on ax (mm)."""
        x, y = X[:, 0] * 1e3, X[:, 1] * 1e3
        if values is None:
            ax.triplot(x, y, tri, lw=0.2, color="k")
            return None
        return ax.tripcolor(x, y, tri, facecolors=values, **kw)

    def zoom(ax):
        x0, x1, y0, y1 = (v * 1e3 for v in ZOOM)
        ax.set(xlim=(x0, x1), ylim=(y0, y1))

    panels = [
        ("mesh", None, {}),
        ("p [Pa]", p, {"cmap": "viridis"}),
        ("|q| [m/s]", q, {"cmap": "magma", "norm": LogNorm(max(q.max() * 1e-6, 1e-30), q.max())}),
    ]
    for title, values, kw in panels:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.5))
        for ax in (a1, a2):
            im = show(ax, values, **kw)
            ax.set_aspect("equal")
            ax.set(xlabel="x [mm]", ylabel="y [mm]")
        zoom(a2)
        a1.set_title(f"{title}, whole annulus" + ("" if values is None else f", t = {t:.2f} s"))
        a2.set_title(f"{title}, zoom at the ventricle wall (thin side)")
        if im is not None:
            fig.colorbar(im, ax=[a1, a2], shrink=0.8)
        stem = title.split()[0].strip("|")
        fig.savefig(HERE / "figures" / f"inspect_{name}_{stem or 'q'}.png", dpi=150)
        plt.close(fig)
    print("saved figures/inspect_*.png")
