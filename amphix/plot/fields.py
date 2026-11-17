"""Fields of a fields file drawn on matplotlib axes (2D, lengths in mm).

    fl = load_fields("results/run_fields.npz")
    field(fl, "p", t=0.5)                                   # the snapshot nearest t = 0.5 s
    field(fl, "cflux", average=(0.0, 1.0), log=True, quiver=20)
    field(fl, "d", component="x", zoom=(5e-3, 6e-3, -5e-4, 5e-4), mesh=True)

    fp = FieldPlot(fl, "p", ax, vmin=0, vmax=140)           # build once ...
    for i in range(len(fl.snap_t)):
        fp.update(i)                                        # ... and redraw only the values

A field is drawn on its DG lattice (lattice.py), or one colour per cell with cells=True. A vector
is drawn by its magnitude (component="mag", the default) or one component ("x", "y"), with
optional direction arrows (quiver=N: about N arrows over the shown region, of equal length; the
colour gives the magnitude). log=True draws a logarithmic colour scale and leaves the non-positive
values out. zoom=(x0, x1, y0, y1) [m] shows only that region. Without vmin/vmax the colour scale
spans the values in the shown region (log: at most DECADES decades below the largest).
"""

from typing import Any

import dolfinx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.tri import Triangulation

from amphix.plot.labels import (
    PlotError,
    component_index,
    component_name,
    label,
    nearest,
    reduce,
    when,
)
from amphix.plot.lattice import MM, lattice
from amphix.post.sampling import Points

DECADES = 6  # the automatic log scale: at most this many decades below the largest value
Zoom = tuple[float, float, float, float]


def mesh(fields: Any, ax: Any = None, zoom: Zoom | None = None, **kwargs: Any) -> Any:
    """Draw the cell edges of a fields file's mesh (mm); return the lines."""
    ax = ax or plt.gca()
    lat = lattice(fields.domain.mesh, 0, cells=True)
    lines = ax.triplot(
        lat.mesh_x, lat.mesh_y, lat.mesh_triangles, **({"lw": 0.2, "color": "k"} | kwargs)
    )
    _frame(ax, zoom)
    return lines


def _frame(ax: Any, zoom: Zoom | None) -> None:
    ax.set_aspect("equal")
    ax.set(xlabel="x [mm]", ylabel="y [mm]")
    if zoom is not None:
        x0, x1, y0, y1 = (v * MM for v in zoom)
        ax.set(xlim=(x0, x1), ylim=(y0, y1))


class FieldPlot:
    """One field of a fields file on an axes: built once, redrawn by update().

    Artists: image (the field), colorbar, edges (mesh=True), arrows (quiver=N).
    """

    def __init__(  # noqa: PLR0913 (the options are keyword-only)
        self,
        fields: Any,
        name: str,
        ax: Any = None,
        *,
        component: str | None = None,
        log: bool = False,
        vmin: float | None = None,
        vmax: float | None = None,
        zoom: Zoom | None = None,
        cells: bool = False,
        mesh: bool = False,
        quiver: int | None = None,
    ) -> None:
        """Build the triangulation, the axes, the colour bar (and edges, arrows) of a field."""
        if name not in fields.names:
            raise PlotError(f"{name}: not in {fields.path} (saved: {', '.join(fields.names)})")
        V = fields.space(name)
        shape = tuple(V.value_shape)
        self.fields, self.name, self.log = fields, name, log
        self.which = component_index(name, shape, component, fields.domain.mesh.geometry.dim)
        self.lattice = lattice(fields.domain.mesh, V.ufl_element().degree, cells)
        self.ax = ax = ax or plt.gca()
        lat = self.lattice
        self.triangulation = Triangulation(lat.x, lat.y, lat.triangles)
        self.fixed = (vmin, vmax)
        norm = LogNorm(vmin, vmax) if log else Normalize(vmin, vmax)
        cmap = "magma" if log else "viridis"
        if lat.per_cell:
            self.image = ax.tripcolor(
                self.triangulation, facecolors=np.ones(lat.ncells), norm=norm, cmap=cmap
            )
        else:
            self.image = ax.tripcolor(
                self.triangulation, np.ones(len(lat.x)), shading="gouraud", norm=norm, cmap=cmap
            )
        self.colorbar = ax.figure.colorbar(
            self.image, ax=ax, label=label(name, component_name(self.which))
        )
        self.edges = None
        if mesh:
            self.edges = ax.triplot(lat.mesh_x, lat.mesh_y, lat.mesh_triangles, lw=0.2, color="k")
        _frame(ax, zoom)
        self.shown = _inside(lat.points, zoom)
        self.arrows: Any = None
        if quiver:
            if len(shape) != 1:
                raise PlotError(f"{name} is not a vector: no arrows")
            self._arrows(quiver, zoom)

    def _arrows(self, n: int, zoom: Zoom | None) -> None:
        """Place about n arrows on a grid over the shown region, inside the mesh."""
        x = self.fields.domain.mesh.geometry.x
        x0, x1, y0, y1 = zoom or (x[:, 0].min(), x[:, 0].max(), x[:, 1].min(), x[:, 1].max())
        w, h = x1 - x0, y1 - y0
        nx = max(1, round(np.sqrt(n * w / h)))
        ny = max(1, round(n / nx))
        gx, gy = np.meshgrid(
            x0 + (np.arange(nx) + 0.5) * w / nx, y0 + (np.arange(ny) + 0.5) * h / ny
        )
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        self._probe = Points(self.fields.domain, pts)
        self._in_mesh = inside = self._probe.count > 0
        step = min(w / nx, h / ny) * MM
        zeros = np.zeros(int(inside.sum()))
        self.arrows = self.ax.quiver(
            pts[inside, 0] * MM,
            pts[inside, 1] * MM,
            zeros,
            zeros,
            angles="xy",
            scale_units="xy",
            scale=1 / (0.7 * step),  # a unit vector is 0.7 grid steps long
            pivot="mid",
            color="w",
            edgecolor="k",
            linewidth=0.5,
        )

    def function(self, what: int | dolfinx.fem.Function) -> dolfinx.fem.Function:
        """Return the Function to draw: snapshot `what`, or `what` itself."""
        if isinstance(what, int | np.integer):
            return self.fields.function(self.name, int(what))
        return what

    def values(self, what: int | dolfinx.fem.Function) -> np.ndarray:
        """Return the plotted scalar at the lattice points (component or magnitude)."""
        return reduce(self.lattice.values(self.function(what)), self.which)

    def limits(self, values: np.ndarray) -> tuple[float, float] | None:
        """Return the automatic colour limits of these values over the shown region (or None)."""
        v = values[self.shown] if self.shown.any() else values
        v = v[np.isfinite(v) & (v > 0)] if self.log else v[np.isfinite(v)]
        if not len(v):
            return None
        lo, hi = float(v.min()), float(v.max())
        return (max(lo, hi * 10.0**-DECADES), hi) if self.log else (lo, hi)

    def update(self, what: int | dolfinx.fem.Function, title: str | None = None) -> None:
        """Redraw with snapshot `what` (an index) or a Function of this field; set the title."""
        f = self.function(what)
        v = self.values(f)
        if self.log:
            bad = ~(v > 0)
            v = np.ma.masked_array(v, bad)
            if not self.lattice.per_cell:  # leave out the sub-triangles that touch a masked value
                self.triangulation.set_mask(bad[self.lattice.triangles].any(axis=1))
        self.image.set_array(v)
        auto = self.limits(np.ma.filled(v, np.nan))
        lo, hi = self.fixed
        if auto is not None and (lo is None or hi is None):
            self.image.set_clim(auto[0] if lo is None else lo, auto[1] if hi is None else hi)
        if self.arrows is not None:
            u = self._probe(f)[self._in_mesh, :2]
            size = np.linalg.norm(u, axis=1, keepdims=True)
            u = np.divide(u, size, out=np.zeros_like(u), where=size > 0)  # directions only
            self.arrows.set_UVC(u[:, 0], u[:, 1])
        if title is None and isinstance(what, int | np.integer):
            title = f"{self.name}, {when(self.name, t=float(self.fields.snap_t[int(what)]))}"
        if title is not None:
            self.ax.set_title(title)


def _inside(points: np.ndarray, zoom: Zoom | None) -> np.ndarray:
    """Return which points are in the zoomed region (all without a zoom)."""
    if zoom is None:
        return np.ones(len(points), dtype=bool)
    x0, x1, y0, y1 = zoom
    x, y = points[:, 0], points[:, 1]
    return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)


def field(
    fields: Any,
    name: str,
    ax: Any = None,
    *,
    t: float | None = None,
    average: tuple[float, float] | None = None,
    slice: Any = None,
    **options: Any,
) -> FieldPlot:
    """Draw a field at the snapshot nearest t (default: the last), or its mean over (t0, t1].

    options: those of FieldPlot; slice is reserved for 3D fields. Return the FieldPlot (its
    artists: image, colorbar, edges, arrows).
    """
    if slice is not None:
        raise NotImplementedError("slices of 3D fields are not implemented yet")
    if t is not None and average is not None:
        raise PlotError("give a time or an average, not both")
    fp = FieldPlot(fields, name, ax, **options)
    if average is None:
        fp.update(nearest(fields, t))
        return fp
    t0, t1 = average
    try:
        mean = fields.average(name, t0, t1)
    except ValueError as e:
        raise PlotError(f"{name}: {e}") from None
    fp.update(mean, f"{name}, {when(name, average=(t0, t1))}")
    return fp
