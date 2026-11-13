"""The Recorder: what a problem's `output` section asks for, recorded during a run (all SI).

    rec = Recorder(model)                # reads model.problem.output (probes, lines, grid, ...)
    model.run([rec])                     # an observer: called at t0 and after every step
    path = rec.save()                    # -> <root>/<output.dir>/<name>.npz (+ <name>_fields.npz)

Recorded every step: the `series` quantities at the probes, and the flow through every named
boundary of the raw flux (bflux_<boundary>) and, when cflux is asked for, of the conservative one
(cbflux_<boundary>). At the `snapshots`: grids (2D), slices (3D) and lines of the series quantities
together with flux and p, the normal jump of the raw flux (`jump`), and the full fields
(output.fields, to <name>_fields.npz). ParaView files every output.paraview steps.

cflux is post.hdiv's reconstruction (the model must implement core.balance.Balance): it is built
when cflux is in `series` or `fields`, and computed once per step. Points outside the mesh give
NaN.
"""

from pathlib import Path
from typing import Any

import dolfinx
import numpy as np
import ufl

from amphix.io.fields import FieldWriter
from amphix.io.paraview import ParaView
from amphix.io.results import save_results
from amphix.post.hdiv import Reconstruction
from amphix.post.integrals import BoundaryIntegrals, NormalJump
from amphix.post.sampling import Points, grid_points, line_points, slice_points

DEFAULT_FIELDS = ["d", "p", "flux"]  # output.fields: true


class Recorder:
    """Records a run as the problem's `output` section asks (see the module docstring)."""

    def __init__(self, model: Any, output: Any = None) -> None:
        """Set up every requested output; call before the first step."""
        self.model, self.o = model, output or model.problem.output
        domain, dim = model.domain, model.domain.dim
        if (self.o.grid and dim != 2) or (self.o.slices and dim != 3):
            raise ValueError("output: `grid` is for 2D meshes, `slices` for 3D meshes")
        names = DEFAULT_FIELDS if self.o.fields is True else list(self.o.fields or [])
        self.rec = Reconstruction(model) if "cflux" in [*self.o.series, *names] else None
        self._cflux_at: int | None = None
        self.probes = Points(domain, list(self.o.probes.values())) if self.o.probes else None
        self.lines = {
            k: (line_points(v), Points(domain, line_points(v))) for k, v in self.o.lines.items()
        }
        self.grid = None
        if self.o.grid:
            pts = grid_points(domain, int(self.o.grid))
            self.grid = (pts, Points(domain, pts))
        self.slices = {
            k: (pts, Points(domain, pts))
            for k, pts in ((k, slice_points(domain, v)) for k, v in self.o.slices.items())
        }
        self.snap_times = self.o.times(model.problem.time)
        self.snap_quantities = sorted(set(self.o.series) | ({"flux", "p"} & set(self._names())))
        self.jump = NormalJump(self.quantity("flux"), model.measures) if self._has("flux") else None
        self.flows = self._flows() if self.o.boundary_flux else {}
        self.fields = FieldWriter(domain, {n: self._source(n) for n in names}) if names else None
        self.paraview = None
        if self.o.paraview:
            quantities = sorted(set(self.snap_quantities) | {"d"} & set(self._names()))
            self.paraview = ParaView(
                domain, {n: self._source(n) for n in quantities}, self.path().with_suffix("")
            )
        self.data: dict[str, list] = {}

    # -- quantities ---------------------------------------------------------------------------
    def _names(self) -> list[str]:
        return [*self.model.state.current, *self.model.quantities, *(["cflux"] if self.rec else [])]

    def _has(self, name: str) -> bool:
        return name in self._names()

    def quantity(self, name: str) -> dolfinx.fem.Function:
        """Return the current SI Function of a quantity (cflux computed at most once per step)."""
        if name != "cflux":
            return self.model.quantity(name)
        if self.rec is None:
            raise KeyError("cflux: ask for it in output.series or output.fields")
        if self._cflux_at != self.model.n:
            self.rec.compute()
            self._cflux_at = self.model.n
        return self.rec.u_si

    def _source(self, name: str) -> Any:
        return lambda: self.quantity(name)

    def _flows(self) -> dict[str, BoundaryIntegrals]:
        m, out = self.model.measures, {}
        for key, name in (("bflux", "flux"), ("cbflux", "cflux")):
            if self._has(name):
                out[key] = BoundaryIntegrals(
                    ufl.inner(self.quantity(name), m.n), self.model.domain, m
                )
        return out

    # -- recording ----------------------------------------------------------------------------
    def _append(self, key: str, value: Any) -> None:
        self.data.setdefault(key, []).append(value)

    def __call__(self, model: Any = None) -> None:
        """Record the current state (at t0 and after every step)."""
        m = model or self.model
        t, dt = m.t, m.problem.time.dt
        self._append("t", t)
        if self.probes is not None:
            for name in self.o.series:
                self._append(f"probe_{name}", self.probes(self.quantity(name)))
        for key, flows in self.flows.items():
            self.quantity("cflux" if key == "cbflux" else "flux")  # refresh the SI copy
            for boundary, value in flows().items():
                self._append(f"{key}_{boundary}", value)
        if any(abs(t - ts) < dt / 2 for ts in self.snap_times):
            self._snapshot(t)
        if self.paraview is not None and m.n % self.o.paraview == 0:
            self.paraview.write(t)

    def _snapshot(self, t: float) -> None:
        self._append("snap_t", t)
        for name in self.snap_quantities:
            f = self.quantity(name)
            if self.grid is not None:
                self._append(f"grid_{name}", self.grid[1](f))
            for sn, (_, pts) in self.slices.items():
                self._append(f"slice_{sn}_{name}", pts(f))
            for ln, (_, pts) in self.lines.items():
                self._append(f"line_{ln}_{name}", pts(f))
        if self.jump is not None:
            self.quantity("flux")
            self._append("jump", self.jump())
        if self.fields is not None:
            self.fields(t)

    def path(self) -> Path:
        """Return <root>/<output.dir>/<name>.npz."""
        p = self.model.problem
        return Path(p.root) / self.o.dir / f"{p.name}.npz"

    def save(self, path: str | Path | None = None) -> Path | None:
        """Write every record (positions and the problem tree too); return the path (rank 0)."""
        if self.paraview is not None:
            self.paraview.close()
            self.paraview = None
        path = Path(path) if path else self.path()
        if self.fields is not None:
            self.fields.save(path.with_name(f"{path.stem}_fields.npz"), self.model.problem)
        mesh = self.model.domain.mesh
        extra: dict[str, Any] = {"ncells": mesh.topology.index_map(mesh.topology.dim).size_global}
        if self.o.probes:
            extra["probe_names"] = np.array(list(self.o.probes))
            extra["probes"] = np.array(list(self.o.probes.values()), dtype=float)
        extra |= {f"line_{k}": pts for k, (pts, _) in self.lines.items()}
        extra |= {f"slice_{k}": pts for k, (pts, _) in self.slices.items()}
        if self.grid is not None:
            extra["grid"] = self.grid[0]
        arrays = extra | {k: np.array(v) for k, v in self.data.items()}
        return save_results(path, arrays, self.model.problem)
