"""The Recorder: what a problem's `output` section asks for, recorded during a run (all SI).

    rec = Recorder(model)                # reads model.problem.output (config.sections.Output)
    model.run([rec])                     # an observer: called at t0 and after every step
    path = rec.save()                    # -> <root>/<output.dir>/<name>.npz (+ <name>_fields.npz)

Nothing is recorded but `t` (every step) unless a block asks for it, and each block records exactly
its quantities at exactly its steps:

    boundary_flux   every step: the flow through every named boundary of the raw flux
                    (bflux_<boundary>) and/or of the conservative one (cbflux_<boundary>)
    probes          every step: probe_<q>, with probe_names and probes (the positions)
    lines           at lines_t: line_<path>_<q>, with line_<path> (the points)
    grid            at grid_t: grid_<q>, with grid (2D)
    slices          at slices_t: slice_<plane>_<q>, with slice_<plane> (3D)
    jump            at jump_t: jump, the normal jump of the raw flux (post.integrals.NormalJump)
    fields          at its own times: the full fields, to <name>_fields.npz (held in memory until
                    save(); fields_bytes() estimates how much)
    paraview        at its times: ParaView files of its quantities

cflux is post.hdiv's reconstruction (the model must implement core.balance.Balance): it is built
when a block asks for cflux or boundary_flux is conservative, and computed once per step. Points
outside the mesh give NaN.
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


class Recorder:
    """Records a run as the problem's `output` section asks (see the module docstring)."""

    def __init__(self, model: Any, output: Any = None) -> None:
        """Set up every requested output; call before the first step."""
        self.model, self.o = model, output or model.problem.output
        o, domain = self.o, model.domain
        wanted = {q for b in o.blocks().values() for q in b.quantities}
        cflux = "cflux" in wanted or "conservative" in o.boundary_flux
        self.rec = Reconstruction(model) if cflux else None
        self._cflux_at: int | None = None
        self.probes = Points(domain, list(o.probes.spec["points"].values())) if o.probes else None
        self.sampled: list[tuple[str, Any, dict[str, tuple[np.ndarray, Points]]]] = []
        for key in ("lines", "grid", "slices"):
            block = getattr(o, key)
            if block is not None:
                pts = {k: (x, Points(domain, x)) for k, x in self._points(key, block.spec).items()}
                self.sampled.append((key, block, pts))
        self.jump = NormalJump(self.quantity("flux"), model.measures) if o.jump else None
        self.flows = self._flows()
        self.fields = None
        if o.fields:
            self.fields = FieldWriter(domain, {n: self._source(n) for n in o.fields.quantities})
        self.paraview = None
        if o.paraview:
            sources = {n: self._source(n) for n in o.paraview.quantities}
            self.paraview = ParaView(domain, sources, self.path().with_suffix(""))
        self.data: dict[str, list] = {}

    def _points(self, key: str, spec: dict) -> dict[str, np.ndarray]:
        """Return {result key: points} of a lines, grid or slices block."""
        domain = self.model.domain
        if key == "lines":
            return {f"line_{k}": line_points(v) for k, v in spec["paths"].items()}
        if key == "grid":
            return {"grid": grid_points(domain, int(spec["n"]))}
        return {f"slice_{k}": slice_points(domain, v) for k, v in spec["planes"].items()}

    def fields_bytes(self) -> int:
        """Return an estimate of the memory the `fields` block holds until save() [bytes]."""
        return self.fields.nbytes(len(self.o.fields.steps)) if self.fields else 0

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
            raise KeyError("cflux: ask for it in an output block or boundary_flux: conservative")
        if self._cflux_at != self.model.n:
            self.rec.compute()
            self._cflux_at = self.model.n
        return self.rec.u_si

    def _source(self, name: str) -> Any:
        return lambda: self.quantity(name)

    def _flows(self) -> dict[str, BoundaryIntegrals]:
        m, out = self.model.measures, {}
        for key, kind, name in (("bflux", "raw", "flux"), ("cbflux", "conservative", "cflux")):
            if kind in self.o.boundary_flux:
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
        t, n = m.t, m.n
        self._append("t", t)
        if self.probes is not None:
            for name in self.o.probes.quantities:
                self._append(f"probe_{name}", self.probes(self.quantity(name)))
        for key, flows in self.flows.items():
            self.quantity("cflux" if key == "cbflux" else "flux")  # refresh the SI copy
            for boundary, value in flows().items():
                self._append(f"{key}_{boundary}", value)
        self._timed(t, n)

    def _timed(self, t: float, n: int) -> None:
        """Record the blocks whose steps include step n."""
        o = self.o
        for key, block, points in self.sampled:
            if n in block.steps:
                self._append(f"{key}_t", t)
                for name in block.quantities:
                    f = self.quantity(name)
                    for prefix, (_, pts) in points.items():
                        self._append(f"{prefix}_{name}", pts(f))
        if self.jump is not None and n in o.jump.steps:
            self.quantity("flux")
            self._append("jump_t", t)
            self._append("jump", self.jump())
        if self.fields is not None and n in o.fields.steps:
            self.fields(t)
        if self.paraview is not None and n in o.paraview.steps:
            self.paraview.write(t)

    def path(self) -> Path:
        """Return <root>/<output.dir>/<name>.npz."""
        p = self.model.problem
        return Path(p.root) / self.o.dir / f"{p.name}.npz"

    def save(self, path: str | Path | None = None) -> Path:
        """Write every record (positions, problem tree); return the path on every rank, written."""
        if self.paraview is not None:
            self.paraview.close()
            self.paraview = None
        path = Path(path) if path else self.path()
        if self.fields is not None:
            self.fields.save(path.with_name(f"{path.stem}_fields.npz"), self.model.problem)
        mesh = self.model.domain.mesh
        extra: dict[str, Any] = {"ncells": mesh.topology.index_map(mesh.topology.dim).size_global}
        if self.o.probes:
            points = self.o.probes.spec["points"]
            extra["probe_names"] = np.array(list(points))
            extra["probes"] = np.array(list(points.values()), dtype=float)
        for _, _, points in self.sampled:
            extra |= {prefix: x for prefix, (x, _) in points.items()}
        arrays = extra | {k: np.array(v) for k, v in self.data.items()}
        return save_results(path, arrays, self.model.problem)
