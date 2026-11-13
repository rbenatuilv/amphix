"""Integrals of a solution: totals, per named boundary, per cell, and the normal jump of a flux.

    integrate(form, comm)                          a compiled scalar form, summed over the ranks
    BoundaryIntegrals(expr, domain, measures)()    {boundary: int_b expr ds} in SI: expr an SI
                                                   expression, times L0^(dim-1) for the area
    PerCell(domain, {name: lambda v: form})()      {name: array}: each form tested with the DG0
                                                   indicator of every owned cell (local order)
    NormalJump(q, measures)()                      ||[[q.n]]|| / ||{q}.n|| over interior facets

The objects compile their forms once and are called every step. A boundary flow is
BoundaryIntegrals(ufl.inner(q, n), ...) with q an SI velocity (m^2/s per unit depth in 2D, m^3/s in
3D). A normal jump of 0 means a conservative (H(div)) flux. PerCell gives cell-by-cell fluid
balances: one entry per owned cell, in dolfinx's local order.
"""

from collections.abc import Callable
from typing import Any

import dolfinx
import numpy as np
import ufl
from mpi4py import MPI

from amphix.core.mesh import Domain, Measures


def integrate(form: Any, comm: MPI.Comm) -> float:
    """Return the value of a compiled scalar form, summed over the ranks."""
    return float(comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM))


class BoundaryIntegrals:
    """{boundary: int_boundary expr ds} in SI, for every named boundary (or the ones given)."""

    def __init__(
        self, expr: Any, domain: Domain, measures: Measures, names: list[str] | None = None
    ) -> None:
        """Compile one form per boundary; expr is an SI expression on the domain's mesh."""
        self.comm = domain.mesh.comm
        self.area = domain.L0 ** (domain.dim - 1)  # ds* (solver units) -> m^(dim - 1)
        names = list(domain.facet_names) if names is None else names
        self.forms = {n: dolfinx.fem.form(expr * measures.ds(n)) for n in names}

    def __call__(self) -> dict[str, float]:
        """Return the integrals now."""
        return {n: self.area * integrate(f, self.comm) for n, f in self.forms.items()}


class PerCell:
    """Per-cell integrals: each term, a function of the DG0 test function v, assembled."""

    def __init__(self, domain: Domain, terms: dict[str, Callable[[Any], Any]]) -> None:
        """Compile the terms; also the cell volumes (solver units) as `volume`."""
        mesh = domain.mesh
        v = ufl.TestFunction(dolfinx.fem.functionspace(mesh, ("DG", 0)))
        self.nloc = mesh.topology.index_map(mesh.topology.dim).size_local
        self.forms = {k: dolfinx.fem.form(term(v)) for k, term in terms.items()}
        self.volume = self._assemble(dolfinx.fem.form(v * ufl.dx(domain=mesh)))

    def _assemble(self, form: Any) -> np.ndarray:
        vec = dolfinx.fem.assemble_vector(form)
        vec.scatter_reverse(dolfinx.la.InsertMode.add)  # contributions of ghost cells' facets
        return vec.array[: self.nloc].copy()

    def __call__(self) -> dict[str, np.ndarray]:
        """Return each term on every owned cell (local order)."""
        return {k: self._assemble(f) for k, f in self.forms.items()}


class NormalJump:
    """||[[q.n]]|| / ||{q}.n|| over the interior facets: 0 for a mass-conserving (H(div)) flux."""

    def __init__(self, q: Any, measures: Measures) -> None:
        """Compile the two integrals of a vector field q."""
        n, dS = measures.n, measures.dS
        self.num = dolfinx.fem.form(ufl.jump(q, n) ** 2 * dS)
        self.den = dolfinx.fem.form(ufl.inner(ufl.avg(q), n("+")) ** 2 * dS)
        self.comm = measures.domain.mesh.comm

    def __call__(self) -> float:
        """Return the relative normal jump now (0 if {q}.n vanishes everywhere)."""
        num, den = integrate(self.num, self.comm), integrate(self.den, self.comm)
        return float(np.sqrt(num / den)) if den > 0 else 0.0
