"""Measures of a Domain by name, the facet normal and cell sizes.

    m = Measures(domain, degree=2 * k + 1)     # quadrature degree of every integral (None: UFL's)
    m.dx()            m.dx("tissue")           whole domain, or named subdomains (one or a list)
    m.ds()            m.ds("pial")             whole boundary, or named boundaries (one or a list)
    m.dS              m.n, m.h, m.hF           interior facets; normal, cell diameter, facet size

h is the cell diameter; on an interior facet hF = 2 h+ h- / (h+ + h-), the harmonic mean of the two
cell diameters. All are in solver units, like the mesh. cell_size and mean_size are in SI.
"""

from collections.abc import Sequence
from typing import Any

import dolfinx
import numpy as np
import ufl
from mpi4py import MPI

from amphix.core.mesh.domain import Domain

Names = str | Sequence[str] | None


class Measures:
    """dx, ds, dS (with one quadrature degree), n, h, hF of a domain."""

    def __init__(self, domain: Domain, degree: int | None = None) -> None:
        """Build the measures; degree is the quadrature degree of every integral (2k + 1)."""
        self.domain, mesh = domain, domain.mesh
        md = {} if degree is None else {"quadrature_degree": degree}
        self._dx = ufl.Measure("dx", domain=mesh, subdomain_data=domain.cell_tags, metadata=md)
        self._ds = ufl.Measure("ds", domain=mesh, subdomain_data=domain.facet_tags, metadata=md)
        self.dS = ufl.Measure("dS", domain=mesh, metadata=md)
        self.n = ufl.FacetNormal(mesh)
        self.h = ufl.CellDiameter(mesh)
        self.hF = 2 * self.h("+") * self.h("-") / (self.h("+") + self.h("-"))

    def dx(self, names: Names = None) -> Any:
        """Return dx over the whole domain, or over the named subdomain(s)."""
        return self._dx if names is None else self._dx(_tags(names, self.domain.cell_tag))

    def ds(self, names: Names = None) -> Any:
        """Return ds over the whole boundary, or over the named boundary (boundaries)."""
        return self._ds if names is None else self._ds(_tags(names, self.domain.facet_tag))


def _tags(names: str | Sequence[str], tag: Any) -> int | tuple[int, ...]:
    """Return the tag of one name, or the tuple of tags of several (for ds(tags))."""
    if isinstance(names, str):
        return int(tag(names))
    return tuple(int(tag(n)) for n in names)


def cell_size(domain: Domain) -> float:
    """Return the largest cell diameter [m] over all ranks."""
    mesh = domain.mesh
    nloc = mesh.topology.index_map(mesh.topology.dim).size_local
    X = mesh.geometry.x[mesh.geometry.dofmaps[0][:nloc]]  # (cells, vertices, 3)
    h = max(
        (
            float(np.linalg.norm(X[:, i] - X[:, j], axis=1).max(initial=0.0))
            for i in range(X.shape[1])
            for j in range(i)
        ),
        default=0.0,
    )
    return float(mesh.comm.allreduce(h, op=MPI.MAX)) * domain.L0


def mean_size(domain: Domain) -> float:
    """Return (|Omega| / ncells)^(1/dim) [m], which follows a change of h better than cell_size."""
    mesh, tdim = domain.mesh, domain.mesh.topology.dim
    one = dolfinx.fem.form(dolfinx.fem.Constant(mesh, 1.0) * ufl.dx(mesh))
    volume = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(one), op=MPI.SUM)
    return float((volume / mesh.topology.index_map(tdim).size_global) ** (1 / tdim)) * domain.L0
