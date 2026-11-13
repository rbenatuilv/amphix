"""Norms in SI, and observed orders of convergence.

    norm(e, "L2", domain, degree)                    ||e||, with e in SI on the domain's mesh
    error(f, exact, "H1", domain, degree)            ||f - exact||, f an SI Function
    Difference(space_a, space_b, L0, ["L2"])(a, b)   {kind: ||a - b||} on the mesh of b
    orders(errors, sizes)                            log(e_i / e_i+1) / log(s_i / s_i+1)

The kinds are "L2", the L2 norm, and "H1", the broken H1 seminorm ||grad_h e||, the gradient taken
cell by cell. The mesh is in solver units (x = L0 x*), so in SI

    ||e||_L2^2 = L0^dim int |e|^2 dx*        ||e||_H1^2 = L0^(dim - 2) int |grad* e|^2 dx*

The quadrature degree is the caller's; deg + 4 (the degree of d plus 4) keeps error norms clear of
quadrature error.

A mesh-independence check remeshes at several h and compares the runs with Difference. Both fields
are interpolated into DG on the mesh of b at interior (Gauss-Legendre) points: b exactly, and a
exactly when the meshes match. Otherwise a is sampled at those points, and points just outside its
mesh (curved boundaries) are evaluated in the nearest cell within `padding`.
"""

from collections.abc import Sequence
from typing import Any

import basix
import basix.ufl
import dolfinx
import numpy as np
import ufl
from mpi4py import MPI

from amphix.core.mesh import Domain

KINDS = ("L2", "H1")


def norm_form(e: Any, kind: str, L0: float, dx: Any) -> Any:
    """Return the UFL integrand of the squared norm (SI) of e: L2, or the broken H1 seminorm."""
    if kind not in KINDS:
        raise ValueError(f"norm kind must be one of {KINDS}, got {kind!r}")
    dim = dx.ufl_domain().geometric_dimension
    if kind == "L2":
        return L0**dim * ufl.inner(e, e) * dx
    return L0 ** (dim - 2) * ufl.inner(ufl.grad(e), ufl.grad(e)) * dx


def _integrate(form: Any, comm: MPI.Comm) -> float:
    return float(comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(form)), op=MPI.SUM))


def _dx(mesh: dolfinx.mesh.Mesh, degree: int | None) -> Any:
    md = {} if degree is None else {"metadata": {"quadrature_degree": degree}}
    return ufl.Measure("dx", domain=mesh, **md)


def norm(e: Any, kind: str, domain: Domain, degree: int | None = None) -> float:
    """Return ||e|| in SI (e an SI expression on the domain's mesh)."""
    mesh = domain.mesh
    return float(np.sqrt(_integrate(norm_form(e, kind, domain.L0, _dx(mesh, degree)), mesh.comm)))


def error(f: Any, exact: Any, kind: str, domain: Domain, degree: int | None = None) -> float:
    """Return ||f - exact|| in SI: f an SI Function, exact an SI expression (model.data(...))."""
    return norm(f - exact, kind, domain, degree)


class Difference:
    """||a - b|| of a in space_from and b in space_to, on the mesh of b.

    relative: divided by ||b|| (when > 0). padding: how far outside the mesh of a a point may be
    and still be evaluated (in the mesh's units: solver units), for curved boundaries.
    """

    def __init__(
        self,
        space_from: dolfinx.fem.FunctionSpace,
        space_to: dolfinx.fem.FunctionSpace,
        L0: float,
        kinds: Sequence[str] = ("L2",),
        degree: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        """Build the common DG space and the interpolation data (options: padding, relative)."""
        options = options or {}
        mesh = space_to.mesh
        deg = max(space_from.ufl_element().degree, space_to.ufl_element().degree)
        el = basix.ufl.element(
            "DG",
            mesh.basix_cell(),
            deg,
            shape=space_to.value_shape,
            lagrange_variant=basix.LagrangeVariant.gl_warped,
        )
        common = dolfinx.fem.functionspace(mesh, el)
        self.a, self.b = dolfinx.fem.Function(common), dolfinx.fem.Function(common)
        nloc = mesh.topology.index_map(mesh.topology.dim).size_local
        self.cells = np.arange(nloc, dtype=np.int32)
        padding = float(options.get("padding", 1e-10))
        self.data = dolfinx.fem.create_interpolation_data(
            common, space_from, self.cells, padding=padding
        )
        dx = _dx(mesh, degree)
        self.forms = {k: dolfinx.fem.form(norm_form(self.a - self.b, k, L0, dx)) for k in kinds}
        self.scales = None
        if options.get("relative", False):
            self.scales = {k: dolfinx.fem.form(norm_form(self.b, k, L0, dx)) for k in kinds}
        self.comm = mesh.comm

    def __call__(self, a: dolfinx.fem.Function, b: dolfinx.fem.Function) -> dict[str, float]:
        """Return {kind: ||a - b||} (or relative)."""
        self.a.interpolate_nonmatching(a, self.cells, self.data)
        self.b.interpolate(b)
        out = {}
        for k, form in self.forms.items():
            num = self.comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM)
            den = (
                self.comm.allreduce(dolfinx.fem.assemble_scalar(self.scales[k]), op=MPI.SUM)
                if self.scales
                else 0.0
            )
            out[k] = float(np.sqrt(num / den)) if den > 0 else float(np.sqrt(num))
        return out


def orders(errors: Sequence[float], sizes: Sequence[float]) -> np.ndarray:
    """Return the observed orders log(e_i / e_i+1) / log(s_i / s_i+1) between successive levels."""
    e, s = np.asarray(errors, dtype=float), np.asarray(sizes, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(e[:-1] / e[1:]) / np.log(s[:-1] / s[1:])
