"""The mixed space of a model: one DG element per field, one linear system per step.

    space = MixedSpace(mesh, {"d": dg_element(mesh, 2, vector=True), "p": dg_element(mesh, 2)})
    u, v = space.trial(), space.test()          # {"d": ..., "p": ...}: trial, test functions
    space.split(solution, fields)               # mixed Function -> one Function per field

Every field lives in the mixed space W (basix.ufl.mixed_element, in the order given), so a step is
one system. Each field also has its collapsed sub-space V[name], where its Function lives, and
dofs[name], the indices of its dofs inside W: local and ghost dofs, as dolfinx collapse() returns
them. owned_dofs(name) keeps only the owned ones (the rows of the step matrix on this rank).
"""

from typing import Any

import basix.ufl
import dolfinx
import numpy as np
import ufl


def dg_element(mesh: dolfinx.mesh.Mesh, degree: int, vector: bool = False) -> Any:
    """Return the discontinuous Lagrange element of a degree (vector-valued if vector)."""
    shape = (mesh.geometry.dim,) if vector else ()
    return basix.ufl.element("DG", mesh.basix_cell(), degree, shape=shape)


class MixedSpace:
    """The mixed space W of some fields, and the collapsed space and dofs of each."""

    def __init__(self, mesh: dolfinx.mesh.Mesh, elements: dict[str, Any]) -> None:
        """Build W from {field: element}; the order of the dict is the order inside W."""
        self.names = list(elements)
        self.W = dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element(list(elements.values())))
        self.V: dict[str, dolfinx.fem.FunctionSpace] = {}
        self.dofs: dict[str, np.ndarray] = {}
        for i, name in enumerate(self.names):
            V, dofs = self.W.sub(i).collapse()
            self.V[name], self.dofs[name] = V, np.asarray(dofs, dtype=np.int64).reshape(-1)

    @property
    def ndofs(self) -> int:
        """Return the global number of unknowns."""
        imap = self.W.dofmap.index_map
        return int(imap.size_global * self.W.dofmap.index_map_bs)

    def owned_dofs(self, name: str) -> np.ndarray:
        """Return the owned dofs of a field inside W (local indices: rows of A on this rank)."""
        nowned = self.W.dofmap.index_map.size_local * self.W.dofmap.index_map_bs
        dofs = self.dofs[name]
        return dofs[dofs < nowned]

    def trial(self) -> dict[str, Any]:
        """Return {field: TrialFunction component}."""
        return dict(zip(self.names, ufl.TrialFunctions(self.W), strict=True))

    def test(self) -> dict[str, Any]:
        """Return {field: TestFunction component}."""
        return dict(zip(self.names, ufl.TestFunctions(self.W), strict=True))

    def functions(self) -> dict[str, dolfinx.fem.Function]:
        """Return a new Function per field, in its collapsed space, named after the field."""
        return {name: dolfinx.fem.Function(self.V[name], name=name) for name in self.names}

    def split(self, solution: dolfinx.fem.Function, into: dict[str, dolfinx.fem.Function]) -> None:
        """Copy each field of a mixed Function into its own Function."""
        for name in self.names:
            into[name].x.array[:] = solution.x.array[self.dofs[name]]
            into[name].x.scatter_forward()

    def join(self, fields: dict[str, dolfinx.fem.Function], into: dolfinx.fem.Function) -> None:
        """Copy Functions per field into a mixed Function (the inverse of split)."""
        for name in self.names:
            into.x.array[self.dofs[name]] = fields[name].x.array
        into.x.scatter_forward()
