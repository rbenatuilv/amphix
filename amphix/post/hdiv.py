"""The conservative velocity of a step: the H(div) (Raviart-Thomas) reconstruction.

    rec = Reconstruction(model)          # any core.balance.Balance model: poro2, poro3
    for t in model.steps():
        u = rec.compute()                # the velocity over the step just taken (SI, RT)

Post-processing: the solver is not touched. After a step, the solution at both ends of the step is
turned into a velocity whose flow through every facet is the solver's own. compute() may be called
at any step; before any step the velocity is zero.

Why. The solver conserves fluid cell by cell with its numerical flux Fhat, a single value on each
facet. The raw velocity (poro2: -kappa grad p_h; poro3: w_t) is a field on the cells: on a facet
the two neighbouring cells see different normal flows, and a transport equation fed with it sees
fluid created or destroyed. In H(div) the normal component is continuous across facets.

The reconstruction. u in RT with normal traces of degree k (model.trace_degree()) such that

    every facet F    int_F (u.n) mu = int_F Fhat mu                  for mu in P_k(F)
    every cell K     int_K u . psi = int_K q_avg . psi + L(psi)       for psi in [P_{k-1}(K)]^d

with Fhat, q_avg and L the model's (core.balance; physics.poro.balance). The facet equations make
every cell's balance exact; the cell equations, with the lifting L of the symmetric SIPG term, make
it exact against every pressure test function: (div u, q)_K = -gained(q). This is the construction
of Ern, Nicaise and Vohralik (2007) and Bastian and Riviere (2003).

Solve. Facet and interior unknowns decouple: the facet block is the normal-trace mass matrix (block
diagonal by facet), the interior block the moments of the RT bubbles (block diagonal by cell). Both
are constant, assembled and factorised once; each compute() assembles two right-hand sides. In
parallel each rank takes its owned dofs (the bubbles of a cell belong to the cell's owner); the
facet integrals on rank boundaries need the ghosted facets (GhostMode.shared_facet).
"""

from typing import Any

import basix.ufl
import dolfinx
import dolfinx.fem.petsc
import numpy as np
import ufl
from petsc4py import PETSc

from amphix.core.balance import Balance
from amphix.core.linalg import make_ksp
from amphix.core.units import VELOCITY


class Reconstruction:
    """The Raviart-Thomas velocity of the last step of a Balance model (module docstring)."""

    def __init__(self, model: Balance) -> None:
        """Build the RT space, the two blocks (factorised once) and the right-hand side forms."""
        mesh, n = model.domain.mesh, model.measures.n
        self.model, self.k = model, model.trace_degree()
        if self.k < 1:
            raise ValueError("flux reconstruction needs normal traces of degree >= 1")
        self.V = dolfinx.fem.functionspace(
            mesh, basix.ufl.element("RT", mesh.basix_cell(), self.k + 1)
        )
        self.u = dolfinx.fem.Function(self.V, name="cflux")  # solver units
        self.u_si = dolfinx.fem.Function(self.V, name="cflux")
        self._factor = model.scales.factor(VELOCITY)

        md = {"quadrature_degree": 2 * self.k + 2}
        dS = ufl.Measure("dS", domain=mesh, metadata=md)
        ds_all = ufl.Measure("ds", domain=mesh, metadata=md)
        dx = ufl.Measure("dx", domain=mesh, metadata=md)
        u, v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        W = dolfinx.fem.functionspace(mesh, ("DG", self.k - 1, (mesh.geometry.dim,)))
        psi = ufl.TestFunction(W)

        # facet equations: normal-trace mass (bubbles have no normal trace: their rows are zero)
        a_facet = (
            ufl.inner(u("+"), n("+")) * ufl.inner(v("+"), n("+")) * dS
            + ufl.inner(u, n) * ufl.inner(v, n) * ds_all
        )
        interior, bnd = model.numerical_flux()
        rhs_facet = interior * ufl.inner(v("+"), n("+")) * dS + sum(
            b * ufl.inner(v, n) * ds for ds, b in bnd
        )
        # cell equations: moments against [P_{k-1}]^d
        a_cell = ufl.inner(u, psi) * dx
        rhs_cell = ufl.inner(model.step_velocities()["avg"], psi) * dx
        lifting = model.flux_lifting(psi, dS)  # poro2: of the symmetric SIPG term
        if lifting is not None:
            rhs_cell += lifting
        self._blocks(a_facet, a_cell, W)
        self.rhs_facet, self.rhs_cell = dolfinx.fem.form(rhs_facet), dolfinx.fem.form(rhs_cell)

    def _blocks(self, a_facet: Any, a_cell: Any, W: Any) -> None:
        """Split the owned RT dofs into facet and interior ones; assemble and factorise both blocks.

        The index sets are global (PETSc numbering); self._f and self._b are the same dofs in the
        local numbering, to move values between the RT vectors and the blocks' vectors.
        """
        mesh, comm = self.V.mesh, self.V.mesh.comm
        tdim = mesh.topology.dim
        ncells = mesh.topology.index_map(tdim).size_local
        imap = self.V.dofmap.index_map
        local_int = np.asarray(self.V.dofmap.dof_layout.entity_dofs(tdim, 0), dtype=np.int32)
        self._b = np.unique(self.V.dofmap.list[:ncells][:, local_int].ravel()).astype(np.int32)
        ndofs = imap.size_local
        assert np.all(self._b < ndofs), "the bubbles of an owned cell must be owned"
        self._f = np.setdiff1d(np.arange(ndofs, dtype=np.int32), self._b)
        start = imap.local_range[0]
        self.is_f = PETSc.IS().createGeneral(self._f + start, comm=comm)
        self.is_b = PETSc.IS().createGeneral(self._b + start, comm=comm)
        bs = W.dofmap.index_map_bs  # all moments of the owned cells
        rows = W.dofmap.index_map.local_range[0] * bs + np.arange(
            W.dofmap.index_map.size_local * bs, dtype=np.int32
        )
        self.is_rows = PETSc.IS().createGeneral(rows, comm=comm)
        A = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a_facet))
        A.assemble()
        B = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a_cell))
        B.assemble()
        self.A_ff = A.createSubMatrix(self.is_f, self.is_f)
        self.B_rb = B.createSubMatrix(self.is_rows, self.is_b)
        self.B_rf = B.createSubMatrix(self.is_rows, self.is_f)
        self.ksp_f, self.ksp_b = make_ksp(self.A_ff), make_ksp(self.B_rb)
        self.uf, self.ub = self.A_ff.createVecRight(), self.B_rb.createVecRight()
        self.bf, self.rhs_b = self.A_ff.createVecLeft(), self.B_rb.createVecLeft()

    def solve(self) -> dolfinx.fem.Function:
        """Compute the velocity of the last step in solver units (self.u) and return it.

        The blocks' vectors hold the owned dofs in the order of the index sets, so the owned part
        of the RT vectors maps to them through the local indices self._f and self._b.
        """
        bf = dolfinx.fem.petsc.assemble_vector(self.rhs_facet)
        bf.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        bc = dolfinx.fem.petsc.assemble_vector(self.rhs_cell)
        bc.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.bf.setArray(bf.getArray(readonly=True)[self._f])  # owned part only
        self.ksp_f.solve(self.bf, self.uf)
        self.B_rf.mult(self.uf, self.rhs_b)  # move the facet part to the right-hand side
        self.rhs_b.aypx(-1.0, bc)  # bc's owned rows are is_rows, in order
        self.ksp_b.solve(self.rhs_b, self.ub)
        x = self.u.x.array
        x[self._f] = self.uf.getArray(readonly=True)
        x[self._b] = self.ub.getArray(readonly=True)
        self.u.x.scatter_forward()  # the ghosts
        bf.destroy()
        bc.destroy()
        return self.u

    def compute(self) -> dolfinx.fem.Function:
        """Return the conservative velocity over the last step, in SI (an RT Function)."""
        self.solve()
        self.u_si.x.array[:] = self.u.x.array / self._factor
        return self.u_si
