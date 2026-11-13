"""The linear solver of a step: the matrix factorised once, the right-hand side every step.

    solver = LinearSolver(a, L, W)                    # LU (MUMPS) by default
    solver = LinearSolver(a, L, W, {"ksp_type": "gmres", "pc_type": "ilu", "ksp_rtol": 1e-12})
    solver.solve(u)                                   # assemble b, back-substitute into u (in W)
    solver.update_matrix()                            # after a Constant of `a` changed (theta)

The step residual is affine in the unknowns, so ufl.lhs / ufl.rhs split it into a (constant while
dt and the coefficients are) and L (new data each step): each step costs one assembly of b and one
back-substitution.

The options are PETSc's own (without the leading dash), added to DIRECT, so any KSP/PC can be
used. Each solver has its own options prefix, so that two solvers in one script do not share
options.
"""

import itertools
from typing import Any, cast

import dolfinx
import dolfinx.fem.petsc
from petsc4py import PETSc

DIRECT = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_14": 80,  # extra MUMPS workspace (percent)
}
_count = itertools.count(1)


def make_ksp(A: PETSc.Mat, options: dict[str, Any] | None = None) -> PETSc.KSP:
    """Return a KSP for A: DIRECT plus the extra PETSc options, under its own prefix."""
    prefix = f"amphix{next(_count)}_"
    ksp = PETSc.KSP().create(A.comm)
    ksp.setOperators(A)
    ksp.setOptionsPrefix(prefix)
    opts = PETSc.Options()
    values = {**DIRECT, **(options or {})}
    for k, v in values.items():
        opts.setValue(prefix + k, v)
    ksp.setFromOptions()
    for k in values:  # read: drop them from the global options database
        opts.delValue(prefix + k)
    return ksp


class LinearSolver:
    """A = assemble(a), factorised once; b = assemble(L) at every solve."""

    def __init__(
        self,
        a: Any,
        L: Any,
        W: dolfinx.fem.FunctionSpace,
        options: dict[str, Any] | None = None,
    ) -> None:
        """Compile the forms (UFL), assemble A, and set up the KSP."""
        self.a, self.L = dolfinx.fem.form(a), dolfinx.fem.form(L)
        self.A = dolfinx.fem.petsc.assemble_matrix(self.a)
        self.A.assemble()
        self.b = dolfinx.fem.petsc.create_vector(W)
        self.ksp = make_ksp(self.A, options)

    def update_matrix(self) -> None:
        """Reassemble A after a Constant in the bilinear form changed; it is refactorised."""
        self.A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(self.A, self.a)
        self.A.assemble()
        self.ksp.setOperators(self.A)

    def assemble_rhs(self) -> PETSc.Vec:
        """Assemble b = L (with the current data) and return it."""
        with self.b.localForm() as bl:
            bl.set(0.0)
        # (the stubs miss the singledispatch overload (Vec, Form) and type the PETSc enums as int)
        dolfinx.fem.petsc.assemble_vector(self.b, self.L)  # type: ignore[arg-type]
        add, reverse = PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE
        self.b.ghostUpdate(addv=add, mode=reverse)  # type: ignore[arg-type]
        return self.b

    def solve(self, u: dolfinx.fem.Function) -> None:
        """Solve A u = b into the Function u (of the mixed space)."""
        self.assemble_rhs()
        self.ksp.solve(self.b, u.x.petsc_vec)
        reason = cast(int, self.ksp.getConvergedReason())  # an int at run time
        if reason <= 0:
            raise RuntimeError(f"linear solve failed, PETSc reason {reason}")
        u.x.scatter_forward()
