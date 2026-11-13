"""core.linalg: LU by default, any PETSc options, refactorisation after a Constant changes, and
failed solves reported."""

import dolfinx
import numpy as np
import pytest
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from amphix.core.linalg import LinearSolver, make_ksp

MESH = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
V = dolfinx.fem.functionspace(MESH, ("DG", 1))
X = ufl.SpatialCoordinate(MESH)


def projection(c=None, options=None):
    """c (u, v) = (f, v) in DG1 with f in DG1: u = f / c exactly."""
    c = c if c is not None else dolfinx.fem.Constant(MESH, 1.0)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    f = 1 + X[0] + 2 * X[1]
    solver = LinearSolver(c * u * v * ufl.dx, f * v * ufl.dx, V, options)
    out = dolfinx.fem.Function(V)
    exact = dolfinx.fem.Function(V)
    exact.interpolate(dolfinx.fem.Expression(f, V.element.interpolation_points))
    return solver, out, exact


def test_direct_by_default():
    solver, u, exact = projection()
    assert solver.ksp.getType() == "preonly" and solver.ksp.getPC().getType() == "lu"
    assert solver.ksp.getPC().getFactorSolverType() == "mumps"
    solver.solve(u)
    assert np.allclose(u.x.array, exact.x.array, rtol=1e-12, atol=1e-14)


def test_any_petsc_options_and_own_prefix():
    """Pluggable: an iterative solver from options alone; options do not leak to other solvers."""
    solver, u, exact = projection(
        options={"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-13}
    )
    assert solver.ksp.getType() == "cg" and solver.ksp.getPC().getType() == "jacobi"
    solver.solve(u)
    assert np.allclose(u.x.array, exact.x.array, rtol=1e-10, atol=1e-12)
    assert projection()[0].ksp.getType() == "preonly"  # a new solver: the default again
    assert not PETSc.Options().getAll()  # nothing left in the global database
    assert make_ksp(solver.A).getOptionsPrefix() != solver.ksp.getOptionsPrefix()


def test_update_matrix_after_a_constant_changed():
    c = dolfinx.fem.Constant(MESH, 1.0)
    solver, u, exact = projection(c)
    solver.solve(u)
    c.value = 4.0
    solver.update_matrix()
    solver.solve(u)
    assert np.allclose(u.x.array, exact.x.array / 4.0, rtol=1e-12, atol=1e-14)


def test_failed_solve_raises():
    solver, u, _ = projection(
        options={"ksp_type": "cg", "pc_type": "none", "ksp_max_it": 1, "ksp_rtol": 1e-14}
    )
    with pytest.raises(RuntimeError, match="linear solve failed"):
        solver.solve(u)
