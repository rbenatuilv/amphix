"""core.timestepping: Newmark on arrays and in UFL; the Rannacher start of theta."""

import dolfinx
import numpy as np
import pytest
from mpi4py import MPI

from amphix.core.timestepping import Newmark, NewmarkState, ThetaMethod

MESH = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)


def motion(t, u0=0.3, v0=-1.2, a=2.5):
    """u, u_t, u_tt of a motion with constant acceleration."""
    return u0 + v0 * t + a * t**2 / 2, v0 + a * t, a


def test_average_acceleration_is_exact_for_constant_acceleration():
    nm, t, dt = Newmark(0.25, 0.5, 0.1), 0.7, 0.1
    (u0, z0, a0), (u1, z1, a1) = motion(t), motion(t + dt)
    assert nm.acceleration(u1, u0, z0, a0) == pytest.approx(a1, rel=1e-12)
    assert nm.velocity(u1, u0, z0, a0) == pytest.approx(z1, rel=1e-12)


@pytest.mark.parametrize("beta, gamma", [(0.25, 0.5), (0.3, 0.6)])
def test_t2_is_the_trapezoid_of_t1(beta, gamma):
    """The velocity update is z^{n+1} = z^n + dt [(1 - gamma) a^n + gamma a^{n+1}], with the
    acceleration update substituted."""
    nm, dt = Newmark(beta, gamma, 0.05), 0.05
    u, u_old, z_old, a_old = 1.3, 0.9, 2.0, -4.0
    a_new = nm.acceleration(u, u_old, z_old, a_old)
    assert nm.velocity(u, u_old, z_old, a_old) == pytest.approx(
        z_old + dt * ((1 - gamma) * a_old + gamma * a_new), rel=1e-13
    )


def test_newmark_state_in_ufl_and_on_arrays():
    V = dolfinx.fem.functionspace(MESH, ("DG", 0))
    nm = Newmark(0.25, 0.5, 0.1)
    s = NewmarkState(V, nm)
    s.u_old.x.array[:], s.z_old.x.array[:], s.a_old.x.array[:] = motion(0.0)
    u = dolfinx.fem.Function(V)
    u.x.array[:] = motion(0.1)[0]
    ufl_z = dolfinx.fem.Function(V)  # the UFL expression, interpolated
    ufl_z.interpolate(dolfinx.fem.Expression(s.velocity(u), V.element.interpolation_points))
    s.advance(u)
    assert np.allclose(ufl_z.x.array, motion(0.1)[1], rtol=1e-12)
    assert np.allclose(s.z_old.x.array, motion(0.1)[1], rtol=1e-12)
    assert np.allclose(s.a_old.x.array, motion(0.1)[2], rtol=1e-12)
    assert np.array_equal(s.u_old.x.array, u.x.array)


def test_rannacher_start():
    th = ThetaMethod(0.5, 2, MESH)
    seen = [(th.before_step(n), float(th.theta.value)) for n in range(1, 6)]
    assert seen == [(False, 1.0), (False, 1.0), (True, 0.5), (False, 0.5), (False, 0.5)]
    assert th.regular() == {th.theta: 0.5}
    plain = ThetaMethod(0.5, 0, MESH)
    assert float(plain.theta.value) == 0.5 and not any(plain.before_step(n) for n in range(1, 4))
    euler = ThetaMethod(1.0, 3, MESH)  # starting with theta = 1 and staying there: no switch
    assert not any(euler.before_step(n) for n in range(1, 6)) and float(euler.theta.value) == 1.0
    assert th.blend(4.0, 2.0) is not None
