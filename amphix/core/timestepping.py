"""Time schemes: Newmark-beta for second-order fields, the theta-method with a Rannacher start.

Newmark, with the new displacement u as the unknown and (u, z, a) the old displacement, velocity
and acceleration:

    a^{n+1} = (u - u^n) / (beta dt^2) - z^n / (beta dt) - (1 - 2 beta) / (2 beta) a^n
    z^{n+1} = gamma / (beta dt) (u - u^n) - (gamma / beta - 1) z^n - dt (gamma / (2 beta) - 1) a^n

Both are affine in u. Inside the forms they are UFL expressions of the trial function
(NewmarkState.velocity, .acceleration); after the solve they are evaluated on the dof arrays
(NewmarkState.advance). (beta, gamma) = (1/4, 1/2): average acceleration, second order, no damping.

Theta-method: a first-order equation is weighted theta at t^{n+1} and 1 - theta at t^n (blend).
With startup_steps = m > 0 the first m steps use theta = 1 (Rannacher start), then the regular
theta. ThetaMethod.before_step(n) switches before step m + 1 and says so, since the matrix depends
on theta and must be refactorised.
"""

from typing import Any

import dolfinx
import numpy as np

Value = Any  # a UFL expression or a NumPy array: the formulas work on both


class Newmark:
    """Newmark-beta: new acceleration and velocity from the new u and the old (u, z, a)."""

    def __init__(self, beta: float, gamma: float, dt: float) -> None:
        """Parameters (beta, gamma) and the step dt (solver units)."""
        self.beta, self.gamma, self.dt = beta, gamma, dt

    def acceleration(self, u: Value, u_old: Value, z_old: Value, a_old: Value) -> Value:
        """Return a^{n+1}."""
        b, dt = self.beta, self.dt
        return (u - u_old) / (b * dt**2) - z_old / (b * dt) - (1 - 2 * b) / (2 * b) * a_old

    def velocity(self, u: Value, u_old: Value, z_old: Value, a_old: Value) -> Value:
        """Return z^{n+1}."""
        b, g, dt = self.beta, self.gamma, self.dt
        return (g / (b * dt)) * (u - u_old) - (g / b - 1) * z_old - dt * (g / (2 * b) - 1) * a_old


class NewmarkState:
    """The old displacement, velocity and acceleration of one Newmark-integrated field."""

    def __init__(self, V: dolfinx.fem.FunctionSpace, newmark: Newmark) -> None:
        """Three Functions of the field's space, at rest."""
        self.newmark = newmark
        self.u_old, self.z_old, self.a_old = (dolfinx.fem.Function(V) for _ in range(3))

    def velocity(self, u: Value) -> Value:
        """Return the new velocity as an expression of the new displacement u."""
        return self.newmark.velocity(u, self.u_old, self.z_old, self.a_old)

    def acceleration(self, u: Value) -> Value:
        """Return the new acceleration as an expression of the new displacement u."""
        return self.newmark.acceleration(u, self.u_old, self.z_old, self.a_old)

    def advance(self, u: dolfinx.fem.Function) -> None:
        """After a solve: the new a and z (from the old state), then u_old <- u."""
        x = u.x.array
        old = (self.u_old.x.array, self.z_old.x.array, self.a_old.x.array)
        a, z = self.newmark.acceleration(x, *old), self.newmark.velocity(x, *old)
        self.a_old.x.array[:], self.z_old.x.array[:], self.u_old.x.array[:] = a, z, x
        for f in (self.a_old, self.z_old, self.u_old):
            f.x.scatter_forward()


class ThetaMethod:
    """theta as a Constant of the forms, with a Rannacher start (startup_steps at theta = 1)."""

    def __init__(self, theta: float, startup_steps: int, mesh: dolfinx.mesh.Mesh) -> None:
        """Theta of the regular steps; the first startup_steps steps use 1."""
        self.target, self.startup_steps = float(theta), int(startup_steps)
        start = 1.0 if self.startup_steps > 0 else self.target
        self.theta = dolfinx.fem.Constant(mesh, np.float64(start))

    def before_step(self, n: int) -> bool:
        """Before step n (1, 2, ...): switch to the regular theta at step startup_steps + 1.

        Returns True when theta changed, so the step matrix must be reassembled.
        """
        if n == self.startup_steps + 1 and float(self.theta.value) != self.target:
            # (dolfinx's stub types Constant.value as a float32 array)
            self.theta.value = self.target  # type: ignore[assignment]
            return True
        return False

    def regular(self) -> dict[dolfinx.fem.Constant, float]:
        """Return {theta: its regular value}: the constants of a regular step's matrix."""
        return {self.theta: self.target}

    def blend(self, new: Value, old: Value) -> Value:
        """Return theta new + (1 - theta) old."""
        return self.theta * new + (1 - self.theta) * old
