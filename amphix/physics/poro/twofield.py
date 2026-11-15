"""Two-field Biot model (poro2): solid displacement d and pore pressure p.

Equations (non-dimensional; c holds the coefficients)

    rho d_tt - div( sigma_e(d) - alpha p I ) = f                       solid momentum
    c0 p_t + alpha div d_t - div( kappa grad p ) + beta p = g           fluid mass

with sigma_e(d) = 2 mu eps(d) + lambda div(d) I and kappa = K / mu_F. The Darcy flux is not an
unknown: it is recovered from p,

    q = -kappa grad p_h            (cellwise, exactly, in DG_{deg_p - 1}; quantity "flux")

so it is discontinuous across elements and not mass-conserving (post.hdiv gives the one that is).

Discretisation: DG P_deg for d, DG P_deg_p for p (default deg_p = deg), SIPG for both operators,
Nitsche for the Dirichlet data. Time: Newmark-beta for d, the theta-method for the mass equation;
with startup_steps > 0 the first steps use theta = 1 (Rannacher start). Quadrature degree
2 deg + 1, whatever deg_p is.

Boundary conditions and sources: physics.poro.boundary. Initial state: rest, or
set_initial(d=..., d_t=..., d_tt=..., p=...) in SI.
"""

from typing import Any, ClassVar

import dolfinx
import ufl
from petsc4py import PETSc

from amphix.core.model import Model
from amphix.core.spaces import dg_element
from amphix.core.timestepping import Newmark, NewmarkState, ThetaMethod
from amphix.core.units import ACCELERATION, DISPLACEMENT, PRESSURE, VELOCITY
from amphix.physics.poro import boundary
from amphix.physics.poro.balance import TwoFieldBalance
from amphix.physics.poro.forms import b_P, darcy, darcy_penalty, elasticity
from amphix.physics.poro.parameters import Discretization, Material


class TwoField(TwoFieldBalance, Model):
    """Two-field Biot (d, p), with its step mass balance (poro.balance)."""

    name = "poro2"
    fields: ClassVar[dict] = {"d": DISPLACEMENT, "p": PRESSURE}
    quantities: ClassVar[dict] = {"flux": VELOCITY, "velocity": VELOCITY}
    sections: ClassVar[dict] = {"material": Material, "discretization": Discretization}
    condition_kinds = boundary.CONDITIONS
    source_kinds = boundary.SOURCES_2F
    time_keys = frozenset({"theta", "startup_steps", "newmark_beta", "newmark_gamma"})

    def elements(self) -> dict[str, Any]:
        """Return DG P_deg (vector) for d and DG P_deg_p for p."""
        disc = self.problem.section("discretization")
        self.k, self.deg_p = disc.deg, disc.pressure_degree(self.name)
        mesh = self.domain.mesh
        return {"d": dg_element(mesh, self.k, vector=True), "p": dg_element(mesh, self.deg_p)}

    def quadrature_degree(self) -> int:
        """Return 2 deg + 1 (deg: the degree of d)."""
        return 2 * self.k + 1

    def initialize(self) -> None:
        """Coefficients, Newmark for d, theta, and the flux and solid velocity as quantities."""
        mesh, tc = self.domain.mesh, self.problem.time
        disc = self.problem.section("discretization")
        scaled = self.problem.section("material").scaled(self.scales)
        scaled.update(eta=disc.eta, zeta=disc.zeta)
        self.c = {k: dolfinx.fem.Constant(mesh, PETSc.ScalarType(v)) for k, v in scaled.items()}
        self.sd = NewmarkState(
            self.space.V["d"], Newmark(tc.newmark_beta, tc.newmark_gamma, self.state.dt)
        )
        self.state.track("d_t", self.sd.z_old, VELOCITY)
        self.state.track("d_tt", self.sd.a_old, ACCELERATION)
        self.theta = ThetaMethod(tc.theta, tc.startup_steps, mesh)
        self.schedule(self.theta)
        Vq = dolfinx.fem.functionspace(mesh, ("DG", max(self.deg_p - 1, 0), (mesh.geometry.dim,)))
        self.derive("flux", -self.c["kappa"] * ufl.grad(self.state.current["p"]), Vq)
        self.derive("velocity", self.sd.z_old)

    def mass_data(self, q: Any) -> Any:
        """Return the right-hand side of the mass equation at the current t.

        The source, the pressure (Nitsche) and flux data, and the boundary velocity of a prescribed
        displacement.
        """
        c, m, n = self.c, self.measures, self.measures.n
        pen = darcy_penalty(c, self.k)
        G = self.sources["fluid_source"] * q * m.dx()
        for ds, gp in boundary.pairs(self.conditions, "pressure"):
            G += -c["kappa"] * ufl.inner(ufl.grad(q), n) * gp * ds + pen / m.h * gp * q * ds
        for ds, gq in boundary.pairs(self.conditions, "flux"):
            G += -gq * q * ds  # kappa grad p . n = -q.n
        for ds, _, gd_t in self._displacement:  # data of the Gamma_d term inside b_P
            G += -c["alpha"] * q * ufl.inner(gd_t, n) * ds
        return G

    def residual(self, u: dict[str, Any], v: dict[str, Any]) -> Any:
        """Return one step: momentum at t^{n+1}, mass with the theta-method."""
        c, m, th, dt = self.c, self.measures, self.theta.theta, self.state.dt
        d, p, vd, q = u["d"], u["p"], v["d"], v["p"]
        sd, p_old = self.sd, self.state.previous["p"]
        self._displacement = boundary.displacement(self.conditions, self.expressions, self.scales)
        ds_d, ds_p = boundary.ds_d(self.conditions), self.conditions.measure("pressure")
        rollers = boundary.rollers(self.conditions)  # d.n = 0: no data in the mass equation

        # solid momentum at t^{n+1}; d_tt written through the new d (Newmark)
        F = (
            c["rho"] * ufl.inner(sd.acceleration(d), vd) * m.dx()
            + elasticity(d, vd, c, m, self._displacement, self.k, rollers=rollers)
            + c["alpha"] * b_P(p, vd, m, ds_d)
            - ufl.inner(self.sources["body_force"], vd) * m.dx()
        )
        for ds, tr in boundary.tractions(self.conditions):
            F -= ufl.inner(tr, vd) * ds

        # fluid mass with the theta-method:
        #   c0 (p - p_old)/dt - alpha b_P(q, z_theta) + theta A(p) + (1 - theta) A(p_old)
        #     = theta G(t^{n+1}) + (1 - theta) G(t^n),    z_theta = theta z^{n+1} + (1 - theta) z^n
        z = th * sd.velocity(d) + (1 - th) * sd.z_old
        G = self.mass_data(q)
        F += (
            c["c0"] * (p - p_old) / dt * q * m.dx()
            - c["alpha"] * b_P(q, z, m, ds_d)
            + th * darcy(p, q, c, m, ds_p, self.k)
            + (1 - th) * darcy(p_old, q, c, m, ds_p, self.k)
            - th * G
            - (1 - th) * self.expressions.at(G, self.state.t_old)
        )
        return F

    def after_step(self) -> None:
        """Advance Newmark: the new d_tt and d_t from the new d."""
        self.sd.advance(self.state.current["d"])

    def after_initial(self) -> None:
        """Make the initial d Newmark's old displacement too (d_t, d_tt were set in place)."""
        self.sd.u_old.x.array[:] = self.state.current["d"].x.array
        self.sd.u_old.x.scatter_forward()
