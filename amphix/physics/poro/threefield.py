"""Three-field Biot model (poro3): solid displacement d, filtration displacement w, pore pressure p.

w is the relative fluid displacement phi (u_fluid - u_solid), so its time derivative is the Darcy
flux, q = w_t: an unknown of the system, H(div)-like, with [[w.n]] penalised.

Equations (non-dimensional; c holds the coefficients)

    rho d_tt + rhoF w_tt - div( sigma_e(d) - alpha p I ) = f_d          solid momentum
    rhoF d_tt + rhoW w_tt + (1/kappa) w_t + grad p        = f_w          fluid momentum
    c0 p + alpha div d + div w                            = f_p          mass

with rhoF the fluid density, rhoW = rhoF tortuosity / phi, kappa = K / mu_F. With rho_f = 0 the
fluid momentum is Darcy's law and the model reduces to the two-field one.

Discretisation: DG P_deg for d and w, DG P_deg_p for p (default deg - 1: P2/P2/P1 is the stable
choice; equal orders lose a rate in p). SIPG elasticity for d; for w only the normal jump is
penalised, with eta (lambda + 2 mu) k^2 / hF. Time: Newmark-beta for both d and w. Quadrature
degree 2 deg + 1.

Boundary conditions (physics.poro.boundary): pressure p = g_p is natural for w (it enters through
-(p, div w)); flux is only w.n = 0, penalised (checked when the model is built).
"""

from typing import Any, ClassVar

import dolfinx
import numpy as np
import ufl
from petsc4py import PETSc

from amphix.core.model import Model
from amphix.core.spaces import dg_element
from amphix.core.timestepping import Newmark, NewmarkState
from amphix.core.units import ACCELERATION, DISPLACEMENT, PRESSURE, VELOCITY
from amphix.physics.poro import boundary
from amphix.physics.poro.balance import ThreeFieldBalance
from amphix.physics.poro.forms import b_P, b_W, elastic_penalty, elasticity
from amphix.physics.poro.parameters import Discretization, Material


class ThreeField(ThreeFieldBalance, Model):
    """Three-field Biot (d, w, p), with its step mass balance (poro.balance)."""

    name = "poro3"
    fields: ClassVar[dict] = {"d": DISPLACEMENT, "w": DISPLACEMENT, "p": PRESSURE}
    quantities: ClassVar[dict] = {"flux": VELOCITY, "velocity": VELOCITY}
    sections: ClassVar[dict] = {"material": Material, "discretization": Discretization}
    condition_kinds = boundary.CONDITIONS
    source_kinds = boundary.SOURCES_3F
    time_keys = frozenset({"newmark_beta", "newmark_gamma"})
    tracked = frozenset({"d_t", "d_tt", "w_t", "w_tt"})
    restarts = False  # its flux is w_t, which a fields file does not give back

    def elements(self) -> dict[str, Any]:
        """Return DG P_deg (vector) for d and w, DG P_deg_p for p."""
        disc = self.problem.section("discretization")
        self.k, self.deg_p = disc.deg, disc.pressure_degree(self.name)
        mesh = self.domain.mesh
        return {
            "d": dg_element(mesh, self.k, vector=True),
            "w": dg_element(mesh, self.k, vector=True),
            "p": dg_element(mesh, self.deg_p),
        }

    def quadrature_degree(self) -> int:
        """Return 2 deg + 1 (deg: the degree of d and w)."""
        return 2 * self.k + 1

    def initialize(self) -> None:
        """Coefficients, Newmark for d and w, the flux w_t, the solid velocity; zero flux only."""
        mesh, tc = self.domain.mesh, self.problem.time
        disc = self.problem.section("discretization")
        scaled = self.problem.section("material").scaled(self.scales)
        scaled.update(eta=disc.eta, zeta=disc.zeta)
        self.c = {k: dolfinx.fem.Constant(mesh, PETSc.ScalarType(v)) for k, v in scaled.items()}
        newmark = Newmark(tc.newmark_beta, tc.newmark_gamma, self.state.dt)
        self.sd, self.sw = (NewmarkState(self.space.V[f], newmark) for f in ("d", "w"))
        for f, s in (("d", self.sd), ("w", self.sw)):
            self.state.track(f"{f}_t", s.z_old, VELOCITY)
            self.state.track(f"{f}_tt", s.a_old, ACCELERATION)
        self.derive("flux", self.sw.z_old)
        self.derive("velocity", self.sd.z_old)
        times = np.linspace(tc.t0, tc.T, 5)  # sample times
        boundary.require_no_roller(self.conditions)
        boundary.require_zero_flux(self.conditions, mesh, self.state.t, times, self.scales.T0)

    def residual(self, u: dict[str, Any], v: dict[str, Any]) -> Any:
        """Return one step: both momentum equations and the mass equation at t^{n+1}."""
        c, m, n = self.c, self.measures, self.measures.n
        d, w, p, vd, vw, q = u["d"], u["w"], u["p"], v["d"], v["w"], v["p"]
        sd, sw = self.sd, self.sw
        pen_w = elastic_penalty(c, self.k)
        displacement = boundary.displacement(self.conditions, self.expressions, self.scales)
        ds_d, ds_q = self.conditions.measure("displacement"), self.conditions.measure("flux")

        # solid momentum
        F = (
            c["rho"] * ufl.inner(sd.acceleration(d), vd) * m.dx()
            + c["rhoF"] * ufl.inner(sw.acceleration(w), vd) * m.dx()
            + elasticity(d, vd, c, m, displacement, self.k)
            + c["alpha"] * b_P(p, vd, m, ds_d)
            - ufl.inner(self.sources["body_force"], vd) * m.dx()
        )
        for ds, tr in boundary.tractions(self.conditions):
            F -= ufl.inner(tr, vd) * ds

        # fluid momentum: inertia + Darcy drag (1/kappa) w_t + grad p
        F += (
            c["rhoF"] * ufl.inner(sd.acceleration(d), vw) * m.dx()
            + c["rhoW"] * ufl.inner(sw.acceleration(w), vw) * m.dx()
            + 1 / c["kappa"] * ufl.inner(sw.velocity(w), vw) * m.dx()
            + b_W(p, vw, m, ds_q)
            + pen_w / m.hF * ufl.jump(w, n) * ufl.jump(vw, n) * m.dS
            - ufl.inner(self.sources["fluid_force"], vw) * m.dx()
        )
        if ds_q is not None:  # w.n = 0, penalised
            F += pen_w / m.h * ufl.inner(w, n) * ufl.inner(vw, n) * ds_q
        for ds, gp in boundary.pairs(self.conditions, "pressure"):  # p given: natural
            F += gp * ufl.inner(vw, n) * ds

        # mass: c0 p + alpha div d + div w = f_p
        F += (
            c["c0"] * p * q * m.dx()
            - c["alpha"] * b_P(q, d, m, ds_d)
            - b_W(q, w, m, ds_q)
            - self.sources["mass_source"] * q * m.dx()
        )
        for ds, gd, _ in displacement:  # data of the Gamma_d term in b_P
            F += c["alpha"] * q * ufl.inner(gd, n) * ds
        return F

    def after_step(self) -> None:
        """Advance Newmark for d and w."""
        self.sd.advance(self.state.current["d"])
        self.sw.advance(self.state.current["w"])

    def after_initial(self) -> None:
        """Make the initial d and w Newmark's old displacements too."""
        for f, s in (("d", self.sd), ("w", self.sw)):
            s.u_old.x.array[:] = self.state.current[f].x.array
            s.u_old.x.scatter_forward()
