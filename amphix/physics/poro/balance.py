"""The step mass balance of the poro models (core.balance.Balance).

Over the last step t^n -> t^{n+1}, with th the theta of that step (poro2), from the solver-unit
state (state.previous holds t^n) and the data at t^n (expressions.at(., t_old)):

    poro2   Fhat = th Fhat(p^{n+1}) + (1 - th) Fhat(p^n), Fhat(p) out of K:
                     interior    -kappa {grad p}.n + pen / hF (p_K - p_N)
                     pressure    -kappa grad p.n + pen / h (p - g_p)
                     flux        g_q
            q_avg = -kappa grad p_th,    L(psi) = sum_F int_F omega kappa (psi.n) [[p_th]]
                    (omega = 1/2 inside, 1 on pressure boundaries, [[p]] = p_K - p_N or p - g_p)
            trace degree deg_p
    poro3   Fhat = {Dw / dt}.n inside, (Dw / dt).n on pressure boundaries, 0 on flux boundaries;
            q_avg = Dw / dt,  L = 0  (Dw = w^{n+1} - w^n);  trace degree deg

and `gained` is the mass equation over the step without its flux part. A model mixes one of the
two classes in.
"""

from typing import Any

import ufl

from amphix.physics.poro import boundary
from amphix.physics.poro.forms import b_P, darcy_penalty

Expr = Any


class TwoFieldBalance:
    """The Balance methods of poro2 (mixed into TwoField)."""

    def trace_degree(self) -> int:
        """Return deg_p: the degree of the normal traces of the reconstructed velocity."""
        return self.deg_p

    def _old(self, e: Expr) -> Expr:
        return self.expressions.at(e, self.state.t_old)

    def numerical_flux(self) -> tuple[Expr, list[tuple[Any, Expr]]]:
        """Return th Fhat(p^{n+1}) + (1 - th) Fhat(p^n), Fhat(p) out of K (the flux of darcy())."""
        c, m, n = self.c, self.measures, self.measures.n
        th, pen = self.theta.theta, darcy_penalty(c, self.k)

        def at_level(p: Expr, at: Any) -> tuple[Expr, list]:
            interior = -c["kappa"] * ufl.inner(
                ufl.avg(ufl.grad(p)), n("+")
            ) + pen / m.hF * ufl.jump(p)
            bnd = [
                (ds, -c["kappa"] * ufl.inner(ufl.grad(p), n) + pen / m.h * (p - at(gp)))
                for ds, gp in boundary.pairs(self.conditions, "pressure")
            ]
            return interior, bnd + [
                (ds, at(gq)) for ds, gq in boundary.pairs(self.conditions, "flux")
            ]

        (i1, b1), (i0, b0) = (
            at_level(self.state.current["p"], lambda e: e),
            at_level(self.state.previous["p"], self._old),
        )
        return th * i1 + (1 - th) * i0, [
            (ds, th * a + (1 - th) * b) for (ds, a), (_, b) in zip(b1, b0, strict=True)
        ]

    def step_velocities(self) -> dict[str, Expr]:
        """Return the end-of-step -kappa grad p^{n+1} and the average th q^{n+1} + (1-th) q^n."""
        kappa, th = self.c["kappa"], self.theta.theta
        q1 = -kappa * ufl.grad(self.state.current["p"])
        q0 = -kappa * ufl.grad(self.state.previous["p"])
        return {"end": q1, "avg": th * q1 + (1 - th) * q0}

    def fluid_gained(self, q: Any) -> Any:
        """Return the mass equation (theta-method) of the step without its flux, tested with q."""
        c, m, n = self.c, self.measures, self.measures.n
        th, p, p0 = self.theta.theta, self.state.current["p"], self.state.previous["p"]
        z_th = th * self.state.current["d_t"] + (1 - th) * self.state.previous["d_t"]
        g = self.sources["fluid_source"]
        G = (
            c["c0"] / self.state.dt * (p - p0) * q * m.dx()
            + c["beta"] * (th * p + (1 - th) * p0) * q * m.dx()
            - (th * g + (1 - th) * self._old(g)) * q * m.dx()
            - c["alpha"] * b_P(q, z_th, m, self.conditions.measure("displacement"))
        )
        for ds, _, gd_t in boundary.displacement(self.conditions, self.expressions, self.scales):
            G += c["alpha"] * q * ufl.inner(th * gd_t + (1 - th) * self._old(gd_t), n) * ds
        return G

    def flux_lifting(self, psi: Any, dS: Any) -> Any:
        """Return the lifting of the SIPG term: sum_F int_F omega kappa (psi.n) [[p_th]]."""
        c, n, th = self.c, self.measures.n, self.theta.theta
        p_th = th * self.state.current["p"] + (1 - th) * self.state.previous["p"]
        L = c["kappa"] * ufl.jump(p_th) * ufl.inner(ufl.avg(psi), n("+")) * dS
        for ds, gp in boundary.pairs(self.conditions, "pressure"):
            g_th = th * gp + (1 - th) * self._old(gp)
            L += c["kappa"] * (p_th - g_th) * ufl.inner(psi, n) * ds
        return L


class ThreeFieldBalance:
    """The Balance methods of poro3 (mixed into ThreeField)."""

    def trace_degree(self) -> int:
        """Return deg: the degree of the normal traces of the reconstructed velocity."""
        return self.k

    def _w_rate(self) -> Expr:
        return (self.state.current["w"] - self.state.previous["w"]) / self.state.dt

    def numerical_flux(self) -> tuple[Expr, list[tuple[Any, Expr]]]:
        """Return {Dw / dt}.n inside, (Dw / dt).n on pressure boundaries (0 on flux ones)."""
        n, Dw = self.measures.n, self._w_rate()
        bnd = [(ds, ufl.inner(Dw, n)) for ds, _ in boundary.pairs(self.conditions, "pressure")]
        return ufl.inner(ufl.avg(Dw), n("+")), bnd

    def step_velocities(self) -> dict[str, Expr]:
        """Return the end-of-step w_t^{n+1} (Newmark) and the step average (w^{n+1} - w^n) / dt."""
        return {"end": self.state.current["w_t"], "avg": self._w_rate()}

    def fluid_gained(self, q: Any) -> Any:
        """Return the mass equation over the step, per unit time, without its flux part."""
        c, m, n, dt = self.c, self.measures, self.measures.n, self.state.dt
        Dd = self.state.current["d"] - self.state.previous["d"]
        fp = self.sources["mass_source"]
        old = self.expressions.at(fp, self.state.t_old)
        G = (
            c["c0"] / dt * (self.state.current["p"] - self.state.previous["p"]) * q * m.dx()
            - (fp - old) / dt * q * m.dx()
            - c["alpha"] / dt * b_P(q, Dd, m, self.conditions.measure("displacement"))
        )
        for ds, gd, _ in boundary.displacement(self.conditions, self.expressions, self.scales):
            gd_old = self.expressions.at(gd, self.state.t_old)
            G += c["alpha"] / dt * q * ufl.inner(gd - gd_old, n) * ds
        return G

    def flux_lifting(self, psi: Any, dS: Any) -> Any:
        """Return None: poro3's reconstruction needs no lifting."""
        return None
