"""Manufactured solutions: the data that make given fields the exact solution.

    mms = Manufactured(problem, {"d": lambda x, t: ..., "p": lambda x, t: ...})   # SI x, t (UFL)
    model = make_model(problem, sources=mms.sources,
                       boundary_data=mms.boundary_data({"left": ("displacement", "pressure"), ...}))
    model.set_initial(**mms.initial)
    error(model.quantity("p"), model.data(mms.exact("p")), "L2", model.domain, deg + 4)

From the exact fields everything else is derived symbolically and exactly, in SI:

    sources         the model's strong form applied to them: poro2 body_force, fluid_source;
                    poro3 body_force, fluid_force, mass_source
    boundary data   displacement d; traction (sigma_e(d) - alpha p I) n; pressure p; flux q.n
                    (a normal_traction cannot be exact: the exact traction has a tangential part)
    initial state   at t0 (not rest): every field, and f_t, f_tt of the Newmark ones (d; d and w)
    exact(name)     a field, "flux" (poro2: q = -K / mu_F grad p; poro3: q = w_t) or "velocity"

Every value is a callable (x, t) of the SI position and time, evaluated by the model on its own
mesh.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import ufl
from ufl.domain import extract_unique_domain

from amphix.core.expressions import expand, spatial_divergence, spatial_gradient, time_derivative

FIELDS = {"poro2": ("d", "p"), "poro3": ("d", "w", "p")}
NEWMARK = {"poro2": ("d",), "poro3": ("d", "w")}  # the second-order fields
SOURCES = {
    "poro2": ("body_force", "fluid_source"),
    "poro3": ("body_force", "fluid_force", "mass_source"),
}
KINDS = ("displacement", "traction", "pressure", "flux")
Exact = Callable[[Any, Any], Any]


class Manufactured:
    """The exact solution of one problem's model, and the data derived from it (callables, SI)."""

    def __init__(self, problem: Any, exact: Mapping[str, Exact]) -> None:
        """Take every unknown of the model as a callable (x, t) -> UFL, in SI."""
        if problem.model not in FIELDS:
            raise ValueError(f"no strong form for {problem.model!r}; known: {sorted(FIELDS)}")
        want = FIELDS[problem.model]
        missing, extra = set(want) - set(exact), set(exact) - set(want)
        if missing or extra:
            raise ValueError(
                f"exact: {problem.model} needs exactly {list(want)}; "
                f"missing {sorted(missing)}, unknown {sorted(extra)}"
            )
        self.model, self.material, self.L0 = (
            problem.model,
            problem.section("material"),
            problem.scales.L0,
        )
        self.fields = dict(exact)

    # -- calculus in SI at (x, t) -----------------------------------------------------------------
    def _at(self, x: Any, t: Any) -> dict[str, Any]:
        return {k: ufl.as_ufl(f(x, t)) for k, f in self.fields.items()}

    def _dt(self, e: Any, t: Any, n: int = 1) -> Any:
        for _ in range(n):
            e = time_derivative(e, t)
        return e

    def _sigma_e(self, u: Any) -> Any:
        m = self.material
        grad, div = spatial_gradient(u, self.L0), spatial_divergence(u, self.L0)
        return 2 * m.mu * ufl.sym(grad) + m.lam * div * ufl.Identity(len(u))

    def _strong(self, F: dict[str, Any], t: Any) -> dict[str, Any]:
        """Return the sources that make F exact."""
        if self.model == "poro3":
            return self._strong3(F, t)
        return self._strong2(F, t)

    def _strong3(self, F: dict[str, Any], t: Any) -> dict[str, Any]:
        """Return the poro3 sources for which F is an exact solution."""
        m, L0 = self.material, self.L0
        d, w, p = F["d"], F["w"], F["p"]
        rhoW = m.rho_f * m.tortuosity / m.phi
        stress = self._sigma_e(d) - m.alpha * p * ufl.Identity(len(d))
        f_d = (
            m.rho * self._dt(d, t, 2) + m.rho_f * self._dt(w, t, 2) - spatial_divergence(stress, L0)
        )
        f_w = (
            m.rho_f * self._dt(d, t, 2)
            + rhoW * self._dt(w, t, 2)
            + m.muF / m.K * self._dt(w, t)
            + spatial_gradient(p, L0)
        )
        f_p = m.c0 * p + m.alpha * spatial_divergence(d, L0) + spatial_divergence(w, L0)
        return {"body_force": f_d, "fluid_force": f_w, "mass_source": f_p}

    def _strong2(self, F: dict[str, Any], t: Any) -> dict[str, Any]:
        """Return the poro2 sources for which F is an exact solution."""
        m, L0 = self.material, self.L0
        d, p = F["d"], F["p"]
        stress = self._sigma_e(d) - m.alpha * p * ufl.Identity(len(d))
        f = m.rho * self._dt(d, t, 2) - spatial_divergence(stress, L0)
        g = (
            m.c0 * self._dt(p, t)
            + m.alpha * spatial_divergence(self._dt(d, t), L0)
            - spatial_divergence(m.K / m.muF * spatial_gradient(p, L0), L0)
            + m.beta * p
        )
        return {"body_force": f, "fluid_source": g}

    def _flux(self, F: dict[str, Any], t: Any) -> Any:
        """Return the exact flux: poro2 q = -kappa grad p, poro3 q = w_t."""
        if self.model == "poro3":
            return self._dt(F["w"], t)
        return -self.material.K / self.material.muF * spatial_gradient(F["p"], self.L0)

    # -- what a model and the error norms need ----------------------------------------------------
    @property
    def sources(self) -> dict[str, Exact]:
        """Return {source: callable}, the strong form of the exact fields."""
        return {
            name: (lambda x, t, name=name: expand(self._strong(self._at(x, t), t)[name]))
            for name in SOURCES[self.model]
        }

    def exact(self, name: str) -> Exact:
        """Return the exact value (SI) of a field, of "flux" or of "velocity" (d_t)."""
        if name == "flux":
            return lambda x, t: expand(self._flux(self._at(x, t), t))
        if name == "velocity":
            return lambda x, t: self._dt(self._at(x, t)["d"], t)
        return self.fields[name]

    def boundary_data(self, kinds: Mapping[str, Sequence[str]]) -> dict[str, dict[str, Exact]]:
        """Return {boundary: {kind: callable}}: the exact data of each condition kind given."""
        return {b: {kind: self._condition(b, kind) for kind in given} for b, given in kinds.items()}

    def _condition(self, boundary: str, kind: str) -> Exact:
        if kind not in KINDS:
            hint = "; use traction" if kind == "normal_traction" else ""
            raise ValueError(f"boundaries.{boundary}: {kind} cannot be exact{hint}")
        m = self.material

        def data(x: Any, t: Any) -> Any:
            F = self._at(x, t)
            n = ufl.FacetNormal(extract_unique_domain(x))
            if kind == "displacement":
                return F["d"]
            if kind == "traction":
                return expand(
                    (self._sigma_e(F["d"]) - m.alpha * F["p"] * ufl.Identity(len(F["d"]))) * n
                )
            if kind == "pressure":
                return F["p"]
            return expand(ufl.inner(self._flux(F, t), n))

        return data

    @property
    def initial(self) -> dict[str, Exact]:
        """Return the exact state at t0 as set_initial keywords (f, and f_t, f_tt if Newmark)."""
        out = dict(self.fields)
        for name in NEWMARK[self.model]:
            f = self.fields[name]
            out[f"{name}_t"] = lambda x, t, f=f: self._dt(ufl.as_ufl(f(x, t)), t)
            out[f"{name}_tt"] = lambda x, t, f=f: self._dt(ufl.as_ufl(f(x, t)), t, 2)
        return out
