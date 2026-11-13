"""The DG forms of poroelasticity, shared by the two- and three-field models.

Non-dimensional (solver units); c is the model's dict of coefficient Constants (Material.scaled plus
eta, zeta), m its core.mesh.Measures. Notation as in core.dg: [[.]] jump, {.} average on interior
facets, n the outward normal, h the cell diameter, hF the harmonic mean of the two on an interior
facet, k the degree of d.

    elasticity(d, v, c, m, displacement, k)   SIPG of -div sigma_e(d), Nitsche d = g_d
    b_P(p, v, m, ds_d)                        DG form of (grad p, v)
    darcy(p, q, c, m, ds_p, k)                SIPG of -div(kappa grad p) + beta p, Nitsche p
    b_W(p, w, m, ds_q)                        DG form of (grad p, w) (three-field)
"""

from typing import Any

import ufl

from amphix.core.dg import penalty, sym_bnd, sym_jump
from amphix.core.mesh import Measures

Expr = Any


def eps(u: Expr) -> Expr:
    """Return the small-strain tensor sym(grad u)."""
    return ufl.sym(ufl.grad(u))


def sigma_e(u: Expr, c: dict[str, Any]) -> Expr:
    """Return the effective (elastic) stress 2 mu eps(u) + lambda div(u) I."""
    return 2 * c["mu"] * eps(u) + c["lam"] * ufl.div(u) * ufl.Identity(len(u))


def elastic_penalty(c: dict[str, Any], k: int) -> Expr:
    """Return the SIPG penalty of elasticity, eta (lambda + 2 mu) k^2 (also poro3's w.n jumps)."""
    return penalty(c["eta"] * (c["lam"] + 2 * c["mu"]), k)


def elasticity(
    d: Expr, v: Expr, c: dict[str, Any], m: Measures, displacement: list, k: int
) -> Expr:
    """Return the SIPG residual of -div sigma_e(d), with Nitsche d = g_d on Gamma_d.

        (sigma_e(d), eps(v)) - <{sigma_e(d)}, [[v n]]> - <{sigma_e(v)}, [[d n]]>
          + pen/hF <[[d n]], [[v n]]>                                    (interior facets)
          - <sigma_e(d), (v n)> - <sigma_e(v), ((d - g_d) n)> + pen/h <(d - g_d) n, v n>  (Gamma_d)

    pen = eta (lambda + 2 mu) k^2; displacement is [(ds, g_d, velocity)] (poro.boundary).
    """
    n, pen = m.n, elastic_penalty(c, k)
    F = (
        ufl.inner(sigma_e(d, c), eps(v)) * m.dx()
        - ufl.inner(ufl.avg(sigma_e(d, c)), sym_jump(v, n)) * m.dS
        - ufl.inner(ufl.avg(sigma_e(v, c)), sym_jump(d, n)) * m.dS
        + pen / m.hF * ufl.inner(sym_jump(d, n), sym_jump(v, n)) * m.dS
    )
    for ds, g, _ in displacement:
        F += (
            -ufl.inner(sigma_e(d, c), sym_bnd(v, n)) * ds
            - ufl.inner(sigma_e(v, c), sym_bnd(d - g, n)) * ds
            + pen / m.h * ufl.inner(sym_bnd(d - g, n), sym_bnd(v, n)) * ds
        )
    return F


def b_P(p: Expr, v: Expr, m: Measures, ds_d: Any) -> Expr:
    """Return the DG form of (grad p, v) = -(p, div v) + <{p}, [[v.n]]> + <p, v.n>_{Gamma_d}.

    Used as +alpha b_P(p, v) in the solid momentum equation and -alpha b_P(q, d_t) in the mass
    equation, so the discrete coupling is exactly antisymmetric (energy-stable). ds_d: the measure
    of the displacement boundaries (None if there are none).
    """
    n = m.n
    F = -p * ufl.div(v) * m.dx() + ufl.avg(p) * ufl.jump(v, n) * m.dS
    if ds_d is not None:
        F += p * ufl.inner(v, n) * ds_d
    return F


def darcy_penalty(c: dict[str, Any], k: int) -> Expr:
    """Return the SIPG penalty of the Darcy operator, zeta kappa k^2 (Fumagalli et al. 2024)."""
    return penalty(c["zeta"] * c["kappa"], k)


def darcy(p: Expr, q: Expr, c: dict[str, Any], m: Measures, ds_p: Any, k: int) -> Expr:
    """Return the SIPG form of -div(kappa grad p) + beta p, Nitsche p = . on Gamma_p.

    Only the part that depends on p; the data are in the model's mass data. ds_p: the measure of the
    pressure boundaries (None if there are none).
    """
    n, pen = m.n, darcy_penalty(c, k)
    F = (
        c["kappa"] * ufl.inner(ufl.grad(p), ufl.grad(q)) * m.dx()
        - c["kappa"] * ufl.inner(ufl.avg(ufl.grad(p)), ufl.jump(q, n)) * m.dS
        - c["kappa"] * ufl.inner(ufl.avg(ufl.grad(q)), ufl.jump(p, n)) * m.dS
        + pen / m.hF * ufl.jump(p) * ufl.jump(q) * m.dS
        + c["beta"] * p * q * m.dx()
    )
    if ds_p is not None:
        F += (
            -c["kappa"] * ufl.inner(ufl.grad(p), n) * q * ds_p
            - c["kappa"] * ufl.inner(ufl.grad(q), n) * p * ds_p
            + pen / m.h * p * q * ds_p
        )
    return F


def b_W(p: Expr, w: Expr, m: Measures, ds_q: Any) -> Expr:
    """Return the DG form of (grad p, w) = -(p, div w) + <{p}, [[w.n]]> + <p, w.n>_{Gamma_q}.

    Three-field only. On pressure boundaries the term <p, w.n> carries the given pressure (the
    right-hand side), so it is not here; ds_q: the measure of the flux boundaries (None if none).
    """
    n = m.n
    F = -p * ufl.div(w) * m.dx() + ufl.avg(p) * ufl.jump(w, n) * m.dS
    if ds_q is not None:
        F += p * ufl.inner(w, n) * ds_q
    return F
