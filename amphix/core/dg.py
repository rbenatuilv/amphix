"""DG building blocks shared by every physics: symmetric jumps and the SIPG penalty scale.

With n the outward normal of each side of a facet:

    sym_jump(v, n) = sym(v+ (x) n+ + v- (x) n-)    [[v]] of a vector field on an interior facet
    sym_bnd(v, n)  = sym(v (x) n)                  the same on a boundary facet
    ufl.jump(v, n) = v+ . n+ + v- . n-             [[v.n]] (vector) or q+ n+ + q- n- (scalar)
    ufl.avg(w)     = (w+ + w-) / 2                 {w}

The interior penalty of an operator with coefficient c and degree k is c k^2 / hF on interior
facets and c k^2 / h on boundary facets (eta (lambda + 2 mu) k^2 for elasticity, zeta kappa k^2
for Darcy); penalty() gives c k^2.
"""

from typing import Any

import ufl

Expr = Any  # ufl.core.expr.Expr


def sym_jump(v: Expr, n: Expr) -> Expr:
    """Return the symmetrised jump [[v (x) n]] of a vector field on an interior facet."""
    return ufl.sym(ufl.outer(v("+"), n("+")) + ufl.outer(v("-"), n("-")))


def sym_bnd(v: Expr, n: Expr) -> Expr:
    """Return the symmetrised v (x) n on a boundary facet."""
    return ufl.sym(ufl.outer(v, n))


def penalty(coefficient: Expr, k: int) -> Expr:
    """Return the SIPG penalty scale coefficient * k^2 (divided by hF inside, h on boundaries)."""
    return coefficient * k**2
