"""The Balance protocol: the step mass balance of a model that conserves fluid cell by cell.

A DG model whose mass equation holds for every pressure test function conserves fluid cell by cell
with its own numerical flux Fhat: testing it with the indicator of a cell K gives, over one step,
gained_K + int_{dK} Fhat = 0 exactly. A model states that balance, over the last step
t^n -> t^{n+1}, through five methods built on its solver-unit state:

    trace_degree()          k, the degree of the normal traces of the H(div) velocity
    numerical_flux()        (interior, [(ds, boundary)]): Fhat out of the '+' cell of each
                            interior facet, and the outward Fhat on each boundary
    step_velocities()       {"end": ..., "avg": ...}: the raw velocities of the step
    fluid_gained(q)         the mass equation over the step without its flux part, tested with q
    flux_lifting(psi, dS)   the cell term L_K of the reconstruction (None if there is none)

post.hdiv.Reconstruction depends only on this protocol; balanced(model) says whether a model (or a
model class) has its five methods. Everything is in solver units.
"""

from typing import Any, Protocol

Expr = Any


class Balance(Protocol):
    """A model with a cell-by-cell mass balance over each step (see the module docstring)."""

    domain: Any
    scales: Any
    measures: Any
    state: Any

    def trace_degree(self) -> int:
        """Return the degree of the normal traces of the reconstructed velocity."""
        ...

    def numerical_flux(self) -> tuple[Expr, list[tuple[Any, Expr]]]:
        """Return Fhat over the last step: (interior, [(ds, boundary)])."""
        ...

    def step_velocities(self) -> dict[str, Expr]:
        """Return the raw velocities over the last step: "end" and "avg"."""
        ...

    def fluid_gained(self, q: Any) -> Any:
        """Return the fluid gained over the last step, per unit time, tested with q (a form)."""
        ...

    def flux_lifting(self, psi: Any, dS: Any) -> Any:
        """Return the lifting of the reconstruction's cell equations (a form), or None."""
        ...


METHODS = ("trace_degree", "numerical_flux", "step_velocities", "fluid_gained", "flux_lifting")


def balanced(model: Any) -> bool:
    """Return True if a model, or a model class, has the methods of Balance."""
    return all(callable(getattr(model, m, None)) for m in METHODS)
