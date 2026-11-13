"""The poro condition and source kinds, and their data as the forms use it.

Every boundary carries one mechanical and one hydraulic condition:

    mechanical   displacement      d = g_d                 Nitsche, and its velocity in the mass
                                                           equation (poro2), or g_d (poro3)
                 traction          (sigma_e - alpha p I) n = t_N    natural, the total traction
                 normal_traction   ... = s n  (a fluid at pressure P on the wall: s = -P)
    hydraulic    pressure          p = g_p                 poro2: Nitsche; poro3: natural
                 flux              outward q.n = g_q       poro2: natural; poro3: only 0 (w.n = 0)

Sources: poro2 reads body_force f [N/m^3] and fluid_source g [1/s]; poro3 reads body_force f_d,
fluid_force f_w [N/m^3] and mass_source f_p [-] (accumulated since t0).

Data, in solver units (core.conditions does the scaling and the completeness checks):

    displacement(conditions, expressions, scales)  [(ds, g_d, velocity of g_d)]  exact d/dt (SI)
    tractions(conditions)                          [(ds, t_N)]  traction and normal_traction
    pairs(conditions, kind)                        [(ds, value)]  pressure, flux
    require_zero_flux(...)                         poro3: every flux value is zero at all times
"""

from collections.abc import Callable
from typing import Any

import dolfinx
import numpy as np
from mpi4py import MPI

from amphix.core.conditions import ConditionKind, Conditions, SourceKind
from amphix.core.expressions import Expressions
from amphix.core.units import (
    BODY_FORCE,
    DISPLACEMENT,
    NONE,
    PRESSURE,
    RATE,
    VELOCITY,
    Scales,
)

CONDITIONS = (
    ConditionKind("displacement", "mechanical", "vector", DISPLACEMENT),
    ConditionKind("traction", "mechanical", "vector", PRESSURE),
    ConditionKind("normal_traction", "mechanical", "normal", PRESSURE),
    ConditionKind("pressure", "hydraulic", "scalar", PRESSURE),
    ConditionKind("flux", "hydraulic", "scalar", VELOCITY),
)
SOURCES_2F = (
    SourceKind("body_force", "vector", BODY_FORCE),
    SourceKind("fluid_source", "scalar", RATE),
)
SOURCES_3F = (
    SourceKind("body_force", "vector", BODY_FORCE),
    SourceKind("fluid_force", "vector", BODY_FORCE),
    SourceKind("mass_source", "scalar", NONE),
)

Pair = tuple[Any, Any]


def displacement(
    conditions: Conditions, expressions: Expressions, scales: Scales
) -> list[tuple[Any, Any, Any]]:
    """Return [(ds, g_d, velocity of g_d)] in solver units; the velocity is the exact d/dt in SI."""
    factor = scales.factor(VELOCITY)
    return [(c.ds, c.value, factor * expressions.ddt(c.si)) for c in conditions.of("displacement")]


def tractions(conditions: Conditions) -> list[Pair]:
    """Return [(ds, t_N)] of the traction and normal_traction boundaries, in the entry order."""
    given = conditions.of("traction") + conditions.of("normal_traction")
    return [(c.ds, c.value) for c in sorted(given, key=_order(conditions))]


def pairs(conditions: Conditions, kind: str) -> list[Pair]:
    """Return [(ds, value)] of one kind (pressure or flux), solver units."""
    return [(c.ds, c.value) for c in conditions.of(kind)]


def _order(conditions: Conditions) -> Callable[[Any], int]:
    """Sort key: the position of a condition's boundary in the entries."""
    order = {name: i for i, name in enumerate(conditions.boundaries)}
    return lambda c: order[c.boundary]


def require_zero_flux(
    conditions: Conditions,
    mesh: dolfinx.mesh.Mesh,
    t: dolfinx.fem.Constant,
    times: np.ndarray,
    T0: float,
) -> None:
    """Raise unless every flux value is 0 at the given times [s] (poro3: only w.n = 0 exists).

    A number must be 0; any other data is checked by the boundary integral of its square at each
    time.
    """
    saved = t.value.copy()
    try:
        for c in conditions.of("flux"):
            if isinstance(c.si, dolfinx.fem.Constant):
                if np.any(c.si.value != 0):
                    raise ValueError(f"boundaries.{c.boundary}.flux: poro3 supports only flux: 0")
                continue
            form = dolfinx.fem.form(c.si**2 * c.ds)
            area = dolfinx.fem.form(dolfinx.fem.Constant(mesh, 1.0) * c.ds)
            size = max(_integrate(area, mesh), 1.0)
            for time in times:
                t.value = time / T0
                if _integrate(form, mesh) > 1e-20 * size:
                    raise ValueError(
                        f"boundaries.{c.boundary}.flux: poro3 supports only "
                        f"zero flux; it is not zero at t = {time:.3g} s"
                    )
    finally:
        t.value = saved


def _integrate(form: Any, mesh: dolfinx.mesh.Mesh) -> float:
    return float(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM))
