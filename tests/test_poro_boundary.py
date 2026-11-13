"""physics.poro.boundary: the kinds, the data of each condition in solver units (with the exact
velocity of a prescribed displacement), the order of the tractions, and the poro3 rule that
only a zero flux exists."""

import dolfinx
import numpy as np
import pytest
import ufl

from amphix.core.conditions import Conditions
from amphix.core.expressions import Expressions
from amphix.core.mesh import Measures, build
from amphix.core.units import Scales
from amphix.physics.poro import boundary as b

SCALES = Scales(L0=2.0, T0=0.5, P0=3.0)
DOMAIN = build({"shape": "rectangle", "corners": [[0, 0], [2, 2]], "n": [2, 2]}, L0=SCALES.L0)
M = Measures(DOMAIN)
T = dolfinx.fem.Constant(DOMAIN.mesh, 0.3)  # t* = 0.3: t = 0.15 s
EX = Expressions(DOMAIN.mesh, SCALES, T)


def conditions(entries):
    return Conditions(entries, b.CONDITIONS, DOMAIN, M, EX, SCALES)


def total(form):
    return dolfinx.fem.assemble_scalar(dolfinx.fem.form(form))


ENTRIES = {
    "left": {"normal_traction": "-2 * t", "pressure": 1.0},
    "bottom": {"displacement": ["0.1 * sin(t) * x", 0], "flux": 0.5},
    "right": {"traction": [3.0, 0], "flux": 0},
    "top": {"normal_traction": -6.0, "pressure": "y"},
}


def test_kinds():
    assert [(k.name, k.family) for k in b.CONDITIONS] == [
        ("displacement", "mechanical"),
        ("traction", "mechanical"),
        ("normal_traction", "mechanical"),
        ("pressure", "hydraulic"),
        ("flux", "hydraulic"),
    ]
    assert [s.name for s in b.SOURCES_2F] == ["body_force", "fluid_source"]
    assert [s.name for s in b.SOURCES_3F] == ["body_force", "fluid_force", "mass_source"]


def test_displacement_and_its_exact_velocity():
    [(ds, g, g_t)] = b.displacement(conditions(ENTRIES), EX, SCALES)
    x, t = SCALES.L0 * ufl.SpatialCoordinate(DOMAIN.mesh), SCALES.T0 * T  # SI
    want_g = ufl.as_vector([0.1 * ufl.sin(t) * x[0], 0]) / SCALES.L0
    want_v = ufl.as_vector([0.1 * ufl.cos(t) * x[0], 0]) * SCALES.T0 / SCALES.L0  # velocity
    for got, want in ((g, want_g), (g_t, want_v)):
        assert total(ufl.inner(got - want, got - want) * ds) < 1e-28


def test_tractions_in_the_order_of_the_boundaries():
    """One traction list in boundary order, normal tractions as s n among them."""
    got = b.tractions(conditions(ENTRIES))
    assert [ds.subdomain_id() for ds, _ in got] == [1, 2, 4]  # left, right, top
    ds, top = got[2]
    assert total(ufl.inner(top, M.n) * ds) == pytest.approx(-6.0 / 3.0)  # -6 Pa n, / P0, length 1
    assert [ds.subdomain_id() for ds, _ in b.pairs(conditions(ENTRIES), "pressure")] == [1, 4]


@pytest.mark.parametrize(
    "flux, ok",
    [
        (0, True),
        ("0", True),
        (lambda x, t: 0 * x[0], True),
        (1e-3, False),
        ("sin(t)", False),  # zero at t0, not later: caught at a later sample
    ],
)
def test_poro3_only_zero_flux(flux, ok):
    entries = {**ENTRIES, "bottom": {"displacement": [0, 0], "flux": flux}}
    times = np.linspace(0.0, 1.0, 5)
    if ok:
        b.require_zero_flux(conditions(entries), DOMAIN.mesh, T, times, SCALES.T0)
    else:
        with pytest.raises(
            ValueError, match=r"boundaries\.(bottom|right)\.flux: poro3 supports only"
        ):
            b.require_zero_flux(conditions(entries), DOMAIN.mesh, T, times, SCALES.T0)
    assert float(T.value) == 0.3  # the time Constant is restored
