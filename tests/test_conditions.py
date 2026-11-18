"""core.conditions: completeness checked against the mesh after the code data are merged, shapes,
scaling, order, and the sources."""

import dolfinx
import pytest
import ufl
from parallel import total

from amphix.core import units as u
from amphix.core.conditions import ConditionKind, Conditions, SourceKind, Sources, merge_entries
from amphix.core.expressions import Expressions
from amphix.core.mesh import Measures, build
from amphix.core.units import Scales

KINDS = (
    ConditionKind("displacement", "mechanical", "vector", u.DISPLACEMENT),
    ConditionKind("traction", "mechanical", "vector", u.PRESSURE),
    ConditionKind("normal_traction", "mechanical", "normal", u.PRESSURE),
    ConditionKind("pressure", "hydraulic", "scalar", u.PRESSURE),
    ConditionKind("flux", "hydraulic", "scalar", u.VELOCITY),
)
SOURCES = (
    SourceKind("body_force", "vector", u.BODY_FORCE),
    SourceKind("fluid_source", "scalar", u.RATE),
)
SCALES = Scales(L0=2.0, T0=0.5, P0=3.0)
DOMAIN = build({"shape": "rectangle", "corners": [[0, 0], [2, 2]], "n": [2, 2]}, L0=SCALES.L0)
M = Measures(DOMAIN)
EX = Expressions(DOMAIN.mesh, SCALES, dolfinx.fem.Constant(DOMAIN.mesh, 0.0))
GOOD = {
    "left": {"displacement": [0, 0], "pressure": "1 + y"},
    "right": {"traction": [3.0, 0], "flux": 0},
    "bottom": {"displacement": ["0.1 * x", 0], "flux": 0.2},
    "top": {"normal_traction": -6.0, "pressure": 0},
}


def conditions(entries):
    return Conditions(entries, KINDS, DOMAIN, M, EX, SCALES)


def test_complete_set():
    c = conditions(GOOD)
    assert [x.boundary for x in c.of("displacement")] == ["left", "bottom"]  # entry order
    assert c.measure("pressure").subdomain_id() == (1, 4)  # left, top
    assert (
        c.measure("normal_traction").subdomain_id() == (4,)
        and c.of("traction")[0].boundary == "right"
    )
    flux = c.of("flux")[1]  # bottom: 0.2 m/s -> 0.2 T0 / L0 in solver units
    assert total(flux.value * flux.ds) == pytest.approx(0.2 * 0.5 / 2.0 * 1.0)  # bottom: length 1
    top = c.of("normal_traction")[0]  # -6 Pa n, n = (0, 1): -6 / P0 in solver units
    assert total(ufl.inner(top.value, M.n) * top.ds) == pytest.approx(-6.0 / 3.0)
    assert total(top.si[1] * top.ds) == pytest.approx(-6.0)  # the SI data is kept
    assert c.measure("traction").subdomain_id() == (2,)


def test_a_kind_nobody_uses_has_no_measure():
    entries = {k: {"displacement": [0, 0], "pressure": 0} for k in GOOD}
    assert conditions(entries).measure("flux") is None and conditions(entries).of("flux") == []


@pytest.mark.parametrize(
    "change, match",
    [
        ({"skull": {"pressure": 0, "displacement": [0, 0]}}, r"unknown boundaries \['skull'\]"),
        ({"top": None}, r"no conditions for boundaries \['top'\]"),
        ({"top": {"pressure": 0}}, r"boundaries\.top: needs exactly one mechanical condition"),
        (
            {"top": {"pressure": 0, "flux": 0, "displacement": [0, 0]}},
            "one hydraulic condition .*got 2",
        ),
        (
            {"top": {"pressure": 0, "dispalcement": [0, 0]}},
            r"unknown condition\(s\) \['dispalcement'\]",
        ),
        (
            {"top": {"pressure": 0, "displacement": [0, 0, 0]}},
            r"top\.displacement: needs 2 components",
        ),
        ({"top": {"pressure": [0, 1], "displacement": [0, 0]}}, r"top\.pressure: needs a scalar"),
        (
            {"top": {"pressure": 0, "displacement": lambda x, t: x[0]}},
            "needs 2 components, got shape",
        ),
        ({"top": {"pressure": 0, "normal_traction": [1, 0]}}, r"normal_traction: needs a scalar"),
    ],
)
def test_errors(change, match):
    entries = {k: v for k, v in {**GOOD, **change}.items() if v is not None}
    with pytest.raises(ValueError, match=match):
        conditions(entries)


def test_code_replaces_whole_entries():
    """A boundary given in code replaces its YAML entry; the YAML may have none at all."""
    code = {"top": {"pressure": lambda x, t: x[0], "displacement": [0, 0]}}
    merged = merge_entries(GOOD, code)
    assert merged["top"] == code["top"] and merged["left"] == GOOD["left"]
    assert list(merged) == ["left", "right", "bottom", "top"]  # YAML order kept
    assert merge_entries({}, GOOD) == GOOD
    c = conditions(merged)
    assert [x.boundary for x in c.of("displacement")] == ["left", "bottom", "top"]


def test_sources():
    s = Sources({"fluid_source": "2 * x"}, SOURCES, EX, SCALES)
    zero = s["body_force"]  # absent: a zero Constant of its shape
    assert isinstance(zero, ufl.core.expr.Expr) and zero.ufl_shape == (2,)
    area = total(dolfinx.fem.Constant(DOMAIN.mesh, 1.0) * M.dx())  # 1 in solver units
    assert total(s["fluid_source"] * M.dx()) == pytest.approx(0.5 * 2 * SCALES.L0 * 0.5 * area)
    with pytest.raises(ValueError, match=r"sources: \['mass_source'\] not read"):
        Sources({"mass_source": 1}, SOURCES, EX, SCALES)
    with pytest.raises(ValueError, match=r"sources\.body_force: needs 2 components"):
        Sources({"body_force": 1.0}, SOURCES, EX, SCALES)
    assert total(Sources({}, SOURCES, EX, SCALES)["fluid_source"] * M.dx()) == 0.0  # absent: zero
