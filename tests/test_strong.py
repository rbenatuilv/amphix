"""physics.poro.strong (Manufactured): the poro models converge at the expected orders in space and
in time against manufactured solutions given in code; and the checks of Manufactured."""

import numpy as np
import pytest
import ufl

import amphix.physics.poro  # noqa: F401 (registers poro2, poro3)
from amphix.config import from_dict
from amphix.core.expressions import FUNCTIONS
from amphix.core.mesh import cell_size
from amphix.core.model import make_model
from amphix.physics.poro.strong import Manufactured
from amphix.post.norms import error, orders

UNIT = {"lambda": 1.0, "mu": 1.0, "alpha": 1.0, "c0": 1.0, "K": 1.0, "muF": 1.0, "rho": 1.0}
FLUID = {"rho_f": 1.0, "phi": 0.5, "tortuosity": 1.0}
SQUARE = {"shape": "rectangle", "corners": [[0, 0], [1, 1]], "n": [4, 4]}

SMOOTH = {  # for the space studies
    "d": ["sin(t)**2 * sin(pi*x) * cos(pi*y)", "sin(t)**2 * sin(pi*y) * cos(pi*x)"],
    "w": ["sin(t)**2 * (cos(pi*x) * sin(pi*y) + 0.2)", "sin(t)**2 * (cos(pi*y) * sin(pi*x) + 0.2)"],
    "p": "sin(t)**2 * (cos(pi*x) * cos(pi*y) + 0.5)",
}
LINEAR = {  # for the time studies: exact in space
    "d": ["exp(t) * (x + 2*y)", "exp(t) * (x - y)"],
    "w": ["cos(t) * (1 + y)", "cos(t) * (2 - x)"],
    "p": "sin(t) * (1 + x + y)",
}
SIDES = {  # poro2 (poro3 has only pressure conditions on the hydraulic side)
    "left": ("displacement", "pressure"),
    "bottom": ("displacement", "flux"),
    "right": ("traction", "flux"),
    "top": ("traction", "pressure"),
}


def formula(value):
    """A field given as a formula string (or a list of them) as a callable (x, t)."""
    if isinstance(value, list):
        parts = [formula(v) for v in value]
        return lambda x, t: ufl.as_vector([f(x, t) for f in parts])

    def f(x, t):
        names = {**FUNCTIONS, "x": x[0], "y": x[1], "t": t}
        return eval(value, {"__builtins__": {}}, names)

    return f


def problem(model, deg, time):
    material = UNIT | (FLUID if model == "poro3" else {})
    return from_dict(
        {
            "model": model,
            "material": material,
            "mesh": SQUARE,
            "discretization": {"deg": deg},
            "time": time,
        },
        "mms",
    )


def study(model, deg, time, exact, levels, norms):
    """Errors at T over a sequence of meshes ({"mesh.n": ...}) or steps ({"time.dt": ...})."""
    base = problem(model, deg, time)
    names = ["d", "w", "p"] if model == "poro3" else ["d", "p"]
    exact = {k: formula(exact[k]) for k in names}
    kinds = SIDES if model == "poro2" else {b: (k[0], "pressure") for b, k in SIDES.items()}
    out, sizes = {}, []
    for overrides in levels:
        p = base.with_(overrides)
        mms = Manufactured(p, exact)
        m = make_model(p, sources=mms.sources, boundary_data=mms.boundary_data(kinds))
        m.set_initial(**mms.initial)
        m.run()
        degree = deg + 4
        for f in [*names, "flux"]:
            for k in norms:
                if not (k == "H1" and f == "flux"):
                    err = error(m.quantity(f), m.data(mms.exact(f)), k, m.domain, degree)
                    out.setdefault(f"{f}_{k}", []).append(err)
        sizes.append(cell_size(m.domain) if "mesh.n" in overrides else m.dt)
    return {k: orders(np.array(v), sizes)[-1] for k, v in out.items()}


SLOW = pytest.mark.slow  # 10-18 s each: the three-field model
CASES = {  # (model, deg, time, exact, levels, norms, expected orders)
    "poro2_space": (
        "poro2",
        1,
        {"T": 0.1, "dt": 0.003125, "theta": 0.5},
        SMOOTH,
        [{"mesh.n": [n, n]} for n in (4, 8, 16)],
        ["L2", "H1"],
        {"d_L2": 2, "d_H1": 1, "p_L2": 2, "p_H1": 1, "flux_L2": 1},
    ),
    "poro2_time": (
        "poro2",
        1,
        {"T": 1.0, "dt": 0.1, "theta": 0.5},
        LINEAR,
        [{"time.dt": dt} for dt in (0.1, 0.05, 0.025)],
        ["L2"],
        {"d_L2": 2, "p_L2": 2, "flux_L2": 2},
    ),
    "poro3_space": (
        "poro3",
        2,
        {"T": 0.1, "dt": 0.0015625},
        SMOOTH,
        [{"mesh.n": [n, n]} for n in (4, 8, 16)],
        ["L2", "H1"],
        {"d_L2": 3, "d_H1": 2, "w_L2": 3, "w_H1": 2, "p_L2": 2, "p_H1": 1, "flux_L2": 3},
    ),
    "poro3_time": (
        "poro3",
        2,
        {"T": 1.0, "dt": 0.1},
        LINEAR,
        [{"time.dt": dt} for dt in (0.1, 0.05, 0.025)],
        ["L2"],
        {"d_L2": 2, "w_L2": 2, "p_L2": 2, "flux_L2": 2},
    ),
}


@pytest.mark.parametrize(
    "case",
    [
        "poro2_space",
        "poro2_time",
        pytest.param("poro3_space", marks=SLOW),
        pytest.param("poro3_time", marks=SLOW),
    ],
)
def test_orders(case):
    *setup, expected = CASES[case]
    rates = study(*setup)
    print(f"\n{case}: " + ", ".join(f"{k} {v:.2f}" for k, v in rates.items()))
    assert set(rates) == set(expected)
    for key, order in expected.items():
        assert rates[key] > order - 0.2, (key, rates[key])


def test_checks():
    p = problem("poro2", 1, {"T": 0.1, "dt": 0.05})
    exact = {k: formula(SMOOTH[k]) for k in ("d", "p")}
    with pytest.raises(ValueError, match=r"needs exactly \['d', 'p'\]; missing \['p'\]"):
        Manufactured(p, {"d": exact["d"]})
    with pytest.raises(ValueError, match="normal_traction cannot be exact; use traction"):
        Manufactured(p, exact).boundary_data({"top": ("normal_traction", "pressure")})
