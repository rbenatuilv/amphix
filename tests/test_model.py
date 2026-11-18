"""core.model, through the toy heat model of tests/toy_heat.py: the life cycle, observers, data
given in code, the two state views, the regular-step matrix, the Rannacher start, and
orders of convergence in space and in time."""

import ast
from pathlib import Path

import dolfinx
import numpy as np
import pytest
import toy_heat
import ufl
from toy_heat import Heat

import amphix.core.model as base
from amphix.config import from_dict
from amphix.core import registry
from amphix.core.model import make_model
from amphix.post.norms import error, orders

L = 1e-2  # the square is L x L (m)
K = 1e-4  # diffusivity (m^2/s)
TREE = {
    "model": "toy_heat",
    "scales": {"L0": L, "T0": 2.0},
    "mesh": {"shape": "rectangle", "corners": [[0, 0], [L, L]], "n": [4, 4]},
    "heat": {"k": K, "deg": 1},
    "time": {"T": 0.2, "dt": 0.05, "theta": 0.5},
}


def problem(**changes):
    tree = {**TREE, **changes}
    for key in ("mesh", "heat", "time"):
        tree[key] = {**TREE[key], **changes.get(key, {})}
    return from_dict(tree, "heat")


# u = exp(-t) sin(pi x / L) cos(pi y / L): f = (2 k pi^2 / L^2 - 1) u
def smooth(x, t):
    return ufl.exp(-t) * ufl.sin(np.pi * x[0] / L) * ufl.cos(np.pi * x[1] / L)


def smooth_source(x, t):
    return (2 * K * np.pi**2 / L**2 - 1) * smooth(x, t)


def smooth_flux(x, t):  # -k du/dx on the right side (n = (1, 0))
    return -K * np.pi / L * ufl.exp(-t) * ufl.cos(np.pi * x[0] / L) * ufl.cos(np.pi * x[1] / L)


def linear(x, t):  # in DG1: only the time error is left
    return ufl.sin(t) * (1 + x[0] / L + 2 * x[1] / L)


def linear_source(x, t):
    return ufl.cos(t) * (1 + x[0] / L + 2 * x[1] / L)


def mms(p, exact, source, flux=None):
    """A model whose data all come from code: no `boundaries:` and no `sources:` in the YAML."""
    bnd = {s: {"temperature": exact} for s in ("left", "bottom", "top", "right")}
    if flux is not None:
        bnd["right"] = {"heat_flux": flux}
    m = make_model(p, sources={"heat_source": source}, boundary_data=bnd)
    m.set_initial(u=exact)
    return m


def final_error(m, exact):
    m.run()
    return error(m.quantity("u"), m.data(exact), "L2", m.domain, 4)


def test_registered_and_built_from_the_registry():
    assert registry.get("toy_heat") is Heat
    m = make_model(
        problem(boundaries={s: {"temperature": 0} for s in ("left", "right", "bottom", "top")})
    )
    assert isinstance(m, Heat) and m.n == 0 and m.t == 0.0 and m.system.ndofs == 32 * 3


def test_boundaries_are_complete_only_after_code_data():
    """The YAML has no boundaries; code gives three; the fourth is missing at build time."""
    p = problem()
    with pytest.raises(ValueError, match=r"no conditions for boundaries \['top'\]"):
        make_model(p, boundary_data={s: {"temperature": 0} for s in ("left", "right", "bottom")})


def test_life_cycle_and_observers():
    m = mms(problem(), smooth, smooth_source, smooth_flux)
    seen = []
    m.run([lambda model: seen.append((model.n, model.t))])
    assert seen == [(n, pytest.approx(0.05 * n)) for n in range(5)]  # t0 and after each step
    assert m.n == 4 and list(m.steps()) == []  # at T: nothing left
    with pytest.raises(RuntimeError, match="only before the first step"):
        m.set_initial(u=0.0)


def test_previous_and_the_two_state_views():
    m = mms(problem(), linear, linear_source)
    m.step()
    before = m.field("u").x.array.copy()
    m.step()
    assert np.array_equal(m.previous("u").x.array, before)  # the start of the last step
    copy = m.field("u")
    copy.x.array[:] = 0.0  # an SI copy: the model's own state is untouched
    assert np.abs(m.state.current["u"].x.array).max() > 0
    g = m.quantity("gradient")  # derived, SI: the gradient of the SI field, in K/m
    u = m.field("u")
    scale = error(g, 0 * g, "L2", m.domain, 4)
    assert scale > 0 and error(g, ufl.grad(u) / L, "L2", m.domain, 4) < 1e-12 * scale
    with pytest.raises(KeyError, match="no quantity"):
        m.quantity("flux")
    with pytest.raises(KeyError, match="not state"):
        make_model(
            problem(),
            boundary_data={s: {"temperature": 0} for s in ("left", "right", "bottom", "top")},
        ).set_initial(v=1)


def test_rannacher_start_and_the_regular_matrix():
    bnd = {s: {"temperature": 0} for s in ("left", "right", "bottom", "top")}
    start = make_model(problem(time={"startup_steps": 2}), boundary_data=bnd)
    plain = make_model(problem(), boundary_data=bnd)
    regular = start.system.matrix(regular=True)  # theta = 1/2 while the model is at theta = 1
    assert float(start.theta.theta.value) == 1.0
    diff = regular.copy()
    diff.axpy(-1.0, plain.system.matrix())
    assert diff.norm() < 1e-14 * regular.norm()
    assert start.system.matrix(regular=True) is not start.system.matrix()
    assert plain.system.matrix(regular=True) is plain.system.matrix()  # already regular: A itself
    thetas = []
    for _ in range(4):
        start.step()
        thetas.append(float(start.theta.theta.value))
    assert thetas == [1.0, 1.0, 0.5, 0.5]
    diff = start.system.matrix().copy()
    diff.axpy(-1.0, plain.system.matrix())
    assert diff.norm() < 1e-14 * regular.norm()  # refactorised at step 3
    owned = start.domain.mesh.comm.allreduce(len(start.system.dofs("u")))  # rows on each rank
    assert owned == start.system.ndofs and start.system.rhs().getSize() > 0


def test_order_in_space():
    """P1, Crank-Nicolson with a small dt: L2 order 2 (the error at T against the exact u)."""
    ns = [8, 16, 32]  # (4 -> 8 is pre-asymptotic: 1.70; 16 -> 32: 1.92)
    e = [
        final_error(
            mms(
                problem(mesh={"n": [n, n]}, time={"T": 0.1, "dt": 0.00625}),
                smooth,
                smooth_source,
                smooth_flux,
            ),
            smooth,
        )
        for n in ns
    ]
    assert orders(e, [1 / n for n in ns])[-1] > 1.9, e


@pytest.mark.parametrize("theta, order", [(0.5, 2.0), (1.0, 1.0)])
def test_order_in_time(theta, order):
    """A field in DG1: only the time error; Crank-Nicolson order 2, backward Euler order 1."""
    dts = [0.1, 0.05, 0.025]
    e = [
        final_error(
            mms(problem(time={"T": 1.0, "dt": dt, "theta": theta}), linear, linear_source), linear
        )
        for dt in dts
    ]
    assert abs(orders(e, dts)[-1] - order) < 0.1, e


def test_no_biot_in_the_base():
    """The base knows no physics: no identifier of core/model.py names a Biot concept."""
    tree = ast.parse(Path(base.__file__).read_text())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}
    names |= {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.ClassDef)}
    names |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    for word in ("displacement", "pressure", "newmark", "theta", "darcy", "biot", "poro", "flux"):
        assert not [n for n in names if word in n.lower()], word
    assert toy_heat.Heat.time_keys == {"theta", "startup_steps"}  # theta lives in the physics


def test_derived_quantity_is_never_a_previous_steps_value():
    """The cache of a derived quantity is keyed on the step number (and on set_initial)."""
    m = make_model(
        problem(),
        sources={"heat_source": linear_source},
        boundary_data={s: {"temperature": linear} for s in ("left", "right", "bottom", "top")},
    )
    space = m.quantity("gradient").function_space
    fresh = dolfinx.fem.Function(space)
    expr = dolfinx.fem.Expression(
        ufl.grad(m.state.current["u"]), space.element.interpolation_points
    )
    zero = m.quantity("gradient").x.array.copy()  # at rest: 0
    m.set_initial(u=lambda x, t: 1 + x[0] / L)  # same n = 0, new state: the cache is stale
    assert not np.array_equal(m.quantity("gradient").x.array, zero)
    last = m.quantity("gradient").x.array.copy()
    for _ in m.steps():
        g = m.quantity("gradient").x.array.copy()
        fresh.interpolate(expr)  # solver units; quantity() is SI: / L0
        assert np.allclose(g, fresh.x.array / L, rtol=1e-13, atol=0)
        assert not np.array_equal(g, last)  # never the previous step's value
        assert m.quantity("gradient").x.array.tobytes() == g.tobytes()  # same step: cached
        last = g


def test_a_simple_physics_writes_three_hooks():
    hooks = {
        "elements",
        "quadrature_degree",
        "initialize",
        "residual",
        "before_step",
        "after_step",
        "regular_constants",
        "derived",
        "after_initial",
    }
    assert hooks & set(vars(Heat)) == {"elements", "initialize", "residual"}
    m = make_model(
        problem(), boundary_data={s: {"temperature": 0} for s in ("left", "right", "bottom", "top")}
    )
    assert m.measures.dx().metadata()["quadrature_degree"] == 3  # default: 2 deg + 1
