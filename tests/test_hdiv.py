"""post.hdiv (the RT reconstruction) on the poro Balance (core.balance, poro.balance): cflux at any
step, conservation facet by facet and cell by cell, and convergence."""

import dolfinx
import numpy as np
import pytest
import ufl
from test_strong import formula

import amphix.physics.poro  # noqa: F401 (registers poro2, poro3)
from amphix.config import from_dict
from amphix.core.model import make_model
from amphix.physics.poro.strong import Manufactured
from amphix.post.hdiv import Reconstruction

MATERIAL = {"lambda": 1.0, "mu": 1.0, "alpha": 1.0, "c0": 1.0, "K": 1.0, "muF": 1.0, "rho": 1.0}
FLUID = {"rho_f": 1.0, "phi": 0.5, "tortuosity": 1.0}


def test_cflux_at_any_step():
    """A reconstruction built after the last step equals one built before the first: the model
    keeps the state at the start of its last step, so cflux needs no history."""
    early, late = (make_model(square("poro2", 1, 1.0)) for _ in range(2))
    rec = Reconstruction(early)
    for _ in early.steps():
        expected = rec.compute().x.array.copy()
    for _ in late.steps():
        pass
    got = Reconstruction(late).compute().x.array
    assert np.abs(got - expected).max() < 1e-12 * np.abs(expected).max()


# ---- conservation ----------------------------------------------------------------------------
def square(model, deg, theta, n=4):
    """Every boundary-condition kind, with time-dependent data."""
    return from_dict(
        {
            "model": model,
            "material": MATERIAL | (FLUID if model == "poro3" else {}),
            "mesh": {"shape": "rectangle", "corners": [[0, 0], [1, 1]], "n": [n, n]},
            "boundaries": {
                "left": {"displacement": [0, 0], "pressure": "sin(t) * (1 + y)"},
                "right": {"traction": [0, 0], "flux": 0},
                "bottom": {
                    "displacement": ["0.1 * sin(t)", 0],
                    "flux": 0 if model == "poro3" else "0.3 * sin(2*t)",
                },
                "top": {"normal_traction": "-sin(t)", "pressure": 0},
            },
            "discretization": {"deg": deg},
            "time": {"T": 0.25, "dt": 0.05} | ({"theta": theta} if model == "poro2" else {}),
        },
        "sq",
    )


def compiled(forms):
    """Compile once, assemble every step."""
    return {k: dolfinx.fem.form(f) for k, f in forms.items()}


def assemble(form):
    """One value per owned cell (with the contributions of the ghost cells' facets)."""
    vec = dolfinx.fem.assemble_vector(form)
    vec.scatter_reverse(dolfinx.la.InsertMode.add)
    return vec.array[: vec.index_map.size_local]


def total(form, comm):
    """A scalar form summed over the ranks."""
    return comm.allreduce(dolfinx.fem.assemble_scalar(form))


@pytest.mark.parametrize(
    "model, deg, theta", [("poro2", 1, 1.0), ("poro2", 2, 0.5), ("poro3", 2, 1.0)]
)
def test_conservative(model, deg, theta):
    """Facet flows = Fhat, no normal jump, div u = -gained."""
    m = make_model(square(model, deg, theta))
    rec = Reconstruction(m)
    u, n, dS = rec.u, m.measures.n, m.measures.dS
    v0 = ufl.TestFunction(dolfinx.fem.functionspace(m.domain.mesh, ("DG", 0)))
    q = ufl.TestFunction(dolfinx.fem.functionspace(m.domain.mesh, ("DG", m.deg_p)))
    interior, bnd = m.numerical_flux()
    mis = (ufl.inner(u("+"), n("+")) - interior) * ufl.jump(v0) * dS + sum(
        (ufl.inner(u, n) - b) * v0 * ds for ds, b in bnd
    )
    if model == "poro3":  # flux walls: no flow
        mis += ufl.inner(u, n) * v0 * m.conditions.measure("flux")
    forms = compiled(
        {
            "mis": mis,
            "through": abs(interior) * (v0("+") + v0("-")) * dS,
            "div": m.fluid_gained(q) + ufl.div(u) * q * m.measures.dx(),
            "div_scale": abs(ufl.div(u)) * q * m.measures.dx(),
        }
    )
    jump = dolfinx.fem.form(ufl.jump(u, n) ** 2 * dS)
    scale = dolfinx.fem.form(ufl.inner(ufl.avg(u), n("+")) ** 2 * dS)
    for _ in m.steps():
        rec.solve()
        a = {k: assemble(f) for k, f in forms.items()}
        assert np.abs(a["mis"]).max() < 1e-12 * a["through"].max()  # facet flows = Fhat
        comm = m.domain.mesh.comm
        assert total(jump, comm) < 1e-24 * total(scale, comm)
        assert np.abs(a["div"]).max() < 1e-11 * np.abs(a["div_scale"]).max()  # div u = -gained


@pytest.mark.parametrize("model, deg", [("poro2", 1), ("poro3", 2)])
def test_cell_balance(model, deg):
    """Every cell's books close with the solver's wall flow and the reconstructed velocity's,
    not with the raw one's (through the Balance methods)."""
    m = make_model(square(model, deg, 1.0))
    rec = Reconstruction(m)
    v = ufl.TestFunction(dolfinx.fem.functionspace(m.domain.mesh, ("DG", 0)))
    meas, n = m.measures, m.measures.n
    interior, bnd = m.numerical_flux()

    def out_of(q):  # q.n out of each cell, from inside it
        return (
            ufl.inner(q("+"), n("+")) * v("+") + ufl.inner(q("-"), n("-")) * v("-")
        ) * meas.dS + ufl.inner(q, n) * v * meas.ds()

    forms = compiled(
        {
            "gained": m.fluid_gained(v),
            "solver": interior * ufl.jump(v) * meas.dS + sum(b * v * ds for ds, b in bnd),
            "throughput": abs(interior) * (v("+") + v("-")) * meas.dS
            + sum(abs(b) * v * ds for ds, b in bnd),
            "rec": out_of(rec.u),
            "end": out_of(m.step_velocities()["end"]),
        }
    )
    for _ in m.steps():
        rec.solve()
        b = {k: assemble(f) for k, f in forms.items()}
        tp = b["throughput"].max()
        assert np.abs(b["gained"] + b["solver"]).max() < 1e-12 * tp
        assert np.abs(b["rec"] - b["solver"]).max() < 1e-12 * tp
        assert np.abs(b["end"] - b["solver"]).max() > 1e-6 * tp


def final_error(model, deg, n, dt, end=0.05, theta=0.5):
    """L2 error at T = end of the reconstructed flux against the exact step-consistent flux."""
    smooth = {
        "d": ["sin(t)**2 * sin(pi*x) * cos(pi*y)", "sin(t)**2 * sin(pi*y) * cos(pi*x)"],
        "w": [
            "sin(t)**2 * (cos(pi*x) * sin(pi*y) + 0.2)",
            "sin(t)**2 * (cos(pi*y) * sin(pi*x) + 0.2)",
        ],
        "p": "sin(t)**2 * (cos(pi*x) * cos(pi*y) + 0.5)",
    }
    fields = ("d", "p") if model == "poro2" else ("d", "w", "p")
    if model == "poro2":
        kinds = {
            "left": ("displacement", "pressure"),
            "bottom": ("displacement", "flux"),
            "right": ("traction", "flux"),
            "top": ("traction", "pressure"),
        }
    else:
        kinds = {
            s: ("displacement" if s in ("left", "bottom") else "traction", "pressure")
            for s in ("left", "bottom", "right", "top")
        }
    p = from_dict(
        {
            "model": model,
            "material": MATERIAL | (FLUID if model == "poro3" else {}),
            "mesh": {"shape": "rectangle", "corners": [[0, 0], [1, 1]], "n": [n, n]},
            "discretization": {"deg": deg},
            "time": {"T": end, "dt": dt} | ({"theta": theta} if model == "poro2" else {}),
        }
    )
    mms = Manufactured(p, {k: formula(smooth[k]) for k in fields})
    m = make_model(p, sources=mms.sources, boundary_data=mms.boundary_data(kinds))
    m.set_initial(**mms.initial)
    rec = Reconstruction(m)
    q_ex = m.data(mms.exact("flux"))  # SI
    th = theta if model == "poro2" else 0.5  # poro3: (w^{n+1} - w^n)/dt ~ the midpoint
    target = th * q_ex + (1 - th) * m.expressions.at(q_ex, m.state.t_old)
    dx = ufl.Measure("dx", domain=m.domain.mesh, metadata={"quadrature_degree": 2 * rec.k + 6})
    form = dolfinx.fem.form(ufl.inner(rec.u_si - target, rec.u_si - target) * dx)
    for _ in m.steps():
        pass
    rec.compute()
    return np.sqrt(abs(total(form, m.domain.mesh.comm)))


@pytest.mark.parametrize(
    "model, deg, ns, dt, order",
    [
        ("poro2", 1, [4, 8, 16], 0.003125, 1),
        ("poro2", 2, [4, 8], 0.0015625, 2),
        ("poro3", 2, [4, 8], 0.0015625, 3),
    ],
)
def test_convergence(model, deg, ns, dt, order):
    """The reconstruction converges at the order of the raw flux: deg_p (poro2), deg + 1 (poro3)."""
    e = np.array([final_error(model, deg, n, dt) for n in ns])
    rate = np.log2(e[-2] / e[-1])
    assert rate > order - 0.35, (e, rate)
