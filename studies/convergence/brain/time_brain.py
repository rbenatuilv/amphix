"""Convergence in TIME of poro2 (2F) and poro3 (3F), BRAIN parameters (cvr22), stable pairs.

Manufactured solution: choose the exact fields, let `Manufactured` derive the sources, boundary
data and initial state that make them exact, solve, and measure the error at T. The fields are
polynomials of low degree in x, which the elements reproduce exactly, so only the time error is
left (a coarse mesh is enough). Halving dt must divide the error by 2^order.

Schemes: 2F Newmark (d) + Crank-Nicolson (p); 3F Newmark (d, w). Both second order.

As time_unit.py, with the cvr22 material and brain scales, a 1 cm square, and exact fields of
physical size (d ~ 1 um, w ~ 1 nm, p ~ 100 Pa). Errors are relative to the exact field's norm.

In this regime d and p are quasi-static: inertia is negligible (rho L^2 / (mu T^2) ~ 2e-4) and the
mass equation is a constraint (c0 tiny). Their time error therefore comes only from w, through
div w in the mass equation, and is far smaller than w's. The exact w has div w != 0 so that this
coupling is active: p then shows a clean second order. With div w = 0, p's error stays at a small
dt-independent level (relative ~1e-8) and shows no order. d's error stays tiny (relative 1e-9 to
1e-8) and its level-to-level orders are irregular; its slope over all levels is about 1.9.

Output: time_brain.png, and the errors and orders, printed.
"""

import matplotlib.pyplot as plt
import ufl

import amphix  # noqa: F401 (registers poro2 and poro3)
from amphix.config import from_dict
from amphix.core.model import make_model
from amphix.physics.poro import Manufactured
from amphix.post.norms import error, norm, orders

sin, cos, exp = ufl.sin, ufl.cos, ufl.exp
L, D, W, P = 1e-2, 1e-6, 1e-9, 100.0  # domain size [m], sizes of d [m], w [m], p [Pa]
STEPS = (0.1, 0.05, 0.025, 0.0125)  # dt [s]
T = 1.0
FIELDS = {"poro2": ["d", "p"], "poro3": ["d", "w", "p"]}
PAIRS = {"poro2": (2, 1), "poro3": (2, 1)}  # (deg, deg_p): the stable pairs

# ---- the exact solution: exact in space (d, w quadratic or less, p linear) --------------------
EXACT = {
    "d": lambda x, t: D * exp(t) * ufl.as_vector([(x[0] + 2 * x[1]) / L, (x[0] - x[1]) / L]),
    "w": lambda x, t: W * cos(t) * ufl.as_vector([1 + (x[0] / L) ** 2, 2 + (x[1] / L) ** 2]),
    "p": lambda x, t: P * sin(t) * (1 + (x[0] + x[1]) / L),
}


# ---- the problem: 1 cm square (coarse), cvr22 brain tissue -------------------------------------
def problem(model, dt):
    time = {"T": T, "dt": dt} | ({"theta": 0.5} if model == "poro2" else {})
    deg, deg_p = PAIRS[model]
    return from_dict(
        {
            "model": model,
            "scales": {"L0": L, "T0": 1.0, "P0": 1500.0},
            "material": {"preset": "cvr22"},
            "mesh": {"shape": "rectangle", "corners": [[0, 0], [L, L]], "n": [4, 4]},
            "discretization": {"deg": deg, "deg_p": deg_p},
            "time": time,
        },
        "mms",
    )


# ---- one run: the relative L2 errors at T ------------------------------------------------------
def errors(model, dt):
    prob = problem(model, dt)
    mms = Manufactured(prob, {f: EXACT[f] for f in FIELDS[model]})
    sides = {s: ("displacement", "pressure") for s in ("left", "right", "bottom", "top")}
    m = make_model(prob, sources=mms.sources, boundary_data=mms.boundary_data(sides))
    m.set_initial(**mms.initial)
    m.run()
    q = 6  # quadrature degree
    return {
        f: error(m.quantity(f), u, "L2", m.domain, q) / norm(u, "L2", m.domain, q)
        for f in FIELDS[model]
        for u in [m.data(mms.exact(f))]
    }


# ---- the study: one panel per model ------------------------------------------------------------
if __name__ == "__main__":
    import pathlib

    HERE = pathlib.Path(__file__).parent
    (HERE / "figures").mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, model in zip(axes, FIELDS, strict=True):
        tag = "2F" if model == "poro2" else "3F"
        table = {}
        for dt in STEPS:
            for f, e in errors(model, dt).items():
                table.setdefault(f, []).append(e)
        print(f"\n{tag}")
        for f, errs in table.items():
            rates = "  ".join(f"{r:5.2f}" for r in orders(errs, STEPS))
            print(f"  {f}  " + "  ".join(f"{e:.2e}" for e in errs) + f"   orders {rates}")
            ax.loglog(STEPS, errs, "-o", label=f"{f} L2 ({orders(errs, STEPS)[-1]:.1f})")
        ax.set(title=tag, xlabel="dt [s]", ylabel="relative error")
        ax.set_xticks(STEPS, [f"{s:g}" for s in STEPS])
        ax.minorticks_off()
        ax.legend(fontsize=8)
    fig.suptitle(
        "Brain parameters (cvr22), convergence in time, stable pairs (last observed order)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(f"{HERE}/figures/time_brain.png", dpi=120)
