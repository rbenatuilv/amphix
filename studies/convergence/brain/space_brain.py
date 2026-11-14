"""Convergence in SPACE of poro2 (2F) and poro3 (3F), BRAIN parameters (cvr22): stable and unstable
pairs.

Manufactured solution: choose the exact fields, let `Manufactured` derive the sources, boundary
data and initial state that make them exact, solve with a tiny dt (so the spatial error
dominates), and measure the error at T. Halving h must divide the error by 2^order.

Pairs (degrees of d/p, or d/w/p):   2F unstable P1/P1, stable P2/P1
                                    3F unstable P2/P2/P2, stable P2/P2/P1
As space_unit.py, with the cvr22 material and brain scales, a 1 cm square, and exact fields of
physical size (d ~ 1 um, w ~ 1 nm, p ~ 100 Pa). Errors are relative to the exact field's norm.
Expected with brain parameters:
    2F  kappa = K / mu_F ~ 1e-13 m^2/(Pa s) no longer stabilises the pressure: the unstable P1/P1
        stalls (p H1 error stuck near 15 %), the stable P2/P1 keeps the optimal orders.
    3F  as with unit parameters: P2/P2/P1 optimal in d and p, P2/P2/P2 loses order in p. New:
        w converges only as fast as grad p (L2 order = degree of p), since w_t ~ -kappa grad p.

Output: space_brain_2F.png, space_brain_3F.png, and the errors and orders, printed.
"""

import matplotlib.pyplot as plt
import ufl

import amphix  # noqa: F401 (registers poro2 and poro3)
from amphix.config import from_dict
from amphix.core.mesh import cell_size
from amphix.core.model import make_model
from amphix.physics.poro import Manufactured
from amphix.post.norms import error, norm, orders

sin, cos, pi = ufl.sin, ufl.cos, ufl.pi
L, D, W, P = 1e-2, 1e-6, 1e-9, 100.0  # domain size [m], sizes of d [m], w [m], p [Pa]
MESHES = (4, 8, 16, 32)  # cells per side
T = 0.1
FIELDS = {"poro2": ["d", "p"], "poro3": ["d", "w", "p"]}
PAIRS = {  # model: {label: (deg, deg_p)}, stable first
    "poro2": {"P2/P1": (2, 1), "P1/P1": (1, 1)},
    "poro3": {"P2/P2/P1": (2, 1), "P2/P2/P2": (2, 2)},
}
DT = {"poro2": 0.003125, "poro3": 0.0015625}


# ---- the exact solution: smooth in x (x, t in SI; X = x / L) ---------------------------------
def X(x):
    return x[0] / L, x[1] / L


EXACT = {
    "d": lambda x, t: (
        D
        * sin(t) ** 2
        * ufl.as_vector(
            [sin(pi * X(x)[0]) * cos(pi * X(x)[1]), sin(pi * X(x)[1]) * cos(pi * X(x)[0])]
        )
    ),
    "w": lambda x, t: (
        W
        * sin(t) ** 2
        * ufl.as_vector(
            [
                cos(pi * X(x)[0]) * sin(pi * X(x)[1]) + 0.2,
                cos(pi * X(x)[1]) * sin(pi * X(x)[0]) + 0.2,
            ]
        )
    ),
    "p": lambda x, t: P * sin(t) ** 2 * (cos(pi * X(x)[0]) * cos(pi * X(x)[1]) + 0.5),
}


# ---- the problem: 1 cm square, cvr22 brain tissue --------------------------------------------
def problem(model, pair, n):
    time = {"T": T, "dt": DT[model]} | ({"theta": 0.5} if model == "poro2" else {})
    return from_dict(
        {
            "model": model,
            "scales": {"L0": L, "T0": 1.0, "P0": 1500.0},
            "material": {"preset": "cvr22"},
            "mesh": {"shape": "rectangle", "corners": [[0, 0], [L, L]], "n": [n, n]},
            "discretization": {"deg": pair[0], "deg_p": pair[1]},
            "time": time,
        },
        "mms",
    )


# ---- one run: the relative errors at T --------------------------------------------------------
def errors(model, pair, n):
    prob = problem(model, pair, n)
    mms = Manufactured(prob, {f: EXACT[f] for f in FIELDS[model]})
    sides = {s: ("displacement", "pressure") for s in ("left", "right", "bottom", "top")}
    m = make_model(prob, sources=mms.sources, boundary_data=mms.boundary_data(sides))
    m.set_initial(**mms.initial)
    m.run()
    q = max(pair) + 4  # quadrature degree, high enough not to pollute the error
    out = {}
    for f in FIELDS[model]:
        u = m.data(mms.exact(f))
        for k in ("L2", "H1"):
            out[f"{f} {k}"] = error(m.quantity(f), u, k, m.domain, q) / norm(u, k, m.domain, q)
    return out, cell_size(m.domain)


# ---- the study: one figure per model, one panel per field --------------------------------------
if __name__ == "__main__":
    import pathlib

    HERE = pathlib.Path(__file__).parent
    (HERE / "figures").mkdir(exist_ok=True)

    for model, pairs in PAIRS.items():
        tag = "2F" if model == "poro2" else "3F"
        fig, axes = plt.subplots(1, len(FIELDS[model]), figsize=(5 * len(FIELDS[model]), 4.5))
        for (label, pair), color in zip(pairs.items(), ("tab:blue", "tab:red"), strict=True):
            table, h = {}, []
            for n in MESHES:
                errs, size = errors(model, pair, n)
                h.append(size)
                for key, e in errs.items():
                    table.setdefault(key, []).append(e)
            print(f"\n{tag} {label}")
            for key, errs in table.items():
                rates = "  ".join(f"{r:5.2f}" for r in orders(errs, h))
                print(f"  {key:6s}  " + "  ".join(f"{e:.2e}" for e in errs) + f"   orders {rates}")
            for ax, field in zip(axes, FIELDS[model], strict=True):
                for kind, style in (("L2", "-o"), ("H1", "--s")):
                    errs = table[f"{field} {kind}"]
                    ax.loglog(
                        h,
                        errs,
                        style,
                        color=color,
                        label=f"{label} {kind} ({orders(errs, h)[-1]:.1f})",
                    )
        for ax, field in zip(axes, FIELDS[model], strict=True):
            ax.set(title=field, xlabel="h [m]", ylabel="relative error")
            ax.legend(fontsize=8)
        fig.suptitle(
            f"Brain parameters (cvr22), {tag} in space. Blue: stable, red: unstable; "
            "solid: L2, dashed: H1; (last observed order)",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(f"{HERE}/figures/space_brain_{tag}.png", dpi=120)
