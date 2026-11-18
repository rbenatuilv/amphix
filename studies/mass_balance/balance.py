"""Mass balance of the fluid velocity, 2F and 3F: the raw velocity against its RT reconstruction.

The solver conserves fluid cell by cell, over every step:

    gained_K + int_{dK} u.n = 0        for every cell K

(gained_K: the fluid stored in K over the step, minus the source). A transport equation sees the
velocity only through its normal flow on the facets, so we test that balance with the facet flow
of two velocities:
    raw  the velocity read off the solution, averaged over the step (2F: the theta-average of
         -kappa grad p, 3F: (w^{n+1} - w^n) / dt). It is discontinuous across facets; each
         facet gets the average of its two sides.
    RT   post.hdiv.Reconstruction: the same velocity made H(div) (Raviart-Thomas), with the
         solver's own flux on every facet.
Imbalance of a cell e_K = gained_K + int_{dK} u.n. We report it relative to the flow through the
cells (int_{dK} |u.n| of the RT velocity): the total, sum |e_K| / sum flow_K, and the worst cell,
max |e_K| / flow_K. Expected: RT at round-off. Raw: far from it in 2F; at round-off
already in 3F, whose facet average of (w^{n+1} - w^n) / dt is exactly the solver's flux.

Case (the vascular source run): the brain annulus, cvr22, stable pairs, the vascular source
g(t) (option 2) over one cardiac cycle, p = 0 on both surfaces, the pial surface fixed, the
ventricle free.

Output: balance.png, and the worst values over the cycle, printed. Runs in parallel too (mpirun):
each rank balances its owned cells, the totals and maxima are reduced over the ranks, and rank 0
prints and draws.
"""

import dolfinx
import matplotlib.pyplot as plt
import numpy as np
import ufl
from mpi4py import MPI

import amphix  # noqa: F401 (registers poro2 and poro3, and the brain presets and waveforms)
from amphix.brain import geometries
from amphix.config import from_dict
from amphix.core.model import make_model
from amphix.post.hdiv import Reconstruction

PAIRS = {"poro2": (2, 1), "poro3": (2, 1)}  # (deg, deg_p): 2F P2/P1, 3F P2/P2/P1
T, DT = 1.0, 0.02  # one cardiac cycle [s]
# The vascular source g(t) = G (sin 2 pi t + 0.5 sin 4 pi t), G = 3.53e-3 1/s (option 2).
# 2F takes g as its fluid_source; 3F's mass equation is in displacement form, so it takes the
# volume added up to t, int_0^t g, as its mass_source.
SOURCE = {
    "poro2": {"fluid_source": {"waveform": "vascular_option2"}},
    "poro3": {
        "mass_source": "3.53e-3 * ((1 - cos(2*pi*t)) / (2*pi) + 0.5 * (1 - cos(4*pi*t)) / (4*pi))"
    },
}


# ---- the problem ---------------------------------------------------------------------------------
def problem(model):
    deg, deg_p = PAIRS[model]
    return from_dict(
        {
            "model": model,
            "scales": {"L0": 1e-2, "T0": 1.0, "P0": 1500.0},
            "material": {"preset": "cvr22"},
            "mesh": geometries.annulus(h=5e-4),
            "boundaries": {
                "pial": {"displacement": [0, 0], "pressure": 0},
                "ventricle": {"normal_traction": 0, "pressure": 0},
            },
            "sources": SOURCE[model],
            "discretization": {"deg": deg, "deg_p": deg_p},
            "time": {"T": T, "dt": DT} | ({"theta": 0.5} if model == "poro2" else {}),
        },
        "balance",
    )


# ---- the balance of every cell, for one velocity -------------------------------------------------
def per_cell(form):
    """Assemble a form linear in a DG0 test function: one value per owned cell."""
    vec = dolfinx.fem.assemble_vector(dolfinx.fem.form(form))
    vec.scatter_reverse(dolfinx.la.InsertMode.add)  # facet terms assembled by the next rank
    return vec.array[: vec.index_map.size_local]


def imbalance(m, u, v, n):
    """e_K = gained_K + sum over the facets of K of the facet flow of u (average of both sides)."""
    flow = ufl.inner(ufl.avg(u), n("+")) * ufl.jump(v) * ufl.dS + ufl.inner(u, n) * v * ufl.ds
    return per_cell(m.fluid_gained(v)) + per_cell(flow)


# ---- the study -----------------------------------------------------------------------------------
if __name__ == "__main__":
    import pathlib

    HERE = pathlib.Path(__file__).parent
    comm = MPI.COMM_WORLD
    root = comm.rank == 0
    if root:
        (HERE / "figures").mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, model in zip(axes, PAIRS, strict=True):
        tag = "2F" if model == "poro2" else "3F"
        m = make_model(problem(model))
        rec = Reconstruction(m)
        V0 = dolfinx.fem.functionspace(m.domain.mesh, ("DG", 0))
        v, n = ufl.TestFunction(V0), m.measures.n
        names = ("raw", "RT")
        times, total, worst = [], {k: [] for k in names}, {k: [] for k in names}
        for t in m.steps():
            u_rt = rec.solve()  # solver units, like the balance
            u_raw = m.step_velocities()["avg"]
            through = per_cell(  # int_{dK} |u.n| of the RT velocity
                abs(ufl.inner(u_rt("+"), n("+"))) * (v("+") + v("-")) * ufl.dS
                + abs(ufl.inner(u_rt, n)) * v * ufl.ds
            )
            flow = comm.allreduce(through.sum(), op=MPI.SUM)
            times.append(t)
            for name, u in (("raw", u_raw), ("RT", u_rt)):
                e = np.abs(imbalance(m, u, v, n))
                total[name].append(comm.allreduce(e.sum(), op=MPI.SUM) / flow)
                worst[name].append(comm.allreduce(np.max(e / through), op=MPI.MAX))
        if not root:
            continue
        print(f"\n{tag}: {m.system.ndofs} dofs, {len(times)} steps; worst over the cycle")
        # raw drawn thick: visible under RT where they coincide (3F)
        for name, color, lw in zip(names, ("tab:red", "tab:blue"), (4, 1.5), strict=True):
            print(f"  {name:3s}  total {max(total[name]):.2e}   worst cell {max(worst[name]):.2e}")
            kw = {"color": color, "lw": lw, "alpha": 0.5 if name == "raw" else 1}
            ax.semilogy(times, total[name], "-", label=f"{name}: total", **kw)
            ax.semilogy(times, worst[name], "--", label=f"{name}: worst cell", **kw)
        ax.set(title=tag, xlabel="t [s]", ylabel="imbalance / flow through the cells")
        ax.legend(fontsize=8)
    if root:
        fig.suptitle(
            "Brain annulus, cvr22, vascular source: fluid created or lost cell by cell", fontsize=10
        )
        fig.tight_layout()
        fig.savefig(f"{HERE}/figures/balance.png", dpi=120)
