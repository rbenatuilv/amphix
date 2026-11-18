"""The RT reconstruction (post.hdiv) on the brain annulus, a few steps: run serially (imported by
the tests) and under mpirun (python tests/hdiv_parallel.py MODEL prints JSON), to compare the two.

Per step: the raw flux and cflux at fixed points, the conservative flow through each wall
(cbflux), and two worst cell values, each over the flow through the cell, over all owned cells
and over the owned cells next to another rank's cells:
    imbalance   gained_K + int_dK u.n of the RT velocity (studies/mass_balance/balance.py): it
                closes to the solver's own cell balance
    rt          int_dK u.n - int_dK Fhat: the RT flow against the solver's flux (round-off)"""

import json
import sys

import numpy as np
import ufl
from mpi4py import MPI

import amphix  # noqa: F401 (registers poro2 and poro3, and the brain presets and waveforms)
from amphix.brain import geometries
from amphix.config import from_dict
from amphix.core.model import make_model
from amphix.post.hdiv import Reconstruction
from amphix.post.integrals import BoundaryIntegrals, PerCell
from amphix.post.sampling import Points

PAIRS = {"poro2": (2, 1), "poro3": (2, 1)}  # 2F P2/P1, 3F P2/P2/P1
SOURCE = {  # the vascular source of the mass balance study
    "poro2": {"fluid_source": {"waveform": "vascular_option2"}},
    "poro3": {
        "mass_source": "3.53e-3 * ((1 - cos(2*pi*t)) / (2*pi) + 0.5 * (1 - cos(4*pi*t)) / (4*pi))"
    },
}
POINTS = [[-8e-3, 0.0], [-5e-3, 2e-3], [0.0, 6e-3], [7e-3, -3e-3], [3e-3, 4e-3], [0.0, -9e-3]]


def problem(model):
    deg, deg_p = PAIRS[model]
    return from_dict(
        {
            "model": model,
            "scales": {"L0": 1e-2, "T0": 1.0, "P0": 1500.0},
            "material": {"preset": "cvr22"},
            "mesh": geometries.annulus(h=2e-3),
            "boundaries": {
                "pial": {"displacement": [0, 0], "pressure": "100 * sin(2*pi*t)"},
                "ventricle": {"normal_traction": 0, "pressure": 0},
            },
            "sources": SOURCE[model],
            "discretization": {"deg": deg, "deg_p": deg_p},
            "time": {"T": 0.08, "dt": 0.02} | ({"theta": 0.5} if model == "poro2" else {}),
        },
        "hdiv_parallel",
    )


def next_to_other_ranks(mesh):
    """Mask of the owned cells that share a facet with a ghost cell (another rank's)."""
    tdim = mesh.topology.dim
    nloc = mesh.topology.index_map(tdim).size_local
    mesh.topology.create_connectivity(tdim - 1, tdim)
    f2c = mesh.topology.connectivity(tdim - 1, tdim)
    mask = np.zeros(nloc, dtype=bool)
    for f in range(mesh.topology.index_map(tdim - 1).size_local):
        cells = f2c.links(f)
        if len(cells) == 2 and max(cells) >= nloc:
            mask[min(cells)] = True
    return mask


def run(model):
    """Return {flux, cflux, cbflux_<wall>, imbalance(_edge), rt(_edge), boundary_cells}."""
    m = make_model(problem(model))
    rec = Reconstruction(m)
    mesh, n = m.domain.mesh, m.measures.n
    points = Points(m.domain, POINTS)
    flows = BoundaryIntegrals(ufl.inner(rec.u_si, n), m.domain, m.measures)
    u = rec.u  # solver units, like the balance
    interior, bnd = m.numerical_flux()
    cells = PerCell(
        m.domain,
        {
            "gained": m.fluid_gained,
            "out": lambda v: (
                ufl.inner(ufl.avg(u), n("+")) * ufl.jump(v) * ufl.dS + ufl.inner(u, n) * v * ufl.ds
            ),
            "solver": lambda v: (
                interior * ufl.jump(v) * m.measures.dS + sum(b * v * ds for ds, b in bnd)
            ),
            "through": lambda v: (
                abs(ufl.inner(u("+"), n("+"))) * (v("+") + v("-")) * ufl.dS
                + abs(ufl.inner(u, n)) * v * ufl.ds
            ),
        },
    )
    edge = next_to_other_ranks(mesh)
    worst = ("imbalance", "imbalance_edge", "rt", "rt_edge")
    out: dict = {"flux": [], "cflux": []} | {k: [] for k in worst}
    out |= {f"cbflux_{b}": [] for b in flows.forms}
    for _ in m.steps():
        rec.compute()
        out["flux"].append(points(m.quantity("flux")).tolist())  # the solver's, raw
        out["cflux"].append(points(rec.u_si).tolist())
        for b, value in flows().items():
            out[f"cbflux_{b}"].append(value)
        c = cells()
        for key, diff in (("imbalance", c["gained"] + c["out"]), ("rt", c["out"] - c["solver"])):
            e = np.abs(diff) / c["through"]
            out[key].append(mesh.comm.allreduce(e.max(), op=MPI.MAX))
            out[f"{key}_edge"].append(mesh.comm.allreduce(e[edge].max(initial=0.0), op=MPI.MAX))
    out["boundary_cells"] = mesh.comm.allreduce(int(edge.sum()), op=MPI.SUM)
    return out


if __name__ == "__main__":
    result = run(sys.argv[1])
    if MPI.COMM_WORLD.rank == 0:
        print(json.dumps(result))
