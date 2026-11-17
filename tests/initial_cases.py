"""The cases of the initial-state tests, run serially (imported) and under mpirun:

python tests/initial_cases.py restart IN.npz      restart from a steady fields file, as JSON
"""

import json
import sys
from pathlib import Path

import numpy as np
import ufl
from mpi4py import MPI

import amphix.physics.poro  # noqa: F401
from amphix.config import from_dict
from amphix.core.model import make_model
from amphix.io.fields import FieldWriter
from amphix.io.initial import apply_initial
from amphix.post.integrals import BoundaryIntegrals

MATERIAL = {"lambda": 1.0, "mu": 1.0, "alpha": 1.0, "c0": 1.0, "K": 1.0, "muF": 1.0, "rho": 1.0}
STEADY = {"T": 1e9, "dt": 1e9, "theta": 1}
CN = {"T": 0.03, "dt": 0.01, "theta": 0.5}


def problem(time, initial=None, n=(6, 3), root=".", output=None):
    """poro2 on a 2 x 1 rectangle, scales != 1: a pressure difference between left and right."""
    tree = {
        "model": "poro2",
        "material": MATERIAL,
        "mesh": {"shape": "rectangle", "corners": [[0, 0], [2, 1]], "n": list(n)},
        "scales": {"L0": 2.0, "T0": 0.5, "P0": 3.0},
        "discretization": {"deg": 1},
        "boundaries": {
            "left": {"pressure": 1.0, "displacement": [0, 0]},
            "right": {"pressure": 0, "displacement": [0, 0]},
            "bottom": {"flux": 0, "roller": 0},
            "top": {"flux": 0, "traction": [0, 0]},
        },
        "time": time,
    }
    if initial is not None:
        tree["initial"] = initial
    if output is not None:
        tree["output"] = output
    return from_dict(tree, "initial", root)


def flows(model):
    """The flow through every boundary now."""
    m = model.measures
    integrals = BoundaryIntegrals(ufl.inner(model.quantity("flux"), m.n), model.domain, m)
    return integrals()


def state(model):
    """The SI dofs of d and p on this rank."""
    return {f: model.field(f).x.array.copy() for f in ("d", "p")}


def steady(folder):
    """One backward-Euler step of 1e9 s, saving d and p; return the fields file and the flows."""
    path = Path(folder) / "steady_fields.npz"
    model = make_model(problem(STEADY))
    model.step()
    writer = FieldWriter(model.domain, {f: (lambda f=f: model.quantity(f)) for f in ("d", "p")})
    writer(model.t)
    writer.save(path, model.problem)
    return path, flows(model)


def restart(path):
    """Crank-Nicolson steps from the saved state: the largest change of d and p (relative to their
    size, over every step and rank) and the flows after the last step."""
    model = make_model(problem(CN, {"from": str(path)}))
    apply_initial(model)
    start, change = state(model), 0.0
    for _ in model.steps():
        for f, v in state(model).items():
            size = model.domain.mesh.comm.allreduce(np.abs(start[f]).max(initial=0), op=MPI.MAX)
            change = max(change, np.abs(v - start[f]).max(initial=0) / size)
    change = model.domain.mesh.comm.allreduce(change, op=MPI.MAX)
    return {"change": change, "flows": flows(model)}


if __name__ == "__main__":
    out = restart(sys.argv[2])
    if MPI.COMM_WORLD.rank == 0:
        print(json.dumps(out))
