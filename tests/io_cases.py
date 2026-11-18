"""The cases of the io tests, run serially (imported) and under mpirun:

python tests/io_cases.py write OUT.npz        a small poro2 run, every field at every step
python tests/io_cases.py measure IN.npz       integrals of a fields file, as JSON
"""

import json
import sys

import dolfinx
import ufl
from meshes import write_two_subdomains  # noqa: F401 (re-exported for the tests)
from mpi4py import MPI

import amphix.physics.poro  # noqa: F401
from amphix.config import from_dict
from amphix.core.mesh import build
from amphix.core.model import make_model
from amphix.io.fields import FieldWriter, load_fields
from amphix.post.hdiv import Reconstruction

MATERIAL = {"lambda": 1.0, "mu": 1.0, "alpha": 1.0, "c0": 1.0, "K": 1.0, "muF": 1.0, "rho": 1.0}
NAMES = ("d", "p", "flux", "velocity", "cflux")


def problem(mesh=None):
    """poro2 on a 2 x 1 rectangle (or another mesh), scales != 1, four steps."""
    mesh = mesh or {"shape": "rectangle", "corners": [[0, 0], [2, 1]], "n": [6, 3]}
    return from_dict(
        {
            "model": "poro2",
            "material": MATERIAL,
            "mesh": mesh,
            "scales": {"L0": 2.0, "T0": 0.5, "P0": 3.0},
            "discretization": {"deg": 1},
            "time": {"T": 0.2, "dt": 0.05, "theta": 0.5},
        },
        "io",
    )


def boundaries(names, moving):
    """Every boundary drained and fixed, one loaded."""
    out = {b: {"pressure": 0, "displacement": [0, 0]} for b in names}
    out[moving] = {"pressure": "sin(pi*t) * (1 + y)", "normal_traction": "-sin(pi*t)"}
    return out


def run_and_write(path, mesh=None, moving="left"):
    """Run, store every field after every step; return the model and its last SI values."""
    p = problem(mesh)
    domain = build(p.mesh, p.scales.L0, p.root)
    model = make_model(p, domain, boundary_data=boundaries(list(domain.facet_names), moving))
    rec = Reconstruction(model)
    sources = {n: (lambda n=n: model.quantity(n)) for n in NAMES[:4]}
    sources["cflux"] = rec.compute
    writer = FieldWriter(model.domain, sources)
    for t in model.steps():
        writer(t)
    writer.save(path, model.problem)
    return model, rec


def measure(path):
    """Integrals of the stored fields on the reloaded domain: total and per name."""
    fl = load_fields(path)
    d, out = fl.domain, {}
    ds = ufl.Measure("ds", domain=d.mesh, subdomain_data=d.facet_tags)
    dx = ufl.Measure("dx", domain=d.mesh, subdomain_data=d.cell_tags)

    def total(form):
        return d.mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(form)))

    p = fl.function("p", len(fl.snap_t) - 1)
    out["p"] = total(p * dx)
    out |= {f"p_ds_{n}": total(p * ds(t)) for n, t in d.facet_names.items()}
    out |= {
        f"one_dx_{n}": total(dolfinx.fem.Constant(d.mesh, 1.0) * dx(t))
        for n, t in d.cell_names.items()
    }
    out["names"] = sorted(d.facet_names) + sorted(d.cell_names)
    return out


if __name__ == "__main__":
    what, path = sys.argv[1], sys.argv[2]
    if what == "write":
        run_and_write(
            path,
            None if len(sys.argv) < 4 else {"shape": "gmsh", "file": sys.argv[3]},
            moving="left" if len(sys.argv) < 4 else "skull",
        )
    else:
        res = measure(path)
        if MPI.COMM_WORLD.rank == 0:
            print(json.dumps(res))
