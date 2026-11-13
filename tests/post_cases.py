"""The cases of the post tests: run serially (imported) and under mpirun (python
tests/post_cases.py prints JSON), so that both can be compared."""

import json

import dolfinx
import numpy as np
import ufl
from mpi4py import MPI

from amphix.core.mesh import Measures, build
from amphix.post.integrals import BoundaryIntegrals, NormalJump, PerCell
from amphix.post.sampling import Points, grid_points, line_points

L0 = 1e-2
SQUARE = {"shape": "rectangle", "corners": [[0, 0], [2e-2, 1e-2]], "n": [8, 4]}
RING = {
    "shape": "annulus",
    "radius": 1e-2,
    "inner": {"center": [3e-3, 0], "radius": 2.5e-3},
    "h": 2e-3,
}


def field(domain, expr_of_x, shape=()):
    """A DG2 Function of an SI expression of the SI position x."""
    V = dolfinx.fem.functionspace(domain.mesh, ("DG", 2, shape))
    f = dolfinx.fem.Function(V)
    x = domain.L0 * ufl.SpatialCoordinate(domain.mesh)
    f.interpolate(dolfinx.fem.Expression(expr_of_x(x), V.element.interpolation_points))
    return f


def results():
    """Point values and integrals on a rectangle and an annulus (every rank gets the same)."""
    out = {}
    for name, section in (("square", SQUARE), ("ring", RING)):
        d = build(section, L0=L0)
        m = Measures(d, 5)
        p = field(d, lambda x: 1 + 3 * x[0] - 2 * x[1] ** 2)  # SI, quadratic: held by DG2
        q = field(d, lambda x: ufl.as_vector([x[1], -(x[0] ** 2)]), (2,))
        pts = np.vstack(
            [
                line_points({"from": [-1e-2, 0], "to": [1e-2, 0], "n": 11}),
                grid_points(d, 4),
                [[5e-3, 5e-3], [1e-2, 5e-3]],
            ]
        )  # also rank seams
        out[f"{name}_p"] = Points(d, pts)(p).tolist()
        out[f"{name}_q"] = Points(d, pts)(q).tolist()
        flows = BoundaryIntegrals(ufl.inner(q, m.n), d, m)()
        out[f"{name}_flows"] = flows
        out[f"{name}_jump"] = NormalJump(q, m)()
        cells = PerCell(d, {"p": lambda v, p=p, m=m: p * v * m.dx()})()
        out[f"{name}_cells"] = d.mesh.comm.allreduce(float(cells["p"].sum()), op=MPI.SUM)
    return out


if __name__ == "__main__":
    res = results()
    if MPI.COMM_WORLD.rank == 0:
        print(json.dumps(res))
