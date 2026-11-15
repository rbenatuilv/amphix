"""The roller condition of poro2 (d.n = 0, tangential traction free): static uniaxial compression
of a rectangle, rollers on top and bottom. The exact solution d_y = 0, d_x linear with strain
-s / (lambda + 2 mu) is in the space (deg 2), so it is reproduced to round-off. A glued wall on top
and bottom (displacement: [0, 0]) holds d_x there and must fail the same check."""

import dolfinx
import numpy as np
import pytest
import ufl
from mpi4py import MPI

import amphix.physics.poro  # noqa: F401 (registers poro2, poro3)
from amphix.config import from_dict
from amphix.core.model import make_model

S, LAM, MU, X1 = 100.0, 2000.0, 1000.0, 2.0  # load [Pa], Lame [Pa], right end [m]
STRAIN = -S / (LAM + 2 * MU)


def compress(wall, model="poro2"):
    """Run to the steady state; return the model (load constant, p drained on left and right)."""
    tree = {
        "model": model,
        "material": {"lambda": LAM, "mu": MU, "alpha": 1.0, "c0": 1.0, "K": 1.0, "muF": 1.0},
        "mesh": {"shape": "rectangle", "corners": [[0, 0], [X1, 1]], "n": [4, 2]},
        "scales": {"L0": 1.0, "T0": 1.0, "P0": 100.0},
        "discretization": {"deg": 2},
        "time": {"T": 3e12, "dt": 1e12},
        "boundaries": {
            "left": {"normal_traction": -S, "pressure": 0},
            "right": {"displacement": [0, 0], "pressure": 0},
            "top": {**wall, "flux": 0},
            "bottom": {**wall, "flux": 0},
        },
    }
    if model == "poro2":
        tree["time"]["theta"] = 1.0
    m = make_model(from_dict(tree, "roller"))
    for _ in m.steps():
        pass
    return m


def errors(m):
    """Max over the mesh of |d_x - exact|, |d_y|, |p|, and the L2 error of d_x on the top."""
    d, p = m.quantity("d"), m.quantity("p")
    V = dolfinx.fem.functionspace(m.domain.mesh, ("DG", 2, (2,)))
    di = dolfinx.fem.Function(V)
    di.interpolate(d)  # same space: the dofs at the DG nodes
    nodes = V.tabulate_dof_coordinates()
    vals = di.x.array.reshape(-1, 2)
    ex = STRAIN * (m.scales.L0 * nodes[:, 0] - X1)
    xs = ufl.SpatialCoordinate(m.domain.mesh) * m.scales.L0
    top = dolfinx.fem.form((d[0] - STRAIN * (xs[0] - X1)) ** 2 * m.measures.ds("top"))
    comm = m.domain.mesh.comm
    top_err = np.sqrt(comm.allreduce(dolfinx.fem.assemble_scalar(top), op=MPI.SUM))
    return (
        comm.allreduce(np.abs(vals[:, 0] - ex).max(), op=MPI.MAX),
        comm.allreduce(np.abs(vals[:, 1]).max(), op=MPI.MAX),
        comm.allreduce(np.abs(p.x.array).max(initial=0.0), op=MPI.MAX),
        top_err,
    )


def test_roller_reproduces_uniaxial_compression():
    dx_err, dy, p, top = errors(compress({"roller": 0}))
    scale = abs(STRAIN) * X1  # the largest displacement
    assert p < 1e-10 * S
    assert dx_err < 1e-10 * scale
    assert dy < 1e-10 * scale
    assert top < 1e-10 * scale


def test_a_glued_wall_fails_the_same_check():
    _, _, _, top = errors(compress({"displacement": [0, 0]}))
    assert top > 1e-3 * abs(STRAIN) * X1


def test_roller_only_zero():
    with pytest.raises(ValueError, match="only roller: 0"):
        compress({"roller": 1e-3})


def test_roller_rejected_by_poro3():
    with pytest.raises(ValueError, match="only in poro2"):
        compress({"roller": 0}, model="poro3")
