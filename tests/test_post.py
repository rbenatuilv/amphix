"""post.integrals and post.sampling: closed forms, NaN outside, the normal jump of raw and
conservative fluxes, and two ranks equal to one (mpirun -n 2)."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import ufl
from parallel import serial_only
from post_cases import L0, RING, SQUARE, field, results
from test_hdiv import square

import amphix.physics.poro  # noqa: F401
from amphix.core.mesh import Measures, build
from amphix.core.model import make_model
from amphix.post.hdiv import Reconstruction
from amphix.post.integrals import BoundaryIntegrals, NormalJump, PerCell
from amphix.post.sampling import Points, grid_points, line_points, slice_points

MPIRUN = shutil.which("mpirun", path=str(Path(sys.executable).parent)) or shutil.which("mpirun")


def test_points_in_si_and_nan_outside():
    d = build(SQUARE, L0=L0)
    p = field(d, lambda x: 1 + 3 * x[0] - 2 * x[1] ** 2)
    pts = [[5e-3, 2.5e-3], [2e-2, 1e-2], [3e-2, 0.0]]  # inside, a corner, outside
    got = Points(d, pts)(p)
    want = [1 + 3 * x - 2 * y**2 for x, y in pts[:2]]
    assert np.allclose(got[:2], want, rtol=1e-12) and np.isnan(got[2])
    q = field(d, lambda x: ufl.as_vector([x[1], -x[0]]), (2,))
    assert Points(d, pts[:1])(q).shape == (1, 2)
    assert np.allclose(Points(d, pts[:1])(q), [[2.5e-3, -5e-3]])


def test_point_sets():
    line = line_points({"from": [0, 0], "to": [1e-2, 2e-2], "n": 5})
    assert line.shape == (5, 2) and np.allclose(line[-1], [1e-2, 2e-2])
    assert len(line_points({"from": [0, 0], "to": [1, 0]})) == 101
    ring = build(RING, L0=L0)
    grid = grid_points(ring, 5)
    assert grid.shape == (25, 2) and np.allclose(grid[0], [-1e-2 + 2e-3, -1e-2 + 2e-3])
    assert np.isnan(Points(ring, grid)(field(ring, lambda x: x[0]))[0])  # the corner: outside
    box = build({"shape": "box", "corners": [[0, 0, 0], [2, 1, 1]], "n": [2, 1, 1]})
    s = slice_points(box, {"normal": "y", "at": 0.5, "n": 5})
    assert s.shape == (25, 3) and np.allclose(s[:, 1], 0.5)
    with pytest.raises(ValueError, match="2D meshes only"):
        grid_points(box, 4)
    with pytest.raises(ValueError, match="3D meshes only"):
        slice_points(ring, {"normal": "z", "at": 0.0})


def test_integrals_closed_forms():
    d = build(SQUARE, L0=L0)
    m = Measures(d, 5)
    q = field(d, lambda x: ufl.as_vector([x[0], 0 * x[0]]), (2,))  # q.n = x on the right: 2e-2
    flows = BoundaryIntegrals(ufl.inner(q, m.n), d, m)()
    assert flows["right"] == pytest.approx(2e-2 * 1e-2) and flows["left"] == pytest.approx(
        0, abs=1e-18
    )
    cells = PerCell(d, {"one": lambda v: v * m.dx()})
    volume = d.mesh.comm.allreduce(cells.volume.sum())  # owned cells on each rank
    assert volume == pytest.approx(2.0) and np.allclose(cells()["one"], cells.volume)
    assert NormalJump(q, m)() < 1e-12  # continuous: no jump


def test_normal_jump_of_raw_and_conservative_fluxes():
    model = make_model(square("poro2", 1, 1.0))
    rec = Reconstruction(model)
    raw, cons = (
        NormalJump(model.quantity("flux"), model.measures),
        NormalJump(rec.u, model.measures),
    )
    for _ in model.steps():
        rec.solve()
        model.quantity("flux")
        assert cons() < 1e-10 < raw()


@serial_only("it compares itself with its own mpirun run")
@pytest.mark.skipif(MPIRUN is None, reason="no mpirun")
def test_two_ranks_equal_one():
    serial = results()
    script = Path(__file__).with_name("post_cases.py")
    run = subprocess.run(
        [MPIRUN, "-n", "2", sys.executable, str(script)],
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    parallel = json.loads(run.stdout.strip().splitlines()[-1])
    for key, value in serial.items():
        a, b = (
            np.asarray(
                parallel[key] if not isinstance(value, dict) else list(parallel[key].values())
            ),
            np.asarray(value if not isinstance(value, dict) else list(value.values())),
        )
        assert np.allclose(a, b, rtol=1e-12, atol=1e-18, equal_nan=True), key
