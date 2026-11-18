"""io.results, io.fields, io.paraview: round trips (values, times, tags), files independent of the
number of ranks, and ParaView output that leaves the run unchanged."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import dolfinx
import numpy as np
import pytest
from io_cases import measure, problem, run_and_write
from meshes import write_two_subdomains
from parallel import serial_only

from amphix.core.mesh import Measures
from amphix.io.fields import load_fields
from amphix.io.paraview import ParaView
from amphix.io.results import load_results, save_results
from amphix.post.integrals import NormalJump
from amphix.post.sampling import Points

MPIRUN = shutil.which("mpirun", path=str(Path(sys.executable).parent)) or shutil.which("mpirun")
CASES = Path(__file__).with_name("io_cases.py")


def mpirun(*args):
    run = subprocess.run(
        [MPIRUN, "-n", "2", sys.executable, str(CASES), *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    return run.stdout


def test_results_round_trip(tmp_path):
    p = problem()
    path = save_results(tmp_path / "r.npz", {"t": [0.0, 0.1], "probe_p": [[1.0], [2.0]]}, p)
    res = load_results(path)
    assert np.array_equal(res["t"], [0.0, 0.1]) and res["probe_p"].shape == (2, 1)
    assert res["problem"] == p.raw
    with pytest.raises(ValueError, match="reserved"):
        save_results(tmp_path / "x.npz", {"problem": 1}, p)


def test_reserved_key_refused_on_every_rank(tmp_path):
    """The check comes before the rank-0 write: a bad call fails on every rank, not on rank 0
    alone (the other ranks would go on, and wait for it)."""
    with pytest.raises(ValueError, match="reserved"):
        save_results(tmp_path / "x.npz", {"problem": 1}, problem())
    assert not (tmp_path / "x.npz").exists()


def test_fields_round_trip(tmp_path):
    """The saved fields are the model's."""
    model, rec = run_and_write(tmp_path / "f.npz")
    fl = load_fields(tmp_path / "f.npz")
    assert np.allclose(fl.snap_t, [0.05, 0.1, 0.15, 0.2]) and fl.names == [
        "d",
        "p",
        "flux",
        "velocity",
        "cflux",
    ]
    assert fl.problem == model.problem.raw
    pts = [[0.3, 0.2], [1.7, 0.9], [1.01, 0.5]]  # m (L0 = 2: the mesh is 4 m x 2 m)
    mine = {n: Points(model.domain, pts)(model.quantity(n)) for n in ("d", "p", "flux", "velocity")}
    mine["cflux"] = Points(model.domain, pts)(rec.compute())
    for name, values in mine.items():  # the last snapshot = the final state
        back = Points(fl.domain, pts)(fl.function(name, 3))
        assert np.allclose(back, values, rtol=1e-13, atol=1e-300), name
    q = fl.function("cflux", 2)  # stored exactly: no normal jump
    assert NormalJump(q, Measures(fl.domain))() < 1e-12
    mean = np.mean([fl.function("cflux", i).x.array for i in (1, 2, 3)], axis=0)
    assert np.allclose(fl.average("cflux", 0.05, 0.2).x.array, mean)
    with pytest.raises(ValueError, match="no saved time"):
        fl.average("p", 0.3, 0.4)


@pytest.mark.parametrize("mesh", ["annulus", "msh"])
def test_tags_round_trip(tmp_path, mesh):
    if mesh == "annulus":
        section = {
            "shape": "annulus",
            "radius": 2.0,
            "inner": {"center": [0.5, 0], "radius": 0.5},
            "h": 0.5,
            "names": {"inner": "cavity", "outer": "wall"},
        }
        moving = "cavity"
    else:
        section, moving = (
            {"shape": "gmsh", "file": str(write_two_subdomains(tmp_path / "two.msh"))},
            "skull",
        )
    model, _ = run_and_write(tmp_path / "f.npz", section, moving)
    fl = load_fields(tmp_path / "f.npz")
    a, b = model.domain, fl.domain
    assert b.facet_names == a.facet_names and b.cell_names == a.cell_names and b.L0 == 1.0

    def integrals(d):  # in SI
        m, one = Measures(d), dolfinx.fem.Constant(d.mesh, 1.0)
        out = {
            f"ds_{n}": d.L0 * dolfinx.fem.assemble_scalar(dolfinx.fem.form(one * m.ds(n)))
            for n in d.facet_names
        }
        out |= {
            f"dx_{n}": d.L0**2 * dolfinx.fem.assemble_scalar(dolfinx.fem.form(one * m.dx(n)))
            for n in d.cell_names
        }
        return out

    want, got = integrals(a), integrals(b)
    assert want.keys() == got.keys() and all(
        np.isclose(got[k], v, rtol=1e-12) for k, v in want.items()
    )


@serial_only("it compares itself with its own mpirun run")
@pytest.mark.skipif(MPIRUN is None, reason="no mpirun")
def test_files_do_not_depend_on_the_ranks(tmp_path):
    """Written on two ranks = written on one; read on two ranks = read on one (tags included)."""
    msh = str(write_two_subdomains(tmp_path / "two.msh"))
    run_and_write(tmp_path / "one.npz", {"shape": "gmsh", "file": msh}, "skull")
    mpirun("write", str(tmp_path / "two.npz"), msh)
    with np.load(tmp_path / "one.npz") as a, np.load(tmp_path / "two.npz") as b:
        for key in ("x", "cells", "cell_values", "facets", "facet_values", "d", "p", "flux"):
            assert np.allclose(a[key], b[key], rtol=1e-12, atol=1e-14), key
    serial = measure(tmp_path / "one.npz")
    parallel = json.loads(mpirun("measure", str(tmp_path / "one.npz")).strip().splitlines()[-1])
    assert parallel["names"] == serial["names"]
    for key, value in serial.items():
        if key != "names":
            assert parallel[key] == pytest.approx(value, rel=1e-12, abs=1e-15), key


def test_paraview_leaves_the_run_unchanged(tmp_path):
    p = problem()
    model, _ = run_and_write(tmp_path / "f.npz")
    x0 = model.domain.mesh.geometry.x.copy()
    flux0 = model.quantity("flux").x.array.copy()
    pv = ParaView(
        model.domain,
        {n: (lambda n=n: model.quantity(n)) for n in ("d", "p", "flux")},
        tmp_path / "pv",
    )
    pv.write(model.t)
    pv.close()
    assert (tmp_path / "pv" / "flux.bp").exists() and (tmp_path / "pv" / "d.bp").exists()
    assert np.array_equal(model.domain.mesh.geometry.x, x0)  # bit for bit
    assert np.array_equal(model.quantity("flux").x.array, flux0)
    assert p.scales.L0 == 2.0
