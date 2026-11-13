"""io.Recorder: the outputs (cflux in probes, fields, ParaView and cbflux; bflux independent of
the probes; a 3D run with slices) and the checks of the output section."""

import dolfinx
import numpy as np
import pytest
import ufl
from test_hdiv import MATERIAL, square

import amphix.physics.poro  # noqa: F401
from amphix.config import from_dict
from amphix.core.model import make_model
from amphix.io.fields import load_fields
from amphix.io.recorder import Recorder
from amphix.io.results import load_results


def with_output(tmp_path, name, **output):
    tree = {**square("poro2", 1, 1.0).raw, "output": {"dir": str(tmp_path), "name": name, **output}}
    return from_dict(tree, name, tmp_path)


def test_cflux_output(tmp_path):
    """cflux in probes, fields (exact DG copy), ParaView and cbflux; the average of the fields."""
    p = with_output(
        tmp_path,
        "cf",
        series=["p", "cflux"],
        probes={"c": [0.5, 0.5]},
        fields=["p", "cflux"],
        snapshots={"from": 0.05, "to": 0.25},
        paraview=2,
    )
    model = make_model(p)
    rec = Recorder(model)
    model.run([rec])
    r = load_results(rec.save())
    assert r["probe_cflux"].shape == (6, 1, 2) and r["problem"] == p.raw
    # the right boundary is impermeable: the conservative flow through it is zero, the raw one not
    assert np.abs(r["cbflux_right"]).max() < 1e-12 * np.abs(r["cbflux_left"]).max()
    assert np.abs(r["bflux_right"]).max() > 1e-6 * np.abs(r["bflux_left"]).max()
    # the bottom carries the given flux 0.3 sin(2t) (backward Euler: its value at the step end)
    assert np.allclose(r["cbflux_bottom"][1:], 0.3 * np.sin(2 * r["t"][1:]), rtol=1e-10)
    fl = load_fields(tmp_path / "cf_fields.npz")
    q = fl.function("cflux", 2)
    n = ufl.FacetNormal(fl.domain.mesh)
    jump = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.jump(q, n) ** 2 * ufl.dS))
    scale = dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(ufl.inner(ufl.avg(q), n("+")) ** 2 * ufl.dS)
    )
    assert jump < 1e-24 * scale
    mean = np.mean([fl.function("cflux", i).x.array for i in (1, 2, 3)], axis=0)
    assert np.allclose(fl.average("cflux", 0.05, 0.2).x.array, mean)
    assert (tmp_path / "cf" / "cflux.bp").exists() and len(r["snap_t"]) == 5 and len(r["jump"]) == 5


def test_bflux_does_not_depend_on_probes(tmp_path):
    """bflux is the flux of the current step whether or not something else asked for the flux."""
    runs = {}
    for name, out in (("none", {}), ("probed", {"probes": {"c": [0.5, 0.5]}, "series": ["flux"]})):
        model = make_model(with_output(tmp_path, name, **out))
        rec = Recorder(model)
        model.run([rec])
        runs[name] = load_results(rec.save())
    keys = [k for k in runs["none"] if k.startswith("bflux_")]
    assert keys and all(
        np.allclose(runs["none"][k], runs["probed"][k], rtol=1e-12, atol=0) for k in keys
    )
    assert np.abs(runs["none"]["bflux_left"][1:]).min() > 0


def test_3d_run_with_slices(tmp_path):
    """A box, probes, a slice, the fields file."""
    sides = ("left", "right", "front", "back", "bottom", "top")
    bnd = {s: {"pressure": 0, "displacement": [0, 0, 0]} for s in sides}
    bnd["left"] = {"pressure": "sin(pi*t) * (1 + y)", "normal_traction": "-sin(pi*t)"}
    p = from_dict(
        {
            "model": "poro2",
            "material": MATERIAL,
            "boundaries": bnd,
            "mesh": {"shape": "box", "corners": [[0, 0, 0], [2, 1, 1]], "n": [2, 1, 1]},
            "scales": {"L0": 2.0, "T0": 0.5, "P0": 3.0},
            "discretization": {"deg": 1},
            "time": {"T": 0.2, "dt": 0.05, "theta": 0.5},
            "output": {
                "dir": str(tmp_path),
                "name": "box",
                "snapshots": [0.1, 0.2],
                "fields": True,
                "probes": {"c": [1.0, 0.5, 0.5]},
                "slices": {"mid": {"normal": "y", "at": 0.5, "n": 5}},
            },
        },
        "box",
        tmp_path,
    )
    model = make_model(p)
    rec = Recorder(model)
    model.run([rec])
    r = load_results(rec.save())
    assert r["slice_mid"].shape == (25, 3) and np.allclose(r["slice_mid"][:, 1], 0.5)
    assert r["slice_mid_p"].shape == (2, 25) and r["slice_mid_flux"].shape == (2, 25, 3)
    assert not np.isnan(r["slice_mid_p"]).any() and r["probe_p"].shape == (5, 1)
    assert load_fields(tmp_path / "box_fields.npz").names == ["d", "p", "flux"]  # fields: true


def test_output_checks(tmp_path):
    with pytest.raises(ValueError, match="`grid` is for 2D meshes"):
        sides = ("left", "right", "front", "back", "bottom", "top")
        p = from_dict(
            {
                "model": "poro2",
                "material": MATERIAL,
                "mesh": {"shape": "box", "corners": [[0, 0, 0], [1, 1, 1]], "n": [1, 1, 1]},
                "boundaries": {s: {"pressure": 0, "displacement": [0, 0, 0]} for s in sides},
                "time": {"T": 0.1, "dt": 0.05},
                "output": {"grid": 4},
            }
        )
        Recorder(make_model(p))
    rec = Recorder(make_model(with_output(tmp_path, "x")))
    with pytest.raises(KeyError, match="cflux: ask for it"):
        rec.quantity("cflux")
