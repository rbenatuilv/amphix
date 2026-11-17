"""io.Recorder: nothing recorded unless asked for; each block records exactly its quantities at
exactly its steps; boundary flows as computed by hand; cflux outputs; a 3D run with slices."""

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
from amphix.post.hdiv import Reconstruction
from amphix.post.integrals import BoundaryIntegrals

META = {"problem", "ncells"}  # what every results file holds besides the records
LINE = {"from": [0.1, 0.5], "to": [0.9, 0.5], "n": 7}


def with_output(tmp_path, name, **output):
    tree = {**square("poro2", 1, 1.0).raw, "output": {"dir": str(tmp_path), "name": name, **output}}
    return from_dict(tree, name, tmp_path)


def record(problem):
    model = make_model(problem)
    rec = Recorder(model)
    model.run([rec])
    return rec, load_results(rec.save())


def test_nothing_requested(tmp_path):
    rec, r = record(with_output(tmp_path, "none"))
    assert set(r) - META == {"t"} and len(r["t"]) == 6
    assert rec.rec is None and rec.fields is None and rec.fields_bytes() == 0
    assert not (tmp_path / "none_fields.npz").exists()


@pytest.mark.parametrize(
    ("block", "value", "keys"),
    [
        (
            "probes",
            {"points": {"c": [0.5, 0.5]}, "quantities": ["d"]},
            {"probe_d", "probe_names", "probes"},
        ),
        (
            "lines",
            {"quantities": ["p", "cflux"], "at": [0.1, 0.2], "paths": {"mid": LINE}},
            {"lines_t", "line_mid", "line_mid_p", "line_mid_cflux"},
        ),
        (
            "grid",
            {"n": 3, "quantities": ["flux"], "at": [0.1, 0.2]},
            {"grid_t", "grid", "grid_flux"},
        ),
        ("jump", {"at": [0.1, 0.2]}, {"jump_t", "jump"}),
        ("fields", {"quantities": ["p"], "at": [0.1, 0.2]}, set()),
        ("paraview", {"quantities": ["p"], "at": [0.1, 0.2]}, set()),
    ],
)
def test_each_block_alone(tmp_path, block, value, keys):
    rec, r = record(with_output(tmp_path, block, **{block: value}))
    assert set(r) - META == {"t", *keys}
    assert (rec.rec is not None) == ("cflux" in value.get("quantities", []))
    for key in keys - {"probe_names", "probes", "line_mid", "grid"}:
        assert len(r[key]) == (6 if block == "probes" else 2), key
    if f"{block}_t" in keys:
        assert np.allclose(r[f"{block}_t"], [0.1, 0.2])
    assert (tmp_path / f"{block}_fields.npz").exists() == (block == "fields")
    if block == "fields":
        fl = load_fields(tmp_path / "fields_fields.npz")
        assert fl.names == ["p"] and np.allclose(fl.snap_t, [0.1, 0.2])
    assert (tmp_path / block / "p.bp").exists() == (block == "paraview")


@pytest.mark.slow
def test_boundary_flows(tmp_path):
    """raw and conservative flows: the boundary integrals of flux and cflux after every step."""
    model = make_model(with_output(tmp_path, "ref"))
    n, rec = model.measures.n, Reconstruction(model)
    raw = BoundaryIntegrals(ufl.inner(model.quantity("flux"), n), model.domain, model.measures)
    cons = BoundaryIntegrals(ufl.inner(rec.u_si, n), model.domain, model.measures)
    ref = {"bflux": [], "cbflux": []}
    for _ in model.steps():
        model.quantity("flux")
        rec.compute()
        ref["bflux"].append(raw())
        ref["cbflux"].append(cons())
    runs = {}
    for kinds in ("raw", "conservative", ["raw", "conservative"]):
        name = kinds if isinstance(kinds, str) else "both"
        runs[name] = record(with_output(tmp_path, name, boundary_flux=kinds))[1]
    keys = {"raw": {"bflux"}, "conservative": {"cbflux"}, "both": {"bflux", "cbflux"}}
    for name, r in runs.items():
        assert {k.split("_")[0] for k in set(r) - META - {"t"}} == keys[name]
        for key in keys[name]:
            for b in model.domain.facet_names:
                got, want = r[f"{key}_{b}"][1:], [step[b] for step in ref[key]]
                assert np.allclose(got, want, rtol=1e-12, atol=1e-14), (name, key, b)
    # the right boundary is impermeable: the conservative flow through it is zero, the raw one not
    r = runs["both"]
    assert np.abs(r["cbflux_right"]).max() < 1e-12 * np.abs(r["cbflux_left"]).max()
    assert np.abs(r["bflux_right"]).max() > 1e-6 * np.abs(r["bflux_left"]).max()
    # the bottom carries the given flux 0.3 sin(2t) (backward Euler: its value at the step end)
    assert np.allclose(r["cbflux_bottom"][1:], 0.3 * np.sin(2 * r["t"][1:]), rtol=1e-10)


def test_blocks_at_their_own_steps(tmp_path):
    """Lines every step, fields every other step, ParaView twice, the jump once: one run."""
    rec, r = record(
        with_output(
            tmp_path,
            "cf",
            boundary_flux="conservative",
            probes={"points": {"c": [0.5, 0.5]}, "quantities": ["p", "cflux"]},
            lines={"quantities": ["cflux"], "at": {}, "paths": {"mid": LINE}},
            fields={"quantities": ["p", "cflux"], "at": {"from": 0.05, "every": 2}},
            jump={"at": [0.25]},
            paraview={"quantities": ["cflux"], "at": {"every": 4}},
        )
    )
    assert r["probe_cflux"].shape == (6, 1, 2) and r["line_mid_cflux"].shape == (6, 7, 2)
    assert np.allclose(r["lines_t"], r["t"]) and np.allclose(r["jump_t"], [0.25])
    with np.load(tmp_path / "cf_fields.npz") as f:
        assert rec.fields_bytes() == f["p"].nbytes + f["cflux"].nbytes  # the estimate is exact
    fl = load_fields(tmp_path / "cf_fields.npz")
    assert np.allclose(fl.snap_t, [0.05, 0.15, 0.25])
    q = fl.function("cflux", 1)  # stored exactly: no normal jump
    n = ufl.FacetNormal(fl.domain.mesh)
    jump = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.jump(q, n) ** 2 * ufl.dS))
    scale = dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(ufl.inner(ufl.avg(q), n("+")) ** 2 * ufl.dS)
    )
    assert jump < 1e-24 * scale
    mean = np.mean([fl.function("cflux", i).x.array for i in (1, 2)], axis=0)
    assert np.allclose(fl.average("cflux", 0.05, 0.25).x.array, mean)
    assert (tmp_path / "cf" / "cflux.bp").exists()


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
                "fields": {"quantities": ["d", "p", "flux"], "at": [0.1, 0.2]},
                "probes": {"points": {"c": [1.0, 0.5, 0.5]}, "quantities": ["p"]},
                "slices": {
                    "quantities": ["p", "flux"],
                    "at": [0.1, 0.2],
                    "planes": {"mid": {"normal": "y", "at": 0.5, "n": 5}},
                },
            },
        },
        "box",
        tmp_path,
    )
    _, r = record(p)
    assert r["slice_mid"].shape == (25, 3) and np.allclose(r["slice_mid"][:, 1], 0.5)
    assert r["slice_mid_p"].shape == (2, 25) and r["slice_mid_flux"].shape == (2, 25, 3)
    assert not np.isnan(r["slice_mid_p"]).any() and r["probe_p"].shape == (5, 1)
    assert np.allclose(r["slices_t"], [0.1, 0.2])
    assert load_fields(tmp_path / "box_fields.npz").names == ["d", "p", "flux"]


def test_cflux_only_when_asked(tmp_path):
    rec = Recorder(make_model(with_output(tmp_path, "x")))
    with pytest.raises(KeyError, match="cflux: ask for it"):
        rec.quantity("cflux")
