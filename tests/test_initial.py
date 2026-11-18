"""The `initial` section: a run started from a saved steady state stays steady (on 1 and 2 ranks),
another mesh is refused, data matches set_initial, and the invalid sections name what is wrong."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from initial_cases import CN, STEADY, problem, restart, state, steady
from parallel import copy_file, serial_only, shared_dir, write_text

import amphix.physics.poro  # noqa: F401
from amphix.cli import run
from amphix.config import ConfigError, from_dict, load
from amphix.core.model import make_model
from amphix.io.fields import FieldWriter
from amphix.io.initial import apply_initial

MPIRUN = shutil.which("mpirun", path=str(Path(sys.executable).parent)) or shutil.which("mpirun")
CASES = Path(__file__).with_name("initial_cases.py")


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    """The steady fields file and the steady flows."""
    return steady(shared_dir(tmp_path_factory, "steady"))


def test_the_saved_state_is_put_back_exactly(saved):
    path, _ = saved
    first = make_model(problem(STEADY))
    first.step()
    model = make_model(problem(CN, {"from": str(path)}))
    apply_initial(model)
    for f, v in state(model).items():
        ref = first.field(f).x.array
        assert np.abs(v - ref).max() < 1e-14 * np.abs(ref).max(), f
    for f in ("d_t", "d_tt"):
        assert not model.field(f).x.array.any(), f


def test_a_steady_start_stays_steady(saved):
    path, flows = saved
    out = restart(path)
    assert out["change"] < 1e-9
    for name, q in flows.items():
        assert abs(out["flows"][name] - q) < 1e-9 * abs(flows["left"]), name


@serial_only("it compares itself with its own mpirun run")
@pytest.mark.slow
@pytest.mark.skipif(MPIRUN is None, reason="no mpirun")
def test_a_steady_start_on_two_ranks(saved):
    path, flows = saved
    run_ = subprocess.run(
        [MPIRUN, "-n", "2", sys.executable, str(CASES), "restart", str(path)],
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    out = json.loads(run_.stdout.strip().splitlines()[-1])
    assert out["change"] < 1e-9
    for name, q in flows.items():
        assert abs(out["flows"][name] - q) < 1e-9 * abs(flows["left"]), name


@pytest.mark.parametrize(
    ("n", "message"), [((4, 3), "cells"), ((6, 3), "vertex")], ids=["cells", "coordinates"]
)
def test_another_mesh_is_refused(saved, n, message):
    path, _ = saved
    p = problem(CN, {"from": str(path)}, n=n)
    if message == "vertex":  # same sizes, other coordinates
        p = p.with_({"mesh.corners": [[0, 0], [2, 1.5]]})
    with pytest.raises(ValueError, match=f"not the mesh of this run.*{message}"):
        apply_initial(make_model(p))


def test_data_is_set_initial():
    values = {"p": "0.5 * (1 - x / 2) + 0.1 * y", "d": ["1e-3 * x * y", 0], "d_t": [0, "1e-2"]}
    by_section = make_model(problem(CN, values))
    apply_initial(by_section)
    by_hand = make_model(problem(CN))
    by_hand.set_initial(**values)
    for m in (by_section, by_hand):
        m.step()
    for f in ("d", "p", "d_t", "d_tt"):
        a, b = by_section.field(f).x.array, by_hand.field(f).x.array
        assert np.abs(b).max() > 0 and np.array_equal(a, b), f


def test_a_model_does_not_step_without_its_initial_state(tmp_path):
    p = problem(CN, {"p": 0.5}, root=tmp_path).with_({"time.T": 0.01})
    with pytest.raises(RuntimeError, match="initial"):
        make_model(p).step()
    assert run(p, verbose=False).is_file()  # amphix.run applies it


def test_from_is_relative_to_the_yaml(saved, tmp_path):
    path, _ = saved
    copy_file(path, tmp_path / "cases" / "results" / "steady_fields.npz")
    write_text(
        tmp_path / "cases" / "case.yaml",
        "model: poro2\ninitial: {from: results/steady_fields.npz}\n",
    )
    tree = {k: v for k, v in problem(CN).raw.items() if k != "model"}
    write_text(tmp_path / "run.yaml", yaml.safe_dump({"base": "cases/case.yaml", **tree}))
    p = load(tmp_path / "run.yaml")
    assert p.initial.path == tmp_path / "cases" / "results" / "steady_fields.npz"
    assert p.initial.index == 0 and p.initial.time == STEADY["T"]


def test_invalid_sections(saved, tmp_path):
    path, _ = saved
    only_p = tmp_path / "only_p.npz"
    m = make_model(problem(STEADY))
    w = FieldWriter(m.domain, {"p": lambda: m.quantity("p")})
    w(0.0)
    w.save(only_p, m.problem)
    cases = {
        "both": ({"from": str(path), "p": 0}, "not both"),
        "unknown": ({"w": [0, 0]}, "initial.w: not state of poro2"),
        "time alone": ({"p": 0, "time": 1.0}, "initial.time: only with `from`"),
        "missing file": ({"from": "nowhere.npz"}, "no file"),
        "unknown time": ({"from": str(path), "time": 2.0}, "2.0 is not saved"),
        "missing field": ({"from": str(only_p)}, r"has no \['d'\]"),
        "bad data": ({"p": {"a": 1}}, "initial.p"),
    }
    for initial, message in cases.values():
        with pytest.raises(ConfigError, match=message):
            problem(CN, initial, root=tmp_path)


def test_poro3_does_not_restart(saved):
    path, _ = saved
    tree = dict(problem(CN).raw, model="poro3", initial={"from": str(path)})
    tree["time"] = {"T": 0.03, "dt": 0.01}
    with pytest.raises(NotImplementedError, match="poro3"):
        from_dict(tree)
