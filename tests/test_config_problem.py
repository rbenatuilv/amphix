"""config: the Problem, the shared sections, a physics registering its own, and the checks that
name the YAML path."""

import warnings

import pytest
import toy_physics

from amphix.config import ConfigError, ConfigWarning, Output, Time, from_dict, load
from amphix.core import registry

TREE = {
    "model": "toy_a",
    "description": "a toy problem",
    "scales": {"L0": 1e-2, "T0": 0.5, "P0": 10.0},
    "mesh": {"shape": "rectangle", "corners": [[0, 0], [1e-2, 1e-2]], "n": [4, 4]},
    "material": {"k": 2.0},
    "params": {"A": 3},
    "loads": {"g": "A * sin(t)"},
    "boundaries": {"left": {"temperature": "g", "fixed": [0, 0]}, "right": {"heat_flux": 0}},
    "sources": {"heat_source": "1e-3 * x"},
    "time": {"T": 0.1, "dt": 0.01, "theta": 0.5},
    "output": {"probes": {"c": [5e-3, 5e-3]}, "snapshots": {"from": 0.02, "to": 0.06, "every": 2}},
    "solver": {"mat_mumps_icntl_14": 100},
}


def problem(**changes):
    return from_dict({**TREE, **changes}, "toy", "/tmp/x")


def test_every_section():
    p = problem()
    assert p.model == "toy_a" and p.name == "toy" and str(p.root) == "/tmp/x"
    assert p.scales.L0 == 1e-2 and p.time == Time(T=0.1, dt=0.01, theta=0.5) and p.time.nsteps == 10
    assert (
        p.section("material").k == 2.0 and p.params == {"A": 3.0} and p.loads == {"g": "A * sin(t)"}
    )
    assert p.boundaries["left"]["temperature"] == "g" and p.solver == {"mat_mumps_icntl_14": 100}
    assert isinstance(p.output, Output) and p.output.times(p.time) == pytest.approx(
        [0.02, 0.04, 0.06]
    )
    with pytest.raises(KeyError, match="no section 'discretization'"):
        p.section("discretization")


def test_with_and_load(tmp_path):
    (tmp_path / "case.yaml").write_text(
        "model: toy_a\nmaterial: {k: 1}\n"
        "mesh: {shape: rectangle, corners: [[0, 0], [1, 1]], n: [2, 2]}\n"
    )
    (tmp_path / "run.yaml").write_text(
        "base: case.yaml\ntime: {T: 1, dt: 0.1}\noutput: {name: renamed}\n"
    )
    p = load(tmp_path / "run.yaml", {"time.dt": 0.05, "material.k": 4})
    assert p.time.nsteps == 20 and p.section("material").k == 4.0 and p.name == "renamed"
    assert p.root == tmp_path and "base" not in p.raw
    q = p.with_({"model": "toy_b", "time.dt": 0.5})
    assert q.model == "toy_b" and q.time.nsteps == 2 and p.time.nsteps == 20


def test_boundaries_may_come_from_code():
    """No `boundaries:` loads; completeness is checked when the model is built."""
    tree = {k: v for k, v in TREE.items() if k != "boundaries"}
    assert from_dict(tree).boundaries == {}


@pytest.mark.parametrize(
    "change, path, match",
    [
        ({"geometry": {}}, "geometry", "unknown top-level key"),
        ({"descripton": "typo"}, "descripton", "unknown top-level key"),
        ({"model": "heat"}, "model", "unknown model 'heat'"),
        ({"mesh": {"shape": "disc", "raduis": 1.0, "h": 0.1}}, "mesh", "unknown key"),
        ({"scales": {"L0": 0}}, "scales.L0", "must be positive"),
        ({"scales": {"l0": 1}}, "scales.l0", "unknown key"),
        ({"time": {"T": 0.1, "dt": 0.03}}, "time", "must be a multiple of dt"),
        ({"time": {"T": 0.1}}, "time", r"missing \['dt'\]"),
        ({"time": {"T": 0.1, "dt": -0.01}}, "time.dt", "must be positive"),
        ({"time": {"T": 0.1, "dt": 0.01, "startup_steps": 1.5}}, "time.startup_steps", "integer"),
        ({"params": {"t": 1.0}}, "params.t", "hide a built-in"),
        ({"params": {"e": 1.0}}, "params.e", "hide a built-in"),
        ({"loads": {"r": "x"}}, "loads.r", "hide a built-in"),
        ({"loads": {"A": "1"}}, "loads.A", "is also a param"),
        ({"params": {"A": "3"}}, "params.A", "must be a number"),
        ({"boundaries": {"left": {"temperatur": 0}}}, "boundaries.left.temperatur", "unknown key"),
        (
            {"boundaries": {"left": {"temperature": 0, "heat_flux": 1}}},
            "boundaries.left",
            r"\['heat_flux', 'temperature'\] are all thermal conditions",
        ),
        (
            {"boundaries": {"left": {"temperature": {"a": 1}}}},
            "boundaries.left.temperature",
            "waveform",
        ),
        (
            {"boundaries": {"left": {"fixed": [0, None]}}},
            r"boundaries.left.fixed\[1\]",
            "must be a number",
        ),
        ({"sources": {"heat_sorce": 1}}, "sources.heat_sorce", "unknown key"),
        ({"material": {"k": -1}}, "material.k", "must be positive"),
        ({"material": {}}, "material", r"missing \['k'\]"),
        ({"output": {"probe": {}}}, "output.probe", "unknown key"),
        (
            {"output": {"snapshots": {"from": 0.1, "until": 0.2}}},
            "output.snapshots.until",
            "unknown key",
        ),
        ({"solver": {"ksp_type": ["gmres"]}}, "solver.ksp_type", "number or a string"),
    ],
)
def test_errors_name_the_path(change, path, match):
    with pytest.raises(ConfigError, match=rf"^{path}: .*{match}"):
        problem(**change)


def test_missing_required_sections():
    for key in ("mesh", "time", "model"):
        with pytest.raises(ConfigError, match=rf"^{key}: missing"):
            from_dict({k: v for k, v in TREE.items() if k != key})


def test_keys_the_model_does_not_read_warn():
    with pytest.warns(ConfigWarning, match=r"time: \['theta'\] not read by toy_b"):
        problem(model="toy_b")
    with pytest.warns(ConfigWarning, match=r"material: \['extra'\] not read by toy_a"):
        problem(material={"k": 1.0, "extra": 2.0})
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # read by toy_b: no warning
        problem(model="toy_b", material={"k": 1.0, "extra": 2.0}, time={"T": 0.1, "dt": 0.01})


def test_registry():
    assert registry.get("toy_a") is toy_physics.ToyA
    registry.register(toy_physics.ToyA)  # the same class again: fine

    class Other(toy_physics.ToyA):
        pass

    with pytest.raises(ValueError, match="already registered by ToyA"):
        registry.register(Other)
    with pytest.raises(KeyError, match="unknown model 'nope'"):
        registry.get("nope")
