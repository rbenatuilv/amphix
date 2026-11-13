"""amphix run (cli.py) and the public API (__init__.py): an end-to-end run of a YAML file that
matches a step-by-step run, the command-line errors, and the re-exported names."""

import importlib.metadata

import numpy as np
import pytest
import yaml
from cases import annulus, cell_values
from test_twofield import run

import amphix
from amphix.cli import main
from amphix.config import from_dict

PUBLIC = [
    "MODELS",
    "ConfigError",
    "Model",
    "Observer",
    "Problem",
    "Reconstruction",
    "Recorder",
    "State",
    "System",
    "ThreeField",
    "TwoField",
    "brain",
    "build_domain",
    "load",
    "load_fields",
    "load_results",
    "make_model",
    "run",
]


def write(path, tree):
    path.write_text(yaml.safe_dump(tree, sort_keys=False))
    return path


def test_end_to_end_from_yaml(tmp_path, capsys):
    """`amphix run` of an annulus YAML file, with --set and --name, gives the boundary flows at
    every step and p at every snapshot of the same problem run step by step in Python."""
    tree = annulus("poro2", h=1e-3)
    path = write(tmp_path / "annulus.yaml", tree)
    argv = ["run", str(path), "--name", "e2e", "--set", "output.fields=[p]"]
    argv += ["--set", "output.snapshots={from: 0.01, to: 0.05}"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "e2e: poro2" in out and "5 steps" in out and "saved" in out
    r = amphix.load_results(tmp_path / "results" / "e2e.npz")
    assert r["problem"]["output"]["name"] == "e2e"
    _, ref = run(from_dict(tree))
    assert np.allclose(r["t"][1:], ref["t"], rtol=0, atol=1e-12)  # the Recorder also records t0
    fields = amphix.load_fields(tmp_path / "results" / "e2e_fields.npz")
    assert np.allclose(fields.snap_t, ref["t"], rtol=0, atol=1e-12)
    for b in ("ventricle", "pial"):
        assert np.allclose(r[f"bflux_{b}"][1:], ref[f"bflux_{b}"], rtol=1e-12, atol=0), b
    p = np.array([cell_values(fields.function("p", i)) for i in range(5)])
    assert np.abs(p - ref["field_p"]).max() < 1e-12 * np.abs(ref["field_p"]).max()


def test_run_from_a_script(tmp_path):
    """amphix.run(problem) is what the command line does; a preset and a waveform need no import
    of amphix.brain any more."""
    tree = {
        "model": "poro2",
        "material": {"preset": "cvr22"},
        "mesh": {"shape": "rectangle", "corners": [[0, 0], [1e-2, 1e-2]], "n": [2, 2]},
        "scales": {"L0": 1e-2, "P0": 1500.0},
        "boundaries": {
            "left": {"displacement": [0, 0], "pressure": {"waveform": "cardiac_pulse"}},
            "right": {"displacement": [0, 0], "pressure": 0},
            "bottom": {"traction": [0, 0], "flux": 0},
            "top": {"traction": [0, 0], "flux": 0},
        },
        "sources": {"fluid_source": {"waveform": "vascular_option1"}},
        "discretization": {"deg": 1},
        "time": {"T": 0.2, "dt": 0.1},
        "output": {"probes": {"c": [5e-3, 5e-3]}},
    }
    problem = amphix.load(write(tmp_path / "square.yaml", tree))
    path = amphix.run(problem, verbose=False)
    assert path == tmp_path / "results" / "square.npz"
    r = amphix.load_results(path)
    assert r["probe_p"].shape == (3, 1) and np.isfinite(r["probe_p"]).all()


def test_command_line_errors(tmp_path, capsys):
    path = write(tmp_path / "bad.yaml", {"model": "heat"})
    assert main(["run", str(path)]) == 2
    assert "model: unknown model 'heat'" in capsys.readouterr().err
    assert main(["run", str(path), "--set", "time.dt"]) == 2
    assert "--set: expected KEY=VALUE" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main([])  # a command is required


def test_public_api():
    assert sorted(amphix.__all__) == sorted(PUBLIC)
    for name in PUBLIC:
        assert hasattr(amphix, name), name
    assert set(amphix.MODELS) >= {"poro2", "poro3"} and "cvr22" in amphix.brain.PRESETS
    scripts = importlib.metadata.entry_points(group="console_scripts", name="amphix")
    assert [s.value for s in scripts] == ["amphix.cli:main"]
