"""config.loader: exponent numbers, base: chains and overrides."""

from pathlib import Path

import pytest

from amphix.config.loader import merge, override, parse_set, read, read_tree, yaml_load
from amphix.config.validate import ConfigError


def test_exponent_numbers_are_numbers(tmp_path):
    """1e-3 is a number (PyYAML alone, following YAML 1.1, reads it as a string)."""
    ov = parse_set(
        [
            "mesh.h=1e-3",
            "x.h=[2E-3, 1e-3]",
            "time.T=5e1",
            "output.name=e5",
            "loads.f=1e-3*t",
            "output.dir=run_1e-3",
        ]
    )
    assert ov["mesh.h"] == 1e-3 and ov["x.h"] == [2e-3, 1e-3] and ov["time.T"] == 50.0
    assert (
        ov["output.name"] == "e5" and ov["loads.f"] == "1e-3*t" and ov["output.dir"] == "run_1e-3"
    )
    assert yaml_load("a: 5e-4\nb: 1.0e-6\nc: -2E+3\nd: '1e-3'") == {
        "a": 5e-4,
        "b": 1e-6,
        "c": -2000.0,
        "d": "1e-3",
    }


@pytest.mark.parametrize("item", ["time.dt", "=3"])
def test_bad_set_items(item):
    with pytest.raises(ConfigError, match=r"^--set: expected KEY=VALUE"):
        parse_set([item])


def test_base_chain(tmp_path):
    """study -> experiment -> case: each base relative to its own file, later files win."""
    for d in ("cases", "app", "study"):
        (tmp_path / d).mkdir()
    (tmp_path / "cases" / "case.yaml").write_text(
        "model: poro2\nmaterial: {mu: 1, lambda: 1, K: 1}\nmesh: {shape: gmsh, file: mesh.msh}\n"
    )
    (tmp_path / "app" / "exp.yaml").write_text(
        "base: ../cases/case.yaml\nmaterial: {K: 2}\ntime: {T: 1, dt: 0.1}\noutput: {dir: out}\n"
    )
    (tmp_path / "study" / "st.yaml").write_text("base: ../app/exp.yaml\nmaterial: {mu: 3}\n")
    tree, name, root = read(tmp_path / "app" / "exp.yaml")
    assert tree["material"] == {"mu": 1, "lambda": 1, "K": 2} and tree["output"] == {"dir": "out"}
    assert name == "exp" and root == tmp_path / "app" and "base" not in tree
    assert Path(tree["mesh"]["file"]) == (tmp_path / "cases" / "mesh.msh").resolve()
    tree = read_tree(tmp_path / "study" / "st.yaml")
    assert tree["material"] == {"mu": 3, "lambda": 1, "K": 2}


def test_base_errors(tmp_path):
    (tmp_path / "a.yaml").write_text("base: b.yaml\n")
    (tmp_path / "b.yaml").write_text("base: a.yaml\n")
    with pytest.raises(ConfigError, match=r"^base: loop a.yaml -> b.yaml -> a.yaml"):
        read(tmp_path / "a.yaml")
    (tmp_path / "c.yaml").write_text("base: missing.yaml\n")
    with pytest.raises(ConfigError, match=r"^base: no file .*missing.yaml \(base of c.yaml\)"):
        read(tmp_path / "c.yaml")
    (tmp_path / "d.yaml").write_text("- 1\n- 2\n")
    with pytest.raises(ConfigError, match="must be a section"):
        read(tmp_path / "d.yaml")
    (tmp_path / "e.yaml").write_text("a: [1, 2\n")
    with pytest.raises(ConfigError, match="not valid YAML"):
        read(tmp_path / "e.yaml")


def test_merge_and_override():
    base = {"material": {"K": 1, "mu": 2}, "output": {"fields": ["d", "p"]}}
    assert merge(base, {"material": {"K": 3}, "output": {"fields": ["p"]}}) == {
        "material": {"K": 3, "mu": 2},
        "output": {"fields": ["p"]},
    }  # lists are replaced
    tree = override(base, {"material.K": 5, "time.dt": 0.01, "output": {"dir": "x"}})
    assert tree == {"material": {"K": 5, "mu": 2}, "time": {"dt": 0.01}, "output": {"dir": "x"}}
    assert base["material"]["K"] == 1  # copies, not views
    with pytest.raises(ConfigError, match=r"^material\.K: is not a section"):
        override(base, {"material.K.x": 1})
