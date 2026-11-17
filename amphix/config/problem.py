"""A problem: one YAML file (with its bases and overrides) -> one frozen, validated `Problem`.

    problem = load("annulus_poro2_pulse.yaml", {"time.dt": 0.005})
    problem.time.dt, problem.section("material").K, problem.with_({"mesh.h": 2.5e-4})

Everything is in SI. The top level holds `model` (required: a registered model name), the shared
sections (config/sections.py), the model's own sections (e.g. material, discretization) and an
optional free `description`. Any other top-level key is an error.
"""

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from amphix.config import loader, sections
from amphix.config.validate import ConfigError
from amphix.core import registry
from amphix.core.units import Scales

SHARED = (
    "scales",
    "mesh",
    "time",
    "output",
    "solver",
    "params",
    "loads",
    "boundaries",
    "sources",
    "initial",
)


@dataclass(frozen=True)
class Problem:
    """A validated problem description, in SI. Sections a model adds are in `sections`."""

    name: str
    root: Path
    model: str
    scales: Scales
    mesh: dict
    time: sections.Time
    output: sections.Output = field(default_factory=sections.Output)
    solver: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    loads: dict = field(default_factory=dict)
    boundaries: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)
    initial: sections.Initial | None = None  # None: start at rest
    sections: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict, repr=False, compare=False)

    def section(self, name: str) -> Any:
        """Return a section the model registered (e.g. "material")."""
        if name not in self.sections:
            raise KeyError(f"{self.model} has no section {name!r}; it has {sorted(self.sections)}")
        return self.sections[name]

    def with_(self, overrides: dict[str, Any]) -> "Problem":
        """Return a copy with dotted-key overrides, validated again."""
        return from_dict(loader.override(self.raw, overrides), self.name, self.root)


def _model(tree: dict) -> Any:
    if "model" not in tree:
        raise ConfigError("model", f"missing; one of {sorted(registry.MODELS)}")
    name = tree["model"]
    try:
        return registry.get(name)
    except KeyError:
        raise ConfigError(
            "model", f"unknown model {name!r}; registered: {sorted(registry.MODELS)}"
        ) from None


def _check_top_level(tree: dict, model: Any) -> None:
    allowed = {"model", "description", *SHARED, *model.sections}
    for key in tree:
        if key not in allowed:
            raise ConfigError(str(key), f"unknown top-level key; allowed {sorted(allowed)}")


def from_dict(tree: dict, name: str = "problem", root: str | Path = ".") -> Problem:
    """Return the Problem of a raw tree (bases already merged): every section validated."""
    model = _model(tree)
    _check_top_level(tree, model)
    for key in ("mesh", "time"):
        if key not in tree:
            raise ConfigError(key, "missing (required)")
    params, loads = sections.names(tree.get("params", {}), tree.get("loads", {}))
    output = sections.Output.from_dict(tree.get("output", {}))
    own = {s: cls.from_dict(tree.get(s, {}), s, model.name) for s, cls in model.sections.items()}
    return Problem(
        name=output.name or name,
        root=Path(root),
        model=model.name,
        scales=sections.scales(tree.get("scales", {})),
        mesh=sections.mesh(tree["mesh"]),
        time=sections.Time.from_dict(tree["time"], model),
        output=output,
        solver=sections.solver(tree.get("solver", {})),
        params=params,
        loads=loads,
        boundaries=sections.boundaries(tree.get("boundaries", {}), model),
        sources=sections.sources(tree.get("sources", {}), model),
        initial=sections.initial(tree.get("initial"), model, Path(root)),
        sections=own,
        raw=copy.deepcopy(tree),
    )


def load(path: str | Path, overrides: dict[str, Any] | None = None) -> Problem:
    """Read a problem file (with its `base:` chain), apply dotted-key overrides, validate."""
    tree, name, root = loader.read(path, overrides)
    return from_dict(tree, name, root)
