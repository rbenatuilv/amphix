"""The sections every problem shares, and the shape checks of the model-dependent ones.

    scales      Scales(L0, T0, P0)               [m, s, Pa], default 1 (solve in SI numbers)
    mesh        dict, checked by core.mesh.check (sizes in SI; required)
    time        Time(T, dt, t0, theta, startup_steps, newmark_beta, newmark_gamma) (required)
    output      Output: what io records (positions in m, times in s)
    solver      {PETSc option: value}, added to the default direct solver
    params      {name: number}                   knobs for formulas (--set params.A=266.6)
    loads       {name: data}                     named data, evaluated in order
    boundaries  {boundary: {kind: data}}         kinds the model has, at most one per family;
                                                 completeness is checked when the model is built
    sources     {name: data}                     names the model reads

A physics adds its own sections through its model class (`sections`): each is a class with
from_dict(raw, path, model).
"""

import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from amphix.config.validate import ConfigError, check_keys, join, mapping, number, unread
from amphix.core import mesh as core_mesh
from amphix.core.expressions import FUNCTIONS, VARIABLES
from amphix.core.units import Scales


class Section(Protocol):
    """A model's own section: built from its raw dict (path for errors, model for warnings)."""

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], path: str, model: str) -> Any:
        """Return the validated section."""
        ...


def scales(raw: Any) -> Scales:
    """Return the `scales` section."""
    raw = mapping("scales", raw)
    check_keys("scales", raw, {"L0", "T0", "P0"})
    return Scales(**{k: number(join("scales", k), v, positive=True) for k, v in raw.items()})


def mesh(raw: Any) -> dict:
    """Return the `mesh` section, checked by core.mesh before any meshing."""
    raw = dict(mapping("mesh", raw))
    try:
        core_mesh.check(raw)
    except ValueError as e:
        raise ConfigError("mesh", str(e).split(": ", 1)[-1]) from e
    return raw


TIME_KEYS = {"T", "dt", "t0", "theta", "startup_steps", "newmark_beta", "newmark_gamma"}


@dataclass(frozen=True)
class Time:
    """Time interval [s], step [s] and the parameters of the time schemes."""

    T: float
    dt: float
    t0: float = 0.0
    theta: float = 1.0  # theta-method: 1 backward Euler, 1/2 Crank-Nicolson
    startup_steps: int = 0  # first steps with theta = 1 (Rannacher start)
    newmark_beta: float = 0.25
    newmark_gamma: float = 0.5
    nsteps: int = field(init=False)

    def __post_init__(self) -> None:
        """Check that [t0, T] is a whole number of steps."""
        n = round((self.T - self.t0) / self.dt)
        if n < 1 or abs(n * self.dt - (self.T - self.t0)) > 1e-9 * max(1.0, abs(self.T)):
            raise ConfigError(
                "time", f"T - t0 = {self.T - self.t0} must be a multiple of dt = {self.dt}"
            )
        object.__setattr__(self, "nsteps", n)

    @classmethod
    def from_dict(cls, raw: Any, model: Any) -> "Time":
        """Return the `time` section; optional keys the model does not read warn."""
        raw = mapping("time", raw)
        check_keys("time", raw, TIME_KEYS, required={"T", "dt"})
        unread("time", set(raw) - {"T", "dt", "t0"} - set(model.time_keys), model.name)
        values: dict[str, Any] = {
            k: number(join("time", k), v, positive=k == "dt") for k, v in raw.items()
        }
        if "startup_steps" in values:
            steps = number("time.startup_steps", raw["startup_steps"], integer=True)
            if steps < 0:
                raise ConfigError("time.startup_steps", f"must be >= 0, got {steps}")
            values["startup_steps"] = steps
        return cls(**values)


OUTPUT_KEYS = {
    "dir",
    "name",
    "series",
    "probes",
    "lines",
    "grid",
    "slices",
    "snapshots",
    "fields",
    "boundary_flux",
    "paraview",
}


@dataclass(frozen=True)
class Output:
    """What the Recorder stores (all positions in m, times in s)."""

    dir: str = "results"
    name: str | None = None  # file stem; default: the YAML stem
    series: list = field(default_factory=lambda: ["flux", "p"])  # at the probes, every step
    probes: dict = field(default_factory=dict)  # name -> [x, y(, z)]
    lines: dict = field(default_factory=dict)  # name -> {from, to, n}
    grid: int | None = None  # 2D: n x n points over the bounding box
    slices: dict = field(default_factory=dict)  # 3D: name -> {normal, at, n}
    snapshots: list | dict = field(default_factory=list)  # times, or {from, to, every}
    fields: bool | list = False  # full fields at the snapshots
    boundary_flux: bool = True
    paraview: int = 0  # every n steps (0: never)

    @classmethod
    def from_dict(cls, raw: Any) -> "Output":
        """Return the `output` section."""
        raw = mapping("output", raw)
        check_keys("output", raw, OUTPUT_KEYS)
        s = raw.get("snapshots", [])
        if isinstance(s, Mapping):
            check_keys("output.snapshots", s, {"from", "to", "every"}, required={"from", "to"})
            if s["to"] < s["from"] or int(s.get("every", 1)) < 1:
                raise ConfigError("output.snapshots", "needs from <= to and every >= 1")
        elif not isinstance(s, Sequence):
            raise ConfigError("output.snapshots", "a list of times or {from, to, every}")
        for key in ("lines", "slices", "probes"):
            mapping(join("output", key), raw.get(key, {}))
        return cls(**raw)

    def times(self, time: Time) -> list[float]:
        """Return the snapshot times [s]: the list given, or every `every`-th step of [from, to]."""
        s = self.snapshots
        if not isinstance(s, Mapping):
            return [float(t) for t in s]
        first, last = (round((s[k] - time.t0) / time.dt) for k in ("from", "to"))
        return [time.t0 + k * time.dt for k in range(first, last + 1, int(s.get("every", 1)))]


def solver(raw: Any) -> dict:
    """Return the `solver` section: PETSc options (no leading dash) with scalar values."""
    raw = mapping("solver", raw)
    for k, v in raw.items():
        if not isinstance(v, str | numbers.Real):
            raise ConfigError(join("solver", k), f"must be a number or a string, got {v!r}")
    return dict(raw)


def data(path: str, value: Any) -> None:
    """Raise unless value is data: a number, a formula, a list of data, or a {waveform: ...}."""
    if isinstance(value, bool) or not isinstance(value, str | numbers.Real | Sequence | Mapping):
        raise ConfigError(path, f"must be a number, a formula, a list or a waveform, got {value!r}")
    if isinstance(value, Mapping) and "waveform" not in value:
        raise ConfigError(path, "a section here must be {waveform: name, ...}")
    if isinstance(value, Sequence) and not isinstance(value, str):
        for i, v in enumerate(value):
            data(f"{path}[{i}]", v)


def names(params: Any, loads: Any) -> tuple[dict, dict]:
    """Return params (numbers) and loads; no built-in name hidden, no name in both."""
    params, loads = dict(mapping("params", params)), dict(mapping("loads", loads))
    reserved = set(FUNCTIONS) | set(VARIABLES)
    for section, given in (("params", params), ("loads", loads)):
        for name in given:
            if name in reserved:
                raise ConfigError(join(section, name), "would hide a built-in formula name")
    for name in set(params) & set(loads):
        raise ConfigError(join("loads", name), "is also a param")
    return {k: number(join("params", k), v) for k, v in params.items()}, loads


def boundaries(raw: Any, model: Any) -> dict:
    """Return `boundaries` checked for shape: known kinds, at most one per family, data values."""
    raw = mapping("boundaries", raw)
    kinds = {k.name: k.family for k in model.condition_kinds}
    for name, value_of in raw.items():
        path = join("boundaries", name)
        entry = mapping(path, value_of)
        check_keys(path, entry, kinds)
        families = [kinds[k] for k in entry]
        for family in {f for f in families if families.count(f) > 1}:
            given = sorted(k for k in entry if kinds[k] == family)
            raise ConfigError(path, f"{given} are all {family} conditions: give one")
        for kind, value in entry.items():
            data(join(path, kind), value)
    return {k: dict(v) for k, v in raw.items()}


def sources(raw: Any, model: Any) -> dict:
    """Return `sources`: names the model reads, data values."""
    raw = mapping("sources", raw)
    check_keys("sources", raw, {k.name for k in model.source_kinds})
    for name, value in raw.items():
        data(join("sources", name), value)
    return dict(raw)
