"""The sections every problem shares, and the shape checks of the model-dependent ones.

    scales      Scales(L0, T0, P0)               [m, s, Pa], default 1 (solve in SI numbers)
    mesh        dict, checked by core.mesh.check (sizes in SI; required)
    time        Time(T, dt, t0, theta, startup_steps, newmark_beta, newmark_gamma) (required)
    output      Output: what io records, and when (positions in m, times in s)
    solver      {PETSc option: value}, added to the default direct solver
    params      {name: number}                   knobs for formulas (--set params.A=266.6)
    loads       {name: data}                     named data, evaluated in order
    boundaries  {boundary: {kind: data}}         kinds the model has, at most one per family;
                                                 completeness is checked when the model is built
    sources     {name: data}                     names the model reads
    initial     Initial: the state at t0         a saved fields file, or data per state key;
                                                 default: rest

A physics adds its own sections through its model class (`sections`): each is a class with
from_dict(raw, path, model).
"""

import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from amphix.config.validate import ConfigError, check_keys, join, mapping, number, unread
from amphix.core import mesh as core_mesh
from amphix.core.balance import balanced
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


def steps(path: str, at: Any, time: Time) -> Sequence[int]:
    """Return the step numbers (0 is t0) of an `at`: a list of times [s], or {from, to, every}.

    from and to are in s (default t0 and T), every in steps (default 1). A time within dt/2 of a
    step is that step; a time farther than dt/2 from every step (outside [t0, T]) is an error.
    """

    def step(key: str, t: Any) -> int:
        t = number(key, t)
        n = min(max(round((t - time.t0) / time.dt), 0), time.nsteps)  # the nearest step
        tn = time.t0 + n * time.dt
        if abs(t - tn) > time.dt / 2 * (1 + 1e-9):
            raise ConfigError(
                key,
                f"{t} s is {abs(t - tn):.6g} s from the nearest step (t = {tn:.10g} s): more "
                f"than dt/2 from every step of [{time.t0}, {time.T}], dt = {time.dt}",
            )
        return n

    if isinstance(at, Mapping):
        check_keys(path, at, {"from", "to", "every"})
        first = step(join(path, "from"), at.get("from", time.t0))
        last = step(join(path, "to"), at.get("to", time.T))
        every = number(join(path, "every"), at.get("every", 1), integer=True)
        if last < first or every < 1:
            raise ConfigError(path, "needs from <= to and every >= 1")
        return range(first, last + 1, int(every))
    if isinstance(at, Sequence) and not isinstance(at, str) and at:
        return tuple(sorted({step(f"{path}[{i}]", t) for i, t in enumerate(at)}))
    raise ConfigError(path, f"a list of times [s] or {{from, to, every}}, got {at!r}")


@dataclass(frozen=True)
class Block:
    """One output block: its quantities, its steps (from `at`) and its own keys (paths, n...)."""

    quantities: tuple[str, ...] = ()
    steps: Sequence[int] = ()  # step numbers, 0 is t0
    spec: dict = field(default_factory=dict)


# block: (its own keys, besides quantities and at), quantities?, at?
BLOCKS: dict[str, tuple[set[str], bool, bool]] = {
    "probes": ({"points"}, True, False),
    "lines": ({"paths"}, True, True),
    "grid": ({"n"}, True, True),
    "slices": ({"planes"}, True, True),
    "fields": (set(), True, True),
    "jump": (set(), False, True),
    "paraview": (set(), True, True),
}
BOUNDARY_FLUX = ("raw", "conservative")


def _quantities(path: str, raw: Any, known: set[str]) -> tuple[str, ...]:
    if isinstance(raw, str) or not isinstance(raw, Sequence) or not raw:
        raise ConfigError(path, f"a list of quantities, some of {sorted(known)}; got {raw!r}")
    for q in raw:
        if q not in known:
            raise ConfigError(path, f"unknown quantity {q!r}; this model has {sorted(known)}")
    return tuple(raw)


def _block(name: str, raw: Any, time: Time, known: set[str]) -> Block:
    path = join("output", name)
    raw = mapping(path, raw)
    own, quantities, timed = BLOCKS[name]
    keys = own | ({"quantities"} if quantities else set()) | ({"at"} if timed else set())
    check_keys(path, raw, keys, required=keys)
    for key in own - {"n"}:
        mapping(join(path, key), raw[key])
    if "n" in own:
        number(join(path, "n"), raw["n"], positive=True, integer=True)
    names = _quantities(join(path, "quantities"), raw["quantities"], known) if quantities else ()
    at = steps(join(path, "at"), raw["at"], time) if timed else ()
    return Block(names, at, {k: raw[k] for k in own})


def _boundary_flux(raw: Any, known: set[str]) -> tuple[str, ...]:
    path = "output.boundary_flux"
    values = [raw] if isinstance(raw, str) else raw
    if isinstance(raw, bool) or not isinstance(values, Sequence):
        raise ConfigError(path, f"none, raw, conservative or [raw, conservative]; got {raw!r}")
    if list(values) == ["none"]:
        return ()
    for v in values:
        if v not in BOUNDARY_FLUX:
            raise ConfigError(path, f"none, raw, conservative or [raw, conservative]; got {v!r}")
    if "raw" in values and "flux" not in known:
        raise ConfigError(path, "raw needs the quantity flux, which this model does not have")
    if "conservative" in values and "cflux" not in known:
        raise ConfigError(
            path, "conservative: the model has no step balance (core.balance.Balance)"
        )
    return tuple(v for v in BOUNDARY_FLUX if v in values)


@dataclass(frozen=True)
class Output:
    """What the Recorder stores: each block, if given, says what it records and when.

        boundary_flux   none (default), raw, conservative or [raw, conservative]: the flow
                        through every named boundary, every step (bflux_*, cbflux_*)
        probes          {points: {name: [x, y(, z)]}, quantities}: every step
        lines           {quantities, at, paths: {name: {from, to, n}}}
        grid            {n, quantities, at}: 2D, n x n points over the bounding box
        slices          {quantities, at, planes: {name: {normal, at, n}}}: 3D
        fields          {quantities, at}: the full fields, to <name>_fields.npz
        jump            {at}: the normal jump of the raw flux
        paraview        {quantities, at}

    `at` is a list of times [s] or {from, to, every} (see steps()). Positions are in m. Nothing is
    recorded but t unless asked for; cflux is the conservative (Raviart-Thomas) velocity.
    """

    dir: str = "results"
    name: str | None = None  # file stem; default: the YAML stem
    boundary_flux: tuple[str, ...] = ()
    probes: Block | None = None
    lines: Block | None = None
    grid: Block | None = None
    slices: Block | None = None
    fields: Block | None = None
    jump: Block | None = None
    paraview: Block | None = None

    @classmethod
    def from_dict(cls, raw: Any, time: Time, model: Any, dim: int) -> "Output":
        """Return the `output` section of a model, its times on the steps of `time`."""
        raw = mapping("output", raw)
        check_keys("output", raw, {"dir", "name", "boundary_flux", *BLOCKS})
        if not set(raw) - {"dir", "name"}:  # nothing recorded: the model need not say more
            return cls(**raw)
        cflux = ["cflux"] if balanced(model) else []
        known = {*model.fields, *model.tracked, *model.quantities, *cflux}
        for name, wrong in (("grid", 3), ("slices", 2)):
            if name in raw and dim == wrong:
                raise ConfigError(
                    join("output", name), f"not for a {dim}D mesh (grid: 2D, slices: 3D)"
                )
        if "jump" in raw and "flux" not in known:
            raise ConfigError(
                "output.jump", "needs the quantity flux, which this model does not have"
            )
        values: dict[str, Any] = {k: raw[k] for k in ("dir", "name") if k in raw}
        values["boundary_flux"] = _boundary_flux(raw.get("boundary_flux", "none"), known)
        values |= {k: _block(k, raw[k], time, known) for k in BLOCKS if k in raw}
        return cls(**values)

    def blocks(self) -> dict[str, Block]:
        """Return {name: block} of the blocks given."""
        return {k: b for k in BLOCKS if (b := getattr(self, k)) is not None}


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


@dataclass(frozen=True)
class Initial:
    """The state at t0, from a saved fields file or from data (otherwise the run starts at rest).

        initial: {from: results/steady_fields.npz, time: last}     # d and p saved by a run
        initial: {p: "gamma * (1 - r / 1e-2)", d: [0, 0]}          # data, evaluated at t0

    From a file: the model's fields at one saved time (`last`, the default, or a time in s), every
    other tracked state (d_t, d_tt) zero; the path is relative to the YAML file. From data: any
    state of the model (its fields, and what it tracks: d_t, d_tt), the others zero. The boundary
    data at t0 should agree with this state; nothing checks it.
    """

    path: Path | None = None  # the fields file
    index: int = -1  # which of its saved times
    time: float = 0.0  # that saved time [s]
    values: dict = field(default_factory=dict)  # or {state: data}


def initial(raw: Any, model: Any, root: Path) -> Initial | None:
    """Return the `initial` section (None: start at rest); a file is checked for the fields."""
    raw = mapping("initial", raw if raw is not None else {})
    if not raw:
        return None
    state = {*model.fields, *model.tracked}
    for key in raw:
        if key not in {"from", "time", *state}:
            raise ConfigError(
                join("initial", key),
                f"not state of {model.name}: give `from` (and `time`) or some of {sorted(state)}",
            )
    given = sorted(set(raw) & state)
    if "from" in raw and given:
        raise ConfigError("initial", f"`from` and {given}: give a fields file or data, not both")
    if "from" in raw:
        return _saved(raw, model, root)
    if "time" in raw:
        raise ConfigError("initial.time", "only with `from`")
    for key, value in raw.items():
        data(join("initial", key), value)
    return Initial(values=dict(raw))


def _saved(raw: Mapping[str, Any], model: Any, root: Path) -> Initial:
    if not model.restarts:
        raise NotImplementedError(f"initial.from: {model.name} cannot start from a fields file")
    path = Path(str(raw["from"]))
    path = path if path.is_absolute() else root / path
    if not path.is_file():
        raise ConfigError("initial.from", f"no file {path}")
    try:
        with np.load(path) as f:
            names, times = [str(n) for n in f["names"]], np.asarray(f["snap_t"], dtype=float)
    except (OSError, ValueError, KeyError) as e:
        raise ConfigError("initial.from", f"{path} is not a fields file ({e})") from e
    missing = sorted(set(model.fields) - set(names))
    if missing:
        raise ConfigError(
            "initial.from", f"{path.name} has no {missing}; it has {names} (output.fields)"
        )
    if len(times) == 0:
        raise ConfigError("initial.from", f"{path.name} holds no saved time (output.fields.at)")
    when = raw.get("time", "last")
    if when == "last":
        index = len(times) - 1
    else:
        t = number("initial.time", when)
        hits = np.flatnonzero(np.abs(times - t) <= 1e-9 * max(1.0, abs(t)))
        if len(hits) == 0:
            raise ConfigError(
                "initial.time", f"{t} is not saved in {path.name}; saved: {times.tolist()} or last"
            )
        index = int(hits[0])
    return Initial(path=path, index=index, time=float(times[index]))
