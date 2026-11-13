"""Boundary conditions and sources for any physics: kinds, completeness, scaled UFL data.

A physics declares the kinds it accepts:

    ConditionKind("displacement", "mechanical", "vector", DISPLACEMENT)
    ConditionKind("normal_traction", "mechanical", "normal", PRESSURE)  # a scalar s: data s n
    ConditionKind("pressure", "hydraulic", "scalar", PRESSURE)
    SourceKind("body_force", "vector", BODY_FORCE)

and every boundary of the mesh carries exactly one condition of each family. The entries come from
the YAML `boundaries` section and from code (`boundary_data=`, which replaces whole entries); the
checks of completeness and shapes run when the model is built:

    conditions = Conditions(entries, kinds, domain, measures, expressions, scales)
    for c in conditions.of("displacement"):   # in the order of the entries
        c.ds, c.value, c.si                    # measure, data in solver units, data in SI (UFL)
    conditions.measure("pressure")             # ds over every boundary of that kind, or None

`si` is kept so that a physics can take exact derivatives in SI (the boundary velocity of a
prescribed displacement: expressions.ddt(c.si)).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from amphix.core.expressions import Data, Expressions
from amphix.core.mesh import Domain, Measures
from amphix.core.units import Kind, Scales

SHAPES = ("scalar", "vector", "normal")


@dataclass(frozen=True)
class ConditionKind:
    """A boundary-condition kind: name, family, shape of its data, and unit kind of the data."""

    name: str
    family: str
    shape: str  # "scalar", "vector", or "normal" (a scalar s whose condition data is s n)
    unit: Kind


@dataclass(frozen=True)
class SourceKind:
    """A volume source: name, shape ("scalar" or "vector") and unit kind."""

    name: str
    shape: str
    unit: Kind


@dataclass(frozen=True)
class Condition:
    """One condition on one boundary: measure, data in solver units, data in SI."""

    boundary: str
    kind: ConditionKind
    ds: Any
    value: Any
    si: Any


def merge_entries(yaml: Mapping[str, dict], code: Mapping[str, dict] | None) -> dict[str, dict]:
    """Return the YAML entries with each boundary given in code replacing its whole entry."""
    out = {name: dict(entry) for name, entry in yaml.items()}
    out.update({name: dict(entry) for name, entry in (code or {}).items()})
    return out


def _check_shape(path: str, expr: Any, shape: str, dim: int) -> None:
    want = (dim,) if shape == "vector" else ()
    if expr.ufl_shape != want:
        what = f"{dim} components" if shape == "vector" else "a scalar"
        raise ValueError(f"{path}: needs {what}, got shape {expr.ufl_shape}")


class Conditions:
    """The boundary conditions of a model, complete and scaled (see the module docstring)."""

    def __init__(
        self,
        entries: Mapping[str, Mapping[str, Data]],
        kinds: Sequence[ConditionKind],
        domain: Domain,
        measures: Measures,
        expressions: Expressions,
        scales: Scales,
    ) -> None:
        """Check completeness against the mesh, then build every condition."""
        self.kinds = {k.name: k for k in kinds}
        self._check_complete(entries, domain)
        self.boundaries = list(entries)  # the entry order
        self._of: dict[str, list[Condition]] = {name: [] for name in self.kinds}
        self._measures, dim = measures, domain.mesh.geometry.dim
        for boundary, entry in entries.items():
            for name, data in entry.items():
                kind, path = self.kinds[name], f"boundaries.{boundary}.{name}"
                si = expressions(data)
                _check_shape(path, si, kind.shape, dim)
                if kind.shape == "normal":
                    si = si * measures.n
                value = scales.factor(kind.unit) * si
                self._of[name].append(Condition(boundary, kind, measures.ds(boundary), value, si))

    def _check_complete(self, entries: Mapping[str, Mapping[str, Data]], domain: Domain) -> None:
        known, given = set(domain.facet_names), set(entries)
        if given - known:
            extra = sorted(given - known)
            raise ValueError(
                f"boundaries: unknown boundaries {extra}; the mesh has {sorted(known)}"
            )
        if known - given:
            raise ValueError(f"boundaries: no conditions for boundaries {sorted(known - given)}")
        families = sorted({k.family for k in self.kinds.values()})
        for boundary, entry in entries.items():
            unknown = set(entry) - set(self.kinds)
            if unknown:
                raise ValueError(
                    f"boundaries.{boundary}: unknown condition(s) {sorted(unknown)}; "
                    f"allowed {sorted(self.kinds)}"
                )
            for family in families:
                n = sum(self.kinds[k].family == family for k in entry)
                if n != 1:
                    options = sorted(k for k, v in self.kinds.items() if v.family == family)
                    raise ValueError(
                        f"boundaries.{boundary}: needs exactly one {family} condition "
                        f"(one of {options}), got {n}"
                    )

    def of(self, kind: str) -> list[Condition]:
        """Return the conditions of one kind, in the order of the entries."""
        return self._of[kind]

    def measure(self, kind: str) -> Any:
        """Return ds over every boundary with a condition of this kind, or None if there is none."""
        names = [c.boundary for c in self._of[kind]]
        return self._measures.ds(names) if names else None


class Sources:
    """The volume sources of a model, scaled: absent ones are zero Constants of their shape."""

    def __init__(
        self,
        given: Mapping[str, Data],
        kinds: Sequence[SourceKind],
        expressions: Expressions,
        scales: Scales,
    ) -> None:
        """Check names and shapes; given are the YAML sources merged with those from code."""
        self.kinds = {k.name: k for k in kinds}
        unknown = set(given) - set(self.kinds)
        if unknown:
            raise ValueError(f"sources: {sorted(unknown)} not read; allowed {sorted(self.kinds)}")
        self._values, dim = {}, expressions.mesh.geometry.dim
        for name, kind in self.kinds.items():
            default: Data = np.zeros(dim).tolist() if kind.shape == "vector" else 0.0
            si = expressions(given.get(name, default))
            _check_shape(f"sources.{name}", si, kind.shape, dim)
            self._values[name] = scales.factor(kind.unit) * si

    def __getitem__(self, name: str) -> Any:
        """Return a source in solver units."""
        return self._values[name]
