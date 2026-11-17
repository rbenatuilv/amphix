"""The registry of models: the YAML `model:` name -> the model class.

A model class registers itself when it is defined (core.model.Model), so a physics only has to be
imported: amphix/__init__.py imports physics.poro, which registers poro2 and poro3. A model defined
in a script registers the same way.

What config/ reads from a registered class, without importing any physics (`Registered`):

    name             the `model:` value
    sections         {section: class with from_dict(raw, path, model)}, e.g. material
    condition_kinds  the boundary-condition kinds (each with .name and .family)
    source_kinds     the source kinds (each with .name)
    time_keys        the optional `time` keys it reads (e.g. theta, startup_steps); the others warn
    fields, tracked  its state (`initial` keys): the fields and the other tracked state (d_t, d_tt)
    restarts         whether `initial` may name a fields file
    quantities       its derived quantities (flux, ...), which `output` may record
"""

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Protocol


class Registered(Protocol):
    """What a model class exposes to config/ (see the module docstring)."""

    name: ClassVar[str]
    sections: ClassVar[Mapping[str, Any]]
    condition_kinds: ClassVar[Sequence[Any]]
    source_kinds: ClassVar[Sequence[Any]]
    time_keys: ClassVar[frozenset[str]]
    fields: ClassVar[Mapping[str, Any]]
    tracked: ClassVar[frozenset[str]]
    restarts: ClassVar[bool]
    quantities: ClassVar[Mapping[str, Any]]


MODELS: dict[str, Any] = {}


def register(cls: Any) -> Any:
    """Register a model class under cls.name; another class with the same name is an error."""
    name = cls.name
    if name in MODELS and MODELS[name] is not cls:
        raise ValueError(f"model {name!r} is already registered by {MODELS[name].__qualname__}")
    MODELS[name] = cls
    return cls


def get(name: str) -> Any:
    """Return the class registered as `name`."""
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(MODELS)}")
    return MODELS[name]
