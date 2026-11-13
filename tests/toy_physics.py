"""A dummy physics for the config tests: what a model class shows to config/ (core.registry)."""

from collections import namedtuple
from dataclasses import dataclass
from typing import ClassVar

from amphix.config.validate import ConfigError, check_keys, number, unread
from amphix.core import registry

Kind = namedtuple("Kind", "name family")
SourceKind = namedtuple("SourceKind", "name")


@dataclass(frozen=True)
class ToyMaterial:
    """A physics section: conductivity k > 0, and `extra`, which only toy_b reads."""

    k: float
    extra: float = 0.0

    @classmethod
    def from_dict(cls, raw, path, model):
        check_keys(path, raw, {"k", "extra"}, required={"k"})
        if model != "toy_b" and "extra" in raw:
            unread(path, ["extra"], model)
        out = cls(**{key: number(f"{path}.{key}", v) for key, v in raw.items()})
        if out.k <= 0:
            raise ConfigError(f"{path}.k", "must be positive")
        return out


class ToyA:
    name = "toy_a"
    sections: ClassVar[dict] = {"material": ToyMaterial}
    condition_kinds = (
        Kind("temperature", "thermal"),
        Kind("heat_flux", "thermal"),
        Kind("fixed", "mechanical"),
    )
    source_kinds = (SourceKind("heat_source"),)
    time_keys = frozenset({"theta", "startup_steps"})


class ToyB(ToyA):
    name = "toy_b"
    time_keys = frozenset()


registry.register(ToyA)
registry.register(ToyB)
