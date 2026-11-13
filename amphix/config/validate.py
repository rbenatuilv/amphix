"""Errors and warnings that name the YAML path of what is wrong.

    raise ConfigError("material.K", "must be positive")   ->  "material.K: must be positive"
    check_keys("time", given, allowed={"T", "dt", ...}, required={"T", "dt"})
    number("mesh.h", 5e-4, positive=True)                  ->  5e-4 (as a float)

Unknown keys are errors; a valid key that the chosen model does not read is a warning (`unread`),
because the same file may serve several models.
"""

import numbers
import warnings
from collections.abc import Iterable, Mapping
from typing import Any


class ConfigError(ValueError):
    """A problem description that cannot be used; the message starts with the YAML path."""

    def __init__(self, path: str, message: str) -> None:
        """Error at `path` (dotted, e.g. "boundaries.cavity.pressure"; "" for the whole file)."""
        self.path = path
        super().__init__(f"{path}: {message}" if path else message)


class ConfigWarning(UserWarning):
    """A valid key that the chosen model does not read."""


def join(path: str, key: object) -> str:
    """Return the dotted path of a key inside a section ("" is the top level)."""
    return f"{path}.{key}" if path else str(key)


def mapping(path: str, value: Any) -> Mapping:
    """Return value if it is a mapping (a YAML section), else raise."""
    if not isinstance(value, Mapping):
        raise ConfigError(path, f"must be a section (key: value pairs), got {value!r}")
    return value


def check_keys(
    path: str, given: Mapping, allowed: Iterable[str], required: Iterable[str] = ()
) -> None:
    """Raise for a key of `given` that is not allowed, or a required key that is missing."""
    allowed, required = set(allowed), set(required)
    unknown = sorted(str(k) for k in set(given) - allowed)
    if unknown:
        where = join(path, unknown[0]) if len(unknown) == 1 else path
        raise ConfigError(where, f"unknown key(s) {unknown}; allowed {sorted(allowed)}")
    missing = sorted(required - set(given))
    if missing:
        raise ConfigError(path, f"missing {missing}")


def number(path: str, value: Any, *, positive: bool = False, integer: bool = False) -> float:
    """Return value as a float (an int if integer), checking it is a number (and > 0)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ConfigError(path, f"must be a number, got {value!r}")
    v = float(value)
    if integer and v != int(v):
        raise ConfigError(path, f"must be an integer, got {value!r}")
    if positive and not v > 0:
        raise ConfigError(path, f"must be positive, got {value!r}")
    return int(v) if integer else v


def unread(path: str, keys: Iterable[str], model: str) -> None:
    """Warn that these keys are valid but not read by the chosen model."""
    keys = sorted(keys)
    if keys:
        warnings.warn(f"{path}: {keys} not read by {model}", ConfigWarning, stacklevel=3)
