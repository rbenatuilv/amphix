"""Describing a problem: shared sections, the loader (`base:` chains, overrides), validation.

Each physics registers its own sections; this layer knows nothing about Biot. Uses only `core`.

    from amphix.config import load
    problem = load("case.yaml", {"time.dt": 0.005})
"""

from amphix.config.loader import parse_set
from amphix.config.problem import Problem, from_dict, load
from amphix.config.sections import Output, Section, Time
from amphix.config.validate import ConfigError, ConfigWarning

__all__ = [
    "ConfigError",
    "ConfigWarning",
    "Output",
    "Problem",
    "Section",
    "Time",
    "from_dict",
    "load",
    "parse_set",
]
