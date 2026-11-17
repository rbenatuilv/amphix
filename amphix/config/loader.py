"""Reading a problem file: YAML with exponent numbers, `base:` chains and dotted overrides.

    tree, name, root = read("annulus_poro2_pulse.yaml", {"time.dt": 0.005, "params.A": 266.6})

`base:` names another YAML, relative to the file that contains it, which this file extends: nested
sections merge key by key (`material: {K: 1.0e-12}` changes only K); any other value, lists
included, is replaced. A base can have its own `base:` (experiment -> case); a loop is an error. A
relative path (`mesh.file`, `initial.from`) stays relative to the file that gives it. The name of
the problem (the file stem) and its root (the folder of `output.dir`) come from the file read, not
from its bases.

Overrides (`--set KEY=VALUE`, or a dict from Python) replace one key each, by its dotted path, after
the bases are merged; a missing path is created. Their values are read as YAML, so
`--set 'output.fields=[p, cflux]'` gives a list.

Numbers: PyYAML follows YAML 1.1, where a float needs a dot, so `1e-3` would be the string "1e-3".
The loader also reads exponent-only numbers (1e-3, 2E5, as YAML 1.2 does) as floats.
"""

import copy
import re
from pathlib import Path
from typing import Any

import yaml

from amphix.config.validate import ConfigError, join, mapping

PATHS = (("mesh", "file"), ("initial", "from"))  # file paths, relative to the YAML that gives them


class _Loader(yaml.SafeLoader):
    """yaml.SafeLoader that also reads exponent-only numbers (1e-3) as floats."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:[0-9][0-9_]*(?:\.[0-9_]*)?|\.[0-9_]+)[eE][-+]?[0-9]+$"),
    list("-+0123456789."),
)


def yaml_load(text: str) -> Any:
    """Return yaml.safe_load(text), with 1e-3 read as a number."""
    return yaml.load(text, Loader=_Loader)


def parse_set(items: list[str]) -> dict[str, Any]:
    """Return --set KEY=VALUE arguments as dotted-key overrides; values are read as YAML."""
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ConfigError("--set", f"expected KEY=VALUE, got {item!r}")
        out[key] = yaml_load(value)
    return out


def merge(base: dict, top: dict) -> dict:
    """Return base updated with top, recursively for nested sections."""
    out = copy.deepcopy(base)
    for k, v in top.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def override(tree: dict, overrides: dict[str, Any] | None) -> dict:
    """Return a copy of tree with each dotted key replaced (missing sections are created)."""
    tree = copy.deepcopy(tree)
    for key, value in (overrides or {}).items():
        parts, node, path = key.split("."), tree, ""
        for part in parts[:-1]:
            path = join(path, part)
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ConfigError(path, f"is not a section, so {key} cannot be set")
        node[parts[-1]] = value
    return tree


def read_tree(path: str | Path, _chain: tuple[Path, ...] = ()) -> dict:
    """Return the tree of a YAML file with its `base:` chain merged in (base first)."""
    path = Path(path).resolve()
    chain = (*_chain, path)
    if path in _chain:
        raise ConfigError("base", "loop " + " -> ".join(p.name for p in chain))
    if not path.is_file():
        where = f" (base of {_chain[-1].name})" if _chain else ""
        raise ConfigError("base" if _chain else "", f"no file {path}{where}")
    try:
        tree = yaml_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        raise ConfigError("", f"{path.name} is not valid YAML: {e}") from e
    tree = dict(mapping("", tree))
    if "base" not in tree:
        return tree
    base_path = path.parent / str(tree.pop("base"))
    base = read_tree(base_path, chain)
    for section, key in PATHS:
        node = base.get(section)
        if isinstance(node, dict) and key in node and not Path(str(node[key])).is_absolute():
            node[key] = str((base_path.parent / str(node[key])).resolve())  # relative to its base
    return merge(base, tree)


def read(path: str | Path, overrides: dict[str, Any] | None = None) -> tuple[dict, str, Path]:
    """Return (tree, name, root) of a problem file: bases merged, then the overrides applied."""
    path = Path(path)
    return override(read_tree(path), overrides), path.stem, path.parent
