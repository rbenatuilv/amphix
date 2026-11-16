"""Meshes with named boundaries and subdomains, from a `mesh` section (sizes in SI).

    build({"shape": "annulus", "radius": 1e-2, "inner": {"center": [3e-3, 0], "radius": 2.5e-3},
           "h": 5e-4, "names": {"inner": "ventricle", "outer": "pial"}}, L0=1e-2) -> Domain

The shapes, the keys each one accepts (besides `shape` and `names`) and the boundaries they name:

    rectangle   corners, n | h, diagonal       left, right, bottom, top
    box         corners, n | h                 left, right, front, back, bottom, top
    disc, ball  center, radius, h              outer
    annulus,    center, radius, inner, h,      outer, inner (inner = {center, radius}: the hole)
    shell       wall
    gmsh        file, dim, facets, cells       the named physical groups of the file

`wall: {h, band, growth}` refines an annulus or shell at both walls: size wall.h within band of
either wall, growing to h at band + growth. `names: {old: new}` renames boundaries and
subdomains. check() rejects an unknown shape or key, and a rectangle or box without n or h, before
any meshing (config calls it at load time).
"""

from pathlib import Path

from amphix.core.mesh.domain import Domain
from amphix.core.mesh.gmsh import from_msh, ghosted_partitioner, round_shape
from amphix.core.mesh.measures import Measures, cell_size, mean_size
from amphix.core.mesh.shapes import box, rectangle

SHAPE_KEYS: dict[str, set[str]] = {
    "rectangle": {"corners", "n", "h", "diagonal"},
    "box": {"corners", "n", "h"},
    "disc": {"center", "radius", "h"},
    "ball": {"center", "radius", "h"},
    "annulus": {"center", "radius", "inner", "h", "wall"},
    "shell": {"center", "radius", "inner", "h", "wall"},
    "gmsh": {"file", "dim", "facets", "cells"},
}
REQUIRED: dict[str, set[str]] = {
    "rectangle": {"corners"},
    "box": {"corners"},
    "disc": {"radius", "h"},
    "ball": {"radius", "h"},
    "annulus": {"radius", "inner", "h"},
    "shell": {"radius", "inner", "h"},
    "gmsh": {"file"},
}

__all__ = [
    "REQUIRED",
    "SHAPE_KEYS",
    "Domain",
    "Measures",
    "box",
    "build",
    "cell_size",
    "check",
    "from_msh",
    "ghosted_partitioner",
    "mean_size",
    "rectangle",
]


def check(section: dict) -> None:
    """Raise ValueError for an unknown shape, an unknown or missing key, before any meshing."""
    shape = section.get("shape")
    if shape not in SHAPE_KEYS:
        raise ValueError(f"mesh.shape: unknown shape {shape!r}; one of {sorted(SHAPE_KEYS)}")
    allowed = SHAPE_KEYS[shape] | {"shape", "names"}
    unknown = set(section) - allowed
    if unknown:
        raise ValueError(
            f"mesh ({shape}): unknown key(s) {sorted(unknown)}; allowed {sorted(allowed)}"
        )
    missing = REQUIRED[shape] - set(section)
    if missing:
        raise ValueError(f"mesh ({shape}): missing {sorted(missing)}")
    if shape in ("rectangle", "box") and not ({"n", "h"} & set(section)):
        raise ValueError(f"mesh ({shape}): give n or h")
    if "inner" in section and set(section["inner"]) != {"center", "radius"}:
        raise ValueError(f"mesh ({shape}): inner takes center and radius")
    if "wall" in section:
        wall = section["wall"]
        if not isinstance(wall, dict) or set(wall) != {"h", "band", "growth"}:
            raise ValueError(f"mesh ({shape}): wall takes h, band and growth")
        if not all(isinstance(v, int | float) and v > 0 for v in wall.values()):
            raise ValueError(f"mesh ({shape}): wall h, band and growth must be positive numbers")


def build(section: dict, L0: float = 1.0, root: str | Path = ".") -> Domain:
    """Return the mesh (solver units) of a mesh section; a relative gmsh file is under root."""
    check(section)
    shape = section["shape"]
    keys = {k: v for k, v in section.items() if k not in ("shape", "names")}
    if shape == "rectangle":
        domain = rectangle(L0=L0, **keys)
    elif shape == "box":
        domain = box(L0=L0, **keys)
    elif shape == "gmsh":
        path = Path(root) / keys.pop("file")
        domain = from_msh(path, L0=L0, **keys)
    else:
        domain = round_shape(shape, L0=L0, **keys)
    tdim = domain.mesh.topology.dim
    domain.mesh.topology.create_connectivity(tdim - 1, tdim)  # facet -> cell, for every DG form
    return domain.renamed(section.get("names", {}))
