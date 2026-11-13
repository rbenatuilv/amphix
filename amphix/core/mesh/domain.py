"""The `Domain`: a mesh in solver units with its named boundaries and subdomains.

The mesh coordinates are non-dimensional (x* = x / L0); `L0` is kept with the mesh so that whoever
measures or samples it can work in SI. Facets and cells carry integer tags, and the names map onto
them:

    facet_names   {"ventricle": 2, "pial": 1}   boundaries: the YAML `boundaries` keys
    cell_names    {"tissue": 1, "fluid": 2}      subdomains (coupled physics), or {}

Every cell has a tag (1 when the mesh has no subdomains), so measures and saved fields can keep
them without special cases.
"""

from dataclasses import dataclass

import dolfinx
import numpy as np


@dataclass(frozen=True)
class Domain:
    """A mesh (solver units), its length scale L0 [m], and named facet and cell tags."""

    mesh: dolfinx.mesh.Mesh
    L0: float
    facet_tags: dolfinx.mesh.MeshTags
    facet_names: dict[str, int]
    cell_tags: dolfinx.mesh.MeshTags
    cell_names: dict[str, int]

    @property
    def dim(self) -> int:
        """Return the topological dimension (2 or 3)."""
        return int(self.mesh.topology.dim)

    def facet_tag(self, name: str) -> int:
        """Return the tag of a named boundary."""
        return _lookup(self.facet_names, name, "boundary")

    def cell_tag(self, name: str) -> int:
        """Return the tag of a named subdomain."""
        return _lookup(self.cell_names, name, "subdomain")

    def renamed(self, names: dict) -> "Domain":
        """Return the same domain with boundaries and subdomains renamed.

        names is {old: new}, for whichever group (facet or cell) has that name; an old name that
        both a facet and a cell group have is ambiguous and raises. Or it is
        {"facets": {old: new}, "cells": {old: new}}, one map per kind.
        """
        if (
            names
            and set(names) <= {"facets", "cells"}
            and all(isinstance(v, dict) for v in names.values())
        ):
            facet_map, cell_map = names.get("facets", {}), names.get("cells", {})
            _check_known(facet_map, self.facet_names, "names.facets")
            _check_known(cell_map, self.cell_names, "names.cells")
        else:
            both = sorted(set(names) & set(self.facet_names) & set(self.cell_names))
            if both:
                raise ValueError(
                    f"names: {both} name both a boundary and a subdomain; "
                    "use names: {facets: {...}, cells: {...}}"
                )
            _check_known(names, {**self.facet_names, **self.cell_names}, "names")
            facet_map = {k: v for k, v in names.items() if k in self.facet_names}
            cell_map = {k: v for k, v in names.items() if k in self.cell_names}
        facets = _apply(facet_map, self.facet_names, "boundaries")
        cells = _apply(cell_map, self.cell_names, "subdomains")
        return Domain(self.mesh, self.L0, self.facet_tags, facets, self.cell_tags, cells)


def _check_known(rename: dict[str, str], names: dict[str, int], where: str) -> None:
    unknown = set(rename) - set(names)
    if unknown:
        raise ValueError(f"{where}: {sorted(unknown)} not in the mesh; it has {sorted(names)}")


def _apply(rename: dict[str, str], names: dict[str, int], what: str) -> dict[str, int]:
    out = {rename.get(k, k): v for k, v in names.items()}
    if len(out) != len(names):  # two groups of one kind would share a name
        raise ValueError(f"names: renaming would give two {what} the same name: {rename}")
    return out


def _lookup(names: dict[str, int], name: str, what: str) -> int:
    if name not in names:
        raise KeyError(f"no {what} named {name!r}; the mesh has {sorted(names)}")
    return names[name]


def all_cells(mesh: dolfinx.mesh.Mesh, tag: int = 1) -> dolfinx.mesh.MeshTags:
    """Return cell tags that give every (local) cell the same tag: a mesh without subdomains."""
    tdim = mesh.topology.dim
    imap = mesh.topology.index_map(tdim)
    cells = np.arange(imap.size_local + imap.num_ghosts, dtype=np.int32)
    return dolfinx.mesh.meshtags(mesh, tdim, cells, np.full(len(cells), tag, dtype=np.int32))
