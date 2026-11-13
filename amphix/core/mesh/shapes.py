"""Structured meshes: a rectangle (triangles) and a box (tetrahedra), each side its own boundary.

    rectangle   left (x = x0), right (x = x1), bottom (y = y0), top (y = y1)
    box         left, right (x), front, back (y), bottom, top (z)

Sizes are in SI; the mesh is returned in solver units (coordinates / L0). The cells are given by
`n` (cells per side) or by a target size `h` (n = round(side / h), at least 1). A 2D corner facet
does not exist (facets are edges), and in 3D an edge belongs to no facet, so every boundary facet
lies on exactly one side (the first one listed wins if it did not).
"""

from collections.abc import Sequence

import dolfinx
import numpy as np
from mpi4py import MPI

from amphix.core.mesh.domain import Domain, all_cells

SIDES_2D = ("left", "right", "bottom", "top")
SIDES_3D = ("left", "right", "front", "back", "bottom", "top")
DIAGONALS = ("left", "right", "crossed", "left_right", "right_left")


def cells_per_side(
    lo: np.ndarray, hi: np.ndarray, n: Sequence[int] | None, h: float | None
) -> list:
    """Return n, or the number of cells of size about h (solver units) along each side."""
    if n is not None:
        return [int(k) for k in n]
    if h is None:
        raise ValueError("give n (cells per side) or h (target size)")
    return [max(1, round((b - a) / h)) for a, b in zip(lo, hi, strict=True)]


def rectangle(
    corners: Sequence[Sequence[float]],
    n: Sequence[int] | None = None,
    h: float | None = None,
    diagonal: str = "right",
    L0: float = 1.0,
) -> Domain:
    """Return the rectangle [x0, x1] x [y0, y1] (corners in m), triangles with this diagonal."""
    lo, hi = (np.asarray(c, dtype=float) / L0 for c in corners)
    if diagonal not in DIAGONALS:
        raise ValueError(f"diagonal must be one of {DIAGONALS}, got {diagonal!r}")
    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        [lo, hi],
        cells_per_side(lo, hi, n, None if h is None else h / L0),
        dolfinx.mesh.CellType.triangle,
        ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
        diagonal=dolfinx.mesh.DiagonalType[diagonal],
    )
    sides = {"left": (0, lo[0]), "right": (0, hi[0]), "bottom": (1, lo[1]), "top": (1, hi[1])}
    return _tagged(mesh, sides, L0)


def box(
    corners: Sequence[Sequence[float]],
    n: Sequence[int] | None = None,
    h: float | None = None,
    L0: float = 1.0,
) -> Domain:
    """Return the box [x0, x1] x [y0, y1] x [z0, z1] (corners in m), 6 tetrahedra per cube."""
    lo, hi = (np.asarray(c, dtype=float) / L0 for c in corners)
    mesh = dolfinx.mesh.create_box(
        MPI.COMM_WORLD,
        [lo, hi],
        cells_per_side(lo, hi, n, None if h is None else h / L0),
        dolfinx.mesh.CellType.tetrahedron,
        ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
    )
    sides = {
        "left": (0, lo[0]),
        "right": (0, hi[0]),
        "front": (1, lo[1]),
        "back": (1, hi[1]),
        "bottom": (2, lo[2]),
        "top": (2, hi[2]),
    }
    return _tagged(mesh, sides, L0)


def _tagged(mesh: dolfinx.mesh.Mesh, sides: dict, L0: float) -> Domain:
    """Tag the boundary facets of each side 1, 2, ... in the order of `sides`."""
    fdim = mesh.topology.dim - 1
    facets, values, names = [], [], {}
    for tag, (name, (axis, value)) in enumerate(sides.items(), start=1):
        f = dolfinx.mesh.locate_entities_boundary(
            mesh, fdim, lambda x, a=axis, v=value: np.isclose(x[a], v)
        )
        facets.append(f)
        values.append(np.full(len(f), tag, dtype=np.int32))
        names[name] = tag
    all_f, idx = np.unique(np.concatenate(facets), return_index=True)  # first side wins
    tags = dolfinx.mesh.meshtags(mesh, fdim, all_f, np.concatenate(values)[idx])
    return Domain(mesh, L0, tags, names, all_cells(mesh), {})
