"""Where a DG field is drawn: the Lagrange lattice of every cell of a triangle mesh.

A DG field of degree k is drawn cell by cell. Each cell is cut at its lattice points (barycentric
coordinates i/k) into k^2 sub-triangles, the field is evaluated at those points of that cell (one
eval over all cells), and its colour is interpolated linearly in each sub-triangle (Gouraud). A P1
field is drawn exactly, a P2 field exactly at its lattice points, and a jump between two cells stays
visible: the points of a shared edge belong to each cell separately. Nothing is projected onto a
continuous space.

With cells=True (and for degree 0), each cell is one triangle with one value, at its centroid.

Only triangle meshes with affine cells (geometry degree 1) on one process are drawn; the mesh is in
m and the triangles are given in mm.
"""

from typing import Any

import dolfinx
import numpy as np

from amphix.plot.labels import PlotError

MM = 1e3  # mm per m


def check(mesh: dolfinx.mesh.Mesh) -> None:
    """Raise unless the mesh can be drawn (2D triangles, affine, one process)."""
    if mesh.topology.dim == 3:
        raise NotImplementedError("plotting 3D fields (slices) is not implemented yet")
    cell = mesh.topology.cell_name()
    if cell != "triangle" or mesh.geometry.dim != 2:
        raise PlotError(
            f"plotting needs a 2D triangle mesh, not {cell} cells in {mesh.geometry.dim}D"
        )
    degree = mesh.geometry.cmaps[0].degree
    if degree != 1:
        raise PlotError(f"plotting needs affine cells (geometry degree 1), not degree {degree}")
    if mesh.comm.size > 1:
        raise PlotError("plotting runs on one process (load the fields with MPI.COMM_SELF)")


def reference(k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the lattice of degree k on the reference triangle and its k^2 sub-triangles.

    Points are (s, r) with s, r = i/k, j/k and i + j <= k; triangles index those points.
    """
    ij = [(i, j) for j in range(k + 1) for i in range(k + 1 - j)]
    index = {p: n for n, p in enumerate(ij)}
    tris = []
    for i, j in ij:
        if i + j < k:  # upward
            tris.append((index[i, j], index[i + 1, j], index[i, j + 1]))
        if i + j < k - 1:  # downward
            tris.append((index[i + 1, j], index[i + 1, j + 1], index[i, j + 1]))
    return np.array(ij, dtype=float) / k, np.array(tris, dtype=np.int64).reshape(-1, 3)


class Lattice:
    """The points of every cell where a field is evaluated, and the triangles drawn between them.

    points (npoints, 3) in m and cells (npoints,) are what Function.eval takes; x, y (mm) and
    triangles are the drawn triangulation; mesh_x, mesh_y, mesh_triangles are the cells themselves.
    """

    def __init__(self, mesh: dolfinx.mesh.Mesh, degree: int, cells: bool = False) -> None:
        """Build the lattice of a field of this degree (cells=True: one point per cell)."""
        check(mesh)
        ncells = mesh.topology.index_map(2).size_local
        nodes = mesh.geometry.dofmaps[0][:ncells]
        x = mesh.geometry.x
        v = x[nodes]  # (ncells, 3 vertices, 3)
        self.ncells, self.degree = ncells, degree
        self.per_cell = cells or degree == 0  # one value per cell, at its centroid
        if self.per_cell:
            ref, sub = np.array([[1.0, 1.0]]) / 3, None
        else:
            ref, sub = reference(degree)
        pts = (
            v[:, None, 0]
            + ref[None, :, 0, None] * (v[:, None, 1] - v[:, None, 0])
            + ref[None, :, 1, None] * (v[:, None, 2] - v[:, None, 0])
        )
        self.points = np.ascontiguousarray(pts.reshape(-1, 3))
        self.cells = np.repeat(np.arange(ncells, dtype=np.int32), len(ref))
        self.mesh_x, self.mesh_y = x[:, 0] * MM, x[:, 1] * MM
        self.mesh_triangles = np.asarray(nodes, dtype=np.int64)
        if sub is None:
            self.x, self.y, self.triangles = self.mesh_x, self.mesh_y, self.mesh_triangles
        else:
            offsets = np.arange(ncells, dtype=np.int64)[:, None, None] * len(ref)
            self.triangles = (sub[None] + offsets).reshape(-1, 3)
            self.x, self.y = self.points[:, 0] * MM, self.points[:, 1] * MM

    def values(self, f: dolfinx.fem.Function) -> np.ndarray:
        """Return f at the lattice points (npoints, value size): each point in its own cell."""
        return f.eval(self.points, self.cells).reshape(len(self.points), -1)


_LATTICES: dict[tuple[int, int, bool], tuple[Any, Lattice]] = {}


def lattice(mesh: dolfinx.mesh.Mesh, degree: int, cells: bool = False) -> Lattice:
    """Return the Lattice of this mesh and degree, built once and reused (the last few meshes)."""
    key = (id(mesh), degree, cells)
    hit = _LATTICES.get(key)
    if hit is not None and hit[0] is mesh:
        return hit[1]
    out = Lattice(mesh, degree, cells)
    if len(_LATTICES) >= 8:
        _LATTICES.pop(next(iter(_LATTICES)))
    _LATTICES[key] = (mesh, out)
    return out
