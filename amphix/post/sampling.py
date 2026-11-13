"""Point values of fields: probes, lines, 2D grids and 3D slices, in SI, in parallel.

    probe = Points(domain, [[-5e-3, 0], [8e-3, 0]])     # positions in m
    probe(model.quantity("p"))                          # (npoints,) or (npoints, value size)

Each point is evaluated by the rank that owns its cell, then the values are combined on every rank,
so every rank gets the same array. A point on the boundary between two ranks' cells is averaged
(the field is single-valued there unless it jumps). Points outside the mesh give NaN.

The point sets (all in m):

    line_points({from: [...], to: [...], n: 101})     n points on a segment
    grid_points(domain, n)                            2D: n x n cell-centred points over the
                                                      bounding box of the mesh
    slice_points(domain, {normal: z, at: 0.0, n: 81}) 3D: n x n points on a plane; reshape(n, n)
                                                      gives [i_b, i_a], (a, b) the in-plane axes in
                                                      xyz order (normal z: (x, y))
"""

from collections.abc import Sequence
from typing import Any

import dolfinx
import numpy as np
from mpi4py import MPI

from amphix.core.mesh import Domain


class Points:
    """Evaluate SI Functions at fixed points given in m (parallel-safe, NaN outside the mesh)."""

    def __init__(self, domain: Domain, points: Sequence[Sequence[float]] | np.ndarray) -> None:
        """Locate the cell of each point once (on the rank that owns it)."""
        mesh = domain.mesh
        pts = np.atleast_2d(np.asarray(points, dtype=float)) / domain.L0  # solver units
        self.n = len(pts)
        self.pts = np.zeros((self.n, 3))
        self.pts[:, : pts.shape[1]] = pts
        self.comm = mesh.comm
        tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
        candidates = dolfinx.geometry.compute_collisions_points(tree, self.pts)
        colliding = dolfinx.geometry.compute_colliding_cells(mesh, candidates, self.pts)
        nloc = mesh.topology.index_map(mesh.topology.dim).size_local
        idx, cells = [], []
        for i in range(self.n):
            owned = [c for c in colliding.links(i) if c < nloc]
            if owned:
                idx.append(i)
                cells.append(owned[0])
        self.idx, self.cells = np.array(idx, dtype=np.int64), np.array(cells, dtype=np.int32)
        count = np.zeros(self.n)
        count[self.idx] = 1
        self.count = np.empty(self.n)
        self.comm.Allreduce(count, self.count, op=MPI.SUM)

    def __call__(self, f: dolfinx.fem.Function) -> np.ndarray:
        """Return (npoints, value size) values of f; value size 1 is squeezed. NaN outside."""
        vs = f.function_space.value_size
        local = np.zeros((self.n, vs))
        if len(self.idx):
            local[self.idx] = f.eval(self.pts[self.idx], self.cells)
        out = np.empty_like(local)
        self.comm.Allreduce(local, out, op=MPI.SUM)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = out / self.count[:, None]  # a point on a rank boundary: the average
        out[self.count == 0] = np.nan
        return out[:, 0] if vs == 1 else out


def line_points(spec: dict[str, Any]) -> np.ndarray:
    """Return n points (default 101) from spec["from"] to spec["to"], in m."""
    a, b = np.asarray(spec["from"], dtype=float), np.asarray(spec["to"], dtype=float)
    s = np.linspace(0.0, 1.0, int(spec.get("n", 101)))[:, None]
    return a + s * (b - a)


def _bounding_box(domain: Domain) -> tuple[np.ndarray, np.ndarray]:
    """Return the bounding box of the mesh over every rank, in m."""
    x, comm = domain.mesh.geometry.x, domain.mesh.comm
    lo, hi = np.empty(3), np.empty(3)
    comm.Allreduce(x.min(axis=0), lo, op=MPI.MIN)
    comm.Allreduce(x.max(axis=0), hi, op=MPI.MAX)
    return lo * domain.L0, hi * domain.L0


def grid_points(domain: Domain, n: int) -> np.ndarray:
    """Return an n x n cell-centred grid over the bounding box of a 2D mesh, in m."""
    if domain.dim != 2:
        raise ValueError("grid_points: 2D meshes only (3D: slice_points)")
    lo, hi = _bounding_box(domain)
    s = (np.arange(n) + 0.5) / n
    gx, gy = np.meshgrid(lo[0] + s * (hi[0] - lo[0]), lo[1] + s * (hi[1] - lo[1]))
    return np.column_stack([gx.ravel(), gy.ravel()])


def slice_points(domain: Domain, spec: dict[str, Any]) -> np.ndarray:
    """Return n x n points (default 81) on the plane {x_normal = at} of a 3D mesh, in m."""
    if domain.dim != 3:
        raise ValueError("slice_points: 3D meshes only (2D: grid_points)")
    axis = "xyz".index(spec["normal"])
    a, b = [i for i in range(3) if i != axis]
    n = int(spec.get("n", 81))
    lo, hi = _bounding_box(domain)
    s = (np.arange(n) + 0.5) / n
    ga, gb = np.meshgrid(lo[a] + s * (hi[a] - lo[a]), lo[b] + s * (hi[b] - lo[b]))
    pts = np.zeros((n * n, 3))
    pts[:, a], pts[:, b], pts[:, axis] = ga.ravel(), gb.ravel(), float(spec["at"])
    return pts
