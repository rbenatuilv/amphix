"""Meshes made with gmsh: round shapes, and any .msh file with physical groups.

    disc (2D), ball (3D)       boundary  outer
    annulus (2D), shell (3D)   boundaries outer, inner: a disc or ball with a hole (off-centre ok)
    from_msh                   the named physical groups of the file: of dimension dim - 1 the
                               boundaries, of dimension dim the subdomains (cell tags)

Sizes are in SI; the mesh is returned in solver units (coordinates / L0). Round shapes use the OCC
kernel with Mesh.CharacteristicLengthMax = h / L0; the larger circle or sphere is `outer`. Every
mesh ghosts the cells across shared facets (DG needs both sides of each facet).
"""

from collections.abc import Callable, Sequence
from pathlib import Path

import dolfinx
import dolfinx.io.gmsh
import numpy as np
from mpi4py import MPI

from amphix.core.mesh.domain import Domain, all_cells


def ghosted_partitioner() -> Callable:
    """Return a partitioner that ghosts the cells across shared facets (DG needs both sides)."""
    # the (mode, max_facet_to_cell_links) overload is registered by singledispatch: mypy misses it
    ghost = dolfinx.mesh.GhostMode.shared_facet
    return dolfinx.mesh.create_cell_partitioner(ghost, 2)  # type: ignore[call-arg]


def round_shape(
    shape: str,
    radius: float,
    h: float,
    center: Sequence[float] | None = None,
    inner: dict | None = None,
    L0: float = 1.0,
) -> Domain:
    """Return a disc, annulus, ball or shell; inner = {center, radius} for the hole (in m)."""
    import gmsh  # noqa: PLC0415 (only round shapes need gmsh)

    dim = 3 if shape in ("ball", "shell") else 2
    if (inner is None) != (shape in ("disc", "ball")):
        raise ValueError(f"{shape}: inner (the hole) is {'not ' * (inner is not None)}needed")
    c = np.zeros(3)
    c[:dim] = np.asarray(center if center is not None else [0.0] * dim, dtype=float) / L0
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        if dim == 2:
            add = lambda c, r: occ.addDisk(*c, r, r)  # noqa: E731
        else:
            add = lambda c, r: occ.addSphere(*c, r)  # noqa: E731
        body = [(dim, add(c, radius / L0))]
        if inner is not None:
            ci = np.zeros(3)
            ci[:dim] = np.asarray(inner["center"], dtype=float) / L0
            body, _ = occ.cut(body, [(dim, add(ci, inner["radius"] / L0))])
        occ.synchronize()
        gmsh.model.addPhysicalGroup(dim, [b[1] for b in body], 1)
        bnd = [e for _, e in gmsh.model.getBoundary(body, oriented=False)]
        box = {e: gmsh.model.getBoundingBox(dim - 1, e) for e in bnd}
        size = {e: box[e][3] - box[e][0] for e in bnd}
        names = {}
        for tag, e in enumerate(sorted(bnd, key=lambda e: -size[e]), start=1):
            names["outer" if tag == 1 else "inner"] = tag  # the larger circle / sphere: outer
            gmsh.model.addPhysicalGroup(dim - 1, [e], tag)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h / L0)
        gmsh.model.mesh.generate(dim)
        md = dolfinx.io.gmsh.model_to_mesh(
            gmsh.model, MPI.COMM_WORLD, 0, gdim=dim, partitioner=ghosted_partitioner()
        )
    finally:
        gmsh.finalize()
    assert md.facet_tags is not None and md.cell_tags is not None  # both groups were added
    return Domain(md.mesh, L0, md.facet_tags, names, md.cell_tags, {})


def from_msh(
    path: str | Path,
    dim: int = 2,
    facets: dict[str, int] | None = None,
    cells: dict[str, int] | None = None,
    L0: float = 1.0,
) -> Domain:
    """Return the mesh of a .msh file (in m) with its named physical groups.

    facets / cells name physical groups that have no name in the file ({name: tag}); a name given
    here wins over the file's.
    """
    md = dolfinx.io.gmsh.read_from_msh(
        Path(path), MPI.COMM_WORLD, 0, gdim=dim, partitioner=ghosted_partitioner()
    )
    md.mesh.geometry.x[:] /= L0
    groups = md.physical_groups
    facet_names = {n: pg.tag for n, pg in groups.items() if pg.dim == dim - 1} | (facets or {})
    cell_names = {n: pg.tag for n, pg in groups.items() if pg.dim == dim} | (cells or {})
    if md.facet_tags is None:
        raise ValueError(f"{path}: no physical groups of dimension {dim - 1} (the boundaries)")
    cell_tags = md.cell_tags if md.cell_tags is not None else all_cells(md.mesh)
    return Domain(md.mesh, L0, md.facet_tags, facet_names, cell_tags, cell_names)
