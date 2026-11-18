"""Full fields saved at chosen times, read back as Functions on any number of ranks.

    writer = FieldWriter(model.domain, {"p": lambda: model.quantity("p"), "cflux": rec.compute})
    writer(t)                                   # after a step: store every field's dofs (SI)
    writer.save("results/run_fields.npz", model.problem)   # rank 0 writes; the path on every rank
    fl = load_fields("results/run_fields.npz")
    fl.function("p", 0), fl.average("cflux", 2.0, 3.0), fl.domain   # mesh in m, with its tags
    restore("results/run_fields.npz", -1, ["d", "p"], model.domain)  # on the run's own mesh

The file holds the mesh (vertices in m), its facet and cell tags with their names, and, for each
quantity, the DG dofs of every cell at every saved time: the exact Function, not a sampling of it.
A field that is not DG (cflux: Raviart-Thomas) is copied exactly into DG of its polynomial degree.
The cells are written in their input order, so a file does not depend on the number of ranks that
wrote it.

The reloaded Domain is in SI (L0 = 1, coordinates in m), ghosted across shared facets, with the
boundary and subdomain names it had. Facet tags need a mesh of degree 1.

restore() puts saved quantities back on the mesh of a new run instead, to start from them. That
mesh must be the saved one: the same cells in the same input order, each with the same vertices
(within 1e-9 of the largest coordinate), on any number of ranks. Each cell then gets back its dofs
as they were saved; another mesh is an error, never interpolated onto.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import basix
import basix.ufl
import dolfinx
import numpy as np
import ufl
import yaml
from mpi4py import MPI

from amphix.core.mesh import Domain, ghosted_partitioner
from amphix.io.results import problem_text, written

Source = Callable[[], dolfinx.fem.Function]


def dg_space(V: dolfinx.fem.FunctionSpace) -> dolfinx.fem.FunctionSpace:
    """Return V if it is DG, else DG of the same polynomial degree and shape (which contains V)."""
    el = V.ufl_element()
    if el.discontinuous:
        return V
    return dolfinx.fem.functionspace(V.mesh, ("DG", el.embedded_superdegree, V.value_shape))


def copy_to(g: dolfinx.fem.Function, f: dolfinx.fem.Function, factor: float = 1.0) -> None:
    """Set g = factor * f, with g in dg_space of f's space (an exact copy)."""
    if g.function_space == f.function_space:
        g.x.array[:] = factor * f.x.array
    else:
        g.interpolate(f)
        g.x.array[:] *= factor
    g.x.scatter_forward()


def _mesh_data(domain: Domain) -> dict[str, Any] | None:
    """Gather the mesh and its tags on rank 0, cells in input order (None on the other ranks)."""
    mesh, comm = domain.mesh, domain.mesh.comm
    tdim, gdim = mesh.topology.dim, mesh.geometry.dim
    nloc = mesh.topology.index_map(tdim).size_local
    gi = np.asarray(mesh.geometry.input_global_indices, dtype=np.int64)
    orig = np.asarray(mesh.topology.original_cell_index[:nloc], dtype=np.int64)
    cell_tag = dict(
        zip(domain.cell_tags.indices.tolist(), domain.cell_tags.values.tolist(), strict=True)
    )
    tags = domain.facet_tags
    facet_nodes = gi[dolfinx.mesh.entities_to_geometry(mesh, tdim - 1, tags.indices)]
    local = (
        gi,
        mesh.geometry.x[:, :gdim] * domain.L0,
        gi[mesh.geometry.dofmaps[0][:nloc]],
        orig,
        np.array([cell_tag.get(c, 0) for c in range(nloc)], dtype=np.int32),
        np.sort(facet_nodes, axis=1),
        tags.values,
    )
    parts = comm.gather(local)
    if comm.rank != 0:
        return None
    ids, first = np.unique(np.concatenate([p[0] for p in parts]), return_index=True)  # shared nodes
    orig_all = np.concatenate([p[3] for p in parts])
    order = np.argsort(orig_all)  # gathered rows -> input order
    facets, idx = np.unique(np.concatenate([p[5] for p in parts]), axis=0, return_index=True)
    cmap = mesh.geometry.cmaps[0]
    return {
        "x": np.concatenate([p[1] for p in parts])[first],
        "cells": np.searchsorted(ids, np.concatenate([p[2] for p in parts]))[order],
        "cell_values": np.concatenate([p[4] for p in parts])[order],
        "facets": np.searchsorted(ids, facets),
        "facet_values": np.concatenate([p[6] for p in parts])[idx],
        "cell": mesh.topology.cell_name(),
        "gdeg": cmap.degree,
        "gvariant": int(cmap.variant),
        "facet_names": json.dumps(domain.facet_names),
        "cell_names": json.dumps(domain.cell_names),
        "_order": order,
    }


class FieldWriter:
    """Collect the DG dofs of some SI Functions at each call; save() writes them (rank 0)."""

    def __init__(self, domain: Domain, sources: dict[str, Source]) -> None:
        """Take {name: a callable returning the current SI Function of that quantity}."""
        self.domain, self.sources = domain, dict(sources)
        self.mesh_data = _mesh_data(domain)
        self.t: list[float] = []
        self.data: dict[str, list] = {n: [] for n in sources}
        self._dg: dict[str, dolfinx.fem.Function] = {}
        self.meta = {}
        for (
            name,
            source,
        ) in self.sources.items():  # known before any snapshot: save() works with none
            V = source().function_space
            if not V.ufl_element().discontinuous:  # cflux (RT): its exact DG copy
                self._dg[name] = dolfinx.fem.Function(dg_space(V))
                V = self._dg[name].function_space
            self.meta[name] = (
                V.ufl_element().degree,
                V.value_shape,
                V.dofmap.list.shape[1] * V.dofmap.bs,
            )

    def nbytes(self, ntimes: int) -> int:
        """Return the memory [bytes] that ntimes calls hold until save() (on rank 0)."""
        ncells = self.domain.mesh.topology.index_map(self.domain.dim).size_global
        return 8 * ntimes * ncells * sum(width for _, _, width in self.meta.values())

    def __call__(self, t: float) -> None:
        """Store every field now (at time t [s])."""
        comm = self.domain.mesh.comm
        nloc = self.domain.mesh.topology.index_map(self.domain.dim).size_local
        self.t.append(t)
        for name, source in self.sources.items():
            f = source()
            if name in self._dg:
                copy_to(self._dg[name], f)
                f = self._dg[name]
            dofmap = f.function_space.dofmap
            vals = f.x.array.reshape(-1, dofmap.bs)[dofmap.list[:nloc]].reshape(nloc, -1)
            parts = comm.gather(vals)
            if comm.rank == 0:
                self.data[name].append(np.concatenate(parts)[self.mesh_data["_order"]])

    def save(self, path: str | Path, problem: Any) -> Path:
        """Write the file (rank 0); return its path on every rank once it is written."""
        return written(path, self.domain.mesh.comm, lambda target: self._write(target, problem))

    def _write(self, path: Path, problem: Any) -> None:
        """Write the file (rank 0 holds the data)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        mesh = {k: v for k, v in self.mesh_data.items() if k != "_order"}
        out = dict(
            mesh,
            snap_t=np.array(self.t),
            names=np.array(list(self.sources)),
            problem=problem_text(problem),
        )
        ncells = len(mesh["cells"])
        for name, (deg, shape, width) in self.meta.items():
            out[name] = np.array(self.data[name]).reshape(
                -1, ncells, width
            )  # (nsnap, ncells, ndofs)
            out[f"{name}_deg"], out[f"{name}_shape"] = deg, np.array(shape, dtype=int)
        np.savez(path, **out)


class Fields:
    """A saved fields file read back: domain (m, with tags), snap_t, names, problem, functions."""

    def __init__(self, path: str | Path, comm: MPI.Comm = MPI.COMM_WORLD) -> None:
        """Read on rank 0, distribute the mesh and the tags (any number of ranks)."""
        self.path, self.comm = Path(path), comm
        f = np.load(self.path, allow_pickle=False) if comm.rank == 0 else None
        self._file = f
        small = None
        if f is not None:
            small = {
                k: f[k].item() if f[k].ndim == 0 else f[k]
                for k in (
                    "cell",
                    "gdeg",
                    "gvariant",
                    "snap_t",
                    "names",
                    "problem",
                    "facet_names",
                    "cell_names",
                )
            }
            small["nn"], small["gdim"] = f["cells"].shape[1], f["x"].shape[1]
            for n in small["names"]:
                small[f"{n}_meta"] = (int(f[f"{n}_deg"]), tuple(int(v) for v in f[f"{n}_shape"]))
        small = comm.bcast(small)
        self.snap_t, self.names = np.asarray(small["snap_t"]), [str(n) for n in small["names"]]
        self.problem = yaml.safe_load(str(small["problem"]))
        self._meta = {n: small[f"{n}_meta"] for n in self.names}
        self.domain = self._domain(small)
        nloc = self.domain.mesh.topology.index_map(self.domain.dim).size_local
        orig = np.asarray(self.domain.mesh.topology.original_cell_index[:nloc], dtype=np.int64)
        self._orig = comm.gather(orig)  # rank 0: the rows of each rank
        self._spaces: dict[str, Any] = {}
        self._cache: dict[str, np.ndarray] = {}

    def _domain(self, small: dict) -> Domain:
        f, comm, gdim = self._file, self.comm, int(small["gdim"])
        cells = (
            f["cells"].astype(np.int64) if f is not None else np.zeros((0, small["nn"]), np.int64)
        )
        x = f["x"] if f is not None else np.zeros((0, gdim))
        e = ufl.Mesh(
            basix.ufl.element(
                "Lagrange",
                str(small["cell"]),
                int(small["gdeg"]),
                shape=(gdim,),
                lagrange_variant=basix.LagrangeVariant(int(small["gvariant"])),
            )
        )
        mesh = dolfinx.mesh.create_mesh(comm, cells, e, x, partitioner=ghosted_partitioner())
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim - 1, tdim)
        facets = f["facets"].astype(np.int64) if f is not None else np.zeros((0, tdim), np.int64)
        values = f["facet_values"] if f is not None else np.zeros(0, np.int32)
        local, vals = dolfinx.io.distribute_entity_data(
            mesh, tdim - 1, facets, values.astype(np.int32)
        )
        facet_tags = dolfinx.mesh.meshtags_from_entities(
            mesh, tdim - 1, dolfinx.graph.adjacencylist(local), vals.astype(np.int32)
        )
        cell_values = comm.bcast(f["cell_values"] if f is not None else None)
        nall = mesh.topology.index_map(tdim).size_local + mesh.topology.index_map(tdim).num_ghosts
        orig = np.asarray(mesh.topology.original_cell_index, dtype=np.int64)[:nall]
        cell_tags = dolfinx.mesh.meshtags(
            mesh, tdim, np.arange(len(orig), dtype=np.int32), cell_values[orig].astype(np.int32)
        )
        return Domain(
            mesh,
            1.0,
            facet_tags,
            json.loads(small["facet_names"]),
            cell_tags,
            json.loads(small["cell_names"]),
        )

    def space(self, name: str) -> dolfinx.fem.FunctionSpace:
        """Return the DG space of a stored quantity."""
        if name not in self._spaces:
            deg, shape = self._meta[name]
            self._spaces[name] = dolfinx.fem.functionspace(self.domain.mesh, ("DG", deg, shape))
        return self._spaces[name]

    def function(self, name: str, i: int) -> dolfinx.fem.Function:
        """Return the quantity `name` at saved time i, as a new Function (SI)."""
        return self._function(name, lambda data: data[i])

    def average(self, name: str, t_from: float, t_to: float) -> dolfinx.fem.Function:
        """Return the mean of `name` over the saved times t_from < t <= t_to, as a new Function.

        For cflux saved at every step of a window, the field at t is the velocity over the step that
        ends at t, so this is exactly its time average over [t_from, t_to].
        """
        tol = 1e-9 * max(1.0, abs(t_to))
        sel = (self.snap_t > t_from + tol) & (self.snap_t <= t_to + tol)
        if not sel.any():
            raise ValueError(f"average: no saved time in ({t_from}, {t_to}]")
        return self._function(name, lambda data: data[sel].mean(axis=0))

    def _function(
        self, name: str, pick: Callable[[np.ndarray], np.ndarray]
    ) -> dolfinx.fem.Function:
        V = self.space(name)
        parts = None
        if self.comm.rank == 0:
            if name not in self._cache:
                self._cache = {name: self._file[name]}  # one quantity in memory at a time
            vals = pick(self._cache[name])
            parts = [vals[o] for o in self._orig]
        mine = self.comm.scatter(parts)
        u = dolfinx.fem.Function(V, name=name)
        nloc, bs = len(mine), V.dofmap.bs
        u.x.array.reshape(-1, bs)[V.dofmap.list[:nloc]] = mine.reshape(nloc, -1, bs)
        u.x.scatter_forward()
        return u


def load_fields(path: str | Path, comm: MPI.Comm = MPI.COMM_WORLD) -> Fields:
    """Return a saved fields file read back (see Fields)."""
    return Fields(path, comm)


def _mesh_error(f: Any, mesh: dolfinx.mesh.Mesh) -> str | None:
    """Return why the saved mesh is not this one by its sizes (rank 0), or None."""
    tdim = mesh.topology.dim
    saved = (f["cells"].shape[0], f["x"].shape[0], f["x"].shape[1], f["cells"].shape[1])
    here = (
        mesh.topology.index_map(tdim).size_global,
        mesh.geometry.index_map().size_global,
        mesh.geometry.dim,
        mesh.geometry.dofmaps[0].shape[1],
    )
    if saved == here:
        return None
    return (
        f"{saved[0]} cells, {saved[1]} nodes in {saved[2]}D ({saved[3]} per cell) saved; "
        f"this mesh has {here[0]}, {here[1]} in {here[2]}D ({here[3]})"
    )


def restore(
    path: str | Path, index: int, names: list[str], domain: Domain
) -> dict[str, dolfinx.fem.Function]:
    """Return saved quantities at saved time `index` as SI Functions on the domain's own mesh.

    The file must hold the domain's mesh (see the module docstring), else ValueError. Each
    Function is in DG of the saved degree, its cells' dofs copied from the file.
    """
    mesh, comm = domain.mesh, domain.mesh.comm
    tdim, gdim = mesh.topology.dim, mesh.geometry.dim
    nloc = mesh.topology.index_map(tdim).size_local
    orig = comm.gather(np.asarray(mesh.topology.original_cell_index[:nloc], dtype=np.int64))
    head, parts = None, None
    if comm.rank == 0:
        with np.load(path, allow_pickle=False) as f:
            error = _mesh_error(f, mesh)
            meta = {n: (int(f[f"{n}_deg"]), tuple(int(v) for v in f[f"{n}_shape"])) for n in names}
            head = (error, meta, float(np.abs(f["x"]).max()))
            if error is None:
                x, cells = f["x"], f["cells"]
                values = {n: f[n][index] for n in names}
                parts = [(x[cells[o]], {n: v[o] for n, v in values.items()}) for o in orig]
    error, meta, size = comm.bcast(head)
    if error is not None:
        raise ValueError(f"{path}: not the mesh of this run: {error}")
    x_saved, values = comm.scatter(parts)
    x_here = mesh.geometry.x[mesh.geometry.dofmaps[0][:nloc], :gdim] * domain.L0
    moved = float(np.abs(x_here - x_saved).max(initial=0.0))
    moved = comm.allreduce(moved, op=MPI.MAX)
    if moved > 1e-9 * size:
        raise ValueError(
            f"{path}: not the mesh of this run: a cell vertex is {moved:.3g} m from the saved one"
        )
    out = {}
    for name, (deg, shape) in meta.items():
        V = dolfinx.fem.functionspace(mesh, ("DG", deg, shape))
        u = dolfinx.fem.Function(V, name=name)
        bs = V.dofmap.bs
        u.x.array.reshape(-1, bs)[V.dofmap.list[:nloc]] = values[name].reshape(nloc, -1, bs)
        u.x.scatter_forward()
        out[name] = u
    return out
