"""Test meshes, and the integral of 1 over every measure of each: run serially (imported by the
tests) and under mpirun (python tests/meshes.py MSH prints JSON), to compare the two."""

import json
import sys

import dolfinx
import gmsh
from mpi4py import MPI

from amphix.core.mesh import Measures, build

ANNULUS = {
    "shape": "annulus",
    "radius": 1.0,
    "inner": {"center": [0.3, 0], "radius": 0.3},
    "h": 0.2,
}


def write_two_subdomains(path):
    """A .msh of two unit squares: cells "tissue" (5) and "fluid" (7), boundary "skull" (3)."""
    if MPI.COMM_WORLD.rank == 0:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        left, right = occ.addRectangle(0, 0, 0, 1, 1), occ.addRectangle(1, 0, 0, 1, 1)
        occ.fragment([(2, left)], [(2, right)])
        occ.synchronize()
        gmsh.model.addPhysicalGroup(2, [left], 5, name="tissue")
        gmsh.model.addPhysicalGroup(2, [right], 7, name="fluid")
        outer = [e for _, e in gmsh.model.getBoundary([(2, left), (2, right)], oriented=False)]
        gmsh.model.addPhysicalGroup(1, outer, 3, name="skull")
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.25)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(path))
        gmsh.finalize()
    MPI.COMM_WORLD.barrier()
    return path


def domains(msh):
    """Every mesh source."""
    return {
        "rectangle": build({"shape": "rectangle", "corners": [[0, 0], [2, 1]], "n": [6, 4]}),
        "box": build({"shape": "box", "corners": [[0, 0, 0], [1, 1, 1]], "n": [2, 2, 2]}),
        "annulus": build(ANNULUS),
        "shell": build(
            {
                "shape": "shell",
                "radius": 1.0,
                "inner": {"center": [0.3, 0, 0], "radius": 0.3},
                "h": 0.5,
            }
        ),
        "msh": build({"shape": "gmsh", "file": str(msh)}),
    }


def integrals(domain):
    """The integral of 1 over dx, dS, every named ds and every named dx, summed over the ranks."""
    m, mesh = Measures(domain), domain.mesh
    one = dolfinx.fem.Constant(mesh, 1.0)

    def total(form):
        return mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(form)), op=MPI.SUM)

    out = {"dx": total(one * m.dx()), "dS": total(one * m.dS), "ds": total(one * m.ds())}
    out |= {f"ds_{n}": total(one * m.ds(n)) for n in domain.facet_names}
    out |= {f"dx_{n}": total(one * m.dx(n)) for n in domain.cell_names}
    return out


def ghosts(domain):
    """The smallest number of ghost cells on any rank (0 in serial; > 0 when facets are ghosted)."""
    mesh = domain.mesh
    return mesh.comm.allreduce(mesh.topology.index_map(mesh.topology.dim).num_ghosts, op=MPI.MIN)


if __name__ == "__main__":
    out = {k: integrals(d) | {"ghosts": ghosts(d)} for k, d in domains(sys.argv[1]).items()}
    if MPI.COMM_WORLD.rank == 0:
        print(json.dumps(out))
