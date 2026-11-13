"""Problems and helpers shared by the tests: the brain annulus under a cardiac pulse."""

import numpy as np

import amphix  # noqa: F401 (registers the models and the presets)
from amphix.brain import geometries


def annulus(model="poro2", deg=1, h=5e-4, steps=5):
    """The annulus of brain.geometries, cvr22 tissue, the ventricle pressure pulsing (Pa)."""
    tree = {
        "model": model,
        "scales": {"L0": 1e-2, "T0": 1.0, "P0": 1500.0},
        "material": {"preset": "cvr22"},
        "loads": {"pc": {"waveform": "cardiac_pulse"}},
        "boundaries": {
            "ventricle": {"pressure": "pc", "normal_traction": "-pc"},
            "pial": {"pressure": 0, "displacement": [0, 0]},
        },
        "discretization": {"deg": deg},
        "time": {"T": steps * 0.01, "dt": 0.01},
        "mesh": geometries.annulus(h),
    }
    if model == "poro2":
        tree["time"]["theta"] = 0.5
    return tree


def cell_order(mesh):
    """The permutation that puts the owned cells in their input (gmsh) order."""
    nloc = mesh.topology.index_map(mesh.topology.dim).size_local
    return np.argsort(np.asarray(mesh.topology.original_cell_index[:nloc]))


def cell_values(f):
    """The dofs of a DG Function on each owned cell, cells in input order: (ncells, dofs)."""
    mesh, dofmap = f.function_space.mesh, f.function_space.dofmap
    order = cell_order(mesh)
    nloc = len(order)
    return f.x.array.reshape(-1, dofmap.bs)[dofmap.list[:nloc]].reshape(nloc, -1)[order]
