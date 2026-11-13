"""physics.poro.TwoField: the discrete solution does not depend on the scales."""

import numpy as np
import ufl
from cases import annulus, cell_values

import amphix.physics.poro  # noqa: F401 (registers poro2)
from amphix.config import from_dict
from amphix.core.mesh import build
from amphix.core.model import make_model
from amphix.post.integrals import BoundaryIntegrals

FIELDS = ("d", "p", "flux")


def run(problem, domain=None):
    """Every field per cell and every boundary flux, after every step."""
    model = make_model(problem, domain)
    flows = BoundaryIntegrals(
        ufl.inner(model.quantity("flux"), model.measures.n), model.domain, model.measures
    )
    out = {"t": []}
    for t in model.steps():
        out["t"].append(t)
        for q in [*model.fields, "flux"]:
            out.setdefault(f"field_{q}", []).append(cell_values(model.quantity(q)))
        model.quantity("flux")  # refresh the SI copy the forms of `flows` are built on
        for name, value in flows().items():
            out.setdefault(f"bflux_{name}", []).append(value)
    return model, {k: np.array(v) for k, v in out.items()}


def test_the_scales_do_not_change_the_solution():
    """Every form is invariant under the non-dimensionalisation, so two sets of scales give the
    same SI solution. One mesh for both, rescaled: gmsh run in other units places the vertices
    slightly differently."""
    tree = annulus("poro2", h=2e-3)
    runs = []
    for scales in ({"L0": 1e-2, "T0": 1.0, "P0": 1500.0}, {"L0": 1e-3, "T0": 0.5, "P0": 1.0}):
        problem = from_dict({**tree, "scales": scales}, "scales")
        domain = build(problem.mesh, 1e-2)
        domain.mesh.geometry.x[:] *= 1e-2 / scales["L0"]  # the same mesh in the new units
        domain = type(domain)(
            domain.mesh,
            scales["L0"],
            domain.facet_tags,
            domain.facet_names,
            domain.cell_tags,
            domain.cell_names,
        )
        runs.append(run(problem, domain)[1])
    for q in FIELDS:
        a, b = runs[0][f"field_{q}"], runs[1][f"field_{q}"]
        assert np.abs(a - b).max() < 1e-10 * np.abs(a).max(), q
