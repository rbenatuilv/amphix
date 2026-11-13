"""core.dg and core.spaces: jumps vanish for continuous fields; the mixed space splits and joins."""

import dolfinx
import numpy as np
import pytest
import ufl

from amphix.core.dg import penalty, sym_bnd, sym_jump
from amphix.core.mesh import Measures, build
from amphix.core.spaces import MixedSpace, dg_element

DOMAIN = build({"shape": "rectangle", "corners": [[0, 0], [1, 1]], "n": [3, 3]})
MESH = DOMAIN.mesh


def total(form):
    return dolfinx.fem.assemble_scalar(dolfinx.fem.form(form))


def vector_dg(expr, degree=1):
    V = dolfinx.fem.functionspace(MESH, ("DG", degree, (2,)))
    f = dolfinx.fem.Function(V)
    f.interpolate(dolfinx.fem.Expression(expr, V.element.interpolation_points))
    return f


def test_jumps():
    m, x = Measures(DOMAIN), ufl.SpatialCoordinate(MESH)
    smooth = vector_dg(ufl.as_vector([x[0] + 2 * x[1], x[0] * x[1]]), degree=2)  # continuous
    assert total(ufl.inner(sym_jump(smooth, m.n), sym_jump(smooth, m.n)) * m.dS) < 1e-28
    cells = dolfinx.fem.functionspace(MESH, ("DG", 0, (2,)))
    rough = dolfinx.fem.Function(cells)
    rough.x.array[:] = np.arange(rough.x.array.size)  # a different constant on every cell
    assert total(ufl.inner(sym_jump(rough, m.n), sym_jump(rough, m.n)) * m.dS) > 1e-3
    c = ufl.as_vector([1.0, 2.0])  # sym(c (x) n) has trace c.n, whose boundary integral is 0
    assert abs(total(ufl.tr(sym_bnd(c, m.n)) * m.ds())) < 1e-14
    assert total(ufl.tr(sym_bnd(c, m.n)) * m.ds("right")) == pytest.approx(1.0)  # n = (1, 0)
    assert penalty(10.0, 2) == 40.0


def test_mixed_space():
    space = MixedSpace(MESH, {"d": dg_element(MESH, 2, vector=True), "p": dg_element(MESH, 1)})
    ncells = MESH.topology.index_map(2).size_local
    assert space.names == ["d", "p"] and space.ndofs == ncells * (6 * 2 + 3)
    d, p = space.dofs["d"], space.dofs["p"]
    assert len(d) == ncells * 12 and len(np.intersect1d(d, p)) == 0  # disjoint, all of W
    assert (
        len(np.union1d(d, p)) == space.W.dofmap.index_map.size_local * space.W.dofmap.index_map_bs
    )
    assert len(space.owned_dofs("d")) + len(space.owned_dofs("p")) == space.ndofs  # serial
    assert set(space.trial()) == set(space.test()) == {"d", "p"}
    fields = space.functions()
    x = ufl.SpatialCoordinate(MESH)
    fields["d"].interpolate(
        dolfinx.fem.Expression(
            ufl.as_vector([x[0], x[1] ** 2]), space.V["d"].element.interpolation_points
        )
    )
    fields["p"].x.array[:] = np.linspace(0, 1, fields["p"].x.array.size)
    mixed = dolfinx.fem.Function(space.W)
    space.join(fields, mixed)
    back = space.functions()
    space.split(mixed, back)
    for name in ("d", "p"):
        assert np.array_equal(back[name].x.array, fields[name].x.array)
    assert total(ufl.inner(mixed.sub(0) - fields["d"], mixed.sub(0) - fields["d"]) * ufl.dx) < 1e-28
