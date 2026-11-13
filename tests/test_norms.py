"""post.norms: SI norms of known functions, errors, Difference across non-matching meshes (the
mesh-independence check), observed orders."""

import dolfinx
import numpy as np
import pytest
import ufl

from amphix.core.mesh import build, cell_size
from amphix.post.norms import Difference, error, norm, orders

L0 = 1e-2
RECT = build({"shape": "rectangle", "corners": [[0, 0], [2e-2, 1e-2]], "n": [8, 4]}, L0=L0)
X = L0 * ufl.SpatialCoordinate(RECT.mesh)  # SI position


def dg(domain, degree, expr, shape=()):
    V = dolfinx.fem.functionspace(domain.mesh, ("DG", degree, shape))
    f = dolfinx.fem.Function(V)
    f.interpolate(dolfinx.fem.Expression(expr, V.element.interpolation_points))
    return f


def test_si_norms_of_known_functions():
    """||x||_L2^2 = Lx^3 Ly / 3 and |x|_H1^2 = Lx Ly, the rectangle being 2 cm x 1 cm."""
    lx, ly = 2e-2, 1e-2
    assert norm(X[0], "L2", RECT, 4) == pytest.approx(np.sqrt(lx**3 * ly / 3), rel=1e-12)
    assert norm(X[0], "H1", RECT, 4) == pytest.approx(np.sqrt(lx * ly), rel=1e-12)
    v = ufl.as_vector([X[1], 2 * X[0]])  # |v|^2 = y^2 + 4 x^2; |grad v|^2 = 1 + 4
    assert norm(v, "L2", RECT, 4) == pytest.approx(np.sqrt(lx * ly**3 / 3 + 4 * lx**3 * ly / 3))
    assert norm(v, "H1", RECT, 4) == pytest.approx(np.sqrt(5 * lx * ly))
    with pytest.raises(ValueError, match="norm kind"):
        norm(X[0], "H2", RECT)


def test_errors():
    f = dg(RECT, 2, X[0] * X[1] + X[0] ** 2)  # DG2 holds it: no error
    assert error(f, X[0] * X[1] + X[0] ** 2, "L2", RECT, 6) < 1e-17
    assert error(f, X[0] * X[1] + X[0] ** 2 + 1e-3, "L2", RECT, 6) == pytest.approx(
        1e-3 * np.sqrt(2e-4), rel=1e-10
    )  # a constant shift: |c| sqrt(area)


def test_difference_on_the_same_and_on_other_meshes():
    fine = build({"shape": "rectangle", "corners": [[0, 0], [2e-2, 1e-2]], "n": [12, 6]}, L0=L0)
    poly = lambda d: (L0 * ufl.SpatialCoordinate(d.mesh))[0] ** 2  # noqa: E731
    a, b = dg(RECT, 2, poly(RECT)), dg(fine, 2, poly(fine))
    diff = Difference(
        a.function_space, b.function_space, L0, ["L2", "H1"], degree=6, options={"relative": True}
    )
    out = diff(a, b)
    assert out["L2"] < 1e-12 and out["H1"] < 1e-11  # both meshes hold x^2 exactly
    smooth = lambda d: ufl.sin(300 * (L0 * ufl.SpatialCoordinate(d.mesh))[0])  # noqa: E731
    out = Difference(a.function_space, b.function_space, L0, ["L2"])(
        dg(RECT, 2, smooth(RECT)), dg(fine, 2, smooth(fine))
    )
    assert 1e-6 < out["L2"] < 1e-2  # two discretisations of a non-polynomial differ a little


def test_difference_across_curved_boundaries():
    """Two meshes of an annulus have different polygons: padding lets the points of one just
    outside the other be evaluated in the nearest cell (a linear field: exactly)."""
    ring = {"shape": "annulus", "radius": 1.0, "inner": {"center": [0.2, 0], "radius": 0.3}}
    coarse, fine = build({**ring, "h": 0.3}), build({**ring, "h": 0.15})
    lin = lambda d: 1 + 2 * ufl.SpatialCoordinate(d.mesh)[0] - ufl.SpatialCoordinate(d.mesh)[1]  # noqa: E731
    a, b = dg(coarse, 1, lin(coarse)), dg(fine, 1, lin(fine))
    diff = Difference(
        a.function_space,
        b.function_space,
        1.0,
        ["L2"],
        options={"padding": cell_size(coarse), "relative": True},
    )
    assert diff(a, b)["L2"] < 1e-12


def test_orders():
    h = np.array([0.4, 0.2, 0.1])
    assert np.allclose(orders(3 * h**2, h), [2.0, 2.0])
    assert np.allclose(orders([1.0, 0.5], [0.1, 0.05]), [1.0])
    assert np.isnan(orders([0.0, 0.0], [0.1, 0.05])[0])  # exact on both: no order
