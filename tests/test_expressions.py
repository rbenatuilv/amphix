"""core.expressions: every kind of value, in SI, and exact SI derivatives in t and x."""

import dolfinx
import numpy as np
import pytest
import ufl
from mpi4py import MPI

from amphix.core.expressions import Expressions, register_waveform
from amphix.core.units import Scales

MESH = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)  # solver units: [0, L0]^2 in SI
SCALES = Scales(L0=1.0e-2, T0=0.5, P0=1500.0)
X = SCALES.L0 * ufl.SpatialCoordinate(MESH)  # the SI position, written by hand
x, y = X[0], X[1]
T_STAR = 0.3
T = SCALES.T0 * T_STAR  # the SI time of the tests


def evaluator(**kw):
    return Expressions(MESH, SCALES, dolfinx.fem.Constant(MESH, T_STAR), **kw)


def integral(e):
    form = dolfinx.fem.form(e * ufl.dx(domain=MESH))
    return MESH.comm.allreduce(dolfinx.fem.assemble_scalar(form))


FLOOR = integral(ufl.inner(X, X))  # the size of x^2: the scale of a zero reference


def assert_same(a, b):
    """a == b as fields: the squared L2 norm of a - b is round-off."""
    err, scale = integral(ufl.inner(a - b, a - b)), integral(ufl.inner(b, b))
    assert err <= 1e-26 * max(scale, FLOOR), (err, scale)


def test_numbers_and_vectors():
    """Numbers stay Constants (never simplified away), also a formula that evaluates to a number."""
    ex = evaluator()
    c = ex(2.5)
    assert isinstance(c, dolfinx.fem.Constant) and float(c.value) == 2.5
    assert isinstance(ex("2 * pi"), dolfinx.fem.Constant) and isinstance(
        ex(0), dolfinx.fem.Constant
    )
    v = ex([1, 2.0])
    assert isinstance(v, dolfinx.fem.Constant) and v.value.tolist() == [1.0, 2.0]
    assert_same(ex([0, "t * y"]), ufl.as_vector([0, T * y]))
    assert ex(np.float64(1.5)).value == 1.5


def test_formulas_params_and_chained_loads():
    ex = evaluator(params={"A": 2}, loads={"ramp": "where(t < 1, t, 1)", "f": "A * ramp * x"})
    assert_same(ex("f"), 2 * T * x)
    assert_same(ex("sqrt(x**2 + y**2) - r"), 0 * x)
    assert_same(ex("max(x, y) + exp(-t)"), ufl.max_value(x, y) + ufl.exp(-T))


def test_z_only_in_3d():
    with pytest.raises(NameError):
        evaluator()("z")
    box = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    ex3 = Expressions(box, SCALES, dolfinx.fem.Constant(box, 0.0))
    assert ex3("z").ufl_shape == ()


def test_callables_and_ufl_get_si_arguments():
    ex = evaluator()
    assert_same(ex(lambda x_, t_: t_ * x_[0] ** 2), T * x**2)
    e = T * y
    assert ex(e) is e


def test_ddt_is_exact():
    ex = evaluator()
    assert_same(ex.ddt(ex("sin(2*pi*t) * x")), 2 * np.pi * ufl.cos(2 * np.pi * T) * x)
    zero = ex.ddt(ex([0.1, 0]))  # a fixed displacement has zero velocity, of the right shape
    assert isinstance(zero, dolfinx.fem.Constant) and zero.value.shape == (2,)
    for constant in (ex(3.0), ex("x * y"), ex(lambda x_, t_: x_[0])):  # constant in time
        zero = ex.ddt(constant)
        assert isinstance(zero, dolfinx.fem.Constant) and zero.value.shape == ()  # not ufl Zero
        assert float(zero.value) == 0.0


def test_grad_and_div_in_si():
    ex = evaluator()
    u = ex(["x**2 * y", "sin(t) * y"])
    assert_same(ex.grad(u), ufl.as_matrix([[2 * x * y, x**2], [0, ufl.sin(T)]]))
    assert_same(ex.div(u), 2 * x * y + ufl.sin(T))
    assert_same(ex.grad(u), ufl.grad(u) / SCALES.L0)  # grad on the mesh, / L0
    sigma = ex([["x * y", "x"], ["y**2", "t * x * y"]])
    assert_same(ex.div(sigma), ufl.as_vector([y, T * x]))  # row-wise, as ufl.div
    assert_same(ex.ddt(ex.div(u)), ufl.cos(T) + 0 * x)  # mixed derivatives
    assert ex.grad(ex(1.0)).value.shape == (2,)


def test_raw_spatial_coordinate_data():
    """grad and div see a dependence on position however the data was built (here: by hand)."""
    ex = evaluator()
    xs = SCALES.L0 * ufl.SpatialCoordinate(MESH)  # not ex.x
    u = ufl.as_vector([xs[0] ** 2, xs[0] * xs[1]])
    assert_same(ex.grad(u), ufl.as_matrix([[2 * x, 0], [y, x]]))
    assert_same(ex.div(u), 3 * x)
    zero = ex.grad(ex([1.0, 2.0]))
    assert isinstance(zero, dolfinx.fem.Constant) and zero.value.shape == (2, 2)


def test_time_dependent_ufl_data_must_use_the_models_t():
    """ddt sees only the model's t (documented): another time Constant counts as constant."""
    ex = evaluator()
    assert_same(ex.ddt(ex.t**2 * x), 2 * T * x)  # the model's t: exact
    foreign = dolfinx.fem.Constant(MESH, T)
    assert float(ex.ddt(foreign**2 * x).value) == 0.0  # the pitfall the docstring warns about


def test_at_another_time():
    ex = evaluator()
    t_old = dolfinx.fem.Constant(MESH, 0.1)
    assert_same(ex.at(ex.ddt(ex("t**2 * x")), t_old), 2 * SCALES.T0 * 0.1 * x)


def test_waveforms():
    register_waveform("test_ramp", lambda t, duration=1.0: t / duration)
    assert_same(evaluator()({"waveform": "test_ramp", "duration": 2.0}), T / 2 + 0 * x)
    with pytest.raises(ValueError, match="unknown waveform"):
        evaluator()({"waveform": "nope"})


def test_bad_value():
    with pytest.raises(TypeError, match="cannot turn"):
        evaluator()(object())
