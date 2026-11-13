"""Data -> UFL expressions, in SI, of the position x and the time t.

A value (`Data`) can be

    a number                 2.5                      (a Constant: never simplified away)
    a list (vector)          [0, "-rho * 9.81"]
    a string                 "A * (1 - cos(2*pi*t)) / 2"
    a callable               lambda x, t: ...         (x and t are UFL, in SI)
    a UFL expression         on the model's mesh, in SI
    a waveform               {"waveform": name, **parameters}   (see register_waveform)

Names available in strings:

    x, y, z    position [m]            t      time [s]          r = |x| (from the origin)
    pi, e, sin, cos, tan, exp, log, sqrt, abs, tanh, sinh, cosh, atan, min, max
    where(cond, a, b)                  e.g. where(t < 0.1, t / 0.1, 1)
    every key of `params`, and every earlier key of `loads` (loads may build on each other)

x = L0 x* is the SI position (x* the mesh coordinate) and t = T0 t* the SI time (t* the time
Constant of the forms). Every derivative of data is exact and in SI:

    ddt(e) = diff(e, t)          t is a ufl.variable
    grad(e) = grad*(e) / L0      on the mesh coordinate, so it works on any expression
    div(u) = div*(u) / L0        row-wise for a matrix, as ufl.div

Time-dependent UFL data must be written with the model's t (`Expressions.t`, which a callable
receives as its second argument): for ddt, an expression of any other time Constant is constant.

at(e, t_old) evaluates data at another time; it needs an expression whose derivatives are already
expanded, which every result of this module is.
"""

import math
import numbers
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeAlias

import dolfinx
import numpy as np
import ufl
from petsc4py import PETSc
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from ufl.algorithms.apply_derivatives import apply_derivatives

from amphix.core.units import Scales

Expr: TypeAlias = Any  # ufl.core.expr.Expr (ufl has no type stubs)
Data: TypeAlias = float | str | Sequence[Any] | Mapping[str, Any] | Callable[..., Any] | Expr

VARIABLES = ("x", "y", "z", "t", "r")
FUNCTIONS: dict[str, Any] = {
    "pi": math.pi,
    "e": math.e,
    "sin": ufl.sin,
    "cos": ufl.cos,
    "tan": ufl.tan,
    "exp": ufl.exp,
    "log": ufl.ln,
    "sqrt": ufl.sqrt,
    "abs": abs,
    "tanh": ufl.tanh,
    "sinh": ufl.sinh,
    "cosh": ufl.cosh,
    "atan": ufl.atan,
    "min": ufl.min_value,
    "max": ufl.max_value,
    "where": ufl.conditional,
}

WAVEFORMS: dict[str, Callable[..., Expr]] = {}


def register_waveform(name: str, f: Callable[..., Expr]) -> None:
    """Make `{"waveform": name, **parameters}` a value: f(t, **parameters), t in SI (UFL)."""
    WAVEFORMS[name] = f


def expand(expr: Expr) -> Expr:
    """Evaluate every derivative now (diff in t, grad), so that no derivative node is left."""
    return apply_derivatives(apply_algebra_lowering(expr))


# The SI derivatives of data, for whoever holds the SI x and t: Expressions, and the callables
# of a manufactured solution (physics.poro.strong). They may return a UFL Zero.
def time_derivative(expr: Expr, t: Expr) -> Expr:
    """Return d/dt of an expression in SI; t is the SI time variable (Expressions.t)."""
    return expand(ufl.diff(expr, t))


def spatial_gradient(expr: Expr, L0: float) -> Expr:
    """Return the SI gradient of an expression on a mesh in solver units (x = L0 x*)."""
    return expand(ufl.grad(expr) / L0)


def spatial_divergence(expr: Expr, L0: float) -> Expr:
    """Return the SI divergence of a vector or a matrix (row-wise, as ufl.div)."""
    if len(expr.ufl_shape) not in (1, 2):
        raise ValueError(f"div needs a vector or a matrix, got shape {expr.ufl_shape}")
    return expand(ufl.div(expr) / L0)


class Expressions:
    """Evaluator of data, bound to one mesh, one time Constant and one set of scales."""

    def __init__(
        self,
        mesh: dolfinx.mesh.Mesh,
        scales: Scales,
        t: dolfinx.fem.Constant,
        params: Mapping[str, float] | None = None,
        loads: Mapping[str, Data] | None = None,
    ) -> None:
        """Bind to `mesh` (solver units) and to `t`, the time Constant of the forms (t* = t/T0)."""
        self.mesh, self.scales, self.t_const = mesh, scales, t
        self.x = scales.L0 * ufl.SpatialCoordinate(mesh)  # SI position
        self.t = ufl.variable(scales.T0 * t)  # SI time
        ns: dict[str, Any] = dict(FUNCTIONS)
        ns.update(x=self.x[0], y=self.x[1], t=self.t, r=ufl.sqrt(ufl.inner(self.x, self.x)))
        if mesh.geometry.dim == 3:
            ns["z"] = self.x[2]
        ns.update({k: float(v) for k, v in (params or {}).items()})
        self.ns = ns
        for name, value in (loads or {}).items():  # in order: a load may use the earlier ones
            self.ns[name] = self(value)

    def _const(self, value: float | np.ndarray) -> dolfinx.fem.Constant:
        return dolfinx.fem.Constant(self.mesh, np.asarray(value, dtype=PETSc.ScalarType))

    def __call__(self, value: Data) -> Expr:
        """Return the UFL expression (SI) of a value."""
        if isinstance(value, ufl.core.expr.Expr):  # (a UFL expression is callable: check it first)
            return value
        if callable(value):
            out = expand(ufl.as_ufl(value(self.x, self.t)))
            return self._zero_or(out, out.ufl_shape)  # a callable may give an exact zero
        if isinstance(value, Mapping):
            return self._waveform(value)
        return self._plain(value)

    def _plain(self, value: float | str | Sequence[Any]) -> Expr:
        """Return a number, a formula or a list (of numbers: a Constant) as UFL."""
        if isinstance(value, numbers.Real):
            return self._const(float(value))
        if isinstance(value, str):
            out = eval(value, {"__builtins__": {}}, self.ns)  # the formula language above
            return self(out) if isinstance(out, numbers.Real) else ufl.as_ufl(out)
        if isinstance(value, Sequence):
            if all(isinstance(v, numbers.Real) for v in value):
                return self._const(np.asarray(value, dtype=float))
            return ufl.as_tensor([self(v) for v in value])  # a vector, or a matrix of rows
        raise TypeError(f"cannot turn {value!r} into an expression")

    def _waveform(self, value: Mapping[str, Any]) -> Expr:
        params = dict(value)
        name = params.pop("waveform", None)
        if name not in WAVEFORMS:
            raise ValueError(f"unknown waveform {name!r}; known: {sorted(WAVEFORMS)}")
        return expand(ufl.as_ufl(WAVEFORMS[name](self.t, **params)))

    def _zero_or(self, out: Expr, shape: tuple[int, ...]) -> Expr:
        """Return out, or a zero Constant of its shape if out is a UFL Zero (never a bare Zero)."""
        if isinstance(out, ufl.constantvalue.Zero):
            return self._const(np.zeros(shape))
        return out

    def ddt(self, expr: Expr) -> Expr:
        """Return the exact time derivative d/dt (SI) of an expression."""
        return self._zero_or(time_derivative(expr, self.t), expr.ufl_shape)

    def grad(self, expr: Expr) -> Expr:
        """Return the exact gradient (SI) of an expression: shape expr.shape + (dim,)."""
        shape = (*expr.ufl_shape, self.mesh.geometry.dim)
        return self._zero_or(spatial_gradient(expr, self.scales.L0), shape)

    def div(self, expr: Expr) -> Expr:
        """Return the exact divergence (SI) of a vector or a matrix (row-wise, as ufl.div)."""
        out = spatial_divergence(expr, self.scales.L0)
        return self._zero_or(out, expr.ufl_shape[:-1])

    def at(self, expr: Expr, t: dolfinx.fem.Constant) -> Expr:
        """Return the same expression with the time Constant replaced by t (e.g. t_old)."""
        return ufl.replace(expr, {self.t_const: t})
