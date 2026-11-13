"""The Model base: one problem, one mesh, one mixed DG space, one linear system per step.

A physics is a subclass that states its class attributes (core.registry: name, fields, sections,
condition_kinds, source_kinds, time_keys, quantities) and writes, at least, three hooks:

    elements()            {field: DG element}                                            required
    initialize()          after the space exists: coefficients, time schemes, and the
                          registrations below (state.track, schedule, derive)            usually
    residual(u, v)        F(trial, test) of one step, affine in the unknowns (solver units) required

The other hooks have defaults:

    quadrature_degree()   2 max(element degree) + 1; None: UFL's estimate
    before_step()         True if any registered schedule switches before step n + 1
    regular_constants()   the merged regular() of the schedules (System.matrix(regular=True))
    after_step()          nothing (poro: the Newmark update)
    derived(name)         the registered derived quantity, interpolated at most once per step
    after_initial()       nothing (poro: copy set_initial into the Newmark states)

Registrations, in initialize(): schedule(s) for an object with before_step(n) -> bool and
regular() -> {Constant: value} (core.timestepping.ThetaMethod); derive(name, expr, space) for a
derived quantity (a UFL expression interpolated on demand, or a Function returned as it is).

One step: before_step (refactorise if it says so); state.previous <- current; t_old <- t,
t <- t0 + (n + 1) dt; solve; split into the fields; after_step; n += 1.

Two views of the state: `state` holds the solver-unit Functions the forms are built on (physics
code); field(), previous() and quantity() return SI copies (scripts, post, io).
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, ClassVar, Protocol

import dolfinx
import dolfinx.fem.petsc
import numpy as np
import ufl

from amphix.core import mesh as core_mesh
from amphix.core import registry
from amphix.core.conditions import ConditionKind, Conditions, SourceKind, Sources, merge_entries
from amphix.core.expressions import Data, Expressions
from amphix.core.linalg import LinearSolver
from amphix.core.spaces import MixedSpace
from amphix.core.units import Kind

Observer = Callable[["Model"], None]


class Schedule(Protocol):
    """Constants that change during a run (a Rannacher start): ThetaMethod is one."""

    def before_step(self, n: int) -> bool:
        """Before step n: update the constants; True if they changed."""
        ...

    def regular(self) -> dict[dolfinx.fem.Constant, float]:
        """Return the constants with their values of a regular step."""
        ...


class State:
    """The solver-unit state of a model: what physics code reads and writes.

    Tracked Functions (`current`), their values at the start of the last step (`previous`), and
    the time Constants of the forms: t (t* of the step) and t_old (t* of its start); dt is dt*.
    """

    def __init__(self, mesh: dolfinx.mesh.Mesh, t0: float, dt: float) -> None:
        """Start empty, at t* = t0 (solver units), with the step dt (solver units)."""
        self.current: dict[str, dolfinx.fem.Function] = {}
        self.previous: dict[str, dolfinx.fem.Function] = {}
        self.kinds: dict[str, Kind] = {}
        self.t = dolfinx.fem.Constant(mesh, np.float64(t0))
        self.t_old = dolfinx.fem.Constant(mesh, np.float64(t0))
        self.dt = dt

    def track(self, name: str, function: dolfinx.fem.Function, kind: Kind) -> None:
        """Keep `function` as state `name` (of a unit kind), copied at the start of each step."""
        self.current[name], self.kinds[name] = function, kind
        self.previous[name] = dolfinx.fem.Function(function.function_space, name=f"{name}_previous")
        self.previous[name].x.array[:] = function.x.array

    def save_previous(self) -> None:
        """Copy every tracked Function into its previous value (the start of a step)."""
        for name, f in self.current.items():
            self.previous[name].x.array[:] = f.x.array


class System:
    """The step system: A of the current step (factorised once), b, the dofs of each field."""

    def __init__(
        self,
        residual: Any,
        space: MixedSpace,
        options: dict[str, Any],
        regular: Callable[[], dict[dolfinx.fem.Constant, float]],
    ) -> None:
        """Split the residual into lhs and rhs (exact: it is affine) and set up the solver."""
        self.space, self._regular = space, regular
        self.solver = LinearSolver(ufl.lhs(residual), ufl.rhs(residual), space.W, options)

    def matrix(self, regular: bool = False) -> Any:
        """Return A of the current step, or of a regular step.

        A regular step has the start-up constants (theta of a Rannacher start) at their regular
        values: A itself when they already are, else a new matrix (the model is not changed).
        """
        constants = self._regular() if regular else {}
        if all(np.array_equal(c.value, v) for c, v in constants.items()):
            return self.solver.A
        saved = {c: c.value.copy() for c in constants}
        try:
            for c, v in constants.items():
                c.value = v  # type: ignore[assignment]
            A = dolfinx.fem.petsc.assemble_matrix(self.solver.a)
            A.assemble()
        finally:
            for c, v in saved.items():
                c.value = v
        return A

    def rhs(self) -> Any:
        """Return b assembled with the current data."""
        return self.solver.assemble_rhs()

    def dofs(self, field: str) -> np.ndarray:
        """Return the owned dofs (rows of A) of a field."""
        return self.space.owned_dofs(field)

    @property
    def ndofs(self) -> int:
        """Return the global number of unknowns."""
        return self.space.ndofs


class Model:
    """Base of every physics (see the module docstring)."""

    name: ClassVar[str]
    fields: ClassVar[dict[str, Kind]] = {}
    quantities: ClassVar[dict[str, Kind]] = {}
    sections: ClassVar[dict[str, Any]] = {}
    condition_kinds: ClassVar[tuple[ConditionKind, ...]] = ()
    source_kinds: ClassVar[tuple[SourceKind, ...]] = ()
    time_keys: ClassVar[frozenset[str]] = frozenset()

    def __init_subclass__(cls, **kw: Any) -> None:
        """Register a subclass that defines a name (the YAML `model:` value)."""
        super().__init_subclass__(**kw)
        if "name" in vars(cls):
            registry.register(cls)

    def __init__(
        self,
        problem: Any,
        domain: core_mesh.Domain | None = None,
        *,
        sources: Mapping[str, Data] | None = None,
        boundary_data: Mapping[str, dict[str, Data]] | None = None,
    ) -> None:
        """Build the mesh (unless given), data, space, state and system of one problem."""
        self.problem, self.scales, tc = problem, problem.scales, problem.time
        self.domain = domain or core_mesh.build(problem.mesh, self.scales.L0, problem.root)
        mesh, T0 = self.domain.mesh, self.scales.T0
        self.n, self.dt = 0, tc.dt
        self.state = State(mesh, tc.t0 / T0, tc.dt / T0)
        self.expressions = Expressions(
            mesh, self.scales, self.state.t, problem.params, problem.loads
        )
        self._schedules: list[Schedule] = []
        self._derived: dict[
            str, tuple[Any, Any, list]
        ] = {}  # name -> (Function, Expression, [key])
        self._version = 0  # bumped by set_initial: the state at n = 0 changed
        elements = self._elements = self.elements()
        self.measures = core_mesh.Measures(self.domain, self.quadrature_degree())
        entries = merge_entries(problem.boundaries, boundary_data)
        self.conditions = Conditions(
            entries, self.condition_kinds, self.domain, self.measures, self.expressions, self.scales
        )
        self.sources = Sources(
            {**problem.sources, **(sources or {})}, self.source_kinds, self.expressions, self.scales
        )
        self.space = MixedSpace(mesh, elements)
        self.solution = dolfinx.fem.Function(self.space.W)
        for name, f in self.space.functions().items():
            self.state.track(name, f, self.fields[name])
        self.initialize()
        F = self.residual(self.space.trial(), self.space.test())
        self.system = System(F, self.space, problem.solver, self.regular_constants)
        self._si: dict[tuple[str, str], dolfinx.fem.Function] = {}

    # -- written by each physics ----------------------------------------------------------------
    def elements(self) -> dict[str, Any]:
        """Return {field: DG element}, in the order of `fields`."""
        raise NotImplementedError

    def quadrature_degree(self) -> int | None:
        """Return 2 max(element degree) + 1, the quadrature degree of every integral."""
        return 2 * max(int(el.degree) for el in self._elements.values()) + 1

    def initialize(self) -> None:
        """Set up coefficients and time schemes once the space exists."""

    def residual(self, u: dict[str, Any], v: dict[str, Any]) -> Any:
        """Return the weak form F(u, v) of one step."""
        raise NotImplementedError

    def before_step(self) -> bool:
        """Prepare step n + 1; return True if the step matrix changed (a schedule switched)."""
        changed = [s.before_step(self.n + 1) for s in self._schedules]  # every one, no shortcut
        return any(changed)

    def after_step(self) -> None:
        """Update the time schemes after a solve."""

    def regular_constants(self) -> dict[dolfinx.fem.Constant, float]:
        """Return {Constant: value} of a regular step: the schedules' regular values."""
        out: dict[dolfinx.fem.Constant, float] = {}
        for s in self._schedules:
            out.update(s.regular())
        return out

    def derived(self, name: str) -> dolfinx.fem.Function:
        """Return the solver-unit Function of a derived quantity, current for this step."""
        if name not in self._derived:
            raise KeyError(f"{self.name} has no quantity {name!r}")
        f, expr, key = self._derived[name]
        if expr is not None and key[0] != (self.n, self._version):  # stale: interpolate again
            f.interpolate(expr)
            key[0] = (self.n, self._version)
        return f

    # -- registrations (in initialize) -----------------------------------------------------------
    def schedule(self, schedule: Schedule) -> None:
        """Register constants that change during the run (before_step, regular_constants)."""
        self._schedules.append(schedule)

    def derive(self, name: str, value: Any, space: Any = None) -> None:
        """Register a derived quantity (`quantities`), returned by derived() and quantity().

        value is a UFL expression, interpolated into `space` on demand and at most once per step
        (the cache is keyed on the step number and on set_initial), or a Function that the physics
        keeps current itself (returned as it is).
        """
        if isinstance(value, dolfinx.fem.Function):
            self._derived[name] = (value, None, [None])
            return
        f = dolfinx.fem.Function(space, name=name)
        expr = dolfinx.fem.Expression(value, space.element.interpolation_points)
        self._derived[name] = (f, expr, [None])

    def after_initial(self) -> None:
        """Copy a new initial state into the time schemes."""

    # -- life cycle -----------------------------------------------------------------------------
    @property
    def t(self) -> float:
        """Return the time [s] of the current state."""
        return self.problem.time.t0 + self.n * self.dt

    def step(self) -> float:
        """Take one time step; return the new time [s]."""
        if self.before_step():
            self.system.solver.update_matrix()
        self.state.save_previous()
        tc, T0 = self.problem.time, self.scales.T0
        self.state.t_old.value = self.state.t.value
        self.state.t.value = (tc.t0 + (self.n + 1) * tc.dt) / T0
        self.system.solver.solve(self.solution)
        self.space.split(self.solution, self.state.current)
        self.after_step()
        self.n += 1
        return self.t

    def steps(self) -> Iterator[float]:
        """Step until T; yield the time [s] after each step."""
        while self.n < self.problem.time.nsteps:
            yield self.step()

    def run(self, observers: Sequence[Observer] = ()) -> None:
        """Run to T, calling every observer at t0 and after every step."""
        for observe in observers:
            observe(self)
        for _ in self.steps():
            for observe in observers:
                observe(self)

    def set_initial(self, **values: Data) -> None:
        """Set tracked state at t0 from SI data (fields, and what the physics tracks)."""
        if self.n != 0:
            raise RuntimeError("set_initial: only before the first step")
        for name, value in values.items():
            if name not in self.state.current:
                raise KeyError(
                    f"set_initial: {name!r} is not state of {self.name}; "
                    f"it has {sorted(self.state.current)}"
                )
            f = self.state.current[name]
            expr = self.scales.factor(self.state.kinds[name]) * self.expressions(value)
            f.interpolate(
                dolfinx.fem.Expression(expr, f.function_space.element.interpolation_points)
            )
            f.x.scatter_forward()
        self.after_initial()
        self.state.save_previous()
        self._version += 1  # derived quantities of the old initial state are stale

    # -- the public state, in SI ----------------------------------------------------------------
    def _copy(self, key: tuple[str, str], f: dolfinx.fem.Function, kind: Kind) -> Any:
        if key not in self._si:
            self._si[key] = dolfinx.fem.Function(f.function_space, name=key[1])
        out = self._si[key]
        out.x.array[:] = f.x.array / self.scales.factor(kind)
        return out

    def field(self, name: str) -> dolfinx.fem.Function:
        """Return an SI copy of a field (or tracked state) at t; writing into it changes nothing."""
        return self._copy(("current", name), self.state.current[name], self.state.kinds[name])

    def previous(self, name: str) -> dolfinx.fem.Function:
        """Return an SI copy of a field (or tracked state) at the start of the last step."""
        return self._copy(("previous", name), self.state.previous[name], self.state.kinds[name])

    def quantity(self, name: str) -> dolfinx.fem.Function:
        """Return an SI copy of a field, tracked state, or derived quantity."""
        if name in self.state.current:
            return self.field(name)
        if name not in self.quantities:
            raise KeyError(f"{self.name} has no quantity {name!r}")
        return self._copy(("derived", name), self.derived(name), self.quantities[name])

    def data(self, value: Data) -> Any:
        """Return any data as UFL (SI) at the model's current time."""
        return self.expressions(value)


def make_model(problem: Any, domain: core_mesh.Domain | None = None, **data: Any) -> Model:
    """Return the model registered as problem.model, built for this problem."""
    return registry.get(problem.model)(problem, domain, **data)
