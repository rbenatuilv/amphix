"""A toy physics for the tests of core.model: the heat equation u_t - div(k grad u) = f.

SIPG in space (penalty eta k deg^2 / h), the theta-method in time with a Rannacher start, Nitsche
for a prescribed temperature, an outward flux -k grad u . n as the natural condition, and the
gradient as a derived quantity. It uses only core: if the Model base needed anything of Biot,
this model could not be written.
"""

from dataclasses import dataclass
from typing import ClassVar

import dolfinx
import numpy as np
import ufl

from amphix.config.validate import check_keys, number
from amphix.core.conditions import ConditionKind, SourceKind
from amphix.core.dg import penalty
from amphix.core.model import Model
from amphix.core.spaces import dg_element
from amphix.core.timestepping import ThetaMethod
from amphix.core.units import RATE, VELOCITY, Kind

TEMPERATURE = Kind("temperature")  # unscaled
DIFFUSIVITY = Kind("diffusivity", length=2, time=-1)  # m^2/s
GRADIENT = Kind("temperature_gradient", length=-1)  # K/m


@dataclass(frozen=True)
class HeatParameters:
    """The `heat` section: diffusivity k [m^2/s], DG degree, SIPG penalty."""

    k: float
    deg: int = 1
    eta: float = 10.0

    @classmethod
    def from_dict(cls, raw, path, model):
        check_keys(path, raw, {"k", "deg", "eta"}, required={"k"})
        return cls(
            k=number(f"{path}.k", raw["k"], positive=True),
            deg=number(f"{path}.deg", raw.get("deg", 1), integer=True),
            eta=number(f"{path}.eta", raw.get("eta", 10.0), positive=True),
        )


class Heat(Model):
    name = "toy_heat"
    fields: ClassVar[dict] = {"u": TEMPERATURE}
    quantities: ClassVar[dict] = {"gradient": GRADIENT}
    sections: ClassVar[dict] = {"heat": HeatParameters}
    condition_kinds = (
        ConditionKind("temperature", "thermal", "scalar", TEMPERATURE),
        ConditionKind("heat_flux", "thermal", "scalar", VELOCITY),
    )  # K m/s
    source_kinds = (SourceKind("heat_source", "scalar", RATE),)
    time_keys = frozenset({"theta", "startup_steps"})

    def elements(self):
        self.par = self.problem.section("heat")
        return {"u": dg_element(self.domain.mesh, self.par.deg)}

    def initialize(self):
        mesh, tc = self.domain.mesh, self.problem.time
        self.k = dolfinx.fem.Constant(
            mesh, np.float64(self.scales.to_solver(self.par.k, DIFFUSIVITY))
        )
        self.theta = ThetaMethod(tc.theta, tc.startup_steps, mesh)
        self.schedule(self.theta)  # before_step and regular_constants: the base's defaults
        V = dolfinx.fem.functionspace(mesh, ("DG", self.par.deg - 1, (mesh.geometry.dim,)))
        self.derive("gradient", ufl.grad(self.state.current["u"]), V)  # derived(): the default

    def operator(self, w, q):
        """SIPG form of -div(k grad w), Nitsche on the temperature boundaries (w part)."""
        m, n, k = self.measures, self.measures.n, self.k
        pen = penalty(self.par.eta * k, self.par.deg)
        F = (
            k * ufl.inner(ufl.grad(w), ufl.grad(q)) * m.dx()
            - k * ufl.inner(ufl.avg(ufl.grad(w)), ufl.jump(q, n)) * m.dS
            - k * ufl.inner(ufl.avg(ufl.grad(q)), ufl.jump(w, n)) * m.dS
            + pen / m.hF * ufl.jump(w) * ufl.jump(q) * m.dS
        )
        ds = self.conditions.measure("temperature")
        if ds is not None:
            F += (
                -k * ufl.inner(ufl.grad(w), n) * q * ds
                - k * ufl.inner(ufl.grad(q), n) * w * ds
                + pen / m.h * w * q * ds
            )
        return F

    def step_data(self, q):
        """The data form of the step at the current t: source, Nitsche data, flux."""
        m, n, k = self.measures, self.measures.n, self.k
        pen = penalty(self.par.eta * k, self.par.deg)
        G = self.sources["heat_source"] * q * m.dx()
        for c in self.conditions.of("temperature"):
            G += -k * ufl.inner(ufl.grad(q), n) * c.value * c.ds + pen / m.h * c.value * q * c.ds
        for c in self.conditions.of("heat_flux"):
            G += -c.value * q * c.ds
        return G

    def residual(self, u, v):
        w, q, th = u["u"], v["u"], self.theta.theta
        old = self.state.previous["u"]
        G = self.step_data(q)
        return (
            (w - old) / self.state.dt * q * self.measures.dx()
            + th * self.operator(w, q)
            + (1 - th) * self.operator(old, q)
            - th * G
            - (1 - th) * self.expressions.at(G, self.state.t_old)
        )
