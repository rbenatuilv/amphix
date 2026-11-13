"""Units: the reference scales of the solve, and the kind (dimension) of every quantity.

Everything given and returned is in SI. The solver works in non-dimensional solver units, defined
by a reference length L0 [m], time T0 [s] and stress P0 [Pa]:

    x* = x / L0,   t* = t / T0,   p* = p / P0,   d* = d / L0,   q* = q T0 / L0, ...

Every quantity has a dimension m^length s^time Pa^pressure (its `Kind`), and is turned into solver
units by one rule,

    q* = q * L0^-length * T0^-time * P0^-pressure                      (Scales.factor)

so a velocity (1, -1, 0) gets T0 / L0, a density kg/m^3 = Pa s^2 / m^2 (-2, 2, 1) gets
L0^2 / (P0 T0^2), the mobility K / mu_F (2, -1, -1) gets P0 T0 / L0^2. A new physics defines its
own kinds. Scales of 1 (the default) mean "solve with the SI numbers as given".
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Kind:
    """The dimension m^length s^time Pa^pressure of a quantity, with a name for messages."""

    name: str
    length: int = 0
    time: int = 0
    pressure: int = 0


# The kinds of the fields and data, with their SI units.
NONE = Kind("none")  # dimensionless (-), e.g. alpha, porosity, mass_source
LENGTH = Kind("length", length=1)  # m
TIME = Kind("time", time=1)  # s
PRESSURE = Kind("pressure", pressure=1)  # Pa (also stresses, tractions, elastic moduli)
DISPLACEMENT = Kind("displacement", length=1)  # m
VELOCITY = Kind("velocity", length=1, time=-1)  # m/s (also Darcy fluxes)
ACCELERATION = Kind("acceleration", length=1, time=-2)  # m/s^2
BODY_FORCE = Kind("body_force", length=-1, pressure=1)  # N/m^3 = Pa/m
RATE = Kind("rate", time=-1)  # 1/s, e.g. fluid_source
# and those of the material coefficients
DENSITY = Kind("density", length=-2, time=2, pressure=1)  # kg/m^3 = Pa s^2/m^2
MOBILITY = Kind("mobility", length=2, time=-1, pressure=-1)  # m^2/(Pa s), kappa = K / mu_F
STORAGE = Kind("storage", pressure=-1)  # 1/Pa, c0
LEAKAGE = Kind("leakage", time=-1, pressure=-1)  # 1/(Pa s), beta


@dataclass(frozen=True)
class Scales:
    """Reference length L0 [m], time T0 [s] and stress P0 [Pa] of the solve (all > 0)."""

    L0: float = 1.0
    T0: float = 1.0
    P0: float = 1.0

    def __post_init__(self) -> None:
        """Reject a scale that is not positive."""
        for name in ("L0", "T0", "P0"):
            if not getattr(self, name) > 0:
                raise ValueError(f"scales.{name} must be positive, got {getattr(self, name)}")

    def factor(self, kind: Kind) -> float:
        """Return the factor that turns an SI value of this kind into solver units."""
        return float(self.L0**-kind.length * self.T0**-kind.time * self.P0**-kind.pressure)

    def to_solver(self, value: float, kind: Kind) -> float:
        """Return an SI value in solver units."""
        return value * self.factor(kind)

    def to_si(self, value: float, kind: Kind) -> float:
        """Return a solver-unit value in SI."""
        return value / self.factor(kind)
