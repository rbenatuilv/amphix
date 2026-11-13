"""The poro sections, `material` (SI) and `discretization`, and the scaled coefficients.

    material:  {mu: 507, lambda: 11567, alpha: 1.0, c0: 1.0e-6, K: 1.0e-16, muF: 0.8e-3}
    discretization: {deg: 2, deg_p: null, eta: 10, zeta: 10}

Elasticity is given by Lamé (`lambda`, `mu`) or by Young and Poisson (`E`, `nu`). Which keys each
model reads:

    both    lambda, mu, alpha, c0, K, muF, rho
    poro2   beta        (leakage)
    poro3   rho_f, phi, tortuosity  (fluid inertia rho_W = rho_f tortuosity / phi)

A key a model does not read, given a value other than its default, warns (the same file serves both
models).

A named parameter set fills the section, and the keys given next to it override it:

    material: {preset: cvr22, K: 2.0e-16}

`E`/`nu` given next to a preset replace its `lambda`/`mu` (and the other way round). A preset gives
each model only the keys it reads, so only keys given in the file can warn as unread. The sets are
registered by `amphix.brain` (`register_preset`).

The scaled coefficients come from Scales.factor of each kind.
"""

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

from amphix.config.validate import ConfigError, check_keys, join, mapping, number, unread
from amphix.core.units import DENSITY, LEAKAGE, MOBILITY, PRESSURE, STORAGE, Scales

BOTH = ("lambda", "mu", "alpha", "c0", "K", "muF", "rho")
READS = {"poro2": (*BOTH, "beta"), "poro3": (*BOTH, "rho_f", "phi", "tortuosity")}
OPTIONAL = {"rho": 0.0, "beta": 0.0, "rho_f": 0.0, "phi": 1.0, "tortuosity": 1.0}  # defaults
PAIRS = ({"lambda", "mu"}, {"E", "nu"})  # the two ways of giving elasticity

PRESETS: dict[str, dict[str, float]] = {}  # name -> material values (SI), filled by amphix.brain


def register_preset(name: str, values: Mapping[str, float]) -> None:
    """Make `material: {preset: name}` the material `values` (SI keys of this section)."""
    check_keys(f"preset {name}", values, {*BOTH, "E", "nu", *OPTIONAL})
    PRESETS[name] = dict(values)


def _with_preset(path: str, raw: dict, model: str) -> dict:
    """Return the preset's values the model reads (if `preset` is given), then the keys given."""
    if "preset" not in raw:
        return raw
    name = raw.pop("preset")
    if not isinstance(name, str) or name not in PRESETS:
        raise ConfigError(
            join(path, "preset"),
            f"unknown preset {name!r}; known {sorted(PRESETS)} (amphix.brain registers them)",
        )
    values = {k: v for k, v in PRESETS[name].items() if k in {*READS[model], "E", "nu"}}
    for given, other in (PAIRS, PAIRS[::-1]):
        if given & set(raw):  # elasticity given the other way: drop the preset's pair
            values = {k: v for k, v in values.items() if k not in other}
    return values | raw


@dataclass(frozen=True)
class Material:
    """Physical parameters in SI."""

    lam: float  # Lame lambda [Pa]
    mu: float  # shear modulus [Pa]
    alpha: float  # Biot-Willis coefficient [-]
    c0: float  # constrained specific storage [1/Pa]
    K: float  # intrinsic permeability [m^2]
    muF: float  # pore-fluid viscosity [Pa s]
    rho: float = 0.0  # mixture density [kg/m^3] (solid inertia)
    rho_f: float = 0.0  # pore-fluid density [kg/m^3] (poro3)
    phi: float = 1.0  # porosity [-] (poro3)
    tortuosity: float = 1.0  # [-] (poro3)
    beta: float = 0.0  # leakage coefficient [1/(Pa s)] (poro2)

    @classmethod
    def from_dict(cls, raw: Any, path: str, model: str) -> "Material":
        """Return the `material` section of a model (poro2 or poro3)."""
        raw = dict(mapping(path, raw))
        check_keys(path, raw, {*BOTH, "E", "nu", *OPTIONAL, "preset"})
        given = set(raw) - {"preset"}  # only these can warn as unread
        values = {k: number(join(path, k), v) for k, v in _with_preset(path, raw, model).items()}
        lame, young = {"lambda", "mu"} & set(values), {"E", "nu"} & set(values)
        if (lame and young) or not (len(lame) == 2 or len(young) == 2):
            raise ConfigError(path, "give either lambda and mu, or E and nu")
        if young:
            E, nu = values.pop("E"), values.pop("nu")
            if not -1 < nu < 0.5:
                raise ConfigError(join(path, "nu"), f"must be in (-1, 0.5), got {nu}")
            values["lambda"], values["mu"] = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
        missing = sorted({"alpha", "c0", "K", "muF"} - set(values))
        if missing:
            raise ConfigError(path, f"missing {missing}")
        for key in ("mu", "K", "muF"):
            if values[key] <= 0:
                raise ConfigError(join(path, key), f"must be positive, got {values[key]}")
        unread(
            path,
            [
                k
                for k in OPTIONAL
                if k not in READS[model] and k in given and values[k] != OPTIONAL[k]
            ],
            model,
        )
        values["lam"] = values.pop("lambda")
        return cls(**values)

    def scaled(self, s: Scales) -> dict[str, float]:
        """Return the non-dimensional coefficients of the equations, by Scales.factor of each kind.

        Only kappa = K / muF enters Darcy's law, so muF disappears; rho_W = rho_f tortuosity / phi.
        """
        density = s.factor(DENSITY)
        return {
            "lam": self.lam * s.factor(PRESSURE),
            "mu": self.mu * s.factor(PRESSURE),
            "alpha": self.alpha,
            "c0": self.c0 * s.factor(STORAGE),
            "kappa": self.K / self.muF * s.factor(MOBILITY),
            "rho": self.rho * density,
            "rhoF": self.rho_f * density,
            "rhoW": self.rho_f * self.tortuosity / self.phi * density,
            "beta": self.beta * s.factor(LEAKAGE),
        }


@dataclass(frozen=True)
class Discretization:
    """DG degrees and SIPG penalties. deg_p None: deg (poro2), deg - 1 (poro3)."""

    deg: int = 2  # degree of d (and w)
    deg_p: int | None = None  # degree of p
    eta: float = 10.0  # elasticity penalty (and the normal-jump penalty of w in poro3)
    zeta: float = 10.0  # Darcy penalty (poro2), times kappa* deg^2 / h

    @classmethod
    def from_dict(cls, raw: Any, path: str, model: str) -> "Discretization":
        """Return the `discretization` section."""
        raw = dict(mapping(path, raw))
        check_keys(path, raw, {f.name for f in fields(cls)})
        values: dict[str, Any] = {}
        for key in ("deg", "deg_p"):
            if raw.get(key) is not None:
                values[key] = number(join(path, key), raw[key], integer=True)
                if values[key] < (1 if key == "deg" else 0):
                    raise ConfigError(join(path, key), f"too small: {values[key]}")
        for key in ("eta", "zeta"):
            if key in raw:
                values[key] = number(join(path, key), raw[key], positive=True)
        if model == "poro3" and "zeta" in raw:
            unread(path, ["zeta"], model)
        return cls(**values)

    def pressure_degree(self, model: str) -> int:
        """Return deg_p, or its default: deg (poro2), deg - 1 (poro3, the stable P2/P2/P1)."""
        if self.deg_p is not None:
            return self.deg_p
        return self.deg if model == "poro2" else max(self.deg - 1, 0)
