"""physics.poro.parameters: the material and discretization sections, and the scaled
coefficients."""

import pytest

from amphix.config.validate import ConfigError, ConfigWarning
from amphix.core.units import Scales
from amphix.physics.poro.parameters import Discretization, Material

BRAIN = {
    "mu": 1500.0,
    "lambda": 11567.0,
    "alpha": 0.49,
    "c0": 1.0e-6,
    "K": 1.57e-11,
    "muF": 3.5e-3,
    "rho": 1000.0,
}
FLUID = {"rho_f": 1000.0, "phi": 0.2, "tortuosity": 1.6}
SCALES = Scales(L0=1.0e-2, T0=1.0, P0=1500.0)


def material(raw, model="poro2"):
    return Material.from_dict(raw, "material", model)


def test_brain_coefficients():
    c = material({**BRAIN, **FLUID}, "poro3").scaled(SCALES)
    assert c["mu"] == pytest.approx(1.0) and c["lam"] == pytest.approx(11567 / 1500)
    assert c["kappa"] == pytest.approx(1.57e-11 / 3.5e-3 * 1500 / 1e-4)
    assert c["rhoW"] == pytest.approx(c["rhoF"] * 1.6 / 0.2)


def test_scaled_equals_the_formulas():
    """Scales.factor of each kind gives the coefficients written out by hand here, to 1e-15."""
    m = Material(
        lam=11567.0,
        mu=1500.0,
        alpha=0.49,
        c0=1e-6,
        K=1.57e-11,
        muF=3.5e-3,
        rho=1000.0,
        rho_f=1000.0,
        phi=0.2,
        tortuosity=1.6,
        beta=2e-9,
    )  # every key (no model)
    s = Scales(L0=3e-3, T0=0.4, P0=700.0)
    c, (L0, T0, P0) = m.scaled(s), (s.L0, s.T0, s.P0)
    inertia = L0**2 / (P0 * T0**2)
    expected = {
        "lam": m.lam / P0,
        "mu": m.mu / P0,
        "alpha": m.alpha,
        "c0": m.c0 * P0,
        "kappa": m.K / m.muF * P0 * T0 / L0**2,
        "rho": m.rho * inertia,
        "rhoF": m.rho_f * inertia,
        "rhoW": m.rho_f * m.tortuosity / m.phi * inertia,
        "beta": m.beta * P0 * T0,
    }
    assert set(c) == set(expected)
    for key, value in expected.items():
        assert c[key] == pytest.approx(value, rel=1e-15), key


def test_young_and_poisson():
    m = material({"E": 3.0, "nu": 0.25, "alpha": 1, "c0": 0, "K": 1, "muF": 1})
    assert m.mu == pytest.approx(1.2) and m.lam == pytest.approx(1.2)


@pytest.mark.parametrize(
    "raw, path, match",
    [
        ({**BRAIN, "E": 3.0, "nu": 0.3}, "material", "either lambda and mu, or E and nu"),
        ({k: v for k, v in BRAIN.items() if k != "lambda"}, "material", "either lambda and mu"),
        (
            {"E": 3.0, "nu": 0.5, "alpha": 1, "c0": 0, "K": 1, "muF": 1},
            "material.nu",
            "in .-1, 0.5",
        ),
        ({k: v for k, v in BRAIN.items() if k != "alpha"}, "material", r"missing \['alpha'\]"),
        ({**BRAIN, "mu": 0.0}, "material.mu", "must be positive"),
        ({**BRAIN, "K": -1.0}, "material.K", "must be positive"),
        ({**BRAIN, "Kappa": 1.0}, "material.Kappa", "unknown key"),
        ({**BRAIN, "rho": "1000"}, "material.rho", "must be a number"),
    ],
)
def test_material_errors(raw, path, match):
    with pytest.raises(ConfigError, match=rf"^{path}: .*{match}"):
        material(raw)


def test_unread_keys_warn():
    with pytest.warns(ConfigWarning, match=r"\['phi', 'rho_f', 'tortuosity'\] not read by poro2"):
        material({**BRAIN, **FLUID}, "poro2")
    with pytest.warns(ConfigWarning, match=r"\['beta'\] not read by poro3"):
        material({**BRAIN, "beta": 1e-9}, "poro3")


def test_discretization():
    d = Discretization.from_dict({}, "discretization", "poro2")
    assert (d.deg, d.deg_p, d.eta, d.zeta) == (2, None, 10.0, 10.0)
    assert d.pressure_degree("poro2") == 2 and d.pressure_degree("poro3") == 1
    d = Discretization.from_dict({"deg": 1, "deg_p": 1}, "discretization", "poro3")
    assert d.pressure_degree("poro3") == 1
    with pytest.raises(ConfigError, match=r"^discretization.kappa_floor: unknown key"):
        Discretization.from_dict({"kappa_floor": 1.0}, "discretization", "poro2")
    with pytest.raises(ConfigError, match=r"^discretization.deg: too small"):
        Discretization.from_dict({"deg": 0}, "discretization", "poro2")
    with pytest.raises(ConfigError, match=r"^discretization.eta: must be positive"):
        Discretization.from_dict({"eta": 0}, "discretization", "poro2")
    with pytest.warns(ConfigWarning, match=r"\['zeta'\] not read by poro3"):
        Discretization.from_dict({"zeta": 3}, "discretization", "poro3")
