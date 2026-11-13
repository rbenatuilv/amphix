"""core.units: one rule (exponents of L0, T0, P0) gives every factor."""

import pytest

from amphix.core import units as u

S = u.Scales(L0=1.0e-2, T0=0.5, P0=1500.0)
L0, T0, P0 = S.L0, S.T0, S.P0


@pytest.mark.parametrize(
    "kind, expected",
    [
        (u.LENGTH, 1 / L0),
        (u.TIME, 1 / T0),
        (u.PRESSURE, 1 / P0),
        (u.DISPLACEMENT, 1 / L0),
        (u.VELOCITY, T0 / L0),
        (u.ACCELERATION, T0**2 / L0),
        (u.BODY_FORCE, L0 / P0),
        (u.RATE, T0),
        (u.NONE, 1.0),
        (u.MOBILITY, P0 * T0 / L0**2),  # kappa
        (u.DENSITY, L0**2 / (P0 * T0**2)),  # rho, rho_f
        (u.STORAGE, P0),  # c0
        (u.LEAKAGE, P0 * T0),  # beta
    ],
)
def test_factors(kind, expected):
    assert S.factor(kind) == pytest.approx(expected, rel=1e-15)


def test_brain_mobility():
    """The scaled mobility of the brain annulus."""
    s = u.Scales(L0=1.0e-2, T0=1.0, P0=1500.0)
    assert s.to_solver(1.57e-11 / 3.5e-3, u.MOBILITY) == pytest.approx(
        1.57e-11 / 3.5e-3 * 1500 / 1e-4
    )


def test_defaults_and_round_trip():
    assert u.Scales().factor(u.DENSITY) == 1.0
    assert S.to_si(S.to_solver(3.7, u.BODY_FORCE), u.BODY_FORCE) == pytest.approx(3.7, rel=1e-15)


@pytest.mark.parametrize("bad", [{"L0": 0.0}, {"T0": -1.0}])
def test_scales_must_be_positive(bad):
    with pytest.raises(ValueError, match="must be positive"):
        u.Scales(**bad)
