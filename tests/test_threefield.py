"""physics.poro.ThreeField: the zero-flux rule of poro3."""

import pytest
from cases import annulus

import amphix.physics.poro  # noqa: F401 (registers poro3)
from amphix.config import from_dict
from amphix.core.model import make_model


@pytest.mark.parametrize("flux", [1e-6, "1e-6 * t"])
def test_only_zero_flux(flux):
    tree = annulus("poro3", deg=2, h=4e-3)
    tree["boundaries"] = {**tree["boundaries"], "pial": {"flux": flux, "displacement": [0, 0]}}
    with pytest.raises(ValueError, match=r"boundaries\.pial\.flux: poro3 supports only"):
        make_model(from_dict(tree))
    tree["boundaries"]["pial"]["flux"] = 0
    assert make_model(from_dict(tree)).name == "poro3"
