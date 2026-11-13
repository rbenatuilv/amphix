"""Poroelasticity (Biot): the two-field model (d, p) and the three-field model (d, w, p).

    parameters   the `material` and `discretization` sections
    boundary     the condition and source kinds, and their data in the forms
    forms        the DG forms: elasticity, Darcy and the couplings
    twofield     poro2
    threefield   poro3
    balance      the step mass balance (for post.hdiv)
    strong       manufactured solutions

Importing this package registers the models.
"""

from amphix.physics.poro.parameters import Discretization, Material
from amphix.physics.poro.strong import Manufactured
from amphix.physics.poro.threefield import ThreeField
from amphix.physics.poro.twofield import TwoField

__all__ = ["Discretization", "Manufactured", "Material", "ThreeField", "TwoField"]
