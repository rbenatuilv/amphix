"""Named brain-tissue parameter sets, each value with its source: `material: {preset: name}`.

    cvr22    Causemann et al. 2022 [CVR22], Fluids Barriers CNS 19:84,
             doi:10.1186/s12987-022-00376-2; alpha = 1 - K_dr / K_s = 1. The densities,
             porosity and tortuosity are from [M25]
    prv23    Piersanti, Rognes and Vinje 2023 [PRV23], PLOS ONE 18(12):e0288668,
             doi:10.1371/journal.pone.0288668, Table 1 (models D and E); alpha = 1. The paper gives
             only K / muF: muF is the CSF viscosity of cvr22. The densities, porosity and
             tortuosity are from [M25]

A set carries every key of both poro models; each model takes only the keys it reads. Keys given
next to the preset override it:

    material: {preset: cvr22, K: 2.0e-16}

Importing this module registers the sets with physics.poro.parameters.register_preset.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from amphix.physics.poro.parameters import register_preset

CVR22 = "Causemann et al. 2022 [CVR22], Fluids Barriers CNS 19:84, doi:10.1186/s12987-022-00376-2"
PRV23 = (
    "Piersanti, Rognes and Vinje 2023 [PRV23], PLOS ONE 18(12):e0288668,"
    " doi:10.1371/journal.pone.0288668, Table 1 (models D and E)"
)
M25 = "[M25]"


@dataclass(frozen=True)
class ParameterSet:
    """A material in SI (keys of the `material` section), with the source of every value."""

    name: str
    values: Mapping[str, float]
    sources: Mapping[str, str]

    def __post_init__(self) -> None:
        """Check that every value has a source, and every source a value."""
        if set(self.values) != set(self.sources):
            raise ValueError(f"{self.name}: values and sources must have the same keys")


CVR22_SET = ParameterSet(
    "cvr22",
    values={
        "mu": 507.0,  # Pa
        "lambda": 11567.0,  # Pa
        "alpha": 1.0,
        "c0": 1.0e-6,  # 1/Pa
        "K": 1.0e-16,  # m^2
        "muF": 0.8e-3,  # Pa s
        "rho": 1000.0,  # kg/m^3
        "rho_f": 1000.0,  # kg/m^3 (poro3)
        "phi": 0.2,  # (poro3)
        "tortuosity": 1.6,  # (poro3)
    },
    sources={
        "mu": f"from E = 1500 Pa, nu = 0.479 {CVR22}, rounded (E / (2 (1 + nu)) = 507.10 Pa)",
        "lambda": f"from E = 1500 Pa, nu = 0.479 {CVR22}, rounded"
        " (E nu / ((1 + nu)(1 - 2 nu)) = 11566.70 Pa)",
        "alpha": "alpha = 1 - K_dr / K_s = 1 (incompressible constituents, K_s >> K_dr)",
        "c0": CVR22,
        "K": f"{CVR22}: intrinsic permeability",
        "muF": f"{CVR22}: CSF viscosity",
        "rho": M25,
        "rho_f": M25,
        "phi": M25,
        "tortuosity": M25,
    },
)

PRV23_SET = ParameterSet(
    "prv23",
    values={
        "mu": 549.2,  # Pa
        "lambda": 54367.0,  # Pa
        "alpha": 1.0,
        "c0": 3.9e-4,  # 1/Pa
        "K": 1.256e-14,  # m^2
        "muF": 0.8e-3,  # Pa s
        "rho": 1000.0,  # kg/m^3
        "rho_f": 1000.0,  # kg/m^3 (poro3)
        "phi": 0.2,  # (poro3)
        "tortuosity": 1.6,  # (poro3)
    },
    sources={
        "mu": f"from E = 1642 Pa, nu = 0.495 {PRV23}, rounded (E / (2 (1 + nu)) = 549.16 Pa)",
        "lambda": f"from E = 1642 Pa, nu = 0.495 {PRV23}, rounded"
        " (E nu / ((1 + nu)(1 - 2 nu)) = 54367.2 Pa)",
        "alpha": PRV23,
        "c0": f"{PRV23}: storage coefficient s",
        "K": f"from the hydraulic conductivity K / muF = 1.57e-5 mm^2/(Pa s) = 1.57e-11 m^2/(Pa s)"
        f" {PRV23}, times muF: K = 1.57e-11 * 0.8e-3",
        "muF": f"{CVR22}: CSF viscosity; {PRV23} gives only K / muF, the ratio poro2 reads",
        "rho": M25,
        "rho_f": M25,
        "phi": M25,
        "tortuosity": M25,
    },
)

PRESETS: dict[str, ParameterSet] = {CVR22_SET.name: CVR22_SET, PRV23_SET.name: PRV23_SET}

for _set in PRESETS.values():
    register_preset(_set.name, _set.values)
