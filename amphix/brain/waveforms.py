"""Brain waveforms of one cardiac cycle per second (t in s), usable as `{waveform: name, ...}` data.

    cardiac_pulse      p_c(t) = A (1 - cos 2 pi t) / 2                [Pa]   A = 133.3 Pa
    transmantle        delta_p(t) = dp sin 2 pi t                     [Pa]   dp = 1.5 Pa
    vascular_option1   g(t) = G sin 2 pi t                            [1/s]  G = 3.53e-3 1/s
    vascular_option2   g(t) = G (sin 2 pi t + weight sin 4 pi t)      [1/s]  weight = 0.5

For example `sources: {fluid_source: {waveform: vascular_option2, G: 3.53e-3}}`. Each function
takes t as UFL (in a model) or as numbers (numpy).

Sources of the values:
- A = 133.3 Pa = 1 mmHg: the pulse amplitude; it starts from rest, with its peak at t = 1/2.
- dp = 1.5 Pa: the cardiac transmantle gradient of 1.46 mmHg/m (Vinje et al., as reported in
  [PRV23]) times the tissue length of the annulus: 194.6 Pa/m x 7.5 mm = 1.46 Pa.
- G = 3.53e-3 1/s: the vascular fluid source per tissue volume, calibrated to a stroke volume of
  1.12e-3 of the tissue volume (1.54 ml over a 1369.54 ml parenchyma, [CVR22]). The stroke volume
  is max - min of int_0^t g over a period, the volume added during inflow: G / pi, for option 1
  and for option 2 while weight <= 0.5. g has zero mean over a period.

Importing this module registers the waveforms with core.expressions.register_waveform.
"""

import math
from typing import Any

import numpy as np
import ufl

from amphix.core.expressions import register_waveform

A_PULSE = 133.3  # Pa, 1 mmHg
DP_TRANSMANTLE = 1.5  # Pa
G_VASCULAR = 3.53e-3  # 1/s
WEIGHT = 0.5  # of the second harmonic (option 2)


def _sin(t: Any) -> Any:
    return ufl.sin(t) if isinstance(t, ufl.core.expr.Expr) else np.sin(t)


def _cos(t: Any) -> Any:
    return ufl.cos(t) if isinstance(t, ufl.core.expr.Expr) else np.cos(t)


def cardiac_pulse(t: Any, A: float = A_PULSE) -> Any:
    """Return the cardiac pulse A (1 - cos 2 pi t) / 2 [Pa]: from rest, peak A at t = 1/2."""
    return A * (1 - _cos(2 * math.pi * t)) / 2


def transmantle(t: Any, dp: float = DP_TRANSMANTLE) -> Any:
    """Return the transmantle pressure difference dp sin 2 pi t [Pa]."""
    return dp * _sin(2 * math.pi * t)


def vascular(t: Any, G: float = G_VASCULAR, option: int = 2, weight: float = WEIGHT) -> Any:
    """Return the vascular fluid source g(t) [1/s] (per tissue volume) of option 1 or 2."""
    if option == 1:
        return G * _sin(2 * math.pi * t)
    if option == 2:
        return G * (_sin(2 * math.pi * t) + weight * _sin(4 * math.pi * t))
    raise ValueError(f"vascular: option must be 1 or 2, got {option!r}")


def vascular_option1(t: Any, G: float = G_VASCULAR) -> Any:
    """Return g(t) = G sin 2 pi t [1/s]."""
    return vascular(t, G, option=1)


def vascular_option2(t: Any, G: float = G_VASCULAR, weight: float = WEIGHT) -> Any:
    """Return g(t) = G (sin 2 pi t + weight sin 4 pi t) [1/s].

    g = G sin 2 pi t (1 + 2 weight cos 2 pi t): for weight <= 0.5 it has the sign of sin 2 pi t,
    the 4 pi t term adds nothing over the inflow half, and the stroke volume is G / pi, as with
    option 1. Above 0.5, g changes sign more often, the stroke volume is not G / pi, and G must be
    recalibrated.
    """
    return vascular(t, G, option=2, weight=weight)


WAVEFORMS = {
    f.__name__: f for f in (cardiac_pulse, transmantle, vascular_option1, vascular_option2)
}
for _name, _f in WAVEFORMS.items():
    register_waveform(_name, _f)
