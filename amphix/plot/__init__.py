"""Plotting saved runs, after the run, from their files alone.

Fields of a fields file (2D) and time series of a results file, with matplotlib (optional:
`pip install amphix[plot]`). Lengths in mm, everything else in SI.

    import amphix
    from amphix import plot

    fl = amphix.load_fields("results/run_fields.npz")
    plot.field(fl, "p", t=0.5, zoom=(5.2e-3, 6.2e-3, -5e-4, 5e-4))   # the snapshot nearest 0.5 s
    plot.field(fl, "cflux", average=(0.0, 1.0), log=True, quiver=20)
    plot.series(amphix.load_results("results/run.npz"), ["probe_p", "cbflux_pial"])

Every function takes an optional matplotlib axes. Uses only `io`, `post` and `core`.
"""

try:
    import matplotlib  # noqa: F401
except ImportError as e:  # pragma: no cover (tested in a subprocess)
    raise ImportError("amphix.plot needs matplotlib: pip install 'amphix[plot]'") from e

from amphix.plot.fields import FieldPlot, field, mesh
from amphix.plot.labels import PlotError, label, nearest, when
from amphix.plot.lattice import Lattice
from amphix.plot.series import series

__all__ = [
    "FieldPlot",
    "Lattice",
    "PlotError",
    "field",
    "label",
    "mesh",
    "nearest",
    "series",
    "when",
]
