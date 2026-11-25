"""Run the 16 unit cases of the brain annulus: python run.py [--only text].

A case is <pset>_<pia>_<load>, e.g. cvr22_fixed_Cc; the result is results/<case>.npz (and
<case>_fields.npz). A case whose result exists is skipped. Also runs under mpirun -n N.
"""

import argparse
from pathlib import Path

import amphix

HERE = Path(__file__).parent

WALL = {  # wall refinement (logbook step 8: wall flows within ~2 % for h <= 4.5 delta)
    "cvr22": {"h": 1.0e-4, "band": 2.0e-4, "growth": 2.0e-3},  # 4.5 delta, delta = 22 um
    "prv23": {"h": 2.0e-4, "band": 4.0e-4, "growth": 2.0e-3},  # 1.8 delta, delta = 0.11 mm
}
PIA = {
    "fixed": {"pressure": "pp", "displacement": [0, 0]},
    "free": {"pressure": "pp", "normal_traction": "-pp"},
}
LOADS = {  # frequency [Hz], common C, difference D
    "Cc": (1.0, 1, 0),
    "Cr": (0.25, 1, 0),
    "Dc": (1.0, 0, 1),
    "Dr": (0.25, 0, 1),
}


def overrides(pset, pia, load):
    """Changes to unit.yaml for one case."""
    freq, C, D = LOADS[load]
    P = 1 / freq  # period
    T = 3 * P  # three periods; the last one is recorded
    last = {"from": T - P, "to": T}
    return {
        "mesh.wall": WALL[pset],
        "material.preset": pset,
        "params": {"freq": freq, "C": C, "D": D},
        "boundaries.pial": PIA[pia],
        "time.T": T,
        "time.dt": P / 100,  # 100 steps per period
        "output.fields.at": [T - 0.75 * P, T - 0.5 * P, T - 0.25 * P, T],
        "output.grid.at": {**last, "every": 5},
        "output.lines.at": {**last, "every": 1},
        "output.name": f"{pset}_{pia}_{load}",
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="run only the cases whose name contains this")
    only = ap.parse_args().only
    for pset in WALL:
        for pia in PIA:
            for load in LOADS:
                name = f"{pset}_{pia}_{load}"
                if only in name and not (HERE / "results" / f"{name}.npz").exists():
                    amphix.run(amphix.load(HERE / "unit.yaml", overrides(pset, pia, load)))
