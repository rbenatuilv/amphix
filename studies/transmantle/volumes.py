"""Fluid volume through the ventricle and the pia over the last two periods of each run.

Reads results/<run>.npz for the runs in RUNS, made from transmantle.yaml with

    amphix run transmantle.yaml --set params.dP=0 --set time.T=1e9 --set time.dt=1e9 \
        --set time.theta=1 --set 'output.fields={quantities: [d, p], at: [1e9]}' --name steady
    amphix run transmantle.yaml --set params.dP=0 \
        --set 'initial={from: results/steady_fields.npz}' --name mean
    amphix run transmantle.yaml --set 'initial={from: results/steady_fields.npz}' --name osc

cbflux_<wall> is the outward flow per unit depth [m^2/s], averaged over each step; summing it
times dt over the steps of the window gives the volume per unit depth [m^2]. Inflow at the
ventricle is minus its outward volume; outflow at the pia is its outward volume; their
difference is the fluid stored. Each run is also compared with `mean` (gamma only).
"""

from pathlib import Path

import numpy as np

RUNS = ["steady", "mean", "osc"]
WINDOW = 2.0  # s: the last two periods of each run

# steady flow through an eccentric annulus (bipolar coordinates), per unit depth [m^2/s]
kappa = 1e-16 / 0.8e-3
R1, R2, e = 2.5e-3, 1.0e-2, 3.0e-3
gamma = 0.292
Q_steady = 2 * np.pi * kappa * gamma / np.arccosh((R1**2 + R2**2 - e**2) / (2 * R1 * R2))


def volumes(path):
    """Return (inflow at the ventricle, outflow at the pia) over the window [m^2]."""
    r = np.load(path)
    t = r["t"]
    dt = t[-1] - t[-2]
    if dt > WINDOW:  # steady run: one huge implicit step, its last flows are the steady values
        return -r["cbflux_ventricle"][-1] * WINDOW, r["cbflux_pial"][-1] * WINDOW
    w = t > t[-1] - WINDOW + 1e-9  # steps ending in (T - WINDOW, T]
    return -r["cbflux_ventricle"][w].sum() * dt, r["cbflux_pial"][w].sum() * dt


if __name__ == "__main__":
    HERE = Path(__file__).parent
    print(f"analytic steady: {Q_steady * WINDOW:.3e} m^2 per unit depth over {WINDOW:.0f} s")
    print(f"{'run':>10} {'in (ventricle)':>15} {'out (pia)':>12} {'stored':>12}")
    res = {}
    for name in RUNS:
        path = HERE / "results" / f"{name}.npz"
        if path.exists():
            v_in, v_out = res[name] = volumes(path)
            print(f"{name:>10} {v_in:15.3e} {v_out:12.3e} {v_in - v_out:12.3e}")
    if "mean" in res:
        print("difference from mean (effect of the oscillation):")
        for name in res.keys() - {"mean", "steady"}:
            d_in, d_out = (res[name][i] - res["mean"][i] for i in (0, 1))
            print(f"{name:>10} in {d_in:.3e}, out {d_out:.3e}")
