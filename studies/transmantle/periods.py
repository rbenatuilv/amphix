"""The start-up residual of the oscillation, period by period.

Case 2 started from the steady state of gamma, run for 20 periods (see transmantle.yaml):

    amphix run transmantle.yaml --set time.T=20 \
        --set 'initial={from: results/steady_fields.npz}' --name case2_20

For each period k it sums the outward flow of each wall over (k - 1, k] s and subtracts the steady
volume of one period (results/steady.npz): what is left is the net effect of the oscillation in
that period. A start-up residual decays with k (as k^-3/2 for a sine start); a real net effect
would level off at a value of its own.

Figure: figures/periods.png. Top: the raw net volume of each period from period K_FIRST on, with
the steady value (case 1). Bottom, every period: its difference from the steady value (absolute
values, log-log; open markers where the difference is negative).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import amphix

RUN = "case2_20"
WALLS = {"ventricle": ("into the tissue at the ventricle", -1), "pial": ("out at the pia", 1)}
SLOPE = -1.5  # decay of the residual of a sine start
K_FIRST = 4  # top row: the first period shown (the earliest ones would squash the scale)


def per_period(r, steady):
    """Return the periods k and {wall: net volume of period k minus the steady one} [m^2]."""
    t = r["t"]
    dt = t[-1] - t[-2]
    k = np.arange(1, round(t[-1]) + 1)
    out = {}
    for w, (_, sign) in WALLS.items():
        q = sign * r[f"cbflux_{w}"]
        v = np.array([q[(t > j - 1 + 1e-9) & (t < j + 1e-9)].sum() * dt for j in k])
        out[w] = v - steady[w]
    return k, out


if __name__ == "__main__":
    HERE = Path(__file__).parent
    (HERE / "figures").mkdir(exist_ok=True)
    s = amphix.load_results(HERE / "results" / "steady.npz")
    steady = {w: sign * s[f"cbflux_{w}"][-1] for w, (_, sign) in WALLS.items()}  # 1 s = 1 period
    k, dv = per_period(amphix.load_results(HERE / "results" / f"{RUN}.npz"), steady)

    print(f"steady volume per period: {steady['pial']:.3e} m^2 per unit depth")
    print(f"{'k':>3} {'ventricle':>11} {'pia':>11}")
    for i, j in enumerate(k):
        print(f"{j:3d} {dv['ventricle'][i]:11.3e} {dv['pial'][i]:11.3e}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for col, (w, (title, _)) in enumerate(WALLS.items()):
        # top: the raw net volume of each period, converging to the steady value
        ax = axes[0, col]
        shown = k >= K_FIRST
        ax.plot(
            k[shown],
            steady[w] + dv[w][shown],
            "o-",
            color="C1",
            label="case 2, net volume of period k",
        )
        ax.axhline(steady[w], color="C0", lw=1.2, label="case 1 (steady)")
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set(title=f"flow {title}", xlabel="period k")

        # bottom: its difference from the steady value, log-log
        ax = axes[1, col]
        v = dv[w]
        neg = v < 0
        ax.loglog(k, np.abs(v), "-", color="C1", label="case 2 - case 1")
        ax.loglog(k[~neg], v[~neg], "o", color="C1")
        ax.loglog(k[neg], -v[neg], "o", mfc="none", color="C1")
        ax.loglog(k, abs(v[-1]) * (k / k[-1]) ** SLOPE, ":", color="k", label=f"k^{SLOPE}")
        ax.axhline(steady["pial"], color="0.5", lw=0.8, label="steady volume per period")
        ax.set(xlabel="period k")
    axes[0, 0].set_ylabel("net volume of period k [m^2 per unit depth]")
    axes[1, 0].set_ylabel("|case 2 - case 1| [m^2 per unit depth]")
    axes[0, 1].legend(fontsize=8)
    axes[1, 1].legend(fontsize=8)
    fig.savefig(HERE / "figures" / "periods.png", dpi=150)
