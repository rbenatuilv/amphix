"""Net volume per period added by the oscillation, against the period number.

Reads results/steady.npz and the oscillating runs in RUNS (e.g. made with time.T = 20, see
volumes.py for the commands). For each period k = 1, 2, ... it sums the outward flow of each
wall over (k - 1, k] s and subtracts the steady volume of one period: what is left is the net
effect of the oscillation in that period. A start-up artefact decays as a power of k (about
-3/2 for a sine start, -5/2 for a cosine start); a real effect would level off.

Figure: figures/periods.png (absolute values, log-log; open markers where the value is
negative).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUNS = {"osc_sin20": "sin start", "osc_cos20": "cos start"}
WALLS = {"ventricle": -1, "pial": 1}  # sign: inflow at the ventricle, outflow at the pia


def steady_flow(path):
    """Return the steady outward flows {wall: m^2/s} of the one-step steady run."""
    r = np.load(path)
    return {w: r[f"cbflux_{w}"][-1] for w in WALLS}


def per_period(path, steady):
    """Return the period numbers and {wall: net volume per period minus the steady one}."""
    r = np.load(path)
    t = r["t"]
    dt = t[-1] - t[-2]
    k = np.arange(1, round(t[-1]) + 1)
    out = {}
    for w, sign in WALLS.items():
        q = r[f"cbflux_{w}"]
        v = [q[(t > j - 1 + 1e-9) & (t < j + 1e-9)].sum() * dt - steady[w] for j in k]
        out[w] = sign * np.array(v)
    return k, out


if __name__ == "__main__":
    HERE = Path(__file__).parent
    (HERE / "figures").mkdir(exist_ok=True)
    steady = steady_flow(HERE / "results" / "steady.npz")
    v_steady = abs(steady["pial"])  # steady volume of one period [m^2]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for (w, _sign), ax in zip(WALLS.items(), axes, strict=True):
        for i, (name, label) in enumerate(RUNS.items()):
            path = HERE / "results" / f"{name}.npz"
            if not path.exists():
                continue
            k, dv = per_period(path, steady)
            dv = dv[w]
            neg = dv < 0
            ax.loglog(k, np.abs(dv), "-", color=f"C{i}", label=label)
            ax.loglog(k[~neg], dv[~neg], "o", color=f"C{i}")
            ax.loglog(k[neg], -dv[neg], "o", mfc="none", color=f"C{i}")
            print(f"{name} {w}: " + " ".join(f"{x:.2e}" for x in dv))
            slope = -1.5 if "sin" in name else -2.5
            ref = abs(dv[-1]) * (k / k[-1]) ** slope
            ax.loglog(k, ref, ":", color=f"C{i}", label=f"slope {slope}")
        ax.axhline(v_steady, color="0.5", lw=0.8, label="steady volume per period")
        ax.set(
            title={"ventricle": "ventricle (inflow)", "pial": "pia (outflow)"}[w],
            xlabel="period k",
        )
    axes[0].set_ylabel("|net volume per period - steady| [m^2 per unit depth]")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "periods.png", dpi=120)
