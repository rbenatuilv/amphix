"""Fluid through the ventricle and the pia: flow over time and volume over the last two periods.

Reads results/case1.npz and results/case2.npz (see transmantle.yaml). cbflux_<wall> is the
outward flow per unit depth [m^2/s], averaged over each step; summed times dt over the steps of
the window it gives the volume per unit depth [m^2]. Inflow at the ventricle is minus its
outward volume, outflow at the pia its outward volume; their difference is the fluid stored.

Figure: figures/volumes.png. Top: the flow through each wall against t, the window shaded.
Bottom: the volumes over the window, per case.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import amphix

RUNS = {"case1": "case 1: gamma only", "case2": "case 2: gamma + dP sin"}
T0, T1 = 3.0, 5.0  # the last two periods [s]


def flows(r):
    """Return t and the flows [m^2/s]: into the tissue at the ventricle, out of it at the pia."""
    return r["t"], -r["cbflux_ventricle"], r["cbflux_pial"]


def volumes(r):
    """Return (inflow at the ventricle, outflow at the pia) over (T0, T1] [m^2]."""
    t, q_in, q_out = flows(r)
    dt = t[-1] - t[-2]
    w = (t > T0 + 1e-9) & (t < T1 + 1e-9)  # steps ending in (T0, T1]
    return q_in[w].sum() * dt, q_out[w].sum() * dt


if __name__ == "__main__":
    HERE = Path(__file__).parent
    (HERE / "figures").mkdir(exist_ok=True)
    res = {k: amphix.load_results(HERE / "results" / f"{k}.npz") for k in RUNS}

    print(f"volumes over ({T0:g}, {T1:g}] s, per unit depth [m^2]")
    print(f"{'':>26} {'in (ventricle)':>15} {'out (pia)':>12} {'in - out':>12}")
    vols = {}
    for k, label in RUNS.items():
        v_in, v_out = vols[k] = volumes(res[k])
        print(f"{label:>26} {v_in:15.3e} {v_out:12.3e} {v_in - v_out:12.3e}")

    fig = plt.figure(figsize=(12, 8), layout="constrained")
    top = fig.subfigures(2, 1, height_ratios=[1, 1])
    ax_v, ax_p = top[0].subplots(1, 2, sharex=True)
    for ax, i, title in (
        (ax_v, 1, "into the tissue at the ventricle"),
        (ax_p, 2, "out at the pia"),
    ):
        for k, label in RUNS.items():
            f = flows(res[k])
            ax.plot(f[0], f[i], label=label)
        ax.axvspan(T0, T1, color="0.9", zorder=0)
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set(title=f"flow {title}", xlabel="t [s]", ylabel="[m^2/s per unit depth]")
        ax.legend(fontsize=8)

    ax = top[1].subplots()
    bars = ["in (ventricle)", "out (pia)", "in - out (stored)"]
    x = np.arange(len(bars))
    width = 0.8 / len(RUNS)
    for j, (k, label) in enumerate(RUNS.items()):
        v_in, v_out = vols[k]
        b = ax.bar(
            x + (j - (len(RUNS) - 1) / 2) * width, [v_in, v_out, v_in - v_out], width, label=label
        )
        ax.bar_label(b, fmt="%.2e", fontsize=8)
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_xticks(x, bars)
    ax.set(
        title=f"volume over the last two periods ({T0:g}-{T1:g} s)", ylabel="[m^2 per unit depth]"
    )
    ax.legend(fontsize=8)
    fig.savefig(HERE / "figures" / "volumes.png", dpi=150)
