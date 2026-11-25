"""Fluid through the ventricle and the pia: flow over time and volume over the last two periods.

Case 1 (gamma only) is steady, so any window gives its steady volume. Case 2 (gamma + dP sin) is
read from a long run started from the steady state of gamma, and its volumes are taken over its
last two periods: the start-up residual of the oscillation decays only as k^-3/2 (periods.py),
so the later the window, the closer case 2 comes to case 1.

    case 1:   results/case1.npz      (transmantle.yaml, see its header)
    case 2:   results/<CASE2>.npz    a long run, e.g. 20 or 100 periods:
        amphix run transmantle.yaml --set time.T=100 \
            --set 'initial={from: results/steady_fields.npz}' --name case2_100

cbflux_<wall> is the outward flow per unit depth [m^2/s], averaged over each step; summed times dt
over the steps of the window it gives the volume per unit depth [m^2]. Inflow at the ventricle is
minus its outward volume, outflow at the pia its outward volume; their difference is the fluid
stored.

Figure: figures/volumes.png. Top: the flow through each wall over the last four periods (case 1
as its steady value), the window shaded. Bottom: the volumes over two periods, per case.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import amphix

CASE2 = "case2_100"  # the long run of case 2 (case2_20 or case2_100)
SPAN = 2.0  # the window: the last two periods [s]
SHOWN = 4.0  # top plots: the last four periods [s]


def flows(r):
    """Return t and the flows [m^2/s]: into the tissue at the ventricle, out of it at the pia."""
    return r["t"], -r["cbflux_ventricle"], r["cbflux_pial"]


def volumes(r):
    """Return (inflow at the ventricle, outflow at the pia) over the last SPAN seconds [m^2]."""
    t, q_in, q_out = flows(r)
    dt = t[-1] - t[-2]
    w = t > t[-1] - SPAN + 1e-9  # the steps ending in the window
    return q_in[w].sum() * dt, q_out[w].sum() * dt


if __name__ == "__main__":
    HERE = Path(__file__).parent
    (HERE / "figures").mkdir(exist_ok=True)
    r1 = amphix.load_results(HERE / "results" / "case1.npz")
    r2 = amphix.load_results(HERE / "results" / f"{CASE2}.npz")
    t_end = r2["t"][-1]
    runs = {
        "case 1: gamma only (steady)": volumes(r1),
        f"case 2: gamma + dP sin ({t_end - SPAN:g}-{t_end:g} s)": volumes(r2),
    }

    print("volumes over two periods, per unit depth [m^2]")
    print(f"{'':>36} {'in (ventricle)':>15} {'out (pia)':>12} {'in - out':>12}")
    for label, (v_in, v_out) in runs.items():
        print(f"{label:>36} {v_in:15.3e} {v_out:12.3e} {v_in - v_out:12.3e}")

    fig = plt.figure(figsize=(12, 8), layout="constrained")
    top = fig.subfigures(2, 1, height_ratios=[1, 1])
    ax_v, ax_p = top[0].subplots(1, 2, sharex=True)
    for ax, i, title in (
        (ax_v, 1, "into the tissue at the ventricle"),
        (ax_p, 2, "out at the pia"),
    ):
        f2 = flows(r2)
        last = f2[0] > t_end - SHOWN - 1e-9
        ax.plot(f2[0][last], f2[i][last], color="C1", lw=0.8, label="case 2: gamma + dP sin")
        ax.axhline(flows(r1)[i][-1], color="C0", lw=1.5, label="case 1: gamma only (steady)")
        ax.axvspan(t_end - SPAN, t_end, color="0.9", zorder=0)
        ax.set(title=f"flow {title}", xlabel="t [s]", ylabel="[m^2/s per unit depth]")
        ax.legend(fontsize=8)

    ax = top[1].subplots()
    bars = ["in (ventricle)", "out (pia)", "in - out (stored)"]
    x = np.arange(len(bars))
    width = 0.8 / len(runs)
    for j, (label, (v_in, v_out)) in enumerate(runs.items()):
        b = ax.bar(
            x + (j - (len(runs) - 1) / 2) * width, [v_in, v_out, v_in - v_out], width, label=label
        )
        ax.bar_label(b, fmt="%.2e", fontsize=8)
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_xticks(x, bars)
    ax.set(title="volume over two periods", ylabel="[m^2 per unit depth]")
    ax.legend(fontsize=8)
    fig.savefig(HERE / "figures" / "volumes.png", dpi=150)
