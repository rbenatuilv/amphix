"""Brain annulus: results at physiological amplitudes, built from the unit cases (run.py).

    python analyse.py

The real loading is a sum of the unit cases (the model is linear):
    heartbeat              = A_Cc Cc + A_Dc Dc
    heartbeat + breathing  = heartbeat + A_Cr Cr + A_Dr Dr

Figures (figures/), one per parameter set:
    flux_<pset>.png   flow through each wall over 4 s: heartbeat, heartbeat + breathing
    split_<pset>.png  heartbeat flow through each wall: common pulse, difference, total
    maps_<pset>.png   heartbeat: amplitude of the pressure and of the fluid speed over the slice
    excursion.png     fluid excursion at each wall, heartbeat vs heartbeat + breathing, against
                      the solute diffusion length
Printed: peak-to-peak displaced volume at each wall over 4 s, and the excursion (volume / length).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import amphix
from amphix.plot import FieldPlot

HERE = Path(__file__).parent
WALLS = {"ventricle": 2 * np.pi * 1.3e-2, "pial": 2 * np.pi * 6.5e-2}  # wall: length [m]
OMEGA = {"c": 2 * np.pi, "r": 0.5 * np.pi}  # heartbeat, breathing [rad/s]
D_SOLUTE = 4e-10  # m^2/s: effective diffusivity of a small solute in the ECS (D / tortuosity^2)

# physiological amplitudes [Pa] (logbook step 9)
AMP = {
    "Cc": 133.3 / 2,  # ICP pulse, 1 mmHg peak to peak
    "Dc": 194.6 * 0.052,  # transmantle gradient x mean thickness
    "Cr": 133.3 / 2 / 2.85,  # provisional: no measured value yet
    "Dr": 69.3 * 0.052,
}


def wall_amplitudes(name):
    """Complex amplitude Z of the flow through each wall, f = Re[Z e^{iwt}], on the last period."""
    r = np.load(HERE / "results" / f"{name}.npz")
    t, w = r["t"], OMEGA[name[-1]]
    dt, last = t[1] - t[0], t > t[-1] - 2 * np.pi / w + 1e-9
    ts = t[last] - dt / 2  # step averages belong to the mid-step
    M = np.column_stack([np.ones_like(ts), ts - ts.mean(), np.sin(w * ts), np.cos(w * ts)])
    out = {}
    for wall in WALLS:
        coef = np.linalg.pinv(M) @ r[f"cbflux_{wall}"][last]
        out[wall] = coef[3] - 1j * coef[2]  # a sin + b cos = Re[(b - ia) e^{iwt}]
    return out


def combined(pset, pia, cases):
    """Sum of AMP[case] x Z[case] for each wall."""
    Z = {c: wall_amplitudes(f"{pset}_{pia}_{c}") for c in cases}
    return {wall: sum(AMP[c] * Z[c][wall] for c in cases) for wall in WALLS}


def signal(Z, w, t):
    return np.real(Z * np.exp(1j * w * t))


def excursion(heart, breath=None):
    """Peak-to-peak displaced volume over one breathing cycle (4 s), from flow amplitudes.

    The displaced volume is the time integral of the flow, Re[Z / (iw) e^{iwt}]. For heartbeat
    alone this is 2|Z|/w, the volume that enters during half a heartbeat and leaves in the other.
    """
    t = np.linspace(0, 4, 4001)
    V = signal(heart / (1j * OMEGA["c"]), OMEGA["c"], t)
    if breath is not None:
        V = V + signal(breath / (1j * OMEGA["r"]), OMEGA["r"], t)
    return V.max() - V.min()


def figure_flux(pset):
    t = np.linspace(0, 4, 801)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), layout="constrained")
    for ax, pia in zip(axes, ("fixed", "free"), strict=True):
        heart = combined(pset, pia, ("Cc", "Dc"))
        breath = combined(pset, pia, ("Cr", "Dr"))
        for (wall, length), colour in zip(WALLS.items(), ("C0", "C1"), strict=True):
            h = signal(heart[wall], OMEGA["c"], t)
            b = signal(breath[wall], OMEGA["r"], t)
            ax.plot(t, h, ":", color=colour, label=f"{wall}, heartbeat")
            ax.plot(t, h + b, "-", color=colour, label=f"{wall}, heartbeat + breathing")
            v_h = excursion(heart[wall])
            v_hb = excursion(heart[wall], breath[wall])
            print(
                f"{pset:>6} {pia:>5} {wall:>9}: "
                f"heartbeat {v_h:.2e} m^2 ({v_h / length * 1e9:6.1f} nm), "
                f"heartbeat + breathing {v_hb:.2e} m^2 ({v_hb / length * 1e9:6.1f} nm)"
            )
        ax.set(title=f"{pset}, {pia} pia", xlabel="t [s]", ylabel="outward flow [m^2/s]")
    axes[0].legend(fontsize=8)
    fig.savefig(HERE / "figures" / f"flux_{pset}.png", dpi=150)


def figure_split(pset):
    t = np.linspace(0, 2, 401)
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), layout="constrained", sharex=True)
    for j, pia in enumerate(("fixed", "free")):
        common = combined(pset, pia, ("Cc",))
        diff = combined(pset, pia, ("Dc",))
        for i, wall in enumerate(WALLS):
            ax = axes[i, j]
            c = signal(common[wall], OMEGA["c"], t)
            d = signal(diff[wall], OMEGA["c"], t)
            ax.plot(t, c, "--", label="common pulse (1 mmHg on both walls)")
            ax.plot(t, d, ":", label="difference (transmantle, at the ventricle)")
            ax.plot(t, c + d, "k-", label="total")
            ax.set(title=f"{pset}, {pia} pia, {wall}", ylabel="outward flow [m^2/s]")
    axes[1, 0].set_xlabel("t [s]")
    axes[1, 1].set_xlabel("t [s]")
    axes[0, 0].legend(fontsize=8)
    fig.savefig(HERE / "figures" / f"split_{pset}.png", dpi=150)


def amplitude(pset, pia, name):
    """Heartbeat amplitude of a field, from its four snapshots at the quarter periods.

    At phases pi/2, pi, 3pi/2, 2pi, a field c + Re[Z e^{iwt}] takes the values c - Im Z, c - Re Z,
    c + Im Z, c + Re Z, so Re Z = (f4 - f2)/2 and Im Z = (f3 - f1)/2. (The velocity snapshots are
    step averages, half a step earlier: a phase shift of 0.03 rad, negligible.)
    """
    Z, fields = 0, None
    for case in ("Cc", "Dc"):
        fl = amphix.load_fields(HERE / "results" / f"{pset}_{pia}_{case}_fields.npz")
        f = [fl.function(name, i).x.array for i in range(4)]
        Z = Z + AMP[case] * ((f[3] - f[1]) / 2 + 1j * (f[2] - f[0]) / 2)
        if fields is None:
            fields = fl
    g = fields.function(name, 0)
    g.x.array[:] = np.abs(Z)  # per component: the speed is their magnitude
    return fields, g


def figure_maps(pset):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), layout="constrained")
    for j, pia in enumerate(("fixed", "free")):
        fl, p = amplitude(pset, pia, "p")
        FieldPlot(fl, "p", axes[0, j]).update(p, f"{pset}, {pia} pia: pressure amplitude [Pa]")
        fl, v = amplitude(pset, pia, "cflux")
        FieldPlot(fl, "cflux", axes[1, j], log=True).update(
            v, f"{pset}, {pia} pia: speed amplitude [m/s]"
        )
    fig.savefig(HERE / "figures" / f"maps_{pset}.png", dpi=150)


def figure_excursion():
    """Fluid excursion (displaced volume / wall length) at each wall: heartbeat vs + breathing."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained", sharey=True)
    for ax, pset in zip(axes, ("cvr22", "prv23"), strict=True):
        labels, heart_nm, both_nm = [], [], []
        for pia in ("fixed", "free"):
            heart = combined(pset, pia, ("Cc", "Dc"))
            breath = combined(pset, pia, ("Cr", "Dr"))
            for wall, length in WALLS.items():
                labels.append(f"{pia} pia\n{wall}")
                heart_nm.append(excursion(heart[wall]) / length * 1e9)
                both_nm.append(excursion(heart[wall], breath[wall]) / length * 1e9)
        x = np.arange(len(labels))
        ax.bar(x - 0.2, heart_nm, 0.4, label="heartbeat")
        ax.bar(x + 0.2, both_nm, 0.4, label="heartbeat + breathing")
        for T, style in ((1, "--"), (4, ":")):
            ax.axhline(
                np.sqrt(D_SOLUTE * T) * 1e9,
                color="k",
                ls=style,
                label=f"solute diffusion length in {T} s",
            )
        ax.set(
            title=pset, xticks=x, xticklabels=labels, yscale="log", ylabel="fluid excursion [nm]"
        )
    axes[0].legend(fontsize=8)
    fig.savefig(HERE / "figures" / "excursion.png", dpi=150)


if __name__ == "__main__":
    (HERE / "figures").mkdir(exist_ok=True)
    for pset in ("cvr22", "prv23"):
        figure_flux(pset)
        figure_split(pset)
        figure_maps(pset)
    figure_excursion()
