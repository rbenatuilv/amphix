"""Run A on the annulus: does a boundary pulse move fluid beyond the wall layers?

Reads results/runA_*.npz (made with annulus.yaml, see the commands in the report) and
measures, on the last cycle, the amplitude of the radial flux and of p - pc along the two
radial lines of the x axis (thick side: pial at x = -10 mm, ventricle at 0.5 mm; thin side:
ventricle at 5.5 mm, pial at 10 mm). On the x axis both walls are crossed at right angles,
so the radial flux is the x component.

Figures:
    annulus_layer.png    cvr22: flux amplitude against distance from each wall, and e^{-d/delta}
    annulus_params.png   cvr22, [M25], [PRV23]: flux amplitude across the thick side
Table (printed): wall flows, bulk / wall flux, and the change between the two meshes.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

A = 66.65  # Pa, pc = A sin(2 pi t)
W = 2 * np.pi
LAM = 11567.0
LINES = {  # name: (x at the start, x at the end, points, wall at the start, wall at the end)
    "thick": (-1.0e-2, 5.0e-4, 2101, "pial", "ventricle"),
    "thin": (5.5e-3, 1.0e-2, 901, "ventricle", "pial"),
}
RUNS = {  # result name: (label, mu, alpha, K, muF, c0)
    "runA_4e-5": ("cvr22", 507.0, 1.0, 1e-16, 0.8e-3, 1e-6),
    "runA_8e-5": ("cvr22, coarse wall", 507.0, 1.0, 1e-16, 0.8e-3, 1e-6),
    "runA_m25": ("[M25]", 1500.0, 0.49, 1.57e-11, 3.5e-3, 1e-6),
    "runA_prv23": ("[PRV23]", 507.0, 1.0, 1.26e-14, 0.8e-3, 3.9e-4),
}


def delta(mu, alpha, K, muF, c0):
    """Penetration depth at 1 Hz [m] (1D estimate)."""
    S = c0 + alpha**2 / (LAM + 2 * mu)
    return np.sqrt(2 * (K / muF) / (W * S))


def fit(t, f):
    """f = c + a sin wt + b cos wt over the rows of t -> a + i b (columns kept apart: NaN stays)."""
    M = np.column_stack([np.ones_like(t), np.sin(W * t), np.cos(W * t)])
    _, a, b = np.linalg.pinv(M) @ f.reshape(len(t), -1)
    return (a + 1j * b).reshape(f.shape[1:])


def last_cycle(t):
    t_end = round(t[-1])
    return (t > t_end - 1 - 1e-6) & (t < t_end - 1e-6)


def measure(path):
    """Amplitudes on the last cycle: radial flux and p - pc on each line, wall flows."""
    r = np.load(path)
    t, ts = r["t"], r["lines_t"]
    dt = t[1] - t[0]
    s, c = last_cycle(ts), last_cycle(t)
    out = {}
    for name in LINES:
        q = fit(ts[s] - dt / 2, r[f"line_{name}_cflux"][s][..., 0])  # step average: mid-step
        p = fit(ts[s], r[f"line_{name}_p"][s]) - A  # pc = A sin: complex amplitude A
        out[name] = (np.abs(q), np.abs(p))
    for wall in ("ventricle", "pial"):
        out[wall] = abs(fit(t[c] - dt / 2, r[f"cbflux_{wall}"][c]))
    return out


if __name__ == "__main__":
    HERE = Path(__file__).parent
    (HERE / "figures").mkdir(exist_ok=True)
    res = {
        k: measure(HERE / "results" / f"{k}.npz")
        for k in RUNS
        if (HERE / "results" / f"{k}.npz").exists()
    }
    xs = {n: np.linspace(x0, x1, npts) for n, (x0, x1, npts, _, _) in LINES.items()}

    # table
    print(
        f"{'run':>20} {'delta':>9} {'vent flow':>10} {'pial flow':>10}"
        f" {'thick mid/wall':>15} {'thin mid/wall':>14}"
    )
    for k, m in res.items():
        d = delta(*RUNS[k][1:])
        ratios = []
        for n in LINES:
            q = m[n][0]
            ratios.append(q[len(q) // 2] / np.nanmax(q))
        print(
            f"{RUNS[k][0]:>20} {d * 1e6:7.1f}um {m['ventricle']:10.3e} {m['pial']:10.3e}"
            f" {ratios[0]:15.2e} {ratios[1]:14.2e}"
        )
    if {"runA_4e-5", "runA_8e-5"} <= res.keys():
        f, c = res["runA_4e-5"], res["runA_8e-5"]
        print("mesh change 8e-5 -> 4e-5:")
        for wall in ("ventricle", "pial"):
            print(f"  {wall} flow {abs(c[wall] / f[wall] - 1):.2%}")
        for n in LINES:
            m = len(f[n][0]) // 2
            print(f"  {n} bulk flux {abs(c[n][0][m] / f[n][0][m] - 1):.2%}")

    # 1. the layer in the brain (cvr22, fine wall mesh)
    if "runA_4e-5" in res:
        d = delta(*RUNS["runA_4e-5"][1:])
        fig, ax = plt.subplots(figsize=(7, 5))
        for n, (x0, x1, _, w0, w1) in LINES.items():
            q, x = res["runA_4e-5"][n][0], xs[n]
            half = len(x) // 2
            for wall, dist, qq in (
                (w0, x[:half] - x0, q[:half]),
                (w1, x1 - x[::-1][:half], q[::-1][:half]),
            ):
                ax.loglog(dist[1:] * 1e3, qq[1:], label=f"{n} side, from the {wall}")
        dd = np.logspace(-6, -3.5, 200)  # 1 um to 0.3 mm
        q0 = np.nanmax(res["runA_4e-5"]["thin"][0])
        ref = q0 * np.exp(-dd / d)
        ax.loglog(dd * 1e3, ref, "k--", label=f"e^(-d/delta), delta = {d * 1e6:.0f} um")
        lo = min(np.nanmin(res["runA_4e-5"][n][0]) for n in LINES)
        ax.set_ylim(lo / 3, q0 * 3)
        ax.set(
            xlabel="distance from the wall [mm]",
            ylabel="amplitude of the radial flux [m/s]",
            title="Run A (cvr22): flux driven by the boundary pulse",
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(HERE / "figures" / "annulus_layer.png", dpi=120)

    # 2. which parameter sets let the pulse reach the bulk (thick side, normalised)
    fig, ax = plt.subplots(figsize=(7, 5))
    for k in ("runA_4e-5", "runA_prv23", "runA_m25"):
        if k in res:
            q = res[k]["thick"][0]
            d = delta(*RUNS[k][1:])
            label = f"{RUNS[k][0]} (delta = {d * 1e3:.2g} mm)"
            ax.semilogy(xs["thick"] * 1e3, q / np.nanmax(q), label=label)
    ax.set(
        xlabel="x [mm]  (pial wall at -10, ventricle at 0.5)",
        ylabel="flux amplitude / its maximum",
        title="Thick side: how far the boundary pulse moves fluid",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "annulus_params.png", dpi=120)
