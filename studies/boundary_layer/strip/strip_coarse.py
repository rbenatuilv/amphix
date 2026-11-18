"""Boundary layer on coarse meshes: the strip of strip.yaml with h up to 20 delta."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import amphix

# exact solution (cvr22, 1 Hz, P = 1 Pa): p = Im[p_hat(x) e^{i w t}]
kappa = 1e-16 / 0.8e-3
S = 1e-6 + 1 / (11567 + 2 * 507)
w = 2 * np.pi
delta = np.sqrt(2 * kappa / (w * S))
L = float(60 * delta)  # plain float: the overrides are saved as YAML; three cells at h = 20 delta
k = (1 + 1j) / delta
x = np.linspace(0, L, 601)  # the points of the line `axis`


def p_hat(x):
    return np.sinh(k * (L - x)) / np.sinh(k * L)


def flow_hat(W):  # outward flow through x = 0 [m^2/s per unit depth]
    return -W * kappa * k / np.tanh(k * L)


def fit(t, f):  # f = a sin wt + b cos wt over the last cycle -> a + i b
    A = np.column_stack([np.sin(w * t), np.cos(w * t)])
    (a, b), *_ = np.linalg.lstsq(A, f, rcond=None)
    return a + 1j * b


def last_cycle(t, t_end):  # exactly one period, [t_end - 1, t_end)
    return (t > t_end - 1 - 1e-6) & (t < t_end - 1e-6)


def run(yaml, hd, dt):
    """Run the strip with h = hd * delta; return the errors of p (in the layer) and of the flow."""
    n = int(max(2, round(L / (hd * delta))))
    W = 2 * L / n  # two square cells across: stretched cells are unstable with SIPG
    overrides = {
        "mesh.corners": [[0.0, 0.0], [L, W]],
        "mesh.n": [n, 2],
        "output.lines.paths.axis": {"from": [0.0, W / 2], "to": [L, W / 2], "n": len(x)},
        "time.dt": dt,
        "output.name": f"strip_coarse_{hd}_dt{dt}",
    }
    r = amphix.load_results(amphix.run(amphix.load(yaml, overrides)))
    t_end = round(r["t"][-1])
    s, t = last_cycle(r["lines_t"], t_end), last_cycle(r["t"], t_end)
    ph = fit(r["lines_t"][s], r["line_axis_p"][s])
    # cbflux is the flux averaged over the step [t - dt, t]: it belongs to the midpoint t - dt/2
    flow = fit(r["t"][t] - dt / 2, r["cbflux_left"][t])
    near = np.isfinite(ph) & (x <= 5 * delta)  # the layer; beyond it p_hat ~ 0
    err_p = np.linalg.norm(ph[near] - p_hat(x[near])) / np.linalg.norm(p_hat(x[near]))
    err_flow = abs(flow - flow_hat(W)) / abs(flow_hat(W))
    return err_p, err_flow


if __name__ == "__main__":
    HERE = Path(__file__).parent
    (HERE / "figures").mkdir(exist_ok=True)

    HD = [0.5, 1, 2, 3, 5, 7, 10, 14, 20]  # h / delta
    DT = 0.005  # s

    res = {}
    for hd in HD:
        res[hd] = run(HERE / "strip.yaml", hd, DT)
        ep, ef = res[hd]
        print(f"h/delta = {hd:5.1f}  p {ep:.2e}  flow {ef:.2e}")

    fig, ax = plt.subplots()
    ax.loglog(HD, [res[hd][0] for hd in HD], "o-", label="p, x < 5 delta")
    ax.loglog(HD, [res[hd][1] for hd in HD], "s-", label="surface flow")
    ax.axhline(0.01, color="0.6", lw=0.8)
    ax.set(
        xlabel="h / delta",
        ylabel="relative error",
        title=f"Under-resolved layer (cvr22, 1 Hz, delta = {delta * 1e6:.1f} um, dt = {DT})",
    )
    ax.legend()
    fig.savefig(HERE / "figures" / "strip_coarse.png", dpi=120)
