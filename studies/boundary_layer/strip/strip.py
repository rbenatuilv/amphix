"""Boundary layer: the strip of strip.yaml against the exact 1D solution."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import amphix

# exact solution (cvr22, 1 Hz, P = 1 Pa): p = Im[p_hat(x) e^{i w t}]
kappa = 1e-16 / 0.8e-3
S = 1e-6 + 1 / (11567 + 2 * 507)
w = 2 * np.pi
delta = np.sqrt(2 * kappa / (w * S))
L, W = 1.11e-4, 1.11e-5  # strip length and width [m]
k = (1 + 1j) / delta


def p_hat(x):
    return np.sinh(k * (L - x)) / np.sinh(k * L)


flow_hat = -W * kappa * k / np.tanh(k * L)  # outward flow through x = 0 [m^2/s per unit depth]
x = np.linspace(0, L, 201)  # the points of the line `axis`


def fit(t, f):  # f = a sin wt + b cos wt over the last cycle -> a + i b
    A = np.column_stack([np.sin(w * t), np.cos(w * t)])
    (a, b), *_ = np.linalg.lstsq(A, f, rcond=None)
    return a + 1j * b


def last_cycle(t, t_end):  # exactly one period, [t_end - 1, t_end)
    return (t > t_end - 1 - 1e-6) & (t < t_end - 1e-6)


def run(yaml, n, dt):
    """Run the strip; return the fitted p_hat on the line and the errors of p and of the flow."""
    name = f"strip_{n}_dt{dt}"
    ny = max(2, round(n * W / L))  # cells across: keeps the cells close to square
    problem = amphix.load(yaml, {"mesh.n": [n, ny], "time.dt": dt, "output.name": name})
    r = amphix.load_results(amphix.run(problem))
    t_end = round(r["t"][-1])
    s, t = last_cycle(r["snap_t"], t_end), last_cycle(r["t"], t_end)
    ph = fit(r["snap_t"][s], r["line_axis_p"][s])
    # cflux is the flux averaged over the step [t - dt, t]: it belongs to the midpoint t - dt/2
    flow = fit(r["t"][t] - dt / 2, r["cbflux_left"][t])
    ok = np.isfinite(ph)  # a line point the mesh search missed gives NaN
    err_p = np.linalg.norm(ph[ok] - p_hat(x[ok])) / np.linalg.norm(p_hat(x[ok]))
    err_flow = abs(flow - flow_hat) / abs(flow_hat)
    return ph, err_p, err_flow


if __name__ == "__main__":
    HERE = Path(__file__).parent
    (HERE / "figures").mkdir(exist_ok=True)

    NS = [3, 4, 5, 7, 10, 14, 20, 28, 40, 57, 80, 113, 160]  # cells along the strip
    DTS = [0.01, 0.005]  # s
    LAYER = [3, 10, 40]  # meshes drawn in the first figure

    res = {}
    for dt in DTS:
        for n in NS:
            res[n, dt] = run(HERE / "strip.yaml", n, dt)
            _, ep, ef = res[n, dt]
            hd = L / n / delta
            print(f"dt = {dt}  n = {n:3d}  h/delta = {hd:.3f}  p {ep:.2e}  flow {ef:.2e}")

    # 1. the layer: amplitude along the strip
    fig, ax = plt.subplots()
    for n in LAYER:
        ph = res[n, DTS[0]][0]
        ax.semilogy(x / delta, np.abs(ph), ".", label=f"h/delta = {L / n / delta:.2f}")
    ax.semilogy(x / delta, np.abs(p_hat(x)), "k-", label="exact")
    ax.set(
        xlabel="x / delta",
        ylabel="amplitude of p [Pa]",
        title=f"Pressure amplitude along the strip (cvr22, 1 Hz, delta = {delta * 1e6:.1f} um)",
    )
    ax.legend()
    fig.savefig(HERE / "figures" / "strip_layer.png", dpi=120)

    # 2. convergence: pressure and surface flow, for both time steps
    hd = [L / n / delta for n in NS]
    fig, ax = plt.subplots()
    for dt, ls in zip(DTS, ["-", "--"], strict=True):
        ax.loglog(hd, [res[n, dt][1] for n in NS], "o" + ls, color="C0", label=f"p, dt = {dt}")
        ax.loglog(hd, [res[n, dt][2] for n in NS], "s" + ls, color="C1", label=f"flow, dt = {dt}")
    ax.set(
        xlabel="h / delta",
        ylabel="relative error",
        title="Mesh convergence of the layer: pressure and surface flow",
    )
    ax.legend()
    fig.savefig(HERE / "figures" / "strip_mesh.png", dpi=120)
