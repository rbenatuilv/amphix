"""The small bulk flux of the annulus (~1e-12 m/s, far from the walls): physics or numerics?

    python interior.py run [CASE ...]     run the cases (default: all), results/interior_<case>.npz
    python interior.py report             table and figures/interior_*.png

Cases (annulus.yaml, wall mesh h = 4e-5 m as in the smoke run, T = 2 s, fit over [1, 2)):

    base         the annulus: off-centre ventricle, clamped pial wall
    h/2          bulk mesh size halved (5e-4 -> 2.5e-4 m), same wall refinement
    dt/2         time step halved (0.01 -> 0.005 s)
    concentric   the same loads and walls with the ventricle at the centre: a Lame problem, whose
                 div u is uniform, so the undrained bulk pressure is uniform and no bulk flow is
                 expected beyond the wall layers

At each interior probe, the last cycle of p, of the raw flux (-K/muF grad p_h) and of cflux (RT)
is fitted with a sin wt + b cos wt (cflux, a step average, at mid-step t - dt/2). The phase is
that of the larger component against pc = (A/2) sin wt, folded to (-90, 90] degrees (0: in phase
or in antiphase with the load); `coherent` is 1 - the residual of the fit / the signal (rms).
p - B pc uses B = alpha / (alpha^2 + c0 M), M = lambda + 2 mu: the 1D undrained coefficient.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import amphix
from amphix import plot
from amphix.core.mesh import build
from amphix.io.fields import load_fields

HERE = Path(__file__).parent
W = 2 * np.pi
A = 133.3 / 2  # Pa: pc = A sin wt
KAPPA = 1e-16 / 0.8e-3  # m^2 / (Pa s)
B = 1.0 / (1.0 + 1e-6 * (11567.0 + 2 * 507.0))  # alpha = 1, c0 = 1e-6 1/Pa
PROBES = {  # m; all >= 0.5 mm (22 delta) from both walls in every case
    "thick_mid": [-4.75e-3, 0.0],
    "thin_mid": [7.75e-3, 0.0],
    "top": [0.0, 6.0e-3],
    "bottom": [3.0e-3, -6.25e-3],
    "near_vent": [6.0e-3, 0.0],  # thin side, 0.5 mm from the ventricle wall
    "near_pial": [-9.0e-3, 0.0],  # thick side, 1 mm from the pial wall
}
CASES = {  # name: (overrides, maps)
    "base": ({}, True),
    "h/2": ({"mesh.h": 2.5e-4}, False),
    "dt/2": ({"time.dt": 0.005}, False),
    "concentric": ({"mesh.inner.center": [0.0, 0.0]}, True),
}
SNAP = 1.25  # s: pc at its peak (fields saved for the maps)
ZOOM = (-6.0e-3, -3.5e-3, -1.25e-3, 1.25e-3)  # m: a bulk window on the thick side


def stem(case):
    return "interior_" + case.replace("/", "")


def run(case):
    overrides, maps = CASES[case]
    output = {
        "dir": "results",
        "name": stem(case),
        "probes": {"points": PROBES, "quantities": ["p", "flux", "cflux"]},
    }
    if maps:
        output["grid"] = {"n": 161, "quantities": ["p", "cflux"], "at": {"from": 1.0, "to": 2.0}}
        output["fields"] = {"quantities": ["p", "flux", "cflux"], "at": [SNAP]}
    base = {"time.T": 2.0, "mesh.wall.h": 4.0e-5, "output": output}
    problem = amphix.load(HERE / "annulus.yaml", base | overrides)
    return amphix.run(problem)


def fit(t, f):
    """f = c + a sin wt + b cos wt over the rows of t -> (a + i b, coherence) per column."""
    f = f.reshape(len(t), -1)
    M = np.column_stack([np.ones_like(t), np.sin(W * t), np.cos(W * t)])
    coef, *_ = np.linalg.lstsq(M, f, rcond=None)
    resid = f - M @ coef
    signal = (M[:, 1:] @ coef[1:]).std(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        coherent = 1 - resid.std(axis=0) / signal
    return coef[1] + 1j * coef[2], coherent


def vector_fit(t, q):
    """(times, points, 2) -> amplitude of the vector, phase [deg] of its larger component, coh."""
    zx, cx = fit(t, q[..., 0])
    zy, cy = fit(t, q[..., 1])
    amp = np.sqrt(np.abs(zx) ** 2 + np.abs(zy) ** 2)
    big = np.where(np.abs(zx) >= np.abs(zy), zx, zy)
    phase = np.degrees(np.angle(big))
    phase = (phase + 90) % 180 - 90  # a component's sign is a direction: fold to (-90, 90]
    return amp, phase, np.minimum(cx, cy)


def last_cycle(t):
    return (t > 1.0 - 1e-6) & (t < 2.0 - 1e-6)


def measure(case):
    r = amphix.load_results(HERE / "results" / f"{stem(case)}.npz")
    t = r["t"]
    dt = t[1] - t[0]
    s = last_cycle(t)
    out = {"h": r["problem"]["mesh"]["h"], "dt": dt, "ncells": int(r["ncells"])}
    out["q"] = vector_fit(t[s] - dt / 2, r["probe_cflux"][s])
    out["flux"] = vector_fit(t[s], r["probe_flux"][s])  # -K/muF grad p at t itself
    zp, _ = fit(t[s], r["probe_p"][s])
    out["p"] = zp
    out["p - B pc"] = zp - B * A  # pc = A sin wt: complex amplitude A
    if "grid" in r:
        g, gt = r["grid"], r["grid_t"]
        sg = last_cycle(gt)
        out["grid"] = g
        out["grid q"] = vector_fit(gt[sg] - dt / 2, r["grid_cflux"][sg])[0]
        out["grid p"] = fit(gt[sg], r["grid_p"][sg])[0] - B * A
    return out


def cell_noise(case):
    """Cell-to-cell variation of cflux and of the raw flux in the bulk at SNAP: the median over
    neighbouring bulk cells of |q(c1) - q(c2)| / |q| (centroid values). A smooth field gives
    about h / L (~0.05 here); noise gives about 1."""
    fl = load_fields(HERE / "results" / f"{stem(case)}_fields.npz")
    i = int(np.argmin(np.abs(fl.snap_t - SNAP)))
    mesh = fl.domain.mesh
    mesh.topology.create_connectivity(1, 2)
    f2c = mesh.topology.connectivity(1, 2)
    x = mesh.geometry.x
    nodes = mesh.geometry.dofmaps[0]
    centre = x[nodes].mean(axis=1)
    cells = np.arange(len(nodes), dtype=np.int32)
    r = np.hypot(centre[:, 0], centre[:, 1])
    c = fl.problem["mesh"]["inner"]["center"]
    rv = np.hypot(centre[:, 0] - c[0], centre[:, 1] - c[1])
    bulk = (r < 9.0e-3) & (rv > 3.5e-3)  # >= 1 mm from both walls
    pairs = np.array([f2c.links(f) for f in range(f2c.num_nodes) if len(f2c.links(f)) == 2])
    pairs = pairs[bulk[pairs].all(axis=1)]
    out = {}
    for name in ("flux", "cflux"):
        q = fl.function(name, i).eval(centre, cells)
        size = np.linalg.norm(q, axis=1)
        jump = np.linalg.norm(q[pairs[:, 0]] - q[pairs[:, 1]], axis=1)
        out[name] = float(np.median(jump / np.maximum(size[pairs].mean(axis=1), 1e-300)))
    return out


def centre(case):
    return (0.0, 0.0) if case == "concentric" else (3.0e-3, 0.0)


def in_bulk(x, case):
    """The points (n, 2) at least 1 mm from both walls."""
    c = centre(case)
    return (np.hypot(x[:, 0], x[:, 1]) < 9.0e-3) & (
        np.hypot(x[:, 0] - c[0], x[:, 1] - c[1]) > 3.5e-3
    )


def smooth_flux(m, case, deg=4):
    """The physical bulk flux: kappa |grad| of a degree-4 polynomial fitted to the in-phase
    p - B pc of the grid over the bulk (the fit filters the cell-scale noise out). Return the
    flux at each probe [m/s], the range of the fit and the rms of its residual [Pa]."""
    g, d = m["grid"], m["grid p"].real
    ok = np.isfinite(d) & in_bulk(g, case)
    x, y = g[ok, 0] / 1e-2, g[ok, 1] / 1e-2  # cm: a well-scaled basis
    terms = [(i, j) for i in range(deg + 1) for j in range(deg + 1 - i)]
    M = np.column_stack([x**i * y**j for i, j in terms])
    c, *_ = np.linalg.lstsq(M, d[ok], rcond=None)
    flux = []
    for px, py in PROBES.values():
        u, v = px / 1e-2, py / 1e-2
        gx = sum(ci * i * u ** (i - 1) * v**j for ci, (i, j) in zip(c, terms, strict=True) if i)
        gy = sum(ci * j * u**i * v ** (j - 1) for ci, (i, j) in zip(c, terms, strict=True) if j)
        flux.append(KAPPA * np.hypot(gx, gy) / 1e-2)
    return np.array(flux), float(np.ptp(M @ c)), float((d[ok] - M @ c).std())


def cell_sizes(case):
    """The median cell size sqrt(2 area) [m] within 0.2 mm of each probe."""
    p = amphix.load(HERE / "annulus.yaml", {"mesh.wall.h": 4.0e-5} | CASES[case][0])
    mesh = build(p.raw["mesh"], 1.0, HERE).mesh
    X = mesh.geometry.x[mesh.geometry.dofmaps[0]][:, :, :2]
    cen = X.mean(axis=1)
    a, b = X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]
    size = np.sqrt(np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]))
    return [float(np.median(size[np.hypot(*(cen - q).T) < 2e-4])) for q in PROBES.values()]


def report():
    (HERE / "figures").mkdir(exist_ok=True)
    res = {c: measure(c) for c in CASES if (HERE / "results" / f"{stem(c)}.npz").exists()}
    print(f"B = {B:.5f}; 1e-12 m/s <-> grad p = {1e-12 / KAPPA:.1f} Pa/m")
    for c, m in res.items():
        print(f"\n{c}: {m['ncells']} cells, h = {m['h']:.2g} m, dt = {m['dt']:.3g} s")
        print(
            f"  {'probe':>10} {'|cflux|':>10} {'phase':>6} {'coh':>5} {'|flux|':>10}"
            f" {'phase':>6} {'|p|':>8} {'|p-B pc|':>9} {'arg':>5}"
        )
        for j, name in enumerate(PROBES):
            q, ph, co = (v[j] for v in m["q"])
            fq, fph, _ = (v[j] for v in m["flux"])
            d = m["p - B pc"][j]
            print(
                f"  {name:>10} {q:10.3e} {ph:6.1f} {co:5.2f} {fq:10.3e} {fph:6.1f}"
                f" {abs(m['p'][j]):8.3f} {abs(d):9.4f} {np.degrees(np.angle(d)):5.0f}"
            )
        d = m["p - B pc"]
        print(f"  interior p - B pc: from {d.real.min():.4f} to {d.real.max():.4f} Pa (in phase)")
        if c in ("base", "h/2"):
            sizes = ", ".join(f"{s * 1e6:.0f}" for s in cell_sizes(c))
            print(f"  cell size at the probes [um]: {sizes}")
        if "grid" in m:
            flux, span, noise = smooth_flux(m, c)
            print(
                f"  smooth p - B pc over the bulk: range {span:.4f} Pa,"
                f" residual {noise:.1e} Pa rms; kappa |grad| at the probes: "
                + ", ".join(f"{v:.1e}" for v in flux)
            )
    for c in ("base", "concentric"):
        if (HERE / "results" / f"{stem(c)}_fields.npz").exists():
            noise = cell_noise(c)
            print(
                f"{c}: cell-to-cell variation in the bulk at t = {SNAP}: "
                + ", ".join(f"{k} {v:.2f}" for k, v in noise.items())
            )
    figures(res)


def figures(res):
    # 1. amplitude maps: |cflux| and p - B pc over the last cycle
    have = [c for c in ("base", "concentric") if c in res and "grid" in res[c]]
    if have:
        fig, axes = plt.subplots(
            2, len(have), figsize=(5.6 * len(have), 9.5), squeeze=False, layout="constrained"
        )
        n = 161
        for k, c in enumerate(have):
            g = res[c]["grid"] * 1e3
            gx, gy = g[:, 0].reshape(n, n), g[:, 1].reshape(n, n)
            q = res[c]["grid q"].reshape(n, n)
            im = axes[0, k].pcolormesh(
                gx, gy, np.log10(q), cmap="magma", vmin=-14, vmax=-8, shading="nearest"
            )
            fig.colorbar(im, ax=axes[0, k], label="log10 amplitude of cflux [m/s]")
            d = res[c]["grid p"].real
            bulk = np.isfinite(d) & in_bulk(res[c]["grid"], c)
            d = (d - d[bulk].mean()).reshape(n, n)  # the deviation from the bulk mean
            lim = 0.05  # Pa: the base bulk spans about 0.08 Pa
            im = axes[1, k].pcolormesh(
                gx, gy, d, cmap="RdBu_r", shading="nearest", vmin=-lim, vmax=lim
            )
            fig.colorbar(im, ax=axes[1, k], label="in-phase amplitude, minus its bulk mean [Pa]")
            axes[0, k].set_title(f"{c}: |cflux| over the last cycle")
            axes[1, k].set_title(f"{c}: p - B pc - bulk mean (B = {B:.4f})")
            for ax in axes[:, k]:
                ax.set_aspect("equal")
                ax.set(xlabel="x [mm]", ylabel="y [mm]")
                for name, (px, py) in PROBES.items():
                    ax.plot(px * 1e3, py * 1e3, "c+", ms=8)
                    ax.annotate(name, (px * 1e3, py * 1e3), fontsize=7, color="c")
        fig.savefig(HERE / "figures" / "interior_maps.png", dpi=120)
        plt.close(fig)

    # 2. the cell scale: cflux and the raw flux in a bulk window, with the mesh
    path = HERE / "results" / f"{stem('base')}_fields.npz"
    if path.exists():
        fl = load_fields(path)
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
        for ax, (name, kw) in zip(
            axes, [("cflux", {}), ("flux", {}), ("cflux", {"cells": True})], strict=True
        ):
            plot.field(fl, name, ax, t=SNAP, zoom=ZOOM, mesh=True, log=True, **kw)
            ax.set_title(f"{name}{' (cell centres)' if kw else ''}, t = {SNAP} s")
        fig.suptitle("base: the bulk on the thick side (pc at its peak)")
        fig.savefig(HERE / "figures" / "interior_cells.png", dpi=120)
        plt.close(fig)

    # 3. amplitude at each probe, per case
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    xs = np.arange(len(PROBES))
    for k, (c, m) in enumerate(res.items()):
        ax.semilogy(xs + 0.1 * (k - 1.5), m["q"][0], "o", label=f"{c}: cflux")
        ax.semilogy(xs + 0.1 * (k - 1.5), m["flux"][0], "x", color=f"C{k}", label=f"{c}: raw flux")
    ax.set_xticks(xs, list(PROBES), rotation=20)
    ax.set(ylabel="amplitude over the last cycle [m/s]", title="Interior flux at the probes")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(HERE / "figures" / "interior_probes.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        for case in sys.argv[2:] or list(CASES):
            print(f"== {case}", flush=True)
            run(case)
    else:
        report()
