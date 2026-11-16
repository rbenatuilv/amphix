"""Smoke-test check of Run A: bulk follows the pulse, flow only near the walls, no NaN."""

import numpy as np

A = 66.65  # Pa, amplitude of pc = A sin(2 pi t)
r = np.load("results/smoke_4e-5.npz")

ts = r["snap_t"]
i = np.argmax(np.sin(2 * np.pi * ts))  # the stored snapshot closest to the pulse peak
pc = A * np.sin(2 * np.pi * ts[i])
print(f"t = {ts[i]:.2f} s   pc = {pc:.2f} Pa")

for line in ("thick", "thin"):
    p = r[f"line_{line}_p"]
    q = np.linalg.norm(r[f"line_{line}_cflux"], axis=-1)
    m = p.shape[1] // 2  # midpoint of the line
    print(
        f"{line}: p mid = {p[i, m]:.2f} Pa ({100 * (p[i, m] / pc - 1):+.2f} %)"
        f"   |q| peak: mid {np.nanmax(q[:, m]):.2e}"
        f"   10 um from each end {np.nanmax(q[:, 2]):.2e} {np.nanmax(q[:, -3]):.2e}"
    )

for wall in ("ventricle", "pial"):
    f = r[f"cbflux_{wall}"]
    print(f"{wall} flow: min {f.min():.2e}  max {f.max():.2e}")

for k in r.files:
    a = r[k]
    if a.dtype.kind == "f" and np.isnan(a).any():
        print(f"NaN in {k} {a.shape}: {np.isnan(a).sum()}")
