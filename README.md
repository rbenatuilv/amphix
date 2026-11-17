# amphix

**brAin MultiPHysIcs on fenicsX.** A Python library for brain-tissue mechanics and fluid flow on
[FEniCSx](https://fenicsproject.org) (dolfinx 0.11).

- **Poroelasticity (Biot):** a two-field model (displacement, pressure; `poro2`) and a three-field
  model (displacement, filtration displacement, pressure; `poro3`), discretised with discontinuous
  Galerkin (SIPG) in space, Newmark and the theta-method in time.
- **A conservative velocity:** an H(div) (Raviart–Thomas) reconstruction of the Darcy flux that
  conserves fluid cell by cell, ready to drive a transport model.
- **Problems described in YAML,** in SI units, with brain parameter sets, cardiac waveforms and
  geometries built in.
- **Outputs:** probes, boundary flows, lines, grids and slices, full fields (reloadable on any number
  of MPI ranks) and ParaView files.

## Installation

```bash
conda create -n amphix -c conda-forge python=3.12 fenics-dolfinx=0.11 mpich python-gmsh pyyaml
conda activate amphix
pip install -e .            # pip install -e ".[plot]" for plotting (matplotlib)
```

## Usage

A problem is a YAML file. This one pulses the ventricle of a 2D annulus of brain tissue:

```yaml
model: poro2                       # two-field Biot: displacement d, pressure p
scales: {L0: 1.0e-2, T0: 1.0, P0: 1500.0}
mesh:
  shape: annulus
  radius: 1.0e-2
  inner: {center: [3.0e-3, 0.0], radius: 2.5e-3}
  h: 1.0e-3
  names: {inner: ventricle, outer: pial}
material: {preset: cvr22}          # brain tissue, Causemann et al. 2022
loads:
  pc: {waveform: cardiac_pulse}    # 133.3 (1 - cos 2 pi t) / 2 Pa
boundaries:
  ventricle: {pressure: pc, normal_traction: -pc}
  pial: {pressure: 0, displacement: [0, 0]}
discretization: {deg: 2}
time: {T: 1.0, dt: 0.01, theta: 0.5}
output:                           # nothing is recorded unless asked for
  boundary_flux: conservative      # the flow through each boundary, every step
  probes:                          # every step
    points: {left: [-5.0e-3, 0.0]}
    quantities: [p, cflux]         # cflux: the conservative velocity
  fields:                          # at = a list of times [s], or {from, to, every (steps)}
    quantities: [p, cflux]
    at: {from: 0.0, to: 1.0, every: 10}
```

Run it from the command line (in parallel with `mpirun -n 4 amphix run ...`):

```bash
amphix run annulus.yaml
amphix run annulus.yaml --set time.dt=0.005 --set mesh.h=5e-4 --name fine
```

or from Python, step by step if needed:

```python
import amphix

problem = amphix.load("annulus.yaml", {"time.dt": 0.005})
amphix.run(problem)  # results/annulus.npz and annulus_fields.npz

model = amphix.make_model(problem)
cflux = amphix.Reconstruction(model)
for t in model.steps():
    p = model.quantity("p")  # SI dolfinx Functions
    u = cflux.compute()

results = amphix.load_results("results/annulus.npz")  # {"probe_p": ..., "cbflux_pial": ...}
fields = amphix.load_fields("results/annulus_fields.npz")  # fields.function("cflux", i)
```

## Plotting

`amphix plot` draws a saved run, after it, from its files alone (it needs matplotlib). Lengths are
in mm, everything else in SI; a DG field is drawn cell by cell on its Lagrange lattice, so P2
detail and jumps between cells stay visible.

```bash
amphix plot annulus p cflux --at 0.5                  # results/annulus_fields.npz, nearest snapshot
amphix plot annulus cflux --log --zoom 5.2e-3 6.2e-3 -5e-4 5e-4 --mesh --quiver 20
amphix plot annulus d --component x --average 0 1     # mean over the snapshots in (0, 1]
amphix plot annulus --series probe_p cbflux_pial      # results/annulus.npz, against t
amphix plot annulus cflux --log --gif --from 1 --to 2 # figures/annulus_cflux.gif, one frame per snapshot
```

The PNG goes to `figures/` (`--out` to choose, `--show` to open a window). `--cells` draws one
colour per cell, `--vmin/--vmax` fix the colour scale; an animation keeps one scale for all its
frames (`--out x.mp4` needs ffmpeg). cflux (and cbflux) saved at t is the mean over the step ending
at t, and is labelled so. The same from Python, on any matplotlib axes:

```python
import matplotlib.pyplot as plt
from amphix import load_fields, load_results, plot

fl = load_fields("results/annulus_fields.npz")
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
plot.field(fl, "p", a1, t=0.5)  # returns a FieldPlot: .update(i) redraws
plot.field(fl, "cflux", a2, t=0.5, log=True, quiver=20)
plot.series(load_results("results/annulus.npz"), ["probe_p"], a3)
plot.animate(fl, "p", "figures/p.gif", t_from=1.0, t_to=2.0, fps=10)
```

## Package layout

| package | contents |
|---|---|
| `amphix.core` | units, expressions, meshes, DG operators, time schemes, linear solvers, the `Model` base |
| `amphix.config` | YAML loading (`base:` chains, overrides) and validation |
| `amphix.physics.poro` | the poroelastic models, their forms, and manufactured solutions |
| `amphix.post` | norms, integrals, point sampling, the H(div) reconstruction |
| `amphix.io` | the `Recorder`, result and field files, ParaView output |
| `amphix.brain` | brain parameter sets, cardiac waveforms and geometries |
| `amphix.plot` | plots of saved fields and time series (matplotlib, optional) |

## Development

```bash
pip install -e ".[dev,plot]"
pre-commit install
python -m pytest                          # fast tests
python -m pytest -m "slow or not slow"    # all tests
```
