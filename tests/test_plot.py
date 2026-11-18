"""amphix.plot and `amphix plot`: DG fields drawn exactly on their lattice with the jumps between
cells, one value per cell, snapshots and averages, components, log scales, zoom and arrows, time
series, the command line, and amphix without matplotlib."""

import subprocess
import sys
import warnings
from types import SimpleNamespace

import dolfinx
import matplotlib
import numpy as np
import pytest
from mpi4py import MPI
from PIL import Image
from test_hdiv import square

import amphix.physics.poro  # noqa: F401
from amphix.cli import main
from amphix.config import from_dict
from amphix.core.mesh import build
from amphix.core.model import make_model
from amphix.io.fields import FieldWriter, load_fields
from amphix.io.recorder import Recorder
from amphix.io.results import load_results
from amphix.plot import FieldPlot, Lattice, PlotError, animate, field, series
from amphix.plot import animation as animation_module
from amphix.plot import lattice as lattice_module

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROBES = {"a": [0.25, 0.5], "b": [0.75, 0.25]}


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """A poro2 run (P2) recorded every step: results/run.npz and results/run_fields.npz."""
    root = tmp_path_factory.mktemp("plot")
    output = {
        "dir": "results",
        "name": "run",
        "boundary_flux": ["raw", "conservative"],
        "probes": {"points": PROBES, "quantities": ["p", "cflux"]},
        "fields": {"quantities": ["d", "p", "cflux"], "at": {}},
    }
    tree = {**square("poro2", 2, 1.0).raw, "output": output}
    model = make_model(from_dict(tree, "run", root))
    rec = Recorder(model)
    model.run([rec])
    rec.save()
    return root


@pytest.fixture(scope="module")
def fl(run):
    return load_fields(run / "results" / "run_fields.npz")


def poly(x):
    return 1.0 + 2.0 * x[0] - 3.0 * x[1] + 0.5 * x[0] ** 2 - x[0] * x[1] + 2.0 * x[1] ** 2


def write(path, deg, expr, jump=False, shape=()):
    """A fields file of one DG field `f` on a 2 m x 1 m rectangle (cell index added: jumps)."""
    domain = build({"shape": "rectangle", "corners": [[0, 0], [2, 1]], "n": [3, 2]})
    V = dolfinx.fem.functionspace(domain.mesh, ("DG", deg, shape))
    f = dolfinx.fem.Function(V)
    f.interpolate(expr)
    if jump:
        ncells = domain.mesh.topology.index_map(2).size_local
        for c in range(ncells):
            f.x.array[V.dofmap.cell_dofs(c)] += 10.0 * (c + 1)
    writer = FieldWriter(domain, {"f": lambda: f})
    writer(0.0)
    writer.save(path, SimpleNamespace(raw={}))
    return load_fields(path, MPI.COMM_SELF)


@pytest.mark.parametrize(("deg", "expr"), [(1, lambda x: 1 + 2 * x[0] - 3 * x[1]), (2, poly)])
def test_exact_at_the_lattice(tmp_path, deg, expr):
    """A P1 or P2 field is drawn with its exact values at every sub-triangle vertex."""
    fields = write(tmp_path / "f.npz", deg, expr)
    fp = field(fields, "f")
    tri = fp.triangulation
    assert len(tri.triangles) == 12 * deg**2  # k^2 sub-triangles in each of 12 cells
    want = expr(np.array([tri.x, tri.y]) / 1e3)  # mm -> m
    assert np.allclose(fp.image.get_array(), want, rtol=0, atol=1e-12)
    assert fp.ax.get_xlabel() == "x [mm]" and fp.ax.get_title() == "f, t = 0 s"


def test_jumps_kept(tmp_path):
    """The points of a shared edge belong to each cell: two values at one place."""
    fields = write(tmp_path / "f.npz", 1, lambda x: x[0], jump=True)
    fp = field(fields, "f")
    xy = np.round(np.column_stack([fp.triangulation.x, fp.triangulation.y]), 9)
    _, where, count = np.unique(xy, axis=0, return_inverse=True, return_counts=True)
    values = np.asarray(fp.image.get_array())
    shared = np.flatnonzero(count > 1)
    assert len(shared) > 0
    for s in shared:
        assert np.ptp(values[where.ravel() == s]) > 1.0


def test_one_value_per_cell(tmp_path):
    fields = write(tmp_path / "f.npz", 1, lambda x: 1 + 2 * x[0] - 3 * x[1])
    fp = field(fields, "f", cells=True, mesh=True)
    lat = fp.lattice
    assert lat.per_cell and len(fp.image.get_array()) == lat.ncells == 12
    centroid = lat.points[:, :2]
    assert np.allclose(fp.image.get_array(), 1 + 2 * centroid[:, 0] - 3 * centroid[:, 1])
    assert fp.edges is not None


def test_unsupported_meshes():
    cube = dolfinx.mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    with pytest.raises(NotImplementedError, match="3D"):
        Lattice(cube, 1)
    quads = dolfinx.mesh.create_unit_square(
        MPI.COMM_SELF, 1, 1, dolfinx.mesh.CellType.quadrilateral
    )
    with pytest.raises(PlotError, match="triangle"):
        Lattice(quads, 1)


def test_snapshots_and_average(fl):
    assert np.allclose(fl.snap_t, [0, 0.05, 0.1, 0.15, 0.2, 0.25])
    with pytest.warns(UserWarning, match=r"nearest one, t = 0\.1 s"):
        fp = field(fl, "p", t=0.11)
    assert fp.ax.get_title() == "p, t = 0.1 s"
    exact = fp.lattice.values(fl.function("p", 2))[:, 0]
    assert np.array_equal(fp.image.get_array(), exact)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert field(fl, "p", t=0.1).ax.get_title() == "p, t = 0.1 s"
    assert field(fl, "p").ax.get_title() == "p, t = 0.25 s"  # default: the last
    fp = field(fl, "cflux", average=(0.05, 0.2), component="x")
    mean = fp.lattice.values(fl.average("cflux", 0.05, 0.2))[:, 0]
    assert np.array_equal(fp.image.get_array(), mean)
    assert fp.ax.get_title() == "cflux, mean over (0.05, 0.2] s"
    assert field(fl, "cflux", t=0.2).ax.get_title() == "cflux, step ending at t = 0.2 s"
    with pytest.raises(PlotError, match="no saved time"):
        field(fl, "p", average=(0.3, 0.4))
    with pytest.raises(PlotError, match="not in"):
        field(fl, "q")
    with pytest.raises(NotImplementedError):
        field(fl, "p", slice={"normal": "z", "at": 0})


def test_components(fl):
    raw = field(fl, "cflux", t=0.2).lattice.values(fl.function("cflux", 4))
    x, y, mag = (
        np.asarray(field(fl, "cflux", t=0.2, component=c).image.get_array())
        for c in ("x", "y", "mag")
    )
    assert np.array_equal(x, raw[:, 0]) and np.array_equal(y, raw[:, 1])
    assert np.allclose(mag, np.hypot(x, y)) and np.abs(x).max() > 0
    fp = field(fl, "d", component="x")
    assert fp.colorbar.ax.get_ylabel() == "d_x [m]"
    assert field(fl, "cflux").colorbar.ax.get_ylabel() == "|cflux| [m/s]"
    with pytest.raises(PlotError, match="no component z on a 2D mesh"):
        field(fl, "cflux", component="z")
    with pytest.raises(PlotError, match="scalar"):
        field(fl, "p", component="x")


def test_log_scale(tmp_path):
    """The zeros of a magnitude are left out; the automatic scale spans at most DECADES
    decades. A signed quantity on a log scale must be positive where it is shown."""

    def vector(x):  # zero at the vertices x = 0 and 2/3 (P1 interpolation)
        return np.stack([np.maximum(x[0] - 0.7, 0.0), np.zeros_like(x[0])])

    fields = write(tmp_path / "f.npz", 1, vector, shape=(2,))
    fp = field(fields, "f", log=True)
    v = fp.image.get_array()
    masked, x = np.ma.getmaskarray(v), np.asarray(fp.triangulation.x)
    assert np.array_equal(masked, v.data <= 0)  # eval may give 1e-17 instead of 0 at x = 2/3
    assert masked[x < 1].sum() >= (x < 600).sum() and not masked[x > 700].any()
    assert isinstance(fp.image.norm, matplotlib.colors.LogNorm)
    assert fp.image.norm.vmax == pytest.approx(1.3) and fp.image.norm.vmin >= 1.3e-6
    hidden = fp.triangulation.mask
    assert hidden.any() and not hidden.all()
    fixed = field(fields, "f", log=True, vmin=1e-3, vmax=2.0)
    assert (fixed.image.norm.vmin, fixed.image.norm.vmax) == (1e-3, 2.0)
    with pytest.raises(PlotError, match=r"log scales are for magnitudes: f_x is not positive"):
        field(fields, "f", log=True, component="x")  # zero on the left
    signed = write(tmp_path / "g.npz", 1, lambda x: x[0] - 0.7)
    with pytest.raises(PlotError, match=r"magnitudes: f is not positive everywhere \(min -0\.7\)"):
        field(signed, "f", log=True)
    assert field(signed, "f", log=True, zoom=(1.0, 2.0, 0.0, 1.0)).image.norm.vmax == pytest.approx(
        1.3
    )
    positive = write(tmp_path / "h.npz", 1, lambda x: x[0] + 1.0)
    assert field(positive, "f", log=True).image.norm.vmin == pytest.approx(1.0)


def test_zoom_and_arrows(tmp_path, fl):
    """The automatic scale spans the shown region; about N arrows of unit length, in the mesh."""
    fields = write(tmp_path / "f.npz", 1, lambda x: x[0])
    fp = field(fields, "f", zoom=(0.0, 0.7, 0.0, 1.0))  # vertices at x = 0, 2/3, 4/3, 2
    assert fp.ax.get_xlim() == (0.0, 700.0) and fp.image.norm.vmax == pytest.approx(2 / 3)
    fp = field(fl, "cflux", t=0.2, quiver=12, log=True)
    xy, u = fp.arrows.get_offsets(), np.hypot(fp.arrows.U, fp.arrows.V)
    assert 8 <= len(xy) <= 16 and np.allclose(u, 1.0)
    assert xy.min() > 0 and xy.max() < 1e3  # mm, inside the unit square
    with pytest.raises(PlotError, match="p is not a vector"):
        field(fl, "p", quiver=10)


def test_series(run):
    r = load_results(run / "results" / "run.npz")
    _, ax = plt.subplots()
    lines = series(r, "probe_p", ax)
    assert [line.get_label() for line in lines] == ["p at a", "p at b"]
    assert np.array_equal(lines[1].get_ydata(), r["probe_p"][:, 1])
    assert ax.get_ylabel() == "p [Pa]" and ax.get_xlabel() == "t [s]"
    _, ax = plt.subplots()
    mag, x = series(r, "probe_cflux", ax)[0], series(r, "probe_cflux", ax, component="x")[0]
    assert np.allclose(mag.get_ydata(), np.linalg.norm(r["probe_cflux"][:, 0], axis=1))
    assert np.array_equal(x.get_ydata(), r["probe_cflux"][:, 0, 0])
    assert x.get_label() == "cflux_x at a"
    _, ax = plt.subplots()
    flows = series(run / "results" / "run.npz", ["bflux_left", "cbflux_left"], ax)
    assert flows[1].get_label() == "cbflux_left, mean over the step ending at t"
    assert ax.get_ylabel() == "[m²/s]"
    with pytest.raises(PlotError, match="cbflux_pial: not a time series"):
        series(r, "cbflux_pial", ax)


def test_command_line(run, monkeypatch, capsys):
    monkeypatch.chdir(run)
    assert main(["plot", "run", "p", "cflux", "--at", "0.12", "--quiver", "9"]) == 0
    out = capsys.readouterr()
    assert (run / "figures" / "run_p_cflux_t0.1.png").is_file()
    assert "saved figures/run_p_cflux_t0.1.png" in out.out
    assert "warning: no snapshot at t = 0.12 s" in out.err
    assert main(["plot", "results/run_fields.npz", "d", "--average", "0", "0.25"]) == 0
    assert (run / "figures" / "run_d_avg0-0.25.png").is_file()
    assert main(["plot", "run", "p", "--cells", "--mesh", "--out", "out/p.png"]) == 0
    assert (run / "out" / "p.png").is_file()
    assert main(["plot", "run", "--series", "probe_p", "cbflux_left"]) == 0
    assert (run / "figures" / "run_probe_p_cbflux_left_series.png").is_file()
    capsys.readouterr()
    for argv, error in [
        (["run", "q"], "q: not in results/run_fields.npz (saved: d, p, cflux)"),
        (["nope", "p"], "results/nope_fields.npz: no such file"),
        (["run", "p", "--component", "x"], "p is a scalar: it has no component x"),
        (["run", "cflux", "--component", "z"], "cflux: no component z on a 2D mesh"),
        (["run", "--series", "probe_p", "--log"], "--series: no fields or --log"),
        (["run", "p", "--log"], "log scales are for magnitudes: p is not positive everywhere"),
        (["run"], "give the fields to draw, or --series KEY ..."),
        (["results/run.npz", "p"], "results/run.npz: not a fields file"),
    ]:
        assert main(["plot", *argv]) == 2, argv
        err = capsys.readouterr().err.strip()
        assert err.startswith(f"amphix plot: {error}") and "\n" not in err, argv


def test_amphix_without_matplotlib(tmp_path):
    """import amphix needs no matplotlib; amphix plot says what is missing."""
    code = (
        "import sys; sys.modules['matplotlib'] = None\n"
        "import amphix\n"
        "from amphix.cli import main\n"
        "sys.exit(main(['plot', 'run', 'p']))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=300,
        check=False,
    )
    assert done.returncode == 2, done.stderr
    assert "amphix plot: amphix.plot needs matplotlib" in done.stderr


def test_fieldplot_updates(fl):
    """Built once, redrawn: the arrays change, the triangulation does not."""
    fp = FieldPlot(fl, "p", vmin=-1.0, vmax=1.0)
    tri = fp.triangulation
    fp.update(1)
    first = np.array(fp.image.get_array())
    fp.update(5)
    assert fp.triangulation is tri and not np.array_equal(first, fp.image.get_array())
    assert fp.ax.get_title() == "p, t = 0.25 s" and fp.image.norm.vmax == 1.0


def test_animate(fl, tmp_path, monkeypatch):
    """One frame per snapshot in [t_from, t_to], one colour scale, one triangulation."""
    built, clims = [], []
    monkeypatch.setattr(lattice_module, "_LATTICES", {})

    class Counted(Lattice):
        def __init__(self, *args, **kwargs):
            built.append(args[1:])
            super().__init__(*args, **kwargs)

    update = FieldPlot.update

    def recorded(self, what, title=None):
        update(self, what, title)
        clims.append((self.image.get_clim(), self.ax.get_title()))

    monkeypatch.setattr(lattice_module, "Lattice", Counted)
    monkeypatch.setattr(FieldPlot, "update", recorded)
    path = animate(fl, "cflux", tmp_path / "a" / "q.gif", t_from=0.05, t_to=0.2, fps=5)
    with Image.open(path) as im:
        assert im.n_frames == 4
    deg = fl.space("cflux").ufl_element().degree
    assert built == [(deg, False)]  # one lattice, built once for the four frames
    assert [title for _, title in clims] == [
        f"cflux, step ending at t = {t} s" for t in (0.05, 0.1, 0.15, 0.2)
    ]
    assert len({clim for clim, _ in clims}) == 1
    values = [FieldPlot(fl, "cflux").values(i) for i in (1, 2, 3, 4)]
    lo, hi = clims[0][0]
    assert lo == min(v.min() for v in values) and hi == max(v.max() for v in values)
    clims.clear()
    animate(fl, "p", tmp_path / "p.gif", t_to=0.1, vmin=-1.0)  # from the first, vmin fixed
    assert len(clims) == 3 and {clim[0] for clim, _ in clims} == {-1.0}
    assert len({clim for clim, _ in clims}) == 1


def test_animate_writers(fl, tmp_path, monkeypatch):
    with pytest.raises(PlotError, match="no snapshot in"):
        animate(fl, "p", tmp_path / "p.gif", t_from=0.3)
    with pytest.raises(PlotError, match=r"\.gif or an \.mp4"):
        animate(fl, "p", tmp_path / "p.avi")
    monkeypatch.setattr(animation_module.animation.writers, "is_available", lambda name: False)
    with pytest.raises(PlotError, match="needs ffmpeg"):
        animate(fl, "p", tmp_path / "p.mp4")


@pytest.mark.slow
def test_animate_mp4(fl, tmp_path):
    if not animation_module.animation.writers.is_available("ffmpeg"):
        pytest.skip("no ffmpeg")
    assert animate(fl, "p", tmp_path / "p.mp4", t_to=0.1).stat().st_size > 0


@pytest.mark.slow
def test_command_line_gif(run, monkeypatch, capsys):
    monkeypatch.chdir(run)
    assert main(["plot", "run", "cflux", "--gif", "--from", "0.1", "--log", "--fps", "4"]) == 0
    with Image.open(run / "figures" / "run_cflux.gif") as im:
        assert im.n_frames == 4
    assert main(["plot", "run", "d", "--gif", "--out", "out/d.gif", "--component", "y"]) == 0
    with Image.open(run / "out" / "d.gif") as im:
        assert im.n_frames == 6
    capsys.readouterr()
    for argv, error in [
        (["run", "p", "d", "--gif"], "--gif: one field, without --at, --average or --show"),
        (["run", "p", "--gif", "--at", "0.1"], "--gif: one field, without --at, --average"),
        (["run", "p", "--from", "0.1"], "--from: only with --gif"),
        (["run", "--series", "probe_p", "--fps", "3"], "--series: no fields or --fps"),
        (["run", "p", "--gif", "--to", "-1"], "no snapshot in [None, -1.0]"),
    ]:
        assert main(["plot", *argv]) == 2, argv
        assert error in capsys.readouterr().err, argv
