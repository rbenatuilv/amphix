"""brain/: a preset resolves to the material it stands for, a material without `preset` is its own
keys, keys next to a preset override it, the cvr22 values and sources, the waveforms (zero mean and
the stroke volume of the vascular source) and the geometries."""

import math
import warnings

import dolfinx
import numpy as np
import pytest
import ufl
from cases import annulus
from mpi4py import MPI

from amphix.brain import PRESETS, geometries, waveforms
from amphix.config import from_dict
from amphix.config.validate import ConfigError, ConfigWarning
from amphix.core.expressions import Expressions
from amphix.core.mesh import Measures, build
from amphix.core.model import make_model
from amphix.core.units import Scales
from amphix.physics.poro import parameters

INLINE = {  # a material with every key of both models
    "mu": 1500.0,
    "lambda": 11567.0,
    "alpha": 0.49,
    "c0": 1.0e-6,
    "K": 1.57e-11,
    "muF": 3.5e-3,
    "rho": 1000.0,
    "rho_f": 1000.0,
    "phi": 0.2,
    "tortuosity": 1.6,
}


@pytest.fixture
def inline():
    """A test-only preset of the INLINE material."""
    parameters.register_preset("test_inline", INLINE)
    yield "test_inline"
    del parameters.PRESETS["test_inline"]


def material(raw, model="poro2"):
    return parameters.Material.from_dict(raw, "material", model)


def silently(raw, model):
    """The material, failing on any warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return material(raw, model)


# ---- a preset resolves to the material it stands for --------------------------------------------
@pytest.mark.parametrize("model", ["poro2", "poro3"])
def test_a_preset_is_its_inline_material(model, inline):
    """The same Material, field for field, as the inline material, and no warning (poro2 does not
    read rho_f, phi, tortuosity: a preset gives it only what it reads)."""
    keys = set(parameters.READS[model]) & set(INLINE)
    given = {k: v for k, v in INLINE.items() if k in keys}
    assert silently({"preset": inline}, model) == material(given, model)
    tree = annulus(model, deg=2, h=4e-3)
    a = from_dict({**tree, "material": {"preset": inline}}).section("material")
    assert a == from_dict({**tree, "material": given}).section("material")


def test_the_presets():
    assert set(PRESETS) == {"cvr22"} and set(parameters.PRESETS) == {"cvr22"}


# ---- without `preset`, the keys given -----------------------------------------------------------
@pytest.mark.parametrize("model", ["poro2", "poro3"])
def test_without_preset(model):
    """With the presets registered, a material without `preset` is its own keys."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = material(INLINE, model)
    assert m.lam == INLINE["lambda"] and m.mu == INLINE["mu"]
    for key in ("alpha", "c0", "K", "muF", "rho", "rho_f", "phi", "tortuosity"):
        assert getattr(m, key) == INLINE[key], key
    assert len(caught) == (1 if model == "poro2" else 0)  # rho_f, phi, tortuosity unread


def test_a_preset_is_not_needed_to_warn():
    """A material given in full warns for the keys the model does not read."""
    with pytest.warns(ConfigWarning, match=r"\['phi', 'rho_f', 'tortuosity'\] not read by poro2"):
        material(INLINE, "poro2")


# ---- keys next to a preset override it -----------------------------------------------------------
def test_keys_override_the_preset(inline):
    cvr = PRESETS["cvr22"].values
    m = silently({"preset": "cvr22", "K": 2.0e-16}, "poro2")
    assert m.K == 2.0e-16 and (m.mu, m.lam, m.alpha, m.muF) == (507.0, 11567.0, 1.0, cvr["muF"])
    m = silently({"preset": "cvr22", "mu": 600.0}, "poro2")
    assert (m.mu, m.lam) == (600.0, 11567.0)
    m = silently({"preset": "cvr22", "E": 3.0, "nu": 0.25}, "poro2")  # E-nu replace the Lamé pair
    assert m.mu == pytest.approx(1.2) and m.lam == pytest.approx(1.2) and m.K == 1e-16
    m = silently({"preset": inline, "phi": 0.4}, "poro3")
    assert m.phi == 0.4 and m.tortuosity == 1.6 and m.K == 1.57e-11


def test_overrides_through_the_problem(inline):
    """Problem.with_ (the --set path) overrides the preset, and can change the preset itself."""
    tree = annulus("poro2", h=4e-3)
    problem = from_dict({**tree, "material": {"preset": "cvr22"}})
    k = problem.with_({"material.K": 3.0e-16}).section("material")
    assert k.K == 3.0e-16 and k.mu == 507.0
    other = problem.with_({"material.preset": inline}).section("material")
    assert other == silently({"preset": inline}, "poro2")


def test_preset_errors_and_warnings():
    with pytest.raises(ConfigError, match=r"^material.preset: unknown preset 'nope'"):
        material({"preset": "nope"})
    with pytest.raises(ConfigError, match=r"^material: give either lambda and mu, or E and nu"):
        material({"preset": "cvr22", "nu": 0.3})  # half a pair
    with pytest.raises(ConfigError, match=r"^material.K: must be positive"):
        material({"preset": "cvr22", "K": 0.0})
    with pytest.warns(ConfigWarning, match=r"\['phi'\] not read by poro2"):  # given, not preset
        material({"preset": "cvr22", "phi": 0.3}, "poro2")


# ---- the parameter sets --------------------------------------------------------------------------
def test_cvr22_values_and_sources():
    cvr = PRESETS["cvr22"]
    assert {k: cvr.values[k] for k in ("alpha", "mu", "lambda", "K", "muF", "c0")} == {
        "alpha": 1.0,
        "mu": 507.0,
        "lambda": 11567.0,
        "K": 1e-16,
        "muF": 0.8e-3,
        "c0": 1e-6,
    }
    E, nu = 1500.0, 0.479  # [CVR22]: mu and lambda are these, rounded to the pascal
    assert abs(E / (2 * (1 + nu)) - cvr.values["mu"]) < 0.5
    assert abs(E * nu / ((1 + nu) * (1 - 2 * nu)) - cvr.values["lambda"]) < 0.5
    for key in ("rho", "rho_f", "phi", "tortuosity"):  # from [M25]
        assert cvr.values[key] == INLINE[key], key
        assert "[M25]" in cvr.sources[key]
    assert set(cvr.sources) == set(cvr.values) and all(cvr.sources.values())
    for key in ("mu", "lambda"):
        assert "from E = 1500 Pa, nu = 0.479 Causemann" in cvr.sources[key]
        assert "rounded" in cvr.sources[key]
    assert "10.1186/s12987-022-00376-2" in cvr.sources["K"]


# ---- waveforms -----------------------------------------------------------------------------------
VASCULAR = ("vascular_option1", "vascular_option2")


T_PERIOD = np.linspace(0.0, 1.0, 200_001)


def added(g):
    """int_0^t g over one period (trapezoids), per tissue volume."""
    return np.concatenate([[0.0], np.cumsum((g[1:] + g[:-1]) / 2 * np.diff(T_PERIOD))])


@pytest.mark.parametrize("name", VASCULAR)
def test_vascular_zero_mean_and_stroke_volume(name):
    """Over one period: zero mean, and a stroke volume (max - min of int_0^t g: the volume added
    during inflow) of G / pi = 1.1236e-3, the calibration target 1.12e-3 (1.54 ml / 1369.54 ml)."""
    v = added(waveforms.WAVEFORMS[name](T_PERIOD))
    G = waveforms.G_VASCULAR
    assert abs(v[-1]) < 1e-12 * G  # the mean times the period (cumsum round-off: 3e-15 G)
    stroke = v.max() - v.min()
    assert stroke == pytest.approx(G / math.pi, rel=1e-9)
    assert round(stroke, 5) == 1.12e-3 == round(1.54 / 1369.54, 5)
    assert v[100_000] == pytest.approx(v.max(), rel=1e-12)  # all in the first half period


def test_option2_stroke_volume_needs_weight_at_most_one_half():
    """G / pi up to weight 0.5; above, g changes sign more often and the swing is larger."""
    G = waveforms.G_VASCULAR
    for weight, same in ((0.25, True), (0.5, True), (0.8, False)):
        v = added(waveforms.vascular_option2(T_PERIOD, weight=weight))
        assert abs(v[-1]) < 1e-12 * G
        assert (abs(v.max() - v.min() - G / math.pi) < 1e-9 * G) == same, weight


def test_pulse_and_transmantle():
    t = np.array([0.0, 0.25, 0.5, 0.75])
    assert np.allclose(waveforms.cardiac_pulse(t), [0.0, 66.65, 133.3, 66.65], atol=1e-12)
    assert np.allclose(waveforms.transmantle(t), [0.0, 1.5, 0.0, -1.5], atol=1e-12)
    assert np.allclose(waveforms.vascular(t, option=2, weight=0.0), waveforms.vascular(t, option=1))
    with pytest.raises(ValueError, match="option must be 1 or 2"):
        waveforms.vascular(t, option=3)


@pytest.mark.parametrize("name", sorted(waveforms.WAVEFORMS))
def test_waveforms_as_data(name):
    """`{waveform: name}` in a model is the numpy function, in SI time (T0 = 0.5)."""
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    t_star = dolfinx.fem.Constant(mesh, 0.0)
    ex = Expressions(mesh, Scales(T0=0.5), t_star)
    form = dolfinx.fem.form(ex({"waveform": name}) * ufl.dx)
    for t in (0.1, 0.3, 0.8):
        t_star.value = t / 0.5
        value = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(form))
        assert value == pytest.approx(waveforms.WAVEFORMS[name](t), rel=1e-12, abs=1e-15)


def test_yaml_waveforms_in_a_model():
    """A YAML problem with a preset, a waveform load and a waveform source builds and steps."""
    sides = {"left": "right", "bottom": "top"}
    problem = from_dict(
        {
            "model": "poro2",
            "material": {"preset": "cvr22"},
            "mesh": {"shape": "rectangle", "corners": [[0, 0], [1e-2, 1e-2]], "n": [2, 2]},
            "scales": {"L0": 1e-2, "P0": 1500.0},
            "loads": {"pc": {"waveform": "cardiac_pulse"}},
            "boundaries": {
                **{a: {"displacement": [0, 0], "pressure": "pc"} for a in sides},
                **{b: {"traction": [0, 0], "flux": 0} for b in sides.values()},
            },
            "sources": {"fluid_source": {"waveform": "vascular_option2", "G": 3.53e-3}},
            "discretization": {"deg": 1},
            "time": {"T": 0.2, "dt": 0.1},
        }
    )
    model = make_model(problem)
    assert [round(t, 12) for t in model.steps()] == [0.1, 0.2]
    assert np.isfinite(model.field("p").x.array).all()


# ---- geometries ----------------------------------------------------------------------------------
def test_annulus():
    domain = build(geometries.annulus(), 1e-2)
    assert domain.dim == 2 and set(domain.facet_names) == {"ventricle", "pial"}
    lengths = boundary_sizes(domain)  # polygons inside the circles: 0.16 % and 0.04 % shorter
    assert lengths["ventricle"] == pytest.approx(2 * math.pi * 2.5e-3, rel=3e-3)
    assert lengths["pial"] == pytest.approx(2 * math.pi * 1e-2, rel=1e-3)


def test_shell():
    domain = build(geometries.shell(), 1e-2)
    assert domain.dim == 3 and set(domain.facet_names) == {"ventricle", "pial"}
    areas = boundary_sizes(domain)  # coarse (h = 2 mm on a 2.5 mm hole): the ventricle 12 % short
    assert areas["ventricle"] == pytest.approx(4 * math.pi * 2.5e-3**2, rel=0.15)
    assert areas["pial"] == pytest.approx(4 * math.pi * 1e-2**2, rel=0.02)


def boundary_sizes(domain):
    """The length (2D) or area (3D) of each named boundary, in SI."""
    m, one = Measures(domain, 1), dolfinx.fem.Constant(domain.mesh, 1.0)
    scale = domain.L0 ** (domain.dim - 1)
    return {
        name: scale
        * domain.mesh.comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(one * m.ds(name)))
        )
        for name in domain.facet_names
    }
