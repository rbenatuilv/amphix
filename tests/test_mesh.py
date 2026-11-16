"""core.mesh: every shape and its names, gmsh import with subdomains, and the checks."""

import numpy as np
import pytest
from meshes import integrals, write_two_subdomains

from amphix.core.mesh import build, check


def counts(domain):
    """Number of boundary facets of each named boundary."""
    values = domain.facet_tags.values
    return {name: int(np.sum(values == tag)) for name, tag in domain.facet_names.items()}


def test_rectangle_and_box():
    rect = build({"shape": "rectangle", "corners": [[0, 0], [2e-2, 1e-2]], "h": 2.5e-3}, L0=1e-2)
    assert rect.mesh.topology.index_map(2).size_global == 2 * 8 * 4  # n = [8, 4] from h
    assert counts(rect) == {"left": 4, "right": 4, "bottom": 8, "top": 8}
    assert np.isclose(rect.mesh.geometry.x[:, 0].max(), 2.0)  # solver units: x / L0
    assert set(np.unique(rect.cell_tags.values)) == {1} and rect.cell_names == {}
    cube = build({"shape": "box", "corners": [[0, 0, 0], [2, 1, 1]], "n": [2, 1, 1]})
    assert cube.mesh.topology.index_map(3).size_global == 12  # 6 tetrahedra per cube
    assert list(cube.facet_names) == ["left", "right", "front", "back", "bottom", "top"]
    assert counts(cube)["left"] == 2 and counts(cube)["bottom"] == 4


@pytest.mark.parametrize(
    "section, names",
    [
        ({"shape": "disc", "radius": 1.0, "h": 0.4}, {"outer"}),
        ({"shape": "ball", "radius": 1.0, "h": 0.5}, {"outer"}),
        (
            {
                "shape": "annulus",
                "radius": 1.0,
                "inner": {"center": [0.3, 0], "radius": 0.3},
                "h": 0.3,
            },
            {"outer", "inner"},
        ),
        (
            {
                "shape": "shell",
                "radius": 1.0,
                "inner": {"center": [0.3, 0, 0], "radius": 0.3},
                "h": 0.4,
            },
            {"outer", "inner"},
        ),
    ],
)
def test_round_shapes(section, names):
    domain = build(section)
    assert set(domain.facet_names) == names
    assert set(np.unique(domain.facet_tags.values)) == set(domain.facet_names.values())
    if "inner" in names:  # the hole is the smaller boundary
        assert counts(domain)["inner"] < counts(domain)["outer"]


def test_renaming():
    domain = build(
        {
            "shape": "annulus",
            "radius": 1.0,
            "inner": {"center": [0, 0], "radius": 0.5},
            "h": 0.4,
            "names": {"inner": "ventricle", "outer": "pial"},
        }
    )
    assert set(domain.facet_names) == {"ventricle", "pial"} and domain.facet_tag("pial") == 1
    with pytest.raises(KeyError, match="no boundary named 'inner'"):
        domain.facet_tag("inner")
    with pytest.raises(ValueError, match="not in the mesh"):
        domain.renamed({"skull": "x"})


def test_gmsh_file_with_subdomains(tmp_path):
    """A square of two cell groups: the physical groups name boundaries and subdomains."""
    write_two_subdomains(tmp_path / "two.msh")
    domain = build(
        {"shape": "gmsh", "file": "two.msh", "names": {"skull": "outer"}}, L0=2.0, root=tmp_path
    )
    assert domain.facet_names == {"outer": 3} and domain.cell_names == {"tissue": 5, "fluid": 7}
    assert set(np.unique(domain.cell_tags.values)) == {5, 7}
    i = integrals(domain)
    assert i["dx_tissue"] == pytest.approx(0.25) and i["dx_fluid"] == pytest.approx(0.25)  # / L0^2


HOLED = {"shape": "annulus", "radius": 1, "h": 0.1, "inner": {"center": [0, 0], "radius": 0.2}}
WALL = {"h": 0.01, "band": 0.05, "growth": 0.1}


@pytest.mark.parametrize(
    "section, match",
    [
        ({"shape": "circle"}, "unknown shape"),
        ({"shape": "disc", "raduis": 1.0, "h": 0.1}, "unknown key"),
        ({"shape": "disc", "h": 0.1}, "missing"),
        ({"shape": "rectangle", "corners": [[0, 0], [1, 1]]}, "give n or h"),
        (
            {"shape": "annulus", "radius": 1, "h": 0.1, "inner": {"center": [0, 0], "r": 0.2}},
            "inner takes",
        ),
        ({"shape": "gmsh", "file": "x.msh", "tags": {"a": 1}}, "unknown key"),  # was tags: facets
        ({"shape": "disc", "radius": 1, "h": 0.1, "wall": WALL}, "unknown key"),  # no hole
        ({**HOLED, "wall": {"h": 0.01, "band": 0.1}}, "wall takes"),
        ({**HOLED, "wall": {**WALL, "h": 0.0}}, "positive numbers"),
    ],
)
def test_checks(section, match):
    with pytest.raises(ValueError, match=match):
        check(section)
