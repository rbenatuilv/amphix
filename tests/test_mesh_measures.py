"""core.mesh measures (by name, h, hF, quadrature degree, SI sizes) and the names: rules."""

import dataclasses

import dolfinx
import numpy as np
import pytest
from meshes import integrals

from amphix.core.mesh import Measures, build, cell_size, mean_size

RECT = {"shape": "rectangle", "corners": [[0, 0], [2e-2, 1e-2]], "n": [4, 2]}  # cells 0.5 x 0.5


def test_measures_by_name():
    d = build(RECT, L0=1e-2)  # solver units: [0, 2] x [0, 1]
    i = integrals(d)
    assert i["dx"] == pytest.approx(2.0) and i["ds_left"] == pytest.approx(1.0)
    assert i["dS"] == pytest.approx(5 + 8 * 0.5 * np.sqrt(2))  # 3 + 2 axis lines, 8 diagonals
    m = Measures(d, degree=3)
    assert m.dx().metadata()["quadrature_degree"] == 3 and m.dS.metadata()["quadrature_degree"] == 3
    assert m.ds(["left", "right"]).subdomain_id() == (1, 2)
    with pytest.raises(KeyError, match="no boundary named"):
        m.ds("pia")


def test_h_hf_and_sizes():
    d = build(RECT, L0=1e-2)
    m, diag = Measures(d), 0.5 * np.sqrt(2)  # every cell: a right triangle of legs 0.5

    def value(form):
        return dolfinx.fem.assemble_scalar(dolfinx.fem.form(form))

    assert value(m.h * m.dx()) == pytest.approx(diag * 2.0)
    assert value(m.hF * m.dS) == pytest.approx(diag * integrals(d)["dS"])
    assert cell_size(d) == pytest.approx(diag * 1e-2)  # SI
    assert mean_size(d) == pytest.approx(np.sqrt(2.0 / 16) * 1e-2)


def test_names_collisions():
    d = build(RECT)
    clash = dataclasses.replace(d, cell_names={"left": 1})  # a cell group named like a boundary
    with pytest.raises(ValueError, match="both a boundary and a subdomain"):
        clash.renamed({"left": "west"})
    two = clash.renamed({"facets": {"left": "west"}, "cells": {"left": "brain"}})
    assert "west" in two.facet_names and two.cell_names == {"brain": 1}
    assert clash.renamed({"right": "east"}).facet_names["east"] == 2  # no clash: plain form works
    with pytest.raises(ValueError, match="same name"):
        d.renamed({"left": "right"})
    with pytest.raises(ValueError, match=r"names\.cells: \['x'\] not in the mesh"):
        d.renamed({"cells": {"x": "y"}})
