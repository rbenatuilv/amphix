"""The package imports, and every layer is documented."""

import importlib

import pytest

import amphix

LAYERS = ["core", "config", "physics", "post", "io", "brain", "plot"]


def test_version():
    assert amphix.__version__ == "0.0.1"


@pytest.mark.parametrize("layer", LAYERS)
def test_layer_is_documented(layer):
    module = importlib.import_module(f"amphix.{layer}")
    assert module.__doc__ and "Uses only" in module.__doc__
