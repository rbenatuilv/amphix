"""config.validate: every error names its YAML path."""

import pytest

from amphix.config.validate import ConfigError, ConfigWarning, check_keys, join, number, unread


def test_error_names_the_path():
    e = ConfigError("material.K", "must be positive")
    assert str(e) == "material.K: must be positive" and e.path == "material.K"
    assert isinstance(e, ValueError) and str(ConfigError("", "empty file")) == "empty file"
    assert (
        join("", "time") == "time" and join("boundaries.cavity", "flux") == "boundaries.cavity.flux"
    )


def test_check_keys():
    check_keys("time", {"T": 1, "dt": 0.1}, {"T", "dt", "t0"}, required={"T", "dt"})
    with pytest.raises(ConfigError, match=r"^time\.dtt: unknown key\(s\) \['dtt'\]"):
        check_keys("time", {"T": 1, "dtt": 0.1}, {"T", "dt"})
    with pytest.raises(ConfigError, match=r"^time: unknown key\(s\) \['a', 'b'\]"):
        check_keys("time", {"a": 1, "b": 2}, {"T"})
    with pytest.raises(ConfigError, match=r"^time: missing \['dt'\]"):
        check_keys("time", {"T": 1}, {"T", "dt"}, required={"T", "dt"})


def test_numbers():
    assert number("mesh.h", 5e-4, positive=True) == 5e-4 and number("x", 3) == 3.0
    assert number("time.startup_steps", 2.0, integer=True) == 2
    for value, kw, match in [
        ("1e-3", {}, "must be a number"),
        (True, {}, "must be a number"),
        (0.0, {"positive": True}, "must be positive"),
        (1.5, {"integer": True}, "must be an integer"),
    ]:
        with pytest.raises(ConfigError, match=rf"^p: {match}"):
            number("p", value, **kw)


def test_unread_is_a_warning():
    with pytest.warns(ConfigWarning, match=r"material: \['rho_f'\] not read by poro2"):
        unread("material", ["rho_f"], "poro2")
