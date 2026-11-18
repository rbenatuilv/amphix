"""post.hdiv under MPI: on 2 and 3 ranks the reconstruction equals the serial one (cflux at fixed
points, the conservative flow through each wall), and every cell's books still close, also the
cells next to another rank's."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from hdiv_parallel import run

MPIRUN = shutil.which("mpirun", path=str(Path(sys.executable).parent)) or shutil.which("mpirun")


def relative(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.abs(a - b).max() / np.abs(a).max()


@pytest.mark.slow
@pytest.mark.skipif(MPIRUN is None, reason="no mpirun")
@pytest.mark.parametrize("model", ["poro2", "poro3"])
def test_ranks_equal_serial(model):
    serial = run(model)
    script = Path(__file__).with_name("hdiv_parallel.py")
    for ranks in (2, 3):
        proc = subprocess.run(
            [MPIRUN, "-n", str(ranks), sys.executable, str(script), model],
            capture_output=True,
            text=True,
            check=True,
            timeout=600,
        )
        par = json.loads(proc.stdout.strip().splitlines()[-1])
        assert par["boundary_cells"] > 0  # the ranks do share facets
        # the solver itself differs at ~1e-11 between rank counts (another LU ordering): cflux
        # must not differ more
        assert relative(serial["flux"], par["flux"]) < 1e-10, ranks
        assert relative(serial["cflux"], par["cflux"]) < 1e-10, ranks
        for key in (k for k in serial if k.startswith("cbflux_")):
            assert relative(serial[key], par[key]) < 1e-11, (ranks, key)
        # every cell, also next to another rank: the RT flow is the solver's to round-off,
        # and the imbalance (the solver's own, ~1e-10 here) at the serial level
        for key in ("rt", "rt_edge"):
            assert max(par[key]) < 1e-11, (ranks, key)
        level = 10 * max(serial["imbalance"])
        for key in ("imbalance", "imbalance_edge"):
            assert max(par[key]) < level, (ranks, key)
