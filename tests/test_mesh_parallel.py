"""Every mesh source ghosts the cells across shared facets (GhostMode.shared_facet): on
two ranks the interior-facet integral and the boundary integrals by name equal the serial ones."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from meshes import domains, ghosts, integrals, write_two_subdomains
from parallel import serial_only

MPIRUN = shutil.which("mpirun", path=str(Path(sys.executable).parent)) or shutil.which("mpirun")


@serial_only("it compares itself with its own mpirun run")
@pytest.mark.skipif(MPIRUN is None, reason="no mpirun")
def test_two_ranks_equal_serial(tmp_path):
    msh = write_two_subdomains(tmp_path / "two.msh")
    serial = {k: integrals(d) for k, d in domains(msh).items()}
    assert all(ghosts(d) == 0 for d in domains(msh).values())  # serial: nothing to ghost
    script = Path(__file__).with_name("meshes.py")
    run = subprocess.run(
        [MPIRUN, "-n", "2", sys.executable, str(script), str(msh)],
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    parallel = json.loads(run.stdout.strip().splitlines()[-1])
    for name, values in serial.items():
        assert parallel[name]["ghosts"] > 0, name  # each rank holds ghost cells
        for key, v in values.items():
            assert parallel[name][key] == pytest.approx(v, rel=1e-12), (name, key)
