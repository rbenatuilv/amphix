"""tmp_path is the same directory on every rank, so that what rank 0 writes the others can read
(under mpirun each rank's pytest has its own temporary root)."""

import re

import pytest
from parallel import shared_dir


@pytest.fixture
def tmp_path(request, tmp_path_factory):
    return shared_dir(tmp_path_factory, re.sub(r"\W", "_", request.node.name)[:30])
