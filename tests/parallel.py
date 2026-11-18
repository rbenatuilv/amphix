"""The suite also runs under mpirun (mpirun -n 2 python -m pytest): helpers for the tests that
read files, print, or reduce values, so that every rank checks the same thing."""

import shutil
from pathlib import Path

import dolfinx
import pytest
from mpi4py import MPI

COMM = MPI.COMM_WORLD


def serial_only(reason):
    """Skip a test under mpirun: it is serial by design (the reason says why)."""
    return pytest.mark.skipif(COMM.size > 1, reason=f"serial by design: {reason}")


def on_root(value):
    """Rank 0's value on every rank (save() and run() return the path on rank 0 only)."""
    return COMM.bcast(value, root=0)


def shared_dir(factory, name):
    """A new temporary directory, the same on every rank (made on rank 0)."""
    return Path(on_root(str(factory.mktemp(name)) if COMM.rank == 0 else None))


def total(form, comm=COMM):
    """A scalar form (UFL or compiled) summed over the ranks."""
    if not isinstance(form, dolfinx.fem.Form):
        form = dolfinx.fem.form(form)
    return comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM)


def write_text(path, text):
    """Write a file on rank 0 (its folders too); return once every rank can read it."""
    path = Path(path)
    if COMM.rank == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    COMM.barrier()
    return path


def copy_file(src, dst):
    """Copy a file on rank 0 (making dst's folders); return once every rank can read it."""
    dst = Path(dst)
    if COMM.rank == 0:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
    COMM.barrier()
    return dst
