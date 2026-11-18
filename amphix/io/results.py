"""Result files: arrays in an .npz, with the problem's YAML tree inside, so a run can be reproduced.

    path = save_results("results/run.npz", {"t": t, "probe_p": values}, problem)   # rank 0 writes
    res = load_results(path)      # {name: array, ..., "problem": the YAML tree as a dict}

Rank 0 writes; every rank gets the path back once the file is written, so any rank can read it
right after (under mpirun too).

Everything stored is in SI. The tree is the problem as loaded (bases merged, overrides applied:
Problem.raw); data given in code is not in it. Only plain arrays are stored, so a file reads back
with numpy alone (no pickle).
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from mpi4py import MPI


def problem_text(problem: Any) -> str:
    """Return the problem's YAML tree as text (what result files store)."""
    return yaml.safe_dump(problem.raw, sort_keys=False)


def save_results(path: str | Path, arrays: dict[str, Any], problem: Any) -> Path:
    """Write arrays and the problem tree to an .npz on rank 0; return the path on every rank."""
    if "problem" in arrays:  # on every rank, before the rank-0 write
        raise ValueError("save_results: 'problem' is reserved for the problem tree")

    def write(target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {k: np.asarray(v) for k, v in arrays.items()}
        np.savez(target, **data, problem=problem_text(problem))

    return written(path, MPI.COMM_WORLD, write)


def written(path: str | Path, comm: MPI.Comm, write: Callable[[Path], None]) -> Path:
    """Run write(path) on rank 0; return rank 0's path on every rank once it is written.

    The broadcast makes the other ranks wait for the file. If the write fails, rank 0 re-raises
    its error and the other ranks raise a RuntimeError with rank 0's message, instead of waiting.
    """
    path = Path(path)
    if comm.rank == 0:
        try:
            write(path)
        except BaseException as e:
            comm.bcast(("failed", f"{type(e).__name__}: {e}"), root=0)
            raise
        return Path(comm.bcast(("written", str(path)), root=0)[1])
    status, value = comm.bcast(None, root=0)
    if status == "failed":
        raise RuntimeError(f"saving {path} failed on rank 0: {value}")
    return Path(value)


def load_results(path: str | Path) -> dict[str, Any]:
    """Return the arrays of a result file, with 'problem' parsed back into a dict."""
    with np.load(path, allow_pickle=False) as f:
        out: dict[str, Any] = {k: f[k] for k in f.files}
    if "problem" in out:
        out["problem"] = yaml.safe_load(str(out["problem"]))
    return out
