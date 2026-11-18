"""Result files: arrays in an .npz, with the problem's YAML tree inside, so a run can be reproduced.

    path = save_results("results/run.npz", {"t": t, "probe_p": values}, problem)   # rank 0 writes
    res = load_results(path)      # {name: array, ..., "problem": the YAML tree as a dict}

Rank 0 writes; every rank gets the path back once the file is written, so any rank can read it
right after (under mpirun too).

Everything stored is in SI. The tree is the problem as loaded (bases merged, overrides applied:
Problem.raw); data given in code is not in it. Only plain arrays are stored, so a file reads back
with numpy alone (no pickle).
"""

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
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: np.asarray(v) for k, v in arrays.items()}
        np.savez(path, **data, problem=problem_text(problem))
    return written(path, comm)


def written(path: str | Path, comm: MPI.Comm) -> Path:
    """Return rank 0's path on every rank, once rank 0 has written it (the broadcast waits)."""
    return Path(comm.bcast(str(path), root=0))


def load_results(path: str | Path) -> dict[str, Any]:
    """Return the arrays of a result file, with 'problem' parsed back into a dict."""
    with np.load(path, allow_pickle=False) as f:
        out: dict[str, Any] = {k: f[k] for k in f.files}
    if "problem" in out:
        out["problem"] = yaml.safe_load(str(out["problem"]))
    return out
