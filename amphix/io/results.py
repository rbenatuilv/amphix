"""Result files: arrays in an .npz, with the problem's YAML tree inside, so a run can be reproduced.

    path = save_results("results/run.npz", {"t": t, "probe_p": values}, problem)   # rank 0 writes
    res = load_results(path)      # {name: array, ..., "problem": the YAML tree as a dict}

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


def save_results(path: str | Path, arrays: dict[str, Any], problem: Any) -> Path | None:
    """Write arrays and the problem tree to an .npz on rank 0; return the path (None elsewhere)."""
    if MPI.COMM_WORLD.rank != 0:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {k: np.asarray(v) for k, v in arrays.items()}
    if "problem" in data:
        raise ValueError("save_results: 'problem' is reserved for the problem tree")
    np.savez(path, **data, problem=problem_text(problem))
    return path


def load_results(path: str | Path) -> dict[str, Any]:
    """Return the arrays of a result file, with 'problem' parsed back into a dict."""
    with np.load(path, allow_pickle=False) as f:
        out: dict[str, Any] = {k: f[k] for k in f.files}
    if "problem" in out:
        out["problem"] = yaml.safe_load(str(out["problem"]))
    return out
