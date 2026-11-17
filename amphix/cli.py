"""Run one problem from the command line, or from a script with `amphix.run`.

    amphix run case.yaml
    amphix run case.yaml --set discretization.deg=1 --set time.dt=0.005
    amphix run case.yaml --set mesh.h=0.00025 --name annulus_fine
    mpirun -n 2 amphix run case.yaml

`--set` takes dotted keys of the YAML tree; values are read as YAML (numbers, lists, strings).
`--name` sets output.name, the stem of the result files. The run records what the `output` section
asks for (io.Recorder) and saves <root>/<output.dir>/<name>.npz (and <name>_fields.npz).
"""

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mpi4py import MPI

from amphix.config import ConfigError, load, parse_set
from amphix.core.model import make_model
from amphix.io.initial import apply_initial
from amphix.io.recorder import Recorder


def run(problem: Any, verbose: bool = True) -> Path | None:
    """Build the problem's model, set its initial state, run it to T with a Recorder, save.

    Return the results path on rank 0 (None on the others).
    """
    talk = verbose and MPI.COMM_WORLD.rank == 0
    start = time.perf_counter()
    model = make_model(problem)
    apply_initial(model)
    if talk:
        mesh = model.domain.mesh
        cells = mesh.topology.index_map(mesh.topology.dim).size_global
        print(
            f"{problem.name}: {problem.model}, {cells} cells, {model.system.ndofs} dofs, "
            f"{problem.time.nsteps} steps (setup {time.perf_counter() - start:.1f} s)",
            flush=True,
        )
    recorder = Recorder(model)
    if talk and recorder.fields is not None:
        names, n = list(problem.output.fields.quantities), len(problem.output.fields.steps)
        print(f"  fields {names} at {n} times: {recorder.fields_bytes() / 1e6:.3g} MB in memory")
    every = max(1, problem.time.nsteps // 10)

    def progress(m: Any) -> None:
        if talk and m.n > 0 and m.n % every == 0:
            print(f"  t = {m.t:.4g} s  ({time.perf_counter() - start:.1f} s)", flush=True)

    model.run([recorder, progress])
    path = recorder.save()
    if talk:
        print(f"saved {path}", flush=True)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line; return the exit status (2 for a problem that cannot be used)."""
    parser = argparse.ArgumentParser(
        prog="amphix", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("run", help="run one problem file and save its results")
    one.add_argument("yaml", help="the problem file")
    one.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="items")
    one.add_argument("--name", help="output file stem (default: output.name or the YAML stem)")
    one.add_argument("--quiet", action="store_true", help="print nothing")
    args = parser.parse_args(argv)
    try:
        overrides = parse_set(args.items)
        if args.name:
            overrides["output.name"] = args.name
        problem = load(args.yaml, overrides)
    except ConfigError as e:
        print(f"amphix: {args.yaml}: {e}", file=sys.stderr)
        return 2
    run(problem, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
