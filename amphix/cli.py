"""Run one problem from the command line, or from a script with `amphix.run`; plot a saved run.

    amphix run case.yaml
    amphix run case.yaml --set discretization.deg=1 --set time.dt=0.005
    amphix run case.yaml --set mesh.h=0.00025 --name annulus_fine
    mpirun -n 2 amphix run case.yaml

`--set` takes dotted keys of the YAML tree; values are read as YAML (numbers, lists, strings).
`--name` sets output.name, the stem of the result files. The run records what the `output` section
asks for (io.Recorder) and saves <root>/<output.dir>/<name>.npz (and <name>_fields.npz).

    amphix plot annulus p cflux --at 0.5 --log --zoom 5.2e-3 6.2e-3 -5e-4 5e-4
    amphix plot annulus cflux --average 0 1 --quiver 20 --out figures/mean_q.png
    amphix plot annulus --series probe_p cbflux_pial
    amphix plot annulus cflux --log --gif --from 1 --to 2 --fps 10

`plot` reads results/<run>_fields.npz (fields) or results/<run>.npz (--series), or the .npz file
given, and writes a PNG (--gif: an animation over the snapshots) to figures/ (plot.command); it
needs matplotlib and one process.
"""

import argparse
import sys
import time
import warnings
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


def _plot_parser(commands: Any) -> None:
    plot = commands.add_parser("plot", help="plot fields or time series of a saved run (PNG)")
    plot.add_argument("run", help="a run name (results/<run>_fields.npz, <run>.npz) or an .npz")
    plot.add_argument("fields", nargs="*", help="the fields to draw, one panel each")
    plot.add_argument("--series", nargs="+", metavar="KEY", help="time series of the results")
    when = plot.add_mutually_exclusive_group()
    when.add_argument("--at", type=float, metavar="T", help="the snapshot nearest T [s]")
    when.add_argument("--average", type=float, nargs=2, metavar=("T0", "T1"), help="mean")
    plot.add_argument("--component", help="mag (default), x, y or z of a vector")
    plot.add_argument("--log", action="store_true", help="logarithmic colour scale")
    plot.add_argument("--vmin", type=float, help="colour scale minimum")
    plot.add_argument("--vmax", type=float, help="colour scale maximum")
    plot.add_argument(
        "--zoom", type=float, nargs=4, metavar=("X0", "X1", "Y0", "Y1"), help="region [m]"
    )
    plot.add_argument("--cells", action="store_true", help="one colour per cell")
    plot.add_argument("--mesh", action="store_true", help="draw the cell edges")
    plot.add_argument("--quiver", type=int, metavar="N", help="about N direction arrows")
    plot.add_argument("--out", help="the output file (default: figures/...)")
    plot.add_argument("--show", action="store_true", help="also open a window")
    plot.add_argument("--gif", action="store_true", help="animate one field over its snapshots")
    plot.add_argument("--from", type=float, dest="t_from", metavar="T0", help="first frame [s]")
    plot.add_argument("--to", type=float, dest="t_to", metavar="T1", help="last frame [s]")
    plot.add_argument("--fps", type=int, help="frames per second (default 10)")


FIELD_ONLY = ("at", "average", "log", "vmin", "vmax", "zoom", "cells", "mesh", "quiver", "gif")
GIF_ONLY = ("t_from", "t_to", "fps")


def _plot(args: argparse.Namespace) -> int:
    """Run `amphix plot`; return the exit status (2 for a plot that cannot be made)."""
    if MPI.COMM_WORLD.size > 1:
        print("amphix plot: runs on one process, not under mpirun", file=sys.stderr)
        return 2
    try:
        from amphix.plot import command  # noqa: PLC0415 (matplotlib is optional)
    except ImportError as e:
        print(f"amphix plot: {e}", file=sys.stderr)
        return 2
    command.backend(args.show)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            path = _draw(args, command)
    except (command.PlotError, NotImplementedError) as e:
        print(f"amphix plot: {e}", file=sys.stderr)
        return 2
    finally:
        for w in caught:
            print(f"amphix plot: warning: {w.message}", file=sys.stderr)
    print(f"saved {path}")
    return 0


def _flag(key: str) -> str:
    return f"--{key.removeprefix('t_')}"  # t_from: --from


def _draw(args: argparse.Namespace, command: Any) -> Path:
    if args.series:
        given = [_flag(k) for k in FIELD_ONLY + GIF_ONLY if getattr(args, k) not in (None, False)]
        if args.fields or given:
            raise command.PlotError(f"--series: no fields or {', '.join(given or ['fields'])}")
        return command.plot_series(
            args.run, args.series, component=args.component, out=args.out, show=args.show
        )
    if not args.fields:
        raise command.PlotError("give the fields to draw, or --series KEY ...")
    options = {
        "component": args.component,
        "log": args.log,
        "vmin": args.vmin,
        "vmax": args.vmax,
        "zoom": tuple(args.zoom) if args.zoom else None,
        "cells": args.cells,
        "mesh": args.mesh,
        "quiver": args.quiver,
    }
    if args.gif:
        if len(args.fields) > 1 or args.at is not None or args.average or args.show:
            raise command.PlotError("--gif: one field, without --at, --average or --show")
        return command.plot_gif(
            args.run,
            args.fields[0],
            t_from=args.t_from,
            t_to=args.t_to,
            fps=10 if args.fps is None else args.fps,
            out=args.out,
            **options,
        )
    given = [_flag(k) for k in GIF_ONLY if getattr(args, k) is not None]
    if given:
        raise command.PlotError(f"{', '.join(given)}: only with --gif")
    return command.plot_fields(
        args.run,
        args.fields,
        at=args.at,
        average=tuple(args.average) if args.average else None,
        out=args.out,
        show=args.show,
        **options,
    )


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
    _plot_parser(commands)
    args = parser.parse_args(argv)
    if args.command == "plot":
        return _plot(args)
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
