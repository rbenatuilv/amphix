"""ParaView output: one VTX (.bp) file per quantity, in SI.

    pv = ParaView(model.domain, {"p": lambda: model.quantity("p"), "d": ...}, "results/run")
    pv.write(model.t)          # every few steps
    pv.close()

The values are copied into DG Functions first (cflux into DG of its polynomial degree, exactly;
derived quantities are evaluated), since both depend on the geometry. The mesh coordinates are then
scaled by L0 only while writing, and restored bit for bit.
"""

from collections.abc import Callable
from pathlib import Path

import dolfinx
import dolfinx.io

from amphix.core.mesh import Domain
from amphix.io.fields import copy_to, dg_space

Source = Callable[[], dolfinx.fem.Function]


class ParaView:
    """VTX writers of some SI quantities, in a folder (one <name>.bp each)."""

    def __init__(self, domain: Domain, sources: dict[str, Source], folder: str | Path) -> None:
        """Open one writer per quantity; sources give the current SI Function of each."""
        self.domain, self.folder = domain, Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.writers = {}
        for name, source in sources.items():
            g = dolfinx.fem.Function(dg_space(source().function_space), name=name)
            path = self.folder / f"{name}.bp"
            self.writers[name] = (source, g, dolfinx.io.VTXWriter(domain.mesh.comm, path, [g]))

    def write(self, t: float) -> None:
        """Write every quantity at time t [s], with the mesh in m for the duration of the write."""
        for source, g, _ in self.writers.values():  # first: derived quantities and the RT -> DG
            copy_to(g, source())  # copy depend on the geometry, so not while it is scaled
        x = self.domain.mesh.geometry.x
        x0 = x.copy()
        x *= self.domain.L0
        try:
            for *_, writer in self.writers.values():
                writer.write(t)
        finally:
            x[:] = x0  # exactly the solver's coordinates again

    def close(self) -> None:
        """Close the files."""
        for *_, writer in self.writers.values():
            writer.close()
        self.writers = {}
