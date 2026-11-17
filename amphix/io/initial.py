"""The `initial` section of a problem, applied to its model before the first step.

    model = make_model(problem)
    apply_initial(model)                        # amphix.run does it
    model.run([rec])

From a fields file: the model's fields at the saved time, put back exactly on the run's mesh
(io.fields.restore: the mesh must be the saved one), the other tracked state (d_t, d_tt) zero. From
data: Model.set_initial evaluates it at t0. Either way the state is scaled as set_initial scales it.
The boundary data at t0 should agree with the initial state: nothing checks it (a steady state saved
by a run, restarted with the same loads, stays steady).
"""

from typing import Any

from amphix.io.fields import restore


def apply_initial(model: Any) -> None:
    """Set the model's state at t0 from its problem's `initial` section (nothing if it has none)."""
    initial = model.problem.initial
    if initial is None:
        return
    if initial.path is None:
        model.set_initial(**initial.values)
        return
    model.set_initial(**restore(initial.path, initial.index, list(model.fields), model.domain))
