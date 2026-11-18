"""amphix: brAin MultiPHysIcs on fenicsX.

Poroelastic (Biot) models of brain tissue on DG finite elements, with a conservative velocity
reconstruction.

    import amphix
    problem = amphix.load("case.yaml", {"time.dt": 0.005})
    path = amphix.run(problem)                 # build, run, record and save

    model = amphix.make_model(problem)         # or step by step
    amphix.apply_initial(model)                # the `initial` section, if any
    rec = amphix.Recorder(model)
    model.run([rec])
    rec.save()

Everything is in SI. Importing the package registers the models (poro2, poro3) and the brain
presets and waveforms. The layers, from the bottom up:

    core/       FEM infrastructure, no physics
    config/     describing a problem
    post/       measuring solutions
    io/         recording and storing
    physics/    the models
    brain/      brain presets: parameters, waveforms, geometries
"""

from amphix import brain
from amphix.cli import run
from amphix.config import ConfigError, Problem, load
from amphix.core.mesh import build as build_domain
from amphix.core.model import Model, Observer, State, System, make_model
from amphix.core.registry import MODELS
from amphix.io.fields import load_fields
from amphix.io.initial import apply_initial
from amphix.io.recorder import Recorder
from amphix.io.results import load_results
from amphix.physics.poro import ThreeField, TwoField
from amphix.post.hdiv import Reconstruction

__version__ = "0.0.1"

__all__ = [
    "MODELS",
    "ConfigError",
    "Model",
    "Observer",
    "Problem",
    "Reconstruction",
    "Recorder",
    "State",
    "System",
    "ThreeField",
    "TwoField",
    "apply_initial",
    "brain",
    "build_domain",
    "load",
    "load_fields",
    "load_results",
    "make_model",
    "run",
]
