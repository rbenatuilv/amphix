"""The brain domain layer.

Named parameter sets with their sources, waveforms and geometry presets. Uses only `physics`,
`config` and `core`.

    parameters   PRESETS: cvr22                 YAML `material: {preset: cvr22}`
    waveforms    cardiac_pulse, cardiac_oscillation, transmantle, vascular (options 1 and 2)
                                                YAML `{waveform: vascular_option2, G: 3.53e-3}`
    geometries   annulus(h, wall), shell(h): mesh sections with boundaries ventricle and pial

Importing this package registers the parameter sets and the waveforms, so that a YAML file can name
them.
"""

from amphix.brain import geometries, parameters, waveforms
from amphix.brain.parameters import PRESETS, ParameterSet

__all__ = ["PRESETS", "ParameterSet", "geometries", "parameters", "waveforms"]
