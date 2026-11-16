"""Brain geometries as `mesh` sections (SI), with boundaries `ventricle` and `pial`.

    annulus(h=5e-4, wall=None)   2D: a disc of radius 1 cm with an off-centre hole (radius 2.5 mm
                                 at x = 3 mm); wall = {h, band, growth} refines both walls
    shell(h=2e-3)                3D: the same section through a ball

The tissue is thick on the left (10.5 mm) and thin on the right (4.5 mm): MEAN_THICKNESS = 7.5 mm
is the ventricle-pial distance (the default length of the transmantle waveform). The hole is the
ventricle, the outer boundary the pial surface. A section is a plain dict:

    tree["mesh"] = geometries.annulus(h=2.5e-4)
"""

RADIUS = 1.0e-2  # m, the outer circle or sphere, centred at the origin
HOLE_RADIUS = 2.5e-3  # m
HOLE_X = 3.0e-3  # m, the centre of the hole on the x axis
MEAN_THICKNESS = 7.5e-3  # m, the mean of 4.5 and 10.5 mm
NAMES = {"inner": "ventricle", "outer": "pial"}


def annulus(h: float = 5.0e-4, wall: dict | None = None) -> dict:
    """Return the 2D annulus mesh section (cell size h in m; wall = {h, band, growth} in m)."""
    section = {
        "shape": "annulus",
        "radius": RADIUS,
        "inner": {"center": [HOLE_X, 0.0], "radius": HOLE_RADIUS},
        "h": h,
        "names": dict(NAMES),
    }
    if wall is not None:
        section["wall"] = dict(wall)
    return section


def shell(h: float = 2.0e-3) -> dict:
    """Return the 3D shell mesh section (cell size h in m; 2 mm is coarse: about 2 cells across)."""
    return {
        "shape": "shell",
        "radius": RADIUS,
        "inner": {"center": [HOLE_X, 0.0, 0.0], "radius": HOLE_RADIUS},
        "h": h,
        "names": dict(NAMES),
    }
