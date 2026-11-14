"""Condition numbers of the 2F and 3F systems, BRAIN parameters (cvr22), stable pairs.

Each time step solves one linear system A u = b, with A fixed while dt is. We build each model on
a sequence of meshes, take its step matrix A, and compute the 2-norm condition number
cond(A) = sigma_max / sigma_min with SLEPc, on the sparse matrix:
    sigma_max(A)  the largest singular value of A
    sigma_min(A)  1 / (the largest singular value of A^-1), A^-1 applied by an LU solve
Only the largest singular value is ever asked for, which SLEPc finds quickly and reliably; the
smallest one directly would need a shift-and-invert setup.

Two versions of each matrix:
    raw      A as amphix solves it (unknowns in solver units: scales L0, T0, P0)
    scaled   D^-1/2 A D^-1/2, D = |diag(A)| (Jacobi scaling): removes the differences of size
             between the blocks, which a solver also removes with any diagonal preconditioner.
The scaled numbers are the fair comparison of the two formulations. Expected: h^-2 growth.

Output: conditioning.png, and the dofs, condition numbers and 3F/2F ratios, printed.
"""

import matplotlib.pyplot as plt
import numpy as np
from petsc4py import PETSc
from slepc4py import SLEPc

import amphix  # noqa: F401 (registers poro2 and poro3)
from amphix.config import from_dict
from amphix.core.linalg import make_ksp
from amphix.core.mesh import cell_size
from amphix.core.model import make_model
from amphix.post.norms import orders

L = 1e-2  # domain size [m]
MESHES = (4, 8, 16, 32)  # cells per side
DT = 0.01  # [s]
PAIRS = {"poro2": (2, 1), "poro3": (2, 1)}  # (deg, deg_p): 2F P2/P1, 3F P2/P2/P1


# ---- the problem: 1 cm square, cvr22 brain tissue, clamped and drained on every side -----------
def problem(model, n):
    deg, deg_p = PAIRS[model]
    sides = {s: {"displacement": [0, 0], "pressure": 0} for s in ("left", "right", "bottom", "top")}
    return from_dict(
        {
            "model": model,
            "scales": {"L0": L, "T0": 1.0, "P0": 1500.0},
            "material": {"preset": "cvr22"},
            "mesh": {"shape": "rectangle", "corners": [[0, 0], [L, L]], "n": [n, n]},
            "boundaries": sides,
            "discretization": {"deg": deg, "deg_p": deg_p},
            "time": {"T": DT, "dt": DT} | ({"theta": 0.5} if model == "poro2" else {}),
        },
        "cond",
    )


# ---- one system: its size and condition numbers (raw, scaled) -----------------------------------
def sigma_max(M):
    """The largest singular value of M (a PETSc matrix, possibly a shell)."""
    svd = SLEPc.SVD().create()
    svd.setOperators(M)
    svd.setWhichSingularTriplets(SLEPc.SVD.Which.LARGEST)
    svd.setDimensions(1)
    svd.setTolerances(1e-10, 1000)
    svd.solve()
    assert svd.getConverged() >= 1, "SLEPc did not converge"
    return svd.getSingularTriplet(0)


class Inverse:
    """A^-1 as a matrix-free operator: y = A^-1 x and y = A^-T x by one LU solve each."""

    def __init__(self, A):
        self.ksp = make_ksp(A)  # amphix's direct solver (MUMPS LU)

    def mult(self, mat, x, y):
        self.ksp.solve(x, y)

    def multTranspose(self, mat, x, y):  # noqa: N802 (PETSc API)
        self.ksp.solveTranspose(x, y)


def cond(A):
    Ainv = PETSc.Mat().createPython(A.getSizes(), Inverse(A), comm=A.comm)
    Ainv.setUp()
    return sigma_max(A) * sigma_max(Ainv)  # = sigma_max / sigma_min


def condition(model, n):
    m = make_model(problem(model, n))
    A = m.system.matrix(regular=True).copy()  # PETSc matrix of a regular time step
    d = A.getDiagonal()
    d.abs()
    d.sqrtabs()
    d.reciprocal()  # d = |diag(A)|^-1/2
    S = A.copy()
    S.diagonalScale(d, d)  # S = D^-1/2 A D^-1/2
    return cell_size(m.domain), A.getSize()[0], cond(A), cond(S)


# ---- the study ----------------------------------------------------------------------------------
if __name__ == "__main__":
    import pathlib

    HERE = pathlib.Path(__file__).parent
    (HERE / "figures").mkdir(exist_ok=True)

    results = {}
    for model in PAIRS:
        tag = "2F" if model == "poro2" else "3F"
        h, ndofs, raw, scaled = zip(*(condition(model, n) for n in MESHES), strict=True)
        results[tag] = (h, raw, scaled)
        print(f"\n{tag}")
        for n, d, r, c in zip(MESHES, ndofs, raw, scaled, strict=True):
            print(f"  n = {n:2d}   dofs {d:6d}   cond raw {r:.2e}   scaled {c:.2e}")
        print(
            "  growth order in 1/h (scaled): " + "  ".join(f"{-r:5.2f}" for r in orders(scaled, h))
        )
    for i, version in ((1, "raw"), (2, "scaled")):
        ratios = [c3 / c2 for c2, c3 in zip(results["2F"][i], results["3F"][i], strict=True)]
        print(f"3F / 2F, {version}: " + "  ".join(f"{r:.3g}" for r in ratios))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, i, version in ((axes[0], 1, "raw"), (axes[1], 2, "scaled (Jacobi)")):
        for tag, res in results.items():
            h, c = res[0], res[i]
            ax.loglog(h, c, "-o", label=f"{tag} (growth {-orders(c, h)[-1]:.1f})")
        h = np.array(results["2F"][0])
        ax.loglog(h, results["2F"][i][-1] * (h / h[-1]) ** -2, "k:", label="h$^{-2}$")
        ax.set(xlabel="h [m]", ylabel="cond(A)", title=version)
        ax.legend()
    fig.suptitle("Brain parameters (cvr22): condition number of the step matrix", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{HERE}/figures/conditioning.png", dpi=120)
