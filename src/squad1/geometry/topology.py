"""2-D SIMP topology optimisation (compliance minimisation, density filter, optimality-criteria update).

Follows the structure of the classic 88-line MATLAB code (Andreassen et al. 2011) with geometric axes
(``x`` right, ``y`` up; density array shape ``(nely, nelx)``, row 0 = bottom). Includes the two benchmark cases
(cantilever, MBB half-beam with the *standard* symmetry boundary conditions) and material loading from a labelled table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from squad1.errors import ConditioningError, ContractError, ConvergenceError

CASES = ("cantilever", "mbb")


@dataclass
class TopologyProblem:
    nelx: int = 60
    nely: int = 20
    volfrac: float = 0.4
    penal: float = 3.0
    rmin: float = 1.5
    case: str = "mbb"
    E0: float = 1.0  # Young's modulus of solid (any consistent unit; compliance scales as 1/E0)
    nu: float = 0.3
    Emin_ratio: float = 1e-9
    load: float = 1.0

    def __post_init__(self) -> None:
        if self.nelx < 4 or self.nely < 4:
            raise ContractError("nelx and nely must be >= 4")
        if not 0.05 <= self.volfrac <= 0.9:
            raise ContractError("volfrac must be in [0.05, 0.9]")
        if self.penal < 1 or self.rmin <= 0 or self.E0 <= 0 or self.load == 0:
            raise ContractError("need penal >= 1, rmin > 0, E0 > 0, load != 0")
        if not -1.0 < self.nu < 0.5:
            raise ContractError("Poisson ratio must be in (-1, 0.5)")
        if self.case not in CASES:
            raise ContractError(f"case must be one of {CASES}")
        if not 1 / 8 <= self.nelx / self.nely <= 8:
            raise ContractError("aspect ratio nelx/nely must be within [1/8, 8]")

    @property
    def ndof(self) -> int:
        return 2 * (self.nelx + 1) * (self.nely + 1)

    def node(self, jx: int, jy: int) -> int:
        return jx * (self.nely + 1) + jy

    def boundary_conditions(self) -> tuple[np.ndarray, np.ndarray]:
        """``(fixed_dofs, load_vector)`` for the chosen benchmark."""
        f = np.zeros(self.ndof)
        if self.case == "cantilever":  # clamp x = 0, point load down at the middle of the free edge
            fixed = np.array(
                [d for jy in range(self.nely + 1) for d in (2 * self.node(0, jy), 2 * self.node(0, jy) + 1)]
            )
            f[2 * self.node(self.nelx, self.nely // 2) + 1] = -self.load
        else:  # MBB half-beam: symmetry plane x = 0 (all ux fixed), roller at bottom-right, load down at top-left
            fixed = np.array([2 * self.node(0, jy) for jy in range(self.nely + 1)] + [2 * self.node(self.nelx, 0) + 1])
            f[2 * self.node(0, self.nely) + 1] = -self.load
        if f[fixed].any():
            raise ContractError("a load acts on a fixed DOF")
        return np.unique(fixed), f


def element_stiffness(nu: float) -> np.ndarray:
    """Unit-square bilinear plane-stress element (E = 1), nodes CCW from bottom-left, dofs (ux, uy) per node."""
    D = 1.0 / (1 - nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    g = 1.0 / np.sqrt(3.0)
    K = np.zeros((8, 8))
    for xi in (-g, g):
        for eta in (-g, g):
            dN = 0.25 * np.array(
                [[-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)], [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]]
            )
            dNdx = dN * 2.0  # unit square: d/dx = 2 d/dxi
            B = np.zeros((3, 8))
            B[0, 0::2] = dNdx[0]
            B[1, 1::2] = dNdx[1]
            B[2, 0::2] = dNdx[1]
            B[2, 1::2] = dNdx[0]
            K += B.T @ D @ B * 0.25  # det J = 1/4
    return K


def _edof_matrix(p: TopologyProblem) -> np.ndarray:
    ex, ey = np.meshgrid(np.arange(p.nelx), np.arange(p.nely))  # (nely, nelx)
    n_bl = ex * (p.nely + 1) + ey
    nodes = np.stack([n_bl, n_bl + (p.nely + 1), n_bl + (p.nely + 1) + 1, n_bl + 1], axis=-1)  # BL, BR, TR, TL
    edof = np.stack([2 * nodes, 2 * nodes + 1], axis=-1).reshape(p.nely * p.nelx, 8)
    return edof


def density_filter(p: TopologyProblem) -> sp.csr_matrix:
    rows, cols, vals = [], [], []
    r = int(np.ceil(p.rmin)) - 1
    ex, ey = np.meshgrid(np.arange(p.nelx), np.arange(p.nely))
    idx = (ey * p.nelx + ex).ravel()
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            w = max(0.0, p.rmin - np.hypot(dx, dy))
            if w == 0:
                continue
            jx, jy = ex + dx, ey + dy
            ok = (jx >= 0) & (jx < p.nelx) & (jy >= 0) & (jy < p.nely)
            rows.append(idx.reshape(p.nely, p.nelx)[ok])
            cols.append((jy * p.nelx + jx)[ok])
            vals.append(np.full(ok.sum(), w))
    H = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(p.nelx * p.nely,) * 2
    )
    return sp.diags(1.0 / np.asarray(H.sum(axis=1)).ravel()) @ H


@dataclass
class TopologyResult:
    density: np.ndarray  # physical (filtered) densities (nely, nelx)
    compliance: list[float]
    volume: float
    iterations: int
    converged: bool
    change: float
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def grayness(self) -> float:
        """Share of elements that are neither solid nor void (4 x (1-x) x, averaged)."""
        return float(np.mean(4 * self.density * (1 - self.density)))


def analyse(p: TopologyProblem, x_phys: np.ndarray) -> tuple[float, np.ndarray]:
    """Finite-element solve; returns ``(compliance, element strain-energy ce)`` for a physical density field."""
    KE = element_stiffness(p.nu)
    edof = _edof_matrix(p)
    fixed, f = p.boundary_conditions()
    free = np.setdiff1d(np.arange(p.ndof), fixed)
    xp = x_phys.reshape(-1)
    Emin = p.E0 * p.Emin_ratio
    E = Emin + xp**p.penal * (p.E0 - Emin)
    iK = np.repeat(edof, 8, axis=1).ravel()
    jK = np.tile(edof, (1, 8)).ravel()
    sK = (KE.ravel()[None, :] * E[:, None]).ravel()
    K = sp.csc_matrix((sK, (iK, jK)), shape=(p.ndof, p.ndof))
    U = np.zeros(p.ndof)
    U[free] = spsolve(K[free][:, free], f[free])
    if not np.isfinite(U).all():
        raise ConvergenceError("FE solve produced non-finite displacements")
    Ue = U[edof]
    ce = np.einsum("ij,jk,ik->i", Ue, KE, Ue)
    return float(f @ U), ce


def optimize(
    p: TopologyProblem, iters: int = 100, tol: float = 1e-2, move: float = 0.2, x0: np.ndarray | None = None
) -> TopologyResult:
    n = p.nelx * p.nely
    H = density_filter(p)
    x = np.full(n, p.volfrac) if x0 is None else np.asarray(x0, dtype=float).reshape(n)
    hist: list[float] = []
    change = 1.0
    it = 0
    xphys = H @ x
    for it in range(1, iters + 1):  # noqa: B007  (``it`` is the reported iteration count after the loop)
        xphys = H @ x
        c, ce = analyse(p, xphys.reshape(p.nely, p.nelx))
        hist.append(c)
        Emin = p.E0 * p.Emin_ratio
        dc = -p.penal * xphys ** (p.penal - 1) * (p.E0 - Emin) * ce
        dc = H.T @ dc
        dv = H.T @ np.ones(n)
        l1, l2 = 0.0, 1e9
        while (l2 - l1) / (l1 + l2) > 1e-4:
            mid = 0.5 * (l1 + l2)
            xn = np.clip(
                x * np.sqrt(np.maximum(-dc, 0) / (dv * mid + 1e-300)),
                np.maximum(0.0, x - move),
                np.minimum(1.0, x + move),
            )
            if (H @ xn).mean() > p.volfrac:
                l1 = mid
            else:
                l2 = mid
        change = float(np.abs(xn - x).max())
        x = xn
        if change < tol:
            break
    xphys = H @ x
    c, _ = analyse(p, xphys.reshape(p.nely, p.nelx))
    hist.append(c)
    return TopologyResult(
        xphys.reshape(p.nely, p.nelx),
        hist,
        float(xphys.mean()),
        it,
        change < tol,
        change,
        {"x_design": x.copy()},
    )


# ------------------------------------------------------------------------------------------ materials
def load_material(
    source: str | Path | Any, formula: str | None = None, target_E_gpa: float | None = None
) -> dict[str, Any]:
    """Pick a material from a labelled elasticity table (columns: formula_pretty, E, nu, density_g_cm3, material_id).

    ``source`` is a ``.parquet``/``.csv`` path or a DataFrame. Selection: exact ``formula``; else nearest
    ``target_E_gpa``; else the stiffest. Validates physical ranges instead of trusting the table.
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError("load_material needs pandas: pip install 'squad1[chem]'") from exc
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"materials table not found: {path}")
        df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    else:
        df = source
    need = ["formula_pretty", "E", "nu", "density_g_cm3"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ConditioningError(f"materials table missing columns: {missing}")
    df = df.dropna(subset=need)
    df = df[(df["E"] > 0) & (df["nu"] > -1) & (df["nu"] < 0.5) & (df["density_g_cm3"] > 0)]
    if df.empty:
        raise ConditioningError("no physically valid rows in the materials table")
    if formula is not None:
        m = df[df["formula_pretty"] == formula]
        if m.empty:
            raise ConditioningError(f"formula {formula!r} not found")
        row = m.iloc[0]
    elif target_E_gpa is not None:
        row = df.loc[(df["E"] - target_E_gpa).abs().idxmin()]
    else:
        row = df.loc[df["E"].idxmax()]
    return {
        "material_id": str(row.get("material_id", "")),
        "formula": str(row["formula_pretty"]),
        "E_gpa": float(row["E"]),
        "nu": float(row["nu"]),
        "density_g_cm3": float(row["density_g_cm3"]),
    }
