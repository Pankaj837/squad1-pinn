"""Strut lattices: graph-canonical (unique nodes/struts), exact voxel-union density, fused watertight solid, checks.

The graph (nodes + struts) is the single source of truth — it is what a truss/FEA solver needs (shared joints) and
what the solid is generated from. Relative density is measured on the *union* of struts (voxel raster), so
overlapping joints and shared cell edges are never counted twice (the earlier concatenated-cylinder mesh
double-counted duplicated struts and node overlaps, overstating density by ~50%).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from squad1.errors import ContractError
from squad1.geometry.voxel_mesh import Mesh, MeshStats, mesh_stats, voxels_to_mesh

_CORNERS = np.array(list(itertools.product((0.0, 1.0), repeat=3)))
_FACE_CENTRES = np.array(
    [[0.5, 0.5, 0.0], [0.5, 0.5, 1.0], [0.5, 0.0, 0.5], [0.5, 1.0, 0.5], [0.0, 0.5, 0.5], [1.0, 0.5, 0.5]]
)
_CENTRE = np.array([[0.5, 0.5, 0.5]])


def _pairs_at_distance(nodes: np.ndarray, d: float, tol: float = 1e-9) -> list[tuple[int, int]]:
    out = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if abs(np.linalg.norm(nodes[i] - nodes[j]) - d) < tol:
                out.append((i, j))
    return out


def _unit_cell(kind: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
    if kind == "simple_cubic":
        nodes = _CORNERS
        return nodes, _pairs_at_distance(nodes, 1.0)
    if kind == "bcc":  # corner-to-centre struts only
        nodes = np.vstack([_CORNERS, _CENTRE])
        return nodes, _pairs_at_distance(nodes, np.sqrt(3) / 2)
    if kind == "bcc_cube":  # bcc + cube edges (the variant in the Topology report)
        nodes = np.vstack([_CORNERS, _CENTRE])
        return nodes, _pairs_at_distance(nodes, np.sqrt(3) / 2) + _pairs_at_distance(nodes, 1.0)
    if kind == "octet":  # FCC nearest neighbours: corner<->face-centre and face-centre<->face-centre
        nodes = np.vstack([_CORNERS, _FACE_CENTRES])
        return nodes, _pairs_at_distance(nodes, np.sqrt(2) / 2)
    raise ContractError(f"unknown unit cell {kind!r}; available: {sorted(UNIT_CELLS)}")


UNIT_CELLS = ("simple_cubic", "bcc", "bcc_cube", "octet")


@dataclass
class LatticeGraph:
    nodes: np.ndarray  # (N, 3) mm
    struts: np.ndarray  # (M, 2) node indices, unique, sorted
    size: tuple[float, float, float]  # bounding box of the cell tiling (mm)

    @property
    def total_length(self) -> float:
        return float(np.linalg.norm(self.nodes[self.struts[:, 0]] - self.nodes[self.struts[:, 1]], axis=1).sum())

    @property
    def volume(self) -> float:
        return float(np.prod(self.size))

    def degrees(self) -> np.ndarray:
        return np.bincount(self.struts.ravel(), minlength=len(self.nodes))


def build_lattice(kind: str, nx: int, ny: int, nz: int, cell_size: float) -> LatticeGraph:
    if min(nx, ny, nz) < 1 or cell_size <= 0:
        raise ContractError("nx, ny, nz must be >= 1 and cell_size > 0")
    base, pairs = _unit_cell(kind)
    node_id: dict[tuple[int, int, int], int] = {}
    nodes: list[np.ndarray] = []
    struts: set[tuple[int, int]] = set()

    def nid(p: np.ndarray) -> int:
        key = tuple(np.round(p * 1e6).astype(np.int64))  # integer key: exact merging of coincident nodes
        if key not in node_id:
            node_id[key] = len(nodes)
            nodes.append(p)
        return node_id[key]

    for ix, iy, iz in itertools.product(range(nx), range(ny), range(nz)):
        ids = [nid((b + np.array([ix, iy, iz])) * cell_size) for b in base]
        for a, b in pairs:
            struts.add((min(ids[a], ids[b]), max(ids[a], ids[b])))
    return LatticeGraph(np.array(nodes), np.array(sorted(struts)), (nx * cell_size, ny * cell_size, nz * cell_size))


def rasterize(graph: LatticeGraph, radius: float, voxel: float) -> np.ndarray:
    """Boolean occupancy of the union of strut capsules (distance to segment <= radius) at voxel size ``voxel``."""
    if radius <= 0 or voxel <= 0:
        raise ContractError("radius and voxel must be positive")
    pad = radius + voxel
    lo = graph.nodes.min(axis=0) - pad
    dims = np.ceil((graph.nodes.max(axis=0) + pad - lo) / voxel).astype(int) + 1
    occ = np.zeros(dims, dtype=bool)
    for a, b in graph.struts:
        p, q = graph.nodes[a], graph.nodes[b]
        bb_lo = np.floor((np.minimum(p, q) - radius - lo) / voxel).astype(int).clip(0)
        bb_hi = np.ceil((np.maximum(p, q) + radius - lo) / voxel).astype(int).clip(max=dims - 1)
        sl = tuple(slice(l_, h_ + 1) for l_, h_ in zip(bb_lo, bb_hi))
        grids = np.meshgrid(
            *[lo[d] + (np.arange(bb_lo[d], bb_hi[d] + 1) + 0.0) * voxel for d in range(3)], indexing="ij"
        )
        pts = np.stack(grids, axis=-1)
        d = q - p
        t = np.clip(((pts - p) @ d) / max(float(d @ d), 1e-30), 0.0, 1.0)
        dist = np.linalg.norm(pts - (p + t[..., None] * d), axis=-1)
        occ[sl] |= dist <= radius
    return occ


@dataclass
class LatticeReport:
    relative_density: float
    n_struts: int
    n_nodes: int
    components: int
    watertight: bool
    volume_positive: bool
    min_feature_ok: bool
    density_in_range: bool

    @property
    def all_passed(self) -> bool:
        return all(
            (
                self.components == 1,
                self.watertight,
                self.volume_positive,
                self.min_feature_ok,
                self.density_in_range,
            )
        )


def build_solid(
    graph: LatticeGraph,
    radius: float,
    voxel: float,
    min_feature: float = 0.3,
    density_range: tuple[float, float] = (0.02, 0.60),
) -> tuple[Mesh, LatticeReport, MeshStats]:
    """Fused solid mesh + manufacturability report. ``min_feature`` is the smallest printable strut *diameter* (mm)."""
    occ = rasterize(graph, radius, voxel)
    lo = graph.nodes.min(axis=0) - (radius + voxel)
    mesh = voxels_to_mesh(occ, voxel, tuple(lo - 0.5 * voxel))
    st = mesh_stats(mesh)
    rel = float(occ.sum() * voxel**3 / graph.volume)
    n_cc = int(ndimage.label(occ, structure=np.ones((3, 3, 3)))[1])
    rep = LatticeReport(
        rel,
        len(graph.struts),
        len(graph.nodes),
        n_cc,
        st.watertight,
        st.volume > 0,
        2 * radius >= min_feature and voxel <= radius / 1.5,
        density_range[0] <= rel <= density_range[1],
    )
    return mesh, rep, st
