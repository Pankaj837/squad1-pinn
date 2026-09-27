"""Voxel occupancy -> closed, outward-oriented triangle mesh (+ mesh checks and binary STL writer).

Pure numpy (no mesh library needed). The surface is built from exposed voxel faces, so it is watertight by
construction once *edge-only* contacts (two voxels touching along an edge with the other two cells around that edge
empty — an edge shared by four faces) are removed by :func:`remove_edge_contacts`.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from squad1.errors import ContractError


@dataclass
class Mesh:
    vertices: np.ndarray  # (V, 3) float64
    faces: np.ndarray  # (F, 3) int64, counter-clockwise seen from outside


@dataclass
class MeshStats:
    n_vertices: int
    n_faces: int
    watertight: bool  # every edge shared by exactly two faces
    volume: float  # signed; > 0 means outward-facing normals
    components: int


def remove_edge_contacts(occ: np.ndarray, max_iter: int = 20) -> np.ndarray:
    """Fill one cell of every diagonal-only 2x2 pattern (in all three axis planes) so no mesh edge is non-manifold."""
    occ = occ.copy()
    for _ in range(max_iter):
        changed = False
        for a in range(3):
            b, c = (a + 1) % 3, (a + 2) % 3
            o = np.moveaxis(occ, (b, c), (0, 1))
            p00, p10, p01, p11 = o[:-1, :-1], o[1:, :-1], o[:-1, 1:], o[1:, 1:]
            diag1 = p00 & p11 & ~p10 & ~p01
            diag2 = p10 & p01 & ~p00 & ~p11
            if diag1.any():
                p10[diag1] = True  # views into `o` (and thus `occ`)
                changed = True
            if diag2.any():
                p00[diag2] = True
                changed = True
        if not changed:
            return occ
    raise ContractError("could not remove edge contacts")  # pragma: no cover


def voxels_to_mesh(occ: np.ndarray, spacing: float = 1.0, origin: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Mesh:
    occ = np.asarray(occ, dtype=bool)
    if occ.ndim != 3:
        raise ContractError(f"occupancy must be 3-D, got shape {occ.shape}")
    if not occ.any():
        raise ContractError("occupancy is empty")
    occ = remove_edge_contacts(occ)
    pad = np.pad(occ, 1)
    quads: list[np.ndarray] = []
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        for sign in (1, -1):
            sl = tuple(
                slice(2, None) if (a == axis and sign == 1) else slice(0, -2) if a == axis else slice(1, -1)
                for a in range(3)
            )
            idx = np.argwhere(occ & ~pad[sl])
            if idx.size == 0:
                continue
            base = idx.copy()
            base[:, axis] += 1 if sign == 1 else 0
            corners = []
            for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                c = base.copy()
                c[:, u] += du
                c[:, v] += dv
                corners.append(c)
            q = np.stack(corners, axis=1)  # (n, 4, 3)
            quads.append(q if sign == 1 else q[:, ::-1])
    q = np.concatenate(quads, axis=0).reshape(-1, 3)
    uniq, inv = np.unique(q, axis=0, return_inverse=True)
    inv = inv.reshape(-1, 4)
    faces = np.concatenate([inv[:, [0, 1, 2]], inv[:, [0, 2, 3]]], axis=0)
    verts = uniq.astype(float) * spacing + np.asarray(origin)
    return Mesh(verts, faces.astype(np.int64))


def mesh_stats(mesh: Mesh) -> MeshStats:
    f = mesh.faces
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    v = mesh.vertices
    vol = float(np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum() / 6.0)
    nv = len(v)
    a = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    b = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    g = coo_matrix((np.ones(len(a)), (a, b)), shape=(nv, nv))
    n_comp = int(connected_components(g, directed=False)[0])
    return MeshStats(len(v), len(f), bool((counts == 2).all()), vol, n_comp)


def write_binary_stl(path: str | Path, mesh: Mesh, name: str = "squad1") -> Path:
    """Write a binary STL (little-endian, per-face normals)."""
    v, f = mesh.vertices, mesh.faces
    tri = v[f]  # (F, 3, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 0)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = np.zeros(len(f), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"], rec["v"] = n, tri
    with open(p, "wb") as fh:
        fh.write(name.encode()[:80].ljust(80, b"\0"))
        fh.write(struct.pack("<I", len(f)))
        fh.write(rec.tobytes())
    return p


def read_binary_stl(path: str | Path) -> Mesh:
    raw = Path(path).read_bytes()
    (n,) = struct.unpack("<I", raw[80:84])
    if len(raw) != 84 + 50 * n:
        raise ContractError("not a valid binary STL (size mismatch)")
    rec = np.frombuffer(raw, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n, offset=84)
    tri = rec["v"].reshape(-1, 3).astype(np.float64)
    uniq, inv = np.unique(np.round(tri, 6), axis=0, return_inverse=True)
    return Mesh(uniq, inv.reshape(-1, 3).astype(np.int64))
