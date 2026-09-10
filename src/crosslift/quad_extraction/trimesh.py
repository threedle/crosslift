from __future__ import annotations

import numpy as np
import torch


class TensorHalfEdge:
    """Half-edge connectivity as flat tensors, built without a Python loop."""

    def __init__(self, F_t: torch.Tensor, device="cpu"):
        self.device = device
        self.F = F_t.to(device)
        self.num_faces = self.F.shape[0]
        self.num_halfedges = self.num_faces * 3

        arange = torch.arange(self.num_halfedges, device=device)
        face_base = (arange // 3) * 3
        mod_idx = arange % 3
        self.next_he = face_base + ((mod_idx + 1) % 3)
        self.prev_he = face_base + ((mod_idx + 2) % 3)
        self.face = arange // 3
        self.he_origin = self.F.reshape(-1)

        u = self.he_origin
        v = self.he_origin[self.next_he]
        min_v = torch.minimum(u, v)
        max_v = torch.maximum(u, v)
        # Pack the undirected edge into one integer so sorting groups twins.
        edge_key = min_v * (max_v.max() + 1) + max_v

        sorted_keys, sort_idx = torch.sort(edge_key)
        has_twin_after = sorted_keys[:-1] == sorted_keys[1:]
        self.twin_he = torch.full(
            (self.num_halfedges,), -1, dtype=torch.long, device=device)
        idx_a = sort_idx[:-1][has_twin_after]
        idx_b = sort_idx[1:][has_twin_after]
        self.twin_he[idx_a] = idx_b
        self.twin_he[idx_b] = idx_a

        unique_keys, inverse_indices = torch.unique(edge_key, return_inverse=True)
        self.edge = inverse_indices
        self.num_edges = unique_keys.shape[0]
        self.num_vertices = int(self.F.max().item()) + 1
        self.is_boundary_he = self.twin_he == -1


class TriMesh:
    """Triangle mesh in double precision, with per-face tangent frames."""

    def __init__(self, vertices, faces, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.V = (
            torch.from_numpy(vertices) if isinstance(vertices, np.ndarray) else vertices
        ).to(dtype=torch.float64, device=device)
        self.F = (
            torch.from_numpy(faces).long() if isinstance(faces, np.ndarray) else faces
        ).to(device)

        self.topology = TensorHalfEdge(self.F, device=self.device)
        self.compute_geometry()

    def compute_geometry(self) -> None:
        v0, v1, v2 = self.V[self.F[:, 0]], self.V[self.F[:, 1]], self.V[self.F[:, 2]]
        e1, e2 = v1 - v0, v2 - v0

        cross = torch.linalg.cross(e1, e2, dim=1)
        norm_cross = torch.linalg.norm(cross, dim=1, keepdim=True)
        # Degenerate triangles: unit denominator instead of inf/NaN.
        norm_cross = torch.where(norm_cross < 1e-12, torch.ones_like(norm_cross), norm_cross)
        self.faceAreas = 0.5 * norm_cross.squeeze(-1)
        self.faceNormals = cross / norm_cross
        self.barycenters = (v0 + v1 + v2) / 3.0

        V_normals = torch.zeros_like(self.V)
        weighted = self.faceNormals * self.faceAreas.unsqueeze(1)
        for i in range(3):
            V_normals.index_add_(0, self.F[:, i], weighted)
        vn_norm = torch.linalg.norm(V_normals, dim=1, keepdim=True)
        vn_norm = torch.where(vn_norm < 1e-12, torch.ones_like(vn_norm), vn_norm)
        self.vertexNormals = V_normals / vn_norm

        # Tangent frame: first edge as x, normal cross x as y.
        x_norm = torch.linalg.norm(e1, dim=1, keepdim=True)
        x_norm = torch.where(x_norm < 1e-12, torch.ones_like(x_norm), x_norm)
        self.FBx = e1 / x_norm
        localY = torch.linalg.cross(self.faceNormals, self.FBx, dim=1)
        y_norm = torch.linalg.norm(localY, dim=1, keepdim=True)
        y_norm = torch.where(y_norm < 1e-12, torch.ones_like(y_norm), y_norm)
        self.FBy = localY / y_norm

    @classmethod
    def from_file(cls, path: str, device=None) -> TriMesh:
        """Load through the same reader the C++ stages use."""
        from crosslift.geometry.io import read_mesh_arrays

        V, F = read_mesh_arrays(path)
        return cls(V, F, device=device)


def slice_component(V, F, face_labels, k, face_arrays=None):
    """Cut connected component ``k`` out as a self-contained mesh.

    Args:
        V: (nV, 3) float64 vertex positions of the full mesh.
        F: (nF, 3) int32 faces of the full mesh.
        face_labels: (nF,) component id per face.
        k: which component to extract.
        face_arrays: ``{name: (nF, ...) array}`` of per-face data to slice in
            lockstep with the faces. ``None`` values pass straight through, so
            optional inputs need no special-casing at the call site.

    Returns:
        ``(V_k, F_k, v_global, f_mask, sliced)`` — the sub-mesh, the original
        vertex ids behind it, the face mask that selected it, and the sliced
        per-face arrays.
    """
    f_mask = face_labels == k
    F_sub = F[f_mask]
    v_global = np.unique(F_sub.reshape(-1))

    remap = -np.ones(V.shape[0], dtype=np.int64)
    remap[v_global] = np.arange(v_global.size)
    F_k = np.ascontiguousarray(remap[F_sub], dtype=np.int32)
    V_k = np.ascontiguousarray(V[v_global], dtype=np.float64)

    sliced = {}
    if face_arrays:
        for name, arr in face_arrays.items():
            sliced[name] = None if arr is None else np.ascontiguousarray(arr[f_mask])
    return V_k, F_k, v_global, f_mask, sliced


def concat_meshes(Vs, Fs):
    """Concatenate per-component meshes, offsetting vertex indices."""
    if not Vs:
        valence = Fs[0].shape[1] if Fs else 4
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, valence), dtype=np.int32),
        )
    offsets = np.cumsum([0] + [v.shape[0] for v in Vs[:-1]])
    return (
        np.vstack(Vs).astype(np.float64),
        np.vstack([f + off for f, off in zip(Fs, offsets)]).astype(np.int32),
    )
