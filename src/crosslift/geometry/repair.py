from __future__ import annotations

import os

import numpy as np


def coincident_face_mask(V, F, decimals=9):
    """``(nF,)`` bool mask, False on faces spanning an earlier face's positions."""
    _, position = np.unique(np.round(V, decimals), axis=0, return_inverse=True)
    corners = np.sort(position.reshape(-1)[F], axis=1)
    _, first = np.unique(corners, axis=0, return_index=True)
    mask = np.zeros(F.shape[0], dtype=bool)
    mask[first] = True
    return mask


def make_manifold(V, F, epsilon=0.0, label=""):
    """Weld duplicate vertices, drop coincident faces, restore manifoldness.

    Args:
        V: (nV, 3) float64 vertex positions.
        F: (nF, 3) int32 faces.
        epsilon: coordinate-wise tolerance for merging vertices.
        label: prefix for the summary line.

    Returns:
        ``(V, F, face_mask)`` where ``face_mask`` selects the surviving faces
        out of the input ``F``.
    """
    import igl

    n_verts, n_faces = V.shape[0], F.shape[0]

    V, _, _, F = igl.remove_duplicate_vertices(V, F.astype(np.int64), epsilon)
    face_mask = coincident_face_mask(V, F)
    F = np.ascontiguousarray(F[face_mask], dtype=np.int64)
    V, F, _ = igl.split_nonmanifold(np.ascontiguousarray(V, np.float64), F)

    V = np.ascontiguousarray(V, dtype=np.float64)
    F = np.ascontiguousarray(F, dtype=np.int32)
    if V.shape[0] != n_verts or F.shape[0] != n_faces:
        print(f"{label}Mesh repair: {n_verts} -> {V.shape[0]} vertices, "
              f"{n_faces} -> {F.shape[0]} faces")
    return V, F, face_mask


def repair_mesh_file(path, out_dir, epsilon=0.0, label="  "):
    """Write a welded, manifold copy of ``path`` into ``out_dir``.

    Args:
        path: mesh to repair.
        out_dir: directory to write the repaired copy into.
        epsilon: coordinate-wise tolerance for merging vertices.
        label: prefix for the summary lines.

    Returns:
        Path to the repaired mesh, or ``path`` unchanged when it needed nothing.
    """
    from crosslift.geometry.io import read_mesh_arrays, write_obj

    V, F = read_mesh_arrays(path)
    repaired_V, repaired_F, _ = make_manifold(V, F, epsilon, label=label)
    if repaired_V.shape[0] == V.shape[0] and repaired_F.shape[0] == F.shape[0]:
        return path

    name = os.path.splitext(os.path.basename(path))[0]
    repaired_path = os.path.join(out_dir, f"{name}_repaired.obj")
    write_obj(repaired_path, repaired_V, repaired_F)
    print(f"{label}Repaired mesh: {repaired_path}")
    return repaired_path
