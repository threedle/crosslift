from __future__ import annotations

import os

import numpy as np

from crosslift.geometry.repair import make_manifold


def _edge_length_for_faces(V, F, target_faces):
    """Equilateral edge length tiling this surface's area in ``target_faces``."""
    corners = V[F]
    area = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
        axis=1).sum()
    return float(np.sqrt(4.0 * area / (np.sqrt(3.0) * target_faces)))


def remesh_isotropic(V, F, target_edge=None, target_faces=None, iterations=10,
                     adaptive=False, label=""):
    """Rebuild the surface with near-uniform triangles.

    Args:
        V: (nV, 3) float64 vertex positions.
        F: (nF, 3) int32 faces.
        target_edge: edge length to aim for, in world units.
        target_faces: approximate face count, used when ``target_edge`` is None.
            Defaults to the input face count.
        iterations: split, collapse, flip and relax passes.
        adaptive: let curvature vary the edge length.
        label: prefix for the summary lines.

    Returns:
        ``(V, F)``, manifold and consistently oriented.
    """
    import pymeshlab as ml

    if target_edge is None:
        if target_faces is None:
            target_faces = F.shape[0]
        target_edge = _edge_length_for_faces(V, F, target_faces)

    mesh_set = ml.MeshSet()
    mesh_set.add_mesh(ml.Mesh(np.ascontiguousarray(V, dtype=np.float64),
                              np.ascontiguousarray(F, dtype=np.int32)))
    mesh_set.meshing_isotropic_explicit_remeshing(
        iterations=iterations, adaptive=adaptive,
        targetlen=ml.PureValue(target_edge))
    current = mesh_set.current_mesh()

    remeshed_V = np.ascontiguousarray(current.vertex_matrix(), dtype=np.float64)
    remeshed_F = np.ascontiguousarray(current.face_matrix(), dtype=np.int32)
    print(f"{label}Remesh: {F.shape[0]} -> {remeshed_F.shape[0]} faces, "
          f"target edge {target_edge:.5g}")
    return make_manifold(remeshed_V, remeshed_F, label=label)[:2]


def remesh_mesh_file(path, out_dir, target_edge=None, target_faces=None,
                     iterations=10, adaptive=False, label="  "):
    """Write an isotropically remeshed copy of ``path`` into ``out_dir``.

    Args:
        path: mesh to remesh.
        out_dir: directory to write the remeshed copy into.
        target_edge: edge length to aim for, in world units.
        target_faces: approximate face count, used when ``target_edge`` is None.
            Defaults to the input face count.
        iterations: split, collapse, flip and relax passes.
        adaptive: let curvature vary the edge length.
        label: prefix for the summary lines.

    Returns:
        Path to the remeshed mesh.
    """
    from crosslift.geometry.io import read_mesh_arrays, write_obj

    V, F = read_mesh_arrays(path)
    V, F = remesh_isotropic(V, F, target_edge, target_faces, iterations,
                            adaptive, label)

    name = os.path.splitext(os.path.basename(path))[0].removesuffix("_repaired")
    remeshed_path = os.path.join(out_dir, f"{name}_remeshed.obj")
    write_obj(remeshed_path, V, F)
    print(f"{label}Remeshed mesh: {remeshed_path}")
    return remeshed_path
