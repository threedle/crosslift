from __future__ import annotations

import numpy as np


def read_mesh_arrays(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a triangle mesh as ``(V, F)`` arrays for the C++ stages
    """
    import igl

    V, F = igl.read_triangle_mesh(path)
    return (
        np.ascontiguousarray(V, dtype=np.float64),
        np.ascontiguousarray(F, dtype=np.int32),
    )

def read_quad_obj(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load an OBJ as ``(V, F)``, keeping only its quad faces.

    Triangles and n-gons in the file are dropped.
    """
    V, F = [], []
    with open(path) as fh:
        for line in fh:
            token = line.split()
            if not token:
                continue
            if token[0] == "v":
                V.append([float(c) for c in token[1:4]])
            elif token[0] == "f" and len(token) == 5:
                F.append([int(c.split("/")[0]) - 1 for c in token[1:]])
    return (
        np.ascontiguousarray(V, dtype=np.float64).reshape(-1, 3),
        np.ascontiguousarray(F, dtype=np.int32).reshape(-1, 4),
    )

def write_obj(path: str, V: np.ndarray, F: np.ndarray) -> None:
    """Write a mesh to OBJ."""
    V = np.asarray(V)
    F = np.asarray(F)
    with open(path, "w") as fh:
        fh.writelines(f"v {v[0]} {v[1]} {v[2]}\n" for v in V)
        # OBJ indices are 1-based.
        fh.writelines(
            "f " + " ".join(str(int(i) + 1) for i in face) + "\n" for face in F
        )


def read_rawfield(path: str) -> tuple[np.ndarray, int]:
    """Read Directional's ``.rawfield`` format.
    
    Returns:
        ``(field, N)`` with field of shape (F, N, 3), float64.
    """
    with open(path, "rb") as fh:
        header = fh.readline().decode("utf-8").strip().split()
        if len(header) != 2:
            raise ValueError(
                f"{path}: expected an ASCII '.rawfield' header of the form "
                f"'<N> <num_faces>', got {header!r}"
            )
        N, num_faces = int(header[0]), int(header[1])
        data = np.loadtxt(fh, dtype=np.float64)

    return data.reshape(num_faces, N, 3), N


def write_rawfield(path: str, field: np.ndarray) -> None:
    """Write an (F, N, 3) field in Directional's ``.rawfield`` format."""
    field = np.asarray(field, dtype=np.float64)
    if field.ndim != 3 or field.shape[2] != 3:
        raise ValueError(f"expected an (F, N, 3) field, got shape {field.shape}")
    num_faces, N = field.shape[0], field.shape[1]
    with open(path, "w") as fh:
        fh.write(f"{N} {num_faces}\n")
        fh.writelines(" ".join(map(str, row)) + "\n" for row in field.reshape(num_faces, -1))
