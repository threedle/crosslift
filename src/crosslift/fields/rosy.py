from __future__ import annotations

import math

import numpy as np
import torch


def expand_rosy(rep, normals, N: int):
    """Expand one representative direction per face into all N.

    Args:
        rep: (F, 3) representative direction per face
        normals: (F, 3) face normals
        N: N-RoSy (4 for cross field)

    Returns:
        (F, N, 3) expanded field
    """
    if isinstance(rep, torch.Tensor):
        v0 = rep / rep.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        n_cross_v = torch.linalg.cross(normals, v0, dim=-1)
        return torch.stack(
            [
                v0 * math.cos(2.0 * math.pi * k / N)
                + n_cross_v * math.sin(2.0 * math.pi * k / N)
                for k in range(N)
            ],
            dim=1,
        )

    rep = np.asarray(rep, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    v0 = rep / np.maximum(np.linalg.norm(rep, axis=-1, keepdims=True), 1e-12)
    n_cross_v = np.cross(normals, v0)
    return np.stack(
        [
            v0 * math.cos(2.0 * math.pi * k / N)
            + n_cross_v * math.sin(2.0 * math.pi * k / N)
            for k in range(N)
        ],
        axis=1,
    )


def to_extrinsic(field, normalize: bool = False) -> np.ndarray:
    """Flatten an (F, N, 3) field to the (F, 3N) layout Directional expects.

    Args:
        field: (F, N, 3) full N-RoSy field.
        normalize: normalize each direction first
    """
    ext = np.ascontiguousarray(field, dtype=np.float64)
    if normalize:
        ext = ext / np.maximum(np.linalg.norm(ext, axis=2, keepdims=True), 1e-12)
    return np.ascontiguousarray(ext.reshape(ext.shape[0], -1))
