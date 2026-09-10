from __future__ import annotations

import cupy as cp
import torch
import torch.nn.functional as F
from cupyx.scipy.sparse import bmat as cp_bmat
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from cupyx.scipy.sparse import diags as cp_diags
from cupyx.scipy.sparse import eye as cp_eye
from cupyx.scipy.sparse import vstack as cp_vstack
from cupyx.scipy.sparse.linalg import lsmr as cp_lsmr
from cupyx.scipy.sparse.linalg import spsolve as cp_spsolve

from crosslift.fields.surface_directions import SurfaceDirections
from crosslift.geometry.mesh import Mesh
from crosslift.rendering.base import RasterOutput

_EPS = 1e-12


# Weights
def view_alignment(
    mesh: Mesh,
    c2w: torch.Tensor,
    projection_type: str = "perspective",
    power: float = 1.0,
) -> torch.Tensor:
    """``(B, F)`` grazing cosine between each face and each camera

    Args:
        mesh: Mesh
        c2w: ``(B, 4, 4)`` camera-to-world
        projection_type: ``'perspective'`` or ``'orthographic'``
        power: Exponent on the cosine, higher values punush grazing angles more

    Returns:
        ``(B, F)`` in ``[0, 1]``
    """
    if projection_type == "perspective":
        centroids = mesh.get_face_centroids()                        # (F, 3)
        to_camera = F.normalize(c2w[:, None, :3, 3] - centroids[None], dim=-1)
    elif projection_type == "orthographic":
        to_camera = F.normalize(c2w[:, :3, 2], dim=-1)[:, None, :]   # (B, 1, 3)
    else:
        raise ValueError(
            f"Unknown projection type {projection_type!r}; "
            f"expected 'perspective' or 'orthographic'"
        )
    alignment = (mesh.face_normals[None] * to_camera).sum(-1).abs().clamp(0.0, 1.0)
    if power != 1.0:
        alignment = alignment.clamp_min(_EPS) ** power
    return alignment


def centroid_weights(
    dirs: SurfaceDirections, mesh: Mesh, raster: RasterOutput
) -> torch.Tensor:
    """``(P,)`` Gaussian falloff from each pixel to its face's centroid
    """
    centroids = mesh.get_face_centroids()                             # (F, 3)
    centroids_h = F.pad(centroids, (0, 1), value=1.0)
    centroids_cam = centroids_h @ raster.w2c.transpose(-1, -2)        # (B, F, 4)
    centroid_at_obs = centroids_cam[dirs.view_idx, dirs.face_idx][:, :3]

    pos_cam = raster.pixel_pos_cam.reshape(-1, 3)[dirs.pixel_idx]     # (P, 3)
    dist_sq = ((pos_cam - centroid_at_obs) ** 2).sum(-1)

    corners = mesh.vertices[mesh.faces.long()]                        # (F, k, 3)
    radius_sq = ((corners - centroids[:, None]) ** 2).sum(-1).max(-1).values
    sigma_sq = radius_sq[dirs.face_idx].clamp_min(_EPS)

    return torch.exp(-dist_sq / (2.0 * sigma_sq))


def compute_coherence_weights(
    face_crosses: torch.Tensor, weights: torch.Tensor, epsilon: float = 1e-9
) -> torch.Tensor:
    """``(F,)`` agreement across views, per face. ``C_f = |sum_b w_b z_b| / sum_b w_b``.
    Args:
        face_crosses: ``(B, F)`` complex, the per-view representative directions
            in N-RoSy space.
        weights: ``(B, F)`` per-view weights
        epsilon: Guard for unseen faces

    Returns:
        ``(F,)`` real in ``[0, 1]``
    """
    invalid = torch.isnan(weights) | torch.isnan(face_crosses)
    weights = torch.where(invalid, 0.0, weights)
    face_crosses = torch.where(invalid, 0.0 + 0.0j, face_crosses)

    total = weights.sum(dim=0)
    magnitude = (face_crosses * weights).sum(dim=0).abs()
    return torch.where(total >= epsilon, magnitude / total, 0.0)


# Solver
def _solve_block(A, b, use_direct_solver: bool, tol: float):
    """Least squares on the real 2x2 block form."""
    if use_direct_solver:
        AtA = A.T @ A
        Atb = A.T @ b
        reg = 1e-10 * cp_eye(AtA.shape[0], format="csc")
        return cp_spsolve((AtA + reg).tocsc(), Atb)
    return cp_lsmr(A, b, atol=tol, btol=tol)[0]

def mixed_linear_solve(
    constraints: torch.Tensor,
    f_idx: torch.Tensor,
    w_c: torch.Tensor,
    mesh: Mesh,
    lambda_s: float,
    lambda_c: float,
    face_mask: torch.Tensor | None = None,
    normalize: bool = True,
    N: int = 4,
    use_direct_solver: bool = False,
    sym_pairs=None,
    sym_coeffs=None,
    lambda_sym: float = 0.0,
    tol: float = 1e-12,
) -> torch.Tensor:
    """Smoothness and mixed hard/soft alignment constraints, per face. Negative ``w_c``
    indicates hard constraints.

    Args:
        constraints: ``(C,)`` complex, already N-RoSy encoded.
        f_idx: ``(C,)`` long, the face each constraint applies to.
        w_c: ``(C,)`` real. **Negative marks a hard constraint.**
        mesh: The mesh object containing edge and face information.
        lambda_s: Smoothness weight.
        lambda_c: Soft-constraint weight.
        face_mask: ``(F,)`` bool, the faces to solve over; ``None`` uses all.
        normalize: Scale the solution to unit magnitude.
        N: N-RoSy symmetry order.
        use_direct_solver: Normal equations with a direct sparse solve instead of
            LSMR. Faster on small systems, much heavier on large ones.
        sym_pairs: ``(P, 2)`` int mirror-face pairs, or None.
        sym_coeffs: ``(P,)`` complex coefficients, so each pair asks for
            ``u_j = c * conj(u_i)``.
        lambda_sym: Symmetry weight; 0 disables the term.
        tol: LSMR ``atol``/``btol``.

    Returns:
        ``(F,)`` complex, zero outside ``face_mask``.
    """
    device = constraints.device
    num_faces = mesh.num_faces

    is_subset = face_mask is not None
    if is_subset:
        mask_cp = cp.asarray(face_mask)
        global_idx = cp.where(mask_cp)[0]
    else:
        mask_cp = cp.ones(num_faces, dtype=cp.bool_)
        global_idx = cp.arange(num_faces, dtype=cp.int32)

    num_solved = int(global_idx.shape[0])
    if num_solved == 0:
        return torch.zeros(num_faces, device=device, dtype=torch.complex64)

    to_local = cp.full((num_faces,), -1, dtype=cp.int32)
    to_local[global_idx] = cp.arange(num_solved, dtype=cp.int32)

    # Constraints
    constraints_cp = cp.asarray(constraints)
    w_c_cp = cp.asarray(w_c)
    local_f = to_local[cp.asarray(f_idx)]

    on_solved = local_f >= 0
    if is_subset and not bool(cp.all(on_solved)):
        constraints_cp = constraints_cp[on_solved]
        w_c_cp = w_c_cp[on_solved]
        local_f = local_f[on_solved]

    is_hard = w_c_cp < 0.0
    hard_f, hard_vals = local_f[is_hard], constraints_cp[is_hard]
    soft_f = local_f[~is_hard]
    soft_vals = constraints_cp[~is_hard]
    soft_w = w_c_cp[~is_hard]
    num_soft = int(soft_vals.shape[0])

    # Smoothness
    edge_faces = cp.asarray(mesh.edge_faces)
    usable = (edge_faces[:, 0] >= 0) & (edge_faces[:, 1] >= 0)
    if is_subset:
        usable &= mask_cp[edge_faces[:, 0]] & mask_cp[edge_faces[:, 1]]

    edge_idx = cp.where(usable)[0]
    num_edges = int(edge_idx.shape[0])
    edge_faces = edge_faces[edge_idx]
    w_k = cp.asarray(mesh.harmonic_weights)[edge_idx]
    conns = cp.asarray(mesh.edge_connections)[edge_idx]

    # Assemble
    rows_smooth = cp.arange(num_edges)
    smooth_scale = cp.sqrt(lambda_s * w_k)
    vals_f = smooth_scale * cp.conj(conns[:, 0]) ** N
    vals_g = -smooth_scale * cp.conj(conns[:, 1]) ** N

    rows_soft = num_edges + cp.arange(num_soft)
    vals_soft = cp.sqrt(soft_w * lambda_c)

    b = cp.zeros(num_edges + num_soft, dtype=cp.complex128)
    b[num_edges:] = soft_vals * vals_soft

    if lambda_c == 0.0 and num_soft > 0:
        vals_soft[0] = cp.sqrt(soft_w[0])
        b[num_edges] = soft_vals[0] * vals_soft[0]

    rows = cp.hstack([rows_smooth, rows_smooth, rows_soft])
    cols = cp.hstack([to_local[edge_faces[:, 0]], to_local[edge_faces[:, 1]], soft_f])
    vals = cp.hstack([vals_f, vals_g, vals_soft])
    shape = (num_edges + num_soft, num_solved)

    A_real = cp_csr_matrix((vals.real, (rows, cols)), shape=shape)
    A_imag = cp_csr_matrix((vals.imag, (rows, cols)), shape=shape)
    A_block = cp_bmat([[A_real, -A_imag], [A_imag, A_real]], format="csr")
    b_block = cp.hstack([b.real, b.imag])

    # Symmetry
    if lambda_sym > 0 and sym_pairs is not None and len(sym_pairs) > 0:
        pairs = cp.asarray(sym_pairs)
        local_i, local_j = to_local[pairs[:, 0]], to_local[pairs[:, 1]]
        pair_keep = (local_i >= 0) & (local_j >= 0)
        local_i, local_j = local_i[pair_keep], local_j[pair_keep]
        c_sym = cp.asarray(sym_coeffs)[pair_keep]
        num_pairs = int(local_i.shape[0])

        if num_pairs > 0:
            s = cp.sqrt(cp.float64(lambda_sym))
            cr, ci = c_sym.real, c_sym.imag
            ones = cp.ones(num_pairs, dtype=cp.float64)
            p_idx = cp.arange(num_pairs)
            rows_sym = cp.hstack(
                [p_idx, p_idx, p_idx, num_pairs + p_idx, num_pairs + p_idx,
                 num_pairs + p_idx]
            )
            cols_sym = cp.hstack(
                [local_j, local_i, num_solved + local_i,
                 num_solved + local_j, local_i, num_solved + local_i]
            )
            vals_sym = s * cp.hstack([ones, -cr, -ci, ones, -ci, cr])
            A_block = cp_vstack(
                [A_block,
                 cp_csr_matrix((vals_sym, (rows_sym, cols_sym)),
                               shape=(2 * num_pairs, 2 * num_solved))],
                format="csr",
            )
            b_block = cp.hstack(
                [b_block, cp.zeros(2 * num_pairs, dtype=b_block.dtype)]
            )

    # Eliminate hard constraints
    num_hard = int(hard_f.shape[0])
    fixed_block = None
    if num_hard > 0:
        fixed = cp.zeros(num_solved, dtype=cp.complex128)
        fixed[hard_f] = hard_vals
        fixed_block = cp.hstack([fixed.real, fixed.imag])
        b_block = b_block - A_block.dot(fixed_block)

        free = cp.ones(2 * num_solved, dtype=cp.float64)
        free[hard_f] = 0.0
        free[num_solved + hard_f] = 0.0
        A_block = A_block.dot(cp_diags([free], [0], format="csr"))

    x_block = _solve_block(A_block, b_block, use_direct_solver, tol)
    if fixed_block is not None:
        x_block = x_block + fixed_block

    x_local = (x_block[:num_solved] + 1j * x_block[num_solved:]).astype(cp.complex64)

    if normalize:
        nonzero = cp.abs(x_local) > _EPS
        x_local[nonzero] = x_local[nonzero] / cp.abs(x_local[nonzero])

    if is_subset:
        x_global = cp.zeros((num_faces,), dtype=cp.complex64)
        x_global[global_idx] = x_local
    else:
        x_global = x_local

    return torch.as_tensor(x_global, device=device)


# Stage I: per-view solve
def solve_view_field(
    dirs: SurfaceDirections,
    mesh: Mesh,
    lambda_s: float = 1.0,
    lambda_c: float = 10.0,
    N: int = 4,
    w_c: torch.Tensor | None = None,
    use_direct_solver: bool = False,
    tol: float = 1e-12,
) -> torch.Tensor:
    """Stage I Solve

    Args:
        dirs: Raw directions, not N-RoSy encoded
        mesh: Mesh being solved over.
        lambda_s: Smoothness weight within a view.
        lambda_c: Weight on the per-pixel constraints.
        N: N-RoSy symmetry order.
        w_c: ``(P,)`` per-observation weights, aligned with ``dirs``. ``None``
            weights every pixel equally.
        use_direct_solver: Passed through to :func:`mixed_linear_solve`.
        tol: LSMR tolerance.

    Returns:
        ``(B, F)`` complex unit directions, zero on faces not visible in a view.
    """
    out = torch.zeros(
        (dirs.num_views, mesh.num_faces), dtype=torch.complex64, device=dirs.device
    )
    if len(dirs) == 0:
        return out

    if w_c is None:
        w_c = torch.ones(len(dirs), dtype=torch.float32, device=dirs.device)
    elif w_c.shape[0] != len(dirs):
        raise ValueError(
            f"w_c has {w_c.shape[0]} entries but there are {len(dirs)} directions"
        )

    for view in range(dirs.num_views):
        selected = dirs.view_idx == view
        if not bool(selected.any()):
            continue
        encoded = mixed_linear_solve(
            constraints=dirs.value[selected] ** N,
            f_idx=dirs.face_idx[selected],
            w_c=w_c[selected],
            mesh=mesh,
            lambda_s=lambda_s,
            lambda_c=lambda_c,
            face_mask=dirs.visible_face_mask[view],
            normalize=False,
            N=N,
            use_direct_solver=use_direct_solver,
            tol=tol,
        )
        decoded = encoded ** (1.0 / N)
        out[view] = decoded / decoded.abs().clamp_min(_EPS)

    return out


# Stage II: global solve
def solve_global_field(
    constraints: torch.Tensor,
    f_idx: torch.Tensor,
    w_c: torch.Tensor,
    mesh: Mesh,
    lambda_s: float,
    lambda_c: float,
    N: int = 4,
    normalize: bool = False,
    use_direct_solver: bool = False,
    sym_pairs=None,
    sym_coeffs=None,
    lambda_sym: float = 0.0,
    tol: float = 1e-12,
) -> torch.Tensor:
    """Solve one field over the whole mesh

    Args:
        constraints: ``(C,)`` complex, N-RoSy encoded. Soft constraints from the
            per-view fields and hard ones (``w_c < 0``) concatenated.
        f_idx: ``(C,)`` long face indices.
        w_c: ``(C,)`` real weights; negative marks a hard constraint.
        mesh: Mesh being solved over.
        lambda_s: Smoothness weight.
        lambda_c: Soft-constraint weight.
        N: N-RoSy symmetry order.
        normalize: Scale the solution to unit magnitude.
        use_direct_solver: Passed to :func:`mixed_linear_solve`.
        sym_pairs: ``(P, 2)`` mirror-face pairs, or None.
        sym_coeffs: ``(P,)`` complex coefficients, or None.
        lambda_sym: Symmetry weight; 0 disables.
        tol: LSMR tolerance.

    Returns:
        ``(F,)`` complex, still N-RoSy encoded.
    """
    solver_kwargs = {
        "mesh": mesh,
        "lambda_s": lambda_s,
        "lambda_c": lambda_c,
        "normalize": normalize,
        "N": N,
        "use_direct_solver": use_direct_solver,
        "sym_pairs": sym_pairs,
        "sym_coeffs": sym_coeffs,
        "lambda_sym": lambda_sym,
        "tol": tol,
    }

    if mesh.num_components == 1:
        return mixed_linear_solve(constraints, f_idx, w_c, **solver_kwargs)

    device = constraints.device
    labels = torch.from_numpy(mesh.face_components).to(device).long()
    constraint_labels = labels[f_idx]
    x = torch.zeros(mesh.num_faces, device=device, dtype=torch.complex64)

    for component in range(mesh.num_components):
        component_mask = labels == component
        selected = constraint_labels == component
        component_constraints = constraints[selected]
        component_f_idx = f_idx[selected]
        component_w_c = w_c[selected]

        if component_constraints.numel() == 0:
            anchor = int(component_mask.nonzero()[0].item())
            component_constraints = torch.tensor(
                [1.0 + 0.0j], dtype=constraints.dtype, device=device
            )
            component_f_idx = torch.tensor(
                [anchor], dtype=f_idx.dtype, device=device
            )
            component_w_c = torch.tensor([-1.0], dtype=w_c.dtype, device=device)

        x = x + mixed_linear_solve(
            component_constraints,
            component_f_idx,
            component_w_c,
            face_mask=component_mask,
            **solver_kwargs,
        )

    return x
