from __future__ import annotations

import os
import time

import numpy as np
import torch

from crosslift.fields.rosy import expand_rosy
from crosslift.quad_extraction.polyvector import (
    PolyVectorField,
    extract_polyvector_roots,
    roots_to_coeffs,
)
from crosslift.quad_extraction.trimesh import TriMesh, slice_component


def log(msg, log_file=None):
    """Print, and optionally append to a log file."""
    print(msg, flush=True)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')


def run_nrosy_curl_correction_direct(
    mesh_path,
    raw_vecs_np,
    N,
    w_align_weights=None,
    w_align=1.0,
    num_iterations=5,
    log_path=None,
):
    """
    General N-RoSy curl correction via direct curl projection on raw fields.

    Args:
        mesh_path: Path to input mesh (.obj)
        raw_vecs_np: (F, N, 3) numpy array of raw field vectors
        N: N-RoSy symmetry order
        w_align_weights: Optional (F,) numpy array of per-face alignment weights.
                         Values < 0 mark hard constraints (frozen faces).
        w_align: Uniform alignment weight (default 1.0, used when w_align_weights is None)
        num_iterations: Number of matching/projection iterations (default 5)
        log_path: Optional path to log file

    Returns:
        curlfree_vecs: (F, N, 3) numpy float64 array of curl-free field vectors
    """
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as cp_csr
    from cupyx.scipy.sparse.linalg import LinearOperator
    from cupyx.scipy.sparse.linalg import cg as cp_cg

    timings = {}

    if log_path:
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"N-RoSy Curl Correction - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")

    log("=" * 60, log_path)
    log("N-ROSY CURL CORRECTION (DIRECT PROJECTION)", log_path)
    log("=" * 60, log_path)

    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Device: {device}", log_path)

    log(f"\nLoading mesh: {mesh_path}", log_path)
    mesh = TriMesh.from_file(mesh_path, device=device)
    num_faces = mesh.F.shape[0]
    log(f"  Vertices: {mesh.V.shape[0]}, Faces: {num_faces}", log_path)
    log(f"  N-RoSy: N={N}", log_path)

    # Convert raw_vecs to torch tensor (F, N, 3)
    raw_vecs = torch.from_numpy(raw_vecs_np).to(dtype=torch.float64, device=device)

    timings['load'] = time.time() - t0

    # Intrinsic 2D representation
    t0 = time.time()
    u = (raw_vecs * mesh.FBx.unsqueeze(1)).sum(dim=2)  # (F, N)
    v = (raw_vecs * mesh.FBy.unsqueeze(1)).sum(dim=2)   # (F, N)
    x0 = torch.stack([u, v], dim=-1).reshape(-1)  # (2*N*F,)
    timings['intrinsic_convert'] = time.time() - t0

    # Inner edge topology
    t0 = time.time()
    he_inner_mask = (mesh.topology.twin_he != -1)
    he_indices = torch.nonzero(he_inner_mask).squeeze()
    twins = mesh.topology.twin_he
    unique_edge_mask = (he_indices < twins[he_indices])
    he_idx = he_indices[unique_edge_mask]
    num_inner_edges = he_idx.shape[0]

    left_faces = mesh.topology.face[he_idx]
    right_faces = mesh.topology.face[twins[he_idx]]

    # Edge vertices and vectors
    tri_idx = he_idx % 3
    face_from_he = he_idx // 3
    v0_idx = mesh.F[face_from_he, tri_idx]
    v1_idx = mesh.F[face_from_he, (tri_idx + 1) % 3]
    edge_vecs = mesh.V[v1_idx] - mesh.V[v0_idx]  # (E, 3)

    # Project edges onto face bases
    ein_left_x = (edge_vecs * mesh.FBx[left_faces]).sum(dim=1)
    ein_left_y = (edge_vecs * mesh.FBy[left_faces]).sum(dim=1)
    ein_right_x = (edge_vecs * mesh.FBx[right_faces]).sum(dim=1)
    ein_right_y = (edge_vecs * mesh.FBy[right_faces]).sum(dim=1)
    timings['edge_topology'] = time.time() - t0

    # Hard-constrained faces (frozen DOFs)
    constrained_dof_mask = None
    if w_align_weights is not None:
        hard_faces = np.where(w_align_weights < 0)[0]
        if len(hard_faces) > 0:
            constrained_dof_mask = torch.zeros(2 * N * num_faces, dtype=torch.bool, device=device)
            for f_idx in hard_faces:
                start = 2 * N * f_idx
                constrained_dof_mask[start:start + 2 * N] = True
            log(f"  Hard-constrained faces: {len(hard_faces)}", log_path)

    # Iterative matching + curl projection
    x_current = x0.clone()

    for iteration in range(num_iterations):
        # Reshape current field to (F, N) complex for matching
        x_reshaped = x_current.view(num_faces, N, 2)
        current_int_field = torch.complex(x_reshaped[..., 0], x_reshaped[..., 1])

        # Curl matching
        # Extrinsic field for curl computation
        ext_left = raw_vecs[left_faces] if iteration == 0 else _int_to_ext(current_int_field[left_faces], mesh, left_faces)
        ext_right = raw_vecs[right_faces] if iteration == 0 else _int_to_ext(current_int_field[right_faces], mesh, right_faces)

        # Edge vectors normalized for matching
        edge_norm = edge_vecs / (torch.linalg.norm(edge_vecs, dim=1, keepdim=True) + 1e-12)

        best_curl = torch.full((num_inner_edges,), float('inf'), dtype=torch.float64, device=device)
        matching = torch.zeros(num_inner_edges, dtype=torch.long, device=device)

        for j in range(N):
            curl_sq = torch.zeros(num_inner_edges, dtype=torch.float64, device=device)
            for k in range(N):
                vec_diff = ext_right[:, (j + k) % N, :] - ext_left[:, k, :]  # (E, 3)
                curl_component = (edge_norm * vec_diff).sum(dim=1)  # (E,)
                curl_sq = curl_sq + curl_component ** 2
            update_mask = curl_sq < best_curl
            best_curl[update_mask] = curl_sq[update_mask]
            matching[update_mask] = j

        # Curl matrix C: (N * E_inner, 2 * N * F)
        E = num_inner_edges
        n_range = torch.arange(N, device=device)

        row_base = torch.arange(E, device=device).unsqueeze(1) * N + n_range  # (E, N)
        rows = row_base.unsqueeze(2).expand(-1, -1, 4).reshape(-1)  # (E*N*4,)

        col_left_u = (2 * N * left_faces.unsqueeze(1) + 2 * n_range).reshape(-1)
        col_left_v = col_left_u + 1
        n_matched = (n_range.unsqueeze(0) + matching.unsqueeze(1) + N) % N  # (E, N)
        col_right_u = (2 * N * right_faces.unsqueeze(1) + 2 * n_matched).reshape(-1)
        col_right_v = col_right_u + 1

        cols = torch.stack([col_left_u.view(E, N), col_left_v.view(E, N),
                           col_right_u.view(E, N), col_right_v.view(E, N)], dim=-1).reshape(-1)

        vals_left_x = -ein_left_x.unsqueeze(1).expand(-1, N).reshape(-1)
        vals_left_y = -ein_left_y.unsqueeze(1).expand(-1, N).reshape(-1)
        vals_right_x = ein_right_x.unsqueeze(1).expand(-1, N).reshape(-1)
        vals_right_y = ein_right_y.unsqueeze(1).expand(-1, N).reshape(-1)
        vals = torch.stack([vals_left_x.view(E, N), vals_left_y.view(E, N),
                           vals_right_x.view(E, N), vals_right_y.view(E, N)], dim=-1).reshape(-1)

        C_shape = (N * E, 2 * N * num_faces)
        indices = torch.stack([rows.long(), cols.long()], dim=0)
        C_sparse = torch.sparse_coo_tensor(indices, vals.double(), C_shape, device=device).coalesce()

        # Convert to CuPy sparse
        C_coo = C_sparse.coalesce()
        C_cp = cp_csr((cp.asarray(C_coo.values()),
                       (cp.asarray(C_coo.indices()[0]), cp.asarray(C_coo.indices()[1]))),
                      shape=C_shape)

        # Curl projection: min ||x - x0||^2 s.t. Cx = 0
        x_cp = cp.asarray(x_current.double())

        # Apply hard constraints: zero out columns for frozen faces
        if constrained_dof_mask is not None:
            free_mask = (~constrained_dof_mask).to(torch.float64)
            free_mask_cp = cp.asarray(free_mask)
            diag_idx = cp.arange(2 * N * num_faces)
            FreeMask = cp_csr((free_mask_cp, (diag_idx, diag_idx)),
                              shape=(2 * N * num_faces, 2 * N * num_faces))
            C_cp = C_cp @ FreeMask

        Cx0 = C_cp @ x_cp
        dim_c = N * E
        reg = 1e-8

        def cct_matvec(v_in, C_cp=C_cp, reg=reg):
            return C_cp @ (C_cp.T @ v_in) + reg * v_in

        CCT_op = LinearOperator((dim_c, dim_c), matvec=cct_matvec, dtype=cp.float64)

        # Diagonal preconditioner
        row_norms_sq = cp.array(C_cp.power(2).sum(axis=1)).flatten() + reg
        M_inv = 1.0 / cp.maximum(row_norms_sq, 1e-12)
        M_precond = LinearOperator((dim_c, dim_c),
                                   matvec=lambda v_in, M_inv=M_inv: M_inv * v_in,
                                   dtype=cp.float64)

        lam, info = cp_cg(CCT_op, Cx0, rtol=1e-10, maxiter=2000, M=M_precond)
        if info != 0:
            lam, info = cp_cg(CCT_op, Cx0, rtol=1e-6, maxiter=5000)

        x_proj_cp = x_cp - C_cp.T @ lam
        x_current = torch.as_tensor(x_proj_cp, device=device, dtype=torch.float64)

        # Compute curl norm for logging
        curl_after = float(cp.linalg.norm(C_cp @ x_proj_cp))
        curl_before = float(cp.linalg.norm(Cx0))
        log(f"  Iter {iteration}: curl {curl_before:.6f} -> {curl_after:.6f}", log_path)

        # Cleanup
        C_cp = None
        del C_coo, Cx0, lam
        cp.get_default_memory_pool().free_all_blocks()

    timings['curl_projection'] = time.time() - t0

    # Extrinsic 3D vectors: vec = u * FBx + v * FBy
    t0 = time.time()
    x_final = x_current.view(num_faces, N, 2)
    curlfree_u = x_final[..., 0]  # (F, N)
    curlfree_v = x_final[..., 1]  # (F, N)

    curlfree_vecs = (curlfree_u.unsqueeze(-1) * mesh.FBx.unsqueeze(1) +
                     curlfree_v.unsqueeze(-1) * mesh.FBy.unsqueeze(1))  # (F, N, 3)

    curlfree_np = curlfree_vecs.cpu().numpy()
    timings['extrinsic_convert'] = time.time() - t0

    total = sum(timings.values())
    log("\n" + "=" * 60, log_path)
    log("TIMING SUMMARY", log_path)
    log("=" * 60, log_path)
    for name, t in timings.items():
        log(f"  {name}: {t:.2f}s", log_path)
    log(f"  TOTAL: {total:.2f}s", log_path)

    return curlfree_np


@torch.no_grad()

def _int_to_ext(int_field, mesh, face_indices):
    """Convert intrinsic (F, N) complex field to extrinsic (F, N, 3) vectors."""
    u = int_field.real  # (F, N)
    v = int_field.imag  # (F, N)
    return u.unsqueeze(-1) * mesh.FBx[face_indices].unsqueeze(1) + \
           v.unsqueeze(-1) * mesh.FBy[face_indices].unsqueeze(1)


@torch.no_grad()
def run_curl_correction_direct(
    mesh_path,
    raw_vecs_np,
    N=4,
    w_align_weights=None,
    w_align=1.0,
    num_iterations=None,
    log_path=None,
    log_interval=5,
    wSmooth=1.0,
    wRoSy=1.0,
    one_root=True,
    skip_initial_solve=False,
    dynamic_annealing=True,
    skip_first_implicit=True,
    tau_threshold=0.01,
    free_magnitude=False,
    magnitude_clamp=None,
    V=None,
    F=None,
):
    """
    Run curl correction on in-memory arrays. No file I/O for field data.

    Args:
        mesh_path: Path to input mesh (.obj). Ignored when V and F are provided.
        raw_vecs_np: (F, N, 3) numpy array of raw field vectors (already normalized)
        N: N-RoSy symmetry order (default 4)
        w_align_weights: Optional (F,) numpy array of per-face alignment weights
        w_align: Uniform alignment weight when w_align_weights is None (default 1.0)
        num_iterations: Number of iterations (None = adaptive)
        log_path: Optional path to log file
        log_interval: Polyvector iteration log interval
        wSmooth: Smoothness weight (default 1.0)
        wRoSy: RoSy weight (default 1.0)
        one_root: If True, use one root per face as constraint (default True)
        skip_initial_solve: If True, skip initial solve (default False)
        dynamic_annealing: If True, use adaptive stopping (default True)
        skip_first_implicit: If True, skip implicit step on first iteration (default True)
        tau_threshold: Convergence threshold (default 0.01)
        free_magnitude: If True, correct with the magnitude free so curl can be
            absorbed as scale rather than rotation (default False)
        magnitude_clamp: Bound the free magnitudes to [1/c, c] (default None)
        V: Optional (nV, 3) float64 vertex array. When provided with F, the mesh
            is built in-memory and `mesh_path` is not read. Used by per-component
            pipelines to avoid writing temporary OBJs for each shell.
        F: Optional (nF, 3) int32 face array; see V.

    Returns:
        curlfree_vecs: (F, N, 3) numpy float64 array of curl-free field vectors
    """
    timings = {}

    if log_path:
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"Curl Correction - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")

    # Load mesh
    log("=" * 60, log_path)
    log("LOADING MESH AND RAW FIELD", log_path)
    log("=" * 60, log_path)

    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Device: {device}", log_path)

    if V is not None and F is not None:
        log("\nBuilding mesh from in-memory V/F", log_path)
        mesh = TriMesh(V, F, device=device)
    else:
        log(f"\nLoading mesh: {mesh_path}", log_path)
        mesh = TriMesh.from_file(mesh_path, device=device)
    log(f"  Vertices: {mesh.V.shape[0]}, Faces: {mesh.F.shape[0]}", log_path)

    # Convert raw_vecs to torch tensor
    raw_vecs = torch.from_numpy(raw_vecs_np).to(dtype=torch.float64, device=device)
    log(f"  N-RoSy: N={N}, Faces: {raw_vecs.shape[0]}", log_path)

    timings['load'] = time.time() - t0

    # Curl correction
    log("\n" + "=" * 60, log_path)
    log(f"POLYVECTOR CURL CORRECTION ({num_iterations} iterations)", log_path)
    log("=" * 60, log_path)

    t0 = time.time()

    # Convert raw field to polyvector coefficients
    log("\nConverting raw field to polyvector coefficients...", log_path)
    FBx = mesh.FBx.unsqueeze(1)
    FBy = mesh.FBy.unsqueeze(1)
    u = (raw_vecs * FBx).sum(dim=2)
    v = (raw_vecs * FBy).sum(dim=2)
    roots = torch.complex(u, v)
    input_coeffs = roots_to_coeffs(roots)

    # Build sparse constraint format: constrain all faces with first vector
    num_faces = mesh.F.shape[0]
    if one_root:
        print("Using one root per face as constraint")
        constSpaces = torch.arange(num_faces, device=device, dtype=torch.int64)
        constVectors_complex = roots[:, 0]
    else:
        print("Using two orthogonal roots per face as constraints")
        with open("log_roots.txt", 'a', encoding='utf-8') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Using two roots per face as constraints\n")
        constSpaces = torch.repeat_interleave(
            torch.arange(num_faces, device=device, dtype=torch.int64), 2
        )
        constVectors_complex = roots[:, :2].reshape(-1)

    # Set up alignment weights
    if w_align_weights is not None:
        log("\nUsing per-face alignment weights", log_path)
        w_align_np = w_align_weights
        if len(w_align_np) != num_faces:
            raise ValueError(f"Weight array has {len(w_align_np)} entries but mesh has {num_faces} faces")
        log(f"  {len(w_align_np)} per-face weights, range: [{w_align_np.min():.4f}, {w_align_np.max():.4f}]", log_path)
        if one_root:
            wAlignment = torch.tensor(w_align_np, device=device, dtype=torch.float64)
        else:
            wAlignment = torch.tensor(np.repeat(w_align_np, 2), device=device, dtype=torch.float64)
    else:
        wAlignment = torch.full((num_faces,), w_align, device=device, dtype=torch.float64)
        if not one_root:
            wAlignment = wAlignment.repeat_interleave(2)
        if w_align != 1.0:
            log(f"\n  Using uniform alignment weight: {w_align}", log_path)

    # Create polyvector field and run optimization
    pv = PolyVectorField(mesh, N=N)

    if skip_initial_solve:
        iter_desc = f"{num_iterations} iterations" if num_iterations else "adaptive iterations"
        log(f"\nRunning curl correction ({iter_desc}, skip initial solve)...", log_path)
    else:
        iter_desc = f"{num_iterations} iterations" if num_iterations else "adaptive iterations"
        log(f"\nRunning polyvector_field optimization ({iter_desc})...", log_path)

    curlfree_coeffs = pv.solve_polyvector(
        input_coeffs,
        constSpaces,
        constVectors_complex,
        wAlignment,
        num_iterations=num_iterations,
        wSmooth=wSmooth,
        wRoSy=wRoSy,
        verbose=True,
        log_interval=log_interval,
        skip_initial_solve=skip_initial_solve,
        dynamic_annealing=dynamic_annealing,
        skip_first_implicit=skip_first_implicit,
        tau_threshold=tau_threshold,
        free_magnitude=free_magnitude,
        magnitude_clamp=magnitude_clamp,
    )

    timings['curl_correction'] = time.time() - t0
    log(f"\nCurl correction time: {timings['curl_correction']:.2f}s", log_path)

    # Convert back to raw field (extrinsic 3D vectors)
    curlfree_roots = extract_polyvector_roots(curlfree_coeffs, N, sort_roots=True)
    curlfree_vecs = curlfree_roots.real.unsqueeze(2) * mesh.FBx.unsqueeze(1) + \
                    curlfree_roots.imag.unsqueeze(2) * mesh.FBy.unsqueeze(1)

    curlfree_np = curlfree_vecs.cpu().numpy()

    # Summary
    total = sum(timings.values())
    log("\n" + "=" * 60, log_path)
    log("TIMING SUMMARY", log_path)
    log("=" * 60, log_path)
    for name, t in timings.items():
        log(f"  {name}: {t:.2f}s", log_path)
    log(f"  TOTAL: {total:.2f}s", log_path)

    return curlfree_np


# Drivers
def run_curl_correction(
    rep_vectors_3d: np.ndarray,
    face_normals: np.ndarray,
    input_mesh_path: str,
    w_align_weights: np.ndarray = None,
    polyvector: bool = False,
    N: int = 4,
    skip_initial_solve: bool = False,
    skip_first_implicit: bool = True,
    dynamic_annealing: bool = True,
    wSmooth: float = 1.0,
    wRoSy: float = 1.0,
    wAlign: float = 1.0,
    tau_threshold: float = 0.01,
    free_magnitude: bool = False,
    magnitude_clamp: float | None = None,
) -> np.ndarray:
    """Curl-correct an N-RoSy field.

    Args:
        rep_vectors_3d: (F, 3) representative directions, or (F, N, 3) when
            ``polyvector`` is set and the full field is already in hand.
        face_normals: (F, 3) face normals. Unused when ``polyvector`` is set.
        input_mesh_path: mesh the field lives on.
        w_align_weights: optional (F,) per-face alignment weights. Negative
            entries mark hard constraints.
        N: symmetry order. N = 4 takes the polyvector route, anything else the
            direct curl projection.
        free_magnitude: correct with the magnitude free, so curl can be absorbed
            as scale variation instead of rotation. N = 4 only.
        magnitude_clamp: bound the free magnitudes to [1/c, c].

    Returns:
        (F, N, 3) float64 curl-free field.
    """
    if polyvector:
        raw_vecs = rep_vectors_3d.astype(np.float64)
        N = raw_vecs.shape[1]
    else:
        raw_vecs = expand_rosy(rep_vectors_3d, face_normals, N).astype(np.float64)

    align_weights = (
        np.asarray(w_align_weights, dtype=np.float64)
        if w_align_weights is not None else None
    )

    print("\n=== Curl Correction (GPU) ===")
    start = time.time()

    if N == 4:
        curlfree_vecs = run_curl_correction_direct(
            mesh_path=os.path.abspath(input_mesh_path),
            raw_vecs_np=raw_vecs,
            N=N,
            w_align_weights=align_weights,
            w_align=wAlign,
            wSmooth=wSmooth,
            wRoSy=wRoSy,
            skip_initial_solve=skip_initial_solve,
            dynamic_annealing=dynamic_annealing,
            skip_first_implicit=skip_first_implicit,
            tau_threshold=tau_threshold,
            free_magnitude=free_magnitude,
            magnitude_clamp=magnitude_clamp,
        )
    else:
        if free_magnitude:
            raise NotImplementedError(
                "free_magnitude is implemented on the N=4 polyvector route only")
        curlfree_vecs = run_nrosy_curl_correction_direct(
            mesh_path=os.path.abspath(input_mesh_path),
            raw_vecs_np=raw_vecs,
            N=N,
            w_align_weights=align_weights,
            w_align=wAlign,
        )

    print(f"Curl correction took {time.time() - start:.2f}s")
    return curlfree_vecs


def run_curl_correction_per_component(
    rep_vectors_3d: np.ndarray,
    face_normals: np.ndarray,
    V: np.ndarray,
    F: np.ndarray,
    face_labels: np.ndarray,
    num_components: int,
    w_align_weights: np.ndarray = None,
    polyvector: bool = False,
    N: int = 4,
    skip_initial_solve: bool = False,
    skip_first_implicit: bool = True,
    dynamic_annealing: bool = True,
    wSmooth: float = 1.0,
    wRoSy: float = 1.0,
    wAlign: float = 1.0,
    tau_threshold: float = 0.01,
    free_magnitude: bool = False,
    magnitude_clamp: float | None = None,
    min_component_faces: int = 4,
) -> np.ndarray:
    """Curl-correct each connected component on its own system.

    Args:
        V, F: full input mesh arrays.
        face_labels: (nF,) component id per face, e.g. ``mesh.face_components``.
        min_component_faces: shells smaller than this pass through untouched.
        Everything else mirrors :func:`run_curl_correction`.

    Returns:
        (nF, N, 3) float64 curl-free field, reassembled.
    """
    if polyvector:
        raw_vecs = rep_vectors_3d.astype(np.float64)
        N = raw_vecs.shape[1]
    else:
        raw_vecs = expand_rosy(rep_vectors_3d, face_normals, N).astype(np.float64)

    if N != 4:
        raise NotImplementedError(
            "run_curl_correction_per_component supports N=4 only; "
            "use run_curl_correction for N != 4."
        )

    curlfree_vecs = np.zeros((F.shape[0], N, 3), dtype=np.float64)
    align_arr = (
        np.asarray(w_align_weights, dtype=np.float64)
        if w_align_weights is not None else None
    )

    print(f"\n=== Curl Correction per-component ({num_components} components) ===")
    start = time.time()

    for k in range(num_components):
        V_k, F_k, _, f_mask, sliced = slice_component(
            V, F, face_labels, k, face_arrays={"raw": raw_vecs, "w": align_arr},
        )
        comp_size = int(f_mask.sum())
        if comp_size < min_component_faces:
            print(f"  component {k}: {comp_size} faces — too small, "
                  "passing through unchanged")
            curlfree_vecs[f_mask] = sliced["raw"]
            continue

        print(f"\n--- component {k}: {V_k.shape[0]} V, {comp_size} F ---")
        curlfree_vecs[f_mask] = run_curl_correction_direct(
            mesh_path=None,
            raw_vecs_np=sliced["raw"],
            N=N,
            w_align_weights=sliced["w"],
            w_align=wAlign,
            wSmooth=wSmooth,
            wRoSy=wRoSy,
            skip_initial_solve=skip_initial_solve,
            dynamic_annealing=dynamic_annealing,
            skip_first_implicit=skip_first_implicit,
            tau_threshold=tau_threshold,
            free_magnitude=free_magnitude,
            magnitude_clamp=magnitude_clamp,
            V=V_k,
            F=F_k,
        )

    print(f"Curl correction (per-component) took {time.time() - start:.2f}s")
    return curlfree_vecs
