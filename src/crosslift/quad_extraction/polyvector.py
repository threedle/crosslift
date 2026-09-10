import math

import cupy as cp
import numpy as np
import torch
from cupyx.scipy.sparse import bmat as cp_bmat
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from cupyx.scipy.sparse import diags as cp_diags
from cupyx.scipy.sparse import vstack as cp_vstack
from cupyx.scipy.sparse.linalg import lsmr as cp_lsmr

from crosslift.quad_extraction.trimesh import TriMesh


def roots_to_coeffs(roots: torch.Tensor):
    """Monic polynomial coefficients from its roots.

    P(z) = (z - r1)...(z - rN) = z^N + c_{N-1} z^{N-1} + ... + c_0

    Args:
        roots: (B, N) complex.

    Returns:
        (B, N) complex, ordered [c_0, c_1, ..., c_{N-1}].
    """
    B, N = roots.shape
    device = roots.device

    coeffs = torch.zeros((B, 1), dtype=roots.dtype, device=device)
    coeffs[:, 0] = 1.0

    for i in range(N):
        r = roots[:, i].unsqueeze(1)
        # Multiply by (z - r): shift up for the z term, scale for the -r term.
        term1 = torch.cat(
            [coeffs, torch.zeros((B, 1), device=device, dtype=roots.dtype)], dim=1)
        term2 = torch.cat(
            [torch.zeros((B, 1), device=device, dtype=roots.dtype), coeffs * -r], dim=1)
        coeffs = term1 + term2

    # coeffs is [c_N, c_{N-1}, ..., c_0] with c_N == 1. Drop the leading 1 and
    # reverse to the [c_0, ...] order callers expect.
    return torch.flip(coeffs[:, 1:], dims=[1])


def roots_to_coeffs_sign_symmetric(roots: torch.Tensor, N: int = 4):
    """Coefficients under sign symmetry, where the odd ones vanish.

    A field closed under negation (v and -v both belong to it) has roots in +/-
    pairs, so P(z) is a polynomial in z^2. Building it from the squared roots of
    the first N/2 directions gives c1 = c3 = 0 exactly rather than approximately.

    Args:
        roots: (F, N) complex roots.
        N: field degree (4 here).

    Returns:
        (F, N) complex [c_0, 0, c_2, 0].
    """
    F = roots.shape[0]

    # w = z^2: (w - w0)(w - w1) = w^2 - (w0 + w1) w + w0 w1
    squared = roots[:, : N // 2] ** 2
    w0, w1 = squared[:, 0], squared[:, 1]

    coeffs = torch.zeros((F, N), dtype=roots.dtype, device=roots.device)
    coeffs[:, 0] = w0 * w1
    coeffs[:, 2] = -(w0 + w1)
    return coeffs


# Dynamic annealing constants
DYNAMIC_DECAY = 0.9          # Decay rate for dynamic annealing
LEGACY_DECAY = 0.8           # Original Directional decay rate
TAU_THRESHOLD = 0.01         # Stop when tau/tau_0 < this
COEFF_THRESHOLD = 1e-6       # Stop when max|dx| < this
MAX_ITERATIONS = 100         # Safety upper bound


def extract_polyvector_roots(x_coeffs: torch.Tensor, N: int, sort_roots: bool = True):
    """
    Extract exact roots for a PolyVector field of degree N.

    For signSymmetry (N even with c1=c3=0): solve the reduced polynomial
    w^(N/2) + c2*w^(N/2-1) + ... + c0 = 0, sort the reduced roots by angle,
    take the square root z = sqrt(w), and append the negatives
    [z0, z1, ..., -z0, -z1, ...].
    """
    num_faces = x_coeffs.shape[0]
    device = x_coeffs.device
    x_coeffs = x_coeffs.to(dtype=torch.complex128)

    # signSymmetry: N even and odd coefficients are zero
    sign_symmetry = N % 2 == 0
    if sign_symmetry:
        odd_coeff_norm = 0.0
        for k in range(1, N, 2):
            odd_coeff_norm += x_coeffs[:, k].abs().sum().item()
        sign_symmetry = sign_symmetry and (odd_coeff_norm < 1e-6)
    
    if sign_symmetry:
        # Durand-Kerner iteration, not eigenvalue decomposition
        actual_n = N // 2  # 2 for N=4
        
        # Reduced coefficients (c0, c2, c4, ...): for N=4, w^2 + c2*w + c0 = 0
        reduced_coeffs = x_coeffs[:, ::2]  # (F, actual_n) = (F, 2) for N=4

        # Initial guess: roots.col(0) = (-c0)^(1/actualN), then multiply by exp(2*pi*i/actualN)
        reduced_roots = torch.zeros((num_faces, actual_n), dtype=torch.complex128, device=device)
        reduced_roots[:, 0] = torch.pow(-reduced_coeffs[:, 0], 1.0 / actual_n)
        for i in range(1, actual_n):
            reduced_roots[:, i] = reduced_roots[:, i-1] * torch.exp(torch.tensor(2.0j * torch.pi / actual_n, device=device))
        
        max_iterations = 1000
        root_tolerance = 1e-8
        for iteration in range(max_iterations):
            max_error = 0.0
            for curr_root in range(actual_n):
                z = reduced_roots[:, curr_root]
                # Evaluate polynomial P(z) = z^n + c_{n-1}*z^{n-1} + ... + c_0
                poly_val = z**actual_n
                for k in range(actual_n):
                    poly_val = poly_val + reduced_coeffs[:, k] * (z ** k)
                
                # Compute denominator: product of (z_curr - z_j) for j != curr
                denom = torch.ones_like(z)
                for j in range(actual_n):
                    if j != curr_root:
                        denom = denom * (z - reduced_roots[:, j])
                
                # Update
                update = poly_val / (denom + 1e-20)
                reduced_roots[:, curr_root] = z - update
                
                max_error = max(max_error, update.abs().max().item())
            
            if max_error < root_tolerance:
                break
        
        sqrt_roots = torch.sqrt(reduced_roots)  # (F, actual_n)

        if sort_roots:
            angles = torch.angle(sqrt_roots)
            sort_idx = torch.argsort(angles, dim=1)
            sqrt_roots = torch.gather(sqrt_roots, 1, sort_idx)

        roots = torch.cat([sqrt_roots, -sqrt_roots], dim=1)  # (F, N)
    else:
        # Standard root extraction for non-signSymmetry
        companion = torch.zeros((num_faces, N, N), dtype=torch.complex128, device=device)
        rows = torch.arange(1, N, device=device)
        cols = torch.arange(0, N - 1, device=device)
        companion[:, rows, cols] = 1.0
        companion[..., N - 1] = -x_coeffs
        
        roots = torch.linalg.eigvals(companion)
        
        if sort_roots:
            angles = torch.angle(roots)
            sort_idx = torch.argsort(angles, dim=1)
            roots = torch.gather(roots, 1, sort_idx)
        
    return roots

class PolyVectorField:
    """Implements PolyVector field optimization on GPU with curl correction."""
    def __init__(self, mesh: TriMesh, N=4):
        self.mesh = mesh
        self.N = N
        self.device = mesh.device

        self.coeffs = torch.zeros((self.mesh.F.shape[0], N), dtype=torch.complex128, device=self.device)  # (F, N) complex
        # Intrinsic field vectors (F, N) complex, in the order rawfield loading sets externally
        self._intField = None
        self.A_ls = None
        self.sqrt_Mass = None
        self.D_smooth = None
        self.C_curl = None # Discrete curl operator
    
    @property
    def intField(self):
        return self._intField
    
    @intField.setter
    def intField(self, value):
        self._intField = value
        
    def compute_connections(self):
        mesh = self.mesh
        he_inner_mask = (mesh.topology.twin_he != -1)
        he_indices = torch.nonzero(he_inner_mask).squeeze()
        
        twins = mesh.topology.twin_he
        unique_edge_mask = (he_indices < twins[he_indices])
        valid_he = he_indices[unique_edge_mask]
        
        f0 = mesh.topology.face[valid_he]
        f1 = mesh.topology.face[twins[valid_he]]
        
        v0 = mesh.topology.he_origin[valid_he]
        v1 = mesh.topology.he_origin[mesh.topology.next_he[valid_he]]
        
        vec_edge = mesh.V[v1] - mesh.V[v0]
        len_edge = torch.linalg.norm(vec_edge, dim=1, keepdim=True)
        vec_edge = vec_edge / (len_edge + 1e-12)
        
        comp_f0 = torch.complex(
            (vec_edge * mesh.FBx[f0]).sum(dim=1),
            (vec_edge * mesh.FBy[f0]).sum(dim=1)
        )
        
        comp_f1 = torch.complex(
            (vec_edge * mesh.FBx[f1]).sum(dim=1),
            (vec_edge * mesh.FBy[f1]).sum(dim=1)
        )
        
        connections = comp_f1 / comp_f0 # (E_inner,)
        
        return valid_he, twins[valid_he], f0, f1, connections

    def precompute(self, wSmooth=1.0, wRoSy=1.0, wAlignment=1.0, constSpaces=None, constVectors_complex=None, wAlignment_array=None):
        """
        Precompute matrices for polyvector solve.

        Args:
            wSmooth: Smoothness weight (scalar)
            wRoSy: RoSy weight (scalar)
            wAlignment: Alignment weight - can be scalar or (F,) array for per-face weights
            constSpaces: Optional (K,) int array of constraint face indices
            constVectors_complex: Optional (K,) complex int array of constraint directions
            wAlignment_array: Optional (K,) float array of per-constraint weights
        """
        he_idx, _twin_idx, f0, f1, rij = self.compute_connections()
        num_faces = self.mesh.F.shape[0]
        num_inner = he_idx.shape[0]
        
        v0 = self.mesh.topology.he_origin[he_idx]
        v1 = self.mesh.topology.he_origin[self.mesh.topology.next_he[he_idx]]
        len_sq = ((self.mesh.V[v1] - self.mesh.V[v0])**2).sum(dim=1)
        area_sum = self.mesh.faceAreas[f0] + self.mesh.faceAreas[f1]
        area_sum[area_sum < 1e-12] = 1.0
        
        # Brandt 2020 harmonic weights for connectionMass
        edge_weight = 3.0 * len_sq / area_sum
        edge_weight = edge_weight.to(torch.complex128)

        # Normalization weights
        self.total_smooth_weight = (edge_weight.abs().sum()).item()
        if self.total_smooth_weight < 1e-8: self.total_smooth_weight = 1.0
        
        self.total_mass = (self.mesh.faceAreas.sum()).item()
        if self.total_mass < 1e-8: self.total_mass = 1.0

        self.total_rosy_weight = float(self.N - 1) * self.total_mass

        sign_symmetry = (self.N % 2 == 0)
        real_n = self.N // 2 if sign_symmetry else self.N
        self.total_constrained_weight = float(real_n) * self.total_mass

        # Smoothness matrix D: D_val = sqrt(wSmooth * edge_weight)
        sqrt_w_smooth = math.sqrt(wSmooth)
        
        all_rows, all_cols, all_vals = [], [], []
        
        for n in range(self.N):
            power = self.N - n
            conn_pow = torch.pow(rij, power)
            row_indices = torch.arange(num_inner, device=self.device) + (n * num_inner)
            sqrt_w = torch.sqrt(edge_weight) * sqrt_w_smooth
            
            all_rows.append(row_indices)
            all_cols.append(f1 + n * num_faces)
            all_vals.append(torch.ones_like(conn_pow) * -1.0 * sqrt_w)
            
            all_rows.append(row_indices)
            all_cols.append(f0 + n * num_faces)
            all_vals.append(conn_pow * sqrt_w) 
            
        # Zero-copy GPU transfer: PyTorch -> CuPy via dlpack
        rows_cp = cp.from_dlpack(torch.cat(all_rows).contiguous())
        cols_cp = cp.from_dlpack(torch.cat(all_cols).contiguous())
        vals_cp = cp.from_dlpack(torch.cat(all_vals).contiguous())
        
        self.D_smooth = cp_csr_matrix((vals_cp, (rows_cp, cols_cp)), 
                                      shape=(self.N * num_inner, self.N * num_faces))

        # RoSy matrix: identity on non-c0 terms, values sqrt(wRoSy * Area)
        sqrt_w_rosy = math.sqrt(wRoSy)
        face_areas_cp = cp.from_dlpack(self.mesh.faceAreas.contiguous())
        sqrt_area = cp.sqrt(face_areas_cp)
        
        rosy_rows, rosy_cols, rosy_vals = [], [], []
        for n in range(1, self.N):
             r_indices = cp.arange(num_faces) + (n * num_faces)
             rosy_rows.append(r_indices)
             rosy_cols.append(r_indices)
             rosy_vals.append(sqrt_area * sqrt_w_rosy)
             
        if len(rosy_rows) > 0:
            r_rows = cp.concatenate(rosy_rows)
            r_cols = cp.concatenate(rosy_cols)
            r_vals = cp.concatenate(rosy_vals)
            self.A_rosy = cp_csr_matrix((r_vals, (r_rows, r_cols)),
                                        shape=(self.N * num_faces, self.N * num_faces))
        else:
            self.A_rosy = cp_csr_matrix((self.N * num_faces, self.N * num_faces))

        # Alignment matrix (IAiA projector): (realN, realN) per face, realN = N/2 under signSymmetry
        sign_symmetry = (self.N % 2 == 0)
        real_n = self.N // 2 if sign_symmetry else self.N
        jump = 2 if sign_symmetry else 1

        self.real_n = real_n
        self.jump = jump
        self.sqrt_area = sqrt_area

        # wAlignment can be scalar or (F,) array
        if isinstance(wAlignment, (int, float)):
            self.wAlignment_perface = cp.full(num_faces, float(wAlignment), dtype=cp.float64)
        elif hasattr(wAlignment, '__len__'):
            if hasattr(wAlignment, 'cpu'):  # PyTorch tensor
                self.wAlignment_perface = cp.asarray(wAlignment.cpu().numpy())
            elif isinstance(wAlignment, cp.ndarray):
                self.wAlignment_perface = wAlignment
            else:  # numpy array or list
                self.wAlignment_perface = cp.asarray(wAlignment, dtype=cp.float64)
        else:
            raise ValueError(f"wAlignment must be scalar or array, got {type(wAlignment)}")

        self.wAlignment_val = float(cp.mean(self.wAlignment_perface))

        # Reduction matrix (reducMat) and RHS (reducRhs): tracks constraints applied per face
        num_space_constraints = cp.zeros(num_faces, dtype=cp.int32)

        # (realN, realN) identity and (realN,) zero per face
        local_space_reduc_mats = [cp.eye(real_n, dtype=cp.complex128) for _ in range(num_faces)]
        local_space_reduc_rhs = [cp.zeros(real_n, dtype=cp.complex128) for _ in range(num_faces)]

        # Hard constraints
        if constSpaces is not None and wAlignment_array is not None and constVectors_complex is not None:
            if hasattr(constSpaces, 'cpu'): c_spaces = cp.asarray(constSpaces.cpu().numpy())
            else: c_spaces = cp.asarray(constSpaces)
            
            if hasattr(constVectors_complex, 'cpu'): c_vecs = cp.asarray(constVectors_complex.cpu().numpy())
            else: c_vecs = cp.asarray(constVectors_complex)
                
            if hasattr(wAlignment_array, 'cpu'): c_weights = cp.asarray(wAlignment_array.cpu().numpy())
            else: c_weights = cp.asarray(wAlignment_array)
            
            for i in range(c_spaces.shape[0]):
                if c_weights[i] >= 0.0:
                    continue  # only hard constraints (w < 0.0)
                    
                face_idx = int(c_spaces[i])
                if num_space_constraints[face_idx] == real_n:
                    continue  # overconstrained, ignore further constraints
                
                curr_space_num_dof = int(num_space_constraints[face_idx])
                
                const_vec_complex = c_vecs[i]
                if self.N % 2 == 0:  # signSymmetry
                    const_vec_complex = const_vec_complex * const_vec_complex

                # Single-step reduction matrix eliminating the last DOF for this face
                rows_single = real_n - curr_space_num_dof
                cols_single = real_n - curr_space_num_dof - 1
                
                single_reduc_mat = cp.zeros((rows_single, cols_single), dtype=cp.complex128)
                single_reduc_rhs = cp.zeros(rows_single, dtype=cp.complex128)
                
                for j in range(cols_single):
                    single_reduc_mat[j, j] = -const_vec_complex
                    single_reduc_mat[j+1, j] = 1.0 + 0j
                single_reduc_rhs[rows_single - 1] = -const_vec_complex
                
                local_space_reduc_rhs[face_idx] = local_space_reduc_mats[face_idx] @ single_reduc_rhs + local_space_reduc_rhs[face_idx]
                local_space_reduc_mats[face_idx] = local_space_reduc_mats[face_idx] @ single_reduc_mat
                
                num_space_constraints[face_idx] += 1

        # Global reduction matrices (reducMat, reducRhs)
        reduc_rows = []
        reduc_cols = []
        reduc_vals = []
        full_dofs = self.N * num_faces
        
        reduc_rhs = cp.zeros(full_dofs, dtype=cp.complex128)
        col_counter = 0
        
        for i in range(num_faces):
            local_mat = local_space_reduc_mats[i]
            local_rhs = local_space_reduc_rhs[i]
            
            for j in range(0, self.N, jump):
                for k in range(local_mat.shape[1]):
                    reduc_rows.append(j * num_faces + i)
                    reduc_cols.append(col_counter + k)
                    reduc_vals.append(local_mat[j // jump, k])
                
                reduc_rhs[j * num_faces + i] = local_rhs[j // jump]
                
            col_counter += local_mat.shape[1]
            
        reduced_dofs = col_counter
        
        if len(reduc_rows) > 0:
            reduc_rows = cp.asarray(reduc_rows, dtype=cp.int32)
            reduc_cols = cp.asarray(reduc_cols, dtype=cp.int32)
            reduc_vals = cp.asarray(reduc_vals, dtype=cp.complex128)
            
            self.reducMat = cp_csr_matrix((reduc_vals, (reduc_rows, reduc_cols)),
                                           shape=(full_dofs, reduced_dofs))
        else:
            self.reducMat = cp_csr_matrix((full_dofs, reduced_dofs), dtype=cp.complex128)
            
        self.reducRhs = reduc_rhs
        self.reduced_dofs = reduced_dofs

        # Mass matrix (diagonal), used for the inertial term
        mass_vals = cp.tile(sqrt_area, self.N)
        self.sqrt_Mass = cp_diags(mass_vals)
        self.M_diag = mass_vals**2

        # CPU precompute for reduced matrices: sparse-sparse products here trigger
        # CuPy spGEMM, which can OOM on large meshes; CPU cost is negligible.
        import scipy.sparse as sp_cpu
        
        D_smooth_cpu = sp_cpu.csr_matrix((cp.asnumpy(self.D_smooth.data), 
                                          cp.asnumpy(self.D_smooth.indices), 
                                          cp.asnumpy(self.D_smooth.indptr)),
                                         shape=self.D_smooth.shape)
        A_rosy_cpu = sp_cpu.csr_matrix((cp.asnumpy(self.A_rosy.data),
                                        cp.asnumpy(self.A_rosy.indices),
                                        cp.asnumpy(self.A_rosy.indptr)),
                                       shape=self.A_rosy.shape)
        reducMat_cpu = sp_cpu.csr_matrix((cp.asnumpy(self.reducMat.data),
                                          cp.asnumpy(self.reducMat.indices),
                                          cp.asnumpy(self.reducMat.indptr)),
                                         shape=self.reducMat.shape)
        sqrt_Mass_cpu = sp_cpu.diags(cp.asnumpy(mass_vals))
        
        D_smooth_red_cpu = D_smooth_cpu @ reducMat_cpu
        A_rosy_red_cpu = A_rosy_cpu @ reducMat_cpu
        sqrt_Mass_red_cpu = sqrt_Mass_cpu @ reducMat_cpu
        
        self.D_smooth_red = cp_csr_matrix(D_smooth_red_cpu.astype(np.complex128))
        self.A_rosy_red = cp_csr_matrix(A_rosy_red_cpu.astype(np.complex128))
        self.sqrt_Mass_red = cp_csr_matrix(sqrt_Mass_red_cpu.astype(np.complex128))

        self.wSmooth = wSmooth
        self.wRoSy = wRoSy
        self.wAlignment = wAlignment
    
    def build_alignment_system(self, constSpaces, constVectors_complex, wAlignment):
        """
        Build IAiA projector alignment matrix and RHS from SPARSE constraint lists.
        
        Args:
            constSpaces: (K,) int tensor of face indices (can have duplicates for multiple constraints)
            constVectors_complex: (K,) complex tensor of constraint directions (2D intrinsic)
            wAlignment: (K,) float tensor of per-constraint weights
            
        Returns:
            A_align: (realN*K, N*F) sparse matrix
            b_align_rhs: (realN*K,) vector
            total_constrained_weight: float, for energy normalization
        """
        num_faces = self.mesh.F.shape[0]
        K = constSpaces.shape[0]
        real_n = self.real_n
        jump = self.jump
        
        if K == 0:
            # No constraints
            A_align = cp_csr_matrix((0, self.N * num_faces), dtype=cp.complex128)
            b_align_rhs = cp.zeros(0, dtype=cp.complex128)
            return A_align, b_align_rhs, 1.0
        
        # Convert to CuPy
        if constSpaces.is_cuda:
            face_idx = cp.asarray(constSpaces.contiguous())
        else:
            face_idx = cp.asarray(constSpaces.cpu().numpy())
        
        if constVectors_complex.is_cuda:
            z = cp.asarray(constVectors_complex.contiguous())
        else:
            z = cp.asarray(constVectors_complex.cpu().numpy())
            
        if wAlignment.is_cuda:
            w_per_constraint = cp.asarray(wAlignment.contiguous())
        else:
            w_per_constraint = cp.asarray(wAlignment.cpu().numpy())
            
        # Filter out hard constraints (w < 0.0), they are handled by reducMat
        soft_mask = w_per_constraint >= 0.0
        face_idx = face_idx[soft_mask]
        z = z[soft_mask]
        w_per_constraint = w_per_constraint[soft_mask]
        K = int(cp.sum(soft_mask))
        
        if K == 0:
            # No soft constraints
            A_align = cp_csr_matrix((0, self.N * num_faces), dtype=cp.complex128)
            b_align_rhs = cp.zeros(0, dtype=cp.complex128)
            return A_align, b_align_rhs, 1.0
        
        # For signSymmetry, we use z^2 as the constraint
        if self.N % 2 == 0:  # signSymmetry
            z = z * z  # z^2
        
        # Each constraint adds realN * w * Area
        face_areas = self.sqrt_area[face_idx] ** 2  # (K,) actual areas
        total_constrained_weight = float(cp.sum(w_per_constraint * face_areas) * real_n)
        if total_constrained_weight < 1e-8:
            total_constrained_weight = 1.0
        
        if real_n == 2:
            # Build IAiA projector (same math, but for K constraints instead of F faces)
            z_conj = cp.conj(z)
            norm_sq = cp.abs(z)**2 + 1.0  # (K,)
            
            iaia_00 = 1.0 / norm_sq
            iaia_01 = z / norm_sq
            iaia_10 = z_conj / norm_sq
            iaia_11 = (cp.abs(z)**2) / norm_sq
            
            rhs_0 = -z * z / norm_sq
            rhs_1 = -z * (cp.abs(z)**2) / norm_sq
            
            # Weight by sqrt(wAlignment * Area)
            sqrt_w = cp.sqrt(w_per_constraint) * self.sqrt_area[face_idx]  # (K,)
            
            # Build sparse matrix (realN*K, N*F)
            rows = []
            cols = []
            vals = []
            
            constraint_idx = cp.arange(K)
            
            # Row layout: [constraint 0 block 0, constraint 1 block 0, ..., constraint 0 block 1, ...]
            # Block (0,0): rows 0..K-1, cols = face_idx in c0 block
            rows.append(0 * K + constraint_idx)
            cols.append(0 * jump * num_faces + face_idx)
            vals.append(iaia_00 * sqrt_w)
            
            # Block (0,1): rows 0..K-1, cols = face_idx in c2 block
            rows.append(0 * K + constraint_idx)
            cols.append(1 * jump * num_faces + face_idx)
            vals.append(iaia_01 * sqrt_w)
            
            # Block (1,0): rows K..2K-1, cols = face_idx in c0 block
            rows.append(1 * K + constraint_idx)
            cols.append(0 * jump * num_faces + face_idx)
            vals.append(iaia_10 * sqrt_w)
            
            # Block (1,1): rows K..2K-1, cols = face_idx in c2 block
            rows.append(1 * K + constraint_idx)
            cols.append(1 * jump * num_faces + face_idx)
            vals.append(iaia_11 * sqrt_w)
            
            rows = cp.concatenate(rows).astype(cp.int32)
            cols = cp.concatenate(cols).astype(cp.int32)
            vals = cp.concatenate(vals)
            
            A_align = cp_csr_matrix((vals, (rows, cols)), 
                                    shape=(real_n * K, self.N * num_faces))
            
            # RHS vector (realN*K,)
            b_align_rhs = cp.zeros(real_n * K, dtype=cp.complex128)
            b_align_rhs[:K] = rhs_0 * sqrt_w
            b_align_rhs[K:] = rhs_1 * sqrt_w
            
        else:
            raise NotImplementedError(f"IAiA projector not implemented for realN={real_n}")
        
        return A_align, b_align_rhs, total_constrained_weight

    def compute_energies(self, coeffs_tensor):
        """Compute smoothness/RoSy/alignment energies. Input: coeffs_tensor (F, N) or flattened."""
        num_faces = self.mesh.F.shape[0]
        # Flatten blocked [c0... | c1... | ...] to match matrix column order
        if coeffs_tensor.shape == (num_faces, self.N):
             _coeffs_flat = coeffs_tensor.T.reshape(-1).contiguous()
             x_vec = cp.asarray(_coeffs_flat)
        else:
             _coeffs_flat = coeffs_tensor.flatten().contiguous()
             x_vec = cp.asarray(_coeffs_flat)

        # E = |D_weighted x|^2 / TotalSmooth; D_smooth already includes sqrt(wSmooth)
        norm_D = cp.linalg.norm(self.D_smooth @ x_vec)**2
        E_smooth = norm_D / self.total_smooth_weight

        # E = |A_rosy x|^2 / TotalRoSy
        norm_R = cp.linalg.norm(self.A_rosy @ x_vec)**2
        E_rosy = norm_R / self.total_rosy_weight

        return E_smooth, E_rosy, 0.0 # alignment energy needs x_input; not tracked here

    def compute_optimality_residual(self, constSpaces, constVectors_complex, wAlignment):
        """
        Optimality residual max|totalLhs * x_reduced - totalRhs|: the primary
        convergence metric, distance from stationarity of the weighted energy.

        Args:
            constSpaces: (K,) int tensor of constraint face indices
            constVectors_complex: (K,) complex tensor of constraint directions
            wAlignment: (K,) float tensor of per-constraint weights
            
        Returns:
            float: Max absolute value of the optimality residual
        """
        # Project current coefficients to reduced DOF space:
        # x_reduced = argmin || M^(1/2) (reducMat * x + reducRhs - x_full) ||^2
        _coeffs_flat = self.coeffs.T.reshape(-1).contiguous()
        x_full = cp.asarray(_coeffs_flat)
        b_proj = self.sqrt_Mass @ (x_full - self.reducRhs)
        
        # Block real/imag formulation for complex LSMR
        A_proj_real = cp_csr_matrix((self.sqrt_Mass_red.data.real.astype(cp.float64), 
                                     self.sqrt_Mass_red.indices, 
                                     self.sqrt_Mass_red.indptr), shape=self.sqrt_Mass_red.shape)
        A_proj_imag = cp_csr_matrix((self.sqrt_Mass_red.data.imag.astype(cp.float64), 
                                     self.sqrt_Mass_red.indices, 
                                     self.sqrt_Mass_red.indptr), shape=self.sqrt_Mass_red.shape)
        A_proj_block = cp_bmat([[A_proj_real, -A_proj_imag], [A_proj_imag, A_proj_real]], format='csr')
        b_proj_block = cp.hstack([b_proj.real.astype(cp.float64), b_proj.imag.astype(cp.float64)])
        
        x_proj_block = cp_lsmr(A_proj_block, b_proj_block, atol=1e-10, btol=1e-10)[0]
        x_reduced = (x_proj_block[:self.reduced_dofs] + 1j * x_proj_block[self.reduced_dofs:]).astype(cp.complex128)
        
        A_align, b_align_rhs, total_constrained_weight = self.build_alignment_system(
            constSpaces, constVectors_complex, wAlignment)

        # Avoid explicitly forming totalLhs = reducMat.H @ (D'D + ...) @ reducMat;
        # compute y = totalLhs * x_reduced via sequential SpMVs instead.
        x_full = self.reducMat @ x_reduced

        Ds_x = self.D_smooth @ x_full
        Ds_enc_x = self.D_smooth.T.conj() @ Ds_x
        term_D = Ds_enc_x / self.total_smooth_weight

        Rs_x = self.A_rosy @ x_full
        Rs_enc_x = self.A_rosy.T.conj() @ Rs_x
        term_R = Rs_enc_x / self.total_rosy_weight

        As_x = A_align @ x_full
        As_enc_x = A_align.T.conj() @ As_x
        term_A = As_enc_x / total_constrained_weight

        totalUnreducedLhs_x = term_D + term_R + term_A
        Lhs_x = self.reducMat.T.conj() @ totalUnreducedLhs_x

        # reducRhs is zero, so the totalLhs @ reducRhs term vanishes
        align_rhs_unreduced = (A_align.T.conj() @ b_align_rhs) / total_constrained_weight
        totalRhs = self.reducMat.T.conj() @ align_rhs_unreduced

        residual = Lhs_x - totalRhs
        return float(cp.abs(residual).max())

    def compute_curl_magnitude(self):
        """
        Compute the curl magnitude of the current field: ||C * x||.
        
        This measures how far the field is from being curl-free.
        Used for logging/diagnostics, not for stopping criterion.
        
        Returns:
            Tuple of (curl_l2: float, curl_max: float)
        """
        mesh = self.mesh
        N = self.N
        
        # Extract roots
        roots = extract_polyvector_roots(self.coeffs, N, sort_roots=True)
        
        # Flatten to 2D intrinsic coordinates
        x = torch.stack([roots.real, roots.imag], dim=-1).reshape(-1)  # (2*N*F,)
        
        # Get matching and connections
        he_inner, matchings = self.principal_matching()
        num_inner_edges = he_inner.shape[0]
        
        if num_inner_edges == 0:
            return 0.0, 0.0
        
        # Get face indices for each inner edge
        twin_he = mesh.topology.twin_he
        face_he = mesh.topology.face
        left_faces = face_he[he_inner]
        right_faces = face_he[twin_he[he_inner]]
        
        # Get edge vertices
        tri_idx = he_inner % 3
        face_from_he = he_inner // 3
        v0_idx = mesh.F[face_from_he, tri_idx]
        v1_idx = mesh.F[face_from_he, (tri_idx + 1) % 3]
        
        # Edge vectors
        edge_vecs = mesh.V[v1_idx] - mesh.V[v0_idx]
        
        # Project edges onto face bases
        ein_left_x = (edge_vecs * mesh.FBx[left_faces]).sum(dim=1)
        ein_left_y = (edge_vecs * mesh.FBy[left_faces]).sum(dim=1)
        ein_right_x = (edge_vecs * mesh.FBx[right_faces]).sum(dim=1)
        ein_right_y = (edge_vecs * mesh.FBy[right_faces]).sum(dim=1)
        
        # Compute curl for each edge and each vector
        # curl = edge · (right_vec[matched] - left_vec)
        curl_values = []
        for n in range(N):
            n_matched = (n + matchings + N) % N
            
            # Left face vector n
            left_u = x[2*N*left_faces + 2*n]
            left_v = x[2*N*left_faces + 2*n + 1]
            
            # Right face vector (matched index)
            right_u = x[2*N*right_faces + 2*n_matched]
            right_v = x[2*N*right_faces + 2*n_matched + 1]
            
            # Curl contribution: edge · (right - left)
            curl_n = ein_right_x * right_u + ein_right_y * right_v - \
                     ein_left_x * left_u - ein_left_y * left_v
            curl_values.append(curl_n)
        
        curl_all = torch.stack(curl_values, dim=1)  # (E, N)
        curl_l2 = float(torch.sqrt((curl_all**2).sum()))
        curl_max = float(torch.abs(curl_all).max())
        
        return curl_l2, curl_max

    def soft_rosy(self, factor=1.0, free_magnitude=False):
        """
        Local step: enforces RoSy (rotational symmetry).
        Interpolates: new = (old + 2*factor*rosy) / (1 + 2*factor)

        Args:
            factor: interpolation strength.
            free_magnitude: leave c0 at its current modulus, projecting onto
                crosses of any size rather than unit ones.
        """
        coeffs_curr = self.coeffs  # (F, N)
        
        # RoSy target: zeros for c1,c2,c3, normalized c0
        rosy_coeffs = torch.zeros_like(coeffs_curr)  # (F, N)
        c0 = coeffs_curr[:, 0]  # (F,)
        if free_magnitude:
            rosy_coeffs[:, 0] = c0
        else:
            c0_mag = torch.abs(c0)
            valid = c0_mag > 1e-12
            rosy_coeffs[valid, 0] = c0[valid] / c0_mag[valid]
            rosy_coeffs[~valid, 0] = c0[~valid]

        alpha = 2.0 * factor
        denom = 1.0 + alpha
        new_coeffs = (coeffs_curr + alpha * rosy_coeffs) / denom
        
        self.coeffs = new_coeffs

    def implicit_step(self, constSpaces, constVectors_complex, wAlignment, curr_implicit_factor):
        """
        Global implicit step. Solves in reduced DOF space (realN*F variables
        instead of N*F).

        Args:
            constSpaces: (K,) int tensor of face indices
            constVectors_complex: (K,) complex tensor of constraint directions
            wAlignment: (K,) float tensor of per-constraint weights
            curr_implicit_factor: current implicit factor for the iteration
        """
        num_faces = self.mesh.F.shape[0]
        reduced_dofs = self.reduced_dofs

        # Project current full DOFs to reduced space (zero-copy)
        _coeffs_flat = self.coeffs.T.reshape(-1).contiguous()
        x_full = cp.asarray(_coeffs_flat)
        b_proj = self.sqrt_Mass @ (x_full - self.reducRhs)
        
        # Block real/imag formulation for complex LSMR
        A_proj_real = cp_csr_matrix((self.sqrt_Mass_red.data.real.astype(cp.float64), 
                                     self.sqrt_Mass_red.indices, 
                                     self.sqrt_Mass_red.indptr), shape=self.sqrt_Mass_red.shape)
        A_proj_imag = cp_csr_matrix((self.sqrt_Mass_red.data.imag.astype(cp.float64), 
                                     self.sqrt_Mass_red.indices, 
                                     self.sqrt_Mass_red.indptr), shape=self.sqrt_Mass_red.shape)
        A_proj_block = cp_bmat([[A_proj_real, -A_proj_imag], [A_proj_imag, A_proj_real]], format='csr')
        b_proj_block = cp.hstack([b_proj.real.astype(cp.float64), b_proj.imag.astype(cp.float64)])
        
        x_proj_block = cp_lsmr(A_proj_block, b_proj_block, atol=1e-10, btol=1e-10)[0]
        x_reduced = (x_proj_block[:reduced_dofs] + 1j * x_proj_block[reduced_dofs:]).astype(cp.complex128)
        
        # Build system matrices in reduced space: A_red = A @ reducMat
        A_align, b_align_rhs, total_constrained_weight = self.build_alignment_system(
            constSpaces, constVectors_complex, wAlignment)
        
        # Scale factors 
        scale_s = math.sqrt(curr_implicit_factor / self.total_smooth_weight)
        scale_r = math.sqrt(curr_implicit_factor / self.total_rosy_weight) if self.total_rosy_weight > 0 else 0
        scale_a = math.sqrt(curr_implicit_factor / total_constrained_weight)
        scale_m = math.sqrt(1.0 / self.total_mass)
        
        # Use PRECOMPUTED reduced matrices (avoids GPU spGEMM)
        D_scaled_red = self.D_smooth_red * scale_s
        R_scaled_red = self.A_rosy_red * scale_r
        A_align_scaled_red = (A_align @ self.reducMat) * scale_a  # A_align changes each iter
        
        # Alignment RHS (already in reduced-aligned form, just scale)
        b_align_scaled = b_align_rhs * scale_a
        
        # Use precomputed reduced mass matrix
        M_scaled_red = self.sqrt_Mass_red * scale_m
        b_M_scaled_red = M_scaled_red @ x_reduced
        
        # Stack for solver in reduced space
        A_stack = cp_vstack([D_scaled_red, R_scaled_red, A_align_scaled_red, M_scaled_red])
        b_stack = cp.hstack([cp.zeros(D_scaled_red.shape[0], dtype=cp.complex128),
                             cp.zeros(R_scaled_red.shape[0], dtype=cp.complex128),
                             b_align_scaled,
                             b_M_scaled_red])
        
        A_complex = A_stack.tocsr()
        
        # Warm start from previous iteration
        x0_block = None
        if (hasattr(self, '_prev_x_reduced') and self._prev_x_reduced is not None
                and self._prev_x_reduced.shape[0] == reduced_dofs):
            x0_real = self._prev_x_reduced.real.astype(cp.float64)
            x0_imag = self._prev_x_reduced.imag.astype(cp.float64)
            x0_block = cp.hstack([x0_real, x0_imag])
        
        # Solve using LSMR (block real/imag formulation for complex system)
        A_real = cp_csr_matrix((A_complex.data.real.astype(cp.float64), A_complex.indices, A_complex.indptr), shape=A_complex.shape)
        A_imag = cp_csr_matrix((A_complex.data.imag.astype(cp.float64), A_complex.indices, A_complex.indptr), shape=A_complex.shape)
        A_block = cp_bmat([[A_real, -A_imag], [A_imag, A_real]], format='csr')
        b_block = cp.hstack([b_stack.real.astype(cp.float64), b_stack.imag.astype(cp.float64)])
        
        x_res_block = cp_lsmr(A_block, b_block, x0=x0_block, atol=1e-10, btol=1e-10)[0]
        x_reduced_new = (x_res_block[:reduced_dofs] + 1j * x_res_block[reduced_dofs:]).astype(cp.complex128)
        
        # Cache for next iteration's warm start
        self._prev_x_reduced = x_reduced_new
        
        # Expand back to full DOFs: x_full = reducMat @ x_reduced + reducRhs
        x_full_new = self.reducMat @ x_reduced_new + self.reducRhs
        
        # Update coeffs - ZERO-COPY via torch.as_tensor
        self.coeffs = torch.as_tensor(x_full_new, device=self.device).reshape(self.N, num_faces).T.contiguous().to(torch.complex128)

    def compute_implicit_factor_from_coeffs(self, coeffs, constSpaces, constVectors_complex, wAlignment):
        """
        Compute initial implicit factor (energy/mass ratio) from existing
        coefficients without running initial_solve.

        Args:
            coeffs: (F, N) complex tensor of polyvector coefficients
            constSpaces: (K,) int tensor of constraint face indices
            constVectors_complex: (K,) complex tensor of constraint directions
            wAlignment: (K,) float tensor of per-constraint weights
            
        Returns:
            curr_implicit_factor: float
        """
        # Project coeffs to reduced DOF space
        _coeffs_flat = coeffs.T.reshape(-1).contiguous()
        x_full = cp.asarray(_coeffs_flat)
        b_proj = self.sqrt_Mass @ (x_full - self.reducRhs)
        
        # Block real/imag formulation for complex LSMR
        A_proj_real = cp_csr_matrix((self.sqrt_Mass_red.data.real.astype(cp.float64), 
                                     self.sqrt_Mass_red.indices, 
                                     self.sqrt_Mass_red.indptr), shape=self.sqrt_Mass_red.shape)
        A_proj_imag = cp_csr_matrix((self.sqrt_Mass_red.data.imag.astype(cp.float64), 
                                     self.sqrt_Mass_red.indices, 
                                     self.sqrt_Mass_red.indptr), shape=self.sqrt_Mass_red.shape)
        A_proj_block = cp_bmat([[A_proj_real, -A_proj_imag], [A_proj_imag, A_proj_real]], format='csr')
        b_proj_block = cp.hstack([b_proj.real.astype(cp.float64), b_proj.imag.astype(cp.float64)])
        
        x_proj_block = cp_lsmr(A_proj_block, b_proj_block, atol=1e-10, btol=1e-10)[0]
        x_reduced = (x_proj_block[:self.reduced_dofs] + 1j * x_proj_block[self.reduced_dofs:]).astype(cp.complex128)
        
        A_align, b_align_rhs, total_constrained_weight = self.build_alignment_system(
            constSpaces, constVectors_complex, wAlignment)

        # Avoid explicit totalLhs construction; use squared norms instead.
        # energy = 0.5 * x^H * totalLhs * x - x^H * totalRhs
        scale_s = math.sqrt(1.0 / self.total_smooth_weight)
        scale_r = math.sqrt(1.0 / self.total_rosy_weight) if self.total_rosy_weight > 0 else 0
        scale_a = math.sqrt(1.0 / total_constrained_weight)

        # totalRhs = reducMat' * ((A' * b) / totAlign)
        align_rhs_unreduced = (A_align.T.conj() @ b_align_rhs) / total_constrained_weight
        totalRhs = self.reducMat.T.conj() @ align_rhs_unreduced

        # x' L x = ||D_red x||^2 + ||R_red x||^2 + ||A_red x||^2
        norm_D = cp.linalg.norm(self.D_smooth_red @ x_reduced * scale_s)**2
        norm_R = cp.linalg.norm(self.A_rosy_red @ x_reduced * scale_r)**2
        A_align_red = (A_align @ self.reducMat) * scale_a
        norm_A = cp.linalg.norm(A_align_red @ x_reduced)**2
        
        term_quadratic = norm_D + norm_R + norm_A

        x_H = x_reduced.conj()
        energy = 0.5 * term_quadratic - (x_H @ totalRhs)
        
        # mass = x^H * (reducMat^H * M * reducMat) * x / totalMass
        sqrt_M_red_x = self.sqrt_Mass_red @ x_reduced
        mass = (sqrt_M_red_x.conj() @ sqrt_M_red_x) / self.total_mass
        
        approx_eig = abs(complex(energy) / complex(mass)) if abs(complex(mass)) > 1e-12 else 1.0
        init_implicit_factor = 0.5
        curr_implicit_factor = init_implicit_factor / approx_eig if approx_eig > 1e-12 else init_implicit_factor
        
        return curr_implicit_factor

    def initial_solve(self, x_input, constSpaces, constVectors_complex, wAlignment):
        """
        Solves the integration-weights system (Smooth + RoSy + Align) globally,
        in reduced DOF space.

        Args:
            x_input: (N*F,) flattened coefficients [c0_vec, c1_vec, c2_vec, c3_vec]
            constSpaces: (K,) int tensor of face indices
            constVectors_complex: (K,) complex tensor of constraint directions
            wAlignment: (K,) float tensor of per-constraint weights
        
        Returns:
            Tuple of (x_full, x_reduced, totalLhs, totalRhs) for exact implicit factor calculation
        """
        print("Running Initial Linear Solve (Reduced DOF Space)...")
        
        reduced_dofs = self.reduced_dofs

        A_align, b_align_rhs, total_constrained_weight = self.build_alignment_system(
            constSpaces, constVectors_complex, wAlignment)

        # Avoid explicitly forming totalLhs = D'D + R'R + A'A via spGEMM; solve
        # the stacked least-squares system directly and return the components
        # needed to compute energy via matrix-vector products.
        scale_s = math.sqrt(1.0 / self.total_smooth_weight)
        scale_r = math.sqrt(1.0 / self.total_rosy_weight) if self.total_rosy_weight > 0 else 0
        scale_a = math.sqrt(1.0 / total_constrained_weight)
        
        D_red = self.D_smooth_red * scale_s
        R_red = self.A_rosy_red * scale_r
        A_align_red = (A_align @ self.reducMat) * scale_a
        b_align_scaled = b_align_rhs * scale_a

        # reducRhs is zero (checked in precompute), so totalRhs = reducMat' * align_rhs
        totalRhs = A_align_red.T.conj() @ b_align_scaled
        
        A_stack = cp_vstack([D_red, R_red, A_align_red])
        b_stack = cp.hstack([cp.zeros(D_red.shape[0], dtype=cp.complex128),
                             cp.zeros(R_red.shape[0], dtype=cp.complex128),
                             b_align_scaled])
        
        A_complex = A_stack.tocsr()
        
        # Solve using LSMR (block real/imag formulation for complex system)
        A_real = cp_csr_matrix((A_complex.data.real.astype(cp.float64), A_complex.indices, A_complex.indptr), shape=A_complex.shape)
        A_imag = cp_csr_matrix((A_complex.data.imag.astype(cp.float64), A_complex.indices, A_complex.indptr), shape=A_complex.shape)
        A_block = cp_bmat([[A_real, -A_imag], [A_imag, A_real]], format='csr')
        b_block = cp.hstack([b_stack.real.astype(cp.float64), b_stack.imag.astype(cp.float64)])
        
        x_res_block = cp_lsmr(A_block, b_block, atol=1e-10, btol=1e-10)[0]
        x_reduced = (x_res_block[:reduced_dofs] + 1j * x_res_block[reduced_dofs:]).astype(cp.complex128)

        del A_stack, A_complex, A_block
        cp.get_default_memory_pool().free_all_blocks()

        x_full = self.reducMat @ x_reduced + self.reducRhs

        return x_full, x_reduced, A_align_red, totalRhs

    def solve_polyvector(self, input_coeffs: torch.Tensor, 
                          constSpaces: torch.Tensor,
                          constVectors_complex: torch.Tensor,
                          wAlignment: torch.Tensor,
                          num_iterations: int | None = None,  # None = stopping criterion
                          wSmooth: float = 1.0,
                          wRoSy: float = 1.0, 
                          verbose: bool = False,
                          log_interval: int = 5,
                          skip_initial_solve: bool = False,
                          dynamic_annealing: bool = True,
                          skip_first_implicit: bool = True,
                          tau_threshold: float = 0.01,
                          free_magnitude: bool = False,
                          magnitude_clamp: float | None = None) -> torch.Tensor:
        """
        Polyvector field solve with dynamic annealing for mesh-independent convergence.

        An initial global solve (reduced DOF space, can be skipped) followed by
        an iteration loop of a global implicit step, soft RoSy projection, and
        curl-free projection.

        dynamic_annealing=True (default): decay=0.9, stops when tau/tau_0 < 0.01
        or |dx| < 1e-6, ~44 iterations expected.
        dynamic_annealing=False (legacy mode): decay=0.8, fixed 30 iterations.

        Args:
            input_coeffs: Input polyvector coefficients (F, N) complex
            constSpaces: (K,) int tensor of face indices for constraints
            constVectors_complex: (K,) complex tensor of constraint directions (2D intrinsic)
            wAlignment: (K,) float tensor of per-constraint weights
            num_iterations: Override iteration count (None = auto based on stopping criterion)
            wSmooth: Smoothness weight (default 1.0)
            wRoSy: Rotational symmetry weight (default 1.0)
            verbose: Print iteration energies if True
            log_interval: How often to print iteration logs (default 5, 0 to disable)
            skip_initial_solve: If True, skip initial solve and use input_coeffs directly
            dynamic_annealing: If True, use tau-ratio stopping; if False, legacy 30 iter mode
            skip_first_implicit: If True (default), skip implicit step on first iteration only,
                                 allowing direct curl projection on the input field
            free_magnitude: let the field trade curl against scale instead of paying
                            for all of it in rotation. The RoSy step stops normalizing,
                            the alignment targets follow the current per-face magnitude
                            so only their direction is enforced, and one global factor
                            per iteration keeps the field from shrinking to nothing.
            magnitude_clamp: bound the free magnitudes to [1/c, c], or None
        
        Returns:
            Optimized polyvector coefficients (F, N) complex
        """
        if dynamic_annealing:
            implicit_decay = DYNAMIC_DECAY  # 0.9
            max_iters = num_iterations if num_iterations else MAX_ITERATIONS
            use_stopping = (num_iterations is None)  # tau-ratio stopping if no explicit count
        else:
            implicit_decay = LEGACY_DECAY  # 0.8
            max_iters = num_iterations if num_iterations else 30
            use_stopping = False  # legacy mode runs all iterations
        
        if self.A_ls is None or self.wSmooth != wSmooth or self.wRoSy != wRoSy:
            self.precompute(wSmooth=wSmooth, wRoSy=wRoSy, wAlignment=1.0, 
                            constSpaces=constSpaces, 
                            constVectors_complex=constVectors_complex, 
                            wAlignment_array=wAlignment)
        
        num_faces = self.mesh.F.shape[0]
        device = input_coeffs.device

        # Convert input (zero-copy)
        _input_flat = input_coeffs.T.reshape(-1).contiguous()
        x_input = cp.asarray(_input_flat)

        import time
        t_implicit = 0.0
        t_rosy = 0.0
        t_curl = 0.0
        t_proj = 0.0

        if verbose:
            mode_str = "dynamic" if dynamic_annealing else "legacy"
            print(f"Annealing mode: {mode_str}, max_iterations={max_iters}, decay={implicit_decay:.4f}")
        
        if skip_initial_solve:
            self.coeffs = input_coeffs.clone()
            if verbose:
                print("Skipping initial solve - using input coefficients directly")
                E_s, E_r, _ = self.compute_energies(self.coeffs)
                print(f"Input: Smooth={float(E_s):.6f}, RoSy={float(E_r):.6f}")

            curr_implicit_factor = self.compute_implicit_factor_from_coeffs(
                input_coeffs, constSpaces, constVectors_complex, wAlignment)
        else:
            x_sol, x_reduced, A_align_red, totalRhs = self.initial_solve(x_input, constSpaces, constVectors_complex, wAlignment)
            self.coeffs = torch.as_tensor(x_sol, device=device).reshape(self.N, num_faces).T.contiguous().to(torch.complex128)
            
            if verbose:
                E_s, E_r, _ = self.compute_energies(self.coeffs)
                print(f"Iter 0: Smooth={float(E_s):.6f}, RoSy={float(E_r):.6f}")

            # energy = 0.5 * x_reduced^H * totalLhs * x_reduced - x_reduced^H * totalRhs,
            # computed via squared norms: x' L x = ||D_red x||^2 + ||R_red x||^2 + ||A_red x||^2
            init_implicit_factor = 0.5
            scale_s = math.sqrt(1.0 / self.total_smooth_weight)
            scale_r = math.sqrt(1.0 / self.total_rosy_weight) if self.total_rosy_weight > 0 else 0
            
            norm_D = cp.linalg.norm(self.D_smooth_red @ x_reduced * scale_s)**2
            norm_R = cp.linalg.norm(self.A_rosy_red @ x_reduced * scale_r)**2
            norm_A = cp.linalg.norm(A_align_red @ x_reduced)**2 # already scaled
            
            term_quadratic = norm_D + norm_R + norm_A

            x_H = x_reduced.conj()
            energy = 0.5 * term_quadratic - (x_H @ totalRhs)

            # mass = x^H * (reducMat^H * M * reducMat) * x / totalMass;
            # sqrt_Mass_red = sqrt(M) @ reducMat
            sqrt_M_red_x = self.sqrt_Mass_red @ x_reduced  # (full_dofs,)
            mass = (sqrt_M_red_x.conj() @ sqrt_M_red_x) / self.total_mass
            
            approx_eig = abs(complex(energy) / complex(mass)) if abs(complex(mass)) > 1e-12 else 1.0
            curr_implicit_factor = init_implicit_factor / approx_eig if approx_eig > 1e-12 else init_implicit_factor
        
        if verbose:
            print(f"Initial implicit coefficient: {curr_implicit_factor:.6f}")

        initial_tau = curr_implicit_factor
        coeffs_prev = None
        actual_iterations = 0
        converged = False
        convergence_reason = ""

        # Alignment targets follow the current per-face scale, leaving only
        # their direction enforced. Hard constraints (w < 0) go through
        # reducMat
        const_vectors_curr = constVectors_complex
        if free_magnitude:
            const_dirs = constVectors_complex / torch.abs(constVectors_complex).clamp_min(1e-12)
            soft_const = wAlignment >= 0.0
            if verbose:
                print(f"Free magnitude: {int(soft_const.sum())} direction-only "
                      f"constraint(s), {int((~soft_const).sum())} still magnitude-pinned")

        for i in range(max_iters):
            # a. Global implicit step (skipped on first iteration if skip_first_implicit=True)
            if i == 0 and skip_first_implicit:
                if verbose:
                    print("Skipping implicit step on first iteration (skip_first_implicit=True)")
            else:
                _t0 = time.perf_counter()
                self.implicit_step(constSpaces, const_vectors_curr, wAlignment, curr_implicit_factor)
                torch.cuda.synchronize()
                t_implicit += time.perf_counter() - _t0
            
            # b. Soft RoSy projection
            _t0 = time.perf_counter()
            self.soft_rosy(factor=curr_implicit_factor, free_magnitude=free_magnitude)
            torch.cuda.synchronize()
            t_rosy += time.perf_counter() - _t0
            
            # c. Curl-free projection (fixed tight tolerance for output parity)
            _t0 = time.perf_counter()
            self.project_curl(cg_tol=1e-10, constSpaces=constSpaces, constVectors_complex=const_vectors_curr, wAlignment=wAlignment)
            torch.cuda.synchronize()
            t_curl += time.perf_counter() - _t0

            # d. Magnitude projection (optional, not in the original algorithm)
            _t0 = time.perf_counter()
            face_magnitude = self.project_magnitude(
                min_magnitude=1e-3, anchor_scale=free_magnitude, clamp=magnitude_clamp)
            if free_magnitude:
                const_vectors_curr = constVectors_complex.clone()
                const_vectors_curr[soft_const] = (
                    const_dirs[soft_const]
                    * face_magnitude[constSpaces[soft_const]].to(const_dirs.dtype))
            torch.cuda.synchronize()
            t_proj += time.perf_counter() - _t0
            
            actual_iterations = i + 1

            coeff_change = 0.0
            if coeffs_prev is not None:
                coeff_change = float(torch.abs(self.coeffs - coeffs_prev).max())
            coeffs_prev = self.coeffs.clone()

            curr_implicit_factor *= implicit_decay

            if verbose and (log_interval > 0 and (i == 0 or (i + 1) % log_interval == 0)):
                E_s, E_r, _ = self.compute_energies(self.coeffs)
                _curl_l2, curl_max = self.compute_curl_magnitude()
                tau_ratio = curr_implicit_factor / initial_tau
                mag_str = ""
                if free_magnitude:
                    mag_str = (f", |v|=[{float(face_magnitude.min()):.2f}, "
                               f"{float(face_magnitude.max()):.2f}]")
                print(f"Iter {i+1}: Smooth={float(E_s):.6f}, RoSy={float(E_r):.6f}, "
                      f"Curl={curl_max:.2e}, tau/tau_0={tau_ratio:.4f}, "
                      f"dx={coeff_change:.2e}{mag_str}")

            # Stopping conditions, only under dynamic annealing with auto-stopping
            if use_stopping and i >= 4:  # minimum 5 iterations
                tau_ratio = curr_implicit_factor / initial_tau

                if tau_ratio < TAU_THRESHOLD:
                    converged = True
                    convergence_reason = f"tau/tau_0={tau_ratio:.4f} < {TAU_THRESHOLD}"
                    if verbose:
                        print(f"Converged at iteration {i+1}: {convergence_reason}")
                    break

                if coeff_change < COEFF_THRESHOLD:
                    converged = True
                    convergence_reason = f"Δx={coeff_change:.2e} < {COEFF_THRESHOLD}"
                    if verbose:
                        print(f"Converged at iteration {i+1}: {convergence_reason}")
                    break
        
        _final_curl_l2, final_curl_max = self.compute_curl_magnitude()
        if verbose:
            final_tau_ratio = curr_implicit_factor / initial_tau
            print(f"Final: Curl={final_curl_max:.2e}, tau/tau_0={final_tau_ratio:.4f}")

        pv_timing = {
            'num_iterations': actual_iterations,
            'max_iterations': max_iters,
            'converged': converged,
            'convergence_reason': convergence_reason,
            'final_tau_ratio': curr_implicit_factor / initial_tau,
            'final_curl_max': final_curl_max,
            'implicit_step_total': t_implicit,
            'soft_rosy_total': t_rosy,
            'project_curl_total': t_curl,
            'loop_total': t_implicit + t_rosy + t_curl,
        }
        if hasattr(self, '_curl_timings') and self._curl_count > 0:
            for k, v in self._curl_timings.items():
                pv_timing[f'curl_{k}'] = v

        convergence_status = f"CONVERGED ({convergence_reason})" if converged else "MAX ITERATIONS"
        print(f"\n==== POLYVECTOR TIMING ({actual_iterations}/{max_iters} iterations, {convergence_status}) ====")
        print(f"  implicit_step:  {t_implicit:.3f}s ({t_implicit/actual_iterations*1000:.1f}ms/iter)")
        print(f"  soft_rosy:      {t_rosy:.3f}s ({t_rosy/actual_iterations*1000:.1f}ms/iter)")
        print(f"  project_curl:   {t_curl:.3f}s ({t_curl/actual_iterations*1000:.1f}ms/iter)")

        if hasattr(self, '_curl_timings') and self._curl_count > 0:
            print("\n  -- project_curl breakdown (avg per call) --")
            for k, v in self._curl_timings.items():
                avg_ms = (v / self._curl_count) * 1000
                pct = (v / t_curl * 100) if t_curl > 0 else 0
                print(f"     {k:20s}: {avg_ms:6.1f}ms ({pct:4.1f}%)")

        print(f"\n  TOTAL LOOP:     {t_implicit + t_rosy + t_curl:.3f}s")
        print("=" * 70)

        self.timing = pv_timing
        return self.coeffs.clone()

        
    def principal_matching(self):
        """
        Compute principal matching: for each inner edge, transport vec0 from
        face0 to face1, find which vector j in face1 has the minimum rotation
        angle from the transported vec0, compute the total effort as the
        product of (vecjg / transvecjf) over all j, and adjust the matching
        based on that effort.
        """
        he_idx, _twin_idx, f0, f1, rij = self.compute_connections()

        # intField preserves vector ordering when available; otherwise fall
        # back to extracting roots from the polynomial coefficients
        if self._intField is not None:
            roots = self._intField.to(dtype=torch.complex128)
        else:
            roots = extract_polyvector_roots(self.coeffs.to(dtype=torch.complex128), self.N, sort_roots=True)

        u0 = roots[f0]  # (E, N) complex, vectors in face f0
        u1 = roots[f1]  # (E, N) complex, vectors in face f1

        u0_trans = u0 * rij.unsqueeze(1)  # transported to f1, (E, N)
        vec0_trans = u0_trans[:, 0]  # (E,) complex

        # j in u1 with minimum rotation angle from vec0_trans
        min_rot_angle = torch.full((he_idx.shape[0],), float('inf'), dtype=torch.float64, device=self.device)
        index_min_from_zero = torch.zeros((he_idx.shape[0],), dtype=torch.int32, device=self.device)
        
        for j in range(self.N):
            rot_angle = torch.angle(u1[:, j] / vec0_trans)  # (E,)
            abs_rot = torch.abs(rot_angle)
            
            update_mask = abs_rot < min_rot_angle
            min_rot_angle[update_mask] = abs_rot[update_mask]
            index_min_from_zero[update_mask] = j

        # Effort: product of (vecjg / transvecjf) over all j
        effort_product = torch.ones(he_idx.shape[0], dtype=torch.complex128, device=self.device)
        for j in range(self.N):
            effort_product = effort_product * (u1[:, j] / u0_trans[:, j])
        effort = torch.angle(effort_product)  # (E,)

        curr_effort = torch.zeros(he_idx.shape[0], dtype=torch.float64, device=self.device)
        for j in range(self.N):
            matched_j = (j + index_min_from_zero + self.N) % self.N
            batch_idx = torch.arange(he_idx.shape[0], device=self.device)
            u1_matched = u1[batch_idx, matched_j]
            curr_effort = curr_effort + torch.angle(u1_matched / u0_trans[:, j])

        adjustment = torch.round((curr_effort - effort) / (2.0 * torch.pi))
        matching = index_min_from_zero.to(torch.float64) - adjustment
        matching = matching.to(torch.int32)
        matching = ((matching % self.N) + self.N) % self.N
        
        return he_idx, matching

    def project_curl(self, cg_tol=1e-10, constSpaces=None, constVectors_complex=None, wAlignment=None):
        """
        Projects field onto curl-free subspace - GPU-optimized version.
        
        Uses matrix-free CG to avoid memory explosion on large meshes.
        
        Args:
            cg_tol: CG solver tolerance (adaptive: looser early, tighter later)
        
        All operations on GPU using vectorized PyTorch/CuPy operations.
        No Python loops, no CPU transfers in hot path.
        """
        import time
        _timings = {}
        
        mesh = self.mesh
        num_faces = mesh.F.shape[0]
        N = self.N
        device = self.device
        
        _t0 = time.perf_counter()
        roots = extract_polyvector_roots(self.coeffs, N, sort_roots=True)  # (F, N) complex
        torch.cuda.synchronize()
        _timings['extract_roots'] = time.perf_counter() - _t0

        # Flatten to 2D intrinsic
        _t0 = time.perf_counter()
        x0 = torch.stack([roots.real, roots.imag], dim=-1).reshape(-1)  # (2*N*F,)
        torch.cuda.synchronize()
        _timings['flatten'] = time.perf_counter() - _t0

        _t0 = time.perf_counter()
        he_inner, matchings = self.principal_matching()
        torch.cuda.synchronize()
        _timings['principal_matching'] = time.perf_counter() - _t0
        num_inner_edges = he_inner.shape[0]

        # Build curl operator on GPU
        _t0 = time.perf_counter()
        twin_he = mesh.topology.twin_he  # GPU
        face_he = mesh.topology.face  # GPU
        
        left_faces = face_he[he_inner]  # (E,)
        right_faces = face_he[twin_he[he_inner]]  # (E,)
        
        # Get edge vertices
        tri_idx = he_inner % 3
        face_from_he = he_inner // 3
        v0_idx = mesh.F[face_from_he, tri_idx]  # (E,)
        v1_idx = mesh.F[face_from_he, (tri_idx + 1) % 3]  # (E,)
        
        # Edge vectors (E, 3)
        edge_vecs = mesh.V[v1_idx] - mesh.V[v0_idx]
        
        # Project edges onto face bases - BATCHED
        ein_left_x = (edge_vecs * mesh.FBx[left_faces]).sum(dim=1)  # (E,)
        ein_left_y = (edge_vecs * mesh.FBy[left_faces]).sum(dim=1)  # (E,)
        ein_right_x = (edge_vecs * mesh.FBx[right_faces]).sum(dim=1)  # (E,)
        ein_right_y = (edge_vecs * mesh.FBy[right_faces]).sum(dim=1)  # (E,)
        
        # Build sparse COO indices - VECTORIZED
        E = num_inner_edges
        n_range = torch.arange(N, device=device)  # (N,)
        
        row_base = torch.arange(E, device=device).unsqueeze(1) * N + n_range  # (E, N)
        rows = row_base.unsqueeze(2).expand(-1, -1, 4).reshape(-1)  # (E*N*4,)
        
        col_left_u = (2 * N * left_faces.unsqueeze(1) + 2 * n_range).reshape(-1)  # (E*N,)
        col_left_v = col_left_u + 1  # (E*N,)
        
        match_expanded = matchings.unsqueeze(1)  # (E, 1)
        n_matched = (n_range.unsqueeze(0) + match_expanded + N) % N  # (E, N)
        col_right_u = (2 * N * right_faces.unsqueeze(1) + 2 * n_matched).reshape(-1)  # (E*N,)
        col_right_v = col_right_u + 1  # (E*N,)
        
        cols = torch.stack([col_left_u.view(E, N), col_left_v.view(E, N),
                           col_right_u.view(E, N), col_right_v.view(E, N)], dim=-1).reshape(-1)
        
        vals_left_x = -ein_left_x.unsqueeze(1).expand(-1, N).reshape(-1)
        vals_left_y = -ein_left_y.unsqueeze(1).expand(-1, N).reshape(-1)
        vals_right_x = ein_right_x.unsqueeze(1).expand(-1, N).reshape(-1)
        vals_right_y = ein_right_y.unsqueeze(1).expand(-1, N).reshape(-1)
        vals = torch.stack([vals_left_x.view(E, N), vals_left_y.view(E, N),
                           vals_right_x.view(E, N), vals_right_y.view(E, N)], dim=-1).reshape(-1)
        
        indices = torch.stack([rows.long(), cols.long()], dim=0)
        C_shape = (N * E, 2 * N * num_faces)
        C_sparse = torch.sparse_coo_tensor(indices, vals.double(), C_shape, device=device).coalesce()
        torch.cuda.synchronize()
        _timings['build_curl_matrix'] = time.perf_counter() - _t0

        # Solve curl projection on GPU
        _t0 = time.perf_counter()
        
        import cupy as cp
        from cupyx.scipy.sparse import csr_matrix as cp_csr
        
        C_coo = C_sparse.coalesce()
        C_indices_gpu = C_coo.indices()
        C_values_gpu = C_coo.values()
        
        C_cp = cp_csr((cp.asarray(C_values_gpu), 
                       (cp.asarray(C_indices_gpu[0]), cp.asarray(C_indices_gpu[1]))),
                      shape=C_shape)
        
        x0_cp = cp.asarray(x0.double())

        # Fix hard-constrained DOFs: zero the corresponding columns of C, so
        # dx = -C.T @ lambda is forced to 0 on those DOFs. A face constrained
        # on any one root freezes all N of its roots, to keep them from
        # jittering the constrained one.
        constrained_mask = None
        if constSpaces is not None and wAlignment is not None and constVectors_complex is not None:
            if hasattr(wAlignment, 'cpu'): c_weights = wAlignment.cpu().numpy()
            else: c_weights = wAlignment
            
            if hasattr(constSpaces, 'cpu'): c_spaces = constSpaces.cpu().numpy()
            else: c_spaces = constSpaces

            hard_mask = c_weights < 0.0
            hard_faces = c_spaces[hard_mask]
            
            if len(hard_faces) > 0:
                constrained_mask = cp.zeros(2 * N * num_faces, dtype=cp.bool_)
                for f_idx in hard_faces:
                    start_idx = int(2 * N * f_idx)
                    end_idx = int(start_idx + 2 * N)
                    constrained_mask[start_idx:end_idx] = True

        if constrained_mask is not None:
            free_mask = (~constrained_mask).astype(cp.float64)
            FreeMaskDiag = cp_csr((free_mask, (cp.arange(2 * N * num_faces), cp.arange(2 * N * num_faces))), shape=(2 * N * num_faces, 2 * N * num_faces))
            C_cp = C_cp @ FreeMaskDiag
        
        Cx0 = C_cp @ x0_cp
        
        dim_c = N * E
        reg = 1e-8

        # Matrix-free CCT: C @ C.T causes spGEMM OOM on large meshes
        from cupyx.scipy.sparse.linalg import LinearOperator
        from cupyx.scipy.sparse.linalg import cg as cp_cg
        
        def cct_matvec(v):
            """Apply (CC^T + reg*I) without forming the matrix."""
            return C_cp @ (C_cp.T @ v) + reg * v
        
        CCT_op = LinearOperator((dim_c, dim_c), matvec=cct_matvec, dtype=cp.float64)
        
        # Diagonal preconditioner from row norms of C (approximates diag of CC^T)
        _t_cct = time.perf_counter()
        row_norms_sq = cp.array(C_cp.power(2).sum(axis=1)).flatten() + reg
        M_inv = 1.0 / cp.maximum(row_norms_sq, 1e-12)
        cp.cuda.Stream.null.synchronize()
        _timings['cct_compute'] = _timings.get('cct_compute', 0) + (time.perf_counter() - _t_cct)
        
        def precond(x):
            return M_inv * x
        
        M = LinearOperator((dim_c, dim_c), matvec=precond, dtype=cp.float64)
        
        # CG Warm Starting: use previous lambda as initial guess
        cg_x0 = None
        if (hasattr(self, '_prev_lambda') and self._prev_lambda is not None
                and self._prev_lambda.shape[0] == dim_c):
            cg_x0 = self._prev_lambda
        
        _t_cg = time.perf_counter()
        lam, info = cp_cg(CCT_op, Cx0, x0=cg_x0, rtol=cg_tol, maxiter=2000, M=M)
        if info != 0:
            # Fallback with looser tolerance if CG doesn't converge
            lam, info = cp_cg(CCT_op, Cx0, rtol=max(cg_tol * 100, 1e-6), maxiter=5000)
        cp.cuda.Stream.null.synchronize()
        _timings['cg_iterations'] = _timings.get('cg_iterations', 0) + (time.perf_counter() - _t_cg)
        
        # Cache lambda for next iteration's warm start
        self._prev_lambda = lam
        
        CTlam = C_cp.T @ lam
        x_proj_cp = x0_cp - CTlam

        del C_coo, C_indices_gpu, C_values_gpu, Cx0, lam, CTlam
        cp.get_default_memory_pool().free_all_blocks()
        
        cp.cuda.Stream.null.synchronize()
        _timings['cg_solve'] = time.perf_counter() - _t0

        # Reconstruct roots, then coefficients under signSymmetry
        _t0 = time.perf_counter()
        x_proj = torch.as_tensor(x_proj_cp, device=device, dtype=torch.float64)
        x_reshaped = x_proj.view(num_faces, N, 2)
        roots_new = torch.complex(x_reshaped[..., 0], x_reshaped[..., 1])

        self.coeffs = roots_to_coeffs_sign_symmetric(roots_new, N)
        torch.cuda.synchronize()
        _timings['reconstruct'] = time.perf_counter() - _t0

        if not hasattr(self, '_curl_timings'):
            self._curl_timings = {k: 0.0 for k in _timings}
            self._curl_count = 0
        for k, v in _timings.items():
            self._curl_timings[k] += v
        self._curl_count += 1

    def project_magnitude(self, min_magnitude=1e-3, anchor_scale=False, clamp=None):
        """
        Projects the polyvector field to ensure each vector has at least min_magnitude.
        
        Args:
            min_magnitude: Minimum allowed magnitude for each vector root.
            anchor_scale: rescale every root by one global factor so the
                area-weighted mean magnitude is 1. Keeps the free field off the
                floor without touching its relative variation.
            clamp: bound magnitudes to [1/clamp, clamp] after anchoring, or None.

        Returns:
            (F,) mean root magnitude per face.
        """
        roots = extract_polyvector_roots(self.coeffs, self.N, sort_roots=True)
        mags = torch.abs(roots)
        changed = False

        too_small_mask = mags < min_magnitude
        if too_small_mask.any():
            # v_new = (v_old / |v_old|) * min_magnitude; epsilon guards a zero root
            scale_factors = min_magnitude / (mags + 1e-16)
            roots[too_small_mask] *= scale_factors[too_small_mask]
            changed = True

        if anchor_scale:
            areas = self.mesh.faceAreas
            per_face = torch.abs(roots).mean(dim=1)
            scale = (per_face * areas).sum() / areas.sum().clamp_min(1e-12)
            if scale > 1e-12:
                roots = roots / scale
                changed = True

        if clamp is not None and clamp > 1.0:
            mags = torch.abs(roots)
            roots = roots * (mags.clamp(1.0 / clamp, clamp) / (mags + 1e-16))
            changed = True

        if changed:
            self.coeffs = roots_to_coeffs_sign_symmetric(roots, self.N)
        return torch.abs(roots).mean(dim=1)