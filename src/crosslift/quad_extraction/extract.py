from __future__ import annotations

import multiprocessing as mp
import queue as queuelib
from dataclasses import dataclass
from time import time

import numpy as np

from crosslift import _core
from crosslift.fields.rosy import expand_rosy, to_extrinsic
from crosslift.geometry.io import read_mesh_arrays, write_obj
from crosslift.quad_extraction.curl import (
    run_curl_correction,
    run_curl_correction_per_component,
)
from crosslift.quad_extraction.quadwild import run_quadwild
from crosslift.quad_extraction.trimesh import concat_meshes, slice_component


@dataclass
class QuadResult:
    """What the MIQ route produces."""

    vertices: np.ndarray            # (Q, 3) float64
    faces: np.ndarray               # (R, 4) int32
    curlfree: np.ndarray            # (F, N, 3) the field this was extracted from
    uv: np.ndarray | None           # (P, 2) float64, or None
    face_uv: np.ndarray | None      # (F, 3) int32, or None

    @property
    def corner_uv(self) -> np.ndarray | None:
        """(3F, 2) per-corner UVs, the layout polyscope wants."""
        if self.uv is None or self.face_uv is None:
            return None
        return self.uv[self.face_uv].reshape(-1, 2)


@dataclass
class IntegrationResult:
    """What the Directional route produces."""

    n_function: np.ndarray          # (cV, N) per cut-vertex parametric values
    n_corner_functions: np.ndarray  # (F, 3N) per-corner parametric values
    cut_vertices: np.ndarray        # (cV, 3)
    cut_faces: np.ndarray           # (F, 3) int32
    curlfree: np.ndarray            # (F, N, 3)

    @property
    def corner_uv(self) -> np.ndarray:
        """First two parametric functions per corner, as a (3F, 2) UV map."""
        F = self.n_corner_functions.shape[0]
        N = self.n_corner_functions.shape[1] // 3
        return self.n_corner_functions.reshape(F, 3, N)[:, :, :2].reshape(-1, 2)


def _gradient_size(V, F, use_adaptive_scaling, adaptive_scale_factor,
                   static_quad_scale, target_quad_count=None,
                   target_quads_per_triangle=None, label=""):
    """The density knob, shared by both parameterizers."""
    if target_quads_per_triangle is not None and target_quad_count is None:
        target_quad_count = round(target_quads_per_triangle * F.shape[0])

    if target_quad_count is not None:
        if target_quad_count <= 0:
            raise ValueError(
                f"target quad count must be positive, got {target_quad_count}")
        p = V[F]
        area = 0.5 * np.linalg.norm(
            np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1).sum()
        diag = np.linalg.norm(V.max(axis=0) - V.min(axis=0))
        gradient_size = diag * np.sqrt(target_quad_count / area)
        print(f"{label}Gradient size for ~{target_quad_count} quads: "
              f"{gradient_size:.2f}")
    elif use_adaptive_scaling:
        gradient_size = _core.adaptive_gradient_size(V, F, adaptive_scale_factor)
        print(f"{label}Adaptive gradient size: {gradient_size:.2f}")
    else:
        gradient_size = static_quad_scale
        print(f"{label}Fixed gradient size: {gradient_size:.2f}")
    return gradient_size


def _qex(V, F, UV, FUV, label=""):
    """Trace the integer isolines of a seamless UV map into quads.

    Quads whose indices fall outside the vertex array are dropped: QEx emits
    them on degenerate patches.
    """
    print(f"{label}QEx quad extraction")
    start = time()
    quad_V, quad_F = _core.qex_extract(V, F, UV, FUV)
    valid = (quad_F >= 0).all(axis=1) & (quad_F < quad_V.shape[0]).all(axis=1)
    if not valid.all():
        print(f"{label}  dropping {(~valid).sum()} quad face(s) with "
              "out-of-range indices")
        quad_F = np.ascontiguousarray(quad_F[valid])
    print(f"{label}  quad vertices: {quad_V.shape[0]}, quad faces: {quad_F.shape[0]} "
          f"({time() - start:.2f}s)")
    return quad_V, quad_F


def _miq_and_qex(V, F, curlfree_vecs, gradient_size, stiffness, direct_round,
                 h_per_face, normalize_frames, label=""):
    """One MIQ + QEx pass over a single connected mesh.

    Returns ``(quad_V, quad_F, UV, FUV)``.
    """
    PD1 = np.ascontiguousarray(curlfree_vecs[:, 0, :], dtype=np.float64)
    PD2 = np.ascontiguousarray(curlfree_vecs[:, 1, :], dtype=np.float64)

    print(f"{label}MIQ parameterization")
    start = time()
    UV, FUV = _core.miq_parameterize(
        V, F, PD1, PD2,
        gradient_size=gradient_size,
        stiffness=stiffness,
        direct_round=direct_round,
        h_per_face=h_per_face,
        normalize_frames=normalize_frames,
    )
    print(f"{label}  UV vertices: {UV.shape[0]}, UV faces: {FUV.shape[0]} "
          f"({time() - start:.2f}s)")

    quad_V, quad_F = _qex(V, F, UV, FUV, label=label)
    return quad_V, quad_F, UV, FUV


def _miq_worker(result_queue, args, kwargs):
    try:
        result_queue.put(_miq_and_qex(*args, **kwargs))
    except BaseException as e:  # noqa: BLE001 - relayed to the parent
        result_queue.put(e)


def _miq_and_qex_isolated(*args, timeout=None, **kwargs):
    """Run :func:`_miq_and_qex` in a child process.

    Raises:
        RuntimeError: the child died on a signal or outlived ``timeout``.
    """
    ctx = mp.get_context("fork")
    result_queue = ctx.Queue()
    proc = ctx.Process(target=_miq_worker, args=(result_queue, args, kwargs))
    proc.start()

    deadline = None if timeout is None else time() + timeout
    payload = None
    while True:
        try:
            payload = result_queue.get(timeout=0.1)
            break
        except queuelib.Empty:
            if not proc.is_alive():
                try:
                    payload = result_queue.get(timeout=1.0)
                except queuelib.Empty:
                    payload = None
                break
            if deadline is not None and time() > deadline:
                proc.kill()
                proc.join()
                raise RuntimeError(f"MIQ exceeded {timeout}s")

    proc.join()
    if payload is None:
        raise RuntimeError(f"MIQ died with exit code {proc.exitcode}")
    if isinstance(payload, BaseException):
        raise payload
    return payload


def extract_quads(
    curlfree_vecs: np.ndarray,
    input_mesh_path: str,
    output_path: str | None = None,
    *,
    use_adaptive_scaling: bool = True,
    adaptive_scale_factor: float = 16.0,
    static_quad_scale: float = 80.0,
    target_quad_count: int | None = None,
    target_quads_per_triangle: float | None = None,
    direct_round: bool = False,
    h_per_face: np.ndarray | None = None,
    miq_stiffness: float = 5.0,
    miq_normalize_frames: bool = True,
    face_labels: np.ndarray | None = None,
    num_components: int = 1,
    min_component_faces: int = 4,
    isolate_miq: bool = True,
    miq_timeout: float | None = None,
) -> QuadResult:
    """Run MIQ + QEx on a curl-free cross field.

    Args:
        curlfree_vecs: (F, 4, 3) curl-free field.
        input_mesh_path: the mesh the field lives on.
        output_path: where to write the quad OBJ, or None to skip.
        use_adaptive_scaling: derive the gradient size from mesh curvature
            instead of using ``static_quad_scale``.
        target_quad_count, target_quads_per_triangle: ask for a quad count and
            let the gradient size be solved for; see :func:`_gradient_size`.
        h_per_face: optional (F,) relative sizing; see ``_core.miq_parameterize``.
        face_labels, num_components: solve each connected component separately.
            Per-component UVs cannot be concatenated, so ``uv`` comes back None.
            The gradient size is rescaled per shell, to its bounding box.
        min_component_faces: shells smaller than this are skipped entirely.
        isolate_miq: run each MIQ + QEx pass in a child process.
        miq_timeout: seconds to allow one isolated pass, or None for no limit.

    Returns:
        A :class:`QuadResult`.
    """
    run_miq = _miq_and_qex_isolated if isolate_miq else _miq_and_qex
    miq_kwargs = {"timeout": miq_timeout} if isolate_miq else {}
    V, F = read_mesh_arrays(input_mesh_path)
    print(f"  Mesh: {V.shape[0]} vertices, {F.shape[0]} faces")

    gradient_size = _gradient_size(
        V, F, use_adaptive_scaling, adaptive_scale_factor, static_quad_scale,
        target_quad_count, target_quads_per_triangle, "  ")

    h_full = None if h_per_face is None else np.ascontiguousarray(h_per_face, np.float64)
    if h_full is not None:
        print(f"  Per-face sizing: mean={h_full.mean():.4g}, "
              f"min={h_full.min():.4g}, max={h_full.max():.4g}")

    single_shell = num_components <= 1 or face_labels is None
    if single_shell:
        quad_V, quad_F, UV, FUV = run_miq(
            V, F, curlfree_vecs, gradient_size, miq_stiffness, direct_round,
            h_full, miq_normalize_frames, label="  ", **miq_kwargs)
    else:
        print(f"  Splitting into {num_components} components for MIQ")
        diag_full = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
        quad_Vs, quad_Fs = [], []
        failed = 0
        for k in range(num_components):
            face_arrays = {"cf": curlfree_vecs, "h": h_full}
            V_k, F_k, _, f_mask, sliced = slice_component(
                V, F, face_labels, k, face_arrays=face_arrays)
            comp_size = int(f_mask.sum())
            if comp_size < min_component_faces:
                print(f"  component {k}: {comp_size} faces — too small, skipping")
                continue

            diag_k = float(np.linalg.norm(V_k.max(axis=0) - V_k.min(axis=0)))
            if diag_k <= 0.0 or diag_full <= 0.0:
                print(f"  component {k}: degenerate extent, skipping")
                continue
            gradient_size_k = gradient_size * diag_k / diag_full

            print(f"  component {k}: {V_k.shape[0]} V, {comp_size} F, "
                  f"gradient size {gradient_size_k:.2f}")
            try:
                quad_V_k, quad_F_k, _, _ = run_miq(
                    V_k, F_k, sliced["cf"], gradient_size_k, miq_stiffness,
                    direct_round, sliced["h"], miq_normalize_frames,
                    label="    ", **miq_kwargs)
            except Exception as e:  # noqa: BLE001 - MIQ/QEx raise anything
                print(f"    component {k} failed, skipping: {e}")
                failed += 1
                continue
            if quad_F_k.shape[0] > 0:
                quad_Vs.append(quad_V_k)
                quad_Fs.append(quad_F_k)

        quad_V, quad_F = concat_meshes(quad_Vs, quad_Fs)
        UV, FUV = None, None
        print(f"  Total: {quad_V.shape[0]} vertices, {quad_F.shape[0]} faces "
              f"across {len(quad_Vs)} components ({failed} failed)")

    if output_path:
        write_obj(output_path, quad_V, quad_F)
        print(f"  Saved quad mesh: {output_path}")

    return QuadResult(vertices=quad_V, faces=quad_F, curlfree=curlfree_vecs,
                      uv=UV, face_uv=FUV)


def _quadwild_scale_fact(V, F, gradient_size):
    """Convert a MIQ gradient size to QuadWild's ``scaleFact``.

    ``scaleFact`` multiplies the mesh's mean edge length to get a target quad
    edge; a gradient size asks for ``bbox diagonal / gradient_size``.
    """
    diag = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
    corners = np.stack([F, np.roll(F, -1, axis=1)], axis=-1).reshape(-1, 2)
    edges = np.unique(np.sort(corners, axis=1), axis=0)
    mean_edge = float(np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1).mean())
    return diag / gradient_size / mean_edge


def extract_quads_quadwild(
    field_vecs: np.ndarray,
    input_mesh_path: str,
    output_path: str | None = None,
    *,
    use_adaptive_scaling: bool = True,
    adaptive_scale_factor: float = 16.0,
    static_quad_scale: float = 80.0,
    target_quad_count: int | None = None,
    target_quads_per_triangle: float | None = None,
    sharp_angle: float = -1.0,
    scale_fact: float = 1.0,
    keep_workdir: bool = False,
) -> QuadResult:
    """Run QuadWild on a cross fiel

    Args:
        field_vecs: (F, 4, 3) field; its first direction is written as PD1.
        input_mesh_path: the mesh the field lives on.
        output_path: where to write the quad OBJ, or None to skip.
        use_adaptive_scaling, adaptive_scale_factor, static_quad_scale,
            target_quad_count, target_quads_per_triangle: density, in MIQ's
            gradient-size units — see :func:`_gradient_size`.
        sharp_angle: normal-deviation threshold in degrees for sharp features.
        scale_fact: extra multiplier on the derived ``scaleFact``; smaller
            gives denser quads.
        keep_workdir: leave QuadWild's scratch directory on disk.

    Returns:
        A :class:`QuadResult`.
    """
    V, F = read_mesh_arrays(input_mesh_path)
    print(f"  Mesh: {V.shape[0]} vertices, {F.shape[0]} faces")

    gradient_size = _gradient_size(
        V, F, use_adaptive_scaling, adaptive_scale_factor, static_quad_scale,
        target_quad_count, target_quads_per_triangle, "  ")
    scale_fact = scale_fact * _quadwild_scale_fact(V, F, gradient_size)

    PD1 = np.ascontiguousarray(field_vecs[:, 0, :], dtype=np.float64)
    quad_V, quad_F = run_quadwild(
        V, F, PD1, output_path,
        sharp_angle=sharp_angle, scale_fact=scale_fact,
        keep_workdir=keep_workdir, label="  ")

    return QuadResult(vertices=quad_V, faces=quad_F, curlfree=field_vecs,
                      uv=None, face_uv=None)


def integrate_nrosy(
    curlfree_vecs: np.ndarray,
    input_mesh_path: str,
    N: int,
    output_path: str | None = None,
    *,
    use_adaptive_scaling: bool = True,
    adaptive_scale_factor: float = 16.0,
    static_quad_scale: float = 80.0,
    target_quad_count: int | None = None,
    target_quads_per_triangle: float | None = None,
    integral_seamless: bool = True,
    round_seams: bool = True,
    normalize_frames: bool = True,
    extract_quads_qex: bool = True,
) -> QuadResult | IntegrationResult:
    """Run Directional's seamless integration on a curl-free N-RoSy field.

    Args:
        curlfree_vecs: (F, N, 3) curl-free field.
        use_adaptive_scaling, adaptive_scale_factor, static_quad_scale,
            target_quad_count, target_quads_per_triangle: density, in MIQ's
            gradient-size units — see :func:`_gradient_size`.
        integral_seamless: make the seam transitions integer translations as
            well as integer rotations.
        round_seams: round the seam transitions instead of the singularities.
        normalize_frames: unitize the field first, so only its direction
            reaches the solver.
        extract_quads_qex: at N = 4, hand the parameterization to libQEx.
            Requires ``integral_seamless``.
        output_path: where to write the OBJ — the quad mesh when QEx runs, the
            cut mesh otherwise. None to skip.

    Returns:
        A :class:`QuadResult` when QEx runs, a :class:`IntegrationResult`
        otherwise.
    """
    V, F = read_mesh_arrays(input_mesh_path)
    ext_field = to_extrinsic(curlfree_vecs, normalize=normalize_frames)

    print(f"  Mesh: {V.shape[0]} vertices, {F.shape[0]} faces, N={N}")
    gradient_size = _gradient_size(
        V, F, use_adaptive_scaling, adaptive_scale_factor, static_quad_scale,
        target_quad_count, target_quads_per_triangle, "  ")
    length_ratio = 1.0 / gradient_size
    print(f"  -> length ratio: {length_ratio:.5f}")

    print(f"  Seamless integration (integral_seamless={integral_seamless})")
    start = time()
    n_function, n_corner, cut_V, cut_F = _core.nrosy_integrate(
        V, F, ext_field, N,
        length_ratio=length_ratio,
        integral_seamless=integral_seamless,
        round_seams=round_seams,
    )
    print(f"    cut mesh: {cut_V.shape[0]} vertices, {cut_F.shape[0]} faces "
          f"({time() - start:.2f}s)")

    result = IntegrationResult(
        n_function=n_function, n_corner_functions=n_corner,
        cut_vertices=cut_V, cut_faces=cut_F, curlfree=curlfree_vecs,
    )

    if extract_quads_qex and N == 4:
        if not integral_seamless:
            raise ValueError(
                "QEx needs the seam transitions to be integer grid "
                "automorphisms; pass integral_seamless=True or "
                "extract_quads_qex=False.")
        UV = np.ascontiguousarray(result.corner_uv, dtype=np.float64)
        FUV = np.arange(3 * F.shape[0], dtype=np.int32).reshape(F.shape[0], 3)
        quad_V, quad_F = _qex(V, F, UV, FUV, label="  ")
        if output_path:
            write_obj(output_path, quad_V, quad_F)
            print(f"  Saved quad mesh: {output_path}")
        return QuadResult(vertices=quad_V, faces=quad_F,
                          curlfree=curlfree_vecs, uv=UV, face_uv=FUV)

    if output_path:
        write_obj(output_path, cut_V, cut_F)
        print(f"  Saved cut mesh: {output_path}")
    return result


def extract_surface(
    rep_vectors_3d: np.ndarray,
    face_normals: np.ndarray,
    input_mesh_path: str,
    output_path: str | None = None,
    *,
    w_align_weights: np.ndarray | None = None,
    N: int = 4,
    method: str = "miq",
    face_labels: np.ndarray | None = None,
    num_components: int = 1,
    skip_curl_correction: bool = False,
    curlfree_vecs: np.ndarray | None = None,
    # Curl correction
    wSmooth: float = 1.0,
    wRoSy: float = 1.0,
    wAlign: float = 1.0,
    tau_threshold: float = 0.01,
    skip_initial_solve: bool = False,
    skip_first_implicit: bool = True,
    dynamic_annealing: bool = True,
    free_magnitude: bool = False,
    magnitude_clamp: float | None = None,
    # Density
    use_adaptive_scaling: bool = True,
    adaptive_scale_factor: float = 16.0,
    static_quad_scale: float = 80.0,
    target_quad_count: int | None = None,
    target_quads_per_triangle: float | None = None,
    # MIQ
    direct_round: bool = False,
    h_per_face: np.ndarray | None = None,
    miq_stiffness: float = 5.0,
    miq_normalize_frames: bool = True,
    isolate_miq: bool = True,
    miq_timeout: float | None = None,
    # QuadWild
    quadwild_sharp_angle: float = -1.0,
    quadwild_scale_fact: float = 1.0,
    quadwild_keep_workdir: bool = False,
    # Seamless integration
    integral_seamless: bool = True,
    round_seams: bool = True,
    integrate_normalize_frames: bool = True,
    integrate_extract_quads: bool = True,
) -> QuadResult | IntegrationResult:
    """Curl-correct a solved field, then parameterize and extract from it.

    Args:
        rep_vectors_3d: (F, 3) representative direction per face, as the solve
            produces it.
        face_normals: (F, 3) face normals.
        w_align_weights: optional (F,) per-face alignment weights for curl
            correction. Negative entries mark hard constraints.
        method: ``"miq"`` (default), ``"quadwild"`` — both N = 4 only — or
            ``"integrate"``.
        face_labels, num_components: connected-component labels, for the
            per-component paths. Both curl correction and MIQ assume a single
            connected surface. QuadWild handles multiple shells itself.
        skip_curl_correction: feed the raw field straight to the parameterizer.
        curlfree_vecs: an already-corrected (F, N, 3) field, to reuse one across
            visualization and extraction.
        free_magnitude: correct with the magnitude free, so the field can absorb
            curl as scale variation. Pair with ``miq_normalize_frames=False``,
            which is what carries that scale into the parameterization.
        magnitude_clamp: bound the free magnitudes to [1/c, c].
        quadwild_sharp_angle, quadwild_scale_fact, quadwild_keep_workdir: see
            :func:`extract_quads_quadwild`.

    Returns:
        A :class:`QuadResult` whenever quads come out, which is every method at
        N = 4; an :class:`IntegrationResult` when integration stops at the
        parameterization.
    """
    banners = {
        "miq": "MIQ + QEx",
        "quadwild": "QuadWild",
        "integrate": "seamless integration",
    }
    if method not in banners:
        raise ValueError(f"unknown extraction method {method!r}; "
                         f"expected one of {sorted(banners)}")
    if method in ("miq", "quadwild") and N != 4:
        raise ValueError(f"{method} extracts quads, so it needs N=4, got N={N}. "
                         "Use method='integrate' for other symmetry orders.")

    total_start = time()
    print("=" * 60)
    print(f"SURFACE EXTRACTION (N={N}: {banners[method]})")
    print("=" * 60)

    per_component = num_components > 1 and face_labels is not None and method == "miq"

    if curlfree_vecs is not None:
        print("\n=== Curl correction (skipped — field supplied by caller) ===")
    elif skip_curl_correction:
        print("\n=== Curl correction (skipped by request) ===")
        curlfree_vecs = expand_rosy(rep_vectors_3d, face_normals, N).astype(np.float64)
    elif per_component and N == 4:
        V, F = read_mesh_arrays(input_mesh_path)
        curlfree_vecs = run_curl_correction_per_component(
            rep_vectors_3d=rep_vectors_3d, face_normals=face_normals,
            V=V, F=F, face_labels=face_labels, num_components=num_components,
            w_align_weights=w_align_weights, N=N,
            skip_initial_solve=skip_initial_solve,
            skip_first_implicit=skip_first_implicit,
            dynamic_annealing=dynamic_annealing,
            wSmooth=wSmooth, wRoSy=wRoSy, wAlign=wAlign,
            tau_threshold=tau_threshold,
            free_magnitude=free_magnitude,
            magnitude_clamp=magnitude_clamp,
        )
    else:
        curlfree_vecs = run_curl_correction(
            rep_vectors_3d=rep_vectors_3d, face_normals=face_normals,
            input_mesh_path=input_mesh_path,
            w_align_weights=w_align_weights, N=N,
            skip_initial_solve=skip_initial_solve,
            skip_first_implicit=skip_first_implicit,
            dynamic_annealing=dynamic_annealing,
            wSmooth=wSmooth, wRoSy=wRoSy, wAlign=wAlign,
            tau_threshold=tau_threshold,
            free_magnitude=free_magnitude,
            magnitude_clamp=magnitude_clamp,
        )

    if method == "miq":
        result = extract_quads(
            curlfree_vecs, input_mesh_path, output_path,
            use_adaptive_scaling=use_adaptive_scaling,
            adaptive_scale_factor=adaptive_scale_factor,
            static_quad_scale=static_quad_scale,
            target_quad_count=target_quad_count,
            target_quads_per_triangle=target_quads_per_triangle,
            direct_round=direct_round,
            h_per_face=h_per_face,
            miq_stiffness=miq_stiffness,
            miq_normalize_frames=miq_normalize_frames,
            face_labels=face_labels,
            num_components=num_components,
            isolate_miq=isolate_miq,
            miq_timeout=miq_timeout,
        )
    elif method == "quadwild":
        result = extract_quads_quadwild(
            curlfree_vecs, input_mesh_path, output_path,
            use_adaptive_scaling=use_adaptive_scaling,
            adaptive_scale_factor=adaptive_scale_factor,
            static_quad_scale=static_quad_scale,
            target_quad_count=target_quad_count,
            target_quads_per_triangle=target_quads_per_triangle,
            sharp_angle=quadwild_sharp_angle,
            scale_fact=quadwild_scale_fact,
            keep_workdir=quadwild_keep_workdir,
        )
    else:
        result = integrate_nrosy(
            curlfree_vecs, input_mesh_path, N, output_path,
            use_adaptive_scaling=use_adaptive_scaling,
            adaptive_scale_factor=adaptive_scale_factor,
            static_quad_scale=static_quad_scale,
            target_quad_count=target_quad_count,
            target_quads_per_triangle=target_quads_per_triangle,
            integral_seamless=integral_seamless,
            round_seams=round_seams,
            normalize_frames=integrate_normalize_frames,
            extract_quads_qex=integrate_extract_quads,
        )

    print(f"\nSurface extraction complete ({time() - total_start:.2f}s)")
    return result
