from __future__ import annotations

import math

import numpy as np


def quad_frames(V, Q):
    """Per-quad centroid and an orthonormal in-plane frame"""
    P = V[Q]                                   # (m, 4, 3)
    c = P.mean(axis=1)                         # (m, 3)
    nrm = np.zeros_like(c)
    for k in range(4):
        a, b = P[:, k], P[:, (k + 1) % 4]
        nrm[:, 0] += (a[:, 1] - b[:, 1]) * (a[:, 2] + b[:, 2])
        nrm[:, 1] += (a[:, 2] - b[:, 2]) * (a[:, 0] + b[:, 0])
        nrm[:, 2] += (a[:, 0] - b[:, 0]) * (a[:, 1] + b[:, 1])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-20)

    helper = np.tile(np.array([0.0, 0.0, 1.0]), (c.shape[0], 1))
    flip = np.abs(nrm[:, 2]) > 0.9
    helper[flip] = np.array([1.0, 0.0, 0.0])
    t1 = np.cross(nrm, helper)
    t1 /= np.maximum(np.linalg.norm(t1, axis=1, keepdims=True), 1e-20)
    t2 = np.cross(nrm, t1)
    return c, t1, t2


def _quad_areas(V, Q):
    def tri(a, b, c):
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    P = V[Q]
    return tri(P[:, 0], P[:, 1], P[:, 2]) + tri(P[:, 0], P[:, 2], P[:, 3])


def _median_edge(V, Q):
    P = V[Q]
    e = np.stack([P[:, (k + 1) % 4] - P[:, k] for k in range(4)], axis=1)
    return float(np.median(np.linalg.norm(e, axis=2)))


def feature_edges(V_ref, F_ref, angle_threshold_deg, min_component_edges=3):
    """Sharp creases and boundaries of a triangle mesh, as vertex-index pairs.

    Args:
        V_ref: (n, 3) vertices.
        F_ref: (m, 3) triangles.
        angle_threshold_deg: dihedral angle above which an edge is a crease.
        min_component_edges: drop connected crease components smaller than this.

    Returns:
        (S, 2) int array of edges.
    """
    F_ref = np.asarray(F_ref, dtype=np.int64)
    pairs = np.sort(
        np.stack([F_ref, np.roll(F_ref, -1, axis=1)], axis=-1).reshape(-1, 2), axis=1)
    edges, inverse = np.unique(pairs, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)

    corner_faces = np.repeat(np.arange(F_ref.shape[0], dtype=np.int64), 3)
    order = np.argsort(inverse, kind="stable")
    sorted_edges, sorted_faces = inverse[order], corner_faces[order]
    edge_faces = np.full((edges.shape[0], 2), -1, dtype=np.int64)
    is_first = np.empty(sorted_edges.shape[0], dtype=bool)
    is_first[0] = True
    is_first[1:] = sorted_edges[1:] != sorted_edges[:-1]
    edge_faces[sorted_edges[is_first], 0] = sorted_faces[is_first]
    edge_faces[sorted_edges[~is_first], 1] = sorted_faces[~is_first]

    p = V_ref[F_ref]
    normals = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-20)

    boundary = edge_faces[:, 1] < 0
    interior = ~boundary
    cos = (normals[edge_faces[interior, 0]] * normals[edge_faces[interior, 1]]).sum(axis=1)
    sharp = np.zeros(edges.shape[0], dtype=bool)
    sharp[interior] = np.arccos(np.clip(cos, -1.0, 1.0)) > math.radians(angle_threshold_deg)
    kept = edges[sharp | boundary]

    if min_component_edges > 1 and kept.shape[0]:
        import scipy.sparse as sp
        from scipy.sparse import csgraph

        n = V_ref.shape[0]
        graph = sp.coo_matrix(
            (np.ones(kept.shape[0]), (kept[:, 0], kept[:, 1])), shape=(n, n))
        _num, labels = csgraph.connected_components(graph, directed=False)
        sizes = np.bincount(labels[kept[:, 0]])
        kept = kept[sizes[labels[kept[:, 0]]] >= min_component_edges]

    return kept


def _feature_targets(V, Q, V_ref, segments, verbose=True):
    """Classify quad vertices against the feature curves of the reference mesh.

    Returns:
        ``(on_feature, at_corner, seg_a, seg_b, corner_pos)``. ``on_feature`` and
        ``at_corner`` are (n,) bool masks; ``seg_a``/``seg_b`` are the endpoints
        of the candidate segments for each feature vertex; ``corner_pos`` holds
        the snapped position for each corner vertex.
    """
    from scipy.spatial import cKDTree

    n = V.shape[0]
    empty = (np.zeros(n, dtype=bool), np.zeros(n, dtype=bool), None, None, None)
    if segments.shape[0] == 0:
        return empty

    quad_edge = _median_edge(V, Q)
    a, b = V_ref[segments[:, 0]], V_ref[segments[:, 1]]

    # Resample the curves finely enough that a nearest-sample lookup can only
    # pick a segment whose true distance is close to the real nearest one.
    lengths = np.linalg.norm(b - a, axis=1)
    steps = np.maximum(np.ceil(lengths / max(0.3 * quad_edge, 1e-12)), 1).astype(np.int64)
    seg_id = np.repeat(np.arange(segments.shape[0]), steps + 1)
    t = np.concatenate([np.linspace(0.0, 1.0, s + 1) for s in steps])
    samples = a[seg_id] + t[:, None] * (b - a)[seg_id]

    tree = cKDTree(samples)
    k = min(8, samples.shape[0])
    dist, idx = tree.query(V, k=k)
    if k == 1:
        dist, idx = dist[:, None], idx[:, None]

    on_feature = dist[:, 0] < 0.5 * quad_edge
    cand = seg_id[idx]                                    # (n, k) segment ids
    seg_a, seg_b = a[cand], b[cand]                       # (n, k, 3)

    # Junctions and endpoints of the feature graph pin completely; a corner that
    # slides along one of its curves is still a lost corner.
    verts, counts = np.unique(segments.reshape(-1), return_counts=True)
    corners = verts[counts != 2]
    at_corner = np.zeros(n, dtype=bool)
    corner_pos = np.zeros_like(V)
    if corners.size:
        c_dist, c_idx = cKDTree(V_ref[corners]).query(V)
        # At most one quad vertex per corner, nearest wins; snapping every
        # vertex in the radius collapses the quads between them.
        near = np.argsort(c_dist)
        near = near[c_dist[near] < 0.35 * quad_edge]
        _unique, first = np.unique(c_idx[near], return_index=True)
        at_corner[near[first]] = True
        corner_pos = V_ref[corners][c_idx]
    on_feature &= ~at_corner

    if verbose:
        print(f"  [quadopt] features: {segments.shape[0]} edge(s), "
              f"{int(on_feature.sum())} vertex/vertices on a curve, "
              f"{int(at_corner.sum())} pinned corner(s)")
    return on_feature, at_corner, seg_a, seg_b, corner_pos


def _project_to_segments(P, A, B):
    """Closest point on each of k segments, per point. P (n,3), A/B (n,k,3)."""
    d = B - A
    t = ((P[:, None] - A) * d).sum(axis=2) / np.maximum((d * d).sum(axis=2), 1e-20)
    closest = A + np.clip(t, 0.0, 1.0)[..., None] * d
    best = np.argmin(np.linalg.norm(closest - P[:, None], axis=2), axis=1)
    return closest[np.arange(P.shape[0]), best]


def optimize_quads(
    V,
    Q,
    V_ref=None,
    F_ref=None,
    iters=100,
    scale_blend=1.0,
    step=0.8,
    project_every=1,
    feature_angle=35.0,
    active_verts=None,
    verbose=True,
):
    """
    Make quads square and (optionally) uniformly sized, without changing
    connectivity or valence.

    Args:
        V: (n, 3) quad-mesh vertex positions.
        Q: (m, 4) quad indices.
        V_ref, F_ref: original triangle mesh to project onto. None disables
            projection (the mesh will then shrink slightly as it smooths).
        iters: local/global iterations.
        scale_blend: 0.0 drives every quad toward the average size of its
            neighbours -- this is what removes the "some quads are huge"
            effect. 1.0 keeps each quad's own size and only removes shear.
            The target is local, unlike the look2cross original's global one,
            so a field with free magnitude keeps its sizing.
        step: relaxation factor per iteration, in (0, 1]. Lower is slower but
            more stable on badly distorted input.
        project_every: project back onto the reference surface every k
            iterations.
        feature_angle: normal deviation in degrees above which a reference edge
            is a crease. Vertices on one are projected onto the curve rather
            than the surface, and junction vertices are pinned. Below ~50 an
            organic mesh's tessellation noise starts qualifying. 0 disables.
        active_verts: optional (n,) bool mask. Only these vertices move; the
            rest are pinned. Use to restrict the repair to the neighborhood of
            irregular vertices. None moves everything.
        verbose: print quality before/after.

    Returns:
        (n, 3) float64 optimized vertex positions.
    """
    import igl

    V = np.asarray(V, dtype=np.float64).copy()
    Q = np.asarray(Q, dtype=np.int64)

    # Canonical square corners as complex numbers on the unit circle, in the
    # same cyclic order as the quad's corners.
    U = np.exp(1j * (np.pi / 4 + np.arange(4) * np.pi / 2))   # (4,)
    U_norm2 = float((np.abs(U) ** 2).sum())

    if active_verts is None:
        movable = np.ones(V.shape[0], dtype=bool)
    else:
        movable = np.asarray(active_verts, dtype=bool)

    tree = None
    if V_ref is not None and F_ref is not None:
        V_ref = np.ascontiguousarray(V_ref, dtype=np.float64)
        F_ref = np.ascontiguousarray(F_ref, dtype=np.int32)
        tree = (V_ref, F_ref)

    on_feature = None
    if tree is not None and feature_angle > 0:
        segments = feature_edges(V_ref, F_ref, feature_angle)
        on_feature, at_corner, seg_a, seg_b, corner_pos = _feature_targets(
            V, Q, V_ref, segments, verbose=verbose)
        V[at_corner] = corner_pos[at_corner]
        movable &= ~at_corner

    def quality(Vc):
        """(shear cosine, area spread) summary of current quad quality."""
        P = Vc[Q]
        e = np.stack([P[:, (k + 1) % 4] - P[:, k] for k in range(4)], axis=1)
        el = np.linalg.norm(e, axis=2)
        # Corner right-angle deviation
        cos = np.abs((e[:, [0, 1, 2, 3]] *
                      -e[:, [3, 0, 1, 2]]).sum(axis=2) /
                     np.maximum(el[:, [0, 1, 2, 3]] * el[:, [3, 0, 1, 2]], 1e-20))
        areas = _quad_areas(Vc, Q)
        return float(cos.mean()), float(areas.max() / max(np.median(areas), 1e-20))

    def project(Vc):
        """Surface for the interior, feature curves for feature vertices."""
        _sq, _fi, closest = igl.point_mesh_squared_distance(
            np.ascontiguousarray(Vc), tree[0], tree[1])
        surface = movable.copy()
        if on_feature is not None:
            surface &= ~on_feature
            move = on_feature & movable
            if move.any():
                Vc[move] = _project_to_segments(Vc[move], seg_a[move], seg_b[move])
        Vc[surface] = closest[surface]
        return Vc

    if verbose:
        sh, sp = quality(V)
        print(f"  [quadopt] before: mean|cos(corner)|={sh:.4f}, "
              f"max/median area={sp:.2f}")

    flat_i = Q.reshape(-1)

    for it in range(iters):
        c, t1, t2 = quad_frames(V, Q)
        P = V[Q]
        rel = P - c[:, None, :]
        A = ((rel * t1[:, None, :]).sum(axis=2)
             + 1j * (rel * t2[:, None, :]).sum(axis=2))     # (m, 4) complex

        # Optimal similarity mapping the canonical square onto this quad.
        w = (np.conj(U)[None, :] * A).sum(axis=1) / U_norm2  # (m,) complex
        mag = np.abs(w)
        good = mag > 1e-20
        direction = np.where(good, w / np.maximum(mag, 1e-20), 1.0 + 0j)

        if scale_blend >= 1.0:
            target_mag = mag
        else:
            # Neighbourhood average rather than a global one, so the field's
            # own sizing survives and only outliers are flattened.
            size_acc = np.zeros(V.shape[0])
            size_cnt = np.zeros(V.shape[0])
            np.add.at(size_acc, flat_i, np.repeat(mag, 4))
            np.add.at(size_cnt, flat_i, 1.0)
            local = (size_acc / np.maximum(size_cnt, 1.0))[Q].mean(axis=1)
            target_mag = scale_blend * mag + (1.0 - scale_blend) * local
        w_new = direction * target_mag                        # (m,) complex

        T = w_new[:, None] * U[None, :]                       # (m, 4) complex
        targets = (c[:, None, :]
                   + T.real[:, :, None] * t1[:, None, :]
                   + T.imag[:, :, None] * t2[:, None, :])     # (m, 4, 3)

        # Global step: each vertex goes to the mean of what its quads want.
        acc = np.zeros_like(V)
        cnt = np.zeros(V.shape[0])
        np.add.at(acc, flat_i, targets.reshape(-1, 3))
        np.add.at(cnt, flat_i, 1.0)

        has = cnt > 0
        goal = V.copy()
        goal[has] = acc[has] / cnt[has, None]

        V[movable] = (1.0 - step) * V[movable] + step * goal[movable]

        if tree is not None and (it + 1) % project_every == 0:
            V = project(V)

    if tree is not None:
        V = project(V)

    if verbose:
        sh, sp = quality(V)
        print(f"  [quadopt] after:  mean|cos(corner)|={sh:.4f}, "
              f"max/median area={sp:.2f}  ({iters} iters, "
              f"scale_blend={scale_blend})")

    return V
