from __future__ import annotations

import copy
import logging
import math
import os
from functools import cached_property

import igl
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.sparse import csgraph

logger = logging.getLogger(__name__)
_EPS = 1e-12
# Default sharpness threshold in radians (~84.3 degrees of normal deviation).
DEFAULT_SHARP_ANGLE = 1.47062890563

def _read_mesh(
    path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Read a mesh from disk, preserving face valence.

    Args:
        path: Path to a mesh file. ``.obj`` additionally yields UVs if present.

    Returns:
        ``(vertices, faces, uvs, face_uvs)`` where ``vertices`` is ``(V, 3)``
        float64, ``faces`` is ``(F, K)`` int64 with ``K in {3, 4}``, and the UV
        arrays are ``(T, 2)`` / ``(F, K)`` or ``None`` when absent.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mesh file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".obj":
        try:
            verts, tex_coords, _, faces, face_tex, _ = igl.readOBJ(path)
        except Exception as exc:  # libigl raises on mixed tri/quad OBJs
            print(exc)
            raise ValueError(
                f"libigl could not read {path} with error: {exc}. Meshes with mixed face valence "
                f"(triangles and quads in one file) are not supported."
            ) from exc
        uvs = tex_coords if tex_coords.size else None
        face_uvs = face_tex if (uvs is not None and face_tex.size) else None
    else:
        verts, faces = igl.read_triangle_mesh(path)
        uvs = face_uvs = None

    if faces.ndim != 2 or faces.shape[0] == 0:
        raise ValueError(f"Mesh at {path} contains no faces")
    if faces.shape[1] not in (3, 4):
        raise ValueError(
            f"Unsupported face valence {faces.shape[1]} in {path}; "
            f"expected triangles or quads"
        )

    return verts, faces.astype(np.int64), uvs, face_uvs


def _load_texture(path: str, device: torch.device) -> torch.Tensor | None:
    """Load the sibling ``*_texture.png`` for an OBJ, if one exists.

    Returns:
        An ``(H, W, 3)`` float tensor in ``[0, 1]``, or ``None`` if no texture
        file sits next to the mesh.
    """
    texture_path = path.replace(".obj", "_texture.png")
    if not os.path.exists(texture_path):
        return None

    from PIL import Image  # Imported lazily to keep module import cheap.

    image = Image.open(texture_path).convert("RGB")
    logger.info("Loaded texture from %s", texture_path)
    return torch.as_tensor(
        np.asarray(image), dtype=torch.float32, device=device
    ).div_(255.0)

class Mesh:
    """
    Attributes:
        vertices: ``(V, 3)`` float32 tensor of positions.
        faces: ``(F, K)`` int32 tensor of vertex indices, ``K in {3, 4}``.
        uvs: ``(T, 2)`` float32 tensor of texture coordinates, or ``None``.
        face_uvs: ``(F, K)`` int32 tensor indexing into ``uvs``, or ``None``.
        texture_map: ``(H, W, 3)`` float32 tensor in ``[0, 1]``, or ``None``.
        device: The torch device holding all tensor attributes.
    """

    def __init__(
        self,
        path: str,
        device: torch.device | str,
        *,
        flip_yz: bool = False,
        flip_normals: bool = False,
        quad_mesh: bool = False,
        load_texture: bool = False,
    ):
        """Load a mesh from disk.

        Args:
            path: Path to the mesh file.
            device: Torch device for the tensor attributes.
            flip_yz: Swap the Y and Z axes, for assets authored Z-up.
            flip_normals: Reverse face winding, flipping all normals.
            quad_mesh: Assert that the loaded mesh is quads. Loading preserves
                whatever valence the file has; this only validates the
                expectation early, rather than failing deep in the pipeline.
            load_texture: Also load the sibling ``*_texture.png``.
        """
        self.path = path
        self.device = torch.device(device)

        verts, faces, uvs, face_uvs = _read_mesh(path)

        if flip_yz:
            verts = verts[:, [0, 2, 1]]
        if flip_normals:
            faces = np.ascontiguousarray(faces[:, ::-1])
            if face_uvs is not None:
                face_uvs = np.ascontiguousarray(face_uvs[:, ::-1])

        if quad_mesh and faces.shape[1] != 4:
            raise ValueError(
                f"Expected a quad mesh at {path}, but loaded faces of valence "
                f"{faces.shape[1]}"
            )

        self.vertices = torch.as_tensor(verts, dtype=torch.float32, device=self.device)
        self.faces = torch.as_tensor(faces, dtype=torch.int32, device=self.device)

        self.uvs = (
            None if uvs is None
            else torch.as_tensor(uvs, dtype=torch.float32, device=self.device)
        )
        self.face_uvs = (
            None if face_uvs is None
            else torch.as_tensor(face_uvs, dtype=torch.int32, device=self.device)
        )
        self.texture_map = _load_texture(path, self.device) if load_texture else None

        logger.info(
            "Loaded %s: %d vertices, %d faces (valence %d)%s",
            os.path.basename(path), self.num_vertices, self.num_faces,
            self.valence, "" if self.uvs is None else ", with UVs",
        )

    @classmethod
    def from_arrays(
        cls,
        vertices: torch.Tensor | np.ndarray,
        faces: torch.Tensor | np.ndarray,
        device: torch.device | str = "cpu",
    ) -> Mesh:
        """Build a mesh directly from arrays, bypassing file I/O.
        """
        mesh = cls.__new__(cls)
        mesh.path = None
        mesh.device = torch.device(device)
        mesh.vertices = torch.as_tensor(
            vertices, dtype=torch.float32, device=mesh.device
        )
        mesh.faces = torch.as_tensor(faces, dtype=torch.int32, device=mesh.device)
        mesh.uvs = None
        mesh.face_uvs = None
        mesh.texture_map = None
        return mesh

    def __repr__(self) -> str:
        name = "in-memory" if self.path is None else os.path.basename(self.path)
        return (
            f"Mesh({name}, V={self.num_vertices}, F={self.num_faces}, "
            f"valence={self.valence}, device={self.device})"
        )

    # Basic properties
    @property
    def num_vertices(self) -> int:
        return self.vertices.shape[0]

    @property
    def num_faces(self) -> int:
        return self.faces.shape[0]

    @property
    def valence(self) -> int:
        """Vertices per face: 3 for triangles, 4 for quads."""
        return self.faces.shape[1]

    @property
    def is_quad(self) -> bool:
        return self.valence == 4

    @cached_property
    def _faces_long(self) -> torch.Tensor:
        """``faces`` as int64, required by ``index_add_`` and advanced indexing."""
        return self.faces.long()

    @cached_property
    def _faces_np(self) -> np.ndarray:
        return self.faces.detach().cpu().numpy().astype(np.int64)

    @property
    def _vertices_np(self) -> np.ndarray:
        """Host copy of the vertices.
        """
        return self.vertices.detach().cpu().numpy().astype(np.float64)

    def _invalidate(self, *names: str) -> None:
        """Drop cached properties so they recompute on next access."""
        for name in names:
            self.__dict__.pop(name, None)

    # Per-face geometry
    @cached_property
    def _face_vector_areas(self) -> torch.Tensor:
        """``(F, 3)`` area-weighted face normals
        """
        corners = self.vertices[self._faces_long]  # (F, K, 3)
        return 0.5 * torch.cross(corners, torch.roll(corners, -1, dims=1), dim=-1).sum(1)

    @cached_property
    def face_areas(self) -> torch.Tensor:
        """``(F,)`` face areas."""
        return self._face_vector_areas.norm(dim=-1)

    @cached_property
    def face_normals(self) -> torch.Tensor:
        """``(F, 3)`` unit face normals."""
        return F.normalize(self._face_vector_areas, dim=-1, eps=_EPS)

    @cached_property
    def vertex_normals(self) -> torch.Tensor:
        """``(V, 3)`` unit vertex normals, area-weighted over incident faces."""
        accumulated = torch.zeros_like(self.vertices)
        accumulated.index_add_(
            0,
            self._faces_long.reshape(-1),
            self._face_vector_areas.repeat_interleave(self.valence, dim=0),
        )
        return F.normalize(accumulated, dim=-1, eps=_EPS)

    @cached_property
    def face_basis(self) -> torch.Tensor:
        """``(F, 3, 2)`` orthonormal tangent basis per face.
        """
        normals = self.face_normals
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(normals)
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand_as(normals)
        # Pick whichever helper axis is least aligned with the normal.
        helper = torch.where((normals[..., 2].abs() > 0.9)[..., None], x_axis, z_axis)
        tangent = F.normalize(torch.cross(normals, helper, dim=-1), dim=-1, eps=_EPS)
        bitangent = torch.cross(normals, tangent, dim=-1)
        return torch.stack([tangent, bitangent], dim=-1)

    def get_face_basis(self) -> torch.Tensor:
        """Alias for :attr:`face_basis`, kept for call-site compatibility."""
        return self.face_basis

    def get_face_centroids(self) -> torch.Tensor:
        """``(F, 3)`` face centroids."""
        return self.vertices[self._faces_long].mean(dim=1)

    # Topology
    @cached_property
    def _edge_topology(self) -> tuple[np.ndarray, np.ndarray]:
        """Build the unique edge list and edge-to-face map.
        Returns:
            ``(edges, edge_faces)`` where ``edges`` is ``(E, 2)`` of sorted
            vertex-index pairs and ``edge_faces`` is ``(E, 2)`` of incident face
            indices, with ``-1`` marking a boundary (only one incident face).
        """
        faces = self._faces_np
        n_faces, valence = faces.shape

        corner_pairs = np.stack([faces, np.roll(faces, -1, axis=1)], axis=-1)
        pairs = np.sort(corner_pairs.reshape(-1, 2), axis=1)
        edges, inverse = np.unique(pairs, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)

        counts = np.bincount(inverse, minlength=edges.shape[0])
        if counts.max(initial=0) > 2:
            offenders = edges[counts > 2][:5].tolist()
            raise ValueError(
                f"Non-manifold mesh: {int((counts > 2).sum())} edge(s) have more "
                f"than 2 incident faces, e.g. {offenders}"
            )

        # Scatter each corner's face index into slot 0 or slot 1 of its edge.
        corner_faces = np.repeat(np.arange(n_faces, dtype=np.int64), valence)
        order = np.argsort(inverse, kind="stable")
        sorted_edges, sorted_faces = inverse[order], corner_faces[order]

        edge_faces = np.full((edges.shape[0], 2), -1, dtype=np.int64)
        if sorted_edges.size:
            is_first = np.empty(sorted_edges.shape[0], dtype=bool)
            is_first[0] = True
            is_first[1:] = sorted_edges[1:] != sorted_edges[:-1]
            edge_faces[sorted_edges[is_first], 0] = sorted_faces[is_first]
            edge_faces[sorted_edges[~is_first], 1] = sorted_faces[~is_first]

        return edges.astype(np.int64), edge_faces

    @property
    def edges(self) -> np.ndarray:
        """``(E, 2)`` unique undirected edges as sorted vertex-index pairs."""
        return self._edge_topology[0]

    @property
    def edge_faces(self) -> np.ndarray:
        """``(E, 2)`` incident face indices per edge; ``-1`` marks a boundary."""
        return self._edge_topology[1]

    @cached_property
    def boundary_edge_mask(self) -> np.ndarray:
        """``(E,)`` bool mask, True where an edge has exactly one incident face."""
        incident = self.edge_faces >= 0
        return incident[:, 0] != incident[:, 1]

    @cached_property
    def _components(self) -> tuple[int, np.ndarray]:
        """Connected components of the face-adjacency graph.
        """
        face_a, face_b = self.edge_faces[:, 0], self.edge_faces[:, 1]
        interior = (face_a >= 0) & (face_b >= 0)
        rows = np.concatenate([face_a[interior], face_b[interior]])
        cols = np.concatenate([face_b[interior], face_a[interior]])
        adjacency = sp.csr_matrix(
            (np.ones(rows.shape[0]), (rows, cols)),
            shape=(self.num_faces, self.num_faces),
        )
        num, labels = csgraph.connected_components(adjacency, directed=False)
        return int(num), labels.astype(np.int64)

    @property
    def num_components(self) -> int:
        """Number of connected components in the face-adjacency graph."""
        return self._components[0]

    @property
    def face_components(self) -> np.ndarray:
        """``(F,)`` int64 component label per face."""
        return self._components[1]

    # Discrete differential quantities
    @cached_property
    def harmonic_weights(self) -> np.ndarray:
        """``(E,)`` harmonic edge weights ``3 * L^2 / (A_1 + A_2)``.

        Boundary edges get weight zero.
        """
        verts = self._vertices_np
        edges, edge_faces = self.edges, self.edge_faces
        lengths = np.linalg.norm(verts[edges[:, 1]] - verts[edges[:, 0]], axis=-1)
        areas = self.face_areas.detach().cpu().numpy().astype(np.float64)

        face_a, face_b = edge_faces[:, 0], edge_faces[:, 1]
        interior = (face_a >= 0) & (face_b >= 0)

        weights = np.zeros(edges.shape[0], dtype=np.float64)
        denominator = (
            np.maximum(areas[face_a[interior]], _EPS)
            + np.maximum(areas[face_b[interior]], _EPS)
        )
        weights[interior] = 3.0 * lengths[interior] ** 2 / denominator
        return weights

    @cached_property
    def edge_connections(self) -> np.ndarray:
        """``(E, 2)`` complex edge directions in each incident face's basis.
        """
        verts = self._vertices_np
        basis = self.face_basis.detach().cpu().numpy().astype(np.float64)
        edge_basis = basis[self.edge_faces]  # (E, 2, 3, 2)

        directions = verts[self.edges[:, 1]] - verts[self.edges[:, 0]]
        norms = np.linalg.norm(directions, axis=-1, keepdims=True)
        directions = directions / np.maximum(norms, _EPS)
        directions = directions[:, None, :]  # (E, 1, 3)

        real = np.sum(edge_basis[..., 0] * directions, axis=-1)  # (E, 2)
        imag = np.sum(edge_basis[..., 1] * directions, axis=-1)  # (E, 2)
        return real + 1j * imag

    @cached_property
    def dihedral_angles(self) -> torch.Tensor:
        """``(E,)`` angle between incident face normals; zero on boundary edges.
        """
        edge_faces = torch.as_tensor(self.edge_faces, device=self.device)
        face_a, face_b = edge_faces[:, 0], edge_faces[:, 1]
        interior = (face_a >= 0) & (face_b >= 0)

        angles = torch.zeros(edge_faces.shape[0], device=self.device)
        normals_a = self.face_normals[face_a[interior]]
        normals_b = self.face_normals[face_b[interior]]
        cos_theta = torch.clamp((normals_a * normals_b).sum(dim=1), -1.0, 1.0)
        angles[interior] = torch.acos(cos_theta)
        return angles

    # Field constraints
    def _pack_constraints(
        self, face_indices: np.ndarray, directions: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize and move a set of per-face N-RoSy constraints to the device."""
        directions = directions / np.maximum(np.abs(directions), _EPS)
        return (
            torch.as_tensor(
                np.ascontiguousarray(face_indices),
                dtype=torch.long, device=self.device,
            ),
            torch.from_numpy(
                np.ascontiguousarray(directions, dtype=np.complex64)
            ).to(self.device),
        )

    def get_sharp_edge_constraints(
        self, N: int, angle_threshold: float = DEFAULT_SHARP_ANGLE
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """N-RoSy constraints aligning the field to sharp creases.

        Args:
            N: N-RoSy symmetry order.
            angle_threshold: Normal-deviation threshold in radians. Edges above
                it are treated as sharp.

        Returns:
            ``(face_indices, directions)`` of shape ``(M,)``
        """
        sharp = torch.nonzero(
            self.dihedral_angles > angle_threshold, as_tuple=False
        ).flatten().cpu().numpy()

        incident_faces = self.edge_faces[sharp]  # (S, 2)
        valid = incident_faces >= 0
        face_indices = incident_faces[valid]
        directions = (self.edge_connections[sharp] ** N)[valid]

        logger.info(
            "Found %d sharp edges (threshold %.1f deg) -> %d face constraints",
            sharp.size, math.degrees(angle_threshold), face_indices.size,
        )
        return self._pack_constraints(face_indices, directions)

    def get_boundary_edge_constraints(self, N: int) -> tuple[torch.Tensor, torch.Tensor]:
        """N-RoSy constraints aligning the field to the mesh boundary.

        Args:
            N: N-RoSy symmetry order.

        Returns:
            ``(face_indices, directions)`` of shape ``(B,)``
        """
        boundary = np.flatnonzero(self.boundary_edge_mask)
        face_a, face_b = self.edge_faces[boundary, 0], self.edge_faces[boundary, 1]
        slots = np.where(face_b < 0, 0, 1)
        faces = np.where(face_b < 0, face_a, face_b)

        _, first_occurrence = np.unique(faces, return_index=True)
        keep = np.sort(first_occurrence)

        directions = self.edge_connections[boundary[keep], slots[keep]] ** N
        logger.info("Found %d boundary-edge faces to hard-constrain.", keep.size)
        return self._pack_constraints(faces[keep], directions)

    # Sampling
    def sample_blue_noise(
        self,
        num_samples: int | None = None,
        min_radius: float | None = None,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Farthest-point sample face centroids for blue-noise coverage.

        Args:
            num_samples: Stop after this many faces
            min_radius: Stop once no unsampled face is farther than this from
                the sampled set.
            seed: Seed for the starting face

        Returns:
            ``(mask, indices)`` where ``mask`` is a ``(F,)`` bool tensor and
            ``indices`` is a long tensor of the sampled faces in sampling order.
        """
        if (num_samples is None) == (min_radius is None):
            raise ValueError(
                "Provide exactly one of num_samples or min_radius "
                f"(got num_samples={num_samples}, min_radius={min_radius})"
            )

        centroids = self.get_face_centroids()
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        start = int(
            torch.randint(
                self.num_faces, (1,), device=self.device, generator=generator
            ).item()
        )

        sampled = torch.zeros(self.num_faces, dtype=torch.bool, device=self.device)
        sampled[start] = True
        dist2 = torch.sum((centroids - centroids[start]) ** 2, dim=-1)
        dist2[sampled] = -1.0

        def _accept(index: int) -> None:
            sampled[index] = True
            torch.minimum(
                dist2,
                torch.sum((centroids - centroids[index]) ** 2, dim=-1),
                out=dist2,
            )
            dist2[sampled] = -1.0

        picked = [start]
        if num_samples is not None:
            if num_samples <= 0:
                raise ValueError(f"num_samples must be positive, got {num_samples}")
            budget = min(num_samples, self.num_faces)
            while len(picked) < budget:
                index = int(torch.argmax(dist2).item())
                if dist2[index] < 0:  # every face already sampled
                    break
                picked.append(index)
                _accept(index)
        else:
            threshold = min_radius ** 2
            while True:
                index = int(torch.argmax(dist2).item())
                if dist2[index] < threshold:
                    break
                picked.append(index)
                _accept(index)

        indices = torch.tensor(picked, dtype=torch.long, device=self.device)
        return sampled, indices

    # Transforms
    def normalize_mesh(
        self, inplace: bool = False, target_scale: float = 1.0, dy: float = 0.0,
        center: torch.Tensor | None = None, scale: torch.Tensor | None = None,
    ) -> Mesh:
        """Center the mesh at the origin and scale it to fit a unit sphere.

        Args:
            inplace: Modify this mesh inplace.
            target_scale: Radius of the bounding sphere after scaling.
            dy: Vertical offset applied after scaling.
            center: Origin to subtract, or None for this mesh's vertex mean.
            scale: Radius to divide by, or None for this mesh's bounding radius.

        Returns:
            The normalized mesh (``self`` when ``inplace``).
        """
        mesh = self if inplace else copy.deepcopy(self)

        verts = mesh.vertices - (mesh.vertices.mean(dim=0) if center is None else center)
        if scale is None:
            scale = verts.norm(dim=1).max()
        verts = verts * (target_scale / scale.clamp_min(_EPS))
        verts[:, 1] += dy
        mesh.vertices = verts

        mesh._invalidate("_face_vector_areas", "face_areas")
        return mesh

    def _apply_rotation(self, rotation: torch.Tensor, inplace: bool) -> Mesh:
        """Rotate every 3D-vector-valued attribute by ``rotation``.
        """
        mesh = self if inplace else copy.deepcopy(self)
        transposed = rotation.T

        mesh.vertices = mesh.vertices @ transposed
        for name in ("vertex_normals", "face_normals", "_face_vector_areas"):
            cached = mesh.__dict__.get(name)
            if cached is not None:
                mesh.__dict__[name] = cached @ transposed

        basis = mesh.__dict__.get("face_basis")
        if basis is not None:
            mesh.__dict__["face_basis"] = torch.einsum("ij,fjk->fik", rotation, basis)

        return mesh

    def _axis_rotation(self, axis: str, degrees: float) -> torch.Tensor:
        """Build a 3x3 rotation matrix about a principal axis."""
        angle = math.radians(degrees)
        cos, sin = math.cos(angle), math.sin(angle)
        matrices = {
            "x": [[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]],
            "y": [[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]],
            "z": [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]],
        }
        if axis not in matrices:
            raise ValueError(f"Unknown rotation axis {axis!r}; expected x, y or z")
        return torch.tensor(matrices[axis], dtype=torch.float32, device=self.device)

    def rotate_mesh_basic(
        self, azim: float, elev: float, inplace: bool = False
    ) -> Mesh:
        """Rotate by an azimuth about Y, then an elevation about X.

        Args:
            azim: Azimuth in degrees (rotation about the Y axis).
            elev: Elevation in degrees (rotation about the X axis).
            inplace: Modify this mesh rather than returning a copy.
        """
        rotation = self._axis_rotation("x", elev) @ self._axis_rotation("y", azim)
        return self._apply_rotation(rotation, inplace)

    def rotate_mesh_euler(
        self,
        angles: tuple[float, float, float],
        order: str = "zxy",
        inplace: bool = False,
    ) -> Mesh:
        """Rotate by Euler angles applied in a given order.

        Args:
            angles: ``(angle_x, angle_y, angle_z)`` in degrees.
            order: Order of application, e.g. ``'zxy'`` rotates about Z first,
                then X, then Y.
            inplace: Modify this mesh rather than returning a copy.
        """
        by_axis = dict(zip("xyz", angles))
        rotation = torch.eye(3, dtype=torch.float32, device=self.device)
        for axis in reversed(order):
            rotation = rotation @ self._axis_rotation(axis, by_axis[axis])
        return self._apply_rotation(rotation, inplace)
