from __future__ import annotations

import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F

from crosslift.geometry.mesh import Mesh
from crosslift.rendering.base import RasterOutput, Renderer, RenderOutput
from crosslift.rendering.utils import (
    get_projection_matrix,
    get_projection_matrix_ortho,
    normalize_inverse_depths,
)

_EPS = 1e-8


class NVDRRender(Renderer, method_name="nvdr"):
    """Differentiable rasterization via nvdiffrast."""

    def __init__(
        self,
        near: float = 0.1,
        far: float = 10.0,
        device: torch.device | str | None = None,
        texture_filter: str = "linear-mipmap-linear",
        opengl: bool = False,
    ):
        """
        Args:
            near: Near clipping plane.
            far: Far clipping plane.
            device: Torch device; defaults to CUDA when available.
            texture_filter: nvdiffrast texture filter mode.
            opengl: Use the OpenGL rasterizer context instead of the CUDA one.
                OpenGL needs a display or EGL; CUDA works headless.
        """
        super().__init__()
        if device is not None:
            self.device = torch.device(device)
        self.near = near
        self.far = far
        self.texture_filter = texture_filter
        self.glctx = dr.RasterizeGLContext() if opengl else dr.RasterizeCudaContext()

    # Shared geometry setup
    def _projection_matrix(
        self, intr: torch.Tensor, width: int, height: int, projection_type: str
    ) -> torch.Tensor:
        """``(4, 4)`` projection matrix for the requested projection type.

        Both helpers take ``(intrinsics, width, height)``.
        """
        if projection_type == "orthographic":
            builder = get_projection_matrix_ortho
        elif projection_type == "perspective":
            builder = get_projection_matrix
        else:
            raise ValueError(
                f"Unknown projection type {projection_type!r}; "
                f"expected 'perspective' or 'orthographic'"
            )
        return builder(intr, width, height, near=self.near, far=self.far)

    def _to_clip(
        self,
        mesh: Mesh,
        w2c: torch.Tensor,
        intr: torch.Tensor,
        height: int,
        width: int,
        projection_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """World -> camera -> clip, shared by :meth:`render` and :meth:`rasterize`.

        Returns:
            ``(vert_cam, vert_clip)``, each ``(B, V, 4)`` homogeneous.
        """
        if mesh.valence != 3:
            raise ValueError(
                f"nvdiffrast rasterizes triangles only, but the mesh has faces of "
                f"valence {mesh.valence}. Triangulate first, or use the "
                f"'polyscope' backend for quad meshes."
            )

        num_views = w2c.shape[0]
        vert_h = F.pad(mesh.vertices, (0, 1), mode="constant", value=1.0)
        vert_cam = vert_h @ w2c.transpose(-1, -2)  # (B, V, 4)

        proj = self._projection_matrix(intr, width, height, projection_type)
        proj = proj[None].expand(num_views, -1, -1)
        vert_clip = vert_cam @ proj.transpose(-1, -2)
        return vert_cam, vert_clip

    # Forward render
    def render(
        self,
        mesh: Mesh,
        w2c: torch.Tensor,
        intr: torch.Tensor,
        height: int = 1024,
        width: int = 1024,
        lighting: bool = True,
        draw_edges: bool = False,
        projection_type: str = "perspective",
        cam_space_lighting: bool = True,
        edge_width: float = 1.0,
        aa: bool = True,
        **kwargs,
    ) -> RenderOutput:
        """Render shaded images of ``mesh`` from every camera in ``w2c``.

        Args:
            mesh: Triangle mesh to render.
            w2c: ``(B, 4, 4)`` world-to-camera matrices.
            intr: ``(4,)`` intrinsics ``[fx, fy, cx, cy]``.
            height: Image height in pixels.
            width: Image width in pixels.
            lighting: Apply Lambertian shading on top of the albedo.
            draw_edges: Overlay wireframe edges.
            projection_type: ``'perspective'`` or ``'orthographic'``.
            cam_space_lighting: Light moves with the camera rather than the world.
            edge_width: Wireframe width in pixels, when ``draw_edges``.
            aa: Apply nvdiffrast antialiasing.

        Returns:
            A :class:`RenderOutput` with channel-first ``(B, C, H, W)`` tensors.
        """
        num_views = w2c.shape[0]
        vert_cam, vert_clip = self._to_clip(
            mesh, w2c, intr, height, width, projection_type
        )

        # Barycentric derivatives are only needed for wireframes, texturing and
        # backprop; skipping them is a meaningful saving on the common path.
        needs_grad_db = (
            draw_edges or torch.is_grad_enabled() or mesh.texture_map is not None
        )
        rast, rast_db = dr.rasterize(
            self.glctx, vert_clip, mesh.faces, (height, width), grad_db=needs_grad_db
        )

        fg = rast[..., 3] > 0                       # (B, H, W)
        alpha = fg.float().unsqueeze(-1)

        # Depth as inverse camera-space z, bounded and comparable across views
        inv_z = 1.0 / (-vert_cam[..., 2:3].contiguous() + _EPS)
        depth = dr.interpolate(inv_z, rast, mesh.faces)[0].reshape(
            num_views, height, width
        )
        depth.masked_fill_(~fg, 0)

        normal = dr.interpolate(
            mesh.vertex_normals.contiguous(), rast, mesh.faces
        )[0].reshape(num_views, height, width, 3)
        normal = F.normalize(normal, dim=-1)
        normal[~fg] = normal.new_tensor([0.5, 0.5, 1.0])
        w2c_rotation = w2c[..., :3, :3]

        albedo = self._shade_albedo(mesh, rast, rast_db, normal, num_views)
        if lighting:
            albedo = self._apply_lighting(
                albedo, normal, w2c_rotation, num_views, cam_space_lighting
            )
        if draw_edges:
            albedo = self._draw_edges(albedo, rast, rast_db, fg, edge_width)

        albedo.masked_fill_(~fg.unsqueeze(-1), 1.0)  # white background
        rgba = torch.cat([albedo, alpha], dim=-1)

        if aa:
            rgba, depth, normal = dr.antialias(
                torch.cat([rgba, depth.unsqueeze(-1), normal], dim=-1),
                rast, vert_clip, mesh.faces,
            ).split([4, 1, 3], dim=-1)
            depth = depth.squeeze(-1)

        renders = rgba[..., :3].permute(0, 3, 1, 2)
        masks = rgba[..., 3:].permute(0, 3, 1, 2)

        depths = depth.unsqueeze(1).expand(-1, 3, -1, -1)
        depths = normalize_inverse_depths(depths, masks)

        # Guidance models expect camera-space normals, so rotate out of world space.
        world_normals = F.normalize(normal, dim=-1).permute(0, 3, 1, 2)
        normal_w2c = torch.inverse(w2c[:, :3, :3]).transpose(-1, -2)
        cam_normals = torch.matmul(
            normal_w2c, world_normals.reshape(num_views, 3, height * width)
        ).reshape(num_views, 3, height, width)
        cam_normals = F.normalize(cam_normals, p=2, dim=1)
        cam_normals = torch.where(masks == 1, cam_normals, -1.0)
        normals = (cam_normals + 1) / 2  # [-1, 1] -> [0, 1] for visualization
        world_normals = torch.where(masks == 1, world_normals, -1.0)
        world_normals = (world_normals + 1) / 2

        return RenderOutput(
            renders=renders, depths=depths, normals=normals,
            world_normals=world_normals, masks=masks
        )

    def _shade_albedo(
        self,
        mesh: Mesh,
        rast: torch.Tensor,
        rast_db: torch.Tensor,
        normal: torch.Tensor,
        num_views: int,
    ) -> torch.Tensor:
        """Sample the texture map if the mesh has one, else a flat grey."""
        if mesh.uvs is None or mesh.texture_map is None:
            return torch.zeros_like(normal) + 0.6

        uv_faces = mesh.face_uvs if mesh.face_uvs is not None else mesh.faces
        uvs, _ = dr.interpolate(mesh.uvs.contiguous(), rast, uv_faces, rast_db=rast_db)

        texture = mesh.texture_map
        if texture.dim() == 3:
            texture = texture.unsqueeze(0)
        if texture.shape[0] == 1 and num_views > 1:
            texture = texture.expand(num_views, -1, -1, -1)
        return dr.texture(texture.contiguous(), uvs)

    @staticmethod
    def _apply_lighting(
        albedo: torch.Tensor,
        normal: torch.Tensor,
        w2c_rotation: torch.Tensor,
        num_views: int,
        cam_space_lighting: bool,
    ) -> torch.Tensor:
        """Lambertian shading with a fixed ambient term."""
        light_dir = torch.tensor([0.0, 1.0, 1.0], device=normal.device, dtype=normal.dtype)
        light_dir = light_dir / light_dir.norm()
        light_color = torch.tensor([1.0, 1.0, 1.0], device=normal.device, dtype=normal.dtype)
        light_intensity, ambient = 0.5, 0.5

        if cam_space_lighting:  # light rides with the camera
            light_dir = light_dir[None].expand(num_views, -1)[:, None, None, :]
            normal_cam = normal @ w2c_rotation.transpose(-1, -2).unsqueeze(1)
            lambertian = (normal_cam * light_dir).sum(dim=-1, keepdim=True).clamp_min(0.0)
        else:
            lambertian = (normal * light_dir).sum(dim=-1, keepdim=True).clamp_min(0.0)

        return albedo * (ambient + lambertian * light_intensity * light_color)

    @staticmethod
    def _draw_edges(
        albedo: torch.Tensor,
        rast: torch.Tensor,
        rast_db: torch.Tensor,
        fg: torch.Tensor,
        edge_width: float,
    ) -> torch.Tensor:
        """Blacken pixels within ``edge_width`` of a triangle edge.

        Distance to an edge is the barycentric coordinate divided by its
        screen-space rate of change, in pixels.
        """
        u, v = rast[..., 0], rast[..., 1]
        w_bary = 1.0 - u - v

        du_dx, du_dy = rast_db[..., 0], rast_db[..., 1]
        dv_dx, dv_dy = rast_db[..., 2], rast_db[..., 3]
        u_rate = torch.sqrt(du_dx**2 + du_dy**2) + _EPS
        v_rate = torch.sqrt(dv_dx**2 + dv_dy**2) + _EPS
        w_rate = torch.sqrt((du_dx + dv_dx) ** 2 + (du_dy + dv_dy) ** 2) + _EPS

        min_dist = torch.min(
            torch.stack([u / u_rate, v / v_rate, w_bary / w_rate], dim=-1), dim=-1
        )[0]
        albedo[(min_dist < edge_width) & fg] = 0.0
        return albedo

    # Rasterization for lifting
    def rasterize(
        self,
        mesh: Mesh,
        w2c: torch.Tensor,
        intr: torch.Tensor,
        height: int = 1024,
        width: int = 1024,
        projection_type: str = "perspective",
        **kwargs,
    ) -> RasterOutput:
        """Rasterize ``mesh`` and report which face each pixel landed on.

        No shading, no antialiasing, no gradients — this exists to feed
        ``fields.lifting``, which needs correspondence rather than an image.

        Args:
            mesh: Triangle mesh to rasterize.
            w2c: ``(B, 4, 4)`` world-to-camera matrices.
            intr: ``(4,)`` intrinsics ``[fx, fy, cx, cy]``.
            height: Image height in pixels.
            width: Image width in pixels.
            projection_type: ``'perspective'`` or ``'orthographic'``.

        Returns:
            A :class:`RasterOutput` carrying the correspondence and the camera
            parameters it was produced with.
        """
        num_views = w2c.shape[0]
        vert_cam, vert_clip = self._to_clip(
            mesh, w2c, intr, height, width, projection_type
        )

        rast, _ = dr.rasterize(
            self.glctx, vert_clip, mesh.faces, (height, width), grad_db=False
        )

        fg_mask = rast[..., 3] > 0                       # (B, H, W)
        # rast[..., 3] holds triangle_id + 1, 0 = background; clamp rather than
        # -1 to keep face_id safe to index with. fg_mask is the only valid gate.
        face_id = (rast[..., 3].long() - 1).clamp_min(0)

        normals = F.normalize(
            dr.interpolate(mesh.vertex_normals.contiguous(), rast, mesh.faces)[0],
            dim=-1,
        )
        pixel_pos_cam = dr.interpolate(
            vert_cam.contiguous(), rast, mesh.faces
        )[0][..., :3]

        # A face is visible in a view if any pixel hit it (scatter, not per-view torch.unique)
        visible_face_mask = torch.zeros(
            (num_views, mesh.num_faces), dtype=torch.bool, device=face_id.device
        )
        view_idx = torch.arange(num_views, device=face_id.device)
        view_idx = view_idx[:, None, None].expand_as(face_id)
        visible_face_mask[view_idx[fg_mask], face_id[fg_mask]] = True

        return RasterOutput(
            face_id=face_id,
            fg_mask=fg_mask,
            normals=normals,
            pixel_pos_cam=pixel_pos_cam,
            visible_face_mask=visible_face_mask,
            w2c=w2c,
            intr=intr,
            projection_type=projection_type,
        )
