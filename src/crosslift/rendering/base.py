"""Renderer interface and the data types every backend returns.

Backends subclass :class:`Renderer` with a ``method_name`` and are constructed
through :meth:`Renderer.create`, which imports the backend module lazily. The
laziness matters: importing the nvdiffrast backend builds a CUDA rasterizer
context, which a polyscope-only run should not pay for.

This module is a leaf — it depends on ``geometry`` and ``rendering.utils`` and
nothing else. In particular it must never import from ``fields``.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from crosslift.geometry.mesh import Mesh
from crosslift.rendering.utils import (
    compute_tight_fit_distance,
    get_camera_intrinsics,
)

# Maps a backend name to the module that defines it, so `create` can import on
# demand. A backend registers itself in `_registry` via __init_subclass__ the
# moment its module is imported.
_BACKEND_MODULES = {
    "nvdr": "crosslift.rendering.nvdr",
    "polyscope": "crosslift.rendering.polyscope",
}


@dataclass
class RenderOutput:
    """Shaded images, channel-first and batched over views."""
    renders: torch.Tensor                  # (B, 3, H, W) RGB in [0, 1]
    depths: torch.Tensor | None = None     # (B, 3, H, W) normalized inverse depth
    normals: torch.Tensor | None = None    # (B, 3, H, W) camera-space, mapped to [0, 1]
    # Same normals in world space. Camera-space normals are view-invariant — a
    # surface facing the camera is (0.5, 0.5, 1) in every view — so they carry no
    # cue for which view a cell is, nor for which cells show the same surface.
    # World space encodes both, which is what multi-view guidance needs.
    world_normals: torch.Tensor | None = None  # (B, 3, H, W) world-space, mapped to [0, 1]
    masks: torch.Tensor | None = None      # (B, 1, H, W) foreground coverage
    extras: dict[str, Any] | None = None

@dataclass
class RasterOutput:
    """Pixel-to-surface correspondence — what lifting needs, and nothing more.

    Attributes:
        face_id: ``(B, H, W)`` long. Index of the face covering each pixel.
            Meaningless where ``fg_mask`` is False: it is clamped to 0 there
            rather than set to -1, so it stays safe to use as an index.
        fg_mask: ``(B, H, W)`` bool. True where a face was hit.
        normals: ``(B, H, W, 3)`` unit world-space normals, interpolated from
            the vertex normals.
        pixel_pos_cam: ``(B, H, W, 3)`` camera-space position of each pixel's
            surface point. The perspective Jacobian needs it.
        visible_face_mask: ``(B, F)`` bool. True where a face was hit by at
            least one pixel in that view.

    The camera parameters ride along so that lifting is a pure function of this
    object plus the mesh: ``lift_vecs(grads, mesh, raster)`` needs no separate
    camera arguments, and cannot be handed a raster and a mismatched camera by
    accident.
    """

    face_id: torch.Tensor
    fg_mask: torch.Tensor
    normals: torch.Tensor
    pixel_pos_cam: torch.Tensor
    visible_face_mask: torch.Tensor
    w2c: torch.Tensor                      # (B, 4, 4) world-to-camera
    intr: torch.Tensor                     # (4,) [fx, fy, cx, cy]
    projection_type: str

    @property
    def num_views(self) -> int:
        return self.face_id.shape[0]

    @property
    def image_size(self) -> tuple[int, int]:
        """``(height, width)`` of the rasterized images."""
        return self.face_id.shape[1], self.face_id.shape[2]


class Renderer(ABC):
    """Base class and registry for rendering backends."""

    _registry: ClassVar[dict[str, type[Renderer]]] = {}

    def __init_subclass__(cls, method_name: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if method_name:
            cls._registry[method_name] = cls

    @classmethod
    def create(cls, method_name: str, **kwargs) -> Renderer:
        """Construct a backend by name, importing its module on first use."""
        if method_name not in cls._registry:
            module_name = _BACKEND_MODULES.get(method_name)
            if module_name is None:
                raise ValueError(
                    f"Unknown renderer {method_name!r}; expected one of "
                    f"{sorted(_BACKEND_MODULES)}"
                )
            importlib.import_module(module_name)
        return cls._registry[method_name](**kwargs)

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------------------------------------------- #
    # Cameras
    # ----------------------------------------------------------------- #

    def get_intrinsics(self, cfg) -> torch.Tensor:
        """``(4,)`` intrinsics ``[fx, fy, cx, cy]`` for the configured image size."""
        return get_camera_intrinsics(
            fov=cfg.render.fov,
            width=cfg.render.render_size,
            height=cfg.render.render_size,
        ).to(self.device)

    def get_camera_transforms(
        self,
        cfg,
        mesh: Mesh,
        cameras: list[float] | None = None,
        margin: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Place cameras around the mesh so it fills the frame.

        Args:
            cfg: Run config; reads ``render.num_views``, ``render.fov``,
                ``render.margin`` and ``render.render_size``.
            mesh: Mesh to frame. Only ``vertices`` is read.
            cameras: Flat ``[azim0, elev0, azim1, elev1, ...]`` overriding the
                configured view count. ``None`` uses ``cfg.render.num_views``.
            margin: Padding around the tight fit. Defaults to ``cfg.render.margin``.

        Returns:
            ``(c2w, w2c)``, each ``(B, 4, 4)``.
        """
        if cameras is not None:
            explicit_views = {"azim": cameras[::2], "elev": cameras[1::2]}
            num_cameras = len(explicit_views["azim"])
            cfg.num_views = num_cameras
        else:
            explicit_views = None
            num_cameras = cfg.render.num_views

        if margin is None:
            margin = cfg.render.margin

        c2w, w2c, _, _ = compute_tight_fit_distance(
            mesh.vertices,
            num_cameras=num_cameras,
            aspect=1.0,
            fov_deg=cfg.render.fov,
            margin=margin,
            width=cfg.render.render_size,
            height=cfg.render.render_size,
            cameras=explicit_views,
        )
        return c2w, w2c

    # ----------------------------------------------------------------- #
    # Backend interface
    # ----------------------------------------------------------------- #

    @abstractmethod
    def render(self, mesh: Mesh, w2c: torch.Tensor, **kwargs) -> RenderOutput:
        """Render shaded images of ``mesh`` from each camera in ``w2c``."""

    def rasterize(self, mesh: Mesh, w2c: torch.Tensor, **kwargs) -> RasterOutput:
        """Rasterize ``mesh`` and return pixel-to-face correspondence.

        Not abstract: only backends that can report per-pixel face coverage
        implement it. Figure-quality backends like polyscope legitimately cannot.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support rasterization; "
            f"use the 'nvdr' backend to produce a RasterOutput"
        )
