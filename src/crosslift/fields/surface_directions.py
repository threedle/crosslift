from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SurfaceDirections:
    """Per-pixel directions gathered onto mesh faces.

    Attributes:
        face_idx: ``(P,)`` long. Face each observation landed on.
        view_idx: ``(P,)`` long. View it came from. Non-decreasing — entries are
            emitted in raster order, so all observations for a view are
            contiguous.
        pixel_idx: ``(P,)`` long. Flat index into a ``(B, H, W)`` image of the
            pixel this came from, so ``raster.pixel_pos_cam.reshape(-1, 3)
            [pixel_idx]`` recovers where on the surface it was seen.
        value: ``(P,)`` complex. Direction in that face's local tangent basis,
            normalized to unit magnitude.
        visible_face_mask: ``(B, F)`` bool. True where a face was rasterized in
            that view.
    """

    face_idx: torch.Tensor
    view_idx: torch.Tensor
    pixel_idx: torch.Tensor
    value: torch.Tensor
    visible_face_mask: torch.Tensor

    def __len__(self) -> int:
        return int(self.face_idx.shape[0])

    @property
    def num_views(self) -> int:
        return int(self.visible_face_mask.shape[0])

    @property
    def num_faces(self) -> int:
        return int(self.visible_face_mask.shape[1])

    @property
    def device(self) -> torch.device:
        return self.face_idx.device

    def select_view(self, view: int) -> SurfaceDirections:
        """The observations from a single view, as a one-view instance.
        """
        keep = self.view_idx == view
        return SurfaceDirections(
            face_idx=self.face_idx[keep],
            view_idx=self.view_idx[keep],
            pixel_idx=self.pixel_idx[keep],
            value=self.value[keep],
            visible_face_mask=self.visible_face_mask[view : view + 1],
        )
