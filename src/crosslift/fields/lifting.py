from __future__ import annotations

import torch

from crosslift.fields.surface_directions import SurfaceDirections
from crosslift.geometry.mesh import Mesh
from crosslift.rendering.base import RasterOutput
from crosslift.rendering.utils import get_projection_matrix, get_projection_matrix_ortho

_EPS = 1e-8

def _signed_clamp(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Push ``|x|`` away from zero without changing its sign."""
    ones = torch.ones_like(x)
    return torch.where(x < 0, -ones, ones) * x.abs().clamp_min(eps)


def _projection_matrix(raster: RasterOutput) -> torch.Tensor:
    """Rebuild the ``(4, 4)`` camera-to-clip matrix the raster was produced with.
    """
    height, width = raster.image_size
    if raster.projection_type == "orthographic":
        return get_projection_matrix_ortho(raster.intr, width, height)
    if raster.projection_type == "perspective":
        return get_projection_matrix(raster.intr, width, height)
    raise ValueError(
        f"Unknown projection type {raster.projection_type!r}; "
        f"expected 'perspective' or 'orthographic'"
    )


def lift_vecs(
    vec_2d: torch.Tensor,
    mesh: Mesh,
    raster: RasterOutput,
    eps: float = _EPS,
) -> SurfaceDirections:
    """Unproject per-pixel 2D directions into per-face tangent coordinates.

    Args:
        vec_2d: ``(B, 1, H, W)`` or ``(B, H, W)`` complex. Per-pixel image-space
            direction.
        mesh:
        raster:
        eps: Floor for divisions

    Returns:
        SurfaceDirections: one entry per surviving pixel, directions on the surface.
    """
    if not torch.is_complex(vec_2d):
        raise TypeError(
            f"vec_2d must be complex (real part = x-direction, imaginary part = "
            f"y-direction), got dtype {vec_2d.dtype}"
        )

    vec = vec_2d
    if vec.dim() == 4:
        if vec.shape[1] != 1:
            raise ValueError(
                f"vec_2d must carry a single direction channel, got shape "
                f"{tuple(vec.shape)}"
            )
        vec = vec.squeeze(1)
    if vec.dim() != 3:
        raise ValueError(
            f"vec_2d must be (B, 1, H, W) or (B, H, W), got shape "
            f"{tuple(vec_2d.shape)}"
        )

    num_views, height, width = vec.shape
    if (num_views, height, width) != (raster.num_views, *raster.image_size):
        raise ValueError(
            f"vec_2d is {num_views} views at {height}x{width} but the raster is "
            f"{raster.num_views} views at {raster.image_size[0]}x"
            f"{raster.image_size[1]}; render and rasterize must agree"
        )

    usable = raster.fg_mask & (vec.abs() > eps)
    flat_idx = usable.reshape(-1).nonzero(as_tuple=True)[0]        # (P,)

    if flat_idx.numel() == 0:
        return _empty_directions(raster)

    view_idx = torch.div(flat_idx, height * width, rounding_mode="floor")
    face_idx = raster.face_id.reshape(-1)[flat_idx]                # (P,)
    pos_cam = raster.pixel_pos_cam.reshape(-1, 3)[flat_idx]        # (P, 3)
    observed = vec.reshape(-1)[flat_idx]                           # (P,) complex

    # Tangent basis of the hit face, in camera space
    w2c_rot = raster.w2c[:, :3, :3]                                # (B, 3, 3)
    basis_cam = torch.einsum("bij,fjk->bfik", w2c_rot, mesh.face_basis)
    tangent, bitangent = basis_cam[view_idx, face_idx].unbind(-1)  # (P, 3) each

    # Screen-space Jacobian
    proj = _projection_matrix(raster).to(pos_cam.dtype)
    row_x, row_y, row_w = proj[0], proj[1], proj[3]

    # Clip coordinates of the surface point
    clip_x = pos_cam @ row_x[:3] + row_x[3]
    clip_y = pos_cam @ row_y[:3] + row_y[3]
    clip_w = _signed_clamp(pos_cam @ row_w[:3] + row_w[3], eps)
    clip_w_sq = clip_w * clip_w

    # JT
    def _screen_delta(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        d_x, d_y, d_w = t @ row_x[:3], t @ row_y[:3], t @ row_w[:3]
        d_col = 0.5 * width * (d_x * clip_w - clip_x * d_w) / clip_w_sq
        d_row = 0.5 * height * (d_y * clip_w - clip_y * d_w) / clip_w_sq
        return d_col, d_row
    jt00, jt10 = _screen_delta(tangent)
    jt01, jt11 = _screen_delta(bitangent)

    det = _signed_clamp(jt00 * jt11 - jt01 * jt10, eps)
    obs_x, obs_y = observed.real, observed.imag
    alpha_t = (jt11 * obs_x - jt01 * obs_y) / det
    alpha_b = (jt00 * obs_y - jt10 * obs_x) / det

    value = torch.complex(alpha_t, alpha_b)
    value = value / value.abs().clamp_min(eps)

    # Validity
    keep = torch.isfinite(value.real) & torch.isfinite(value.imag)
    if not bool(keep.all()):
        face_idx, view_idx = face_idx[keep], view_idx[keep]
        pixel_idx, value = flat_idx[keep], value[keep]
    else:
        pixel_idx = flat_idx

    return SurfaceDirections(
        face_idx=face_idx,
        view_idx=view_idx,
        pixel_idx=pixel_idx,
        value=value,
        visible_face_mask=raster.visible_face_mask,
    )


def _empty_directions(raster: RasterOutput) -> SurfaceDirections:
    """A well-formed result with no entries, for a fully background batch."""
    device = raster.face_id.device
    empty_long = torch.empty(0, dtype=torch.long, device=device)
    return SurfaceDirections(
        face_idx=empty_long,
        view_idx=empty_long.clone(),
        pixel_idx=empty_long.clone(),
        value=torch.empty(0, dtype=torch.complex64, device=device),
        visible_face_mask=raster.visible_face_mask,
    )
