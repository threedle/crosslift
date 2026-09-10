from __future__ import annotations

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from crosslift import _core
from crosslift.fields.rosy import to_extrinsic
from crosslift.rendering.base import Renderer


def anchor_cameras(cfg) -> list[float]:
    base = cfg.render.cameras if cfg.render.cameras is not None else [0.0, 0.0]
    if cfg.render.anchor is None:
        return base
    return [angle + base[i % 2] for i, angle in enumerate(cfg.render.anchor)]


def _draw_arrows(ax, grad: torch.Tensor, bg: np.ndarray, step: int, scale: float) -> None:
    """Draw one view's arrows onto ``ax``.

    Args:
        grad: (H, W) complex directions.
        bg: (H, W, 3) background image.
    """
    H, W = grad.shape
    device = grad.device
    n_y, n_x = math.ceil(H / step), math.ceil(W / step)
    empty = torch.iinfo(torch.int64).max

    ax.imshow(bg, origin="upper", zorder=1)
    ax.set_aspect("equal")
    ax.axis("off")

    yx = torch.nonzero(grad.abs() > 1e-8)  # (P, 2) [y, x]

    cell_y = (yx[:, 0] / step).round().clamp(0, n_y - 1)  # (P,)
    cell_x = (yx[:, 1] / step).round().clamp(0, n_x - 1)  # (P,)
    dist = torch.hypot(yx[:, 0] - cell_y * step, yx[:, 1] - cell_x * step)  # (P,)
    near = torch.nonzero(dist < step * 0.75).squeeze(1)  # (Q,) indices into yx
    if near.numel() == 0:
        return

    # per-cell min over distance, keyed on the point index to break ties
    cell = (cell_y * n_x + cell_x).long()[near]  # (Q,)
    key = (dist[near] * 1e3).long() * near.numel() + torch.arange(
        near.numel(), device=device)  # (Q,)
    best = torch.full((n_y * n_x,), empty, dtype=torch.int64, device=device)
    best.scatter_reduce_(0, cell, key, reduce="amin", include_self=False)
    sel = near[best[best != empty] % near.numel()]  # (K,) indices into yx

    vecs = grad[yx[sel, 0], yx[sel, 1]]  # (K,) complex
    vecs = vecs / vecs.abs().clamp_min(1e-12)
    ax.quiver(
        yx[sel, 1].cpu().numpy(), yx[sel, 0].cpu().numpy(),
        vecs.real.cpu().numpy(), vecs.imag.cpu().numpy(),
        color="red", angles="xy", scale_units="xy", scale=1.0 / scale,
        pivot="tail", linewidth=0.6, minlength=0, zorder=3,
        headwidth=3, headlength=3, headaxislength=2.5,
    )


def gradient_arrows(grads: torch.Tensor, overlay: torch.Tensor, file_path: str,
                    view_dir: str | None = None,
                    step: int = 100, scale: float = 50.0) -> None:
    """Draw the image-space alignment directions as red arrows over each view.

    One arrow per grid cell: of the vectors inside a cell, the one nearest the
    cell center is drawn, at its true pixel location.

    Args:
        grads: (B, 1, H, W) complex directions, zero where there is no guidance.
        overlay: (B, 3, H, W) background image per view.
        file_path: where to write the sheet of views.
        view_dir: if set, also write each view as ``view_{i}.png`` here.
        step: grid spacing in pixels.
        scale: arrow length in pixels.
    """
    B = grads.shape[0]
    bg = overlay[:, :3].permute(0, 2, 3, 1).detach().cpu().numpy()

    fig, axs = plt.subplots(1, B, figsize=(6 * B, 6), sharex=True, sharey=True)
    axs = np.atleast_1d(axs).ravel()
    for b in range(B):
        _draw_arrows(axs[b], grads[b, 0], bg[b], step, scale)
    plt.tight_layout()
    plt.savefig(file_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    if view_dir is None:
        return
    os.makedirs(view_dir, exist_ok=True)
    for b in range(B):
        fig, ax = plt.subplots(figsize=(6, 6))
        _draw_arrows(ax, grads[b, 0], bg[b], step, scale)
        plt.tight_layout()
        plt.savefig(os.path.join(view_dir, f"view_{b}.png"),
                    dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close(fig)


class Figures:
    def __init__(self, mesh, cfg, out_dir: str, prefix: str, renderer=None):
        self.mesh = mesh
        self.cfg = cfg
        self.out_dir = out_dir
        self.prefix = prefix
        self.N = cfg.N

        self.renderer = renderer or Renderer.create("polyscope")
        intr = self.renderer.get_intrinsics(cfg)
        _, self.w2c_all = self.renderer.get_camera_transforms(cfg, mesh, margin=1.05)
        _, self.w2c_anchor = self.renderer.get_camera_transforms(
            cfg, mesh, anchor_cameras(cfg), margin=1.05)

        size = cfg.viz.render_size
        self.render_kwargs = {
            "height": size, "width": size, "intr": intr, "fov": 30.0,
        }

        if cfg.viz.sparse_visualization:
            mesh.sampled_faces_mask, mesh.sampled_face_indices = mesh.sample_blue_noise(
                num_samples=cfg.viz.num_face_viz)

        self._streamline_seeds = None
        self._seeds_chosen = False

    # plumbing
    def _path(self, name: str, ext: str = "png") -> str:
        return os.path.join(self.out_dir, f"{self.prefix}_{name}.{ext}")

    def _save(self, renders, name: str) -> str:
        path = self._path(name)
        torchvision.utils.save_image(renders, path)
        return path

    def _render(self, w2c, **kwargs):
        return self.renderer.render(self.mesh, w2c, **kwargs, **self.render_kwargs).renders

    def _render_both(self, name: str, **kwargs) -> None:
        """Save the all-views sheet and the transparent anchor view."""
        self._save(self._render(self.w2c_all, **kwargs), name)
        self._save(
            self._render(self.w2c_anchor, transparent_bg=True, **kwargs),
            f"{name}_anchor",
        )

    # figures
    def cross_field(self, x_real: torch.Tensor, name: str = "cross_field") -> None:
        """Draw the field as N-RoSy tangent crosses.

        Args:
            x_real: (F, 2) real view of the per-face complex coefficient, in the
                face's own tangent basis.
        """
        self._save(
            self._render(self.w2c_all, cross_field=x_real.clone(), n_sym=self.N),
            name,
        )

    def cross_field_3d(self, vec_3d, name: str = "cross_field") -> None:
        """Draw a field as 3D directions instead of tangent coefficients.
        """
        vec_3d = np.asarray(vec_3d, dtype=np.float64)
        if vec_3d.ndim == 3:
            vec_3d = vec_3d[:, 0, :]
        basis_2d = self.mesh.get_face_basis().detach().cpu().numpy()  # (F, 3, 2)
        coeffs = (vec_3d[:, :, None] * basis_2d).sum(axis=-2)         # (F, 2)
        self.cross_field(
            torch.from_numpy(coeffs.astype(np.float32)).to(self.mesh.device), name)

    def sharp_features(self, f_idx: torch.Tensor, directions: torch.Tensor,
                       name: str = "sharp_features") -> None:
        """Draw the hard constraints.

        Args:
            f_idx: (K,) constrained face indices.
            directions: (K,) complex directions in each face's tangent basis.
        """
        if f_idx.numel() == 0:
            return
        field = torch.zeros(
            (self.mesh.num_faces, 2), dtype=torch.float32, device=self.mesh.device)
        field[f_idx] = torch.view_as_real(directions).to(field.dtype)
        self._save(
            self._render(self.w2c_all, sharp_features=field, n_sym=self.N), name)

    def parameterization(self, corner_uv: np.ndarray, mesh=None,
                         name: str = "param") -> None:
        """Draw the UV map as a grid on the surface.

        Args:
            corner_uv: (3F, 2) per-corner UVs.
            mesh: the mesh the UVs index
        """
        if corner_uv is None:
            return
        target = mesh or self.mesh
        for tag, w2c, transparent in (("", self.w2c_all, False),
                                      ("_anchor", self.w2c_anchor, True)):
            renders = self.renderer.render(
                target, w2c, uv_coords=corner_uv, transparent_bg=transparent,
                **self.render_kwargs).renders
            self._save(renders, f"{name}{tag}")

    def quad_mesh(self, quad_mesh, name: str = "quads") -> None:
        """Draw an extracted quad mesh as a wireframe."""
        for tag, w2c, transparent in (("", self.w2c_all, False),
                                      ("_anchor", self.w2c_anchor, True)):
            renders = self.renderer.render(
                quad_mesh, w2c, draw_edges=True, transparent_bg=transparent,
                **self.render_kwargs).renders
            self._save(renders, f"{name}{tag}")

    # streamlines
    def streamline_seeds(self) -> np.ndarray | None:
        if self._seeds_chosen:
            return self._streamline_seeds
        self._seeds_chosen = True

        viz = self.cfg.viz
        if viz.streamline_seed_radius > 0:
            bbox_min = self.mesh.vertices.min(dim=0).values
            bbox_max = self.mesh.vertices.max(dim=0).values
            bbdiag = torch.norm(bbox_max - bbox_min).item()
            _, seed_idx = self.mesh.sample_blue_noise(
                min_radius=viz.streamline_seed_radius * bbdiag)
        else:
            if viz.streamline_num_seeds > 0:
                num_seeds = viz.streamline_num_seeds
            elif viz.streamline_seed_ratio > 0:
                num_seeds = max(1, round(self.mesh.num_faces * viz.streamline_seed_ratio))
            else:
                return None
            _, seed_idx = self.mesh.sample_blue_noise(num_samples=num_seeds)

        self._streamline_seeds = seed_idx.cpu().numpy().astype(np.int32)
        return self._streamline_seeds

    def streamlines(self, field: np.ndarray, name: str = "streamlines") -> None:
        """Trace and draw streamlines.
        """
        viz = self.cfg.viz
        print(f"Computing {name}...")
        try:
            ext = to_extrinsic(field, normalize=True)
            nodes, edges, colors = _core.trace_streamlines(
                self.mesh.vertices.cpu().numpy().astype(np.float64),
                self.mesh.faces.cpu().numpy().astype(np.int32),
                ext, self.N,
                num_steps=viz.streamline_num_steps,
                dist_ratio=viz.streamline_dist_ratio,
                seed_faces=self.streamline_seeds(),
            )
        except Exception as e:
            print(f"  streamline tracing failed: {e}")
            return

        np.savez(self._path(name, "npz"), nodes=nodes, edges=edges, colors=colors)
        self._render_both(
            name, streamlines=(nodes, edges, colors),
            streamline_radius=viz.streamline_radius, n_sym=self.N)
