import polyscope as ps
import numpy as np
import torch
from crosslift.rendering.base import Renderer, RenderOutput


class PolyscopeRender(Renderer, method_name="polyscope"):
    def __init__(
        self,
        device=None,
        psr=None
    ):
        super().__init__()
        if device is not None:
            self.device = device
        if psr is not None:
            self.ps_wrapper = psr
        else:
            ps.set_allow_headless_backends(True)
            ps.init()
            self.ps_wrapper = ps
        self.ps_wrapper.set_up_dir("y_up")
        self.ps_wrapper.set_front_dir("z_front")
        self.ps_wrapper.set_ground_plane_mode("shadow_only")
    
    def render(
        self, mesh, w2c, height=1024, width=1024,
        draw_edges=False, fov=30, transparent_bg=False,
        shadow_strength=0.25, ssaa=1, smooth_shading=True,
        material="clay", mesh_color=(0.5, 0.7, 0.95),
        edge_color=(0, 0, 0), edge_width=1.0, cross_field=None,
        uv_coords=None, texture_map=None, sharp_features=None,
        streamlines=None, streamline_radius=0.0015,
        scalar_quantity=None, scalar_quantity_name="scalar",
        scalar_cmap="viridis", singularities=None,
        singularity_radius=0.006, n_sym=4,
        symmetry_plane=None, symmetry_curve_radius=0.0025, **kwargs
    ):
        num_cameras, _, _ = w2c.size()
        device = mesh.vertices.device
        intr = self.ps_wrapper.CameraIntrinsics(fov_vertical_deg=float(fov), aspect=width/height)
        self.ps_wrapper.set_window_size(width, height)
        self.ps_wrapper.set_shadow_darkness(shadow_strength)
        self.ps_wrapper.set_shadow_blur_iters(8)
        self.ps_wrapper.set_SSAA_factor(ssaa)

        # Register mesh
        v_np = mesh.vertices.detach().cpu().numpy()
        f_np = mesh.faces.detach().cpu().numpy()
        ps_mesh = self.ps_wrapper.register_surface_mesh(
            "Mesh", v_np, f_np, edge_width=edge_width if draw_edges else 0.0)
        if hasattr(mesh, 'uvs') and mesh.uvs is not None \
            and hasattr(mesh, 'texture_map') and mesh.texture_map is not None:
            uvs_np = mesh.uvs.detach().cpu().numpy()
            uvs_np[:, 1] = 1.0 - uvs_np[:, 1]
            if hasattr(mesh, 'face_uvs') and mesh.face_uvs is not None:
                face_uvs_np = mesh.face_uvs.detach().cpu().numpy()
            else:
                face_uvs_np = mesh.faces.detach().cpu().numpy()
            corner_uvs = uvs_np[face_uvs_np]
            corner_uvs_flat = corner_uvs.reshape(-1, 2)
            tex_map_np = mesh.texture_map.detach().cpu().numpy()
            ps_mesh.add_parameterization_quantity("param", corner_uvs_flat, 
                defined_on='corners', enabled=False)
            ps_mesh.add_color_quantity("texture_map", tex_map_np, 
                defined_on='texture', param_name="param",
                filter_mode='nearest', enabled=True)
        if cross_field is not None:
            length = 0.01 # originally was 0.02
            if hasattr(mesh, 'sampled_faces_mask'):
                cross_field[~mesh.sampled_faces_mask] = 0
            cross_field = cross_field.detach().cpu().numpy()
            basis_2d = mesh.get_face_basis().detach().cpu().numpy()
            mesh_color = (0.7, 0.7, 0.7)
            ps_mesh.add_tangent_vector_quantity(
                "cross field", cross_field, basis_2d[..., 0], basis_2d[..., 1],
                n_sym=n_sym, defined_on="faces", color=(0,0,1), length=length, enabled=True
            )
        if streamlines is not None:
            mesh_color = (1, 1, 1)
            sl_nodes, sl_edges, sl_colors = streamlines
            curves = ps.register_curve_network(
                "streamlines", sl_nodes, sl_edges,
                radius=streamline_radius, enabled=True)
            colors = np.zeros((len(sl_edges), 3))
            # Generate N evenly-spaced hues for streamline coloring
            hues = np.linspace(0, 1, n_sym, endpoint=False)
            palette = np.array([[
                max(0, min(1, abs(h * 6 - 3) - 1)),  # R
                max(0, min(1, 2 - abs(h * 6 - 2))),   # G
                max(0, min(1, 2 - abs(h * 6 - 4)))    # B
            ] for h in hues])
            for i in range(len(sl_edges)):
                colors[i] = palette[sl_colors[i] % n_sym]
            curves.add_color_quantity(
                "line colors", colors, defined_on='edges', enabled=True)
        if uv_coords is not None:
            # Wipe non-parameterization related structures
            self.ps_wrapper.remove_all_structures()
            ps_param_mesh = ps.register_surface_mesh(
                "parameterization", v_np, f_np, enabled=True
            )
            ps_param_mesh.add_parameterization_quantity(
                "param", uv_coords, defined_on='corners', enabled=True,
                checker_size=1, viz_style='grid'
            )
        if sharp_features is not None:
            length = 0.01 # originally was 0.02
            if hasattr(mesh, 'sampled_faces_mask'):
                sharp_features[~mesh.sampled_faces_mask] = 0
            sharp_features = sharp_features.detach().cpu().numpy()
            basis_2d = mesh.get_face_basis().detach().cpu().numpy()
            mesh_color = (0.7, 0.7, 0.7)
            ps_mesh.add_tangent_vector_quantity(
                "sharp features", sharp_features, basis_2d[..., 0], basis_2d[..., 1],
                n_sym=n_sym, defined_on="faces", color=(0,0,1), length=length, enabled=True
            )

        if singularities is not None:
            # (positions (k,3), index m_v (k,)). Red = positive index
            # (valence 5 in the quad mesh), blue = negative (valence 3).
            sing_pos, sing_idx = singularities
            sing_pos = np.asarray(sing_pos, dtype=np.float64).reshape(-1, 3)
            sing_idx = np.asarray(sing_idx).reshape(-1)
            if sing_pos.shape[0] > 0:
                pc = ps.register_point_cloud(
                    "singularities", sing_pos, radius=singularity_radius,
                    enabled=True,
                )
                colors = np.where(
                    (sing_idx > 0)[:, None],
                    np.array([[0.90, 0.15, 0.15]]),
                    np.array([[0.15, 0.30, 0.95]]),
                )
                pc.add_color_quantity("index sign", colors, enabled=True)

        if scalar_quantity is not None:
            if isinstance(scalar_quantity, torch.Tensor):
                scalar_quantity = scalar_quantity.detach().cpu().numpy()
            scalar_quantity = np.asarray(scalar_quantity, dtype=np.float64).reshape(-1)
            ps_mesh.add_scalar_quantity(
                scalar_quantity_name, scalar_quantity,
                defined_on="faces", cmap=scalar_cmap, enabled=True,
            )

        # Symmetry plane: orange wireframe rectangle, plus the red curve where
        # it cuts the surface. Drawn as curves, not a translucent quad, since
        # the luminance-derived alpha below turns semi-transparency grey.
        if symmetry_plane is not None:
            from crosslift.symmetry import plane_mesh_intersection

            sp_normal, sp_origin = symmetry_plane
            n = np.asarray(sp_normal, dtype=np.float64).reshape(3)
            n = n / max(float(np.linalg.norm(n)), 1e-12)
            o = np.asarray(sp_origin, dtype=np.float64).reshape(3)

            helper = (np.array([1.0, 0.0, 0.0]) if abs(n[2]) > 0.9
                      else np.array([0.0, 0.0, 1.0]))
            t1 = np.cross(n, helper)
            t1 = t1 / max(float(np.linalg.norm(t1)), 1e-12)
            t2 = np.cross(n, t1)

            extent = float(np.linalg.norm(v_np - o, axis=1).max()) * 1.15
            corners = np.stack([
                o - extent * t1 - extent * t2,
                o + extent * t1 - extent * t2,
                o + extent * t1 + extent * t2,
                o - extent * t1 + extent * t2,
            ])
            outline = ps.register_curve_network(
                "symmetry plane", corners,
                np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32),
                radius=symmetry_curve_radius * 0.6, enabled=True,
            )
            outline.set_color((0.98, 0.62, 0.10))
            outline.set_material("flat")

            curve_nodes, curve_edges = plane_mesh_intersection(v_np, f_np, n, o)
            if curve_edges.shape[0] > 0:
                curve = ps.register_curve_network(
                    "symmetry curve", curve_nodes, curve_edges,
                    radius=symmetry_curve_radius, enabled=True,
                )
                curve.set_color((0.85, 0.05, 0.05))
                curve.set_material("flat")

        ps_mesh.set_color(mesh_color)
        ps_mesh.set_edge_color(edge_color)
        ps_mesh.set_material(material)
        ps_mesh.set_smooth_shade(smooth_shading)

        renders = []
        for i in range(num_cameras):
            extr = self.ps_wrapper.CameraExtrinsics(mat=w2c[i].cpu().numpy())
            cam_params = self.ps_wrapper.CameraParameters(intr, extr)
            self.ps_wrapper.set_view_camera_parameters(cam_params)
            img_array = self.ps_wrapper.screenshot_to_buffer(transparent_bg=False)
            img_trans = self.ps_wrapper.screenshot_to_buffer(transparent_bg=True)
            img_rgb = img_array[:, :, :3].astype(np.float32) / 255.0
            alpha_trans = img_trans[:, :, 3].astype(np.float32) / 255.0
            luminance = np.mean(img_rgb, axis=2)
            shadow_alpha = 1.0 - luminance
            final_alpha = np.maximum(alpha_trans, shadow_alpha)
            final_alpha = (np.clip(final_alpha, 0, 1) * 255).astype(np.uint8)
            h, w, _ = img_array.shape
            final_rgb = np.zeros((h, w, 3), dtype=np.uint8)
            mash_mask = alpha_trans > 1.8*shadow_strength
            final_rgb[mash_mask] = img_trans[mash_mask, :3]
            if transparent_bg:
                img_array = np.dstack((final_rgb, final_alpha[:, :, None]))
            else:
                # Composite on white: shadows look better than transparent_bg=False directly
                white_bg = np.ones_like(final_rgb) * 255.0
                norm_alpha = final_alpha[:, :, None] / 255.0
                img_array = (final_rgb * norm_alpha + white_bg * (1.0 - norm_alpha)).astype(np.uint8)
            renders.append(img_array)
        # convert to torch tensor of (B, C, H, W)
        renders = np.stack(renders, axis=0).astype(np.float32) / 255.0
        renders = torch.from_numpy(renders).permute(0, 3, 1, 2).to(self.device)

        # clear polyscope state for next render
        self.ps_wrapper.remove_all_structures()

        return RenderOutput(renders=renders)
