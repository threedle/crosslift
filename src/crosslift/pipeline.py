import math
import os
from pathlib import Path

import draccus
import torch
import torchvision

from crosslift.config import Config
from crosslift.fields.lifting import lift_vecs
from crosslift.fields.rosy import expand_rosy
from crosslift.fields.solve import (
    centroid_weights,
    compute_coherence_weights,
    solve_global_field,
    solve_view_field,
    view_alignment,
)
from crosslift.geometry.io import read_mesh_arrays, write_obj, write_rawfield
from crosslift.geometry.mesh import Mesh
from crosslift.geometry.remesh import remesh_mesh_file
from crosslift.geometry.repair import repair_mesh_file
from crosslift.guidance.base import VisualGuidance
from crosslift.guidance.gradients import extract_gradients
from crosslift.quad_extraction.extract import QuadResult, extract_surface
from crosslift.quad_extraction.optimize import optimize_quads
from crosslift.rendering.base import Renderer
from crosslift.utils.utils import load_view_images
from crosslift.viz import Figures, gradient_arrows


@torch.no_grad()
def run(cfg: Config) -> None:
    """Run the pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mesh_name = os.path.basename(cfg.path).split(".")[0]
    Path(cfg.run_name).mkdir(parents=True, exist_ok=True)
    os.makedirs(os.path.join(cfg.run_name, "targets"), exist_ok=True)

    with open(cfg.run_name + "/config.yml", "w") as f:
        draccus.dump(cfg, f)

    # Mesh
    mesh_path = cfg.path
    if cfg.repair_mesh:
        mesh_path = repair_mesh_file(cfg.path, cfg.run_name)
    if cfg.remesh.enabled:
        mesh_path = remesh_mesh_file(
            mesh_path, cfg.run_name,
            target_edge=cfg.remesh.target_edge,
            target_faces=cfg.remesh.target_faces,
            iterations=cfg.remesh.iterations,
            adaptive=cfg.remesh.adaptive,
        )
    mesh = Mesh(mesh_path, device)
    render_mesh = mesh if mesh_path == cfg.path else Mesh(cfg.path, device)
    meshes = [mesh] if render_mesh is mesh else [mesh, render_mesh]
    for target in meshes:
        if cfg.render.rot_angles is not None:
            target.rotate_mesh_euler(cfg.render.rot_angles, cfg.render.rot_order, inplace=True)
    # Both meshes take the solve mesh's transform
    center = mesh.vertices.mean(dim=0)
    scale = (mesh.vertices - center).norm(dim=1).max()
    for target in meshes:
        target.normalize_mesh(inplace=True, center=center, scale=scale)

    # Hard constraints
    hard_f_idx = torch.empty((0,), device=device, dtype=torch.long)   # (M,) long
    hard_dirs = torch.empty((0,), device=device, dtype=torch.cfloat)  # (M,) complex, direction**N

    if cfg.guidance.sharp_edge_constraints:
        hard_f_idx, hard_dirs = mesh.get_sharp_edge_constraints(
            angle_threshold=math.radians(cfg.solve.edge_angle_threshold), N=cfg.N
        )  # hard_f_idx: (M,) long, hard_dirs: (M,) complex

    if cfg.guidance.boundary_constraints:
        boundary_f_idx, boundary_dirs = mesh.get_boundary_edge_constraints(N=cfg.N)  # (K,) long, (K,) complex
        if boundary_f_idx.numel() > 0:
            # only pin faces not already constrained by the sharp edge constraints
            fresh = ~torch.isin(boundary_f_idx, hard_f_idx)          # (K,) bool
            hard_f_idx = torch.cat([hard_f_idx, boundary_f_idx[fresh]])  # (M,) long
            hard_dirs = torch.cat([hard_dirs, boundary_dirs[fresh]])     # (M,) complex

    # Cameras + forward render
    renderer = Renderer.create("nvdr")
    intr = renderer.get_intrinsics(cfg)
    c2w, w2c = renderer.get_camera_transforms(cfg, mesh)
    render_kwargs = {
        "height": cfg.render.render_size,
        "width": cfg.render.render_size,
        "intr": intr,
        "fov": cfg.render.fov,
    }
    render_out = renderer.render(render_mesh, w2c, **render_kwargs)
    renders = render_out.renders
    normals = render_out.normals
    world_normals = render_out.world_normals
    depths = render_out.depths
    masks = render_out.masks
    torchvision.utils.save_image(renders, os.path.join(cfg.run_name, "targets/renders.png"))
    torchvision.utils.save_image(normals, os.path.join(cfg.run_name, "targets/normals.png"))
    torchvision.utils.save_image(
        world_normals, os.path.join(cfg.run_name, "targets/world_normals.png"))
    torchvision.utils.save_image(depths, os.path.join(cfg.run_name, "targets/depths.png"))
    torchvision.utils.save_image(masks, os.path.join(cfg.run_name, "targets/masks.png"))

    # Visual guidance
    guidance = VisualGuidance.create(cfg.guidance.method, device=device, cfg=cfg)
    guidance_kwargs = {
        "renders": renders,
        "normals": normals,
        "world_normals": world_normals,
        "depths": depths,
        "masks": masks,
        "image_dir": cfg.guidance.user_images_dir,
    }
    images = guidance.get_images(**guidance_kwargs)

    # Image space gradients
    grads = extract_gradients(images=images, masks=masks) # (B, 1, H, W) complex
    if cfg.viz.save_all:
        gradient_arrows(
            grads, images, os.path.join(cfg.run_name, "targets/guidance_arrows.png"),
            view_dir=(os.path.join(cfg.run_name, "targets/guidance_arrows")
                      if cfg.guidance.save_view_images else None))

    # Lift to mesh surface
    raster = renderer.rasterize(mesh, w2c, intr, projection_type=cfg.render.projection_type)
    surface_vecs = lift_vecs(grads, mesh, raster)  # SurfaceDirections (P,) complex unit directions
    
    # Stage I interpolation per view
    surface_grads = solve_view_field(
        surface_vecs,
        mesh,
        lambda_s=cfg.render.lambda_s,
        lambda_c=cfg.render.lambda_c,
        N=cfg.N,
        w_c=centroid_weights(surface_vecs, mesh, raster),
    )  # (B, F) complex, unit directions (zero on unseen faces from the given view)
    surface_grads = surface_grads ** cfg.N  # (B, F) complex, N-RoSy encoded
    align = view_alignment(
        mesh, c2w, cfg.render.projection_type, cfg.solve.view_align_power
    ) # (B, F)
    coherence = compute_coherence_weights(surface_grads, align) # (F,)
    constraints = surface_grads.flatten() # (B*F,) complex, N-RoSy encoded
    f_idx = torch.arange(mesh.num_faces, device=device)[None].expand(
        surface_grads.shape[0], -1
    ).flatten() # (B*F,) long
    w_c = (align * coherence[None]).flatten() # (B*F,) real
    valid = (constraints.abs() > 1e-12) & torch.isfinite(constraints) # (B*F,) bool
    constraints, f_idx, w_c = constraints[valid], f_idx[valid], w_c[valid] # (C,) each

    # Hard constraints, w_c = -1
    if hard_f_idx.numel() > 0:
        constraints = torch.cat([constraints, hard_dirs]) # (C+M,) complex
        f_idx = torch.cat([f_idx, hard_f_idx]) # (C+M,) long
        w_c = torch.cat([w_c, torch.full_like(hard_f_idx, -1.0, dtype=w_c.dtype)]) # (C+M,) real, -1 = hard

    # Optional auxiliary user-drawn line constraints
    if cfg.guidance.aux_user_lines_dir is not None:
        aux_images = load_view_images(
            cfg.guidance.aux_user_lines_dir, cfg.render.num_views,
            cfg.render.render_size, device,
        )

        aux_grads = extract_gradients(images=aux_images, masks=masks)  # (B, 1, H, W) complex
        aux_surface_vecs = lift_vecs(aux_grads, mesh, raster)  # SurfaceDirections (P,) complex unit directions
        aux_surface_grads = solve_view_field(
            aux_surface_vecs,
            mesh,
            lambda_s=cfg.render.lambda_s,
            lambda_c=cfg.render.lambda_c,
            N=cfg.N,
            w_c=centroid_weights(aux_surface_vecs, mesh, raster),
        )  # (B, F) complex, unit directions (zero on unseen faces from the given view)
        aux_surface_grads = aux_surface_grads ** cfg.N # (B, F) complex, N-RoSy encoded

        aux_constraints = aux_surface_grads.flatten() # (B*F,) complex, N-RoSy encoded
        aux_f_idx = torch.arange(mesh.num_faces, device=device)[None].expand(
            aux_surface_grads.shape[0], -1
        ).flatten() # (B*F,) long
        aux_valid = (aux_constraints.abs() > 1e-12) & torch.isfinite(aux_constraints) # (B*F,) bool
        aux_constraints, aux_f_idx = aux_constraints[aux_valid], aux_f_idx[aux_valid] # (U,) each

        constraints = torch.cat([constraints, aux_constraints]) # (C+M+U,) complex
        f_idx = torch.cat([f_idx, aux_f_idx]) # (C+M+U,) long
        w_c = torch.cat([w_c, torch.full_like(aux_f_idx, cfg.solve.lambda_u, dtype=w_c.dtype)]) # (C+M+U,) real

    # Stage 2 interpolation global solve
    x = solve_global_field(
        constraints,
        f_idx,
        w_c,
        mesh,
        lambda_s=cfg.solve.lambda_s,
        lambda_c=cfg.solve.lambda_c,
        N=cfg.N,
        normalize=False,
        use_direct_solver=cfg.solve.use_direct_solver,
    ) # (F,) complex, N-RoSy encoded

    x_encoded = x.clone() # (F,) complex, N-RoSy encoded
    x = x ** (1.0 / cfg.N) # (F,) complex, decoded direction
    if cfg.solve.normalize:
        x = x / x.abs().clamp_min(1e-12) # (F,) complex, unit magnitude
    x_real = torch.view_as_real(x) # (F, 2) real, [real, imag]


    # Complex coefficients -> 3D vectors in each face's tangent plane.
    tangent, bitangent = mesh.face_basis.unbind(-1) # (F, 3) each
    x_3d = x.real[:, None] * tangent + x.imag[:, None] * bitangent # (F, 3) real
    
    rep_np = x_3d.cpu().numpy() # (F, 3)
    normals_np = mesh.face_normals.cpu().numpy()   # (F, 3)
    raw_field = expand_rosy(rep_np, normals_np, cfg.N)  # (F, N, 3), full N-RoSy field

    # Figures
    figures = Figures(mesh, cfg, cfg.run_name, mesh_name)
    if cfg.viz.save_all:
        figures.cross_field(x_real)
        figures.sharp_features(hard_f_idx, hard_dirs)
    write_rawfield(os.path.join(cfg.run_name, f"{mesh_name}.rawfield"), raw_field)
    figures.streamlines(raw_field)

    # Quad extraction
    if cfg.quad.extract:
        # Curl correction alignment weights. |x| = solve confidence.
        w_align = x_encoded.abs() * mesh.face_areas
        if hard_f_idx.numel() > 0:
            w_align[hard_f_idx] = -1.0

        makes_quads = cfg.N == 4 and (
            cfg.quad.method in ("miq", "quadwild")
            or cfg.quad.integrate_extract_quads)
        suffix = "quads" if makes_quads else "cut_mesh"
        output_path = os.path.join(cfg.run_name, f"{mesh_name}_{suffix}.obj")
        result = extract_surface(
            rep_vectors_3d=rep_np,
            face_normals=normals_np,
            input_mesh_path=mesh_path,
            output_path=output_path,
            w_align_weights=w_align.cpu().numpy(),
            N=cfg.N,
            method=cfg.quad.method,
            face_labels=mesh.face_components,
            num_components=mesh.num_components,
            skip_curl_correction=cfg.quad.skip_curl_correction,
            wSmooth=cfg.quad.cc_wSmooth,
            wRoSy=cfg.quad.cc_wRoSy,
            wAlign=cfg.quad.cc_wAlign,
            tau_threshold=cfg.quad.cc_tau_threshold,
            skip_initial_solve=cfg.quad.cc_skip_initial_solve,
            skip_first_implicit=cfg.quad.cc_skip_first_implicit,
            dynamic_annealing=cfg.quad.cc_dynamic_annealing,
            free_magnitude=cfg.quad.cc_free_magnitude,
            magnitude_clamp=cfg.quad.cc_magnitude_clamp,
            use_adaptive_scaling=cfg.quad.use_adaptive_scaling,
            adaptive_scale_factor=cfg.quad.adaptive_scale_factor,
            static_quad_scale=cfg.quad.static_quad_scale,
            target_quad_count=cfg.quad.target_quad_count,
            target_quads_per_triangle=cfg.quad.target_quads_per_triangle,
            direct_round=cfg.quad.direct_round,
            miq_stiffness=cfg.quad.miq_stiffness,
            miq_normalize_frames=cfg.quad.miq_normalize_frames,
            isolate_miq=cfg.quad.miq_isolate,
            miq_timeout=cfg.quad.miq_timeout,
            quadwild_sharp_angle=cfg.quad.quadwild_sharp_angle,
            quadwild_scale_fact=cfg.quad.quadwild_scale_fact,
            quadwild_keep_workdir=cfg.quad.quadwild_keep_workdir,
            integral_seamless=cfg.quad.integrate_seamless,
            round_seams=cfg.quad.integrate_round_seams,
            integrate_normalize_frames=cfg.quad.integrate_normalize_frames,
            integrate_extract_quads=cfg.quad.integrate_extract_quads,
        )

        write_rawfield(
            os.path.join(cfg.run_name, f"{mesh_name}_curlfree.rawfield"),
            result.curlfree,
        )
        # The same figures again on the corrected field. Seeds are shared with
        # the run above, so the two are directly comparable.
        if cfg.viz.save_all:
            figures.cross_field_3d(result.curlfree, name="cross_field_curl_corrected")
            figures.parameterization(result.corner_uv)
        if cfg.viz.save_all or not cfg.quad.skip_curl_correction:
            figures.streamlines(result.curlfree, name="streamlines_curl_corrected")

        if isinstance(result, QuadResult) and result.faces.shape[0] > 0:
            quad_mesh = Mesh(output_path, device, quad_mesh=True)
            quad_mesh.normalize_mesh(inplace=True)
            figures.quad_mesh(quad_mesh)

            if cfg.quad.smooth_iters > 0:
                print("\n=== Quad smoothing ===")
                V_ref, F_ref = read_mesh_arrays(mesh_path)
                smoothed = optimize_quads(
                    result.vertices, result.faces, V_ref, F_ref,
                    iters=cfg.quad.smooth_iters,
                    scale_blend=cfg.quad.smooth_scale_blend,
                    step=cfg.quad.smooth_step,
                    project_every=cfg.quad.smooth_project_every,
                    feature_angle=cfg.quad.smooth_feature_angle,
                )
                smooth_path = os.path.join(cfg.run_name, f"{mesh_name}_quads_smoothed.obj")
                write_obj(smooth_path, smoothed, result.faces)
                print(f"  Saved smoothed quad mesh: {smooth_path}")
                smooth_mesh = Mesh(smooth_path, device, quad_mesh=True)
                smooth_mesh.normalize_mesh(inplace=True)
                figures.quad_mesh(smooth_mesh, name="quads_smoothed")


if __name__ == "__main__":
    cfg = Config()
    run(cfg)