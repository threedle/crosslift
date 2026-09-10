import torch
import torch.nn.functional as F
import numpy as np


def get_camera_intrinsics(fov, width, height):
    # convert fov from degrees to radians
    fov = torch.tensor(fov) if not torch.is_tensor(fov) else fov
    fov = torch.deg2rad(fov)
    # focal length = image width / (2 * tan(fov/2))
    fx = width / (2 * torch.tan(fov / 2))
    fy = height / (2 * torch.tan(fov / 2))
    cx = width / 2
    cy = height / 2
    intrinsics = torch.tensor([fx, fy, cx, cy])
    return intrinsics

def look_at(center, target, up):
    f = F.normalize(target - center, dim=-1)
    s = F.normalize(torch.cross(f, up, dim=-1), dim=-1)
    u = F.normalize(torch.cross(s, f, dim=-1), dim=-1)
    m = torch.stack([s, u, -f], dim=-1)
    return m

def get_camera_transforms(centers, target, up, device=None):
    r_c2w = look_at(centers, target, up)
    t = centers.unsqueeze(-1)
    r_w2c = r_c2w.transpose(-1, -2)
    hom = torch.tensor([0, 0, 0, 1], device=r_c2w.device).expand(t.size(0), 1, -1)
    c2w = torch.cat([r_c2w, t], dim=-1)
    c2w = torch.cat([c2w, hom], dim=-2)
    w2c = torch.cat([r_w2c, -r_w2c @ t], dim=-1)
    w2c = torch.cat([w2c, hom], dim=-2)
    return c2w.to(device), w2c.to(device)

def get_projection_matrix(intrinsics, width, height, near=0.1, far=10, device=None):
    if device is None:
        device = intrinsics.device
    # Projection matrix
    # [ 2fx/w      0      (-2cx/w+1)        0      ]
    # [   0     -2fy/h    (-2cy/h+1)        0      ]
    # [   0        0     -(f+n)/(f-n) -(2fn)/(f-n) ]
    # [   0        0         -1             0      ]
    fx, fy, cx, cy = intrinsics.unbind(-1)
    proj = torch.zeros(4, 4, device=device)
    proj[0, 0] = 2 * fx / width
    proj[0, 2] = 1 - (2 * cx / width)
    proj[1, 1] = -2 * fy / height # negative for OpenGL convention
    proj[1, 2] = 1 - (2 * cy / height)
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -(2 * far * near) / (far - near)
    proj[3, 2] = -1
    return proj

def get_projection_matrix_ortho(
    intrinsics,
    width,
    height,
    near=0.1,
    far=10.0,
    device=None,
    scale=1.0,  # how much of the normalized mesh fills the view
):
    """
    Orthographic projection compatible with your look_at / w2c convention:

      - Camera looks along -Z in camera space.
      - Mesh is roughly inside [-1, 1]^3 after normalize_mesh().
      - We define a symmetric box in *camera space*:
            X ∈ [-scale, scale]
            Y ∈ [-scale/aspect, scale/aspect]
            Z ∈ [-far, -near]  (Z is negative in camera space)

      - Maps that box to NDC ∈ [-1, 1]^3 for nvdiffrast.
    """
    if device is None:
        device = intrinsics.device
    dtype = intrinsics.dtype

    aspect = width / float(height)

    # Symmetric box in camera space (world units, not pixels)
    right  = scale
    left   = -scale
    top    = scale / aspect
    bottom = -top

    # Standard OpenGL-style glOrtho matrix:
    # [ 2/(r-l)       0          0       -(r+l)/(r-l) ]
    # [    0       2/(t-b)       0       -(t+b)/(t-b) ]
    # [    0          0      -2/(f-n)    -(f+n)/(f-n) ]
    # [    0          0          0              1     ]

    proj = torch.zeros(4, 4, device=device, dtype=dtype)

    proj[0, 0] =  2.0 / (right - left)
    proj[0, 3] = -(right + left) / (right - left)

    proj[1, 1] =  2.0 / (top - bottom)
    proj[1, 3] = -(top + bottom) / (top - bottom)

    proj[2, 2] = -2.0 / (far - near)
    proj[2, 3] = -(far + near) / (far - near)

    proj[3, 3] = 1.0

    return proj

def n_fibonacci_views(num_views, distance=2.5):
    """
    Generate `num_views` camera centers approximately evenly spaced on the sphere
    using a Fibonacci sphere construction.

    Returns:
        centers: (num_views, 3) camera centers
        target:  (num_views, 3) look-at points (all zeros, origin)
        up:      (num_views, 3) per-view up vectors
        labels:  list[str] of length num_views, each in
                 ["side", "front", "side", "back", "top", "bottom"]
                 depending on which canonical direction the view is closest to.
    """
    if num_views < 1:
        raise ValueError("num_views must be at least 1.")

    # Fibonacci sphere directions
    i = torch.arange(num_views, dtype=torch.float32)
    phi = (1.0 + 5.0 ** 0.5) / 2.0  # golden ratio

    y = 1.0 - 2.0 * (i + 0.5) / float(num_views)  # (n,), evenly spaced
    r = torch.sqrt(torch.clamp(1.0 - y * y, min=0.0))  # radius at that y

    theta = 2.0 * np.pi * i / phi

    x = r * torch.cos(theta)
    z = r * torch.sin(theta)

    dirs = torch.stack([x, y, z], dim=-1)  # (n, 3), unit-ish directions
    centers = dirs * distance
    target = torch.zeros_like(centers)

    # Up: mostly +Z, alternate near poles (|z| > 0.9)
    up = torch.tensor([0.0, 1.0, 0.0]).expand_as(centers).clone()
    alternate_up = torch.tensor([0.0, 0.0, 1.0])
    mask = dirs[:, 1].abs() > 0.9
    up[mask] = alternate_up

    # Labels from nearest canonical direction
    canon_dirs = torch.tensor([
        [ 1.0,  0.0,  0.0],  # +X → "side"
        [-1.0,  0.0,  0.0],  # -X → "side"
        [ 0.0,  1.0,  0.0],  # +Y → "top"
        [ 0.0, -1.0,  0.0],  # -Y → "bottom"
        [ 0.0,  0.0,  1.0],  # +Z → "front"
        [ 0.0,  0.0, -1.0],  # -Z → "back"
    ])
    canon_labels = ["side", "side", "top", "bottom", "front", "back"]

    d_norm = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)  # (n,3)
    dots = d_norm @ canon_dirs.T  # (n, 6)
    idx = torch.argmax(dots, dim=-1)  # (n,)

    labels = [canon_labels[int(k)] for k in idx]

    return centers, target, up, labels

def single_view(azim=0, elev=0, distance=2.5):
    """
    Generate a single view at a given azimuthal angle (in degrees)
    around the Y axis. Azim of zero should be looking along -Z axis.
    Elev of zero is at the equator, positive elev moves up toward +Y axis.

    Returns:
        centers: (1, 3) camera center
        target:  (1, 3) look-at point (origin)
        up:      (1, 3) up vector
        labels:  list[str] of length 1, always ["single"]
    """
    theta = torch.deg2rad(torch.tensor(azim)) + np.pi / 2
    phi = torch.deg2rad(torch.tensor(elev))
    x = distance * torch.cos(phi) * torch.cos(theta)
    y = distance * torch.sin(phi)
    z = distance * torch.cos(phi) * torch.sin(theta)
    centers = torch.tensor([[x, y, z]], dtype=torch.float32)
    target = torch.zeros_like(centers)
    up = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
    labels = ["single"]
    return centers, target, up, labels

def n_views(azim, elev, distance=2.5, up_dir=None):
    """
    Generate a n views at the given azim and elev angles (in degrees)

    Returns:
        centers: (1, 3) camera center
        target:  (1, 3) look-at point (origin)
        up:      (1, 3) up vector
        labels:  list[str] of length 1, always ["single"]
    """
    if len(azim) != len(elev):
        raise ValueError("azim and elev must have the same length")
    if not torch.is_tensor(azim):
        azim = torch.tensor(azim)
    if not torch.is_tensor(elev):
        elev = torch.tensor(elev)
    theta = torch.deg2rad(azim) + np.pi / 2
    phi = torch.deg2rad(elev)
    x = torch.cos(phi) * torch.cos(theta) # (n,)
    y = torch.sin(phi) # (n,)
    z = torch.cos(phi) * torch.sin(theta) # (n,)
    dirs = torch.stack([x, y, z], dim=-1) # (n, 3)
    centers = centers = dirs * distance # (n, 3)
    target = torch.zeros_like(centers)
    up = torch.tensor([0.0, 1.0, 0.0]).expand_as(centers).clone()
    alternate_up = torch.tensor([0.0, 0.0, -1.0])
    if up_dir is not None:
        alternate_up = torch.tensor(up_dir).to(up.device)
        up = alternate_up.expand_as(centers).clone()
    mask = (torch.abs(torch.cos(phi)) < 0.1)
    up[mask] = alternate_up

    # Labels from nearest canonical direction
    canon_dirs = torch.tensor([
        [ 1.0,  0.0,  0.0],  # +X → "side"
        [-1.0,  0.0,  0.0],  # -X → "side"
        [ 0.0,  1.0,  0.0],  # +Y → "top"
        [ 0.0, -1.0,  0.0],  # -Y → "bottom"
        [ 0.0,  0.0,  1.0],  # +Z → "front"
        [ 0.0,  0.0, -1.0],  # -Z → "back"
    ])
    canon_labels = ["side", "side", "top", "bottom", "front", "back"]

    d_norm = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)  # (n,3)
    dots = d_norm @ canon_dirs.T  # (n, 6)
    idx = torch.argmax(dots, dim=-1)  # (n,)

    labels = [canon_labels[int(k)] for k in idx]
    return centers, target, up, labels

def n_canonical_views(num_views, distance=2.5):
    """
    Generate `num_views` canonical views:
      - `num_views - 2` views equally spaced around the equator (y = 0),
        i.e. side-on rotations.
      - 2 extra views: top and bottom.
    """
    if num_views < 2:
        raise ValueError("num_views must be at least 2 (top and bottom).")
    print(f"Generating {num_views} canonical view(s).")

    num_side = num_views - 2  # side views on the equator

    centers_list = []
    up_list = []
    labels = []

    # Side views: equally spaced around the y=0 circle
    if num_side > 0:
        theta = torch.arange(num_side, dtype=torch.float32) * (-2.0 * np.pi / num_side) + np.pi

        x = torch.cos(theta) * distance
        y = torch.zeros_like(x)
        z = torch.sin(theta) * distance

        centers_side = torch.stack([x, y, z], dim=-1)  # (num_side, 3)
        centers_list.append(centers_side)

        up_side = torch.tensor([0.0, 1.0, 0.0]).expand(num_side, -1).clone()
        up_list.append(up_side)

        canon_dirs = torch.tensor([
            [-1.0,  0.0, 0.0],  # -X → "left"
            [ 0.0,  0.0, 1.0],  # +Z → "front"
            [ 1.0,  0.0, 0.0],  # +X → "right"
            [ 0.0,  0.0, -1.0], # -Z → "back"
        ])
        canon_labels = ["left", "front", "right", "back"]

        for v in centers_side:
            d = v / (v.norm() + 1e-8)
            dots = canon_dirs @ d        # (4,)
            idx = int(torch.argmax(dots))
            labels.append(canon_labels[idx])

    # Top and bottom, -Z as up near the poles
    centers_top_bottom = torch.tensor([
        [0.0,  distance, 0.0],
        [0.0, -distance, 0.0],
    ], dtype=torch.float32)
    up_top_bottom = torch.tensor([
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
    ], dtype=torch.float32)

    centers_list.append(centers_top_bottom)
    up_list.append(up_top_bottom)

    labels.extend(["top", "bottom"])

    centers = torch.cat(centers_list, dim=0)
    up = torch.cat(up_list, dim=0)
    target = torch.zeros_like(centers)

    return centers, target, up, labels

def random_views(num_cameras, distance=2.5):
    theta = torch.rand(num_cameras) * 2 * np.pi # 0 to 2pi
    phi = torch.rand(num_cameras) * np.pi - np.pi / 2 # -pi/2 to pi/2
    pos = torch.stack([theta.cos() * phi.cos(), theta.sin() * phi.cos(), phi.sin()], dim=-1) # (N, 3)
    centers = pos * distance
    target = torch.zeros_like(centers)
    up = torch.FloatTensor([0, 0, 1]).expand_as(centers).clone()
    # if camera is too close to z axis, use alternate up vector
    alternate_up = torch.FloatTensor([0, 1, 0])
    mask = (pos[:, 2].abs() > 0.9)
    up[mask] = alternate_up
    return centers, target, up

def compute_bounding_sphere(verts, fov_deg=45, margin=1.0):
    """
    Computes a uniform safe distance for a camera to orbit a mesh 
    using a bounding sphere approach.
    """
    min_bound = torch.min(verts, dim=0)[0]
    max_bound = torch.max(verts, dim=0)[0]
    mesh_center = (min_bound + max_bound) / 2.0

    # Bounding sphere radius: max distance from center to any vertex
    verts_centered = verts - mesh_center
    radius = torch.max(torch.norm(verts_centered, dim=-1))

    # Distance to fit that sphere inside the viewing frustum at this FOV
    fov_rad = torch.deg2rad(torch.tensor(float(fov_deg), device=verts.device))
    base_distance = radius / torch.sin(fov_rad / 2.0)
    
    distance = (base_distance * margin).to("cpu").item()

    return distance

def orbit_camera_transforms(
    initial_azim, initial_elev, n_frames, distance=None, up_dir=None, verts=None,
    fov_deg=30, margin=1.0
):
    """
    Generate evenly spaced camera views orbiting around a mesh.
    
    Args:
        initial_azim (float): Starting azimuth angle in degrees.
        initial_elev (float): Constant elevation angle in degrees.
        n_frames (int): Number of camera frames to generate for the orbit.
        distance (float): Camera distance from the origin.
        up_dir (list or None): Optional up vector.
        verts (torch.Tensor or None): Optional vertex positions of the mesh, used to
                                        compute tight fit distance if distance is None.
        
    Returns:
        centers: (n_frames, 3) camera centers
        target:  (n_frames, 3) look-at points (origin)
        up:      (n_frames, 3) up vectors
        labels:  list[str] of length n_frames with canonical directions
    """
    azim = torch.linspace(initial_azim, initial_azim + 360, n_frames + 1)[:-1]
    elev = torch.full((n_frames,), float(initial_elev))
    cameras = {'azim': azim, 'elev': elev}
    if distance is None and verts is not None:
        c2w, w2c, labels, target = compute_tight_fit_distance(
            verts, num_cameras=n_frames, aspect=1.0, fov_deg=fov_deg, margin=margin,
            device=verts.device, cameras=cameras, up_dir=up_dir, uniform=True,
            center="visual"
        )
    else:
        print(f"Using provided orbit distance: {distance:.3f}")
        centers, target, up, labels = n_views(azim, elev, distance=distance, up_dir=up_dir)
        c2w, w2c = get_camera_transforms(centers, target, up)

    return c2w, w2c

def compute_tight_fit_distance(
    verts, num_cameras=1, aspect=1.0, fov_deg=45, margin=1.0, width=1024, height=1024,
    device=None, cameras=None, up_dir=None, uniform=False, center="visual"
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # First find approximate distance
    base_distance = 1.0
    if cameras is not None:
        centers, target, up, labels = n_views(cameras['azim'], cameras['elev'], distance=base_distance, up_dir=up_dir)
    else:
        if num_cameras == 1:
            centers, target, up, labels = single_view(azim=0, elev=0, distance=base_distance)
        elif num_cameras <= 6:
            centers, target, up, labels = n_canonical_views(num_cameras, distance=base_distance)
        else:
            centers, target, up, labels = n_canonical_views(6, distance=base_distance)
            centers_extra, target_extra, up_extra, labels_extra = n_fibonacci_views(num_cameras - 6, distance=base_distance)
            centers = torch.cat([centers, centers_extra], dim=0)
            up = torch.cat([up, up_extra], dim=0)
            target = torch.cat([target, target_extra], dim=0)
            labels = labels + labels_extra
    c2w, w2c = get_camera_transforms(centers, target, up, device=device)
    
    min_bound = torch.min(verts, dim=0)[0]
    max_bound = torch.max(verts, dim=0)[0]
    visual_center = (min_bound + max_bound) / 2.0
    origin = torch.zeros(3, device=device)
    verts_centered = verts - visual_center
    base_distance = 1.0
    
    vert_h = F.pad(torch.Tensor(verts_centered), (0, 1), mode='constant', value=1.0)
    v_cam_h = vert_h @ w2c.transpose(-1, -2)
    v_cam = v_cam_h[..., :3] # (num_cameras, num_verts, 3)
    intrinsics = get_camera_intrinsics(fov=fov_deg, width=width, height=height)
    proj = get_projection_matrix(intrinsics, width, height, device=device)[None].expand(num_cameras, -1, -1)
    S_x = proj[:, 0, 0].unsqueeze(-1)
    S_y = proj[:, 1, 1].unsqueeze(-1)

    dist_req_x = (v_cam[..., 0].abs() * S_x.abs()) + v_cam[..., 2]
    dist_req_y = (v_cam[..., 1].abs() * S_y.abs()) + v_cam[..., 2]
    max_dist_x = torch.max(dist_req_x, dim=1)[0]
    max_dist_y = torch.max(dist_req_y, dim=1)[0]
    dist_needed = torch.max(max_dist_x, max_dist_y)
    if uniform:
        dist_needed = torch.full_like(dist_needed, torch.max(dist_needed))
    final_distances = ((base_distance + dist_needed) * margin).unsqueeze(-1)

    if center == "mass":
        print("Centering on center of mass.")
        # Tilt correction: place the camera, project the raw (untilted-center)
        # vertices through it, and measure NDC to detect clipping.
        origin = torch.zeros(3, device=device)
        view_dir_unit = F.normalize(centers, dim=1).to(device)

        test_cam_pos = origin + (view_dir_unit * final_distances)
        final_target = visual_center.unsqueeze(0).expand(num_cameras, -1).to(device)
        
        _, test_w2c = get_camera_transforms(test_cam_pos, final_target, up.to(device), device=device)

        raw_vert_h = F.pad(verts.clone().to(device), (0, 1), mode='constant', value=1.0)
        test_v_cam_h = raw_vert_h @ test_w2c.transpose(-1, -2)
        test_v_cam = test_v_cam_h[..., :3]

        depth = test_v_cam[..., 2].abs()  # abs: OpenGL Z is behind the lens
        ndc_x = (test_v_cam[..., 0].abs() * S_x.abs()) / depth
        ndc_y = (test_v_cam[..., 1].abs() * S_y.abs()) / depth
        
        max_ndc = torch.max(
            torch.max(ndc_x, dim=1)[0],
            torch.max(ndc_y, dim=1)[0]
        ) # (num_cameras,)
        
        if uniform:
            max_ndc = torch.full_like(max_ndc, torch.max(max_ndc))

        # max_ndc > 1.0 means the mesh clipped; scale distance by that ratio
        final_distances = final_distances * max_ndc.unsqueeze(-1) * margin
        final_cam_pos = origin + (view_dir_unit * final_distances)
    else:
        print("Centering on visual center (bounding box center).")
        view_dir_unit = F.normalize(centers, dim=1).to(device)
        final_cam_pos = visual_center + (view_dir_unit * final_distances)
        final_target = visual_center.unsqueeze(0).expand(num_cameras, -1).to(device)

    final_c2w, final_w2c = get_camera_transforms(
        final_cam_pos,
        final_target,
        up.to(device),
        device=device
    )
    return final_c2w, final_w2c, labels, final_target

def normalize_inverse_depths(
    depths: torch.Tensor,
    masks: torch.Tensor,
    global_norm: bool = False,
) -> torch.Tensor:
    """
    Args:
        depths: (B, 3, H, W) tensor of inverse depths (larger = nearer)
        masks: (B, 1, H, W) tensor of foreground masks (1.0 = FG, 0.0 = BG)
        global_norm: If True, normalizes based on min/max of the entire batch.
                     If False, normalizes each image independently.
    Returns:
        depths_norm: (B, 3, H, W) tensor of normalized inverse depths
    """
    # Masked copies to find min/max of foreground pixels only (one channel;
    # depth is consistent across channels)
    d_for_max = depths[:, 0:1].clone()
    d_for_min = depths[:, 0:1].clone()
    d_for_max[masks < 0.5] = -float('inf')
    d_for_min[masks < 0.5] = float('inf')

    if global_norm:
        d_max = torch.amax(d_for_max, dim=(0, -2, -1), keepdim=True)
        d_min = torch.amin(d_for_min, dim=(0, -2, -1), keepdim=True)
    else:
        d_max = torch.amax(d_for_max, dim=(-2, -1), keepdim=True)
        d_min = torch.amin(d_for_min, dim=(-2, -1), keepdim=True)

    # Empty scenes: all pixels masked out
    is_empty = (d_min == float('inf'))
    d_min[is_empty] = 0.0
    d_max[is_empty] = 1.0

    denom = d_max - d_min
    denom[denom < 1e-6] = 1.0 

    depths_norm = (depths - d_min) / denom

    # Background (below d_min) went negative; clamp it out and mask
    depths_norm = torch.clamp(depths_norm, min=0.0, max=1.0)
    depths_norm = depths_norm * masks

    return depths_norm