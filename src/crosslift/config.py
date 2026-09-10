from dataclasses import dataclass, field


@dataclass
class Guidance:
    overwrite_target: bool = False
    batch_size: int = 6
    user_images_dir: str | None = None
    aux_user_lines_dir: str | None = None
    save_view_images: bool = False
    method: str = "gemini" # "flux", "gemini", or "image"
    gemini_model: str = "gemini-3.1-flash-image" # "gemini-3-pro-image"
    gemini_image_size: str = "4K" # "512px", "1K", "2K" or "4K"
    # "+"-joined guidance_kwargs keys, in the order Gemini receives them. The
    # first is the image it edits.
    gemini_inputs: str = "renders" # + "renders+normals" (cam space) or "renders+world_normals" (world space), "renders+depths"
    gemini_label_views: bool = True # include which view each grid element is in the prompt
    sharp_edge_constraints: bool = False
    boundary_constraints: bool = False

@dataclass
class Render:
    num_views: int = 6
    fov: float = 30.0
    render_size: int = 1024
    projection_type: str = "perspective" # "perspective" or "orthographic"
    lambda_s: float = 1.0
    lambda_c: float = 1000.0
    rot_angles: tuple[float, float, float] | None = None # in degrees, rotation to apply to the mesh before rendering (x, y, z)
    rot_order: str = 'xyz' # order of rotations to apply, e.g., 'xyz', 'zyx', etc.
    cameras: list[float] | None = None
    anchor: list[float] | None = None #[azim, elev, ...], offset from cameras[:2]
    margin: float = 1.0

@dataclass
class Solve:
    lambda_s: float = 0.1
    lambda_c: float = 1.0
    lambda_u: float = 100.0 # constraint weight for user-specified directions
    edge_angle_threshold: float = 35  # degrees
    view_align_power: float = 2.0
    normalize: bool = True
    use_direct_solver: bool = False


@dataclass
class Remesh:
    enabled: bool = False
    target_faces: int | None = None  # approximate, None keeps the input face count
    target_edge: float | None = None  # overrides target_faces
    iterations: int = 10
    adaptive: bool = False # let curvature vary the edge length


@dataclass
class Quad:
    extract: bool = True
    method: str = "miq" # "miq" (MIQ+QEx), "quadwild", or "integrate" (Directional)

    # Curl correction
    skip_curl_correction: bool = True
    cc_skip_initial_solve: bool = True
    cc_skip_first_implicit: bool = True
    cc_dynamic_annealing: bool = True
    cc_wSmooth: float = 1.0
    cc_wAlign: float = 1.0
    cc_wRoSy: float = 1.0
    cc_tau_threshold: float = 0.01
    # Absorb curl as scale, not rotation. Pair with miq_normalize_frames=False.
    cc_free_magnitude: bool = False
    cc_magnitude_clamp: float | None = None  # bound magnitudes to [1/c, c]

    # Density
    use_adaptive_scaling: bool = True
    adaptive_scale_factor: float = 16.0
    static_quad_scale: float = 80.0
    # Override use_adaptive_scaling
    target_quad_count: int | None = None
    target_quads_per_triangle: float | None = None

    # MIQ
    direct_round: bool = False  # greedy rounding: faster, worse
    miq_stiffness: float = 5.0
    miq_normalize_frames: bool = True
    miq_isolate: bool = True  # run each pass in a child process
    miq_timeout: float | None = None  # seconds per pass, None for no limit

    # QuadWild. Binaries and configs come from $QUADWILD_DIR.
    quadwild_sharp_angle: float = -1.0  # normal deviation (deg), <=0 disables
    # Extra multiplier on the scaleFact derived from the density settings above
    quadwild_scale_fact: float = 1.25
    quadwild_keep_workdir: bool = False

    # Post-extraction smoothing
    smooth_iters: int = 0  # 0 disables
    smooth_scale_blend: float = 1.0  # 0 drives every quad to one global size
    smooth_step: float = 0.8
    smooth_project_every: int = 1
    # Normal deviation in degrees marking a crease the smoother holds vertices
    # on. 0 disables.
    smooth_feature_angle: float = 60.0

    # Directional integration
    integrate_seamless: bool = True
    integrate_round_seams: bool = False
    integrate_normalize_frames: bool = True
    integrate_extract_quads: bool = True


@dataclass
class Visualization:
    sparse_visualization: bool = True
    num_face_viz: int = 3500
    render_size: int = 1024
    save_all: bool = False # visualize all intermediate results, not just the final ones

    # Streamlines
    streamline_seed_radius: float = 0.042
    streamline_num_seeds: int = 0
    streamline_seed_ratio: float = 0.0
    streamline_dist_ratio: float = 3.0
    streamline_num_steps: int = 150
    streamline_radius: float = 0.001 # relative to the scene bbox


@dataclass
class Config:
    path: str = "./data/meshes/angel.obj"
    run_name: str = "./outputs/angel"
    N: int = 4
    seed: int = 42
    repair_mesh: bool = False  # weld duplicate vertices, restore manifoldness
    guidance: Guidance = field(default_factory=Guidance)
    remesh: Remesh = field(default_factory=Remesh)
    render: Render = field(default_factory=Render)
    solve: Solve = field(default_factory=Solve)
    quad: Quad = field(default_factory=Quad)
    viz: Visualization = field(default_factory=Visualization)