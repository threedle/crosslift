import io
import os

import torch
import torchvision
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from torchvision.transforms import functional as TF

from crosslift.guidance.base import VisualGuidance

GRID_ROWS, GRID_COLS = 3, 2
MIN_SILHOUETTE_IOU = 0.9

INTENT = ("Add a wireframe quad mesh to {target}. Make the edge flow capture the semantic "
          "features of the object. Keep singularities minimal.")
ALIGNMENT = ("Keep the exact same camera view, object orientation, position and scale; change "
             "only the surface.")
APPEARANCE = ("Draw the mesh as thin black edges on a light gray surface, quads only, and keep "
              "those edges the highest-contrast feature in the image. Keep the background pure white.")
GRID = (f"a {GRID_ROWS}-row by {GRID_COLS}-column grid holding 6 renders of one object seen from "
        f"6 camera angles")
GRID_CONSISTENCY = ("The edge flow must agree across the views: where a feature reappears in "
                    "another cell it must carry the same edge flow.")
DESCRIPTIONS = {
    "renders": "the shaded render",
    "normals": "a camera-space surface-normal map",
    "world_normals": "a world-space surface-direction map",
    "depths": "a depth map",
}
USAGE = {
    "normals": ("Read image {i} as geometry alone: it gives surface orientation and curvature, so "
                "run the edge flow along the principal curvature directions it reveals."),
    "world_normals": ("Image {i} colors each pixel by the direction that piece of surface faces in "
                      "the object's own frame, not by how it looks: equal colors mean equally "
                      "oriented surface, so it tells you which side of the object a view is "
                      "looking at, and when two views are showing you the same surface. Use it to "
                      "run the edge flow along the curvature, and to keep that flow consistent "
                      "wherever one surface reappears in another view."),
    "depths": "Image {i} gives distance from the camera; read it as geometry alone.",
}
REFERENCE_ONLY = ("Edit image 1 and return a single image in image 1's layout. Every other image is "
                  "reference only: never reproduce its colors or its dark background.")
CANONICAL_VIEWS = ["left side", "front", "right side", "back", "top", "bottom"]
VIEW_PLACEMENT = ("Place each feature only where its own camera can see it: a feature on one side "
                  "of the object belongs in the cells that face that side, and is foreshortened or "
                  "hidden in the rest — never repeat a front-facing feature in a cell that is not "
                  "looking at the front.")
POLE_ORIENTATION = ("The top and bottom cells look straight along the object's up-axis, with the "
                    "object's front toward the bottom edge of the frame, so anything facing front "
                    "is edge-on there and must not be drawn as if seen head-on.")


def label_views(cfg):
    """The 6 canonical view names, or None when the cameras are not those 6."""
    if not cfg.guidance.gemini_label_views:
        return None
    if cfg.render.num_views != len(CANONICAL_VIEWS):
        return None
    return CANONICAL_VIEWS


def build_prompt(cfg, inputs):
    """Assemble the prompt for one call, naming every image it carries and every camera it saw.

    Args:
        inputs: guidance_kwargs keys, in the order the images are attached.
    """
    named = ", ".join(f"image {i + 1} is {DESCRIPTIONS[k]}" for i, k in enumerate(inputs))
    solo = len(inputs) == 1
    views = label_views(cfg)

    parts = [f"This image is {GRID}." if solo else
             (f"Each of these images is {GRID}, cell for cell the same views in the same "
              f"order: {named}.")]
    if views:
        parts += [(f"Read left to right then top to bottom, the cells are the object's "
                   f"{', '.join(views[:-2])}, then its {views[-2]} and its {views[-1]}."),
                  VIEW_PLACEMENT, POLE_ORIENTATION]
    parts += [INTENT.format(target="the object in all 6 cells"
                            + ("" if solo else " of image 1") + ", leaving none untextured"),
              GRID_CONSISTENCY]
    parts += [USAGE[k].format(i=i + 1) for i, k in enumerate(inputs) if k in USAGE]
    if not solo:
        parts.append(REFERENCE_ONLY)
    return " ".join(parts + [ALIGNMENT, APPEARANCE])


def generate(client, cfg, images, prompt, aspect_ratio, seed):
    """Edit images[0], conditioned on any images after it, returning the generated PIL image."""
    response = client.models.generate_content(
        model=cfg.guidance.gemini_model,
        contents=[*images, prompt],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            seed=seed,
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio, image_size=cfg.guidance.gemini_image_size),
        ),
    )
    data = next((p.inline_data.data for p in response.parts if p.inline_data), None)
    if data is None:
        raise RuntimeError("Gemini returned no image")
    return Image.open(io.BytesIO(data))


def to_grid(images):
    return TF.to_pil_image(torchvision.utils.make_grid(images.cpu(), nrow=GRID_COLS, padding=0))


def from_grid(grid, num_views):
    """Inverse of to_grid: crop the returned grid back into per-view images."""
    grid = TF.to_tensor(grid.convert("RGB"))
    h, w = grid.shape[1] // GRID_ROWS, grid.shape[2] // GRID_COLS
    views = [grid[:, r * h:(r + 1) * h, c * w:(c + 1) * w]
             for r in range(GRID_ROWS) for c in range(GRID_COLS)]
    return torch.stack(views[:num_views])


def silhouette_iou(images, masks):
    """Per-view overlap between the generated silhouette and the render it was conditioned on."""
    fg = (images.mean(dim=1, keepdim=True) < 0.95).float().cpu()
    m = (masks > 0.5).float().cpu()
    return (fg * m).sum(dim=(1, 2, 3)) / ((fg + m) > 0).float().sum(dim=(1, 2, 3)).clamp_min(1)


def check_alignment(images, masks):
    """Warn per view when the generated silhouette drifts off the render it was conditioned on."""
    iou = silhouette_iou(images, masks)
    for i, v in enumerate(iou.tolist()):
        if v < MIN_SILHOUETTE_IOU:
            print(f"WARNING: view {i} silhouette IoU {v:.3f} — gradients will lift onto wrong faces")
    return iou


class GeminiGuidance(VisualGuidance, method_name="gemini"):
    def __init__(self, cfg, device=None, **kwargs):
        super().__init__(device=device, **kwargs)
        self.cfg = cfg

    def get_images(self, **kwargs):
        """Given renders, use Gemini to add a quad mesh wireframe to each view"""
        cfg = self.cfg
        cache_path = os.path.join(cfg.run_name, "targets/guidance_images.pt")
        if os.path.exists(cache_path) and not cfg.guidance.overwrite_target:
            images = torch.load(cache_path)
        else:
            load_dotenv()
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY is not set")
            client = genai.Client(api_key=api_key)
            inputs = cfg.guidance.gemini_inputs.split("+")
            prompt = build_prompt(cfg, inputs)
            grids = [to_grid(kwargs[k]) for k in inputs]
            seeds = (cfg.seed, cfg.seed + 1)
            for attempt, seed in enumerate(seeds):
                grid = generate(client, cfg, grids, prompt, f"{GRID_COLS}:{GRID_ROWS}", seed)
                images = from_grid(grid, cfg.render.num_views)
                images = TF.resize(images, (cfg.render.render_size, cfg.render.render_size))
                drifted = silhouette_iou(images, kwargs["masks"]) < MIN_SILHOUETTE_IOU
                if not drifted.any() or attempt == len(seeds) - 1:
                    break
                print(f"Silhouette drift on view(s) {drifted.nonzero().flatten().tolist()} at "
                      f"seed {seed}; regenerating with seed {seeds[attempt + 1]}.")
            torch.save(images, cache_path)

        images = images.to(self.device)
        check_alignment(images, kwargs["masks"])
        self.save_images(images)
        return images
