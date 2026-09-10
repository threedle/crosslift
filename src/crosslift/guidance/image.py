import torch
import os
from crosslift.guidance.base import VisualGuidance
from crosslift.utils.utils import load_view_images


class ImageGuidance(VisualGuidance, method_name="image"):
    def __init__(self, cfg, device=None, **kwargs):
        super().__init__(device=device, **kwargs)
        self.cfg = cfg
    
    def get_images(self, **kwargs):
        image_dir = kwargs.get("image_dir")
        images = load_view_images(
            image_dir, self.cfg.render.num_views, self.cfg.render.render_size, self.device
        )

        # Save images
        self.save_images(images)
        torch.save(images, os.path.join(self.cfg.run_name, "targets/guidance_images.pt"))
        return images
