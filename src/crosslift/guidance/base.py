from __future__ import annotations
import os
import torch
import torchvision
import cmap
import numpy as np
from abc import ABC, abstractmethod
from typing import ClassVar

from crosslift.utils.utils import save_view_images

_BACKEND_MODULES = {
            "flux": "crosslift.guidance.flux",
            "gemini": "crosslift.guidance.gemini",
            "image": "crosslift.guidance.image",
        }

class VisualGuidance(ABC):
    _registry: ClassVar[dict[str, type[VisualGuidance]]] = {}

    def __init_subclass__(cls, method_name: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if method_name:
            cls._registry[method_name] = cls
    
    @classmethod
    def create(cls, method_name: str, device=None, **kwargs) -> VisualGuidance:

        if method_name not in cls._registry:
            module_name = _BACKEND_MODULES.get(method_name)
            if module_name:
                import importlib
                importlib.import_module(module_name)
            else:
                raise ValueError(f"Unknown method '{method_name}'.")

        return cls._registry[method_name](device=device, **kwargs)

    def __init__(self, device=None, **kwargs):
        super().__init__(**kwargs)
        if device is not None:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def save_images(self, images, name="guidance_images"):
        """Save the view grid, and the individual views when save_view_images is set."""
        targets = os.path.join(self.cfg.run_name, "targets")
        torchvision.utils.save_image(images, os.path.join(targets, f"{name}.png"))
        if self.cfg.guidance.save_view_images:
            save_view_images(images, os.path.join(targets, name))

    @abstractmethod
    def get_images(self, *args, **kwargs):
        """Get images using the visual guidance."""
        pass
