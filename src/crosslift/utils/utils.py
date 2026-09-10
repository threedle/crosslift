import os

import torch
import torchvision
from PIL import Image
from torchvision import transforms


def load_image(image_path: str) -> Image.Image:
    """
    Load an image from the specified path and convert it to RGB format.

    Args:
        image_path (str): Path to the image file.
        
    Returns:
        tensor_image (torch.Tensor): The image converted to a tensor
    """
    image = Image.open(image_path).convert("RGB")
    transform = transforms.ToTensor()
    tensor_image = transform(image).unsqueeze(0)
    return tensor_image


def load_view_images(
    image_dir: str, num_views: int, render_size: int, device: torch.device | str | None = None
) -> torch.Tensor:
    """Load ``view_{i}.png`` for ``i in range(num_views)``, resized to ``render_size``.

    A missing view falls back to a blank white image, so a directory that
    only covers a few cameras still produces a full ``(num_views, 3, H, W)``
    batch.
    """
    images = []
    for i in range(num_views):
        img_path = os.path.join(image_dir, f"view_{i}.png")
        if not os.path.exists(img_path):
            print(f"Image {img_path} not found, using blank image instead.")
            images.append(torch.ones(1, 3, render_size, render_size))
        else:
            image = load_image(img_path)
            images.append(
                torchvision.transforms.functional.resize(image, (render_size, render_size))
            )
    images = torch.cat(images, dim=0)
    return images if device is None else images.to(device)


def save_view_images(images: torch.Tensor, image_dir: str) -> None:
    """Write ``images`` (B, 3, H, W) as the ``view_{i}.png`` layout ``load_view_images`` reads."""
    os.makedirs(image_dir, exist_ok=True)
    for i, image in enumerate(images):
        torchvision.utils.save_image(image, os.path.join(image_dir, f"view_{i}.png"))
