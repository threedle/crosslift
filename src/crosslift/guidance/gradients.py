import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

def shrink_masks(masks: torch.Tensor, n_pixels: int) -> torch.Tensor:
    """
    Shrinks binary masks by n_pixels.

    Args:
        masks (torch.Tensor): Input masks of shape (B, 1, H, W) 
                              containing 0.0s and 1.0s.
        n_pixels (int): The number of pixels to shrink/erode by. 
                        n_pixels=1 shrinks by 1 pixel (using a 3x3 kernel).
                        n_pixels=2 shrinks by 2 pixels (using a 5x5 kernel).

    Returns:
        torch.Tensor: The eroded masks, of the same shape as input.
    """
    if n_pixels == 0:
        return masks

    kernel_size = 2 * n_pixels + 1
    
    padding = n_pixels

    inverted_masks = 1.0 - masks
    dilated_background = F.max_pool2d(
        inverted_masks,
        kernel_size=kernel_size,
        stride=1,
        padding=padding
    )

    eroded_masks = 1.0 - dilated_background
    
    return eroded_masks

def scale_down(input_tensor: torch.Tensor, scale_factor: int) -> torch.Tensor:
    """
    Downscales a tensor by averaging, then normalizes the result.

    Args:
        input_tensor: Input tensor of shape (B, 1, H, W) and complex dtype,
        scale_factor: The positive integer by which to scale down the width and 
                      height

    Returns:
        A tensor of shape (B, 1, H/scale_factor, W/scale_factor) and complex dtype,
        where abs(output) is 1 or 0.
    """
    # Average the image along its real and imaginary components
    x_real = torch.real(input_tensor)
    x_imag = torch.imag(input_tensor)
    avg_real = F.avg_pool2d(x_real, kernel_size=scale_factor, stride=scale_factor)
    avg_imag = F.avg_pool2d(x_imag, kernel_size=scale_factor, stride=scale_factor)
    avg_tensor = torch.complex(avg_real, avg_imag)

    # Normalize results
    magnitude = torch.abs(avg_tensor)
    normalized_tensor = avg_tensor / magnitude
    output = torch.nan_to_num(normalized_tensor, nan=0.0)

    return output

def extract_gradients(
    images: torch.Tensor, # (B, 1, h, w)
    masks: torch.Tensor,
    integ_ks: int = 11,
    eps: float = 1e-12,
    max_visible_pixel_brightness: int = 30,
    coherence_threshold: float = 0.5,
    image_scale_factor: int = 2,
    mask_shrink_pixels: int = 4,
    border_padding: int = 10,
) -> torch.Tensor:
    """Extract the alignment directions from the input image using per-pixel gradients.

    Args:
        images: (B, 1, h, w) tensor of grayscale images to extract gradients from.
        masks: (B, 1, h, w) binary masks indicating where to extract gradients.
        integ_ks: Kernel size for coherence calculation.
        eps: Small value to avoid division by zero in coherence calculation.
        max_visible_pixel_brightness: Maximum brightness of pixels to consider for
            gradient extraction.
        coherence_threshold: Minimum coherence value to consider a gradient valid.
        image_scale_factor: Scale factor for subpixel gradient extraction.
        mask_shrink_pixels: Number of pixels to shrink the masks by before extracting
            gradients. Avoids sillhouette artifacts.
        border_padding: Number of pixels to pad around the border to avoid edge
            artifacts from mask shrinking at the image borders.
    Returns:
        Tensor of shape (B, 1, h, w) containing the extracted gradients as complex
            numbers. The real part is the x-grad and the imag part is the y-grad.
    """
    kernel = torch.tensor(
        [[-3., 0., 3.],
         [-10.,0.,10.],
         [-3., 0., 3.]]
    ).view(1,1,3,3).to(images.device)

    # Convert images to grayscale
    w = torch.tensor([0.2126, 0.7152, 0.0722], device=images.device, dtype=images.dtype).view(1,3,1,1)
    images = (images[:, :3] * w).sum(dim=1, keepdim=True)

    # Prevent artifacts at mask and border edges
    masks[masks < 0.5] = 0
    masks[masks >= 0.5] = 1
    if mask_shrink_pixels > 0:
        print(f"Shrinking masks by {mask_shrink_pixels} pixels for gradient extraction.")
        masks = shrink_masks(masks, mask_shrink_pixels)
    if border_padding > 0:
        print(f"Adding border padding of {border_padding} pixels for gradient extraction.")
        masks[:, :, :border_padding, :] = 0
        masks[:, :, -border_padding:, :] = 0
        masks[:, :, :, :border_padding] = 0
        masks[:, :, :, -border_padding:] = 0
    if image_scale_factor > 1:
        images = F.interpolate(images, scale_factor=image_scale_factor, mode='bilinear')
        masks = F.interpolate(masks, scale_factor=image_scale_factor, mode='bilinear')
    
    # Use kernels to extract alignment directions
    kx = kernel
    ky = kernel.transpose(-1, -2)
    pad_grad = kernel.shape[-1] // 2

    dx = F.conv2d(images, kx, padding=pad_grad)
    dy = F.conv2d(images, ky, padding=pad_grad)
    grad = torch.complex(dx, dy)
    grad = grad * 1j # rotate 90 deg to align gradient with grid lines
    grad[masks == 0] = 0
    mag = grad.abs()

    # Filter gradients
    filtered_grad = grad.clone()
    filtered_grad[masks == 0] = 0
    
    max_mags = torch.amax(mag, dim=(1, 2, 3), keepdim=True)
    filter_threshold = max_mags * max_visible_pixel_brightness / 255.0
    filtered_grad[mag < filter_threshold] = 0

    dx_filtered = filtered_grad.real
    dy_filtered = filtered_grad.imag
    
    pad_coh = integ_ks // 2
    Jxx = F.avg_pool2d(dx_filtered*dx_filtered, integ_ks, stride=1, padding=pad_coh)
    Jyy = F.avg_pool2d(dy_filtered*dy_filtered, integ_ks, stride=1, padding=pad_coh)
    Jxy = F.avg_pool2d(dx_filtered*dy_filtered, integ_ks, stride=1, padding=pad_coh)
    trace = Jxx + Jyy
    delta = torch.sqrt((Jxx - Jyy)**2 + 4.0*Jxy**2 + eps)
    lam1 = 0.5*(trace + delta)
    lam2 = 0.5*(trace - delta)
    coh  = (lam1 - lam2) / (lam1 + lam2 + eps)
    
    filtered_grad[coh < coherence_threshold] = 0

    if image_scale_factor > 1:
        filtered_grad = scale_down(filtered_grad, image_scale_factor)
    filtered_grad = filtered_grad / filtered_grad.abs().clamp_min(1e-12)
    return filtered_grad