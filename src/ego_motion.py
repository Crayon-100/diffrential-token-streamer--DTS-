"""Global Motion Estimator & Spatial Token Grid Warper.

Implements the Adaptive Dual-Path Ego-Motion Engine for Vision Transformers:
1. estimate_global_motion: Downsampled sub-pixel camera translation estimation via OpenCV phase correlation (< 0.1 ms).
2. warp_token_grid: Differentiable bilinear spatial token grid warping via F.grid_sample.
"""

from typing import Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


# Cache Hanning window to avoid reallocation
_HANNING_WINDOWS = {}


def _get_hanning_window(size: Tuple[int, int]) -> np.ndarray:
    if size not in _HANNING_WINDOWS:
        _HANNING_WINDOWS[size] = cv2.createHanningWindow(size, cv2.CV_32F)
    return _HANNING_WINDOWS[size]


def _to_grayscale_numpy(frame: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Converts a torch tensor or numpy array to a 2D float32 grayscale image."""
    if isinstance(frame, torch.Tensor):
        t = frame.detach().cpu()
        if t.ndim == 4:
            t = t[0]  # [3, H, W]
        if t.ndim == 3:
            # Check if channel-first [C, H, W]
            if t.shape[0] in (1, 3):
                t = t.permute(1, 2, 0)  # [H, W, C]
            arr = t.numpy()
        else:
            arr = t.numpy()
    else:
        arr = np.asarray(frame)

    if arr.ndim == 3:
        if arr.shape[2] == 3:
            gray = cv2.cvtColor((arr * 255).astype(np.uint8) if arr.dtype != np.uint8 else arr, cv2.COLOR_RGB2GRAY)
        elif arr.shape[2] == 1:
            gray = arr.squeeze(2)
        else:
            gray = arr[:, :, 0]
    else:
        gray = arr

    if gray.dtype != np.float32:
        gray = gray.astype(np.float32)

    return gray


def estimate_global_motion(
    frame_prev: Union[torch.Tensor, np.ndarray],
    frame_curr: Union[torch.Tensor, np.ndarray],
    downsample_size: Tuple[int, int] = (64, 64),
    orig_size: Tuple[int, int] = (224, 224),
) -> Tuple[float, float, float]:
    """Estimates sub-pixel camera translation (dx, dy) using downsampled phase correlation.

    Args:
        frame_prev: Previous video frame (Tensor or ndarray).
        frame_curr: Current video frame (Tensor or ndarray).
        downsample_size: Downsampling resolution for ultra-fast correlation (default: 64x64).
        orig_size: Original image frame resolution (default: 224x224).

    Returns:
        Tuple of:
        - dx: Horizontal shift in original pixel space (positive = right, negative = left).
        - dy: Vertical shift in original pixel space (positive = down, negative = up).
        - response: Phase correlation peak response confidence [0, 1].
    """
    if isinstance(downsample_size, int):
        downsample_size = (downsample_size, downsample_size)
    if isinstance(orig_size, int):
        orig_size = (orig_size, orig_size)

    g_prev = _to_grayscale_numpy(frame_prev)
    g_curr = _to_grayscale_numpy(frame_curr)

    # Downsample to 64x64 for < 0.1 ms execution
    s_prev = cv2.resize(g_prev, downsample_size, interpolation=cv2.INTER_AREA)
    s_curr = cv2.resize(g_curr, downsample_size, interpolation=cv2.INTER_AREA)

    hann = _get_hanning_window(downsample_size)
    shift, response = cv2.phaseCorrelate(s_prev, s_curr, hann)

    scale_x = orig_size[1] / float(downsample_size[1])
    scale_y = orig_size[0] / float(downsample_size[0])

    dx = float(shift[0] * scale_x)
    dy = float(shift[1] * scale_y)

    return dx, dy, float(response)


def warp_token_grid(
    tokens_ref: torch.Tensor,
    dx: float,
    dy: float,
    patch_size: int = 14,
    patch_grid: Tuple[int, int] = (16, 16),
    padding_mode: str = "border",
) -> torch.Tensor:
    """Spatially warps the reference token grid to align with camera ego-motion.

    Args:
        tokens_ref: Reference token tensor of shape [B, N, D] or [N, D].
        dx: Horizontal camera displacement in pixels (original image coordinates).
        dy: Vertical camera displacement in pixels (original image coordinates).
        patch_size: Pixel dimension per patch (default: 14 for DINOv2-ViT).
        patch_grid: Spatial patch dimensions (H_p, W_p) (default: (16, 16)).
        padding_mode: Padding mode for grid_sample ('border', 'zeros', 'reflection').

    Returns:
        Warped token tensor with identical shape and dtype on the same device.
    """
    # Shortcut for stationary frames
    if abs(dx) < 1e-4 and abs(dy) < 1e-4:
        return tokens_ref.clone()

    orig_dim = tokens_ref.ndim
    t = tokens_ref if orig_dim == 3 else tokens_ref.unsqueeze(0)  # [B, N, D]
    b, n, d = t.shape
    h_p, w_p = patch_grid

    if n != h_p * w_p:
        raise ValueError(f"Token count {n} does not match patch grid {h_p}x{w_p}={h_p*w_p}")

    # Convert image-space displacement into patch units
    dx_p = float(dx) / float(patch_size)
    dy_p = float(dy) / float(patch_size)

    # Rearrange to spatial feature map: [B, D, H_p, W_p]
    t_feat = rearrange(t, "b (h w) d -> b d h w", h=h_p, w=w_p)

    device = tokens_ref.device
    dtype = tokens_ref.dtype

    # Generate sampling grid
    y_coords, x_coords = torch.meshgrid(
        torch.arange(h_p, dtype=torch.float32, device=device),
        torch.arange(w_p, dtype=torch.float32, device=device),
        indexing="ij",
    )

    # In current frame (x, y), sample from reference location (x - dx_p, y - dy_p)
    src_x = x_coords - dx_p
    src_y = y_coords - dy_p

    # Normalize to [-1, 1] range for align_corners=True
    grid_x = 2.0 * src_x / max(w_p - 1, 1) - 1.0
    grid_y = 2.0 * src_y / max(h_p - 1, 1) - 1.0

    # Stack to [B, H_p, W_p, 2]
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)

    # Differentiable bilinear spatial warping
    warped_feat = F.grid_sample(
        t_feat,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )

    warped_tokens = rearrange(warped_feat, "b d h w -> b (h w) d").to(dtype=dtype)

    if orig_dim == 2:
        warped_tokens = warped_tokens.squeeze(0)

    return warped_tokens
