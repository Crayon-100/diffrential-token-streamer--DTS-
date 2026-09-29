"""Phase A: The Slicer (Edge Feature Extractor)

Extracts discrete semantic patch tokens from video frames using frozen DINOv2-ViT-Small.
Also extracts last-layer self-attention weights from the [CLS] token to patches as a
foreground saliency prior map.
"""

from dataclasses import dataclass
from typing import Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torchvision.transforms.functional as TF


@dataclass(frozen=True)
class SlicerOutput:
    """Output container for the Slicer module."""
    tokens: torch.Tensor          # Latent patch tokens [B, N, D]
    patch_grid: Tuple[int, int]   # (H_patches, W_patches) where N = H_patches * W_patches
    embedding_dim: int            # Latent dimension D (384 for ViT-Small)
    saliency: torch.Tensor        # Normalized foreground saliency map A_cls [B, N] in [0, 1]


class DINOv2Slicer(nn.Module):
    """Slicer module wrapping frozen DINOv2-ViT-Small for latent patch token extraction.
    
    Architecture Specification:
        - Backbone: DINOv2-ViT-Small (frozen)
        - Patch Size: 14 x 14
        - Latent Dimension (D): 384
        - Output: Sequence of latent patch tokens Z in R^{B x N x D}
        - Saliency Prior: Last-layer self-attention from [CLS] to patches in R^{B x N}
    """

    PATCH_SIZE: int = 14
    EMBEDDING_DIM: int = 384

    # ImageNet normalization statistics
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        device: Optional[Union[str, torch.device]] = None,
        backbone: Optional[nn.Module] = None,
    ) -> None:
        """Initialize the DINOv2 Slicer.

        Args:
            model_name: DINOv2 model variant from torch.hub (default: 'dinov2_vits14').
            device: Target torch device. If None, auto-selects CUDA if available else CPU.
            backbone: Optional pre-loaded or mock ViT backbone for offline testing.
        """
        super().__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if backbone is not None:
            self.backbone = backbone
        else:
            # Load official DINOv2 model from torch hub
            self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)

        # Strictly freeze backbone weights and set to eval mode
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone.to(self.device)

        # Register ImageNet normalization buffers
        self.register_buffer(
            "mean",
            torch.tensor(self.IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor(self.IMAGENET_STD, device=self.device).view(1, 3, 1, 1),
            persistent=False,
        )

    def preprocess_frame(self, frame: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Preprocesses input frame into standardized tensor ready for DINOv2.

        Accepts:
            - [H, W, C] or [C, H, W] or [B, C, H, W] tensor in range [0, 1] or [0, 255].
        Returns:
            - Normalized tensor [B, 3, H_aligned, W_aligned] on self.device where dimensions
              are exact multiples of 14.
            - Grid dimensions (H_patches, W_patches).
        """
        x = frame.to(dtype=torch.float32)

        # Standardize input dimensions
        if x.ndim == 3:
            # Check if channel last: [H, W, C] where C is 3
            if x.shape[-1] == 3 and x.shape[0] != 3:
                x = rearrange(x, "h w c -> c h w")
            x = x.unsqueeze(0)  # [1, C, H, W]
        elif x.ndim == 4:
            # Check if channel last: [B, H, W, C]
            if x.shape[-1] == 3 and x.shape[1] != 3:
                x = rearrange(x, "b h w c -> b c h w")
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {frame.shape}")

        if x.shape[1] != 3:
            raise ValueError(f"Expected 3 color channels (RGB), got {x.shape[1]}")

        # Scale to [0, 1] if values are in [0, 255]
        if x.max() > 1.0:
            x = x / 255.0

        # Ensure spatial dimensions are multiples of PATCH_SIZE (14)
        _, _, h, w = x.shape
        h_aligned = (h // self.PATCH_SIZE) * self.PATCH_SIZE
        w_aligned = (w // self.PATCH_SIZE) * self.PATCH_SIZE

        if h_aligned == 0 or w_aligned == 0:
            raise ValueError(
                f"Image resolution {h}x{w} is smaller than patch size {self.PATCH_SIZE}."
            )

        if h != h_aligned or w != w_aligned:
            # Resize image to nearest multiple of 14 using bilinear interpolation
            x = F.interpolate(
                x,
                size=(h_aligned, w_aligned),
                mode="bilinear",
                align_corners=False,
            )

        # Move to target device and apply ImageNet normalization
        x = x.to(self.device)
        x = (x - self.mean) / self.std

        h_patches = h_aligned // self.PATCH_SIZE
        w_patches = w_aligned // self.PATCH_SIZE

        return x, (h_patches, w_patches)

    def _compute_saliency_from_attn(
        self, attn_input: torch.Tensor, expected_n: int
    ) -> torch.Tensor:
        """Computes normalized CLS-to-patch attention saliency map."""
        attn_mod = self.backbone.blocks[-1].attn
        b, n_tokens, c = attn_input.shape
        head_dim = c // attn_mod.num_heads
        scale = head_dim ** -0.5
        qkv = attn_mod.qkv(attn_input).reshape(b, n_tokens, 3, attn_mod.num_heads, head_dim)
        q, k, v = torch.unbind(qkv, 2)
        q = q.transpose(1, 2)  # [B, num_heads, N, head_dim]
        k = k.transpose(1, 2)  # [B, num_heads, N, head_dim]

        attn_scores = (q @ k.transpose(-2, -1)) * scale  # [B, num_heads, N, N]
        attn_weights = attn_scores.softmax(dim=-1)        # [B, num_heads, N, N]

        # CLS token is at index 0; spatial patches are at the end:
        cls_patch_attn = attn_weights[:, :, 0, -expected_n:].mean(dim=1)  # [B, N]

        # Normalize to [0, 1] per batch item
        min_val = cls_patch_attn.amin(dim=-1, keepdim=True)
        max_val = cls_patch_attn.amax(dim=-1, keepdim=True)
        saliency = (cls_patch_attn - min_val) / (max_val - min_val + 1e-8)
        return saliency

    @torch.no_grad()
    def forward(self, frame: torch.Tensor) -> SlicerOutput:
        """Extracts latent patch tokens and CLS attention saliency map from frame tensor.

        Args:
            frame: Input image or batch tensor.

        Returns:
            SlicerOutput containing:
                - tokens: Latent patch tokens [B, N, D]
                - patch_grid: (H_patches, W_patches)
                - embedding_dim: D (384 for ViT-Small)
                - saliency: Normalized [CLS] attention saliency prior [B, N] in [0, 1]
        """
        x, (h_patches, w_patches) = self.preprocess_frame(frame)
        b = x.shape[0]
        expected_n = h_patches * w_patches

        # Hook to capture input to the last attention block
        saved_attn_input = {}
        hook_handle = None
        has_blocks = (
            hasattr(self.backbone, "blocks")
            and len(self.backbone.blocks) > 0
            and hasattr(self.backbone.blocks[-1], "attn")
        )

        if has_blocks:
            def hook_fn(module, inp, out):
                saved_attn_input["input"] = inp[0]
            hook_handle = self.backbone.blocks[-1].attn.register_forward_hook(hook_fn)

        try:
            # Extract features using DINOv2
            if hasattr(self.backbone, "forward_features"):
                features = self.backbone.forward_features(x)
                if isinstance(features, dict) and "x_norm_patchtokens" in features:
                    patch_tokens = features["x_norm_patchtokens"]
                elif isinstance(features, dict) and "x_prenorm" in features:
                    patch_tokens = features["x_prenorm"][:, -expected_n:, :]
                else:
                    raise RuntimeError(
                        f"Unexpected features dict returned from backbone: {features.keys()}"
                    )
            elif hasattr(self.backbone, "get_intermediate_layers"):
                layers = self.backbone.get_intermediate_layers(
                    x, n=1, return_class_token=True
                )
                if isinstance(layers[0], (tuple, list)):
                    patch_tokens = layers[0][0]
                else:
                    patch_tokens = layers[0]
            else:
                out = self.backbone(x)
                if out.ndim == 3 and out.shape[1] > expected_n:
                    patch_tokens = out[:, -expected_n:, :]
                else:
                    patch_tokens = out
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        # Compute or fallback saliency
        if "input" in saved_attn_input:
            saliency = self._compute_saliency_from_attn(saved_attn_input["input"], expected_n)
        elif hasattr(self.backbone, "mock_saliency"):
            saliency = self.backbone.mock_saliency.to(self.device)
        else:
            # Neutral default fallback
            saliency = torch.full((b, expected_n), 0.5, device=self.device)

        # Validate extracted token shapes: [B, N, D]
        if patch_tokens.shape[1] != expected_n:
            raise ValueError(
                f"Extracted token sequence length {patch_tokens.shape[1]} does not match "
                f"expected patch count {expected_n} ({h_patches}x{w_patches})."
            )

        return SlicerOutput(
            tokens=patch_tokens,
            patch_grid=(h_patches, w_patches),
            embedding_dim=patch_tokens.shape[-1],
            saliency=saliency,
        )
