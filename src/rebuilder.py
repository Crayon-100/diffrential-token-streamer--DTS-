"""Phase D: The Rebuilder (Server Execution)

Receives boolean spatial mask M and discrete RVQ indices, looks up continuous vectors
from an identical codebook, selectively updates temporal token cache where M=1, and
executes downstream attention layers.
"""

from dataclasses import dataclass
from typing import Optional, Union, Tuple, Dict, Any
import torch
import torch.nn as nn
from src.packer import Packer, TransmissionPacket, from_bytes
from src.ego_motion import warp_token_grid


@dataclass(frozen=True)
class RebuilderOutput:
    """Output container for the Rebuilder module."""
    refreshed_tokens: torch.Tensor      # Complete refreshed token sequence Z_server [B, N, D]
    downstream_output: torch.Tensor     # Output after downstream attention layers [B, N, D]
    num_refreshed: int                  # Number of tokens updated (M = 1)
    num_cached: int                     # Number of tokens retained from cache (M = 0)
    cache_refresh_ratio: float          # num_refreshed / total_tokens


class Rebuilder(nn.Module):
    """Rebuilder module for server-side split-inference execution.

    Architecture Specification:
        - Receives boolean spatial mask M and RVQ integer indices.
        - Looks up continuous vectors from identical codebook.
        - Updates temporal token cache by overwriting only tokens where M = 1.
        - Feeds refreshed token sequence into downstream attention layers.
    """

    def __init__(
        self,
        packer: Optional[Packer] = None,
        dim: int = 384,
        num_quantizers: int = 4,
        codebook_size: int = 256,
        codebook_path: Optional[str] = None,
        num_heads: int = 6,
        num_downstream_layers: int = 2,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Initialize the Rebuilder.

        Args:
            packer: Pre-existing Packer module (shares codebook weights). If None, initializes
                    an identical codebook.
            dim: Latent token dimension (default: 384).
            num_quantizers: Number of RVQ stages Q (default: 4).
            codebook_size: Vocabulary size V (default: 256).
            codebook_path: Optional path to frozen pretrained codebook checkpoint.
            num_heads: Attention heads for downstream transformer (default: 6, since 384/6 = 64).
            num_downstream_layers: Number of downstream transformer encoder layers (default: 2).
            device: Torch device (defaults to CUDA if available else CPU).
        """
        super().__init__()
        self.dim = int(dim)
        self.num_quantizers = int(num_quantizers)
        self.codebook_size = int(codebook_size)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 1. Server-side codebook for index lookup
        if packer is not None:
            self.codebook = packer
        else:
            self.codebook = Packer(
                dim=self.dim,
                num_quantizers=self.num_quantizers,
                codebook_size=self.codebook_size,
                kmeans_init=False,
                codebook_path=codebook_path,
                device=self.device,
            )

        # 2. Downstream Transformer Attention Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=num_heads,
            dim_feedforward=self.dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.downstream_attention = nn.TransformerEncoder(
            encoder_layer, num_layers=num_downstream_layers
        ).to(self.device)
        self.downstream_attention.eval()

        # 3. Persistent Temporal Token Cache: [B, N, D]
        self.register_buffer("token_cache", None, persistent=False)
        self.patch_grid: Optional[Tuple[int, int]] = None

    @property
    def packer(self) -> Packer:
        """Alias for codebook module."""
        return self.codebook

    @classmethod
    def load_pretrained(
        cls,
        path: str = "models/rvq_codebook_davis_train.pt",
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> "Rebuilder":
        """Loads a Rebuilder instance with frozen pretrained codebook weights."""
        packer = Packer.load_pretrained(path=path, device=device)
        return cls(packer=packer, device=device, **kwargs)

    def initialize_cache(
        self,
        tokens: torch.Tensor,
        patch_grid: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Initializes the temporal token cache (e.g. from an initial keyframe).

        Args:
            tokens: Initial full token tensor of shape [B, N, D] or [N, D].
            patch_grid: Spatial patch dimensions (H_p, W_p).
        """
        t = tokens.detach().to(self.device)
        if t.ndim == 2:
            t = t.unsqueeze(0)  # [1, N, D]

        self.token_cache = t.clone()
        self.patch_grid = patch_grid

    def is_cache_initialized(self) -> bool:
        """Returns True if the server token cache is initialized."""
        return self.token_cache is not None

    def reset_cache(self) -> None:
        """Clears the temporal token cache."""
        self.token_cache = None
        self.patch_grid = None

    def decode_tokens(self, indices: torch.Tensor) -> torch.Tensor:
        """Decodes RVQ discrete indices into continuous token vectors."""
        return self.codebook.decode(indices)

    @torch.no_grad()
    def forward(
        self,
        packet: Union[TransmissionPacket, bytes, bytearray, Tuple[torch.Tensor, torch.Tensor]],
        patch_grid: Optional[Tuple[int, int]] = None,
        motion: Optional[Tuple[float, float]] = None,
    ) -> RebuilderOutput:
        """Reconstructs dynamic tokens, refreshes temporal cache, and executes downstream attention.

        Args:
            packet: TransmissionPacket, raw binary wire bytes, or tuple of (mask, indices).
            patch_grid: Optional patch grid shape (H_p, W_p).
            motion: Optional camera translation tuple (dx, dy) in pixels.

        Returns:
            RebuilderOutput with refreshed token sequence and downstream attention activations.
        """
        # Parse inputs
        motion_vec = motion
        if isinstance(packet, (bytes, bytearray)):
            packet = from_bytes(bytes(packet))

        if isinstance(packet, TransmissionPacket):
            mask = packet.unpack_mask().to(self.device)
            indices = packet.unpack_indices(self.device)
            grid = packet.patch_grid
            k_active = packet.num_active
            n_patches = packet.num_patches
            if motion_vec is None and packet.motion is not None:
                motion_vec = packet.motion
        else:
            raw_mask, indices = packet
            mask = raw_mask.flatten().bool().to(self.device)
            indices = indices.to(self.device)
            k_active = int(mask.sum().item())
            n_patches = int(mask.numel())
            grid = patch_grid

        eff_grid = grid if grid is not None else (self.patch_grid if self.patch_grid is not None else (16, 16))

        # 1. Ensure cache is ready
        if not self.is_cache_initialized():
            # If uninitialized, allocate zeros [1, N, D]
            self.token_cache = torch.zeros(
                (1, n_patches, self.dim), dtype=torch.float32, device=self.device
            )
            self.patch_grid = eff_grid
        else:
            # Warp persistent token cache if camera motion occurred
            if motion_vec is not None:
                dx, dy = float(motion_vec[0]), float(motion_vec[1])
                if abs(dx) > 1e-4 or abs(dy) > 1e-4:
                    self.token_cache = warp_token_grid(
                        self.token_cache, dx=dx, dy=dy, patch_grid=eff_grid
                    )

        # 2. Decode active tokens via codebook lookup
        if k_active > 0:
            z_hat_active = self.codebook.decode(indices)  # [K, D]

            # 3. Selective In-Place Cache Refresh
            # Overwrite only where M = 1 (dynamic/active)
            self.token_cache[0, mask, :] = z_hat_active.to(self.token_cache.dtype)

        # Tokens where M = 0 remain unchanged in self.token_cache!
        num_refreshed = k_active
        num_cached = n_patches - k_active
        refresh_ratio = num_refreshed / n_patches if n_patches > 0 else 0.0

        # 4. Feed refreshed token sequence into downstream attention layers
        refreshed_tokens = self.token_cache.clone()
        downstream_activations = self.downstream_attention(refreshed_tokens)

        return RebuilderOutput(
            refreshed_tokens=refreshed_tokens,
            downstream_output=downstream_activations,
            num_refreshed=num_refreshed,
            num_cached=num_cached,
            cache_refresh_ratio=refresh_ratio,
        )
