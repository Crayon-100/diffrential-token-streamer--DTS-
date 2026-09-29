"""Phase B: The Bouncer (Semantic Temporal Filter)

Filters static/redundant tokens across successive video frames using cosine similarity
and optional Saliency-Gated dynamic prior weighting.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from src.ego_motion import warp_token_grid


@dataclass(frozen=True)
class BouncerOutput:
    """Output container for the Bouncer module."""
    mask: torch.Tensor                  # Boolean mask [B, N]: True for active/dynamic tokens
    drop_mask: torch.Tensor             # Boolean mask [B, N]: True for static/dropped tokens
    active_tokens: torch.Tensor         # Dynamic tokens subset Z_active [K, D] or [B, K_max, D]
    cosine_similarities: torch.Tensor   # Per-token cosine similarity S [B, N]
    spatial_mask: Optional[torch.Tensor]# 2D spatial boolean mask [B, H_p, W_p] if grid provided
    stats: Dict[str, Any]               # Summary metrics (drop_ratio, active_count, etc.)
    dynamic_scores: Optional[torch.Tensor] = None  # Saliency-weighted dynamic score Delta [B, N]
    saliency: Optional[torch.Tensor] = None        # Saliency prior map [B, N]
    warped_reference: Optional[torch.Tensor] = None # Warped reference tokens (if motion path used)


class Bouncer(nn.Module):
    """Bouncer module for temporal semantic filtering of vision transformer tokens.

    Supports operational profiles:
    - 'static': Fast-Path (direct cache comparison, 0 ms overhead).
    - 'motion': Warp-Path (align reference tokens using warp_token_grid).
    - 'auto'  : Adaptive dual-path (Warp-Path if motion >= motion_threshold else Fast-Path).
    """

    def __init__(
        self,
        threshold: float = 0.95,
        saliency_gated: bool = False,
        gamma: float = 1.0,
        tau_dynamic: float = 0.0008,
        tau_hard_change: float = 0.30,
        profile: str = "auto",
        motion_threshold: float = 0.5,
        eps: float = 1e-8,
    ) -> None:
        """Initialize the Bouncer.

        Args:
            threshold: Cosine similarity threshold for standard mode (default: 0.95).
            saliency_gated: Whether to activate saliency-gated dynamic scoring by default.
            gamma: Exponent modulating foreground saliency emphasis (default: 1.0).
            tau_dynamic: Dynamic score threshold for saliency-gated activation (default: 0.0008).
            tau_hard_change: Hard difference threshold triggering transmission regardless of saliency (default: 0.30).
            profile: Operational profile in ['static', 'motion', 'auto'] (default: 'auto').
            motion_threshold: Camera motion magnitude threshold epsilon in pixels (default: 0.5).
            eps: Numerical stability constant.
        """
        super().__init__()
        self.threshold = float(threshold)
        self.saliency_gated = bool(saliency_gated)
        self.gamma = float(gamma)
        self.tau_dynamic = float(tau_dynamic)
        self.tau_hard_change = float(tau_hard_change)
        self.profile = str(profile).lower()
        self.motion_threshold = float(motion_threshold)
        self.eps = float(eps)
        self.register_buffer("edge_raw_shadow", None, persistent=False)
        self.register_buffer("edge_replica_cache", None, persistent=False)
        self.edge_patch_grid: Optional[Tuple[int, int]] = None
        self._last_raw_tokens: Optional[torch.Tensor] = None

    def initialize_edge_cache(
        self,
        raw_tokens: torch.Tensor,
        reconstructed_tokens: Optional[torch.Tensor] = None,
        patch_grid: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Initializes both the Edge Raw Shadow and the Edge Replica Cache.

        Args:
            raw_tokens: Initial uncompressed float tokens [B, N, D] or [N, D] (transmitted keyframe).
            reconstructed_tokens: Initial reconstructed tokens [B, N, D] or [N, D] from RVQ.
                                  If None, falls back to raw_tokens.
            patch_grid: Spatial patch dimensions (H_p, W_p).
        """
        t_raw = raw_tokens.detach()
        if t_raw.ndim == 2:
            t_raw = t_raw.unsqueeze(0)

        t_rec = reconstructed_tokens.detach() if reconstructed_tokens is not None else t_raw.clone()
        if t_rec.ndim == 2:
            t_rec = t_rec.unsqueeze(0)

        self.edge_raw_shadow = t_raw.clone()
        self.edge_replica_cache = t_rec.clone()
        self.edge_patch_grid = patch_grid

    def is_edge_cache_initialized(self) -> bool:
        """Returns True if the Edge Dual-Caches are initialized."""
        return self.edge_raw_shadow is not None and self.edge_replica_cache is not None

    def reset_edge_cache(self) -> None:
        """Resets both Edge Dual-Caches."""
        self.edge_raw_shadow = None
        self.edge_replica_cache = None
        self.edge_patch_grid = None
        self._last_raw_tokens = None

    def update_edge_cache(
        self,
        active_reconstructed: torch.Tensor,
        mask: torch.Tensor,
        active_raw: Optional[torch.Tensor] = None,
        motion: Optional[Tuple[float, float]] = None,
        patch_grid: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Dual-Cache Closed-Loop Update:
        1. Warps BOTH edge_raw_shadow and edge_replica_cache if camera motion occurred.
        2. Scatter-updates active uncompressed float tokens into edge_raw_shadow.
        3. Scatter-updates active de-quantized RVQ tokens into edge_replica_cache.

        Args:
            active_reconstructed: De-quantized tokens Z_hat in R^[K, D] from the Packer.
            mask: Boolean spatial mask [N] where True indicates active tokens.
            active_raw: Optional ground-truth uncompressed float tokens in R^[K, D].
                        If None, extracts from self._last_raw_tokens.
            motion: Optional camera translation tuple (dx, dy) in pixels.
            patch_grid: Optional spatial patch dimensions (H_p, W_p).
        """
        if self.edge_raw_shadow is None or self.edge_replica_cache is None:
            raise RuntimeError("Edge dual-caches are not initialized. Call initialize_edge_cache first.")

        grid = patch_grid or self.edge_patch_grid or (16, 16)

        # 1. Dual-Grid Warping: Warp BOTH caches if camera motion occurred
        if motion is not None:
            dx, dy = float(motion[0]), float(motion[1])
            if abs(dx) > 1e-4 or abs(dy) > 1e-4:
                self.edge_raw_shadow = warp_token_grid(
                    self.edge_raw_shadow, dx=dx, dy=dy, patch_grid=grid
                )
                self.edge_replica_cache = warp_token_grid(
                    self.edge_replica_cache, dx=dx, dy=dy, patch_grid=grid
                )

        # 2. Scatter-update active patches
        mask_flat = mask.flatten().bool()
        k_active = int(mask_flat.sum().item())
        if k_active > 0:
            if active_raw is not None:
                raw_to_scatter = active_raw
            elif self._last_raw_tokens is not None:
                raw_to_scatter = self._last_raw_tokens[0, mask_flat, :]
            else:
                raw_to_scatter = active_reconstructed

            self.edge_raw_shadow[0, mask_flat, :] = raw_to_scatter.to(
                self.edge_raw_shadow.device, dtype=self.edge_raw_shadow.dtype
            )
            self.edge_replica_cache[0, mask_flat, :] = active_reconstructed.to(
                self.edge_replica_cache.device, dtype=self.edge_replica_cache.dtype
            )

    def compute_similarity(
        self, tokens_current: torch.Tensor, tokens_previous: torch.Tensor
    ) -> torch.Tensor:
        """Computes per-patch cosine similarity between current and previous/reference tokens.

        Args:
            tokens_current: Z_t of shape [B, N, D] or [N, D].
            tokens_previous: Z_ref of shape [B, N, D] or [N, D].

        Returns:
            Cosine similarities S of shape [B, N] in range [-1.0, 1.0].
        """
        if tokens_current.shape != tokens_previous.shape:
            raise ValueError(
                f"Shape mismatch: current tokens {tokens_current.shape} != "
                f"previous tokens {tokens_previous.shape}"
            )

        # Ensure batch dimension [B, N, D]
        if tokens_current.ndim == 2:
            tokens_current = tokens_current.unsqueeze(0)
            tokens_previous = tokens_previous.unsqueeze(0)
        elif tokens_current.ndim != 3:
            raise ValueError(
                f"Expected 2D or 3D token tensor, got shape {tokens_current.shape}"
            )

        similarity = F.cosine_similarity(
            tokens_current, tokens_previous, dim=-1, eps=self.eps
        )  # [B, N]

        return similarity

    def forward(
        self,
        tokens_current: torch.Tensor,
        tokens_previous: Optional[torch.Tensor] = None,
        patch_grid: Optional[Tuple[int, int]] = None,
        saliency: Optional[torch.Tensor] = None,
        gamma: Optional[float] = None,
        tau_dynamic: Optional[float] = None,
        tau_hard_change: Optional[float] = None,
        motion: Optional[Tuple[float, float]] = None,
        profile: Optional[str] = None,
    ) -> BouncerOutput:
        """Filters tokens based on temporal cosine similarity or saliency gating.

        Supports operational profiles:
        - 'static': Fast-Path (direct cache comparison, 0 ms overhead).
        - 'motion': Warp-Path (align reference tokens using warp_token_grid).
        - 'auto'  : Adaptive dual-path (Warp-Path if motion >= motion_threshold else Fast-Path).

        Args:
            tokens_current: Z_t of shape [B, N, D] or [N, D].
            tokens_previous: Optional Z_ref of shape [B, N, D] or [N, D]. If None, uses self.edge_replica_cache.
            patch_grid: Optional (H_patches, W_patches) to reshape mask into 2D spatial grid.
            saliency: Optional [B, N] or [N] normalized [CLS] attention prior in [0, 1].
            gamma: Optional override for saliency exponent.
            tau_dynamic: Optional override for dynamic score threshold.
            tau_hard_change: Optional override for hard change threshold.
            motion: Optional camera translation tuple (dx, dy) in pixels.
            profile: Optional override for operational profile ('static', 'motion', 'auto').

        Returns:
            BouncerOutput with boolean mask, active tokens, and diagnostics.
        """
        orig_dim = tokens_current.ndim
        if orig_dim == 2:
            z_curr = tokens_current.unsqueeze(0)
        else:
            z_curr = tokens_current
        self._last_raw_tokens = z_curr

        if tokens_previous is None:
            if self.is_edge_cache_initialized():
                # Lever B: Decouple physical motion detection from quantization distortion
                # Compare raw-to-raw against uncompressed edge_raw_shadow!
                tokens_previous = self.edge_raw_shadow
            else:
                raise ValueError(
                    "tokens_previous was not provided and edge_raw_shadow is not initialized. "
                    "Either pass tokens_previous or call initialize_edge_cache first."
                )

        if tokens_previous.ndim == 2:
            z_prev = tokens_previous.unsqueeze(0)
        else:
            z_prev = tokens_previous

        grid_shape = patch_grid if patch_grid is not None else (16, 16)

        # Resolve operational profile and motion routing
        prof = (profile if profile is not None else self.profile).lower()
        if motion is not None:
            dx, dy = float(motion[0]), float(motion[1])
            motion_mag = (dx ** 2 + dy ** 2) ** 0.5
        else:
            dx, dy = 0.0, 0.0
            motion_mag = 0.0

        # Dual-Path Routing
        warped_ref = None
        if prof == "motion":
            path = "warp"
            z_ref_eff = warp_token_grid(z_prev, dx=dx, dy=dy, patch_grid=grid_shape)
            warped_ref = z_ref_eff
        elif prof == "static":
            path = "fast"
            z_ref_eff = z_prev
        else:  # 'auto'
            if motion is not None and motion_mag >= self.motion_threshold:
                path = "warp"
                z_ref_eff = warp_token_grid(z_prev, dx=dx, dy=dy, patch_grid=grid_shape)
                warped_ref = z_ref_eff
            else:
                path = "fast"
                z_ref_eff = z_prev

        # 1. Compute cosine similarity per patch: [B, N] against effective reference
        sim = self.compute_similarity(z_curr, z_ref_eff)

        # 2. Filtering Logic
        use_saliency = (saliency is not None) or self.saliency_gated
        delta_scores = None

        if use_saliency:
            g = float(gamma if gamma is not None else self.gamma)
            t_dyn = float(tau_dynamic if tau_dynamic is not None else self.tau_dynamic)
            t_hard = float(tau_hard_change if tau_hard_change is not None else self.tau_hard_change)

            if saliency is not None:
                sal = saliency if saliency.ndim == 2 else saliency.unsqueeze(0)
                sal = sal.to(sim.device)
            else:
                sal = torch.ones_like(sim)

            # Difference: 1.0 - CosSim (range 0 to 2)
            diff = (1.0 - sim).clamp(min=0.0)

            # Saliency-gated dynamic score: Delta_i = diff * (saliency ** gamma)
            delta_scores = diff * (sal.clamp(0.0, 1.0) ** g)

            # Dual-threshold activation:
            # Active if Delta > tau_dynamic OR diff > tau_hard_change
            active_mask = (delta_scores > t_dyn) | (diff > t_hard)
            drop_mask = ~active_mask
        else:
            # Standard Cosine Threshold Mode:
            drop_mask = sim > self.threshold
            active_mask = ~drop_mask

        # 3. Extract isolated dynamic tokens Z_active
        active_tokens = z_curr[active_mask]  # [K, D]

        # 4. Generate 2D spatial mask if patch_grid provided
        spatial_mask = None
        if patch_grid is not None:
            hp, wp = patch_grid
            if hp * wp != sim.shape[1]:
                raise ValueError(
                    f"Patch grid {hp}x{wp}={hp*wp} does not match sequence length {sim.shape[1]}"
                )
            spatial_mask = rearrange(active_mask, "b (h w) -> b h w", h=hp, w=wp)

        # 5. Calculate statistics
        total_tokens = int(active_mask.numel())
        active_count = int(active_mask.sum().item())
        dropped_count = total_tokens - active_count
        drop_ratio = dropped_count / total_tokens if total_tokens > 0 else 0.0
        keep_ratio = active_count / total_tokens if total_tokens > 0 else 0.0

        stats = {
            "total_tokens": total_tokens,
            "active_tokens": active_count,
            "dropped_tokens": dropped_count,
            "drop_ratio": drop_ratio,
            "keep_ratio": keep_ratio,
            "mean_similarity": float(sim.mean().item()),
            "min_similarity": float(sim.min().item()),
            "max_similarity": float(sim.max().item()),
            "threshold": self.threshold,
            "saliency_gated": use_saliency,
            "profile": prof,
            "ego_motion_path": path,
            "motion": (dx, dy) if motion is not None else None,
            "motion_magnitude": motion_mag,
        }

        return BouncerOutput(
            mask=active_mask.squeeze(0) if orig_dim == 2 else active_mask,
            drop_mask=drop_mask.squeeze(0) if orig_dim == 2 else drop_mask,
            active_tokens=active_tokens,
            cosine_similarities=sim.squeeze(0) if orig_dim == 2 else sim,
            spatial_mask=spatial_mask.squeeze(0) if (spatial_mask is not None and orig_dim == 2) else spatial_mask,
            stats=stats,
            dynamic_scores=delta_scores.squeeze(0) if (delta_scores is not None and orig_dim == 2) else delta_scores,
            saliency=saliency.squeeze(0) if (saliency is not None and orig_dim == 2) else saliency,
            warped_reference=warped_ref.squeeze(0) if (warped_ref is not None and orig_dim == 2) else warped_ref,
        )
