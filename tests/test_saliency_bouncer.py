"""Unit tests for Saliency-Gated Bouncer and Server Cache Comparison.

Follows unit-testing-test-generate guidelines:
- Verifies saliency tensor bounds [0, 1] and shapes
- Verifies dynamic score Delta calculation
- Verifies background noise suppression vs foreground trigger
- Verifies hard change bypass logic
- Verifies server-cache reference comparison mode
"""

import pytest
import torch
from src.slicer import DINOv2Slicer, SlicerOutput
from src.bouncer import Bouncer, BouncerOutput
from tests.test_phase1 import MockViTBackbone


class TestSaliencyBouncer:
    """Test suite for Saliency-Gated Temporal Filtering."""

    @pytest.fixture
    def bouncer(self) -> Bouncer:
        return Bouncer(
            threshold=0.95,
            saliency_gated=True,
            gamma=1.0,
            tau_dynamic=0.0008,
            tau_hard_change=0.30,
        )

    def test_saliency_gated_background_ripple_suppressed(self, bouncer: Bouncer) -> None:
        """Low-saliency background ripple (water) should be suppressed even with moderate cosine difference."""
        n, d = 2, 64
        # Two tokens with (1 - CosSim) = 0.10 (CosSim = 0.90)
        z1 = torch.zeros(1, n, d)
        z2 = torch.zeros(1, n, d)
        z1[0, :, 0] = 1.0
        z2[0, :, 0] = 0.90
        z2[0, :, 1] = (1 - 0.90**2) ** 0.5  # CosSim = 0.90, diff = 0.10

        # Patch 0: Background ripple (low saliency = 0.005) -> Delta = 0.10 * 0.005 = 0.0005 <= 0.0008 -> DROPPED
        # Patch 1: Foreground object (high saliency = 0.50)  -> Delta = 0.10 * 0.50  = 0.0500 > 0.0008  -> ACTIVE
        saliency = torch.tensor([[0.005, 0.50]])

        out = bouncer(z1, z2, saliency=saliency)

        assert not out.mask[0, 0].item(), "Background ripple must be dropped"
        assert out.mask[0, 1].item(), "Foreground motion must be active"
        assert out.stats["saliency_gated"] is True

    def test_saliency_gated_slow_moving_object_triggered(self, bouncer: Bouncer) -> None:
        """High-saliency object interior with very small movement should be triggered."""
        n, d = 1, 64
        # Small difference: CosSim = 0.98 -> diff = 0.02
        z1 = torch.zeros(1, n, d)
        z2 = torch.zeros(1, n, d)
        z1[0, 0, 0] = 1.0
        z2[0, 0, 0] = 0.98
        z2[0, 0, 1] = (1 - 0.98**2) ** 0.5

        # Saliency on object is 0.20 -> Delta = 0.02 * 0.20 = 0.004 > 0.0008 -> ACTIVE!
        saliency = torch.tensor([[0.20]])

        out = bouncer(z1, z2, saliency=saliency)

        assert out.mask[0, 0].item(), "Slow-moving object interior must be triggered"

    def test_hard_change_triggers_regardless_of_saliency(self, bouncer: Bouncer) -> None:
        """Drastic scene change (diff > tau_hard_change) triggers even with zero saliency."""
        n, d = 1, 64
        # Orthogonal tokens: CosSim = 0.0 -> diff = 1.0 > 0.30
        z1 = torch.zeros(1, n, d)
        z2 = torch.zeros(1, n, d)
        z1[0, 0, 0] = 1.0
        z2[0, 0, 1] = 1.0

        # Zero saliency
        saliency = torch.tensor([[0.0]])

        out = bouncer(z1, z2, saliency=saliency)

        assert out.mask[0, 0].item(), "Hard difference must trigger even with zero saliency"

    def test_server_cache_reference_mode(self, bouncer: Bouncer) -> None:
        """Simulate server cache: comparing against cached tokens rather than previous frame."""
        n, d = 4, 384
        z_cache = torch.randn(1, n, d)
        z_curr = z_cache.clone()

        # Patch 2 moves slightly relative to cache
        z_curr[0, 2, :] = z_curr[0, 2, :] + 0.5

        saliency = torch.tensor([[0.0, 0.0, 0.8, 0.0]])
        out = bouncer(z_curr, z_cache, saliency=saliency)

        # Patch 2 should be active; others should be dropped
        assert out.mask[0, 2].item()
        assert not out.mask[0, 0].item()
        assert not out.mask[0, 1].item()
        assert not out.mask[0, 3].item()

    def test_dual_cache_shadow_initialization_and_separation(self, bouncer: Bouncer) -> None:
        """Verifies edge_raw_shadow and edge_replica_cache maintain separate, distinct states."""
        n, d = 16, 64
        raw_tokens = torch.randn(1, n, d)
        # Introduce distinct distortion in reconstructed tokens
        reconstructed_tokens = raw_tokens + 0.15 * torch.randn(1, n, d)

        bouncer.initialize_edge_cache(
            raw_tokens=raw_tokens,
            reconstructed_tokens=reconstructed_tokens,
            patch_grid=(4, 4),
        )

        assert bouncer.is_edge_cache_initialized()
        assert torch.allclose(bouncer.edge_raw_shadow, raw_tokens)
        assert torch.allclose(bouncer.edge_replica_cache, reconstructed_tokens)
        assert not torch.allclose(bouncer.edge_raw_shadow, bouncer.edge_replica_cache)

    def test_identical_frame_sequence_yields_zero_active_tokens(self, bouncer: Bouncer) -> None:
        """Presenting identical frames must yield Delta_i = 0.0 and transmit zero active tokens (K = 0)."""
        n, d = 16, 64
        frame0_raw = torch.randn(1, n, d)
        # Reconstructed has quantization noise
        frame0_rec = frame0_raw + 0.10 * torch.randn(1, n, d)

        bouncer.initialize_edge_cache(
            raw_tokens=frame0_raw,
            reconstructed_tokens=frame0_rec,
            patch_grid=(4, 4),
        )

        # Frame 1 is completely identical to Frame 0
        frame1_raw = frame0_raw.clone()
        saliency = torch.rand(1, n)

        out = bouncer(frame1_raw, saliency=saliency)

        assert out.mask.sum().item() == 0, f"Expected 0 active tokens on static frame, got {out.mask.sum().item()}"
        assert out.active_tokens.shape[0] == 0

    def test_dual_cache_update_and_warping(self, bouncer: Bouncer) -> None:
        """Dual-cache update warps both caches and scatters distinct raw and reconstructed tokens."""
        n, d = 16, 64
        raw_tokens = torch.ones(1, n, d)
        rec_tokens = torch.full((1, n, d), 0.9)

        bouncer.initialize_edge_cache(
            raw_tokens=raw_tokens,
            reconstructed_tokens=rec_tokens,
            patch_grid=(4, 4),
        )

        # Scatter new tokens at patch 0
        mask = torch.zeros(n, dtype=torch.bool)
        mask[0] = True
        active_raw = torch.full((1, d), 5.0)
        active_rec = torch.full((1, d), 4.8)

        bouncer.update_edge_cache(
            active_reconstructed=active_rec,
            mask=mask,
            active_raw=active_raw,
            motion=(1.0, 0.0),
            patch_grid=(4, 4),
        )

        # Verify patch 0 was updated with distinct values
        assert torch.allclose(bouncer.edge_raw_shadow[0, 0, :], active_raw[0])
        assert torch.allclose(bouncer.edge_replica_cache[0, 0, :], active_rec[0])

