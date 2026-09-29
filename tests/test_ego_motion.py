"""Unit testing suite for Phase I: Adaptive Dual-Path Ego-Motion Engine.

Tests:
1. Global motion estimation with phase correlation (accuracy & latency).
2. Differentiable token grid warping (integer and sub-pixel shifts).
3. Bouncer operational profiles ('static', 'motion', 'auto') and dual-path routing.
4. TransmissionPacket motion metadata encoding & wire byte accounting.
5. Rebuilder cache shifting under motion prior to active token scatter.
"""

import time
import pytest
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from src.ego_motion import estimate_global_motion, warp_token_grid
from src.bouncer import Bouncer
from src.packer import Packer, TransmissionPacket
from src.rebuilder import Rebuilder


def create_synthetic_texture(h=224, w=224):
    """Generates a structured image with distinct landmarks and smooth gradients."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(img, (30, 40), (100, 120), (200, 180, 150), -1)
    cv2.circle(img, (150, 160), 35, (100, 220, 80), -1)
    cv2.circle(img, (70, 180), 20, (180, 50, 230), -1)
    cv2.putText(img, "DINO", (60, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    # Add gentle gradients
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img_f = np.clip(img.astype(np.float32) + x[..., None] * 0.2 + y[..., None] * 0.1, 0, 255).astype(np.uint8)
    return img_f


class TestGlobalMotionEstimator:
    """Tests for phase correlation global camera translation estimator."""

    def test_zero_motion_identical_frames(self):
        """Identical frames should produce near-zero motion (< 0.1 px)."""
        frame = create_synthetic_texture(224, 224)
        dx, dy, resp = estimate_global_motion(frame, frame, downsample_size=64)
        assert abs(dx) < 0.1, f"Expected dx near 0, got {dx}"
        assert abs(dy) < 0.1, f"Expected dy near 0, got {dy}"
        assert resp > 0.99

    def test_synthetic_panning_accuracy(self):
        """Synthetic translation should be recovered accurately within sub-pixel bounds."""
        frame = create_synthetic_texture(224, 224)

        test_shifts = [
            (7.0, 0.0),    # +7 px horizontal (0.5 patch)
            (-14.0, 0.0),  # -14 px horizontal (-1.0 patch)
            (0.0, 10.5),   # +10.5 px vertical
            (-7.0, 7.0),   # Diagonal shift (0.5 patch diagonal)
        ]

        for true_dx, true_dy in test_shifts:
            M = np.float32([[1, 0, true_dx], [0, 1, true_dy]])
            shifted_frame = cv2.warpAffine(
                frame, M, (224, 224), borderMode=cv2.BORDER_REFLECT
            )

            est_dx, est_dy, resp = estimate_global_motion(frame, shifted_frame, downsample_size=64)

            err_x = abs(est_dx - true_dx)
            err_y = abs(est_dy - true_dy)

            # 64x64 downsampling gives sub-pixel precision in downsampled domain (~0.3px in 64x64 -> ~1.0px in 224x224)
            assert err_x < 1.2, f"Shift ({true_dx}, {true_dy}): err_x={err_x:.3f} >= 1.2 (est={est_dx:.2f})"
            assert err_y < 1.2, f"Shift ({true_dx}, {true_dy}): err_y={err_y:.3f} >= 1.2 (est={est_dy:.2f})"
            assert resp > 0.85, f"Response {resp:.3f} below 0.85"

    def test_estimation_latency(self):
        """Phase correlation on 64x64 frames must execute in under 1.0 ms."""
        frame1 = create_synthetic_texture(224, 224)
        frame2 = np.roll(frame1, 5, axis=1)

        # Warmup
        for _ in range(5):
            estimate_global_motion(frame1, frame2, downsample_size=64)

        iterations = 50
        start = time.perf_counter()
        for _ in range(iterations):
            estimate_global_motion(frame1, frame2, downsample_size=64)
        elapsed_ms = ((time.perf_counter() - start) / iterations) * 1000

        assert elapsed_ms < 1.0, f"Average latency {elapsed_ms:.3f} ms exceeds 1.0 ms target"


class TestTokenGridWarping:
    """Tests for spatial token grid warping via F.grid_sample."""

    def test_zero_shift_identity(self):
        """Zero shift should preserve the input tokens perfectly."""
        tokens = torch.randn(1, 256, 384)
        warped = warp_token_grid(tokens, dx=0.0, dy=0.0, patch_grid=(16, 16))
        assert torch.allclose(tokens, warped, atol=1e-5)

    def test_shape_preservation(self):
        """Warping must accept [N, D] and [B, N, D] and preserve shapes."""
        tokens_2d = torch.randn(256, 384)
        warped_2d = warp_token_grid(tokens_2d, dx=5.0, dy=-3.0, patch_grid=(16, 16))
        assert warped_2d.shape == (256, 384)

        tokens_3d = torch.randn(2, 256, 384)
        warped_3d = warp_token_grid(tokens_3d, dx=5.0, dy=-3.0, patch_grid=(16, 16))
        assert warped_3d.shape == (2, 256, 384)

    def test_integer_patch_shift_alignment(self):
        """Shifting by 14 pixels (1 patch) horizontally aligns corresponding patches."""
        # Create a unique 2D position-encoded token grid [1, 16, 16, 384]
        h_p, w_p, d = 16, 16, 64
        grid_tokens = torch.zeros(1, h_p * w_p, d)
        for y in range(h_p):
            for x in range(w_p):
                grid_tokens[0, y * w_p + x, 0] = float(x)
                grid_tokens[0, y * w_p + x, 1] = float(y)

        # Shift by +14 pixels in x -> +1.0 patch in x
        warped = warp_token_grid(grid_tokens, dx=14.0, dy=0.0, patch_grid=(16, 16), patch_size=14)
        warped_reshaped = warped.view(1, h_p, w_p, d)

        # For x in [2, 14], current frame patch x sampled from reference patch (x - 1)
        for y in range(2, 14):
            for x in range(2, 14):
                val_x = warped_reshaped[0, y, x, 0].item()
                val_y = warped_reshaped[0, y, x, 1].item()
                assert abs(val_x - (x - 1.0)) < 0.1, f"At ({y},{x}): expected x-1={x-1}, got {val_x}"
                assert abs(val_y - y) < 0.1, f"At ({y},{x}): expected y={y}, got {val_y}"


class TestBouncerOperationalProfiles:
    """Tests for Bouncer operational profiles and dual-path routing."""

    def test_profile_static_bypasses_warp(self):
        """'static' profile should strictly use fast-path regardless of motion."""
        bouncer = Bouncer(profile="static")
        z_curr = torch.randn(1, 256, 384)
        z_prev = torch.randn(1, 256, 384)

        out = bouncer(z_curr, z_prev, motion=(10.0, 5.0), patch_grid=(16, 16))
        assert out.stats["profile"] == "static"
        assert out.stats["ego_motion_path"] == "fast"
        assert out.warped_reference is None

    def test_profile_motion_forces_warp(self):
        """'motion' profile should strictly use warp-path even for tiny motion."""
        bouncer = Bouncer(profile="motion")
        z_curr = torch.randn(1, 256, 384)
        z_prev = torch.randn(1, 256, 384)

        out = bouncer(z_curr, z_prev, motion=(0.1, 0.1), patch_grid=(16, 16))
        assert out.stats["profile"] == "motion"
        assert out.stats["ego_motion_path"] == "warp"
        assert out.warped_reference is not None

    def test_profile_auto_threshold_routing(self):
        """'auto' profile routes to fast-path when < 0.5px, and warp-path when >= 0.5px."""
        bouncer = Bouncer(profile="auto", motion_threshold=0.5)
        z_curr = torch.randn(1, 256, 384)
        z_prev = torch.randn(1, 256, 384)

        # Sub-threshold motion: sqrt(0.2^2 + 0.2^2) ~ 0.28 < 0.5
        out_sub = bouncer(z_curr, z_prev, motion=(0.2, 0.2), patch_grid=(16, 16))
        assert out_sub.stats["ego_motion_path"] == "fast"
        assert out_sub.warped_reference is None

        # Above-threshold motion: sqrt(3.0^2 + 4.0^2) = 5.0 >= 0.5
        out_super = bouncer(z_curr, z_prev, motion=(3.0, 4.0), patch_grid=(16, 16))
        assert out_super.stats["ego_motion_path"] == "warp"
        assert out_super.warped_reference is not None

    def test_warp_reduces_spurious_drops_on_panned_tokens(self):
        """When reference tokens are panned, warping them should restore high cosine similarity."""
        h_p, w_p, d = 16, 16, 64
        # Spatial pattern
        z_ref = torch.randn(1, h_p * w_p, d)
        # Simulate panned current frame: tokens after camera moves by +14 px
        dx_px = 14.0
        z_curr = warp_token_grid(z_ref, dx=dx_px, dy=0.0, patch_grid=(16, 16), patch_size=14)

        bouncer_fast = Bouncer(threshold=0.90, profile="static")
        bouncer_warp = Bouncer(threshold=0.90, profile="motion")

        out_fast = bouncer_fast(z_curr, z_ref, motion=(dx_px, 0.0), patch_grid=(16, 16))
        out_warp = bouncer_warp(z_curr, z_ref, motion=(dx_px, 0.0), patch_grid=(16, 16))

        # Warped reference should have significantly higher similarity than unwarped reference
        assert out_warp.stats["mean_similarity"] > out_fast.stats["mean_similarity"]


class TestPackerAndRebuilderEgoMotion:
    """Tests for TransmissionPacket motion metadata and Rebuilder cache shifting."""

    def test_packer_wire_bytes_with_motion(self):
        """TransmissionPacket includes 8 extra wire bytes when motion is provided."""
        packer = Packer(dim=64, num_quantizers=2, codebook_size=64, device="cpu")
        z_active = torch.randn(10, 64)
        mask = torch.zeros(256, dtype=torch.bool)
        mask[:10] = True

        out_no_motion = packer(z_active, mask, frame_id=1, motion=None)
        out_with_motion = packer(z_active, mask, frame_id=1, motion=(4.5, -2.1))

        assert out_no_motion.packet.motion is None
        assert out_with_motion.packet.motion == (4.5, -2.1)
        assert out_with_motion.packet.wire_bytes == out_no_motion.packet.wire_bytes + 8

    def test_rebuilder_cache_warping(self):
        """Rebuilder warps persistent cache before applying dynamic token scatter."""
        packer = Packer(dim=64, num_quantizers=2, codebook_size=64, device="cpu")
        rebuilder = Rebuilder(
            packer=packer, dim=64, num_quantizers=2, codebook_size=64, num_heads=4, device="cpu"
        )

        # Initial cache
        initial_cache = torch.randn(1, 256, 64)
        rebuilder.initialize_cache(initial_cache, patch_grid=(16, 16))

        # Frame 2: K = 0 active tokens (fully static scene), but camera moved by dx=14, dy=0
        mask = torch.zeros(256, dtype=torch.bool)
        z_active = torch.empty((0, 64))
        p_out = packer(z_active, mask, frame_id=2, motion=(14.0, 0.0), patch_grid=(16, 16))

        rebuilder_out = rebuilder(p_out.packet)

        # The cache should now be equal to warped initial_cache
        expected_warped = warp_token_grid(initial_cache, dx=14.0, dy=0.0, patch_grid=(16, 16))
        assert torch.allclose(rebuilder.token_cache, expected_warped, atol=1e-5)
