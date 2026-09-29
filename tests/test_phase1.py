"""Unit and integration tests for Phase 1 (Slicer & Bouncer).

Follows unit-testing-test-generate guidelines:
- Tests happy path and boundary conditions
- Tests tensor dimensions and shape invariants
- Tests backbone freeze condition
- Tests offline mock backbone and numerical accuracy
"""

import pytest
import torch
import torch.nn as nn
from src.slicer import DINOv2Slicer, SlicerOutput
from src.bouncer import Bouncer, BouncerOutput


class MockViTBackbone(nn.Module):
    """Mock ViT backbone for fast, offline unit testing without downloading weights."""
    def __init__(self, embed_dim: int = 384, patch_size: int = 14) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        # Dummy linear layer to verify freezing
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward_features(self, x: torch.Tensor) -> dict:
        # x: [B, 3, H, W]
        b, c, h, w = x.shape
        feat = self.proj(x)  # [B, D, H/14, W/14]
        feat = feat.flatten(2).transpose(1, 2)  # [B, N, D]
        return {
            "x_norm_clstoken": torch.zeros(b, self.embed_dim, device=x.device),
            "x_norm_patchtokens": feat,
        }


class TestDINOv2Slicer:
    """Test suite for Phase A: The Slicer."""

    @pytest.fixture
    def mock_slicer(self) -> DINOv2Slicer:
        mock_model = MockViTBackbone(embed_dim=384, patch_size=14)
        return DINOv2Slicer(device="cpu", backbone=mock_model)

    def test_backbone_is_frozen(self, mock_slicer: DINOv2Slicer) -> None:
        """Verify that all parameters in the backbone are frozen (requires_grad=False)."""
        assert not mock_slicer.backbone.training, "Backbone must be in eval() mode"
        for name, param in mock_slicer.backbone.named_parameters():
            assert not param.requires_grad, f"Parameter {name} must have requires_grad=False"

    def test_output_tensor_shapes_standard_frame(self, mock_slicer: DINOv2Slicer) -> None:
        """Verify output tensor shapes for standard 224x224 input."""
        # 224 / 14 = 16 patches per dim -> 256 tokens total
        frame = torch.rand(224, 224, 3)
        out = mock_slicer(frame)

        assert isinstance(out, SlicerOutput)
        assert out.tokens.shape == (1, 256, 384)
        assert out.patch_grid == (16, 16)
        assert out.embedding_dim == 384

    def test_channel_order_handling(self, mock_slicer: DINOv2Slicer) -> None:
        """Test that both [H, W, C] and [C, H, W] inputs are correctly handled."""
        hwc = torch.rand(140, 140, 3)
        chw = torch.rand(3, 140, 140)

        out_hwc = mock_slicer(hwc)
        out_chw = mock_slicer(chw)

        # 140 / 14 = 10 patches -> 100 tokens
        assert out_hwc.tokens.shape == (1, 100, 384)
        assert out_chw.tokens.shape == (1, 100, 384)
        assert out_hwc.patch_grid == (10, 10)

    def test_arbitrary_resolution_resizing(self, mock_slicer: DINOv2Slicer) -> None:
        """Test that images not divisible by 14 are resized to valid patch multiples."""
        odd_frame = torch.rand(230, 215, 3)
        out = mock_slicer(odd_frame)

        # 230 -> 224 (16 patches), 215 -> 210 (15 patches)
        # N = 16 * 15 = 240
        assert out.patch_grid == (16, 15)
        assert out.tokens.shape == (1, 240, 384)

    def test_invalid_input_shapes_raise_error(self, mock_slicer: DINOv2Slicer) -> None:
        """Test that invalid dimensions or channel counts raise ValueError."""
        with pytest.raises(ValueError):
            mock_slicer(torch.rand(224, 224))  # 2D

        with pytest.raises(ValueError):
            mock_slicer(torch.rand(4, 224, 224))  # 4 channels


class TestBouncer:
    """Test suite for Phase B: The Bouncer."""

    @pytest.fixture
    def bouncer(self) -> Bouncer:
        return Bouncer(threshold=0.95)

    def test_identical_tokens_100_percent_dropped(self, bouncer: Bouncer) -> None:
        """If current tokens are identical to previous tokens, all must be dropped."""
        b, n, d = 1, 256, 384
        z = torch.randn(b, n, d)

        out = bouncer(z, z, patch_grid=(16, 16))

        assert isinstance(out, BouncerOutput)
        assert out.mask.sum().item() == 0, "No tokens should be active"
        assert out.drop_mask.all(), "All tokens should be flagged as dropped"
        assert out.stats["drop_ratio"] == 1.0
        assert out.stats["active_tokens"] == 0
        assert out.active_tokens.shape == (0, d)
        assert torch.allclose(out.cosine_similarities, torch.ones(b, n), atol=1e-4)

    def test_orthogonal_tokens_0_percent_dropped(self, bouncer: Bouncer) -> None:
        """If tokens are orthogonal (cosine similarity = 0.0 <= 0.95), all must be kept."""
        b, n, d = 1, 64, 128
        # Create orthogonal tokens: e.g. [1, 0] vs [0, 1]
        z_curr = torch.zeros(b, n, d)
        z_prev = torch.zeros(b, n, d)
        z_curr[:, :, : d // 2] = 1.0
        z_prev[:, :, d // 2 :] = 1.0

        out = bouncer(z_curr, z_prev, patch_grid=(8, 8))

        assert out.mask.all(), "All tokens should be active"
        assert out.drop_mask.sum().item() == 0, "No tokens should be dropped"
        assert out.stats["drop_ratio"] == 0.0
        assert out.stats["active_tokens"] == n
        assert out.active_tokens.shape == (n, d)
        assert torch.allclose(out.cosine_similarities, torch.zeros(b, n), atol=1e-4)

    def test_threshold_boundary_behavior(self, bouncer: Bouncer) -> None:
        """Verify strict threshold behavior: S > 0.95 is dropped, S <= 0.95 is active."""
        d = 64
        # Token A
        t1 = torch.zeros(1, 2, d)
        t1[0, 0, 0] = 1.0
        t1[0, 1, 0] = 1.0

        # Token B: exactly engineered cosine similarities
        t2 = torch.zeros(1, 2, d)
        # Patch 0: S = 0.96 (> 0.95 -> dropped)
        t2[0, 0, 0] = 0.96
        t2[0, 0, 1] = (1 - 0.96**2) ** 0.5
        # Patch 1: S = 0.94 (<= 0.95 -> active)
        t2[0, 1, 0] = 0.94
        t2[0, 1, 1] = (1 - 0.94**2) ** 0.5

        out = bouncer(t1, t2)

        # Patch 0 should be dropped (mask=False, drop_mask=True)
        assert not out.mask[0, 0].item()
        assert out.drop_mask[0, 0].item()

        # Patch 1 should be active (mask=True, drop_mask=False)
        assert out.mask[0, 1].item()
        assert not out.drop_mask[0, 1].item()

        assert out.stats["dropped_tokens"] == 1
        assert out.stats["active_tokens"] == 1

    def test_spatial_mask_grid(self, bouncer: Bouncer) -> None:
        """Verify spatial 2D mask has shape [B, H_p, W_p]."""
        z1 = torch.randn(1, 100, 384)
        z2 = torch.randn(1, 100, 384)

        out = bouncer(z1, z2, patch_grid=(10, 10))
        assert out.spatial_mask is not None
        assert out.spatial_mask.shape == (1, 10, 10)

    def test_mismatched_shapes_raise_error(self, bouncer: Bouncer) -> None:
        """Verify mismatched shapes raise a ValueError."""
        z1 = torch.randn(1, 100, 384)
        z2 = torch.randn(1, 50, 384)
        with pytest.raises(ValueError):
            bouncer(z1, z2)


class TestPhase1Integration:
    """Integration test connecting Slicer and Bouncer."""

    def test_end_to_end_slicer_bouncer(self) -> None:
        mock_model = MockViTBackbone(embed_dim=384, patch_size=14)
        slicer = DINOv2Slicer(device="cpu", backbone=mock_model)
        bouncer = Bouncer(threshold=0.95)

        # Frame 1 and Frame 2
        f1 = torch.rand(224, 224, 3)
        f2 = f1.clone()
        # Alter a small 28x28 region (4 patches out of 256)
        f2[0:28, 0:28, :] = 1.0 - f2[0:28, 0:28, :]

        out1 = slicer(f1)
        out2 = slicer(f2)

        bouncer_out = bouncer(out2.tokens, out1.tokens, patch_grid=out1.patch_grid)

        # Total tokens = 256
        assert bouncer_out.stats["total_tokens"] == 256
        # Most of the frame is identical, so dropped tokens should be high (> 200)
        assert bouncer_out.stats["dropped_tokens"] >= 200
        # Altered patches should be flagged active
        assert bouncer_out.stats["active_tokens"] > 0
        assert bouncer_out.active_tokens.shape[-1] == 384
