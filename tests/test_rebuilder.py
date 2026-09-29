"""Unit and integration tests for Phase D (The Rebuilder - Server Execution & KV-Cache Refresh).

Follows unit-testing-test-generate guidelines:
- Tests temporal cache initialization and reset
- Tests selective cache overwriting where M = 1
- Tests static token preservation where M = 0
- Tests downstream attention execution
- Tests full split-inference pipeline integration
"""

import pytest
import torch
from src.packer import Packer, TransmissionPacket
from src.rebuilder import Rebuilder, RebuilderOutput
from src.slicer import DINOv2Slicer
from src.bouncer import Bouncer
from tests.test_phase1 import MockViTBackbone


class TestRebuilder:
    """Test suite for Phase D: The Rebuilder."""

    @pytest.fixture
    def packer_and_rebuilder(self) -> tuple[Packer, Rebuilder]:
        packer = Packer(dim=384, num_quantizers=4, codebook_size=256, device="cpu")
        rebuilder = Rebuilder(packer=packer, dim=384, num_quantizers=4, codebook_size=256, device="cpu")
        return packer, rebuilder

    def test_cache_initialization_and_reset(self, packer_and_rebuilder: tuple[Packer, Rebuilder]) -> None:
        """Verify cache initialization state and resetting."""
        _, rebuilder = packer_and_rebuilder
        assert not rebuilder.is_cache_initialized()

        dummy_tokens = torch.randn(1, 256, 384)
        rebuilder.initialize_cache(dummy_tokens, patch_grid=(16, 16))

        assert rebuilder.is_cache_initialized()
        assert rebuilder.token_cache.shape == (1, 256, 384)
        assert torch.equal(rebuilder.token_cache, dummy_tokens)

        rebuilder.reset_cache()
        assert not rebuilder.is_cache_initialized()

    def test_selective_cache_overwriting(self, packer_and_rebuilder: tuple[Packer, Rebuilder]) -> None:
        """Verify that only tokens where M=1 are updated, while M=0 are strictly preserved."""
        packer, rebuilder = packer_and_rebuilder
        n, d = 256, 384

        # Initial cache filled with constant value 42.0
        initial_cache = torch.full((1, n, d), 42.0)
        rebuilder.initialize_cache(initial_cache, patch_grid=(16, 16))

        # Create active tokens for only 2 patches (indices 10 and 20)
        mask = torch.zeros(n, dtype=torch.bool)
        mask[10] = True
        mask[20] = True

        z_active = torch.randn(2, d)
        packer_out = packer(z_active, mask, frame_id=1, patch_grid=(16, 16))

        # Rebuilder execution
        out = rebuilder(packer_out.packet)

        assert isinstance(out, RebuilderOutput)
        assert out.num_refreshed == 2
        assert out.num_cached == 254

        refreshed = out.refreshed_tokens[0]

        # Verify static tokens (M=0) remained strictly unchanged
        static_indices = [i for i in range(n) if i not in (10, 20)]
        assert torch.allclose(refreshed[static_indices], torch.full((254, d), 42.0))

        # Verify active tokens (M=1) were refreshed with dequantized vectors
        z_hat_expected = packer.decode(packer_out.indices)
        assert torch.allclose(refreshed[10], z_hat_expected[0], atol=1e-5)
        assert torch.allclose(refreshed[20], z_hat_expected[1], atol=1e-5)

    def test_zero_active_tokens_preserves_entire_cache(
        self, packer_and_rebuilder: tuple[Packer, Rebuilder]
    ) -> None:
        """When K=0 (100% static frame), the entire cache should be preserved."""
        packer, rebuilder = packer_and_rebuilder
        n, d = 256, 384

        initial_cache = torch.randn(1, n, d)
        rebuilder.initialize_cache(initial_cache)

        empty_mask = torch.zeros(n, dtype=torch.bool)
        empty_z = torch.empty((0, d))
        packer_out = packer(empty_z, empty_mask, frame_id=2)

        out = rebuilder(packer_out.packet)

        assert out.num_refreshed == 0
        assert out.num_cached == n
        assert torch.equal(out.refreshed_tokens, initial_cache)

    def test_downstream_attention_execution(
        self, packer_and_rebuilder: tuple[Packer, Rebuilder]
    ) -> None:
        """Verify refreshed tokens pass cleanly through downstream attention layers."""
        packer, rebuilder = packer_and_rebuilder
        n, d = 64, 384
        tokens = torch.randn(1, n, d)
        rebuilder.initialize_cache(tokens)

        mask = torch.zeros(n, dtype=torch.bool)
        mask[5] = True
        z_active = torch.randn(1, d)
        packer_out = packer(z_active, mask, frame_id=3, patch_grid=(8, 8))

        out = rebuilder(packer_out.packet)

        assert out.downstream_output.shape == (1, n, d)
        assert torch.isfinite(out.downstream_output).all()


class TestEndToEndSplitInference:
    """Full system test chaining all 4 phases: Slicer -> Bouncer -> Packer -> Rebuilder."""

    def test_full_split_inference_pipeline(self) -> None:
        mock_model = MockViTBackbone(embed_dim=384, patch_size=14)
        slicer = DINOv2Slicer(device="cpu", backbone=mock_model)
        bouncer = Bouncer(threshold=0.95)
        packer = Packer(dim=384, num_quantizers=4, codebook_size=256, device="cpu")
        rebuilder = Rebuilder(packer=packer, dim=384, num_quantizers=4, codebook_size=256, device="cpu")

        # Frame 0 (Keyframe)
        frame0 = torch.rand(224, 224, 3)
        out0 = slicer(frame0)
        rebuilder.initialize_cache(out0.tokens, patch_grid=out0.patch_grid)
        assert rebuilder.is_cache_initialized()

        # Frame 1 (Differential frame with moving object)
        frame1 = frame0.clone()
        frame1[28:70, 28:70, :] = 1.0 - frame1[28:70, 28:70, :]

        # Edge Execution:
        out1 = slicer(frame1)
        b_out = bouncer(out1.tokens, out0.tokens, patch_grid=out1.patch_grid)
        p_out = packer(b_out.active_tokens, b_out.mask, frame_id=1, patch_grid=out1.patch_grid)

        # Wire Transmission:
        packet = p_out.packet
        assert packet.compression_ratio > 100.0

        # Server Execution:
        server_out = rebuilder(packet)

        assert server_out.refreshed_tokens.shape == (1, 256, 384)
        assert server_out.downstream_output.shape == (1, 256, 384)
        assert server_out.num_refreshed == b_out.stats["active_tokens"]
        assert server_out.num_cached == b_out.stats["dropped_tokens"]
