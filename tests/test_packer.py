"""Unit and integration tests for Phase C (The Packer - Residual Vector Quantization).

Follows unit-testing-test-generate guidelines:
- Tests quantization shapes, types, and codebook range
- Tests zero-token edge cases (K=0)
- Tests reconstruction fidelity and decoder
- Tests bit-packing and transmission packet serialization
- End-to-end pipeline test
"""

import pytest
import torch
import torch.nn.functional as F
from src.packer import Packer, PackerOutput, TransmissionPacket
from src.slicer import DINOv2Slicer
from src.bouncer import Bouncer
from tests.test_phase1 import MockViTBackbone


class TestPacker:
    """Test suite for Phase C: The Packer."""

    @pytest.fixture
    def packer(self) -> Packer:
        return Packer(dim=384, num_quantizers=4, codebook_size=256, device="cpu")

    def test_quantization_shapes_and_types(self, packer: Packer) -> None:
        """Verify RVQ produces integer indices [K, Q] and quantized vectors [K, D]."""
        k, d, q = 20, 384, 4
        z_active = torch.randn(k, d)

        quantized, indices, commit_loss = packer.quantize(z_active)

        assert indices.shape == (k, q)
        assert indices.dtype in (torch.int32, torch.int64)
        assert quantized.shape == (k, d)
        assert commit_loss is not None

    def test_codebook_indices_range(self, packer: Packer) -> None:
        """Verify all discrete indices fall strictly within [0, codebook_size - 1]."""
        z_active = torch.randn(50, 384)
        _, indices, _ = packer.quantize(z_active)

        assert indices.min().item() >= 0
        assert indices.max().item() < packer.codebook_size

    def test_zero_active_tokens_edge_case(self, packer: Packer) -> None:
        """Verify K=0 (100% static frame) is handled cleanly without exceptions."""
        empty_z = torch.empty((0, 384))
        empty_mask = torch.zeros(256, dtype=torch.bool)

        packer_out = packer(empty_z, empty_mask, frame_id=0, patch_grid=(16, 16))

        assert packer_out.indices.shape == (0, packer.num_quantizers)
        assert packer_out.quantized.shape == (0, packer.dim)
        assert packer_out.packet.num_active == 0

        # Decoder should also handle K=0 cleanly
        recon = packer.decode(packer_out.indices)
        assert recon.shape == (0, packer.dim)

    def test_decoder_reconstruction_shape_and_fidelity(self, packer: Packer) -> None:
        """Verify decoder reconstructs continuous vectors [K, D] from indices."""
        k, d = 15, 384
        z_active = torch.randn(k, d)

        _, indices, _ = packer.quantize(z_active)
        z_recon = packer.decode(indices)

        assert z_recon.shape == (k, d)
        # Cosine similarity between active and reconstructed should be positive
        cos_sim = F.cosine_similarity(z_active, z_recon, dim=-1)
        assert (cos_sim > 0.0).all(), "Reconstructed vectors should correlate with inputs"

    def test_mask_bitpack_roundtrip(self) -> None:
        """Verify that bit-packing a boolean mask produces exact roundtrip recovery."""
        n = 256
        # Random boolean mask with known active tokens
        mask = torch.rand(n) > 0.8
        packed = Packer.pack_boolean_mask(mask)

        # 256 bits = 32 bytes
        assert len(packed) == 32

        # Unpack via packet helper
        dummy_indices = torch.zeros((mask.sum().item(), 4), dtype=torch.int64)
        packet = TransmissionPacket(
            frame_id=1,
            num_patches=n,
            num_active=int(mask.sum().item()),
            num_quantizers=4,
            codebook_size=256,
            patch_grid=(16, 16),
            packed_mask=packed,
            indices=dummy_indices,
            raw_bytes=n * 384 * 4,
            wire_bytes=100,
            compression_ratio=3932.16,
        )

        unpacked = packet.unpack_mask()
        assert torch.equal(mask, unpacked), "Unpacked mask must exactly match original"

    def test_packet_payload_and_compression_metrics(self, packer: Packer) -> None:
        """Verify wire bytes calculation and compression ratio."""
        n, d, k = 256, 384, 16
        z_active = torch.randn(k, d)
        mask = torch.zeros(n, dtype=torch.bool)
        mask[:k] = True

        out = packer(z_active, mask, frame_id=5, patch_grid=(16, 16))
        packet = out.packet

        # Raw size: 256 * 384 * 4 = 393,216 bytes
        expected_raw = 256 * 384 * 4
        assert packet.raw_bytes == expected_raw

        # Wire size: Header(18) + Mask(32) + Indices(16 * 4 * 1 = 64) = 114 bytes
        expected_wire = 18 + 32 + (k * 4 * 1)
        assert packet.wire_bytes == expected_wire
        assert len(packet.to_bytes()) == packet.wire_bytes
        assert packet.compression_ratio > 3000.0

    def test_byte_serialization_roundtrip(self, packer: Packer) -> None:
        """Verify binary serialization to_bytes and from_bytes preserves all state."""
        n, d, k = 256, 384, 25
        z_active = torch.randn(k, d)
        mask = torch.zeros(n, dtype=torch.bool)
        mask[:k] = True

        out = packer(z_active, mask, frame_id=42, patch_grid=(16, 16))
        packet = out.packet

        raw_payload = packet.to_bytes()
        assert isinstance(raw_payload, bytes)
        assert len(raw_payload) == packet.wire_bytes

        # Roundtrip deserialization
        restored = TransmissionPacket.from_bytes(raw_payload)
        assert restored.frame_id == packet.frame_id
        assert restored.num_patches == packet.num_patches
        assert restored.num_active == packet.num_active
        assert restored.num_quantizers == packet.num_quantizers
        assert restored.codebook_size == packet.codebook_size
        assert restored.patch_grid == packet.patch_grid
        assert restored.wire_bytes == packet.wire_bytes
        assert restored.raw_bytes == packet.raw_bytes
        assert torch.equal(packet.unpack_mask(), restored.unpack_mask())
        assert torch.equal(packet.unpack_indices(), restored.unpack_indices())


class TestPhaseCEndToEnd:
    """Integration test connecting Slicer, Bouncer, and Packer."""

    def test_full_pipeline_slicer_bouncer_packer(self) -> None:
        mock_model = MockViTBackbone(embed_dim=384, patch_size=14)
        slicer = DINOv2Slicer(device="cpu", backbone=mock_model)
        bouncer = Bouncer(threshold=0.95)
        packer = Packer(dim=384, num_quantizers=4, codebook_size=256, device="cpu")

        # Two consecutive frames
        f1 = torch.rand(224, 224, 3)
        f2 = f1.clone()
        f2[50:100, 50:100, :] = 1.0 - f2[50:100, 50:100, :]

        # Slicer
        out1 = slicer(f1)
        out2 = slicer(f2)

        # Bouncer
        b_out = bouncer(out2.tokens, out1.tokens, patch_grid=out1.patch_grid)

        # Packer
        p_out = packer(
            z_active=b_out.active_tokens,
            mask=b_out.mask,
            frame_id=1,
            patch_grid=out1.patch_grid,
        )

        assert p_out.indices.shape[0] == b_out.stats["active_tokens"]
        assert p_out.indices.shape[1] == 4
        assert p_out.packet.compression_ratio > 100.0

        # Server-side reconstruction test
        z_hat = packer.decode(p_out.indices)
        assert z_hat.shape == b_out.active_tokens.shape
