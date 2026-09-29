"""Unit and Regression Tests for Gate 2 Evaluation Suite.

Covers:
1. Frozen Trained Codebook: loading, SHA256 checksum verification, and protocol ID.
2. Official DAVIS Metrics: Jaccard (J), Boundary F-measure (F), and mean J&F mathematical invariants.
3. DINOv2 Label Propagator: shape contracts, nearest-neighbor matching, and memory bank dynamics.
4. H.264 Codec Baseline: bitrate calculation, FFmpeg encode/decode roundtrip, and frame fidelity.
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.packer import Packer, DEFAULT_CODEBOOK_PATH, DEFAULT_CODEBOOK_ID
from src.rebuilder import Rebuilder
from src.label_propagation import (
    compute_jaccard,
    compute_boundary_f_measure,
    compute_davis_metrics,
    DINOv2LabelPropagator,
)
from src.codec_baseline import (
    compute_bitrate_kbps,
    encode_frames_to_h264,
    decode_h264_to_frames,
)


class TestGate2CodebookIntegrity:
    """Verifies that the frozen trained RVQ codebook loads correctly and matches checksum."""

    def test_default_codebook_exists(self):
        codebook_file = Path(DEFAULT_CODEBOOK_PATH)
        assert codebook_file.exists(), f"Codebook file not found at {DEFAULT_CODEBOOK_PATH}"
        assert codebook_file.stat().st_size > 0

    def test_packer_pretrained_loading_and_checksum(self):
        packer = Packer.load_pretrained(DEFAULT_CODEBOOK_PATH, device="cpu")
        assert packer.codebook_id == DEFAULT_CODEBOOK_ID
        assert packer.codebook_weights is not None
        assert packer.codebook_weights.shape == (4, 256, 384)
        assert hasattr(packer, "mean_token_norm")
        assert packer.mean_token_norm > 40.0

    def test_rebuilder_pretrained_loading(self):
        rebuilder = Rebuilder.load_pretrained(DEFAULT_CODEBOOK_PATH, device="cpu")
        assert rebuilder.packer.codebook_id == DEFAULT_CODEBOOK_ID
        assert rebuilder.packer.codebook_weights.shape == (4, 256, 384)

    def test_packer_rvq_quantize_and_decode(self):
        packer = Packer.load_pretrained(DEFAULT_CODEBOOK_PATH, device="cpu")
        rebuilder = Rebuilder.load_pretrained(DEFAULT_CODEBOOK_PATH, device="cpu")

        # Test on tokens from the codebook manifold normalized to mean_token_norm
        norm_dirs = F.normalize(packer.codebook_weights[0, :10], dim=-1)
        tokens = norm_dirs * packer.mean_token_norm

        quantized, indices, commit_loss = packer.quantize(tokens)
        assert indices.shape == (10, 4)
        assert (indices >= 0).all() and (indices < 256).all()

        rec = rebuilder.decode_tokens(indices)
        assert rec.shape == (10, 384)
        # Quantized and decoded match cosine alignment and value magnitude
        cos_sim_q_rec = torch.cosine_similarity(quantized, rec, dim=-1).mean().item()
        assert cos_sim_q_rec > 0.9999
        cos_sim = torch.cosine_similarity(tokens, rec, dim=-1).mean().item()
        assert cos_sim > 0.95, f"Expected RVQ cos_sim > 0.95 on manifold, got {cos_sim:.4f}"


class TestDAVISMetricsInvariants:
    """Tests mathematical correctness of Jaccard (J) and Boundary F-measure (F)."""

    def test_jaccard_identical_masks(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:60, 20:60] = 1
        j = compute_jaccard(mask, mask)
        assert j == 1.0

    def test_jaccard_disjoint_masks(self):
        m1 = np.zeros((100, 100), dtype=np.uint8)
        m2 = np.zeros((100, 100), dtype=np.uint8)
        m1[10:30, 10:30] = 1
        m2[60:80, 60:80] = 1
        j = compute_jaccard(m1, m2)
        assert j == 0.0

    def test_jaccard_partial_overlap(self):
        m1 = np.zeros((10, 10), dtype=np.uint8)
        m2 = np.zeros((10, 10), dtype=np.uint8)
        m1[:6, :] = 1  # 60 pixels
        m2[2:8, :] = 1  # 60 pixels
        # Intersection: rows 2-5 (4 rows * 10 = 40 pixels)
        # Union: rows 0-7 (8 rows * 10 = 80 pixels)
        # IoU = 40 / 80 = 0.5
        j = compute_jaccard(m1, m2)
        assert abs(j - 0.5) < 1e-6

    def test_boundary_f_identical_masks(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:60, 20:60] = 1
        f = compute_boundary_f_measure(mask, mask, bound_th=2.0)
        assert f == 1.0

    def test_boundary_f_disjoint_masks(self):
        m1 = np.zeros((100, 100), dtype=np.uint8)
        m2 = np.zeros((100, 100), dtype=np.uint8)
        m1[10:30, 10:30] = 1
        m2[60:80, 60:80] = 1
        f = compute_boundary_f_measure(m1, m2, bound_th=2.0)
        assert f == 0.0

    def test_compute_davis_metrics_composite(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 1
        res = compute_davis_metrics(mask, mask)
        assert res["jaccard"] == 1.0
        assert res["f_measure"] == 1.0
        assert res["j_and_f"] == 1.0


class TestLabelPropagator:
    """Verifies DINOv2 label propagation logic and shape constraints."""

    def test_propagator_initialization_and_propagation(self):
        propagator = DINOv2LabelPropagator(top_k=3, temperature=0.10, device="cpu")

        # Synthetic Frame 0 tokens [1, 256, 384] and GT mask [224, 224]
        frame0_tokens = torch.randn(1, 256, 384)
        gt_mask = np.zeros((224, 224), dtype=np.uint8)
        # Patch aligned mask: patches 4..12 -> pixels 4*14 to 12*14 = 56 to 168
        gt_mask[56:168, 56:168] = 1

        propagator.initialize(frame0_tokens=frame0_tokens, frame0_mask=gt_mask)
        assert propagator.memory_tokens is not None
        assert propagator.memory_labels is not None
        assert propagator.memory_tokens.shape == (256, 384)
        assert propagator.memory_labels.shape == (256,)

        # Propagate on identical tokens -> prediction should match GT very closely
        pred_pixel, pred_grid = propagator.propagate(frame0_tokens, update_memory=False)
        assert pred_pixel.shape == (224, 224)
        assert pred_grid.shape == (16, 16)
        assert pred_pixel.dtype == np.uint8

        j = compute_jaccard(pred_pixel, gt_mask)
        assert j > 0.85, f"Expected high IoU on identical tokens, got {j:.4f}"

    def test_propagator_uninitialized_error(self):
        propagator = DINOv2LabelPropagator(device="cpu")
        with pytest.raises(RuntimeError):
            propagator.propagate(torch.randn(1, 256, 384))


class TestCodecBaselinePipeline:
    """Tests H.264 video compression baseline utilities."""

    def test_compute_bitrate_kbps_calculation(self):
        # 10,000 bytes over 50 frames at 25 fps:
        # duration = 50 / 25 = 2.0 sec
        # bits = 80,000 bits
        # kbps = 80,000 / (2.0 * 1000) = 40.0 kbps
        kbps = compute_bitrate_kbps(total_wire_bytes=10000, num_frames=50, fps=25.0)
        assert abs(kbps - 40.0) < 1e-5

    def test_ffmpeg_encode_decode_roundtrip(self):
        # Generate 10 synthetic RGB frames of shape (224, 224, 3)
        frames = [
            np.full((224, 224, 3), fill_value=i * 20, dtype=np.uint8)
            for i in range(10)
        ]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_mp4:
            tmp_path = tmp_mp4.name

        try:
            target_kbps = 100.0  # 100 kbps
            file_bytes = encode_frames_to_h264(
                frames_rgb=frames,
                target_kbps=target_kbps,
                output_mp4_path=tmp_path,
                fps=25.0,
            )
            assert file_bytes > 0
            assert Path(tmp_path).exists()

            decoded = decode_h264_to_frames(tmp_path)
            assert len(decoded) == 10
            assert decoded[0].shape == (224, 224, 3)
        finally:
            p = Path(tmp_path)
            if p.exists():
                p.unlink()


class TestDualCacheShadowPipeline:
    """Verifies that the Dual-Cache Shadow Architecture eliminates RVQ saturation."""

    def test_dual_cache_prevents_quantization_noise_saturation(self):
        from src.bouncer import Bouncer

        packer = Packer.load_pretrained(DEFAULT_CODEBOOK_PATH, device="cpu")
        bouncer = Bouncer(
            threshold=0.10,
            saliency_gated=True,
            gamma=1.0,
            tau_dynamic=0.005,
            tau_hard_change=0.15,
        )

        # Generate Frame 0 tokens on ViT manifold
        frame0_raw = F.normalize(torch.randn(1, 256, 384), dim=-1) * packer.mean_token_norm
        mask_all = torch.ones(256, dtype=torch.bool)
        p0 = packer(frame0_raw[0], mask_all, is_keyframe=True)
        frame0_rec = p0.quantized.unsqueeze(0)

        # Verify that quantization introduces significant cosine difference
        quant_diff = (1.0 - F.cosine_similarity(frame0_raw, frame0_rec, dim=-1)).mean().item()
        assert quant_diff > 0.10, f"Expected RVQ quantization distortion > 0.10, got {quant_diff}"

        # Initialize Dual-Cache Shadow Bouncer
        bouncer.initialize_edge_cache(
            raw_tokens=frame0_raw,
            reconstructed_tokens=frame0_rec,
            patch_grid=(16, 16),
        )

        # Frame 1 is STATIC (identical to Frame 0)
        frame1_raw = frame0_raw.clone()
        saliency = torch.rand(1, 256)

        out = bouncer(frame1_raw, saliency=saliency)

        # Under the old single-cache architecture, all 256 patches were triggered due to quant_diff > 0.10
        # Under the Dual-Cache Shadow architecture, 0 patches are triggered!
        assert out.mask.sum().item() == 0, f"Expected 0 active tokens on static frame, got {out.mask.sum().item()}"

