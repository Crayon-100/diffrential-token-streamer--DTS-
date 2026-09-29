"""Unit tests for Downstream Task Fidelity Evaluation Module.

Verifies:
- DownstreamTaskProbe initialization, layer dimensions, and output shapes
- Metric calculation invariants on identical and perturbed token representations
- Semantic prototype calibration
- Markdown report formatting
"""

from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

from src.downstream_eval import (
    DownstreamTaskProbe,
    SequenceFidelityMetrics,
    MultiSequenceFidelityResult,
    format_fidelity_markdown_table,
    build_semantic_prototypes,
)
from src.slicer import DINOv2Slicer


class TestDownstreamTaskProbe:
    """Test suite for DownstreamTaskProbe."""

    @pytest.fixture
    def probe(self) -> DownstreamTaskProbe:
        return DownstreamTaskProbe(
            dim=384,
            num_heads=6,
            num_layers=2,
            num_classes=10,
            device="cpu",
        )

    def test_probe_output_shapes(self, probe: DownstreamTaskProbe) -> None:
        """Verify probe forward pass shapes and normalization invariants."""
        tokens = torch.randn(1, 256, 384)
        logits, pooled_feat, downstream_tokens = probe(tokens)

        assert logits.shape == (1, 10), "Logits must match [B, num_classes]"
        assert pooled_feat.shape == (1, 384), "Pooled feature must match [B, dim]"
        assert downstream_tokens.shape == (1, 256, 384), "Downstream tokens must match [B, N, dim]"

        norm = torch.linalg.norm(pooled_feat, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5), "Pooled features must be unit normalized"

    def test_evaluate_frame_fidelity_identical_tokens(self, probe: DownstreamTaskProbe) -> None:
        """Identical raw and cache tokens must yield 100% fidelity and 0 MSE."""
        z = torch.randn(1, 256, 384)
        metrics = probe.evaluate_frame_fidelity(z, z)

        assert pytest.approx(metrics["token_cos"], abs=1e-5) == 1.0
        assert pytest.approx(metrics["mse"], abs=1e-5) == 0.0
        assert pytest.approx(metrics["feat_cos"], abs=1e-5) == 1.0
        assert metrics["top1_match"] == 1.0
        assert metrics["top5_match"] == 1.0

    def test_evaluate_frame_fidelity_perturbed_tokens(self, probe: DownstreamTaskProbe) -> None:
        """Slightly perturbed tokens should maintain high cosine fidelity and bounded MSE."""
        z_raw = torch.randn(1, 256, 384)
        noise = torch.randn_like(z_raw) * 0.10
        z_cache = z_raw + noise

        metrics = probe.evaluate_frame_fidelity(z_raw, z_cache)

        assert 0.80 <= metrics["token_cos"] <= 1.0, "Slight noise should preserve >80% cos-sim"
        assert metrics["mse"] > 0.0, "Perturbed tokens must have non-zero MSE"
        assert 0.85 <= metrics["feat_cos"] <= 1.0, "Downstream features should remain highly aligned"
        assert metrics["top1_match"] in (0.0, 1.0)
        assert metrics["top5_match"] in (0.0, 1.0)

    def test_format_fidelity_markdown_table(self) -> None:
        """Verify markdown report generation formatting."""
        seq_metric = SequenceFidelityMetrics(
            sequence="blackswan",
            num_frames=50,
            compression_ratio=1086.7,
            bandwidth_savings_pct=99.91,
            mean_token_cosine_fidelity=0.8625,
            mean_latent_mse=1.6884,
            mean_downstream_feature_fidelity=0.9757,
            top1_task_alignment_pct=100.0,
            top5_task_alignment_pct=100.0,
        )

        multi_res = MultiSequenceFidelityResult(
            sequences=[seq_metric],
            total_frames=50,
            overall_compression_ratio=1086.7,
            overall_bandwidth_savings_pct=99.91,
            macro_token_cosine_fidelity=0.8625,
            macro_latent_mse=1.6884,
            macro_downstream_feature_fidelity=0.9757,
            macro_top1_task_alignment_pct=100.0,
            macro_top5_task_alignment_pct=100.0,
        )

        md = format_fidelity_markdown_table(multi_res)
        assert "| Sequence | Frames | Bandwidth Reduction" in md
        assert "| **blackswan** |" in md
        assert "1,086.7x" in md
        assert "100.00%" in md
        assert "| **OVERALL (AVG / TOTAL)** |" in md

    def test_downstream_null_baseline_frozen_cache(self, capsys: pytest.CaptureFixture) -> None:
        """Null baseline 1: Frozen frame-0 cache under evolving scene dynamics.

        Demonstrates probe metric sensitivity: if server cache remains frozen at frame 0
        while the video evolves to a different semantic state, fidelity degrades and
        Top-1 task match drops to 0.0%.
        """
        torch.manual_seed(42)
        dim, num_patches, num_classes = 384, 256, 10
        probe = DownstreamTaskProbe(dim=dim, num_classes=num_classes, device="cpu")

        # Frame 0 state and target dynamic state
        z_0 = torch.randn(1, num_patches, dim)
        z_target = torch.randn(1, num_patches, dim)
        z_frozen = z_0.clone()

        # Calibrate probe prototypes to downstream representations of z_0 and z_target
        _, h_0, _ = probe(z_0)
        _, h_t, _ = probe(z_target)
        distractors = F.normalize(torch.randn(num_classes - 2, dim), dim=-1)
        probe.prototypes.copy_(torch.cat([h_0, h_t, distractors], dim=0))

        num_frames = 12
        token_cos_history, feat_cos_history, top1_history = [], [], []

        print("\n--- Frozen Frame-0 Cache Null Baseline Trajectory ---")
        for t in range(num_frames):
            alpha = t / (num_frames - 1)
            z_curr = (1.0 - alpha) * z_0 + alpha * z_target
            metrics = probe.evaluate_frame_fidelity(z_curr, z_frozen)
            token_cos_history.append(metrics["token_cos"])
            feat_cos_history.append(metrics["feat_cos"])
            top1_history.append(metrics["top1_match"])
            print(
                f"Frame {t:02d} (alpha={alpha:.2f}): Token Cos={metrics['token_cos']:.3f}, "
                f"Feat Cos={metrics['feat_cos']:.3f}, Top-1 Match={metrics['top1_match']}"
            )

        mean_top1 = sum(top1_history) / len(top1_history)
        print(f"Overall Frozen Cache Top-1 Accuracy: {mean_top1 * 100:.1f}%")

        # Frame 0 must match
        assert top1_history[0] == 1.0, "Frame 0 compared to itself must match"
        # Late frames must diverge as the video changes class
        assert top1_history[-1] == 0.0, "Probe must detect semantic class flip under frozen cache"
        assert token_cos_history[-1] < token_cos_history[0], "Token cosine must degrade over time"
        assert mean_top1 < 1.0, "Frozen cache cannot maintain 100% Top-1 alignment across evolving scene"

    def test_downstream_null_baseline_wrong_video(self) -> None:
        """Null baseline 2: Tokens from completely mismatched video sequences.

        Demonstrates probe metric sensitivity: feeding tokens from Video B into a cache
        expected for Video A results in 0.0% Top-1 alignment and low cosine similarity.
        """
        torch.manual_seed(123)
        dim, num_patches, num_classes = 384, 256, 10
        probe = DownstreamTaskProbe(dim=dim, num_classes=num_classes, device="cpu")

        # Two distinct video sequences with different latent representations
        z_video_a = torch.randn(1, num_patches, dim)
        z_video_b = torch.randn(1, num_patches, dim)

        # Calibrate probe prototypes so class 0 represents Video A and class 1 represents Video B
        _, h_a, _ = probe(z_video_a)
        _, h_b, _ = probe(z_video_b)
        distractors = F.normalize(torch.randn(num_classes - 2, dim), dim=-1)
        probe.prototypes.copy_(torch.cat([h_a, h_b, distractors], dim=0))

        metrics = probe.evaluate_frame_fidelity(z_video_a, z_video_b)
        print("\n--- Wrong Video Null Baseline Telemetry ---")
        print(f"Token Cosine Fidelity : {metrics['token_cos']:.4f}")
        print(f"Latent MSE            : {metrics['mse']:.4f}")
        print(f"Downstream Feat CosSim: {metrics['feat_cos']:.4f}")
        print(f"Pred Video A Class    : {metrics['pred_raw']}")
        print(f"Pred Video B Class    : {metrics['pred_rec']}")
        print(f"Top-1 Task Match      : {metrics['top1_match']:.1f}")

        assert metrics["pred_raw"] == 0, "Video A must predict calibrated class 0"
        assert metrics["pred_rec"] == 1, "Video B must predict calibrated class 1"
        assert metrics["top1_match"] == 0.0, "Wrong video must yield 0.0% Top-1 task alignment"
        assert metrics["token_cos"] < 0.30, "Mismatched video tokens must have low cosine similarity"
