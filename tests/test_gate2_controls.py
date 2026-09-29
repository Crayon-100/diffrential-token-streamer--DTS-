"""Unit and regression tests for Gate 2 Critical Controls & Honest Baselines."""

import math
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

# Ensure repository root is on sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.dense_control import (
    compute_floor_b_copy_mask,
    evaluate_dense_rvq_control,
    evaluate_dense_rvq_frame_skip,
    evaluate_frozen_cache_baseline,
    FloorBResult,
)
from src.packer import Packer, TransmissionPacket, DEFAULT_CODEBOOK_PATH
from src.rebuilder import Rebuilder
from src.run_gate2_controls import (
    SequenceControlsEvaluation,
    generate_gate2_controls_report,
)


class TestGate2Controls:
    """Test suite for Gate 2 Controls and Floor Baselines."""

    @pytest.fixture(autouse=True)
    def setup_fixtures(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.packer = Packer.load_pretrained(DEFAULT_CODEBOOK_PATH, device=self.device)
        self.rebuilder = Rebuilder(packer=self.packer, device=self.device)

    def test_dense_q1_packet_byte_size(self):
        """Dense Q=1 must produce exactly 306 bytes per 256-patch frame."""
        z_tokens = torch.randn(256, 384, device=self.device)
        mask = torch.ones(256, dtype=torch.bool, device=self.device)

        p_out = self.packer(
            z_active=z_tokens,
            mask=mask,
            frame_id=1,
            patch_grid=(16, 16),
            num_quantizers=1,
        )

        assert p_out.packet.num_quantizers == 1
        assert p_out.packet.num_active == 256
        # Header (18) + Mask (32) + Indices (256 * 1) = 306 bytes
        assert p_out.packet.wire_bytes == 306
        assert len(p_out.packet.payload) == 306

        # Rebuilder decodes cleanly
        reb_out = self.rebuilder(p_out.packet)
        assert reb_out.refreshed_tokens.shape == (1, 256, 384)

    def test_dense_q2_packet_byte_size(self):
        """Dense Q=2 must produce exactly 562 bytes per 256-patch frame."""
        z_tokens = torch.randn(256, 384, device=self.device)
        mask = torch.ones(256, dtype=torch.bool, device=self.device)

        p_out = self.packer(
            z_active=z_tokens,
            mask=mask,
            frame_id=1,
            patch_grid=(16, 16),
            num_quantizers=2,
        )

        assert p_out.packet.num_quantizers == 2
        assert p_out.packet.num_active == 256
        # Header (18) + Mask (32) + Indices (256 * 2) = 562 bytes
        assert p_out.packet.wire_bytes == 562
        assert len(p_out.packet.payload) == 562

        # Rebuilder decodes cleanly
        reb_out = self.rebuilder(p_out.packet)
        assert reb_out.refreshed_tokens.shape == (1, 256, 384)

    def test_floor_b_copy_mask(self):
        """Floor B must compute valid non-empty J&F metrics on a real sequence."""
        res = compute_floor_b_copy_mask(
            sequence="blackswan",
            root_dir="data/DAVIS",
        )

        assert isinstance(res, FloorBResult)
        assert res.num_frames == 49  # 50 frames total, 49 evaluated against frame 0
        assert 0.0 <= res.mean_jaccard <= 1.0
        assert 0.0 <= res.mean_f_measure <= 1.0
        assert 0.0 <= res.mean_j_and_f <= 1.0
        assert res.wire_kbps == 0.0

    def test_floor_a_frozen_cache_evaluation(self):
        """Floor A evaluates static repeated frame 0 tokens across the sequence."""
        dummy_f0_tokens = torch.randn(1, 256, 384, device=self.device)
        prop_res, total_bytes, kbps = evaluate_frozen_cache_baseline(
            sequence="blackswan",
            frame0_reconstructed_tokens=dummy_f0_tokens,
            num_frames=50,
            frame0_wire_bytes=1074,
            root_dir="data/DAVIS",
            fps=25.0,
            device=self.device,
        )

        assert prop_res.num_frames == 49
        assert total_bytes == 1074
        # 1074 * 8 / (2.0s * 1000) = 4.296 kbps
        assert abs(kbps - 4.296) < 0.01
        assert 0.0 <= prop_res.mean_j_and_f <= 1.0

    def test_report_generation_uses_median_delta(self, tmp_path):
        """Report generator must correctly compute median delta vs H.264."""
        # Create dummy evaluations where deltas are skewed
        # deltas: [+10.0%, +20.0%, +30.0%, +100.0%] -> median should be (+20 + +30) / 2 = +25.0%
        # (while mean would be +40.0%)
        evals = [
            SequenceControlsEvaluation(
                sequence="seq1", num_frames=10,
                raw_jf=0.50, raw_j=0.50, raw_f=0.50, raw_kbps=100.0,
                gated_q4_jf=0.44, gated_q4_j=0.44, gated_q4_f=0.44, gated_q4_kbps=70.0,
                gated_q4_wire_bytes=700, mean_k=50.0, mean_k_pct=19.5, retention_vs_oracle_pct=88.0,
                dense_q1_jf=0.40, dense_q1_j=0.40, dense_q1_f=0.40, dense_q1_kbps=61.2, dense_q1_wire_bytes=612,
                dense_q2_jf=0.42, dense_q2_j=0.42, dense_q2_f=0.42, dense_q2_kbps=112.4, dense_q2_wire_bytes=1124,
                h264_jf=0.40, h264_j=0.40, h264_f=0.40, h264_achieved_kbps=70.0, h264_file_bytes=700,
                delta_vs_h264_pct=10.0, delta_vs_h264_abs=0.04,
                floor_a_jf=0.30, floor_a_j=0.30, floor_a_f=0.30, floor_a_kbps=5.0, floor_a_wire_bytes=100,
                floor_b_jf=0.25, floor_b_j=0.25, floor_b_f=0.25, floor_b_kbps=0.0,
            ),
            SequenceControlsEvaluation(
                sequence="seq2", num_frames=10,
                raw_jf=0.50, raw_j=0.50, raw_f=0.50, raw_kbps=100.0,
                gated_q4_jf=0.48, gated_q4_j=0.48, gated_q4_f=0.48, gated_q4_kbps=70.0,
                gated_q4_wire_bytes=700, mean_k=50.0, mean_k_pct=19.5, retention_vs_oracle_pct=96.0,
                dense_q1_jf=0.40, dense_q1_j=0.40, dense_q1_f=0.40, dense_q1_kbps=61.2, dense_q1_wire_bytes=612,
                dense_q2_jf=0.42, dense_q2_j=0.42, dense_q2_f=0.42, dense_q2_kbps=112.4, dense_q2_wire_bytes=1124,
                h264_jf=0.40, h264_j=0.40, h264_f=0.40, h264_achieved_kbps=70.0, h264_file_bytes=700,
                delta_vs_h264_pct=20.0, delta_vs_h264_abs=0.08,
                floor_a_jf=0.30, floor_a_j=0.30, floor_a_f=0.30, floor_a_kbps=5.0, floor_a_wire_bytes=100,
                floor_b_jf=0.25, floor_b_j=0.25, floor_b_f=0.25, floor_b_kbps=0.0,
            ),
            SequenceControlsEvaluation(
                sequence="seq3", num_frames=10,
                raw_jf=0.50, raw_j=0.50, raw_f=0.50, raw_kbps=100.0,
                gated_q4_jf=0.52, gated_q4_j=0.52, gated_q4_f=0.52, gated_q4_kbps=70.0,
                gated_q4_wire_bytes=700, mean_k=50.0, mean_k_pct=19.5, retention_vs_oracle_pct=104.0,
                dense_q1_jf=0.40, dense_q1_j=0.40, dense_q1_f=0.40, dense_q1_kbps=61.2, dense_q1_wire_bytes=612,
                dense_q2_jf=0.42, dense_q2_j=0.42, dense_q2_f=0.42, dense_q2_kbps=112.4, dense_q2_wire_bytes=1124,
                h264_jf=0.40, h264_j=0.40, h264_f=0.40, h264_achieved_kbps=70.0, h264_file_bytes=700,
                delta_vs_h264_pct=30.0, delta_vs_h264_abs=0.12,
                floor_a_jf=0.30, floor_a_j=0.30, floor_a_f=0.30, floor_a_kbps=5.0, floor_a_wire_bytes=100,
                floor_b_jf=0.25, floor_b_j=0.25, floor_b_f=0.25, floor_b_kbps=0.0,
            ),
            SequenceControlsEvaluation(
                sequence="seq4", num_frames=10,
                raw_jf=0.50, raw_j=0.50, raw_f=0.50, raw_kbps=100.0,
                gated_q4_jf=0.80, gated_q4_j=0.80, gated_q4_f=0.80, gated_q4_kbps=70.0,
                gated_q4_wire_bytes=700, mean_k=50.0, mean_k_pct=19.5, retention_vs_oracle_pct=160.0,
                dense_q1_jf=0.40, dense_q1_j=0.40, dense_q1_f=0.40, dense_q1_kbps=61.2, dense_q1_wire_bytes=612,
                dense_q2_jf=0.42, dense_q2_j=0.42, dense_q2_f=0.42, dense_q2_kbps=112.4, dense_q2_wire_bytes=1124,
                h264_jf=0.40, h264_j=0.40, h264_f=0.40, h264_achieved_kbps=70.0, h264_file_bytes=700,
                delta_vs_h264_pct=100.0, delta_vs_h264_abs=0.40,
                floor_a_jf=0.30, floor_a_j=0.30, floor_a_f=0.30, floor_a_kbps=5.0, floor_a_wire_bytes=100,
                floor_b_jf=0.25, floor_b_j=0.25, floor_b_f=0.25, floor_b_kbps=0.0,
            ),
        ]

        report_file = tmp_path / "test_report.md"
        report = generate_gate2_controls_report(evals, output_path=str(report_file))

        # Check that median delta (+25.00%) appears, not the skewed mean (+40.00%)
        assert "+25.00%" in report
        assert "Median Delta vs H.264" in report
        assert report_file.exists()

    def test_dense_rvq_frame_skip(self):
        """Dense RVQ frame-skipping (1/2 rate) must reduce wire bytes by ~50%."""
        dummy_tokens = [torch.randn(1, 256, 384, device=self.device) for _ in range(10)]
        prop_res, total_bytes, kbps = evaluate_dense_rvq_frame_skip(
            sequence="blackswan",
            raw_tokens_stream=dummy_tokens,
            packer=self.packer,
            num_quantizers=1,
            skip_interval=2,
            root_dir="data/DAVIS",
            fps=25.0,
            device=self.device,
        )
        assert prop_res.num_frames == 9  # 10 frames total, 9 evaluated
        # 10 frames with skip_interval=2: frames 0, 2, 4, 6, 8 transmitted = 5 packets
        # Frame 0: 306 bytes, Frame 2, 4, 6, 8: 306 bytes each -> 5 * 306 = 1530 bytes
        assert total_bytes == 5 * 306
        assert 0.0 <= prop_res.mean_j_and_f <= 1.0

    def test_lossless_codec_control(self, tmp_path):
        """Lossless H.264 (-qp 0 -pix_fmt yuv444p) J&F must match Raw ViT Oracle within 0.005 tolerance."""
        import tempfile
        from src.davis_loader import DAVISSequenceLoader
        from src.slicer import DINOv2Slicer
        from src.codec_baseline import encode_frames_to_h264, decode_h264_to_frames
        from src.label_propagation import evaluate_sequence_label_propagation

        slicer = DINOv2Slicer(device=self.device)
        loader = DAVISSequenceLoader(sequence="blackswan", root_dir="data/DAVIS")
        num_test_frames = 10

        raw_frames_rgb: List[np.ndarray] = []
        raw_tokens_stream: List[torch.Tensor] = []

        for i in range(num_test_frames):
            item = loader[i]
            frame_u8 = (item.frame.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            raw_frames_rgb.append(frame_u8)

            frame_tensor = item.frame.unsqueeze(0).to(self.device)
            s_out = slicer(frame_tensor)
            raw_tokens_stream.append(s_out.tokens.detach().cpu())

        # Evaluate Raw ViT Oracle on the 10 frames
        oracle_prop = evaluate_sequence_label_propagation(
            sequence="blackswan",
            tokens_stream=raw_tokens_stream,
            root_dir="data/DAVIS",
            device=self.device,
        )

        # Encode losslessly with FFmpeg libx264 (-qp 0 -pix_fmt yuv444p)
        h264_file = tmp_path / "lossless_test.h264"
        file_bytes = encode_frames_to_h264(
            frames_rgb=raw_frames_rgb,
            target_kbps=0.0,
            output_path=str(h264_file),
            fps=25.0,
            lossless=True,
            qp=0,
        )
        assert file_bytes > 0
        assert h264_file.exists()

        # Decode lossless stream
        decoded_frames = decode_h264_to_frames(
            str(h264_file),
            expected_frames=num_test_frames,
            width=224,
            height=224,
        )
        assert len(decoded_frames) == num_test_frames

        # Extract tokens from decoded frames
        lossless_tokens_stream: List[torch.Tensor] = []
        for dec_frame in decoded_frames:
            dec_t = torch.from_numpy(dec_frame).permute(2, 0, 1).unsqueeze(0).to(self.device)
            s_out = slicer(dec_t)
            lossless_tokens_stream.append(s_out.tokens.detach().cpu())

        # Evaluate label propagation on lossless decoded tokens
        lossless_prop = evaluate_sequence_label_propagation(
            sequence="blackswan",
            tokens_stream=lossless_tokens_stream,
            root_dir="data/DAVIS",
            device=self.device,
        )

        # Verify token fidelity and J&F parity within 0.005 tolerance
        cos_sims = [
            float(torch.nn.functional.cosine_similarity(raw_tokens_stream[i], lossless_tokens_stream[i], dim=-1).mean())
            for i in range(num_test_frames)
        ]
        avg_cos_sim = float(np.mean(cos_sims))
        assert avg_cos_sim > 0.995, f"Expected near-perfect cosine similarity, got {avg_cos_sim:.5f}"

        jf_diff = abs(lossless_prop.mean_j_and_f - oracle_prop.mean_j_and_f)
        assert jf_diff < 0.005, (
            f"Lossless J&F ({lossless_prop.mean_j_and_f:.4f}) diverged from "
            f"Oracle J&F ({oracle_prop.mean_j_and_f:.4f}) by {jf_diff:.4f} >= 0.005"
        )

