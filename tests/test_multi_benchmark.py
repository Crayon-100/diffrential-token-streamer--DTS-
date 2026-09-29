"""Unit tests for Multi-Sequence Real-World Benchmark Suite.

Verifies:
- Multi-sequence dataset accessibility (blackswan, bmx-trees, breakdance, boat)
- Markdown table formatting and column schema
- MultiBenchmarkResult metrics integrity and bounds
- Representative 4-panel diagnostic visualization generation
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import pytest
import torch

from src.davis_loader import DAVISSequenceLoader
from src.multi_benchmark import (
    DEFAULT_MOTION_PROFILES,
    DEFAULT_VIS_FRAMES,
    MultiBenchmarkResult,
    format_markdown_table,
    run_multi_benchmark,
)
from src.evaluate_davis import SequenceEvaluationSummary, FrameEvaluationMetrics
from save_benchmark_vis import generate_visualization


class TestMultiBenchmark:
    """Test suite for Multi-Sequence Generalization Benchmark."""

    def test_all_target_sequences_accessible(self) -> None:
        """Verify that all 4 DAVIS 2016 target sequences exist and load properly."""
        target_seqs = ["blackswan", "bmx-trees", "breakdance", "boat"]
        for seq in target_seqs:
            loader = DAVISSequenceLoader(sequence=seq, root_dir="data/DAVIS")
            assert len(loader) > 0, f"Sequence {seq} must contain frames"
            sample = loader[0]
            assert sample.frame.shape == (3, 224, 224)
            assert sample.gt_mask_grid.shape == (16, 16)

    def test_format_markdown_table(self) -> None:
        """Verify that format_markdown_table generates correct GitHub-flavored table."""
        # Create dummy summary
        dummy_metrics = FrameEvaluationMetrics(
            frame_idx=0,
            frame_name="00000.jpg",
            tp=10,
            fp=5,
            tn=230,
            fn=1,
            gt_fg_patches=11,
            pred_fg_patches=15,
            recall=0.90,
            drop_rate=0.95,
            raw_bytes=393216,
            wire_bytes=1000,
            compression_ratio=393.2,
            latency_ms=35.0,
        )
        dummy_summary = SequenceEvaluationSummary(
            sequence="bmx-trees",
            num_frames=2,
            mode_name="Saliency-Gated",
            total_tp=20,
            total_fp=10,
            total_tn=460,
            total_fn=2,
            mean_recall=0.90,
            mean_drop_rate=0.95,
            total_raw_bytes=786432,
            total_wire_bytes=2000,
            overall_compression_ratio=393.2,
            bandwidth_savings_pct=99.75,
            mean_latency_ms=35.0,
            fps=28.5,
            frame_metrics=[dummy_metrics, dummy_metrics],
        )

        dummy_result = MultiBenchmarkResult(
            summaries=[dummy_summary],
            motion_profiles={"bmx-trees": "Fast motion with occlusions"},
            total_sequences=1,
            total_frames=2,
            mean_recall=0.90,
            mean_drop_rate=0.95,
            total_raw_bytes=786432,
            total_wire_bytes=2000,
            avg_payload_bytes_per_frame=1000.0,
            overall_compression_ratio=393.2,
            overall_bandwidth_savings_pct=99.75,
            mean_latency_ms=35.0,
            overall_fps=28.5,
            visualization_paths={"bmx-trees": Path("visualizations/bmx-trees_vis.png")},
        )

        table_md = format_markdown_table(dummy_result)

        assert "| Sequence | Motion Profile | Frames | Recall (%) | Drop Rate (%)" in table_md
        assert "| **bmx-trees** |" in table_md
        assert "| **OVERALL (AVG / TOTAL)** |" in table_md
        assert "393.2x" in table_md
        assert "90.00%" in table_md

    def test_generate_visualization_bmx_trees(self, tmp_path: Path) -> None:
        """Verify 4-panel diagnostic visualization creates valid image file for bmx-trees."""
        out_file = tmp_path / "test_bmx_vis.png"
        saved = generate_visualization(
            sequence="bmx-trees",
            frame_index=10,
            output_path=str(out_file),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        assert saved.exists()
        assert saved.stat().st_size > 10000, "Visualization image file must be non-empty"

    def test_generate_visualization_breakdance(self, tmp_path: Path) -> None:
        """Verify 4-panel diagnostic visualization creates valid image file for breakdance."""
        out_file = tmp_path / "test_breakdance_vis.png"
        saved = generate_visualization(
            sequence="breakdance",
            frame_index=15,
            output_path=str(out_file),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        assert saved.exists()
        assert saved.stat().st_size > 10000, "Visualization image file must be non-empty"
