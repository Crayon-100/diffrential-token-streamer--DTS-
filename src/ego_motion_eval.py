"""Evaluation & Comparison Suite: Baseline vs. Adaptive Dual-Path Ego-Motion Engine.

Evaluates:
1. 'blackswan' (Static / smooth camera gliding - verifies zero regression).
2. 'bmx-trees' (Active camera panning & tracking - demonstrates background leakage suppression).

Profiles compared:
- Baseline: Static Fast-Path (direct cache comparison, 0 ms warp overhead)
- Adaptive: Auto Dual-Path (sub-pixel phase correlation + F.grid_sample token warping)
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Ensure repository root is on sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import torch
from src.slicer import DINOv2Slicer
from src.bouncer import Bouncer
from src.packer import Packer
from src.rebuilder import Rebuilder
from src.evaluate_davis import evaluate_sequence, SequenceEvaluationSummary


def run_ego_motion_comparison(
    sequences: List[str] = ["blackswan", "bmx-trees"],
    root_dir: str = "data/DAVIS",
    tau_dynamic: float = 0.004,
    tau_hard_change: float = 0.30,
    gamma: float = 1.0,
    device: str = "cuda",
) -> Dict[str, Dict[str, SequenceEvaluationSummary]]:
    """Runs comparative evaluation of Baseline vs Adaptive Ego-Motion."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 95)
    print("  ADAPTIVE DUAL-PATH EGO-MOTION BENCHMARK COMPARISON (DAVIS 2016)")
    print(f"  Sequences: {sequences} | Device: {dev} | tau_dyn: {tau_dynamic}")
    print("=" * 95 + "\n")

    slicer = DINOv2Slicer(device=dev)
    results = {}

    for seq in sequences:
        results[seq] = {}
        for profile in ["static", "auto"]:
            print(f"--> Running {seq} with profile '{profile.upper()}'...")
            packer = Packer(device=dev)
            rebuilder = Rebuilder(packer=packer, device=dev)
            bouncer = Bouncer(
                threshold=0.95,
                saliency_gated=True,
                gamma=gamma,
                tau_dynamic=tau_dynamic,
                tau_hard_change=tau_hard_change,
                profile=profile,
                motion_threshold=0.5,
            )

            summary = evaluate_sequence(
                sequence=seq,
                root_dir=root_dir,
                threshold=0.95,
                use_saliency=True,
                use_server_cache=True,
                gamma=gamma,
                tau_dynamic=tau_dynamic,
                tau_hard_change=tau_hard_change,
                profile=profile,
                motion_threshold=0.5,
                device=dev,
                warmup=True,
                slicer=slicer,
                bouncer=bouncer,
                packer=packer,
                rebuilder=rebuilder,
            )
            results[seq][profile] = summary
            avg_payload = summary.total_wire_bytes / summary.num_frames
            print(
                f"    [{profile.upper():6s}] Recall: {summary.mean_recall * 100:.2f}% | "
                f"Drop Rate: {summary.mean_drop_rate * 100:.2f}% | "
                f"Comp: {summary.overall_compression_ratio:.1f}x | "
                f"Avg Wire: {avg_payload:.1f} B | "
                f"Latency: {summary.mean_latency_ms:.2f} ms ({summary.fps:.1f} FPS)"
            )

    return results


def print_comparison_table(results: Dict[str, Dict[str, SequenceEvaluationSummary]]) -> str:
    """Formats Markdown before vs after comparison table."""
    lines = [
        "## Adaptive Dual-Path Ego-Motion Engine: Before vs. After Benchmark",
        "",
        "| Sequence | Configuration | Profile | Recall (%) | Drop Rate (%) | Avg Payload (B) | Compression (x) | Latency (ms) | FPS | Status |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]

    for seq, profiles in results.items():
        base = profiles["static"]
        adapt = profiles["auto"]

        base_payload = base.total_wire_bytes / base.num_frames
        adapt_payload = adapt.total_wire_bytes / adapt.num_frames

        # Compare metrics
        recall_diff = (adapt.mean_recall - base.mean_recall) * 100
        drop_diff = (adapt.mean_drop_rate - base.mean_drop_rate) * 100
        comp_diff = adapt.overall_compression_ratio - base.overall_compression_ratio

        if seq == "blackswan":
            status = "Verified (Zero Regression)" if recall_diff >= -0.5 else "Degraded"
        else:
            status = f"+{drop_diff:.1f}% Drop Rate" if drop_diff > 0 else "Neutral"

        lines.append(
            f"| **{seq}** | Baseline | STATIC | {base.mean_recall * 100:.2f}% | "
            f"{base.mean_drop_rate * 100:.2f}% | {base_payload:,.1f} B | "
            f"{base.overall_compression_ratio:.1f}x | {base.mean_latency_ms:.2f} ms | "
            f"{base.fps:.1f} | Baseline Reference |"
        )
        lines.append(
            f"| **{seq}** | Ego-Motion Engine | **AUTO** | **{adapt.mean_recall * 100:.2f}%** ({recall_diff:+.2f}%) | "
            f"**{adapt.mean_drop_rate * 100:.2f}%** ({drop_diff:+.2f}%) | {adapt_payload:,.1f} B | "
            f"**{adapt.overall_compression_ratio:.1f}x** ({comp_diff:+.1f}x) | {adapt.mean_latency_ms:.2f} ms | "
            f"**{adapt.fps:.1f}** | **{status}** |"
        )

    table_md = "\n".join(lines)
    print("\n" + table_md + "\n")
    return table_md


if __name__ == "__main__":
    results = run_ego_motion_comparison()
    print_comparison_table(results)
