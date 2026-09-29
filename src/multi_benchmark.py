"""Multi-Sequence Real-World Generalization Benchmark Suite for DAVIS 2016.

Evaluates the Saliency-Gated Differential Token Streamer across diverse motion profiles:
1. 'blackswan'   : Smooth non-rigid gliding on reflective water surface
2. 'bmx-trees'   : Fast non-linear bicycle motion with severe tree occlusions & panning
3. 'breakdance'  : Articulated human motion, rapid limb transitions & floor contact
4. 'boat'        : Rigid vehicle motion on active water surface with wake ripples

Computes publication-ready telemetry:
- Foreground Recall (%)
- Background Drop Rate (%)
- Average Wire Payload per frame (Bytes)
- Bandwidth Compression Factor (x reduction vs raw 393,216-byte float tokens)
- Inference Latency (ms/frame) and FPS on NVIDIA GPU
- Saves Markdown comparison report and 4-panel diagnostic visualizations to visualizations/
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import matplotlib
matplotlib.use("Agg")

import torch
from src.slicer import DINOv2Slicer
from src.bouncer import Bouncer
from src.packer import Packer
from src.rebuilder import Rebuilder
from src.evaluate_davis import evaluate_sequence, SequenceEvaluationSummary
from save_benchmark_vis import generate_visualization

# Motion profile metadata descriptions
DEFAULT_MOTION_PROFILES: Dict[str, str] = {
    "blackswan": "Smooth non-rigid gliding on reflective water surface",
    "bmx-trees": "Fast non-linear motion with tree occlusions & camera panning",
    "breakdance": "Articulated human motion, rapid limb transitions & floor contact",
    "boat": "Rigid vehicle displacement across active water wake & ripples",
}

# Representative dynamic frames for 4-panel diagnostic visualization
DEFAULT_VIS_FRAMES: Dict[str, int] = {
    "blackswan": 15,
    "bmx-trees": 20,
    "breakdance": 25,
    "boat": 20,
}


@dataclass
class MultiBenchmarkResult:
    """Aggregate multi-sequence benchmark results."""
    summaries: List[SequenceEvaluationSummary]
    motion_profiles: Dict[str, str]
    total_sequences: int
    total_frames: int
    mean_recall: float
    mean_drop_rate: float
    total_raw_bytes: int
    total_wire_bytes: int
    avg_payload_bytes_per_frame: float
    overall_compression_ratio: float
    overall_bandwidth_savings_pct: float
    mean_latency_ms: float
    overall_fps: float
    visualization_paths: Dict[str, Path]


def format_markdown_table(result: MultiBenchmarkResult) -> str:
    """Formats a publication-ready Markdown table of the benchmark comparison."""
    lines = [
        "# Multi-Sequence Real-World Generalization Benchmark: DAVIS 2016",
        "",
        "Evaluation of the **Differential Token Streamer** with **Saliency-Gated Temporal Filtering** across diverse motion dynamics.",
        "",
        "| Sequence | Motion Profile | Frames | Recall (%) | Drop Rate (%) | Avg Payload (B) | Compression (x) | Bandwidth Saved | Latency (ms) | FPS |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for s in result.summaries:
        profile = result.motion_profiles.get(s.sequence, "Moving object video sequence")
        avg_payload = s.total_wire_bytes / s.num_frames if s.num_frames > 0 else 0
        lines.append(
            f"| **{s.sequence}** | {profile} | {s.num_frames} | "
            f"**{s.mean_recall * 100:.2f}%** | {s.mean_drop_rate * 100:.2f}% | "
            f"{avg_payload:,.1f} B | **{s.overall_compression_ratio:,.1f}x** | "
            f"{s.bandwidth_savings_pct:.2f}% | {s.mean_latency_ms:.2f} ms | "
            f"**{s.fps:.1f}** |"
        )

    # Summary Aggregate Row
    lines.append(
        f"| **OVERALL (AVG / TOTAL)** | *Macro Multi-Sequence Aggregate* | **{result.total_frames}** | "
        f"**{result.mean_recall * 100:.2f}%** | **{result.mean_drop_rate * 100:.2f}%** | "
        f"**{result.avg_payload_bytes_per_frame:,.1f} B** | **{result.overall_compression_ratio:,.1f}x** | "
        f"**{result.overall_bandwidth_savings_pct:.2f}%** | **{result.mean_latency_ms:.2f} ms** | "
        f"**{result.overall_fps:.1f}** |"
    )

    summary_map = {s.sequence: s for s in result.summaries}
    lines.append("")
    lines.append("### Key Architectural Insights Across Motion Dynamics:")

    insight_idx = 1
    if "blackswan" in summary_map and "boat" in summary_map:
        bs = summary_map["blackswan"]
        bt = summary_map["boat"]
        lines.append(
            f"{insight_idx}. **Reflective Water Surfaces (`blackswan`, `boat`)**: Saliency gating effectively filters out ambient water ripple flicker "
            f"(Drop Rates **{bs.mean_drop_rate * 100:.2f}%** and **{bt.mean_drop_rate * 100:.2f}%**), "
            f"delivering **{bs.overall_compression_ratio:,.1f}x** and **{bt.overall_compression_ratio:,.1f}x** bandwidth compression."
        )
        insight_idx += 1
    elif any("swan" in s.lower() or "boat" in s.lower() or "water" in s.lower() for s in summary_map):
        for seq_name, s in summary_map.items():
            if any(k in seq_name.lower() for k in ["swan", "boat", "water"]):
                lines.append(
                    f"{insight_idx}. **Water / Surface Reflections (`{seq_name}`)**: Drop Rate **{s.mean_drop_rate * 100:.2f}%** with **{s.overall_compression_ratio:,.1f}x** compression."
                )
                insight_idx += 1

    if "breakdance" in summary_map:
        bd = summary_map["breakdance"]
        lines.append(
            f"{insight_idx}. **Articulated Human Motion (`breakdance`)**: Rapid limb extensions and acrobatics achieve high foreground tracking "
            f"(**{bd.mean_recall * 100:.2f}% Recall**) with **{bd.overall_compression_ratio:,.1f}x** compression."
        )
        insight_idx += 1

    if "bmx-trees" in summary_map:
        bmx = summary_map["bmx-trees"]
        lines.append(
            f"{insight_idx}. **Camera Panning & Occlusions (`bmx-trees`)**: Camera translation induces global background optical motion across tree trunks. "
            f"The RVQ packer achieves **{bmx.overall_compression_ratio:,.1f}x** compression while retaining **{bmx.mean_recall * 100:.2f}% Foreground Recall**."
        )
        insight_idx += 1

    lines.append(
        f"{insight_idx}. **Real-Time Edge Inference**: Mean edge throughput of **{result.overall_fps:.1f} FPS** "
        f"(**{result.mean_latency_ms:.2f} ms/frame**) on target hardware across {result.total_frames} evaluated frames."
    )

    if result.visualization_paths:
        lines.append("")
        lines.append("### Diagnostic Visualizations:")
        for seq, path in result.visualization_paths.items():
            lines.append(f"- **{seq}**: [`{path.as_posix()}`]({path.as_posix()})")

    return "\n".join(lines)


def run_multi_benchmark(
    sequences: Optional[List[str]] = None,
    root_dir: str = "data/DAVIS",
    tau_dynamic: float = 0.0005,
    tau_hard_change: float = 0.30,
    gamma: float = 1.0,
    profile: str = "auto",
    use_server_cache: bool = False,
    output_report: Optional[str] = "BENCHMARK_REPORT.md",
    save_vis: bool = True,
    vis_dir: str = "visualizations",
    device: Optional[torch.device] = None,
) -> MultiBenchmarkResult:
    """Executes the complete multi-sequence generalization benchmark."""
    if sequences is None:
        sequences = ["blackswan", "bmx-trees", "breakdance", "boat"]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vis_path = Path(vis_dir)
    vis_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 90)
    print("  STARTING MULTI-SEQUENCE REAL-WORLD GENERALIZATION BENCHMARK (DAVIS 2016)")
    print(f"  Target Sequences  : {', '.join(sequences)}")
    print(f"  Inference Device  : {device}")
    print(f"  Dynamic Threshold : tau_dynamic={tau_dynamic}, tau_hard={tau_hard_change}, gamma={gamma}")
    print("=" * 90 + "\n")

    # Instantiate shared pipeline modules once on target device
    print("Initializing pipeline modules on device...")
    slicer = DINOv2Slicer(model_name="dinov2_vits14", device=device)
    bouncer = Bouncer(
        threshold=0.95,
        saliency_gated=True,
        gamma=gamma,
        tau_dynamic=tau_dynamic,
        tau_hard_change=tau_hard_change,
    )
    packer = Packer(dim=384, num_quantizers=4, codebook_size=256, kmeans_init=True, device=device)
    rebuilder = Rebuilder(packer=packer, dim=384, num_quantizers=4, codebook_size=256, device=device)

    # Warmup
    if device.type == "cuda":
        dummy = torch.rand(1, 3, 224, 224, device=device)
        dummy_out = slicer(dummy)
        _ = bouncer(dummy_out.tokens, dummy_out.tokens)
        torch.cuda.synchronize()

    summaries: List[SequenceEvaluationSummary] = []
    vis_paths: Dict[str, Path] = {}

    for seq in sequences:
        print(f"--> Evaluating Sequence: '{seq}' ({DEFAULT_MOTION_PROFILES.get(seq, '')})...")
        t0 = time.time()
        summary = evaluate_sequence(
            sequence=seq,
            root_dir=root_dir,
            threshold=0.95,
            use_saliency=True,
            use_server_cache=use_server_cache,
            gamma=gamma,
            tau_dynamic=tau_dynamic,
            tau_hard_change=tau_hard_change,
            profile=profile,
            device=device,
            warmup=False,
            slicer=slicer,
            bouncer=bouncer,
            packer=packer,
            rebuilder=rebuilder,
        )
        elapsed = time.time() - t0
        summaries.append(summary)

        print(
            f"    Done ({summary.num_frames} frames in {elapsed:.2f}s). "
            f"Recall: {summary.mean_recall * 100:.2f}%, "
            f"Drop Rate: {summary.mean_drop_rate * 100:.2f}%, "
            f"Compression: {summary.overall_compression_ratio:.1f}x, "
            f"FPS: {summary.fps:.1f}"
        )

        # Generate representative 4-panel visualization
        if save_vis:
            frame_to_vis = DEFAULT_VIS_FRAMES.get(seq, min(15, summary.num_frames - 1))
            out_img = vis_path / f"{seq}_vis.png"
            generate_visualization(
                sequence=seq,
                frame_index=frame_to_vis,
                gamma=gamma,
                tau_dynamic=tau_dynamic,
                tau_hard_change=tau_hard_change,
                output_path=str(out_img),
                device=device,
                slicer=slicer,
                bouncer=bouncer,
            )
            vis_paths[seq] = out_img
            print(f"    Saved diagnostic visual artifact: {out_img}")

    # Compute Macro Aggregate Metrics
    total_frames = sum(s.num_frames for s in summaries)
    mean_recall = sum(s.mean_recall for s in summaries) / len(summaries) if summaries else 0.0
    mean_drop_rate = sum(s.mean_drop_rate for s in summaries) / len(summaries) if summaries else 0.0
    total_raw_bytes = sum(s.total_raw_bytes for s in summaries)
    total_wire_bytes = sum(s.total_wire_bytes for s in summaries)
    overall_compression = total_raw_bytes / total_wire_bytes if total_wire_bytes > 0 else 0.0
    overall_savings = (1.0 - total_wire_bytes / total_raw_bytes) * 100.0 if total_raw_bytes > 0 else 0.0
    avg_payload = total_wire_bytes / total_frames if total_frames > 0 else 0.0
    mean_latency = sum(s.mean_latency_ms for s in summaries) / len(summaries) if summaries else 0.0
    overall_fps = 1000.0 / mean_latency if mean_latency > 0 else 0.0

    result = MultiBenchmarkResult(
        summaries=summaries,
        motion_profiles=DEFAULT_MOTION_PROFILES,
        total_sequences=len(summaries),
        total_frames=total_frames,
        mean_recall=mean_recall,
        mean_drop_rate=mean_drop_rate,
        total_raw_bytes=total_raw_bytes,
        total_wire_bytes=total_wire_bytes,
        avg_payload_bytes_per_frame=avg_payload,
        overall_compression_ratio=overall_compression,
        overall_bandwidth_savings_pct=overall_savings,
        mean_latency_ms=mean_latency,
        overall_fps=overall_fps,
        visualization_paths=vis_paths,
    )

    # Format Markdown Report
    report_md = format_markdown_table(result)
    print("\n" + "=" * 90)
    print("  MULTI-SEQUENCE BENCHMARK REPORT")
    print("=" * 90)
    print(report_md)

    if output_report:
        report_file = Path(output_report)
        report_file.write_text(report_md, encoding="utf-8")
        print(f"\nSaved Markdown Benchmark Report to: {report_file.resolve()}\n")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Sequence Real-World Benchmark on DAVIS 2016")
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=["blackswan", "bmx-trees", "breakdance", "boat"],
        help="List of DAVIS sequence names to evaluate",
    )
    parser.add_argument("--root_dir", type=str, default="data/DAVIS", help="Path to DAVIS dataset")
    parser.add_argument("--tau_dynamic", type=float, default=0.0005, help="Dynamic score threshold")
    parser.add_argument("--tau_hard", type=float, default=0.30, help="Hard cosine difference threshold")
    parser.add_argument(
        "--profile",
        type=str,
        default="auto",
        choices=["static", "motion", "auto"],
        help="Operational profile for ego-motion engine: 'static' (Fast-Path), 'motion' (Warp-Path), or 'auto' (Adaptive Dual-Path)",
    )
    parser.add_argument("--gamma", type=float, default=1.0, help="Foreground saliency gating exponent")
    parser.add_argument("--use_server_cache", action="store_true", help="Compare against server cache rather than frame t-1")
    parser.add_argument("--output_report", type=str, default="BENCHMARK_REPORT.md", help="Output Markdown report path")
    parser.add_argument("--vis_dir", type=str, default="visualizations", help="Directory to save 4-panel visual plots")
    parser.add_argument("--no_vis", action="store_true", help="Disable saving visualizations")
    args = parser.parse_args()

    run_multi_benchmark(
        sequences=args.sequences,
        root_dir=args.root_dir,
        tau_dynamic=args.tau_dynamic,
        tau_hard_change=args.tau_hard,
        gamma=args.gamma,
        profile=args.profile,
        use_server_cache=args.use_server_cache,
        output_report=args.output_report,
        save_vis=not args.no_vis,
        vis_dir=args.vis_dir,
    )


if __name__ == "__main__":
    main()
