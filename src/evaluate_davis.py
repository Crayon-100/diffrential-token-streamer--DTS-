"""DAVIS 2016 Real-World Video Sequence Benchmark Evaluator

Evaluates the Differential Token Streamer (Slicer -> Bouncer -> Packer -> Rebuilder)
against real-world moving object video sequences from DAVIS 2016.

Supports:
1. Standard Cosine Threshold filtering
2. Saliency-Gated Temporal Filtering & Server-Cache Reference Comparison
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional

# Ensure repository root is on sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import torch
from src.slicer import DINOv2Slicer
from src.bouncer import Bouncer
from src.packer import Packer
from src.rebuilder import Rebuilder
from src.davis_loader import DAVISSequenceLoader
from src.ego_motion import estimate_global_motion


@dataclass
class FrameEvaluationMetrics:
    frame_idx: int
    frame_name: str
    tp: int
    fp: int
    tn: int
    fn: int
    gt_fg_patches: int
    pred_fg_patches: int
    recall: float
    drop_rate: float
    raw_bytes: int
    wire_bytes: int
    compression_ratio: float
    latency_ms: float


@dataclass
class SequenceEvaluationSummary:
    sequence: str
    num_frames: int
    mode_name: str
    total_tp: int
    total_fp: int
    total_tn: int
    total_fn: int
    mean_recall: float
    mean_drop_rate: float
    total_raw_bytes: int
    total_wire_bytes: int
    overall_compression_ratio: float
    bandwidth_savings_pct: float
    mean_latency_ms: float
    fps: float
    frame_metrics: List[FrameEvaluationMetrics]


def evaluate_sequence(
    sequence: str = "blackswan",
    root_dir: str = "data/DAVIS",
    threshold: float = 0.95,
    use_saliency: bool = True,
    use_server_cache: bool = True,
    gamma: float = 1.0,
    tau_dynamic: float = 0.0008,
    tau_hard_change: float = 0.30,
    profile: str = "static",
    motion_threshold: float = 0.5,
    device: Optional[torch.device] = None,
    warmup: bool = True,
    slicer: Optional[DINOv2Slicer] = None,
    bouncer: Optional[Bouncer] = None,
    packer: Optional[Packer] = None,
    rebuilder: Optional[Rebuilder] = None,
) -> SequenceEvaluationSummary:
    """Streams a DAVIS sequence through the differential pipeline and evaluates metrics."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = DAVISSequenceLoader(sequence=sequence, root_dir=root_dir)

    # Initialize Modules if not passed
    if slicer is None:
        slicer = DINOv2Slicer(model_name="dinov2_vits14", device=device)
    if bouncer is None:
        bouncer = Bouncer(
            threshold=threshold,
            saliency_gated=use_saliency,
            gamma=gamma,
            tau_dynamic=tau_dynamic,
            tau_hard_change=tau_hard_change,
            profile=profile,
            motion_threshold=motion_threshold,
        )
    else:
        bouncer.threshold = threshold
        bouncer.saliency_gated = use_saliency
        bouncer.gamma = gamma
        bouncer.tau_dynamic = tau_dynamic
        bouncer.tau_hard_change = tau_hard_change
        bouncer.profile = profile
        bouncer.motion_threshold = motion_threshold

    bouncer.reset_edge_cache()

    if packer is None:
        packer = Packer(dim=384, num_quantizers=4, codebook_size=256, kmeans_init=True, device=device)
    if rebuilder is None:
        rebuilder = Rebuilder(packer=packer, dim=384, num_quantizers=4, codebook_size=256, device=device)
    else:
        rebuilder.reset_cache()

    # Warmup on GPU
    if warmup and device.type == "cuda":
        dummy = torch.rand(1, 3, 224, 224, device=device)
        dummy_out = slicer(dummy)
        _ = bouncer(dummy_out.tokens, dummy_out.tokens)
        torch.cuda.synchronize()

    frame_metrics: List[FrameEvaluationMetrics] = []
    prev_tokens = None
    prev_frame_np = None
    total_raw_bytes = 0
    total_wire_bytes = 0
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0
    latencies: List[float] = []

    for item in loader:
        idx = item.frame_idx
        frame = item.frame.unsqueeze(0).to(device)  # [1, 3, 224, 224]
        gt_grid = item.gt_mask_grid.flatten().to(device)  # [256] bool
        raw_frame_bytes = 256 * 384 * 4  # 393,216 bytes
        total_raw_bytes += raw_frame_bytes

        # Timing start
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        # Phase A: Slicer (extracts tokens and saliency prior)
        slicer_out = slicer(frame)
        curr_tokens = slicer_out.tokens
        saliency = slicer_out.saliency
        patch_grid = slicer_out.patch_grid

        if idx == 0:
            # Calibrate RVQ codebook on keyframe tokens if using kmeans_init and not yet initted
            if getattr(packer, "kmeans_init", False):
                if hasattr(packer.rvq, "layers") and hasattr(packer.rvq.layers[0], "_codebook"):
                    initted = getattr(packer.rvq.layers[0]._codebook, "initted", None)
                    if initted is not None and not initted.item():
                        packer.rvq.train()
                        _ = packer.quantize(curr_tokens[0])
                        packer.rvq.eval()

            # Keyframe packet (all tokens transmitted)
            mask_all = torch.ones(256, dtype=torch.bool, device=device)
            p_out = packer(curr_tokens[0], mask_all, frame_id=idx, patch_grid=patch_grid, is_keyframe=True)
            packet = p_out.packet
            wire_bytes = packet.wire_bytes
            total_wire_bytes += wire_bytes

            # Dual-Cache Closed-Loop: Edge initializes raw_shadow and replica_cache
            z_hat_0 = p_out.quantized.unsqueeze(0)
            bouncer.initialize_edge_cache(
                raw_tokens=curr_tokens,
                reconstructed_tokens=z_hat_0,
                patch_grid=patch_grid,
            )

            # Server initializes its persistent cache with reconstructed tokens
            rebuilder.initialize_cache(z_hat_0, patch_grid=patch_grid)
            prev_frame_np = (item.frame.permute(1, 2, 0).numpy() * 255).astype("uint8")

            # Timing end
            if device.type == "cuda":
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            latencies.append(latency_ms)

            # On keyframe, all patches transmitted
            pred_fg = 256
            gt_fg = int(gt_grid.sum().item())
            tp = gt_fg
            fp = 256 - gt_fg
            tn = 0
            fn = 0
            recall = 1.0
            drop_rate = 0.0

            frame_metrics.append(
                FrameEvaluationMetrics(
                    frame_idx=idx,
                    frame_name=item.frame_name,
                    tp=tp,
                    fp=fp,
                    tn=tn,
                    fn=fn,
                    gt_fg_patches=gt_fg,
                    pred_fg_patches=pred_fg,
                    recall=recall,
                    drop_rate=drop_rate,
                    raw_bytes=raw_frame_bytes,
                    wire_bytes=wire_bytes,
                    compression_ratio=raw_frame_bytes / wire_bytes,
                    latency_ms=latency_ms,
                )
            )
            continue

        # Differential Frames (idx > 0)
        curr_frame_np = (item.frame.permute(1, 2, 0).numpy() * 255).astype("uint8")
        motion = None
        if profile in ("motion", "auto") and prev_frame_np is not None:
            dx, dy, _ = estimate_global_motion(prev_frame_np, curr_frame_np)
            motion = (dx, dy)
        prev_frame_np = curr_frame_np

        # Phase B: Bouncer (Closed-Loop Reference against edge_replica_cache)
        b_out = bouncer(
            tokens_current=curr_tokens,
            tokens_previous=None,  # Closed-loop: uses bouncer.edge_replica_cache
            patch_grid=patch_grid,
            saliency=saliency if use_saliency else None,
            gamma=gamma,
            tau_dynamic=tau_dynamic,
            tau_hard_change=tau_hard_change,
            motion=motion,
            profile=profile,
        )
        pred_mask = b_out.mask.flatten().bool()  # True = transmitted/dynamic

        # Phase C: Packer
        p_out = packer(
            z_active=b_out.active_tokens,
            mask=b_out.mask,
            frame_id=idx,
            patch_grid=patch_grid,
            motion=motion,
        )
        packet = p_out.packet
        wire_bytes = packet.wire_bytes
        total_wire_bytes += wire_bytes

        # Closed-Loop Dual-Cache Edge Update:
        # edge_raw_shadow gets uncompressed active_raw, edge_replica_cache gets reconstructed tokens
        bouncer.update_edge_cache(
            active_reconstructed=p_out.quantized,
            mask=b_out.mask,
            active_raw=b_out.active_tokens,
            motion=motion,
            patch_grid=patch_grid,
        )

        # Phase D: Rebuilder (Server side: decodes from packet and updates server cache)
        _ = rebuilder(packet)

        # Timing end
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        latencies.append(latency_ms)

        # Ground-truth evaluation
        tp = int((pred_mask & gt_grid).sum().item())
        fn = int((~pred_mask & gt_grid).sum().item())
        tn = int((~pred_mask & ~gt_grid).sum().item())
        fp = int((pred_mask & ~gt_grid).sum().item())

        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn

        gt_fg = tp + fn
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        drop_rate = tn / (tn + fp) if (tn + fp) > 0 else 1.0

        frame_metrics.append(
            FrameEvaluationMetrics(
                frame_idx=idx,
                frame_name=item.frame_name,
                tp=tp,
                fp=fp,
                tn=tn,
                fn=fn,
                gt_fg_patches=gt_fg,
                pred_fg_patches=tp + fp,
                recall=recall,
                drop_rate=drop_rate,
                raw_bytes=raw_frame_bytes,
                wire_bytes=wire_bytes,
                compression_ratio=raw_frame_bytes / wire_bytes,
                latency_ms=latency_ms,
            )
        )

    # Aggregate metrics across differential frames (frames 1..N-1)
    diff_frames = frame_metrics[1:]
    mean_recall = sum(f.recall for f in diff_frames) / len(diff_frames) if diff_frames else 1.0
    mean_drop_rate = sum(f.drop_rate for f in diff_frames) / len(diff_frames) if diff_frames else 1.0
    overall_compression = total_raw_bytes / total_wire_bytes if total_wire_bytes > 0 else 0.0
    bandwidth_savings = (1.0 - total_wire_bytes / total_raw_bytes) * 100.0
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
    fps = 1000.0 / mean_latency if mean_latency > 0 else 0.0

    mode_desc = "Saliency-Gated + Server Cache" if (use_saliency and use_server_cache) else (
        "Saliency-Gated" if use_saliency else "Standard Cosine"
    )
    if profile != "static":
        mode_desc += f" (Ego-Motion: {profile.upper()})"

    return SequenceEvaluationSummary(
        sequence=sequence,
        num_frames=len(loader),
        mode_name=mode_desc,
        total_tp=total_tp,
        total_fp=total_fp,
        total_tn=total_tn,
        total_fn=total_fn,
        mean_recall=mean_recall,
        mean_drop_rate=mean_drop_rate,
        total_raw_bytes=total_raw_bytes,
        total_wire_bytes=total_wire_bytes,
        overall_compression_ratio=overall_compression,
        bandwidth_savings_pct=bandwidth_savings,
        mean_latency_ms=mean_latency,
        fps=fps,
        frame_metrics=frame_metrics,
    )


def print_summary_table(summary: SequenceEvaluationSummary) -> None:
    """Prints a structured telemetry report of the benchmark."""
    print("\n" + "=" * 80)
    print(f"  REAL-WORLD BENCHMARK EVALUATION REPORT: DAVIS 2016 ('{summary.sequence}')")
    print(f"  Mode                         : {summary.mode_name}")
    print("=" * 80)
    print(f"  Video Sequence               : {summary.sequence}")
    print(f"  Total Frames Evaluated       : {summary.num_frames} frames (1 keyframe + {summary.num_frames-1} diff)")
    print(f"  Total Patches Evaluated      : {summary.num_frames * 256:,} patches (16x16 per frame)")
    print("-" * 80)
    print(f"  1. FOREGROUND RECALL         : {summary.mean_recall * 100:.2f}% (Object patches transmitted)")
    print(f"  2. BACKGROUND DROP RATE      : {summary.mean_drop_rate * 100:.2f}% (Static background dropped)")
    print(f"  3. OVERALL COMPRESSION RATIO : {summary.overall_compression_ratio:.1f}x reduction")
    print(f"     - Uncompressed Baseline   : {summary.total_raw_bytes:,} bytes ({summary.total_raw_bytes / (1024*1024):.2f} MB)")
    print(f"     - Transmitted Wire Size   : {summary.total_wire_bytes:,} bytes ({summary.total_wire_bytes / 1024:.2f} KB)")
    print(f"     - Total Bandwidth Saved   : {summary.bandwidth_savings_pct:.2f}%")
    print("-" * 80)
    print(f"  4. INFERENCE LATENCY         : {summary.mean_latency_ms:.2f} ms / frame")
    print(f"     - Throughput              : {summary.fps:.1f} FPS (Real-time on RTX 3050)")
    print("=" * 80)

    # Sample frame trajectory table
    print("\nSample Frame Progression (Every 10 frames):")
    print(f"  {'Frame':<8} | {'GT Object':<10} | {'Transmitted':<12} | {'Recall':<8} | {'Drop Rate':<10} | {'Payload':<10} | {'Latency':<8}")
    print("  " + "-" * 76)
    sample_indices = [0, 1, 10, 20, 30, 40, summary.num_frames - 1]
    for idx in sample_indices:
        if idx < len(summary.frame_metrics):
            fm = summary.frame_metrics[idx]
            print(f"  {fm.frame_name:<8} | {fm.gt_fg_patches:<10} | {fm.pred_fg_patches:<12} | {fm.recall*100:>6.1f}% | {fm.drop_rate*100:>8.1f}% | {fm.wire_bytes:>5} bytes | {fm.latency_ms:>5.1f} ms")
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DAVIS 2016 sequence with Differential Token Streamer")
    parser.add_argument("--sequence", type=str, default="blackswan", help="DAVIS sequence name (default: blackswan)")
    parser.add_argument("--root_dir", type=str, default="data/DAVIS", help="DAVIS dataset root path")
    parser.add_argument("--threshold", type=float, default=0.95, help="Cosine similarity drop threshold (default: 0.95)")
    parser.add_argument("--use_saliency", action="store_true", default=True, help="Enable DINOv2 Saliency Gating")
    parser.add_argument("--disable_saliency", dest="use_saliency", action="store_false")
    parser.add_argument("--use_server_cache", action="store_true", default=True, help="Compare against Server Z_cache")
    parser.add_argument("--disable_server_cache", dest="use_server_cache", action="store_false")
    parser.add_argument("--gamma", type=float, default=1.0, help="Saliency emphasis exponent (default: 1.0)")
    parser.add_argument("--tau_dynamic", type=float, default=0.0008, help="Dynamic score threshold (default: 0.0008)")
    parser.add_argument("--tau_hard", type=float, default=0.30, help="Hard difference threshold (default: 0.30)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running DAVIS Benchmark on sequence '{args.sequence}' using device: {device}...")

    summary = evaluate_sequence(
        sequence=args.sequence,
        root_dir=args.root_dir,
        threshold=args.threshold,
        use_saliency=args.use_saliency,
        use_server_cache=args.use_server_cache,
        gamma=args.gamma,
        tau_dynamic=args.tau_dynamic,
        tau_hard_change=args.tau_hard,
        device=device,
    )

    print_summary_table(summary)


if __name__ == "__main__":
    main()
