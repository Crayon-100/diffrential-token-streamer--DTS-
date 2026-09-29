"""Dense RVQ Controls and Static Floor Baselines for Scientific Verification.

Implements experimental controls and floor baselines for Gate 2:
1. Dense Q=1: Gating OFF (all 256 tokens sent every frame) with RVQ Stage 1 (~61.2 kbps).
2. Dense Q=2: Gating OFF (all 256 tokens sent every frame) with RVQ Stages 1 & 2 (~112.4 kbps).
3. Floor A (Frozen Cache): Transmit Frame 0 keyframe, then 0 tokens; evaluate from static cache.
4. Floor B (Copy Frame 0 Mask): Naive zero-motion baseline; copy Frame 0 GT mask forward.
"""

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Union

# Ensure repository root is on sys.path
root_dir_path = str(Path(__file__).resolve().parent.parent)
if root_dir_path not in sys.path:
    sys.path.insert(0, root_dir_path)

import numpy as np
import torch

from src.codec_baseline import compute_bitrate_kbps
from src.davis_loader import DAVISSequenceLoader
from src.label_propagation import (
    compute_davis_metrics,
    evaluate_sequence_label_propagation,
    FrameLabelPropagationResult,
    SequenceLabelPropagationResult,
)
from src.packer import Packer, DEFAULT_CODEBOOK_PATH
from src.rebuilder import Rebuilder


@dataclass
class FloorBResult:
    """Evaluation result for Floor B (Copy Frame 0 Mask)."""
    sequence: str
    num_frames: int
    mean_jaccard: float
    mean_f_measure: float
    mean_j_and_f: float
    wire_kbps: float = 0.0


def compute_floor_b_copy_mask(
    sequence: str,
    root_dir: str = "data/DAVIS",
    bound_th: Optional[float] = None,
) -> FloorBResult:
    """Evaluates Floor B: copies Frame 0 ground-truth mask to all subsequent frames.

    Args:
        sequence: DAVIS sequence name.
        root_dir: Path to DAVIS dataset.
        bound_th: Boundary distance threshold. If None, uses official 0.008 * diagonal.

    Returns:
        FloorBResult with mean J, mean F, and mean J&F.
    """
    loader = DAVISSequenceLoader(sequence=sequence, root_dir=root_dir)
    n_frames = len(loader)
    if n_frames < 2:
        raise ValueError(f"Sequence {sequence} requires at least 2 frames.")

    item0 = loader[0]
    gt_mask_0 = item0.gt_mask_native.numpy().astype(np.uint8) if item0.gt_mask_native is not None else item0.gt_mask_pixel.numpy().astype(np.uint8)
    eff_bound_th = bound_th
    if eff_bound_th is None:
        diag = np.sqrt(float(gt_mask_0.shape[0] ** 2 + gt_mask_0.shape[1] ** 2))
        eff_bound_th = 0.008 * diag

    j_list: List[float] = []
    f_list: List[float] = []

    for t in range(1, n_frames):
        item_t = loader[t]
        gt_mask_t = item_t.gt_mask_native.numpy().astype(np.uint8) if item_t.gt_mask_native is not None else item_t.gt_mask_pixel.numpy().astype(np.uint8)
        metrics = compute_davis_metrics(pred_mask=gt_mask_0, gt_mask=gt_mask_t, bound_th=eff_bound_th)
        j_list.append(metrics["jaccard"])
        f_list.append(metrics["f_measure"])

    mean_j = float(np.mean(j_list)) if j_list else 0.0
    mean_f = float(np.mean(f_list)) if f_list else 0.0
    mean_jf = (mean_j + mean_f) / 2.0

    return FloorBResult(
        sequence=sequence,
        num_frames=n_frames - 1,
        mean_jaccard=mean_j,
        mean_f_measure=mean_f,
        mean_j_and_f=mean_jf,
        wire_kbps=0.0,
    )


def evaluate_frozen_cache_baseline(
    sequence: str,
    frame0_reconstructed_tokens: torch.Tensor,
    num_frames: int,
    frame0_wire_bytes: int,
    root_dir: str = "data/DAVIS",
    fps: float = 25.0,
    device: Optional[torch.device] = None,
) -> Tuple[SequenceLabelPropagationResult, int, float]:
    """Evaluates Floor A: Frame 0 keyframe sent, 0 tokens sent on subsequent frames.

    Args:
        sequence: DAVIS sequence name.
        frame0_reconstructed_tokens: Reconstructed tokens [1, 256, 384] from Frame 0.
        num_frames: Total frames in sequence.
        frame0_wire_bytes: Payload size of Frame 0 keyframe.
        root_dir: Path to DAVIS dataset.
        fps: Video framerate.
        device: Torch computation device.

    Returns:
        Tuple of (SequenceLabelPropagationResult, total_wire_bytes, wire_kbps).
    """
    total_wire_bytes = frame0_wire_bytes  # 0 bytes for frames 1..N-1
    wire_kbps = compute_bitrate_kbps(total_wire_bytes, num_frames=num_frames, fps=fps)

    # Static stream: server cache is identical to frame 0 for all frames
    static_stream = [frame0_reconstructed_tokens.detach().cpu()] * num_frames

    prop_result = evaluate_sequence_label_propagation(
        sequence=sequence,
        tokens_stream=static_stream,
        root_dir=root_dir,
        device=device,
    )

    return prop_result, total_wire_bytes, wire_kbps


def evaluate_dense_rvq_control(
    sequence: str,
    raw_tokens_stream: List[torch.Tensor],
    num_quantizers: int,
    packer: Packer,
    root_dir: str = "data/DAVIS",
    fps: float = 25.0,
    device: Optional[torch.device] = None,
) -> Tuple[SequenceLabelPropagationResult, int, float]:
    """Evaluates Dense RVQ control (Gating OFF, all 256 tokens transmitted per frame).

    Args:
        sequence: DAVIS sequence name.
        raw_tokens_stream: Extracted raw ViT tokens [1, 256, 384] for all frames.
        num_quantizers: RVQ codebook stages to use (1 for Dense Q=1, 2 for Dense Q=2).
        packer: Shared Packer instance.
        root_dir: Path to DAVIS dataset.
        fps: Video framerate.
        device: Torch computation device.

    Returns:
        Tuple of (SequenceLabelPropagationResult, total_wire_bytes, wire_kbps).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rebuilder = Rebuilder(packer=packer, device=device)
    n_frames = len(raw_tokens_stream)
    mask_all = torch.ones(256, dtype=torch.bool, device=device)

    dense_tokens_stream: List[torch.Tensor] = []
    total_wire_bytes = 0

    for idx, curr_tokens in enumerate(raw_tokens_stream):
        tokens_dev = curr_tokens.to(device)
        is_keyframe = (idx == 0)

        p_out = packer(
            z_active=tokens_dev[0],
            mask=mask_all,
            frame_id=idx,
            patch_grid=(16, 16),
            is_keyframe=is_keyframe,
            num_quantizers=num_quantizers,
        )
        total_wire_bytes += p_out.packet.wire_bytes

        # Rebuilder decodes and updates cache
        _ = rebuilder(p_out.packet)
        dense_tokens_stream.append(rebuilder.token_cache.clone().detach().cpu())

    wire_kbps = compute_bitrate_kbps(total_wire_bytes, num_frames=n_frames, fps=fps)

    prop_result = evaluate_sequence_label_propagation(
        sequence=sequence,
        tokens_stream=dense_tokens_stream,
        root_dir=root_dir,
        device=device,
    )

    return prop_result, total_wire_bytes, wire_kbps


def evaluate_dense_rvq_frame_skip(
    sequence: str,
    raw_tokens_stream: List[torch.Tensor],
    packer: Packer,
    num_quantizers: int = 1,
    skip_interval: int = 2,
    root_dir: str = "data/DAVIS",
    fps: float = 25.0,
    device: Optional[torch.device] = None,
) -> Tuple[SequenceLabelPropagationResult, int, float]:
    """Evaluates Dense RVQ with frame-skipping (1/2 rate: every 2nd frame transmitted).

    On skipped frames, 0 bytes are transmitted and the server holds the previous token cache.

    Args:
        sequence: DAVIS sequence name.
        raw_tokens_stream: Extracted raw ViT tokens [1, 256, 384] for all frames.
        packer: Shared Packer instance.
        num_quantizers: Codebook stages (default: 1 for 1 byte/token).
        skip_interval: Transmit every N-th frame (default: 2 -> half rate).
        root_dir: Path to DAVIS dataset.
        fps: Video framerate.
        device: Torch computation device.

    Returns:
        Tuple of (SequenceLabelPropagationResult, total_wire_bytes, wire_kbps).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rebuilder = Rebuilder(packer=packer, device=device)
    n_frames = len(raw_tokens_stream)
    mask_all = torch.ones(256, dtype=torch.bool, device=device)

    tokens_stream: List[torch.Tensor] = []
    total_wire_bytes = 0

    for idx, curr_tokens in enumerate(raw_tokens_stream):
        tokens_dev = curr_tokens.to(device)
        is_keyframe = (idx == 0)
        should_transmit = (idx % skip_interval == 0)

        if should_transmit:
            p_out = packer(
                z_active=tokens_dev[0],
                mask=mask_all,
                frame_id=idx,
                patch_grid=(16, 16),
                is_keyframe=is_keyframe,
                num_quantizers=num_quantizers,
            )
            total_wire_bytes += p_out.packet.wire_bytes
            _ = rebuilder(p_out.packet)
        # On skipped frames: 0 wire bytes transmitted, cache is retained!

        tokens_stream.append(rebuilder.token_cache.clone().detach().cpu())

    wire_kbps = compute_bitrate_kbps(total_wire_bytes, num_frames=n_frames, fps=fps)

    prop_result = evaluate_sequence_label_propagation(
        sequence=sequence,
        tokens_stream=tokens_stream,
        root_dir=root_dir,
        device=device,
    )

    return prop_result, total_wire_bytes, wire_kbps

