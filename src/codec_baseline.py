"""Honest Video Codec Baseline (H.264 / libx264 at Matched Bitrates).

Encodes raw video frames using standard H.264 (libx264) constrained to the exact
wire bitrate achieved by the Differential Token Streamer.
Decodes compressed frames, extracts DINOv2 tokens, and runs identical DAVIS
label propagation to measure visual perception fidelity under traditional video compression.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional, Tuple, Union

# Ensure repository root is on sys.path
root_dir_path = str(Path(__file__).resolve().parent.parent)
if root_dir_path not in sys.path:
    sys.path.insert(0, root_dir_path)

import cv2
import imageio_ffmpeg
import numpy as np
import torch

from src.davis_loader import DAVISSequenceLoader
from src.slicer import DINOv2Slicer
from src.label_propagation import (
    evaluate_sequence_label_propagation,
    SequenceLabelPropagationResult,
)


@dataclass
class CodecBaselineResult:
    """Benchmark results for H.264 video compression at matched bitrate."""
    sequence: str
    target_kbps: float
    achieved_kbps: float
    target_wire_bytes: int
    h264_file_bytes: int
    h264_jaccard: float
    h264_f_measure: float
    h264_j_and_f: float
    propagation_result: SequenceLabelPropagationResult


def compute_bitrate_kbps(total_wire_bytes: int, num_frames: int, fps: float = 25.0) -> float:
    """Computes equivalent video streaming bitrate in kbps (kilobits per second)."""
    if num_frames == 0 or fps <= 0:
        return 0.0
    duration_sec = num_frames / fps
    total_bits = total_wire_bytes * 8
    kbps = total_bits / (duration_sec * 1000.0)
    return float(kbps)


def encode_frames_to_h264(
    frames_rgb: List[np.ndarray],
    target_kbps: float,
    output_mp4_path: str,
    fps: float = 25.0,
) -> int:
    """Encodes a sequence of RGB frames [H, W, 3] to H.264 using FFmpeg libx264.

    Args:
        frames_rgb: List of RGB uint8 numpy arrays of shape (H, W, 3).
        target_kbps: Target bitrate in kbps.
        output_mp4_path: Filepath to write the MP4 container.
        fps: Frame rate (default: 25.0).

    Returns:
        Encoded file size in bytes.
    """
    if not frames_rgb:
        raise ValueError("frames_rgb list cannot be empty")

    h, w, c = frames_rgb[0].shape
    assert c == 3, f"Expected 3 color channels, got {c}"

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    # Calculate bitrates and buffer sizes (clamp target_kbps to minimum 10 kbps)
    bitrate_int = max(10, int(round(target_kbps)))
    bufsize_int = max(20, bitrate_int * 2)

    cmd = [
        ffmpeg_exe,
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-b:v", f"{bitrate_int}k",
        "-maxrate", f"{bitrate_int}k",
        "-bufsize", f"{bufsize_int}k",
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        output_mp4_path,
    ]

    # Stream frames into FFmpeg stdin
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    raw_bytes = b"".join(frame.tobytes() for frame in frames_rgb)
    stdout, stderr = process.communicate(input=raw_bytes)

    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg encoding failed with code {process.returncode}:\n{stderr.decode('utf-8', errors='ignore')}")

    out_file = Path(output_mp4_path)
    if not out_file.exists():
        raise FileNotFoundError(f"Encoded H.264 file was not created: {output_mp4_path}")

    return out_file.stat().st_size


def decode_h264_to_frames(mp4_path: str) -> List[np.ndarray]:
    """Decodes an H.264 MP4 file back into a list of RGB uint8 numpy frames."""
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file for decoding: {mp4_path}")

    frames: List[np.ndarray] = []
    while True:
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    return frames


def evaluate_h264_baseline_on_sequence(
    sequence: str,
    target_wire_bytes: int,
    root_dir: str = "data/DAVIS",
    fps: float = 25.0,
    slicer: Optional[DINOv2Slicer] = None,
    device: Optional[torch.device] = None,
) -> CodecBaselineResult:
    """Evaluates H.264 video compression at matched bitrate on a DAVIS sequence.

    Steps:
    1. Loads raw RGB frames from sequence.
    2. Computes target bitrate in kbps matching `target_wire_bytes`.
    3. Encodes frames to H.264 (libx264).
    4. Decodes compressed frames back to RGB.
    5. Extracts DINOv2 patch tokens from compressed frames.
    6. Runs semi-supervised label propagation and evaluates J&F.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if slicer is None:
        slicer = DINOv2Slicer(device=device)

    loader = DAVISSequenceLoader(sequence=sequence, root_dir=root_dir)
    n_frames = len(loader)

    # Ingest original raw RGB frames
    raw_rgb_frames = [loader[i].raw_image.resize((224, 224)) for i in range(n_frames)]
    np_rgb_frames = [np.array(img, dtype=np.uint8) for img in raw_rgb_frames]

    # Calculate target bitrate
    target_kbps = compute_bitrate_kbps(target_wire_bytes, num_frames=n_frames, fps=fps)

    # Encode with FFmpeg libx264 in a temporary MP4 file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_mp4:
        tmp_path = tmp_mp4.name

    try:
        encoded_size = encode_frames_to_h264(
            frames_rgb=np_rgb_frames,
            target_kbps=target_kbps,
            output_mp4_path=tmp_path,
            fps=fps,
        )
        achieved_kbps = compute_bitrate_kbps(encoded_size, num_frames=n_frames, fps=fps)

        # Decode compressed video back into RGB frames
        decoded_rgb_frames = decode_h264_to_frames(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # Extract DINOv2 tokens from decoded H.264 frames
    h264_tokens: List[torch.Tensor] = []
    with torch.no_grad():
        for frame_rgb in decoded_rgb_frames:
            # Normalize to [0, 1] tensor [1, 3, 224, 224]
            t_frame = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            out = slicer(t_frame.to(device))
            h264_tokens.append(out.tokens.cpu())

    # Run label propagation on H.264 tokens
    prop_result = evaluate_sequence_label_propagation(
        sequence=sequence,
        tokens_stream=h264_tokens,
        root_dir=root_dir,
        device=device,
    )

    return CodecBaselineResult(
        sequence=sequence,
        target_kbps=target_kbps,
        achieved_kbps=achieved_kbps,
        target_wire_bytes=target_wire_bytes,
        h264_file_bytes=encoded_size,
        h264_jaccard=prop_result.mean_jaccard,
        h264_f_measure=prop_result.mean_f_measure,
        h264_j_and_f=prop_result.mean_j_and_f,
        propagation_result=prop_result,
    )
