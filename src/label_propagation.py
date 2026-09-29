"""Semi-Supervised Label Propagation & Official DAVIS 2016 Evaluation Metrics (J & F).

Standard computer vision task: Given ground-truth segmentation on Frame 0,
propagate object masks to subsequent frames using patch token affinity.
Computes official DAVIS metrics:
- Jaccard Index (J): Region similarity / Intersection over Union (IoU)
- Contour Accuracy (F): Boundary F-measure via Euclidean distance transform
- Mean (J & F): Overall benchmark score
"""

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Union

# Ensure repository root is on sys.path
root_dir_path = str(Path(__file__).resolve().parent.parent)
if root_dir_path not in sys.path:
    sys.path.insert(0, root_dir_path)

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.davis_loader import DAVISSequenceLoader


def compute_jaccard(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Computes Region Jaccard Index (IoU) between binary masks.

    Args:
        pred_mask: Binary prediction mask [H, W] (bool or uint8).
        gt_mask: Binary ground truth mask [H, W] (bool or uint8).

    Returns:
        Jaccard score in [0.0, 1.0].
    """
    m = (pred_mask > 0).astype(bool)
    g = (gt_mask > 0).astype(bool)

    intersection = np.logical_and(m, g).sum()
    union = np.logical_or(m, g).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(intersection / union)


def compute_boundary_f_measure(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    bound_th: float = 2.0,
) -> float:
    """Computes Contour Accuracy (Boundary F-measure) between binary masks.

    Matches official DAVIS benchmark definition (Perazzi et al.):
    - Extracts 1-pixel boundary contours using morphological gradient.
    - Computes Euclidean distance transform on inverted boundary maps.
    - Computes precision and recall within distance threshold `bound_th`.
    - Returns harmonic mean F.

    Args:
        pred_mask: Binary prediction mask [H, W].
        gt_mask: Binary ground truth mask [H, W].
        bound_th: Maximum boundary distance threshold in pixels (default: 2.0).

    Returns:
        F-measure in [0.0, 1.0].
    """
    m = (pred_mask > 0).astype(np.uint8)
    g = (gt_mask > 0).astype(np.uint8)

    # 3x3 structuring element for 1-pixel contour extraction
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bound_m = m - cv2.erode(m, kernel)
    bound_g = g - cv2.erode(g, kernel)

    num_m = int(np.sum(bound_m == 1))
    num_g = int(np.sum(bound_g == 1))

    # Handle empty contour edge cases
    if num_m == 0 and num_g == 0:
        return 1.0
    if num_m == 0 or num_g == 0:
        return 0.0

    # Euclidean distance transform from ground truth boundaries
    dist_g = cv2.distanceTransform(1 - bound_g, cv2.DIST_L2, 3)
    # Euclidean distance transform from predicted boundaries
    dist_m = cv2.distanceTransform(1 - bound_m, cv2.DIST_L2, 3)

    # Precision: fraction of predicted boundary pixels within threshold of GT boundary
    precision = float(np.sum(dist_g[bound_m == 1] <= bound_th) / num_m)

    # Recall: fraction of GT boundary pixels within threshold of predicted boundary
    recall = float(np.sum(dist_m[bound_g == 1] <= bound_th) / num_g)

    if precision + recall == 0.0:
        return 0.0

    f_measure = 2.0 * (precision * recall) / (precision + recall)
    return float(f_measure)


def compute_davis_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    bound_th: float = 2.0,
) -> Dict[str, float]:
    """Computes complete DAVIS benchmark metrics: J, F, and mean J&F."""
    j = compute_jaccard(pred_mask, gt_mask)
    f = compute_boundary_f_measure(pred_mask, gt_mask, bound_th=bound_th)
    return {
        "jaccard": j,
        "f_measure": f,
        "j_and_f": (j + f) / 2.0,
    }


@dataclass
class FrameLabelPropagationResult:
    """Metrics for a single frame of label propagation."""
    frame_idx: int
    jaccard: float
    f_measure: float
    j_and_f: float
    pred_mask: np.ndarray
    gt_mask: np.ndarray


@dataclass
class SequenceLabelPropagationResult:
    """Aggregated label propagation benchmark result for a video sequence."""
    sequence: str
    num_frames: int
    mean_jaccard: float
    mean_f_measure: float
    mean_j_and_f: float
    frame_results: List[FrameLabelPropagationResult]


class DINOv2LabelPropagator:
    """Nearest-neighbor label propagator operating on DINOv2 patch tokens."""

    def __init__(
        self,
        top_k: int = 5,
        temperature: float = 0.10,
        upsample_size: Tuple[int, int] = (224, 224),
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Initialize the label propagator.

        Args:
            top_k: Number of nearest memory patches for label voting (default: 5).
            temperature: Softmax scaling temperature for similarity weighting (default: 0.10).
            upsample_size: Output pixel resolution (default: (224, 224)).
            device: Torch computation device.
        """
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.upsample_size = upsample_size
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.memory_tokens: Optional[torch.Tensor] = None
        self.memory_labels: Optional[torch.Tensor] = None

    def initialize(
        self,
        frame0_tokens: torch.Tensor,
        frame0_mask: Union[torch.Tensor, np.ndarray],
    ) -> None:
        """Stores Keyframe (Frame 0) patch tokens and ground-truth mask.

        Args:
            frame0_tokens: ViT patch tokens [1, 256, D] or [256, D].
            frame0_mask: Ground-truth mask, either boolean grid [16, 16] or pixel mask [224, 224].
        """
        tok = frame0_tokens.to(self.device)
        if tok.ndim == 3:
            tok = tok.squeeze(0)  # [256, D]

        self.memory_tokens = F.normalize(tok, dim=-1)  # [256, D]

        if isinstance(frame0_mask, np.ndarray):
            m_t = torch.from_numpy(frame0_mask)
        else:
            m_t = frame0_mask

        m_t = m_t.to(self.device).float()
        if m_t.shape == (16, 16):
            labels = m_t.flatten()  # [256]
        elif m_t.shape == (224, 224):
            # Pool 224x224 to 16x16 patch grid
            m_pooled = F.adaptive_avg_pool2d(m_t.view(1, 1, 224, 224), (16, 16))
            labels = (m_pooled.flatten() > 0.10).float()
        else:
            labels = m_t.flatten().float()

        self.memory_labels = labels

    def propagate(
        self,
        current_tokens: torch.Tensor,
        update_memory: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Propagates segmentation labels to current frame tokens.

        Args:
            current_tokens: Latent tokens for frame t [1, 256, D] or [256, D].
            update_memory: Whether to add current predictions to memory bank.

        Returns:
            Tuple of:
            - pred_mask_pixel: High-resolution binary mask [224, 224] (uint8).
            - pred_mask_grid: 16x16 patch-level binary grid [16, 16] (bool).
        """
        if self.memory_tokens is None or self.memory_labels is None:
            raise RuntimeError("Propagator must be initialized with frame 0 first via initialize()")

        tok = current_tokens.to(self.device)
        if tok.ndim == 3:
            tok = tok.squeeze(0)  # [256, D]

        z_curr = F.normalize(tok, dim=-1)  # [256, D]

        # 1. Cosine similarity affinity against memory tokens: [256, N_mem]
        affinity = torch.matmul(z_curr, self.memory_tokens.T)

        # 2. Top-k nearest neighbor retrieval and softmax weighting
        k_val = min(self.top_k, self.memory_tokens.shape[0])
        topk_sim, topk_idx = affinity.topk(k_val, dim=-1)
        weights = F.softmax(topk_sim / self.temperature, dim=-1)  # [256, K]

        # 3. Vote on patch labels
        topk_labels = self.memory_labels[topk_idx]  # [256, K]
        pred_prob_grid = (weights * topk_labels).sum(dim=-1).view(1, 1, 16, 16)  # [1, 1, 16, 16]

        # 4. Patch grid binary prediction
        pred_grid_bool = (pred_prob_grid.squeeze().cpu() > 0.5).numpy()

        # 5. Bilinear upsampling to pixel mask [224, 224]
        h_up, w_up = self.upsample_size
        pred_prob_pixel = F.interpolate(
            pred_prob_grid, size=(h_up, w_up), mode="bilinear", align_corners=False
        )
        pred_pixel_u8 = (pred_prob_pixel.squeeze().cpu() > 0.5).numpy().astype(np.uint8)

        # Optional memory update for temporal continuity
        if update_memory:
            # Maintain keyframe (anchor) + latest frame
            key_tokens = self.memory_tokens[:256]
            key_labels = self.memory_labels[:256]
            curr_pred_labels = (pred_prob_grid.flatten() > 0.5).float()
            self.memory_tokens = torch.cat([key_tokens, z_curr], dim=0)
            self.memory_labels = torch.cat([key_labels, curr_pred_labels], dim=0)

        return pred_pixel_u8, pred_grid_bool

    def reset(self) -> None:
        """Clears memory bank."""
        self.memory_tokens = None
        self.memory_labels = None


def evaluate_sequence_label_propagation(
    sequence: str,
    tokens_stream: List[torch.Tensor],
    root_dir: str = "data/DAVIS",
    bound_th: float = 2.0,
    top_k: int = 5,
    device: Optional[Union[str, torch.device]] = None,
) -> SequenceLabelPropagationResult:
    """Evaluates nearest-neighbor label propagation on a sequence of tokens.

    Args:
        sequence: DAVIS sequence name (e.g. 'blackswan').
        tokens_stream: List of DINOv2 tokens for all frames (idx 0 to T-1).
        root_dir: DAVIS dataset root path.
        bound_th: Boundary distance threshold (default: 2.0 px).
        top_k: Nearest neighbors for label propagation.
        device: Computation device.

    Returns:
        SequenceLabelPropagationResult with mean J, mean F, and mean J&F.
    """
    loader = DAVISSequenceLoader(sequence=sequence, root_dir=root_dir)
    n_frames = min(len(loader), len(tokens_stream))
    if n_frames < 2:
        raise ValueError(f"Sequence {sequence} requires at least 2 frames for evaluation.")

    propagator = DINOv2LabelPropagator(top_k=top_k, device=device)

    # Initialize with Frame 0
    frame0_tokens = tokens_stream[0]
    frame0_mask = loader[0].gt_mask_pixel.numpy()
    propagator.initialize(frame0_tokens=frame0_tokens, frame0_mask=frame0_mask)

    frame_results: List[FrameLabelPropagationResult] = []
    j_list: List[float] = []
    f_list: List[float] = []

    for t in range(1, n_frames):
        tokens_t = tokens_stream[t]
        gt_pixel = loader[t].gt_mask_pixel.numpy().astype(np.uint8)

        pred_pixel, _ = propagator.propagate(tokens_t, update_memory=False)
        metrics = compute_davis_metrics(pred_pixel, gt_pixel, bound_th=bound_th)

        j_list.append(metrics["jaccard"])
        f_list.append(metrics["f_measure"])

        frame_results.append(
            FrameLabelPropagationResult(
                frame_idx=t,
                jaccard=metrics["jaccard"],
                f_measure=metrics["f_measure"],
                j_and_f=metrics["j_and_f"],
                pred_mask=pred_pixel,
                gt_mask=gt_pixel,
            )
        )

    mean_j = float(np.mean(j_list)) if j_list else 0.0
    mean_f = float(np.mean(f_list)) if f_list else 0.0
    mean_jf = (mean_j + mean_f) / 2.0

    return SequenceLabelPropagationResult(
        sequence=sequence,
        num_frames=len(frame_results),
        mean_jaccard=mean_j,
        mean_f_measure=mean_f,
        mean_j_and_f=mean_jf,
        frame_results=frame_results,
    )
