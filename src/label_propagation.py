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
    bound_th: Optional[float] = None,
) -> float:
    """Computes Contour Accuracy (Boundary F-measure) between binary masks.

    Matches official DAVIS benchmark definition (Perazzi et al.):
    - Extracts 1-pixel boundary contours using morphological gradient.
    - Computes Euclidean distance transform on inverted boundary maps.
    - Computes precision and recall within distance threshold `bound_th` (default: 0.008 * diagonal).
    - Returns harmonic mean F.

    Args:
        pred_mask: Binary prediction mask [H, W].
        gt_mask: Binary ground truth mask [H, W].
        bound_th: Boundary distance threshold. If None, uses official 0.008 * image diagonal.

    Returns:
        F-measure in [0.0, 1.0].
    """
    m = (pred_mask > 0).astype(np.uint8)
    g = (gt_mask > 0).astype(np.uint8)

    if bound_th is None:
        h, w = m.shape
        diag = np.sqrt(float(h**2 + w**2))
        bound_th = 0.008 * diag

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
    bound_th: Optional[float] = None,
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
    """Sensitive nearest-neighbor label propagator operating on DINOv2 patch tokens.

    Enhancements for Gate 2.5:
    - Temporal Memory Context Queue: Frame 0 anchor + last M=3 recent frame predictions.
    - Spatial Locality Prior: Restricts cosine similarity matching to a localized window
      (R=4 patches) around the previous frame's foreground mask location.
    - Continuous Soft Labels: Ingests ground-truth mask pooled to soft fraction prior to propagation.
    - Soft Probability Upsampling: Bilinearly upsamples similarity logits to native 480p resolution
      before thresholding.
    """

    def __init__(
        self,
        top_k: int = 5,
        temperature: float = 0.08,
        memory_queue_size: int = 3,
        locality_radius: float = 4.0,
        mask_threshold: float = 0.35,
        upsample_size: Optional[Tuple[int, int]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Initialize the sensitive label propagator.

        Args:
            top_k: Number of nearest memory patches for label voting (default: 5).
            temperature: Softmax scaling temperature for similarity weighting (default: 0.08).
            memory_queue_size: Number of recent frame predictions to retain (default: M=3).
            locality_radius: Spatial search radius in patches around previous mask (default: R=4.0).
            mask_threshold: Binary probability threshold after soft upsampling (default: 0.35).
            upsample_size: Target native output resolution (default: None, inferred from mask).
            device: Torch computation device.
        """
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.memory_queue_size = int(memory_queue_size)
        self.locality_radius = float(locality_radius)
        self.mask_threshold = float(mask_threshold)
        self.upsample_size = upsample_size
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

        # Anchor Keyframe (Frame 0)
        self.anchor_tokens: Optional[torch.Tensor] = None
        self.anchor_labels: Optional[torch.Tensor] = None
        self.mask_size: Tuple[int, int] = (224, 224)

        # Temporal Context Queue (recent frames t-1, t-2, ...)
        self.queue_tokens: List[torch.Tensor] = []
        self.queue_labels: List[torch.Tensor] = []
        self.prev_mask: Optional[torch.Tensor] = None

        # 16x16 patch spatial coordinates
        grid_y, grid_x = torch.meshgrid(
            torch.arange(16, device=self.device), torch.arange(16, device=self.device), indexing="ij"
        )
        self.coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=-1).float()  # [256, 2]

    @property
    def memory_tokens(self) -> Optional[torch.Tensor]:
        """Backwards compatibility alias for Frame 0 anchor tokens."""
        return self.anchor_tokens

    @property
    def memory_labels(self) -> Optional[torch.Tensor]:
        """Backwards compatibility alias for Frame 0 anchor labels."""
        return self.anchor_labels

    def initialize(
        self,
        frame0_tokens: torch.Tensor,
        frame0_mask: Union[torch.Tensor, np.ndarray],
    ) -> None:
        """Stores Keyframe (Frame 0) patch tokens and continuous ground-truth mask.

        Args:
            frame0_tokens: ViT patch tokens [1, 256, D] or [256, D].
            frame0_mask: Ground-truth mask, boolean grid [16, 16] or pixel mask [H, W].
        """
        tok = frame0_tokens.to(self.device)
        if tok.ndim == 3:
            tok = tok.squeeze(0)  # [256, D]

        self.anchor_tokens = F.normalize(tok, dim=-1)  # [256, D]

        if isinstance(frame0_mask, np.ndarray):
            m_t = torch.from_numpy(frame0_mask)
        else:
            m_t = frame0_mask

        m_t = m_t.to(self.device).float()
        if m_t.ndim == 2:
            self.mask_size = (int(m_t.shape[0]), int(m_t.shape[1]))
            if m_t.shape != (16, 16):
                # Soft continuous pooling from high-resolution mask: [1, 1, H, W] -> [16, 16]
                m_pooled = F.adaptive_avg_pool2d(m_t.unsqueeze(0).unsqueeze(0), (16, 16))
                labels = m_pooled.squeeze().flatten()  # [256] in [0, 1]
            else:
                labels = m_t.flatten()
        else:
            labels = m_t.flatten().float()

        self.anchor_labels = labels
        self.queue_tokens = []
        self.queue_labels = []
        self.prev_mask = (self.anchor_labels > 0.20)

    def propagate(
        self,
        current_tokens: torch.Tensor,
        upsample_size: Optional[Tuple[int, int]] = None,
        update_memory: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
        """Propagates segmentation labels to current frame tokens.

        Args:
            current_tokens: Latent tokens for frame t [1, 256, D] or [256, D].
            upsample_size: Output resolution (H, W). Defaults to self.upsample_size.
            update_memory: Whether to enqueue current frame into memory queue (default: True).

        Returns:
            Tuple of:
            - pred_mask_pixel: High-resolution binary mask [H, W] (uint8).
            - pred_mask_grid: 16x16 patch-level binary grid [16, 16] (bool).
            - pred_prob_grid: 16x16 soft probability grid tensor [16, 16].
        """
        if self.anchor_tokens is None or self.anchor_labels is None:
            raise RuntimeError("Propagator must be initialized with frame 0 first via initialize()")

        tok = current_tokens.to(self.device)
        if tok.ndim == 3:
            tok = tok.squeeze(0)  # [256, D]

        z_curr = F.normalize(tok, dim=-1)  # [256, D]

        # 1. Assemble memory bank: Frame 0 Anchor + Recent Context Queue
        all_tokens = [self.anchor_tokens] + self.queue_tokens
        all_labels = [self.anchor_labels] + self.queue_labels
        mem_tokens = torch.cat(all_tokens, dim=0)  # [N_mem, D]
        mem_labels = torch.cat(all_labels, dim=0)  # [N_mem]

        # 2. Cosine similarity affinity: [256, N_mem]
        affinity = torch.matmul(z_curr, mem_tokens.T)

        # 3. Spatial Locality Prior around previous frame's mask location
        if self.locality_radius is not None and self.prev_mask is not None and self.prev_mask.sum() > 0:
            fg_coords = self.coords[self.prev_mask]  # [N_fg, 2]
            dists = torch.cdist(self.coords, fg_coords).min(dim=-1).values  # [256]
            valid_patches = dists <= self.locality_radius
        else:
            valid_patches = torch.ones(256, dtype=torch.bool, device=self.device)

        # 4. Top-k nearest neighbor retrieval and softmax weighting
        k_val = min(self.top_k, mem_tokens.shape[0])
        topk_sim, topk_idx = affinity.topk(k_val, dim=-1)
        weights = F.softmax(topk_sim / self.temperature, dim=-1)  # [256, K]

        # 5. Vote on patch labels with soft weights
        topk_labels = mem_labels[topk_idx]  # [256, K]
        pred_prob_flat = (weights * topk_labels).sum(dim=-1)  # [256]
        pred_prob_gated = torch.where(valid_patches, pred_prob_flat, torch.zeros_like(pred_prob_flat))
        pred_prob_grid = pred_prob_gated.view(1, 1, 16, 16)

        # 6. Soft Probability Bilinear Upsampling to native resolution before thresholding
        target_size = upsample_size if upsample_size is not None else (
            self.upsample_size if self.upsample_size is not None else getattr(self, "mask_size", (224, 224))
        )
        pred_prob_pixel = F.interpolate(
            pred_prob_grid, size=target_size, mode="bilinear", align_corners=False
        )
        pred_pixel_u8 = (pred_prob_pixel.squeeze().cpu() > self.mask_threshold).numpy().astype(np.uint8)
        pred_grid_bool = (pred_prob_gated.view(16, 16).cpu() > self.mask_threshold).numpy()

        # 7. Update temporal memory queue
        if update_memory and self.memory_queue_size > 0:
            if len(self.queue_tokens) >= self.memory_queue_size:
                self.queue_tokens.pop(0)
                self.queue_labels.pop(0)
            self.queue_tokens.append(z_curr)
            self.queue_labels.append(pred_prob_gated.detach())
            self.prev_mask = (pred_prob_gated > 0.20)

        return pred_pixel_u8, pred_grid_bool

    def reset(self) -> None:
        """Clears memory bank and queue."""
        self.anchor_tokens = None
        self.anchor_labels = None
        self.queue_tokens.clear()
        self.queue_labels.clear()
        self.prev_mask = None


def evaluate_sequence_label_propagation(
    sequence: str,
    tokens_stream: List[torch.Tensor],
    root_dir: str = "data/DAVIS",
    bound_th: Optional[float] = None,
    top_k: int = 5,
    temperature: float = 0.08,
    memory_queue_size: int = 3,
    locality_radius: float = 4.0,
    mask_threshold: float = 0.35,
    device: Optional[Union[str, torch.device]] = None,
) -> SequenceLabelPropagationResult:
    """Evaluates nearest-neighbor label propagation on a sequence of tokens at native resolution.

    Args:
        sequence: DAVIS sequence name (e.g. 'blackswan').
        tokens_stream: List of DINOv2 tokens for all frames (idx 0 to T-1).
        root_dir: DAVIS dataset root path.
        bound_th: Boundary distance threshold. If None, uses official 0.008 * diagonal.
        top_k: Nearest neighbors for label propagation (default: 5).
        temperature: Softmax scaling temperature (default: 0.08).
        memory_queue_size: Temporal context queue depth (default: M=3).
        locality_radius: Spatial search radius in patches (default: R=4.0).
        mask_threshold: Probability threshold for binary prediction (default: 0.35).
        device: Computation device.

    Returns:
        SequenceLabelPropagationResult with mean J, mean F, and mean J&F.
    """
    loader = DAVISSequenceLoader(sequence=sequence, root_dir=root_dir)
    n_frames = min(len(loader), len(tokens_stream))
    if n_frames < 2:
        raise ValueError(f"Sequence {sequence} requires at least 2 frames for evaluation.")

    item0 = loader[0]
    native_size = item0.native_size
    eff_bound_th = bound_th
    if eff_bound_th is None:
        diag = np.sqrt(float(native_size[0] ** 2 + native_size[1] ** 2))
        eff_bound_th = 0.008 * diag

    propagator = DINOv2LabelPropagator(
        top_k=top_k,
        temperature=temperature,
        memory_queue_size=memory_queue_size,
        locality_radius=locality_radius,
        mask_threshold=mask_threshold,
        upsample_size=native_size,
        device=device,
    )

    # Initialize with Frame 0 native annotation
    frame0_tokens = tokens_stream[0]
    frame0_mask = item0.gt_mask_native.float().numpy() if item0.gt_mask_native is not None else item0.gt_mask_pixel.numpy()
    propagator.initialize(frame0_tokens=frame0_tokens, frame0_mask=frame0_mask)

    frame_results: List[FrameLabelPropagationResult] = []
    j_list: List[float] = []
    f_list: List[float] = []

    for t in range(1, n_frames):
        tokens_t = tokens_stream[t]
        item_t = loader[t]
        gt_pixel = item_t.gt_mask_native.numpy().astype(np.uint8) if item_t.gt_mask_native is not None else item_t.gt_mask_pixel.numpy().astype(np.uint8)

        pred_pixel, _ = propagator.propagate(
            tokens_t,
            upsample_size=native_size,
            update_memory=True,
        )
        metrics = compute_davis_metrics(pred_pixel, gt_pixel, bound_th=eff_bound_th)

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

