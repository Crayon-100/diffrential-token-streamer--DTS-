"""DAVIS 2016 Video Sequence Dataset Loader

Provides dataset ingestion for real-world video sequences from the DAVIS 2016 dataset.
Resizes frames to 224x224 and pools ground-truth segmentation masks into 16x16 patch grids.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union, List
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


@dataclass(frozen=True)
class DAVISFrameItem:
    """Container for a single processed DAVIS frame."""
    frame: torch.Tensor             # Standardized RGB tensor [3, 224, 224] in [0, 1]
    gt_mask_grid: torch.Tensor      # Boolean ground-truth grid [16, 16] (True = dynamic object patch)
    gt_mask_pixel: torch.Tensor     # Binary pixel mask [224, 224]
    frame_name: str                 # e.g. '00000.jpg'
    frame_idx: int                  # 0, 1, 2, ...
    raw_image: Image.Image          # Original PIL Image for visualization


class DAVISSequenceLoader:
    """Iterator and loader for a DAVIS 2016 video sequence.

    Path structure:
        - Frames: <root_dir>/JPEGImages/480p/<sequence>/*.jpg
        - Annotations: <root_dir>/Annotations/480p/<sequence>/*.png
    """

    def __init__(
        self,
        sequence: str = "blackswan",
        root_dir: Union[str, Path] = "data/DAVIS",
        image_size: Tuple[int, int] = (224, 224),
        patch_size: int = 14,
        fg_threshold: float = 0.10,
    ) -> None:
        """Initialize the DAVIS sequence loader.

        Args:
            sequence: Sequence name (e.g. 'blackswan').
            root_dir: Root path to DAVIS dataset.
            image_size: Target (height, width) for frames (default: (224, 224)).
            patch_size: ViT patch size (default: 14).
            fg_threshold: Minimum foreground pixel ratio to classify a patch as dynamic object (default: 0.10).
        """
        self.sequence = sequence
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.patch_size = patch_size
        self.fg_threshold = fg_threshold

        self.jpeg_dir = self.root_dir / "JPEGImages" / "480p" / sequence
        self.anno_dir = self.root_dir / "Annotations" / "480p" / sequence

        if not self.jpeg_dir.exists():
            raise FileNotFoundError(f"JPEG directory not found: {self.jpeg_dir}")
        if not self.anno_dir.exists():
            raise FileNotFoundError(f"Annotations directory not found: {self.anno_dir}")

        # Find matching frame files sorted alphabetically
        self.frame_files = sorted(list(self.jpeg_dir.glob("*.jpg")))
        if not self.frame_files:
            raise ValueError(f"No .jpg frames found in {self.jpeg_dir}")

        # Verify annotations exist for each frame
        self.anno_files: List[Optional[Path]] = []
        for f in self.frame_files:
            anno_path = self.anno_dir / f"{f.stem}.png"
            self.anno_files.append(anno_path if anno_path.exists() else None)

    def __len__(self) -> int:
        return len(self.frame_files)

    @staticmethod
    def mask_to_patch_grid(
        mask_tensor: torch.Tensor,
        patch_size: int = 14,
        fg_threshold: float = 0.10,
    ) -> torch.Tensor:
        """Converts a binary pixel mask [H, W] into a patch grid [H_p, W_p] boolean tensor.

        A patch is marked True if > fg_threshold (e.g. 10%) of its pixels belong to foreground.
        Uses 2D average pooling for efficient and exact area-fraction computation.
        """
        # mask_tensor: [H, W] with 1.0 = fg, 0.0 = bg
        x = mask_tensor.unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]

        # Calculate foreground fraction per patch using average pooling
        patch_fg_fraction = F.avg_pool2d(
            x, kernel_size=patch_size, stride=patch_size
        ).squeeze()  # [H_p, W_p]

        # Apply threshold (> 10% foreground pixels)
        return patch_fg_fraction > fg_threshold

    def __getitem__(self, idx: int) -> DAVISFrameItem:
        frame_path = self.frame_files[idx]
        anno_path = self.anno_files[idx]

        # Load RGB Frame
        raw_img = Image.open(frame_path).convert("RGB")
        resized_img = raw_img.resize(self.image_size, Image.BILINEAR)

        # Convert to PyTorch float tensor [3, 224, 224] in [0, 1]
        frame_tensor = TF.to_tensor(resized_img)

        # Load Annotation Mask
        if anno_path is not None and anno_path.exists():
            raw_mask = Image.open(anno_path)
            resized_mask = raw_mask.resize(self.image_size, Image.NEAREST)
            mask_np = np.array(resized_mask)
            # Binary mask: foreground > 0
            binary_mask = torch.from_numpy(mask_np > 0).float()
            gt_grid = self.mask_to_patch_grid(
                binary_mask, patch_size=self.patch_size, fg_threshold=self.fg_threshold
            )
        else:
            # Fallback if no annotation file
            binary_mask = torch.zeros(self.image_size, dtype=torch.float32)
            h_p = self.image_size[0] // self.patch_size
            w_p = self.image_size[1] // self.patch_size
            gt_grid = torch.zeros((h_p, w_p), dtype=torch.bool)

        return DAVISFrameItem(
            frame=frame_tensor,
            gt_mask_grid=gt_grid,
            gt_mask_pixel=binary_mask.bool(),
            frame_name=frame_path.name,
            frame_idx=idx,
            raw_image=raw_img,
        )

    def __iter__(self) -> Iterator[DAVISFrameItem]:
        for i in range(len(self)):
            yield self[i]
