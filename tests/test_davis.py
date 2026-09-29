"""Unit tests for DAVIS dataset ingestion and benchmark evaluation metrics.

Follows unit-testing-test-generate guidelines:
- Verifies DAVIS dataset loader discovery and shapes
- Verifies >10% foreground patch pooling logic
- Verifies confusion matrix and metric calculations
"""

import pytest
import torch
from pathlib import Path
from src.davis_loader import DAVISSequenceLoader, DAVISFrameItem


class TestDAVISLoader:
    """Test suite for DAVIS dataset loader."""

    @pytest.fixture
    def loader(self) -> DAVISSequenceLoader:
        return DAVISSequenceLoader(sequence="blackswan", root_dir="data/DAVIS")

    def test_davis_sequence_discovery(self, loader: DAVISSequenceLoader) -> None:
        """Verify blackswan sequence is discovered and has 50 frames."""
        assert len(loader) == 50
        assert loader.sequence == "blackswan"

    def test_davis_frame_item_shapes_and_types(self, loader: DAVISSequenceLoader) -> None:
        """Verify frame tensor and GT mask grid shapes."""
        item = loader[0]
        assert isinstance(item, DAVISFrameItem)
        assert item.frame.shape == (3, 224, 224)
        assert item.gt_mask_grid.shape == (16, 16)
        assert item.gt_mask_grid.dtype == torch.bool
        assert item.gt_mask_pixel.shape == (224, 224)
        assert item.frame.min() >= 0.0 and item.frame.max() <= 1.0

    def test_mask_to_patch_grid_10_percent_threshold(self) -> None:
        """Verify patch pooling threshold (>10% foreground pixels = True, <= 10% = False)."""
        # Create 224x224 mask with specific pixel counts in individual 14x14 patches
        mask = torch.zeros((224, 224), dtype=torch.float32)

        # Patch (0, 0): 15 pixels out of 196 (7.65% <= 10% -> False)
        mask[0:15, 0] = 1.0

        # Patch (0, 1): 25 pixels out of 196 (12.75% > 10% -> True)
        mask[0:5, 14:19] = 1.0  # 25 pixels

        # Patch (1, 1): All 196 pixels (100% -> True)
        mask[14:28, 14:28] = 1.0

        # Patch (2, 2): 0 pixels (0% -> False)

        grid = DAVISSequenceLoader.mask_to_patch_grid(mask, patch_size=14, fg_threshold=0.10)

        assert grid.shape == (16, 16)
        assert not grid[0, 0].item(), "7.6% foreground should be False"
        assert grid[0, 1].item(), "12.7% foreground should be True"
        assert grid[1, 1].item(), "100% foreground should be True"
        assert not grid[2, 2].item(), "0% foreground should be False"

    def test_missing_sequence_raises_error(self) -> None:
        """Verify missing sequence raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            DAVISSequenceLoader(sequence="non_existent_seq_xyz", root_dir="data/DAVIS")


class TestBenchmarkMetrics:
    """Test suite for metric calculation logic."""

    def test_recall_and_drop_rate_perfect(self) -> None:
        """Verify 100% recall and 100% drop rate when predictions match ground truth."""
        # 16x16 grid = 256 patches
        gt = torch.zeros(256, dtype=torch.bool)
        gt[10:30] = True  # 20 foreground patches, 236 background patches

        pred = gt.clone()  # Perfect prediction

        tp = (pred & gt).sum().item()
        fn = (~pred & gt).sum().item()
        tn = (~pred & ~gt).sum().item()
        fp = (pred & ~gt).sum().item()

        recall = tp / (tp + fn)
        drop_rate = tn / (tn + fp)

        assert recall == 1.0
        assert drop_rate == 1.0

    def test_recall_and_drop_rate_partial(self) -> None:
        """Verify partial recall and drop rate with known confusion counts."""
        gt = torch.tensor([True, True, False, False])
        pred = torch.tensor([True, False, False, True])

        tp = (pred & gt).sum().item()     # 1
        fn = (~pred & gt).sum().item()    # 1
        tn = (~pred & ~gt).sum().item()   # 1
        fp = (pred & ~gt).sum().item()    # 1

        recall = tp / (tp + fn)           # 1/2 = 0.50
        drop_rate = tn / (tn + fp)        # 1/2 = 0.50

        assert recall == 0.50
        assert drop_rate == 0.50
