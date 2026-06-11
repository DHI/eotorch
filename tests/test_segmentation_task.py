import torch
from torch import nn

import eotorch.data.tasks as tasks_module
from eotorch.data.tasks import PatchSegmentationTask


class DummySegModel(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def test_patch_segmentation_task_supports_binary_metrics(monkeypatch):
    """PatchSegmentationTask should initialize metrics when num_classes is 1."""
    monkeypatch.setitem(tasks_module.CLF_MODEL_MAPPING, "dummyseg", DummySegModel)

    task = PatchSegmentationTask(
        num_classes=1,
        in_channels=4,
        model="dummyseg",
        loss="bce",
        ignore_index=0,
    )

    metric_names = set(task.train_metrics.keys())
    assert "train/mIoU" in metric_names
    assert "train/F1_Score" in metric_names
    assert "train/Accuracy" in metric_names
    assert "train/Pixel_Accuracy" in metric_names


def test_patch_segmentation_task_validation_accepts_3d_binary_targets(monkeypatch):
    """validation_step should handle [N, H, W] targets for BCE binary segmentation."""
    monkeypatch.setitem(tasks_module.CLF_MODEL_MAPPING, "dummyseg", DummySegModel)

    task = PatchSegmentationTask(
        num_classes=1,
        in_channels=4,
        model="dummyseg",
        loss="bce",
        task="binary",
        ignore_index=0,
    )

    x = torch.randn(2, 4, 16, 16)
    y = torch.randint(0, 2, (2, 16, 16), dtype=torch.long)

    task.validation_step((x, y))
