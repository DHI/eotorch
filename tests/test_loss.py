import pytest
import torch
from torch import nn

import eotorch.data.tasks as tasks_module
from eotorch.data.tasks import PatchSegmentationTask
from eotorch.models.loss import (
    BCEDiceBoundaryLoss,
    BCEDiceLoss,
    BoundaryDistanceLoss,
    GaussianNLL,
    MultiClassCEDiceBoundaryDistanceLoss,
    MultiClassCEDiceBoundaryLoss,
)


class DummySegModel(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _vertical_boundary_target(height: int = 40, width: int = 64, boundary_x: int = 32) -> torch.Tensor:
    """[1, H, W] long target: columns < boundary_x are class 0, the rest class 1."""
    target = torch.zeros((1, height, width), dtype=torch.long)
    target[:, :, boundary_x:] = 1
    return target


def _confident_logits_for_boundary(
    height: int = 40, width: int = 64, boundary_x: int = 32, magnitude: float = 10.0
) -> torch.Tensor:
    """[1, 2, H, W] logits confidently predicting a vertical boundary at boundary_x."""
    logits = torch.full((1, 2, height, width), -magnitude)
    logits[:, 0, :, :boundary_x] = magnitude
    logits[:, 1, :, boundary_x:] = magnitude
    return logits.clone().requires_grad_(True)


# ---------------------------------------------------------------------------
# GaussianNLL
# ---------------------------------------------------------------------------


def test_gaussian_nll_mean_reduction_is_finite_and_differentiable():
    mean = torch.zeros(4, requires_grad=True)
    variance = torch.ones(4, requires_grad=True)
    target = torch.tensor([0.5, -0.5, 1.0, -1.0])

    loss_fn = GaussianNLL(reduction="mean")
    loss = loss_fn(mean, variance, target)

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()
    assert mean.grad is not None
    assert variance.grad is not None
    assert torch.isfinite(mean.grad).all()
    assert torch.isfinite(variance.grad).all()


def test_gaussian_nll_none_reduction_preserves_shape():
    mean = torch.zeros(4)
    variance = torch.ones(4)
    target = torch.tensor([0.5, -0.5, 1.0, -1.0])

    loss_fn = GaussianNLL(reduction="none")
    loss = loss_fn(mean, variance, target)

    assert loss.shape == mean.shape
    assert torch.isfinite(loss).all()


# ---------------------------------------------------------------------------
# BCEDiceLoss
# ---------------------------------------------------------------------------


def test_bce_dice_loss_forward_backward():
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    targets = torch.randint(0, 2, (2, 16, 16), dtype=torch.long)

    loss_fn = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5, pos_weight=5.0)
    loss = loss_fn(logits, targets)

    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_bce_dice_loss_rejects_weights_not_summing_to_one():
    with pytest.raises(AssertionError):
        BCEDiceLoss(bce_weight=0.5, dice_weight=0.6)


# ---------------------------------------------------------------------------
# BCEDiceBoundaryLoss
# ---------------------------------------------------------------------------


def test_bce_dice_boundary_loss_forward_backward():
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    targets = torch.randint(0, 2, (2, 16, 16), dtype=torch.long)

    loss_fn = BCEDiceBoundaryLoss(boundary_weight=0.3, region_weight=0.7)
    loss = loss_fn(logits, targets)

    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_bce_dice_boundary_loss_rejects_weights_not_summing_to_one():
    with pytest.raises(AssertionError):
        BCEDiceBoundaryLoss(boundary_weight=0.3, region_weight=0.5)


# ---------------------------------------------------------------------------
# MultiClassCEDiceBoundaryLoss (Sobel-gradient boundary term)
# ---------------------------------------------------------------------------


def test_multiclass_ce_dice_boundary_loss_forward_backward():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 16, 16, requires_grad=True)
    targets = torch.randint(0, 3, (2, 16, 16), dtype=torch.long)

    loss_fn = MultiClassCEDiceBoundaryLoss(num_classes=3, ignore_index=0)
    loss = loss_fn(logits, targets)

    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_multiclass_ce_dice_boundary_loss_requires_multiple_classes():
    with pytest.raises(AssertionError):
        MultiClassCEDiceBoundaryLoss(num_classes=1)


def test_multiclass_ce_dice_boundary_loss_rejects_weights_not_summing_to_one():
    with pytest.raises(AssertionError):
        MultiClassCEDiceBoundaryLoss(num_classes=3, boundary_weight=0.3, region_weight=0.5)


# ---------------------------------------------------------------------------
# BoundaryDistanceLoss
# ---------------------------------------------------------------------------


def test_boundary_distance_loss_forward_backward():
    """Forward pass returns a finite scalar and gradients flow back to the logits."""
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 32, 32, requires_grad=True)
    targets = _vertical_boundary_target(height=32, width=32, boundary_x=16).repeat(2, 1, 1)
    targets[1, :, 20:] = 2  # give the second sample a 3-way boundary

    loss_fn = BoundaryDistanceLoss(num_classes=3)
    loss = loss_fn(logits, targets)

    assert torch.isfinite(loss)
    assert loss.ndim == 0

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_boundary_distance_loss_is_monotonic_in_boundary_offset():
    """The core property this loss exists for: predicting the boundary further from
    the true position must cost more than predicting it just slightly off."""
    true_boundary_x = 32
    target = _vertical_boundary_target(boundary_x=true_boundary_x)

    small_offset_logits = _confident_logits_for_boundary(boundary_x=true_boundary_x + 2)
    large_offset_logits = _confident_logits_for_boundary(boundary_x=true_boundary_x + 20)

    loss_fn = BoundaryDistanceLoss(num_classes=2, max_distance=24.0)

    small_offset_loss = loss_fn(small_offset_logits, target)
    large_offset_loss = loss_fn(large_offset_logits, target)

    assert large_offset_loss > small_offset_loss


def test_boundary_distance_loss_requires_multiple_classes():
    with pytest.raises(AssertionError):
        BoundaryDistanceLoss(num_classes=1)


# ---------------------------------------------------------------------------
# MultiClassCEDiceBoundaryDistanceLoss
# ---------------------------------------------------------------------------


def test_multiclass_ce_dice_boundary_distance_loss_forward_backward():
    """Combined loss returns a finite scalar and gradients flow back to the logits."""
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 32, 32, requires_grad=True)
    targets = _vertical_boundary_target(height=32, width=32, boundary_x=16).repeat(2, 1, 1)

    loss_fn = MultiClassCEDiceBoundaryDistanceLoss(num_classes=3, ignore_index=None)
    loss = loss_fn(logits, targets)

    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_multiclass_ce_dice_boundary_distance_loss_requires_multiple_classes():
    with pytest.raises(AssertionError):
        MultiClassCEDiceBoundaryDistanceLoss(num_classes=1)


# ---------------------------------------------------------------------------
# PatchSegmentationTask wiring: every custom loss name in configure_losses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "loss_name, num_classes, expected_cls",
    [
        ("bce_dice", 1, BCEDiceLoss),
        ("bce_dice_boundary", 1, BCEDiceBoundaryLoss),
        ("ce_dice_boundary", 3, MultiClassCEDiceBoundaryLoss),
        ("ce_dice_boundary_distance", 3, MultiClassCEDiceBoundaryDistanceLoss),
    ],
)
def test_patch_segmentation_task_wires_custom_losses(monkeypatch, loss_name, num_classes, expected_cls):
    monkeypatch.setitem(tasks_module.SEG_MODEL_MAPPING, "dummyseg", DummySegModel)

    task = PatchSegmentationTask(
        num_classes=num_classes,
        in_channels=4,
        model="dummyseg",
        loss=loss_name,
        ignore_index=0,
    )

    assert isinstance(task.criterion, expected_cls)

    x = torch.randn(2, 4, 32, 32)
    y = torch.randint(0, max(num_classes, 2), (2, 32, 32), dtype=torch.long)

    y_hat = task(x)
    loss = task.criterion(y_hat, y)

    assert torch.isfinite(loss)
    loss.backward()
