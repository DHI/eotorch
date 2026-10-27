"""Tests for the segmentation losses."""

from types import SimpleNamespace

import pytest
import torch

from eotorch.models.callbacks import LossWeightScheduler
from eotorch.models.loss import (
    BoundaryDistanceLoss,
    BoundaryLoss,
    CombinedLoss,
    segmentation_loss,
)

LOSSES = ["ce", "dice", "focal", "jaccard", "tversky", "boundary", "boundary_distance"]


def _vertical_boundary_target(boundary_x: int = 32) -> torch.Tensor:
    """(1, 40, 64) target: columns left of boundary_x are class 0, the rest class 1."""
    target = torch.zeros((1, 40, 64), dtype=torch.long)
    target[:, :, boundary_x:] = 1
    return target


def _logits_for_boundary(boundary_x: int) -> torch.Tensor:
    """(1, 2, 40, 64) logits confidently predicting a vertical boundary at boundary_x."""
    logits = torch.full((1, 2, 40, 64), -10.0)
    logits[:, 0, :, :boundary_x] = 10.0
    logits[:, 1, :, boundary_x:] = 10.0
    return logits


@pytest.mark.parametrize("name", LOSSES)
def test_multiclass_losses_backpropagate(name):
    logits = torch.randn(2, 3, 32, 32, requires_grad=True)
    targets = torch.randint(0, 3, (2, 32, 32))
    targets[:, :4] = 0  # some ignored pixels

    loss = segmentation_loss(name, "multiclass", num_classes=3, ignore_index=0)(
        logits, targets
    )
    loss.backward()

    assert loss.ndim == 0 and torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("name", ["bce", "dice", "focal", "boundary", "boundary_distance"])
def test_binary_losses_accept_squeezed_logits(name):
    """Binary segmentation passes (B, H, W) logits and float targets, as TorchGeo does."""
    logits = torch.randn(2, 32, 32, requires_grad=True)
    targets = torch.randint(0, 2, (2, 32, 32)).float()

    loss = segmentation_loss(name, "binary", num_classes=1)(logits, targets)
    loss.backward()

    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def test_boundary_distance_grows_with_boundary_offset():
    target = _vertical_boundary_target(boundary_x=32)
    loss_fn = BoundaryDistanceLoss(max_distance=24)

    small = loss_fn(_logits_for_boundary(34), target)
    large = loss_fn(_logits_for_boundary(52), target)
    exact = loss_fn(_logits_for_boundary(32), target)

    assert exact < small < large


def test_boundary_distance_ignores_ignore_index_as_boundary():
    """Unlabelled pixels next to a class must not create a boundary for it."""
    target = torch.zeros((1, 40, 64), dtype=torch.long)  # 0 = ignored
    target[:, :, 20:44] = 1
    logits = torch.zeros((1, 2, 40, 64))
    logits[:, 1] = 10.0  # confidently class 1 everywhere

    # Class 1 is the only labelled class, so there is no boundary to learn from
    assert BoundaryDistanceLoss(ignore_index=0)(logits, target) == 0


def test_boundary_loss_is_zero_for_perfect_prediction():
    target = _vertical_boundary_target(boundary_x=32)
    assert BoundaryLoss()(_logits_for_boundary(32), target) < 1e-3
    assert BoundaryLoss()(_logits_for_boundary(48), target) > 1e-3


def test_combined_loss_is_weighted_sum():
    logits = torch.randn(2, 3, 16, 16)
    targets = torch.randint(0, 3, (2, 16, 16))
    combined = segmentation_loss({"ce": 0.25, "dice": 2.0}, "multiclass", num_classes=3)

    ce = segmentation_loss("ce", "multiclass", num_classes=3)(logits, targets)
    dice = segmentation_loss("dice", "multiclass", num_classes=3)(logits, targets)

    assert isinstance(combined, CombinedLoss)
    torch.testing.assert_close(combined(logits, targets), 0.25 * ce + 2.0 * dice)


def test_loss_kwargs_are_passed_per_loss():
    loss = segmentation_loss(
        {"focal": 1.0, "boundary_distance": 1.0},
        "multiclass",
        num_classes=3,
        loss_kwargs={"focal": {"gamma": 1.5}, "boundary_distance": {"max_distance": 8}},
    )
    assert loss.losses["focal"].focal_loss_fn.keywords["gamma"] == 1.5
    assert loss.losses["boundary_distance"].max_distance == 8


def test_unknown_loss():
    with pytest.raises(ValueError, match="Unknown loss"):
        segmentation_loss("nope", "multiclass", num_classes=3)


def test_loss_weight_scheduler_ramps_weight():
    criterion = segmentation_loss(
        {"ce": 1.0, "boundary": 0.0}, "multiclass", num_classes=3
    )
    module = SimpleNamespace(criterion=criterion)
    scheduler = LossWeightScheduler("boundary", start=0.0, end=0.3, warmup_epochs=10)

    for epoch, expected in [(0, 0.0), (5, 0.15), (10, 0.3), (20, 0.3)]:
        scheduler.on_train_epoch_start(SimpleNamespace(current_epoch=epoch), module)
        assert criterion.weights["boundary"] == pytest.approx(expected)
    assert criterion.weights["ce"] == 1.0
