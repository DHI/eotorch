from __future__ import annotations

from typing import Any

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianNLL(nn.Module):
    """
    Gaussian negative log likelihood to fit the mean and variance to p(y|x)
    Note: We estimate the heteroscedastic variance. Hence, we include the var_i of sample i in the sum
    over all samples N. Furthermore, the constant log term is discarded.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.eps = 1e-8
        self.reduction = reduction

    def __call__(
        self, mean: torch.Tensor, variance: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        The exponential activation is applied already within the network to directly output variances.

        Parameters:
            mean (torch.Tensor):
                Predicted mean values.
            variance (torch.Tensor):
                Predicted variance.
            target (torch.Tensor):
                Ground truth labels.

        Returns:
            torch.Tensor:
                Gaussian negative log likelihood
        """
        variance = variance + self.eps
        if self.reduction == "mean":
            return torch.mean(
                0.5 / variance * (mean - target) ** 2 + 0.5 * torch.log(variance)
            )
        elif self.reduction == "none":
            return 0.5 / variance * (mean - target) ** 2 + 0.5 * torch.log(variance)


def _probabilities_and_targets(
    logits: torch.Tensor, targets: torch.Tensor, ignore_index: int | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Probabilities and one-hot targets (B, C, H, W) and a mask of valid pixels (B, 1, H, W).

    Binary segmentation logits may come as (B, H, W), with C = 1 for the positive class.
    """
    if logits.ndim == 3:
        logits = logits.unsqueeze(1)
    targets = targets.long()
    valid = torch.ones_like(targets, dtype=torch.bool)
    if ignore_index is not None:
        valid = targets != ignore_index
        targets = torch.where(valid, targets, 0)

    if logits.shape[1] == 1:
        probs = logits.sigmoid()
        one_hot = targets.unsqueeze(1).float()
    else:
        probs = logits.softmax(dim=1)
        one_hot = F.one_hot(targets, logits.shape[1]).permute(0, 3, 1, 2).float()
    return probs, one_hot, valid.unsqueeze(1)


class BoundaryLoss(nn.Module):
    """L1 distance between the edges of the predicted probabilities and the targets.

    Edges are Sobel gradient magnitudes, which emphasizes contour quality without having
    to extract contours.
    """

    def __init__(self, ignore_index: int | None = None):
        super().__init__()
        self.ignore_index = ignore_index
        sobel = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        self.register_buffer(
            "kernels", torch.stack([sobel, sobel.T]).unsqueeze(1), persistent=False
        )

    def _edges(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        gradients = F.conv2d(x.reshape(b * c, 1, h, w), self.kernels, padding=1)
        return gradients.pow(2).sum(dim=1).add(1e-8).sqrt().reshape(b, c, h, w)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs, one_hot, valid = _probabilities_and_targets(
            logits, targets, self.ignore_index
        )
        return F.l1_loss(self._edges(probs * valid), self._edges(one_hot * valid))


class BoundaryDistanceLoss(nn.Module):
    """Boundary loss of Kervadec et al., "Boundary loss for highly unbalanced segmentation", MIDL 2019.

    The predicted probability of each class is weighted by the signed distance to the
    boundary of that class in the target (negative inside, positive outside), so
    probability far outside the true region costs more than probability just past its
    edge. Distances are chessboard distances, clamped to ``max_distance`` pixels and
    scaled to [-1, 1]. Pixels of ``ignore_index`` don't count as a boundary.
    """

    def __init__(self, max_distance: int = 24, ignore_index: int | None = None):
        super().__init__()
        self.max_distance = max_distance
        self.ignore_index = ignore_index

    def _distance_to(self, mask: torch.Tensor) -> torch.Tensor:
        """Distance of each pixel to the closest pixel in mask, up to max_distance."""
        distance = torch.zeros(mask.shape, device=mask.device)
        reached = mask.float()
        for _ in range(self.max_distance):
            distance += 1 - reached
            reached = F.max_pool2d(reached, kernel_size=3, stride=1, padding=1)
        return distance

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs, one_hot, valid = _probabilities_and_targets(
            logits, targets, self.ignore_index
        )
        with torch.no_grad():
            inside = one_hot.bool() & valid
            outside = ~inside & valid
            distance = torch.where(
                inside, -self._distance_to(outside), self._distance_to(inside)
            )
            # A class that is absent, or covers all valid pixels, has no boundary
            has_boundary = inside.flatten(2).any(-1) & outside.flatten(2).any(-1)
            distance *= has_boundary[..., None, None] / self.max_distance
        return (probs * distance * valid).sum() / (valid.sum() * probs.shape[1]).clamp(
            min=1
        )


class CombinedLoss(nn.Module):
    """Weighted sum of losses, e.g. ``weights={"ce": 0.5, "dice": 0.5, "boundary": 0.3}``.

    The weights can be changed during training, see
    :class:`eotorch.models.callbacks.LossWeightScheduler`.
    """

    def __init__(self, losses: dict[str, nn.Module], weights: dict[str, float]):
        super().__init__()
        self.losses = nn.ModuleDict(losses)
        self.weights = dict(weights)

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return sum(
            self.weights[name] * loss(y_hat, y) for name, loss in self.losses.items()
        )


def segmentation_loss(
    loss: str | dict[str, float],
    task: str,
    num_classes: int | None = None,
    ignore_index: int | None = None,
    class_weights: torch.Tensor | list[float] | None = None,
    pos_weight: torch.Tensor | float | None = None,
    loss_kwargs: dict[str, dict[str, Any]] | None = None,
) -> nn.Module:
    """Create a segmentation loss by name, or a :class:`CombinedLoss` from a dict of weights.

    Args:
        loss: One of "ce", "bce", "dice", "focal", "jaccard", "tversky", "boundary" and
            "boundary_distance", or a dict mapping several of them to their weights.
        task: One of "binary", "multiclass" or "multilabel".
        num_classes: Number of classes.
        ignore_index: Label value to ignore. Not supported by "bce".
        class_weights: Weight of each class, only used by "ce".
        pos_weight: Weight of positive examples, only used by "bce".
        loss_kwargs: Extra keyword arguments per loss name,
            e.g. ``{"focal": {"gamma": 2.0}, "boundary_distance": {"max_distance": 16}}``.
    """
    loss_kwargs = loss_kwargs or {}
    if isinstance(loss, dict):
        losses = {
            name: segmentation_loss(
                name,
                task,
                num_classes,
                ignore_index,
                class_weights,
                pos_weight,
                loss_kwargs,
            )
            for name in loss
        }
        return CombinedLoss(losses, loss)

    kwargs = loss_kwargs.get(loss, {})
    match loss:
        case "ce":
            if class_weights is not None:
                class_weights = torch.as_tensor(class_weights, dtype=torch.float32)
            return nn.CrossEntropyLoss(
                weight=class_weights,
                ignore_index=-100 if ignore_index is None else ignore_index,
                **kwargs,
            )
        case "bce":
            if pos_weight is not None:
                pos_weight = torch.as_tensor(pos_weight, dtype=torch.float32)
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight, **kwargs)
        case "dice":
            return smp.losses.DiceLoss(task, ignore_index=ignore_index, **kwargs)
        case "focal":
            kwargs = {"normalized": True, "alpha": 0.8, "gamma": 3.0, **kwargs}
            return smp.losses.FocalLoss(task, ignore_index=ignore_index, **kwargs)
        case "tversky":
            kwargs = {"alpha": 0.3, "beta": 0.7, **kwargs}
            return smp.losses.TverskyLoss(task, ignore_index=ignore_index, **kwargs)
        case "jaccard":
            # JaccardLoss takes the classes to use instead of a class index to ignore
            classes = None
            if task == "multiclass" and ignore_index is not None:
                classes = [i for i in range(num_classes) if i != ignore_index]
            return smp.losses.JaccardLoss(task, classes=classes, **kwargs)
        case "boundary":
            return BoundaryLoss(ignore_index=ignore_index)
        case "boundary_distance":
            return BoundaryDistanceLoss(ignore_index=ignore_index, **kwargs)
    raise ValueError(f"Unknown loss: {loss}")
