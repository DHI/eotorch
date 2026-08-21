import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


def _to_pos_weight(pos_weight: float | torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    weight = torch.as_tensor(pos_weight, device=reference.device, dtype=reference.dtype)
    return weight.flatten()


class GaussianNLL(nn.Module):
    """
    Gaussian negative log likelihood to fit the mean and variance to p(y|x)
    Note: We estimate the heteroscedastic variance. Hence, we include the var_i of sample i in the sum
    over all samples N. Furthermore, the constant log term is discarded.

    Args:
        reduction: "mean" returns a scalar averaged over all elements; "none"
            returns the elementwise loss with no reduction. Defaults to "mean".
    """
    def __init__(self, reduction : str = 'mean'):
        super().__init__()
        self.eps = 1e-8
        self.reduction = reduction

    def __call__(self, mean : torch.Tensor, variance : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        """
        The exponential activation is applied already within the network to directly output variances.

        Args:
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
        if self.reduction == 'mean':
            return torch.mean(0.5 / variance * (mean - target)**2 + 0.5 * torch.log(variance))
        elif self.reduction == 'none':
            return 0.5 / variance * (mean - target)**2 + 0.5 * torch.log(variance)


class BCEDiceLoss(nn.Module):
    """
    Combined BCE and Dice loss optimized for edge detection and imbalanced binary segmentation.
    
    This loss combines the pixel-level precision of binary cross-entropy with the
    region-level overlap insensitivity of Dice loss, making it well-suited for
    detecting thin structures (e.g., edges) in highly imbalanced datasets.

    Args:
        bce_weight (float): Weight for BCE component (default: 0.5). Range [0, 1];
            bce_weight + dice_weight must sum to 1.
        dice_weight (float): Weight for Dice component (default: 0.5). Range [0, 1];
            bce_weight + dice_weight must sum to 1.
        pos_weight (float or None): Weight for positive class in BCE. Useful for imbalanced datasets.
            For 99.9% negatives / 0.1% positives, use ~999. Default: None.
        smooth (float): Smoothing constant for Dice to avoid division by zero. Default: 1.0.
        from_logits (bool): If True, assumes input is logits; if False, assumes probabilities. Default: True.
    
    Example:
        >>> loss_fn = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5, pos_weight=999.0)
        >>> logits = model(x)  # [B, 1, H, W]
        >>> targets = y  # [B, 1, H, W]
        >>> loss = loss_fn(logits, targets)
    """
    
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: float | None = None,
        smooth: float = 1.0,
        from_logits: bool = True,
    ) -> None:
        super().__init__()
        assert 0 <= bce_weight <= 1, "bce_weight must be in [0, 1]"
        assert 0 <= dice_weight <= 1, "dice_weight must be in [0, 1]"
        assert abs((bce_weight + dice_weight) - 1.0) < 1e-6, "bce_weight + dice_weight must sum to 1"
        
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.from_logits = from_logits
        
        # BCE component
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean') if from_logits else nn.BCELoss(reduction='mean')
        self.pos_weight = pos_weight
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute combined BCE+Dice loss.
        
        Args:
            logits: Model output [B, C, H, W]. For binary: C=1.
            targets: Ground truth [B, C, H, W] or [B, H, W] (will be unsqueezed).
        
        Returns:
            Combined loss scalar.
        """
        # Ensure targets have the same shape as logits
        if targets.ndim == logits.ndim - 1:
            targets = targets.unsqueeze(1)
        
        # BCE Loss
        if self.pos_weight is not None:
            bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=_to_pos_weight(self.pos_weight, logits))
        else:
            bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float())
        
        # Dice Loss
        # Convert logits to probabilities if needed
        if self.from_logits:
            probs = torch.sigmoid(logits)
        else:
            probs = logits
        
        # Flatten spatial dimensions
        probs_flat = probs.view(-1)
        targets_flat = targets.float().view(-1)
        
        # Dice coefficient
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_coeff
        
        # Combined loss
        combined_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        return combined_loss


class BCEDiceBoundaryLoss(nn.Module):
    """
    Boundary-aware combined loss for binary segmentation:
    BCE + Dice + boundary alignment term.

    Final loss:
        L = region_weight * (bce_weight * BCE + dice_weight * Dice) + boundary_weight * Boundary

    The boundary term compares image-gradient magnitudes of prediction probabilities
    and targets, which emphasizes contour quality without requiring explicit contour extraction.

    Args:
        bce_weight: BCE weight within the region term. bce_weight + dice_weight must sum to 1.
        dice_weight: Dice weight within the region term. bce_weight + dice_weight must sum to 1.
        boundary_weight: Weight of boundary term in the final combined loss.
            boundary_weight + region_weight must sum to 1.
        region_weight: Weight of region (BCE+Dice) term in the final combined loss.
            boundary_weight + region_weight must sum to 1.
        pos_weight: Optional positive-class weighting for BCE.
        smooth: Dice smoothing constant.
        from_logits: Whether model output is logits.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        boundary_weight: float = 0.3,
        region_weight: float = 0.7,
        pos_weight: float | None = None,
        smooth: float = 1.0,
        from_logits: bool = True,
    ) -> None:
        super().__init__()
        assert 0 <= bce_weight <= 1, "bce_weight must be in [0, 1]"
        assert 0 <= dice_weight <= 1, "dice_weight must be in [0, 1]"
        assert abs((bce_weight + dice_weight) - 1.0) < 1e-6, "bce_weight + dice_weight must sum to 1"
        assert 0 <= boundary_weight <= 1, "boundary_weight must be in [0, 1]"
        assert 0 <= region_weight <= 1, "region_weight must be in [0, 1]"
        assert abs((boundary_weight + region_weight) - 1.0) < 1e-6, (
            "boundary_weight + region_weight must sum to 1"
        )

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.region_weight = region_weight
        self.pos_weight = pos_weight
        self.smooth = smooth
        self.from_logits = from_logits

    @staticmethod
    def _gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
        """Compute Sobel gradient magnitude for [N, 1, H, W] tensors."""
        sobel_x = torch.tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
            device=x.device,
            dtype=x.dtype,
        ).unsqueeze(0)
        sobel_y = torch.tensor(
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
            device=x.device,
            dtype=x.dtype,
        ).unsqueeze(0)

        gx = F.conv2d(x, sobel_x, padding=1)
        gy = F.conv2d(x, sobel_y, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-8)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model output [B, 1, H, W].
            targets: Ground truth [B, 1, H, W] or [B, H, W] (will be unsqueezed).

        Returns:
            Combined region + boundary loss scalar.
        """
        if targets.ndim == logits.ndim - 1:
            targets = targets.unsqueeze(1)

        targets = targets.float()

        # Region term: BCE + Dice
        if self.pos_weight is not None:
            bce = F.binary_cross_entropy_with_logits(
                logits,
                targets,
                pos_weight=_to_pos_weight(self.pos_weight, logits),
            )
        else:
            bce = F.binary_cross_entropy_with_logits(logits, targets)

        probs = torch.sigmoid(logits) if self.from_logits else logits
        probs_flat = probs.reshape(-1)
        targets_flat = targets.reshape(-1)
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()
        dice = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)
        region_loss = self.bce_weight * bce + self.dice_weight * dice

        # Boundary term: align contour strength between prediction and target.
        pred_grad = self._gradient_magnitude(probs)
        target_grad = self._gradient_magnitude(targets)
        boundary_loss = F.l1_loss(pred_grad, target_grad)

        return self.region_weight * region_loss + self.boundary_weight * boundary_loss


class MultiClassCEDiceBoundaryLoss(nn.Module):
    """
    Boundary-aware combined loss for multiclass segmentation:
    CE + multiclass Dice + boundary alignment term.

    Final loss:
        L = region_weight * (ce_weight * CE + dice_weight * Dice) + boundary_weight * Boundary

    Args:
        num_classes: Number of classes.
        ce_weight: Cross-entropy weight inside region term. ce_weight + dice_weight must sum to 1.
        dice_weight: Dice weight inside region term. ce_weight + dice_weight must sum to 1.
        boundary_weight: Boundary term weight in final loss.
            boundary_weight + region_weight must sum to 1.
        region_weight: Region term weight in final loss.
            boundary_weight + region_weight must sum to 1.
        class_weights: Optional class weights for CE.
        ignore_index: Optional ignore index for CE/Dice.
        smooth: Dice smoothing constant.
        from_logits: Whether model output is logits.
    """

    def __init__(
        self,
        num_classes: int,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        boundary_weight: float = 0.3,
        region_weight: float = 0.7,
        class_weights: torch.Tensor | None = None,
        ignore_index: int | None = None,
        smooth: float = 1.0,
        from_logits: bool = True,
    ) -> None:
        super().__init__()
        assert num_classes > 1, "MultiClassCEDiceBoundaryLoss requires num_classes > 1"
        assert 0 <= ce_weight <= 1, "ce_weight must be in [0, 1]"
        assert 0 <= dice_weight <= 1, "dice_weight must be in [0, 1]"
        assert abs((ce_weight + dice_weight) - 1.0) < 1e-6, "ce_weight + dice_weight must sum to 1"
        assert 0 <= boundary_weight <= 1, "boundary_weight must be in [0, 1]"
        assert 0 <= region_weight <= 1, "region_weight must be in [0, 1]"
        assert abs((boundary_weight + region_weight) - 1.0) < 1e-6, (
            "boundary_weight + region_weight must sum to 1"
        )

        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.region_weight = region_weight
        self.class_weights = class_weights
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.from_logits = from_logits

    @staticmethod
    def _gradient_magnitude_per_channel(x: torch.Tensor) -> torch.Tensor:
        """Compute Sobel gradient magnitude for [N, C, H, W] tensors (grouped conv)."""
        channels = x.shape[1]
        sobel_x = torch.tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
            device=x.device,
            dtype=x.dtype,
        ).unsqueeze(0).repeat(channels, 1, 1, 1)
        sobel_y = torch.tensor(
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
            device=x.device,
            dtype=x.dtype,
        ).unsqueeze(0).repeat(channels, 1, 1, 1)

        gx = F.conv2d(x, sobel_x, padding=1, groups=channels)
        gy = F.conv2d(x, sobel_y, padding=1, groups=channels)
        return torch.sqrt(gx * gx + gy * gy + 1e-8)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model output, [N, num_classes, H, W].
            targets: Ground-truth class indices, [N, H, W] (or [N, 1, H, W]).

        Returns:
            Combined region + boundary loss scalar.
        """
        if targets.ndim == logits.ndim and targets.shape[1] == 1:
            targets = targets.squeeze(1)

        targets = targets.long()

        # CE term
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            ignore_index=self.ignore_index if self.ignore_index is not None else -100,
        )

        probs = F.softmax(logits, dim=1) if self.from_logits else logits

        # One-hot targets for Dice + boundary
        valid_mask = None
        if self.ignore_index is not None:
            valid_mask = (targets != self.ignore_index)
            safe_targets = targets.clone()
            safe_targets[~valid_mask] = 0
        else:
            safe_targets = targets

        target_one_hot = F.one_hot(safe_targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        if valid_mask is not None:
            valid_mask_f = valid_mask.unsqueeze(1).float()
            probs = probs * valid_mask_f
            target_one_hot = target_one_hot * valid_mask_f

        # Multiclass Dice (macro over classes)
        probs_flat = probs.reshape(probs.shape[0], probs.shape[1], -1)
        targets_flat = target_one_hot.reshape(target_one_hot.shape[0], target_one_hot.shape[1], -1)
        intersection = (probs_flat * targets_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)
        dice_per_class = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice = dice_per_class.mean()

        region_loss = self.ce_weight * ce + self.dice_weight * dice

        # Boundary term on per-class probability maps
        pred_grad = self._gradient_magnitude_per_channel(probs)
        target_grad = self._gradient_magnitude_per_channel(target_one_hot)
        boundary_loss = F.l1_loss(pred_grad, target_grad)

        return self.region_weight * region_loss + self.boundary_weight * boundary_loss


def _class_boundary_distance_np(
    labels: np.ndarray,
    c: int,
    max_distance: float,
    ignore_index: int | None = None,
) -> np.ndarray:
    """Signed Euclidean distance to the boundary of class `c`, for a single [H, W] label array.

    Negative inside class `c`, positive outside, zero at the boundary; clamped to
    [-max_distance, max_distance] and normalized by max_distance so it stays O(1),
    comparable in scale to CE/Dice.

    Without `ignore_index`, this is a plain one-vs-rest signed distance transform:
    "outside" is simply every non-`c` pixel. With `ignore_index` set, "outside" is
    narrowed to pixels that are validly some *other* real class (not `c`, not
    `ignore_index`) — otherwise, wherever `ignore_index` pixels (e.g. no-data /
    unlabeled background) sit closer to a class-`c` region than any genuinely
    different class does, the *inside* distance would reflect proximity to that
    no-data region rather than the real inter-class boundary. This matters in
    particular when patches are sampled from a buffered region around the true
    boundary, which puts unlabeled background pixels directly adjacent to real
    classes throughout the dataset, not just as a rare edge case.

    The *outside* distance doesn't need this narrowing: it's already exactly
    "distance to the nearest class-`c` pixel" for any querying pixel, regardless
    of what other classes (or no-data) happen to be nearby.

    A degenerate mask (`c` absent, or no distinguishable "other" class present)
    has no positional signal to give and returns zeros.
    """
    mask_c = labels == c
    other = ~mask_c if ignore_index is None else (labels != c) & (labels != ignore_index)
    if not mask_c.any() or not other.any():
        return np.zeros(labels.shape, dtype=np.float32)

    dist_to_c = distance_transform_edt(~mask_c)
    dist_to_other = distance_transform_edt(~other)
    signed = np.where(mask_c, -dist_to_other, dist_to_c)
    signed = np.clip(signed, -max_distance, max_distance) / max_distance
    return signed.astype(np.float32)


def compute_class_distance_maps(
    labels: np.ndarray,
    num_classes: int,
    max_distance: float,
    ignore_index: int | None = None,
) -> np.ndarray:
    """Per-class signed distance maps for a single [H, W] integer label array.

    Pure numpy/scipy, no torch or GPU involved — intended for precomputing
    distance maps once (e.g. alongside label patches on disk) instead of
    recomputing them from scratch on every training step, since the result
    only depends on the label, not on model predictions. Precomputed maps can
    be fed back in via `BoundaryDistanceLoss`/`MultiClassCEDiceBoundaryDistanceLoss`'s
    `distance_maps` argument to skip the redundant recomputation entirely.

    `ignore_index`'s own channel is left all-zero (never used, since
    `BoundaryDistanceLoss` masks out `ignore_index` pixels entirely) and is
    also excluded from what counts as "outside" for every other class's inside
    distance — see `_class_boundary_distance_np` for why that matters.

    Args:
        labels: Integer class-index array, shape (H, W).
        num_classes: Number of classes; output has one map per class.
        max_distance: Distance (in pixels) at which the signed distance map
            saturates (see `_class_boundary_distance_np`).
        ignore_index: Optional class value to exclude from boundary geometry
            entirely (e.g. no-data / unlabeled background), matching the
            `ignore_index` passed to the loss this feeds into.

    Returns:
        np.ndarray: float32 array of shape (num_classes, H, W).
    """
    maps = np.zeros((num_classes, *labels.shape), dtype=np.float32)
    for c in range(num_classes):
        if c == ignore_index:
            continue
        maps[c] = _class_boundary_distance_np(labels, c, max_distance, ignore_index=ignore_index)
    return maps


class BoundaryDistanceLoss(nn.Module):
    """
    Kervadec-style boundary distance loss for multiclass segmentation.

    Kervadec, H.; Bouchtiba, J.; Desrosiers, C.; Granger, E.; Dolz, J.; Ben Ayed, I.:
    "Boundary loss for highly unbalanced segmentation." MIDL 2019.

    Unlike a gradient-magnitude boundary term (see MultiClassCEDiceBoundaryLoss),
    which only rewards an edge of similar sharpness *somewhere* with no positional
    signal, this term is directly informative about *where* the boundary should be:
    predicted probability mass far outside the true region for a class costs
    proportionally more than probability mass placed just past the edge.

    Per-class signed distance maps are derived only from the target mask and are
    therefore treated as constants (computed under torch.no_grad()); only the
    predicted probabilities carry gradient.

    Args:
        num_classes: Number of classes.
        max_distance: Distance (in pixels) at which the signed distance map
            saturates. Bounds the term's magnitude and concentrates gradient signal
            near the boundary, the region that matters for this loss.
        ignore_index: Optional ignore index, excluded from the mean.
        from_logits: Whether model output is logits.
    """

    def __init__(
        self,
        num_classes: int,
        max_distance: float = 24.0,
        ignore_index: int | None = None,
        from_logits: bool = True,
    ) -> None:
        super().__init__()
        assert num_classes > 1, "BoundaryDistanceLoss requires num_classes > 1"
        self.num_classes = num_classes
        self.max_distance = max_distance
        self.ignore_index = ignore_index
        self.from_logits = from_logits

    @torch.no_grad()
    def _distance_maps(self, targets: torch.Tensor) -> torch.Tensor:
        """Per-sample, per-class signed distance maps for a [N, H, W] long target.

        Delegates to `compute_class_distance_maps` with this loss's own
        `ignore_index`, so the on-the-fly path computes the exact same
        geometry as a precomputed `distance_maps` argument would (see
        `compute_class_distance_maps` for why `ignore_index` needs to be
        excluded from boundary geometry, not just masked out afterward).
        """
        n, h, w = targets.shape
        targets_np = targets.detach().cpu().numpy()
        maps = np.stack([
            compute_class_distance_maps(targets_np[i], self.num_classes, self.max_distance, ignore_index=self.ignore_index)
            for i in range(n)
        ])
        return torch.as_tensor(maps, dtype=torch.float32, device=targets.device)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        distance_maps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: Model output, [N, num_classes, H, W].
            targets: Ground-truth class indices, [N, H, W] (or [N, 1, H, W]).
            distance_maps: Optional precomputed per-class signed distance maps,
                [N, num_classes, H, W] (see `compute_class_distance_maps`). When
                given, the on-the-fly `scipy.ndimage.distance_transform_edt`
                computation is skipped entirely — use this to avoid recomputing
                the same maps every step when they only depend on `targets`,
                which doesn't change across epochs for a fixed dataset.

        Returns:
            Boundary distance loss scalar.
        """
        if targets.ndim == logits.ndim and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        targets = targets.long()

        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
        else:
            valid_mask = torch.ones_like(targets, dtype=torch.bool)

        if distance_maps is None:
            distance_maps = self._distance_maps(targets)
        else:
            distance_maps = distance_maps.to(device=logits.device, dtype=torch.float32)

        probs = F.softmax(logits, dim=1) if self.from_logits else logits

        valid_mask_f = valid_mask.unsqueeze(1).float()
        weighted = probs * distance_maps * valid_mask_f
        denom = (valid_mask_f.sum() * self.num_classes).clamp(min=1.0)
        return weighted.sum() / denom


class MultiClassCEDiceBoundaryDistanceLoss(nn.Module):
    """
    Boundary-aware combined loss for multiclass segmentation using a distance-based
    boundary term instead of gradient-magnitude matching:
    CE + multiclass Dice + boundary distance term.

    Final loss:
        L = region_weight * (ce_weight * CE + dice_weight * Dice) + boundary_weight * BoundaryDistance

    A like-for-like drop-in for MultiClassCEDiceBoundaryLoss (same constructor
    shape, plus max_distance), so the two can be compared directly: this term
    rewards predicted probability mass by its distance to the true boundary, rather
    than by matching edge sharpness with no positional signal.

    Args:
        num_classes: Number of classes.
        ce_weight: Cross-entropy weight inside region term. ce_weight + dice_weight must sum to 1.
        dice_weight: Dice weight inside region term. ce_weight + dice_weight must sum to 1.
        boundary_weight: Boundary term weight in final loss.
            boundary_weight + region_weight must sum to 1.
        region_weight: Region term weight in final loss.
            boundary_weight + region_weight must sum to 1.
        max_distance: Distance (in pixels) at which the signed distance map
            saturates (see BoundaryDistanceLoss).
        class_weights: Optional class weights for CE.
        ignore_index: Optional ignore index for CE/Dice/boundary.
        smooth: Dice smoothing constant.
        from_logits: Whether model output is logits.
    """

    def __init__(
        self,
        num_classes: int,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        boundary_weight: float = 0.3,
        region_weight: float = 0.7,
        max_distance: float = 24.0,
        class_weights: torch.Tensor | None = None,
        ignore_index: int | None = None,
        smooth: float = 1.0,
        from_logits: bool = True,
    ) -> None:
        super().__init__()
        assert num_classes > 1, "MultiClassCEDiceBoundaryDistanceLoss requires num_classes > 1"
        assert 0 <= ce_weight <= 1, "ce_weight must be in [0, 1]"
        assert 0 <= dice_weight <= 1, "dice_weight must be in [0, 1]"
        assert abs((ce_weight + dice_weight) - 1.0) < 1e-6, "ce_weight + dice_weight must sum to 1"
        assert 0 <= boundary_weight <= 1, "boundary_weight must be in [0, 1]"
        assert 0 <= region_weight <= 1, "region_weight must be in [0, 1]"
        assert abs((boundary_weight + region_weight) - 1.0) < 1e-6, (
            "boundary_weight + region_weight must sum to 1"
        )

        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.region_weight = region_weight
        self.class_weights = class_weights
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.from_logits = from_logits
        self.boundary_loss = BoundaryDistanceLoss(
            num_classes=num_classes,
            max_distance=max_distance,
            ignore_index=ignore_index,
            from_logits=from_logits,
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        distance_maps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: Model output, [N, num_classes, H, W].
            targets: Ground-truth class indices, [N, H, W] (or [N, 1, H, W]).
            distance_maps: Optional precomputed per-class signed distance maps
                forwarded to the boundary term (see `BoundaryDistanceLoss.forward`
                and `compute_class_distance_maps`), skipping its on-the-fly
                `scipy.ndimage.distance_transform_edt` computation.

        Returns:
            Combined region + boundary distance loss scalar.
        """
        if targets.ndim == logits.ndim and targets.shape[1] == 1:
            targets = targets.squeeze(1)

        targets = targets.long()

        # CE term
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            ignore_index=self.ignore_index if self.ignore_index is not None else -100,
        )

        probs = F.softmax(logits, dim=1) if self.from_logits else logits

        # One-hot targets for Dice
        valid_mask = None
        if self.ignore_index is not None:
            valid_mask = (targets != self.ignore_index)
            safe_targets = targets.clone()
            safe_targets[~valid_mask] = 0
        else:
            safe_targets = targets

        target_one_hot = F.one_hot(safe_targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        dice_probs = probs
        dice_targets = target_one_hot
        if valid_mask is not None:
            valid_mask_f = valid_mask.unsqueeze(1).float()
            dice_probs = dice_probs * valid_mask_f
            dice_targets = dice_targets * valid_mask_f

        # Multiclass Dice (macro over classes)
        probs_flat = dice_probs.reshape(dice_probs.shape[0], dice_probs.shape[1], -1)
        targets_flat = dice_targets.reshape(dice_targets.shape[0], dice_targets.shape[1], -1)
        intersection = (probs_flat * targets_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)
        dice_per_class = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice = dice_per_class.mean()

        region_loss = self.ce_weight * ce + self.dice_weight * dice

        # Boundary term: penalize predicted probability mass by distance to the true boundary
        boundary_loss = self.boundary_loss(logits, targets, distance_maps=distance_maps)

        return self.region_weight * region_loss + self.boundary_weight * boundary_loss