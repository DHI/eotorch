import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianNLL(nn.Module):
    """
    Gaussian negative log likelihood to fit the mean and variance to p(y|x)
    Note: We estimate the heteroscedastic variance. Hence, we include the var_i of sample i in the sum
    over all samples N. Furthermore, the constant log term is discarded.
    """
    def __init__(self, reduction : str = 'mean'):
        super().__init__()
        self.eps = 1e-8
        self.reduction = reduction

    def __call__(self, mean : torch.Tensor, variance : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
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
    
    Reference: Combines approaches from:
    - He et al. (2020): "Rethinking the U-Net architecture for multimodal biomedical image segmentation"
    - Lin et al. (2017): "Focal Loss for Dense Object Detection"
    
    Args:
        bce_weight (float): Weight for BCE component (default: 0.5). Range [0, 1].
        dice_weight (float): Weight for Dice component (default: 0.5). Range [0, 1].
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
            bce_loss = F.binary_cross_entropy_with_logits(
                logits, targets.float(), pos_weight=torch.tensor([self.pos_weight], device=logits.device)
            )
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

    This follows the practical recipe:
        L = lambda_region * (w_bce * BCE + w_dice * Dice) + lambda_boundary * L_boundary

    The boundary term compares image-gradient magnitudes of prediction probabilities
    and targets, which emphasizes contour quality without requiring explicit contour extraction.

    Args:
        bce_weight: BCE weight within the region term.
        dice_weight: Dice weight within the region term.
        boundary_weight: Weight of boundary term in the final combined loss.
        region_weight: Weight of region (BCE+Dice) term in the final combined loss.
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
        if targets.ndim == logits.ndim - 1:
            targets = targets.unsqueeze(1)

        targets = targets.float()

        # Region term: BCE + Dice
        if self.pos_weight is not None:
            bce = F.binary_cross_entropy_with_logits(
                logits,
                targets,
                pos_weight=torch.tensor([self.pos_weight], device=logits.device, dtype=logits.dtype),
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