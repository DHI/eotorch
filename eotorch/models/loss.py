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