"""
DINOv3 backbone with frozen features passed to UPerNet decoder.

Provides a minimal integration of Hugging Face DINOv3 as a frozen backbone
with SMP's UPerNet decoder for semantic segmentation tasks.
"""

from __future__ import annotations

import re
from typing import Any

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from transformers import AutoModel


class DINOv3UPerNet(nn.Module):
    """Combines frozen DINOv3 from Hugging Face with trainable SMP UPerNet decoder.

    Strategy:
    1. Extract frozen patch embeddings from DINOv3 (shape: [B, hidden_size, H', W']).
    2. Use SMP's ResNet-based UPerNet encoder to process these embeddings.
    3. Pass through the trained UPerNet decoder to produce class logits.

    Args:
        num_classes: Number of output segmentation classes.
        dinov3_model_name: Hugging Face identifier (e.g., 'facebook/dinov3-vitl14-pretrained').
            Short model IDs are also supported for known variants, including:
            - 'dinov3-vitl16-pretrain-sat493m'
            - 'dinov3-vit7b16-pretrain-sat493m'
        in_channels: Input channels (must match DINOv3 input, usually 3).
        decoder_channels: Decoder hidden channels (default: 256).
        freeze_backbone: If True, freeze DINOv3 weights (default: True).

    Example:
        >>> model = DINOv3UPerNet(
        ...     num_classes=3,
        ...     dinov3_model_name='facebook/dinov3-vitl14-pretrained',
        ...     freeze_backbone=True,
        ... )
        >>> x = torch.randn(2, 3, 224, 224)
        >>> logits = model(x)  # [2, 3, 224, 224]
    """

    _KNOWN_SHORT_MODEL_IDS: set[str] = {
        "dinov3-vitl14-pretrained",
        "dinov3-vitl16-pretrained",
        "dinov3-base14-pretrained",
        "dinov3-convnext-base-pretrained",
        "dinov3-vitl16-pretrain-sat493m",
        "dinov3-vit7b16-pretrain-sat493m",
    }

    @staticmethod
    def _resolve_model_name(model_name: str) -> str:
        """Resolve short DINOv3 IDs to fully qualified Hugging Face IDs."""
        if "/" in model_name:
            return model_name

        if model_name in DINOv3UPerNet._KNOWN_SHORT_MODEL_IDS:
            return f"facebook/{model_name}"

        return model_name

    @staticmethod
    def _infer_patch_size(model_name: str) -> int:
        """Infer patch size from model identifier, defaulting to 14."""
        # Covers names like "vitl16", "vit7b16", "base14", etc.
        match = re.search(r"(?:vit\w*|base)(14|16)", model_name)
        if match:
            return int(match.group(1))

        # Fallback for arbitrary names that still include 14/16 tokens.
        if "16" in model_name:
            return 16
        if "14" in model_name:
            return 14

        return 14

    def __init__(
        self,
        num_classes: int,
        dinov3_model_name: str = "facebook/dinov3-vitl16-pretrain-sat493m",
        in_channels: int = 3,
        decoder_channels: int = 256,
        freeze_backbone: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dinov3_model_name = self._resolve_model_name(dinov3_model_name)
        self.in_channels = in_channels
        self.decoder_channels = decoder_channels
        self.out_channels = num_classes

        # Load frozen DINOv3 backbone from Hugging Face
        self.dinov3_backbone = AutoModel.from_pretrained(self.dinov3_model_name)
        self.hidden_size = self.dinov3_backbone.config.hidden_size

        if freeze_backbone:
            for param in self.dinov3_backbone.parameters():
                param.requires_grad = False

        self.patch_size = self._infer_patch_size(self.dinov3_model_name)

        # Use SMP's UPerNet with ResNet50 encoder, but replace the encoder with DINOv3 features
        self.upernet = smp.UPerNet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
            decoder_channels=decoder_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: extract DINOv3 features, feed through UPerNet decoder.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Logits [B, num_classes, H, W].
        """
        batch_size, _, h, w = x.shape

        # Extract DINOv3 features (frozen)
        with torch.no_grad():
            dino_out = self.dinov3_backbone(x, output_hidden_states=True)

        # last_hidden_state shape: [B, num_patches + 1, hidden_size]
        # num_patches = (h // patch_size) * (w // patch_size)
        features = dino_out.last_hidden_state

        # Remove CLS token (first token), keep only patch tokens
        patch_tokens = features[:, 1:, :]  # [B, num_patches, hidden_size]

        # Reshape to spatial feature map
        num_h = h // self.patch_size
        num_w = w // self.patch_size
        patch_map = patch_tokens.view(
            batch_size, num_h, num_w, self.hidden_size
        ).permute(0, 3, 1, 2)  # [B, hidden_size, num_h, num_w]

        # Upsample features back to input resolution
        upsampled = torch.nn.functional.interpolate(
            patch_map, size=(h, w), mode="bilinear", align_corners=False
        )  # [B, hidden_size, h, w]

        logits = self.upernet(x)

        return logits


__all__ = ["DINOv3UPerNet"]
