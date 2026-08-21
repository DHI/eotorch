"""
DINOv3 backbone with frozen features driving a UPerNet decoder.

Provides a minimal integration of Hugging Face DINOv3 as a frozen backbone
feeding SMP's UPerNet decoder for semantic segmentation tasks.
"""

from __future__ import annotations

import re
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.decoders.upernet.decoder import UPerNetDecoder
from transformers import AutoBackbone


class _ViTFeaturePyramid(nn.Module):
    """ViTDet-style "simple feature pyramid": builds a 4-level {1/4, 1/8, 1/16, 1/32}
    feature pyramid (of the input image) from a single-scale ViT feature map, for
    feeding into a CNN-style decoder such as `UPerNetDecoder`, which expects a
    multi-scale pyramid rather than a ViT's single-resolution token grid.

    Reference: Li, Mao, Girshick, He, "Exploring Plain Vision Transformer
    Backbones for Object Detection", ECCV 2022 — uniform channel width across
    all output levels, resampled from the one ViT feature map via
    transposed-conv/identity/maxpool branches.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.stride4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=2, stride=2),
        )
        self.stride8 = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.stride16 = nn.Identity()
        self.stride32 = nn.MaxPool2d(kernel_size=2, stride=2)

        branch_channels = [in_channels // 4, in_channels // 2, in_channels, in_channels]
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
                for c in branch_channels
            ]
        )

    def forward(
        self, feature: torch.Tensor, target_sizes: list[tuple[int, int]]
    ) -> list[torch.Tensor]:
        """Resample `feature` into 4 levels and interpolate each to its exact target size.

        Args:
            feature: Single-scale ViT feature map, [B, in_channels, H', W'].
            target_sizes: Four (height, width) pairs, one per pyramid level
                (strides 4, 8, 16, 32 of the original input), used as a final
                alignment step since patch_size doesn't always divide the
                input size into clean powers of two.

        Returns:
            Four feature maps, each [B, out_channels, *target_sizes[i]].
        """
        levels = [self.stride4(feature), self.stride8(feature), self.stride16(feature), self.stride32(feature)]
        return [
            proj(F.interpolate(level, size=size, mode="bilinear", align_corners=False))
            for proj, level, size in zip(self.projections, levels, target_sizes)
        ]


class DINOv3UPerNet(nn.Module):
    """Combines a frozen DINOv3 backbone (Hugging Face) with a UPerNet decoder.

    Strategy:
    1. Extract a single-scale frozen feature map from DINOv3 via `AutoBackbone`
       (shape: [B, hidden_size, H', W'], CLS/register tokens already stripped).
    2. Turn it into a 4-level feature pyramid with `_ViTFeaturePyramid`
       (ViTDet-style "simple feature pyramid").
    3. Feed the pyramid into `UPerNetDecoder` + `SegmentationHead` to produce
       class logits, then resize to the exact input resolution.

    Args:
        num_classes: Number of output segmentation classes.
        dinov3_model_name: Hugging Face identifier (e.g., 'facebook/dinov3-vitl14-pretrained').
            Short model IDs are also supported for known variants, including:
            - 'dinov3-vitl16-pretrain-sat493m'
            - 'dinov3-vit7b16-pretrain-sat493m'
        in_channels: Input channels (must match DINOv3 input, usually 3).
        decoder_channels: Decoder hidden channels, and the uniform channel width
            used across all 4 feature-pyramid levels (default: 256).
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
        """Infer patch size from model identifier, defaulting to 14. Informational only —
        the feature pyramid interpolates to exact target sizes regardless of patch size."""
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
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dinov3_model_name = self._resolve_model_name(dinov3_model_name)
        self.in_channels = in_channels
        self.decoder_channels = decoder_channels
        self.out_channels = num_classes

        # Load frozen DINOv3 backbone from Hugging Face; a single (last-stage) feature map,
        # already reshaped to (B, hidden_size, H', W') with CLS/register tokens stripped.
        self.dinov3_backbone = AutoBackbone.from_pretrained(
            self.dinov3_model_name, out_indices=(-1,)
        )
        # Keep backward-compatible attribute name used in some tests/integrations.
        self.backbone = self.dinov3_backbone
        self.hidden_size = self.dinov3_backbone.config.hidden_size

        if freeze_backbone:
            for param in self.dinov3_backbone.parameters():
                param.requires_grad = False
            self.dinov3_backbone.eval()

        self.patch_size = self._infer_patch_size(self.dinov3_model_name)

        self.feature_pyramid = _ViTFeaturePyramid(
            in_channels=self.hidden_size, out_channels=decoder_channels
        )
        # First two entries are placeholders UPerNetDecoder discards internally
        # (it only uses encoder_channels[2:], the 1/4-1/32 stages).
        self.decoder = UPerNetDecoder(
            encoder_channels=[decoder_channels] * 6,
            encoder_depth=5,
            decoder_channels=decoder_channels,
        )
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels,
            out_channels=num_classes,
            kernel_size=1,
            upsampling=4,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: DINOv3 features -> feature pyramid -> UPerNet decoder -> head.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Logits [B, num_classes, H, W].
        """
        _, _, h, w = x.shape

        with torch.no_grad():
            feature = self.dinov3_backbone(x).feature_maps[-1]

        target_sizes = [
            (max(h // 4, 1), max(w // 4, 1)),
            (max(h // 8, 1), max(w // 8, 1)),
            (max(h // 16, 1), max(w // 16, 1)),
            (max(h // 32, 1), max(w // 32, 1)),
        ]
        pyramid = self.feature_pyramid(feature, target_sizes)
        # Placeholder slots for the decoder's unused 1/1 and 1/2 stages.
        decoder_out = self.decoder([pyramid[0], pyramid[0], *pyramid])
        logits = self.segmentation_head(decoder_out)

        if logits.shape[-2:] != (h, w):
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)

        return logits


__all__ = ["DINOv3UPerNet"]
