from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.decoders.upernet.decoder import UPerNetDecoder


class _ViTFeaturePyramid(nn.Module):
    """ViTDet-style "simple feature pyramid" (Li et al., ECCV 2022).

    Resamples the single-scale feature map of a ViT into the four levels
    (1/4, 1/8, 1/16 and 1/32 of the input) that a CNN decoder such as UPerNet expects.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.resamplers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2),
                    nn.GELU(),
                    nn.ConvTranspose2d(in_channels // 2, in_channels // 4, 2, stride=2),
                ),
                nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2),
                nn.Identity(),
                nn.MaxPool2d(2, stride=2),
            ]
        )
        self.projections = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(c, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            for c in [in_channels // 4, in_channels // 2, in_channels, in_channels]
        )

    def forward(self, feature: torch.Tensor, input_size: tuple[int, int]) -> list:
        h, w = input_size
        levels = []
        for i, (resample, project) in enumerate(zip(self.resamplers, self.projections)):
            # The ViT patch size does not always divide the input into powers of two
            size = (max(h // 2 ** (i + 2), 1), max(w // 2 ** (i + 2), 1))
            level = F.interpolate(resample(feature), size=size, mode="bilinear")
            levels.append(project(level))
        return levels


class DINOv3UPerNet(nn.Module):
    """DINOv3 backbone (from Hugging Face) with a UPerNet decoder for segmentation.

    The backbone is available as ``encoder``, so it can be frozen with the
    ``freeze_backbone`` option of the eotorch tasks, which is recommended. It expects
    3-band input, normalized the way the chosen DINOv3 weights were trained.
    Requires the ``transformers`` package.

    Parameters:
        in_channels (int):
            Number of input channels. Must be 3.
        num_classes (int):
            Number of output channels.
        model_name (str, optional):
            Hugging Face model id of the DINOv3 backbone.
        decoder_channels (int, optional):
            Number of channels in the feature pyramid and decoder. Defaults to 256.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        model_name: str = "facebook/dinov3-vitl16-pretrain-sat493m",
        decoder_channels: int = 256,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoBackbone
        except ImportError as e:
            raise ImportError(
                "DINOv3UPerNet requires the transformers package: pip install transformers"
            ) from e
        if in_channels != 3:
            raise ValueError(f"DINOv3 expects 3 input channels, got {in_channels}.")

        # Last-stage feature map, reshaped to (B, hidden_size, H', W')
        self.encoder = AutoBackbone.from_pretrained(model_name, out_indices=(-1,))
        self.feature_pyramid = _ViTFeaturePyramid(
            self.encoder.config.hidden_size, decoder_channels
        )
        # UPerNetDecoder ignores its first two (1/1 and 1/2 resolution) inputs
        self.decoder = UPerNetDecoder(
            encoder_channels=[decoder_channels] * 6,
            encoder_depth=5,
            decoder_channels=decoder_channels,
        )
        self.segmentation_head = SegmentationHead(
            decoder_channels, num_classes, kernel_size=1, upsampling=4
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        feature = self.encoder(x).feature_maps[-1]
        pyramid = self.feature_pyramid(feature, input_size)
        logits = self.segmentation_head(
            self.decoder([pyramid[0], pyramid[0], *pyramid])
        )
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear")
        return logits
