from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import torch.nn as nn

if TYPE_CHECKING:
    import torch


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple | int = (3, 3),
        stride: Tuple | int = 1,
        padding: Tuple | int | str = "same",
        norm_layer: nn.Module = nn.BatchNorm2d,
        bn: bool = True,
        relu: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.bn = bn
        self.relu = nn.ReLU(inplace=True) if relu else None

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            **kwargs,
        )
        self.batchnorm = norm_layer(num_features=out_channels)

    def forward(self, inputs: torch.Tensor):
        x = self.conv(inputs)
        if self.bn:
            x = self.batchnorm(x)
        if self.relu:
            x = self.relu(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int] | List[tuple] = [3, 3, 1],
        norm_layer: nn.Module = nn.BatchNorm2d,
        **kwargs,
    ):
        super().__init__()
        self.kernel_sizes = kernel_sizes

        self.conv1 = Conv2d(
            in_channels, out_channels, kernel_size=kernel_sizes[0], **kwargs
        )
        self.conv2 = Conv2d(
            out_channels, out_channels, kernel_size=kernel_sizes[1], **kwargs
        )
        self.conv3 = Conv2d(
            out_channels, out_channels, kernel_size=kernel_sizes[2], **kwargs
        )
        self.ReLU = nn.ReLU(inplace=True)
        self.batchnorm = norm_layer(num_features=out_channels)

    def forward(self, inputs: torch.Tensor):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + self.batchnorm(inputs)
        x = self.ReLU(x)
        return x


class SeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple | int = (3, 3),
        stride: Tuple | int = 1,
        padding: Tuple | int | str = "same",
        norm_layer: nn.Module = nn.BatchNorm2d,
        bn: bool = True,
        relu: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.bn = bn
        self.relu = relu

        self.depthwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            **kwargs,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=(1, 1), stride=1, padding=1
        )
        self.batchnorm = norm_layer(num_features=out_channels)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        if self.bn:
            x = self.batchnorm(x)
        if self.relu:
            x = self.ReLU(x)
        return x
