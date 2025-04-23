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
        norm_momentum: float = 0.1,  # Add momentum parameter
        bn: bool = True,
        relu: bool = True,
        dropout_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.bn = bn
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else None

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            **kwargs,
        )
        self.batchnorm = norm_layer(
            num_features=out_channels, momentum=norm_momentum
        )  # Pass momentum

    def forward(self, inputs: torch.Tensor):
        x = self.conv(inputs)
        if self.bn:
            x = self.batchnorm(x)
        if self.relu:
            x = self.relu(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int] | List[tuple] = [3, 3, 1],
        norm_layer: nn.Module = nn.BatchNorm2d,
        norm_momentum: float = 0.1,  # Add momentum parameter
        **kwargs,
    ):
        super().__init__()
        self.kernel_sizes = kernel_sizes

        # If in_channels is different from out_channels, we need a projection shortcut
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                relu=False,
                bn=True,
                norm_layer=norm_layer,
                norm_momentum=norm_momentum,  # Pass momentum
                **kwargs,
            )

        # Define convolutional layers using the momentum
        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_sizes[0],
            relu=True,
            bn=True,
            norm_layer=norm_layer,
            norm_momentum=norm_momentum,  # Pass momentum
            **kwargs,
        )
        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_sizes[1],
            relu=True,
            bn=True,
            norm_layer=norm_layer,
            norm_momentum=norm_momentum,  # Pass momentum
            **kwargs,
        )
        self.conv3 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_sizes[2],
            relu=False,
            bn=True,
            norm_layer=norm_layer,
            norm_momentum=norm_momentum,  # Pass momentum
            **kwargs,
        )
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor):
        shortcut = self.shortcut(inputs)  # Apply shortcut projection if needed

        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)  # Output of conv blocks (BN applied, no ReLU yet)

        x = x + shortcut  # Add the shortcut connection
        x = self.ReLU(x)  # Apply final ReLU activation
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
        norm_momentum: float = 0.1,  # Add momentum parameter
        bn: bool = True,
        relu: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.bn = bn
        self.relu = relu

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,  # Corrected for depthwise convolution
            groups=in_channels,  # Added groups for depthwise
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            **kwargs,
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,  # Corrected padding
        )
        self.batchnorm = norm_layer(
            num_features=out_channels, momentum=norm_momentum
        )  # Pass momentum
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        if self.bn:
            x = self.batchnorm(x)
        if self.relu:
            x = self.ReLU(x)
        return x
