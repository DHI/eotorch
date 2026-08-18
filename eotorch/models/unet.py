import torch
import torch.nn as nn

from eotorch.models.layers import Conv2d


class DoubleConv(nn.Module):
    """Two consecutive 3x3 conv-BN-ReLU blocks."""

    def __init__(self, in_channels: int, out_channels: int, norm_momentum: float = 0.1):
        super().__init__()
        self.conv1 = Conv2d(
            in_channels, out_channels, kernel_size=3, padding="same", norm_momentum=norm_momentum
        )
        self.conv2 = Conv2d(
            out_channels, out_channels, kernel_size=3, padding="same", norm_momentum=norm_momentum
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""

    def __init__(self, in_channels: int, out_channels: int, norm_momentum: float = 0.1):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, norm_momentum=norm_momentum)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(self.maxpool(inputs))


class Up(nn.Module):
    """Upscaling then double conv, consuming the matching encoder skip connection."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        norm_momentum: float = 0.1,
    ):
        super().__init__()
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv = DoubleConv(
            in_channels + skip_channels, out_channels, norm_momentum=norm_momentum
        )

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(inputs)
        x = torch.cat((x, skip), dim=-3)
        return self.conv(x)


class UNetBody(nn.Module):
    """Shared encoder/decoder used by :class:`ClfUNet` and :class:`RegUNet`.

    A plain (non-residual) UNet where the filter count at the first stage is set by
    ``num_filters`` and doubles at every one of the ``depth`` downsampling steps, then
    halves back down through the same number of upsampling steps. Every channel count in
    the network is derived from ``num_filters``, so the total parameter count can be
    reduced arbitrarily by lowering ``num_filters`` (and/or ``depth``) -- unlike an
    ImageNet-pretrained encoder such as smp's UNet with a ResNet18 backbone, which has a
    fixed ~12M parameters.
    """

    def __init__(
        self,
        in_channels: int,
        num_filters: int = 32,
        depth: int = 4,
        norm_momentum: float = 0.1,
    ):
        super().__init__()
        filters = [num_filters * 2**i for i in range(depth + 1)]

        self.in_conv = DoubleConv(in_channels, filters[0], norm_momentum=norm_momentum)
        self.downs = nn.ModuleList(
            Down(filters[i], filters[i + 1], norm_momentum=norm_momentum) for i in range(depth)
        )
        self.ups = nn.ModuleList(
            Up(filters[i + 1], filters[i], filters[i], norm_momentum=norm_momentum)
            for i in reversed(range(depth))
        )
        self.out_channels = filters[0]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        skips = [self.in_conv(inputs)]
        for down in self.downs:
            skips.append(down(skips[-1]))

        x = skips[-1]
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(x, skip)
        return x


class UNet(nn.Module):
    """
    A lightweight UNet for segmentation and regression tasks with a fully
    configurable filter count.

    Unlike smp's UNet with a pretrained ResNet encoder (~12M parameters even with the
    smallest ResNet18 backbone), the size of this network is entirely controlled by
    ``num_filters`` (and ``depth``), making it practical to shrink well below that for
    small datasets or limited compute.

    Parameters:
        in_channels (int):
            Number of input channels.
        num_classes (int, optional):
            Number of output channels. Defaults to 1, suitable for single-target
            regression. Set explicitly for segmentation (number of classes) or
            multi-output regression.
        num_filters (int, optional):
            Number of filters at the first encoder / last decoder stage. Doubles at
            every downsampling step and halves back down through the decoder.
            Defaults to 32.
        depth (int, optional):
            Number of downsampling/upsampling steps. Defaults to 4.
        norm_momentum (float, optional):
            Momentum for normalization layers. Defaults to 0.1.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 1,
        num_filters: int = 32,
        depth: int = 4,
        norm_momentum: float = 0.1,
    ):
        super().__init__()
        self.body = UNetBody(
            in_channels, num_filters=num_filters, depth=depth, norm_momentum=norm_momentum
        )
        self.output = nn.Conv2d(self.body.out_channels, num_classes, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.body(inputs)
        return self.output(x)
