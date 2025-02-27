import torch.nn as nn

from eotorch.models.blocks import (
    PreActivateDoubleConv,
    PreActivateResBlock,
    PreActivateResUpBlock,
)


class DeepResUNet(nn.Module):
    def __init__(self, n_classes: int, n_channels: int, filters: int = 128):
        super(DeepResUNet, self).__init__()

        self.down_conv1 = PreActivateResBlock(n_channels, filters)
        self.down_conv2 = PreActivateResBlock(filters, filters)
        self.down_conv3 = PreActivateResBlock(filters, filters)
        self.down_conv4 = PreActivateResBlock(filters, filters)

        self.double_conv = PreActivateDoubleConv(filters, filters)

        self.up_conv4 = PreActivateResUpBlock(filters, filters)
        self.up_conv3 = PreActivateResUpBlock(filters, filters)
        self.up_conv2 = PreActivateResUpBlock(filters, filters)
        self.up_conv1 = PreActivateResUpBlock(filters, filters)

        self.conv_last = nn.Conv2d(filters, n_classes, kernel_size=1)

    def forward(self, x):
        x, skip1_out = self.down_conv1(x)
        x, skip2_out = self.down_conv2(x)
        x, skip3_out = self.down_conv3(x)
        x, skip4_out = self.down_conv4(x)
        x = self.double_conv(x)
        x = self.up_conv4(x, skip4_out)
        x = self.up_conv3(x, skip3_out)
        x = self.up_conv2(x, skip2_out)
        x = self.up_conv1(x, skip1_out)
        x = self.conv_last(x)
        return x
