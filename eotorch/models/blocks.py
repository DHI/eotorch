import torch
import torch.nn as nn


class PreActivateResUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(PreActivateResUpBlock, self).__init__()
        self.ch_avg = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.up_sample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.ch_avg = nn.Sequential(
            nn.Conv2d(
                2 * in_channels, out_channels, kernel_size=1, stride=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.double_conv = PreActivateDoubleConv(
            2 * in_channels, out_channels, kernel_size
        )

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x) + self.ch_avg(x)


class PreActivateResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(PreActivateResBlock, self).__init__()
        self.ch_avg = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.double_conv = PreActivateDoubleConv(in_channels, out_channels, kernel_size)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        identity = self.ch_avg(x)
        out = self.double_conv(x)
        out = out + identity
        return self.down_sample(out), out


class PreActivateDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(PreActivateDoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1),
        )

    def forward(self, x):
        return self.double_conv(x)
