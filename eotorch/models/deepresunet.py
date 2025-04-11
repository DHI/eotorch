import torch
import torch.nn as nn

from eotorch.models.layers import Conv2d, ResBlock


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.resblock1 = ResBlock(in_channels, out_channels, kernel_sizes=[3, 3, 1])
        self.resblock2 = ResBlock(out_channels, out_channels, kernel_sizes=[3, 3, 1])
        self.maxpool = nn.MaxPool2d((2, 2))

    def forward(self, inputs: torch.Tensor):
        x = self.resblock1(inputs)
        skip = self.resblock2(x)
        x = self.maxpool(skip)
        return x, skip


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv = Conv2d(
            2 * in_channels, out_channels, kernel_size=(1, 1), padding="valid"
        )
        self.resblock1 = ResBlock(in_channels, out_channels, kernel_sizes=[3, 3, 1])
        self.resblock2 = ResBlock(out_channels, out_channels, kernel_sizes=[3, 3, 1])

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor):
        x = self.upsample(inputs)
        x = torch.cat((x, skip), dim=-3)
        x = self.conv(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        return x


class RegDeepResUNet(nn.Module):
    """
    A fully convolutional neural network for regression tasks.

    DeepResUNet is adapated from the architecture proposed in:
    Yi, Y.; Zhang, Z.; Zhang, W.; Zhang, C.; Li, W.; Zhao, T.: Semantic Segmentation of Urban Buildings from VHR Remote
    Sensing Imagery Using a Deep Convolutional Neural Network.
    Remote Sens. 2019, 11, 1774. https://doi.org/10.3390/rs11151774.

    Parameters:
        in_channels (int):
            Number of input channels.
        num_filters (int, optional):
            Number of filters. Defaults to 128.
        static_filters (bool, optional):
            Keep the number of filters consistent for each layer. If False, the number of filters
            are doubled after each encoder block and halved again after each decoder block.
            Defaults to True.
    """

    def __init__(
        self,
        in_channels: int,
        num_filters: int = 128,
        static_filters: bool = True,
    ):
        super().__init__()
        self.num_filters = (
            [num_filters] * 4
            if static_filters
            else list(reversed([num_filters // (2**i) for i in range(4)]))
        )

        # Encoder
        self.input_conv = Conv2d(
            in_channels,
            self.num_filters[0],
            kernel_size=(5, 5),
            padding="same",
            stride=(1, 1),
        )
        self.maxpool = nn.MaxPool2d((2, 2))
        self.encoder1 = Encoder(self.num_filters[0], self.num_filters[1])
        self.encoder2 = Encoder(self.num_filters[1], self.num_filters[2])
        self.encoder3 = Encoder(self.num_filters[2], self.num_filters[3])

        # Bridge
        self.resblock1 = ResBlock(
            self.num_filters[3], self.num_filters[3], kernel_sizes=[3, 3, 1]
        )
        self.resblock2 = ResBlock(
            self.num_filters[3], self.num_filters[3], kernel_sizes=[3, 3, 1]
        )

        # Decoder
        self.decoder1 = Decoder(self.num_filters[3], self.num_filters[3])
        self.decoder2 = Decoder(self.num_filters[3], self.num_filters[2])
        self.decoder3 = Decoder(self.num_filters[2], self.num_filters[1])
        self.decoder4 = Decoder(self.num_filters[1], self.num_filters[0])

        self.output = nn.Conv2d(
            self.num_filters[0],
            1,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding="same",
        )
        self.activation = nn.Linear()

    def forward(self, inputs):
        skip1 = self.input_conv(inputs)
        x = self.maxpool(skip1)
        x, skip2 = self.encoder1(x)
        x, skip3 = self.encoder2(x)
        x, skip4 = self.encoder3(x)

        x = self.resblock1(x)
        x = self.resblock2(x)

        x = self.decoder1(x, skip4)
        x = self.decoder2(x, skip3)
        x = self.decoder3(x, skip2)
        x = self.decoder4(x, skip1)

        x = self.output(x)
        # x = self.activation(x)
        return x


class ClfDeepResUNet(nn.Module):
    """
    A fully convolutional neural network for semantic segmentation tasks.

    DeepResUNet is adapated from the architecture proposed in:
    Yi, Y.; Zhang, Z.; Zhang, W.; Zhang, C.; Li, W.; Zhao, T.: Semantic Segmentation of Urban Buildings from VHR Remote
    Sensing Imagery Using a Deep Convolutional Neural Network.
    Remote Sens. 2019, 11, 1774. https://doi.org/10.3390/rs11151774.

    Parameters:
        in_channels (int):
            Number of input channels.
        num_filters (int, optional):
            Number of self.num_filters. Defaults to 128.
        static_filters (bool, optional):
            Keep the number of self.num_filters consistent for each layer. If False, the number of self.num_filters
            are doubled after each encoder block and halved again after each decoder block.
            Defaults to True.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        num_filters: int = 128,
        static_filters: bool = True,
    ):
        super().__init__()
        self.num_filters = (
            [num_filters] * 4
            if static_filters
            else list(reversed([num_filters // (2**i) for i in range(4)]))
        )

        # Encoder
        self.input_conv = Conv2d(
            in_channels,
            self.num_filters[0],
            kernel_size=(5, 5),
            padding="same",
            stride=(1, 1),
        )
        self.maxpool = nn.MaxPool2d((2, 2))
        self.encoder1 = Encoder(self.num_filters[0], self.num_filters[1])
        self.encoder2 = Encoder(self.num_filters[1], self.num_filters[2])
        self.encoder3 = Encoder(self.num_filters[2], self.num_filters[3])

        # Bridge
        self.resblock1 = ResBlock(
            self.num_filters[3], self.num_filters[3], kernel_sizes=[3, 3, 1]
        )
        self.resblock2 = ResBlock(
            self.num_filters[3], self.num_filters[3], kernel_sizes=[3, 3, 1]
        )

        # Decoder
        self.decoder1 = Decoder(self.num_filters[3], self.num_filters[3])
        self.decoder2 = Decoder(self.num_filters[3], self.num_filters[2])
        self.decoder3 = Decoder(self.num_filters[2], self.num_filters[1])
        self.decoder4 = Decoder(self.num_filters[1], self.num_filters[0])

        self.output = nn.Conv2d(
            self.num_filters[0],
            num_classes,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding="same",
        )
        self.activation = nn.Softmax()

    def forward(self, inputs):
        skip1 = self.input_conv(inputs)
        x = self.maxpool(skip1)
        x, skip2 = self.encoder1(x)
        x, skip3 = self.encoder2(x)
        x, skip4 = self.encoder3(x)

        x = self.resblock1(x)
        x = self.resblock2(x)

        x = self.decoder1(x, skip4)
        x = self.decoder2(x, skip3)
        x = self.decoder3(x, skip2)
        x = self.decoder4(x, skip1)

        x = self.output(x)
        # x = self.activation(x)
        return x
