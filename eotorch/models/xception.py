from pathlib import Path
from typing import List

import torch
import torch.nn as nn

from eotorch.models.layers import Conv2d, SeparableConv2d


class PointwiseBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_filters: List[int],
    ):
        assert len(num_filters) == 3, (
            "num_filters requires a list of exactly three values."
        )
        super().__init__()

        self.in_channels = in_channels
        self.num_filters = num_filters

        self.conv1 = Conv2d(in_channels, num_filters[0], kernel_size=(1, 1), relu=True)
        self.conv2 = Conv2d(in_channels, num_filters[1], kernel_size=(1, 1), relu=True)
        self.conv3 = Conv2d(in_channels, num_filters[2], kernel_size=(1, 1), relu=False)
        self.conv_shortcut = Conv2d(
            in_channels, num_filters[2], kernel_size=(1, 1), relu=False
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor):
        if self.in_channels == self.num_filters[-1]:
            shortcut = inputs
        else:
            shortcut = self.conv_shortcut(inputs)

        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + shortcut
        x = self.relu(x)
        return x


class SepConvBlock(nn.Module):
    def __init__(self, in_channels: int, num_filters: int):
        super().__init__()

        self.in_channels = in_channels
        self.num_filters = num_filters

        self.relu = nn.ReLU(inplace=True)
        self.sepconv1 = SeparableConv2d(in_channels, num_filters)
        self.sepconv2 = SeparableConv2d(num_filters, num_filters, relu=False)
        self.conv_shortcut = Conv2d(
            in_channels, num_filters, kernel_size=(1, 1), relu=False
        )

    def forward(self, inputs: torch.Tensor):
        if self.in_channels == self.num_filters[-1]:
            # identity shortcut
            shortcut = inputs
        else:
            shortcut = self.conv_shortcut(inputs)

        x = self.relu(inputs)
        x = self.sepconv1(x)
        x = self.sepconv2(x)
        x = x + shortcut
        return x


class AuxModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_filters: int,
    ):
        self.residual = Conv2d(
            in_channels, num_filters, kernel_size=(1, 1), bn=False, bias=True
        )
        self.reslayer1 = ResLayer(num_filters, num_filters)
        self.reslayer2 = ResLayer(num_filters, num_filters)
        self.reslayer3 = ResLayer(num_filters, num_filters)
        self.reslayer4 = ResLayer(num_filters, num_filters)
        self.output = nn.Conv2d(num_filters, 1, kernel_size=(1, 1), padding="same")

    def forward(self, inputs: torch.Tensor):
        x = self.residual(inputs)
        x = self.reslayer1(x)
        x = self.reslayer2(x)
        x = self.reslayer3(x)
        x = self.reslayer4(x)
        x = self.output(x)
        return x


class ResLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = Conv2d(
            in_channels, out_channels, kernel_size=(1, 1), bn=False, bias=True
        )
        self.dropout = nn.Dropout(0.5)
        self.conv2 = Conv2d(
            out_channels, out_channels, kernel_size=(1, 1), bn=False, bias=True
        )

    def forward(self, inputs: torch.Tensor):
        x = self.conv1(inputs)
        x = self.dropout(x)
        x = self.conv2(x)
        x = x + inputs
        return x


def ELUplus1(x: torch.Tensor):
    elu = nn.ELU(inplace=False)(x)
    return torch.add(elu, 1.0)


def clamp_exp(x: torch.Tensor, min_x: float = -100, max_x: float = 10):
    x = torch.clamp(x, min=min_x, max=max_x)
    return torch.exp(x)


class XceptionSDB(nn.Module):
    """
    A custom fully convolutional neural network designed for pixel-wise analysis of Sentinel-2 satellite images.

    "XceptionSDB" builds on the separable convolution described by Chollet (2017) who proposed the Xception network.
    Any kind of down sampling is avoided (no pooling, striding, etc.).

    This architecture is adapted from:
    Lang, N., Schindler, K., Wegner, J.D.: Country-wide high-resolution vegetation height mapping with Sentinel-2,
    Remote Sensing of Environment, vol. 233 (2019) <https://arxiv.org/abs/1904.13270>
    Lang, N., Jetz, W., Schindler, K., & Wegner, J. D. (2022). A high-resolution canopy height model of the Earth.
    arXiv preprint arXiv:2204.08322.

    Parameters:
        in_channels (int):
            Number of input channels
        num_filters (int, optional):
            Number of filters. Defaults to 128.
        n_sepconv_filters (int, optional):
            Number of filters in the separable convolutional layers. Defaults to 728.
        sepconv_blocks (int, optional):
            Number of separable convolution blocks. Defaults to 8.
        aux_layers (int, optional):
            Number of auxiliary layers from the input. Note that auxiliary layers are read as the last
            `aux_layers` layers of the input. Defaults to 0.
        aux_filters (int, optional):
            Number of filters in the auxiliary model.
        long_skip (bool, optional):
            Add a long skip (residual) connection from the entry block to the last feature.
        returns (str, optional):
            The return type. Either 'targets' for mean predictions only or 'variances' for both means
            and variances. Defaults to 'variances'.
        detach_variance (bool, optional):
            Detach the graph before computing the variance. Defaults to False.
        min_variance (float, optional):
            Shift the output variance by `min_var`. Defaults to 0..
        trainable (bool, optional):
            All parameters in the model can be trained. If False, only the parameters in the last output
            layer(s) can be trained. Used for finetuning results. Defaults to True.
        var_activation (str, optional):
            Activation function applied to the variance. One of 'relu', 'elu', or 'exp'. Defaults to 'relu'.
        model_weights_path (Path | str | None, optional):
            Path to load pretrained model weights used to initialize.

    """

    def __init__(
        self,
        in_channels: int,
        num_filters: int = 128,
        n_sepconv_filters: int = 728,
        sepconv_blocks: int = 8,
        aux_layers: int = 0,
        aux_filters: int = 128,
        long_skip: bool = False,
        returns: str = "variances",
        detach_variance: bool = False,
        min_variance: float = 0.0,
        trainable: bool = True,
        var_activation: str = "relu",
        model_weights_path: Path | str | None = None,
    ):
        super().__init__()

        assert n_sepconv_filters > filters * 2, (
            "The separable convolution filter size in \
            is smaller than the final filter size in the point-wise block."
        )
        filters = [num_filters, num_filters * 2, n_sepconv_filters]

        activations = {
            "relu": nn.ReLU(inplace=True),
            "elu": ELUplus1,
            "exp": clamp_exp,
        }

        self.n_sepconv_filters = n_sepconv_filters
        self.sepconv_blocks = sepconv_blocks
        self.aux_layers = aux_layers

        self.long_skip = long_skip
        self.returns = returns
        self.detach_variance = detach_variance
        self.min_variance = min_variance
        self.var_activation = activations[var_activation]

        if self.aux_layers > 0:
            self.aux_model = AuxModel(aux_layers, aux_filters)

        # Convolutional layers
        self.entry_block = PointwiseBlock(
            in_channels, filters=[num_filters, num_filters * 2, n_sepconv_filters]
        )
        self.sepconvblock = self._make_sepconv_blocks()
        self.mean = Conv2d(
            n_sepconv_filters, 1, kernel_size=(1, 1), relu=False, bn=False
        )
        self.variance = Conv2d(
            n_sepconv_filters, 1, kernel_size=(1, 1), relu=False, bn=False
        )

        if not trainable:
            for param in self.parameters():
                param.requires_grad = False
            for param in self.aux_model.parameters():
                param.requires_grad = False

            # unfreeze the last layer(s) of the regressor
            for param_mean, param_var in zip(
                self.mean.parameters(), self.variance.parameters()
            ):
                param_mean.requires_grad = True
                param_var.requires_grad = True

        if model_weights_path is not None:
            print("Loading pretrained model weights from:")
            print(model_weights_path)
            self._load_model_weights(Path(model_weights_path))

    def forward(self, inputs: torch.Tensor):
        if self.aux_layers:
            aux_inputs = inputs[:, -self.aux_layers :, ...]
            aux_output = self.aux_model(aux_inputs)
            x = inputs[:, : self.aux_layers, ...]
        else:
            x = inputs

        x = self.entry_block(x)
        if self.long_skip:
            shortcut = x
        x = self.sepconvblock(x)
        if self.long_skip:
            x = x + shortcut

        mean = self.mean(x)
        if self.aux_layers:
            mean = mean + aux_output

        if self.returns == "targets":
            return mean
        elif self.returns == "variances":
            if self.detach_variance:
                x = x.detach()
            variance = self.variance(x)
            variance = self._constrain_variance(variance)
            return mean, variance

    def _constrain_variance(self, variance: torch.Tensor):
        variance = self.var_activation(variance)
        variance = variance + self.min_variance
        return variance

    def _make_sepconv_blocks(self):
        blocks = []
        for _ in range(self.sepconv_blocks):
            blocks.append(SepConvBlock(self.n_sepconv_filters, self.n_sepconv_filters))

        return nn.Sequential(*blocks)

    def _load_model_weights(self, model_weights_path: Path):
        checkpoint = torch.load(model_weights_path)
        model_weights = checkpoint["model_state_dict"]
        self.load_state_dict(model_weights)
