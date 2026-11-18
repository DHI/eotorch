from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution with bias."""
    return nn.Conv2d(
        in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=True
    )


def _drop_unused_shortcut_keys(module, state_dict, prefix, *args) -> None:
    """Load-state-dict pre-hook dropping shortcut weights of identity-shortcut blocks.

    The reference implementation always creates ``conv_shortcut``/``bn_shortcut`` even when
    the identity shortcut is used, so its checkpoints contain these (unused) weights.
    """
    for key in list(state_dict):
        if key.startswith((f"{prefix}conv_shortcut.", f"{prefix}bn_shortcut.")):
            del state_dict[key]


class _ResidualBlock(nn.Module):
    """Shortcut handling shared by the Xception blocks.

    Unlike the reference, the projection shortcut is only created when it is used, so
    there are no parameters without gradients (which would require
    ``find_unused_parameters=True`` under DDP). Reference checkpoints still load.
    """

    def _make_shortcut(self, in_channels: int, out_channels: int, norm_layer) -> None:
        self.identity_shortcut = in_channels == out_channels
        if self.identity_shortcut:
            self._register_load_state_dict_pre_hook(_drop_unused_shortcut_keys, with_module=True)
        else:
            self.conv_shortcut = conv1x1(in_channels, out_channels)
            self.bn_shortcut = norm_layer(out_channels)

    def _shortcut(self, x: torch.Tensor) -> torch.Tensor:
        if self.identity_shortcut:
            return x
        return self.bn_shortcut(self.conv_shortcut(x))


class SeparableConv2d(nn.Module):
    """Depthwise separable convolution (depthwise followed by pointwise), without bias."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class PointwiseBlock(_ResidualBlock):
    """Entry block: three 1x1 conv + BN layers with a residual connection.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    filters : list of int
        Output channels of the three 1x1 convolutions.
    norm_layer : type of nn.Module, optional
        Normalization layer. Defaults to ``nn.BatchNorm2d``.
    """

    def __init__(self, in_channels: int, filters: list[int], norm_layer=nn.BatchNorm2d):
        super().__init__()
        if len(filters) != 3:
            raise ValueError("filters requires a list of exactly three values.")
        self.in_channels = in_channels
        self.filters = filters

        self.conv1 = conv1x1(in_channels, filters[0])
        self.bn1 = norm_layer(filters[0])
        self.conv2 = conv1x1(filters[0], filters[1])
        self.bn2 = norm_layer(filters[1])
        self.conv3 = conv1x1(filters[1], filters[2])
        self.bn3 = norm_layer(filters[2])
        self.relu = nn.ReLU(inplace=True)
        self._make_shortcut(in_channels, filters[2], norm_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self._shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = out + shortcut
        return self.relu(out)


class SepConvBlock(_ResidualBlock):
    """Residual block: (ReLU, SepConv 3x3, BN) x 2 plus shortcut.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    filters : list of int
        Output channels of the two separable convolutions.
    norm_layer : type of nn.Module, optional
        Normalization layer. Defaults to ``nn.BatchNorm2d``.
    """

    def __init__(self, in_channels: int, filters: list[int], norm_layer=nn.BatchNorm2d):
        super().__init__()
        if len(filters) != 2:
            raise ValueError("filters requires a list of exactly two values.")
        self.in_channels = in_channels
        self.filters = filters

        self.sepconv1 = SeparableConv2d(in_channels, filters[0], kernel_size=3)
        self.bn1 = norm_layer(filters[0])
        self.sepconv2 = SeparableConv2d(filters[0], filters[1], kernel_size=3)
        self.bn2 = norm_layer(filters[1])
        # Not inplace: the block input is also used by the shortcut.
        self.relu = nn.ReLU(inplace=False)
        self._make_shortcut(in_channels, filters[1], norm_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self._shortcut(x)

        out = self.bn1(self.sepconv1(self.relu(x)))
        out = self.bn2(self.sepconv2(self.relu(out)))

        return out + shortcut


def ELUplus1(x: torch.Tensor) -> torch.Tensor:
    """ELU shifted by +1, a smooth positive activation for variances."""
    return nn.functional.elu(x) + 1.0


def clamp_exp(x: torch.Tensor, min_x: float = -100, max_x: float = 10) -> torch.Tensor:
    """Exponential with clamped input, for log-variance outputs."""
    return torch.exp(torch.clamp(x, min=min_x, max=max_x))


class XceptionSDB(nn.Module):
    """
    A fully convolutional network for pixel-wise regression on Sentinel-2 images.

    "XceptionS2" builds on the separable convolution described by Chollet (2017) who proposed the Xception
    network. Any kind of down sampling is avoided (no pooling, striding, etc.), so every block runs at full
    patch resolution with ``num_sepconv_filters`` channels; reduce ``num_sepconv_filters`` (the global canopy
    height model uses 256), ``num_sepconv_blocks`` or the patch/batch size if memory is tight.

    Mirrors ``XceptionS2`` from https://github.com/langnico/global-canopy-height-model without the
    geographic prior. Adapted from:
    Lang, N., Schindler, K., Wegner, J.D.: Country-wide high-resolution vegetation height mapping with Sentinel-2,
    Remote Sensing of Environment, vol. 233 (2019) <https://arxiv.org/abs/1904.13270>
    Lang, N., Jetz, W., Schindler, K., & Wegner, J. D. (2023). A high-resolution canopy height model of the Earth.
    Nature Ecology & Evolution <https://arxiv.org/abs/2204.08322>

    Parameters:
        in_channels (int):
            Number of input channels.
        out_channels (int, optional):
            Number of output channels (set >1 for multi-task learning). Defaults to 1.
        num_sepconv_blocks (int, optional):
            Number of separable convolution blocks. Defaults to 8.
        num_sepconv_filters (int, optional):
            Number of filters in the separable convolution blocks. Defaults to 728.
        returns (str, optional):
            What ``forward`` returns. Defaults to 'targets'.

            - 'targets': the predictions (means) only.
            - 'variances_exp': ``(predictions, variances)`` with the variance head predicting
              log-variances, passed through a clamped exponential (used for the global
              canopy height model).
            - 'variances': ``(predictions, variances)`` with ``var_activation`` applied to the
              variance head.
            - 'var_from_second_moments': ``(predictions, variances, second_moments)`` with
              variances derived as ``second_moments - predictions**2``.
        var_activation (str, optional):
            Activation applied to the variances for 'variances' and 'var_from_second_moments'.
            One of 'relu', 'elu' (ELU + 1) or 'exp'. Defaults to 'relu'.
        min_var (float, optional):
            Shift the output variance by adding ``min_var`` (not applied for 'variances_exp').
            Defaults to 0.
        detach_var_input (bool, optional):
            Detach the graph before computing the variance. Defaults to False.
        long_skip (bool, optional):
            Add a long skip (residual) connection from the entry block features to the last
            features. Defaults to False.
        manual_init (bool, optional):
            Xavier-uniform init of convolutions and unit/zero init of batch norms. Defaults to False.
        freeze_features (bool, optional):
            Freeze the feature extractor; only the last linear layers (unless frozen below) are trained.
            Defaults to False.
        freeze_last_mean (bool, optional):
            With ``freeze_features``, also freeze the prediction (mean) layer. Defaults to False.
        freeze_last_var (bool, optional):
            With ``freeze_features``, also freeze the variance layer. Defaults to False.
        model_weights_path (Path | str | None, optional):
            Path to pretrained weights used to initialize; a reference checkpoint with a
            'model_state_dict' key or a plain state dict.
    """

    RETURNS = ("targets", "variances_exp", "variances", "var_from_second_moments")

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        num_sepconv_blocks: int = 8,
        num_sepconv_filters: int = 728,
        returns: str = "targets",
        var_activation: str = "relu",
        min_var: float = 0.0,
        detach_var_input: bool = False,
        long_skip: bool = False,
        manual_init: bool = False,
        freeze_features: bool = False,
        freeze_last_mean: bool = False,
        freeze_last_var: bool = False,
        model_weights_path: Path | str | None = None,
    ):
        super().__init__()

        var_activations = {
            "relu": nn.functional.relu,
            "elu": ELUplus1,
            "exp": clamp_exp,
        }
        if returns not in self.RETURNS:
            raise ValueError(f"returns must be one of {self.RETURNS}, got {returns!r}.")
        if var_activation not in var_activations:
            raise ValueError(
                f"var_activation must be one of {sorted(var_activations)}, got {var_activation!r}."
            )

        self.num_sepconv_blocks = num_sepconv_blocks
        self.num_sepconv_filters = num_sepconv_filters
        self.returns = returns
        self.min_var = min_var
        self.detach_var_input = detach_var_input
        self.long_skip = long_skip
        self.freeze_features = freeze_features
        self.freeze_last_mean = freeze_last_mean
        self.freeze_last_var = freeze_last_var
        self.var_activation = var_activations[var_activation]

        self.entry_block = PointwiseBlock(in_channels, filters=[128, 256, num_sepconv_filters])
        self.sepconv_blocks = self._make_sepconv_blocks()

        self.predictions = conv1x1(num_sepconv_filters, out_channels)
        self.variances = conv1x1(num_sepconv_filters, out_channels)
        self.second_moments = conv1x1(num_sepconv_filters, out_channels)

        if manual_init:
            self._manual_init()

        if freeze_features:
            for param in self.parameters():
                param.requires_grad = False
            if not freeze_last_mean:
                for param in self.predictions.parameters():
                    param.requires_grad = True
            if not freeze_last_var:
                for param in self.variances.parameters():
                    param.requires_grad = True

        if model_weights_path is not None:
            print(f"Loading pretrained model weights from: {model_weights_path}")
            self._load_model_weights(model_weights_path)

    def forward(self, x: torch.Tensor):
        x = self.entry_block(x)
        if self.long_skip:
            shortcut = x
        x = self.sepconv_blocks(x)
        if self.long_skip:
            x = x + shortcut
        predictions = self.predictions(x)

        if self.returns == "targets":
            return predictions

        if self.returns == "variances_exp":
            return predictions, clamp_exp(self.variances(x))

        if self.detach_var_input:
            x = x.detach()

        if self.returns == "variances":
            return predictions, self._constrain_variances(self.variances(x))

        # var_from_second_moments
        second_moments = self.second_moments(x)
        variances = self._constrain_variances(second_moments - predictions**2)
        return predictions, variances, second_moments

    def _make_sepconv_blocks(self) -> nn.Sequential:
        return nn.Sequential(
            *[
                SepConvBlock(
                    self.num_sepconv_filters,
                    filters=[self.num_sepconv_filters, self.num_sepconv_filters],
                )
                for _ in range(self.num_sepconv_blocks)
            ]
        )

    def _manual_init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _constrain_variances(self, variances: torch.Tensor) -> torch.Tensor:
        return self.var_activation(variances) + self.min_var

    def _load_model_weights(self, model_weights_path: Path | str) -> None:
        checkpoint = torch.load(model_weights_path, map_location="cpu")
        self.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
