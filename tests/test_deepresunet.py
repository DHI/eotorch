"""Tests for the DeepResUNet model."""

import pytest
import torch

from eotorch.models import DeepResUNet


@pytest.mark.parametrize("static_filters", [True, False])
def test_forward_shape(static_filters):
    model = DeepResUNet(
        in_channels=6, num_classes=7, num_filters=32, static_filters=static_filters
    )
    x = torch.randn(2, 6, 64, 64)
    out = model(x)
    assert out.shape == (2, 7, 64, 64)
    out.sum().backward()


def test_pyramid_filter_widths():
    model = DeepResUNet(in_channels=3, num_filters=64, static_filters=False)
    assert model.num_filters == [8, 16, 32, 64]
    # Full-resolution decoder concatenates upsampled (16) + skip (8) channels
    assert model.decoder4.conv.conv.in_channels == 16 + 8
    assert model.decoder4.conv.conv.out_channels == 8


def test_static_filters_state_dict_shapes_unchanged():
    """Static-filter decoders keep the 2 * in_channels concat conv, so old checkpoints load."""
    model = DeepResUNet(in_channels=3, num_filters=16)
    for name in ["decoder1", "decoder2", "decoder3", "decoder4"]:
        conv = getattr(model, name).conv.conv
        assert conv.weight.shape == (16, 32, 1, 1)
