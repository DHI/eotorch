"""Tests for the models implemented in eotorch."""

import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from eotorch.models import DeepResUNet, DINOv3UPerNet, UNet


@pytest.mark.parametrize("static_filters", [True, False])
def test_deepresunet_output_shape(static_filters):
    model = DeepResUNet(
        in_channels=4, num_classes=3, num_filters=16, static_filters=static_filters
    )
    assert model(torch.randn(2, 4, 64, 64)).shape == (2, 3, 64, 64)


def test_unet_output_shape():
    model = UNet(in_channels=4, num_classes=2, num_filters=8, depth=3)
    assert model(torch.randn(2, 4, 64, 64)).shape == (2, 2, 64, 64)


class _FakeBackbone(nn.Module):
    """Stands in for a Hugging Face AutoBackbone with a 16 pixel patch size."""

    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.proj = nn.Conv2d(3, hidden_size, kernel_size=16, stride=16)

    def forward(self, x):
        return SimpleNamespace(feature_maps=[self.proj(x)])


@pytest.fixture
def fake_transformers(monkeypatch):
    auto_backbone = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: _FakeBackbone()
    )
    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(AutoBackbone=auto_backbone)
    )


@pytest.mark.parametrize("size", [64, 96, 100])
def test_dinov3_upernet_output_shape(fake_transformers, size):
    model = DINOv3UPerNet(in_channels=3, num_classes=4, decoder_channels=16)
    assert model(torch.randn(2, 3, size, size)).shape == (2, 4, size, size)


def test_dinov3_upernet_backbone_trains_unless_frozen(fake_transformers):
    model = DINOv3UPerNet(in_channels=3, num_classes=2, decoder_channels=16)
    model(torch.randn(2, 3, 64, 64)).sum().backward()
    assert model.encoder.proj.weight.grad is not None


def test_dinov3_upernet_requires_three_channels(fake_transformers):
    with pytest.raises(ValueError, match="3 input channels"):
        DINOv3UPerNet(in_channels=4, num_classes=2)
