"""
Tests for DINOv3 + UPerNet integration with PatchSegmentationTask.

Tests that:
1. DINOv3UPerNet model instantiates correctly
2. PatchSegmentationTask can use it as a model
3. Backbone is frozen, decoder is trainable
4. Forward pass produces correct output shapes, with DINOv3 features actually
   driving the decoder (not a discarded, unused side-computation)
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from eotorch.data import PatchSegmentationTask
from eotorch.models import DINOv3UPerNet, SEG_MODEL_MAPPING


class _FakeDinoBackbone(nn.Module):
    """Stand-in for a real HF `AutoBackbone`: same call signature/output shape
    (`.feature_maps`, a list with one (B, hidden_size, H', W') tensor), but tiny
    and with a real trainable parameter so freeze/train behavior is genuinely testable.
    """

    def __init__(self, hidden_size: int = 8, feat_hw: tuple[int, int] = (4, 4)):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.hidden_size = hidden_size
        self.feat_hw = feat_hw
        self.dummy_param = nn.Parameter(torch.randn(hidden_size))

    def forward(self, x, **kwargs):
        b = x.shape[0]
        feat = torch.randn(b, self.hidden_size, *self.feat_hw) + self.dummy_param.view(1, -1, 1, 1)
        return SimpleNamespace(feature_maps=[feat])


def _patch_autobackbone(hidden_size: int = 8, feat_hw: tuple[int, int] = (4, 4)):
    return patch(
        "eotorch.models.dinov3_upernet.AutoBackbone.from_pretrained",
        return_value=_FakeDinoBackbone(hidden_size=hidden_size, feat_hw=feat_hw),
    )


class TestDINOv3UPerNetModel:
    """Test DINOv3UPerNet model instantiation and forward pass."""

    def test_dinov3_upernet_instantiation(self):
        """Test that DINOv3UPerNet can be instantiated."""
        with _patch_autobackbone():
            model = DINOv3UPerNet(
                num_classes=3,
                dinov3_model_name="facebook/dinov3-vitl14-pretrained",
                in_channels=3,
                decoder_channels=16,
                freeze_backbone=True,
            )
        assert model is not None
        assert model.num_classes == 3
        assert model.out_channels == 3

    def test_dinov3_upernet_in_seg_mapping(self):
        """Test that DINOv3UPerNet is registered in SEG_MODEL_MAPPING."""
        assert "dinov3_upernet" in SEG_MODEL_MAPPING
        assert SEG_MODEL_MAPPING["dinov3_upernet"] == DINOv3UPerNet

    def test_dinov3_upernet_backbone_frozen(self):
        """Backbone parameters should have requires_grad=False and be in eval mode."""
        with _patch_autobackbone():
            model = DINOv3UPerNet(
                num_classes=3,
                dinov3_model_name="facebook/dinov3-vitl14-pretrained",
                decoder_channels=16,
                freeze_backbone=True,
            )

        assert not model.dinov3_backbone.training
        assert all(not p.requires_grad for p in model.dinov3_backbone.parameters())

    def test_dinov3_upernet_freeze_backbone_false(self):
        """Backbone parameters should stay trainable when freeze_backbone=False."""
        with _patch_autobackbone():
            model = DINOv3UPerNet(
                num_classes=3,
                dinov3_model_name="facebook/dinov3-vitl14-pretrained",
                decoder_channels=16,
                freeze_backbone=False,
            )

        assert all(p.requires_grad for p in model.dinov3_backbone.parameters())

    def test_dinov3_upernet_decoder_trainable(self):
        """Decoder/head/feature-pyramid parameters should be trainable regardless of freeze_backbone."""
        with _patch_autobackbone():
            model = DINOv3UPerNet(
                num_classes=3,
                dinov3_model_name="facebook/dinov3-vitl14-pretrained",
                freeze_backbone=True,
                decoder_channels=16,
            )

        trainable_params = sum(
            p.numel()
            for module in (model.feature_pyramid, model.decoder, model.segmentation_head)
            for p in module.parameters()
            if p.requires_grad
        )
        assert trainable_params > 0, "Decoder path should have trainable parameters"


class TestDINOv3UPerNetOutputShapes:
    """Test that output shapes are correct and DINOv3 features genuinely drive them."""

    @pytest.mark.parametrize("input_size", [(224, 224), (200, 200)])
    def test_dinov3_upernet_output_shape_multiclass(self, input_size):
        """Output shape matches input resolution, for sizes divisible and not divisible by patch_size."""
        with _patch_autobackbone(hidden_size=8, feat_hw=(4, 4)):
            model = DINOv3UPerNet(
                num_classes=5,
                dinov3_model_name="facebook/dinov3-vitl14-pretrained",
                decoder_channels=16,
            )
        model.eval()

        x = torch.randn(2, 3, *input_size)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 5, *input_size), f"Got {output.shape}"

    def test_dinov3_upernet_output_shape_binary(self):
        """Test output shape for binary segmentation."""
        with _patch_autobackbone(hidden_size=8, feat_hw=(4, 4)):
            model = DINOv3UPerNet(
                num_classes=1,
                dinov3_model_name="facebook/dinov3-vitl14-pretrained",
                decoder_channels=16,
            )
        model.eval()

        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 1, 224, 224), f"Got {output.shape}"

    def test_dinov3_upernet_gradients_flow_to_decoder_only(self):
        """Backward pass should populate grads on the decoder path but never on the frozen backbone."""
        with _patch_autobackbone(hidden_size=8, feat_hw=(4, 4)):
            model = DINOv3UPerNet(
                num_classes=2,
                dinov3_model_name="facebook/dinov3-vitl14-pretrained",
                decoder_channels=16,
                freeze_backbone=True,
            )

        x = torch.randn(2, 3, 64, 64)  # batch_size > 1: BatchNorm needs >1 sample per channel in train mode
        output = model(x)
        output.sum().backward()

        assert any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.segmentation_head.parameters()
        )
        assert model.dinov3_backbone.dummy_param.grad is None

    def test_dinov3_upernet_output_depends_on_backbone_features(self):
        """The decoder output must actually depend on the DINOv3 feature map, not just on
        raw pixels — this is the property that was broken when features were discarded."""
        with _patch_autobackbone(hidden_size=8, feat_hw=(4, 4)):
            model = DINOv3UPerNet(
                num_classes=2,
                dinov3_model_name="facebook/dinov3-vitl14-pretrained",
                decoder_channels=16,
            )
        model.eval()

        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            baseline = model(x).clone()

            # Perturbing only the backbone's output (not x) must change the final logits.
            original_forward = model.dinov3_backbone.forward

            def _perturbed_forward(inp, **kwargs):
                out = original_forward(inp, **kwargs)
                return SimpleNamespace(feature_maps=[fm + 5.0 for fm in out.feature_maps])

            model.dinov3_backbone.forward = _perturbed_forward
            perturbed = model(x)

        assert not torch.allclose(baseline, perturbed)


class TestDINOv3UPerNetPatchSegmentationIntegration:
    """Test integration of DINOv3UPerNet with PatchSegmentationTask."""

    @patch("eotorch.data.tasks.get_init_args")
    def test_patch_segmentation_task_uses_dinov3_upernet(self, mock_get_init_args):
        """Test that PatchSegmentationTask can instantiate DINOv3UPerNet."""
        mock_get_init_args.return_value = [
            "num_classes",
            "in_channels",
            "decoder_channels",
            "freeze_backbone",
        ]

        with _patch_autobackbone(hidden_size=8, feat_hw=(4, 4)):
            model = PatchSegmentationTask(
                num_classes=3,
                in_channels=3,
                model="dinov3_upernet",
                num_filters=16,
                freeze_backbone=True,
                loss="dice",
                lr=1e-4,
            )

        assert model.model is not None
        assert isinstance(model.model, DINOv3UPerNet)
        assert model.model.num_classes == 3

    @patch("eotorch.data.tasks.get_init_args")
    def test_patch_segmentation_task_binary_dinov3_upernet(self, mock_get_init_args):
        """Test binary segmentation with DINOv3+UPerNet."""
        mock_get_init_args.return_value = [
            "num_classes",
            "in_channels",
            "decoder_channels",
            "freeze_backbone",
        ]

        with _patch_autobackbone(hidden_size=8, feat_hw=(4, 4)):
            model = PatchSegmentationTask(
                num_classes=1,
                in_channels=3,
                task="binary",
                model="dinov3_upernet",
                loss="bce",
                lr=1e-4,
            )

        assert hasattr(model, "train_metrics")
        assert hasattr(model, "val_metrics")
        assert "mIoU" in model.train_metrics
        assert "F1_Score" in model.train_metrics

    @patch("eotorch.data.tasks.get_init_args")
    def test_patch_segmentation_task_multiclass_dinov3_upernet(self, mock_get_init_args):
        """Test multiclass segmentation with DINOv3+UPerNet."""
        mock_get_init_args.return_value = [
            "num_classes",
            "in_channels",
            "decoder_channels",
            "freeze_backbone",
        ]

        with _patch_autobackbone(hidden_size=8, feat_hw=(4, 4)):
            model = PatchSegmentationTask(
                num_classes=5,
                in_channels=3,
                task="multiclass",
                model="dinov3_upernet",
                loss="ce",
                lr=1e-4,
            )

        assert hasattr(model, "train_metrics")
        assert "mIoU" in model.train_metrics
        assert "F1_Score" in model.train_metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
