from pathlib import Path

import pytest
import torch
from torch import nn

import eotorch.data.tasks as tasks_module
from eotorch.data.tasks import RegressionTask


class DummyRegModel(nn.Module):
    def __init__(self, in_channels: int, num_outputs: int = 1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_outputs, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DummyTrainMetrics(nn.Module):
    def forward(self, preds: torch.Tensor, target: torch.Tensor):
        return {"train/MSE": torch.tensor(0.0)}


class DummySmpUnet(nn.Module):
    def __init__(self, in_channels: int, classes: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(in_channels, 4, kernel_size=1))
        self.decoder = nn.Sequential(nn.Conv2d(4, 4, kernel_size=1))
        self.head = nn.Conv2d(4, classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return self.head(x)


def test_regression_task_training_step_runs_with_local_model_mapping(monkeypatch):
    """RegressionTask should use local model mapping and run a train step."""
    monkeypatch.setitem(tasks_module.REG_MODEL_MAPPING, "dummyreg", DummyRegModel)

    task = RegressionTask(
        in_channels=3,
        num_outputs=1,
        model="dummyreg",
        loss="mse",
    )
    task.train_metrics = DummyTrainMetrics()

    batch = {
        "image": torch.randn(2, 3, 8, 8),
        "mask": torch.randn(2, 8, 8),
    }

    loss = task.training_step(batch, batch_idx=0)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_regression_predict_on_tif_file_uses_checkpoint_patch_size(monkeypatch):
    """predict_on_tif_file should forward patch size from checkpoint datamodule params."""

    called = {}

    class DummyLoadedModule:
        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def predict_value(self, batch):
            return batch

    def fake_load_from_checkpoint(path, map_location=None, **kwargs):
        called["load_kwargs"] = kwargs
        return DummyLoadedModule()

    def fake_torch_load(path, weights_only=False, map_location=None):
        return {
            "hyper_parameters": {
                "in_channels": 3,
                "num_outputs": 1,
                "model": "deepresunet",
                "loss": "mse",
                "lr": 1e-3,
            },
            "datamodule_hyper_parameters": {"patch_size": 64},
            "state_dict": {},
        }

    def fake_predict_on_tif_generic(**kwargs):
        called["predict_kwargs"] = kwargs
        return Path("predictions/out.tif")

    monkeypatch.setattr(RegressionTask, "load_from_checkpoint", staticmethod(fake_load_from_checkpoint))
    monkeypatch.setattr(tasks_module.torch, "load", fake_torch_load)

    import eotorch.inference.inference as inference_module

    monkeypatch.setattr(inference_module, "predict_on_tif_generic", fake_predict_on_tif_generic)

    output = RegressionTask.predict_on_tif_file(
        tif_file_path="input.tif",
        checkpoint_path="model.ckpt",
        batch_size=2,
        progress_bar=False,
    )

    assert output == Path("predictions/out.tif")
    assert called["predict_kwargs"]["patch_size"] == 64
    assert called["predict_kwargs"]["batch_size"] == 2
    assert "in_channels" in called["load_kwargs"]
    assert "num_outputs" in called["load_kwargs"]


def test_regression_predict_on_tif_file_requires_model_hparams(monkeypatch):
    """predict_on_tif_file should fail when required regression hparams are missing."""

    def fake_torch_load(path, weights_only=False, map_location=None):
        return {
            "hyper_parameters": {},
            "datamodule_hyper_parameters": {"patch_size": 64},
            "state_dict": {},
        }

    monkeypatch.setattr(tasks_module.torch, "load", fake_torch_load)

    with pytest.raises(ValueError, match="Missing required model hyperparameters"):
        RegressionTask.predict_on_tif_file(
            tif_file_path="input.tif",
            checkpoint_path="model.ckpt",
            progress_bar=False,
        )


def test_regression_task_supports_smp_unet(monkeypatch):
    """RegressionTask should support SMP UNet fallback and freeze flags."""
    called = {}

    def fake_unet(encoder_name, encoder_weights, in_channels, classes):
        called["encoder_name"] = encoder_name
        called["encoder_weights"] = encoder_weights
        called["in_channels"] = in_channels
        called["classes"] = classes
        return DummySmpUnet(in_channels=in_channels, classes=classes)

    monkeypatch.setattr(tasks_module.smp, "Unet", fake_unet)

    task = RegressionTask(
        in_channels=3,
        num_outputs=2,
        model="unet",
        backbone="resnet18",
        weights=True,
        freeze_backbone=True,
        freeze_decoder=True,
    )

    assert called["encoder_name"] == "resnet18"
    assert called["encoder_weights"] == "imagenet"
    assert called["in_channels"] == 3
    assert called["classes"] == 2
    assert all(not p.requires_grad for p in task.model.encoder.parameters())
    assert all(not p.requires_grad for p in task.model.decoder.parameters())
