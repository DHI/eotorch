from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torchgeo.trainers import (
    SemanticSegmentationTask as TorchGeoSemanticSegmentationTask,
)
from torchvision.models._api import WeightsEnum

from eotorch.models import MODEL_MAPPING
from eotorch.utils import get_init_args


class SemanticSegmentationTask(TorchGeoSemanticSegmentationTask):
    """
    Task based on TorchGeo's SemanticSegmentationTask, but with the ability to use custom models.
    We still have access to all the architectures and backbones from TorchGeo.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        num_filters: int = 128,
        model: str = "deepresunet",
        backbone: str = "resnet50",
        weights: WeightsEnum | str | bool | None = None,
        loss: str = "ce",
        class_weights: Tensor | None = None,
        ignore_index: int | None = None,
        lr: float = 1e-3,
        patience: int = 10,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        model_kwargs: dict[str, Any] = None,
    ):
        self.model_kwargs = model_kwargs or {}
        super().__init__(
            model=model,
            backbone=backbone,
            weights=weights,
            in_channels=in_channels,
            num_classes=num_classes,
            num_filters=num_filters,
            loss=loss,
            class_weights=class_weights,
            ignore_index=ignore_index,
            lr=lr,
            patience=patience,
            freeze_backbone=freeze_backbone,
            freeze_decoder=freeze_decoder,
        )

    def configure_models(self) -> None:
        model: str = self.hparams["model"]
        weights = self.weights

        if model.lower() in MODEL_MAPPING:
            model_cls = MODEL_MAPPING[model]

            init_args = get_init_args(model_cls)
            for param in init_args:
                if (
                    param in self.hparams
                ):  # all the args from __init__ are in hparams, like num_classes, in_channels, etc.
                    self.model_kwargs[param] = self.hparams[param]

            print(f"Initializing model {model} with kwargs {self.model_kwargs}")
            self.model: nn.Module = model_cls(**self.model_kwargs)

            if weights:
                if isinstance(weights, (str, Path)):
                    weights = torch.load(weights)
                self.model.load_state_dict(weights)
                print(f"Weights loaded successfully from {weights}")

        else:
            super().configure_models()

    def training_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute the training loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            The loss tensor.
        """

        x = batch["image"]
        if (x == self.hparams["ignore_index"]).all():
            return None
        y = batch["mask"]
        batch_size = x.shape[0]
        y_hat = self(x)
        loss: Tensor = self.criterion(y_hat, y)

        self.log("train_loss", loss, batch_size=batch_size, prog_bar=True)
        self.train_metrics(y_hat, y)
        self.log_dict(self.train_metrics, batch_size=batch_size)
        return loss
