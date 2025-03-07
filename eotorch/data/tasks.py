from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch import Tensor, nn
from torchgeo.datasets import RGBBandsMissingError, unbind_samples
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

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Compute the validation loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        if (x == self.hparams["ignore_index"]).all():
            return None

        y = batch["mask"]
        batch_size = x.shape[0]
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, batch_size=batch_size)
        self.val_metrics(y_hat, y)
        self.log_dict(self.val_metrics, batch_size=batch_size)

        if (
            batch_idx < 10
            and hasattr(self.trainer, "datamodule")
            and hasattr(self.trainer.datamodule, "plot")
            and self.logger
            and hasattr(self.logger, "experiment")
            and hasattr(self.logger.experiment, "add_figure")
        ):
            datamodule = self.trainer.datamodule
            batch["prediction"] = y_hat.argmax(dim=1)
            for key in ["image", "mask", "prediction"]:
                batch[key] = batch[key].cpu()
            sample = unbind_samples(batch)[0]

            fig: Figure | None = None
            try:
                fig = datamodule.plot(sample)
            except RGBBandsMissingError:
                pass

            if fig:
                summary_writer = self.logger.experiment
                summary_writer.add_figure(
                    f"image/{batch_idx}", fig, global_step=self.global_step
                )
                plt.close()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        if (x == self.hparams["ignore_index"]).all():
            return None

        y = batch["mask"]
        batch_size = x.shape[0]
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss, batch_size=batch_size)
        self.test_metrics(y_hat, y)
        self.log_dict(self.test_metrics, batch_size=batch_size)

    @staticmethod
    def get_prediction_func(checkpoint_path: str | Path) -> Callable:
        lightning_module = SemanticSegmentationTask.load_from_checkpoint(
            checkpoint_path
        )

        def predict(image: np.ndarray) -> np.ndarray:
            image = torch.from_numpy(image)
            if image.ndim == 3:
                image = image.unsqueeze(0)

            image = image.permute(0, -1, 1, 2)
            with torch.no_grad():
                return lightning_module(image).cpu().numpy()

        return predict

    @staticmethod
    def predict_on_tif_file(
        tif_file_path: str | Path,
        weights_path: str | Path,
        patch_size: int = 64,
        overlap: int = 2,
        class_mapping: dict[int, str] = None,
        func_supports_batching: bool = True,
        batch_size: int = 8,
        out_file_path: str | Path = None,
        show_results: bool = False,
        ax: plt.Axes = None,
        argmax_dim: int = -3,
    ):
        """
        Use a trained model to predict segmentation classes on a TIF file

        Parameters
        ----------
        tif_file_path : str | Path
            Path to the input TIF file.
        weights_path : str | Path
            Path to the model weights used for prediction.
        patch_size : int
            Integer size of the patch to use for prediction.
        overlap : int
            Overlap factor between patches (larger values increase overlap).
        class_mapping : dict[int, str], optional
            Mapping from predicted class indices to class names for visualization. Used for plotting only.
        func_supports_batching : bool
            Whether the prediction_func supports batched processing.
        batch_size : int
            The batch size used for prediction (ignored if func_supports_batching is False).
        out_file_path : str | Path, optional
            Output path for saving the results. Writes to a "predictions" subfolder if None.
        show_results : bool
            If True, display the prediction output in a notebook environment.
        ax : plt.Axes, optional
            Matplotlib Axes object for plotting if show_results is True.

        Returns
        -------
        Path
            The path to the produced TIF file or plot visualization (when show_results=True).
        """
        from eotorch.inference import predict_on_tif

        return predict_on_tif(
            tif_file_path=tif_file_path,
            prediction_func=SemanticSegmentationTask.get_prediction_func(weights_path),
            patch_size=patch_size,
            overlap=overlap,
            class_mapping=class_mapping,
            func_supports_batching=func_supports_batching,
            batch_size=batch_size,
            out_file_path=out_file_path,
            show_results=show_results,
            ax=ax,
            argmax_dim=argmax_dim,
        )
