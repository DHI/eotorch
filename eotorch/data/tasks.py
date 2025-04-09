from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchgeo.datasets import RGBBandsMissingError, unbind_samples
from torchgeo.trainers import (
    SemanticSegmentationTask as TorchGeoSemanticSegmentationTask,
)

from eotorch.models import MODEL_MAPPING
from eotorch.utils import get_init_args

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from torch import Tensor, nn
    from torchvision.models._api import WeightsEnum


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
        y = batch["mask"].squeeze()

        if ignore_index := self.hparams["ignore_index"]:
            # filter out patches with only ignore_index in the mask
            valid_patches = ~(y == ignore_index).all(dim=(-2, -1)).squeeze()
            # Skip processing if all pixels in the patch are ignore_index
            if not valid_patches.any():
                return None
            x = x[valid_patches]
            y = y[valid_patches]

        if x.ndim > 4:
            x = x.squeeze(0)

        print(np.unique(y.cpu().numpy(), return_counts=True))
        batch_size = x.shape[0]
        y_hat = self(x)
        print(
            np.unique(
                np.argmax(y_hat.detach().cpu().numpy(), axis=-3).astype("uint8"),
                return_counts=True,
            )
        )
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
        y = batch["mask"].squeeze()

        if ignore_index := self.hparams["ignore_index"]:
            valid_patches = ~(y == ignore_index).all(dim=(-2, -1)).squeeze()
            if not valid_patches.any():
                return None
            x = x[valid_patches]
            y = y[valid_patches]

        if x.ndim > 4:
            x = x.squeeze(0)

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

            # Get prediction and ensure it's in the right format
            # Use dimension 1 for the argmax to operate on the channel dimension
            # but preserve the batch dimension
            predictions = y_hat.argmax(dim=1)

            # Create a new batch dictionary with the valid subset
            valid_batch = {
                "image": x.cpu(),
                "mask": y.unsqueeze(1).cpu(),  # Add channel dimension back
                "prediction": predictions.unsqueeze(1)
                .detach()
                .cpu(),  # Add channel dimension to match mask
            }

            # Only sample from first example for visualization
            sample_idx = 0
            sample = unbind_samples(valid_batch)[sample_idx]

            # Verify that sample has a prediction and remove extra dimension for plotting
            if "prediction" in sample and sample["prediction"] is not None:
                # Remove the channel dimension for plotting
                sample["prediction"] = sample["prediction"].squeeze(0)

            else:
                print(f"Warning: Sample {batch_idx} missing prediction key")

            fig: Figure | None = None
            try:
                fig = datamodule.plot(sample)
            except RGBBandsMissingError:
                print(f"RGB bands missing in batch {batch_idx}, skipping visualization")
                pass
            except Exception as e:
                print(f"Error plotting sample from batch {batch_idx}: {e}")
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
        y = batch["mask"].squeeze()
        if ignore_index := self.hparams["ignore_index"]:
            valid_patches = ~(y == ignore_index).all(dim=(-2, -1)).squeeze()
            if not valid_patches.any():
                return None
            x = x[valid_patches]
            y = y[valid_patches]

        if x.ndim > 4:
            x = x.squeeze(0)

        batch_size = x.shape[0]
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss, batch_size=batch_size)
        self.test_metrics(y_hat, y)
        self.log_dict(self.test_metrics, batch_size=batch_size)

    def get_prediction_func(
        self, device=None, reduce_zero_label: bool = None
    ) -> Callable:
        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cuda = device.type == "cuda"

        in_channels = self.hparams.get("in_channels")
        if hasattr(self, "datamodule_params"):
            reduce_zero_label = self.datamodule_params.get("dataset_args", {}).get(
                "reduce_zero_label"
            )
            print(
                "Found reduce_zero_label = True in datamodule_params. Adding 1 to class predictions."
            )
        else:
            reduce_zero_label = reduce_zero_label or False

        def predict(image: np.ndarray) -> np.ndarray:
            in_channels_checked = False
            image = torch.from_numpy(image)
            if image.ndim == 3:
                image = image.unsqueeze(0)

            if in_channels and not in_channels_checked:
                if image.shape[1] != in_channels:
                    raise ValueError(
                        f"Input image has {image.shape[1]} channels, but model expects {in_channels}."
                    )
                in_channels_checked = True
            if cuda:
                image = image.cuda()
            with torch.inference_mode():
                batch_logits = self(image.float()).cpu().numpy()
                class_pred = np.argmax(batch_logits, axis=-3).astype("uint8")
                if reduce_zero_label:
                    class_pred += 1

                return class_pred

        return predict

    @staticmethod
    def predict_on_tif_file(
        tif_file_path: str | Path,
        checkpoint_path: str | Path,
        patch_size: int = 64,
        overlap: int = 2,
        class_mapping: dict[int, str] = None,
        func_supports_batching: bool = True,
        batch_size: int = 8,
        out_file_path: str | Path = None,
        show_results: bool = False,
        ax: plt.Axes = None,
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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lightning_module = SemanticSegmentationTask.load_from_checkpoint(
            checkpoint_path,
            map_location=device,
        )
        data_module_params = torch.load(
            checkpoint_path, weights_only=False, map_location=device
        ).get("datamodule_hyper_parameters")
        if data_module_params:
            lightning_module.datamodule_params = data_module_params
            dataset_args = data_module_params.get("dataset_args", {})
            class_mapping = class_mapping or dataset_args.get("class_mapping")
            patch_size = data_module_params.get("patch_size") or patch_size

        return predict_on_tif(
            tif_file_path=tif_file_path,
            prediction_func=lightning_module.get_prediction_func(
                device=device,
                reduce_zero_label=True,
            ),
            patch_size=patch_size,
            overlap=overlap,
            class_mapping=class_mapping,
            func_supports_batching=func_supports_batching,
            batch_size=batch_size,
            out_file_path=out_file_path,
            show_results=show_results,
            ax=ax,
        )
