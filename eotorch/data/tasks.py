from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import segmentation_models_pytorch as smp
import torch
from lightning import LightningModule
from matplotlib import pyplot as plt
from torchgeo.datasets import RGBBandsMissingError, unbind_samples
from torchgeo.datasets.utils import array_to_tensor
from torchgeo.trainers import (
    SemanticSegmentationTask as TorchGeoSemanticSegmentationTask,
    RegressionTask as TorchGeoRegressionTask,
)
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassJaccardIndex,
)
from torchmetrics.wrappers import ClasswiseWrapper

from eotorch.models import CLF_MODEL_MAPPING, REG_MODEL_MAPPING
from eotorch.utils import get_init_args

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from torch import Tensor, nn
    from torchvision.models._api import WeightsEnum


class SemanticSegmentationTask(TorchGeoSemanticSegmentationTask):
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
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        model_kwargs: dict[str, Any] = None,
        lr_scheduler: dict[str, Any] = None,
        class_names: list[str] | None = None,
    ):
        """
        Task based on TorchGeo's SemanticSegmentationTask, but with the ability to use custom models.
        We still have access to all the architectures and backbones from TorchGeo.

        Args:
            num_classes: Number of output classes for segmentation.
            in_channels: Number of input channels.
            num_filters: Number of filters in the model.
            model: Model architecture to use (e.g., 'deepresunet', 'unet').
            backbone: Backbone network for encoder-decoder architectures.
            weights: Pretrained weights to load.
            loss: Loss function ('ce', 'bce', 'jaccard', 'focal', 'dice').
            class_weights: Optional weights for each class in the loss.
            ignore_index: Label index to ignore in loss and metrics.
            lr: Learning rate.
            freeze_backbone: Whether to freeze the backbone.
            freeze_decoder: Whether to freeze the decoder.
            model_kwargs: Additional keyword arguments for model initialization.
            lr_scheduler: The learning rate scheduler configuration. If None, a default scheduler will be used.
                The default scheduler is ReduceLROnPlateau with the following parameters:
                * mode: "min"
                * factor: 0.2
                * patience: 10
                * min_lr: 1e-6
                * monitor: "val_loss"

                The way to specify the default scheduler would be:
                lr_scheduler = {"type": "ReduceLROnPlateau", "mode": "min", "factor": 0.2, "patience": 10, "min_lr": 1e-6, "monitor": "val_loss"}

                For available schedulers, see https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate

                Some examples of other schedulers:
                * CosineAnnealingLR:
                    lr_scheduler = {"type": "CosineAnnealingLR", "T_max": 100, "eta_min": 1e-6}
            class_names: List of class names for metric labeling. If None, uses numeric indices.

        """
        self.model_kwargs = model_kwargs or {}
        self.lr_scheduler = lr_scheduler or {}
        self.class_names = class_names
        self.save_hyperparameters()
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
            freeze_backbone=freeze_backbone,
            freeze_decoder=freeze_decoder,
        )

    def configure_models(self) -> None:
        model: str = self.hparams["model"]
        weights = self.weights

        if model.lower() in CLF_MODEL_MAPPING:
            model_cls = CLF_MODEL_MAPPING[model]

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

    def configure_losses(self) -> None:
        """Initialize the loss criterion."""
        ignore_index: int | None = self.hparams["ignore_index"]
        match self.hparams["loss"]:
            case "ce":
                from torch import nn

                ignore_value = -1000 if ignore_index is None else ignore_index
                self.criterion: nn.Module = nn.CrossEntropyLoss(
                    ignore_index=ignore_value, weight=self.hparams["class_weights"]
                )
            case "bce":
                from torch import nn

                self.criterion = nn.BCEWithLogitsLoss()
            case "jaccard":
                # JaccardLoss requires a list of classes to use instead of a class
                # index to ignore.
                classes = [
                    i for i in range(self.hparams["num_classes"]) if i != ignore_index
                ]

                self.criterion = smp.losses.JaccardLoss(
                    mode=self.hparams["task"], classes=classes
                )
            case "focal":
                self.criterion = smp.losses.FocalLoss(
                    mode=self.hparams["task"],
                    ignore_index=ignore_index,
                    normalized=True,
                    alpha=0.8,
                    gamma=3.0,
                )
            case "dice":
                self.criterion = smp.losses.DiceLoss(
                    mode=self.hparams["task"],
                    ignore_index=ignore_index,
                    from_logits=True,
                )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams["lr"],
            weight_decay=1e-4,  # Add some regularization
        )
        monitor = None
        if not self.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=10, min_lr=1e-6, factor=0.2
            )

        else:
            scheduler_type = self.lr_scheduler.pop("type")
            if hasattr(torch.optim.lr_scheduler, scheduler_type):
                scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_type)
                if "monitor" in self.lr_scheduler:
                    monitor = self.lr_scheduler.pop("monitor")
                scheduler = scheduler_cls(optimizer, **self.lr_scheduler)
            else:
                raise ValueError(
                    f"Scheduler {scheduler_type} not found in torch.optim.lr_scheduler"
                )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": monitor or self.monitor,
            },
        }

    def configure_metrics(self) -> None:
        """Initialize the performance metrics.

        Uses metric naming consistent with TerraTorch for easier comparison:
        * mIoU: Mean Intersection over Union (macro average)
        * F1_Score: Macro-averaged F1 score
        * Accuracy: Macro-averaged accuracy
        * Pixel_Accuracy: Micro-averaged accuracy (per-pixel)
        * IoU_<class>: Per-class IoU using ClasswiseWrapper
        * Class_Accuracy_<class>: Per-class accuracy using ClasswiseWrapper

        .. note::
           * 'Micro' averaging gives equal weight to each pixel
           * 'Macro' averaging gives equal weight to each class
        """
        num_classes = self.hparams["num_classes"]
        ignore_index = self.hparams["ignore_index"]
        class_names = self.hparams.get("class_names") or [
            str(i) for i in range(num_classes)
        ]

        metrics = MetricCollection(
            {
                # Macro-averaged metrics (consistent with TerraTorch naming)
                "mIoU": MulticlassJaccardIndex(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="macro",
                ),
                "F1_Score": MulticlassF1Score(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="macro",
                ),
                "Accuracy": MulticlassAccuracy(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="macro",
                ),
                # Micro-averaged (pixel-level) accuracy
                "Pixel_Accuracy": MulticlassAccuracy(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="micro",
                ),
                # Per-class metrics using ClasswiseWrapper (like TerraTorch)
                "IoU": ClasswiseWrapper(
                    MulticlassJaccardIndex(
                        num_classes=num_classes,
                        ignore_index=ignore_index,
                        average=None,
                    ),
                    labels=class_names,
                    prefix="IoU_",
                ),
                "Class_Accuracy": ClasswiseWrapper(
                    MulticlassAccuracy(
                        num_classes=num_classes,
                        ignore_index=ignore_index,
                        average=None,
                    ),
                    labels=class_names,
                    prefix="Class_Accuracy_",
                ),
                "Class_F1": ClasswiseWrapper(
                    MulticlassF1Score(
                        num_classes=num_classes,
                        ignore_index=ignore_index,
                        average=None,
                    ),
                    labels=class_names,
                    prefix="Class_F1_",
                ),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

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

        self.log("input_std", x.std(), on_step=True, on_epoch=False, prog_bar=False)

        batch_size = x.shape[0]
        y_hat = self(x)
        y_hat_hard = y_hat.argmax(dim=1)
        loss: Tensor = self.criterion(y_hat, y)

        self.log(
            "train_loss", loss, batch_size=batch_size, prog_bar=True, on_epoch=True
        )

        # Update metrics (TerraTorch pattern: update in step, log at epoch end)
        self.train_metrics.update(y_hat_hard, y)

        return loss

    def on_train_epoch_end(self) -> None:
        """Called at the end of the training epoch to log accumulated metrics."""
        # Compute and log all metrics at epoch end (TerraTorch pattern)
        computed = self.train_metrics.compute()
        self.log_dict(computed, on_epoch=True, prog_bar=False)
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        """Called at the end of the validation epoch to log accumulated metrics."""
        # Compute and log all metrics at epoch end (TerraTorch pattern)
        computed = self.val_metrics.compute()
        self.log_dict(computed, on_epoch=True, prog_bar=False)
        self.val_metrics.reset()

    def on_test_epoch_end(self) -> None:
        """Called at the end of the test epoch to log accumulated metrics."""
        # Compute and log all metrics at epoch end (TerraTorch pattern)
        computed = self.test_metrics.compute()
        self.log_dict(computed, on_epoch=True, prog_bar=False)
        self.test_metrics.reset()

    def validation_step(
        self, batch: Any, batch_idx: int = None, dataloader_idx: int = 0
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
        y_hat_hard = y_hat.argmax(dim=1)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, batch_size=batch_size)

        # Update metrics (TerraTorch pattern: update in step, log at epoch end)
        self.val_metrics.update(y_hat_hard, y)

        if (
            batch_idx < 10
            and hasattr(self.trainer, "datamodule")
            and hasattr(self.trainer.datamodule, "plot")
            and self.logger
            and hasattr(self.logger, "experiment")
            and hasattr(self.logger.experiment, "add_figure")
        ):
            datamodule = self.trainer.datamodule

            # Create a new batch dictionary with the valid subset
            valid_batch = {
                "image": x.cpu(),
                "mask": y.unsqueeze(1).cpu(),  # Add channel dimension back
                "prediction": y_hat_hard.unsqueeze(1)
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
        y_hat_hard = y_hat.argmax(dim=1)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss, batch_size=batch_size)

        # Update metrics (TerraTorch pattern: update in step, log at epoch end)
        self.test_metrics.update(y_hat_hard, y)

    def predict_step(self, batch: Tensor) -> Tensor:
        """Compute the predicted class probabilities.

        Args:
            batch: The output of your DataLoader.

        Returns:
            Output predicted probabilities.
        """
        # x = batch["image"]

        with torch.inference_mode():
            y_hat: Tensor = self(batch)

            match self.hparams["task"]:
                case "binary" | "multilabel":
                    y_hat = y_hat.sigmoid()
                case "multiclass":
                    y_hat = y_hat.softmax(dim=1)

            return y_hat
            # return y_hat.cpu().numpy()

    def predict_class(self, batch: Tensor | np.ndarray) -> np.ndarray:
        if isinstance(batch, np.ndarray):
            batch = array_to_tensor(batch)
        batch = batch.to(self.device)
        probs = self.predict_step(batch)
        preds = probs.argmax(dim=1).cpu().numpy().astype("uint8")
        if hasattr(self, "reduce_zero_label") and self.reduce_zero_label:
            preds += 1

        return preds

    @staticmethod
    def predict_on_tif_file(
        tif_file_path: str | Path,
        checkpoint_path: str | Path,
        func_supports_batching: bool = True,
        batch_size: int = 8,
        out_file_path: str | Path = None,
        show_results: bool = False,
        ax: plt.Axes = None,
        progress_bar: bool = True,
        patch_size: int = None,
        class_mapping: dict[int, str] = None,
    ):
        """
        Use a trained model to predict segmentation classes on a TIF file

        Parameters
        ----------
        tif_file_path : str | Path
            Path to the input TIF file.
        weights_path : str | Path
            Path to the model weights used for prediction.
        overlap : int
            Overlap factor between patches (larger values increase overlap).
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
        patch_size : int
            Integer size of the patch to use for prediction. Will be read from the checkpoint by default. Should only
            be set if the checkpoint does not contain the patch size.
        class_mapping : dict[int, str], optional
            Mapping from predicted class indices to class names for visualization. Used for plotting only.
            Will be read from the checkpoint if by default. Should only be set if the checkpoint does not contain the class mapping.

        Returns
        -------
        Path
            The path to the produced TIF file or plot visualization (when show_results=True).
        """
        from eotorch.inference.inference import predict_on_tif_generic

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        lightning_module = SemanticSegmentationTask.load_from_checkpoint(
            checkpoint_path,
            map_location=device,
        )
        lightning_module.to(device)
        lightning_module.eval()
        data_module_params = torch.load(
            checkpoint_path, weights_only=False, map_location=device
        ).get("datamodule_hyper_parameters")
        if data_module_params:
            dataset_args = data_module_params.get("dataset_args", {})
            class_mapping = class_mapping or dataset_args.get("class_mapping")
            lightning_module.datamodule_params = data_module_params
            lightning_module.info_logged = False
            reduce_zero_label = dataset_args.get("reduce_zero_label", False)

        if (not data_module_params) and (not patch_size):
            raise ValueError(
                "No datamodule_hyper_parameters found in checkpoint and patch_size is not set. "
                "Please provide a valid checkpoint with datamodule_hyper_parameters or set patch_size."
            )
        if not (_ps := data_module_params.get("patch_size")):
            patch_size_info = " (not found in checkpoint)"
        else:
            patch_size = _ps
            patch_size_info = " (found in model checkpoint)"

        patch_size_info = f"Using patch size: {_ps}" + patch_size_info
        print(patch_size_info)
        if reduce_zero_label:
            print(
                "Found reduce_zero_label = True in datamodule_params. Adding 1 to class predictions."
            )
            lightning_module.reduce_zero_label = True

        return predict_on_tif_generic(
            tif_file_path=tif_file_path,
            prediction_func=lightning_module.predict_class,
            show_results=show_results,
            patch_size=_ps,
            batch_size=batch_size,
            progress_bar=progress_bar,
            out_file_path=out_file_path,
            ax=ax,
            class_mapping=class_mapping,
            func_supports_batching=func_supports_batching,
        )


class RegressionTask(TorchGeoRegressionTask):
    """Task for regression problems, inheriting from TorchGeo's RegressionTask."""
    
    def __init__(
        self,
        in_channels: int,
        num_filters: int = 128,
        model: str = "deepresunet",
        backbone: str = "resnet50",
        weights: WeightsEnum | str | bool | None = None,
        num_outputs: int = 1,
        loss: str = "mse",
        lr: float = 1e-3,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        model_kwargs: dict[str, Any] = None,
        lr_scheduler: dict[str, Any] = None,
    ):
        """Initialize a regression task.
        
        Inherits from TorchGeo's RegressionTask and adds support for custom models,
        learning rate schedulers, and additional metrics.

        Args:
            in_channels: Number of input channels
            model: Name of the model architecture to use
            backbone: Name of the backbone to use (for applicable models)
            weights: Initial model weights
            loss: Loss function to use ('mse', 'mae', etc.)
            lr: Learning rate
            freeze_backbone: Whether to freeze backbone weights
            freeze_decoder: Whether to freeze decoder weights
            model_kwargs: Additional model-specific arguments
            lr_scheduler: Learning rate scheduler configuration. If None, uses default:
                        ReduceLROnPlateau with:
                        * mode: "min"
                        * factor: 0.2
                        * patience: 10
                        * min_lr: 1e-6
                        * monitor: "val_loss"
        """
        self.model_kwargs = model_kwargs or {}
        self.lr_scheduler = lr_scheduler or {}

        super().__init__(
            model=model,
            backbone=backbone,
            weights=weights,
            in_channels=in_channels,
            num_outputs=num_outputs,
            num_filters=num_filters,
            loss=loss,
            lr=lr,
            freeze_backbone=freeze_backbone,
            freeze_decoder=freeze_decoder,
        )

    def configure_models(self) -> None:
        model: str = self.hparams["model"]
        weights = self.weights

        # Model selection
        if model.lower() in REG_MODEL_MAPPING:
            model_cls = REG_MODEL_MAPPING[model]
           
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

    def configure_losses(self) -> None:
        """Initialize the loss criterion for regression."""
        match self.hparams["loss"]:
            case "mse":
                self.criterion = torch.nn.MSELoss()
            case "mae":
                self.criterion = torch.nn.L1Loss()
            case "huber":
                self.criterion = torch.nn.HuberLoss()
            case "smoothl1":
                self.criterion = torch.nn.SmoothL1Loss()
            case _:
                raise ValueError(f"Unknown loss: {self.hparams['loss']}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams["lr"],
            weight_decay=1e-4,  # Add some regularization
        )
        monitor = None
        if not self.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=10, min_lr=1e-6, factor=0.2
            )

        else:
            scheduler_type = self.lr_scheduler.pop("type")
            if hasattr(torch.optim.lr_scheduler, scheduler_type):
                scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_type)
                if "monitor" in self.lr_scheduler:
                    monitor = self.lr_scheduler.pop("monitor")
                scheduler = scheduler_cls(optimizer, **self.lr_scheduler)
            else:
                raise ValueError(
                    f"Scheduler {scheduler_type} not found in torch.optim.lr_scheduler"
                )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": monitor or self.monitor,
            },
        }

    def training_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Modified training step for regression tasks."""
        x = batch["image"]
        y = batch["mask"]
        
        if x.ndim > 4:
            x = x.squeeze(0)
            
        batch_size = x.shape[0]
        y_hat = self(x)
        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)
        loss = self.criterion(y_hat, y)
        
        self.log("train_loss", loss, batch_size=batch_size, prog_bar=True, on_epoch=True)
        self.train_metrics(y_hat, y)
        self.log_dict(self.train_metrics, batch_size=batch_size, on_epoch=True)
        
        return loss

    def validation_step(self, batch: dict, batch_idx: int = None, dataloader_idx: int = 0) -> None:
        """Modified validation step for regression tasks."""
        x = batch["image"]
        y = batch["mask"]
        
        if x.ndim > 4:
            x = x.squeeze(0)
            
        batch_size = x.shape[0]
        y_hat = self(x)
        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)
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
            if self.target_key == 'mask':
                y = y.squeeze(dim=1)
                y_hat = y_hat.squeeze(dim=1)

            # Create a new batch dictionary with the valid subset
            valid_batch = {
                "image": x.cpu(),
                "mask": y.cpu(),  # Add channel dimension back
                "prediction": y_hat
                .detach()
                .cpu(),  # Add channel dimension to match mask
            }

            # Only sample from first example for visualization
            sample = unbind_samples(valid_batch)[0]

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

    def test_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Modified test step for regression tasks."""
        x = batch["image"]
        y = batch["mask"]
        
        if x.ndim > 4:
            x = x.squeeze(0)
            
        batch_size = x.shape[0]
        y_hat = self(x)
        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)
        loss = self.criterion(y_hat, y)
        
        self.log("test_loss", loss, batch_size=batch_size)
        self.test_metrics(y_hat, y)
        self.log_dict(self.test_metrics, batch_size=batch_size)

    def predict_step(self, batch: Tensor) -> Tensor:
        """Override to return raw values instead of probabilities."""
        with torch.inference_mode():
            return self(batch)

    @staticmethod
    def predict_on_tif_file(       
        tif_file_path: str | Path,
        checkpoint_path: str | Path,
        func_supports_batching: bool = True,
        batch_size: int = 8,
        out_file_path: str | Path = None,
        show_results: bool = False,
        ax: plt.Axes = None,
        progress_bar: bool = True,
        patch_size: int = None,
    ):
        """
        Perform predictions on a GeoTIFF file using a trained regression model.

        Parameters:
            tif_file_path (str | Path): 
                Path to the input GeoTIFF file to perform predictions on.
            checkpoint_path (str | Path): 
                Path to the trained model checkpoint file .
            func_supports_batching (bool, optional):
                Whether the prediction function supports batch processing. Defaults to True.
            batch_size (int, optional):
                Number of patches to process simultaneously. Only used if func_supports_batching is True. Defaults to 8.
            out_file_path (str | Path, optional):
                Path where the output predictions should be saved as a GeoTIFF. If None, output is not saved to file. Defaults to None.
            show_results (bool, optional):
                Whether to display the prediction results using matplotlib. Defaults to False.
            ax (plt.Axes, optional):
                Matplotlib axes on which to plot the results. Only used if show_results is True. Defaults to None.
            progress_bar (bool, optional): 
                Whether to display a progress bar during prediction. Defaults to True.
            patch_size (int, optional):
                Size of image patches to process. If None, tries to get from checkpoint datamodule parameters. Defaults to None.

        Raises:
            ValueError:
                If no patch_size is provided and it cannot be found in the checkpoint's datamodule parameters

        Returns:
            numpy.ndarray:
                Array containing the prediction results for the entire GeoTIFF image
        """        
        from eotorch.inference.inference import predict_on_tif_generic

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lightning_module = RegressionTask.load_from_checkpoint(
            checkpoint_path,
            map_location=device,
        )
        lightning_module.to(device)
        lightning_module.eval()
        
        data_module_params = torch.load(
            checkpoint_path, weights_only=False, map_location=device
        ).get("datamodule_hyper_parameters")
        
        if not data_module_params and not patch_size:
            raise ValueError(
                "No datamodule_hyper_parameters found in checkpoint and patch_size is not set. "
                "Please provide a valid checkpoint with datamodule_hyper_parameters or set patch_size."
            )

        patch_size = data_module_params.get("patch_size", patch_size)
        print(f"Using patch size: {patch_size}")

        return predict_on_tif_generic(
            tif_file_path=tif_file_path,
            prediction_func=lambda x: lightning_module.predict_step(x).cpu().numpy(),
            show_results=show_results,
            patch_size=patch_size,
            batch_size=batch_size,
            progress_bar=progress_bar,
            out_file_path=out_file_path,
            ax=ax,
            func_supports_batching=func_supports_batching,
        )


class PatchSegmentationTask(LightningModule):
    """Lightning module for semantic segmentation with DeepResUNet backbone."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        num_filters: int = 256,
        model: str = 'deepresunet',
        backbone: str = 'resnet34',
        loss: str = 'dice',
        task: str = 'multiclass',
        class_weights: Tensor | None = None,
        weights: WeightsEnum | str | bool | None = None,
        ignore_index: int | None = None,
        lr: float = 1e-3,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        model_kwargs: dict[str, Any] = None,
        lr_scheduler: dict[str, Any] | None = None,
        class_names: list[str] | None = None,
    ) -> None:
        """Initialize model, optimization config, and metric collections."""
        super().__init__()
        self.weights = weights
        self.task = task
        self.class_names = class_names
        self.model_kwargs = model_kwargs or {}
        self.lr_scheduler = lr_scheduler or {}

        self.save_hyperparameters(ignore={"weights"})

        self.configure_models()
        self.configure_losses()
        self.configure_metrics()

    def configure_models(self) -> None:
        model: str = self.hparams["model"]
        backbone: str = self.hparams['backbone']
        weights = self.weights

        if model.lower() in CLF_MODEL_MAPPING:
            model_cls = CLF_MODEL_MAPPING[model]

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
            in_channels: int = self.hparams['in_channels']
            num_classes: int = self.hparams['num_classes']
            num_filters: int = self.hparams['num_filters']

            match model.lower():
                case 'unet':
                    self.model = smp.Unet(
                        encoder_name=backbone,
                        encoder_weights='imagenet' if weights is True else None,
                        in_channels=in_channels,
                        classes=num_classes,
                    )
                case 'unet++' | 'unetplusplus':
                    self.model = smp.UnetPlusPlus(
                        encoder_name=backbone,
                        encoder_weights='imagenet' if weights is True else None,
                        in_channels=in_channels,
                        classes=num_classes,
                        decoder_channels=[num_filters]+[num_filters := num_filters // 2 for _ in range(4)],
                    )
                case 'upernet':
                    self.model = smp.UPerNet(
                        encoder_name=backbone,
                        encoder_weights='imagenet' if weights is True else None,
                        in_channels=in_channels,
                        classes=num_classes,
                        decoder_channels=num_filters,
                    )
                case 'segformer':
                    self.model = smp.Segformer(
                        encoder_name=backbone,
                        encoder_weights='imagenet' if weights is True else None,
                        in_channels=in_channels,
                        classes=num_classes,
                        decoder_segmentation_channels=num_filters,
                    )
                case 'dpt':
                    self.model = smp.DPT(
                        encoder_name='tu-vit_base_patch16_224.augreg_in21k',
                        encoder_weights='imagenet' if weights is True else None,
                        in_channels=in_channels,
                        classes=num_classes,
                        decoder_intermediate_channels=[num_filters * 2**i for i in [0, 1, 2, 2]],
                        decoder_fusion_channels=num_filters,
                    )
                case _:
                    raise ValueError(f"Unknown model: {model}")

            # Freeze backbone
            if self.hparams['freeze_backbone']:
                for param in self.model.encoder.parameters():
                    param.requires_grad = False

            # Freeze decoder
            if self.hparams['freeze_decoder']:
                for param in self.model.decoder.parameters():
                    param.requires_grad = False

    def configure_losses(self) -> None:
        """Configure the training loss function based on the selected loss name."""
        ignore_index: int | None = self.hparams["ignore_index"]
        class_weights: Tensor | None = self.hparams["class_weights"]
        if class_weights is not None and not isinstance(class_weights, Tensor):
            class_weights = torch.tensor(class_weights, dtype=torch.float32)

        match self.hparams["loss"]:
            case "ce":
                from torch import nn

                ignore_value = -1000 if ignore_index is None else ignore_index
                self.criterion: nn.Module = nn.CrossEntropyLoss(
                    ignore_index=ignore_value, weight=class_weights
                )
            case "bce":
                from torch import nn

                self.criterion = nn.BCEWithLogitsLoss()
            case "jaccard":
                # JaccardLoss requires a list of classes to use instead of a class
                # index to ignore.
                classes = [
                    i for i in range(self.hparams["num_classes"]) if i != ignore_index
                ]

                self.criterion = smp.losses.JaccardLoss(
                    mode=self.hparams["task"], classes=classes
                )
            case "focal":
                self.criterion = smp.losses.FocalLoss(
                    mode=self.hparams["task"],
                    ignore_index=ignore_index,
                    normalized=True,
                    alpha=0.8,
                    gamma=3.0,
                )
            case "dice":
                self.criterion = smp.losses.DiceLoss(
                    mode=self.hparams["task"],
                    ignore_index=ignore_index,
                    from_logits=True,
                )
            case _:
                raise ValueError(f"Unknown loss: {self.hparams['loss']}")

    def configure_optimizers(self) -> dict[str, Any]:
        """Build optimizer and scheduler configuration for Lightning."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"], weight_decay=1e-4)
        monitor = "val_loss"
        if not self.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=10, min_lr=1e-6, factor=0.2
            )
        else:
            scheduler_config = dict(self.lr_scheduler)
            scheduler_type = scheduler_config.pop("type", None)
            if not scheduler_type:
                raise ValueError("lr_scheduler config must include a 'type' key")
            if "monitor" in scheduler_config:
                monitor = scheduler_config.pop("monitor")
            if not hasattr(torch.optim.lr_scheduler, scheduler_type):
                raise ValueError(
                    f"Scheduler {scheduler_type} not found in torch.optim.lr_scheduler"
                )
            scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_type)
            scheduler = scheduler_cls(optimizer, **scheduler_config)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": monitor,
            },
        }
    
    def configure_metrics(self) -> None:
        """Create train/validation/test metric collections."""
        num_classes = self.hparams["num_classes"]
        ignore_index = self.hparams["ignore_index"]
        class_names = self.hparams.get("class_names") or [
            str(i) for i in range(num_classes)
        ]
        
        metrics = MetricCollection(
            {
                # Macro-averaged metrics (consistent with TerraTorch naming)
                "mIoU": MulticlassJaccardIndex(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="macro",
                ),
                "F1_Score": MulticlassF1Score(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="macro",
                ),
                "Accuracy": MulticlassAccuracy(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="macro",
                ),
                # Micro-averaged (pixel-level) accuracy
                "Pixel_Accuracy": MulticlassAccuracy(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="micro",
                ),
                # Per-class metrics using ClasswiseWrapper (like TerraTorch)
                "IoU": ClasswiseWrapper(
                    MulticlassJaccardIndex(
                        num_classes=num_classes,
                        ignore_index=ignore_index,
                        average=None,
                    ),
                    labels=class_names,
                    prefix="IoU_",
                ),
                "Class_Accuracy": ClasswiseWrapper(
                    MulticlassAccuracy(
                        num_classes=num_classes,
                        ignore_index=ignore_index,
                        average=None,
                    ),
                    labels=class_names,
                    prefix="Class_Accuracy_",
                ),
                "Class_F1": ClasswiseWrapper(
                    MulticlassF1Score(
                        num_classes=num_classes,
                        ignore_index=ignore_index,
                        average=None,
                    ),
                    labels=class_names,
                    prefix="Class_F1_",
                ),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")


    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Run one training step and log aggregate/per-class training metrics."""
        x, y = batch
        y_hat = self(x)
        batch_size = x.shape[0]
        loss = self.criterion(y_hat, y)
        
        self.log("train_loss", loss, batch_size=batch_size, prog_bar=True, on_epoch=True)
        computed = self.train_metrics(y_hat, y)
        self.log_dict(computed, batch_size=batch_size)

        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor]) -> None:
        """Run one validation step and log aggregate/per-class validation metrics."""
        x, y = batch

        batch_size = x.shape[0]
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.log("val_loss", loss, batch_size=batch_size)
        computed = self.val_metrics(y_hat, y)
        self.log_dict(computed, batch_size=batch_size)

    def predict_step(self, batch: Tensor) -> Tensor:
        """Predict class probabilities for an input batch."""
        with torch.inference_mode():
            y_hat: Tensor = self(batch)

            match self.hparams["task"]:
                case "binary" | "multilabel":
                    y_hat = y_hat.sigmoid()
                case "multiclass":
                    y_hat = y_hat.softmax(dim=1)

            return y_hat

    def predict_class(self, batch: Tensor | np.ndarray) -> np.ndarray:
        """Predict class indices for a tensor or ndarray batch."""
        if isinstance(batch, np.ndarray):
            batch = array_to_tensor(batch)
        batch = batch.to(self.device)
        probs = self.predict_step(batch)
        preds = probs.argmax(dim=1).cpu().numpy().astype("uint8")
        if hasattr(self, "reduce_zero_label") and self.reduce_zero_label:
            preds += 1
        return preds

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the segmentation backbone."""
        return self.model(x)
    
    @staticmethod
    def predict_on_tif_file(
        tif_file_path: str | Path,
        checkpoint_path: str | Path,
        func_supports_batching: bool = True,
        batch_size: int = 8,
        out_file_path: str | Path | None = None,
        show_results: bool = False,
        progress_bar: bool = True,
        patch_size: int | None = None,
        class_mapping: dict[int, str] | None = None,
    ) -> Path | str:
        """Run tiled inference on a TIF file using a saved model checkpoint."""
        from eotorch.inference.inference import predict_on_tif_generic

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        reduce_zero_label = False
        checkpoint_data = torch.load(
            checkpoint_path, weights_only=False, map_location=device
        )
        hparams = checkpoint_data.get("hyper_parameters", {})
        data_module_params = checkpoint_data.get("datamodule_hyper_parameters")
        state_dict = checkpoint_data.get("state_dict", {})

        if "num_classes" not in hparams and "model.output.weight" in state_dict:
            hparams["num_classes"] = int(state_dict["model.output.weight"].shape[0])
        if "in_channels" not in hparams and "model.input_conv.conv.weight" in state_dict:
            hparams["in_channels"] = int(state_dict["model.input_conv.conv.weight"].shape[1])
        if "num_filters" not in hparams and "model.input_conv.conv.weight" in state_dict:
            hparams["num_filters"] = int(state_dict["model.input_conv.conv.weight"].shape[0])
        if "lr" not in hparams and "learning_rate" in hparams:
            hparams["lr"] = hparams["learning_rate"]

        if ("num_classes" not in hparams) and data_module_params:
            dataset_args = data_module_params.get("dataset_args", {})
            class_mapping_from_data = dataset_args.get("class_mapping")
            if class_mapping_from_data:
                hparams["num_classes"] = len(class_mapping_from_data)

        missing_required = [k for k in ("num_classes", "in_channels") if k not in hparams]
        if missing_required:
            raise ValueError(
                "Missing required model hyperparameters in checkpoint: "
                + ", ".join(missing_required)
                + ". Please provide a checkpoint that includes hyper_parameters or includes model state_dict keys "
                + "for input/output layers."
            )

        init_keys = [
            "num_classes",
            "in_channels",
            "num_filters",
            "loss",
            "task",
            "ignore_index",
            "lr",
            "lr_scheduler",
        ]
        init_kwargs = {k: hparams[k] for k in init_keys if k in hparams}

        lightning_module = PatchSegmentationTask.load_from_checkpoint(
            checkpoint_path,
            map_location=device,
            **init_kwargs,
        )
        lightning_module.to(device)
        lightning_module.eval()
        if data_module_params:
            dataset_args = data_module_params.get("dataset_args", {})
            class_mapping = class_mapping or dataset_args.get("class_mapping")
            lightning_module.datamodule_params = data_module_params
            lightning_module.info_logged = False
            reduce_zero_label = dataset_args.get("reduce_zero_label", False)

        if (not data_module_params) and (not patch_size):
            raise ValueError(
                "No datamodule_hyper_parameters found in checkpoint and patch_size is not set. "
                "Please provide a valid checkpoint with datamodule_hyper_parameters or set patch_size."
            )
        _ps = patch_size
        if data_module_params and data_module_params.get("patch_size"):
            _ps = data_module_params.get("patch_size")

        if not _ps:
            patch_size_info = " (not found in checkpoint)"
        else:
            patch_size = _ps
            patch_size_info = " (found in model checkpoint)"

        patch_size_info = f"Using patch size: {_ps}" + patch_size_info
        print(patch_size_info)
        if reduce_zero_label:
            print(
                "Found reduce_zero_label = True in datamodule_params. Adding 1 to class predictions."
            )
            lightning_module.reduce_zero_label = True

        return predict_on_tif_generic(
            tif_file_path=tif_file_path,
            prediction_func=lightning_module.predict_class,
            show_results=show_results,
            patch_size=_ps,
            batch_size=batch_size,
            progress_bar=progress_bar,
            out_file_path=out_file_path,
            class_mapping=class_mapping,
            func_supports_batching=func_supports_batching,
        )