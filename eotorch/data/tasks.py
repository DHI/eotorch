from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import segmentation_models_pytorch as smp
import torch
from matplotlib import pyplot as plt
from torchgeo.datasets import RGBBandsMissingError, unbind_samples
from torchgeo.datasets.utils import array_to_tensor
from torchgeo.trainers import (
    SemanticSegmentationTask as TorchGeoSemanticSegmentationTask,
)
from torchmetrics import Accuracy, F1Score, JaccardIndex, MetricCollection

from eotorch.models import MODEL_MAPPING
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
    ):
        """
        Task based on TorchGeo's SemanticSegmentationTask, but with the ability to use custom models.
        We still have access to all the architectures and backbones from TorchGeo.

        Args:
            lr_scheduler : dict
                The learning rate scheduler to use. If None, a default scheduler will be used.
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

        """
        self.model_kwargs = model_kwargs or {}
        self.lr_scheduler = lr_scheduler or {}
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

        * :class:`~torchmetrics.Accuracy`: Overall accuracy (micro average):
          The number of true positives divided by the dataset size. Higher values are better.
        * :class:`~torchmetrics.Accuracy`: Per-class accuracy (no average across classes):
          Average of the accuracy calculated for each class separately. Higher values are better.
        * :class:`~torchmetrics.JaccardIndex`: Overall IoU (micro average):
          Intersection over union averaged across all pixels. Higher values are better.
        * :class:`~torchmetrics.JaccardIndex`: Per-class IoU (no average across classes):
          Average of the IoU calculated for each class separately. Higher values are better.

        .. note::
           * 'Micro' averaging gives equal weight to each pixel and is suitable for overall performance
             evaluation but may not reflect minority class accuracy in imbalanced datasets.
        """
        kwargs = {
            "task": self.hparams["task"],
            "num_classes": self.hparams["num_classes"],
            "num_labels": self.hparams["num_labels"],
            "ignore_index": self.hparams["ignore_index"],
        }
        metrics = MetricCollection(
            {
                # Overall accuracy (micro average) - existing metric
                "overall_acc": Accuracy(
                    multidim_average="global", average="micro", **kwargs
                ),
                "macro_acc": Accuracy(
                    multidim_average="global", average="macro", **kwargs
                ),
                # Per-class accuracy (macro average) - new metric
                "per_class_acc": Accuracy(
                    multidim_average="global", average=None, **kwargs
                ),
                "overall_f1": F1Score(
                    average="micro",
                    **kwargs,
                ),
                "macro_f1": F1Score(
                    average="macro",
                    **kwargs,
                ),
                "per_class_f1": F1Score(
                    average=None,
                    **kwargs,
                ),
                # Overall IoU (micro average) - existing metric
                "overall_iou": JaccardIndex(average="micro", **kwargs),
                # Per-class IoU (macro average) - new metric
                "per_class_iou": JaccardIndex(average=None, **kwargs),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

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
        loss: Tensor = self.criterion(y_hat, y)

        self.log(
            "train_loss", loss, batch_size=batch_size, prog_bar=True, on_epoch=True
        )
        self.train_metrics(y_hat, y)

        # Log regular metrics excluding per-class accuracy and IoU
        self.log_dict(
            {
                k: v
                for k, v in self.train_metrics.items()
                if k
                not in {
                    "train_per_class_acc",
                    "train_per_class_iou",
                    "train_per_class_f1",
                }
            },
            batch_size=batch_size,
            on_epoch=True,
        )
        return loss

    def on_train_epoch_end(self) -> None:
        """Called at the end of the training epoch to log accumulated metrics."""
        # Compute the per-class accuracies
        per_class_acc = self.train_metrics["train_per_class_acc"].compute()
        per_class_iou = self.train_metrics["train_per_class_iou"].compute()
        per_class_f1 = self.train_metrics["train_per_class_f1"].compute()

        # Log to tensorboard using a single writer call with a tag per class
        if self.logger and hasattr(self.logger, "experiment"):
            writer = self.logger.experiment

            # For accuracies - create separate tags for each class to avoid legend clutter
            for i, acc in enumerate(per_class_acc):
                writer.add_scalar(
                    f"train_class_accuracy/{i + 1}",
                    acc.item(),
                    self.current_epoch,
                )

            # For IoUs - create separate tags for each class
            for i, iou in enumerate(per_class_iou):
                writer.add_scalar(
                    f"train_class_iou/{i + 1}",
                    iou.item(),
                    self.current_epoch,
                )

            # For F1 scores - create separate tags for each class
            for i, f1 in enumerate(per_class_f1):
                writer.add_scalar(
                    f"train_class_f1/{i + 1}",
                    f1.item(),
                    self.current_epoch,
                )

    def on_validation_epoch_end(self) -> None:
        """Called at the end of the validation epoch to log accumulated metrics."""
        # Compute the per-class accuracies
        per_class_acc = self.val_metrics["val_per_class_acc"].compute()
        per_class_iou = self.val_metrics["val_per_class_iou"].compute()
        per_class_f1 = self.val_metrics["val_per_class_f1"].compute()

        if self.logger and hasattr(self.logger, "experiment"):
            writer = self.logger.experiment

            # For accuracies - create separate tags for each class to avoid legend clutter
            for i, acc in enumerate(per_class_acc):
                writer.add_scalar(
                    f"val_class_accuracy/{i + 1}",
                    acc.item(),
                    self.current_epoch,
                )

            # For IoUs - create separate tags for each class
            for i, iou in enumerate(per_class_iou):
                writer.add_scalar(
                    f"val_class_iou/{i + 1}",
                    iou.item(),
                    self.current_epoch,
                )

            # For F1 scores - create separate tags for each class
            for i, f1 in enumerate(per_class_f1):
                writer.add_scalar(
                    f"val_class_f1/{i + 1}",
                    f1.item(),
                    self.current_epoch,
                )

    def on_test_epoch_end(self) -> None:
        """Called at the end of the test epoch to log accumulated metrics."""
        # Compute the per-class accuracies
        per_class_acc = self.test_metrics["test_per_class_acc"].compute()
        per_class_iou = self.test_metrics["test_per_class_iou"].compute()
        per_class_f1 = self.test_metrics["test_per_class_f1"].compute()

        if self.logger and hasattr(self.logger, "experiment"):
            writer = self.logger.experiment

            # For accuracies - create separate tags for each class to avoid legend clutter
            for i, acc in enumerate(per_class_acc):
                writer.add_scalar(
                    f"test_class_accuracy/{i + 1}",
                    acc.item(),
                    self.current_epoch,
                )

            # For IoUs - create separate tags for each class
            for i, iou in enumerate(per_class_iou):
                writer.add_scalar(
                    f"test_class_iou/{i + 1}",
                    iou.item(),
                    self.current_epoch,
                )

            # For F1 scores - create separate tags for each class
            for i, f1 in enumerate(per_class_f1):
                writer.add_scalar(
                    f"test_class_f1/{i + 1}",
                    f1.item(),
                    self.current_epoch,
                )

    def validation_step(
        self, batch: Any, batch_idx: int = None, dataloader_idx: int = 0
    ) -> None:
        """Compute the validation loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.s
            dataloader_idx: Index of the current dataloader.
        """
        # x = batch
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
        # return y_hat.argmax(dim=1)
        # return y_hat
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, batch_size=batch_size)
        self.val_metrics(y_hat, y)
        self.log_dict(
            {
                k: v
                for k, v in self.val_metrics.items()
                if k
                not in {"val_per_class_acc", "val_per_class_iou", "val_per_class_f1"}
            },
            batch_size=batch_size,
        )

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
        self.log_dict(
            {
                k: v
                for k, v in self.test_metrics.items()
                if k
                not in {"test_per_class_acc", "test_per_class_iou", "test_per_class_f1"}
            },
            batch_size=batch_size,
        )

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
            checkpoint_path
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

        predict_on_tif_generic(
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
