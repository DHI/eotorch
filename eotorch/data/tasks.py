from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import segmentation_models_pytorch as smp
import torch
from matplotlib import pyplot as plt
from torch import Tensor, nn
from torchgeo.datasets import RGBBandsMissingError
from torchgeo.datasets.utils import array_to_tensor
from torchgeo.models import FCN, get_weight
from torchgeo.tasks import PixelwiseRegression, SemanticSegmentation
from torchgeo.tasks import utils as torchgeo_task_utils
from torchmetrics import Accuracy, F1Score, JaccardIndex, MetricCollection
from torchmetrics.wrappers import ClasswiseWrapper
from torchvision.models._api import WeightsEnum

from eotorch.models import MODEL_MAPPING
from eotorch.models.loss import segmentation_loss
from eotorch.utils import get_init_args

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def create_model(
    model: str,
    in_channels: int,
    out_channels: int,
    backbone: str = "resnet50",
    weights: WeightsEnum | str | bool | None = None,
    num_filters: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> nn.Module:
    """Create one of the eotorch models, TorchGeo's FCN, or any smp model by name.

    Args:
        model: Name of an eotorch model (see eotorch.models.MODEL_MAPPING), "fcn", or an
            architecture of `smp.create_model
            <https://smp.readthedocs.io/en/latest/models.html>`__, e.g. "unet",
            "unetplusplus", "deeplabv3plus", "upernet", "segformer" or "dpt".
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        backbone: Encoder of smp models.
        weights: For eotorch models, a state dict of the whole model or the path to one.
            For smp models, True for ImageNet weights, or a TorchGeo weight enum, its
            name, or the path to a checkpoint, to initialize the encoder with.
        num_filters: Number of filters for models with that parameter. Defaults to
            the model's own default.
        model_kwargs: Additional keyword arguments for the model,
            e.g. ``{"decoder_channels": [128, 64, 32, 16, 8]}`` for "unet".
    """
    model_kwargs = model_kwargs or {}

    if model in MODEL_MAPPING:
        model_cls = MODEL_MAPPING[model]
        if num_filters is not None and "num_filters" in get_init_args(model_cls):
            model_kwargs = {"num_filters": num_filters, **model_kwargs}
        net = model_cls(
            in_channels=in_channels, num_classes=out_channels, **model_kwargs
        )
        if weights:
            if isinstance(weights, (str, Path)):
                weights = torch.load(weights, weights_only=True)
            net.load_state_dict(weights)
        return net

    if model == "fcn":
        return FCN(in_channels, out_channels, num_filters=num_filters or 64)

    net = smp.create_model(
        model,
        encoder_name=backbone,
        encoder_weights="imagenet" if weights is True else None,
        in_channels=in_channels,
        classes=out_channels,
        **model_kwargs,
    )
    if weights and weights is not True:
        if isinstance(weights, WeightsEnum):
            state_dict = weights.get_state_dict(progress=True)
        elif os.path.exists(weights):
            _, state_dict = torchgeo_task_utils.extract_backbone(weights)
        else:
            state_dict = get_weight(weights).get_state_dict(progress=True)
        net.encoder.load_state_dict(state_dict)
    return net


class _TaskMixin:
    """Functionality shared by SegmentationTask and RegressionTask."""

    #: Number of output channels of the model
    num_outputs: int

    def configure_models(self) -> None:
        self.model = create_model(
            self.hparams["model"],
            in_channels=self.hparams["in_channels"],
            out_channels=self.num_outputs,
            backbone=self.hparams["backbone"],
            weights=self.weights,
            num_filters=self.hparams["num_filters"],
            model_kwargs=self.hparams["model_kwargs"],
        )
        for part in ("backbone", "decoder"):
            if not self.hparams[f"freeze_{part}"]:
                continue
            module = getattr(
                self.model, "encoder" if part == "backbone" else part, None
            )
            if module is None:
                raise ValueError(
                    f"Model {self.hparams['model']} has no {part} to freeze."
                )
            for param in module.parameters():
                param.requires_grad = False

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams["lr"], weight_decay=1e-4
        )
        config = dict(
            self.hparams["lr_scheduler"]
            or {
                "type": "ReduceLROnPlateau",
                "patience": 10,
                "min_lr": 1e-6,
                "factor": 0.2,
            }
        )
        monitor = config.pop("monitor", self.monitor)
        scheduler_type = config.pop("type")
        scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_type, None)
        if scheduler_cls is None:
            raise ValueError(
                f"Scheduler {scheduler_type} not found in torch.optim.lr_scheduler"
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler_cls(optimizer, **config),
                "monitor": monitor,
            },
        }

    def _log_figure(self, batch_idx: int, sample: dict[str, Tensor]) -> None:
        """Log a plot of a validation sample and its prediction to TensorBoard."""
        datamodule = getattr(self.trainer, "datamodule", None)
        experiment = getattr(self.logger, "experiment", None)
        if batch_idx >= 10 or not hasattr(datamodule, "plot"):
            return
        if not hasattr(experiment, "add_figure"):
            return

        fig: Figure | None = None
        try:
            fig = datamodule.plot({k: v.detach().cpu() for k, v in sample.items()})
        except RGBBandsMissingError:
            pass
        except Exception as e:
            print(f"Error plotting sample from batch {batch_idx}: {e}")
        if fig:
            experiment.add_figure(
                f"image/{batch_idx}", fig, global_step=self.global_step
            )
            plt.close()

    def _to_tensor(self, batch: Tensor | np.ndarray) -> Tensor:
        if isinstance(batch, np.ndarray):
            batch = array_to_tensor(batch)
        return batch.float().to(self.device)

    def _inference_settings(self, dataset_args: dict[str, Any]) -> dict[str, Any]:
        """Keyword arguments for predict_on_tif_generic, given the saved dataset args."""
        raise NotImplementedError

    @classmethod
    def predict_on_tif_file(
        cls,
        tif_file_path: str | Path,
        checkpoint_path: str | Path,
        out_file_path: str | Path | None = None,
        batch_size: int = 8,
        patch_size: int | None = None,
        show_results: bool = False,
        ax: plt.Axes | None = None,
        progress_bar: bool = True,
        **kwargs: Any,
    ) -> Path:
        """
        Use a trained model to predict on a TIF file, patch by patch.

        Args:
            tif_file_path: Path to the input TIF file.
            checkpoint_path: Path to the model checkpoint.
            out_file_path: Output path. Writes to a "predictions" subfolder if None.
            batch_size: Number of patches to predict at once.
            patch_size: Size of the patches. Read from the checkpoint by default.
            show_results: If True, plot the prediction.
            ax: Matplotlib Axes to plot on if show_results is True.
            progress_bar: If True, show a progress bar.
            **kwargs: Passed on to eotorch.inference.inference.predict_on_tif_generic,
                e.g. class_mapping to override the one saved in the checkpoint.

        Returns:
            The path to the produced TIF file, or the plot if show_results is True.
        """
        from eotorch.inference.inference import predict_on_tif_generic

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        module = cls.load_from_checkpoint(checkpoint_path, map_location=device).eval()
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
        datamodule_params = checkpoint.get("datamodule_hyper_parameters") or {}

        patch_size = patch_size or datamodule_params.get("patch_size")
        if not patch_size:
            raise ValueError(
                "No patch_size found in the checkpoint, please provide it as an argument."
            )

        settings = module._inference_settings(datamodule_params.get("dataset_args", {}))
        return predict_on_tif_generic(
            tif_file_path=tif_file_path,
            patch_size=patch_size,
            batch_size=batch_size,
            out_file_path=out_file_path,
            show_results=show_results,
            ax=ax,
            progress_bar=progress_bar,
            **{**settings, **kwargs},
        )


class SegmentationTask(_TaskMixin, SemanticSegmentation):
    #: Set when predicting with a model trained on labels with reduce_zero_label=True
    reduce_zero_label = False

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        model: str = "deepresunet",
        backbone: str = "resnet50",
        weights: WeightsEnum | str | bool | None = None,
        task: str = "multiclass",
        num_filters: int | None = None,
        loss: str | dict[str, float] = "ce",
        loss_kwargs: dict[str, dict[str, Any]] | None = None,
        class_weights: Tensor | list[float] | None = None,
        pos_weight: Tensor | float | None = None,
        ignore_index: int | None = None,
        lr: float = 1e-3,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        model_kwargs: dict[str, Any] | None = None,
        lr_scheduler: dict[str, Any] | None = None,
        class_names: list[str] | None = None,
    ):
        """
        Semantic segmentation, based on TorchGeo's SemanticSegmentation task.

        Args:
            num_classes: Number of classes. Use 1 for binary segmentation.
            in_channels: Number of input channels.
            model: Model name, see :func:`create_model`.
            backbone: Encoder of smp models.
            weights: Initial weights, see :func:`create_model`.
            task: One of "binary", "multiclass" or "multilabel".
            num_filters: Number of filters for models with that parameter.
            loss: Loss name, or a dict of loss names and weights to combine,
                see :func:`eotorch.models.loss.segmentation_loss`.
            loss_kwargs: Extra keyword arguments per loss name.
            class_weights: Weight of each class in the "ce" loss.
            pos_weight: Weight of positive examples in the "bce" loss.
            ignore_index: Label value to ignore in the loss and metrics. Patches
                containing only this value are skipped.
            lr: Learning rate.
            freeze_backbone: Whether to freeze the encoder of the model.
            freeze_decoder: Whether to freeze the decoder of the model.
            model_kwargs: Additional keyword arguments for the model.
            lr_scheduler: Learning rate scheduler configuration: the name of a scheduler
                in torch.optim.lr_scheduler as "type", optionally the metric to
                "monitor", and the arguments of the scheduler. The default is
                {"type": "ReduceLROnPlateau", "patience": 10, "min_lr": 1e-6, "factor": 0.2}.
                Another example: {"type": "CosineAnnealingLR", "T_max": 100, "eta_min": 1e-6}
            class_names: Class names, used for the names of the per-class metrics.
        """
        self.weights = weights
        self.num_outputs = num_classes
        super().__init__(
            model=model,
            backbone=backbone,
            weights=weights,
            in_channels=in_channels,
            task=task,
            num_classes=num_classes,
            num_labels=num_classes,
            labels=class_names,
            num_filters=num_filters,
            pos_weight=pos_weight,
            loss=loss,
            class_weights=class_weights,
            ignore_index=ignore_index,
            lr=lr,
            freeze_backbone=freeze_backbone,
            freeze_decoder=freeze_decoder,
        )

    def configure_losses(self) -> None:
        self.criterion = segmentation_loss(
            self.hparams["loss"],
            task=self.hparams["task"],
            num_classes=self.hparams["num_classes"],
            ignore_index=self.hparams["ignore_index"],
            class_weights=self.hparams["class_weights"],
            pos_weight=self.hparams["pos_weight"],
            loss_kwargs=self.hparams["loss_kwargs"],
        )

    def configure_metrics(self) -> None:
        """Initialize the metrics, named consistently with TerraTorch.

        * mIoU, F1_Score, Accuracy: averaged over classes (macro)
        * Pixel_Accuracy: averaged over pixels (micro)
        * IoU_<class>, Class_Accuracy_<class>, Class_F1_<class>: per class
          (not for binary segmentation)
        """
        task = self.hparams["task"]
        num_classes = self.hparams["num_classes"]
        args = {
            "task": task,
            "num_classes": num_classes,
            "num_labels": num_classes,
            "ignore_index": self.hparams["ignore_index"],
        }
        metrics = {
            "mIoU": JaccardIndex(**args, average="macro"),
            "F1_Score": F1Score(**args, average="macro"),
            "Accuracy": Accuracy(**args, average="macro"),
            "Pixel_Accuracy": Accuracy(**args, average="micro"),
        }
        if task != "binary":
            labels = self.hparams["class_names"] or [str(i) for i in range(num_classes)]
            for name, metric_cls, prefix in [
                ("IoU", JaccardIndex, "IoU_"),
                ("Class_Accuracy", Accuracy, "Class_Accuracy_"),
                ("Class_F1", F1Score, "Class_F1_"),
            ]:
                metrics[name] = ClasswiseWrapper(
                    metric_cls(**args, average=None), labels=labels, prefix=prefix
                )
        metrics = MetricCollection(metrics, compute_groups=False)
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def _step(self, batch: dict[str, Tensor], stage: str) -> tuple:
        x, y = batch["image"], batch["mask"]
        if y.ndim == 4:
            y = y.squeeze(1)

        ignore_index = self.hparams["ignore_index"]
        if ignore_index is not None:
            labelled = (y != ignore_index).flatten(1).any(1)
            if not labelled.any():
                return None, x, y, None
            x, y = x[labelled], y[labelled]

        # (B, C, H, W) logits, or (B, H, W) for binary segmentation
        y_hat = self(x).squeeze(1)
        target = y if self.hparams["task"] == "multiclass" else y.float()
        loss = self.criterion(y_hat, target)
        self.log(
            f"{stage}_loss",
            loss,
            batch_size=x.shape[0],
            prog_bar=stage == "train",
            on_epoch=True,
        )
        getattr(self, f"{stage}_metrics").update(y_hat, y)
        return loss, x, y, y_hat

    def _hard_predictions(self, y_hat: Tensor) -> Tensor:
        if self.hparams["task"] == "multiclass":
            return y_hat.argmax(dim=1)
        return (y_hat > 0).long()

    def training_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        return self._step(batch, "train")[0]

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        loss, x, y, y_hat = self._step(batch, "val")
        if loss is not None:
            sample = {
                "image": x[0],
                "mask": y[0].unsqueeze(0),
                "prediction": self._hard_predictions(y_hat)[0],
            }
            self._log_figure(batch_idx, sample)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self._step(batch, "test")

    @torch.inference_mode()
    def predict_probabilities(self, batch: Tensor | np.ndarray) -> Tensor:
        """Class probabilities (B, C, H, W) for a batch of images."""
        y_hat = self(self._to_tensor(batch))
        if self.hparams["task"] == "multiclass":
            return y_hat.softmax(dim=1)
        return y_hat.sigmoid()

    def predict_class(self, batch: Tensor | np.ndarray) -> np.ndarray:
        """Class indices (B, H, W) for a batch of images, or (B, L, H, W) for multilabel."""
        probs = self.predict_probabilities(batch)
        if self.hparams["task"] == "multiclass":
            classes = probs.argmax(dim=1)
        else:
            classes = (probs > 0.5).squeeze(1)
        classes = classes.to(torch.uint8).cpu().numpy()
        return classes + 1 if self.reduce_zero_label else classes

    def _inference_settings(self, dataset_args: dict[str, Any]) -> dict[str, Any]:
        self.reduce_zero_label = dataset_args.get("reduce_zero_label", False)
        count = self.num_outputs if self.hparams["task"] == "multilabel" else 1
        return {
            "prediction_func": self.predict_class,
            "class_mapping": dataset_args.get("class_mapping"),
            "count": count,
        }


class RegressionTask(_TaskMixin, PixelwiseRegression):
    def __init__(
        self,
        in_channels: int,
        num_outputs: int = 1,
        model: str = "deepresunet",
        backbone: str = "resnet50",
        weights: WeightsEnum | str | bool | None = None,
        num_filters: int | None = None,
        loss: str = "mse",
        nodata_value: float | None = None,
        lr: float = 1e-3,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        model_kwargs: dict[str, Any] | None = None,
        lr_scheduler: dict[str, Any] | None = None,
        output_names: list[str] | None = None,
    ):
        """
        Pixelwise regression, based on TorchGeo's PixelwiseRegression task.

        Args:
            in_channels: Number of input channels.
            num_outputs: Number of values to predict per pixel.
            model: Model name, see :func:`create_model`.
            backbone: Encoder of smp models.
            weights: Initial weights, see :func:`create_model`.
            num_filters: Number of filters for models with that parameter.
            loss: One of "mse", "mae", "huber" or "smoothl1".
            nodata_value: Target value of pixels to exclude from the loss and metrics,
                e.g. for sparse labels.
            lr: Learning rate.
            freeze_backbone: Whether to freeze the encoder of the model.
            freeze_decoder: Whether to freeze the decoder of the model.
            model_kwargs: Additional keyword arguments for the model.
            lr_scheduler: Learning rate scheduler configuration, see SegmentationTask.
            output_names: Names of the outputs, used for the metrics if num_outputs > 1.
        """
        self.weights = weights
        self.num_outputs = num_outputs
        super().__init__(
            model=model,
            backbone=backbone,
            weights=weights,
            in_channels=in_channels,
            num_outputs=num_outputs,
            labels=output_names,
            num_filters=num_filters,
            loss=loss,
            lr=lr,
            freeze_backbone=freeze_backbone,
            freeze_decoder=freeze_decoder,
        )

    def configure_losses(self) -> None:
        losses = {
            "mse": nn.MSELoss,
            "mae": nn.L1Loss,
            "huber": nn.HuberLoss,
            "smoothl1": nn.SmoothL1Loss,
        }
        if self.hparams["loss"] not in losses:
            raise ValueError(f"Unknown loss: {self.hparams['loss']}")
        self.criterion = losses[self.hparams["loss"]]()

    def _valid_pixels(self, y_hat: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """Pixels as rows of (N, num_outputs), without those where the target is nodata."""
        y_hat = y_hat.movedim(1, -1).flatten(0, -2)
        y = y.movedim(1, -1).flatten(0, -2)
        if self.hparams["nodata_value"] is not None:
            valid = (y != self.hparams["nodata_value"]).all(dim=-1)
            y_hat, y = y_hat[valid], y[valid]
        # Metrics for a single output expect a vector
        return y_hat.squeeze(-1), y.squeeze(-1)

    def _step(self, batch: dict[str, Tensor], stage: str) -> tuple:
        x, y = batch["image"], batch["mask"]
        if y.ndim == 3:
            y = y.unsqueeze(1)
        y_hat_map = self(x)
        y_hat, y_valid = self._valid_pixels(y_hat_map, y)
        if not len(y_valid):
            return None, x, y, y_hat_map

        loss = self.criterion(y_hat, y_valid.float())
        self.log(
            f"{stage}_loss",
            loss,
            batch_size=x.shape[0],
            prog_bar=stage == "train",
            on_epoch=True,
        )
        getattr(self, f"{stage}_metrics").update(y_hat, y_valid)
        return loss, x, y, y_hat_map

    def training_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        return self._step(batch, "train")[0]

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        loss, x, y, y_hat = self._step(batch, "val")
        if loss is not None and self.num_outputs == 1:
            sample = {"image": x[0], "mask": y[0, 0], "prediction": y_hat[0, 0]}
            self._log_figure(batch_idx, sample)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self._step(batch, "test")

    @torch.inference_mode()
    def predict_value(self, batch: Tensor | np.ndarray) -> np.ndarray:
        """Predicted values (B, num_outputs, H, W) for a batch of images."""
        return self(self._to_tensor(batch)).float().cpu().numpy()

    def _inference_settings(self, dataset_args: dict[str, Any]) -> dict[str, Any]:
        nodata = self.hparams["nodata_value"]
        return {
            "prediction_func": self.predict_value,
            "dtype": "float32",
            "count": self.num_outputs,
            "nodata": np.nan if nodata is None else nodata,
        }
