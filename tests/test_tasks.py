"""Tests for SegmentationTask and RegressionTask, from training to inference."""

from pathlib import Path

import lightning as L
import numpy as np
import pytest
import rasterio as rst
import torch

from eotorch.data import (
    RasterDataModule,
    RegressionTask,
    SegmentationTask,
    get_regression_dataset,
    get_segmentation_dataset,
)
from eotorch.data.geodatasets import (
    PlottableClassificationDataset,
    PlottableImageDataset,
    PlottableRegressionDataset,
)

TEST_DATA = Path(__file__).parent / "data"
NODATA = -9999.0


@pytest.fixture(autouse=True)
def _restore_class_attrs():
    """get_*_dataset configures the dataset classes, so restore them."""
    classes = (
        PlottableImageDataset,
        PlottableClassificationDataset,
        PlottableRegressionDataset,
    )
    saved = {
        cls: {k: getattr(cls, k) for k in cls._class_config if hasattr(cls, k)}
        for cls in classes
    }
    yield
    for cls, attrs in saved.items():
        for k, v in attrs.items():
            setattr(cls, k, v)


def _fit(task, dataset, tmp_path) -> Path:
    datamodule = RasterDataModule(
        train_dataset=dataset, val_dataset=dataset, batch_size=2, patch_size=32
    )
    trainer = L.Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(task, datamodule=datamodule)
    checkpoint = tmp_path / "model.ckpt"
    trainer.save_checkpoint(checkpoint)
    return checkpoint


def test_segmentation_fit_and_predict(tmp_path):
    dataset = get_segmentation_dataset(
        images_dir=TEST_DATA,
        labels_dir=TEST_DATA,
        image_glob="image_*.tif",
        label_glob="label_*.tif",
        all_image_bands=("red", "green", "blue"),
        class_mapping={1: "a", 2: "b", 3: "c", 4: "d"},
        cache_size=0,
    )
    task = SegmentationTask(
        num_classes=4,
        in_channels=3,
        model="unet_lite",
        num_filters=4,
        model_kwargs={"depth": 2},
        loss={"ce": 1.0, "boundary_distance": 0.5},
        ignore_index=-1,
        class_names=["a", "b", "c", "d"],
    )
    checkpoint = _fit(task, dataset, tmp_path)

    loaded = SegmentationTask.load_from_checkpoint(checkpoint)
    assert loaded.hparams["loss"] == {"ce": 1.0, "boundary_distance": 0.5}
    assert loaded.hparams["model_kwargs"] == {"depth": 2}

    out = SegmentationTask.predict_on_tif_file(
        TEST_DATA / "image_0.tif",
        checkpoint,
        out_file_path=tmp_path / "pred.tif",
        progress_bar=False,
    )
    with rst.open(out) as src:
        pred = src.read(1)
        assert src.dtypes[0] == "uint8"
    assert pred.shape == (100, 100)
    # Trained with reduce_zero_label, so classes are shifted back to 1-4
    assert set(np.unique(pred)) <= {1, 2, 3, 4}

    blended = SegmentationTask.predict_on_tif_file(
        TEST_DATA / "image_0.tif",
        checkpoint,
        out_file_path=tmp_path / "blended.tif",
        progress_bar=False,
        blend=True,
    )
    with rst.open(blended) as src:
        assert set(np.unique(src.read(1))) <= {1, 2, 3, 4}


def test_regression_fit_and_predict(tmp_path):
    # Continuous labels derived from the class labels, with 0 as nodata
    for label in TEST_DATA.glob("label_*.tif"):
        with rst.open(label) as src:
            data, profile = src.read().astype("float32"), src.profile
        data = np.where(data == 0, NODATA, data * 1.5)
        profile.update(dtype="float32", nodata=NODATA)
        with rst.open(tmp_path / f"target_{label.name}", "w", **profile) as dst:
            dst.write(data)

    dataset = get_regression_dataset(
        images_dir=TEST_DATA,
        labels_dir=tmp_path,
        image_glob="image_*.tif",
        label_glob="target_*.tif",
        all_image_bands=("red", "green", "blue"),
        nodata_value=NODATA,
        cache_size=0,
    )
    task = RegressionTask(
        in_channels=3,
        model="unet_lite",
        num_filters=4,
        nodata_value=NODATA,
        loss="huber",
    )
    checkpoint = _fit(task, dataset, tmp_path)

    out = RegressionTask.predict_on_tif_file(
        TEST_DATA / "image_0.tif",
        checkpoint,
        out_file_path=tmp_path / "pred.tif",
        progress_bar=False,
    )
    with rst.open(out) as src:
        pred = src.read(1)
        assert src.dtypes[0] == "float32" and src.nodata == NODATA
    assert pred.shape == (100, 100) and np.isfinite(pred).all()


def test_regression_loss_excludes_nodata_pixels():
    task = RegressionTask(
        in_channels=3, model="unet_lite", num_filters=4, nodata_value=NODATA
    )
    x = torch.randn(2, 3, 16, 16)
    y = torch.full((2, 1, 16, 16), NODATA)

    assert task.training_step({"image": x, "mask": y}, 0) is None

    y[:, :, :8] = 1.0
    loss = task.training_step({"image": x, "mask": y}, 0)
    expected = ((task(x)[:, :, :8] - 1.0) ** 2).mean()
    torch.testing.assert_close(loss, expected)


def test_segmentation_skips_unlabelled_patches():
    task = SegmentationTask(
        num_classes=3, in_channels=3, model="unet_lite", num_filters=4, ignore_index=0
    )
    batch = {
        "image": torch.randn(2, 3, 32, 32),
        "mask": torch.zeros(2, 1, 32, 32).long(),
    }
    assert task.training_step(batch, 0) is None

    batch["mask"][1, :, :4] = 2
    assert torch.isfinite(task.training_step(batch, 0))


def test_binary_segmentation():
    task = SegmentationTask(
        num_classes=1,
        in_channels=3,
        task="binary",
        model="unet_lite",
        num_filters=4,
        loss={"bce": 1.0, "dice": 1.0},
    )
    batch = {
        "image": torch.randn(2, 3, 16, 16),
        "mask": torch.randint(0, 2, (2, 1, 16, 16)),
    }
    loss = task.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert set(task.train_metrics.compute()) >= {"train/mIoU", "train/F1_Score"}
    assert task.predict_class(batch["image"]).shape == (2, 16, 16)


def test_smp_model_with_kwargs_and_frozen_encoder():
    task = SegmentationTask(
        num_classes=3,
        in_channels=5,
        model="unetplusplus",
        backbone="resnet18",
        model_kwargs={"decoder_channels": [64, 32, 16, 8, 4]},
        freeze_backbone=True,
    )
    assert task(torch.randn(2, 5, 64, 64)).shape == (2, 3, 64, 64)
    assert not any(p.requires_grad for p in task.model.encoder.parameters())
    assert all(p.requires_grad for p in task.model.decoder.parameters())


def test_freezing_model_without_encoder_fails():
    with pytest.raises(ValueError, match="no backbone to freeze"):
        SegmentationTask(num_classes=3, in_channels=3, freeze_backbone=True)


def test_lr_scheduler_config():
    scheduler = {"type": "CosineAnnealingLR", "T_max": 10, "monitor": "val/mIoU"}
    task = SegmentationTask(
        num_classes=3,
        in_channels=3,
        model="unet_lite",
        num_filters=4,
        lr_scheduler=scheduler,
    )
    for _ in range(2):  # must not consume the config
        config = task.configure_optimizers()
    assert isinstance(
        config["lr_scheduler"]["scheduler"], torch.optim.lr_scheduler.CosineAnnealingLR
    )
    assert config["lr_scheduler"]["monitor"] == "val/mIoU"


def test_eotorch_models_have_no_imagenet_weights():
    with pytest.raises(ValueError, match="No pretrained weights"):
        SegmentationTask(num_classes=3, in_channels=3, weights=True)
