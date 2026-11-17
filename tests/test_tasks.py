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

TEST_DATA = Path(__file__).parent / "data"
NODATA = -9999.0


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
        assert src.nodata == 0
    assert pred.shape == (100, 100)
    # Label 0 was nodata in training, so classes are shifted back to 1-4
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

    with_probabilities = SegmentationTask.predict_on_tif_file(
        TEST_DATA / "image_0.tif",
        checkpoint,
        out_file_path=tmp_path / "with_probabilities.tif",
        progress_bar=False,
        save_probabilities=True,
    )
    with rst.open(with_probabilities) as src:
        np.testing.assert_array_equal(src.read(1), pred)
    with rst.open(tmp_path / "with_probabilities_probabilities.tif") as src:
        assert src.descriptions == ("a", "b", "c", "d")
        probs = src.read()
    np.testing.assert_allclose(probs.sum(axis=0), 1, rtol=1e-5)
    np.testing.assert_array_equal(probs.argmax(axis=0) + 1, pred)


def test_segmentation_fits_with_default_label_handling(tmp_path):
    """By default, label 0 is unlabelled: it is read as -1, which the task ignores."""
    dataset = get_segmentation_dataset(
        images_dir=TEST_DATA,
        labels_dir=TEST_DATA,
        image_glob="image_*.tif",
        label_glob="label_*.tif",
        cache_size=0,
    )
    task = SegmentationTask(
        num_classes=4, in_channels=3, model="unet_lite", num_filters=4
    )
    assert task.hparams["ignore_index"] == -1
    _fit(task, dataset, tmp_path)


def test_predict_with_checkpoint_from_old_sampler_config(tmp_path):
    """Checkpoints saved with the old train_sampler_config can still be used for inference."""
    dataset = get_segmentation_dataset(
        images_dir=TEST_DATA,
        labels_dir=TEST_DATA,
        image_glob="image_*.tif",
        label_glob="label_*.tif",
        cache_size=0,
    )
    task = SegmentationTask(
        num_classes=4, in_channels=3, model="unet_lite", num_filters=4
    )
    checkpoint_path = _fit(task, dataset, tmp_path)

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    datamodule_params = checkpoint["datamodule_hyper_parameters"]
    del datamodule_params["sampling"]
    datamodule_params["train_sampler_config"] = {
        "type": "RandomPatchSampler",
        "length": 10,
    }
    torch.save(checkpoint, checkpoint_path)

    out = SegmentationTask.predict_on_tif_file(
        TEST_DATA / "image_0.tif",
        checkpoint_path,
        out_file_path=tmp_path / "pred.tif",
        progress_bar=False,
    )
    with rst.open(out) as src:
        assert src.shape == (100, 100)


def test_regression_fit_and_predict(tmp_path):
    dataset = get_regression_dataset(
        images_dir=TEST_DATA,
        labels_dir=TEST_DATA,
        image_glob="image_*.tif",
        label_glob="target_*.tif",
        all_image_bands=("red", "green", "blue"),
        cache_size=0,
    )
    task = RegressionTask(in_channels=3, model="unet_lite", num_filters=4, loss="huber")
    checkpoint = _fit(task, dataset, tmp_path)

    out = RegressionTask.predict_on_tif_file(
        TEST_DATA / "image_0.tif",
        checkpoint,
        out_file_path=tmp_path / "pred.tif",
        progress_bar=False,
    )
    with rst.open(out) as src:
        pred = src.read(1)
        assert src.dtypes[0] == "float32" and np.isnan(src.nodata)
    assert pred.shape == (100, 100) and np.isfinite(pred).all()


def test_regression_loss_excludes_pixels_without_target():
    task = RegressionTask(in_channels=3, model="unet_lite", num_filters=4)
    x = torch.randn(2, 3, 16, 16)
    y = torch.full((2, 1, 16, 16), torch.nan)

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


@pytest.mark.parametrize(
    "dataset_args, nodata, label_offset",
    [
        ({"nodata_value": 0, "label_offset": 1}, 0, 1),
        ({"nodata_value": None, "label_offset": 0}, None, 0),
        ({"nodata_value": 255, "label_offset": 3}, 255, 3),
        # A nodata value that does not fit into the uint8 predictions
        ({"nodata_value": -9999, "label_offset": 0}, None, 0),
        # Checkpoints from before nodata_value replaced reduce_zero_label
        ({"reduce_zero_label": True}, 0, 1),
        ({"reduce_zero_label": False}, None, 0),
        ({}, None, 0),
    ],
)
def test_predictions_use_the_label_values_of_the_files(
    dataset_args, nodata, label_offset
):
    task = SegmentationTask(
        num_classes=2, in_channels=3, model="unet_lite", num_filters=4
    )
    settings = task._inference_settings(dataset_args)
    assert settings["nodata"] == nodata

    probabilities = np.zeros((2, 4, 4), dtype="float32")
    probabilities[1] = 1
    classes = task.classes_from_probabilities(probabilities)
    assert (classes == 1 + label_offset).all()


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


def _write_target(path: Path, nodata: float | None, dtype: str = "float32") -> None:
    """A target of 5 everywhere, covering only the left half of image_0.tif."""
    with rst.open(TEST_DATA / "image_0.tif") as src:
        profile = {**src.profile, "count": 1, "dtype": dtype, "nodata": nodata}
    profile["width"] //= 2
    with rst.open(path, "w", **profile) as dst:
        dst.write(np.full((profile["height"], profile["width"]), 5, dtype=dtype), 1)


@pytest.mark.parametrize(
    "declared_nodata, dtype", [(None, "float32"), (None, "uint8"), (NODATA, "float32")]
)
def test_regression_targets_are_nan_outside_the_label_files(
    tmp_path, declared_nodata, dtype
):
    """Without a nodata_value, areas without labels must not be read as a target of 0."""
    _write_target(tmp_path / "target.tif", declared_nodata, dtype)
    dataset = get_regression_dataset(
        images_dir=TEST_DATA,
        labels_dir=tmp_path,
        image_glob="image_0.tif",
        label_glob="target.tif",
        cache_size=0,
    )
    xmin, ymin, xmax, ymax = dataset.datasets[0].index.total_bounds

    mask = dataset.datasets[1][xmin:xmax, ymin:ymax]["mask"]

    assert mask.dtype == torch.float32
    assert (mask[:, :50] == 5).all()
    assert mask[:, 50:].isnan().all()


@pytest.mark.parametrize(
    "class_mapping, expected_class", [({0: "a", 5: "b"}, 5), ({5: "a", 6: "b"}, 0)]
)
def test_segmentation_labels_are_unlabelled_outside_the_label_files(
    tmp_path, class_mapping, expected_class
):
    """With a nodata_value, areas without labels must not be read as class 0."""
    _write_target(tmp_path / "labels.tif", nodata=None, dtype="uint8")
    dataset = get_segmentation_dataset(
        images_dir=TEST_DATA,
        labels_dir=tmp_path,
        image_glob="image_0.tif",
        label_glob="labels.tif",
        cache_size=0,
        nodata_value=255,
        class_mapping=class_mapping,
    )
    xmin, ymin, xmax, ymax = dataset.datasets[0].index.total_bounds

    mask = dataset.datasets[1][xmin:xmax, ymin:ymax]["mask"]

    assert mask.dtype == torch.long
    assert (mask[:, :50] == expected_class).all()
    assert (mask[:, 50:] == -1).all()


def test_regression_nodata_pixels_are_read_as_nan():
    """Both for a nodata_value that is given and for the one declared in the files."""
    with rst.open(TEST_DATA / "target_3.tif") as src:
        stored = src.read(1)
        xmin, ymin, xmax, ymax = src.bounds
        assert src.nodata == NODATA and (stored == NODATA).any()

    for nodata_value in (None, NODATA):
        dataset = get_regression_dataset(
            images_dir=TEST_DATA,
            labels_dir=TEST_DATA,
            image_glob="image_3.tif",
            label_glob="target_3.tif",
            nodata_value=nodata_value,
            cache_size=0,
        )
        label_ds = dataset.datasets[1]
        assert label_ds.nodata_value == NODATA

        mask = label_ds[xmin:xmax, ymin:ymax]["mask"]

        np.testing.assert_array_equal(mask.isnan().numpy(), stored == NODATA)
        np.testing.assert_array_equal(label_ds.values_as_in_files(mask), stored)
