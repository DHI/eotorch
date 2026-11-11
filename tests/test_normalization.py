"""Tests for normalization, on the fly during training and inference, and of files."""

import glob
from pathlib import Path

import lightning as L
import numpy as np
import pytest
import rasterio as rst
import torch

from eotorch.data import RasterDataModule, SegmentationTask, get_segmentation_dataset
from eotorch.processing.normalize import band_statistics, zscore_normalize

TEST_DATA = Path(__file__).parent / "data"


@pytest.fixture
def dataset():
    """Dataset using the blue and red band of the test images."""
    return get_segmentation_dataset(
        images_dir=TEST_DATA,
        labels_dir=TEST_DATA,
        image_glob="image_*.tif",
        label_glob="label_*.tif",
        all_image_bands=("red", "green", "blue"),
        bands_to_return=("blue", "red"),
        cache_size=0,
    )


def test_band_statistics(dataset):
    mean, std = band_statistics(dataset)

    data = np.concatenate(
        [
            rst.open(f).read([3, 1]).reshape(2, -1)
            for f in sorted(glob.glob(str(TEST_DATA / "image_*.tif")))
        ],
        axis=1,
    )
    np.testing.assert_allclose(mean, data.mean(axis=1))
    np.testing.assert_allclose(std, data.std(axis=1))


def test_datamodule_normalizes_batches(dataset):
    datamodule = RasterDataModule(
        train_dataset=dataset, patch_size=32, mean=np.array([10.0, 20.0]), std=[2, 4]
    )
    image = torch.full((2, 2, 8, 8), 30.0)

    batch = datamodule.on_after_batch_transfer({"image": image}, 0)

    assert batch["image"][:, 0].eq(10.0).all() and batch["image"][:, 1].eq(2.5).all()
    assert datamodule.hparams["mean"] == [10.0, 20.0]
    assert datamodule.hparams["std"] == [2.0, 4.0]


@pytest.mark.parametrize("stage", ["training", "validating"])
def test_augmentations_are_applied_after_normalization(dataset, stage):
    """Augmentations are set as in TorchGeo, e.g. train_aug for training only."""
    import kornia.augmentation as K
    from types import SimpleNamespace

    datamodule = RasterDataModule(
        train_dataset=dataset, patch_size=32, mean=[10.0, 20.0], std=[2, 4]
    )
    datamodule.train_aug = K.AugmentationSequential(
        K.RandomHorizontalFlip(p=1.0), data_keys=None, keepdim=True
    )
    trainer_state = dict.fromkeys(
        ["training", "validating", "sanity_checking", "testing", "predicting"], False
    )
    datamodule.trainer = SimpleNamespace(**{**trainer_state, stage: True})
    image = torch.arange(8.0).expand(2, 2, 8, 8) + 10

    batch = datamodule.on_after_batch_transfer({"image": image.clone()}, 0)

    normalized = (image[:, 0] - 10) / 2
    if stage == "training":
        normalized = normalized.flip(-1)
    torch.testing.assert_close(batch["image"][:, 0], normalized)


def test_inference_selects_and_normalizes_bands(dataset, tmp_path):
    mean, std = band_statistics(dataset)
    datamodule = RasterDataModule(
        train_dataset=dataset,
        val_dataset=dataset,
        batch_size=2,
        patch_size=32,
        mean=mean,
        std=std,
    )
    task = SegmentationTask(
        num_classes=4, in_channels=2, model="unet_lite", num_filters=4, ignore_index=-1
    )
    trainer = L.Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(task, datamodule=datamodule)
    checkpoint = tmp_path / "model.ckpt"
    trainer.save_checkpoint(checkpoint)

    loaded = SegmentationTask.load_from_checkpoint(checkpoint)
    loaded._set_input_preprocessing(
        torch.load(checkpoint, weights_only=False)["datamodule_hyper_parameters"]
    )
    raw = torch.rand(1, 3, 4, 4) * 255
    expected = (raw[:, [2, 0]] - torch.tensor(mean).view(-1, 1, 1)) / torch.tensor(
        std
    ).view(-1, 1, 1)
    torch.testing.assert_close(loaded._prepare_input(raw), expected)

    # The test image has 3 bands, the model expects 2
    out = SegmentationTask.predict_on_tif_file(
        TEST_DATA / "image_0.tif",
        checkpoint,
        out_file_path=tmp_path / "pred.tif",
        progress_bar=False,
    )
    with rst.open(out) as src:
        assert src.shape == (100, 100)


def test_zscore_normalize_counts_valid_pixels_per_band(tmp_path):
    """The image is smaller than the 256 px windows the files are read in."""
    with rst.open(TEST_DATA / "image_0.tif") as src:
        profile = src.profile
        data = src.read().astype("float64")
    data[1, :50] = 0  # zeros are left out, here only in the second band
    image = tmp_path / "image.tif"
    with rst.open(image, "w", **profile) as dst:
        dst.write(data.astype(profile["dtype"]))

    stats = zscore_normalize(image, tmp_path / "out", bands=[1, 2])

    valid = np.where(data[:2] == 0, np.nan, data[:2])
    np.testing.assert_allclose(stats["mean"], np.nanmean(valid, axis=(1, 2)))
    np.testing.assert_allclose(stats["std"], np.nanstd(valid, axis=(1, 2)))
    with rst.open(tmp_path / "out" / "image_zscore.tif") as src:
        assert src.count == 2
        normalized = src.read(2)[50:]
    np.testing.assert_allclose(normalized.mean(), 0, atol=1e-4)
