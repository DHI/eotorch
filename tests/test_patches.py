"""Tests for generating patch files and training on them."""

from pathlib import Path

import lightning as L
import numpy as np
import pytest
import rasterio as rst
from rasterio.warp import Resampling, calculate_default_transform, reproject

from eotorch.data import RasterDataModule, SegmentationTask, get_segmentation_dataset
from eotorch.data.geodatasets import (
    PlottableClassificationDataset,
    PlottableImageDataset,
)
from eotorch.data.samplers import FileSampler
from eotorch.processing import generate_patches_from_files

TEST_DATA = Path(__file__).parent / "data"
PATCH_SIZE = 32


@pytest.fixture(autouse=True)
def _restore_class_attrs():
    """get_segmentation_dataset configures the dataset classes, so restore them."""
    saved = {
        cls: {k: getattr(cls, k) for k in cls._class_config if hasattr(cls, k)}
        for cls in (PlottableImageDataset, PlottableClassificationDataset)
    }
    yield
    for cls, attrs in saved.items():
        for k, v in attrs.items():
            setattr(cls, k, v)


@pytest.fixture
def patch_dir(tmp_path):
    for i in range(2):
        generate_patches_from_files(
            TEST_DATA / f"image_{i}.tif",
            TEST_DATA / f"label_{i}.tif",
            tmp_path,
            patch_size=PATCH_SIZE,
            val_fraction=0.3,
            n_random_offsets=2,
            empty_img_threshold=None,
            show_progress=False,
        )
    return tmp_path


def _patch_dataset(directory):
    return get_segmentation_dataset(
        images_dir=directory,
        labels_dir=directory,
        image_glob="*_image.tif",
        label_glob="*_label.tif",
        all_image_bands=("red", "green", "blue"),
        cache_size=0,
    )


def _bounds(path):
    with rst.open(path) as src:
        return src.bounds


def _overlap(a, b):
    return (
        a.left < b.right and b.left < a.right and a.bottom < b.top and b.bottom < a.top
    )


def test_patches_are_georeferenced_and_split(patch_dir):
    train = sorted((patch_dir / "train").glob("*_image.tif"))
    val = sorted((patch_dir / "val").glob("*_image.tif"))
    assert train and val

    with rst.open(train[0]) as src, rst.open(TEST_DATA / "image_0.tif") as full:
        assert src.shape == (PATCH_SIZE, PATCH_SIZE)
        assert src.dtypes == full.dtypes and src.crs == full.crs
        window = rst.windows.from_bounds(*src.bounds, full.transform)
        np.testing.assert_array_equal(src.read(), full.read(window=window))

    for t in train:
        for v in val:
            if t.name.split("_")[1] == v.name.split("_")[1]:  # from the same image
                assert not _overlap(_bounds(t), _bounds(v))


def test_labels_are_reprojected_onto_image_grid(tmp_path):
    # Label raster with half the resolution of the image
    with rst.open(TEST_DATA / "label_0.tif") as src:
        transform, width, height = calculate_default_transform(
            src.crs, src.crs, src.width, src.height, *src.bounds, resolution=20
        )
        labels = np.zeros((1, height, width), dtype="uint8")
        reproject(
            rst.band(src, 1),
            labels,
            dst_transform=transform,
            dst_crs=src.crs,
            resampling=Resampling.mode,
        )
        profile = {
            **src.profile,
            "transform": transform,
            "width": width,
            "height": height,
        }
    with rst.open(tmp_path / "coarse_label.tif", "w", **profile) as dst:
        dst.write(labels)

    generate_patches_from_files(
        TEST_DATA / "image_0.tif",
        tmp_path / "coarse_label.tif",
        tmp_path / "patches",
        patch_size=PATCH_SIZE,
        val_fraction=0,
        show_progress=False,
    )
    label_patches = list((tmp_path / "patches" / "train").glob("*_label.tif"))
    image_patch = label_patches[0].with_name(
        label_patches[0].name.replace("_label", "_image")
    )
    assert _bounds(label_patches[0]) == _bounds(image_patch)


def test_file_sampler_samples_each_image_file_once(patch_dir):
    dataset = _patch_dataset(patch_dir / "train")
    n_files = len(list((patch_dir / "train").glob("*_image.tif")))

    sampler = FileSampler(dataset, shuffle=True)
    # Overlapping patches create extra rows in the index of the IntersectionDataset
    assert len(dataset.index) > n_files
    assert len(sampler) == len(list(sampler)) == n_files
    for index in sampler:
        assert dataset[index]["image"].shape == (3, PATCH_SIZE, PATCH_SIZE)


def test_train_and_predict_on_patches(patch_dir, tmp_path):
    datamodule = RasterDataModule(
        train_dataset=_patch_dataset(patch_dir / "train"),
        val_dataset=_patch_dataset(patch_dir / "val"),
        batch_size=4,
        patch_size=None,
    )
    assert datamodule.hparams["patch_size"] == PATCH_SIZE

    task = SegmentationTask(
        num_classes=4, in_channels=3, model="unet_lite", num_filters=4, ignore_index=-1
    )
    trainer = L.Trainer(
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(task, datamodule=datamodule)
    assert trainer.num_training_batches == -(-len(datamodule.train_sampler) // 4)

    checkpoint = tmp_path / "model.ckpt"
    trainer.save_checkpoint(checkpoint)
    out = SegmentationTask.predict_on_tif_file(
        TEST_DATA / "image_2.tif",
        checkpoint,
        out_file_path=tmp_path / "pred.tif",
        progress_bar=False,
    )
    with rst.open(out) as src:
        assert src.shape == (100, 100)
