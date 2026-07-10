"""Tests for SegmentationDataModule setup and data loading."""

from pathlib import Path

import pytest
import torch
from torchgeo.datasets import IntersectionDataset

from eotorch.data.datamodules import SegmentationDataModule, get_dataset_args
from eotorch.data.geodatasets import (
    ClassificationRasterDataset,
    PlottableClassificationDataset,
    PlottableImageDataset,
)

TEST_DATA = Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def _restore_class_attrs():
    """Save and restore class-level attributes modified by tests."""
    saved = {
        "img_glob": PlottableImageDataset.filename_glob,
        "img_all_bands": PlottableImageDataset.all_bands,
        "img_rgb_bands": PlottableImageDataset.rgb_bands,
        "lbl_glob": PlottableClassificationDataset.filename_glob,
    }
    yield
    PlottableImageDataset.filename_glob = saved["img_glob"]
    PlottableImageDataset.all_bands = saved["img_all_bands"]
    PlottableImageDataset.rgb_bands = saved["img_rgb_bands"]
    PlottableClassificationDataset.filename_glob = saved["lbl_glob"]


@pytest.fixture
def image_dataset():
    PlottableImageDataset.filename_glob = "image_*.tif"
    PlottableImageDataset.all_bands = ["red", "green", "blue"]
    PlottableImageDataset.rgb_bands = ["red", "green", "blue"]
    return PlottableImageDataset(paths=str(TEST_DATA), cache_size=1)


@pytest.fixture
def label_dataset():
    PlottableClassificationDataset.filename_glob = "label_*.tif"
    return PlottableClassificationDataset(
        paths=str(TEST_DATA), cache_size=1, reduce_zero_label=False
    )


@pytest.fixture
def segmentation_dataset(image_dataset, label_dataset):
    return ClassificationRasterDataset(image_dataset, label_dataset)


@pytest.fixture
def datamodule(segmentation_dataset):
    return SegmentationDataModule(
        train_dataset=segmentation_dataset,
        batch_size=2,
        patch_size=32,
        num_workers=0,
    )


def test_datamodule_setup_fit(datamodule):
    datamodule.setup(stage="fit")
    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is not None
    assert datamodule.train_sampler is not None
    assert datamodule.val_sampler is not None


def test_datamodule_setup_with_val_dataset(segmentation_dataset):
    """When val_dataset is provided, no automatic split should happen."""
    dm = SegmentationDataModule(
        train_dataset=segmentation_dataset,
        val_dataset=segmentation_dataset,
        batch_size=2,
        patch_size=32,
        num_workers=0,
    )
    dm.setup(stage="fit")
    # val_dataset should remain the one we passed in
    assert dm.val_dataset is segmentation_dataset


def test_datamodule_train_dataloader(datamodule):
    datamodule.setup(stage="fit")
    loader = datamodule.train_dataloader()
    batch = next(iter(loader))
    assert "image" in batch
    assert batch["image"].ndim == 4  # (B, C, H, W)
    assert batch["image"].shape[-1] == 32
    assert batch["image"].shape[-2] == 32


def test_datamodule_val_dataloader(datamodule):
    datamodule.setup(stage="fit")
    loader = datamodule.val_dataloader()
    batch = next(iter(loader))
    assert "image" in batch
    assert "mask" in batch


def test_datamodule_predict_setup(image_dataset):
    dm = SegmentationDataModule(
        predict_dataset=image_dataset,
        batch_size=4,
        patch_size=32,
        num_workers=0,
    )
    dm.setup(stage="predict")
    assert dm.predict_sampler is not None
    loader = dm.predict_dataloader()
    batch = next(iter(loader))
    assert "image" in batch


def test_datamodule_custom_sampler(segmentation_dataset):
    dm = SegmentationDataModule(
        train_dataset=segmentation_dataset,
        val_dataset=segmentation_dataset,
        batch_size=2,
        patch_size=32,
        num_workers=0,
        train_sampler_config={"type": "RandomGeoSampler", "length": 10},
    )
    dm.setup(stage="fit")
    loader = dm.train_dataloader()
    batches = list(loader)
    # With length=10 and batch_size=2, we should get 5 batches
    assert len(batches) == 5


def test_datamodule_invalid_sampler(segmentation_dataset):
    dm = SegmentationDataModule(
        train_dataset=segmentation_dataset,
        val_dataset=segmentation_dataset,
        batch_size=2,
        patch_size=32,
        num_workers=0,
        train_sampler_config={"type": "NonExistentSampler"},
    )
    with pytest.raises(ValueError, match="not found in torchgeo.samplers"):
        dm.setup(stage="fit")
