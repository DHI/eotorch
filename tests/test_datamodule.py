"""Tests for RasterDataModule setup and data loading."""

from pathlib import Path

import pytest
from torchgeo.datasets import RasterDataset

from eotorch.data.datamodules import RasterDataModule, get_dataset_args
from eotorch.data.geodatasets import (
    ClassificationRasterDataset,
    PlottableClassificationDataset,
    PlottableImageDataset,
)

TEST_DATA = Path(__file__).parent / "data"


def _image_dataset(filename_glob: str = "image_*.tif") -> PlottableImageDataset:
    return PlottableImageDataset(
        paths=str(TEST_DATA),
        filename_glob=filename_glob,
        all_bands=["red", "green", "blue"],
        rgb_bands=["red", "green", "blue"],
        cache_size=1,
    )


@pytest.fixture
def image_dataset():
    return _image_dataset()


@pytest.fixture
def label_dataset():
    return PlottableClassificationDataset(
        paths=str(TEST_DATA),
        filename_glob="label_*.tif",
        cache_size=1,
        nodata_value=None,
    )


@pytest.fixture
def segmentation_dataset(image_dataset, label_dataset):
    return ClassificationRasterDataset(image_dataset, label_dataset)


@pytest.fixture
def datamodule(segmentation_dataset):
    return RasterDataModule(
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


def test_datamodule_val_split_at_init(segmentation_dataset):
    """Without a val_dataset, a file-wise split is held out when the module is created."""
    n_files = len(segmentation_dataset)
    dm = RasterDataModule(train_dataset=segmentation_dataset, patch_size=32)
    assert dm.val_dataset is not None
    assert len(dm.train_dataset) + len(dm.val_dataset) == n_files
    assert len(dm.val_dataset) >= 1


def test_datamodule_val_split_none_raises(segmentation_dataset):
    with pytest.raises(ValueError, match="val_split=None"):
        RasterDataModule(
            train_dataset=segmentation_dataset, patch_size=32, val_split=None
        )


def test_datamodule_single_file_without_val_raises():
    single_file_dataset = _image_dataset("image_0.tif")
    assert len(single_file_dataset) == 1
    with pytest.raises(ValueError, match="at least 2"):
        RasterDataModule(train_dataset=single_file_dataset, patch_size=32)


def test_datamodule_setup_with_val_dataset(segmentation_dataset):
    """When val_dataset is provided, no automatic split should happen."""
    dm = RasterDataModule(
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
    dm = RasterDataModule(
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
    dm = RasterDataModule(
        train_dataset=segmentation_dataset,
        val_dataset=segmentation_dataset,
        batch_size=2,
        patch_size=32,
        num_workers=0,
        train_sampler_config={"type": "RandomPatchSampler", "length": 10},
    )
    dm.setup(stage="fit")
    loader = dm.train_dataloader()
    batches = list(loader)
    # With length=10 and batch_size=2, we should get 5 batches
    assert len(batches) == 5


def test_datamodule_batch_sampler(segmentation_dataset):
    dm = RasterDataModule(
        train_dataset=segmentation_dataset,
        val_dataset=segmentation_dataset,
        batch_size=2,
        patch_size=32,
        num_workers=0,
        drop_last=True,
        train_sampler_config={"type": "RandomBatchGeoSampler", "length": 6},
    )
    dm.setup(stage="fit")
    batches = list(dm.train_dataloader())
    assert len(batches) == 3
    assert all(batch["image"].shape[0] == 2 for batch in batches)


def test_datamodule_tuple_patch_size(segmentation_dataset, image_dataset):
    dm = RasterDataModule(
        train_dataset=segmentation_dataset,
        val_dataset=segmentation_dataset,
        predict_dataset=image_dataset,
        batch_size=2,
        patch_size=(32, 16),
        num_workers=0,
    )
    dm.setup(stage="fit")
    dm.setup(stage="predict")
    batch = next(iter(dm.train_dataloader()))
    assert batch["image"].shape[-2:] == (32, 16)
    assert next(iter(dm.predict_dataloader()))["image"].shape[-2:] == (32, 16)


class _TorchGeoImages(RasterDataset):
    filename_glob = "image_*.tif"


class _TorchGeoLabels(RasterDataset):
    filename_glob = "label_*.tif"
    is_image = False


def test_dataset_args_of_torchgeo_datasets():
    """Plain TorchGeo datasets lack some eotorch settings, but their args are saved."""
    images = _TorchGeoImages(str(TEST_DATA))
    args = get_dataset_args(images & _TorchGeoLabels(str(TEST_DATA)))
    assert args["image_glob"] == "image_*.tif"
    assert args["label_glob"] == "label_*.tif"
    assert args["cache_size"] is None
    assert "class_mapping" not in args

    dm = RasterDataModule(train_dataset=images, val_dataset=images, patch_size=32)
    assert dm.hparams["dataset_args"]["image_glob"] == "image_*.tif"


def test_dataset_args_of_other_datasets_raise():
    with pytest.raises(TypeError, match="RasterDataset or IntersectionDataset"):
        get_dataset_args(object())


def test_datamodule_invalid_sampler(segmentation_dataset):
    dm = RasterDataModule(
        train_dataset=segmentation_dataset,
        val_dataset=segmentation_dataset,
        batch_size=2,
        patch_size=32,
        num_workers=0,
        train_sampler_config={"type": "NonExistentSampler"},
    )
    with pytest.raises(ValueError, match="not found in torchgeo.samplers"):
        dm.setup(stage="fit")


def test_preview_data_sampling(datamodule, monkeypatch):
    import folium
    import matplotlib.pyplot as plt

    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gcf()))

    preview = datamodule.preview_data_sampling(max_samples=4, count_labels=True)

    assert isinstance(preview, folium.Map)
    # One pie chart of the label distribution for training and one for validation
    (figure,) = shown
    titles = [ax.get_title() for ax in figure.axes]
    assert titles == ["Label Distribution - Train", "Label Distribution - Validation"]
    plt.close("all")
