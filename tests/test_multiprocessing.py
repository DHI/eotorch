"""Datasets must survive pickling, so DataLoader workers can be spawned (as on Windows)."""

import pickle
from pathlib import Path

import pytest
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples
from torchgeo.samplers import GriddedPatchSampler

from eotorch.data import get_segmentation_dataset
from eotorch.data.geodatasets import (
    PlottableClassificationDataset,
    PlottableImageDataset,
)

TEST_DATA = Path(__file__).parent / "data"


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
def dataset():
    return get_segmentation_dataset(
        images_dir=TEST_DATA,
        labels_dir=TEST_DATA,
        image_glob="image_*.tif",
        label_glob="label_*.tif",
        all_image_bands=("red", "green", "blue"),
        class_mapping={0: "a", 1: "b", 2: "c"},
        cache_size=1,
    )


def test_pickle_keeps_class_config(dataset):
    data = pickle.dumps(dataset)
    # Simulate a freshly spawned process, where the class attributes have their defaults
    PlottableImageDataset.filename_glob = "*"
    PlottableClassificationDataset.class_mapping = None

    restored = pickle.loads(data)
    image_ds, label_ds = restored.datasets
    assert image_ds.filename_glob == "image_*.tif"
    assert label_ds.class_mapping == {0: "a", 1: "b", 2: "c"}
    assert (
        restored[next(iter(GriddedPatchSampler(restored, size=32)))]["image"].shape[0]
        == 3
    )


def test_spawned_workers(dataset):
    loader = DataLoader(
        dataset,
        sampler=GriddedPatchSampler(dataset, size=32),
        batch_size=2,
        num_workers=2,
        multiprocessing_context="spawn",
        collate_fn=stack_samples,
    )
    batch = next(iter(loader))
    assert batch["image"].shape == (2, 3, 32, 32)
