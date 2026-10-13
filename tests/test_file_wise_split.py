from pathlib import Path

import pandas as pd
import pytest
import torch
from torchgeo.datasets import RasterDataset

from eotorch.data.geodatasets import (
    ClassificationRasterDataset,
    PlottableClassificationDataset,
    PlottableImageDataset,
)
from eotorch.data.splits import file_wise_split


# Subclasses, so the class attributes don't leak into other test modules
class ImageDataset(PlottableImageDataset):
    filename_glob = "image_*.tif"
    all_bands = ["red", "green", "blue"]
    rgb_bands = ["red", "green", "blue"]


class LabelDataset(PlottableClassificationDataset):
    filename_glob = "label_*.tif"


@pytest.fixture(scope="module")
def dataset_files():
    data_dir = Path(__file__).parent / "data"

    files = []
    for i in range(4):
        img_path = data_dir / f"image_{i}.tif"
        lbl_path = data_dir / f"label_{i}.tif"
        files.append((img_path, lbl_path))

    return files


def test_split_raster_dataset_ratios(dataset_files):
    img_paths = [str(f[0]) for f in dataset_files]

    # Define a minimal subclass to avoid abstract method errors if any
    class SimpleRasterDataset(RasterDataset):
        filename_glob = "*.tif"

    ds = SimpleRasterDataset(paths=img_paths)

    splits = file_wise_split(ds, ratios_or_counts=[0.5, 0.5])

    assert len(splits) == 2
    assert len(splits[0]) + len(splits[1]) == 4
    # Ensure no overlap
    files0 = set(splits[0].paths)
    files1 = set(splits[1].paths)
    assert files0.isdisjoint(files1)


def test_split_intersection_dataset_ratios(dataset_files):
    data_dir = dataset_files[0][0].parent

    img_ds = ImageDataset(paths=str(data_dir))
    lbl_ds = LabelDataset(paths=str(data_dir))

    # Create ClassificationRasterDataset
    ds = ClassificationRasterDataset(img_ds, lbl_ds)

    # This is expected to fail before the fix
    splits = file_wise_split(ds, ratios_or_counts=[0.5, 0.5])

    assert len(splits) == 2
    assert len(splits[0]) + len(splits[1]) == 4

    # Check that we can access items in the splits (verifies reconstruction)
    assert len(splits[0]) > 0
    _ = splits[0][
        splits[0].bounds
    ]  # Access by bounds of the whole dataset (simplified)
    # Or just check index
    assert not splits[0].index.empty


def test_split_explicit_files(dataset_files):
    img_paths = [str(f[0]) for f in dataset_files]

    class SimpleRasterDataset(RasterDataset):
        filename_glob = "*.tif"

    ds = SimpleRasterDataset(paths=img_paths)

    val_files = [img_paths[0]]
    test_files = [img_paths[1]]

    splits = file_wise_split(ds, val_img_files=val_files, test_img_files=test_files)

    assert len(splits) == 3  # train, val, test

    # Check val
    assert len(splits[1]) == 1
    assert str(Path(splits[1].paths[0]).resolve()) == str(Path(val_files[0]).resolve())

    # Check test
    assert len(splits[2]) == 1
    assert str(Path(splits[2].paths[0]).resolve()) == str(Path(test_files[0]).resolve())

    # Check train (remainder)
    assert len(splits[0]) == 2


def test_split_intersection_dataset_explicit_files(dataset_files):
    data_dir = dataset_files[0][0].parent
    img_ds = ImageDataset(paths=str(data_dir))
    lbl_ds = LabelDataset(paths=str(data_dir))
    ds = ClassificationRasterDataset(img_ds, lbl_ds)

    val_file = Path(dataset_files[0][0]).resolve()
    train_ds, val_ds = file_wise_split(ds, val_img_files=[str(val_file)])

    assert [Path(p).resolve() for p in val_ds.datasets[0].paths] == [val_file]
    assert val_file not in [Path(p).resolve() for p in train_ds.datasets[0].paths]
    assert len(train_ds.datasets[0].paths) == 3

    # The split datasets keep a temporal index, so samplers and indexing work
    for split in (train_ds, val_ds):
        assert isinstance(split.index.index, pd.IntervalIndex)
        sample = split[split.bounds]
        assert "image" in sample and "mask" in sample


def test_intersection_dataset_keeps_transforms_and_temporal_index(dataset_files):
    data_dir = dataset_files[0][0].parent

    def transforms(sample):
        sample["transformed"] = True
        return sample

    ds = ClassificationRasterDataset(
        ImageDataset(paths=str(data_dir)),
        LabelDataset(paths=str(data_dir)),
        transforms=transforms,
    )

    assert ds.transforms is transforms
    assert isinstance(ds.index.index, pd.IntervalIndex)

    sample = ds[ds.bounds]
    assert sample["transformed"]
    # Bounds are a tensor so Lightning can move samples between devices
    assert isinstance(sample["bounds"], torch.Tensor)
    assert sample["bounds"].dtype == torch.float64
