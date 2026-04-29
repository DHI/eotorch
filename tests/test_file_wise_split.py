from pathlib import Path

import pytest
from torchgeo.datasets import RasterDataset

from eotorch.data.geodatasets import (
    LabelledRasterDataset,
    PlottabeLabelDataset,
    PlottableImageDataset,
)
from eotorch.data.splits import file_wise_split


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
    PlottableImageDataset.filename_glob = "image_*.tif"
    PlottableImageDataset.all_bands = ["red", "green", "blue"]
    PlottableImageDataset.rgb_bands = ["red", "green", "blue"]

    PlottabeLabelDataset.filename_glob = "label_*.tif"

    data_dir = dataset_files[0][0].parent

    img_ds = PlottableImageDataset(paths=str(data_dir))
    lbl_ds = PlottabeLabelDataset(paths=str(data_dir))

    # Create LabelledRasterDataset
    ds = LabelledRasterDataset(img_ds, lbl_ds)

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
