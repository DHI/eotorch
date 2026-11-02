from pathlib import Path

import pytest

from eotorch.data import get_regression_dataset, get_segmentation_dataset

DATA_DIR = Path(__file__).parent / "data"


def _dataset(get_dataset, i: int, **kwargs):
    # Selecting files by path rather than glob, as get_*_dataset stores the globs on
    # the dataset classes, which would leak into other tests
    return get_dataset(
        images_dir=DATA_DIR / f"image_{i}.tif",
        labels_dir=DATA_DIR / f"label_{i}.tif",
        **kwargs,
    )


@pytest.mark.parametrize(
    "get_dataset", [get_segmentation_dataset, get_regression_dataset]
)
def test_union(get_dataset):
    ds1, ds2 = _dataset(get_dataset, 0), _dataset(get_dataset, 1)
    union = ds1 | ds2

    assert type(union) is type(ds1)
    for i in range(2):
        assert len(union.datasets[i].index) == 2
        assert union.datasets[i].crs == ds1.crs
    assert len(union) >= len(ds1)
    union[union.bounds]


def test_union_keeps_label_settings():
    ds1, ds2 = (
        _dataset(get_segmentation_dataset, i, reduce_zero_label=False) for i in range(2)
    )
    assert not (ds1 | ds2).datasets[1].reduce_zero_label


def test_union_rejects_other_types():
    ds = _dataset(get_segmentation_dataset, 0)
    with pytest.raises(TypeError):
        ds | _dataset(get_regression_dataset, 0)
    with pytest.raises(TypeError):
        ds.datasets[0] | ds


def test_intersection():
    ds = _dataset(get_segmentation_dataset, 0)
    intersection = ds & ds

    assert type(intersection) is type(ds)
    assert len(intersection) == len(ds)
