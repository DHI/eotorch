from pathlib import Path

import pytest

from eotorch.data import get_regression_dataset, get_segmentation_dataset

DATA_DIR = Path(__file__).parent / "data"


def _dataset(get_dataset, i: int, **kwargs):
    labels = "target" if get_dataset is get_regression_dataset else "label"
    return get_dataset(
        images_dir=DATA_DIR / f"image_{i}.tif",
        labels_dir=DATA_DIR / f"{labels}_{i}.tif",
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


def test_datasets_keep_their_own_settings():
    ds1 = _dataset(
        get_segmentation_dataset, 0, class_mapping={1: "a"}, rgb_bands=("b", "g", "r")
    )
    ds2 = _dataset(get_segmentation_dataset, 1, class_mapping={5: "z"})
    assert ds1.datasets[0].rgb_bands == ("b", "g", "r")
    assert ds1.datasets[1].class_mapping == {1: "a"}
    assert ds2.datasets[1].class_mapping == {5: "z"}


def test_union_keeps_label_settings():
    ds1, ds2 = (
        _dataset(
            get_segmentation_dataset, i, reduce_zero_label=False, class_mapping={1: "a"}
        )
        for i in range(2)
    )
    union_labels = (ds1 | ds2).datasets[1]
    assert not union_labels.reduce_zero_label
    assert union_labels.class_mapping == {1: "a"}


def test_union_rejects_other_types():
    ds = _dataset(get_segmentation_dataset, 0)
    with pytest.raises(TypeError):
        ds | _dataset(get_regression_dataset, 0)
    with pytest.raises(TypeError):
        ds.datasets[0] | ds


@pytest.mark.parametrize(
    "get_dataset", [get_segmentation_dataset, get_regression_dataset]
)
def test_intersection_is_not_supported(get_dataset):
    ds = _dataset(get_dataset, 0)
    with pytest.raises(TypeError, match="can't be intersected"):
        ds & ds
