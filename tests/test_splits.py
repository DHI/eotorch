"""Test functions for the splits module."""

import json
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box

from eotorch.data import get_segmentation_dataset
from eotorch.data.splits import aoi_split

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def tile_dataset():
    return get_segmentation_dataset(
        images_dir=DATA_DIR / "image_0.tif", labels_dir=DATA_DIR / "label_0.tif"
    )


def _left_half(ds):
    minx, miny, maxx, maxy = ds.index.total_bounds
    return minx, miny, (minx + maxx) / 2, maxy


def _aoi_over_left_half(ds):
    """AOI box covering the left half of the dataset, in the dataset CRS."""
    minx, miny, midx, maxy = _left_half(ds)
    return box(minx - 100, miny - 100, midx, maxy + 100)


def _split_bounds(ds, **kwargs):
    train, val = aoi_split(dataset=ds, **kwargs)
    assert len(train) > 0
    return tuple(val.index.total_bounds)


def test_aoi_split_reprojects_aoi_file(tile_dataset, tmp_path):
    aoi_file = tmp_path / "aoi.geojson"
    gpd.GeoDataFrame(
        geometry=[_aoi_over_left_half(tile_dataset)], crs=tile_dataset.crs
    ).to_crs("EPSG:4326").to_file(aoi_file)

    bounds = _split_bounds(tile_dataset, aoi_files=aoi_file)
    # The bounding box of the reprojected AOI is slightly larger than the original
    assert bounds == pytest.approx(_left_half(tile_dataset), abs=50)


def test_aoi_split_file_without_crs_uses_dataset_crs(tile_dataset, tmp_path):
    aoi_file = tmp_path / "aoi.shp"
    gpd.GeoDataFrame(geometry=[_aoi_over_left_half(tile_dataset)]).to_file(aoi_file)

    bounds = _split_bounds(tile_dataset, aoi_files=aoi_file)
    assert bounds == pytest.approx(_left_half(tile_dataset))


def test_aoi_split_crs_for_file_without_crs(tile_dataset, tmp_path):
    aoi_file = tmp_path / "aoi.shp"
    aoi = gpd.GeoSeries([_aoi_over_left_half(tile_dataset)], crs=tile_dataset.crs)
    gpd.GeoDataFrame(
        geometry=aoi.to_crs("EPSG:4326").set_crs(None, allow_override=True)
    ).to_file(aoi_file)

    bounds = _split_bounds(tile_dataset, aoi_files=aoi_file, crs="EPSG:4326")
    assert bounds == pytest.approx(_left_half(tile_dataset), abs=50)


def test_aoi_split_crs_overrides_file_crs(tile_dataset, tmp_path):
    # GeoJSON without CRS information is read as EPSG:4326, although these
    # coordinates are in the dataset CRS
    aoi_file = tmp_path / "aoi.geojson"
    geometry = _aoi_over_left_half(tile_dataset).__geo_interface__
    feature = {"type": "Feature", "properties": {}, "geometry": geometry}
    aoi_file.write_text(
        json.dumps({"type": "FeatureCollection", "features": [feature]})
    )

    with pytest.warns(UserWarning, match="is read as EPSG:4326"):
        bounds = _split_bounds(tile_dataset, aoi_files=aoi_file, crs=tile_dataset.crs)
    assert bounds == pytest.approx(_left_half(tile_dataset))


def test_aoi_split_restricts_image_and_label_datasets(tmp_path):
    ds = get_segmentation_dataset(
        images_dir=[DATA_DIR / "image_0.tif", DATA_DIR / "image_3.tif"],
        labels_dir=[DATA_DIR / "label_0.tif", DATA_DIR / "label_3.tif"],
    )
    footprints = ds.datasets[0].index
    tile_0, tile_3 = (
        footprints.geometry[footprints.filepath.str.endswith(f"image_{i}.tif")].iloc[0]
        for i in (0, 3)
    )
    # The AOI covers all of tile 3 and the left half of tile 0
    minx, miny, maxx, maxy = tile_0.bounds
    aoi_file = tmp_path / "aoi.geojson"
    gpd.GeoDataFrame(
        geometry=[
            tile_3.buffer(100),
            box(minx - 100, miny - 100, (minx + maxx) / 2, maxy + 100),
        ],
        crs=ds.crs,
    ).to_file(aoi_file)

    train, val = aoi_split(dataset=ds, aoi_files=aoi_file)

    def files(split):
        return [
            sorted(Path(f).name for f in sub.index.filepath) for sub in split.datasets
        ]

    assert files(train) == [["image_0.tif"], ["label_0.tif"]]
    assert files(val) == [
        ["image_0.tif", "image_3.tif"],
        ["label_0.tif", "label_3.tif"],
    ]
    assert sorted(Path(p).name for p in train.datasets[0].paths) == ["image_0.tif"]
    # Files cut by the AOI are clipped to the part within each split
    for split in (train, val):
        area = split.index.union_all()
        for sub in split.datasets:
            assert sub.index.union_all().symmetric_difference(area).area < 1e-6
