"""Tests for rasterizing vector labels."""

import geopandas as gpd
import numpy as np
import rasterio as rst
from rasterio.transform import from_origin
from shapely.geometry import box

from eotorch.processing.rasterize import DataframeSource


def test_labels_are_cropped_with_a_margin_on_all_sides(tmp_path):
    # A 1000 x 1000 px raster of 1 m pixels, with a 100 x 100 px label from 400 to 500
    transform = from_origin(0, 1000, 1, 1)
    labels = gpd.GeoDataFrame(
        {"class": [1]}, geometry=[box(400, 500, 500, 600)], crs="EPSG:32632"
    )
    source = DataframeSource(
        labels, "class", transform=transform, shape=(1000, 1000), crs="EPSG:32632"
    )
    out = tmp_path / "labels.tif"
    source.rasterize_polygons(out, min_size=0)

    with rst.open(out) as src:
        data = src.read(1)
        left, bottom, right, top = src.bounds
    assert (left, bottom, right, top) == (350, 450, 550, 650)
    rows, cols = np.nonzero(data)
    assert rows.min() == cols.min() == 50
    assert data.shape[0] - 1 - rows.max() == data.shape[1] - 1 - cols.max() == 50


def test_cropped_labels_are_at_least_min_size(tmp_path):
    # A 20 x 20 px label in the top left corner of the raster
    labels = gpd.GeoDataFrame(
        {"class": [1]}, geometry=[box(0, 980, 20, 1000)], crs="EPSG:32632"
    )
    source = DataframeSource(
        labels,
        "class",
        transform=from_origin(0, 1000, 1, 1),
        shape=(1000, 1000),
        crs="EPSG:32632",
    )
    out = tmp_path / "labels.tif"
    source.rasterize_polygons(out, min_size=300)

    with rst.open(out) as src:
        assert src.shape == (300, 300)
        assert src.bounds.left == 0 and src.bounds.top == 1000
