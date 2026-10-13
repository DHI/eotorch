"""Tests for patch placement in generate_train_val_patches."""

import geopandas as gpd
import numpy as np
import pytest
import rasterio as rst
import shapely
from rasterio.transform import from_origin

from eotorch.processing.patch import generate_train_val_patches

PATCH_SIZE = 32
ORIGIN_X, ORIGIN_Y = 500000.0, 6200000.0
CRS = 'EPSG:25832'


@pytest.fixture
def sparse_scene(tmp_path):
    """A 320 x 320 px raster with small, scattered polygons.

    Each polygon lies within a single 32 px block. The last one ends exactly on a block
    edge, so the neighbouring block only touches it.
    """
    img_path = tmp_path / 'image.tif'
    rng = np.random.default_rng(0)
    with rst.open(
        img_path, 'w', driver='GTiff', width=320, height=320, count=3, dtype='uint8',
        crs=CRS, transform=from_origin(ORIGIN_X, ORIGIN_Y, 1, 1),
    ) as dst:
        dst.write(rng.integers(1, 255, (3, 320, 320), dtype=np.uint8))

    def px_box(col, row, size):
        return shapely.box(ORIGIN_X + col, ORIGIN_Y - row - size, ORIGIN_X + col + size, ORIGIN_Y - row)

    polygons = [px_box(10, 10, 8), px_box(100, 40, 5), px_box(200, 260, 12), px_box(280, 140, 6)]
    polygons += [px_box(40 + 60 * i, 200, 4) for i in range(4)]  # the last spans cols 220-224
    shp_path = tmp_path / 'target.shp'
    gpd.GeoDataFrame(geometry=polygons, crs=CRS).to_file(shp_path)

    return img_path, shp_path, polygons


def _patch_footprints(patch_dir):
    footprints = []
    for path in sorted(patch_dir.glob('*_image.tiff')):
        with rst.open(path) as src:
            footprints.append((src.bounds, src.transform))
    return footprints


def _generate(img_path, shp_path, tmp_path, n_random_offsets, **kwargs):
    train_dir, val_dir = tmp_path / 'train', tmp_path / 'val'
    generate_train_val_patches(
        src_path=img_path,
        feature_shps=[shp_path],
        train_patch_dir=train_dir,
        val_patch_dir=val_dir,
        patch_size=PATCH_SIZE,
        val_fraction=0.25,
        n_random_offsets=n_random_offsets,
        indexes=[1, 2, 3],
        label_sources=[shp_path],
        class_map={'target': 1},
        random_seed=1,
        **kwargs,
    )
    return train_dir, val_dir


@pytest.mark.parametrize('n_random_offsets', [0, 3])
def test_all_patches_contain_labels(sparse_scene, tmp_path, n_random_offsets):
    img_path, shp_path, polygons = sparse_scene
    features = shapely.unary_union(polygons)
    train_dir, val_dir = _generate(img_path, shp_path, tmp_path, n_random_offsets)

    for patch_dir in (train_dir, val_dir):
        footprints = _patch_footprints(patch_dir)
        assert footprints
        for bounds, _ in footprints:
            overlap = features.intersection(shapely.box(*bounds)).area
            assert overlap > 0, f'{patch_dir.name} patch {bounds} misses all features'

        for label_path in patch_dir.glob('*_label.tiff'):
            with rst.open(label_path) as src:
                assert (src.read(1) > 0).any(), f'{label_path.name} has no labelled pixels'


def test_full_coverage_keeps_small_features_in_view(sparse_scene, tmp_path):
    img_path, shp_path, polygons = sparse_scene
    train_dir, _ = _generate(img_path, shp_path, tmp_path, n_random_offsets=5, min_feature_coverage=1.0)

    footprints = _patch_footprints(train_dir)
    assert footprints
    for bounds, _ in footprints:
        patch = shapely.box(*bounds)
        for poly in polygons:
            if poly.intersection(patch).area > 0:
                assert patch.contains(poly), f'patch {bounds} cuts off part of {poly.bounds}'


def test_zero_offsets_gives_one_grid_aligned_patch_per_block(sparse_scene, tmp_path):
    img_path, shp_path, _ = sparse_scene
    train_dir, val_dir = _generate(img_path, shp_path, tmp_path, n_random_offsets=0)

    train = _patch_footprints(train_dir)
    val = _patch_footprints(val_dir)

    blocks = set()
    for bounds, _ in train + val:
        col = bounds.left - ORIGIN_X
        row = ORIGIN_Y - bounds.top
        assert col % PATCH_SIZE == 0 and row % PATCH_SIZE == 0, f'patch at ({row}, {col}) is not grid-aligned'
        blocks.add((row, col))

    assert len(blocks) == len(train) + len(val), 'blocks should not be sampled twice'
    assert len(blocks) == 8, 'expected one block per polygon, excluding the block that only touches one'


def test_random_offsets_add_patches_within_half_a_patch(sparse_scene, tmp_path):
    img_path, shp_path, _ = sparse_scene
    aligned_dir, _ = _generate(img_path, shp_path, tmp_path / 'aligned', n_random_offsets=0)
    offset_dir, _ = _generate(img_path, shp_path, tmp_path / 'offset', n_random_offsets=3)

    n_aligned = len(_patch_footprints(aligned_dir))
    offset = _patch_footprints(offset_dir)
    assert len(offset) > n_aligned

    for bounds, _ in offset:
        col = (bounds.left - ORIGIN_X) % PATCH_SIZE
        row = (ORIGIN_Y - bounds.top) % PATCH_SIZE
        # distance to the nearest grid line is at most half a patch
        assert min(col, PATCH_SIZE - col) <= PATCH_SIZE // 2
        assert min(row, PATCH_SIZE - row) <= PATCH_SIZE // 2
