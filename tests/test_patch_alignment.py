"""Tests for aligning label rasters to image rasters in generate_patches_from_files."""

from pathlib import Path

import numpy as np
import pytest
import rasterio as rst
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

from eotorch.processing.patch import (
    _read_aligned_image_and_labels,
    generate_patches_from_files,
)
from rasterio.enums import Resampling

UTM = "EPSG:32632"
WEST, NORTH = 500_000.0, 6_200_000.0


def _write(path: Path, data: np.ndarray, transform, crs=UTM, nodata=None) -> Path:
    if data.ndim == 2:
        data = data[None]
    with rst.open(
        path, "w", driver="GTiff", height=data.shape[1], width=data.shape[2],
        count=data.shape[0], dtype=data.dtype, crs=crs, transform=transform, nodata=nodata,
    ) as dst:
        dst.write(data)
    return path


def _image(tmp_path: Path, size: int = 64, res: float = 10.0) -> Path:
    rng = np.random.default_rng(0)
    data = rng.uniform(1, 2, (3, size, size)).astype("float32")
    return _write(tmp_path / "img.tif", data, from_origin(WEST, NORTH, res, res), nodata=-9999)


def test_same_grid_reads_labels_unchanged(tmp_path):
    img = _image(tmp_path)
    labels = np.arange(64 * 64, dtype="int32").reshape(64, 64)
    lbl = _write(tmp_path / "lbl.tif", labels, from_origin(WEST, NORTH, 10, 10))

    image, out_labels, meta, _ = _read_aligned_image_and_labels(img, lbl, Resampling.nearest)

    assert image.shape == (3, 64, 64)
    np.testing.assert_array_equal(out_labels, labels)
    assert meta["transform"] == from_origin(WEST, NORTH, 10, 10)


def test_coarser_partial_labels_crop_image_and_upsample(tmp_path):
    img = _image(tmp_path)
    # 20 m labels covering image rows/cols 16..48, each label pixel unique
    labels = np.arange(16 * 16, dtype="uint8").reshape(16, 16)
    lbl = _write(tmp_path / "lbl.tif", labels, from_origin(WEST + 160, NORTH - 160, 20, 20))

    image, out_labels, meta, _ = _read_aligned_image_and_labels(img, lbl, Resampling.nearest)

    assert image.shape == (3, 32, 32)
    assert out_labels.shape == (32, 32)
    assert meta["transform"] == from_origin(WEST + 160, NORTH - 160, 10, 10)
    np.testing.assert_array_equal(out_labels, np.kron(labels, np.ones((2, 2), dtype="uint8")))

    with rst.open(img) as src:
        np.testing.assert_array_equal(image, src.read()[:, 16:48, 16:48])


def test_different_crs_labels_are_reprojected(tmp_path):
    img = _image(tmp_path, size=128)
    # Constant-valued labels in WGS84 covering the middle of the image
    with rst.open(img) as src:
        l, b, r, t = transform_bounds(UTM, "EPSG:4326", *src.bounds)
    dx, dy = (r - l) / 4, (t - b) / 4
    labels = np.full((50, 50), 3, dtype="uint8")
    lbl = _write(
        tmp_path / "lbl.tif", labels,
        from_origin(l + dx, t - dy, 2 * dx / 50, 2 * dy / 50), crs="EPSG:4326", nodata=0,
    )

    image, out_labels, meta, _ = _read_aligned_image_and_labels(img, lbl, Resampling.nearest)

    assert meta["crs"] == rst.CRS.from_string(UTM)
    assert out_labels.shape == image.shape[1:]
    assert 0 < out_labels.shape[0] < 128 and 0 < out_labels.shape[1] < 128
    # Reprojected bounds are not axis-aligned in UTM, so allow a thin nodata rim
    assert set(np.unique(out_labels)) <= {0, 3}
    assert (out_labels == 3).mean() > 0.95


def test_resampling_accepts_name(tmp_path):
    img = _image(tmp_path)
    labels = np.array([[0.0, 1.0], [0.0, 1.0]], dtype="float32")
    lbl = _write(tmp_path / "lbl.tif", labels, from_origin(WEST, NORTH, 320, 320))

    out_dir = tmp_path / "patches"
    generate_patches_from_files(
        img, lbl, out_dir, patch_size=64, val_fraction=0.5, n_random_offsets=0,
        empty_img_threshold=None, show_progress=False, resampling="bilinear",
    )

    patch = next((out_dir / "val").glob("*_label.tiff"), None) or next(
        (out_dir / "train").glob("*_label.tiff")
    )
    with rst.open(patch) as src:
        values = np.unique(src.read(1))
    # Bilinear interpolation produces values strictly between the two classes
    assert ((values > 0) & (values < 1)).any()


def test_patches_lie_inside_label_extent(tmp_path):
    img = _image(tmp_path, size=128)
    labels = np.ones((40, 40), dtype="uint8")
    lbl_tf = from_origin(WEST + 300, NORTH - 300, 20, 20)
    lbl = _write(tmp_path / "lbl.tif", labels, lbl_tf, nodata=255)

    out_dir = tmp_path / "patches"
    generate_patches_from_files(
        img, lbl, out_dir, patch_size=16, n_random_offsets=1,
        empty_img_threshold=None, show_progress=False,
    )

    with rst.open(lbl) as src:
        lb = src.bounds
    patch_paths = list(out_dir.rglob("*.tiff"))
    assert patch_paths
    for p in patch_paths:
        with rst.open(p) as src:
            b = src.bounds
            assert src.crs == rst.CRS.from_string(UTM)
            assert src.res == (10.0, 10.0)
            # Label patches carry the label nodata, not the image's (-9999 won't fit uint8)
            assert src.nodata == (255 if p.stem.endswith("_label") else -9999)
        assert lb.left <= b.left and b.right <= lb.right
        assert lb.bottom <= b.bottom and b.top <= lb.top


def test_no_overlap_raises(tmp_path):
    img = _image(tmp_path)
    lbl = _write(
        tmp_path / "lbl.tif", np.ones((10, 10), dtype="uint8"),
        from_origin(WEST + 10_000, NORTH, 10, 10),
    )
    with pytest.raises(ValueError, match="do not overlap"):
        _read_aligned_image_and_labels(img, lbl, Resampling.nearest)
