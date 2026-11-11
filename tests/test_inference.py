"""Tests for patch-wise inference on TIF files."""

import math
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
import rasterio as rst
from rasterio.windows import Window

from eotorch.inference.inference import predict_on_tif_generic
from eotorch.inference.inference_utils import (
    PatchBlender,
    blend_weights,
    patch_generator,
    patches_along_axis,
)

TEST_IMAGE = Path(__file__).parent / "data" / "image_0.tif"


def _first_band(batch: np.ndarray) -> np.ndarray:
    """A "model" whose prediction is the first band of its input."""
    return batch[:, 0].astype("float32")


def _distance_to_patch_edge(batch: np.ndarray) -> np.ndarray:
    """A "model" whose prediction is the distance of each pixel to the patch edge."""
    patch_size = batch.shape[-1]
    distance = np.minimum(np.arange(patch_size), np.arange(patch_size)[::-1])
    distance_2d = np.minimum.outer(distance, distance).astype("float32")
    return np.broadcast_to(distance_2d, (len(batch), patch_size, patch_size))


def _read(path) -> np.ndarray:
    with rst.open(path) as src:
        return src.read(1)


def _write_image(path: Path, height: int, width: int) -> Path:
    data = np.arange(height * width, dtype="float32").reshape(1, height, width)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32632",
        "transform": rst.transform.from_origin(0, height, 1, 1),
    }
    with rst.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


@pytest.mark.parametrize(
    "height, width, patch_size, overlap, batch_size",
    [
        (128, 128, 64, 2, 4),  # patches end exactly at the image edge
        (100, 100, 64, 2, 4),  # the last row and column are moved back
        (128, 128, 64, 1, 3),  # no overlap, last batch only partly filled
        (128, 128, 64, 4, 8),
        (250, 180, 64, 2, 5),
        (20, 100, 64, 2, 4),  # smaller than a patch, padded
    ],
)
def test_patch_generator_covers_image_once(
    tmp_path, height, width, patch_size, overlap, batch_size
):
    image = _write_image(tmp_path / "image.tif", height, width)
    with pytest.warns(UserWarning) if height < patch_size else nullcontext():
        batches = list(patch_generator(image, patch_size, overlap, batch_size))

    windows = [
        (w.row_off, w.col_off) for _, batch_windows in batches for w in batch_windows
    ]
    assert len(windows) == len(set(windows))
    covered = np.zeros((height, width), dtype=bool)
    for row, col in windows:
        covered[row : row + patch_size, col : col + patch_size] = True
    assert covered.all()

    stride = patch_size // overlap
    n_patches = patches_along_axis(height, patch_size, stride) * patches_along_axis(
        width, patch_size, stride
    )
    assert len(windows) == n_patches
    assert len(batches) == math.ceil(n_patches / batch_size)
    assert all(batch.shape[1:] == (1, patch_size, patch_size) for batch, _ in batches)


@pytest.mark.parametrize("overlap", [2, 4])
def test_only_the_center_of_patches_is_kept(tmp_path, overlap):
    out = predict_on_tif_generic(
        TEST_IMAGE,
        _distance_to_patch_edge,
        patch_size=32,
        overlap=overlap,
        dtype="float32",
        nodata=None,
        out_file_path=tmp_path / "pred.tif",
        progress_bar=False,
    )
    # Half of the overlap is cut from each side of a patch, except at the image border
    buffer = (32 - 32 // overlap) // 2
    assert _read(out)[buffer:-buffer, buffer:-buffer].min() >= buffer


@pytest.mark.parametrize("blend", [False, True])
def test_images_smaller_than_a_patch_are_padded(tmp_path, blend):
    image = _write_image(tmp_path / "small.tif", height=20, width=30)
    with pytest.warns(UserWarning, match="smaller than the patch size"):
        out = predict_on_tif_generic(
            image,
            _first_band,
            patch_size=32,
            dtype="float32",
            nodata=None,
            out_file_path=tmp_path / "pred.tif",
            progress_bar=False,
            blend=blend,
        )
    np.testing.assert_allclose(_read(out), _read(image), rtol=1e-6)


@pytest.fixture
def first_band():
    with rst.open(TEST_IMAGE) as src:
        return src.read(1).astype("float32")


def test_blend_weights():
    weights = blend_weights(16)
    assert weights.shape == (16, 16) and (weights > 0).all()
    assert weights[8, 8] == weights.max() and weights[0, 0] == weights.min()
    np.testing.assert_allclose(weights, weights.T)


def test_patch_blender_averages_overlapping_patches(tmp_path):
    blender = PatchBlender(tmp_path, height=8, width=12, patch_size=8)
    blender.add(np.full((8, 8), 1.0), Window(0, 0, 8, 8))
    blender.add(np.full((8, 8), 3.0), Window(4, 0, 8, 8))

    ((_, values, covered),) = blender.chunks()
    assert covered.all()
    np.testing.assert_allclose(values[0, :, :4], 1, rtol=1e-6)
    np.testing.assert_allclose(values[0, :, -4:], 3, rtol=1e-6)
    # The overlap is weighted towards the center of each patch
    assert 1 < values[0, 4, 4] < values[0, 4, 7] < 3


@pytest.mark.parametrize("overlap", [1, 2, 4])
@pytest.mark.parametrize("blend", [False, True])
def test_predictions_are_stitched_correctly(tmp_path, first_band, blend, overlap):
    out = predict_on_tif_generic(
        TEST_IMAGE,
        _first_band,
        patch_size=32,
        overlap=overlap,
        dtype="float32",
        out_file_path=tmp_path / "pred.tif",
        progress_bar=False,
        blend=blend,
    )
    np.testing.assert_allclose(_read(out), first_band, rtol=1e-6)


def test_blending_does_not_depend_on_patch_order(tmp_path, first_band):
    batches = list(patch_generator(TEST_IMAGE, patch_size=32, overlap=2, batch_size=1))
    random.Random(0).shuffle(batches)

    out = predict_on_tif_generic(
        TEST_IMAGE,
        _first_band,
        data_and_window_generator=iter(batches),
        patch_size=32,
        dtype="float32",
        out_file_path=tmp_path / "pred.tif",
        progress_bar=False,
        blend=True,
    )
    np.testing.assert_allclose(_read(out), first_band, rtol=1e-6)


def test_blended_probabilities_are_postprocessed(tmp_path, first_band):
    def probabilities(batch):
        bright = (batch[:, 0] > 100).astype("float32")
        return np.stack([1 - bright, bright], axis=1)

    out = predict_on_tif_generic(
        TEST_IMAGE,
        probabilities,
        patch_size=32,
        postprocess=lambda probs: probs.argmax(axis=-3).astype("uint8"),
        out_file_path=tmp_path / "pred.tif",
        progress_bar=False,
        blend=True,
    )
    np.testing.assert_array_equal(_read(out), first_band > 100)


@pytest.mark.parametrize("blend", [False, True])
def test_probabilities_are_saved_next_to_classes(tmp_path, first_band, blend):
    def probabilities(batch):
        bright = (batch[:, 0] > 100).astype("float32")
        return np.stack([1 - bright, bright], axis=1)

    out = predict_on_tif_generic(
        TEST_IMAGE,
        probabilities,
        patch_size=32,
        postprocess=lambda probs: probs.argmax(axis=-3).astype("uint8"),
        out_file_path=tmp_path / "pred.tif",
        progress_bar=False,
        blend=blend,
        class_mapping={0: "dark", 1: "bright"},
        save_probabilities=True,
    )
    np.testing.assert_array_equal(_read(out), first_band > 100)
    with rst.open(tmp_path / "pred_probabilities.tif") as src:
        assert src.count == 2 and src.dtypes[0] == "float32"
        assert src.descriptions == ("dark", "bright")
        probs = src.read()
    np.testing.assert_allclose(probs[1], first_band > 100, atol=1e-6)
    np.testing.assert_allclose(probs.sum(axis=0), 1, rtol=1e-6)


def test_probability_bands_are_unnamed_without_one_class_per_band(tmp_path):
    # A binary model predicts one probability band, but the mapping has two classes
    predict_on_tif_generic(
        TEST_IMAGE,
        lambda batch: (batch[:, :1] > 100).astype("float32"),
        patch_size=32,
        postprocess=lambda probs: (probs[..., 0, :, :] > 0.5).astype("uint8"),
        class_mapping={0: "dark", 1: "bright"},
        out_file_path=tmp_path / "pred.tif",
        progress_bar=False,
        save_probabilities=True,
    )
    with rst.open(tmp_path / "pred_probabilities.tif") as src:
        assert src.count == 1 and src.descriptions == (None,)


def test_probabilities_are_not_saved_by_default(tmp_path):
    predict_on_tif_generic(
        TEST_IMAGE,
        _first_band,
        patch_size=32,
        dtype="float32",
        out_file_path=tmp_path / "pred.tif",
        progress_bar=False,
    )
    assert not (tmp_path / "pred_probabilities.tif").exists()
