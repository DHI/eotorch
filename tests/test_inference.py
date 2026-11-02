"""Tests for patch-wise inference on TIF files."""

import random
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
)

TEST_IMAGE = Path(__file__).parent / "data" / "image_0.tif"


def _first_band(batch: np.ndarray) -> np.ndarray:
    """A "model" whose prediction is the first band of its input."""
    return batch[:, 0].astype("float32")


def _read(path) -> np.ndarray:
    with rst.open(path) as src:
        return src.read(1)


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


@pytest.mark.parametrize("blend", [False, True])
def test_predictions_are_stitched_correctly(tmp_path, first_band, blend):
    out = predict_on_tif_generic(
        TEST_IMAGE,
        _first_band,
        patch_size=32,
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
        save_probabilities=True,
        probability_band_names=["dark", "bright"],
    )
    np.testing.assert_array_equal(_read(out), first_band > 100)
    with rst.open(tmp_path / "pred_probabilities.tif") as src:
        assert src.count == 2 and src.dtypes[0] == "float32"
        assert src.descriptions == ("dark", "bright")
        probs = src.read()
    np.testing.assert_allclose(probs[1], first_band > 100, atol=1e-6)
    np.testing.assert_allclose(probs.sum(axis=0), 1, rtol=1e-6)


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
