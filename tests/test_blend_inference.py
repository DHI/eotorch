import random

import numpy as np
import pytest
import rasterio as rst
from rasterio.transform import from_origin

from eotorch.inference import inference_utils as iu
from eotorch.inference.inference import predict_on_tif_generic

PATCH_SIZE = 64
OVERLAP = 2
HEIGHT, WIDTH = 300, 410  # not multiples of the stride, so the last row/col get clamped


@pytest.fixture
def tif_path(tmp_path):
    path = tmp_path / "input.tif"
    rng = np.random.default_rng(0)
    data = rng.uniform(1, 2, size=(1, HEIGHT, WIDTH)).astype(np.float32)
    with rst.open(
        path,
        "w",
        driver="GTiff",
        height=HEIGHT,
        width=WIDTH,
        count=1,
        dtype="float32",
        nodata=0,
        crs="EPSG:32632",
        transform=from_origin(500000, 6000000, 10, 10),
    ) as dst:
        dst.write(data)
    return path


def _predict(batch: np.ndarray) -> np.ndarray:
    """Two-band float prediction with a per-patch offset, so overlapping patches disagree."""
    patch_mean = batch.mean(axis=(1, 2, 3), keepdims=True)
    return np.concatenate([batch + patch_mean, 2 * batch - patch_mean], axis=1).astype(np.float32)


def _batches(tif_path, batch_size=4):
    return list(iu.patch_generator(tif_path, PATCH_SIZE, OVERLAP, batch_size))


def _reference_blend(batches) -> np.ndarray:
    """Full-raster weighted overlap-add, i.e. the original in-memory algorithm."""
    weight_2d = iu.blend_weight_2d(PATCH_SIZE)
    accum = np.zeros((2, HEIGHT, WIDTH), dtype=np.float32)
    weight_accum = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    for batch, windows in batches:
        pred = _predict(batch)
        for p, w in zip(pred, windows):
            r, c = int(w.row_off), int(w.col_off)
            accum[:, r:r + PATCH_SIZE, c:c + PATCH_SIZE] += p * weight_2d
            weight_accum[r:r + PATCH_SIZE, c:c + PATCH_SIZE] += weight_2d
    return accum / weight_accum[None]


def _run(tif_path, out_path, generator, blend=True):
    predict_on_tif_generic(
        tif_path,
        _predict,
        data_and_window_generator=generator,
        patch_size=PATCH_SIZE,
        overlap=OVERLAP,
        out_file_path=out_path,
        progress_bar=False,
        blend=blend,
    )
    with rst.open(out_path) as src:
        return src.read()


def test_streamed_blend_matches_full_raster_blend(tif_path, tmp_path):
    batches = _batches(tif_path)
    out = _run(tif_path, tmp_path / "out.tif", iter(batches))
    assert out.shape == (2, HEIGHT, WIDTH)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, _reference_blend(batches), rtol=1e-5, atol=1e-5)


def test_streamed_blend_bottom_up_order(tif_path, tmp_path):
    batches = _batches(tif_path)
    top_down = _run(tif_path, tmp_path / "top_down.tif", iter(batches))
    bottom_up = [(b[::-1], w[::-1]) for b, w in reversed(batches)]
    out = _run(tif_path, tmp_path / "bottom_up.tif", iter(bottom_up))
    np.testing.assert_allclose(out, top_down, rtol=1e-5, atol=1e-5)


def test_streamed_blend_rejects_unordered_windows(tif_path, tmp_path):
    batches = _batches(tif_path, batch_size=1)
    random.Random(0).shuffle(batches)
    with pytest.raises(RuntimeError, match="monotonic row order"):
        _run(tif_path, tmp_path / "shuffled.tif", iter(batches))


def test_hard_crop_without_blend_still_runs(tif_path, tmp_path):
    out = _run(tif_path, tmp_path / "hard.tif", None, blend=False)
    assert out.shape == (2, HEIGHT, WIDTH)
    assert np.isfinite(out).all()
