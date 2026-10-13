import numpy as np
import pytest
import rasterio as rst
import torch
from rasterio.transform import from_origin

from eotorch.processing.normalize import band_scaling
from eotorch.processing.patch import compute_patch_stats
from eotorch.transforms import BandScaling, Compose, Normalize, SelectBands

LIMITS = {1: (10.0, 200.0), 2: (0.0, 100.0), 3: (50.0, 250.0)}


def _random_image(shape, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, shape).astype(np.float32)
    img[..., :5, :5] = 0  # nodata corner
    return img


def test_band_scaling_matches_band_scaling_function():
    img = _random_image((3, 32, 32))
    expected = band_scaling(img, LIMITS)

    out = BandScaling(LIMITS)(image=img)['image']

    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6)


def test_band_scaling_handles_batches():
    batch = np.stack([_random_image((3, 16, 16), seed=s) for s in range(4)])
    scaling = BandScaling(LIMITS)

    out = scaling(image=batch)['image']

    assert out.shape == batch.shape
    for i in range(len(batch)):
        np.testing.assert_allclose(out[i].numpy(), band_scaling(batch[i], LIMITS), rtol=1e-6)


def test_band_scaling_accepts_sequence_and_custom_nodata():
    img = np.full((3, 4, 4), 255.0, dtype=np.float32)
    img[:, 0, 0] = 7

    out = BandScaling(list(LIMITS.values()), nodata=7)(image=img)['image']

    assert (out[:, 0, 0] == 0).all()
    assert (out[:, 1:, 1:] == 1).all()


def test_band_scaling_passes_mask_through():
    mask = np.ones((8, 8), dtype=np.uint8)

    out = BandScaling(LIMITS)(image=_random_image((3, 8, 8)), mask=mask)

    assert out['mask'] is mask


def test_select_bands():
    img = np.arange(5)[:, None, None] * np.ones((5, 2, 2))
    batch = np.stack([img, img])

    out = SelectBands([4, 1])(image=batch)['image']

    assert out.shape == (2, 2, 2, 2)
    assert (out[:, 0] == 3).all() and (out[:, 1] == 0).all()


def test_compose_chains_transforms_and_keeps_extra_keys():
    img = _random_image((3, 16, 16))
    mean, std = [0.5, 0.4, 0.3], [0.2, 0.25, 0.3]
    distance = np.zeros((16, 16))

    out = Compose(BandScaling(LIMITS), Normalize(mean, std))(image=img, mask=None, distance=distance)

    expected = (band_scaling(img, LIMITS) - np.array(mean)[:, None, None]) / np.array(std)[:, None, None]
    np.testing.assert_allclose(out['image'].numpy(), expected, rtol=1e-5)
    assert out['distance'] is distance


def test_compute_patch_stats_with_transform(tmp_path):
    images = [_random_image((3, 16, 16), seed=s) for s in range(3)]
    for i, img in enumerate(images):
        with rst.open(
            tmp_path / f'p{i}_image.tiff', 'w', driver='GTiff', width=16, height=16, count=3,
            dtype='float32', transform=from_origin(0, 16, 1, 1),
        ) as dst:
            dst.write(img)

    mean, std = compute_patch_stats(tmp_path, show_progress=False, transform=BandScaling(LIMITS))

    scaled = np.stack([band_scaling(img, LIMITS) for img in images])
    raw = np.stack(images)
    for b in range(3):
        valid = scaled[:, b][raw[:, b] != 0]
        assert mean[b] == pytest.approx(valid.mean(), rel=1e-5)
        assert std[b] == pytest.approx(valid.std(), rel=1e-4)


def test_band_scaling_output_is_float_tensor():
    out = BandScaling(LIMITS)(image=np.ones((3, 2, 2), dtype=np.uint8))['image']

    assert isinstance(out, torch.Tensor) and out.dtype == torch.float32
