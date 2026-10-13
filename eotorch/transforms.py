from typing import Any, Callable, Sequence

import torch
import torch.nn as nn


def _unpack_image_and_mask(image, mask, args):
    """Accept both ``transform(image=..., mask=...)`` and positional calls."""
    if image is None and len(args) >= 1:
        image = args[0]
    if mask is None and len(args) >= 2:
        mask = args[1]
    return image, mask


class BandScaling(nn.Module):
    """Scale image bands to ``[0, 1]`` using fixed per-band limits.

    The on-the-fly equivalent of :func:`eotorch.processing.normalize.band_scaling`
    with a dict of limits: each band is mapped linearly from ``(low, high)`` to
    ``(0, 1)`` and clipped to ``[1e-5, 1]``, so valid pixels never become exactly 0.
    Pixels equal to ``nodata`` are set to 0.

    Works on a single ``(C, H, W)`` sample, as passed by
    ``eotorch.data.PatchDataModule``, and on a ``(B, C, H, W)`` batch, as passed by
    :func:`eotorch.inference.inference.predict_on_tif_generic` via ``transforms``.

    Parameters
    ----------
    limits : dict[int, tuple[float, float]] | Sequence[tuple[float, float]]
        Per-band ``(low, high)`` values, either as a dict keyed by 1-based band
        position (as in ``band_scaling``) or as a sequence in band order.
    nodata : float | None, optional
        Input value marking missing data, which is set to 0 in the output.
        If None, no pixels are treated as nodata. Defaults to 0.
    """

    def __init__(
        self,
        limits: dict[int, tuple[float, float]] | Sequence[tuple[float, float]],
        nodata: float | None = 0.0,
    ):
        super().__init__()
        if isinstance(limits, dict):
            limits = [limits[k] for k in sorted(limits)]
        low, high = zip(*limits)
        self.register_buffer('low', torch.as_tensor(low, dtype=torch.float32).view(-1, 1, 1))
        self.register_buffer('high', torch.as_tensor(high, dtype=torch.float32).view(-1, 1, 1))
        self.nodata = nodata

    def forward(self, image=None, mask=None, *args, **kwargs) -> dict[str, Any]:
        image, mask = _unpack_image_and_mask(image, mask, args)
        if image is None:
            raise ValueError('BandScaling expects an image.')

        image = torch.as_tensor(image, dtype=torch.float32)
        scaled = ((image - self.low) / (self.high - self.low)).clamp(1e-5, 1)
        if self.nodata is not None:
            scaled = torch.where(image == self.nodata, torch.zeros_like(scaled), scaled)

        return {'image': scaled, 'mask': mask}


class SelectBands(nn.Module):
    """Select a subset of image bands.

    Useful at inference when the input raster has more bands than the model was
    trained on, e.g. as the first entry of ``transforms`` in
    :func:`eotorch.inference.inference.predict_on_tif_generic`.

    Parameters
    ----------
    indexes : Sequence[int]
        1-based band indexes to keep, in output order.
    """

    def __init__(self, indexes: Sequence[int]):
        super().__init__()
        self.indexes = [i - 1 for i in indexes]

    def forward(self, image=None, mask=None, *args, **kwargs) -> dict[str, Any]:
        image, mask = _unpack_image_and_mask(image, mask, args)
        if image is None:
            raise ValueError('SelectBands expects an image.')

        image = torch.as_tensor(image)
        return {'image': image[..., self.indexes, :, :], 'mask': mask}


class Compose(nn.Module):
    """Apply several image/mask transforms in order.

    Each transform is called as ``transform(image=..., mask=..., **extra)`` and must
    return a dict; its entries update the inputs to the next transform. Extra keyword
    arguments (e.g. ``distance`` from ``eotorch.data.PatchDataModule``) are passed
    through to every transform.

    Parameters
    ----------
    *transforms : Callable
        Transforms to apply, in order.
    """

    def __init__(self, *transforms: Callable[..., dict[str, Any]]):
        super().__init__()
        self.transforms = list(transforms)

    def forward(self, image=None, mask=None, *args, **kwargs) -> dict[str, Any]:
        image, mask = _unpack_image_and_mask(image, mask, args)
        sample = {'image': image, 'mask': mask, **kwargs}
        for transform in self.transforms:
            sample.update(transform(**sample))
        return sample


class Normalize(nn.Module):
    """Normalize patch image bands using precomputed per-band mean/std.

    Rescales image bands to zero mean / unit variance (roughly ``[-1, 1]`` for a
    dataset without heavy outliers) using the dataset's own per-band statistics,
    as expected by DINOv3. Typically applied after :class:`BandScaling` (see
    :class:`Compose`). Get ``mean`` and ``std`` from
    :func:`eotorch.processing.patch.compute_patch_stats`, passing the same
    preceding transforms as its ``transform`` argument.

    Compatible with ``eotorch.data.PatchDataModule``'s ``transform`` hook,
    which calls ``transform(image=img, mask=label)`` per-sample with ``img``
    a ``(C, H, W)`` array and ``label`` a ``(H, W)`` array, and expects a
    dict with ``'image'``/``'mask'`` keys back.

    Parameters
    ----------
    mean : Sequence[float]
        Per-band mean, one value per channel.
    std : Sequence[float]
        Per-band standard deviation, one value per channel.
    """

    def __init__(self, mean: Any, std: Any):
        super().__init__()
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32).view(-1, 1, 1))
        self.register_buffer('std', torch.as_tensor(std, dtype=torch.float32).view(-1, 1, 1))

    def forward(self, image=None, mask=None, *args, **kwargs) -> dict[str, Any]:
        image, mask = _unpack_image_and_mask(image, mask, args)
        if image is None:
            raise ValueError('Normalize expects an image.')

        image = torch.as_tensor(image, dtype=torch.float32)
        image = (image - self.mean) / self.std

        return {'image': image, 'mask': mask}
