from typing import Any

import torch
import torch.nn as nn


class Normalize(nn.Module):
    """Normalize patch image bands using precomputed per-band mean/std.

    Patches produced by :func:`eotorch.processing.patch.generate_train_val_patches`
    are scaled to ``[0, 1]``. This rescales them to zero mean / unit variance
    (roughly ``[-1, 1]`` for a dataset without heavy outliers) using the
    dataset's own per-band statistics, as expected by DINOv3. Get ``mean``
    and ``std`` from :func:`eotorch.processing.patch.compute_patch_stats`.

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
        if image is None and len(args) >= 1:
            image = args[0]
        if mask is None and len(args) >= 2:
            mask = args[1]
        if image is None:
            raise ValueError('Normalize expects an image.')

        image = torch.as_tensor(image, dtype=torch.float32)
        image = (image - self.mean) / self.std

        return {'image': image, 'mask': mask}
