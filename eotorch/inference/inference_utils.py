from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import rasterio as rst
from affine import Affine
from rasterio.windows import Window, from_bounds


def prediction_to_numpy(pred: Any) -> np.ndarray:
    """
    Convert the prediction to a numpy array.

    Parameters
    ----------
        pred: prediction

    Returns
    ----------
        numpy array
    """
    import torch

    if isinstance(pred, np.ndarray):
        return pred
    elif isinstance(pred, torch.Tensor):
        # check if the tensor is on GPU and / or requires gradients
        if pred.is_cuda:
            pred = pred.cpu()
        if pred.requires_grad:
            pred = pred.detach()
        return pred.numpy()

    else:
        raise ValueError(
            "Invalid prediction type. Must be numpy array or torch tensor."
        )


def crop_np_to_window(arr: np.ndarray, w_buffered: Window, w_unbuffered: Window):
    left = int(round(w_unbuffered.col_off - w_buffered.col_off))
    top = int(round(w_unbuffered.row_off - w_buffered.row_off))
    right = left + int(round(w_unbuffered.width))
    bottom = top + int(round(w_unbuffered.height))

    if arr.ndim == 3:
        return arr[:, top:bottom, left:right]
    return arr[top:bottom, left:right]


def blend_weight_2d(patch_size: int, min_weight: float = 1e-3) -> np.ndarray:
    """2-D Hann-window taper for weighted overlap-add tile blending.

    Peaks at the patch center and tapers toward (but never exactly to) zero
    at its edges, so overlapping patches' predictions can be combined as a
    weighted average that favors each patch's more reliable center over its
    edges -- which are biased by that patch's own zero-padded convolution
    boundary, since each patch is run through the model independently. A hard
    crop-and-tile instead keeps a fixed "trusted" region per patch and abuts
    them edge-to-edge; wherever two patches' independently-computed kept
    regions don't quite agree, that shows up as a visible seam. Blending
    smooths that disagreement out instead.

    Parameters
    ----------
    patch_size : int
        Side length of the (square) patch.
    min_weight : float, optional
        Floor applied to the raw Hann window (which reaches exactly 0 at its
        endpoints), so a pixel touched by only one patch -- e.g. right at the
        image border, where there's no neighboring patch to compensate -- still
        gets a strictly positive weight and doesn't produce a division by zero
        when normalizing the accumulated weighted sum.

    Returns
    -------
    np.ndarray
        float32 array of shape (patch_size, patch_size).
    """
    taper = np.clip(np.hanning(patch_size), min_weight, None)
    return np.outer(taper, taper).astype(np.float32)


class BlendAccumulator:
    """Streaming weighted overlap-add of patch predictions into an open raster.

    Instead of holding a full-raster accumulator, the output is split into
    horizontal row chunks of height `chunk_height` that are created lazily when
    a patch first touches them. Once a patch no longer overlaps a buffered chunk,
    that chunk is normalized and written to `dest`, then freed. This is exact as
    long as patches arrive in monotonic row order (top-down, as with
    `patch_generator`, or bottom-up, as with torchgeo's GridGeoSampler); a patch
    touching an already-written chunk raises a RuntimeError instead of silently
    producing a wrong result. Memory scales with raster width x
    (patch_size + 2 * chunk_height), not with the full raster size.

    Parameters
    ----------
    dest : rasterio DatasetWriter
        Open output raster to write normalized chunks to.
    num_bands : int
        Number of output bands.
    patch_size : int
        Side length of the (square) predicted patches.
    chunk_height : int
        Height in rows of each buffered chunk.
    dtype : str
        Output data type that normalized chunks are cast to.
    nodata_value : int | float
        Value written where no patch contributed any weight.
    """

    def __init__(
        self,
        dest,
        num_bands: int,
        patch_size: int,
        chunk_height: int,
        dtype: str,
        nodata_value,
    ):
        self.dest = dest
        self.num_bands = num_bands
        self.patch_size = patch_size
        self.chunk_height = max(1, int(chunk_height))
        self.dtype = dtype
        self.nodata_value = nodata_value
        self.height, self.width = dest.height, dest.width
        self.weight_2d = blend_weight_2d(patch_size)
        self._chunks: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._flushed: set[int] = set()

    def _chunk_rows(self, idx: int) -> tuple[int, int]:
        start = idx * self.chunk_height
        return start, min(start + self.chunk_height, self.height)

    def add(self, pred: np.ndarray, window: Window):
        """Add one patch prediction (bands x patch x patch, or patch x patch)."""
        patch_pred = pred if pred.ndim == 3 else pred[None]
        row_off, col_off = int(window.row_off), int(window.col_off)
        row_end = min(row_off + self.patch_size, self.height)
        first, last = row_off // self.chunk_height, (row_end - 1) // self.chunk_height
        touched = range(first, last + 1)
        if any(idx in self._flushed for idx in touched):
            raise RuntimeError(
                "blend=True requires the data_and_window_generator to yield windows "
                "in monotonic row order (top-down or bottom-up), but a window at "
                f"row offset {row_off} touches rows that have already been written. "
                "Use blend=False for generators with a different ordering."
            )
        weighted = patch_pred * self.weight_2d
        col_end = col_off + self.patch_size
        for idx in touched:
            if idx not in self._chunks:
                start, stop = self._chunk_rows(idx)
                self._chunks[idx] = (
                    np.zeros((self.num_bands, stop - start, self.width), dtype=np.float32),
                    np.zeros((stop - start, self.width), dtype=np.float32),
                )
            accum, weight_accum = self._chunks[idx]
            start, stop = self._chunk_rows(idx)
            lo, hi = max(row_off, start), min(row_end, stop)
            accum[:, lo - start:hi - start, col_off:col_end] += weighted[:, lo - row_off:hi - row_off]
            weight_accum[lo - start:hi - start, col_off:col_end] += self.weight_2d[lo - row_off:hi - row_off]

    def _flush(self, idx: int):
        accum, weight_accum = self._chunks.pop(idx)
        blended = np.divide(
            accum,
            weight_accum[None],
            out=np.full_like(accum, self.nodata_value, dtype=np.float32),
            where=weight_accum[None] > 0,
        ).astype(self.dtype)
        start, stop = self._chunk_rows(idx)
        self.dest.write(blended, window=Window(0, start, self.width, stop - start))
        self._flushed.add(idx)

    def flush_except(self, row_lo: int, row_hi: int):
        """Write out every buffered chunk not overlapping rows [row_lo, row_hi)."""
        for idx in list(self._chunks):
            start, stop = self._chunk_rows(idx)
            if stop <= row_lo or start >= row_hi:
                self._flush(idx)

    def flush_all(self):
        """Write out every remaining buffered chunk."""
        for idx in list(self._chunks):
            self._flush(idx)


def buffered_to_unbuffered(
    window: Window,
    buffer: int,
    # patch_size: int,
    img_height: int,
    img_width: int,
):
    """
    Convert a buffered window to an unbuffered window.

    Parameters
    ----------
        window: buffered window
        buffer: buffer size
        img_height: height of the original image
        img_width: width of the original image

    Returns
    ----------
        unbuffered window
    """

    top, left = window.row_off, window.col_off
    window_height, window_width = window.height, window.width
    if top > 0:  # This is not the first row, so so we want the buffer on top
        window_height -= buffer
        top += buffer
    elif top < 0:
        window_height = max(window_height + top, 0)
        top = 0

    if left > 0:  # This is not the first column, so we want the buffer on the left
        window_width -= buffer
        left += buffer
    elif left < 0:
        window_width = max(window_width + left, 0)
        left = 0
    bottom = top + window_height
    right = left + window_width
    if (
        bottom < img_height
    ):  # This is not the last row, so we want the buffer on the bottom
        window_height -= buffer
    elif bottom > img_height:
        # we cannot have a window that is larger than the image
        # so we need to adjust the height
        window_height = img_height - top
    if (
        right < img_width
    ):  # This is not the last column, so we want the buffer on the right
        window_width -= buffer
    elif right > img_width:
        # we cannot have a window that is larger than the image
        # so we need to adjust the width
        window_width = img_width - left

    return Window(left, top, window_width, window_height)


def buffered_to_unbuffered_new(
    window: Window, buffer: int, img_height: int, img_width: int, transform: Affine
) -> Tuple[Window, Affine]:
    """
    Convert a buffered window to an unbuffered window by removing the buffer
    from the window's boundaries. Ensures the unbuffered window stays within the
    image dimensions and updates the transform accordingly.

    Parameters
    ----------
        window: rasterio.windows.Window, buffered window
        buffer: int, buffer size
        img_height: int, height of the original image
        img_width: int, width of the original image
        transform: Affine, original transform of the image

    Returns
    ----------
        Tuple[rasterio.windows.Window, Affine], unbuffered window and updated transform
    """
    top, left = window.row_off, window.col_off
    window_height, window_width = window.height, window.width

    if top > 0:  # Remove buffer from top if not the first row
        top += buffer
        window_height -= buffer
    if left > 0:  # Remove buffer from left if not the first column
        left += buffer
        window_width -= buffer
    if (
        top + window_height < img_height
    ):  # Remove buffer from bottom if not the last row
        window_height = min(window_height, img_height - top)
    if (
        left + window_width < img_width
    ):  # Remove buffer from right if not the last column
        window_width = min(window_width, img_width - left)

    # Ensure the window height and width do not become negative
    window_height = max(0, window_height)
    window_width = max(0, window_width)

    # Update the transform for the new window
    new_transform = transform * Affine.translation(
        left - window.col_off, top - window.row_off
    )

    window = Window(left, top, window_width, window_height)

    return window, new_transform


def eotorch_patch_generator(
    tif_file_path: str | Path,
    checkpoint_path: str | Path,
    batch_size: int = 8,
):
    """
    Similar to the patch_generator function, but uses the exact same way of loading the image data
    as we do during training with eotorch.

    Parameters
    ----------
        tif_file_path: path to the tif file to predict on
        checkpoint_path: path to a SemanticSegmentationTask checkpoint; its
            datamodule_hyper_parameters determine dataset args (bands, CRS, res,
            class_mapping, etc.) and patch_size
        batch_size: number of patches per yielded batch

    Yields
    ----------
        batch: image tensor batch, loaded and transformed exactly as during training
        windows: list of rasterio windows, indicating the location of each patch
                in the original tif file
    """
    import torch

    from eotorch.data import SegmentationDataModule
    from eotorch.data.geodatasets import get_segmentation_dataset
    from eotorch.data.tasks import SemanticSegmentationTask

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lightning_module = SemanticSegmentationTask.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
    )
    lightning_module.eval()
    data_module_params = torch.load(
        checkpoint_path, weights_only=False, map_location=device
    ).get("datamodule_hyper_parameters")
    if not data_module_params:
        raise ValueError(
            "No datamodule_hyper_parameters found in checkpoint. "
            "Please provide a valid checkpoint with datamodule_hyper_parameters."
        )
    lightning_module.datamodule_params = data_module_params
    dataset_args = data_module_params.get("dataset_args", {})

    pred_ds = get_segmentation_dataset(
        images_dir=tif_file_path,
        all_image_bands=dataset_args.get("all_image_bands"),
        rgb_bands=dataset_args.get("rgb_bands"),
        crs=dataset_args.get("crs"),
        res=dataset_args.get("res"),
        reduce_zero_label=dataset_args.get("reduce_zero_label"),
        class_mapping=dataset_args.get("class_mapping"),
        cache_size=0,
        # cache_size=1,
        bands_to_return=dataset_args.get("bands_to_return"),
    )
    module = SegmentationDataModule(
        # val_dataset=pred_ds,
        predict_dataset=pred_ds,
        patch_size=data_module_params.get("patch_size"),
        batch_size=batch_size,
    )
    # module.setup(stage="validate")
    module.setup(stage="predict")
    # data_loader = module.val_dataloader()
    data_loader = module.predict_dataloader()

    with rst.open(tif_file_path) as src:
        transform = src.transform

    for batch in data_loader:
        pixel_windows = [
            from_bounds(
                left=bounds[0],
                bottom=bounds[2],
                right=bounds[1],
                top=bounds[3],
                transform=transform,
            )
            for bounds in batch["bounds"]
        ]
        batch = module.transfer_batch_to_device(batch, device)
        yield batch["image"], pixel_windows
        # yield batch["image"].to(device), pixel_windows


def patch_generator(
    tif_file_path: Union[str, Path],
    patch_size: Optional[int] = 64,
    overlap: Optional[int] = 2,
    batch_size: Optional[int] = 32,
):
    """
    Generator that yields patches of a tif file.

    Parameters
    ----------
        tif_file_path: path to tif file
        patch_size: size of the patches to use for prediction
        overlap: overlap between patches, 2 means 50% overlap
        batch_size: number of patches to accumulate before yielding a batch

    Yields
    ----------
        batch: numpy array of shape (batch_size, patch_size, patch_size, n_channels)
        windows: list of rasterio windows, indicating the location of each patch
                in the original tif file
    """
    with rst.open(tif_file_path) as src:
        height = src.height
        width = src.width

        batch, windows = [], []
        last_row = False
        nan_inputs_logged = False
        for top in range(0, height, patch_size // overlap):
            row_finished = False
            bottom = top + patch_size
            if bottom > height:
                bottom = height
                top = bottom - patch_size
                last_row = True
            for left in range(0, width, patch_size // overlap):
                right = left + patch_size
                if right > width:
                    right = width
                    left = right - patch_size
                    row_finished = True

                window = Window(left, top, patch_size, patch_size)
                windows.append(window)
                img_data = src.read(window=window)
                if np.isnan(img_data).any():
                    if not nan_inputs_logged:
                        print(
                            f"NaN values found in the input image which are not equal to the nodata value. They will be replaced with the nodata value of the file {src.nodata}."
                        )
                        nan_inputs_logged = True
                    img_data = np.nan_to_num(img_data, nan=src.nodata)
                batch.append(img_data)
                if len(batch) == batch_size:
                    yield np.array(batch), windows
                    batch, windows = [], []
                if row_finished:
                    break
            if last_row:
                if len(batch) > 0:
                    yield np.array(batch), windows
                else:
                    break
