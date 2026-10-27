from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import rasterio as rst
from rasterio.windows import Window


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

    return arr[..., top:bottom, left:right]


def blend_weights(patch_size: int) -> np.ndarray:
    """2D Hann window for blending overlapping patch predictions, highest at the center.

    Predictions near the edges of a patch are less reliable, as the model sees less
    context there. The minimum weight keeps pixels at the image border, which may be
    covered by a single patch edge only, from getting no weight at all.
    """
    taper = np.clip(np.hanning(patch_size), 1e-3, None)
    return np.outer(taper, taper).astype("float32")


class PatchBlender:
    """Weighted average of overlapping patch predictions.

    The weighted sums are kept in memory-mapped files in a directory, so large rasters
    don't have to fit in memory.
    """

    def __init__(self, directory: str | Path, height: int, width: int, patch_size: int):
        self.directory = Path(directory)
        self.shape = (height, width)
        self.patch_weights = blend_weights(patch_size)
        self.weights = np.lib.format.open_memmap(
            self.directory / "weights.npy", mode="w+", dtype="float32", shape=self.shape
        )
        # Created for the first prediction, once the number of bands is known
        self.sums = None

    def add(self, prediction: np.ndarray, window: Window) -> None:
        """Add the prediction ([bands,] height, width) of the patch at window."""
        prediction = prediction.reshape(-1, *prediction.shape[-2:])
        if self.sums is None:
            self.sums = np.lib.format.open_memmap(
                self.directory / "sums.npy",
                mode="w+",
                dtype="float32",
                shape=(len(prediction), *self.shape),
            )
        row, col = int(window.row_off), int(window.col_off)
        rows = slice(row, row + prediction.shape[-2])
        cols = slice(col, col + prediction.shape[-1])
        self.sums[:, rows, cols] += prediction * self.patch_weights
        self.weights[rows, cols] += self.patch_weights

    def chunks(self, chunk_height: int = 256):
        """Yield the window, blended values and covered pixels for each chunk of rows."""
        height, width = self.shape
        for row in range(0, height, chunk_height):
            rows = slice(row, min(row + chunk_height, height))
            weights = np.array(self.weights[rows])
            covered = weights > 0
            values = self.sums[:, rows] / np.where(covered, weights, 1)
            yield Window(0, row, width, rows.stop - row), values, covered


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
        patch_size: patch size

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
