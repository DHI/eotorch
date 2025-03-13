from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import rasterio as rst
from affine import Affine
from rasterio.windows import Window, from_bounds


def crop_np_to_window(arr: np.ndarray, w_buffered: Window, w_unbuffered: Window):
    left = int(round(w_unbuffered.col_off - w_buffered.col_off))
    top = int(round(w_unbuffered.row_off - w_buffered.row_off))
    right = left + int(round(w_unbuffered.width))
    bottom = top + int(round(w_unbuffered.height))

    return arr[top:bottom, left:right]


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
    if left > 0:  # This is not the first column, so we want the buffer on the left
        window_width -= buffer
        left += buffer
    bottom = top + window_height
    right = left + window_width
    if (
        bottom < img_height
    ):  # This is not the last row, so we want the buffer on the bottom
        window_height -= buffer
    if (
        right < img_width
    ):  # This is not the last column, so we want the buffer on the right
        window_width -= buffer

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
                batch.append(img_data)
                if len(batch) == batch_size:
                    yield np.array(batch).transpose(0, 2, 3, 1), windows
                    batch, windows = [], []
                if row_finished:
                    break
            if last_row:
                if len(batch) > 0:
                    yield np.array(batch).transpose(0, 2, 3, 1), windows
                    break
                else:
                    break


def window_to_np_idc(window: Window) -> Tuple[slice, slice]:
    """Convert a window to numpy array indices."""
    return (
        slice(int(round(window.row_off)), int(round(window.row_off + window.height))),
        slice(int(round(window.col_off)), int(round(window.col_off + window.width))),
    )
