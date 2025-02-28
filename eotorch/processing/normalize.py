# System features
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

# Data management
import numpy as np
import rasterio as rst
from rasterio.enums import Resampling

from eotorch.bandindex import BAND_INDEX
from eotorch.utils import get_outpath, slice_image

logger = logging.getLogger(__name__)


def normalize(
    img_path: Path | str,
    sensor: str,
    limits: Dict | Tuple = (1, 99),
    out_res: float | None = None,
    bands: List | None = None,
    out_dir: str | None = None,
    out_path: str | None = None,
    sample_size: int | float = 0.5,
) -> None:
    """
    Normalizes images within given bounds (percentile or fixed) and converts it to float values in range (0, 1).

    If no `out_res` is specified, the data will be read and written in chunks, otherwise the whole image is read
    and resampled and then written.

    Parameters:
        img_path (Path |str):
            Path of the input image.
        sensor (str):
            Name of the sensor to find the correct band order.
        limits (Dict | Tuple, optional):
            Either a dictionary of manually defined bounds for each band given in the format {band : (lower, upper)}
            or a tuple of the percentile cutoffs for all bands. Defaults to (1, 99).
        out_res (float, optional):
            Output resolution of image. No resampling is done if None. If an `out_res` is specified, the input type of
            `img_path` must be a string pointing to a file. Defaults to None.
        bands (List, optional):
            List of named bands to include from the output image. The bands are stacked in the order they are listed.
            If None, includes all. Defaults to None.
        out_dir (str, optional):
            Output directory. If None is specified, the output file is placed in the current working directory. Defaults to None.
        out_path (str, optional):
            Full output path. If None is specified, the output file is placed in the `out_dir`
            and named '[input_image_name]_norm.tif'. Defaults to None.
        sample_size (int | float, optional):
            Sample size for finding the scaling limits of the input image. Values <= 1 will use a fraction of the image for
            limit estimation, while values above 1 will read `sample_size` number of blocks to estimate the limits from.
            Defaults to 0.5.
    """
    img_path = Path(img_path)
    out_path = get_outpath(
        img_path, out_dir, out_path, subfolder="input", suffix="_norm"
    )

    band_index = BAND_INDEX[sensor]['bandmap']
    if bands is not None:
        band_index = {band: band_index[band] for band in bands}
    band_list = [b + 1 for b in list(band_index.values())]

    if isinstance(limits, tuple):
        limits = sample_limits(img_path, limits, band_list, sample_size)

    if out_res is not None:
        image, profile = resample(
            img_path, res=out_res, resampling="bilinear", indexes=band_list
        )
        profile.update(driver="GTiff", dtype="float32", count=len(band_list), nodata=0)
        image = np.nan_to_num(image, nan=0.0)
        out_img = band_scaling(image, limits)
        with rst.open(out_path, "w", **profile) as dst:
            dst.write(out_img)

    else:
        with rst.open(img_path) as src:
            profile = src.profile.copy()
            profile.update(
                driver="GTiff", dtype="float32", count=len(band_list), nodata=0
            )
            windows = slice_image(*src.shape, as_windows=True)
            with rst.open(out_path, "w", **profile) as dst:
                for w in windows:
                    image = src.read(indexes=band_list, window=w)
                    image = np.nan_to_num(image, nan=0.0)
                    out_img = band_scaling(image, limits)
                    dst.write(out_img, window=w)


def sample_limits(
    img_path: Path | str,
    limits: Tuple,
    band_list: List,
    sample_size: int | float,
) -> Dict:
    """
    Finds the percentile `limits` of the input image by sampling a number of internal tiles/blocks.

    Parameters:
        img_path (str):
            Path of the input image.
        limits (Tuple):
            A tuple of lower and upper percentile cutoffs.
        band_list (List):
            List of band indexes to generate limits for.
        sample_size (int | float):
            Sample size for finding the scaling limits of the input image. Values <= 1 will use a fraction of the image for
            limit estimation, while values above 1 will read `sample_size` number of blocks to estimate the limits from.
            Defaults to 1.0.

    Raises:
        ValueError:
            The `sample_size` is larger than the total number of internal tiles/blocks in the image.

    Returns:
        Dict:
            Cutoff values for each band in the `band_list`.
    """
    img_path = Path(img_path)
    with rst.open(img_path) as src:
        windows = slice_image(*src.shape, as_windows=True)
        sample_size = (
            int(len(windows) * sample_size) if sample_size <= 1 else sample_size
        )
        if sample_size > len(windows):
            raise ValueError(
                "The number of samples specified exceed the total number of blocks."
            )
        sample_arr = np.zeros(
            (sample_size, len(band_list), 256, 256), dtype=src.meta["dtype"]
        )
        for n, w in enumerate(random.sample(windows, sample_size)):
            sample_arr[n] = src.read(indexes=band_list, window=w)
    return {
        i + 1: (
            np.percentile(sample_arr[:, i, ...], limits[0]).astype(src.meta["dtype"]),
            np.percentile(sample_arr[:, i, ...], limits[1]).astype(src.meta["dtype"]),
        )
        for i in range(len(band_list))
    }


def band_scaling(
    img: np.ndarray,
    limits: Tuple | Dict,
) -> np.ndarray:
    """
    Band scaling using either percentile cutoffs (e.g. 2nd and 98th percentile) common to visualization
    or manually defined bounds for each band.

    Parameters
    ----------
        img (np.ndarray):
            The array/image to be rescaled. Can be either 2D or 3D.
        limits (Dict, optional):
            Either a dictionary of manually defined bounds for each band given in the format {band : (lower, upper)}
            or a tuple of the percentile cutoffs. Defaults to (2, 98).
        heuristic (bool, optional):
            Reduce the memory overhead by calculating statistics from a random subset of pixels. Defaults to False.
        n_samples (int, optional):
            Number of sample pixels to use when applying heuristics to find the limits. Defaults to 1e5.

    Returns
    ----------
        np.ndarray: Rescaled image as float in range 0, 1
    """
    if isinstance(limits, tuple):
        lower, upper = limits

    if img.ndim == 2:
        img = img[np.newaxis, ...]

    # Normalization based on the lower and upper nth percentile
    out_img = np.zeros(img.shape, dtype="float32")
    for n, band in enumerate(img):
        if isinstance(limits, dict):
            bottom, top = limits[n + 1]
        else:
            bottom = np.percentile(band[band > 0], lower)
            top = np.percentile(band[band > 0], upper)

        out_img[n] = band.astype("float32") - bottom
        out_img[n] = out_img[n] / (top - bottom)

    out_img = np.clip(out_img, 1e-5, 1)
    out_img[img == 0] = 0

    return out_img


def resample(
    img_path: Path | str, 
    res: float, 
    resampling: str = 'nearest', 
    **kwargs
) -> Tuple[np.ndarray, dict]:
    """
    Resamples the input raster to the target spatial resolution using the specified resampling method.

    Parameters:
        img_path (Path | str): 
            Input raster to be resampled.
        res (float): 
            Target spatial output resolution in georeferenced units.
        resampling (str, optional): 
            Resampling method to use when up- or downsampling the image. Defaults to 'nearest'.

    Returns:
        Tuple[np.ndarray, dict]: 
            Tuple of resampled data array and updated metadata.
    """    
    rs_method = {
        "nearest": Resampling(0),
        "bilinear": Resampling(1),
        "cubic": Resampling(2),
        "cubic_spline": Resampling(3),
        "lanczos": Resampling(4),
        "average": Resampling(5),
        "mode": Resampling(6),
        "gauss": Resampling(7),
        "max": Resampling(8),
        "min": Resampling(9),
        "med": Resampling(10),
        "q1": Resampling(11),
        "q3": Resampling(12),
        "sum": Resampling(13),
        "rms": Resampling(14),
    }
    img_path = Path(img_path)
    with rst.open(img_path) as src:
        profile = src.meta.copy()
        scale_factor = res / src.res[0]
        n_bands = (
            src.count if kwargs.get("indexes") is None else len(kwargs.get("indexes"))
        )
        out_image = src.read(
            out_shape=(
                n_bands,
                int(src.height / scale_factor),
                int(src.width / scale_factor),
            ),
            resampling=rs_method[resampling],
            **kwargs,
        )
        transform = src.transform * src.transform.scale(
            (src.width / out_image.shape[-1]), (src.height / out_image.shape[-2])
        )
    profile.update(
        {
            "transform": transform,
            "height": out_image.shape[-2],
            "width": out_image.shape[-1],
            "count": out_image.shape[0] if np.ndim(out_image) == 3 else 1,
        }
    )

    return out_image, profile
