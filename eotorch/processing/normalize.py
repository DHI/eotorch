# System features
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple, Union

# Data management
import numpy as np
import rasterio as rst
from alive_progress import alive_bar
from rasterio.enums import Resampling

from eotorch.bandindex import BAND_INDEX
from eotorch.utils import slice_image

logger = logging.getLogger(__name__)


def normalize(
    img_path: str | Path,
    sensor: str = None,
    limits: Dict | Tuple = (1, 99),
    out_res: float | None = None,
    bands: List | None = None,
    out_path: str | Path = None,
    sample_size: int | float = 0.5,
) -> Path:
    """
    Normalizes images within given bounds (percentile or fixed) and converts it to float values in range (0, 1).

    If no `out_res` is specified, the data will be read and written in chunks, otherwise the whole image is read
    and resampled and then written.

    Parameters:
        img_path (str | Path):
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
        out_path (str | Path, optional):
            Full output path. If None is specified, the output file is placed in the `out_dir`
            and named '[input_image_name]_norm.tif'. Defaults to None.
        sample_size (int | float, optional):
            Sample size for finding the scaling limits of the input image. Values <= 1 will use a fraction of the image for
            limit estimation, while values above 1 will read `sample_size` number of blocks to estimate the limits from.
            Defaults to 0.5.

    Returns:

        out_path: The path to the output normalized image.
    """
    img_path = Path(img_path)
    out_path = (
        Path(out_path) if out_path else img_path.parent / f"{img_path.stem}_norm.tif"
    )
    out_path.parent.mkdir(exist_ok=True, parents=True)

    if sensor:
        band_index = BAND_INDEX[sensor]["bandmap"]
        if bands is not None:
            band_index = {band: band_index[band] for band in bands}
        band_list = [b + 1 for b in list(band_index.values())]
    else:
        with rst.open(img_path) as src:
            band_list = list(range(1, src.count + 1))

    if isinstance(limits, tuple):
        limits = sample_limits(
            img_path=img_path,
            limits=limits,
            sample_size=sample_size,
            band_list=band_list,
        )

    if out_res is not None:
        image, profile = resample(
            img_path, res=out_res, resampling="bilinear", indexes=band_list
        )
        profile.update(
            driver="GTiff",
            dtype="float32",
            count=len(band_list),
            nodata=0,
        )
        image = np.nan_to_num(image, nan=0.0)
        out_img = band_scaling(image, limits)
        with rst.open(out_path, "w", **profile) as dst:
            dst.write(out_img)

    else:
        with rst.open(img_path) as src:
            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                dtype="float32",
                count=len(band_list),
                nodata=0,
            )
            windows = slice_image(*src.shape, as_windows=True)
            with rst.open(out_path, "w", **profile) as dst:
                for w in windows:
                    image = src.read(indexes=band_list, window=w)
                    image = np.nan_to_num(image, nan=0.0)
                    out_img = band_scaling(image, limits)
                    dst.write(out_img, window=w)

    return out_path


def sample_limits(
    img_path: Path | str,
    limits: Tuple,
    sample_size: int | float,
    band_list: List = None,
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

        if not band_list:
            band_list = list(range(1, src.count + 1))
        if sample_size > len(windows):
            raise ValueError(
                "The number of samples specified exceed the total number of blocks."
            )
        sample_arr = np.zeros(
            (sample_size, len(band_list), 256, 256), dtype=src.meta["dtype"]
        )
        for n, w in enumerate(random.sample(windows, sample_size)):
            sample_arr[n] = src.read(indexes=band_list, window=w)

        # handle nan values
        sample_arr = np.nan_to_num(sample_arr, nan=src.meta["nodata"])
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
    img_path: Path | str, res: float, resampling: str = "nearest", **kwargs
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


def zscore_normalize(
    input_files: Union[str, Path, List[Union[str, Path]]],
    out_dir: Union[str, Path],
    bands: List[int] = None,
    save_stats: bool = False,
    sample_size: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Applies z-score normalization to a set of images based on mean and standard deviation
    computed across all input files.

    This normalization method transforms the data so that it has zero mean and unit variance:
        normalized = (x - mean) / std

    This is the normalization method used in Prithvi-EO-2.0 fine-tuning examples.

    Parameters:
        input_files (str | Path | List[str | Path]):
            Either a single file path, a directory path (all .tif files will be processed),
            or a list of file paths to normalize.
        out_dir (str | Path):
            Directory where normalized images will be saved. Will be created if it doesn't exist.
        bands (List[int], optional):
            List of 1-based band indices to process. If None, all bands are processed.
            Defaults to None.
        save_stats (bool, optional):
            If True, saves the computed mean and std statistics to a JSON file in out_dir.
            Defaults to False.
        sample_size (float, optional):
            Fraction of pixels to use for computing statistics (0-1). Use < 1 for large datasets
            to speed up computation. Defaults to 1.0 (use all pixels).

    Returns:
        Dict[str, np.ndarray]:
            Dictionary with keys 'mean' and 'std', each containing a numpy array of per-band statistics.

    Example:
        >>> # Normalize all images in a directory
        >>> stats = zscore_normalize(
        ...     input_files="/path/to/images",
        ...     out_dir="/path/to/normalized",
        ...     save_stats=True
        ... )
        >>> print(f"Mean per band: {stats['mean']}")
        >>> print(f"Std per band: {stats['std']}")

        >>> # Normalize specific files
        >>> files = ["/path/to/img1.tif", "/path/to/img2.tif"]
        >>> stats = zscore_normalize(
        ...     input_files=files,
        ...     out_dir="/path/to/normalized",
        ...     bands=[1, 2, 3, 4],  # Only process first 4 bands
        ...     save_stats=True
        ... )
    """
    # Parse input files
    input_files = Path(input_files) if isinstance(input_files, str) else input_files

    if isinstance(input_files, Path):
        if input_files.is_dir():
            # Get all .tif files in directory
            file_list = sorted(input_files.glob("*.tif")) + sorted(
                input_files.glob("*.tiff")
            )
            if not file_list:
                raise ValueError(
                    f"No .tif or .tiff files found in directory: {input_files}"
                )
        else:
            # Single file
            file_list = [input_files]
    else:
        # List of files
        file_list = [Path(f) for f in input_files]

    if not file_list:
        raise ValueError("No input files provided")

    logger.info(f"Processing {len(file_list)} files for z-score normalization")

    # Create output directory
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine bands to process
    with rst.open(file_list[0]) as src:
        n_bands = src.count
        if bands is None:
            bands = list(range(1, n_bands + 1))
        else:
            if max(bands) > n_bands:
                raise ValueError(
                    f"Band index {max(bands)} exceeds number of bands ({n_bands})"
                )

    logger.info(
        f"Computing statistics for {len(bands)} bands across {len(file_list)} files"
    )

    # First pass: compute mean and std across all files
    band_sums = np.zeros(len(bands), dtype=np.float64)
    band_sq_sums = np.zeros(len(bands), dtype=np.float64)
    total_pixels = 0

    with alive_bar(len(file_list), title="Computing statistics", unit="files") as bar:
        for file_path in file_list:
            with rst.open(file_path) as src:
                # Read data in chunks if needed
                if sample_size < 1.0:
                    # Sample random windows
                    windows = list(slice_image(*src.shape, as_windows=True))
                    n_samples = max(1, int(len(windows) * sample_size))
                    windows = random.sample(windows, n_samples)
                else:
                    windows = slice_image(*src.shape, as_windows=True)

                for window in windows:
                    data = src.read(indexes=bands, window=window)
                    # Mask out nodata values if present
                    if src.nodata is not None:
                        mask = data == src.nodata
                        data = data.astype(np.float64)
                        data[mask] = np.nan
                    else:
                        data = data.astype(np.float64)

                    # Also mask out zeros (common nodata indicator)
                    # data[data == 0] = np.nan

                    # Compute sums per band
                    for i in range(len(bands)):
                        valid_pixels = ~np.isnan(data[i])
                        if i == 0:  # Count pixels only once
                            total_pixels += np.sum(valid_pixels)
                        band_sums[i] += np.nansum(data[i])
                        band_sq_sums[i] += np.nansum(data[i] ** 2)

            bar()

    if total_pixels == 0:
        raise ValueError("No valid pixels found in input files")

    # Compute mean and std
    means = band_sums / total_pixels
    stds = np.sqrt((band_sq_sums / total_pixels) - (means**2))

    # Avoid division by zero
    stds[stds < 1e-10] = 1.0

    logger.info(f"Computed statistics from {total_pixels:,} pixels")
    logger.info(f"Means: {means}")
    logger.info(f"Stds: {stds}")

    # Save statistics if requested
    if save_stats:
        stats_file = out_dir / "zscore_stats.json"
        stats_dict = {
            "means": means.tolist(),
            "stds": stds.tolist(),
            "bands": bands,
            "n_files": len(file_list),
            "total_pixels": int(total_pixels),
        }
        with open(stats_file, "w") as f:
            json.dump(stats_dict, f, indent=2)
        logger.info(f"Statistics saved to {stats_file}")

    # Second pass: normalize and save files
    logger.info("Normalizing and saving files...")
    with alive_bar(len(file_list), title="Normalizing files", unit="files") as bar:
        for file_path in file_list:
            out_path = out_dir / f"{file_path.stem}_zscore.tif"

            with rst.open(file_path) as src:
                profile = src.profile.copy()
                profile.update(
                    dtype="float32",
                    # nodata=np.nan,
                )

                # Process in chunks to handle large files
                windows = slice_image(*src.shape, as_windows=True)

                with rst.open(out_path, "w", **profile) as dst:
                    for window in windows:
                        data = src.read(indexes=bands, window=window)

                        # Mask out nodata values
                        # if src.nodata is not None:
                        # mask = data != src.nodata
                        # data = data.astype(np.float32)
                        # data[~mask] = np.nan
                        # else:
                        # data = data.astype(np.float32)

                        # Mask out zeros
                        # data[data == 0] = np.nan

                        # Apply z-score normalization
                        for i in range(len(bands)):
                            data[i] = (data[i] - means[i]) / stds[i]

                        dst.write(data, window=window)

            logger.info(f"Saved normalized file: {out_path}")
            bar()

    logger.info(f"Z-score normalization complete. Files saved to {out_dir}")

    return {"mean": means, "std": stds}
