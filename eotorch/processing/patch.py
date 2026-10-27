import random
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import rasterio as rst
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from tqdm import tqdm


def _read_aligned(
    img_path: str | Path, label_path: str | Path, resampling: Resampling
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], float | int]:
    """Read an image, and its labels reprojected onto the grid of the image if needed.

    Returns the image, the labels, the image metadata and the label nodata value
    (0 if the label file has none), which fills image areas without labels.
    """
    with rst.open(img_path) as img_src, rst.open(label_path) as label_src:
        image, meta = img_src.read(), img_src.meta.copy()
        label_nodata = 0 if label_src.nodata is None else label_src.nodata
        if (img_src.crs, img_src.transform, img_src.shape) == (
            label_src.crs,
            label_src.transform,
            label_src.shape,
        ):
            return image, label_src.read(1), meta, label_nodata

        labels = np.full(img_src.shape, label_nodata, dtype=label_src.dtypes[0])
        reproject(
            source=rst.band(label_src, 1),
            destination=labels,
            dst_transform=img_src.transform,
            dst_crs=img_src.crs,
            dst_nodata=label_nodata,
            resampling=resampling,
        )
    return image, labels, meta, label_nodata


def _should_skip_patch(
    image: np.ndarray,
    labels: np.ndarray,
    image_nodata: float | int | None,
    label_nodata: float | int,
    empty_img_threshold: float | None,
    empty_label_threshold: float | None,
    value_threshold: float | int | None,
) -> bool:
    """Whether a patch is too empty to be useful for training."""
    empty_img = (image == (0 if image_nodata is None else image_nodata)).all(axis=0)
    empty_labels = labels == label_nodata
    if empty_img.all() or empty_labels.all():
        return True
    if empty_img_threshold is not None and empty_img.mean() > empty_img_threshold:
        return True
    if (
        empty_label_threshold is not None
        and empty_labels.mean() > empty_label_threshold
    ):
        return True
    return value_threshold is not None and (labels <= value_threshold).all()


def _overlaps_block(r: int, c: int, blocks: set[tuple[int, int]], size: int) -> bool:
    """Whether a patch at (r, c) overlaps any of the blocks on the grid of the given size."""
    rows = {r // size * size, (r + size - 1) // size * size}
    cols = {c // size * size, (c + size - 1) // size * size}
    return any((row, col) in blocks for row in rows for col in cols)


def generate_patches_from_files(
    img_path: str | Path,
    label_path: str | Path,
    out_dir: str | Path,
    patch_size: int = 128,
    val_fraction: float = 0.2,
    n_random_offsets: int = 0,
    random_seed: int = 42,
    empty_img_threshold: float | None = 0.5,
    empty_label_threshold: float | None = None,
    value_threshold: float | int | None = None,
    resampling: str = "nearest",
    show_progress: bool = True,
) -> None:
    """
    Split an image and its labels into patch files for training and validation.

    The image is divided into non-overlapping blocks of ``patch_size``. A random
    ``val_fraction`` of the blocks is written to ``out_dir / "val"``, the others to
    ``out_dir / "train"``, each with ``n_random_offsets`` extra patches shifted by up to
    half a patch for more variety. Shifted patches never overlap a validation block.

    The patches are GeoTIFFs named ``<image name>_<n>_image.tif`` and
    ``<image name>_<n>_label.tif``. Load them with
    ``get_segmentation_dataset(..., image_glob="*_image.tif", label_glob="*_label.tif")``
    (or get_regression_dataset) and use one patch per sample with
    ``RasterDataModule(..., patch_size=None)``.

    Vector labels can be rasterized onto the image grid first, e.g. with
    :class:`eotorch.processing.rasterize.FileSource`.

    Parameters
    ----------
    img_path : str | Path
        Path to the image raster, which defines the grid of the patches.
    label_path : str | Path
        Path to the label raster. It may differ from the image in extent, resolution
        and CRS, in which case it is reprojected onto the image grid.
    out_dir : str | Path
        Output directory, in which "train" and "val" directories are created.
    patch_size : int, default=128
        Patch width/height in pixels.
    val_fraction : float, default=0.2
        Fraction of the blocks used for validation.
    n_random_offsets : int, default=0
        Number of randomly shifted patches per training block, in addition to the
        block itself.
    random_seed : int, default=42
        Seed for choosing the validation blocks and the offsets.
    empty_img_threshold : float | None, default=0.5
        Skip patches with a larger fraction of image nodata pixels. Patches without
        any valid image or label pixels are always skipped.
    empty_label_threshold : float | None, default=None
        Skip patches with a larger fraction of label nodata pixels.
    value_threshold : float | int | None, default=None
        Skip patches where all label values are less than or equal to this value.
    resampling : str, default="nearest"
        Name of the rasterio resampling method used to reproject the labels, e.g.
        "nearest" or "mode" for classes, or "bilinear" for continuous values.
    show_progress : bool, default=True
        Whether to show a progress bar.
    """
    image, labels, meta, label_nodata = _read_aligned(
        img_path, label_path, Resampling[resampling]
    )
    height, width = labels.shape
    blocks = [
        (r, c)
        for r in range(0, height - patch_size + 1, patch_size)
        for c in range(0, width - patch_size + 1, patch_size)
    ]
    rng = random.Random(random_seed)
    val_blocks = set(rng.sample(blocks, round(len(blocks) * val_fraction)))

    half = patch_size // 2
    train_locations = []
    for r, c in blocks:
        if (r, c) in val_blocks:
            continue
        train_locations.append((r, c))
        for _ in range(n_random_offsets):
            train_locations.append(
                (r + rng.randint(-half, half), c + rng.randint(-half, half))
            )
    train_locations = [
        (r, c)
        for r, c in train_locations
        if 0 <= r <= height - patch_size
        and 0 <= c <= width - patch_size
        and not _overlaps_block(r, c, val_blocks, patch_size)
    ]

    stem = Path(img_path).stem
    for split, locations in (("train", train_locations), ("val", sorted(val_blocks))):
        split_dir = Path(out_dir) / split
        split_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for r, c in tqdm(locations, desc=split, disable=not show_progress):
            rows, cols = slice(r, r + patch_size), slice(c, c + patch_size)
            image_patch, label_patch = image[:, rows, cols], labels[rows, cols]
            if _should_skip_patch(
                image_patch,
                label_patch,
                meta["nodata"],
                label_nodata,
                empty_img_threshold,
                empty_label_threshold,
                value_threshold,
            ):
                continue

            profile = {
                **meta,
                "driver": "GTiff",
                "height": patch_size,
                "width": patch_size,
                "transform": window_transform(
                    Window(c, r, patch_size, patch_size), meta["transform"]
                ),
            }
            with rst.open(split_dir / f"{stem}_{n}_image.tif", "w", **profile) as dst:
                dst.write(image_patch)
            label_profile = {
                **profile,
                "count": 1,
                "dtype": label_patch.dtype,
                "nodata": label_nodata,
            }
            with rst.open(
                split_dir / f"{stem}_{n}_label.tif", "w", **label_profile
            ) as dst:
                dst.write(label_patch, 1)
            n += 1


def clear_patches(wildcard: str, patch_dir: str | Path) -> None:
    """
    Deletes existing patches matching the scene name.

    Parameters:
        wildcard (str):
            Glob wildcard used to match patch files.
        patch_dir (str | Path):
            Patch directory.
    """
    for path in glob(str(Path(patch_dir) / wildcard)):
        Path(path).unlink()
