import os
import random
from glob import glob
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rst
import shapely
from alive_progress import alive_bar
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import box as shapely_box
from tqdm import tqdm

from eotorch.processing.normalize import band_scaling
from eotorch.io import read_vector


def _should_skip_patch(
    image: np.ndarray,
    labels: np.ndarray,
    slice_obj: tuple[slice, slice],
    patch_size: int,
    meta: dict[str, Any],
    value_threshold: float | int | None,
    empty_img_threshold: float | None,
    empty_label_threshold: float | None,
    frac_empty_patches: float,
) -> tuple[bool, str | None]:
    """
    Determine if a patch should be skipped based on filtering criteria.
    
    Parameters
    ----------
    image : np.ndarray
        Input image array with shape (bands, height, width).
    labels : np.ndarray
        Label array with shape (height, width).
    slice_obj : tuple[slice, slice]
        2D slice tuple indicating the patch location.
    patch_size : int
        Patch width/height in pixels.
    meta : dict[str, Any]
        Raster metadata containing at least the nodata value.
    value_threshold : float | int | None
        Skip patch when all label values are less than or equal to this threshold.
    empty_img_threshold : float | None
        Skip patch when empty image ratio (all bands == nodata) exceeds this threshold.
    empty_label_threshold : float | None
        Skip patch when empty label ratio (label == 0) exceeds this threshold.
    frac_empty_patches : float
        Fraction of empty patches to retain even if they exceed the empty_threshold.

    Returns
    -------
    tuple[bool, str | None]
        (should_skip, reason). `reason` is populated only when skipped.
    """
    def _dice_roll(prob: float) -> bool:
        return ~(np.random.random(1) <= prob)[0]
        
    # Skip if all nodata or all zeros
    if (image[:, *slice_obj] == meta['nodata']).all() or (image[:, *slice_obj] == 0).all():
        return True, 'all_nodata_or_zero'
    
    if value_threshold is not None:
        if (labels[slice_obj] <= value_threshold).all():
            if _dice_roll(frac_empty_patches):
                return True, f'all_labels_leq_value_threshold({value_threshold})'
    
    if empty_img_threshold is not None:
        img_empty_ratio = ((image[:, *slice_obj] == meta['nodata']).all(axis=0).sum() / patch_size**2)
        if img_empty_ratio > empty_img_threshold:
            return True, (
                f'img_empty_ratio({img_empty_ratio:.4f})>'
                f'empty_img_threshold({empty_img_threshold})'
            )
        
    if empty_label_threshold is not None:
        label_empty_ratio = labels[slice_obj][labels[slice_obj]==0].size / patch_size**2
        if label_empty_ratio > empty_label_threshold:
            if _dice_roll(frac_empty_patches):
                return True, (
                    f'label_empty_ratio({label_empty_ratio:.4f})>'
                    f'empty_label_threshold({empty_label_threshold})'
                )
        
    return False, None


def _write_patch(
    image: np.ndarray,
    labels: np.ndarray,
    slice_obj: tuple[slice, slice],
    patch_index: int,
    img_stem: str,
    out_dir: Path,
    meta: dict[str, Any],
    patch_size: int,
) -> None:
    """
    Write a single patch pair (feature and label) to disk.
    
    Parameters
    ----------
    image : np.ndarray
        Input image array with shape (bands, height, width).
    labels : np.ndarray
        Label array with shape (height, width).
    slice_obj : tuple[slice, slice]
        2D slice tuple indicating the patch location.
    patch_index : int
        Numeric index for naming the patch pair.
    img_stem : str
        Image filename stem used in output filenames.
    out_dir : Path
        Output directory for written patch files.
    meta : dict[str, Any]
        Source image metadata used to derive output metadata.
    patch_size : int
        Patch width/height in pixels.
    """
    feature_name = f'{img_stem}_{patch_index}_image.tiff'
    label_name = f'{img_stem}_{patch_index}_label.tiff'
    
    x, y = slice_obj
    
    # Write feature patch
    out_meta = meta_from_origin(image, x.start, y.start, meta, patch_size, dtype='float32')
    with rst.open(out_dir / feature_name, 'w', **out_meta) as dst:
        dst.write(image[:, *slice_obj].astype('float32'))
    
    # Write label patch
    out_meta = meta_from_origin(labels, x.start, y.start, meta, patch_size, dtype=labels.dtype)
    with rst.open(out_dir / label_name, 'w', **out_meta) as dst:
        dst.write(labels[slice_obj], 1)


def generate_patches_from_files(
    img_path: str | Path,
    label_path: str | Path,
    out_dir: str | Path,
    patch_size: int = 128,
    val_fraction: float = 0.2,
    n_random_offsets: int = 2,
    random_seed: int = 42,
    empty_img_threshold: float | None = 0.5,
    empty_label_threshold: float | None = None,
    value_threshold: float | int | None = None,
    frac_empty_patches: float = 0,
    show_progress: bool = True,
    log_skipped_patches: bool = False,
) -> None:
    """
    Generate and save training and validation patches from image and label files.

    A non-overlapping grid of `patch_size` blocks is laid over the raster. A fraction
    of the blocks are reserved for validation and written grid-aligned (one patch per
    block) to `out_dir / 'val'`. The remaining blocks are used for training: each is
    sampled `n_random_offsets` times with a random pixel offset and written to
    `out_dir / 'train'`, skipping any offset patch that would overlap a validation
    block. This mirrors the grid + random-offset sampling used by
    `generate_train_val_patches`.

    Parameters
    ----------
    img_path : str | Path
        Path to the normalized image raster.
    label_path : str | Path
        Path to the label raster.
    out_dir : str | Path
        Output directory. Training patches are written to `out_dir / 'train'`,
        validation patches to `out_dir / 'val'`.
    patch_size : int, default=128
        Patch width/height in pixels.
    val_fraction : float, default=0.2
        Fraction of grid blocks reserved for validation.
    n_random_offsets : int, default=2
        Number of randomly offset patches to generate per training block.
        Setting to 0 will generate one patch per training block (grid-aligned).
    random_seed : int, default=42
        Seed for reproducibility of the validation block selection and random offsets.
    empty_img_threshold : float | None, default=0.5
        Maximum allowed empty-image ratio before skipping a patch.
    empty_label_threshold : float | None, default=None
        Maximum allowed empty-label ratio before skipping a patch.
    value_threshold : float | int | None, default=None
        Skip patch when all label values are <= this threshold.
    frac_empty_patches : float, default=0
        Fraction of empty patches to retain even if they exceed the `empty_label_threshold` and `value_threshold`.
    show_progress : bool, default=True
        Whether to display a progress bar while writing patches.
    log_skipped_patches : bool, default=False
        Whether to print info logs with patch index and skip reason.
    """
    out_dir = Path(out_dir)
    train_dir = out_dir / 'train'
    val_dir = out_dir / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    n_random_offsets += 1  # Ensure at least one patch per training block

    with rst.open(img_path) as img_src, rst.open(label_path) as label_src:
        image = img_src.read()
        labels = label_src.read(indexes=1)
        height, width = img_src.shape
        meta = img_src.meta.copy()

    blocks = [
        (r, c)
        for r in range(0, height - patch_size + 1, patch_size)
        for c in range(0, width - patch_size + 1, patch_size)
    ]

    rng = random.Random(random_seed)
    img_stem = Path(img_path).stem

    # Validation locations are selected first to avoid overlap with training patches.
    # Only one validation patch per block is generated (grid-aligned).
    n_val = max(1, round(len(blocks) * val_fraction))
    val_block_list = rng.sample(blocks, n_val) if len(blocks) > n_val else blocks
    val_locations = set(val_block_list)
    train_block_list = [b for b in blocks if b not in val_locations]

    def _skip_and_write(r: int, c: int, index: int, dest_dir: Path) -> bool:
        if r + patch_size > height or c + patch_size > width:
            return False

        s = (slice(r, r + patch_size), slice(c, c + patch_size))

        should_skip, reason = _should_skip_patch(
            image,
            labels,
            s,
            patch_size,
            meta,
            value_threshold,
            empty_img_threshold,
            empty_label_threshold,
            frac_empty_patches,
        )
        if should_skip:
            if log_skipped_patches:
                msg = f"[INFO] Skipped patch at (row={r}, col={c}) ({img_stem}): {reason}"
                if show_progress:
                    tqdm.write(msg)
                else:
                    print(msg)
            return False

        _write_patch(image, labels, s, index, img_stem, dest_dir, meta, patch_size)
        return True

    n_val_written = 0
    val_iterator = tqdm(sorted(val_locations), desc='Validation') if show_progress else sorted(val_locations)
    for vr, vc in val_iterator:
        if _skip_and_write(vr, vc, n_val_written, val_dir):
            n_val_written += 1

    n_train_written = 0
    candidates = [(br, bc) for br, bc in train_block_list for _ in range(n_random_offsets)]
    train_iterator = tqdm(candidates, desc='Training') if show_progress else candidates

    for br, bc in train_iterator:
        offset_r = rng.randint(0, patch_size)
        offset_c = rng.randint(0, patch_size)

        r = br + offset_r
        c = bc + offset_c

        if _overlaps_any_val(r, c, val_locations, patch_size):
            continue

        if _skip_and_write(r, c, n_train_written, train_dir):
            n_train_written += 1


def meta_from_origin(
    image: np.ndarray,
    x: float,
    y: float,
    meta: dict[str, Any],
    patch_size: int,
    dtype: str | np.dtype[Any] | None = None,
) -> dict[str, Any]:
    """
    Generates new metadata of a patch from origin coordinates.

    Parameters
    ----------
    image : np.ndarray
        Input image features.
    x : float
        x-coordinate of corner pixel.
    y : float
        y-coordinate of corner pixel.
    meta : dict[str, Any]
        Original metadata.
    patch_size : int
        Patch width/height in pixels.
    dtype : str | np.dtype[Any] | None, optional
        dtype of the output. If None, infers the dtype from the metadata. Defaults to None.

    Returns
    -------
    dict[str, Any]
        Patch metadata.
    """
    x_size = meta['transform'][0]
    y_size = -meta['transform'][4] if meta['transform'][4] < 0 else meta['transform'][4]
    origin = meta['transform'][2]+(y*y_size), meta['transform'][5]-(x*x_size)

    out_meta = meta.copy()
    dtype = meta['dtype'] if dtype is None else dtype
    if np.ndim(image) == 3:
        count = image.shape[0]
    elif np.ndim(image) == 2:
        count = 1
    
    out_meta.update({
        'width' : patch_size,
        'height' : patch_size,
        'count' : count,
        'transform' : from_origin(*origin, x_size, y_size),
        'dtype' : dtype
    })
    
    return out_meta
    

def clear_patches(wildcard: str, patch_dir: str | Path) -> None:
    """
    Deletes existing patches matching the scene name.

    Parameters
    ----------
    wildcard : str
        Glob wildcard used to match patch files.
    patch_dir : str | Path
        Patch directory.
    """
    img_paths = glob(os.path.join(patch_dir, f'{wildcard}'))
    for img_path in img_paths:
        os.remove(img_path)


def get_feature_overlapping_blocks(
    vrt_path: str | Path,
    features_union: shapely.Geometry,
    patch_size: int,
) -> list[tuple[int, int]]:
    """Return block origins on a non-overlapping grid that intersect a geometry.

    Parameters
    ----------
    vrt_path : str | Path
        Path to the source raster / VRT.
    features_union : shapely.Geometry
        Union of all feature geometries, in the same CRS as the raster.
    patch_size : int
        Patch side length in pixels.

    Returns
    -------
    list[tuple[int, int]]
        Block origins (row, col) in pixel coordinates.
    """
    with rst.open(vrt_path) as src:
        height, width = src.height, src.width
        tf = src.transform

    x_res, y_res = tf.a, abs(tf.e)

    def block_box(r, c):
        west = tf.c + c * x_res
        north = tf.f - r * y_res
        return shapely_box(west, north - patch_size * y_res, west + patch_size * x_res, north)

    blocks = []
    for r in range(0, height - patch_size + 1, patch_size):
        for c in range(0, width - patch_size + 1, patch_size):
            if features_union.intersects(block_box(r, c)):
                blocks.append((r, c))
    return blocks


def _overlaps_any_val(
    r: int,
    c: int,
    val_block_set: set[tuple[int, int]],
    patch_size: int,
) -> bool:
    """Check whether a patch at pixel (r, c) overlaps any validation block.

    Checks only the 3x3 grid neighbourhood, so the lookup is O(9) regardless
    of how many validation blocks exist.

    Parameters
    ----------
    r : int
        Row index (in pixels) of the patch origin.
    c : int
        Column index (in pixels) of the patch origin.
    val_block_set : set[tuple[int, int]]
        Set of validation block origins (row, col) on the same grid.
    patch_size : int
        Patch side length in pixels.

    Returns
    -------
    bool
        True if the patch overlaps any validation block.
    """
    br_base = (r // patch_size) * patch_size
    bc_base = (c // patch_size) * patch_size
    for nr in (br_base - patch_size, br_base, br_base + patch_size):
        for nc in (bc_base - patch_size, bc_base, bc_base + patch_size):
            if (nr, nc) in val_block_set:
                if (r < nr + patch_size and nr < r + patch_size and
                        c < nc + patch_size and nc < c + patch_size):
                    return True
    return False


def _load_label_geoms(
    label_sources: str | Path | list[str | Path] | gpd.GeoDataFrame | list[gpd.GeoDataFrame],
    class_map: dict[str, int],
    crs: rst.CRS | str,
) -> dict[int, list[shapely.Geometry]]:
    """Load geometries from shapefiles or geodataframes and map to class values.

    Parameters
    ----------
    label_sources : str | Path | list | gpd.GeoDataFrame | list[gpd.GeoDataFrame]
        Single or multiple shapefiles (as paths) or geodataframes.
    class_map : dict[str, int]
        Mapping of class names to raster values, e.g. {'water': 1, 'land': 2, 'structures': 3}.
    crs : rst.CRS | str
        Target CRS to reproject geometries to.

    Returns
    -------
    dict[int, list[shapely.Geometry]]
        Mapping of class values to lists of geometries: {1: [geom1, ...], 2: [geom2, ...], ...}.
    """
    if isinstance(label_sources, (str, Path)):
        label_sources = [label_sources]
    elif isinstance(label_sources, gpd.GeoDataFrame):
        label_sources = [label_sources]

    geoms_by_class = {v: [] for v in class_map.values()}

    for source in label_sources:
        if isinstance(source, (str, Path)):
            gdf = read_vector(source).to_crs(crs)
        else:
            gdf = source.to_crs(crs)

        if isinstance(source, (str, Path)):
            class_name = Path(source).stem
        else:
            class_name = None
            for class_n in class_map.keys():
                if class_n in gdf.columns:
                    class_name = class_n
                    break

        if class_name in class_map:
            class_value = class_map[class_name]
            geoms_by_class[class_value].extend(gdf.geometry.tolist())

    return geoms_by_class


def _rasterize_patch_labels(
    r: int,
    c: int,
    patch_size: int,
    tf: rst.Affine,
    geoms_by_class: dict[int, list[shapely.Geometry]],
) -> np.ndarray:
    """Rasterize label geometries for a patch window.

    Parameters
    ----------
    r : int
        Row index (in pixels) of the patch origin.
    c : int
        Column index (in pixels) of the patch origin.
    patch_size : int
        Patch side length in pixels.
    tf : rst.Affine
        Raster transform.
    geoms_by_class : dict[int, list[shapely.Geometry]]
        Mapping of class values to geometry lists.

    Returns
    -------
    np.ndarray
        Label array of shape (patch_size, patch_size), dtype uint8.
    """
    window = rst.windows.Window(c, r, patch_size, patch_size)
    window_tf = rst.windows.transform(window, tf)

    label_patch = np.zeros((patch_size, patch_size), dtype='uint8')

    # Rasterize each class in order
    for class_value, geom_list in sorted(geoms_by_class.items()):
        if geom_list:
            rasterize(
                [(g, class_value) for g in geom_list],
                out=label_patch,
                transform=window_tf,
                default_value=class_value,
            )

    return label_patch


def _write_image_label_patch(
    src: rst.DatasetReader,
    r: int,
    c: int,
    patch_size: int,
    out_path_img: str | Path,
    out_path_label: str | Path | None,
    geoms_by_class: dict[int, list[shapely.Geometry]] | None = None,
    limits: dict[int, tuple[float, float]] | tuple[float, float] | None = None,
    indexes: list[int] = [1, 2, 3, 4]
) -> bool:
    """Write both image and label patches for a window.

    Parameters
    ----------
    src : rst.DatasetReader
        Open raster source.
    r : int
        Row index (in pixels) of the patch origin.
    c : int
        Column index (in pixels) of the patch origin.
    patch_size : int
        Patch side length in pixels.
    out_path_img : str | Path
        Output path for the image patch.
    out_path_label : str | Path | None
        Output path for the label patch. If None, labels are not written.
    geoms_by_class : dict[int, list[shapely.Geometry]] | None, optional
        Mapping of class values to geometry lists for rasterization.
    limits : dict[int, tuple[float, float]] | tuple[float, float] | None, optional
        Normalization limits for band_scaling.
    indexes : list[int], optional
        List of band indexes to read from the source raster. Defaults to [1, 2, 3, 4].

    Returns
    -------
    bool
        False if the patch is entirely nodata/zero; True if written successfully.
    """
    tf = src.transform
    window = rst.windows.Window(c, r, patch_size, patch_size)
    patch = src.read(indexes=indexes, window=window)

    nodata = src.nodata
    if (patch == 0).all() or (nodata is not None and (patch == nodata).all()):
        return False

    x_res, y_res = tf.a, abs(tf.e)
    west = tf.c + c * x_res
    north = tf.f - r * y_res

    if limits is not None:
        patch = band_scaling(patch, limits)

    out_meta = src.meta.copy()
    out_meta.update({
        'width': patch_size,
        'height': patch_size,
        'count': len(indexes),
        'dtype': 'float32',
        'transform': from_origin(west, north, x_res, y_res),
        'compress': 'LZW',
        'driver': 'GTiff',
    })
    with rst.open(out_path_img, 'w', **out_meta) as dst:
        dst.write(patch.astype('float32'))

    # Write label patch if geometries provided
    if out_path_label is not None and geoms_by_class is not None:
        label_patch = _rasterize_patch_labels(r, c, patch_size, tf, geoms_by_class)

        label_meta = out_meta.copy()
        label_meta.update({
            'count': 1,
            'dtype': 'uint8',
            'nodata': 0,
        })
        with rst.open(out_path_label, 'w', **label_meta) as dst:
            dst.write(label_patch, 1)

    return True


def generate_train_val_patches(
    src_path: str | Path,
    feature_shps: list[str | Path] | gpd.GeoDataFrame | list[gpd.GeoDataFrame] | gpd.GeoSeries | list[gpd.GeoSeries] | shapely.Geometry | list[shapely.Geometry],
    train_patch_dir: str | Path,
    val_patch_dir: str | Path,
    patch_size: int,
    val_fraction: float = 0.2,
    n_random_offsets: int = 2,
    indexes: list[int] = [1, 2, 3, 4],
    random_seed: int = 42,
    limits: dict[int, tuple[float, float]] | tuple[float, float] | None = None,
    label_sources: str | Path | list[str | Path] | gpd.GeoDataFrame | list[gpd.GeoDataFrame] | None = None,
    class_map: dict[str, int] | None = None,
) -> None:
    """Generate random training and validation image patches within feature regions.

    Patches are randomly sampled from locations that intersect features. Validation
    patches are grid-aligned (one per block). Training patches are randomly positioned
    to ensure diversity while avoiding overlap with validation patches.

    Parameters
    ----------
    src_path : str | Path
        Path to the source raster / VRT.
    feature_shps : list[str | Path] | gpd.GeoDataFrame | list[gpd.GeoDataFrame] | shapely.Geometry | list[shapely.Geometry]
        Shapefiles, GeoDataFrames, or geometries whose features determine which regions contain patches.
    train_patch_dir : str | Path
        Output directory for training patches.
    val_patch_dir : str | Path
        Output directory for validation patches.
    patch_size : int
        Patch side length in pixels.
    val_fraction : float, optional
        Fraction of blocks reserved for validation.
    n_random_offsets : int, optional
        Number of randomly offset patches to generate per training block.
        Setting to 0 will generate one patch per block (grid-aligned).
    indexes : list[int], optional
        List of band indexes to read from the source raster. Defaults to [1, 2, 3, 4].
    random_seed : int, optional
        Seed for reproducibility.
    limits : dict[int, tuple[float, float]] | tuple[float, float] | None, optional
        Normalization limits for band_scaling. If dict, format is {band : (lower, upper)}.
        If tuple, (lower, upper) percentile cutoffs applied to all bands.
        If None, no normalization is applied.
    label_sources : str | Path | list | gpd.GeoDataFrame | list[gpd.GeoDataFrame] | None, optional
        Shapefile paths or GeoDataFrames containing label geometries. Can be a single
        source or list of sources. If None, labels are not written.
    class_map : dict[str, int] | None, optional
        Mapping of class names to raster values (e.g., {'water': 1, 'land': 2}).
        Required if label_sources is provided.
    """
    train_patch_dir = Path(train_patch_dir)
    val_patch_dir = Path(val_patch_dir)
    train_patch_dir.mkdir(parents=True, exist_ok=True)
    val_patch_dir.mkdir(parents=True, exist_ok=True)

    n_random_offsets += 1  # Ensure at least one patch per training block

    with rst.open(src_path) as src:
        crs = src.crs

    if not isinstance(feature_shps, (list, tuple)):
        feature_shps = [feature_shps]
    if isinstance(feature_shps[0], (str, Path)):
        features = pd.concat([read_vector(shp).to_crs(crs).geometry for shp in feature_shps])
    if isinstance(feature_shps[0], gpd.GeoDataFrame):
        features = pd.concat([gdf.to_crs(crs).geometry for gdf in feature_shps])
    if isinstance(feature_shps[0], (shapely.Geometry, gpd.GeoSeries)):
        features = feature_shps
    features_union = shapely.unary_union(features)

    geoms_by_class = None
    if label_sources is not None and class_map is not None:
        geoms_by_class = _load_label_geoms(label_sources, class_map, crs)

    print('Building feature-overlapping block grid...')
    blocks = get_feature_overlapping_blocks(src_path, features_union, patch_size)
    print(f'Feature-overlapping blocks: {len(blocks)}')

    rng = random.Random(random_seed)

    with rst.open(src_path) as src:
        height, width = src.height, src.width

        # Validation locations are selected first to avoid overlap with training patches
        # Only one validation patch per block is generated (grid-aligned)
        n_val = max(1, round(len(blocks) * val_fraction))
        val_block_list = rng.sample(blocks, n_val) if len(blocks) > n_val else blocks[:n_val]
        val_locations = set((br, bc) for br, bc in val_block_list)
        train_block_list = [b for b in blocks if b not in val_locations]

        print(f'Train blocks: {len(train_block_list)},  Val blocks: {len(val_locations)}')

        target_count = len(train_block_list) * n_random_offsets  # Target number of training patches to generate

        val_written = 0
        with alive_bar(len(val_locations), title='Validation', unit=' patches') as bar:
            for i, (vr, vc) in enumerate(sorted(val_locations)):
                img_path = val_patch_dir / f'val_{i:05d}_image.tiff'
                lbl_path = val_patch_dir / f'val_{i:05d}_label.tiff' if geoms_by_class else None
                if _write_image_label_patch(src, vr, vc, patch_size, img_path, lbl_path, geoms_by_class, limits, indexes=indexes):
                    val_written += 1
                bar()

        train_written = 0
        with alive_bar(target_count, title='Training ', unit=' patches') as bar:
            for br, bc in train_block_list:
                for _ in range(n_random_offsets):
                    # Random offset within the block
                    offset_r = rng.randint(0, patch_size)
                    offset_c = rng.randint(0, patch_size)

                    r = br + offset_r
                    c = bc + offset_c

                    # Check bounds
                    if r + patch_size > height or c + patch_size > width:
                        bar()
                        continue

                    # Check if patch overlaps any validation block
                    if _overlaps_any_val(r, c, val_locations, patch_size):
                        bar()
                        continue

                    img_path = train_patch_dir / f'train_{train_written:05d}_image.tiff'
                    lbl_path = train_patch_dir / f'train_{train_written:05d}_label.tiff' if geoms_by_class else None
                    if _write_image_label_patch(src, r, c, patch_size, img_path, lbl_path, geoms_by_class, limits, indexes=indexes):
                        train_written += 1
                    bar()

    print(f'Done — {train_written} training patches, {val_written} validation patches.')


def compute_patch_stats(
    patch_dirs: str | Path | list[str | Path],
    image_suffix: str = 'image',
    nodata: float | None = 0.0,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-band mean and standard deviation across a set of patch rasters.

    Statistics are accumulated in a single pass (running sum and sum-of-squares
    per band), so patches are never all held in memory at once.

    Parameters
    ----------
    patch_dirs : str | Path | list[str | Path]
        One or more directories to scan for patches (e.g. both the train and
        validation patch directories, so stats reflect the full dataset).
    image_suffix : str, optional
        Suffix identifying image patches, matched as ``*_{image_suffix}.tiff``.
    nodata : float | None, optional
        Pixel value to exclude from the statistics. Patches written by
        :func:`generate_train_val_patches` are zero-padded outside valid data
        (via ``band_scaling``), so this defaults to ``0.0``. Pass None to
        include every pixel.
    show_progress : bool, optional
        Whether to display a progress bar while scanning patches.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Per-band mean and standard deviation, each of shape ``(bands,)``.
    """
    if isinstance(patch_dirs, (str, Path)):
        patch_dirs = [patch_dirs]

    patch_paths = [
        p for d in patch_dirs for p in sorted(Path(d).glob(f'*_{image_suffix}.tiff'))
    ]
    if not patch_paths:
        raise FileNotFoundError(f"No '*_{image_suffix}.tiff' patches found in {patch_dirs}")

    with rst.open(patch_paths[0]) as src:
        n_bands = src.count

    total_sum = np.zeros(n_bands, dtype=np.float64)
    total_sumsq = np.zeros(n_bands, dtype=np.float64)
    total_count = np.zeros(n_bands, dtype=np.int64)

    iterator = tqdm(patch_paths, desc='Computing patch stats') if show_progress else patch_paths
    for path in iterator:
        with rst.open(path) as src:
            arr = src.read().astype(np.float64)

        for b in range(n_bands):
            band = arr[b]
            valid = band[band != nodata] if nodata is not None else band.ravel()
            total_sum[b] += valid.sum()
            total_sumsq[b] += np.square(valid).sum()
            total_count[b] += valid.size

    mean = total_sum / total_count
    std = np.sqrt(total_sumsq / total_count - mean**2)

    return mean, std