import os
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import rasterio as rst
from tqdm import tqdm
from rasterio.transform import from_origin


def _should_skip_patch(
    image: np.ndarray,
    labels: np.ndarray,
    slice_obj: tuple[slice, slice],
    patch_size: int,
    meta: dict[str, Any],
    value_threshold: float | int | None,
    empty_threshold: float | None,
) -> bool:
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
    empty_threshold : float | None
        Skip patch when empty label ratio (label == 0) exceeds this threshold.
    
    Returns:
        bool: True if patch should be skipped, False otherwise
    """
    # Skip if all nodata or all zeros
    if (image[:, *slice_obj] == meta['nodata']).all() or (image[:, *slice_obj] == 0).all():
        return True
    
    if value_threshold is not None:
        if (labels[slice_obj] <= value_threshold).all():
            return True
    
    elif empty_threshold is not None:
        empty_ratio = labels[slice_obj][labels[slice_obj]==0].size / patch_size**2
        if empty_ratio > empty_threshold:
            return True
    
    return False


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
    feature_name = f'{img_stem}_{patch_index}_feature.tiff'
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
    stride: int = 1,
    overlap: bool = True,
    empty_threshold: float | None = 0.2,
    value_threshold: float | int | None = None,
    show_progress: bool = True,
) -> None:
    """
    Generate and save patches from image and label files.
    
    Parameters
    ----------
    img_path : str | Path
        Path to the normalized image raster.
    label_path : str | Path
        Path to the label raster.
    out_dir : str | Path
        Output directory where patch files are written.
    patch_size : int, default=128
        Patch width/height in pixels.
    stride : int, default=1
        The stride for moving the patch window.
    overlap : bool, default=True
        Whether to allow overlapping patches.
    empty_threshold : float | None, default=0.2
        Maximum allowed empty-label ratio before skipping a patch.
    value_threshold : float | int | None, default=None
        Skip patch when all label values are <= this threshold.
    show_progress : bool, default=True
        Whether to display a progress bar while writing patches.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with rst.open(img_path) as img_src, rst.open(label_path) as label_src:
        image = img_src.read()
        labels = label_src.read(indexes=1)
        shape = img_src.shape
        meta = img_src.meta.copy()
    
    slices = discrete_patch(*shape, size=patch_size, stride=stride, overlap=overlap)
    iterator = tqdm(enumerate(slices), total=len(slices)) if show_progress else enumerate(slices)
    
    for n, s in iterator:
        # Skip patch if it doesn't meet criteria
        if _should_skip_patch(image, labels, s, patch_size, meta, value_threshold, empty_threshold):
            continue
        
        _write_patch(image, labels, s, n, Path(img_path).stem, out_dir, meta, patch_size)


def discrete_patch(
    height: int,
    width: int,
    size: int = 256,
    stride: int = 1,
    overlap: bool = True,
) -> list[tuple[slice, slice]]:
    """
    Generates discrete slices of a 2D array of a given dimension (size*size).

    Parameters:
        height (int): 
            Height of array.
        width (int): 
            Width of array.
        size (int, optional): 
            Size of slices/subsets/patches. Defaults to 256.
        stride (int, optional): 
            The overlap between patches. The stride is defined as the number of 
            windows that will pass over a subset, i.e. a `stride` of 1 will have no overlap,
            while a stride of 2 will result in 50% overlap between two patches in one dimension
            for half the slices. Defaults to 1.
        overlap (bool, optional): 
            Trims edges of an array if the height/width is not divisible by the size. 
            Setting this to False with an array not equally divisible by `size` will result in 
            irregular slices. Defaults to True.
        offset (int, optional): 
            Shifts the slices (offset, offset) `offset` pixels diagonally towards the bottom right. 
            Defaults to 0.

    Returns:
        list[tuple[slice, slice]]:
            Sequence of 2D row/column slice tuples.
    """    
    dims = []
    for dim in (height, width):
        dim_bounds = []
        for b in range(0, dim, size // stride):
            if (overlap and stride==1) or (stride>1):
                bounds = (b, b+size)
                bounds = (dim-size, dim) if bounds[1] > dim else bounds
                try:
                    if not bounds == dim_bounds[-1]:
                        dim_bounds.append(bounds)
                except IndexError:
                    dim_bounds.append(bounds)
            elif (b+(size//stride)) < dim:
                bounds = (b, b+size)
                bounds = (dim-size, dim) if bounds[1] > dim else bounds
                dim_bounds.append(bounds)
                
        out_bounds = [(i,j) for i,j in dim_bounds[:-1]]
        out_bounds.append(dim_bounds[-1])
        dims.append(out_bounds)

    slices = []
    for width in dims[1]:
        for height in dims[0]:
            slices.append((slice(*height), slice(*width)))
            
    return slices


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

    Parameters:
        image (np.ndarray): 
            Input image features.
        x (float): 
            x-coordinate of corner pixel.
        y (float): 
            y-coordinate of corner pixel.
        meta (dict[str, Any]): 
            Original metadata.
        patch_size (int): 
            Patch width/height in pixels.
        dtype (str | np.dtype[Any] | None, optional): 
            dtype of the output. If None, infers the dtype from the metadata. Defaults to None.

    Returns:
        dict[str, Any]: 
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

    Parameters:
        wildcard (str): 
            Glob wildcard used to match patch files.
        patch_dir (str | Path): 
            Patch directory.
    """    
    img_paths = glob(os.path.join(patch_dir, f'{wildcard}'))
    for img_path in img_paths:
        os.remove(img_path)