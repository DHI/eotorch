import os
from pathlib import Path
from typing import Sequence

from rasterio.windows import Window


def slice_image(
    height: int,
    width: int,
    size: int = 256,
    stride: int = 1,
    overlap: bool = True,
    offset: int = 0,
    as_windows: bool = False,
) -> Sequence[slice]:
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
        Sequence[slice]:
            Sequence of 2D slices or if specified, rasterio Window objects.
    """
    assert offset < size, "Offset must be less than size."
    dims = []
    for dim in (height, width):
        dim_bounds = []
        for b in range(0, dim, size // stride):
            if (overlap and stride == 1) or (stride > 1):
                bounds = (b, b + size)
                bounds = (dim - size, dim) if bounds[1] > dim else bounds
                try:
                    if not bounds == dim_bounds[-1]:
                        dim_bounds.append(bounds)
                except IndexError:
                    dim_bounds.append(bounds)
            elif (b + (size // stride)) < dim:
                bounds = (b, b + size)
                bounds = (dim - size, dim) if bounds[1] > dim else bounds
                dim_bounds.append(bounds)

        out_bounds = [(i + offset, j + offset) for i, j in dim_bounds[:-1]]
        out_bounds.append(dim_bounds[-1])
        dims.append(out_bounds)

    slices = []
    for width in dims[1]:
        for height in dims[0]:
            if as_windows:
                slices.append(Window.from_slices(slice(*height), slice(*width)))
            else:
                slices.append((slice(*height), slice(*width)))

    return slices


def get_outpath(img_path, out_dir, out_path, subfolder=None, suffix=""):
    scene = Path(img_path).stem
    if subfolder is None:
        out_dir = os.getcwd()
    else:
        out_dir = os.path.join(os.getcwd(), subfolder) if out_dir is None else out_dir
        Path(out_dir).mkdir(exist_ok=True)
    out_path = (
        os.path.join(out_dir, f"{scene}{suffix}.tif") if out_path is None else out_path
    )
    return out_path
