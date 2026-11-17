import inspect
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyproj
import shapely
import shapely.geometry
import shapely.ops
from rasterio.windows import Window


def window_to_np_idc(window: Window) -> tuple[slice, slice]:
    """Convert a window to numpy array indices."""
    return (
        slice(round(window.row_off), round(window.row_off + window.height)),
        slice(round(window.col_off), round(window.col_off + window.width)),
    )


def is_nodata(array: np.ndarray, nodata_value: float | None) -> np.ndarray:
    """Boolean array of the pixels equal to the nodata value, which may be NaN or None."""
    if nodata_value is None:
        return np.zeros(array.shape, dtype=bool)
    if np.isnan(nodata_value):
        # NaN is never equal to itself, so it is found with isnan
        return np.isnan(array)
    return array == nodata_value


def get_init_args(cls):
    """Returns the argument names of a class's __init__ method, excluding 'self'."""
    signature = inspect.signature(cls.__init__)  # Get the signature of __init__
    return [
        param.name for param in signature.parameters.values() if param.name != "self"
    ]


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
                # Moved back to end at the edge, but not before the start of the array
                bounds = (max(dim - size, 0), dim) if bounds[1] > dim else bounds
                if not dim_bounds or bounds != dim_bounds[-1]:
                    dim_bounds.append(bounds)
            elif (b + (size // stride)) < dim:
                bounds = (b, b + size)
                # Moved back to end at the edge, but not before the start of the array
                bounds = (max(dim - size, 0), dim) if bounds[1] > dim else bounds
                dim_bounds.append(bounds)

        out_bounds = [(i + offset, j + offset) for i, j in dim_bounds[:-1]]
        out_bounds.append(dim_bounds[-1])
        dims.append(out_bounds)

    slices = []
    for cols in dims[1]:
        for rows in dims[0]:
            if as_windows:
                slices.append(Window.from_slices(slice(*rows), slice(*cols)))
            else:
                slices.append((slice(*rows), slice(*cols)))

    return slices


def _format_filepaths(filepaths):
    """Format filepaths for display in plot titles."""
    if not filepaths:
        return ""

    # If only one filepath, show it directly
    if len(filepaths) == 1:
        path = Path(filepaths[0])
        # Show truncated path with at least two parent directories if path is too long
        if len(str(path)) > 40:
            # Check if it's in home directory or current working directory
            home_dir = Path.home()
            cwd = Path.cwd()

            if home_dir == path.parent or cwd == path.parent:
                return f".../{path.name}"

            # Include at least two parent directories
            parent_parts = list(path.parents)
            if len(parent_parts) >= 2:
                return f".../{parent_parts[1].name}/{parent_parts[0].name}/{path.name}"
            elif len(parent_parts) == 1:
                return f".../{parent_parts[0].name}/{path.name}"
            else:
                return f".../{path.name}"
        return str(path)

    # Convert all paths to Path objects
    paths = [Path(p) for p in filepaths]

    # Find common prefix to avoid repetition
    try:
        common_path = Path(Path(*paths[0].parts[:1]).anchor)
        for i in range(1, len(paths[0].parts)):
            potential_common = Path(*paths[0].parts[: i + 1])
            if all(str(potential_common) in str(p) for p in paths):
                common_path = potential_common
            else:
                break
    except (IndexError, ValueError):
        common_path = None

    if common_path and len(str(common_path)) > 10:
        formatted = f"{common_path}/\n"
        # Show only the unique parts after the common prefix
        unique_parts = [str(p).replace(str(common_path), "").lstrip("/") for p in paths]

        # If there are too many files, limit display
        if len(unique_parts) > 3:
            formatted += "\n".join(unique_parts[:2])
            formatted += f"\n...and {len(unique_parts) - 2} more files"
        else:
            formatted += "\n".join(unique_parts)
        return formatted

    # If no common prefix or too short, show first few and indicate total count
    if len(paths) > 3:
        return (
            "\n".join([str(p) for p in paths[:2]])
            + f"\n...and {len(paths) - 2} more files"
        )

    return "\n".join([str(p) for p in paths])


def torchgeo_bb_to_shapely(
    bounds,
    bbox_crs: str | pyproj.CRS,
    target_crs: str | pyproj.CRS = "EPSG:4326",
) -> shapely.geometry.Polygon:
    """
    Box of the given bounds as a Shapely Polygon in target_crs.

    Parameters:
        bounds (tuple or Tensor):
            Bounds as (minx, miny, maxx, maxy), e.g. of a shapely geometry, or the
            "bounds" Tensor of a TorchGeo sample.
        bbox_crs (str or pyproj.CRS):
            The coordinate reference system of the bounds.
        target_crs (str or pyproj.CRS, optional):
            The target coordinate reference system. Defaults to "EPSG:4326".
    """
    import torch

    if isinstance(bounds, torch.Tensor):
        # TorchGeo samples: [xmin, xmax, xres, ymin, ymax, yres, tmin, tmax, tres]
        xmin, xmax, _, ymin, ymax = bounds[:5].tolist()
        bounds = (xmin, ymin, xmax, ymax)
    polygon = shapely.geometry.box(*bounds)

    if pyproj.CRS(bbox_crs) == pyproj.CRS(target_crs):
        return polygon
    transformer = pyproj.Transformer.from_crs(bbox_crs, target_crs, always_xy=True)
    return shapely.ops.transform(transformer.transform, polygon)


def convert_bounds(bbox, invert_y=False):
    """
    Helper method for changing bounding box representation to leaflet notation

    ``(lon1, lat1, lon2, lat2) -> ((lat1, lon1), (lat2, lon2))``
    """
    x1, y1, x2, y2 = bbox
    if invert_y:
        y1, y2 = y2, y1
    return ((y1, x1), (y2, x2))
