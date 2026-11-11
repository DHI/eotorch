import inspect
from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyproj
import shapely
import shapely.geometry
import shapely.ops
from rasterio.windows import Window


def transform_index(
    index: gpd.GeoDataFrame, current_crs: str | pyproj.CRS, new_crs: str | pyproj.CRS
) -> gpd.GeoDataFrame:
    """
    Transform a Geopandas GeoDataFrame index to a new coordinate reference system (CRS).

    Parameters:
        index (gpd.GeoDataFrame):
            The Geopandas GeoDataFrame index to transform.
        current_crs (str or pyproj.CRS):
            The current coordinate reference system.
        new_crs (str or pyproj.CRS):
            The new coordinate reference system.

    Returns:
        gpd.GeoDataFrame:
            A new Geopandas GeoDataFrame with transformed coordinates.
    """
    if isinstance(current_crs, str):
        current_crs = pyproj.CRS(current_crs)

    if isinstance(new_crs, str):
        new_crs = pyproj.CRS(new_crs)

    # Create a copy of the index and transform to new CRS
    new_index = index.copy()
    new_index = new_index.to_crs(new_crs)

    return new_index


def window_to_np_idc(window: Window) -> tuple[slice, slice]:
    """Convert a window to numpy array indices."""
    return (
        slice(int(round(window.row_off)), int(round(window.row_off + window.height))),
        slice(int(round(window.col_off)), int(round(window.col_off + window.width))),
    )


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
                try:
                    if not bounds == dim_bounds[-1]:
                        dim_bounds.append(bounds)
                except IndexError:
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
    for width in dims[1]:
        for height in dims[0]:
            if as_windows:
                slices.append(Window.from_slices(slice(*height), slice(*width)))
            else:
                slices.append((slice(*height), slice(*width)))

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
    bbox,
    bbox_crs: str | pyproj.CRS,
    target_crs: str | pyproj.CRS = "EPSG:4326",
):
    """
    Convert a list of bound coordinates, tuple, Tensor, or a BoundingBox object to a Shapely Polygon.

    Parameters:
        bbox (list, tuple, Tensor, or BoundingBox):
            The bounding box to convert.
            - If a Tensor: expected format [xmin, xmax, xres, ymin, ymax, yres, tmin, tmax, tres]
              (torchgeo v0.9 _slice_to_tensor format)
            - If a list: should be in the format [minx, maxx, miny, maxy] (torchgeo format)
            - If a tuple: should be in the format (minx, miny, maxx, maxy) (geopandas bounds format)
            - If a BoundingBox object: should have the attributes minx, maxx, miny, maxy.
            May contain more than 4 values (e.g. mint, maxt).
        bbox_crs (str or pyproj.CRS):
            The coordinate reference system of the input bounding box.
        target_crs (str or pyproj.CRS, optional):
            The target coordinate reference system. Defaults to "EPSG:4326".

    Returns:
        shapely.geometry.Polygon:
            The converted Shapely Polygon.
    """
    import torch

    if isinstance(target_crs, str):
        target_crs = pyproj.CRS(target_crs)

    if isinstance(bbox_crs, str):
        bbox_crs = pyproj.CRS(bbox_crs)

    # Create a shapely polygon from the bbox coordinates
    if isinstance(bbox, torch.Tensor):
        # torchgeo v0.9 _slice_to_tensor format: [xmin, xmax, xres, ymin, ymax, yres, tmin, tmax, tres]
        polygon = shapely.geometry.box(
            minx=bbox[0].item(),
            miny=bbox[3].item(),
            maxx=bbox[1].item(),
            maxy=bbox[4].item(),
        )
    elif isinstance(bbox, (list, tuple, np.ndarray)):
        # Handle both list and tuple formats
        # For geopandas geometry.bounds: (minx, miny, maxx, maxy)
        # For torchgeo list format: [minx, maxx, miny, maxy]
        if len(bbox) >= 4:
            if isinstance(bbox, tuple):
                # geopandas bounds format: (minx, miny, maxx, maxy)
                polygon = shapely.geometry.box(
                    minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3]
                )
            else:
                # torchgeo list format: [minx, maxx, miny, maxy]
                polygon = shapely.geometry.box(
                    minx=bbox[0], miny=bbox[2], maxx=bbox[1], maxy=bbox[3]
                )
        else:
            raise ValueError(
                f"Bounding box must have at least 4 coordinates, got {len(bbox)}"
            )
    else:
        # BoundingBox object with attributes
        polygon = shapely.geometry.box(
            minx=bbox.minx, miny=bbox.miny, maxx=bbox.maxx, maxy=bbox.maxy
        )

    # If the CRS is the same, no need to transform
    if bbox_crs == target_crs:
        return polygon

    # Create transformer for the coordinate conversion
    transformer = pyproj.Transformer.from_crs(bbox_crs, target_crs, always_xy=True)

    # Transform the polygon to the target CRS
    transformed_polygon = shapely.ops.transform(transformer.transform, polygon)

    return transformed_polygon


def convert_bounds(bbox, invert_y=False):
    """
    Helper method for changing bounding box representation to leaflet notation

    ``(lon1, lat1, lon2, lat2) -> ((lat1, lon1), (lat2, lon2))``
    """
    x1, y1, x2, y2 = bbox
    if invert_y:
        y1, y2 = y2, y1
    return ((y1, x1), (y2, x2))
