from __future__ import annotations

import copy
import functools
import logging
import os
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rst
import torch
from matplotlib import cm
from pyproj import CRS
from rasterio.plot import show
from shapely import Polygon
from shapely.geometry import box as shapely_box
from torchgeo.datasets import (
    IntersectionDataset,
    RasterDataset,
    RGBBandsMissingError,
)
from torchgeo.datasets.utils import GeoSlice, array_to_tensor

from eotorch.bandindex import BAND_INDEX
from eotorch.inference.inference_utils import prediction_to_numpy
from eotorch.plot import label_map, plot_dataset_index, plot_numpy_array, plot_samples
from eotorch.utils import _format_filepaths

if TYPE_CHECKING:
    from torch import Tensor


def split_raster_into_dense_areas(
    file_path: str | Path,
    polygon: Polygon,
    min_length: int,
    nodata_value: int | float,
    polygon_crs: CRS = None,
) -> list[Polygon]:
    """Split a raster into multiple areas with higher label density.

    Args:
        file_path: Path to the raster file
        polygon: Shapely polygon defining the area to split
        min_length: Minimum length (in pixels) of the shortest side of polygon box
        nodata_value: Value indicating no data in the label file
        polygon_crs: CRS of the input polygon. If provided, raster will be warped to this CRS
                     for processing, and results will be in polygon CRS

    Returns:
        List of polygons representing areas with higher label density (in polygon CRS)
    """
    with rst.open(file_path) as src:
        # Use WarpedVRT to work in polygon CRS if provided, otherwise use raster CRS
        target_crs = polygon_crs if polygon_crs is not None else src.crs

        if src.crs != target_crs:
            # Use WarpedVRT to read raster data in the target CRS
            vrt = rst.vrt.WarpedVRT(src, crs=target_crs)
            transform = vrt.transform
            bounds = vrt.bounds
            width = vrt.width
            height = vrt.height
            data_source = vrt
        else:
            # Use original raster if CRS matches
            transform = src.transform
            bounds = src.bounds
            width = src.width
            height = src.height
            data_source = src

        # Convert polygon to pixel coordinates (now in the same CRS as raster data)
        minx, miny, maxx, maxy = polygon.bounds

        # Clip to raster bounds
        minx = max(minx, bounds.left)
        miny = max(miny, bounds.bottom)
        maxx = min(maxx, bounds.right)
        maxy = min(maxy, bounds.top)

        # Convert to pixel coordinates
        col_min, row_max = ~transform * (minx, miny)
        col_max, row_min = ~transform * (maxx, maxy)

        # Ensure integer pixel coordinates and fix ordering
        col_min, col_max = int(min(col_min, col_max)), int(max(col_min, col_max))
        row_min, row_max = int(min(row_min, row_max)), int(max(row_min, row_max))

        # Ensure we stay within raster bounds
        col_min = max(0, col_min)
        col_max = min(width, col_max)
        row_min = max(0, row_min)
        row_max = min(height, row_max)

        # Check if we have a valid window
        if col_min >= col_max or row_min >= row_max:
            return []

        # Read the data for the area
        window = rst.windows.Window(
            col_min, row_min, col_max - col_min, row_max - row_min
        )
        data = data_source.read(1, window=window)

        # Find areas with labels (not nodata)
        labeled_mask = data != nodata_value

        # If no labels found, return empty list
        if not labeled_mask.any():
            return []

        # Find bounding boxes of labeled regions
        polygons = []

        # Simple approach: create rectangles around labeled pixels
        labeled_rows, labeled_cols = np.where(labeled_mask)

        if len(labeled_rows) == 0:
            return []

        # Group nearby labeled pixels into rectangles
        # Create a simple grid-based approach
        min_row, max_row = labeled_rows.min(), labeled_rows.max()
        min_col, max_col = labeled_cols.min(), labeled_cols.max()

        # Calculate grid size based on min_length
        height = max_row - min_row + 1
        width = max_col - min_col + 1

        # Determine how many splits are needed
        rows_splits = max(1, height // min_length)
        cols_splits = max(1, width // min_length)

        row_step = height // rows_splits
        col_step = width // cols_splits

        # Create a grid of cells and check which ones contain labels
        labeled_cells = np.zeros((rows_splits, cols_splits), dtype=bool)
        cell_bounds = {}

        for i in range(rows_splits):
            for j in range(cols_splits):
                # Calculate grid cell boundaries
                cell_row_min = min_row + i * row_step
                cell_row_max = (
                    min_row + (i + 1) * row_step if i < rows_splits - 1 else max_row + 1
                )
                cell_col_min = min_col + j * col_step
                cell_col_max = (
                    min_col + (j + 1) * col_step if j < cols_splits - 1 else max_col + 1
                )

                # Check if this cell contains any labels
                cell_mask = labeled_mask[
                    cell_row_min:cell_row_max, cell_col_min:cell_col_max
                ]
                if cell_mask.any():
                    labeled_cells[i, j] = True
                    cell_bounds[(i, j)] = (
                        cell_row_min,
                        cell_row_max,
                        cell_col_min,
                        cell_col_max,
                    )

        # Find rectangular groups of neighboring cells
        visited = np.zeros_like(labeled_cells, dtype=bool)

        for i in range(rows_splits):
            for j in range(cols_splits):
                if labeled_cells[i, j] and not visited[i, j]:
                    # Find the largest rectangle starting from this cell
                    rect_cells = _find_largest_rectangle(labeled_cells, visited, i, j)

                    if rect_cells:
                        # Calculate bounds for the merged rectangle
                        min_i = min(cell[0] for cell in rect_cells)
                        max_i = max(cell[0] for cell in rect_cells)
                        min_j = min(cell[1] for cell in rect_cells)
                        max_j = max(cell[1] for cell in rect_cells)

                        # Get pixel bounds
                        cell_row_min, _, cell_col_min, _ = cell_bounds[(min_i, min_j)]
                        _, cell_row_max, _, cell_col_max = cell_bounds[(max_i, max_j)]

                        # Convert to geographic coordinates
                        geo_minx, geo_maxy = transform * (
                            col_min + cell_col_min,
                            row_min + cell_row_min,
                        )
                        geo_maxx, geo_miny = transform * (
                            col_min + cell_col_max,
                            row_min + cell_row_max,
                        )

                        # Create polygon
                        cell_polygon = shapely_box(
                            geo_minx, geo_miny, geo_maxx, geo_maxy
                        )

                        # Intersect with original polygon to ensure we stay within bounds
                        intersected = polygon.intersection(cell_polygon)
                        if not intersected.is_empty and hasattr(intersected, "bounds"):
                            polygons.append(intersected)

        # If no grid cells had labels, create one rectangle around all labeled pixels
        if not polygons:
            geo_minx, geo_maxy = transform * (col_min + min_col, row_min + min_row)
            geo_maxx, geo_miny = transform * (
                col_min + max_col + 1,
                row_min + max_row + 1,
            )
            cell_polygon = shapely_box(geo_minx, geo_miny, geo_maxx, geo_maxy)
            intersected = polygon.intersection(cell_polygon)
            if not intersected.is_empty:
                polygons.append(intersected)

        # Close VRT if we created one
        if src.crs != target_crs:
            data_source.close()

        return polygons


def _find_largest_rectangle(
    labeled_cells: np.ndarray, visited: np.ndarray, start_i: int, start_j: int
) -> list[tuple[int, int]]:
    """Find the largest rectangle of neighboring labeled cells starting from a given position.

    Args:
        labeled_cells: 2D boolean array indicating which cells contain labels
        visited: 2D boolean array tracking which cells have been processed
        start_i: Starting row index
        start_j: Starting column index

    Returns:
        List of (i, j) tuples representing the cells in the rectangle
    """
    rows, cols = labeled_cells.shape

    # Find the maximum width rectangle starting from this row
    max_area = 0
    best_rect = []

    # Try different heights starting from current position
    for height in range(1, rows - start_i + 1):
        # Check if we can extend to this height
        can_extend = True
        for h in range(height):
            if (
                start_i + h >= rows
                or not labeled_cells[start_i + h, start_j]
                or visited[start_i + h, start_j]
            ):
                can_extend = False
                break

        if not can_extend:
            break

        # Find maximum width for this height
        width = 0
        for w in range(cols - start_j):
            # Check if all cells in this column for the current height are labeled and unvisited
            valid_column = True
            for h in range(height):
                if (
                    start_i + h >= rows
                    or start_j + w >= cols
                    or not labeled_cells[start_i + h, start_j + w]
                    or visited[start_i + h, start_j + w]
                ):
                    valid_column = False
                    break

            if valid_column:
                width += 1
            else:
                break

        # Check if this rectangle is better than previous ones
        area = height * width
        if area > max_area:
            max_area = area
            best_rect = [
                (start_i + h, start_j + w) for h in range(height) for w in range(width)
            ]

    # Mark all cells in the best rectangle as visited
    for i, j in best_rect:
        visited[i, j] = True

    return best_rect


class SamplePlotMixin:
    """Mixin class for datasets that can plot samples."""

    def plot_samples(
        self, n: int = 3, patch_size: int = 256, show_filepaths: bool = False
    ):
        """Plot n random samples, leaving out samples without data."""
        return plot_samples(
            self, n=n, patch_size=patch_size, show_filepaths=show_filepaths
        )


class CustomCacheRasterDataset(RasterDataset, SamplePlotMixin):
    """
    Implementing RasterDatset with customisable cache size, as it is hardcoded to 128 in torchgeo's RasterDataset.
    """

    def __init__(
        self,
        paths: str | Path | Iterable[str] | Iterable[Path] = "data",
        crs: CRS | None = None,
        res: float | None = None,
        bands: Sequence[str] | None = None,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cache_size: int = 20,
        filename_glob: str | None = None,
        filename_regex: str | None = None,
        date_format: str | None = None,
        all_bands: Sequence[str] | None = None,
        rgb_bands: Sequence[str] | None = None,
        separate_files: bool | None = None,
    ) -> None:
        # TorchGeo defines these as class attributes, to be overridden by subclassing.
        # Set on the instance instead, so each dataset keeps its own. None keeps the
        # class attribute, so subclassing still works.
        settings = {
            "filename_glob": filename_glob,
            "filename_regex": filename_regex,
            "date_format": date_format,
            "all_bands": all_bands,
            "rgb_bands": rgb_bands,
            "separate_files": separate_files,
        }
        for name, value in settings.items():
            if value is not None:
                setattr(self, name, value)
        super().__init__(paths, crs, res, bands, transforms, cache=cache_size > 0)
        self.cache_size = cache_size
        # If True, a query matching the footprint of a file exactly reads only that
        # file, instead of merging every file it touches. Used for pre-patched data,
        # see RasterDataModule(patch_size=None).
        self.single_file_reads = False
        self._init_cache()
        # In order to avoid spamming "Skipping source" Log Messages in Rasterio Merge
        logging.getLogger("rasterio").setLevel(logging.WARNING)

    def _init_cache(self) -> None:
        self._cached_load_warp_file = functools.lru_cache(maxsize=self.cache_size)(
            self._load_warp_file
        )
        # Lookup table of file footprints, built on first use and whenever the index changes
        self._footprints: dict[tuple[int, ...], list[tuple[pd.Interval, str]]] = {}
        self._footprints_index = None

    def __getstate__(self) -> dict[str, Any]:
        # DataLoader workers started with "spawn" (the only option on Windows) pickle the
        # dataset. The per-instance lru_cache can't be pickled, so rebuild it afterwards.
        state = self.__dict__.copy()
        for cache in ("_cached_load_warp_file", "_footprints", "_footprints_index"):
            del state[cache]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._init_cache()

    def __getitem__(self, index: GeoSlice) -> dict[str, Any]:
        if self.single_file_reads and not self.separate_files:
            x, y, t = self._disambiguate_slice(index)
            filepath = self._file_with_footprint(x, y, t)
            if filepath is not None:
                return self._read_file(filepath, (x, y, t))
        return super().__getitem__(index)

    def _pixel_keys(self, bounds: np.ndarray) -> list[tuple[int, ...]]:
        """Lookup keys for (minx, miny, maxx, maxy) rows of bounds."""
        xres, yres = self.res
        x0, y0 = self._footprints_origin
        # In whole pixels, float noise in the coordinates can't break exact dict lookups
        pixels = np.rint((bounds - [x0, y0, x0, y0]) / [xres, yres, xres, yres])
        return [tuple(row) for row in pixels.astype(int).tolist()]

    def _file_with_footprint(self, x: slice, y: slice, t: slice) -> str | None:
        """The file whose footprint matches the query to within half a pixel, if any."""
        if self._footprints_index is not self.index:
            bounds = self.index.geometry.bounds.to_numpy()
            # Pixels are counted from the corner of a file, as the coordinates of a
            # grid aren't necessarily multiples of its resolution
            self._footprints_origin = bounds[0, :2]
            self._footprints = defaultdict(list)
            for key, interval, filepath in zip(
                self._pixel_keys(bounds), self.index.index, self.index.filepath
            ):
                self._footprints[key].append((interval, filepath))
            self._footprints_index = self.index

        (key,) = self._pixel_keys(np.array([[x.start, y.start, x.stop, y.stop]]))
        query_interval = pd.Interval(t.start, t.stop)
        for interval, filepath in self._footprints.get(key, []):
            if interval.overlaps(query_interval):
                return filepath
        return None

    def _read_file(self, filepath: str, index: GeoSlice) -> dict[str, Any]:
        """Same as RasterDataset.__getitem__, but reading a single file."""
        x, y, _ = index
        src = (
            self._cached_load_warp_file(filepath, self.crs)
            if self.cache
            else self._load_warp_file(filepath, self.crs)
        )
        shape = round((y.stop - y.start) / y.step), round((x.stop - x.start) / x.step)
        if src.shape == shape:
            # Already on the grid of the query, so skip the slower resampling of a merge
            data = array_to_tensor(src.read(self.band_indexes))
        else:
            data = self._merge_or_stack([filepath], index, self.band_indexes)
        transform = rst.transform.from_origin(x.start, y.stop, x.step, y.step)
        sample = {
            "bounds": self._slice_to_tensor(index),
            "transform": torch.tensor(transform),
        }

        data = data.to(self.dtype)
        if self.is_image:
            sample["image"] = data
        else:
            sample["mask"] = data.squeeze(-3)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def filepaths_at(self, bounds: Tensor) -> list[str]:
        """Files spatially overlapping a sample's "bounds" tensor."""
        xmin, xmax, _, ymin, ymax = bounds.tolist()[:5]
        return self.index.cx[xmin:xmax, ymin:ymax].filepath.tolist()

    def _repr_html_(self):
        print(
            f"Dataset containing {len(self)} items, crs: {self.crs.name}, res: {self.res}"
        )
        return plot_dataset_index(self)._repr_html_()

    def __or__(self, other: CustomCacheRasterDataset) -> CustomCacheRasterDataset:
        """
        Create a new CustomCacheRasterDataset that is the union of this dataset and another.

        The two datasets must have compatible configurations for resolution, selected bands,
        definition of all bands, is_image status, and separate_files status.
        The CRS can differ; the new dataset will adopt the CRS of the left operand ('self').
        The 'other' dataset's index will be transformed to 'self.crs' using GeoDataset's
        built-in CRS transformation logic if their CRSs differ.

        The new dataset inherits all other settings, e.g. transforms and cache_size, from 'self'.
        The index of the new dataset is a combination of the indices from 'self' and
        the (potentially transformed) index from 'other'.

        Args:
            other: Another CustomCacheRasterDataset instance to union with.

        Returns:
            A new CustomCacheRasterDataset instance representing the union.

        Raises:
            TypeError: If 'other' is not a CustomCacheRasterDataset instance.
            ValueError: If the datasets have incompatible data-defining configurations
                        or if the union results in an empty dataset.
        """
        if not isinstance(other, CustomCacheRasterDataset):
            raise TypeError(
                f"Can only union with another CustomCacheRasterDataset, not {type(other).__name__}"
            )

        # --- Compatibility checks (excluding CRS) ---
        # Attributes to check for equality, with user-friendly names for error messages
        compatibility_attrs = {
            "bands": "Selected bands (bands attribute)",
            "all_bands": "'all_bands' attribute",
            "is_image": "'is_image' attribute",
            "separate_files": "'separate_files' attribute",
            "reduce_zero_label": "'reduce_zero_label' attribute",
        }

        for attr_name, display_name in compatibility_attrs.items():
            self_attr = getattr(self, attr_name, None)
            other_attr = getattr(other, attr_name, None)
            if self_attr != other_attr:
                raise ValueError(
                    f"{display_name} must be the same for union. "
                    f"Self: {self_attr}, Other: {other_attr}"
                )

        # Deepcopy 'other' and transform its CRS if necessary using GeoDataset's setter
        other_transformed = copy.deepcopy(other)
        if other_transformed.crs != self.crs:
            other_transformed.crs = self.crs  # This invokes GeoDataset.crs.
        if other_transformed.res != self.res:
            other_transformed.res = self.res

        # --- Construct the items for the new_dataset.index ---
        # Combine indexes from both datasets, keeping their time intervals
        combined_index = pd.concat([self.index, other_transformed.index])

        if combined_index.empty:
            raise ValueError(
                "Union results in an empty dataset based on merged index items."
            )

        def _ensure_list_of_paths(
            paths_attr: str | os.PathLike | Iterable[str] | Iterable[os.PathLike],
        ):
            if isinstance(paths_attr, (str, os.PathLike)):
                return [paths_attr]
            if isinstance(paths_attr, Iterable) and not isinstance(
                paths_attr, (str, bytes)
            ):
                return list(paths_attr)
            raise TypeError(f"Unsupported type for paths attribute: {type(paths_attr)}")

        self_paths_list = _ensure_list_of_paths(self.paths)
        other_paths_list = _ensure_list_of_paths(other.paths)
        # Ensure paths in constructor_paths are strings, as expected by some internal logic.
        combined_constructor_paths = sorted(
            list(set(map(str, self_paths_list + other_paths_list)))
        )

        # A copy keeps all settings of 'self', only the files change
        new_dataset = copy.deepcopy(self)
        new_dataset.paths = combined_constructor_paths
        new_dataset.index = combined_index
        return new_dataset


class PlottableImageDataset(CustomCacheRasterDataset):
    def plot(self, sample, ax=None, **kwargs):
        predictions = sample.get("prediction")

        if ax is None:
            n_cols = 2 if predictions is not None else 1
            fig, ax = plt.subplots(1, n_cols, figsize=(n_cols * 4, 4))
        image_ax = ax if predictions is None else ax[0]

        # The sample only holds the selected bands, in the order of self.bands
        try:
            rgb_indices = [self.bands.index(band) for band in self.rgb_bands]
        except ValueError as e:
            raise RGBBandsMissingError() from e

        image = sample["image"][rgb_indices]

        if show_filepaths := kwargs.pop("show_filepaths", False):
            filepaths = self.filepaths_at(sample["bounds"])
            formatted_paths = _format_filepaths(filepaths)
            image_ax.set_title(formatted_paths, fontsize=8, wrap=True)

        show(
            image.numpy(),
            ax=image_ax,
            adjust=True,
            title="Image" if not show_filepaths else None,
            **kwargs,
        )
        if predictions is not None:
            plot_numpy_array(
                array=prediction_to_numpy(predictions),
                ax=ax[-1],
                title="Prediction",
                **kwargs,
            )

        plt.tight_layout()


class PlottableLabelDataset(CustomCacheRasterDataset):
    """Base class of the label datasets for classification and regression."""

    is_image = False
    nodata_value: float = 0
    class_mapping: dict | None = None
    colormap = cm.viridis

    def values_as_in_files(self, values: Tensor) -> np.ndarray:
        """Label values of a mask or prediction as they are stored in the label files."""
        return values.numpy()

    def plot(self, sample, ax=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots()
        if show_filepaths := kwargs.pop("show_filepaths", False):
            filepaths = self.filepaths_at(sample["bounds"])
            formatted_paths = _format_filepaths(filepaths)
            ax.set_title(formatted_paths, fontsize=8, wrap=True)

        plot_numpy_array(
            array=self.values_as_in_files(sample["mask"]),
            ax=ax,
            class_mapping=self.class_mapping,
            colormap=self.colormap,
            nodata_value=self.nodata_value,
            title="Label" if not show_filepaths else None,
            **kwargs,
        )

    def preview_labels(self):
        """
        Visualize the label rasters on an interactive map. This can be useful
        for drawing rectangles for validation/test splitting based on where certain
        raster values are present.

        Can take a long time to load if have many/large label rasters.
        """

        return label_map(self.files)

    def split_sparse_index(self, min_length: int = 300) -> None:
        """Apply split index to create more densely labeled dataset rows.

        Args:
            min_length: Minimum length (in pixels) of the shortest side of polygon box
        """
        new_rows = []

        for temporal_interval, row in self.index.iterrows():
            filepath = row.filepath
            polygon = row.geometry

            # Split the area into more densely labeled regions
            split_polygons = split_raster_into_dense_areas(
                file_path=filepath,
                polygon=polygon,
                min_length=min_length,
                nodata_value=self.nodata_value,
                polygon_crs=self.index.crs,
            )

            # Create new rows for each split polygon
            for split_polygon in split_polygons:
                new_row = row.copy()
                new_row.geometry = split_polygon
                new_rows.append((temporal_interval, new_row))

        # Create new index if we have split polygons
        if new_rows:
            # Create new GeoDataFrame
            new_data = []
            new_temporal_intervals = []

            for temporal_interval, row in new_rows:
                new_data.append(row)
                new_temporal_intervals.append(temporal_interval)

            new_gdf = gpd.GeoDataFrame(new_data, crs=self.index.crs)
            new_gdf.index = pd.IntervalIndex(new_temporal_intervals, name="datetime")

            self.index = new_gdf


class PlottableClassificationDataset(PlottableLabelDataset):
    colormap = cm.tab20
    dataset_type = "classification"

    def __init__(
        self,
        *args,
        reduce_zero_label: bool = True,
        class_mapping: dict[int, str] | None = None,
        **kwargs,
    ) -> None:
        """
        reduce_zero_label (bool): Subtract 1 from all labels. Useful when labels start from 1 instead of the
        expected 0. Defaults to True.
        class_mapping (dict): Mapping of class indices to class names. Used for visualization.
        Other arguments are passed on to CustomCacheRasterDataset."""

        super().__init__(*args, **kwargs)
        self.reduce_zero_label = reduce_zero_label
        if class_mapping is not None:
            self.class_mapping = class_mapping

    def __getitem__(self, index) -> dict[str, Any]:
        sample = super().__getitem__(index)

        if self.reduce_zero_label:
            sample["mask"] -= 1

        return sample

    def values_as_in_files(self, values: Tensor) -> np.ndarray:
        """Label values of a mask or prediction as they are stored in the label files."""
        values = values.numpy()
        if self.reduce_zero_label:
            values = values + 1  # not +=, which would change the tensor
        return values

    def get_pixel_counts(self):
        """
        Get the pixel counts for each class in the dataset.
        This reads all the files in the dataset. To get the counts for only
        the areas with an intersection with the images, use ClassificationRasterDataset.get_label_pixel_counts()

        Returns:
            A dictionary mapping class indices or class names to pixel counts.
        """
        pixel_counts = defaultdict(int)

        for temporal_interval, row in self.index.iterrows():
            # Create GeoSlice from geometry bounds and temporal interval
            bounds = row.geometry.bounds  # (minx, miny, maxx, maxy)
            geo_slice = (
                slice(bounds[0], bounds[2]),  # x: minx to maxx
                slice(bounds[1], bounds[3]),  # y: miny to maxy
                slice(temporal_interval.left, temporal_interval.right),  # t
            )
            counts = self._get_pixel_counts(geo_slice)
            for key, cnt in counts.items():
                pixel_counts[key] += cnt
        return dict(pixel_counts)

    def _get_pixel_counts(self, index) -> dict[int, int]:
        out_dict = {}
        mask = self.values_as_in_files(self.__getitem__(index)["mask"])
        if self.nodata_value is not None:
            mask = mask[mask != self.nodata_value]
        unique, counts = np.unique(mask, return_counts=True)
        counts = dict(zip(unique, counts))

        for cls, cnt in counts.items():
            if self.class_mapping is not None:
                key = self.class_mapping.get(int(cls), int(cls))
            else:
                key = cls
            out_dict[key] = int(cnt)

        return out_dict


class PlottableRegressionDataset(PlottableLabelDataset):
    nodata_value: float = 0.0
    dataset_type = "regression"

    @property
    def dtype(self) -> torch.dtype:
        """Override base RasterDataset dtype to use float32 for regression values."""
        return torch.float32

    def __init__(self, *args, nodata_value: float | None = None, **kwargs) -> None:
        """
        nodata_value (float): Label value of pixels without a target.
        Other arguments are passed on to CustomCacheRasterDataset."""
        super().__init__(*args, **kwargs)
        if nodata_value is not None:
            self.nodata_value = nodata_value


class LabelledRasterDataset(IntersectionDataset, SamplePlotMixin):
    """Base class of the datasets of images and their labels.

    The first dataset holds the images, the second one the labels.
    """

    def plot(self, sample: dict[str, Any], **kwargs):
        predictions = sample.get("prediction")
        n_cols = 3 if predictions is not None else 2
        fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 4, 4))

        # The image dataset would plot the prediction itself, so it is left out here
        sample_without_prediction = {
            k: v for k, v in sample.items() if k != "prediction"
        }
        for i, dataset in enumerate(self.datasets):
            if hasattr(dataset, "plot"):
                dataset.plot(sample_without_prediction, ax=axes[i], **kwargs)
            else:
                raise NotImplementedError("Dataset must be plottable")

        if predictions is not None:
            label_ds: PlottableLabelDataset = self.datasets[1]
            plot_numpy_array(
                array=label_ds.values_as_in_files(predictions),
                ax=axes[-1],
                class_mapping=label_ds.class_mapping,
                colormap=label_ds.colormap,
                nodata_value=label_ds.nodata_value,
                title="Prediction",
                **kwargs,
            )

        plt.tight_layout()

        return fig

    def _repr_html_(self):
        print(
            f"Dataset containing {len(self)} items, crs: {self.crs.name}, res: {self.res}"
        )
        m = plot_dataset_index(self.datasets[0], name="Images", color="blue")
        m = plot_dataset_index(self.datasets[1], m, name="Labels", color="green")
        return plot_dataset_index(
            self, m, name="Intersection", color="pink"
        )._repr_html_()

    def preview_labels(self):
        """
        Visualize the label rasters on an interactive map. This can be useful
        for drawing rectangles for validation/test splitting based on where certain
        raster values are present.

        Can take a long time to load if have many/large label rasters.
        """
        return self.datasets[1].preview_labels()

    def __or__(self, other: LabelledRasterDataset) -> LabelledRasterDataset:
        """Combine the files of two datasets of the same type, images with images and
        labels with labels."""
        if not isinstance(other, type(self)):
            raise TypeError(
                f"Cannot union {type(self).__name__} with {type(other).__name__}"
            )

        return type(self)(
            self.datasets[0] | other.datasets[0],
            self.datasets[1] | other.datasets[1],
            collate_fn=self.collate_fn,
            transforms=self.transforms,
        )

    def __and__(self, other):
        # TorchGeo's & would stack the images and labels of both datasets on top of
        # each other
        raise TypeError(
            "Datasets of images and labels can't be intersected with '&'. "
            "Use '|' to combine their files."
        )


class ClassificationRasterDataset(LabelledRasterDataset):
    def get_label_pixel_counts(self):
        """
        Get the pixel counts for each class in the label dataset.

        Returns:
            A dictionary mapping class indices or class names to pixel counts.
        """
        label_ds: PlottableClassificationDataset = self.datasets[1]
        pixel_counts = defaultdict(int)
        for temporal_interval, row in self.index.iterrows():
            bounds = row.geometry.bounds  # (minx, miny, maxx, maxy)
            geo_slice = (
                slice(bounds[0], bounds[2]),  # x: minx to maxx
                slice(bounds[1], bounds[3]),  # y: miny to maxy
                slice(temporal_interval.left, temporal_interval.right),  # t
            )
            counts = label_ds._get_pixel_counts(geo_slice)
            for key, cnt in counts.items():
                pixel_counts[key] += cnt
        return dict(pixel_counts)


class RegressionRasterDataset(LabelledRasterDataset):
    pass


def _image_dataset(
    *,
    images_dir: str | Path | Iterable[str] | Iterable[Path],
    image_glob: str,
    image_filename_regex: str | None,
    image_date_format: str | None,
    all_image_bands: tuple[str],
    rgb_bands: tuple[str],
    sensor_name: str | None,
    crs: CRS | None,
    res: float | None,
    bands_to_return: tuple[str] | None,
    image_transforms: Callable[[dict[str, Any]], dict[str, Any]] | None,
    cache_size: int,
    image_separate_files: bool,
) -> PlottableImageDataset:
    """The image dataset of get_segmentation_dataset and get_regression_dataset."""
    if sensor_name:
        if sensor_name not in BAND_INDEX:
            raise ValueError(f"Sensor {sensor_name} not found in BAND_INDEX")
        if all_image_bands:
            print(
                "Warning: param all_image_bands will be ignored as sensor_name is provided"
            )
        all_image_bands = tuple(BAND_INDEX[sensor_name]["bandmap"].keys())
        if res:
            print("Warning: param res will be ignored as sensor_name is provided")
        res = BAND_INDEX[sensor_name]["res"]

    return PlottableImageDataset(
        paths=images_dir,
        filename_glob=image_glob,
        filename_regex=image_filename_regex,
        date_format=image_date_format,
        all_bands=all_image_bands,
        rgb_bands=rgb_bands,
        separate_files=image_separate_files,
        bands=bands_to_return,
        transforms=image_transforms,
        cache_size=cache_size,
        res=res,
        crs=crs,
    )


def get_segmentation_dataset(
    images_dir: str | Path | Iterable[str] | Iterable[Path],
    labels_dir: str | Path | Iterable[str] | Iterable[Path] | None = None,
    image_glob="*.tif",
    label_glob="*.tif",
    image_filename_regex: str | None = None,
    label_filename_regex: str | None = None,
    image_date_format: str | None = None,
    label_date_format: str | None = None,
    all_image_bands: tuple[str] = (),
    rgb_bands: tuple[str] = ("red", "green", "blue"),
    sensor_name: str | None = None,
    crs: CRS = None,
    res: float | None = None,
    bands_to_return: tuple[str] | None = None,
    image_transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    label_transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    class_mapping: dict[int, str] | None = None,
    cache_size: int = 20,
    reduce_zero_label: bool = True,
    split_sparse_labels: bool = False,
    split_sparse_labels_min_size: int = 300,
    image_separate_files: bool = False,
) -> PlottableImageDataset | ClassificationRasterDataset:
    r"""
    Create a segmentation dataset from images and labels. Labels are optional.

    Args:
        images_dir (str or Path): Path to the directory containing the images.
        labels_dir (str or Path): Path to the directory containing the labels.
        image_glob (str): Glob pattern for the image files. Defaults to "*.tif". Modify if not all .tif files in the directory should be used.
        label_glob (str): Glob pattern for the label files. Defaults to "*.tif". Modify if not all .tif files in the directory should be used.
        image_filename_regex (str): Regular expression to extract metadata from image filenames.
                                    See https://torchgeo.readthedocs.io/en/stable/tutorials/custom_raster_dataset.html#filename_regex
        label_filename_regex (str): Regular expression to extract metadata from label filenames.
        image_date_format (str): Date format of the date extracted from image filenames. Should be specified if image_filename_regex captures a date.
                            See https://torchgeo.readthedocs.io/en/stable/tutorials/custom_raster_dataset.html#date_format
                            Example for matching files from the same year, when the filenames have the format "image_YYYY_3.tif" and "label_YYYY_whatever.tif":
                                image_filename_regex=r'.*_(?P<date>\d{4})_.*',
                                label_filename_regex=r'.*_(?P<date>\d{4})_.*',
                                image_date_format='%Y',
                                label_date_format='%Y'
        label_date_format (str): Date format of the date extracted from label filenames. Should be specified if label_filename_regex captures a date.
        all_image_bands (tuple): All bands in the image files.
        rgb_bands (tuple): Bands to use for RGB visualization.
        sensor_name (str): Name of the sensor to use. Overrides all_image_bands and res. For valid sensor names, see BAND_INDEX in eotorch.bandindex.py.
        crs (CRS): Coordinate reference system of the data.
        res (float): Resolution of the data.
        bands_to_return (tuple): Bands to return from the dataset.
        image_transforms (callable): Transforms to apply to the images.
        label_transforms (callable): Transforms to apply to the labels.
        class_mapping (dict): Mapping of class indices to class names. Used for visualization.
        cache_size (int): Size of the memory file cache to use for image and label file reading.
        reduce_zero_label (bool): Subtract 1 from all labels. Useful when labels start from 1 instead of the expected 0.
        split_sparse_labels (bool): If True, will try to split the label rasters into multiple boxes, in order to reduce the number of nodata
                                    pixels in the label rasters. This is useful for large label rasters with sparse labels.
        split_sparse_labels_min_size (int): Minimum length (in pixels) of the shortest side of polygon box to be created when splitting sparse labels.
        image_separate_files (bool): Set to True if you images are stored in individual files, e.g. red.tif, green.tif, blue.tif.

    Returns:
        PlottableImageDataset or ClassificationRasterDataset: The dataset.
    """

    image_ds = _image_dataset(
        images_dir=images_dir,
        image_glob=image_glob,
        image_filename_regex=image_filename_regex,
        image_date_format=image_date_format,
        all_image_bands=all_image_bands,
        rgb_bands=rgb_bands,
        sensor_name=sensor_name,
        crs=crs,
        res=res,
        bands_to_return=bands_to_return,
        image_transforms=image_transforms,
        cache_size=cache_size,
        image_separate_files=image_separate_files,
    )
    if not labels_dir:
        return image_ds

    # The labels are read on the grid of the images
    label_ds = PlottableClassificationDataset(
        paths=labels_dir,
        filename_glob=label_glob,
        filename_regex=label_filename_regex,
        date_format=label_date_format,
        class_mapping=class_mapping,
        reduce_zero_label=reduce_zero_label,
        transforms=label_transforms,
        cache_size=cache_size,
        res=image_ds.res,
        crs=image_ds.crs,
    )
    if split_sparse_labels:
        label_ds.split_sparse_index(split_sparse_labels_min_size)

    return ClassificationRasterDataset(image_ds, label_ds)


def get_regression_dataset(
    images_dir: str | Path | Iterable[str] | Iterable[Path],
    labels_dir: str | Path | Iterable[str] | Iterable[Path] | None = None,
    image_glob="*.tif",
    label_glob="*.tif",
    image_filename_regex: str | None = None,
    label_filename_regex: str | None = None,
    image_date_format: str | None = None,
    label_date_format: str | None = None,
    all_image_bands: tuple[str] = (),
    rgb_bands: tuple[str] = ("red", "green", "blue"),
    sensor_name: str | None = None,
    crs: CRS = None,
    res: float | None = None,
    nodata_value: float = 0.0,
    bands_to_return: tuple[str] | None = None,
    image_transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    label_transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    cache_size: int = 20,
    split_sparse_labels: bool = False,
    split_sparse_labels_min_size: int = 300,
    image_separate_files: bool = False,
) -> PlottableImageDataset | RegressionRasterDataset:
    r"""
    Create a regression dataset from images and labels. Labels are optional.

    Args:
        images_dir (str or Path): Path to the directory containing the images.
        labels_dir (str or Path): Path to the directory containing the labels.
        image_glob (str): Glob pattern for the image files. Defaults to "*.tif". Modify if not all .tif files in the directory should be used.
        label_glob (str): Glob pattern for the label files. Defaults to "*.tif". Modify if not all .tif files in the directory should be used.
        image_filename_regex (str): Regular expression to extract metadata from image filenames.
                                    See https://torchgeo.readthedocs.io/en/stable/tutorials/custom_raster_dataset.html#filename_regex
        label_filename_regex (str): Regular expression to extract metadata from label filenames.
        image_date_format (str): Date format of the date extracted from image filenames. Should be specified if image_filename_regex captures a date.
                            See https://torchgeo.readthedocs.io/en/stable/tutorials/custom_raster_dataset.html#date_format
                            Example for matching files from the same year, when the filenames have the format "image_YYYY_3.tif" and "label_YYYY_whatever.tif":
                                image_filename_regex=r'.*_(?P<date>\d{4})_.*',
                                label_filename_regex=r'.*_(?P<date>\d{4})_.*',
                                image_date_format='%Y',
                                label_date_format='%Y'
        label_date_format (str): Date format of the date extracted from label filenames. Should be specified if label_filename_regex captures a date.
        all_image_bands (tuple): All bands in the image files.
        rgb_bands (tuple): Bands to use for RGB visualization.
        sensor_name (str): Name of the sensor to use. Overrides all_image_bands and res. For valid sensor names, see BAND_INDEX in eotorch.bandindex.py.
        crs (CRS): Coordinate reference system of the data.
        res (float): Resolution of the data.
        nodata_value (float): Label value of pixels without a target, which are left out
                              of the loss and metrics.
        bands_to_return (tuple): Bands to return from the dataset.
        image_transforms (callable): Transforms to apply to the images.
        label_transforms (callable): Transforms to apply to the labels.
        cache_size (int): Size of the memory file cache to use for image and label file reading.
        split_sparse_labels (bool): If True, will try to split the label rasters into multiple boxes, in order to reduce the number of nodata
                                    pixels in the label rasters. This is useful for large label rasters with sparse labels.
        split_sparse_labels_min_size (int): Minimum length (in pixels) of the shortest side of polygon box to be created when splitting sparse labels.
        image_separate_files (bool): Set to True if you images are stored in individual files, e.g. red.tif, green.tif, blue.tif.

    Returns:
        PlottableImageDataset or RegressionRasterDataset: The dataset.
    """

    image_ds = _image_dataset(
        images_dir=images_dir,
        image_glob=image_glob,
        image_filename_regex=image_filename_regex,
        image_date_format=image_date_format,
        all_image_bands=all_image_bands,
        rgb_bands=rgb_bands,
        sensor_name=sensor_name,
        crs=crs,
        res=res,
        bands_to_return=bands_to_return,
        image_transforms=image_transforms,
        cache_size=cache_size,
        image_separate_files=image_separate_files,
    )
    if not labels_dir:
        return image_ds

    # The labels are read on the grid of the images
    label_ds = PlottableRegressionDataset(
        paths=labels_dir,
        filename_glob=label_glob,
        filename_regex=label_filename_regex,
        date_format=label_date_format,
        nodata_value=nodata_value,
        transforms=label_transforms,
        cache_size=cache_size,
        res=image_ds.res,
        crs=image_ds.crs,
    )
    if split_sparse_labels:
        label_ds.split_sparse_index(split_sparse_labels_min_size)

    return RegressionRasterDataset(image_ds, label_ds)
