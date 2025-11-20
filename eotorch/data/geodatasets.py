from __future__ import annotations

import copy
import functools
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence, cast

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
    GeoDataset,
    IntersectionDataset,
    RasterDataset,
    concat_samples,
)
from torchgeo.datasets.utils import BoundingBox

from eotorch.bandindex import BAND_INDEX
from eotorch.inference.inference_utils import prediction_to_numpy
from eotorch.plot import label_map, plot_dataset_index, plot_numpy_array, plot_samples
from eotorch.utils import _format_filepaths, transform_index

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
        self,
        n: int = 3,
        patch_size: int = 256,
        show_filepaths: bool = False,
        nodata_val: int | float = 0,
    ):
        # if the dataset is an IntersectionDataset, get the nodata value from the label dataset
        if isinstance(self, IntersectionDataset) and hasattr(
            self.datasets[1], "nodata_value"
        ):
            nodata_val = self.datasets[1].nodata_value

        return plot_samples(
            self,
            n=n,
            patch_size=patch_size,
            nodata_val=nodata_val,
            show_filepaths=show_filepaths,
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
    ) -> None:
        super().__init__(paths, crs, res, bands, transforms, cache=cache_size > 0)
        self.cache_size = cache_size
        self._cached_load_warp_file = functools.lru_cache(maxsize=cache_size)(
            self._load_warp_file
        )
        # In order to avoid spamming "Skipping source" Log Messages in Rasterio Merge
        logging.getLogger("rasterio").setLevel(logging.WARNING)

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        interval = pd.Interval(query.mint, query.maxt)
        index = self.index.iloc[self.index.index.overlaps(interval)]
        index = index.cx[query.minx : query.maxx, query.miny : query.maxy]  # type: ignore[misc]

        filepaths = index.filepath.tolist()
        if index.empty:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        if self.separate_files:
            data_list: list[Tensor] = []
            filename_regex = re.compile(self.filename_regex, re.VERBOSE)
            for band in self.bands:
                band_filepaths = []
                for filepath in index.filepath:
                    filename = os.path.basename(filepath)
                    directory = os.path.dirname(filepath)
                    match = re.match(filename_regex, filename)
                    if match:
                        if "band" in match.groupdict():
                            start = match.start("band")
                            end = match.end("band")
                            filename = filename[:start] + band + filename[end:]
                    filepath = os.path.join(directory, filename)
                    band_filepaths.append(filepath)
                data_list.append(self._merge_files(band_filepaths, query))
            data = torch.cat(data_list)
        else:
            data = self._merge_files(filepaths, query, self.band_indexes)

        sample = {"crs": self.crs, "bounds": query}

        data = data.to(self.dtype)

        if self.is_image:
            sample["image"] = data
            sample["image"] = data
            sample["image_filepaths"] = filepaths
        else:
            sample["mask"] = data.squeeze(0)
            sample["mask"] = data
            sample["mask_filepaths"] = filepaths

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

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

        The new dataset inherits transforms and cache_size from 'self'.
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
        }

        for attr_name, display_name in compatibility_attrs.items():
            self_attr = getattr(self, attr_name)
            other_attr = getattr(other, attr_name)
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
        # Combine indexes from both datasets
        combined_index = pd.concat(
            [self.index, other_transformed.index], ignore_index=True
        )

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

        # Create the new CustomCacheRasterDataset instance.
        # It will initially build an index based on combined_constructor_paths and new_crs.

        new_ds_class = self.__class__
        new_ds_class.all_bands = self.all_bands
        new_ds_class.rgb_bands = self.rgb_bands
        new_ds_class.separate_files = self.separate_files
        new_ds_class.is_image = self.is_image
        new_dataset = new_ds_class(
            paths=combined_constructor_paths,  # type: ignore[arg-type]
            crs=self.crs,
            res=self.res,
            bands=self.bands,  # Bands (checked for equality)
            transforms=self.transforms,  # Inherit from self
            cache_size=self.cache_size,  # Inherit from self
        )

        # Replace the automatically generated index with our carefully constructed one.
        new_dataset.index = combined_index

        return new_dataset


class PlottableImageDataset(CustomCacheRasterDataset):
    def plot(self, sample, ax=None, **kwargs):
        predictions = sample.get("prediction")

        if ax is None:
            # _, ax = plt.subplots()
            n_cols = 2 if predictions is not None else 1
            fig, ax = plt.subplots(1, n_cols, figsize=(n_cols * 4, 4))

        rgb_indices = []
        for band in self.rgb_bands:
            rgb_indices.append(self.all_bands.index(band))

        image = sample["image"][rgb_indices]

        if show_filepaths := kwargs.pop("show_filepaths", False):
            filepaths = sample["image_filepaths"]
            formatted_paths = _format_filepaths(filepaths)
            ax.set_title(formatted_paths, fontsize=8, wrap=True)

        show(
            image.numpy(),
            ax=ax if predictions is None else ax[0],
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


class PlottabeLabelDataset(CustomCacheRasterDataset):
    colormap = cm.tab20
    nodata_value = 0
    class_mapping: dict = None
    is_image = False

    def __init__(
        self,
        paths: str | Path | Iterable[str] | Iterable[Path] = "data",
        crs: CRS | None = None,
        res: float | None = None,
        bands: Sequence[str] | None = None,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cache_size: int = 20,
        reduce_zero_label: bool = True,
    ) -> None:
        """
        reduce_zero_label (bool): Subtract 1 from all labels. Useful when labels start from 1 instead of the
        expected 0. Defaults to True."""

        super().__init__(paths, crs, res, bands, transforms, cache_size)
        self.reduce_zero_label = reduce_zero_label

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        sample = super().__getitem__(query)

        if self.reduce_zero_label:
            sample["mask"] -= 1

        return sample

    def plot(self, sample, ax=None, **kwargs):
        vals = sample["mask"].numpy()
        if self.reduce_zero_label:
            vals += 1
        if show_filepaths := kwargs.pop("show_filepaths", False):
            filepaths = sample["mask_filepaths"]
            formatted_paths = _format_filepaths(filepaths)
            ax.set_title(formatted_paths, fontsize=8, wrap=True)

        plot_numpy_array(
            array=vals,
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

    def get_pixel_counts(self):
        """
        Get the pixel counts for each class in the dataset.
        This reads all the files in the dataset. To get the counts for only
        the areas with an intersection with the images, use LabelledRasterDataset.get_label_pixel_counts()

        Returns:
            A dictionary mapping class indices or class names to pixel counts.
        """
        pixel_counts = defaultdict(int)

        # Use GeoPandas iteration instead of R-tree intersection
        for temporal_interval, row in self.index.iterrows():
            # Create bounding box from geometry bounds
            bounds = row.geometry.bounds  # (minx, miny, maxx, maxy)
            # Get temporal bounds from the index (which is a pd.Interval)
            bbox = BoundingBox(
                bounds[0],
                bounds[2],
                bounds[1],
                bounds[3],
                temporal_interval.left,
                temporal_interval.right,
            )
            counts = self._get_pixel_counts(bbox)
            for key, cnt in counts.items():
                pixel_counts[key] += cnt
        return dict(pixel_counts)

    def _get_pixel_counts(self, bbox: BoundingBox) -> dict[int, int]:
        out_dict = {}
        sample = self.__getitem__(bbox)
        mask = sample["mask"]
        if hasattr(mask, "numpy"):
            mask = mask.numpy()
        if self.reduce_zero_label:
            mask = mask + 1
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


class LabelledRasterDataset(
    IntersectionDataset,
    SamplePlotMixin,
):
    def __init__(
        self,
        dataset1: GeoDataset,
        dataset2: GeoDataset,
        collate_fn: Callable[
            [Sequence[dict[str, Any]]], dict[str, Any]
        ] = concat_samples,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(dataset1, dataset2, collate_fn, transforms)

        # rename index columns resulting from merge, for more clarity
        if "filepath_1" in self.index.columns:
            self.index = self.index.rename(
                columns={"filepath_1": "image_filepath", "filepath_2": "label_filepath"}
            )
        if "datetime_1" in self.index.columns:
            self.index = self.index.rename(
                columns={
                    "datetime_1": "image_datetime",
                    "datetime_2": "label_datetime",
                }
            )

    def plot(self, sample: dict[str, Any], **kwargs):
        predictions = sample.pop("prediction", None)
        n_cols = 3 if predictions is not None else 2
        fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 4, 4))

        for i, dataset in enumerate(self.datasets):
            if hasattr(dataset, "plot"):
                dataset.plot(sample, ax=axes[i], **kwargs)
            else:
                raise NotImplementedError("Dataset must be plottable")

        if predictions is not None:
            label_ds: PlottabeLabelDataset = self.datasets[-1]
            data = predictions.numpy()
            if label_ds.reduce_zero_label:
                data += 1
            plot_numpy_array(
                array=data,
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

        if isinstance(self.datasets[-1], PlottabeLabelDataset):
            label_ds: PlottabeLabelDataset = self.datasets[-1]
            return label_ds.preview_labels()
        else:
            raise NotImplementedError(
                "Label dataset must be of type PlottabeLabelDataset"
            )

    def get_label_pixel_counts(self):
        """
        Get the pixel counts for each class in the label dataset.

        Returns:
            A dictionary mapping class indices or class names to pixel counts.
        """
        if isinstance(self.datasets[-1], PlottabeLabelDataset):
            label_ds: PlottabeLabelDataset = self.datasets[-1]
            pixel_counts = defaultdict(int)
            # Use GeoPandas iteration instead of R-tree intersection
            for temporal_interval, row in self.index.iterrows():
                bounds = row.geometry.bounds  # (minx, miny, maxx, maxy)
                # Get temporal bounds from the index (which is a pd.Interval)
                bbox = BoundingBox(
                    bounds[0],
                    bounds[2],
                    bounds[1],
                    bounds[3],
                    temporal_interval.left,
                    temporal_interval.right,
                )
                counts = label_ds._get_pixel_counts(bbox)
                for key, cnt in counts.items():
                    pixel_counts[key] += cnt
            return dict(pixel_counts)

        else:
            raise NotImplementedError(
                "Label dataset must be of type PlottabeLabelDataset"
            )

    def __or__(self, other: LabelledRasterDataset) -> LabelledRasterDataset:
        """
        Overload the | operator to create a new LabelledRasterDataset from two LabelledRasterDatasets.
        """
        if not isinstance(other, LabelledRasterDataset):
            raise TypeError(
                f"Cannot union {type(self).__name__} with {type(other).__name__}"
            )

        return LabelledRasterDataset(
            self.datasets[0] | other.datasets[0],
            self.datasets[1] | other.datasets[1],
            collate_fn=self.collate_fn,
            transforms=self.transforms,
        )

    def __and__(self, other: LabelledRasterDataset) -> LabelledRasterDataset:
        return LabelledRasterDataset(
            IntersectionDataset(self.datasets[0], other.datasets[0]),
            IntersectionDataset(self.datasets[1], other.datasets[1]),
            collate_fn=self.collate_fn,
            transforms=self.transforms,
        )

    @property
    def crs(self) -> CRS:
        """:term:`coordinate reference system (CRS)` of both datasets.

        Returns:
            The :term:`coordinate reference system (CRS)`.
        """
        return self.index.crs
        # return self._crs

    @crs.setter
    def crs(self, new_crs: CRS) -> None:
        # super().__setattr__("crs", new_crs)
        old_crs = self.crs
        index_needs_update = new_crs != old_crs
        # self._crs = new_crs
        self.datasets[0].crs = new_crs
        self.datasets[1].crs = new_crs

        if index_needs_update and len(self.index) > 0:
            self.index = transform_index(
                index=self.index, current_crs=old_crs, new_crs=new_crs
            )


def get_segmentation_dataset(
    images_dir: str | Path | Iterable[str] | Iterable[Path],
    labels_dir: str | Path | Iterable[str] | Iterable[Path] = None,
    image_glob="*.tif",
    label_glob="*.tif",
    image_filename_regex: str = None,
    label_filename_regex: str = None,
    image_date_format: str = None,
    label_date_format: str = None,
    all_image_bands: tuple[str] = (),
    rgb_bands: tuple[str] = ("red", "green", "blue"),
    sensor_name: str = None,
    crs: CRS = None,
    res: float = None,
    bands_to_return: tuple[str] = None,
    image_transforms: Callable[[dict[str, Any]], dict[str, Any]] = None,
    label_transforms: Callable[[dict[str, Any]], dict[str, Any]] = None,
    class_mapping: dict[int, str] = None,
    cache_size: int = 20,
    reduce_zero_label: bool = True,
    split_sparse_labels: bool = False,
    split_sparse_labels_min_size: int = 300,
    image_separate_files: bool = False,
) -> PlottableImageDataset | LabelledRasterDataset:
    r"""
    Create a segmentation dataset from images and labels. Labels are optional.

    Args:
        images_dir (str or Path): Path to the directory containing the images.
        labels_dir (str or Path): Path to the directory containing the labels.
        image_glob (str): Glob pattern for the image files. Defaults to "*.tif". Modify if not all .tif files in the directory should be used.
        label_glob (str): Glob pattern for the label files. Defaults to "*.tif". Modify if not all .tif files in the directory should be used.
        image_filename_regex (str): Regular expression to extract metadata from image filenames.
                                    See https://torchgeo.readthedocs.io/en/stable/tutorials/custom_raster_dataset.html#filename_regex
        date_format (str): Date format to extract metadata from image filenames. Should be specified if image_filename_regex is used.
                            See https://torchgeo.readthedocs.io/en/stable/tutorials/custom_raster_dataset.html#date_format
                            Example for matching files from the same year, when the filenames have the format "image_YYYY_3.tif" and "label_YYYY_whatever.tif":
                                image_filename_regex=r'.*_(?P<date>\d{4})_.*',
                                label_filename_regex=r'.*_(?P<date>\d{4})_.*',
                                date_format='%Y'
        label_filename_regex (str): Regular expression to extract metadata from label filenames.
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
        PlottableImageDataset or LabelledRasterDataset: The dataset.
    """

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

    image_filename_regex = image_filename_regex or r".*"
    label_filename_regex = label_filename_regex or r".*"
    image_date_format = image_date_format or "%Y%m%d"
    label_date_format = label_date_format or "%Y%m%d"

    image_ds_class = PlottableImageDataset
    image_ds_class.filename_glob = image_glob
    image_ds_class.all_bands = all_image_bands
    image_ds_class.rgb_bands = rgb_bands
    image_ds_class.separate_files = image_separate_files
    image_ds_class.filename_regex = image_filename_regex
    image_ds_class.date_format = image_date_format

    image_ds = image_ds_class(
        paths=images_dir,
        bands=bands_to_return,
        transforms=image_transforms,
        cache_size=cache_size,
        res=res,
        crs=crs,
    )

    if labels_dir:
        label_ds_class = PlottabeLabelDataset
        label_ds_class.filename_glob = label_glob
        label_ds_class.class_mapping = class_mapping
        label_ds_class.filename_regex = label_filename_regex
        label_ds_class.date_format = label_date_format

        label_ds = label_ds_class(
            paths=labels_dir,
            transforms=label_transforms,
            cache_size=cache_size,
            reduce_zero_label=reduce_zero_label,
            res=res,
            crs=crs,
        )
        if split_sparse_labels:
            label_ds.split_sparse_index(split_sparse_labels_min_size)

        return LabelledRasterDataset(image_ds, label_ds)

    return image_ds
