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
from shapely.geometry import box as shapely_box
from torchgeo.datasets import IntersectionDataset, RasterDataset
from torchgeo.datasets.utils import BoundingBox

from eotorch.bandindex import BAND_INDEX
from eotorch.inference.inference_utils import prediction_to_numpy
from eotorch.plot import label_map, plot_dataset_index, plot_numpy_array, plot_samples
from eotorch.utils import _format_filepaths, tranform_index

if TYPE_CHECKING:
    from torch import Tensor


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
        # Use GeoPandas spatial indexing instead of R-tree
        if hasattr(self.index, "cx"):
            # New GeoPandas approach
            query_geom = shapely_box(query.minx, query.miny, query.maxx, query.maxy)
            intersecting_items = self.index[self.index.geometry.intersects(query_geom)]
            filepaths = cast(list[str], intersecting_items["filepath"].tolist())
        else:
            # Legacy R-tree approach for backwards compatibility
            hits = self.index.intersection(tuple(query), objects=True)
            filepaths = cast(list[str], [hit.object for hit in hits])

        if not filepaths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        if self.separate_files:
            data_list: list[Tensor] = []
            filename_regex = re.compile(self.filename_regex, re.VERBOSE)
            for band in self.bands:
                band_filepaths = []
                for filepath in filepaths:
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
            sample["image_filepaths"] = filepaths
        else:
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

        # --- Paths for the new dataset's constructor ---
        # This ensures new_dataset.files can be correctly computed by RasterDataset parent.
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
        for _, row in self.index.iterrows():
            # Create bounding box from geometry bounds
            bounds = row.geometry.bounds  # (minx, miny, maxx, maxy)
            bbox = BoundingBox(
                bounds[0], bounds[2], bounds[1], bounds[3], 0, 1
            )  # Add time dimension
            counts = self._get_pixel_counts(bbox)
            for key, cnt in counts.items():
                pixel_counts[key] += cnt
        return dict(pixel_counts)

    def _get_pixel_counts(self, bbox: BoundingBox) -> dict[int, int]:
        import numpy as np

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


class LabelledRasterDataset(
    IntersectionDataset,
    SamplePlotMixin,
):
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
            for _, row in self.index.iterrows():
                bounds = row.geometry.bounds  # (minx, miny, maxx, maxy)
                bbox = BoundingBox(
                    bounds[0], bounds[2], bounds[1], bounds[3], 0, 1
                )  # Add time dimension
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
            self.index = tranform_index(
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

        return LabelledRasterDataset(image_ds, label_ds)

    return image_ds
