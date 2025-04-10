from __future__ import annotations

import functools
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence, cast

import matplotlib.pyplot as plt
import torch
from matplotlib import cm
from rasterio.plot import show
from torchgeo.datasets import IntersectionDataset, RasterDataset
from torchgeo.datasets.utils import BoundingBox

from eotorch.bandindex import BAND_INDEX
from eotorch.plot import label_map, plot_dataset_index, plot_numpy_array
from eotorch.utils import _format_filepaths

if TYPE_CHECKING:
    from rasterio.crs import CRS
    from torch import Tensor


class CustomCacheRasterDataset(RasterDataset):
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
        print(f"Dataset containing {len(self)} files, crs: {self.crs}, res: {self.res}")
        return plot_dataset_index(self)._repr_html_()


class PlottableImageDataset(CustomCacheRasterDataset):
    def plot(self, sample, ax=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots()

        rgb_indices = []
        for band in self.rgb_bands:
            rgb_indices.append(self.all_bands.index(band))

        image = sample["image"][rgb_indices]

        if show_filepaths := kwargs.pop("show_filepaths", False):
            filepaths = sample["image_filepaths"]
            formatted_paths = _format_filepaths(filepaths)
            ax.set_title(formatted_paths, fontsize=8, wrap=True)

        return show(
            image.numpy(),
            ax=ax,
            adjust=True,
            title="Image" if not show_filepaths else None,
            **kwargs,
        )


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


class LabelledRasterDataset(IntersectionDataset):
    def plot(self, sample: dict[str, Any], **kwargs):
        n_cols = 2 if "prediction" not in sample else 3
        fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 4, 4))

        for i, dataset in enumerate(self.datasets):
            if isinstance(dataset, (PlottableImageDataset, PlottabeLabelDataset)):
                dataset.plot(sample, ax=axes[i], **kwargs)
            else:
                raise NotImplementedError("Dataset must be plottable")

        if "prediction" in sample:
            label_ds: PlottabeLabelDataset = self.datasets[-1]
            data = sample["prediction"].numpy()
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
        print(f"Dataset containing {len(self)} files, crs: {self.crs}, res: {self.res}")
        return plot_dataset_index(self)._repr_html_()

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

    image_ds_class = PlottableImageDataset
    image_ds_class.filename_glob = image_glob
    image_ds_class.all_bands = all_image_bands
    image_ds_class.rgb_bands = rgb_bands
    image_ds_class.separate_files = image_separate_files
    if image_filename_regex:
        image_ds_class.filename_regex = image_filename_regex
    if image_date_format:
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
        if label_filename_regex:
            label_ds_class.filename_regex = label_filename_regex
        if label_date_format:
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
