from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib import cm
from rasterio.plot import show
from torchgeo.datasets import IntersectionDataset, RasterDataset
from torchgeo.datasets.utils import BoundingBox

from eotorch.bandindex import BAND_INDEX
from eotorch.plot.plot import plot_numpy_array

if TYPE_CHECKING:
    from rasterio.crs import CRS


class CustomCacheRasterDataset(RasterDataset):
    """
    Implementing RasterDatset with customisable cache size, as it is hardcoded to 128 in torchgeo's RasterDataset.
    """

    def __init__(
        self,
        paths: Path | Iterable[Path] = "data",
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


class PlottableImageDataset(CustomCacheRasterDataset):
    def plot(self, sample, ax=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots()

        rgb_indices = []
        for band in self.rgb_bands:
            rgb_indices.append(self.all_bands.index(band))

        image = sample["image"][rgb_indices]
        return show(image.numpy(), ax=ax, adjust=True, **kwargs)


class PlottabeLabelDataset(CustomCacheRasterDataset):
    colormap = cm.tab20
    nodata_value = 0
    class_mapping: dict = None
    is_image = False

    def __init__(
        self,
        paths: Path | Iterable[Path] = "data",
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

        plot_numpy_array(
            array=vals,
            ax=ax,
            class_mapping=self.class_mapping,
            colormap=self.colormap,
            nodata_value=self.nodata_value,
            **kwargs,
        )


class LabelledRasterDataset(IntersectionDataset):
    def plot(self, sample: dict[str, Any], **kwargs):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        for i, dataset in enumerate(self.datasets):
            if isinstance(dataset, (PlottableImageDataset, PlottabeLabelDataset)):
                dataset.plot(sample, ax=axes[i], **kwargs)
            else:
                raise NotImplementedError("Dataset must be plottable")

        plt.tight_layout()

        return fig


def get_segmentation_dataset(
    images_dir: str | Path,
    labels_dir: str | Path = None,
    image_glob="*.tif",
    label_glob="*.tif",
    all_image_bands: tuple[str] = (),
    rgb_bands: tuple[str] = ("red", "green", "blue"),
    sensor_name: str = None,
    crs: CRS | None = None,
    res: float | None = None,
    bands_to_return: tuple[str] = None,
    image_transforms: Callable[[dict[str, Any]], dict[str, Any]] = None,
    label_transforms: Callable[[dict[str, Any]], dict[str, Any]] = None,
    class_mapping: dict[int, str] = None,
    cache_size: int = 20,
    reduce_zero_label: bool = True,
) -> PlottableImageDataset | LabelledRasterDataset:
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
