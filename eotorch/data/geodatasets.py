from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from rasterio.crs import CRS
from rasterio.plot import show
from torchgeo.datasets import IntersectionDataset, RasterDataset
from torchgeo.datasets.utils import BoundingBox


class PlottableImageDataset(RasterDataset):
    def plot(self, sample, ax=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots()

        rgb_indices = []
        for band in self.rgb_bands:
            rgb_indices.append(self.all_bands.index(band))

        image = sample["image"][rgb_indices]
        return show(image.numpy(), ax=ax, adjust=True, **kwargs)


class PlottabeLabelDataset(RasterDataset):
    colormap = cm.tab20
    nodata_value = 0
    class_mapping = None
    is_image = False

    def __init__(
        self,
        paths: Path | Iterable[Path] = "data",
        crs: CRS | None = None,
        res: float | None = None,
        bands: Sequence[str] | None = None,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cache: bool = True,
        reduce_zero_label: bool = True,
    ) -> None:
        """
        reduce_zero_label (bool): Subtract 1 from all labels. Useful when labels start from 1 instead of the
        expected 0. Defaults to True."""

        super().__init__(paths, crs, res, bands, transforms, cache)
        self.reduce_zero_label = reduce_zero_label

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        sample = super().__getitem__(query)

        if self.reduce_zero_label:
            sample["mask"] -= 1

        return sample

    def plot(self, sample, ax=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots()

        vals = sample["mask"].numpy()
        if self.reduce_zero_label:
            vals += 1
        values = np.unique(vals.ravel()).tolist()
        plot_values = set(values + [self.nodata_value])
        bounds = list(plot_values) + [max(values) + 1]
        norm = plt.matplotlib.colors.BoundaryNorm(bounds, self.colormap.N)
        ax = show(vals, ax=ax, cmap=self.colormap, norm=norm, **kwargs)

        class_mapping = self.class_mapping or {v: str(v) for v in values}

        if (self.nodata_value in values) and (self.nodata_value not in class_mapping):
            class_mapping[self.nodata_value] = "No Data"

        im = ax.get_images()[0]

        colors = {v: im.cmap(im.norm(v)) for v in plot_values}

        legend_patches = [
            mpatches.Patch(color=colors[i], label=class_mapping[i]) for i in values
        ]
        ax.legend(
            handles=legend_patches,
            bbox_to_anchor=(1.05, 1),
            loc=2,
            borderaxespad=0.0,
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
    bands_to_return: tuple[str] = None,
    image_transforms: Callable[[dict[str, Any]], dict[str, Any]] = None,
    label_transforms: Callable[[dict[str, Any]], dict[str, Any]] = None,
    class_mapping: dict[int, str] = None,
    cache: bool = True,
    reduce_zero_label: bool = True,
) -> PlottableImageDataset | LabelledRasterDataset:
    image_ds_class = PlottableImageDataset
    image_ds_class.filename_glob = image_glob
    image_ds_class.all_bands = all_image_bands
    image_ds_class.rgb_bands = rgb_bands

    image_ds = image_ds_class(
        paths=images_dir,
        bands=bands_to_return,
        transforms=image_transforms,
        cache=cache,
    )

    if labels_dir:
        label_ds_class = PlottabeLabelDataset
        label_ds_class.filename_glob = label_glob
        label_ds_class.class_mapping = class_mapping

        label_ds = label_ds_class(
            paths=labels_dir,
            transforms=label_transforms,
            cache=cache,
            reduce_zero_label=reduce_zero_label,
        )

        return LabelledRasterDataset(image_ds, label_ds)

    return image_ds
