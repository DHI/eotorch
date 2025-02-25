from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from rasterio.plot import show
from torchgeo.datasets import IntersectionDataset, RasterDataset


class PlottableRasterDataset(RasterDataset):
    colormap = cm.tab20
    nodata_value = 0
    class_mapping = None

    def plot(self, sample, ax=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots()

        if self.is_image:
            rgb_indices = []
            for band in self.rgb_bands:
                rgb_indices.append(self.all_bands.index(band))

            image = sample["image"][rgb_indices]
            return show(image.numpy(), ax=ax, adjust=True, **kwargs)

        vals = sample["mask"].numpy()
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
            if isinstance(dataset, PlottableRasterDataset):
                dataset.plot(sample, ax=axes[i], **kwargs)
            else:
                raise NotImplementedError("Dataset must be plottable")

        plt.tight_layout()

        return fig


class SegmentationRasterDataset:
    """
    A dataset for semantic segmentation of raster data.
    Can be used with or without labels.

    Parameters
    ----------
        images_dir: str or Path, path to the directory containing images
        labels_dir: str or Path, path to the directory containing labels
        image_glob: str, glob pattern for images
        label_glob: str, glob pattern for labels
        image_kwargs: dict, keyword arguments for the image dataset, used to override defaults and set custom values
        label_kwargs: dict, keyword arguments for the label dataset, used to override defaults and set custom values

    """

    @classmethod
    def create(
        cls,
        images_dir: str | Path,
        labels_dir: str | Path = None,
        image_glob="*.tif",
        label_glob="*.tif",
        image_kwargs: dict[str, Any] = None,
        label_kwargs: dict[str, Any] = None,
    ):
        image_kwargs = image_kwargs or {}
        label_kwargs = label_kwargs or {}

        image_ds = PlottableRasterDataset(paths=images_dir)
        image_ds.filename_glob = image_glob
        for k, v in image_kwargs.items():
            setattr(image_ds, k, v)

        if labels_dir:
            label_ds = PlottableRasterDataset(paths=labels_dir)
            label_ds.is_image = False
            label_ds.filename_glob = label_glob
            for k, v in label_kwargs.items():
                setattr(label_ds, k, v)

            return LabelledRasterDataset(image_ds, label_ds)

        else:
            return image_ds
