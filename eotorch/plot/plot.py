from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np
import rasterio as rst
from matplotlib import cm
from matplotlib import pyplot as plt
from rasterio.plot import show
from rasterio.windows import get_data_window

from eotorch import utils


def plot_class_raster(
    tif_file_path: str | Path,
    ax: plt.Axes = None,
    class_mapping: dict[int, str] = None,
    colormap=cm.tab20,
    data_window_only: bool = False,
    **kwargs,
) -> None:
    """
    Plot a raster image with class labels.
    """
    with rst.open(tif_file_path) as src:
        array = src.read(1)
        plot_numpy_array(
            array,
            ax=ax,
            class_mapping=class_mapping,
            nodata_value=src.nodata,
            colormap=colormap,
            data_window_only=data_window_only,
            **kwargs,
        )


def plot_numpy_array(
    array: np.ndarray,
    ax: plt.Axes = None,
    class_mapping: dict[int, str] = None,
    nodata_value: int = 0,
    colormap=cm.tab20,
    data_window_only: bool = False,
    **kwargs,
):
    if ax is None:
        _, ax = plt.subplots()

    if data_window_only:
        window = get_data_window(array, nodata=nodata_value)
        plot_array = array[utils.window_to_np_idc(window)]
    else:
        plot_array = array

    values = np.unique(plot_array.ravel()).tolist()
    plot_values = set(values + [nodata_value])
    if class_mapping:
        bounds = list(range(len(plot_values) + 1))
    else:
        bounds = list(plot_values) + [max(values) + 1]
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, len(bounds))
    ax = show(plot_array, ax=ax, cmap=colormap, norm=norm, **kwargs)

    class_mapping = (
        class_mapping.copy() if class_mapping else {v: str(v) for v in values}
    )

    if (nodata_value in values) and (nodata_value not in class_mapping):
        class_mapping[nodata_value] = "No Data"

    im = ax.get_images()[0]

    colors = {v: im.cmap(im.norm(v)) for v in plot_values}

    legend_patches = [
        mpatches.Patch(color=colors[i], label=class_mapping.get(i, str(i)))
        for i in values
    ]
    ax.legend(
        handles=legend_patches,
        bbox_to_anchor=(1.05, 1),
        loc=2,
        borderaxespad=0.0,
    )


def plot_samples(dataset, n: int = 3, patch_size: int = 256, nodata_val: int = 0):
    from torch.utils.data import DataLoader
    from torchgeo.datasets import stack_samples, unbind_samples
    from torchgeo.samplers import RandomGeoSampler

    assert hasattr(dataset, "plot"), "Dataset must have a plot method"
    # sampler = GridGeoSampler(dataset, size=patch_size, stride=100)
    sampler = RandomGeoSampler(dataset, size=patch_size)
    dataloader = DataLoader(
        dataset, sampler=sampler, collate_fn=stack_samples, batch_size=1
    )
    _iterator = iter(dataloader)
    samples = []
    for _ in range(100):
        try:
            batch = next(_iterator)
            sample = unbind_samples(batch)[0]
            if (
                (sample["image"] == 0).all()
                or (sample["mask"] == nodata_val).all()
                or (sample["mask"] == -1).all()
            ):
                continue
            samples.append(sample)
            if len(samples) == n:
                break
        except StopIteration:
            break

    if not samples:
        print("No valid samples found.")
        return

    for i, sample in enumerate(samples):
        dataset.plot(sample)
        plt.suptitle(f"Sample {i + 1}")
        plt.show()
