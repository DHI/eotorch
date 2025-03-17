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
    """
    Plot a numpy array with class labels and consistent colors.

    Args:
        array: The input array to plot
        ax: Matplotlib axes to plot on (creates new figure if None)
        class_mapping: Dictionary mapping class values to their string labels
        nodata_value: Value to treat as no data
        colormap: Matplotlib colormap to use
        data_window_only: If True, crops to non-nodata region
        **kwargs: Additional arguments for rasterio's show function
    """
    if ax is None:
        _, ax = plt.subplots()

    if data_window_only:
        window = get_data_window(array, nodata=nodata_value)
        plot_array = array[utils.window_to_np_idc(window)]
    else:
        plot_array = array

    values = np.unique(plot_array.ravel()).tolist()

    # Determine min/max class values based on mapping or array values
    if class_mapping:
        class_keys = sorted(list(class_mapping.keys()))
        min_class = min(class_keys) if class_keys else 0
        max_class = max(class_keys) if class_keys else 0
    else:
        min_class = min(values) if values else 0
        max_class = max(values) if values else 0

    # Ensure nodata_value is included in the range if negative
    if nodata_value < min_class:
        min_class = nodata_value

    # Make sure the maximum class index is at least 7 to ensure colormap compatibility
    max_class = max(max_class, 7)

    # Create a more stable boundary norm - reliable for categorical data
    bounds = np.arange(min_class - 0.5, max_class + 1.5)
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, colormap.N)

    # Plot using rasterio's show function
    ax = show(plot_array, ax=ax, cmap=colormap, norm=norm, **kwargs)

    # Prepare class mapping
    if class_mapping is None:
        class_mapping = {v: str(v) for v in values}
    else:
        class_mapping = class_mapping.copy()

    # Add nodata to class mapping if needed
    if (nodata_value in values) and (
        nodata_value not in class_mapping
        or class_mapping[nodata_value] == str(nodata_value)
    ):
        class_mapping[nodata_value] = "No Data"

    # Create deterministic color mapping based on class values
    colors = {}
    for cls_val in range(min_class, max_class + 1):
        colors[cls_val] = colormap(norm(cls_val))

    # Create legend only for values present in this image
    legend_patches = [
        mpatches.Patch(color=colors[i], label=class_mapping.get(i, str(i)))
        for i in values
        if i in colors
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
