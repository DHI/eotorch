from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np
import rasterio as rst
from matplotlib import cm
from matplotlib import pyplot as plt
from rasterio.plot import show


def plot_numpy_array(
    array: np.ndarray,
    ax: plt.Axes = None,
    class_mapping: dict[int, str] = None,
    nodata_value: int = 0,
    colormap=cm.tab20,
    **kwargs,
):
    if ax is None:
        _, ax = plt.subplots()

    values = np.unique(array.ravel()).tolist()
    plot_values = set(values + [nodata_value])
    bounds = list(plot_values) + [max(values) + 1]
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, colormap.N)
    ax = show(array, ax=ax, cmap=colormap, norm=norm, **kwargs)

    class_mapping = class_mapping.copy() or {v: str(v) for v in values}

    if (nodata_value in values) and (nodata_value not in class_mapping):
        class_mapping[nodata_value] = "No Data"

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


def plot_samples(dataset, n: int = 3, patch_size: int = 256):
    from torch.utils.data import DataLoader
    from torchgeo.datasets import stack_samples, unbind_samples
    from torchgeo.samplers import RandomGeoSampler

    assert hasattr(dataset, "plot"), "Dataset must have a plot method"
    # sampler = GridGeoSampler(dataset, size=patch_size, stride=100)
    sampler = RandomGeoSampler(dataset, size=patch_size)
    dataloader = DataLoader(
        dataset, sampler=sampler, collate_fn=stack_samples, batch_size=n
    )
    batch = next(iter(dataloader))
    samples = unbind_samples(batch)
    for i, sample in enumerate(samples):
        dataset.plot(sample)
        plt.suptitle(f"Sample {i + 1}")
        plt.show()
