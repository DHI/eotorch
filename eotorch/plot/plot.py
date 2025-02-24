from pathlib import Path

import numpy as np
import rasterio as rst
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchgeo.datasets import GeoDataset, stack_samples, unbind_samples
from torchgeo.samplers import RandomGeoSampler


def plot_predictions_pyplot(
    predictions_path: str | Path, classes: dict[int, str], ax: plt.Axes = None
):
    """
    Plot predictions using matplotlib.pyplot.

    Parameters
    ----------
        predictions_path: str or Path, path to the predictions file
        classes: dict, mapping of class indices to class names
        ax: matplotlib.axes.Axes, axes to plot on
    """

    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(nrows=1, ncols=1)

    with rst.open(predictions_path) as src:
        predictions = src.read(1)
        no_data_val = src.nodata
    im = ax.imshow(predictions, interpolation="none")
    values = np.unique(predictions.ravel()).tolist()

    if no_data_val not in values:
        plot_values = values + [no_data_val]
    else:
        plot_values = values

    if not classes:
        classes = {v: str(v) for v in values}
    if (no_data_val in values) and (no_data_val not in classes):
        classes[no_data_val] = "No Data"

    colors = [im.cmap(im.norm(value)) for value in plot_values]
    patches = [
        mpatches.Patch(color=colors[i], label=classes[i])
        for i in values
        # mpatches.Patch(color=colors[i - 1], label=classes[i - 1]) for i in values
    ]
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)


def plot_samples(dataset: GeoDataset, n: int = 3, patch_size: int = 256):
    assert hasattr(dataset, "plot"), "Dataset must have a plot method"
    # sampler = GridGeoSampler(dataset, size=patch_size, stride=100)
    sampler = RandomGeoSampler(dataset, size=patch_size)
    dataloader = DataLoader(
        dataset, sampler=sampler, collate_fn=stack_samples, batch_size=n
    )
    batch = next(iter(dataloader))
    samples = unbind_samples(batch)
    for i, sample in enumerate(samples):
        dataset.plot(sample, title=f"Sample {i + 1}")
        plt.show()
