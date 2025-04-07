from pathlib import Path

import folium
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rst
from matplotlib import cm
from rasterio.plot import show
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds, get_data_window
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples, unbind_samples
from torchgeo.samplers import BatchGeoSampler

from eotorch import utils


def convert_bounds(bbox, invert_y=False):
    """
    Helper method for changing bounding box representation to leaflet notation

    ``(lon1, lat1, lon2, lat2) -> ((lat1, lon1), (lat2, lon2))``
    """
    x1, y1, x2, y2 = bbox
    if invert_y:
        y1, y2 = y2, y1
    return ((y1, x1), (y2, x2))


def plot_dataset_index(dataset, map=None, color="olive", name=None):
    """
    Plot the boundaries of a geo dataset on a folium map.

    Args:
        dataset: The geo dataset to visualize
        map: Optional existing folium map to add to
        color: Color to use for the dataset boundaries
        name: Name to use in the layer control legend (defaults to "Dataset Boundaries")

    Returns:
        folium.Map: The map with dataset boundaries visualized
    """
    objs = dataset.index.intersection(dataset.index.bounds, objects=True)

    map = map or folium.Map()

    # Create a feature group with a name for the legend
    feature_group_name = name or f"Dataset Boundaries ({color})"
    feature_group = folium.FeatureGroup(name=feature_group_name)

    style_dict = dict(fill=False, weight=5, opacity=0.7, color=color)

    for o in objs:
        folium.GeoJson(
            utils.torchgeo_bb_to_shapely(o.bounds, bbox_crs=dataset.crs),
            style_function=lambda x, style=style_dict: style,
        ).add_to(feature_group)

    # Add the feature group to the map
    feature_group.add_to(map)

    # Add click for marker functionality
    folium.ClickForMarker().add_to(map)

    # Fit bounds to show all content
    map.fit_bounds(
        bounds=convert_bounds(
            utils.torchgeo_bb_to_shapely(
                dataset.index.bounds, bbox_crs=dataset.crs
            ).bounds
        )
    )

    # Always add layer control
    folium.LayerControl().add_to(map)

    return map


def visualize_samplers(
    datasets: list,
    samplers: list,
    max_samples: int = None,
    max_files: int = None,
    class_mapping: dict = None,
):
    dataloaders = []
    for dataset, sampler in zip(datasets, samplers):
        if isinstance(sampler, BatchGeoSampler):
            batch_sampler = sampler
            sampler = None
        else:
            batch_sampler = None

        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            sampler=sampler,
            collate_fn=stack_samples,
        )
        dataloaders.append(dataloader)

    file_mapping = {}
    colors = ["r", "g", "b", "c", "m", "y", "k"]
    for j, (color, dataloader) in enumerate(zip(colors, dataloaders)):
        for i, batch in enumerate(dataloader, start=1):
            if (max_samples is not None) and (i > max_samples):
                break

            samples = unbind_samples(batch)

            for sample in samples:
                for mask_path in sample["mask_filepaths"]:
                    if mask_path not in file_mapping:
                        if max_files is not None and len(file_mapping) >= max_files:
                            continue
                        with rst.open(mask_path) as src:
                            with WarpedVRT(src, crs=datasets[j].crs) as vrt:
                                bounds = vrt.bounds
                                transform = rst.transform.from_bounds(
                                    west=bounds[0],
                                    south=bounds[1],
                                    east=bounds[2],
                                    north=bounds[3],
                                    width=vrt.width,
                                    height=vrt.height,
                                )
                                file_mapping[mask_path] = {
                                    "fig": plot_class_raster(
                                        tif_file_path=mask_path,
                                        title=mask_path,
                                        class_mapping=class_mapping,
                                    ),
                                    "height": vrt.height,
                                    "width": vrt.width,
                                    "transform": transform,
                                }
                    bounds = sample["bounds"]
                    height, width = (
                        file_mapping[mask_path]["height"],
                        file_mapping[mask_path]["width"],
                    )
                    pixel_bounds = from_bounds(
                        left=bounds[0],
                        bottom=bounds[2],
                        right=bounds[1],
                        top=bounds[3],
                        transform=file_mapping[mask_path]["transform"],
                    )
                    fig = file_mapping[mask_path]["fig"]
                    xy = (
                        pixel_bounds.col_off / width,
                        pixel_bounds.row_off / height,
                    )
                    ax = fig.gca()
                    rect = plt.Rectangle(
                        xy=xy,
                        width=pixel_bounds.width / width,
                        height=pixel_bounds.height / height,
                        fill=False,
                        color=color,
                        # alpha=0.5,
                        # zorder=1000,
                        transform=fig.transFigure,
                        # transform=ax.transAxes,
                        figure=fig,
                        lw=2,
                    )
                    ax.add_patch(rect)
    plt.show(block=True)


def plot_samplers_on_map(
    datasets: list,
    samplers: list,
    map=None,
    max_samples: int = None,
    max_files: int = None,
    colors: list = None,
    dataset_colors: list = None,
    dataset_names: list = None,
):
    """
    Visualize samplers on a folium map rather than on matplotlib plots.

    Args:
        datasets: List of datasets to visualize
        samplers: List of samplers to visualize (must match datasets length)
        map: Optional existing folium map to add to
        max_samples: Maximum number of samples to plot per dataset-sampler pair
        max_files: Maximum number of files to include in visualization
        colors: List of colors for the samples from each dataset
        dataset_colors: List of colors for the dataset boundaries
        dataset_names: List of names to use for each dataset in the legend

    Returns:
        folium.Map: The map with samplers visualized
    """
    from alive_progress import alive_it
    from torch.utils.data import DataLoader
    from torchgeo.datasets import stack_samples, unbind_samples
    from torchgeo.samplers import BatchGeoSampler

    if colors is None:
        colors = ["red", "green", "blue", "purple", "orange", "cadetblue", "black"]

    if dataset_colors is None:
        dataset_colors = [
            "olive",
            "darkgreen",
            "darkblue",
            "darkred",
            "darkpurple",
            "darkorange",
            "gray",
        ]

    if dataset_names is None:
        dataset_names = [f"Dataset {i + 1}" for i in range(len(datasets))]

    # Create a folium map if not provided
    map = map or folium.Map()

    # Dictionary to track processed files to avoid duplicates
    processed_files = {}

    # Create separate feature groups for dataset boundaries
    for i, dataset in enumerate(datasets):
        color = dataset_colors[i % len(dataset_colors)]
        name = f"{dataset_names[i]} Boundary"

        # Create a feature group for dataset boundaries
        boundary_group = folium.FeatureGroup(name=name)

        # Add dataset boundaries to the feature group
        style_dict = dict(fill=False, weight=5, opacity=0.7, color=color)

        # Add dataset boundaries without progress bar
        for o in dataset.index.intersection(dataset.index.bounds, objects=True):
            folium.GeoJson(
                utils.torchgeo_bb_to_shapely(o.bounds, bbox_crs=dataset.crs),
                style_function=lambda x, style=style_dict: style,
            ).add_to(boundary_group)

        # Add the feature group to the map
        boundary_group.add_to(map)

    # Create dataloaders
    dataloaders = []

    for dataset, sampler in zip(datasets, samplers):
        if isinstance(sampler, BatchGeoSampler):
            batch_sampler = sampler
            sampler = None
        else:
            batch_sampler = None

        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            sampler=sampler,
            collate_fn=stack_samples,
        )
        dataloaders.append(dataloader)

        # Try to estimate total number of samples for progress bar
        # if hasattr(sampler, "__len__"):
        #     total_samples_expected += len(sampler)
        # elif hasattr(batch_sampler, "__len__"):
        #     total_samples_expected += len(batch_sampler)

    # Set max_samples based on estimate if not provided

    print("Sampling bounding boxes and adding them to map...")

    # Process each dataset and sampler
    for j, (color, dataloader) in enumerate(zip(colors, dataloaders)):
        # Create a feature group for this dataset's samples
        feature_group_name = f"{dataset_names[j]} Samples ({colors[j]})"
        sample_group = folium.FeatureGroup(name=feature_group_name)

        total = (
            min(max_samples, len(samplers[j]))
            if max_samples is not None
            else len(samplers[j])
        )
        sample_count = 0

        # Create an iterator that will display progress
        dataloader_with_progress = alive_it(
            dataloader,
            total=total,
            title=f"Adding {feature_group_name}",
            force_tty=True,
            max_cols=100,
            finalize=lambda bar: bar.title(f"Added {feature_group_name}"),
        )

        for batch in dataloader_with_progress:
            samples = unbind_samples(batch)

            for sample in samples:
                # Create geometric representation of sample bounds
                bounds = sample["bounds"]
                poly = utils.torchgeo_bb_to_shapely(bounds, bbox_crs=datasets[j].crs)

                # Define style for the sample polygon
                style_function = lambda x, c=color: {
                    "fillColor": c,
                    "color": c,
                    "weight": 2,
                    "fillOpacity": 0.2,
                }

                # Add the sample polygon to the sample group
                folium.GeoJson(
                    poly,
                    style_function=style_function,
                    tooltip=f"Sample from {dataset_names[j]}",
                ).add_to(sample_group)

                sample_count += 1

                # Process mask files if needed for additional visualization
                if max_files is not None and "mask_filepaths" in sample:
                    for mask_path in sample["mask_filepaths"]:
                        if (
                            mask_path not in processed_files
                            and len(processed_files) < max_files
                        ):
                            # Mark this file as processed
                            processed_files[mask_path] = True

                # Check if we've reached the maximum sample count
                if max_samples is not None and sample_count >= max_samples:
                    break

            # Break out of batch loop if max_samples reached
            if max_samples is not None and sample_count > max_samples:
                break

        # Add the sample feature group to the map
        sample_group.add_to(map)

    # Fit the map bounds to show all content
    if datasets:
        # Use the bounds of the first dataset as a fallback
        map.fit_bounds(
            bounds=convert_bounds(
                utils.torchgeo_bb_to_shapely(
                    datasets[0].index.bounds, bbox_crs=datasets[0].crs
                ).bounds
            )
        )

    # Always add the layer control
    folium.LayerControl().add_to(map)

    return map


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
        return plot_numpy_array(
            array,
            ax=ax,
            class_mapping=class_mapping,
            nodata_value=int(src.nodata),
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

    plt.ioff()
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

    return ax.get_figure()


def plot_samples(
    dataset,
    n: int = 3,
    patch_size: int = 256,
    nodata_val: int = 0,
    show_filepaths: bool = False,
):
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
        dataset.plot(sample, show_filepaths=show_filepaths)
        plt.suptitle(f"Sample {i + 1}")
        plt.show()
