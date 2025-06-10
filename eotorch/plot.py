from pathlib import Path

import folium
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio as rst
from folium.plugins import Draw, MeasureControl
from folium.raster_layers import ImageOverlay
from matplotlib import cm
from rasterio.plot import show
from rasterio.vrt import WarpedVRT
from rasterio.windows import get_data_window
from torchgeo.datasets import IntersectionDataset

from eotorch import utils


def plot_raster_histogram(raster_file_path: str | Path, band: int = 1, bins: int = 256):
    """
    Plot a histogram of pixel values for a specific band in a raster file.
    Args:
        raster_file_path: Path to the raster file
        band: Band number to plot (default is 1)
        bins: Number of bins for the histogram (default is 256)
    """
    with rst.open(raster_file_path) as src:
        # Read the specified band
        data = src.read(band)

        # Flatten the data and remove NaN values
        data_flat = data.flatten()
        data_flat = data_flat[~np.isnan(data_flat)]

        # Plot the histogram
        plt.figure(figsize=(10, 6))
        plt.hist(data_flat, bins=bins, color="blue", alpha=0.7)
        plt.title(f"Histogram of Band {band} in {raster_file_path}")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.grid()
        plt.show()


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
    if map is None:
        map = folium.Map()
        # map = folium.Map(crs="EPSG4326")
        # Assume a new map needs controls
        has_draw_control, has_measure_control = False, False
    else:
        # Check if controls already exist on the provided map
        # We will handle LayerControl separately later
        has_draw_control = any(
            isinstance(child, Draw) for child in map._children.values()
        )
        has_measure_control = any(
            isinstance(child, MeasureControl) for child in map._children.values()
        )

    # Create a feature group with a name for the legend including the color
    base_name = f"{name} ({color})" if name else f"Dataset Boundaries ({color})"
    feature_group_name = base_name

    existing_feature_group_names = [
        getattr(child, "layer_name", None)  # Use getattr for safety
        for child in map._children.values()
        if isinstance(child, folium.FeatureGroup)
    ]

    count = 1
    while feature_group_name in existing_feature_group_names:
        feature_group_name = f"{base_name} ({count})"
        count += 1

    feature_group = folium.FeatureGroup(name=feature_group_name, show=True)

    style_dict = dict(
        fill=True, fillColor=color, fillOpacity=0.3, weight=2, opacity=0.7, color=color
    )

    current_bounds = None
    index_gdf = dataset.index

    if index_gdf.empty:
        print(
            f"Warning: No objects found in dataset '{name}'. Skipping bounds fitting and GeoJson addition."
        )
        geoms_list = []
    else:
        geoms_list = index_gdf.geometry.tolist()
        bounds = index_gdf.total_bounds  # returns [minx, miny, maxx, maxy]
        current_bounds = tuple(bounds)

    if (
        current_bounds and geoms_list
    ):  # Only add GeoJson if bounds were determined and objects exist
        for geom in geoms_list:  # Iterate over the geometries
            try:  # Add try-except for individual geometry processing
                # Get original bounds before transformation
                original_bounds = geom.bounds  # (minx, miny, maxx, maxy)

                # Convert to lat/lon if needed
                if dataset.crs != 4326:  # Assuming CRS 4326 is WGS84
                    # Transform geometry to WGS84 for folium
                    # gdf_temp = gpd.GeoDataFrame([1], geometry=[geom], crs="EPSG:4326")
                    gdf_temp = gpd.GeoDataFrame([1], geometry=[geom], crs=dataset.crs)
                    gdf_temp = gdf_temp.to_crs(4326)
                    geom_transformed = gdf_temp.geometry.iloc[0]
                    transformed_bounds = geom_transformed.bounds
                else:
                    geom_transformed = geom
                    transformed_bounds = original_bounds

                def create_style_function(style):
                    return lambda x: style

                # Create tooltip with bounds information
                tooltip_text = (
                    f"Bounds ({dataset.crs.name}):<br>"
                    f"Min X: {original_bounds[0]:.6f}<br>"
                    f"Min Y: {original_bounds[1]:.6f}<br>"
                    f"Max X: {original_bounds[2]:.6f}<br>"
                    f"Max Y: {original_bounds[3]:.6f}"
                )

                if dataset.crs != 4326:
                    tooltip_text += (
                        f"<br><br>Bounds (WGS84):<br>"
                        f"Min Lon: {transformed_bounds[0]:.6f}<br>"
                        f"Min Lat: {transformed_bounds[1]:.6f}<br>"
                        f"Max Lon: {transformed_bounds[2]:.6f}<br>"
                        f"Max Lat: {transformed_bounds[3]:.6f}"
                    )

                folium.GeoJson(
                    geom_transformed,
                    style_function=create_style_function(style_dict),
                    tooltip=tooltip_text,
                ).add_to(feature_group)
            except Exception as e:
                print(
                    f"Warning: Could not process geometry in dataset '{name}'. Error: {e}"
                )

    # Add the feature group to the map (even if empty, so it appears in LayerControl)
    feature_group.add_to(map)

    # Fit bounds only if we successfully added geometries
    if current_bounds:
        map_bounds = map.get_bounds()
        if map_bounds:
            map.fit_bounds(map_bounds)
        else:  # Fallback if map has no bounds yet (first dataset)
            map.fit_bounds(utils.convert_bounds(current_bounds))

    # Add Draw and Measure controls only if they are needed
    if not has_draw_control:
        Draw(
            export=True,
            draw_options={
                "polyline": False,
                "polygon": False,
                "circle": False,
                "marker": False,
                "circlemarker": False,
            },
        ).add_to(map)
    if not has_measure_control:
        MeasureControl(position="bottomleft").add_to(map)

    # Remove existing LayerControl if present, then add a new one
    # This ensures the control reflects all added layers
    existing_lc = None
    for key, child in list(map._children.items()):  # Iterate over a copy of items
        if isinstance(child, folium.LayerControl):
            existing_lc = key
            break
    if existing_lc:
        del map._children[existing_lc]

    folium.LayerControl().add_to(map)  # Always add/re-add LayerControl

    return map


def plot_samplers_on_map(
    datasets: list,
    samplers: list,
    map=None,
    max_samples: int = None,
    max_files: int = None,
    colors: list = None,
    dataset_colors: list = None,
    dataset_names: list = None,
    count_labels: bool = False,
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
        count_labels: If True, count the distribution of label pixels across samples

    Returns:
        If count_labels is False:
            folium.Map: The map with samplers visualized
        If count_labels is True:
            tuple: (folium.Map, dict) containing the map and label counts per dataset
    """
    import torch
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

    # Dictionary to track label counts if requested, one dict per dataset
    label_counts_per_dataset = (
        {name: {} for name in dataset_names} if count_labels else None
    )

    # Create separate feature groups for dataset boundaries
    for i, dataset in enumerate(datasets):
        color = dataset_colors[i % len(dataset_colors)]
        name = f"{dataset_names[i]} Boundary"

        # Create a feature group for dataset boundaries
        boundary_group = folium.FeatureGroup(name=name)

        # Add dataset boundaries to the feature group
        style_dict = dict(fill=False, weight=5, opacity=0.7, color=color)

        # Add dataset boundaries without progress bar
        for _, row in dataset.index.iterrows():
            geom_bounds = row.geometry.bounds
            folium.GeoJson(
                utils.torchgeo_bb_to_shapely(geom_bounds, bbox_crs=dataset.crs),
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

    # Process each dataset and sampler
    print("Sampling bounding boxes and adding them to map...")

    for dataset_idx, (color, dataloader) in enumerate(zip(colors, dataloaders)):
        # Create a feature group for this dataset's samples
        current_dataset_name = dataset_names[dataset_idx]
        feature_group_name = f"{current_dataset_name} Samples ({colors[dataset_idx]})"
        sample_group = folium.FeatureGroup(name=feature_group_name)

        total = (
            min(max_samples, len(samplers[dataset_idx]))
            if max_samples is not None
            else len(samplers[dataset_idx])
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
                poly = utils.torchgeo_bb_to_shapely(
                    bounds, bbox_crs=datasets[dataset_idx].crs
                )

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
                    tooltip=f"Sample from {current_dataset_name}",
                ).add_to(sample_group)

                # Count label pixels if requested
                if count_labels and "mask" in sample:
                    mask = sample["mask"]
                    # Skip -1 values which are typically ignored or no-data values
                    valid_mask = mask != -1

                    # Adjust mask values if IntersectionDataset with reduce_zero_label=True was used
                    # This ensures label counts match the original class values if 0 was mapped to -1
                    current_dataset = datasets[dataset_idx]
                    if isinstance(current_dataset, IntersectionDataset) and hasattr(
                        current_dataset.datasets[1], "reduce_zero_label"
                    ):
                        if current_dataset.datasets[1].reduce_zero_label:
                            mask = mask + 1  # Shift labels back up by 1

                    if torch.any(valid_mask):
                        # Only count valid (non -1) pixels
                        filtered_mask = mask[valid_mask]
                        unique_values, counts = torch.unique(
                            filtered_mask, return_counts=True
                        )

                        # Update counts for this dataset
                        dataset_label_counts = label_counts_per_dataset[
                            current_dataset_name
                        ]
                        for val, count in zip(
                            unique_values.cpu().numpy(), counts.cpu().numpy()
                        ):
                            # Aggregate counts for each label value
                            dataset_label_counts[val] = (
                                dataset_label_counts.get(val, 0) + count
                            )

                sample_count += 1

                # Process mask files if needed for additional visualization (e.g., debugging)
                # Limits the number of unique mask files processed across all samples/datasets
                if max_files is not None and "mask_filepaths" in sample:
                    for mask_path in sample["mask_filepaths"]:
                        if (
                            mask_path not in processed_files
                            and len(processed_files) < max_files
                        ):
                            # Mark this file as processed to avoid duplicates
                            processed_files[mask_path] = True

                # Check if we've reached the maximum sample count for this dataset
                if max_samples is not None and sample_count >= max_samples:
                    break

            # Break out of batch loop if max_samples reached for this dataset
            if max_samples is not None and sample_count >= max_samples:
                break

        # Add the sample feature group to the map
        sample_group.add_to(map)

    # Fit map bounds to encompass all datasets
    try:
        all_bounds = []
        for dataset in datasets:
            if not dataset.index.empty:
                # Get bounds in the dataset's original CRS
                dataset_bounds = (
                    dataset.index.total_bounds
                )  # [minx, miny, maxx, maxy] numpy array

                # Convert to WGS84 (EPSG:4326) for folium
                if dataset.crs != pyproj.CRS("EPSG:4326"):
                    # Convert numpy array to tuple to ensure correct format interpretation
                    # total_bounds returns [minx, miny, maxx, maxy], convert to tuple for correct handling
                    bounds_tuple = tuple(dataset_bounds)  # (minx, miny, maxx, maxy)
                    bounds_poly = utils.torchgeo_bb_to_shapely(
                        bounds_tuple, bbox_crs=dataset.crs, target_crs="EPSG:4326"
                    )
                    transformed_bounds = bounds_poly.bounds  # (minx, miny, maxx, maxy)
                else:
                    transformed_bounds = tuple(dataset_bounds)

                all_bounds.append(transformed_bounds)

        if all_bounds:
            # Calculate the overall bounds encompassing all datasets
            min_x = min(bounds[0] for bounds in all_bounds)
            min_y = min(bounds[1] for bounds in all_bounds)
            max_x = max(bounds[2] for bounds in all_bounds)
            max_y = max(bounds[3] for bounds in all_bounds)

            # Convert to folium format: ((lat1, lon1), (lat2, lon2))
            # Note: folium expects (lat, lon) which is (y, x)
            folium_bounds = ((min_y, min_x), (max_y, max_x))
            map.fit_bounds(bounds=folium_bounds)

    except Exception as e:
        print(f"Warning: Could not fit map bounds for datasets. Error: {e}")
        # Continue without fitting bounds

    # Always add the layer control
    folium.LayerControl().add_to(map)

    Draw(
        export=True,
        draw_options={
            "polyline": False,
            "polygon": False,
            "circle": False,
            "marker": False,
            "circlemarker": False,
        },
    ).add_to(map)
    MeasureControl(position="bottomleft").add_to(map)

    if count_labels:
        return map, label_counts_per_dataset
    else:
        return map


def plot_class_raster(
    tif_file_path: str | Path,
    ax: plt.Axes = None,
    class_mapping: dict[int, str] = None,
    colormap=cm.tab20,
    data_window_only: bool = False,
    **kwargs,
) -> plt.Axes:  # Return the axes object
    """
    Plot a raster image with class labels. Manages figure creation and display.
    """
    show_plot = False
    if ax is None:
        fig, ax = plt.subplots()
        show_plot = True

    with rst.open(tif_file_path) as src:
        array = src.read(1)
        nodata = src.nodata  # Get nodata value from source

        # Call plot_numpy_array, passing the determined ax and nodata value
        plot_numpy_array(
            array,
            ax=ax,  # Always pass an ax
            class_mapping=class_mapping,
            nodata_value=int(nodata)
            if nodata is not None
            else None,  # Pass nodata value
            colormap=colormap,
            data_window_only=data_window_only,
            **kwargs,
        )

    if show_plot:
        plt.show()

    return ax  # Return the axes object used for plotting


def plot_numpy_array(
    array: np.ndarray,
    ax: plt.Axes = None,
    class_mapping: dict[int, str] = None,
    nodata_value: int = None,  # Changed: Allow None
    colormap=cm.tab20,
    data_window_only: bool = False,
    nodata_color: str = "black",
    **kwargs,
):
    """
    Plot a numpy array with class labels onto a given Axes object.

    Args:
        array: The input array to plot
        ax: Matplotlib axes to plot on.
        class_mapping: Dictionary mapping class values to their string labels
        nodata_value: Value to treat as no data. If None, treat as no nodata.
        colormap: Matplotlib colormap to use for data values
        data_window_only: If True, crops to non-nodata region
        nodata_color: Color to use for the nodata value.
        **kwargs: Additional arguments for rasterio's show function
    """
    if ax is None:
        fig, ax = plt.subplots()

    if data_window_only and nodata_value is not None:
        window = get_data_window(array, nodata=nodata_value)
        plot_array = array[utils.window_to_np_idc(window)]
    else:
        plot_array = array

    # Create a masked array if nodata_value is specified
    if nodata_value is not None:
        masked_array = np.ma.masked_equal(plot_array, nodata_value)
        nodata_present = nodata_value in np.unique(plot_array.ravel())
        array_to_plot = masked_array  # Plot the masked array
    else:
        # If no nodata value, plot the original array and don't use masking features
        masked_array = plot_array  # Use unmasked array for calculations
        nodata_present = False
        array_to_plot = plot_array  # Plot the unmasked array

    # Get unique values present in the data (excluding nodata if applicable)
    data_values = np.unique(
        masked_array.compressed() if nodata_value is not None else masked_array.ravel()
    ).tolist()

    # Determine min/max class values based *only* on data values and mapping
    potential_data_values = set(data_values)
    if class_mapping:
        potential_data_values.update(
            k for k in class_mapping.keys() if nodata_value is None or k != nodata_value
        )

    numeric_data_values = {
        v for v in potential_data_values if isinstance(v, (int, np.integer))
    }

    cmap = plt.get_cmap(colormap).copy()
    if nodata_value is not None:
        cmap.set_bad(color=nodata_color)  # Set nodata color only if nodata exists

    if not numeric_data_values:
        # Handle case with only nodata or empty array
        norm = None
        # Plot showing only nodata color or blank if no nodata
        show(array_to_plot, ax=ax, cmap=cmap, vmin=-1, vmax=0, **kwargs)

    else:
        min_class = min(numeric_data_values)
        max_class = max(numeric_data_values)

        bounds = np.arange(min_class - 0.5, max_class + 1.5)
        if len(bounds) < 2:
            bounds = np.array([min_class - 0.5, min_class + 0.5])
            max_class = min_class

        norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        # Pass the array_to_plot (masked or unmasked)
        show(array_to_plot, ax=ax, cmap=cmap, norm=norm, **kwargs)

    # --- Legend Handling ---
    legend_patches = []
    effective_class_mapping = {}
    if class_mapping:
        effective_class_mapping.update(
            {
                k: v
                for k, v in class_mapping.items()
                if nodata_value is None or k != nodata_value
            }
        )

    for v in data_values:
        if v not in effective_class_mapping:
            effective_class_mapping[v] = str(v)

    present_data_values = sorted(
        list(set(v for v in data_values if isinstance(v, (int, np.integer))))
    )

    if norm:
        for i in present_data_values:
            if i >= bounds[0] and i < bounds[-1]:
                color = cmap(norm(i))
                label = effective_class_mapping.get(i, str(i))
                legend_patches.append(mpatches.Patch(color=color, label=label))

    if nodata_present:  # Add nodata legend only if it was actually present
        legend_patches.append(mpatches.Patch(color=nodata_color, label="No Data"))

    if legend_patches:
        ax.legend(
            handles=legend_patches,
            bbox_to_anchor=(1.05, 1),
            loc=2,
            borderaxespad=0.0,
        )


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
    sampler = RandomGeoSampler(
        dataset,
        size=patch_size,
    )
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
                (("image" in sample) and (sample["image"] == 0).all())
                or ("mask" in sample and (sample["mask"] == nodata_val).all())
                or ("mask" in sample and (sample["mask"] == -1).all())
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
        if isinstance(dataset, IntersectionDataset):
            plt.suptitle(f"Sample {i + 1}")
        else:
            if not show_filepaths:
                plt.title(f"Sample {i + 1}")
        plt.show()


def label_map(label_file_paths: str | list, map: folium.Map = None):
    """
    Visualize label maps on a folium map.

    Args:
        label_file_paths: List of paths to label files
        map: Optional existing folium map to add to

    Returns:
        folium.Map: The map with label maps visualized
    """
    map = map or folium.Map()

    if not isinstance(label_file_paths, list):
        label_file_paths = [label_file_paths]

    # Keep track of bounds
    bounds_list = []

    # Pre-compute the colormap once - using tab20 which has good color distinction for categorical data
    # Create a static dictionary for faster lookup
    tab20_colors = {}
    for i in range(20):
        # Pre-compute and store as RGB float values
        tab20_colors[i] = [float(c) for c in cm.tab20(i / 20.0)[:3]]

    # Fast lookup colormap function
    def fast_colormap(x):
        return tab20_colors[int(x) % 20]

    # Process files in a single loop
    for label_file_path in label_file_paths:
        with rst.open(label_file_path) as src:
            # Read the data and get bounds in a single operation with VRT
            with WarpedVRT(src, crs="EPSG:4326") as vrt:
                bounds = utils.convert_bounds(
                    vrt.bounds
                )  # Convert directly to save a step
                array = vrt.read(1)

        # Store bounds for map fitting
        bounds_list.append(bounds)

        # Create overlay and add to map in one step
        ImageOverlay(
            image=array,
            bounds=bounds,
            colormap=fast_colormap,
            opacity=0.7,
            name=Path(label_file_path).name,
            mercator_project=True,
        ).add_to(map)

    # Calculate map bounds if we have any overlays
    if bounds_list:
        min_lat = min(bounds[0][0] for bounds in bounds_list)
        max_lat = max(bounds[1][0] for bounds in bounds_list)
        min_lon = min(bounds[0][1] for bounds in bounds_list)
        max_lon = max(bounds[1][1] for bounds in bounds_list)
        map.fit_bounds(bounds=((min_lat, min_lon), (max_lat, max_lon)))

    # Add controls
    folium.LayerControl().add_to(map)
    Draw(
        export=True,
        draw_options={
            "polyline": False,
            "polygon": False,
            "circle": False,
            "marker": False,
            "circlemarker": False,
        },
    ).add_to(map)
    MeasureControl(position="bottomleft").add_to(map)

    return map
