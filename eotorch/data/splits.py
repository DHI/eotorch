from copy import deepcopy
from math import isclose
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import geopandas as gpd
import pandas as pd
import torch
from pyproj import CRS, Transformer
from shapely.geometry import box as shapely_box
from torch import Generator, default_generator, randperm
from torchgeo.datasets import GeoDataset, IntersectionDataset, RasterDataset
from torchgeo.datasets.splits import _fractions_to_lengths

from eotorch.data.geodatasets import LabelledRasterDataset


def _format_paths(files: Union[str, Path, List[str], List[Path], None]) -> List[Path]:
    """Formats various file input types into a list of resolved Path objects."""
    if files is None:
        return []
    if isinstance(files, (str, Path)):
        files = [files]
    if isinstance(files, list):
        files = [Path(i) for i in files]
    return [i.resolve() for i in files]


def _get_split_items(
    dataset_items: List[Tuple[Any, Any]],  # (geometry, filepath) tuples
    dataset_files: List[Path],
    val_files: List[Path],
    test_files: List[Path],
) -> Tuple[List[Tuple[Any, Any]], List[Tuple[Any, Any]], List[Tuple[Any, Any]]]:
    """Validates split files and returns items partitioned into train, val, test."""

    def _validate_and_collect_items(files_param: List[Path]) -> List[Tuple[Any, Any]]:
        """Checks if files exist in the dataset and collects corresponding items."""
        out_list = []
        file_found = False
        for f in files_param:
            try:
                idx = dataset_files.index(f)
                out_list.append(dataset_items[idx])
                file_found = True
            except ValueError:
                print(f"Warning: File {str(f)} not found in dataset. Skipping.")

        if files_param and not file_found:
            raise ValueError(
                "None of the provided files were found in the dataset. Please check the files."
            )
        return out_list

    val_items = _validate_and_collect_items(val_files)
    test_items = _validate_and_collect_items(test_files)

    val_test_items = set(val_items + test_items)
    train_items = [item for item in dataset_items if item not in val_test_items]

    return train_items, val_items, test_items


def _build_split_indexes(
    train_items: List[Tuple[Any, Any]],
    val_items: List[Tuple[Any, Any]],
    test_items: List[Tuple[Any, Any]],
    original_index: gpd.GeoDataFrame,
) -> List[gpd.GeoDataFrame]:
    """Builds GeoPandas GeoDataFrame indexes for train, validation, and test splits."""
    new_indexes = []

    def _create_index(items: List[Tuple[Any, Any]]) -> gpd.GeoDataFrame:
        """Helper to create and populate a GeoPandas index."""
        if not items:
            # Return empty GeoDataFrame with correct structure and temporal index
            empty_gdf = gpd.GeoDataFrame(
                {"filepath": []}, geometry=[], crs=original_index.crs
            )
            # Create empty IntervalIndex to match original structure
            empty_temporal_index = pd.IntervalIndex.from_tuples(
                [], closed="both", name="datetime"
            )
            empty_gdf.index = empty_temporal_index
            return empty_gdf

        # Extract filepaths from items to find corresponding rows in original index
        item_filepaths = {filepath for _, filepath in items}

        # Find matching rows in original index
        mask = original_index["filepath"].isin(item_filepaths)
        filtered_index = original_index[mask].copy()

        return filtered_index

    if train_items:
        new_indexes.append(_create_index(train_items))
    if val_items:
        new_indexes.append(_create_index(val_items))
    if test_items:
        new_indexes.append(_create_index(test_items))

    return new_indexes


def _build_split_datasets(
    original_dataset: RasterDataset, new_indexes: List[gpd.GeoDataFrame]
) -> List[RasterDataset]:
    """Builds the final list of split datasets using the provided indexes."""
    new_datasets = []
    for index in new_indexes:
        if isinstance(original_dataset, IntersectionDataset):
            # Handle IntersectionDataset (e.g., LabelledRasterDataset)
            img_ds_orig = original_dataset.datasets[0]
            label_ds_orig = original_dataset.datasets[1]

            # Create new image dataset for the split
            img_ds_new = deepcopy(img_ds_orig)
            img_ds_new.index = index
            if len(index) > 0:
                img_ds_new.paths = index["filepath"].tolist()
            else:
                img_ds_new.paths = []

            # Create new label dataset index based on intersection with new image dataset bounds
            img_bounds = img_ds_new.bounds

            # For GeoPandas, we need to intersect with the image bounds
            if not label_ds_orig.index.empty:
                # Create bounding box geometry for intersection
                img_bbox_geom = shapely_box(
                    img_bounds.minx, img_bounds.miny, img_bounds.maxx, img_bounds.maxy
                )

                # Find labels that intersect with the image bounds
                label_intersections = label_ds_orig.index[
                    label_ds_orig.index.geometry.intersects(img_bbox_geom)
                ]

                # Create new label dataset for the split
                label_ds_new = deepcopy(label_ds_orig)
                label_ds_new.index = label_intersections.copy()
                if len(label_intersections) > 0:
                    label_ds_new.paths = label_intersections["filepath"].tolist()
                else:
                    label_ds_new.paths = []
            else:
                # Empty label dataset
                label_ds_new = deepcopy(label_ds_orig)
                empty_gdf = gpd.GeoDataFrame(
                    {"filepath": []}, geometry=[], crs=label_ds_orig.index.crs
                )
                # Create empty IntervalIndex to match original structure
                empty_temporal_index = pd.IntervalIndex.from_tuples(
                    [], closed="both", name="datetime"
                )
                empty_gdf.index = empty_temporal_index
                label_ds_new.index = empty_gdf
                label_ds_new.paths = []

            # Create the final IntersectionDataset for the split
            ds = original_dataset.__class__(
                img_ds_new,
                label_ds_new,
                collate_fn=original_dataset.collate_fn,
                transforms=original_dataset.transforms,
            )
        else:
            # Handle single RasterDataset
            ds = deepcopy(original_dataset)
            ds.index = index
            if len(index) > 0:
                ds.paths = index["filepath"].tolist()
            else:
                ds.paths = []

        new_datasets.append(ds)
    return new_datasets


def file_wise_split(
    dataset: RasterDataset,
    val_img_files: Union[str, Path, List[str], List[Path], None] = None,
    test_img_files: Union[str, Path, List[str], List[Path], None] = None,
    ratios_or_counts: Optional[Sequence[float]] = None,
) -> List[RasterDataset]:
    """
    Splits a dataset based on image files, ensuring no file overlap between splits.

    Args:
        dataset: The dataset to be split (RasterDataset or IntersectionDataset).
        val_img_files: Files for the validation set.
        test_img_files: Files for the test set.
        ratios_or_counts: Ratios or counts for random splitting (overrides file lists).

    Returns:
        A list of datasets [train, validation (optional), test (optional)].

    Raises:
        ValueError: If input parameters are invalid or files are not found.
    """
    use_ratios = ratios_or_counts is not None
    use_files = (val_img_files is not None) or (test_img_files is not None)

    if not use_ratios and not use_files:
        raise ValueError(
            "Either ratios_or_counts or val_img_files/test_img_files must be provided."
        )

    if use_ratios and use_files:
        print(
            "Warning: Both ratios_or_counts and file lists provided. "
            "Using ratios_or_counts for splitting."
        )
        use_files = False

    if use_ratios:
        # Perform ratio-based random split
        return random_bbox_assignment(
            dataset=dataset,
            lengths=ratios_or_counts,
            generator=torch.Generator().manual_seed(0),
        )

    # Perform file-based split
    val_files = _format_paths(val_img_files)
    test_files = _format_paths(test_img_files)

    # Determine the base image dataset
    if isinstance(dataset, IntersectionDataset):
        img_dataset = dataset.datasets[0]
    elif isinstance(dataset, RasterDataset):
        img_dataset = dataset
    else:
        raise TypeError(
            "Dataset must be a RasterDataset or IntersectionDataset subclass."
        )

    # Get all items and corresponding file paths from the image dataset index
    dataset_items = [
        (row.geometry, row.filepath) for _, row in img_dataset.index.iterrows()
    ]
    dataset_files = [Path(filepath).resolve() for _, filepath in dataset_items]

    # Partition items into train, validation, and test sets
    train_items, val_items, test_items = _get_split_items(
        dataset_items, dataset_files, val_files, test_files
    )

    # Build GeoPandas indexes for each split
    new_indexes = _build_split_indexes(
        train_items, val_items, test_items, img_dataset.index
    )

    # Build the final datasets using the original dataset structure and new indexes
    new_datasets = _build_split_datasets(dataset, new_indexes)

    return new_datasets


def random_bbox_assignment(
    dataset: GeoDataset,
    lengths: Sequence[float],
    generator: Generator | None = default_generator,
) -> list[GeoDataset]:
    """
    Reimplementation of the random_bbox_assignment function from torchgeo, but with assurance that
    if a split percentage is provided, at least one bbox is assigned to each split.

    Split a GeoDataset randomly assigning its index's BoundingBoxes.

    This function will go through each BoundingBox in the GeoDataset's index and
    randomly assign it to new GeoDatasets.

    Args:
        dataset: dataset to be split
        lengths: lengths or fractions of splits to be produced
        generator: (optional) generator used for the random permutation

    Returns:
        A list of the subset datasets.
    """

    if not (isclose(sum(lengths), 1) or isclose(sum(lengths), len(dataset))):
        raise ValueError(
            "Sum of input lengths must equal 1 or the length of dataset's index."
        )

    if any(n <= 0 for n in lengths):
        raise ValueError("All items in input lengths must be greater than 0.")

    if isclose(sum(lengths), 1):
        lengths = _fractions_to_lengths(lengths, len(dataset))
    lengths = cast(Sequence[int], lengths)
    zero_length_idces = [i for i, length in enumerate(lengths) if length == 0]
    non_zero_length_idces = [i for i, length in enumerate(lengths) if length > 0]
    if zero_length_idces:
        for i in zero_length_idces:
            lengths[i] = 1
            lengths[non_zero_length_idces[0]] -= 1
            if isclose(lengths[non_zero_length_idces[0]], 0):
                raise ValueError(
                    "Not enough bboxes to assign to all splits. "
                    "Please provide a larger dataset or reduce the number of splits."
                )

    # Get all items from the dataset index
    items_data = [(row.geometry, row.filepath) for _, row in dataset.index.iterrows()]

    # Randomly permute the items
    indices = randperm(len(items_data), generator=generator)
    shuffled_items = [items_data[i] for i in indices]

    # Create new indexes by filtering the original index
    new_indexes = []
    start_idx = 0

    for length in lengths:
        end_idx = start_idx + length
        split_items = shuffled_items[start_idx:end_idx]

        if split_items:
            # Extract filepaths to filter original index
            split_filepaths = {filepath for _, filepath in split_items}
            mask = dataset.index["filepath"].isin(split_filepaths)
            filtered_index = dataset.index[mask].copy()
        else:
            # Create empty GeoDataFrame with correct structure
            empty_gdf = gpd.GeoDataFrame(
                {"filepath": []}, geometry=[], crs=dataset.index.crs
            )
            # Create empty IntervalIndex to match original structure
            empty_temporal_index = pd.IntervalIndex.from_tuples(
                [], closed="both", name="datetime"
            )
            empty_gdf.index = empty_temporal_index
            filtered_index = empty_gdf

        new_indexes.append(filtered_index)
        start_idx = end_idx

    # Create new datasets
    new_datasets = []
    for index in new_indexes:
        ds = deepcopy(dataset)
        ds.index = index
        if len(index) > 0:
            ds.paths = index["filepath"].tolist()
        else:
            ds.paths = []
        new_datasets.append(ds)

    return new_datasets


def aoi_split(
    dataset: GeoDataset,
    val_aois: Union[
        Sequence[Tuple[float, float, float, float]],
        str,
        Path,
        None,
    ] = None,
    test_aois: Union[
        Sequence[Tuple[float, float, float, float]],
        str,
        Path,
        None,
    ] = None,
    as_lists: bool = False,
    crs: Optional[Union[str, Dict[str, Any]]] = "EPSG:4326",
    buffer_size: float = 0.0,
) -> List[GeoDataset]:
    """Split a GeoDataset into train, validation, and test datasets based on areas of interest (AOIs).

    This function performs a purely spatial split and preserves the temporal information
    from the original dataset exactly as it is.

    This function divides a dataset into multiple datasets:
    - First dataset: Contains all data outside the provided AOIs (training dataset)
    - Second dataset (optional): Contains data that intersects with validation AOIs
    - Third dataset (optional): Contains data that intersects with test AOIs

    The AOIs can be provided in different formats:
    - List of (minx, maxx, miny, maxy) tuples (spatial only)
    - Path to a GeoJSON file containing Polygon features

    The function supports transforming between coordinate reference systems. If your AOIs are
    specified in lat/lon (EPSG:4326) but your dataset is in a different CRS, you can specify
    the CRS parameters to handle the transformation automatically.

    A buffer can be applied around the AOIs to ensure that samples from training and validation/test
    datasets do not overlap. The buffer size is specified in the same units as the dataset's CRS
    (typically meters).

    Args:
        dataset: Dataset to be split
        val_aois: Sequence of validation areas of interest as (minx, maxx, miny, maxy) tuples,
                  or a path to a GeoJSON file containing validation regions
        test_aois: Sequence of test areas of interest as (minx, maxx, miny, maxy) tuples,
                   or a path to a GeoJSON file containing test regions
        as_lists: If True, return a list of train items and lists of val/test items
                 instead of creating new datasets
        crs: The CRS of the input AOIs (e.g., "EPSG:4326" for lat/lon). If None,
             assumes AOIs are in the same CRS as the dataset. Default is "EPSG:4326".
        buffer_size: Size of buffer (in the dataset's CRS units, typically meters)
                    to apply around AOIs. This creates a gap between training and validation/test
                    regions to prevent patch overlap. Default is 0.0 (no buffer).

    Returns:
        A list of datasets, where the first is the training dataset (outside all AOIs),
        the second (if val_aois is not None) is the validation dataset, and
        the third (if test_aois is not None) is the test dataset.
        If as_lists is True, returns lists of items for each dataset instead.

    Raises:
        ValueError: If validation and test AOIs overlap with each other
        ImportError: If coordinate transformation is requested but pyproj is not available
        ValueError: If a GeoJSON file is invalid or doesn't contain Polygon features
    """

    if isinstance(dataset, LabelledRasterDataset):
        # If the dataset is a LabelledRasterDataset, we need to split the image and label datasets

        img_datasets = aoi_split(
            dataset.datasets[0],
            val_aois=val_aois,
            test_aois=test_aois,
            as_lists=as_lists,
            crs=crs,
            buffer_size=buffer_size,
        )
        label_datasets = aoi_split(
            dataset.datasets[1],
            val_aois=val_aois,
            test_aois=test_aois,
            as_lists=as_lists,
            crs=crs,
            buffer_size=buffer_size,
        )

        datasets_to_return = []
        for img_ds, label_ds in zip(img_datasets, label_datasets):
            datasets_to_return.append(
                LabelledRasterDataset(
                    img_ds,
                    label_ds,
                    collate_fn=dataset.collate_fn,
                    transforms=dataset.transforms,
                )
            )

        return datasets_to_return

    # Process AOIs
    roi_polygons = []  # Will hold all Polygon objects
    buffered_roi_polygons = []  # Will hold buffered versions of all ROIs
    aoi_categories = []  # Will track whether each ROI is for validation (0) or test (1)

    # Process validation AOIs if provided
    if val_aois is not None:
        val_roi_polygons, val_buffered_polygons = _process_aois(
            val_aois, dataset, crs, buffer_size
        )
        roi_polygons.extend(val_roi_polygons)
        buffered_roi_polygons.extend(val_buffered_polygons)
        aoi_categories.extend([0] * len(val_roi_polygons))  # 0 for validation

    # Process test AOIs if provided
    if test_aois is not None:
        test_roi_polygons, test_buffered_polygons = _process_aois(
            test_aois, dataset, crs, buffer_size
        )
        roi_polygons.extend(test_roi_polygons)
        buffered_roi_polygons.extend(test_buffered_polygons)
        aoi_categories.extend([1] * len(test_roi_polygons))  # 1 for test

    # If no AOIs were provided, return the original dataset
    if not roi_polygons:
        if as_lists:
            # Get all items from the dataset index
            all_hits = [
                (row.geometry, row.filepath) for _, row in dataset.index.iterrows()
            ]
            return [all_hits]
        return [dataset]

    # Check for overlapping AOIs between validation and test sets
    if val_aois is not None and test_aois is not None:
        val_count = len([c for c in aoi_categories if c == 0])
        for i in range(val_count):
            for j in range(val_count, len(roi_polygons)):
                if (
                    roi_polygons[i].intersects(roi_polygons[j])
                    and not roi_polygons[i].intersection(roi_polygons[j]).is_empty
                ):
                    raise ValueError("Validation and test AOIs cannot overlap.")

    # Create indexes for train and AOI categories using GeoPandas
    train_items = []
    category_items = [[] for _ in range(len(set(aoi_categories)))]

    # Get all items from dataset
    all_items = [(row.geometry, row.filepath) for _, row in dataset.index.iterrows()]

    # Process each item in the dataset
    for geom, filepath in all_items:
        # Check if this item is in any AOI
        for aoi_idx, (roi, buffered_roi) in enumerate(
            zip(roi_polygons, buffered_roi_polygons)
        ):
            category = aoi_categories[aoi_idx]
            # Map category to the correct index in category_items
            if val_aois is not None and test_aois is not None:
                category_idx = category  # Both validation (0) and test (1) are present
            elif val_aois is not None:
                category_idx = 0  # Only validation, so it's at index 0
            else:  # test_aois is not None
                category_idx = 0  # Only test, so it's at index 0

            # Check intersection with original ROI for AOI datasets
            if geom.intersects(roi):
                category_items[category_idx].append((geom, filepath))
                break  # Item assigned to an AOI, don't check others

        # If item doesn't intersect any AOI, check if it should be added to training
        # (i.e., it doesn't intersect any buffered ROI)
        else:
            in_buffered_roi = any(
                geom.intersects(buffered_roi) for buffered_roi in buffered_roi_polygons
            )
            if not in_buffered_roi:
                train_items.append((geom, filepath))

    if as_lists:
        result = [train_items]
        for items in category_items:
            result.append(items)
        return result

    # Create datasets from the items
    new_datasets = []

    # Create train dataset by filtering original index
    if train_items:
        train_filepaths = {filepath for _, filepath in train_items}
        mask = dataset.index["filepath"].isin(train_filepaths)
        train_gdf = dataset.index[mask].copy()
    else:
        # Create empty GeoDataFrame with correct structure
        train_gdf = gpd.GeoDataFrame(
            {"filepath": []}, geometry=[], crs=dataset.index.crs
        )
        # Create empty IntervalIndex to match original structure
        empty_temporal_index = pd.IntervalIndex.from_tuples(
            [], closed="both", name="datetime"
        )
        train_gdf.index = empty_temporal_index

    train_ds = deepcopy(dataset)
    train_ds.index = train_gdf
    if len(train_gdf) > 0:
        train_ds.paths = train_gdf["filepath"].tolist()
    else:
        train_ds.paths = []
    new_datasets.append(train_ds)

    # Create category datasets (validation and/or test) by filtering original index
    for items in category_items:
        if items:
            cat_filepaths = {filepath for _, filepath in items}
            mask = dataset.index["filepath"].isin(cat_filepaths)
            cat_gdf = dataset.index[mask].copy()
        else:
            # Create empty GeoDataFrame with correct structure
            cat_gdf = gpd.GeoDataFrame(
                {"filepath": []}, geometry=[], crs=dataset.index.crs
            )
            # Create empty IntervalIndex to match original structure
            empty_temporal_index = pd.IntervalIndex.from_tuples(
                [], closed="both", name="datetime"
            )
            cat_gdf.index = empty_temporal_index

        ds = deepcopy(dataset)
        ds.index = cat_gdf
        if len(cat_gdf) > 0:
            ds.paths = cat_gdf["filepath"].tolist()
        else:
            ds.paths = []
        new_datasets.append(ds)

    return new_datasets


def _process_aois(
    aois: Union[
        Sequence[Tuple[float, float, float, float]],
        str,
        Path,
    ],
    dataset: GeoDataset,
    crs: Optional[Union[str, Dict[str, Any]]] = "EPSG:4326",
    buffer_size: float = 0.0,
) -> Tuple[List, List]:
    """Helper function to process AOIs into Polygon objects.

    Args:
        aois: Areas of interest in one of the supported formats
        dataset: Dataset to be split
        crs: The CRS of the input AOIs
        buffer_size: Size of buffer to apply around AOIs

    Returns:
        Tuple of (roi_polygons, buffered_roi_polygons)
    """
    from shapely.geometry import box

    roi_polygons = []
    buffered_roi_polygons = []

    # Handle GeoJSON file input (if aois is a string or Path)
    if isinstance(aois, (str, Path)):
        geojson_path = Path(aois)
        if not geojson_path.exists():
            raise ValueError(f"GeoJSON file not found: {geojson_path}")

        try:
            # Read the GeoJSON file using geopandas
            gdf = gpd.read_file(geojson_path)

            # Filter to only keep Polygon geometries
            gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]

            if gdf.empty:
                raise ValueError("No valid Polygon features found in GeoJSON file")

            # Create bounding boxes from geometries
            aoi_list = []
            for geom in gdf.geometry:
                bounds = geom.bounds  # Returns (minx, miny, maxx, maxy)
                # Reorder to (minx, maxx, miny, maxy) as expected by the function
                aoi_list.append((bounds[0], bounds[2], bounds[1], bounds[3]))

            aois = aoi_list
        except Exception as e:
            raise ValueError(f"Error processing GeoJSON file: {str(e)}")

    # Handle CRS transformation if needed
    need_transform = False
    transformer = None

    if crs is not None:
        # Set up transformer if CRSs are different
        if crs != dataset.crs:
            need_transform = True
            source_crs = CRS.from_user_input(crs)
            target_crs = CRS.from_user_input(dataset.crs)
            transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

    for aoi in aois:
        if len(aoi) == 4:
            # Spatial only (minx, maxx, miny, maxy)
            minx, maxx, miny, maxy = aoi

            # Transform coordinates if needed
            if need_transform:
                minx, miny = transformer.transform(minx, miny)
                maxx, maxy = transformer.transform(maxx, maxy)

            # Create polygon from bounding box
            polygon = box(minx, miny, maxx, maxy)
            roi_polygons.append(polygon)

            # Create buffered polygon for training set determination
            buffered_polygon = box(
                minx - buffer_size,
                miny - buffer_size,
                maxx + buffer_size,
                maxy + buffer_size,
            )
            buffered_roi_polygons.append(buffered_polygon)
        else:
            raise ValueError("AOIs must be tuples of length 4 (minx, maxx, miny, maxy)")

    # Check for overlapping AOIs within the same category
    for i, roi in enumerate(roi_polygons):
        if any(
            roi.intersects(x) and not roi.intersection(x).is_empty
            for x in roi_polygons[i + 1 :]
        ):
            raise ValueError(
                "AOIs within the same category (validation or test) cannot overlap."
            )

    return roi_polygons, buffered_roi_polygons
