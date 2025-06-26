from copy import deepcopy
from math import isclose
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import geopandas as gpd
import pandas as pd
import torch
from pyproj import CRS, Transformer
from shapely.geometry import Polygon, box
from torch import Generator, default_generator, randperm
from torchgeo.datasets import GeoDataset, IntersectionDataset, RasterDataset
from torchgeo.datasets.splits import _fractions_to_lengths


def _transform_bounds_to_crs(bounds, source_crs, target_crs):
    """Transform bounding box coordinates from source CRS to target CRS.

    Args:
        bounds: Tuple of (minx, miny, maxx, maxy) in source CRS
        source_crs: Source CRS as string (e.g., "EPSG:4326")
        target_crs: Target CRS as string (e.g., "EPSG:3857")

    Returns:
        Tuple of (minx, miny, maxx, maxy) in target CRS
    """
    if source_crs == target_crs:
        return bounds

    try:
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        minx, miny, maxx, maxy = bounds

        # Transform all four corners to handle potential rotation/skew
        corners = [
            (minx, miny),  # bottom-left
            (minx, maxy),  # top-left
            (maxx, miny),  # bottom-right
            (maxx, maxy),  # top-right
        ]

        transformed_corners = [transformer.transform(x, y) for x, y in corners]

        # Get new bounds from transformed corners
        xs, ys = zip(*transformed_corners)
        return (min(xs), min(ys), max(xs), max(ys))

    except Exception as e:
        raise ValueError(
            f"Error transforming bounds from {source_crs} to {target_crs}: {str(e)}"
        )


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
                img_bbox_geom = box(
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
        (row["geometry"], row["filepath"]) for _, row in img_dataset.index.iterrows()
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
    items_data = [
        (row["geometry"], row["filepath"]) for _, row in dataset.index.iterrows()
    ]

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
    aoi_files: str | Path | Sequence[Union[str, Path]],
    crs: str | Dict[str, Any] = "EPSG:4326",
    buffer_size_metres: int = 0,
) -> List[GeoDataset]:
    """Split a GeoDataset based on provided AOI files with cleaner implementation.

    Args:
        dataset: Dataset to be split
        aoi_files: Single filepath or sequence of filepaths to shapefile or .geojson files
        crs: The CRS of the input AOIs (e.g., "EPSG:4326" for lat/lon). If None,
             assumes AOIs are in the same CRS as the dataset. Default is "EPSG:4326".
        buffer_size: Size of buffer in meters to apply around AOIs. Default is 0.

    Returns:
        A list of datasets where the first is the remainder (training dataset),
        followed by one dataset for each provided file.
    """

    # Handle LabelledRasterDataset/IntersectionDataset by processing the intersection directly
    is_intersection_dataset = isinstance(dataset, IntersectionDataset)

    # Normalize files to list of Path objects
    if isinstance(aoi_files, (str, Path)):
        file_list = [Path(aoi_files)]
    else:
        file_list = [Path(f) for f in aoi_files]

    # Load all AOI geometries from files and convert to bounding boxes
    all_aoi_polygons = []
    for file_path in file_list:
        if not file_path.exists():
            raise ValueError(f"AOI file not found: {file_path}")

        try:
            gdf = gpd.read_file(file_path)
            gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]

            if gdf.empty:
                raise ValueError(f"No valid Polygon features found in {file_path}")

            # Instead of reprojecting geometries, transform only bounds to dataset CRS
            rectangles = []
            for geom in gdf.geometry:
                bounds = geom.bounds  # minx, miny, maxx, maxy in original CRS

                # Transform bounds to dataset CRS if needed
                if crs is not None and crs != dataset.crs:
                    transformed_bounds = _transform_bounds_to_crs(
                        bounds, crs, dataset.crs
                    )
                else:
                    transformed_bounds = bounds

                # Create rectangle from transformed bounds
                minx, miny, maxx, maxy = transformed_bounds
                rectangle = box(minx, miny, maxx, maxy)
                rectangles.append(rectangle)

            all_aoi_polygons.append(rectangles)

        except Exception as e:
            raise ValueError(f"Error processing AOI file {file_path}: {str(e)}")

    # Convert buffer size to dataset CRS units
    buffer_size_crs = _convert_buffer_to_crs_units(buffer_size_metres, dataset.crs)

    # Create buffered versions for training exclusion
    all_buffered_polygons = []
    for polygons in all_aoi_polygons:
        if buffer_size_crs > 0:
            buffered = [poly.buffer(buffer_size_crs) for poly in polygons]
        else:
            buffered = polygons.copy()
        all_buffered_polygons.append(buffered)

    # Process each geometry in the dataset
    remainder_items = []
    aoi_datasets_items = [[] for _ in file_list]

    for idx, row in dataset.index.iterrows():
        dataset_geom = row["geometry"]

        # Handle different filepath column structures
        if "filepath" in row.index:
            filepath = row["filepath"]
        elif "filepath_1" in row.index:
            # For IntersectionDataset, use the first dataset's filepath as the identifier
            filepath = row["filepath_1"]
        else:
            # Fallback: use the index as identifier
            filepath = str(idx)

        # Track which AOI files this geometry intersects with
        intersected_aois = set()

        # Check intersection with each AOI file
        for file_idx, polygons in enumerate(all_aoi_polygons):
            for polygon in polygons:
                if dataset_geom.intersects(polygon):
                    intersection = dataset_geom.intersection(polygon)
                    intersection = _ensure_valid_polygon(intersection)
                    if intersection is not None and not intersection.is_empty:
                        aoi_datasets_items[file_idx].append(
                            (intersection, filepath, idx)
                        )
                        intersected_aois.add(file_idx)

        # For remainder: split dataset geometry into boxes excluding all buffered AOIs
        remainder_geoms = _split_dataset_geometry_excluding_aois(
            dataset_geom, all_buffered_polygons
        )

        for geom in remainder_geoms:
            if geom is not None and not geom.is_empty:
                remainder_items.append((geom, filepath, idx))

    # Create datasets
    result_datasets = []

    # Create remainder dataset (first in list)
    remainder_gdf = _create_geodataframe_from_items(remainder_items, dataset)
    remainder_ds = deepcopy(dataset)
    remainder_ds.index = remainder_gdf

    # Update paths based on dataset type
    if is_intersection_dataset:
        # For IntersectionDataset: update sub-dataset paths only
        _update_intersection_dataset_paths(remainder_ds, remainder_gdf)
    else:
        # For regular datasets: update paths on the dataset itself
        remainder_ds.paths = (
            remainder_gdf["filepath"].unique().tolist()
            if len(remainder_gdf) > 0
            else []
        )

    result_datasets.append(remainder_ds)

    # Create datasets for each AOI file
    for items in aoi_datasets_items:
        aoi_gdf = _create_geodataframe_from_items(items, dataset)
        aoi_ds = deepcopy(dataset)
        aoi_ds.index = aoi_gdf

        # Update paths based on dataset type
        if is_intersection_dataset:
            # For IntersectionDataset: update sub-dataset paths only
            _update_intersection_dataset_paths(aoi_ds, aoi_gdf)
        else:
            # For regular datasets: update paths on the dataset itself
            aoi_ds.paths = (
                aoi_gdf["filepath"].unique().tolist() if len(aoi_gdf) > 0 else []
            )

        result_datasets.append(aoi_ds)

    return result_datasets


def _split_dataset_geometry_excluding_aois(dataset_geom, all_buffered_polygons):
    """Split dataset geometry into boxes excluding all AOI polygons to avoid holes."""
    current_geoms = [dataset_geom]

    # Process each AOI file's polygons
    for buffered_polygons in all_buffered_polygons:
        for buffered_polygon in buffered_polygons:
            new_geoms = []
            for geom in current_geoms:
                if geom.intersects(buffered_polygon):
                    # Split into maximum 4 boxes around the AOI
                    split_boxes = _split_geometry_into_boxes(geom, buffered_polygon)
                    new_geoms.extend(split_boxes)
                else:
                    new_geoms.append(geom)
            current_geoms = new_geoms

    return [_ensure_valid_polygon(geom) for geom in current_geoms]


def _create_geodataframe_from_items(items, original_dataset):
    """Create GeoDataFrame from items with proper structure."""
    if not items:
        # Create empty GeoDataFrame with correct structure
        empty_gdf = gpd.GeoDataFrame(
            {"filepath": []}, geometry=[], crs=original_dataset.index.crs
        )
        empty_temporal_index = pd.IntervalIndex.from_tuples(
            [], closed="both", name="datetime"
        )
        empty_gdf.index = empty_temporal_index
        return empty_gdf

    # Create new GeoDataFrame with split geometries
    data = []
    temporal_indices = []

    for geom, filepath, temporal_idx in items:
        # Ensure geometry is valid for raster operations
        if geom.geom_type in ["Point", "LineString"]:
            continue
        elif geom.geom_type == "GeometryCollection":
            # Extract only Polygon/MultiPolygon parts
            polygons = [
                g for g in geom.geoms if g.geom_type in ["Polygon", "MultiPolygon"]
            ]
            if not polygons:
                continue
            from shapely.ops import unary_union

            geom = unary_union(polygons)
            if geom.is_empty:
                continue

        data.append({"filepath": filepath, "geometry": geom})
        temporal_indices.append(temporal_idx)

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(data, crs=original_dataset.index.crs)
    gdf.index = pd.Index(temporal_indices, name=original_dataset.index.index.name)

    return gdf


def _ensure_valid_polygon(geom):
    """Ensure geometry is a valid Polygon or MultiPolygon for raster operations."""
    if geom.is_empty:
        return None

    # Handle different geometry types
    if geom.geom_type in ["Polygon", "MultiPolygon"]:
        return geom
    elif geom.geom_type == "GeometryCollection":
        # Extract only Polygon/MultiPolygon parts
        polygons = [g for g in geom.geoms if g.geom_type in ["Polygon", "MultiPolygon"]]
        if not polygons:
            return None
        from shapely.ops import unary_union

        result = unary_union(polygons)
        return result if not result.is_empty else None
    else:
        # Skip Point, LineString, etc. as they can't represent raster coverage
        return None


def _convert_buffer_to_crs_units(buffer_meters: float, dataset_crs: str) -> float:
    """Convert buffer size from meters to dataset CRS units.

    Args:
        buffer_meters: Buffer size in meters
        dataset_crs: CRS of the dataset

    Returns:
        Buffer size in dataset CRS units
    """
    if buffer_meters == 0:
        return 0

    try:
        crs_obj = CRS.from_user_input(dataset_crs)

        # If CRS is already in meters, return as-is
        if crs_obj.axis_info[0].unit_name in ["metre", "meter", "m"]:
            return buffer_meters

        # If CRS is in degrees (like EPSG:4326), convert meters to degrees
        # Using rough approximation: 1 degree ≈ 111,320 meters at equator
        if crs_obj.axis_info[0].unit_name in ["degree", "degrees"]:
            return buffer_meters / 111320.0

        # For other units, return the buffer as-is (user's responsibility)
        return buffer_meters

    except Exception:
        # If we can't determine the CRS units, assume it's in the same units as provided
        return buffer_meters


def _split_geometry_into_boxes(
    original_geom: Polygon, subtract_geom: Polygon
) -> List[Polygon]:
    """
    Split a geometry into multiple box geometries that exclude the subtract_geom area.

    Instead of using geometric difference (which can create holes), this function
    creates multiple rectangular geometries that cover the original area while
    excluding the subtract area.

    Args:
        original_geom: The original geometry to split
        subtract_geom: The geometry to exclude from the original

    Returns:
        List of box geometries that cover the original area excluding the subtract area
    """
    if not original_geom.intersects(subtract_geom):
        # No intersection, return original geometry
        return [original_geom]

    # Get bounds of the original geometry
    minx, miny, maxx, maxy = original_geom.bounds

    # Get bounds of the subtract geometry
    sub_minx, sub_miny, sub_maxx, sub_maxy = subtract_geom.bounds

    boxes = []

    # Create boxes around the subtract geometry:
    # Left box
    if minx < sub_minx:
        left_box = box(minx, miny, min(sub_minx, maxx), maxy)
        if original_geom.intersects(left_box):
            intersection = original_geom.intersection(left_box)
            intersection = _ensure_valid_polygon(intersection)
            if intersection is not None and not intersection.is_empty:
                boxes.append(intersection)

    # Right box
    if maxx > sub_maxx:
        right_box = box(max(sub_maxx, minx), miny, maxx, maxy)
        if original_geom.intersects(right_box):
            intersection = original_geom.intersection(right_box)
            intersection = _ensure_valid_polygon(intersection)
            if intersection is not None and not intersection.is_empty:
                boxes.append(intersection)

    # Bottom box (between left and right boundaries of subtract_geom)
    if miny < sub_miny:
        bottom_box = box(
            max(minx, sub_minx), miny, min(maxx, sub_maxx), min(sub_miny, maxy)
        )
        if original_geom.intersects(bottom_box):
            intersection = original_geom.intersection(bottom_box)
            intersection = _ensure_valid_polygon(intersection)
            if intersection is not None and not intersection.is_empty:
                boxes.append(intersection)

    # Top box (between left and right boundaries of subtract_geom)
    if maxy > sub_maxy:
        top_box = box(
            max(minx, sub_minx), max(sub_maxy, miny), min(maxx, sub_maxx), maxy
        )
        if original_geom.intersects(top_box):
            intersection = original_geom.intersection(top_box)
            intersection = _ensure_valid_polygon(intersection)
            if intersection is not None and not intersection.is_empty:
                boxes.append(intersection)

    return boxes


def _update_intersection_dataset_paths(intersection_dataset, index_gdf):
    """Update the paths of sub-datasets in an IntersectionDataset based on the intersection index.

    Args:
        intersection_dataset: The IntersectionDataset to update
        index_gdf: The GeoDataFrame index containing the geometries after splitting
    """
    if len(index_gdf) == 0:
        # Empty dataset - set empty paths for both sub-datasets
        intersection_dataset.datasets[0].paths = []
        intersection_dataset.datasets[1].paths = []
        return

    # Get the bounds of the intersection geometries to determine relevant files
    intersection_bounds = index_gdf.total_bounds  # [minx, miny, maxx, maxy]
    intersection_bbox = box(
        intersection_bounds[0],
        intersection_bounds[1],
        intersection_bounds[2],
        intersection_bounds[3],
    )

    # Update image dataset paths
    img_dataset = intersection_dataset.datasets[0]
    if not img_dataset.index.empty:
        img_mask = img_dataset.index.geometry.intersects(intersection_bbox)
        img_paths = img_dataset.index.loc[img_mask, "filepath"].unique().tolist()
        img_dataset.paths = img_paths
    else:
        img_dataset.paths = []

    # Update label dataset paths
    label_dataset = intersection_dataset.datasets[1]
    if not label_dataset.index.empty:
        label_mask = label_dataset.index.geometry.intersects(intersection_bbox)
        label_paths = label_dataset.index.loc[label_mask, "filepath"].unique().tolist()
        label_dataset.paths = label_paths
    else:
        label_dataset.paths = []
