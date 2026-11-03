import warnings
from collections.abc import Sequence
from copy import deepcopy
from glob import glob
from math import isclose
from pathlib import Path
from typing import Any, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from pyproj import CRS, Transformer
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
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


def _format_paths(files: str | Path | list[str] | list[Path] | None) -> list[Path]:
    """Formats various file input types into a list of resolved Path objects."""
    if files is None:
        return []
    if isinstance(files, (str, Path)):
        files = [files]
    if isinstance(files, list):
        files = [Path(i) for i in files]
    return [i.resolve() for i in files]


def _get_split_items(
    dataset_items: list[tuple[Any, Any]],  # (geometry, filepath) tuples
    dataset_files: list[Path],
    val_files: list[Path],
    test_files: list[Path],
) -> tuple[list[tuple[Any, Any]], list[tuple[Any, Any]], list[tuple[Any, Any]]]:
    """Validates split files and returns items partitioned into train, val, test."""

    def _validate_and_collect_items(files_param: list[Path]) -> list[tuple[Any, Any]]:
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
    train_items: list[tuple[Any, Any]],
    val_items: list[tuple[Any, Any]],
    test_items: list[tuple[Any, Any]],
    original_index: gpd.GeoDataFrame,
) -> list[gpd.GeoDataFrame]:
    """Builds GeoPandas GeoDataFrame indexes for train, validation, and test splits."""
    new_indexes = []

    def _create_index(items: list[tuple[Any, Any]]) -> gpd.GeoDataFrame:
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


def _build_split_datasets[DatasetT: GeoDataset](
    original_dataset: DatasetT, new_indexes: list[gpd.GeoDataFrame]
) -> list[DatasetT]:
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
            img_bounds = img_ds_new.bounds  # tuple of (x_slice, y_slice, t_slice)

            # For GeoPandas, we need to intersect with the image bounds
            if not label_ds_orig.index.empty:
                # Create bounding box geometry for intersection
                x_slice, y_slice, _ = img_bounds
                img_bbox_geom = box(
                    x_slice.start, y_slice.start, x_slice.stop, y_slice.stop
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


def file_wise_split[DatasetT: GeoDataset](
    dataset: DatasetT,
    val_img_files: str | Path | list[str] | list[Path] | None = None,
    test_img_files: str | Path | list[str] | list[Path] | None = None,
    val_img_glob: str | Path | None = None,
    test_img_glob: str | Path | None = None,
    ratios_or_counts: Sequence[float] | None = None,
) -> list[DatasetT]:
    """
    Splits a dataset based on image files, ensuring no file overlap between splits.

    Parameters:
        dataset (RasterDataset):
            The dataset to be split (RasterDataset or IntersectionDataset).
        val_img_files (str | Path | list[str] | list[Path] | None, optional):
            Files for the validation set. Defaults to None.
        test_img_files (str | Path | list[str] | list[Path] | None, optional):
            Files for the test set. Defaults to None.
        val_img_glob (str | Path | None, optional):
            glob pattern to find images for the validation set. Overrides `val_img_files`. Defaults to None.
        test_img_glob (str | Path | None, optional):
            glob pattern to find images for the test set. Overrides `test_img_files`. Defaults to None.
        ratios_or_counts (Sequence[float] | None, optional):
            Ratios or counts for random splitting (overrides file lists). Defaults to None.

    Raises:
        ValueError:
            If input parameters are invalid or files are not found.
        TypeError:
            If dataset is not a RasterDataset or IntersectionDataset.

    Returns:
        list[RasterDataset]:
            A list of datasets [train, validation (optional), test (optional)].
    """

    use_ratios = ratios_or_counts is not None
    use_files = (
        val_img_files or test_img_files or val_img_glob or test_img_glob
    ) is not None

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
    val_img_files = glob(val_img_glob) if val_img_glob is not None else val_img_files
    test_img_files = (
        glob(test_img_glob) if test_img_glob is not None else test_img_files
    )
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


def random_bbox_assignment[DatasetT: GeoDataset](
    dataset: DatasetT,
    lengths: Sequence[float],
    generator: Generator | None = default_generator,
) -> list[DatasetT]:
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

    # Determine the dataset to split (handle IntersectionDataset)
    if isinstance(dataset, IntersectionDataset):
        split_dataset = dataset.datasets[0]
    else:
        split_dataset = dataset

    if not (isclose(sum(lengths), 1) or isclose(sum(lengths), len(split_dataset))):
        raise ValueError(
            "Sum of input lengths must equal 1 or the length of dataset's index."
        )

    if any(n <= 0 for n in lengths):
        raise ValueError("All items in input lengths must be greater than 0.")

    if isclose(sum(lengths), 1):
        lengths = _fractions_to_lengths(lengths, len(split_dataset))
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
        (row["geometry"], row["filepath"]) for _, row in split_dataset.index.iterrows()
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
            mask = split_dataset.index["filepath"].isin(split_filepaths)
            filtered_index = split_dataset.index[mask].copy()
        else:
            # Create empty GeoDataFrame with correct structure
            empty_gdf = gpd.GeoDataFrame(
                {"filepath": []}, geometry=[], crs=split_dataset.index.crs
            )
            # Create empty IntervalIndex to match original structure
            empty_temporal_index = pd.IntervalIndex.from_tuples(
                [], closed="both", name="datetime"
            )
            empty_gdf.index = empty_temporal_index
            filtered_index = empty_gdf

        new_indexes.append(filtered_index)
        start_idx = end_idx

    return _build_split_datasets(dataset, new_indexes)


def aoi_split[DatasetT: GeoDataset](
    dataset: DatasetT,
    aoi_files: str | Path | Sequence[str | Path],
    crs: str | dict[str, Any] | CRS | None = None,
    buffer_size_metres: int = 0,
) -> list[DatasetT]:
    """Split a GeoDataset based on provided AOI files.

    Each polygon in an AOI file is replaced by its bounding box. The dataset footprints
    inside the boxes of an AOI file make up that file's dataset; everything outside all
    (buffered) boxes makes up the remainder. AOIs in another CRS than the dataset are
    reprojected to the dataset CRS.

    Args:
        dataset: Dataset to be split
        aoi_files: Single filepath or sequence of filepaths to shapefile or .geojson files
        crs: The CRS of the coordinates in the AOI files (e.g., "EPSG:32634"). Overrides
            the CRS declared in the files, so only needed for files that declare no CRS
            or the wrong one, e.g. GeoJSON files with projected coordinates (GeoJSON files
            without CRS information are always read as "EPSG:4326"). If None, the CRS
            declared in each file is used, or the dataset CRS for files that declare
            none. Default is None.
        buffer_size_metres: Size of buffer in meters to apply around AOIs. The buffer
            is only removed from the remainder. Default is 0.

    Returns:
        A list of datasets where the first is the remainder (training dataset),
        followed by one dataset for each provided file.
    """
    if isinstance(aoi_files, (str, Path)):
        aoi_files = [aoi_files]
    aoi_boxes = [_read_aoi_boxes(Path(f), crs, dataset.crs) for f in aoi_files]

    buffer_size = _convert_buffer_to_crs_units(buffer_size_metres, dataset.crs)
    buffered_boxes = [
        aoi.buffer(buffer_size) if buffer_size > 0 else aoi
        for boxes in aoi_boxes
        for aoi in boxes
    ]

    # Footprints of each split, as (row position in dataset.index, footprint)
    remainder_items = []
    aoi_items = [[] for _ in aoi_boxes]
    for pos, footprint in enumerate(dataset.index.geometry):
        for items, boxes in zip(aoi_items, aoi_boxes):
            for aoi in boxes:
                part = _ensure_valid_polygon(footprint.intersection(aoi))
                if part is not None:
                    items.append((pos, part))
        for part in _split_geometry_excluding_aois(footprint, buffered_boxes):
            remainder_items.append((pos, part))

    result_datasets = []
    for items in [remainder_items, *aoi_items]:
        split = deepcopy(dataset)
        split.index = dataset.index.iloc[[pos for pos, _ in items]].copy()
        split.index["geometry"] = gpd.GeoSeries(
            [part for _, part in items], crs=dataset.index.crs
        ).values
        if isinstance(dataset, IntersectionDataset):
            for ds in split.datasets:
                _restrict_to_footprints(ds, split.index)
        else:
            split.paths = split.index["filepath"].unique().tolist()
        result_datasets.append(split)

    return result_datasets


def _read_aoi_boxes(file_path: Path, crs, dataset_crs) -> list[Polygon]:
    """Bounding boxes of the polygons in an AOI file, in the dataset CRS.

    The file's coordinates are taken to be in `crs` if given, otherwise in the CRS
    declared in the file, or in the dataset CRS if the file declares none.
    """
    if not file_path.exists():
        raise ValueError(f"AOI file not found: {file_path}")

    try:
        gdf = gpd.read_file(file_path)
        if crs is not None:
            crs = CRS.from_user_input(crs)
            if gdf.crs is not None and gdf.crs != crs:
                warnings.warn(
                    f"AOI file {file_path} is read as {gdf.crs.to_string()}, but "
                    f"crs={crs.to_string()} was given. Using {crs.to_string()}.",
                    stacklevel=3,
                )
            gdf = gdf.set_crs(crs, allow_override=True)
        elif gdf.crs is None:
            gdf = gdf.set_crs(dataset_crs)
        gdf = gdf.to_crs(dataset_crs)
        gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
        if gdf.empty:
            raise ValueError(f"No valid Polygon features found in {file_path}")
    except Exception as e:
        raise ValueError(f"Error processing AOI file {file_path}: {e}") from e

    return [box(*geom.bounds) for geom in gdf.geometry]


def _split_geometry_excluding_aois(geom, aois) -> list[Polygon]:
    """Split a geometry into boxes excluding all AOI polygons to avoid holes."""
    parts = [geom]
    for aoi in aois:
        parts = [
            box_ for part in parts for box_ in _split_geometry_into_boxes(part, aoi)
        ]
    parts = (_ensure_valid_polygon(part) for part in parts)
    return [part for part in parts if part is not None]


def _ensure_valid_polygon(geom):
    """Ensure geometry is a valid Polygon or MultiPolygon for raster operations.

    Returns None if the geometry has no area, e.g. if it is empty, a Point or a LineString.
    """
    if geom.is_empty:
        return None
    if geom.geom_type in ["Polygon", "MultiPolygon"]:
        return geom
    if geom.geom_type == "GeometryCollection":
        polygons = [g for g in geom.geoms if g.geom_type in ["Polygon", "MultiPolygon"]]
        result = unary_union(polygons)
        return None if result.is_empty else result
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
) -> list[Polygon]:
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
        return [original_geom]

    minx, miny, maxx, maxy = original_geom.bounds
    sub_minx, sub_miny, sub_maxx, sub_maxy = subtract_geom.bounds

    # Boxes left and right of the subtract geometry, then below and above it
    # (between its left and right boundaries)
    boxes = []
    if minx < sub_minx:
        boxes.append(box(minx, miny, min(sub_minx, maxx), maxy))
    if maxx > sub_maxx:
        boxes.append(box(max(sub_maxx, minx), miny, maxx, maxy))
    if miny < sub_miny:
        boxes.append(
            box(max(minx, sub_minx), miny, min(maxx, sub_maxx), min(sub_miny, maxy))
        )
    if maxy > sub_maxy:
        boxes.append(
            box(max(minx, sub_minx), max(sub_maxy, miny), min(maxx, sub_maxx), maxy)
        )

    parts = (_ensure_valid_polygon(original_geom.intersection(b)) for b in boxes)
    return [part for part in parts if part is not None]


def _restrict_to_footprints(dataset, footprints: gpd.GeoDataFrame) -> None:
    """Clip the index rows of a dataset to the given footprints.

    Each row is replaced by its parts within the footprints it overlaps in both space
    (by more than a shared edge) and time, one row per footprint. Samples are read
    from within the footprints only, so the rest of a file isn't part of the split.
    The datasets of an intersection are restricted the same way.

    Args:
        dataset: Dataset to restrict, e.g. the image or label dataset of a split
        footprints: Index of the split, with the footprints of its samples
    """
    index = dataset.index
    rows, matches = footprints.sindex.query(index.geometry, predicate="intersects")
    order = np.argsort(rows, kind="stable")
    rows, matches = rows[order], matches[order]
    parts = index.geometry.iloc[rows].intersection(
        footprints.geometry.iloc[matches], align=False
    )
    keep = (parts.area > 0).to_numpy() & np.array(
        [a.overlaps(b) for a, b in zip(index.index[rows], footprints.index[matches])],
        dtype=bool,
    )
    clipped = index.iloc[rows[keep]].copy()
    clipped.geometry = parts[keep].values
    dataset.index = clipped

    if hasattr(dataset, "datasets"):
        for ds in dataset.datasets:
            _restrict_to_footprints(ds, footprints)
    else:
        dataset.paths = dataset.index["filepath"].unique().tolist()
