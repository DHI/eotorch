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
from pyproj import CRS
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from torch import Generator, default_generator, randperm
from torchgeo.datasets import GeoDataset, IntersectionDataset, RasterDataset
from torchgeo.datasets.splits import _fractions_to_lengths


def _format_paths(files: str | Path | list[str] | list[Path] | None) -> list[Path]:
    """Formats various file input types into a list of resolved Path objects."""
    if files is None:
        return []
    if isinstance(files, (str, Path)):
        files = [files]
    if isinstance(files, list):
        files = [Path(i) for i in files]
    return [i.resolve() for i in files]


def _rows_of_files(image_files: pd.Series, files: list[Path]) -> pd.Series:
    """Rows of the image index whose (resolved) file is one of the given files."""
    for f in files:
        if f not in image_files.values:
            print(f"Warning: File {str(f)} not found in dataset. Skipping.")
    rows = image_files.isin(files)
    if files and not rows.any():
        raise ValueError(
            "None of the provided files were found in the dataset. Please check the files."
        )
    return rows


def _with_rows(dataset: RasterDataset, index: gpd.GeoDataFrame) -> RasterDataset:
    """Copy of a raster dataset with only the given rows of its index."""
    split = deepcopy(dataset)
    split.index = index.copy()
    split.paths = index["filepath"].unique().tolist()
    return split


def _build_split_datasets[DatasetT: GeoDataset](
    dataset: DatasetT, image_indexes: list[gpd.GeoDataFrame]
) -> list[DatasetT]:
    """One dataset for each index of image rows, with the settings of dataset.

    The split of a dataset of images and labels keeps the labels that overlap its images.
    """
    if not isinstance(dataset, IntersectionDataset):
        return [_with_rows(dataset, index) for index in image_indexes]

    images, labels = dataset.datasets
    splits = []
    for index in image_indexes:
        overlapping = labels.index.geometry.intersects(index.union_all())
        split = type(dataset)(
            _with_rows(images, index),
            _with_rows(labels, labels.index[overlapping]),
            collate_fn=dataset.collate_fn,
            transforms=dataset.transforms,
        )
        splits.append(split)
    return splits


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

    # All rows of a file go to the same split
    image_index = img_dataset.index
    image_files = image_index["filepath"].map(lambda f: Path(f).resolve())
    val_rows = _rows_of_files(image_files, val_files)
    test_rows = _rows_of_files(image_files, test_files)

    image_indexes = [image_index[~(val_rows | test_rows)]]
    if val_files:
        image_indexes.append(image_index[val_rows])
    if test_files:
        image_indexes.append(image_index[test_rows])
    return _build_split_datasets(dataset, image_indexes)


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

    # Randomly assign the rows of the index to the splits
    order = randperm(len(split_dataset.index), generator=generator).tolist()
    new_indexes = []
    start = 0
    for length in lengths:
        new_indexes.append(split_dataset.index.iloc[order[start : start + length]])
        start += length

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
