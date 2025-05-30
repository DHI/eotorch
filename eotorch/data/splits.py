from copy import deepcopy
from math import isclose
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import geopandas as gpd
import torch
from pyproj import CRS, Transformer
from rtree.index import Index, Item, Property
from torch import Generator, default_generator, randperm
from torchgeo.datasets import GeoDataset, IntersectionDataset, RasterDataset
from torchgeo.datasets.splits import _fractions_to_lengths
from torchgeo.datasets.utils import BoundingBox

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
    dataset_items: List[Item],
    dataset_files: List[Path],
    val_files: List[Path],
    test_files: List[Path],
) -> Tuple[List[Item], List[Item], List[Item]]:
    """Validates split files and returns items partitioned into train, val, test."""

    def _validate_and_collect_items(files_param: List[Path]) -> List[Item]:
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
    train_items: List[Item], val_items: List[Item], test_items: List[Item]
) -> List[Index]:
    """Builds R-tree indexes for train, validation, and test splits."""
    new_indexes = []

    def _create_index(items: List[Item]) -> Index:
        """Helper to create and populate an R-tree index."""
        idx = Index(interleaved=False, properties=Property(dimension=3))
        for i, item in enumerate(items):
            idx.insert(i, item.bounds, item.object)
        return idx

    if train_items:
        new_indexes.append(_create_index(train_items))
    if val_items:
        new_indexes.append(_create_index(val_items))
    if test_items:
        new_indexes.append(_create_index(test_items))

    return new_indexes


def _build_split_datasets(
    original_dataset: RasterDataset, new_indexes: List[Index]
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
                img_ds_new.paths = [
                    i.object for i in index.intersection(index.bounds, objects=True)
                ]
            else:
                img_ds_new.paths = []

            # Create new label dataset index based on intersection with new image dataset bounds
            img_bounds = img_ds_new.bounds
            label_hits = label_ds_orig.index.intersection(
                tuple(img_bounds), objects=True
            )
            label_idx_new = Index(interleaved=False, properties=Property(dimension=3))
            for hit in label_hits:
                hit_bbox = BoundingBox(*hit.bounds)
                intersection_bbox = hit_bbox & img_bounds
                if intersection_bbox.area > 0:
                    label_idx_new.insert(len(label_idx_new), hit.bounds, hit.object)

            # Create new label dataset for the split
            label_ds_new = deepcopy(label_ds_orig)
            label_ds_new.index = label_idx_new
            if len(label_idx_new) > 0:
                label_ds_new.paths = [
                    i.object
                    for i in label_idx_new.intersection(
                        label_idx_new.bounds, objects=True
                    )
                ]
            else:
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
                ds.paths = [
                    i.object for i in index.intersection(index.bounds, objects=True)
                ]
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
    dataset_items = list(
        img_dataset.index.intersection(img_dataset.index.bounds, objects=True)
    )
    dataset_files = [Path(item.object).resolve() for item in dataset_items]

    # Partition items into train, validation, and test sets
    train_items, val_items, test_items = _get_split_items(
        dataset_items, dataset_files, val_files, test_files
    )

    # Build R-tree indexes for each split
    new_indexes = _build_split_indexes(train_items, val_items, test_items)

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
    hits = list(dataset.index.intersection(dataset.index.bounds, objects=True))

    hits = [hits[i] for i in randperm(sum(lengths), generator=generator)]

    new_indexes = [
        Index(interleaved=False, properties=Property(dimension=3)) for _ in lengths
    ]

    for i, length in enumerate(lengths):
        for j in range(length):
            hit = hits.pop()
            new_indexes[i].insert(j, hit.bounds, hit.object)

    new_datasets = []
    for index in new_indexes:
        ds = deepcopy(dataset)
        ds.index = index
        new_datasets.append(ds)

    return new_datasets


def aoi_split(
    dataset: GeoDataset,
    val_aois: Union[
        Sequence[BoundingBox],
        Sequence[Tuple[float, float, float, float]],
        Sequence[Tuple[float, float, float, float, float, float]],
        str,
        Path,
        None,
    ] = None,
    test_aois: Union[
        Sequence[BoundingBox],
        Sequence[Tuple[float, float, float, float]],
        Sequence[Tuple[float, float, float, float, float, float]],
        str,
        Path,
        None,
    ] = None,
    time_bounds: Optional[Tuple[float, float]] = None,
    as_lists: bool = False,
    crs: Optional[Union[str, Dict[str, Any]]] = "EPSG:4326",
    buffer_size: float = 0.0,
) -> List[GeoDataset]:
    """Split a GeoDataset into train, validation, and test datasets based on areas of interest (AOIs).

    This function divides a dataset into multiple datasets:
    - First dataset: Contains all data outside the provided AOIs (training dataset)
    - Second dataset (optional): Contains data that intersects with validation AOIs
    - Third dataset (optional): Contains data that intersects with test AOIs

    The AOIs can be provided in different formats:
    - List of TorchGeo BoundingBox objects
    - List of (minx, maxx, miny, maxy) tuples (spatial only), or for lat/lon: (minlon, maxlon, minlat, maxlat)
    - List of (minx, maxx, miny, maxy, mint, maxt) tuples (spatiotemporal)
    - Path to a GeoJSON file containing Polygon features

    If AOIs are spatial only (4-tuples) and time_bounds is provided, the time dimension
    will be added to create complete 6D bounding boxes.

    The function supports transforming between coordinate reference systems. If your AOIs are
    specified in lat/lon (EPSG:4326) but your dataset is in a different CRS, you can specify
    the CRS parameters to handle the transformation automatically.

    A buffer can be applied around the AOIs to ensure that samples from training and validation/test
    datasets do not overlap. The buffer size is specified in the same units as the dataset's CRS
    (typically meters).

    Args:
        dataset: Dataset to be split
        val_aois: Sequence of validation areas of interest in one of the supported formats,
                  or a path to a GeoJSON file containing validation regions
        test_aois: Sequence of test areas of interest in one of the supported formats,
                   or a path to a GeoJSON file containing test regions
        time_bounds: Optional time bounds to use if aois are spatial only
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
            time_bounds=time_bounds,
            as_lists=as_lists,
            crs=crs,
            buffer_size=buffer_size,
        )
        label_datasets = aoi_split(
            dataset.datasets[1],
            val_aois=val_aois,
            test_aois=test_aois,
            time_bounds=time_bounds,
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
    roi_boxes = []  # Will hold all BoundingBox objects
    buffered_roi_boxes = []  # Will hold buffered versions of all ROIs
    aoi_categories = []  # Will track whether each ROI is for validation (0) or test (1)

    # Process validation AOIs if provided
    if val_aois is not None:
        val_roi_boxes, val_buffered_boxes = _process_aois(
            val_aois, dataset, time_bounds, crs, buffer_size
        )
        roi_boxes.extend(val_roi_boxes)
        buffered_roi_boxes.extend(val_buffered_boxes)
        aoi_categories.extend([0] * len(val_roi_boxes))  # 0 for validation

    # Process test AOIs if provided
    if test_aois is not None:
        test_roi_boxes, test_buffered_boxes = _process_aois(
            test_aois, dataset, time_bounds, crs, buffer_size
        )
        roi_boxes.extend(test_roi_boxes)
        buffered_roi_boxes.extend(test_buffered_boxes)
        aoi_categories.extend([1] * len(test_roi_boxes))  # 1 for test

    # If no AOIs were provided, return the original dataset
    if not roi_boxes:
        if as_lists:
            all_hits = list(
                dataset.index.intersection(dataset.index.bounds, objects=True)
            )
            return [all_hits]
        return [dataset]

    # Check for overlapping AOIs between validation and test sets
    if val_aois is not None and test_aois is not None:
        val_count = len([c for c in aoi_categories if c == 0])
        for i in range(val_count):
            for j in range(val_count, len(roi_boxes)):
                if (
                    roi_boxes[i].intersects(roi_boxes[j])
                    and (roi_boxes[i] & roi_boxes[j]).area > 0
                ):
                    raise ValueError("Validation and test AOIs cannot overlap.")

    # Create indexes for train and AOI categories
    train_idx = Index(interleaved=False, properties=Property(dimension=3))

    # Create category indexes - either just validation, just test, or both
    category_indexes = []
    if val_aois is not None:
        val_idx = Index(interleaved=False, properties=Property(dimension=3))
        category_indexes.append(val_idx)
    if test_aois is not None:
        test_idx = Index(interleaved=False, properties=Property(dimension=3))
        category_indexes.append(test_idx)

    # Get all hits from dataset
    all_hits = list(dataset.index.intersection(dataset.index.bounds, objects=True))

    # For each AOI category, prepare to collect items that intersect
    category_items = [[] for _ in category_indexes]

    # Process each item in the dataset
    for item_idx, hit in enumerate(all_hits):
        box = BoundingBox(*hit.bounds)
        roi_coverage = []  # List to store parts of box covered by ROIs

        # Check if this item is in any AOI
        for aoi_idx, (roi, buffered_roi) in enumerate(
            zip(roi_boxes, buffered_roi_boxes)
        ):
            category = aoi_categories[aoi_idx]
            # Map category to the correct index in category_indexes
            # If we only have val_aois, validation (category 0) maps to index 0
            # If we only have test_aois, test (category 1) maps to index 0
            # If we have both, validation (category 0) maps to index 0, test (category 1) maps to index 1
            if val_aois is not None and test_aois is not None:
                category_idx = category  # Both validation (0) and test (1) are present
            elif val_aois is not None:
                category_idx = 0  # Only validation, so it's at index 0
            else:  # test_aois is not None
                category_idx = 0  # Only test, so it's at index 0

            # Check intersection with original ROI for AOI datasets
            if box.intersects(roi):
                new_box = box & roi
                if new_box.area > 0:
                    # Get the appropriate category index
                    category_items[category_idx].append((item_idx, hit, new_box))

            # Check intersection with buffered ROI for training exclusion
            if box.intersects(buffered_roi):
                new_box = box & buffered_roi
                if new_box.area > 0:
                    # Use the buffered ROI for determining what's excluded from training
                    roi_coverage.append(new_box)

        # If box is not covered by any buffered ROI, add the whole item to training
        if not roi_coverage:
            train_idx.insert(len(train_idx), hit.bounds, hit.object)
        else:
            # Compute the remainder of the box after removing all buffered ROI intersections
            # Start with the original box
            remainder = [box]

            # Subtract each ROI intersection from the remainder
            for roi_box in roi_coverage:
                new_remainder = []
                for r in remainder:
                    # If r and roi_box don't overlap, keep r as is
                    if not r.intersects(roi_box) or (r & roi_box).area == 0:
                        new_remainder.append(r)
                    else:
                        # Otherwise, compute the difference by breaking down the box
                        # This creates up to 4 new boxes for each dimension

                        # X-direction boxes (left and right of intersection)
                        if r.minx < roi_box.minx:
                            new_remainder.append(
                                BoundingBox(
                                    r.minx, roi_box.minx, r.miny, r.maxy, r.mint, r.maxt
                                )
                            )
                        if r.maxx > roi_box.maxx:
                            new_remainder.append(
                                BoundingBox(
                                    roi_box.maxx, r.maxx, r.miny, r.maxy, r.mint, r.maxt
                                )
                            )

                        # Y-direction boxes (below and above intersection)
                        # We need to avoid adding regions already covered by X-direction boxes
                        if r.miny < roi_box.miny:
                            new_remainder.append(
                                BoundingBox(
                                    max(r.minx, roi_box.minx),
                                    min(r.maxx, roi_box.maxx),
                                    r.miny,
                                    roi_box.miny,
                                    r.mint,
                                    r.maxt,
                                )
                            )
                        if r.maxy > roi_box.maxy:
                            new_remainder.append(
                                BoundingBox(
                                    max(r.minx, roi_box.minx),
                                    min(r.maxx, roi_box.maxx),
                                    roi_box.maxy,
                                    r.maxy,
                                    r.mint,
                                    r.maxt,
                                )
                            )

                        # Time-direction boxes (before and after intersection)
                        # Add only if we have time dimension and there's an actual difference
                        if r.mint < roi_box.mint:
                            new_remainder.append(
                                BoundingBox(
                                    max(r.minx, roi_box.minx),
                                    min(r.maxx, roi_box.maxx),
                                    max(r.miny, roi_box.miny),
                                    min(r.maxy, roi_box.maxy),
                                    r.mint,
                                    roi_box.mint,
                                )
                            )
                        if r.maxt > roi_box.maxt:
                            new_remainder.append(
                                BoundingBox(
                                    max(r.minx, roi_box.minx),
                                    min(r.maxx, roi_box.maxx),
                                    max(r.miny, roi_box.miny),
                                    min(r.maxy, roi_box.maxy),
                                    roi_box.maxt,
                                    r.maxt,
                                )
                            )

                remainder = new_remainder

            # Add all remaining parts to the training set
            for i, r in enumerate(remainder):
                # Only add boxes with area > 0
                if r.area > 0:
                    train_idx.insert(len(train_idx), tuple(r), hit.object)

    # Populate the category indexes
    for cat_idx, items in enumerate(category_items):
        for j, (_, hit, new_box) in enumerate(items):
            category_indexes[cat_idx].insert(j, tuple(new_box), hit.object)

    if as_lists:
        # Check if train_idx has any items before accessing bounds
        if len(train_idx) > 0:
            train_list = [
                hit for hit in train_idx.intersection(train_idx.bounds, objects=True)
            ]
        else:
            train_list = []

        cat_lists = []
        for idx in category_indexes:
            if len(idx) > 0:
                cat_lists.append(
                    [hit for hit in idx.intersection(idx.bounds, objects=True)]
                )
            else:
                cat_lists.append([])
        return [train_list] + cat_lists

    # Create datasets from indexes
    new_datasets = []

    # Create train dataset
    train_ds = deepcopy(dataset)
    train_ds.index = train_idx
    if hasattr(train_ds, "paths"):
        # Check if train_idx has any items before accessing bounds
        if len(train_idx) > 0:
            train_ds.paths = [
                i.object for i in train_idx.intersection(train_idx.bounds, objects=True)
            ]
        else:
            train_ds.paths = []
    new_datasets.append(train_ds)

    # Create category datasets (validation and/or test)
    for index in category_indexes:
        ds = deepcopy(dataset)
        ds.index = index
        if hasattr(ds, "paths"):
            # Check if index has any items before accessing bounds
            if len(index) > 0:
                ds.paths = [
                    i.object for i in index.intersection(index.bounds, objects=True)
                ]
            else:
                ds.paths = []
        new_datasets.append(ds)

    return new_datasets


def _process_aois(
    aois: Union[
        Sequence[BoundingBox],
        Sequence[Tuple[float, float, float, float]],
        Sequence[Tuple[float, float, float, float, float, float]],
        str,
        Path,
    ],
    dataset: GeoDataset,
    time_bounds: Optional[Tuple[float, float]] = None,
    crs: Optional[Union[str, Dict[str, Any]]] = "EPSG:4326",
    buffer_size: float = 0.0,
) -> Tuple[List[BoundingBox], List[BoundingBox]]:
    """Helper function to process AOIs into BoundingBox objects.

    Args:
        aois: Areas of interest in one of the supported formats
        dataset: Dataset to be split
        time_bounds: Optional time bounds to use if aois are spatial only
        crs: The CRS of the input AOIs
        buffer_size: Size of buffer to apply around AOIs

    Returns:
        Tuple of (roi_boxes, buffered_roi_boxes)
    """
    roi_boxes = []
    buffered_roi_boxes = []

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

    # Get dataset bounds for time dimension if needed
    if time_bounds is None and any(
        len(aoi) == 4 for aoi in aois if not isinstance(aoi, BoundingBox)
    ):
        try:
            _, _, _, _, mint, maxt = dataset.bounds
            time_bounds = (mint, maxt)
        except (ValueError, AttributeError):
            # If no time dimension in dataset bounds
            time_bounds = (0, 1)  # Default time bounds

    for aoi in aois:
        if isinstance(aoi, BoundingBox):
            # Transform BoundingBox coordinates if needed
            if need_transform:
                minx, miny = transformer.transform(aoi.minx, aoi.miny)
                maxx, maxy = transformer.transform(aoi.maxx, aoi.maxy)
                transformed_box = BoundingBox(
                    minx, maxx, miny, maxy, aoi.mint, aoi.maxt
                )
                roi_boxes.append(transformed_box)
                # Create buffered box for training set determination
                buffered_box = BoundingBox(
                    minx - buffer_size,
                    maxx + buffer_size,
                    miny - buffer_size,
                    maxy + buffer_size,
                    aoi.mint,
                    aoi.maxt,
                )
                buffered_roi_boxes.append(buffered_box)
            else:
                roi_boxes.append(aoi)
                # Create buffered box for training set determination
                buffered_box = BoundingBox(
                    aoi.minx - buffer_size,
                    aoi.maxx + buffer_size,
                    aoi.miny - buffer_size,
                    aoi.maxy + buffer_size,
                    aoi.mint,
                    aoi.maxt,
                )
                buffered_roi_boxes.append(buffered_box)
        elif len(aoi) == 4:
            # Spatial only (minx, maxx, miny, maxy)
            minx, maxx, miny, maxy = aoi

            # Transform coordinates if needed
            if need_transform:
                minx, miny = transformer.transform(minx, miny)
                maxx, maxy = transformer.transform(maxx, maxy)

            if time_bounds:
                mint, maxt = time_bounds
                roi_boxes.append(BoundingBox(minx, maxx, miny, maxy, mint, maxt))
                # Create buffered box for training set determination
                buffered_roi_boxes.append(
                    BoundingBox(
                        minx - buffer_size,
                        maxx + buffer_size,
                        miny - buffer_size,
                        maxy + buffer_size,
                        mint,
                        maxt,
                    )
                )
            else:
                raise ValueError("time_bounds must be provided for spatial-only AOIs")
        elif len(aoi) == 6:
            # Complete spatiotemporal (minx, maxx, miny, maxy, mint, maxt)
            minx, maxx, miny, maxy, mint, maxt = aoi

            # Transform coordinates if needed
            if need_transform:
                minx, miny = transformer.transform(minx, miny)
                maxx, maxy = transformer.transform(maxx, maxy)

            roi_boxes.append(BoundingBox(minx, maxx, miny, maxy, mint, maxt))
            # Create buffered box for training set determination
            buffered_roi_boxes.append(
                BoundingBox(
                    minx - buffer_size,
                    maxx + buffer_size,
                    miny - buffer_size,
                    maxy + buffer_size,
                    mint,
                    maxt,
                )
            )
        else:
            raise ValueError(
                "AOIs must be BoundingBox objects or tuples of length 4 (spatial) or 6 (spatiotemporal)"
            )

    # Check for overlapping AOIs within the same category
    for i, roi in enumerate(roi_boxes):
        if any(roi.intersects(x) and (roi & x).area > 0 for x in roi_boxes[i + 1 :]):
            raise ValueError(
                "AOIs within the same category (validation or test) cannot overlap."
            )

    return roi_boxes, buffered_roi_boxes
