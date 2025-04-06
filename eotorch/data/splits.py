from copy import deepcopy
from math import isclose
from pathlib import Path
from typing import Sequence, cast

import torch
from rtree.index import Index, Property
from torch import Generator, default_generator, randperm
from torchgeo.datasets import GeoDataset, IntersectionDataset, RasterDataset
from torchgeo.datasets.splits import _fractions_to_lengths

from eotorch.data.geodatasets import LabelledRasterDataset


def file_wise_split(
    dataset: RasterDataset,
    val_img_files=None,
    test_img_files=None,
    ratios_or_counts=None,
):
    """
    File based splitting of a dataset.
    This means that the existing dataset will be split into 2 or 3 new datasets, with the new
    datasets having no overlap in terms of input imagery.
    Since the split is based on files, it is still possible to have spatial overlap when dealing
    with multiple timestamps of the same region.
    The user can either specify the exact (image) files to be used for validation and test datasets, or
    specify the ratios or counts of the splits.
    If ratios_or_counts are provided, the function will randomly assign the images to the splits.
    If val_img_files and test_img_files are provided, the function will use these files to create
    the validation and test datasets.
    The train dataset will be created from the remaining images.
    The function will return a list of datasets, with the first dataset being the train dataset.

    Args:
        dataset: The dataset to be split.
        val_img_files: A list of image files to be used for validation.
        test_img_files: A list of image files to be used for testing.
        ratios_or_counts: A list of ratios or counts to be used for splitting the dataset.
        If ratios_or_counts are provided, val_img_files and test_img_files will be ignored.
    Returns:
        A list of datasets, with the first dataset being the train dataset.
    """

    ratio_based_split, file_based_split = False, False
    if ratios_or_counts is not None:
        ratio_based_split = True
    if (val_img_files is not None) or (test_img_files is not None):
        file_based_split = True
    assert ratio_based_split or file_based_split, (
        "Either ratios_or_counts or train_files, val_files, test_files should be provided."
    )
    if ratio_based_split and file_based_split:
        print(
            "Both ratios_or_counts and train_files, val_files or test_files were provided. "
            "Only ratios_or_counts will be used for splitting."
        )
        file_based_split = False

    if ratio_based_split:
        return random_bbox_assignment(
            dataset=dataset,
            lengths=ratios_or_counts,
            generator=torch.Generator().manual_seed(0),
        )

    def _file_param_format(files) -> list[Path]:
        if files is None:
            return []
        if isinstance(files, str):
            files = [files]
        if isinstance(files, list):
            files = [Path(i) for i in files]
        if isinstance(files, Path):
            files = [files]
        return [i.resolve() for i in files]

    val_img_files, test_img_files = (
        _file_param_format(val_img_files),
        _file_param_format(test_img_files),
    )
    if isinstance(dataset, (IntersectionDataset)):
        img_dataset = dataset.datasets[0]
    elif isinstance(dataset, RasterDataset):
        img_dataset = dataset
    else:
        raise ValueError(
            "Dataset should be either a RasterDataset or IntersectionDataset."
        )
    dataset_items = list(
        img_dataset.index.intersection(img_dataset.index.bounds, objects=True)
    )
    dataset_files = [Path(i.object).resolve() for i in dataset_items]

    def _check_files(files_param: list[Path], out_list: list):
        if not files_param:
            return out_list
        file_found = False
        for f in files_param:
            if f not in dataset_files:
                print(f"File {str(f)} not found in dataset. Skipping")
            else:
                file_found = True
                out_list.append(dataset_items[dataset_files.index(f)])

        if not file_found:
            raise ValueError(
                "None of the provided files found in the dataset. Please check the files."
            )
        return out_list

    val_items, test_items = (
        _check_files(val_img_files, []),
        _check_files(test_img_files, []),
    )
    train_items = [
        i for i in dataset_items if i not in val_items and i not in test_items
    ]

    print([t.object for t in train_items])

    train_idx = Index(interleaved=False, properties=Property(dimension=3))
    for i in train_items:
        train_idx.insert(len(train_idx), i.bounds, i.object)
    new_indexes = [train_idx]
    if val_items:
        val_idx = Index(interleaved=False, properties=Property(dimension=3))
        for i in val_items:
            val_idx.insert(len(val_idx), i.bounds, i.object)
        new_indexes.append(val_idx)
    if test_items:
        test_idx = Index(interleaved=False, properties=Property(dimension=3))
        for i in test_items:
            test_idx.insert(len(test_idx), i.bounds, i.object)
        new_indexes.append(test_idx)

    new_datasets = []
    for index in new_indexes:
        if isinstance(dataset, (IntersectionDataset)):
            img_ds = dataset.datasets[0]
        else:
            img_ds = dataset
        ds = deepcopy(img_ds)
        ds.index = index
        ds.paths = [i.object for i in index.intersection(index.bounds, objects=True)]

        if isinstance(dataset, (LabelledRasterDataset, IntersectionDataset)):
            ds = dataset.__class__(
                ds,
                dataset.datasets[1],
                collate_fn=dataset.collate_fn,
                transforms=dataset.transforms,
            )

        new_datasets.append(ds)

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
