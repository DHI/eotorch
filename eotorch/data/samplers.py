from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torchgeo.datasets import GeoDataset, IntersectionDataset
from torchgeo.samplers import GeoSampler


class FileSampler(GeoSampler):
    """Samples the full extent of each image file of a dataset.

    Meant for datasets that were split into patches beforehand, one patch per file,
    e.g. with :func:`eotorch.processing.generate_patches_from_files`. TorchGeo's
    PreChippedGeoSampler is deprecated, and its replacements sample from the combined
    area of all files, which doesn't work for separate, possibly overlapping patches.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        shuffle: bool = False,
        generator: torch.Generator | None = None,
    ) -> None:
        """
        Args:
            dataset: Dataset to sample from.
            shuffle: Whether to sample the files in a new random order every epoch.
            generator: Random number generator used for shuffling.
        """
        # The index of an IntersectionDataset has a row for every overlapping pair of
        # image and label files, so use the index of the images.
        if isinstance(dataset, IntersectionDataset):
            images = dataset.datasets[0].index
            labels = dataset.datasets[1].index.union_all()
            # Image files without labels can't be read as a sample
            self.index = images[images.geometry.intersection(labels).area > 0]
        else:
            self.index = dataset.index
        self.shuffle = shuffle
        self.generator = generator

    def __iter__(self) -> Iterator[tuple[slice, slice, slice]]:
        order = range(len(self))
        if self.shuffle:
            order = torch.randperm(len(self), generator=self.generator).tolist()
        for i in order:
            xmin, ymin, xmax, ymax = self.index.geometry.iloc[i].bounds
            # The time interval tells apart files with the same footprint
            interval = self.index.index[i]
            yield (
                slice(xmin, xmax),
                slice(ymin, ymax),
                slice(interval.left, interval.right),
            )

    def __len__(self) -> int:
        return len(self.index)


@dataclass(frozen=True)
class GridSampling:
    """Sample training patches on a regular grid that covers each file.

    Args:
        overlap: Fraction of the patch size by which neighbouring patches overlap,
            from 0 (patches touch) to just below 1.
    """

    overlap: float = 0.5

    def __post_init__(self) -> None:
        if not 0 <= self.overlap < 1:
            raise ValueError(
                f"overlap must be at least 0 and below 1, got {self.overlap}."
            )


@dataclass(frozen=True)
class RandomSampling:
    """Sample training patches at random locations, newly drawn every epoch.

    Args:
        samples_per_epoch: Number of patches per epoch. If None, about the number of
            non-overlapping patches that fit into the dataset.
    """

    samples_per_epoch: int | None = None

    def __post_init__(self) -> None:
        if self.samples_per_epoch is not None and self.samples_per_epoch < 1:
            raise ValueError(
                f"samples_per_epoch must be at least 1, got {self.samples_per_epoch}."
            )


@dataclass(frozen=True)
class FileSampling:
    """Use each file as one sample, in every split (train, val, test and predict).

    For datasets that were split into patches beforehand, one patch per file, e.g.
    with :func:`eotorch.processing.generate_patches_from_files`. The patch size is
    read from the files. Training files are shuffled every epoch.
    """
