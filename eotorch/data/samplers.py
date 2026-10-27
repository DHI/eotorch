from collections.abc import Iterator

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
            dataset = dataset.datasets[0]
        self.bounds = [geometry.bounds for geometry in dataset.index.geometry]
        self.shuffle = shuffle
        self.generator = generator

    def __iter__(self) -> Iterator[tuple[slice, slice]]:
        order = range(len(self))
        if self.shuffle:
            order = torch.randperm(len(self), generator=self.generator).tolist()
        for i in order:
            xmin, ymin, xmax, ymax = self.bounds[i]
            yield slice(xmin, xmax), slice(ymin, ymax)

    def __len__(self) -> int:
        return len(self.bounds)
