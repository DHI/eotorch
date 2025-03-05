from typing import Any

import torch
from torchgeo.datamodules import GeoDataModule
from torchgeo.datasets import (
    GeoDataset,
    IntersectionDataset,
    RasterDataset,
    random_grid_cell_assignment,
)
from torchgeo.samplers import GridGeoSampler, RandomBatchGeoSampler, RandomGeoSampler

from eotorch.data.geodatasets import get_segmentation_dataset


def get_dataset_args(ds):
    args = {}
    if isinstance(ds, RasterDataset):
        image_ds = ds
    elif isinstance(ds, IntersectionDataset):
        image_ds: RasterDataset = ds.datasets[0]
        label_ds: RasterDataset = ds.datasets[1]

        args["labels_dir"] = label_ds.paths
        args["class_mapping"] = label_ds.class_mapping
        args["label_glob"] = label_ds.filename_glob
        args["label_transforms"] = label_ds.transforms

    args["images_dir"] = image_ds.paths
    args["all_image_bands"] = image_ds.all_bands
    args["rgb_bands"] = image_ds.rgb_bands
    args["image_glob"] = image_ds.filename_glob
    args["bands_to_return"] = image_ds.bands
    args["image_transforms"] = image_ds.transforms
    args["cache"] = image_ds.cache

    return args


class SegmentationDataModule(GeoDataModule):
    def __init__(
        self,
        dataset: type[GeoDataset] | RasterDataset | IntersectionDataset,
        batch_size: int = 1,
        patch_size: int | tuple[int, int] = 64,
        length: int | None = None,
        num_workers: int = 0,
        **kwargs: Any,
    ):
        if isinstance(dataset, (RasterDataset, IntersectionDataset)):
            dataset_kwargs = get_dataset_args(dataset)
            dataset_class = type(dataset)

        else:
            dataset_kwargs = kwargs
            dataset_class = dataset
        super().__init__(
            dataset_class, batch_size, patch_size, length, num_workers, **dataset_kwargs
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """

        self.dataset = get_segmentation_dataset(**self.kwargs)

        generator = torch.Generator().manual_seed(0)
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = random_grid_cell_assignment(
            dataset=self.dataset,
            fractions=[0.8, 0.1, 0.1],
            grid_size=6,
            generator=generator,
        )

        if stage in ["fit"]:
            self.train_batch_sampler = RandomBatchGeoSampler(
                self.train_dataset, self.patch_size, self.batch_size, self.length
            )
        if stage in ["fit", "validate"]:
            self.val_sampler = GridGeoSampler(
                self.val_dataset, self.patch_size, self.patch_size
            )
        if stage in ["test"]:
            self.test_sampler = GridGeoSampler(
                self.test_dataset, self.patch_size, self.patch_size
            )

    def val_dataloader(self):  # -> DataLoader[dict[str, Tensor]]:
        loader = self._dataloader_factory("val")
        # loader.num_workers = 0
        loader.num_workers = 2
        return loader
