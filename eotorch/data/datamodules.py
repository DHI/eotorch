from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo.datamodules import GeoDataModule
from torchgeo.datasets import (
    GeoDataset,
    IntersectionDataset,
    RasterDataset,
    random_grid_cell_assignment,
)
from torchgeo.samplers import BatchGeoSampler, GridGeoSampler, RandomBatchGeoSampler

from eotorch.data.geodatasets import (
    PlottabeLabelDataset,
    PlottableImageDataset,
    get_segmentation_dataset,
)


def get_dataset_args(ds):
    args = {}
    if isinstance(ds, RasterDataset):
        image_ds = ds
    elif isinstance(ds, IntersectionDataset):
        image_ds: PlottableImageDataset = ds.datasets[0]
        label_ds: PlottabeLabelDataset = ds.datasets[1]

        args["labels_dir"] = label_ds.paths
        args["class_mapping"] = label_ds.class_mapping
        args["label_glob"] = label_ds.filename_glob
        args["label_transforms"] = label_ds.transforms
        args["reduce_zero_label"] = label_ds.reduce_zero_label
        args["label_filename_regex"] = label_ds.filename_regex
        args["label_date_format"] = label_ds.date_format

    args["images_dir"] = image_ds.paths
    args["all_image_bands"] = image_ds.all_bands
    args["rgb_bands"] = image_ds.rgb_bands
    args["image_glob"] = image_ds.filename_glob
    args["bands_to_return"] = image_ds.bands
    args["image_transforms"] = image_ds.transforms
    args["cache_size"] = image_ds.cache_size
    args["image_separate_files"] = image_ds.separate_files
    args["image_filename_regex"] = image_ds.filename_regex
    args["image_date_format"] = image_ds.date_format
    args["crs"] = image_ds.crs
    args["res"] = image_ds.res
    return args


class SegmentationDataModule(GeoDataModule):
    def __init__(
        self,
        dataset: type[GeoDataset] | RasterDataset | IntersectionDataset,
        batch_size: int = 1,
        patch_size: int | tuple[int, int] = 64,
        length: int | None = None,
        num_workers: int = 0,
        persistent_workers: bool = False,
        pin_memory: bool = False,
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
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory

        def _format_dict_for_yaml(d: dict):
            out_dict = {}
            for k, v in d.items():
                if isinstance(v, Path):
                    out_dict[k] = str(v)
                else:
                    out_dict[k] = v
            return out_dict

        self.save_hyperparameters(
            {
                "dataset_args": _format_dict_for_yaml(dataset_kwargs),
                "batch_size": batch_size,
                "patch_size": patch_size,
                "length": length,
                "num_workers": num_workers,
                "persistent_workers": persistent_workers,
                "pin_memory": pin_memory,
            }
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

    # def val_dataloader(self):  # -> DataLoader[dict[str, Tensor]]:
    #     loader = self._dataloader_factory("val")
    #     # loader.num_workers = 0
    #     loader.num_workers = 2
    #     return loader

    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
        """
        Same as GeoDataModule._dataloader_factory but allows for customization of dataloader behavior.

        Implement one or more PyTorch DataLoaders.

        Args:
            split: Either 'train', 'val', 'test', or 'predict'.

        Returns:
            A collection of data loaders specifying samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset or sampler, or if the dataset or sampler has length 0.
        """
        dataset = self._valid_attribute(f"{split}_dataset", "dataset")
        sampler = self._valid_attribute(
            f"{split}_batch_sampler", f"{split}_sampler", "batch_sampler", "sampler"
        )
        batch_size = self._valid_attribute(f"{split}_batch_size", "batch_size")

        if isinstance(sampler, BatchGeoSampler):
            batch_size = 1
            batch_sampler = sampler
            sampler = None
        else:
            batch_sampler = None

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            # prefetch_factor=1,
        )
