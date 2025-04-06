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
from eotorch.data.splits import file_wise_split


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
    """

    A data module for segmentation tasks using GeoDataModule.
    Inherits from GeoDataModule and provides additional generic functionality for
    segmentation datasets.

    The following customizations are made:
    - In TorchGeo's standard approach when initializing subclasses of GeoDataModule, the dataset is passed as a class type
      (e.g., RasterDataset, IntersectionDataset) instead of an instance. This is due to the fact that the dataset
      might have to be initialized on multiple different workers / nodes. However, this requires the user to
      provide all the necessary parameters for dataset initialization, which might not be intuitive for
      users who are not familiar with the library. Instead, we would like to allow users of the high-level
      interfaces of eotorch to first initialize a dataset, inspect some samples from it, and then just pass
      the dataset instance directly to the data module. This behavior is supported by SegmentationDataModule.
    - Parameters such as the number of workers, persistent workers, and pin memory can be set modified when
        initializing the data module. This allows for more control over the data loading process.

    """

    def __init__(
        self,
        train_dataset: type[GeoDataset] | RasterDataset | IntersectionDataset,
        val_dataset: type[GeoDataset] | RasterDataset | IntersectionDataset = None,
        test_dataset: type[GeoDataset] | RasterDataset | IntersectionDataset = None,
        batch_size: int = 1,
        patch_size: int | tuple[int, int] = 64,
        length: int | None = None,
        num_workers: int = 0,
        persistent_workers: bool = False,
        pin_memory: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            # train_cls,
            train_dataset.__class__,
            batch_size,
            patch_size,
            length,
            num_workers,  # **dataset_kwargs
        )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        if not self.val_dataset:
            print("No val dataset provided. Performing default train / val split.")
            # TODO: Change this to > 10 once more splits are supported, for fewer files we probably want an ROI based split
            number_of_files_threshold = 1
            if len(self.train_dataset) > number_of_files_threshold:
                self.train_dataset, self.val_dataset = file_wise_split(
                    dataset=self.train_dataset,
                    ratios_or_counts=[0.9, 0.1],
                )
                print(
                    f"""
Since the number of files in the training dataset ({len(self.train_dataset)}) is greater than 
{number_of_files_threshold}, a  train / val split based on files was performed.
{len(self.val_dataset)} file(s) were assigned to the validation dataset.
If this is not desired, please provide a val_dataset to the data module.
                    """
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
                "dataset_args": _format_dict_for_yaml(get_dataset_args(train_dataset)),
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

        if stage in ["fit"]:
            self.train_sampler = GridGeoSampler(
                self.train_dataset, self.patch_size, self.patch_size / 2
            )
            # self.train_batch_sampler = RandomBatchGeoSampler(
            #     self.train_dataset, self.patch_size, self.batch_size, self.length
            # )
        if stage in ["fit", "validate"]:
            self.val_sampler = GridGeoSampler(
                self.val_dataset, self.patch_size, self.patch_size
            )
        if stage in ["test"]:
            self.test_sampler = GridGeoSampler(
                self.test_dataset, self.patch_size, self.patch_size
            )

    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
        """
        Same as GeoDataModule._dataloader_factory but allows for customization of dataloader behavior.

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

    def transfer_batch_to_device(
        self, batch: dict[str, Tensor], device: torch.device, dataloader_idx: int
    ) -> dict[str, Tensor]:
        """Transfer batch to device.

        Defines how custom data types are moved to the target device.

        Args:
            batch: A batch of data that needs to be transferred to a new device.
            device: The target device as defined in PyTorch.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A reference to the data on the new device.
        """
        # Non-Tensor values cannot be moved to a device

        for key in {"image_filepaths", "mask_filepaths"}:
            if key in batch:
                del batch[key]

        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        return batch
