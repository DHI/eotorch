from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo import samplers
from torchgeo.datamodules import GeoDataModule
from torchgeo.datasets import (
    GeoDataset,
    IntersectionDataset,
    RasterDataset,
)
from torchgeo.samplers.utils import _to_tuple

from eotorch.data.geodatasets import (
    CustomCacheRasterDataset,
    PlottableClassificationDataset,
)
from eotorch.data.samplers import FileSampler
from eotorch.data.splits import file_wise_split


def normalize_image(
    image: Tensor, mean: Sequence[float], std: Sequence[float]
) -> Tensor:
    """Normalize each band of a batch of images (B, C, H, W)."""
    mean = torch.as_tensor(mean, dtype=torch.float32, device=image.device)
    std = torch.as_tensor(std, dtype=torch.float32, device=image.device)
    return (image.float() - mean.view(-1, 1, 1)) / std.view(-1, 1, 1)


def get_dataset_args(ds: RasterDataset | IntersectionDataset) -> dict[str, Any]:
    """Settings of a dataset, saved in the checkpoint so inference can load data the same way.

    Plain TorchGeo datasets don't have the settings that only eotorch datasets have,
    such as class_mapping, so these are left out or set to None.
    """
    args = {}
    if isinstance(ds, RasterDataset):
        image_ds = ds
    elif isinstance(ds, IntersectionDataset):
        image_ds, label_ds = ds.datasets

        args["labels_dir"] = label_ds.paths
        args["label_glob"] = label_ds.filename_glob
        args["label_transforms"] = label_ds.transforms
        args["label_filename_regex"] = label_ds.filename_regex
        args["label_date_format"] = label_ds.date_format
        if isinstance(label_ds, PlottableClassificationDataset):
            args["class_mapping"] = label_ds.class_mapping
            args["reduce_zero_label"] = label_ds.reduce_zero_label
            args["nodata_value"] = label_ds.nodata_value
    else:
        raise TypeError(
            f"Expected a RasterDataset or IntersectionDataset, got {type(ds).__name__}"
        )

    args["images_dir"] = image_ds.paths
    args["all_image_bands"] = image_ds.all_bands
    args["rgb_bands"] = image_ds.rgb_bands
    args["image_glob"] = image_ds.filename_glob
    args["bands_to_return"] = image_ds.bands
    args["image_transforms"] = image_ds.transforms
    args["cache_size"] = getattr(image_ds, "cache_size", None)
    args["image_separate_files"] = image_ds.separate_files
    args["image_filename_regex"] = image_ds.filename_regex
    args["image_date_format"] = image_ds.date_format
    args["crs"] = image_ds.crs
    args["res"] = image_ds.res
    return args


class RasterDataModule(GeoDataModule):
    """
    A data module for segmentation and regression on raster datasets.

    The following customizations are made to TorchGeo's GeoDataModule:
    - In TorchGeo's standard approach when initializing subclasses of GeoDataModule, the dataset is passed as a class type
      (e.g., RasterDataset, IntersectionDataset) instead of an instance. This is due to the fact that the dataset
      might have to be initialized on multiple different workers / nodes. However, this requires the user to
      provide all the necessary parameters for dataset initialization, which might not be intuitive for
      users who are not familiar with the library. Instead, we would like to allow users of the high-level
      interfaces of eotorch to first initialize a dataset, inspect some samples from it, and then just pass
      the dataset instance directly to the data module. This behavior is supported by RasterDataModule.
    - Parameters such as the number of workers, persistent workers, and pin memory can be set modified when
        initializing the data module. This allows for more control over the data loading process.
    - The dataset arguments are saved in the checkpoint, so inference can load data the same way.
    - If no validation dataset is given, a file-wise validation split is held out from the
      training dataset when the data module is created (see val_split).
    - Images are normalized with the given mean and std instead of being divided by 255.
      Augmentations are applied after that and can be set as in TorchGeo, e.g.
      datamodule.train_aug = K.AugmentationSequential(K.RandomHorizontalFlip(),
      data_keys=None, keepdim=True).
    - Label pixels that augmentations pad, e.g. the corners after a rotation, are set
      to -1 for classes and to NaN for regression targets, which the eotorch tasks
      leave out of the loss and metrics.

    """

    def __init__(
        self,
        train_dataset: type[GeoDataset] | RasterDataset | IntersectionDataset = None,
        val_dataset: type[GeoDataset] | RasterDataset | IntersectionDataset = None,
        test_dataset: type[GeoDataset] | RasterDataset | IntersectionDataset = None,
        predict_dataset: type[GeoDataset] | RasterDataset | IntersectionDataset = None,
        batch_size: int = 1,
        patch_size: int | tuple[int, int] | None = 64,
        num_workers: int = 0,
        persistent_workers: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        train_sampler_config: dict[str, Any] | None = None,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        val_split: float | None = 0.1,
    ):
        """
        Initialize the RasterDataModule.

        Args:
            train_dataset: The training dataset.
            val_dataset: The validation dataset (optional).
            test_dataset: The test dataset (optional).
            batch_size: The batch size for data loading.
            patch_size: The size of the patches to be extracted from the images. Set to
                None if the datasets are already split into patches, one per file (e.g.
                with eotorch.processing.generate_patches_from_files). Each file is then
                one sample, read from that file only (overlapping files are not
                merged), and the train_sampler_config is not used.
            num_workers: The number of worker processes for data loading.
            persistent_workers: Whether to use persistent workers for data loading.
            pin_memory: Whether to pin memory for data loading.
            drop_last: Whether to drop the last incomplete batch. This is useful when
                using models with BatchNorm layers that fail with batch_size=1.
            train_sampler_config: The training sampler configuration (optional).
                If None, a default sampler will be used. The default is a GriddedPatchSampler with the following parameters:
                size=patch_size, stride=patch_size / 2.
                The way to specify the default sampler would be:
                train_sampler_config = {"type": "GriddedPatchSampler", "stride": patch_size / 2}

                Patch size does not need to be specified here, as it is already a parameter of the data module.
                It will be passed to the sampler as the 'size' arg automatically.
                Any sampler from torchgeo.samplers can be used here. Batch samplers, such
                as RandomBatchGeoSampler, also get the batch_size of the data module,
                unless it is given in the configuration.
                Example of some other sampler configurations:
                train_sampler_config = {"type": "RandomPatchSampler", "length": length}
                train_sampler_config = {"type": "RandomBatchGeoSampler", "length": length}
            mean: Mean of each image band. If given together with std, images are
                normalized on the fly as (image - mean) / std, which is also applied when
                predicting with the trained model. See
                eotorch.processing.normalize.band_statistics. For scaling from a range
                (low, high) to (0, 1), use mean=low and std=high - low.
            std: Standard deviation of each image band.
            val_split: Fraction of training files to hold out for validation when
                val_dataset is not given. The split is file-wise and seeded, so it is
                reproducible. Set to None to require an explicit val_dataset.
        """
        self.file_sampling = patch_size is None
        if self.file_sampling:
            _, ymin, _, ymax = FileSampler(train_dataset).index.geometry.iloc[0].bounds
            patch_size = round((ymax - ymin) / train_dataset.res[1])

        super().__init__(
            dataset_class=train_dataset.__class__,
            batch_size=batch_size,
            patch_size=patch_size,
            num_workers=num_workers,
        )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.predict_dataset = predict_dataset
        self.train_sampler_config = train_sampler_config
        self.val_split = val_split
        if val_dataset is None and train_dataset is not None:
            self.train_dataset, self.val_dataset = self._split_off_val(
                train_dataset, val_split
            )

        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        # Plain floats, to be saved in the checkpoint
        mean = None if mean is None else [float(m) for m in mean]
        std = None if std is None else [float(s) for s in std]
        self.mean, self.std = mean, std
        # Images are normalized with mean and std in on_after_batch_transfer. TorchGeo's
        # default aug would also divide them by 255, so it is replaced by doing nothing.
        # Augmentations can be set as in TorchGeo, e.g. as train_aug.
        self.aug = torch.nn.Identity()

        def _format_dict_for_yaml(d: dict):
            out_dict = {}
            for k, v in d.items():
                if isinstance(v, Path):
                    out_dict[k] = str(v)
                else:
                    out_dict[k] = v
            return out_dict

        if self.train_dataset is not None:
            dataset_args = get_dataset_args(self.train_dataset)
            self.save_hyperparameters(
                {
                    "dataset_args": _format_dict_for_yaml(dataset_args),
                    "batch_size": batch_size,
                    "patch_size": patch_size,
                    "num_workers": num_workers,
                    "persistent_workers": persistent_workers,
                    "pin_memory": pin_memory,
                    "drop_last": drop_last,
                    "train_sampler_config": train_sampler_config,
                    "mean": self.mean,
                    "std": self.std,
                    "val_split": val_split,
                }
            )

    @staticmethod
    def _split_off_val(
        dataset: GeoDataset, val_split: float | None
    ) -> tuple[GeoDataset, GeoDataset]:
        """Hold out a file-wise validation split from the training dataset."""
        if val_split is None:
            raise ValueError(
                "No val_dataset given and val_split=None. Pass a val_dataset or set val_split."
            )
        n_total = len(dataset)
        if n_total < 2:
            raise ValueError(
                f"No val_dataset given, and the training dataset has {n_total} file(s), "
                "but a file-wise validation split needs at least 2. Pass a val_dataset, "
                "e.g. created with eotorch.data.splits.file_wise_split or an ROI-based split."
            )
        train_dataset, val_dataset = file_wise_split(
            dataset=dataset, ratios_or_counts=[1 - val_split, val_split]
        )
        print(
            f"No val_dataset given: holding out {len(val_dataset)} of {n_total} files "
            f"for validation (val_split={val_split})."
        )
        return train_dataset, val_dataset

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """

        if stage in ["fit", "validate"]:
            self.val_sampler = self._eval_sampler(self.val_dataset)

        if stage in ["fit"]:
            if self.file_sampling:
                self.train_sampler = self._file_sampler(
                    self.train_dataset, shuffle=True
                )
            elif self.train_sampler_config is None:
                self.train_sampler = samplers.GriddedPatchSampler(
                    self.train_dataset,
                    size=self.patch_size,
                    stride=self._half_patch_size(),
                )
            else:
                # Initialize the sampler based on the provided configuration
                local_config = self.train_sampler_config.copy()
                sampler_type = local_config.pop("type")
                if hasattr(samplers, sampler_type):
                    sampler_class = getattr(samplers, sampler_type)
                    if issubclass(sampler_class, samplers.BatchGeoSampler):
                        local_config.setdefault("batch_size", self.batch_size)
                    self.train_sampler = sampler_class(
                        self.train_dataset,
                        size=self.patch_size,
                        **local_config,
                    )
                else:
                    raise ValueError(
                        f"Sampler type '{sampler_type}' not found in torchgeo.samplers."
                    )

        if stage in ["test"]:
            self.test_sampler = self._eval_sampler(self.test_dataset)
        if stage in ["predict"]:
            self.predict_sampler = self._eval_sampler(
                self.predict_dataset, stride=self._half_patch_size()
            )

    def _half_patch_size(self) -> tuple[float, float]:
        height, width = _to_tuple(self.patch_size)
        return height / 2, width / 2

    def _eval_sampler(self, dataset: GeoDataset, stride: float | None = None):
        if self.file_sampling:
            return self._file_sampler(dataset)
        return samplers.GriddedPatchSampler(
            dataset, size=self.patch_size, stride=stride
        )

    @staticmethod
    def _file_sampler(dataset: GeoDataset, shuffle: bool = False) -> FileSampler:
        # Each sample is a whole file, so read only that file instead of merging it
        # with every file it touches, which is much faster.
        for ds in getattr(dataset, "datasets", [dataset]):
            if isinstance(ds, CustomCacheRasterDataset):
                ds.single_file_reads = True
        return FileSampler(dataset, shuffle=shuffle)

    def preview_data_sampling(
        self,
        max_samples: int = 100,
        map=None,
        count_labels: bool = False,
        class_mapping: dict | None = None,
    ):
        """
        Visualize the dataset splits and their samplers on an interactive map.

        This method allows users to verify their data splits before training by
        displaying the datasets and how they are sampled using the configured samplers.

        Args:
            max_samples: Maximum number of samples to display per dataset.
                Set to a reasonable value to keep the map responsive.
            map: Optional existing folium map to add the visualization to.
                If None, a new map will be created.
            count_labels: If True, also count the label pixels of the samples and show
                their distribution per dataset as pie charts.
            class_mapping: Optional mapping of class values to names for the pie charts.
                If None, the class_mapping of the training labels is used.

        Returns:
            folium.Map: The interactive map with dataset boundaries and samplers visualized.
        """
        from eotorch.plot import plot_label_distribution, plot_samplers_on_map

        # Ensure samplers are set up
        if self.train_sampler is None:
            print("Setting up samplers for visualization...")
            self.setup(stage="fit")
            if self.test_dataset is not None and self.test_sampler is None:
                self.setup(stage="test")

        # Collect datasets and samplers
        datasets, samplers, names = [], [], []
        if self.train_dataset is not None and self.train_sampler is not None:
            datasets.append(self.train_dataset)
            samplers.append(self.train_sampler)
            names.append("Train")

        if self.val_dataset is not None and self.val_sampler is not None:
            datasets.append(self.val_dataset)
            samplers.append(self.val_sampler)
            names.append("Validation")

        if self.test_dataset is not None and self.test_sampler is not None:
            datasets.append(self.test_dataset)
            samplers.append(self.test_sampler)
            names.append("Test")

        if not datasets:
            raise ValueError(
                "No datasets with samplers available. "
                "Make sure to initialize the data module properly."
            )

        if not count_labels:
            return plot_samplers_on_map(
                datasets=datasets,
                samplers=samplers,
                map=map,
                max_samples=max_samples,
                dataset_names=names,
            )

        map_with_samplers, label_counts_per_dataset = plot_samplers_on_map(
            datasets=datasets,
            samplers=samplers,
            map=map,
            max_samples=max_samples,
            dataset_names=names,
            count_labels=True,
        )
        if class_mapping is None and isinstance(datasets[0], IntersectionDataset):
            class_mapping = getattr(datasets[0].datasets[1], "class_mapping", None)
        plot_label_distribution(label_counts_per_dataset, class_mapping)
        return map_with_samplers

    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
        """
        Same as GeoDataModule._dataloader_factory, but with the persistent_workers,
        pin_memory and drop_last settings of this data module.

        Args:
            split: Either 'train', 'val', 'test', or 'predict'.

        Returns:
            A collection of data loaders specifying samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset or sampler, or if the dataset or sampler has length 0.
        """
        dataset = self._valid_attribute(f"{split}_dataset", "dataset")
        sampler = self._valid_attribute(f"{split}_sampler", "sampler")
        batch_size = self._valid_attribute(f"{split}_batch_size", "batch_size")

        # A batch sampler yields whole batches, so it replaces sampler and batch_size
        batch_sampler = None
        if isinstance(sampler, samplers.BatchGeoSampler):
            batch_size, batch_sampler, sampler = 1, sampler, None

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            # drop_last can't be used together with a batch sampler
            drop_last=split == "train" and self.drop_last and batch_sampler is None,
        )

    def on_after_batch_transfer(
        self, batch: dict[str, Tensor], dataloader_idx: int
    ) -> dict[str, Tensor]:
        """Normalize the images after they are transferred to the device, then apply
        the augmentations of TorchGeo's data module."""
        if self.mean is not None:
            batch["image"] = normalize_image(batch["image"], self.mean, self.std)
        if "mask" not in batch:
            return super().on_after_batch_transfer(batch, dataloader_idx)

        # Augmentations such as rotations pad the mask with 0, which would be read as
        # class 0 or as a target of 0. An extra channel of ones is augmented with the
        # mask, to tell afterwards which pixels were padded.
        mask = batch["mask"]
        has_channels = mask.ndim == 4
        channels = mask if has_channels else mask.unsqueeze(1)
        inside = torch.ones_like(channels[:, :1])
        batch["mask"] = torch.cat([channels, inside], dim=1)

        batch = super().on_after_batch_transfer(batch, dataloader_idx)

        channels, inside = batch["mask"][:, :-1], batch["mask"][:, -1:]
        # The values of pixels without a label, as in the eotorch datasets
        no_label = torch.nan if channels.is_floating_point() else -1
        channels = channels.masked_fill(inside == 0, no_label)
        batch["mask"] = channels if has_channels else channels.squeeze(1)
        return batch
