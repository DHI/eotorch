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

from eotorch.data.geodatasets import (
    PlottableClassificationDataset,
    PlottableImageDataset,
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


def get_dataset_args(ds):
    args = {}
    if isinstance(ds, RasterDataset):
        image_ds = ds
    elif isinstance(ds, IntersectionDataset):
        image_ds: PlottableImageDataset = ds.datasets[0]
        label_ds: PlottableClassificationDataset = ds.datasets[1]

        args["labels_dir"] = label_ds.paths
        args["label_glob"] = label_ds.filename_glob
        args["label_transforms"] = label_ds.transforms
        args["label_filename_regex"] = label_ds.filename_regex
        args["label_date_format"] = label_ds.date_format
        if label_ds.dataset_type == "classification":
            args["class_mapping"] = label_ds.class_mapping
            args["reduce_zero_label"] = label_ds.reduce_zero_label

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
        train_sampler_config: dict[str, Any] = None,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
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
                one sample, and the train_sampler_config is not used.
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
                Any sampler from torchgeo.samplers can be used here. The sampler will be initialized with the following parameters.
                Example of some other sampler configurations:
                train_sampler_config = {"type": "RandomPatchSampler", "length": length}
            mean: Mean of each image band. If given together with std, images are
                normalized on the fly as (image - mean) / std, which is also applied when
                predicting with the trained model. See
                eotorch.processing.normalize.band_statistics. For scaling from a range
                (low, high) to (0, 1), use mean=low and std=high - low.
            std: Standard deviation of each image band.
        """
        self.file_sampling = patch_size is None
        if self.file_sampling:
            _, ymin, _, ymax = FileSampler(train_dataset).bounds[0]
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

        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        # Plain floats, to be saved in the checkpoint
        mean = None if mean is None else [float(m) for m in mean]
        std = None if std is None else [float(s) for s in std]
        self.mean, self.std = mean, std

        def _format_dict_for_yaml(d: dict):
            out_dict = {}
            for k, v in d.items():
                if isinstance(v, Path):
                    out_dict[k] = str(v)
                else:
                    out_dict[k] = v
            return out_dict

        if self.train_dataset is not None:
            try:
                dataset_args = get_dataset_args(self.train_dataset)
            except Exception:
                print(
                    "Could not get dataset args from train_dataset, not saving to disk."
                )
                dataset_args = {}
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
                }
            )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """

        if stage in ["fit", "validate"]:
            if self.val_dataset is None:
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

            self.val_sampler = self._eval_sampler(self.val_dataset)

        if stage in ["fit"]:
            if self.file_sampling:
                self.train_sampler = FileSampler(self.train_dataset, shuffle=True)
            elif self.train_sampler_config is None:
                self.train_sampler = samplers.GriddedPatchSampler(
                    self.train_dataset, size=self.patch_size, stride=self.patch_size / 2
                )
            else:
                # Initialize the sampler based on the provided configuration
                local_config = self.train_sampler_config.copy()
                sampler_type = local_config.pop("type")
                if hasattr(samplers, sampler_type):
                    sampler_class = getattr(samplers, sampler_type)
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
                self.predict_dataset, stride=self.patch_size / 2
            )

    def _eval_sampler(self, dataset: GeoDataset, stride: float | None = None):
        if self.file_sampling:
            return FileSampler(dataset)
        return samplers.GriddedPatchSampler(
            dataset, size=self.patch_size, stride=stride
        )

    def preview_data_sampling(
        self,
        max_samples: int = 100,
        map=None,
        count_labels: bool = False,
        class_mapping: dict = None,
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
            count_labels: If True, also count and plot the distribution of label pixels
                across all datasets as a pie chart.
            class_mapping: Optional mapping of class values to names for labeling.
                If None, will attempt to use class_mapping from datasets.

        Returns:
            If count_labels is False:
                folium.Map: The interactive map with dataset boundaries and samplers visualized.
            If count_labels is True:
                tuple: (folium.Map, dict) containing the map and dictionary of pie chart figures per dataset.
        """
        import matplotlib.pyplot as plt

        from eotorch.plot import plot_samplers_on_map

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

        # If no class_mapping is provided but count_labels is True,
        # try to get it from datasets
        if class_mapping is None and count_labels:
            for dataset in datasets:
                if hasattr(dataset, "class_mapping"):
                    class_mapping = dataset.class_mapping
                if hasattr(dataset, "datasets"):
                    # If it's an IntersectionDataset, get the class mapping from the label dataset
                    for ds in dataset.datasets:
                        if hasattr(ds, "class_mapping"):
                            class_mapping = ds.class_mapping
                            break
                    break

        # Get the map with samplers (and label counts if requested)
        if count_labels:
            map_with_samplers, label_counts_per_dataset = plot_samplers_on_map(
                datasets=datasets,
                samplers=samplers,
                map=map,
                max_samples=max_samples,
                dataset_names=names,
                count_labels=True,
            )

            # First, collect all unique class values across all datasets to create a consistent color map
            all_class_values = set()
            for label_counts in label_counts_per_dataset.values():
                all_class_values.update(label_counts.keys())

            # Create a consistent color map for all classes
            import matplotlib.colors as mcolors

            # Get a colormap with distinct colors
            color_list = list(mcolors.TABLEAU_COLORS.values())
            # Extend the color list if we have more classes than colors
            while len(color_list) < len(all_class_values):
                color_list.extend(mcolors.TABLEAU_COLORS.values())

            # Map each class value to a specific color
            class_color_map = {
                val: color_list[i % len(color_list)]
                for i, val in enumerate(sorted(all_class_values))
            }

            # Get datasets with valid label counts
            valid_datasets = [
                name for name, counts in label_counts_per_dataset.items() if counts
            ]

            if valid_datasets:
                # Create a single figure with subfigures for each dataset
                n_datasets = len(valid_datasets)
                fig, axes = plt.subplots(1, n_datasets, figsize=(n_datasets * 6, 6))

                # Handle case where there's only one dataset (axes won't be an array)
                if n_datasets == 1:
                    axes = [axes]

                # Function for autopct display
                def make_autopct(values):
                    def autopct(pct):
                        total = sum(values)
                        val = int(round(pct * total / 100.0))
                        return f"{pct:.1f}%\n({val:,})"

                    return autopct

                # Create pie charts for each dataset
                for i, dataset_name in enumerate(valid_datasets):
                    label_counts = label_counts_per_dataset[dataset_name]

                    # Get labels, values and colors
                    labels = []
                    values = []
                    colors = []

                    for label_val, count in label_counts.items():
                        if class_mapping and label_val in class_mapping:
                            label_name = class_mapping[label_val]
                        else:
                            label_name = f"Class {label_val}"

                        labels.append(label_name)
                        values.append(count)
                        colors.append(class_color_map[label_val])

                    # Create pie chart on the appropriate subfigure
                    if values:
                        axes[i].pie(
                            values,
                            labels=labels,
                            autopct=make_autopct(values),
                            shadow=False,
                            startangle=90,
                            colors=colors,
                        )
                        axes[i].axis("equal")
                        axes[i].set_title(f"Label Distribution - {dataset_name}")

                plt.tight_layout()
                plt.show()
            else:
                print("No labels found in datasets. Returning only the map.")
            return map_with_samplers
        else:
            return plot_samplers_on_map(
                datasets=datasets,
                samplers=samplers,
                map=map,
                max_samples=max_samples,
                dataset_names=names,
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
        sampler = self._valid_attribute(f"{split}_sampler", "sampler")
        batch_size = self._valid_attribute(f"{split}_batch_size", "batch_size")

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last if split == "train" else False,
        )

    def on_after_batch_transfer(
        self, batch: dict[str, Tensor], dataloader_idx: int
    ) -> dict[str, Tensor]:
        """Normalize the images after they are transferred to the device."""
        if self.mean is not None:
            batch["image"] = normalize_image(batch["image"], self.mean, self.std)
        return batch
