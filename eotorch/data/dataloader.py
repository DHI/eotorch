from pathlib import Path
from glob import glob
from typing import Any, Callable
import warnings

import numpy as np
import pandas as pd
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torch import Tensor
import rasterio as rst


def _load_patches(
    patch_dir: str | Path,
    image_suffix: str = "feature",
    label_suffix: str = "label",
    distance_suffix: str | None = None,
) -> pd.DataFrame:
    """
    Create a dataframe of feature/label(/distance) patch paths in a patch directory.

    Parameters:
        patch_dir (str | Path): Path to the directory containing patches.
        image_suffix (str): Suffix for feature patch files. Defaults to "feature".
        label_suffix (str): Suffix for label patch files. Defaults to "label".
        distance_suffix (str | None): Suffix for precomputed distance-map patch
            files (matched as ``*_{distance_suffix}.npy``), e.g. as written by
            an external distance-map precompute step. If None, no distance column
            is added.

    Returns:
        pd.DataFrame:
            Dataframe with feature and label path columns, plus a distance
            column when `distance_suffix` is given.
    """
    patch_path = Path(patch_dir)
    feature_paths = glob(str(patch_path / f'*_{image_suffix}.tiff'))
    label_paths = glob(str(patch_path / f'*_{label_suffix}.tiff'))

    feature_map = {
        Path(path).name.removesuffix(f'_{image_suffix}.tiff'): path
        for path in feature_paths
    }
    label_map = {
        Path(path).name.removesuffix(f'_{label_suffix}.tiff'): path
        for path in label_paths
    }

    patch_keys = sorted(feature_map.keys() & label_map.keys())
    if not patch_keys:
        warnings.warn(f'No patches found in {patch_path}', UserWarning, stacklevel=2)
        columns = ['feature', 'label'] + (['distance'] if distance_suffix else [])
        return pd.DataFrame(columns=columns)

    if len(patch_keys) != len(feature_map) or len(patch_keys) != len(label_map):
        warnings.warn(
            f'Ignoring unmatched feature/label patches in {patch_path}',
            UserWarning,
            stacklevel=2,
        )

    data = {
        'feature': [feature_map[key] for key in patch_keys],
        'label': [label_map[key] for key in patch_keys],
    }

    if distance_suffix is not None:
        distance_paths = glob(str(patch_path / f'*_{distance_suffix}.npy'))
        distance_map = {
            Path(path).name.removesuffix(f'_{distance_suffix}.npy'): path
            for path in distance_paths
        }
        missing = [key for key in patch_keys if key not in distance_map]
        if missing:
            raise FileNotFoundError(
                f"Missing precomputed distance maps for {len(missing)} patch(es) in {patch_path} "
                f"(expected '<key>_{distance_suffix}.npy'). Run the distance-map precompute step "
                "first, or omit distance_suffix."
            )
        data['distance'] = [distance_map[key] for key in patch_keys]

    return pd.DataFrame(data)


class DatasetFromPatches(Dataset):
    """Torch dataset over feature/label (and optionally distance-map) patch files on disk."""

    def __init__(
        self,
        patch_dir: str | Path,
        transform: Callable[..., Any] | None = None,
        image_suffix: str = "feature",
        label_suffix: str = "label",
        distance_suffix: str | None = None,
    ):
        """Index the patches in `patch_dir` and infer patch size from the first feature file."""
        self.patches = _load_patches(
            patch_dir, image_suffix=image_suffix, label_suffix=label_suffix, distance_suffix=distance_suffix
        )
        self.transform = transform
        self.distance_suffix = distance_suffix

        self.patch_size = None
        if len(self.patches) > 0:
            with rst.open(self.patches.iloc[0]['feature']) as feature_src:
                self.patch_size = feature_src.width

    def __len__(self):
        """Number of indexed patches."""
        return len(self.patches)

    def __getitem__(self, idx):
        """Load and transform one patch, returning (image, label) or (image, label, distance_map)."""
        row = self.patches.iloc[idx]
        with rst.open(row['feature']) as feature_src, rst.open(row['label']) as label_src:
            img = feature_src.read()
            label = label_src.read(indexes=1)

        if self.transform is not None:
            try:
                transformed = self.transform(image=img, mask=label)
            except TypeError:
                transformed = self.transform(img, label)

            if isinstance(transformed, dict):
                img = transformed.get('image', img)
                label = transformed.get('mask', transformed.get('label', label))
            elif isinstance(transformed, tuple) and len(transformed) == 2:
                img, label = transformed
            else:
                raise ValueError(
                    'Transform must return either (image, label) or a dict with image/mask keys.'
                )

        label_tensor = Tensor(label)
        if np.issubdtype(label.dtype, np.floating):
            label_tensor = label_tensor.float()
        else:
            label_tensor = label_tensor.long()

        if self.distance_suffix is not None:
            distance_map = np.load(row['distance'])
            return Tensor(img), label_tensor, Tensor(distance_map)

        return Tensor(img), label_tensor
    

class PatchDataModule(LightningDataModule):
    """Lightning DataModule wrapping train/val `DatasetFromPatches` with an optional random val split."""

    def __init__(
        self,
        train_patch_dir: str | Path,
        val_patch_dir: str | Path | None = None,
        batch_size: int = 8,
        val_fraction: float = 0.2,
        transform: Callable[..., Any] | None = None,
        image_suffix: str = "feature",
        label_suffix: str = "label",
        distance_suffix: str | None = None,
        num_workers: int = 0,
        persistent_workers: bool = True,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.train_dataset = DatasetFromPatches(
            train_patch_dir, transform=transform, image_suffix=image_suffix, label_suffix=label_suffix, distance_suffix=distance_suffix
        )
        self.val_dataset = DatasetFromPatches(
            val_patch_dir, transform=transform, image_suffix=image_suffix, label_suffix=label_suffix, distance_suffix=distance_suffix
        ) if val_patch_dir is not None else None
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.patch_size = self.train_dataset.patch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers and num_workers > 0
        self.pin_memory = pin_memory
        self._train_split = None
        self._val_split = None

        self.save_hyperparameters(
            {
                "train_patch_dir": str(train_patch_dir),
                "val_patch_dir": str(val_patch_dir) if val_patch_dir is not None else None,
                "batch_size": batch_size,
                "patch_size": self.patch_size,
                "val_fraction": val_fraction,
                "image_suffix": image_suffix,
                "label_suffix": label_suffix,
                "distance_suffix": distance_suffix,
                "num_workers": num_workers,
                "persistent_workers": self.persistent_workers,
                "pin_memory": pin_memory,
            }
        )

    def setup(self, stage=None):
        """Assign train/val splits: the provided val_dataset if given, else a random split of train_dataset."""
        if self.val_dataset is not None:
            self._train_split = self.train_dataset
            self._val_split = self.val_dataset
            return None

        dataset_size = len(self.train_dataset)
        val_size = int(dataset_size * self.val_fraction)

        if self.val_fraction > 0 and dataset_size > 0 and val_size == 0:
            val_size = 1

        train_size = dataset_size - val_size
        self._train_split, self._val_split = random_split(self.train_dataset, [train_size, val_size])
    
    def train_dataloader(self):
        """DataLoader over the training split, shuffled."""
        dataset = self._train_split or self.train_dataset
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        """DataLoader over the validation split."""
        dataset = self._val_split or self.val_dataset
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self):
        """DataLoader used for prediction: val split if available, else the training data."""
        dataset = self._val_split or self.val_dataset or self.train_dataset
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )