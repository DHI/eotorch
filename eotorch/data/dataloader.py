from pathlib import Path
from glob import glob
from typing import Any, Callable
import warnings

import pandas as pd
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torch import Tensor
import rasterio as rst


def _load_patches(patch_dir: str | Path, image_suffix: str = "feature", label_suffix: str = "label") -> pd.DataFrame:
    """
    Create a dataframe of feature/label patch paths in a patch directory.

    Parameters:
        patch_dir (str | Path): Path to the directory containing patches.
        image_suffix (str): Suffix for feature patch files. Defaults to "feature".
        label_suffix (str): Suffix for label patch files. Defaults to "label".

    Returns:
        pd.DataFrame:
            Dataframe with feature and label path columns.
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
        return pd.DataFrame(columns=['feature', 'label'])

    if len(patch_keys) != len(feature_map) or len(patch_keys) != len(label_map):
        warnings.warn(
            f'Ignoring unmatched feature/label patches in {patch_path}',
            UserWarning,
            stacklevel=2,
        )

    return pd.DataFrame(
        {
            'feature': [feature_map[key] for key in patch_keys],
            'label': [label_map[key] for key in patch_keys],
        }
    )


class DatasetFromPatches(Dataset):
    def __init__(
        self,
        patch_dir: str | Path,
        transform: Callable[..., Any] | None = None,
        image_suffix: str = "feature",
        label_suffix: str = "label",
    ):
        self.patches = _load_patches(patch_dir, image_suffix=image_suffix, label_suffix=label_suffix)
        self.transform = transform
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        with rst.open(self.patches.iloc[idx, 0]) as feature_src, rst.open(self.patches.iloc[idx, 1]) as label_src:
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

        return Tensor(img), Tensor(label).long()
    

class PatchDataModule(LightningDataModule):
    def __init__(
        self, 
        train_patch_dir: str | Path,
        val_patch_dir: str | Path | None = None,
        batch_size: int = 8,
        val_fraction: float = 0.2,
        transform: Callable[..., Any] | None = None,
        image_suffix: str = "feature",
        label_suffix: str = "label",
    ):
        super().__init__()
        self.train_dataset = DatasetFromPatches(train_patch_dir, transform=transform, image_suffix=image_suffix, label_suffix=label_suffix)
        self.val_dataset = DatasetFromPatches(val_patch_dir, transform=transform, image_suffix=image_suffix, label_suffix=label_suffix) if val_patch_dir is not None else None
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self._train_split = None
        self._val_split = None
    
    def setup(self, stage=None):
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
        dataset = self._train_split or self.train_dataset
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        dataset = self._val_split or self.val_dataset
        return DataLoader(dataset, batch_size=self.batch_size)
    
    def predict_dataloader(self):
        dataset = self._val_split or self.val_dataset or self.train_dataset
        return DataLoader(dataset, batch_size=self.batch_size)