from pathlib import Path
from glob import glob

import pandas as pd
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torch import Tensor
import rasterio as rst


def _load_patches(wildcard : str, patch_dir : str) -> pd.DataFrame:
    """
    Create a dataframe of feature/label patch paths matching a wildcard.

    Parameters:
        wildcard (str): 
            Glob wildcard to match patch filenames.
        patch_dir (str): 
            Patch directory.

    Returns:
        pd.DataFrame:
            Dataframe with feature and label path columns.
    """    
    df = pd.DataFrame().from_dict({
        'feature' : glob(rf'{patch_dir}/{wildcard}_*feature.tiff'),
        'label' : glob(rf'{patch_dir}/{wildcard}_*label.tiff')
    })
    if df.empty:
        print('Warning: No patches found', UserWarning, stacklevel=2)
        return pd.DataFrame(columns=['feature', 'label'])
    else:
        return df


class DatasetFromPatches(Dataset):
    def __init__(self, wildcard, patch_dir):
        self.patches = _load_patches(wildcard, patch_dir)
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        with rst.open(self.patches.iloc[idx, 0]) as feature_src, rst.open(self.patches.iloc[idx, 1]) as label_src:
            img = feature_src.read()
            label = label_src.read(indexes=1)
        return Tensor(img), Tensor(label).long()
    

class PatchDataModule(LightningDataModule):
    def __init__(
        self, 
        wildcard: str, 
        patch_dir: str | Path,
        batch_size: int = 8,
        split : list = [0.8, 0.2]
    ):
        super().__init__()
        self.dataset = DatasetFromPatches(wildcard, patch_dir)
        self.batch_size = batch_size
        self.split = split
    
    def setup(self, stage=None):
        self.train, self.val = random_split(self.dataset, self.split)
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)
    
    def predict_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)