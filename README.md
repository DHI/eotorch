# EOTorch

**EOTorch** is a Python library for geospatial deep learning built on top of [TorchGeo](https://github.com/microsoft/torchgeo) and [PyTorch Lightning](https://lightning.ai/). It provides a streamlined workflow for common Earth observation (EO) tasks: semantic segmentation and pixelwise regression. 

## Key Features

- 🗺️ **Easy dataset creation** from GeoTIFF images and labels with automatic CRS/resolution handling
- 🔀 **Flexible data splitting** - file-wise random splits or spatial (AOI-based) splits  
- 📊 **Built-in visualization** - plot samples, preview datasets on interactive maps
- 🧠 **Flexible training** - eotorch and [segmentation_models_pytorch](https://smp.readthedocs.io) models, combinable losses, sparse labels
- 🧩 **Pre-patched data** - generate patch files once and train on them with the same tools
- ⚖️ **On-the-fly normalization** - no need to write normalized copies of your images
- 🚀 **Seamless inference** - run predictions on large rasters with automatic patching and optional blending of overlapping patches
- 🔧 **TorchGeo compatible** - works with existing TorchGeo components while simplifying common workflows

---

## Table of Contents

- [Installation](#installation)
- [Core Concepts](#core-concepts)
  - [1. Dataset Creation with `get_segmentation_dataset()`](#1-dataset-creation-with-get_segmentation_dataset)
  - [2. Data Splitting with `splits`](#2-data-splitting-with-splits)
  - [3. Data Loading with `RasterDataModule`](#3-data-loading-with-rasterdatamodule)
  - [4. Training with `SegmentationTask` and `RegressionTask`](#4-training-with-segmentationtask-and-regressiontask)
  - [5. Inference with `predict_on_tif_file()`](#5-inference-with-predict_on_tif_file)
  - [6. Preprocessing with `eotorch.processing`](#6-preprocessing-with-eotorchprocessing)
- [Key Differences from TorchGeo](#key-differences-from-torchgeo)
- [API Reference](#api-reference)
- [Examples](#examples)

---

## Installation

### Basic Installation
To install the base package (with core functionality, excluding extras):

```bash
pip install git+https://github.com/DHI/eotorch.git
```

## Optional Dependencies

EOTorch supports several optional features. You can install them optional extras.

### Available extras

| Extra       | Description                                 |
|-------------|---------------------------------------------|
| `dev`       | Developer tools (formatting, testing, docs) |
| `test`      | Testing tools (pytest, coverage, mypy)      |
| `notebooks` | Jupyter notebook support                    |
| `dinov3`    | The `dinov3_upernet` model (`transformers`) |
| `cuda126`   | GPU support for Windows with CUDA 12.6      |

#### Example: Install with optional extras for Jupyter notebook and Windows GPU Support
```bash
pip install "git+https://github.com/DHI/eotorch.git[cuda126,notebooks]"
```

## ⚠ Windows GPU Installation Notes

If you're using **Windows with an NVIDIA GPU**, follow these guidelines:

- Your **GPU driver** must support **CUDA 12.6**. You can check your driver compatibility by running:

  ```bash
  nvidia-smi
  ```

  Look for the `Driver Version` and `CUDA Version`. As long as the driver supports CUDA **12.6 or newer**, the `cuda126` optional dependency *should* work.

- The `cuda126` optional dependency installs:
  - `torch>=2.7.0`
  - `torchvision>=0.22.0`
  using the official [PyTorch CUDA 12.6 wheels](https://download.pytorch.org/whl/cu126).
---

## Development
To set up a local development environment for a new project using UV:

```bash
uv init
uv add "git+https://github.com/DHI/eotorch.git[<add_extras_here>]"
```

---

## Examples

You can find usage examples in the [notebooks](notebooks/) directory.

* For segmentation use cases, check [notebooks/segmentation.ipynb](notebooks/segmentation.ipynb)

---

## Core Concepts

### 1. Dataset Creation with `get_segmentation_dataset()`

The primary entry point for creating datasets. Unlike TorchGeo where you typically subclass `RasterDataset`, EOTorch provides a factory function that handles most common use cases:

```python
from eotorch.data import get_segmentation_dataset

# Create a dataset with images and labels
ds = get_segmentation_dataset(
    images_dir="path/to/images",
    labels_dir="path/to/labels",
    all_image_bands=("B", "G", "R", "NIR"),  # Band names in your files
    rgb_bands=("R", "G", "B"),                # Bands for visualization
    class_mapping={1: "Water", 2: "Forest", 3: "Urban"},
    reduce_zero_label=True,  # Shift labels (1,2,3) → (0,1,2) for training
)

# Visualize samples to verify correct image-label pairing
ds.plot_samples(n=3, patch_size=256, show_filepaths=True)
```

#### Temporal Matching via Regex

When images and labels have temporal information in filenames, use regex to match them. The key is to define a named capture group `(?P<date>...)` that extracts the date portion from the filename.

```python
ds = get_segmentation_dataset(
    images_dir="path/to/images",
    labels_dir="path/to/labels",
    # Match files by date extracted from filename
    image_filename_regex=r"(?P<date>\d{8})_S2[AB]_.*\.tif",
    label_filename_regex=r"(?P<date>\d{8})_.*\.tif",
    image_date_format="%Y%m%d",
    label_date_format="%Y%m%d",
    # ...
)
```

##### Common Regex Patterns

| Filename Example | Regex Pattern | Date Format |
|------------------|---------------|-------------|
| `20240315_S2A_T32UNF.tif` | `r"(?P<date>\d{8})_S2[AB]_.*\.tif"` | `%Y%m%d` |
| `scene_2024-03-15_rgb.tif` | `r"scene_(?P<date>\d{4}-\d{2}-\d{2})_.*\.tif"` | `%Y-%m-%d` |
| `tile_15MAR2024_processed.tif` | `r"tile_(?P<date>\d{2}[A-Z]{3}\d{4})_.*\.tif"` | `%d%b%Y` |
| `S2B_20240315T102341.tif` | `r"S2[AB]_(?P<date>\d{8})T\d+\.tif"` | `%Y%m%d` |
| `image_2024_001.tif` (day of year) | `r"image_(?P<date>\d{4}_\d{3})\.tif"` | `%Y_%j` |
| `data_202403.tif` (monthly) | `r"data_(?P<date>\d{6})\.tif"` | `%Y%m` |

##### Detailed Examples

**Example 1: Sentinel-2 style filenames**
```
Images:  20240315_S2A_32UNF_20240315_0_L2A.tif
Labels:  20240315_labels.tif
```
```python
image_filename_regex=r"(?P<date>\d{8})_S2[AB]_\w+_\d{8}_\d+_L2A.*\.tif"
label_filename_regex=r"(?P<date>\d{8})_labels\.tif"
image_date_format="%Y%m%d"
label_date_format="%Y%m%d"
```

**Example 2: ISO date format with dashes**
```
Images:  satellite_2024-03-15_B04.tif
Labels:  mask_2024-03-15.tif
```
```python
image_filename_regex=r"satellite_(?P<date>\d{4}-\d{2}-\d{2})_.*\.tif"
label_filename_regex=r"mask_(?P<date>\d{4}-\d{2}-\d{2})\.tif"
image_date_format="%Y-%m-%d"
label_date_format="%Y-%m-%d"
```

**Example 3: Year-only matching (for annual composites)**
```
Images:  landcover_2023_composite.tif
Labels:  training_2023_v2.tif
```
```python
image_filename_regex=r"landcover_(?P<date>\d{4})_.*\.tif"
label_filename_regex=r"training_(?P<date>\d{4})_.*\.tif"
image_date_format="%Y"
label_date_format="%Y"
```

> **Tip**: If your images and labels don't have temporal information to match, you can omit the regex parameters entirely. EOTorch will then match based on spatial overlap (CRS and bounds).

### 2. Data Splitting with `splits`

EOTorch provides two main splitting strategies:

#### File-wise Split
Assigns entire image files to train/val/test sets, ensuring no spatial leakage between splits.

**Option 1: Random split by ratio**
```python
from eotorch.data import splits

# Split 80% train, 20% validation (randomly assigned)
train_ds, val_ds = splits.file_wise_split(
    dataset=ds, 
    ratios_or_counts=[0.8, 0.2]
)

# Three-way split: 70% train, 15% val, 15% test
train_ds, val_ds, test_ds = splits.file_wise_split(
    dataset=ds,
    ratios_or_counts=[0.7, 0.15, 0.15]
)
```

**Option 2: Specify exact files**
```python
# Explicitly assign files to validation set (remaining go to train)
train_ds, val_ds = splits.file_wise_split(
    dataset=ds,
    val_img_files=["scene1.tif", "scene2.tif"]
)

# Specify both validation and test files
train_ds, val_ds, test_ds = splits.file_wise_split(
    dataset=ds,
    val_img_files=["val_scene1.tif", "val_scene2.tif"],
    test_img_files=["test_scene1.tif"]
)

# Or use glob patterns to select files
train_ds, val_ds = splits.file_wise_split(
    dataset=ds,
    val_img_glob="/path/to/data/*_2024_*.tif"  # All 2024 scenes for validation
)
```

#### AOI Split (Spatial)
Split based on geographic regions defined in GeoJSON/Shapefile:

```python
train_ds, val_ds = splits.aoi_split(
    dataset=ds,
    aoi_files="validation_region.geojson",
    buffer_size_metres=100,  # Optional buffer to prevent spatial leakage
)
```

> **Tip: Creating AOI files interactively**  
> In a Jupyter notebook, simply display your dataset variable (`ds`) to see an interactive map showing all image and label extents. You can use this map to visually identify regions, then draw and export polygons using the draw tool of the folium map that appears.

### 3. Data Loading with `RasterDataModule`

The data module wraps your datasets for PyTorch Lightning training, for both segmentation and regression:

```python
from eotorch.data import RasterDataModule

module = RasterDataModule(
    train_dataset=train_ds,
    val_dataset=val_ds,
    batch_size=16,
    patch_size=224,         # Size of patches sampled from images
    num_workers=4,          # Also works on Windows
    pin_memory=True,
)
```

#### Custom Samplers

Control how patches are sampled during training:

```python
module = RasterDataModule(
    train_dataset=train_ds,
    # Default: GriddedPatchSampler with stride=patch_size/2
    # For random sampling:
    train_sampler_config={"type": "RandomPatchSampler", "length": 5000},
    # ...
)
```

#### On-the-fly Normalization

Instead of writing normalized copies of your images, pass the mean and standard deviation of each band. The images are then normalized on the GPU as `(image - mean) / std`, and the same is done automatically when predicting with the trained model:

```python
from eotorch.processing.normalize import band_statistics

mean, std = band_statistics(train_ds)  # ignores nodata pixels
module = RasterDataModule(train_dataset=train_ds, val_dataset=val_ds, mean=mean, std=std)
```

To scale a fixed range `(low, high)` of each band to `(0, 1)`, use `mean=low` and `std=high - low`.

#### Pre-patched Data

If your data is already split into patches, one patch per file (see [Patch Generation](#patch-generation)), load the patch directories like any other dataset and set `patch_size=None` to use each file as one sample:

```python
train_ds = get_segmentation_dataset(
    images_dir="patches/train", labels_dir="patches/train",
    image_glob="*_image.tif", label_glob="*_label.tif", ...
)
val_ds = get_segmentation_dataset(images_dir="patches/val", labels_dir="patches/val", ...)

module = RasterDataModule(train_dataset=train_ds, val_dataset=val_ds, patch_size=None, batch_size=16)
```

### 4. Training with `SegmentationTask` and `RegressionTask`

Both tasks are based on the corresponding TorchGeo tasks and can be trained with a Lightning `Trainer`:

```python
from lightning.pytorch import Trainer
from eotorch.data import SegmentationTask

task = SegmentationTask(
    num_classes=3,
    in_channels=4,
    model="unet",                # see below
    backbone="resnet34",
    loss={"ce": 0.5, "dice": 0.5},
    ignore_index=-1,             # e.g. unlabelled pixels with reduce_zero_label=True
    class_names=["Water", "Forest", "Urban"],
)
Trainer(max_epochs=50).fit(task, datamodule=module)
```

**Models**: `model` is one of the eotorch models (`"deepresunet"`, `"unet_lite"`, a small UNet whose size is set by `num_filters`, and `"dinov3_upernet"`), TorchGeo's `"fcn"`, or any architecture of [segmentation_models_pytorch](https://smp.readthedocs.io/en/latest/models.html) such as `"unet"`, `"unetplusplus"`, `"deeplabv3plus"`, `"upernet"`, `"segformer"` or `"dpt"`. Model-specific arguments go into `model_kwargs`, e.g. `model_kwargs={"decoder_channels": [128, 64, 32, 16, 8]}`. Pretrained encoders can be loaded with `weights` (`True` for ImageNet, or a TorchGeo weight enum), and frozen with `freeze_backbone=True`.

**Losses**: `"ce"`, `"bce"`, `"dice"`, `"focal"`, `"jaccard"`, `"tversky"`, `"boundary"` and `"boundary_distance"` (Kervadec et al.), or a weighted combination of them as a dict. Extra arguments per loss go into `loss_kwargs`, e.g. `loss_kwargs={"focal": {"gamma": 2.0}}`. The `LossWeightScheduler` callback in `eotorch.models.callbacks` can ramp up the weight of one term, which is useful for the boundary losses.

**Binary segmentation**: use `num_classes=1, task="binary"` and e.g. `loss="bce"`.

**Regression** works the same way. Pixels where the target equals `nodata_value` are left out of the loss and metrics, which allows training on sparse labels:

```python
from eotorch.data import RegressionTask, get_regression_dataset

ds = get_regression_dataset(images_dir=..., labels_dir=..., nodata_value=-9999, ...)
task = RegressionTask(in_channels=6, model="unet", loss="huber", nodata_value=-9999)
```

### 5. Inference with `predict_on_tif_file()`

Run predictions on large rasters directly from a trained checkpoint. This method automatically handles patching, batching, and stitching, and reads configuration (patch size, class mapping, etc.) from the checkpoint:

```python
from eotorch.data import SegmentationTask

result_path = SegmentationTask.predict_on_tif_file(
    tif_file_path="large_image.tif",
    checkpoint_path="checkpoints/best-model.ckpt",
    batch_size=8,
    blend=True,               # Optional: blend overlapping patches to avoid seams
    show_results=True,        # Display preview in notebook
    out_file_path="predictions/output.tif",  # Optional, auto-generated if not set
)
```

The method automatically:
- Loads the model from the checkpoint
- Reads `patch_size` and `class_mapping` from saved hyperparameters
- Selects and normalizes the image bands as during training
- Handles `reduce_zero_label` correction if it was used during training
- Outputs a GeoTIFF with the same CRS and extent as the input

With `blend=True`, overlapping patch predictions (class probabilities for segmentation) are averaged with weights that decrease towards the patch edges, instead of only keeping the center of each patch. `RegressionTask.predict_on_tif_file()` works the same way and writes float32 values.

> **Note**: For custom models or more control, use `predict_on_tif_generic()` from `eotorch.inference.inference` which accepts any prediction function.

### 6. Preprocessing with `eotorch.processing`

EOTorch provides utilities for common preprocessing tasks:

#### Normalization

**Z-score normalization** (recommended for foundation models like Prithvi):
```python
from eotorch.processing import zscore_normalize

# Computes mean/std across all images and applies: (x - mean) / std
stats = zscore_normalize(
    input_files="path/to/images",      # Directory, file, or list of files
    out_dir="path/to/normalized",
    save_stats=True,                   # Saves mean/std to JSON for inference
    sample_size=1.0,                   # Use < 1.0 to sample for speed on large datasets
)
# Output files have "_zscore" suffix
```

**Percentile normalization** (scales to 0-1 range):
```python
from eotorch.processing import normalize

# Clips to percentile bounds and scales to [0, 1]
normalize(
    img_path="image.tif",
    limits=(1, 99),            # Percentile cutoffs (or dict of per-band limits)
    out_path="image_norm.tif",
    out_res=10.0,              # Optional: resample to target resolution
)
```

#### Patch Generation

Split an image and its labels into patch files once, e.g. to train on them repeatedly or to share a fixed training set. The labels may have a different extent, resolution or CRS than the image, and are reprojected onto the image grid:

```python
from eotorch.processing import generate_patches_from_files

generate_patches_from_files(
    img_path="image.tif",
    label_path="labels.tif",      # e.g. rasterized vector labels, see below
    out_dir="patches",            # creates patches/train and patches/val
    patch_size=256,
    val_fraction=0.2,             # random blocks of the image held out for validation
    n_random_offsets=2,           # extra, randomly shifted training patches per block
    empty_label_threshold=0.99,   # skip patches with (almost) no labels
)
```

See [Pre-patched Data](#pre-patched-data) for training on the patches.

#### Rasterization

Convert vector annotations (Shapefile, GeoJSON) to raster labels:

**From Shapefiles** (one file per class):
```python
from eotorch.processing.rasterize import FileSource

# Create rasterizer with class files aligned to a reference image
source = FileSource(
    classes=["water", "forest", "urban"],  # Looks for water.shp, forest.shp, etc.
    root_dir="path/to/shapefiles",
    img_path="reference_image.tif"         # Matches CRS, resolution, and extent
)

# Generate raster labels (class 1=water, 2=forest, 3=urban)
source.rasterize_polygons(
    out_path="labels.tif",
    exclude_nodata_bounds=True,    # Crop to data extent
    class_order=[3, 2, 1],         # Optional: control overlap priority
)
```

You can also use `FileSource` without a reference image by providing the raster metadata directly:
```python
from rasterio.crs import CRS
from affine import Affine

source = FileSource(
    classes=["water", "forest", "urban"],
    root_dir="path/to/shapefiles",
    transform=Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 6000000.0),  # 10m resolution
    shape=(1024, 1024),                                            # Height x Width
    crs=CRS.from_epsg(32632)                                       # UTM zone 32N
)
```

**From GeoDataFrame** (single file with class column):
```python
from eotorch.processing.rasterize import DataframeSource
import geopandas as gpd

gdf = gpd.read_file("annotations.geojson")

source = DataframeSource(
    dataframe=gdf,
    class_column="class_id",       # Column containing integer class labels
    img_path="reference_image.tif"
)

source.rasterize_polygons(out_path="labels.tif")
```

---

## Key Differences from TorchGeo

EOTorch builds on TorchGeo but differs in several important ways:

| Aspect | TorchGeo | EOTorch |
|--------|----------|---------|
| **Dataset Creation** | Subclass `RasterDataset` | Use `get_segmentation_dataset()` factory |
| **DataModule Init** | Pass dataset *class* type | Pass dataset *instances* directly |
| **Visualization** | Manual setup | Built-in `plot_samples()`, `preview_labels()` |
| **Splitting** | Basic `random_bbox_assignment` | File-wise and AOI-based splits |
| **Class Labels** | Manual handling | `reduce_zero_label` auto-shifts labels |
| **Cache Control** | Fixed 128 files | Configurable `cache_size` per dataset |

### Dataset Instance vs Class Pattern

**TorchGeo's approach** (pass class type to DataModule):
```python
# TorchGeo requires passing the class, not an instance
datamodule = SomeGeoDataModule(
    dataset_class=MyRasterDataset,  # Class type
    batch_size=8,
    **dataset_kwargs  # Args passed to class constructor
)
```

**EOTorch's approach** (pass instance after inspection):
```python
# EOTorch lets you create and inspect first
ds = get_segmentation_dataset(...)
ds.plot_samples(n=3)  # Verify everything looks correct

train_ds, val_ds = splits.file_wise_split(ds, [0.8, 0.2])

# Then pass instances to DataModule
module = RasterDataModule(
    train_dataset=train_ds,  # Instance
    val_dataset=val_ds,      # Instance
)
```

This pattern allows you to:
- Visualize and verify your dataset before training
- Apply splits that maintain the same dataset configuration
- Easily experiment with different split strategies

---

## API Reference

### Data Module: `eotorch.data`

| Function/Class | Description |
|----------------|-------------|
| `get_segmentation_dataset()` | Create a dataset from image/label directories |
| `get_regression_dataset()` | Create a dataset with continuous labels |
| `RasterDataModule` | Lightning DataModule for segmentation and regression |
| `SegmentationTask` | Semantic segmentation task |
| `RegressionTask` | Pixelwise regression task |
| `splits.file_wise_split()` | Split dataset by files (random or specified) |
| `splits.aoi_split()` | Split dataset by geographic regions |

### Inference: `eotorch.inference`

| Function | Description |
|----------|-------------|
| `predict_on_tif_generic()` | Run patched inference on large GeoTIFFs |

### Processing: `eotorch.processing`

| Function | Description |
|----------|-------------|
| `normalize.band_statistics()` | Per-band mean/std for on-the-fly normalization |
| `generate_patches_from_files()` | Split an image and labels into patch files |
| `zscore_normalize()` | Z-score normalization with stats export |
| `normalize()` | Percentile-based normalization |

### Visualization: `eotorch.plot`

| Function | Description |
|----------|-------------|
| `plot_class_raster()` | Plot classification results with legend |
| `plot_samples()` | Plot random samples from a dataset |

---

## Integration with TerraTorch / Prithvi-EO-2.0

EOTorch integrates seamlessly with [TerraTorch](https://github.com/IBM/terratorch) for fine-tuning foundation models like Prithvi-EO-2.0:

```python
from terratorch.datasets import HLSBands
from terratorch.tasks import SemanticSegmentationTask
from lightning.pytorch import Trainer

# Define bands (must match your data)
BANDS = [HLSBands.RED, HLSBands.GREEN, HLSBands.BLUE]

# Create the model task
model = SemanticSegmentationTask(
    model_args=dict(
        backbone="prithvi_eo_v2_300",
        backbone_pretrained=True,
        backbone_bands=BANDS,
        backbone_num_frames=1,
        decoder="UperNetDecoder",
        decoder_channels=256,
        num_classes=3,
        necks=[
            {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
            {"name": "ReshapeTokensToImage"},
        ],
    ),
    loss="focal",
    lr=1e-4,
    optimizer="AdamW",
    ignore_index=-1,
    freeze_backbone=True,
    model_factory="EncoderDecoderFactory",
)

# Train with EOTorch DataModule
trainer = Trainer(max_epochs=50, accelerator="auto")
trainer.fit(model, datamodule=module)
```

### Inference with TerraTorch Models

After training, use `predict_on_tif_generic()` from EOTorch to run inference on large rasters. Since TerraTorch's `SemanticSegmentationTask` has a different interface than EOTorch's, you need to provide a custom prediction function:

```python
import torch
from terratorch.tasks import SemanticSegmentationTask
from eotorch.inference.inference import predict_on_tif_generic

# Load the trained checkpoint
checkpoint_path = "results/Prithvi/baseline/checkpoints/best-checkpoint.ckpt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SemanticSegmentationTask.load_from_checkpoint(checkpoint_path, map_location=device)
model.to(device)
model.eval()

# Define a prediction function compatible with predict_on_tif_generic
def predict_func(batch):
    """Prediction function for TerraTorch models."""
    with torch.no_grad():
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch, dtype=torch.float32)
        batch = batch.to(device)
        
        # TerraTorch models return a ModelOutput object
        logits = model.model(batch)
        logits = logits.output  # Extract the actual tensor
        
        predictions = torch.argmax(logits, dim=1)
        
        # If reduce_zero_label was used during training, shift predictions back
        predictions += 1  # Adjust if your labels were shifted
        
    return predictions

# Run inference on a large raster
result = predict_on_tif_generic(
    tif_file_path="path/to/input_image.tif",
    prediction_func=predict_func,
    patch_size=224,              # Must match training patch size
    batch_size=8,
    out_file_path="predictions/output.tif",
    show_results=True,           # Display preview in notebook
    class_mapping=class_mapping, # For visualization
)
```

#### Key Differences from EOTorch Models

| Aspect | EOTorch `SegmentationTask` | TerraTorch `SemanticSegmentationTask` |
|--------|----------------------------|--------------------------------------|
| **Inference method** | Built-in `predict_on_tif_file()` | Use `predict_on_tif_generic()` with custom function |
| **Model output** | Direct tensor | `ModelOutput` object (access via `.output`) |
| **Checkpoint storage** | Stores `datamodule_hyper_parameters` | Does not auto-store EOTorch params |


---

## Examples

You can find usage examples in the [notebooks](notebooks/) directory:

| Notebook | Description |
|----------|-------------|
| [segmentation.ipynb](notebooks/segmentation.ipynb) | End-to-end segmentation workflow with EOTorch |
| [prithvi_eo_2.ipynb](notebooks/prithvi_eo_2.ipynb) | Fine-tuning Prithvi-EO-2.0 with TerraTorch |

### Minimal Example (EOTorch only)

```python
from pathlib import Path
from lightning.pytorch import Trainer
from eotorch.data import RasterDataModule, SegmentationTask, get_segmentation_dataset, splits

# 1. Create dataset
ds = get_segmentation_dataset(
    images_dir=Path("data/images"),
    labels_dir=Path("data/labels"),
    all_image_bands=("R", "G", "B"),
    rgb_bands=("R", "G", "B"),
    class_mapping={1: "Class A", 2: "Class B"},
    reduce_zero_label=True,
)

# 2. Verify with visualization
ds.plot_samples(n=2, patch_size=256)

# 3. Split data
train_ds, val_ds = splits.file_wise_split(ds, ratios_or_counts=[0.8, 0.2])

# 4. Create data module
module = RasterDataModule(
    train_dataset=train_ds,
    val_dataset=val_ds,
    batch_size=16,
    patch_size=224,
    num_workers=4,
)

# 5. Train
task = SegmentationTask(num_classes=2, in_channels=3, model="unet", ignore_index=-1)
trainer = Trainer(max_epochs=10)
trainer.fit(task, datamodule=module)
```

---

## Troubleshooting

### Memory Issues

| Problem | Solution |
|---------|----------|
| Out of RAM | Reduce `num_workers` and/or `cache_size` in dataset |
| Out of GPU memory | Reduce `batch_size` |
| Low GPU utilization | Increase `num_workers` (while monitoring RAM) |

### Common Pitfalls

**Label indexing**: Remember that `reduce_zero_label=True` shifts labels down by 1. If your labels are `{0, 1, 2}`, they become `{-1, 0, 1}` where -1 is typically `ignore_index`.

**Band ordering**: `all_image_bands` defines the order bands are stored in files; `rgb_bands` defines which bands to use for visualization.

**Windows users**: Multiple data loading workers (`num_workers > 0`) are supported. In scripts, put the training code under `if __name__ == "__main__":`, as worker processes re-import the script on Windows.

---