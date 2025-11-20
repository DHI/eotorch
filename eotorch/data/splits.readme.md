# Dataset Splitting Strategies

The `eotorch.data.splits` module provides flexible strategies for splitting geospatial datasets into training, validation, and test sets. These strategies are designed to handle both simple `RasterDataset` and complex `IntersectionDataset` (e.g., paired images and labels) while respecting file boundaries and spatial constraints.

## 1. File-wise Split (`file_wise_split`)

This function splits a dataset based on image files, ensuring that all patches/samples from a single image file end up in the same split. This prevents data leakage where spatially adjacent patches from the same image might otherwise end up in different splits.

### Option A: Ratio-based Splitting

You can split the dataset randomly based on ratios or counts.

```python
from eotorch.data import splits

# Split into 80% train, 20% validation
train_ds, val_ds = splits.file_wise_split(
    dataset=ds, 
    ratios_or_counts=[0.8, 0.2]
)

# Split into 70% train, 15% validation, 15% test
train_ds, val_ds, test_ds = splits.file_wise_split(
    dataset=ds, 
    ratios_or_counts=[0.7, 0.15, 0.15]
)
```

### Option B: Explicit File Lists

You can explicitly specify which files belong to the validation and test sets. All remaining files will be assigned to the training set.

**Using file paths:**
```python
val_files = ["/path/to/image_1.tif", "/path/to/image_2.tif"]
test_files = ["/path/to/image_3.tif"]

train_ds, val_ds, test_ds = splits.file_wise_split(
    dataset=ds,
    val_img_files=val_files,
    test_img_files=test_files
)
```

**Using glob patterns:**
```python
# Use all images matching "2023*" for validation
train_ds, val_ds = splits.file_wise_split(
    dataset=ds,
    val_img_glob="**/2023*.tif"
)
```

## 2. AOI-based Split (`aoi_split`)

This function splits a dataset based on geographic Areas of Interest (AOIs) provided as vector files (e.g., GeoJSON, Shapefile). This is useful for creating spatially disjoint splits, such as "hold-out regions" for testing.

The function returns a list of datasets: `[remainder_dataset, aoi_dataset_1, aoi_dataset_2, ...]`.
- The first dataset contains all data *outside* the provided AOIs (typically used for training).
- Subsequent datasets contain data *inside* each provided AOI file.

### Usage

```python
from eotorch.data import splits

# Split into training (outside AOI) and validation (inside AOI)
train_ds, val_ds = splits.aoi_split(
    dataset=ds,
    aoi_files="validation_zones.geojson",
    buffer_size_metres=100  # Optional buffer
)
```

### Buffer Zone
The `buffer_size_metres` parameter allows you to define a buffer zone around your AOIs. Data within this buffer zone is excluded from the training set (remainder dataset). This is crucial for preventing spatial autocorrelation leakage when your training samples are close to your validation region.

### Splitting Logic

Instead of using a simple geometric difference (which can create complex polygons with holes that are hard to index), the function splits the original dataset geometry into up to 4 rectangular boxes around the subtract geometry (AOI):

```text
┌─────────────────────────────────┐
│           TOP BOX               │
├─────┬───────────────────┬───────┤
│LEFT │   SUBTRACT_GEOM   │ RIGHT │
│ BOX │       (AOI)       │  BOX  │
├─────┴───────────────────┴───────┤
│          BOTTOM BOX             │
└─────────────────────────────────┐
```

This ensures that the resulting training dataset consists of valid rectangular bounds suitable for R-tree indexing.
