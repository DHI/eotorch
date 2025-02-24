## Example of how to use SegmentationRasterDataset:

```python
from eotorch.datasets.geo import SegmentationRasterDataset
from eotorch.plot.plot import plot_samples

class_mapping = {
    1: "Baresoil",
    2: "Buildings",
    3: "Coniferous Trees",
    4: "Deciduous Trees",
    5: "Grass",
    6: "Impervious",
    7: "Water",
}

bla = SegmentationRasterDataset.create(
    images_dir="dev_data/sr_data",
    labels_dir="dev_data/labels",
    image_kwargs=dict(
        all_bands=("B02", "B03", "B04", "B08", "B11", "B12"),
        rgb_bands=("B04", "B03", "B02"),
    ),
    label_kwargs=dict(
        class_mapping=class_mapping,
    ),
)
plot_samples(bla, n=2, patch_size=256)
```

![alt text](media/sample_1.png)
![alt text](media/sample_2.png)