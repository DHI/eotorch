import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from eotorch.plot import plot_numpy_array

# Use non-interactive backend for testing
matplotlib.use("Agg")


@pytest.fixture
def class_mapping():
    return {
        0: "Background",
        1: "Baresoil",
        2: "Buildings",
        3: "Coniferous Trees",
        4: "Deciduous Trees",
        5: "Grass",
        6: "Impervious",
        7: "Water",
    }


def test_plot_numpy_array_with_class_mapping(class_mapping):
    """Test that the function correctly uses class mapping for labels."""
    array = np.array([[1, 2, 3], [0, 4, 5]])

    fig, ax = plt.subplots()
    plot_numpy_array(array, ax=ax, class_mapping=class_mapping)

    # Check legend labels match the class mapping
    legend = ax.get_legend()
    legend_texts = [text.get_text() for text in legend.get_texts()]

    # Legend should contain labels for classes in the array
    expected_labels = [class_mapping[i] for i in sorted(np.unique(array))]
    assert set(legend_texts) == set(expected_labels)

    plt.close(fig)


def test_plot_numpy_array_with_nodata():
    """Test that nodata values are correctly handled."""
    array = np.array([[1, 2, 3], [0, -1, 5]])
    nodata_value = -1

    fig, ax = plt.subplots()
    plot_numpy_array(array, ax=ax, nodata_value=nodata_value)

    # Check that the legend includes "No Data" label
    legend = ax.get_legend()
    legend_texts = [text.get_text() for text in legend.get_texts()]
    assert "No Data" in legend_texts

    plt.close(fig)


def test_plot_numpy_array_data_window_only():
    """Test that data_window_only parameter works correctly."""
    # Create array with data surrounded by nodata
    array = np.zeros((5, 5), dtype=np.int32)
    array[1:4, 1:4] = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    fig, ax = plt.subplots()
    plot_numpy_array(array, ax=ax, nodata_value=0, data_window_only=True)

    # The image shown should be the 3x3 center of the array
    img = ax.get_images()[0]
    assert img.get_array().shape == (3, 3), "Data window should be 3x3"

    plt.close(fig)


def test_plot_numpy_array_color_consistency_with_class_mapping(class_mapping):
    """Test that colors are consistent when class_mapping is provided."""
    # Create two arrays with different classes
    array1 = np.array([[1, 2, 3], [0, 4, 5]])  # Contains some classes
    array2 = np.array([[6, 7, 0]])  # Contains different classes

    # Plot both arrays with the same class mapping
    fig1, ax1 = plt.subplots()
    plot_numpy_array(array1, ax=ax1, class_mapping=class_mapping)
    img1 = ax1.get_images()[0]

    fig2, ax2 = plt.subplots()
    plot_numpy_array(array2, ax=ax2, class_mapping=class_mapping)
    img2 = ax2.get_images()[0]

    # Extract colors for class 0 (present in both arrays)
    color_class0_img1 = img1.cmap(img1.norm(0))
    color_class0_img2 = img2.cmap(img2.norm(0))

    # Colors should be the same for the same class
    assert np.allclose(color_class0_img1, color_class0_img2), (
        "Color for class 0 should be consistent"
    )

    plt.close(fig1)
    plt.close(fig2)


def test_plot_numpy_array_empty_array():
    """Test that the function handles empty arrays properly."""
    array = np.array([[0, 0], [0, 0]])

    fig, ax = plt.subplots()
    plot_numpy_array(array, ax=ax, nodata_value=0)

    # There should still be a legend with at least the nodata class
    legend = ax.get_legend()
    assert legend is not None, "Legend should be present even with empty data"

    plt.close(fig)


@pytest.mark.parametrize("labels", ["label", "target"])
def test_plot_samples_with_patches_as_large_as_the_files(monkeypatch, labels):
    from pathlib import Path

    from eotorch.data import get_regression_dataset, get_segmentation_dataset

    # Segmentation labels (label_*.tif) or regression labels with nodata (target_*.tif)
    get_dataset = get_segmentation_dataset
    kwargs = {}
    if labels == "target":
        get_dataset = get_regression_dataset
        kwargs = {"nodata_value": -9999.0}
    data = Path(__file__).parent / "data"
    ds = get_dataset(
        images_dir=data,
        labels_dir=data,
        image_glob="image_*.tif",
        label_glob=f"{labels}_*.tif",
        all_image_bands=("red", "green", "blue"),
        **kwargs,
    )
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gcf()))

    # The test files are 100 x 100 pixels
    ds.plot_samples(n=2, patch_size=100, show_filepaths=True)

    assert len(shown) == 2
    plt.close("all")


@pytest.mark.parametrize("labels", ["label", "target"])
def test_plotting_does_not_change_the_sample(labels):
    from pathlib import Path

    from eotorch.data import get_regression_dataset, get_segmentation_dataset

    get_dataset = get_segmentation_dataset
    kwargs = {"reduce_zero_label": True}
    if labels == "target":
        get_dataset = get_regression_dataset
        kwargs = {"nodata_value": -9999.0}
    data = Path(__file__).parent / "data"
    ds = get_dataset(
        images_dir=data,
        labels_dir=data,
        image_glob="image_0.tif",
        label_glob=f"{labels}_0.tif",
        all_image_bands=("red", "green", "blue"),
        **kwargs,
    )
    sample = ds[ds.bounds]
    sample["prediction"] = sample["mask"].clone()
    original = {k: v.clone() for k, v in sample.items()}

    for _ in range(2):
        ds.plot(sample)
        plt.close("all")

    assert sample.keys() == original.keys()
    for key, value in original.items():
        assert (sample[key] == value).all(), key


def test_plot_numpy_array_with_more_classes_than_colors():
    array = np.arange(30).reshape(5, 6)

    fig, ax = plt.subplots()
    plot_numpy_array(array, ax=ax, nodata_value=0)

    assert len(ax.get_legend().get_texts()) >= 29
    plt.close(fig)


def _segmentation_dataset(**kwargs):
    from pathlib import Path

    from eotorch.data import get_segmentation_dataset

    data = Path(__file__).parent / "data"
    return get_segmentation_dataset(
        images_dir=data / "image_0.tif",
        labels_dir=data / "label_0.tif",
        all_image_bands=("red", "green", "blue"),
        **kwargs,
    )


def _plotted_image(image_dataset) -> np.ndarray:
    sample = image_dataset[image_dataset.bounds]
    fig, ax = plt.subplots()
    image_dataset.plot(sample, ax=ax)
    plotted = ax.get_images()[0].get_array()
    plt.close(fig)
    return plotted


def test_image_plot_finds_rgb_bands_among_the_selected_bands():
    images = _segmentation_dataset().datasets[0]
    reordered = _segmentation_dataset(bands_to_return=("blue", "green", "red"))
    np.testing.assert_array_equal(
        _plotted_image(reordered.datasets[0]), _plotted_image(images)
    )


def test_image_plot_without_rgb_bands_raises():
    from torchgeo.datasets import RGBBandsMissingError

    images = _segmentation_dataset(bands_to_return=("blue", "green")).datasets[0]
    with pytest.raises(RGBBandsMissingError):
        images.plot(images[images.bounds])
    plt.close("all")


@pytest.mark.parametrize("labels", ["label", "target"])
def test_plots_show_filepaths_without_given_axes(labels):
    from pathlib import Path

    from eotorch.data import get_regression_dataset

    if labels == "label":
        ds = _segmentation_dataset()
    else:
        data = Path(__file__).parent / "data"
        ds = get_regression_dataset(
            images_dir=data / "image_0.tif",
            labels_dir=data / "target_0.tif",
            all_image_bands=("red", "green", "blue"),
            nodata_value=-9999.0,
        )
    sample = ds[ds.bounds]
    image_ds, label_ds = ds.datasets

    label_ds.plot(sample, show_filepaths=True)
    assert f"{labels}_0.tif" in plt.gcf().axes[0].get_title()

    image_ds.plot({**sample, "prediction": sample["mask"]}, show_filepaths=True)
    assert "image_0.tif" in plt.gcf().axes[0].get_title()
    plt.close("all")
