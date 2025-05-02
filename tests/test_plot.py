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
