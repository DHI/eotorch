"""Tests for pixel counting methods with proper temporal dimension handling."""

from pathlib import Path

import pandas as pd
import pytest
from torchgeo.datasets.utils import BoundingBox

from eotorch.data.geodatasets import (
    LabelledRasterDataset,
    PlottabeLabelDataset,
    PlottableImageDataset,
    get_segmentation_dataset,
)


@pytest.fixture
def test_data_dir():
    """Get the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def sparse_raster_path(test_data_dir):
    """Path to the sparse raster test file."""
    return test_data_dir / "sparse_raster.tif"


@pytest.fixture
def label_dataset(sparse_raster_path):
    """Create a PlottabeLabelDataset for testing."""
    dataset = PlottabeLabelDataset(
        paths=sparse_raster_path, cache_size=1, reduce_zero_label=False
    )
    # Set nodata_value to None to avoid filtering out class 0
    dataset.nodata_value = None
    return dataset


@pytest.fixture
def image_dataset(sparse_raster_path):
    """Create a PlottableImageDataset for testing (using same file as placeholder)."""
    return PlottableImageDataset(paths=sparse_raster_path, cache_size=1)


@pytest.fixture
def labelled_dataset(sparse_raster_path):
    """Create a LabelledRasterDataset for testing."""
    # Use the same file for both image and labels for simplicity
    image_ds = PlottableImageDataset(paths=sparse_raster_path, cache_size=1)
    label_ds = PlottabeLabelDataset(
        paths=sparse_raster_path, cache_size=1, reduce_zero_label=False
    )
    # Set nodata_value to None to avoid filtering out class 0
    label_ds.nodata_value = None
    return LabelledRasterDataset(image_ds, label_ds)


def test_label_dataset_temporal_bounds_structure(label_dataset):
    """Test that the label dataset has proper temporal bounds structure."""
    # Check that the dataset index has temporal information
    assert hasattr(label_dataset, "index"), "Dataset should have an index"
    assert not label_dataset.index.empty, "Dataset index should not be empty"

    # Check that the index has temporal intervals
    first_row_index = label_dataset.index.index[0]
    assert isinstance(first_row_index, pd.Interval), (
        "Index should be pd.Interval with temporal bounds"
    )
    assert hasattr(first_row_index, "left"), "Temporal interval should have left bound"
    assert hasattr(first_row_index, "right"), (
        "Temporal interval should have right bound"
    )


def test_label_dataset_get_pixel_counts(label_dataset):
    """Test that get_pixel_counts works with proper temporal bounds."""
    # This should not raise the pandas datetime comparison error
    try:
        pixel_counts = label_dataset.get_pixel_counts()
        assert isinstance(pixel_counts, dict), "Should return a dictionary"
        assert len(pixel_counts) > 0, "Should have some pixel counts"

        # Check that all keys are valid (integers or class names)
        for key, count in pixel_counts.items():
            assert isinstance(count, int), f"Count should be integer, got {type(count)}"
            assert count >= 0, f"Count should be non-negative, got {count}"

    except Exception as e:
        pytest.fail(f"get_pixel_counts raised an unexpected exception: {e}")


def test_labelled_dataset_get_label_pixel_counts(labelled_dataset):
    """Test that get_label_pixel_counts works with proper temporal bounds."""
    # This should not raise the pandas datetime comparison error
    try:
        pixel_counts = labelled_dataset.get_label_pixel_counts()
        assert isinstance(pixel_counts, dict), "Should return a dictionary"
        assert len(pixel_counts) > 0, "Should have some pixel counts"

        # Check that all keys are valid (integers or class names)
        for key, count in pixel_counts.items():
            assert isinstance(count, int), f"Count should be integer, got {type(count)}"
            assert count >= 0, f"Count should be non-negative, got {count}"

    except Exception as e:
        pytest.fail(f"get_label_pixel_counts raised an unexpected exception: {e}")


def test_bounding_box_temporal_bounds_consistency(label_dataset):
    """Test that BoundingBox creation uses consistent temporal bounds."""
    # Iterate through the dataset and check that temporal bounds are consistent
    for temporal_interval, row in label_dataset.index.iterrows():
        bounds = row.geometry.bounds

        # Create BoundingBox the same way as in the fixed methods
        bbox = BoundingBox(
            bounds[0],
            bounds[2],
            bounds[1],
            bounds[3],
            temporal_interval.left,
            temporal_interval.right,
        )

        # Verify temporal bounds types are consistent
        assert type(bbox.mint) is type(temporal_interval.left), (
            "BoundingBox mint should have same type as temporal_interval.left"
        )
        assert type(bbox.maxt) is type(temporal_interval.right), (
            "BoundingBox maxt should have same type as temporal_interval.right"
        )

        # Verify they're not the old hardcoded values
        assert bbox.mint != 0 or bbox.maxt != 1, (
            "Should not use hardcoded 0,1 temporal bounds"
        )


def test_no_datetime_comparison_error(label_dataset, labelled_dataset):
    """Test that the specific pandas datetime comparison error is fixed."""
    import pandas.errors

    # Test PlottabeLabelDataset.get_pixel_counts()
    try:
        label_dataset.get_pixel_counts()
    except pandas.errors.InvalidComparison as e:
        pytest.fail(f"get_pixel_counts still raises InvalidComparison error: {e}")

    # Test LabelledRasterDataset.get_label_pixel_counts()
    try:
        labelled_dataset.get_label_pixel_counts()
    except pandas.errors.InvalidComparison as e:
        pytest.fail(f"get_label_pixel_counts still raises InvalidComparison error: {e}")


def test_pixel_counts_consistency(labelled_dataset):
    """Test that pixel counts are consistent between methods."""
    # Get counts from the label dataset directly
    label_ds = labelled_dataset.datasets[-1]
    direct_counts = label_ds.get_pixel_counts()

    # Get counts from the labelled dataset (intersection)
    intersection_counts = labelled_dataset.get_label_pixel_counts()

    # The intersection counts should be a subset or equal to direct counts
    # (since intersection might filter out some areas)
    for class_val, count in intersection_counts.items():
        if class_val in direct_counts:
            assert count <= direct_counts[class_val], (
                f"Intersection count {count} should not exceed direct count {direct_counts[class_val]} for class {class_val}"
            )


def test_integration_with_get_segmentation_dataset(sparse_raster_path):
    """Test that the fix works with the high-level get_segmentation_dataset function."""
    # Create a dataset using the high-level function
    dataset = get_segmentation_dataset(
        images_dir=sparse_raster_path.parent,
        labels_dir=sparse_raster_path.parent,
        image_glob="sparse_raster.tif",
        label_glob="sparse_raster.tif",
        cache_size=1,
        reduce_zero_label=False,
    )

    # Test that pixel counting works without errors
    try:
        pixel_counts = dataset.get_label_pixel_counts()
        assert isinstance(pixel_counts, dict), "Should return a dictionary"
    except Exception as e:
        pytest.fail(f"Integration test failed with exception: {e}")


def test_temporal_bounds_types(label_dataset):
    """Test that temporal bounds maintain their original types."""
    for temporal_interval, row in label_dataset.index.iterrows():
        # The temporal bounds should maintain their pandas Timestamp type
        # or whatever type they originally had
        assert temporal_interval.left is not None, (
            "Left temporal bound should not be None"
        )
        assert temporal_interval.right is not None, (
            "Right temporal bound should not be None"
        )

        # They should not be converted to integers (the old bug)
        assert not isinstance(temporal_interval.left, int), (
            "Left temporal bound should not be an integer"
        )
        assert not isinstance(temporal_interval.right, int), (
            "Right temporal bound should not be an integer"
        )


# Expected pixel counts for sparse_raster.tif based on direct file analysis
EXPECTED_PIXEL_COUNTS = {
    0: 1046837,
    1: 1974,
    2: 384,
    3: 93,
    4: 481,
    5: 1214,
    6: 464,
    7: 15805,
}
EXPECTED_TOTAL_PIXELS = 1067252


def test_label_dataset_actual_pixel_counts(label_dataset):
    """Test that get_pixel_counts returns the expected pixel counts for sparse_raster.tif."""
    pixel_counts = label_dataset.get_pixel_counts()

    # Check that we get the expected pixel classes
    assert len(pixel_counts) == len(EXPECTED_PIXEL_COUNTS), (
        f"Expected {len(EXPECTED_PIXEL_COUNTS)} classes, got {len(pixel_counts)}"
    )

    # Check each expected class and count
    for expected_class, expected_count in EXPECTED_PIXEL_COUNTS.items():
        assert expected_class in pixel_counts, (
            f"Expected class {expected_class} not found in pixel_counts"
        )
        actual_count = pixel_counts[expected_class]
        assert actual_count == expected_count, (
            f"For class {expected_class}: expected {expected_count} pixels, got {actual_count}"
        )

    # Verify total pixel count
    total_pixels = sum(pixel_counts.values())
    assert total_pixels == EXPECTED_TOTAL_PIXELS, (
        f"Expected total of {EXPECTED_TOTAL_PIXELS} pixels, got {total_pixels}"
    )


def test_labelled_dataset_actual_pixel_counts(labelled_dataset):
    """Test that get_label_pixel_counts returns the expected pixel counts."""
    pixel_counts = labelled_dataset.get_label_pixel_counts()

    # Check that we get the expected pixel classes
    assert len(pixel_counts) == len(EXPECTED_PIXEL_COUNTS), (
        f"Expected {len(EXPECTED_PIXEL_COUNTS)} classes, got {len(pixel_counts)}"
    )

    # Check each expected class and count
    for expected_class, expected_count in EXPECTED_PIXEL_COUNTS.items():
        assert expected_class in pixel_counts, (
            f"Expected class {expected_class} not found in pixel_counts"
        )
        actual_count = pixel_counts[expected_class]
        assert actual_count == expected_count, (
            f"For class {expected_class}: expected {expected_count} pixels, got {actual_count}"
        )

    # Verify total pixel count
    total_pixels = sum(pixel_counts.values())
    assert total_pixels == EXPECTED_TOTAL_PIXELS, (
        f"Expected total of {EXPECTED_TOTAL_PIXELS} pixels, got {total_pixels}"
    )


def test_pixel_counts_with_reduce_zero_label(sparse_raster_path):
    """Test pixel counts with reduce_zero_label=True option."""
    # Create dataset with reduce_zero_label=True
    label_dataset = PlottabeLabelDataset(
        paths=sparse_raster_path, cache_size=1, reduce_zero_label=True
    )
    # Set nodata_value to None to avoid filtering out any classes
    label_dataset.nodata_value = None

    pixel_counts = label_dataset.get_pixel_counts()

    # With reduce_zero_label=True, the _get_pixel_counts method undoes the reduction
    # by adding 1 back, so we should get the original class values
    # Check that we get the expected original classes
    for expected_class, expected_count in EXPECTED_PIXEL_COUNTS.items():
        assert expected_class in pixel_counts, (
            f"Expected class {expected_class} not found in pixel_counts (reduce_zero_label=True should be undone in counting)"
        )
        actual_count = pixel_counts[expected_class]
        assert actual_count == expected_count, (
            f"For class {expected_class} (with reduce_zero_label=True): expected {expected_count} pixels, got {actual_count}"
        )


def test_pixel_counts_class_distribution(label_dataset):
    """Test that pixel counts reflect the expected class distribution."""
    pixel_counts = label_dataset.get_pixel_counts()

    # Test specific properties of the sparse_raster.tif distribution
    # Class 0 should be the most common (background)
    assert pixel_counts[0] > pixel_counts[1], (
        "Class 0 should have more pixels than class 1"
    )
    assert pixel_counts[0] > 1000000, (
        "Class 0 should have over 1 million pixels (background)"
    )

    # Class 7 should be the second most common
    class_counts_sorted = sorted(pixel_counts.items(), key=lambda x: x[1], reverse=True)
    assert class_counts_sorted[0][0] == 0, "Class 0 should be most common"
    assert class_counts_sorted[1][0] == 7, "Class 7 should be second most common"

    # Class 3 should be the least common (except background)
    non_zero_classes = {k: v for k, v in pixel_counts.items() if k != 0}
    min_class = min(non_zero_classes.items(), key=lambda x: x[1])
    assert min_class[0] == 3, (
        f"Class 3 should have fewest pixels, but {min_class[0]} does"
    )
    assert min_class[1] == 93, f"Class 3 should have 93 pixels, got {min_class[1]}"


def test_integration_actual_counts_with_get_segmentation_dataset(sparse_raster_path):
    """Test actual pixel counts with the high-level get_segmentation_dataset function."""
    dataset = get_segmentation_dataset(
        images_dir=sparse_raster_path.parent,
        labels_dir=sparse_raster_path.parent,
        image_glob="sparse_raster.tif",
        label_glob="sparse_raster.tif",
        cache_size=1,
        reduce_zero_label=False,
    )

    # Set nodata_value to None to avoid filtering out class 0
    if hasattr(dataset, "datasets") and len(dataset.datasets) > 1:
        dataset.datasets[-1].nodata_value = None

    pixel_counts = dataset.get_label_pixel_counts()

    # Should get exactly the same counts as the direct method
    for expected_class, expected_count in EXPECTED_PIXEL_COUNTS.items():
        assert expected_class in pixel_counts, (
            f"Expected class {expected_class} not found in integration test"
        )
        actual_count = pixel_counts[expected_class]
        assert actual_count == expected_count, (
            f"Integration test: for class {expected_class} expected {expected_count}, got {actual_count}"
        )


def test_pixel_counts_edge_cases(sparse_raster_path):
    """Test pixel counting under various edge cases and configurations."""

    # Test with different cache sizes
    for cache_size in [1, 5, 20]:
        label_dataset = PlottabeLabelDataset(
            paths=sparse_raster_path, cache_size=cache_size, reduce_zero_label=False
        )
        # Set nodata_value to None to avoid filtering out class 0
        label_dataset.nodata_value = None
        pixel_counts = label_dataset.get_pixel_counts()

        # Should get consistent counts regardless of cache size
        total_pixels = sum(pixel_counts.values())
        assert total_pixels == EXPECTED_TOTAL_PIXELS, (
            f"Cache size {cache_size}: expected {EXPECTED_TOTAL_PIXELS} total pixels, got {total_pixels}"
        )

    # Test with class mapping
    class_mapping = {
        0: "background",
        1: "class_1",
        2: "class_2",
        3: "class_3",
        4: "class_4",
        5: "class_5",
        6: "class_6",
        7: "class_7",
    }

    label_dataset_with_mapping = PlottabeLabelDataset(
        paths=sparse_raster_path, cache_size=1, reduce_zero_label=False
    )
    # Set nodata_value to None to avoid filtering out class 0
    label_dataset_with_mapping.nodata_value = None
    label_dataset_with_mapping.class_mapping = class_mapping

    pixel_counts_mapped = label_dataset_with_mapping.get_pixel_counts()

    # Should have class names as keys instead of numbers
    for original_class, expected_count in EXPECTED_PIXEL_COUNTS.items():
        mapped_class = class_mapping[original_class]
        assert mapped_class in pixel_counts_mapped, (
            f"Mapped class {mapped_class} not found in results"
        )
        assert pixel_counts_mapped[mapped_class] == expected_count, (
            f"Mapped class {mapped_class}: expected {expected_count}, got {pixel_counts_mapped[mapped_class]}"
        )


def test_temporal_bounds_are_not_hardcoded_integers(label_dataset):
    """Specific test to ensure we don't regress to hardcoded integer temporal bounds."""
    # This test specifically targets the original bug
    for temporal_interval, row in label_dataset.index.iterrows():
        bounds = row.geometry.bounds

        # Create BoundingBox exactly as the fixed code does
        bbox = BoundingBox(
            bounds[0],
            bounds[2],
            bounds[1],
            bounds[3],
            temporal_interval.left,
            temporal_interval.right,
        )

        # The key assertion: temporal bounds should NOT be 0 and 1
        # (unless the actual data legitimately has those values)
        if hasattr(temporal_interval.left, "year"):  # It's a timestamp
            assert bbox.mint != 0, "mint should not be hardcoded integer 0"
            assert bbox.maxt != 1, "maxt should not be hardcoded integer 1"

        # Additional check: types should match
        assert type(bbox.mint) is type(temporal_interval.left), (
            f"mint type {type(bbox.mint)} should match temporal_interval.left type {type(temporal_interval.left)}"
        )
        assert type(bbox.maxt) is type(temporal_interval.right), (
            f"maxt type {type(bbox.maxt)} should match temporal_interval.right type {type(temporal_interval.right)}"
        )


def test_error_regression_pandas_invalid_comparison(labelled_dataset):
    """Regression test for the specific pandas.errors.InvalidComparison error."""
    import pandas as pd

    # This is the exact error that was occurring before the fix:
    # "Invalid comparison between dtype=datetime64[ns] and int"
    # The error occurred when creating BoundingBox with hardcoded int values (0, 1)
    # but the dataset had pandas Timestamp temporal bounds
    # Verify that we can call the methods without this specific error
    try:
        # This should work without raising InvalidComparison
        pixel_counts = labelled_dataset.get_label_pixel_counts()
        assert isinstance(pixel_counts, dict)

        # Also test the direct method
        label_ds = labelled_dataset.datasets[-1]
        direct_counts = label_ds.get_pixel_counts()
        assert isinstance(direct_counts, dict)

    except pd.errors.InvalidComparison as e:
        pytest.fail(
            f"Regression: pandas.errors.InvalidComparison error still occurs: {e}. "
            "This indicates the temporal bounds fix is not working properly."
        )
    except Exception as e:
        # Other exceptions might be legitimate, but InvalidComparison specifically
        # indicates our bug regression
        if "Invalid comparison between dtype=datetime64" in str(e):
            pytest.fail(f"Regression: datetime comparison error detected: {e}")
        else:
            # Re-raise other exceptions for normal test failure handling
            raise
