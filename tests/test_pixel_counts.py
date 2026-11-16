"""Tests for pixel counting methods with proper temporal dimension handling."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio

from eotorch.data.geodatasets import (
    ClassificationRasterDataset,
    PlottableClassificationDataset,
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
    """Create a PlottableClassificationDataset for testing."""
    return PlottableClassificationDataset(
        paths=sparse_raster_path, cache_size=1, nodata_value=None
    )


@pytest.fixture
def image_dataset(sparse_raster_path):
    """Create a PlottableImageDataset for testing (using same file as placeholder)."""
    return PlottableImageDataset(paths=sparse_raster_path, cache_size=1)


@pytest.fixture
def labelled_dataset(sparse_raster_path):
    """Create a ClassificationRasterDataset for testing."""
    # Use the same file for both image and labels for simplicity
    image_ds = PlottableImageDataset(paths=sparse_raster_path, cache_size=1)
    label_ds = PlottableClassificationDataset(
        paths=sparse_raster_path, cache_size=1, nodata_value=None
    )
    return ClassificationRasterDataset(image_ds, label_ds)


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
    pixel_counts = label_dataset.get_pixel_counts()
    assert isinstance(pixel_counts, dict), "Should return a dictionary"
    assert len(pixel_counts) > 0, "Should have some pixel counts"

    for count in pixel_counts.values():
        assert isinstance(count, int), f"Count should be integer, got {type(count)}"
        assert count >= 0, f"Count should be non-negative, got {count}"


def test_labelled_dataset_get_label_pixel_counts(labelled_dataset):
    """Test that get_label_pixel_counts works with proper temporal bounds."""
    # This should not raise the pandas datetime comparison error
    pixel_counts = labelled_dataset.get_label_pixel_counts()
    assert isinstance(pixel_counts, dict), "Should return a dictionary"
    assert len(pixel_counts) > 0, "Should have some pixel counts"

    for count in pixel_counts.values():
        assert isinstance(count, int), f"Count should be integer, got {type(count)}"
        assert count >= 0, f"Count should be non-negative, got {count}"


def test_geoslice_temporal_bounds_consistency(label_dataset):
    """Test that GeoSlice creation uses consistent temporal bounds."""
    # Iterate through the dataset and check that temporal bounds are consistent
    for temporal_interval, row in label_dataset.index.iterrows():
        bounds = row.geometry.bounds

        # Create GeoSlice the same way as in the fixed methods
        geo_slice = (
            slice(bounds[0], bounds[2]),  # x: minx to maxx
            slice(bounds[1], bounds[3]),  # y: miny to maxy
            slice(temporal_interval.left, temporal_interval.right),  # t
        )

        # Verify temporal bounds types are consistent
        assert type(geo_slice[2].start) is type(temporal_interval.left), (
            "GeoSlice t.start should have same type as temporal_interval.left"
        )
        assert type(geo_slice[2].stop) is type(temporal_interval.right), (
            "GeoSlice t.stop should have same type as temporal_interval.right"
        )

        # Verify they're not the old hardcoded values
        assert geo_slice[2].start != 0 or geo_slice[2].stop != 1, (
            "Should not use hardcoded 0,1 temporal bounds"
        )


def test_no_datetime_comparison_error(label_dataset, labelled_dataset):
    """Test that the specific pandas datetime comparison error is fixed."""
    import pandas.errors

    # Test PlottableClassificationDataset.get_pixel_counts()
    try:
        label_dataset.get_pixel_counts()
    except pandas.errors.InvalidComparison as e:
        pytest.fail(f"get_pixel_counts still raises InvalidComparison error: {e}")

    # Test ClassificationRasterDataset.get_label_pixel_counts()
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
        nodata_value=None,
    )

    # Test that pixel counting works without errors
    pixel_counts = dataset.get_label_pixel_counts()
    assert isinstance(pixel_counts, dict), "Should return a dictionary"


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
    0: 997500,
    1: 52296,
    2: 10579,
    3: 2884,
    4: 3993,
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


def test_pixel_counts_with_shifted_labels(sparse_raster_path):
    """Test pixel counts with label 0 as nodata, where the classes are shifted by 1."""
    label_dataset = PlottableClassificationDataset(
        paths=sparse_raster_path, cache_size=1, nodata_value=0
    )
    # Set nodata_value to None to avoid filtering out any classes
    label_dataset.nodata_value = None

    pixel_counts = label_dataset.get_pixel_counts()

    # The _get_pixel_counts method undoes the shift of the labels
    # by adding 1 back, so we should get the original class values
    # Check that we get the expected original classes
    for expected_class, expected_count in EXPECTED_PIXEL_COUNTS.items():
        assert expected_class in pixel_counts, (
            f"Expected class {expected_class} not found in pixel_counts (the label shift should be undone in counting)"
        )
        actual_count = pixel_counts[expected_class]
        assert actual_count == expected_count, (
            f"For class {expected_class} (with shifted labels): expected {expected_count} pixels, got {actual_count}"
        )


def test_pixel_counts_class_distribution(label_dataset):
    """Test that pixel counts reflect the expected class distribution."""
    pixel_counts = label_dataset.get_pixel_counts()

    # Test specific properties of the sparse_raster.tif distribution
    # Class 0 should be the most common (background)
    assert pixel_counts[0] > pixel_counts[1], (
        "Class 0 should have more pixels than class 1"
    )
    assert pixel_counts[0] > 900000, "Class 0 should have over 900k pixels (background)"

    # Class 1 should be the second most common
    class_counts_sorted = sorted(pixel_counts.items(), key=lambda x: x[1], reverse=True)
    assert class_counts_sorted[0][0] == 0, "Class 0 should be most common"
    assert class_counts_sorted[1][0] == 1, "Class 1 should be second most common"

    # Class 3 should be the least common (except background)
    non_zero_classes = {k: v for k, v in pixel_counts.items() if k != 0}
    min_class = min(non_zero_classes.items(), key=lambda x: x[1])
    assert min_class[0] == 3, (
        f"Class 3 should have fewest pixels, but {min_class[0]} does"
    )
    assert min_class[1] == 2884, f"Class 3 should have 2884 pixels, got {min_class[1]}"


def test_integration_actual_counts_with_get_segmentation_dataset(sparse_raster_path):
    """Test actual pixel counts with the high-level get_segmentation_dataset function."""
    dataset = get_segmentation_dataset(
        images_dir=sparse_raster_path.parent,
        labels_dir=sparse_raster_path.parent,
        image_glob="sparse_raster.tif",
        label_glob="sparse_raster.tif",
        cache_size=1,
        nodata_value=None,
    )

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
        label_dataset = PlottableClassificationDataset(
            paths=sparse_raster_path, cache_size=cache_size, nodata_value=None
        )
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

    label_dataset_with_mapping = PlottableClassificationDataset(
        paths=sparse_raster_path, cache_size=1, nodata_value=None
    )
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

        # Create GeoSlice exactly as the fixed code does
        geo_slice = (
            slice(bounds[0], bounds[2]),
            slice(bounds[1], bounds[3]),
            slice(temporal_interval.left, temporal_interval.right),
        )

        # The key assertion: temporal bounds should NOT be 0 and 1
        # (unless the actual data legitimately has those values)
        if hasattr(temporal_interval.left, "year"):  # It's a timestamp
            assert geo_slice[2].start != 0, "t.start should not be hardcoded integer 0"
            assert geo_slice[2].stop != 1, "t.stop should not be hardcoded integer 1"

        # Additional check: types should match
        assert type(geo_slice[2].start) is type(temporal_interval.left), (
            f"t.start type {type(geo_slice[2].start)} should match temporal_interval.left type {type(temporal_interval.left)}"
        )
        assert type(geo_slice[2].stop) is type(temporal_interval.right), (
            f"t.stop type {type(geo_slice[2].stop)} should match temporal_interval.right type {type(temporal_interval.right)}"
        )


def test_error_regression_pandas_invalid_comparison(labelled_dataset):
    """Regression test for the specific pandas.errors.InvalidComparison error."""
    import pandas as pd

    # This is the exact error that was occurring before the fix:
    # "Invalid comparison between dtype=datetime64[ns] and int"
    # The error occurred when using hardcoded int values (0, 1) for temporal bounds
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


@pytest.mark.parametrize("nodata_value, label_offset", [(0, 1), (None, 0)])
def test_classes_start_at_0_for_the_model(
    sparse_raster_path, nodata_value, label_offset
):
    """With label 0 as nodata the classes start at 1 in the files, so they are shifted."""
    dataset = PlottableClassificationDataset(
        paths=sparse_raster_path, cache_size=1, nodata_value=nodata_value
    )
    with rasterio.open(sparse_raster_path) as src:
        stored = src.read(1).astype(int)
        xmin, ymin, xmax, ymax = src.bounds

    mask = dataset[xmin:xmax, ymin:ymax]["mask"].squeeze()

    assert dataset.nodata_value == nodata_value
    assert dataset.label_offset == label_offset
    np.testing.assert_array_equal(mask.numpy(), stored - label_offset)
    np.testing.assert_array_equal(dataset.values_as_in_files(mask), stored)


def test_label_0_is_nodata_by_default(sparse_raster_path):
    dataset = PlottableClassificationDataset(paths=sparse_raster_path, cache_size=1)
    assert dataset.nodata_value == 0
    assert 0 not in dataset.get_pixel_counts()


def test_sparse_split_needs_a_nodata_value(sparse_raster_path):
    dataset = PlottableClassificationDataset(
        paths=sparse_raster_path, cache_size=1, nodata_value=None
    )
    with pytest.raises(ValueError, match="nodata_value"):
        dataset.split_sparse_index()


def test_other_nodata_values_need_a_class_mapping(sparse_raster_path):
    """Whether the classes start at 0 or 1 is only known from the class mapping."""
    with pytest.raises(ValueError, match="class_mapping"):
        PlottableClassificationDataset(paths=sparse_raster_path, nodata_value=255)


def test_smallest_class_of_the_mapping_becomes_0(sparse_raster_path):
    """Here label 1 is nodata, and the classes are 2, 3 and 4."""
    dataset = PlottableClassificationDataset(
        paths=sparse_raster_path,
        cache_size=1,
        nodata_value=1,
        class_mapping={1: "unlabelled", 2: "a", 3: "b", 4: "c"},
    )
    with rasterio.open(sparse_raster_path) as src:
        stored = src.read(1).astype(int)
        xmin, ymin, xmax, ymax = src.bounds

    mask = dataset[xmin:xmax, ymin:ymax]["mask"].squeeze()

    assert dataset.label_offset == 2
    assert (mask[stored == 1] == -1).all()
    assert (mask[stored == 2] == 0).all()
    assert (mask[stored == 4] == 2).all()
    restored = dataset.values_as_in_files(mask)
    np.testing.assert_array_equal(restored[stored != 0], stored[stored != 0])
    assert 1 not in dataset.get_pixel_counts()
