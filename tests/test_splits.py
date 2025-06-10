"""Test functions for the splits module."""

import pytest

from eotorch.data.splits import _transform_bounds_to_crs


def test_transform_bounds_same_crs():
    """Test bounds transformation when source and target CRS are the same."""
    bounds = (0, 0, 10, 10)
    crs = "EPSG:4326"

    result = _transform_bounds_to_crs(bounds, crs, crs)

    assert result == bounds


def test_transform_bounds_different_crs():
    """Test bounds transformation between different CRS."""
    # Test transforming from WGS84 to Web Mercator
    bounds_wgs84 = (-1, -1, 1, 1)  # Small area around 0,0 in degrees
    source_crs = "EPSG:4326"
    target_crs = "EPSG:3857"

    result = _transform_bounds_to_crs(bounds_wgs84, source_crs, target_crs)

    # Check that transformation occurred (coordinates should be much larger in Web Mercator)
    assert result != bounds_wgs84
    assert abs(result[0]) > 100000  # Web Mercator coordinates should be large
    assert abs(result[1]) > 100000
    assert abs(result[2]) > 100000
    assert abs(result[3]) > 100000


def test_transform_bounds_invalid_crs():
    """Test bounds transformation with invalid CRS."""
    bounds = (0, 0, 10, 10)

    with pytest.raises(ValueError, match="Error transforming bounds"):
        _transform_bounds_to_crs(bounds, "INVALID:CRS", "EPSG:4326")


def test_bounds_preserve_order():
    """Test that transformed bounds maintain proper min/max order."""
    # Use a bounds that might get rotated during transformation
    bounds = (10, 20, 15, 25)  # minx, miny, maxx, maxy
    source_crs = "EPSG:4326"
    target_crs = "EPSG:3857"

    result = _transform_bounds_to_crs(bounds, source_crs, target_crs)

    # Ensure the result maintains proper bounds order
    assert result[0] < result[2]  # minx < maxx
    assert result[1] < result[3]  # miny < maxy
