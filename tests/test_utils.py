import pyproj
import shapely.geometry
from torchgeo.datasets.utils import BoundingBox

from eotorch.utils import torchgeo_bb_to_shapely


def test_torchgeo_bb_to_shapely_simple_conversion():
    """Test basic conversion of BoundingBox to Shapely Polygon without CRS change."""
    # Create a sample bounding box
    bbox = BoundingBox(minx=10.0, maxx=20.0, miny=30.0, maxy=40.0, mint=0.0, maxt=1.0)

    # Convert to shapely with same CRS
    polygon = torchgeo_bb_to_shapely(bbox, "EPSG:4326", "EPSG:4326")

    # Check that the coordinates are preserved
    assert isinstance(polygon, shapely.geometry.Polygon)
    assert polygon.bounds == (10.0, 30.0, 20.0, 40.0)


def test_torchgeo_bb_to_shapely_crs_conversion():
    """Test conversion with CRS transformation."""
    # Create a bounding box in UTM Zone 32N (EPSG:32632)
    # Coordinates approximately cover an area near Copenhagen, Denmark
    bbox = BoundingBox(
        minx=700000.0, maxx=710000.0, miny=6175000.0, maxy=6185000.0, mint=0.0, maxt=1.0
    )

    # Convert to WGS84 (EPSG:4326)
    polygon = torchgeo_bb_to_shapely(bbox, "EPSG:32632", "EPSG:4326")

    # Check that CRS conversion happened (bounds will be different)
    assert isinstance(polygon, shapely.geometry.Polygon)

    # The bounds should now be in WGS84 and be approximately in this range
    # (specific values depend on exact UTM zone and Earth model)
    minx, miny, maxx, maxy = polygon.bounds
    assert 12.0 < minx < 13.0  # Longitude range for this area
    assert 55.5 < miny < 56.0  # Latitude range for this area
    assert 12.0 < maxx < 13.0
    assert 55.5 < maxy < 56.0
    assert maxx > minx
    assert maxy > miny


def test_torchgeo_bb_to_shapely_pyproj_crs_objects():
    """Test using pyproj.CRS objects instead of strings."""
    # Create a sample bounding box
    bbox = BoundingBox(minx=10.0, maxx=20.0, miny=30.0, maxy=40.0, mint=0.0, maxt=1.0)

    # Create pyproj.CRS objects
    crs1 = pyproj.CRS("EPSG:4326")
    crs2 = pyproj.CRS("EPSG:4326")

    # Convert to shapely with CRS objects
    polygon = torchgeo_bb_to_shapely(bbox, crs1, crs2)

    # Check that the coordinates are preserved
    assert isinstance(polygon, shapely.geometry.Polygon)
    assert polygon.bounds == (10.0, 30.0, 20.0, 40.0)


def test_torchgeo_bb_to_shapely_default_target_crs():
    """Test using the default target CRS (WGS84)."""
    # Create a bounding box in Web Mercator (EPSG:3857)
    bbox = BoundingBox(
        minx=1113194.9,
        maxx=2226389.8,
        miny=5635549.2,
        maxy=6671175.8,
        mint=0.0,
        maxt=1.0,
    )

    # Convert to default (WGS84)
    polygon = torchgeo_bb_to_shapely(bbox, "EPSG:3857")

    # Check that CRS conversion happened
    assert isinstance(polygon, shapely.geometry.Polygon)

    # The bounds should now be in WGS84
    minx, miny, maxx, maxy = polygon.bounds
    # Web Mercator to WGS84 conversion validation
    assert -180.0 <= minx <= 180.0  # Valid longitude range
    assert -90.0 <= miny <= 90.0  # Valid latitude range
    assert -180.0 <= maxx <= 180.0
    assert -90.0 <= maxy <= 90.0
    assert maxx > minx
    assert maxy > miny
