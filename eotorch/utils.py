import inspect
import json
import math
from pathlib import Path
from typing import Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import pyproj
import rasterio as rst
import shapely
import shapely.geometry
import shapely.ops
from pyproj import Geod
from rasterio.windows import Window
from rtree.index import Index, Property
from shapely import MultiPolygon, Polygon
from shapely.ops import transform
from torchgeo.datasets.utils import BoundingBox


class Region:
    def __init__(
        self, polygon: Polygon, crs: str | pyproj.CRS, name: str | None = None, **kwargs
    ):
        self.polygon = polygon
        self.name = name
        self.crs = pyproj.CRS(crs)  # will raise CRSError if invalid

        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def create(
        cls,
        source: str | Path | dict | Polygon | list[Polygon],
        crs: str = None,
        name: str = None,
        dissolve: bool = True,
        **kwargs,
    ):
        if isinstance(source, Region):
            return source
        if isinstance(source, str):
            if source.startswith("{"):
                return cls.from_json(source)
            source = Path(source)

        if isinstance(source, Path):
            if not source.is_file():
                raise FileNotFoundError(f"{str(source)} is not a file")
            if source.suffix == ".shp":
                return cls.from_shapefile(
                    source, crs=crs, dissolve=dissolve, name=name, **kwargs
                )
            elif source.suffix == ".json":
                return cls.from_json(source)
            elif source.suffix == ".geojson":
                return cls.from_geojson(source, crs=crs, name=name, **kwargs)
            elif source.suffix in {".tif", ".tiff"}:
                return cls.from_raster_bounds(source, **kwargs)
            else:
                raise ValueError(
                    f"Unsupported file type: {source.suffix}. Supported types are .shp, .json, .tif"
                )

        if isinstance(source, dict):
            return cls.from_dict(source)

        if isinstance(source, Polygon):
            return cls.from_polygon(source, crs=crs, name=name, **kwargs)

        if isinstance(source, list):
            return cls.from_multiple_polygons(
                source, crs=crs, dissolve=dissolve, name=name, **kwargs
            )

    @classmethod
    def from_shapefile(
        cls,
        path: str | Path,
        crs: str = None,
        dissolve: bool = True,
        name: str = None,
        **kwargs,
    ):
        """
        Create a region from a shapefile. The shapefile must contain exactly one polygon.
        If the CRS is not provided, it will be attempted to be read from the shapefile.
        If the CRS is not provided and not in the shapefile, a ValueError will be raised.

        Args:
            path (str | Path): The path to the shapefile
            crs (str, optional): The CRS of the shapefile. Defaults to None.
            dissolve (bool, optional): Whether to dissolve multiple polygons into one. Defaults to True.
            name (str, optional): The name of the region. Defaults to None.

        Returns:
            Region: The region object

        Raises:
            ValueError: If the shapefile does not contain exactly one polygon
            ValueError: If the CRS is not provided and not in the shapefile
        """
        df = gpd.read_file(path)
        if (crs is None) and (df.crs is None):
            raise ValueError("Either crs or crs in shapefile must be provided")
        if crs is None:
            crs = df.crs
        if len(df) == 1:
            return cls.from_polygon(
                polygon=df.geometry[0], crs=crs, name=name, **kwargs
            )
        else:
            return cls.from_multiple_polygons(
                df.geometry, crs=crs, dissolve=dissolve, name=name, **kwargs
            )

    @classmethod
    def from_polygon(
        cls, polygon: Polygon, crs: str, name: str | None = None, **kwargs
    ):
        """
        Create a region from a shapely polygon.

        Args:
            polygon (Polygon): The polygon
            crs (str): The CRS of the polygon
            name (str, optional): The name of the region. Defaults to None.

        Returns:
            Region: The region object
        """
        return cls(polygon=polygon, crs=crs, name=name, **kwargs)

    @classmethod
    def from_multiple_polygons(
        cls,
        polygons: list[Polygon] | gpd.GeoSeries,
        crs: str,
        dissolve: bool = True,
        name: str | None = None,
        **kwargs,
    ) -> Union["Region", list["Region"]]:
        """
        Create a region from multiple shapely polygons.

        Args:
            polygons (list[Polygon] | gpd.GeoSeries): The polygons
            crs (str): The CRS of the polygons
            dissolve (bool, optional): Whether to dissolve the polygons into one. Defaults to True.
                                        If True, returns a single region. If False, returns a list of regions.
            name (str, optional): The name of the region. Defaults to None.

        Returns:
            Region | list[Region]: The region object or a list of region objects
        """

        if isinstance(polygons, list):
            polygons = gpd.GeoSeries(polygons)

        if dissolve:
            try:
                polygon = polygons.to_frame().dissolve().geometry[0]
                return cls.from_polygon(polygon=polygon, crs=crs, name=name, **kwargs)
            except Exception as e:
                raise ValueError("Unable not dissolve polygons") from e

        return [
            cls.from_polygon(polygon=p, crs=crs, name=name, **kwargs) for p in polygons
        ]

    @classmethod
    def from_dict(cls, dict_obj: dict):
        """
        Create a region from a dictionary. The dictionary must contain the keys "geometry", "crs" and "properties".
        The "geometry" key must contain a shapely geometry object, the "crs" key must contain a pyproj CRS object and
        the "properties" key must contain the properties of the region.

        Args:
            dict_obj (dict): The dictionary

        Returns:
            Region: The region object
        """
        pol = shapely.geometry.shape(dict_obj["geometry"])
        crs = pyproj.CRS.from_json_dict(dict_obj["crs"])
        return cls.from_polygon(polygon=pol, crs=crs, **dict_obj["properties"])

    @classmethod
    def from_json(cls, json_data: str | Path):
        """
        Create a region from a JSON string or a JSON file.

        Args:
            json_data (str | Path): The JSON string or file

        Returns:
            Region: The region object
        """
        if isinstance(json_data, str) and json_data.startswith("{"):
            json_data = json.loads(json_data)
        else:
            with open(json_data) as f:
                json_data = json.load(f)

        return cls.from_dict(json_data)
    
    @classmethod
    def from_geojson(cls, geojson_path: str | Path, crs: str = None, name: str = None, **kwargs):
        """
        Create a region from a GeoJSON file.

        Args:
            geojson_path (str | Path): The path to the GeoJSON file
            crs (str, optional): The CRS of the GeoJSON. Defaults to None.
            name (str, optional): The name of the region. Defaults to None.

        Returns:
            Region: The region object
        """
        with open(geojson_path) as f:
            geojson_data = json.load(f)

        if crs is None:
            crs = geojson_data.get("crs", {}).get("properties", {}).get("name")
            if crs is None:
                raise ValueError("CRS not provided in GeoJSON file, please specify a CRS.")

        polygon = shapely.geometry.shape(geojson_data["features"][0]["geometry"])
        return cls.from_polygon(polygon=polygon, crs=crs, name=name, **kwargs)

    @classmethod
    def from_raster_bounds(
        cls,
        raster_path: str | Path,
        exclude_nodata: bool = False,
    ):
        """
        Create a region from the bounds of a raster file.
        This is useful for creating a region that covers the entire raster file, e.g.
        for fetching data matching the bounds of an existing raster file.

        Args:
            raster_path (str | Path): The path to the raster file
            exclude_nodata (bool, optional): Whether to exclude nodata values when creating the region. Defaults to False.

        Returns:
            Region: The region object
        """

        # TODO: Add support for blob storage paths

        from rasterio.coords import BoundingBox
        from rasterio.windows import bounds, get_data_window

        with rst.open(raster_path) as src:
            crs = src.crs

            if exclude_nodata:
                data_window = get_data_window(src.read(masked=True))
                _bounds = BoundingBox(*bounds(data_window, src.transform))
            else:
                _bounds = src.bounds

        return cls.from_polygon(
            Polygon(
                (
                    (_bounds.left, _bounds.top),
                    (_bounds.right, _bounds.top),
                    (_bounds.right, _bounds.bottom),
                    (_bounds.left, _bounds.bottom),
                )
            ),
            crs=crs,
        )

    @property
    def bounds(self):
        return self.polygon.bounds

    @property
    def number_of_points(self):
        if isinstance(self.polygon, MultiPolygon):
            return sum([len(p.exterior.coords) for p in self.polygon.geoms])
        return len(self.polygon.exterior.coords)

    @property
    def is_in_meters(self):
        return self.crs.axis_info[0].unit_name == "metre"

    @property
    def is_in_degrees(self):
        return self.crs.axis_info[0].unit_name == "degree"

    # def to_db(self):
    #     raise NotImplementedError

    def get_area(self, geodesic: bool = False, unit: str = "km^2") -> float:
        if not geodesic:
            area = self.polygon.area
        else:
            geod = Geod(ellps="WGS84")
            area = abs(geod.geometry_area_perimeter(self.polygon)[0])
        if unit == "m^2":
            return area
        if unit == "km^2":
            return area / 1_000_000
        if unit == "ha":
            return area / 10_000
        if unit == "acres":
            return area / 4_046.86
        raise ValueError("Invalid unit")

    def to_dict(self, compact: bool = False):
        """
        Returns a dictionary representation of the region.

        Args:
            compact (bool, optional): Whether to return a compact representation of the CRS. Defaults to False.

        Returns:
            dict: The dictionary representation
        """
        dict_to_return = {
            "properties": {
                k: v for k, v in self.__dict__.items() if k not in {"polygon", "crs"}
            },
            "crs": self.crs.to_json_dict() if not compact else self.crs.to_dict(),
            "geometry": json.loads(shapely.to_geojson(self.polygon, indent=4)),
        }
        return dict_to_return

    def to_json(self, file_path: str | Path | None = None, compact: bool = False):
        """
        Writes the region to a JSON file or returns a JSON string.

        Args:
            file_path (str | Path | None, optional): The path to the file. Defaults to None.
            If None, the JSON string will be returned.
            compact (bool, optional): Whether to return a compact representation of the CRS. Defaults to False.

        Returns:
            str: The JSON string if file_path is None
            str: The path to the file if file_path is provided

        """
        json_data = self.to_dict(compact=compact)
        if not file_path:
            return json.dumps(json_data, indent=4)

        with open(file_path, "w") as f:
            json.dump(json_data, f, indent=4)

        return file_path

    def with_metres_buffer(self, buffer_in_m: int) -> "Region":
        """
        Returns a new region with a buffer of buffer_in_m around the polygon.

        Args:
            buffer_in_m (int): The buffer in meters

        Returns:
            Region: The new region

        Raises:
            ValueError: If the current CRS is not in meters or degrees
        """
        if not self.is_in_meters and not self.is_in_degrees:
            raise ValueError(
                "Currently only supports CRS with units of metres or degrees"
            )
        if self.is_in_degrees:  # This is just rough, not exact
            buffer_in_m = max(
                buffer_in_m / 111_111,
                buffer_in_m / (111_111 * math.cos(self.bounds[1])),
            )
        # return buffer_in_m

        return Region.from_polygon(
            polygon=self.polygon.buffer(buffer_in_m),
            crs=self.crs,
            name=self.name,
        )

    def with_pixel_buffer(self, pixel_buffer: int, pixel_size_m: int = 10) -> "Region":
        """
        Wrapper around with_metres_buffer that takes a pixel buffer and pixel size in meters.

        Args:
            pixel_buffer (int): The buffer in pixels
            pixel_size_m (int, optional): The size of a pixel in meters. Defaults to 10.

        Returns:
            Region: The new region
        """
        number_of_meters = pixel_buffer * pixel_size_m
        return self.with_metres_buffer(number_of_meters)

    def crop_raster(self, raster_path: str | Path, out_path: str | Path = None):
        """
        Crops a raster file to the region.

        Args:
            raster_path (str | Path): The path to the raster file
            out_path (str | Path, optional): The path to save the cropped raster. Setting out_path to None
            will overwrite the original raster. Defaults to None.

        Returns:
            str: The path to the cropped raster
        """

        from rasterio.mask import mask

        if not Path(raster_path).is_file():
            raise FileNotFoundError(f"{raster_path} is not a file")

        with rst.open(raster_path) as src:
            out_image, out_transform = mask(
                src, [self.polygon], crop=True, all_touched=True
            )
            out_meta = src.meta.copy()
            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                }
            )

            if not out_path:
                out_path = raster_path

            with rst.open(out_path, "w", **out_meta) as dest:
                dest.write(out_image)

    def to_crs(self, crs: str | pyproj.CRS) -> "Region":
        if isinstance(crs, str):
            crs = pyproj.CRS(crs)

        projection = pyproj.Transformer.from_proj(
            self.crs,  # source crs
            crs,  # target_crs
            always_xy=True,
        )
        return Region(
            polygon=transform(projection.transform, self.polygon),
            crs=crs,
            name=self.name,
        )

    def to_latlon(self) -> "Region":
        """
        Returns a new region in EPSG:4326 (lat/lon).
        """
        return self.to_crs("EPSG:4326")

    def __repr__(self):
        return f"Region(name={self.name}, crs={self.crs.name})"

    def _repr_svg_(self):
        print(self.__repr__())
        return self.polygon._repr_svg_()

    def difference(self, other: "Region") -> "Region":
        """
        Returns a new region that is the difference of the current region and another region.
        """
        return Region(
            polygon=self.polygon.difference(other.polygon),
            crs=self.crs,
            name=self.name,
        )

    def intersection(self, other: "Region") -> "Region":
        """
        Returns a new region that is the intersection of the current region and another region.
        """
        return Region(
            polygon=self.polygon.intersection(other.polygon),
            crs=self.crs,
            name=self.name,
        )

    def get_folium_map(
        self,
        draw_full_geometry: bool = True,
        draw_bounds: bool = False,
        map=None,
        **kwargs,
    ):
        import folium

        map = map or folium.Map()

        style_dict = dict(fill=False, weight=5, opacity=0.7, color="olive")
        style_dict.update(kwargs)

        _tmp = self.to_latlon()

        if draw_full_geometry:
            folium.GeoJson(
                _tmp.polygon,
                style_function=lambda x: style_dict,
                name=self.name or "region geometry",
            ).add_to(map)
        if draw_bounds:
            folium.GeoJson(
                shapely.geometry.box(*_tmp.bounds),
                style_function=lambda x: style_dict,
                name="Bounds",
            ).add_to(map)

        folium.ClickForMarker().add_to(map)
        map.fit_bounds(bounds=convert_bounds(_tmp.bounds))

        return map

    def plot_tif(
        self,
        tif_file_path: str | Path,
        red_band_idx: int = 3,
        green_band_idx: int = 2,
        blue_band_idx: int = 1,
        map=None,
    ):
        import folium
        import matplotlib.pyplot as plt

        cm = None

        if not map:
            map = self.get_folium_map()

        with rst.open(tif_file_path) as src:
            number_of_bands = src.count

            window = rst.windows.from_bounds(*self.bounds, src.transform)

            if number_of_bands == 1:
                data = src.read(1, window=window)
                # cm = plt.get_cmap("twilight")

            else:
                data = src.read(
                    [red_band_idx, green_band_idx, blue_band_idx], window=window
                ).transpose(1, 2, 0)
        bounds = self.to_latlon().bounds
        folium.raster_layers.ImageOverlay(
            image=data,
            bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
            # origin="lower",
            # colormap=cm,
            # colormap=lambda x: (1, 0, 0, x),
            name=Path(tif_file_path).name,
            mercator_project=True,
        ).add_to(map)

        # add layer control
        folium.LayerControl().add_to(map)

        return map

def tranform_index(
    index: Index, current_crs: str | pyproj.CRS, new_crs: str | pyproj.CRS
) -> Index:
    """
    Transform an R-tree index to a new coordinate reference system (CRS).

    Parameters:
        index (Index):
            The R-tree index to transform.
        new_crs (str or pyproj.CRS):
            The new coordinate reference system.

    Returns:
        Index:
            A new R-tree index with transformed coordinates.
    """
    if isinstance(current_crs, str):
        current_crs = pyproj.CRS(current_crs)

    if isinstance(new_crs, str):
        new_crs = pyproj.CRS(new_crs)

    new_index = Index(interleaved=False, properties=Property(dimension=3))

    project = pyproj.Transformer.from_crs(
        pyproj.CRS(str(current_crs)), pyproj.CRS(str(new_crs)), always_xy=True
    ).transform
    for hit in index.intersection(index.bounds, objects=True):
        old_minx, old_maxx, old_miny, old_maxy, mint, maxt = hit.bounds
        old_box = shapely.geometry.box(old_minx, old_miny, old_maxx, old_maxy)
        new_box = shapely.ops.transform(project, old_box)
        new_minx, new_miny, new_maxx, new_maxy = new_box.bounds
        new_bounds = (new_minx, new_maxx, new_miny, new_maxy, mint, maxt)
        new_index.insert(hit.id, new_bounds, hit.object)

    return new_index


def window_to_np_idc(window: Window) -> Tuple[slice, slice]:
    """Convert a window to numpy array indices."""
    return (
        slice(int(round(window.row_off)), int(round(window.row_off + window.height))),
        slice(int(round(window.col_off)), int(round(window.col_off + window.width))),
    )


def get_init_args(cls):
    """Returns the argument names of a class's __init__ method, excluding 'self'."""
    signature = inspect.signature(cls.__init__)  # Get the signature of __init__
    return [
        param.name for param in signature.parameters.values() if param.name != "self"
    ]


def slice_image(
    height: int,
    width: int,
    size: int = 256,
    stride: int = 1,
    overlap: bool = True,
    offset: int = 0,
    as_windows: bool = False,
) -> Sequence[slice]:
    """
    Generates discrete slices of a 2D array of a given dimension (size*size).

    Parameters:
        height (int):
            Height of array.
        width (int):
            Width of array.
        size (int, optional):
            Size of slices/subsets/patches. Defaults to 256.
        stride (int, optional):
            The overlap between patches. The stride is defined as the number of
            windows that will pass over a subset, i.e. a `stride` of 1 will have no overlap,
            while a stride of 2 will result in 50% overlap between two patches in one dimension
            for half the slices. Defaults to 1.
        overlap (bool, optional):
            Trims edges of an array if the height/width is not divisible by the size.
            Setting this to False with an array not equally divisible by `size` will result in
            irregular slices. Defaults to True.
        offset (int, optional):
            Shifts the slices (offset, offset) `offset` pixels diagonally towards the bottom right.
            Defaults to 0.

    Returns:
        Sequence[slice]:
            Sequence of 2D slices or if specified, rasterio Window objects.
    """
    assert offset < size, "Offset must be less than size."
    dims = []
    for dim in (height, width):
        dim_bounds = []
        for b in range(0, dim, size // stride):
            if (overlap and stride == 1) or (stride > 1):
                bounds = (b, b + size)
                bounds = (dim - size, dim) if bounds[1] > dim else bounds
                try:
                    if not bounds == dim_bounds[-1]:
                        dim_bounds.append(bounds)
                except IndexError:
                    dim_bounds.append(bounds)
            elif (b + (size // stride)) < dim:
                bounds = (b, b + size)
                bounds = (dim - size, dim) if bounds[1] > dim else bounds
                dim_bounds.append(bounds)

        out_bounds = [(i + offset, j + offset) for i, j in dim_bounds[:-1]]
        out_bounds.append(dim_bounds[-1])
        dims.append(out_bounds)

    slices = []
    for width in dims[1]:
        for height in dims[0]:
            if as_windows:
                slices.append(Window.from_slices(slice(*height), slice(*width)))
            else:
                slices.append((slice(*height), slice(*width)))

    return slices


def _format_filepaths(filepaths):
    """Format filepaths for display in plot titles."""
    if not filepaths:
        return ""

    # If only one filepath, show it directly
    if len(filepaths) == 1:
        path = Path(filepaths[0])
        # Show truncated path with at least two parent directories if path is too long
        if len(str(path)) > 40:
            # Check if it's in home directory or current working directory
            home_dir = Path.home()
            cwd = Path.cwd()

            if home_dir == path.parent or cwd == path.parent:
                return f".../{path.name}"

            # Include at least two parent directories
            parent_parts = list(path.parents)
            if len(parent_parts) >= 2:
                return f".../{parent_parts[1].name}/{parent_parts[0].name}/{path.name}"
            elif len(parent_parts) == 1:
                return f".../{parent_parts[0].name}/{path.name}"
            else:
                return f".../{path.name}"
        return str(path)

    # Convert all paths to Path objects
    paths = [Path(p) for p in filepaths]

    # Find common prefix to avoid repetition
    try:
        common_path = Path(Path(*paths[0].parts[:1]).anchor)
        for i in range(1, len(paths[0].parts)):
            potential_common = Path(*paths[0].parts[: i + 1])
            if all(str(potential_common) in str(p) for p in paths):
                common_path = potential_common
            else:
                break
    except (IndexError, ValueError):
        common_path = None

    if common_path and len(str(common_path)) > 10:
        formatted = f"{common_path}/\n"
        # Show only the unique parts after the common prefix
        unique_parts = [str(p).replace(str(common_path), "").lstrip("/") for p in paths]

        # If there are too many files, limit display
        if len(unique_parts) > 3:
            formatted += "\n".join(unique_parts[:2])
            formatted += f"\n...and {len(unique_parts) - 2} more files"
        else:
            formatted += "\n".join(unique_parts)
        return formatted

    # If no common prefix or too short, show first few and indicate total count
    if len(paths) > 3:
        return (
            "\n".join([str(p) for p in paths[:2]])
            + f"\n...and {len(paths) - 2} more files"
        )

    return "\n".join([str(p) for p in paths])


def torchgeo_bb_to_shapely(
    bbox: list | BoundingBox,
    bbox_crs: str | pyproj.CRS,
    target_crs: str | pyproj.CRS = "EPSG:4326",
):
    """
    Convert a list of bound coordinates or a BoundingBox object to a Shapely Polygon.

    Parameters:
        bbox (list or BoundingBox):
            The bounding box to convert. If a list, it should be in the format
            [minx, maxx, miny, maxy], it may also contain more than 4 values (e.g. mint, maxt).
        If a BoundingBox object, it should have the attributes minx, maxx, miny, maxy.
        bbox_crs (str or pyproj.CRS):
            The coordinate reference system of the input bounding box.
        target_crs (str or pyproj.CRS, optional):
            The target coordinate reference system. Defaults to "EPSG:4326".

    Returns:
        shapely.geometry.Polygon:
            The converted Shapely Polygon.
    """
    if isinstance(target_crs, str):
        target_crs = pyproj.CRS(target_crs)

    if isinstance(bbox_crs, str):
        bbox_crs = pyproj.CRS(bbox_crs)

    # Create a shapely polygon from the bbox coordinates
    if isinstance(bbox, list):
        polygon = shapely.geometry.box(
            minx=bbox[0], miny=bbox[2], maxx=bbox[1], maxy=bbox[3]
        )
    else:
        polygon = shapely.geometry.box(
            minx=bbox.minx, miny=bbox.miny, maxx=bbox.maxx, maxy=bbox.maxy
        )

    # If the CRS is the same, no need to transform
    if bbox_crs == target_crs:
        return polygon

    # Create transformer for the coordinate conversion
    transformer = pyproj.Transformer.from_crs(bbox_crs, target_crs, always_xy=True)

    # Transform the polygon to the target CRS
    transformed_polygon = shapely.ops.transform(transformer.transform, polygon)

    return transformed_polygon
    
def convert_bounds(bbox, invert_y=False):
    """
    Helper method for changing bounding box representation to leaflet notation

    ``(lon1, lat1, lon2, lat2) -> ((lat1, lon1), (lat2, lon2))``
    """
    x1, y1, x2, y2 = bbox
    if invert_y:
        y1, y2 = y2, y1
    return ((y1, x1), (y2, x2))
