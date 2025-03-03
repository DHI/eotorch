from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import rasterio as rst
from pyogrio import read_dataframe
from rasterio.features import rasterize

from ..utils import get_outpath

if TYPE_CHECKING:
    import rasterio.Affine
    import rasterio.CRS
    from shapely.geometry import MultiPolygon, Polygon

logger = logging.getLogger(__name__)


class VectorSource(ABC):
    """
    A source class for rasterizing vector shapes.
    """

    def __init__(
        self,
        transform: rasterio.Affine | None = None,
        shape: tuple[int, int] | None = None,
        crs: rasterio.CRS | str | None = None,
    ):
        self.transform = transform
        self.shape = shape
        self.crs = crs
        self.geoms = None
        self.labels = None
        self.profile = {
            "driver": "GTiff",
            "interleave": "band",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "compress": "lzw",
            "nodata": 0,
            "transform": transform,
            "height": shape[0],
            "width": shape[1],
            "count": 1,
            "crs": crs,
        }

    @abstractmethod
    def _get_geoms(self) -> list:
        pass

    def rasterize_polygons(self, out_path: Path | str, **kwargs):
        geoms = self._get_geoms()
        labels = np.zeros(self.shape, dtype="uint8")
        for i, geom in enumerate(geoms, start=1):
            try:
                rasterize(
                    geom,
                    out=labels,
                    transform=self.transform,
                    default_value=i,
                    **kwargs,
                )
            except ValueError:
                pass
        self.profile.update(dtype="uint8")

        with rst.open(Path(out_path), "w", **self.profile) as dst:
            dst.write(labels, 1)


class FileSource(VectorSource):
    """
    A :class:`.VectorSource` for reading vector geometries from files.
    """

    def __init__(
        self,
        classes: list,
        root_dir: Path | str,
        img_path: Path | str | None = None,
        transform: rasterio.Affine | None = None,
        shape: tuple[int, int] | None = None,
        crs: rasterio.CRS | str | None = None,
    ):
        if img_path is not None and None in [transform, shape, crs]:
            transform, shape, crs = infer_meta(img_path)

        super().__init__(transform, shape, crs)
        self.classes = classes
        self.root_dir = Path(root_dir)

    def _get_geoms(self) -> list:
        features = []
        for c in self.classes:
            features.append(read_shp(c, self.root_dir, self.crs))
        return features


class DataframeSource(VectorSource):
    """
    A :class:`.VectorSource` for reading vector geometries from dataframe.
    """

    def __init__(
        self,
        dataframe: gpd.GeoDataFrame,
        class_column: str,
        img_path: Path | str | None = None,
        transform: rasterio.Affine | None = None,
        shape: tuple[int, int] | None = None,
        crs: rasterio.CRS | str | None = None,
    ):
        if img_path is not None and None in [transform, shape, crs]:
            transform, shape, crs = infer_meta(img_path)

        super().__init__(transform, shape, crs)
        self.gdf = dataframe
        self.class_column = class_column

    def _get_geoms(self) -> list:
        self.gdf = self.gdf.to_crs(self.crs)
        features = [g.geometry for _, g in self.gdf.groupby([self.class_column])]
        return features


def infer_meta(img_path: Path | str) -> tuple[rasterio.Affine, tuple, rasterio.CRS]:
    """
    Infer basic raster metadata information from a raster path.

    Parameters:
        img_path (Path | str):
            Raster image path.

    Returns:
        tuple[rasterio.Affine, tuple, rasterio.CRS]:
            Tuple of raster transform, shape, and coordinate system.
    """
    img_path = Path(img_path)
    with rst.open(img_path) as src:
        transform = src.transform
        shape = src.shape
        crs = src.crs
    return transform, shape, crs


def read_shp(
    fn: Path | str, root_dir: Path | str, crs: rasterio.CRS | str
) -> list[Polygon | MultiPolygon]:
    """
    Read shapefiles matching either the filename `fn` or looks for the shapefile/geojson
    matching the stem name without extension, e.g. 'trees', in the `root_dir`.

    Parameters:
        fn (Path | str):
            Filename of vector file or name of class matching the stem name of the file without extension.
        root_dir (Path | str):
            Root directory to look for vector files in.
        crs (rasterio.crs.CRS | str):
            Coordinate system to project files to.

    Raises:
        ValueError:
            The file could either not be found or the file format is not one of the supported extensions.

    Returns:
        list[Polygon | MultiPolygon]:
            List of geometries
    """
    fn = Path(fn)
    root_dir = Path(root_dir)
    try:
        if fn.suffix == "":
            if os.path.exists(os.path.join(root_dir, f"{fn}.shp")):
                shp = (
                    read_dataframe(os.path.join(root_dir, f"{fn}.shp"))
                    .to_crs(crs)
                    .geometry
                )
            elif os.path.exists(os.path.join(root_dir, f"{fn}.geojson")):
                shp = (
                    read_dataframe(os.path.join(root_dir, f"{fn}.geojson"))
                    .to_crs(crs)
                    .geometry
                )
            else:
                raise ValueError(
                    f"The file format is either not supported or the file {fn} cannot be found."
                )
        else:
            shp = read_dataframe(fn).to_crs(crs).geometry
    except:
        logger.warning("The file does not exist:", os.path.join(root_dir, f"{fn}.shp"))
        shp = []
    return shp
