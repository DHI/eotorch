from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, List

import geopandas as gpd
import numpy as np
import rasterio as rst
from pyogrio import read_dataframe
from rasterio.features import rasterize
from rasterio.windows import Window, get_data_window
from rasterio.windows import transform as window_transform

from eotorch import utils

if TYPE_CHECKING:
    import rasterio.CRS
    from affine import Affine
    from shapely.geometry import MultiPolygon, Polygon

logger = logging.getLogger(__name__)


class VectorSource(ABC):
    """
    A source class for rasterizing vector shapes.
    """

    def __init__(
        self,
        transform: Affine | None = None,
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

    def rasterize_polygons(
        self,
        out_path: Path | str,
        exclude_nodata_bounds: bool = True,
        min_size: int = 300,
        class_order: List[int] = None,
        **kwargs,
    ):
        geoms = self._get_geoms()
        labels = np.zeros(self.shape, dtype="uint8")

        if class_order is None:
            class_order = list(range(1, len(geoms) + 1))

        class_order = [c for c in class_order if c in geoms]

        idc_not_in_class_order = [i for i in geoms.keys() if i not in class_order]

        for i in idc_not_in_class_order:
            geom = geoms[i]
            rasterize(
                geom,
                out=labels,
                transform=self.transform,
                default_value=i,
                **kwargs,
            )

        for i in class_order:
            geom = geoms[i]
            rasterize(
                geom,
                out=labels,
                transform=self.transform,
                default_value=i,
                **kwargs,
            )
        self.profile.update(dtype="uint8")

        if exclude_nodata_bounds:
            window = get_data_window(labels, nodata=0)
            # make sure the window is at least of size min_size
            window_height = max(window.height, min_size)
            window_width = max(window.width, min_size)
            col_off = max(0, window.col_off - 50)
            row_off = max(0, window.row_off - 50)
            window = Window(
                col_off=col_off,
                row_off=row_off,
                height=window_height,
                width=window_width,
            )

            labels = labels[utils.window_to_np_idc(window)]
            # Update the transform to reflect the new origin after cropping
            updated_transform = window_transform(window, self.transform)
            self.profile.update(
                height=labels.shape[0],
                width=labels.shape[1],
                transform=updated_transform,
            )

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
        transform: Affine | None = None,
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
        features = {i: f for i, f in enumerate(features, start=1)}
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
        features = {c: g.geometry for c, g in self.gdf.groupby(self.class_column)}
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
