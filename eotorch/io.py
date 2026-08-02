from pathlib import Path
import geopandas as gpd

def read_vector(path: str | Path) -> gpd.GeoDataFrame:
    """Read a vector file into a GeoDataFrame.

    Parameters
    ----------
    path : str | Path
        Path to vector file. Supported formats include all vector formats supported by 
        GeoPandas through `read_file` and pyarrow's geoparquet.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with geometry and attributes from the input file.
    """
    path = Path(path)
    if path.suffix == '.parquet':
        return gpd.read_parquet(path)
    else:
        return gpd.read_file(path)
    

def write_vector(gdf: gpd.GeoDataFrame, path: str | Path) -> None:
    """Write a GeoDataFrame to a vector file.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame to write.
    path : str | Path
        Output path. Supported formats include all vector formats supported by 
        GeoPandas through `to_file` and pyarrow's geoparquet.
    """
    path = Path(path)
    if path.suffix == '.parquet':
        gdf.to_parquet(path)
    else:
        gdf.to_file(path)