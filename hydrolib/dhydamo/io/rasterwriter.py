"""Write rasters."""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

DRIVERS = {".tif": "GTiff", ".tiff": "GTiff", ".asc": "AAIGrid"}


def write_raster(
    path: Path | str,
    array: np.ndarray,
    transform: Affine,
    crs: CRS | str | None = None,
    nodata: float = -999.0,
    compress: bool = True,
) -> Path:
    """Write a single band raster.

    The format follows from the extension: GeoTIFF for ``.tif`` and ``.tiff``, ASCII
    grid for ``.asc``. Missing folders are created.

    Parameters
    ----------
    path : Path or str
        Output file.
    array : np.ndarray
        2D array with values.
    transform : Affine
        Raster transform.
    crs : CRS or str, optional
        Coordinate reference system. Defaults to None.
    nodata : float, optional
        Nodata value. Defaults to -999.0.
    compress : bool, optional
        Deflate compression for GeoTIFF. Defaults to True.

    Returns
    -------
    Path
        Path of the written raster.

    Raises
    ------
    ValueError
        If the extension is not supported.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in DRIVERS:
        raise ValueError(f"Unsupported raster extension '{path.suffix}', use one of {list(DRIVERS)}.")
    driver = DRIVERS[suffix]
    profile = {
        "driver": driver,
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": array.dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
    }
    if driver == "GTiff" and compress:
        profile["compress"] = "deflate"
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)
    return path
