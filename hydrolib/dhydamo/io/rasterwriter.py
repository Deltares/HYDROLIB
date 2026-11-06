from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

DRIVERS = {".tif": "GTiff", ".tiff": "GTiff", ".asc": "AAIGrid"}


def write_raster(
    path: Path | str,
    array: np.ndarray,
    transform: Affine,
    crs=None,
    nodata: float = -999.0,
    compress: bool = True,
) -> Path:
    """Write a single band raster. The format follows from the extension (.tif/.tiff or .asc).

    Args:
        path (Path | str): output file
        array (np.ndarray): 2D array with values
        transform (Affine): raster transform
        crs (optional): coordinate reference system. Defaults to None.
        nodata (float, optional): nodata value. Defaults to -999.0.
        compress (bool, optional): deflate compression for GeoTIFF. Defaults to True.

    Returns:
        Path: path of the written raster
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in DRIVERS:
        raise ValueError(
            f"Unsupported raster extension '{path.suffix}', use one of {list(DRIVERS)}."
        )
    driver = DRIVERS[suffix]
    profile = dict(
        driver=driver,
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    )
    if driver == "GTiff" and compress:
        profile["compress"] = "deflate"
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)
    return path
