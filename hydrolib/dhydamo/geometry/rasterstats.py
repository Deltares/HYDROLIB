import logging
from itertools import product

import geopandas as gpd
import numpy as np
import pandas as pd
import PIL.Image
import PIL.ImageDraw
import rasterio

from rasterio.windows import Window
from pathlib import Path
from typing import Union

from hydrolib.dhydamo.geometry import common

logger = logging.getLogger(__name__)


class RasterPart:

    def __init__(self, f, xmin, ymin, xmax, ymax):
        self.f = f
        # Indices, not coordinates
        self.xmin = max(xmin, 0)
        self.xmax = min(xmax, self.f.shape[1])
        self.ymin = max(ymin, 0)
        self.ymax = min(ymax, self.f.shape[0])

        self.set_window()

        self.get_corners()

    @classmethod
    def from_bounds(cls, f, bnds):
        # Convert xy bounds to indices
        idxs = list(f.index(bnds[0], bnds[1]))[::-1] + list(f.index(bnds[2], bnds[3]))[::-1]
        return cls(f, min(idxs[0], idxs[2]), min(idxs[1], idxs[3]), max(idxs[0], idxs[2]), max(idxs[1], idxs[3]))

    def set_window(self):
        self.window = Window(col_off=self.xmin, row_off=self.ymin, width=(self.xmax - self.xmin), height=(self.ymax - self.ymin))
        self.shape = (self.ymax - self.ymin, self.xmax - self.xmin)


    def get_corners(self):
        x0 = self.f.xy(self.ymax, self.xmin)[0]
        x1 = self.f.xy(self.ymax, self.xmax)[0]
        y0 = self.f.xy(self.ymin, self.xmax)[1]
        y1 = self.f.xy(self.ymax, self.xmax)[1]

        self.lowerleft = (min(x0, x1), min(y0, y1))
        self.upperright = (max(x0, x1), max(y0, y1))

    def get_xy_range(self):
        x = np.linspace(self.lowerleft[0], self.upperright[0], (self.xmax - self.xmin), endpoint=False)
        y = np.linspace(self.lowerleft[1], self.upperright[1], (self.ymax - self.ymin), endpoint=False)[::-1]
        #TODO: FIX y-DIRECTION
        return x, y

    def read(self, layeridx):
        arr = self.f.read(layeridx, window=self.window)
        return arr

    def get_pts_in_part(self, pts, buffer=0):
        self.get_corners()
        # Select points within part + buffer
        idx = (
            (pts[:, 0] > self.lowerleft[0] - buffer) &
            (pts[:, 0] < self.upperright[0] + buffer) &
            (pts[:, 1] > self.lowerleft[1] - buffer) &
            (pts[:, 1] < self.upperright[1] + buffer)
        )
        return idx

    def get_mask(self, polygon):

        valid = geometry_to_mask(polygon, self.lowerleft, abs(self.f.transform.a), self.shape)
        return valid

def geometry_to_mask(polygons, lowerleft, cellsize, shape):

    # Initialize mask
    mask = np.zeros(shape)

    for polygon in common.as_polygon_list(polygons):
        # Create from exterior
        mask += get_mask(polygon.exterior, lowerleft, cellsize, shape)
        # Subtract interiors
        for interior in polygon.interiors:
            mask -= get_mask(interior, lowerleft, cellsize, shape, outline=0)

    mask = (mask == 1)

    return mask


def get_mask(linestring, lowerleft, cellsize, shape, outline=1):

    # Create array from coordinate sequence
    path = np.vstack(linestring.coords[:])

    # Convert to (0,0) and step size 1
    path[:, 0] -= lowerleft[0]
    path[:, 1] -= lowerleft[1] + cellsize
    path /= cellsize
    # Convert from array to tuple list
    path = list(zip(*zip(*path)))

    # Create mask
    maskIm = PIL.Image.new('L', (shape[1], shape[0]), 0)
    PIL.ImageDraw.Draw(maskIm).polygon(path, outline=outline, fill=1)
    mask = np.array(maskIm)[::-1]

    return mask


def raster_in_parts(f: rasterio.io.DatasetReader, ncols: int, nrows: int, facedata) -> RasterPart:
    """Certain rasters are too big to read into memory at once.
    This function helps splitting them in equal parts of (+- ncols x nrows pixels)

    If facedata is given, each part is extended such that whole faces
    are covered by the parts
    
    Args:
        f (_type_): _description_
        ncols (_type_): _description_
        nrows (_type_): _description_
        facedata (_type_): _description_

    Yields:
        _type_: _description_
    """
    nx = max(1, f.shape[1] // ncols)
    ny = max(1, f.shape[0] // nrows)

    xparts = np.linspace(0, f.shape[1], nx+1).astype(int)
    yparts = np.linspace(0, f.shape[0], ny+1).astype(int)

    pts = facedata[['facex', 'facey']].values

    for ix, iy in product(range(nx), range(ny)):

        part = RasterPart(f, xmin=xparts[ix], ymin=yparts[iy], xmax=xparts[ix+1], ymax=yparts[iy+1])

        if facedata is not None:
            # For each part, get points in part.
            # For narrow/diagonal shapes the points in part can be limited
            idx = part.get_pts_in_part(pts)
            if not idx.any():
                continue

            crds = facedata['crds'].tolist()
            ll = list(zip(*[crds[i].min(axis=0) for i in np.where(idx)[0]]))
            ur = list(zip(*[crds[i].max(axis=0) for i in np.where(idx)[0]]))
            bounds = (min(ll[0]), min(ll[1]), max(ur[0]), max(ur[1]))

            # Get new part based on extended bounds
            part = RasterPart.from_bounds(f, bounds)

            # Add the cell centers within the window as index to the part
            part.idx = idx

        yield part

def rasterize_cells(facedata, prt):

    # Initialize mask
    # Create mask
    maskIm = PIL.Image.new('I', (prt.shape[1], prt.shape[0]), 0)
    todraw = PIL.ImageDraw.Draw(maskIm)

    cellsize = abs(prt.f.transform.a)

    for row in facedata.itertuples():

        # Create array from coordinate sequence
        path = row.crds.copy()
        # Convert to (0,0) and step size 1
        path[:, 0] -= (prt.lowerleft[0] - 0.5 * cellsize)
        path[:, 1] -= (prt.lowerleft[1] + 0.5 * cellsize)
        path /= cellsize
        # Convert from array to tuple list
        path = list(zip(*zip(*path)))

        # Create mask
        todraw.polygon(path, outline=row.Index, fill=row.Index)

    return np.array(maskIm, dtype=np.int32)[::-1]

def check_geodateframe_rasterstats(facedata):
    """
    Check for type, columns and coordinates
    """
    if not isinstance(facedata, gpd.GeoDataFrame):
        raise TypeError('facedata should be type GeoDataFrame')

    # Check if facedata has required columns
    if ('facex' not in facedata.columns) or ('facey' not in facedata.columns):
        xy = list(zip(*[pt.coords[0] for pt in facedata.geometry.centroid]))
        facedata['facex'] = xy[0]
        facedata['facey'] = xy[1]

    # Check if coordinates are present.
    if 'crds' not in facedata.columns:
        facedata['crds'] =[row.coords[:] for row in facedata.geometry]


def raster_stats_fine_cells(rasterpath: Union[str, Path], facedata, stats=['mean']):
    """
    Calculate statistic from a raster, where the raster resoltion is (much)
    smaller than the cell size.

    Parameters
    ----------
    rasterpath : str
        Path to raster file
    facedata : geopandas.GeoDataFrame
        Dataframe with polygons in which the raster statistics are derived.
    stats : list
        List of statistics to retrieve. Should be numpy functions that require one argument
    """

    # Create empty array for stats
    stat_array = {stat: {} for stat in stats + ['count']}

    # Check geometries
    check_geodateframe_rasterstats(facedata)

    # Open raster file
    with rasterio.open(rasterpath, 'r') as f:

        # Split file in parts based on shape
        parts = raster_in_parts(f, ncols=250, nrows=250, facedata=facedata)

        for prt in parts:

            # Get values from array
            arr = prt.read(1)
            valid = (arr != f.nodata)
            if not valid.any():
                continue
            
            # Rasterize the cells in the part
            cellidx_sel = rasterize_cells(facedata.loc[prt.idx], prt)
            assert cellidx_sel.shape == valid.shape
            cellidx_sel[~valid] = 0
            valid = (cellidx_sel != 0)

            for cell_idx in np.unique(cellidx_sel):
                if cell_idx == 0:
                    continue

                # Mask
                cellmask = (cellidx_sel == cell_idx)
                if not cellmask.any():
                    continue

                # Get bottom values
                bottom = arr[cellmask]

                # For each statistic, get the values and add
                for stat in stats:
                    stat_array[stat][cell_idx] = getattr(np, stat)(bottom)

                stat_array['count'][cell_idx] = len(bottom)

        # Cast to pandas dataframe
        df = pd.DataFrame.from_dict(stat_array).reindex(index=facedata.index)

    return df

def waterdepth_ahn(dempath, facedata, outpath, column):
    """
    Function that combines a dem and water levels to a water
    depth raster. No sub grid correction is done.

    Parameters
    ----------
    dempath : str
        Path to raster file with terrain level
    facedata : gpd.GeoDataFrame
        GeoDataFrame with at least the cell geometry and
        a column with water levels
    outpath : str
        Path to output raster file
    column : str
        Name of the column with the water level data
    """

    # Open raster file
    with rasterio.open(dempath, 'r') as f:
        first = True
        out_meta = f.meta.copy()

        # Split file in parts based on shape
        parts = raster_in_parts(f, ncols=250, nrows=250, facedata=facedata)

        for prt in parts:

            # Get values from array
            arr = prt.read(1)
            valid = (arr != f.nodata)
            if not valid.any():
                continue

            cellidx_sel = rasterize_cells(facedata.loc[prt.idx], prt)
            cellidx_sel[~valid] = 0
            valid = (cellidx_sel != 0)

            # Create array to assign water levels
            wlev_subgr = np.zeros(cellidx_sel.shape, dtype=out_meta['dtype'])

            for cell_idx in np.unique(cellidx_sel):
                if cell_idx == 0:
                    continue
                # Mask
                cellmask = (cellidx_sel == cell_idx)
                if not cellmask.any():
                    continue
                # Add values for cell to raster
                wlev_subgr[cellmask] = facedata.at[cell_idx, column]

            # Write to output raster
            with rasterio.open(outpath, 'w' if first else 'r+', **out_meta) as dst:
                # Determine water depth
                if not first:
                    wdep_subgr = dst.read(1, window=prt.window)
                else:
                    wdep_subgr = np.ones_like(wlev_subgr) * f.nodata
                    first = False
                wdep_subgr[valid] = np.maximum(wlev_subgr - arr, 0).astype(out_meta['dtype'])[valid]
                dst.write(wdep_subgr[None, :, :], window=prt.window)

    compress(outpath)


def compress(path):
    """
    Function re-save an existing raster file with compression.

    Parameters
    ----------
    path : str
        Path to raster file. File is overwritten with compress variant.
    """
    # Compress
    with rasterio.open(path, 'r') as f:
        arr = f.read()
        out_meta = f.meta.copy()
        out_meta['compress'] = 'deflate'
    with rasterio.open(path, 'w', **out_meta) as f:
        f.write(arr)
