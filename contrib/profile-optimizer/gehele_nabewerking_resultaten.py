from netCDF4 import Dataset
import numpy as np
import pandas as pd
import datetime as dt
import plotly.graph_objects as go
import os
from datetime import datetime
import math
import gdal
import geopandas as gpd
import pandas as pd
import os
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats
import rasterio
import rasterio.mask
import scipy.stats as st
from scipy.stats import t
import os
from osgeo import gdal
import tqdm
from psutil import virtual_memory
from ahn_downloader.config import MEMORYLIMIT, CACHELIMIT

def plot_position(i,N=105):
    pp = 1/((i-0.3)/(N+0.4))
    return pp

x_values= []
for i in range(1,55):
    x_values.append(plot_position(i))
x2_values = x_values
x2_values.sort()

def get_a_b(wl,n):
    y = np.log(x2_values[(54-n):54])
    x = wl
    #gradient, intercept, r_value, p_value, std_err = \
    res= stats.linregress(y, x)
    b= res.slope
    a= res.intercept
    tinv = lambda p, df: abs(t.ppf(p/2, df))
    ts = tinv(0.05, len(x) - 2)
    print(ts)
    print(b)
    print(a)

    c = ts * res.intercept_stderr
    d = ts * res.stderr

    print(f"slope (95%): {b:.6f} +/- {ts * res.stderr:.6f}")
    print(f"intercept (95%): {a:.6f}" f" +/- {ts * res.intercept_stderr:.6f}")
    #b = np.sqrt((6*np.var(wl))/(np.pi**2))
    #a = (np.mean(wl) - (0.5772*b))
    return a, b,c,d


def get_gumbel(a, b, t):
    x = a+b*np.log(t)
    return x

def gumbel_tabel(wl, t_list):
    a, b,c,d = get_a_b(wl,10)
    wl_list = []
    for t in t_list:
        x = get_gumbel(a, b, t)
        wl_list.append(x)
    wl_s = pd.Series(wl_list, index=t_list)
    wl_min_list = []
    wl_max_list = []
    for t in t_list:
        x = get_gumbel(a-c, b-d, t)
        wl_min_list.append(x)
    wl_min_s = pd.Series(wl_min_list, index=t_list)
    for t in t_list:
        x = get_gumbel(a+c, b+d, t)
        wl_max_list.append(x)
    wl_max_s = pd.Series(wl_max_list, index=t_list)
    return wl_s, wl_min_s,wl_max_s

def stat(x):
    df1 = x.describe().transpose()
    df1["rho"] = x.corr(method="spearman")
    df1["skew"] = x.skew()
    return df1

def calculate_return_period(df, datcolname):
    # Import data
    data = df
    data_orig = data

    values_sorted=data_orig[datcolname].sort_values()
    # Compute the distributions
    table_rp = [
        10,25,50,100
        ]
    distgumbel2,mini,maxi = gumbel_tabel(values_sorted[44:54],table_rp)

    distgumbel2 = distgumbel2.values

    return distgumbel2

def array_to_str(ar):
    return ''.join(str(s, encoding='UTF-8') for s in ar)

def get_node_list(nc, var_node_id):
    """Get list of the node names/ids from the name/id variabele. The input variable ends with _name or _id"""
    nodes = nc.variables[var_node_id]
    node_list = []
    for i in range(len(nodes)):
        node_list.append(array_to_str(nodes[i,:].data))
    return node_list

def set_gdal_cache_limit(gdal_cachemax=None):
    if gdal_cachemax is None:
        mem = virtual_memory()
        gdal_cachemax = int(CACHELIMIT * mem.free * 1e-6)
    gdal.SetConfigOption('GDAL_CACHEMAX', str(gdal_cachemax))
    print(f"GDAL_CACHEMAX set to {gdal_cachemax}mb")
    return

def get_memorylimit(memorylimit=None):
    if memorylimit is None:
        mem = virtual_memory()
        memorylimit = int(MEMORYLIMIT * mem.free * 1e-6)
    return memorylimit

def mosaic_raster(fnames: list, filename: str, export_folder: str, clip_polygon: str, cachelimit: int = None,
                  memorylimit=None, file_extension = "tif"):
    """
    Mosaic geotiffs using GDAL warp which makes is possible to limit the memory use. This is required for raster larger than the systems memory.
    Args:
        fnames: list of filenames. Note that zip files has a unique notition starting with '/vsizip/'
        filename: name of the export file
        export_folder: folder to export the raster to

    Returns: filename of the output raster

    """

    def progress_callback(complete, message, unknown):
        pbar.update(complete)
        return 1

    print(f"Mosaic raster with GDAL")

    if file_extension == "tif":
        co = ["TILED=YES"]
    else:
        co = None

    if clip_polygon is None:
        crop = False
    else:
        crop = True

    set_gdal_cache_limit(cachelimit)

    fname_output = os.path.join(export_folder, filename + "." + file_extension)
    with tqdm.tqdm(total=1) as pbar:
        ds = gdal.Warp(fname_output, fnames, copyMetadata=True, warpMemoryLimit=get_memorylimit(memorylimit),
                       multithread=False, callback=progress_callback, callback_data=pbar, cropToCutline=crop,
                       cutlineDSName=clip_polygon, creationOptions=co)
        ds = None
    return fname_output


path_in_l= r"D:\NBW_berekeningen\landelijk"
koppel_AE_ID = pd.read_csv(r"C:\Users\908800\Box\BH3371 NBW Oosterw\BH3371 NBW Oosterw WIP\04_opzet_model\NBW_doorrekening\input/link_AE_model.csv")

#inladen 1e resultaat om ID te bepalen
nr=1
i= nr-1
outputfile= (path_in_l + "\\" + str(nr)) + "\\fm\\DFM_OUTPUT_import\import_fou.nc"
his = r"D:\NBW_berekeningen\landelijk\1\fm\DFM_OUTPUT_import\import_fou.nc"
if not os.path.exists(outputfile):
    print(nr)
nchis = Dataset(outputfile, mode='r')

listvariables = list(nchis.variables.keys())
max_wl = nchis.variables[str("mesh1d_fourier003_max")][:].data[:]
ID = np.arange(1, len(max_wl)+1, 1)

result_NBW=pd.DataFrame(pd.np.empty((55, 1669)) * pd.np.nan,columns = ID)

for nr in range(1,56):
    i= nr-1
    outputfile= (path_in_l + "\\" + str(nr)) + "\\fm\\DFM_OUTPUT_import\import_fou.nc"
    #his = r"D:\NBW_berekeningen\landelijk\1\fm\DFM_OUTPUT_import\import_fou.nc"
    if not os.path.exists(outputfile):
        print(nr)
    nchis = Dataset(outputfile, mode='r')

    listvariables = list(nchis.variables.keys())
    max_wl = nchis.variables[str("mesh1d_fourier003_max")][:].data[:]


    result_NBW.iloc[i]=max_wl
    #x = nchis.variables["mesh1d_FlowElemContour_x"][:].data[:, 0][0]
    print(nr)

colnames = ["T10","T25","T50","T100"]
result_NBW_HHT=pd.DataFrame(pd.np.empty((1669, 4)) * pd.np.nan,columns = colnames)

for i in range(0,1669):
    print(i)
    result_NBW_HHT.iloc[i]=calculate_return_period(result_NBW,i+1)

#shapefile maken met de waterstanden en herhalingstijden
#ID = get_node_list(nchis,"network_node_id")
x = nchis.variables["mesh1d_FlowElemContour_x"][:].data[:,0]
y = nchis.variables["mesh1d_FlowElemContour_y"][:].data[:,0]
nr = np.arange(1,len(x)+1,1)

df_tot = pd.DataFrame(list(zip(nr,result_NBW_HHT["T10"],result_NBW_HHT["T25"],result_NBW_HHT["T50"],result_NBW_HHT["T100"], x, y)),
                      columns =["ID","T10","T25","T50","T100", 'x', "y"])
#verwijderen laatste 8 rijen, dit zijn namelijk -999 waarden
N=8
df_tot = df_tot.iloc[:-N , :]

gdf_tot= gpd.GeoDataFrame(
    df_tot, geometry=gpd.points_from_xy(df_tot.x, df_tot.y))

output_pad = r"C:\Users\908800\Box\BH3371 NBW Oosterw\BH3371 NBW Oosterw WIP\04_opzet_model\NBW_doorrekening\output\waterstand_herhaling_v2.shp"
gdf_tot.to_file(output_pad)

#nu hebben we per rekenpunt de waterstand per herhalingstijd. nu daar grids van maken per ae en interpoleren

#inladen afwaterende eenheden
# per afwaterende eenheid alleen de rekenpunten selecteren die daarbinnen zitten
# dan met inverse distance weighing een grid maken van 5 x 5
# voor allemaal doen en dan samenvoegen per herhalingstijd
# dan per herhalingstijd het maaiveld eraf halen -> overstromingskaart
#maaiveld nodig van 5 bij 5 dicht geinterpoleerd voor de afwaterende eenheden

#
pad_ae = r"C:\Users\908800\Box\BH3371 NBW Oosterw\BH3371 NBW Oosterw WIP\04_opzet_model\modellen\basis_met_B_watergangen\08_input\shapefiles\Afwaterende_eenheden2.shp"
AE = gpd.read_file(pad_ae)
rp = gpd.read_file(output_pad)
rp=rp.set_crs(28992)

for i in AE.index:
    print(i)
    poly = AE.loc[[i]]
    #nu per row intersection met
    test=gpd.overlay(rp,poly)
    if len(test)>0:
        pad_temp_shape = r"C:\Users\908800\Box\BH3371 NBW Oosterw\BH3371 NBW Oosterw WIP\04_opzet_model\NBW_doorrekening\output\temp\z.shp"
        test.to_file(pad_temp_shape)
        pad_uit = r"C:\Users\908800\Box\BH3371 NBW Oosterw\BH3371 NBW Oosterw WIP\04_opzet_model\NBW_doorrekening\output\rasters\waterstand"
        for j in ["T10","T25","T50","T100"]:
            pad_uit_t = pad_uit + "\\" +j+"\\"+ str(i) +".tif"
            idw= gdal.Grid(pad_uit_t,pad_temp_shape,zfield=j,algorithm="invdist:power=3:radius1=2000:radius2=2000",
                      outputBounds=poly.bounds.values[0])
            idw = None

            #inladen grid
            with rasterio.open(pad_uit_t, "r") as EM:
                out_meta = EM.meta
                nodata_value = EM.nodata
                out_image, out_transform = rasterio.mask.mask(EM, poly.geometry,crop=True)
                out_meta.update({"driver": "GTiff",
                                 "height": out_image.shape[1],
                                 "width": out_image.shape[2],
                                 "transform": out_transform,
                                "nodata":0})

            with rasterio.open(pad_uit_t, "w", **out_meta) as dest:
                dest.write(out_image)

list_T10=[]
for path in os.listdir(r"C:\Users\908800\Box\BH3371 NBW Oosterw\BH3371 NBW Oosterw WIP\04_opzet_model\NBW_doorrekening\output\rasters\waterstand\T10"):
    full_path = os.path.join(r"C:\Users\908800\Box\BH3371 NBW Oosterw\BH3371 NBW Oosterw WIP\04_opzet_model\NBW_doorrekening\output\rasters\waterstand\T10", path)
    list_T10.append(full_path)

list_T25=[]
for path in os.listdir(r"C:\Users\908800\Box\BH3371 NBW Oosterw\BH3371 NBW Oosterw WIP\04_opzet_model\NBW_doorrekening\output\rasters\waterstand\T25"):
    full_path = os.path.join(r"C:\Users\908800\Box\BH3371 NBW Oosterw\BH3371 NBW Oosterw WIP\04_opzet_model\NBW_doorrekening\output\rasters\waterstand\T25", path)
    list_T25.append(full_path)

list_T50=[]
for path in os.listdir(r"C:\Users\908800\Box\BH3371 NBW Oosterw\BH3371 NBW Oosterw WIP\04_opzet_model\NBW_doorrekening\output\rasters\waterstand\T50"):
    full_path = os.path.join(r"C:\Users\908800\Box\BH3371 NBW Oosterw\BH3371 NBW Oosterw WIP\04_opzet_model\NBW_doorrekening\output\rasters\waterstand\T50", path)
    list_T50.append(full_path)

list_T100=[]
for path in os.listdir(r"C:\Users\908800\Box\BH3371 NBW Oosterw\BH3371 NBW Oosterw WIP\04_opzet_model\NBW_doorrekening\output\rasters\waterstand\T100"):
    full_path = os.path.join(r"C:\Users\908800\Box\BH3371 NBW Oosterw\BH3371 NBW Oosterw WIP\04_opzet_model\NBW_doorrekening\output\rasters\waterstand\T100", path)
    list_T100.append(full_path)

poly_clip= os.path.join(r'C:\Users\908800\Box\BH3371 NBW Oosterw\BH3371 NBW Oosterw WIP\02_WSA\03_GIS\rasters\maaiveld','AE_dissolved.shp')

mosaic_raster(list_T10,"T10",pad_uit,poly_clip)
mosaic_raster(list_T25,"T25",pad_uit,poly_clip)
mosaic_raster(list_T50,"T50",pad_uit,poly_clip)
mosaic_raster(list_T100,"T100",pad_uit,poly_clip)

#einde, hieronder om helemaal in 1 keer te doen

gdf_tot=gpd.read_file(output_pad)
poly_clip2 = gpd.read_file(poly_clip)
for j in ["T10", "T25", "T50", "T100"]:
    j="T10"
    pad_uit_t = pad_uit + "\\" + j +"in_1.tif"
    idw = gdal.Grid(pad_uit_t, output_pad, zfield=j, algorithm="invdist:power=3:radius1=2000:radius2=2000",
                    outputBounds=poly_clip2.bounds.values[0])
    idw = None

    # inladen grid
    with rasterio.open(pad_uit_t, "r") as EM:
        out_meta = EM.meta
        nodata_value = EM.nodata
        out_image, out_transform = rasterio.mask.mask(EM, poly_clip2.geometry, crop=True)
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform,
                         "nodata": 0})

    with rasterio.open(pad_uit_t, "w", **out_meta) as dest:
        dest.write(out_image)