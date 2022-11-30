import glob
import os
import sys
from tqdm.auto import tqdm
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.fill import fillnodata
from pathlib import Path

class DTM:
    
    def __init__(self, path):
        self.root = Path(path)
    
    def compose_ahn(self, ahn_path = None, extent_path=None, max_fill_distance=0, output_path=None):              
        
         # if no paths are provided, use the defaults
        if ahn_path is None:
            ahn_path = self.root / "AHN" 
        
        if extent_path is None:
            extent_path = self.root / 'OLO' / 'OLO_stroomgebied_met_buffer.shp'
        
        # make output and tepm directories if necessary
        if output_path is None:
            output_path  = Path(self.root) / 'output'
            output_path.mkdir(parents=True, exist_ok=True)
        temp_path  = Path(self.root) / 'temp'   
        temp_path.mkdir(parents=True, exist_ok=True)
               
        # load extent
        extent = gpd.read_file(extent_path)                
        
        # fill holes per tile
        #intermediates = [rasterio.open(pad) for pad in glob.glob(ahn_path)]
        intermediates = [rasterio.open(pad) for pad in ahn_path.glob('*.tif')]
        # print(intermediates)
        
        for intermediate in tqdm(intermediates, total=len(intermediates)):    
            ahn_tile = intermediate.read(1)
            intermediate_mask_nodata = ~(ahn_tile == intermediate.nodata)
            out_meta = intermediate.meta.copy()
            out_meta['compression'] = 'lzw'
            #filename = intermediate.name.rsplit('/',1)[-1][:-4] + "_intermediate.tif"
            filename = os.path.split(intermediate.name)[-1][:-4] + "_intermediate.tif"
            filled_intermediate = fillnodata(
                image = ahn_tile,
                mask = intermediate_mask_nodata,
                max_search_distance=max_fill_distance)
            with rasterio.open(temp_path/filename, "w", **out_meta) as dest:
                dest.write(filled_intermediate,1)                
            intermediate.close()
        
        # open and merge rasters
        opened = [rasterio.open(pad) for pad in glob.glob(os.path.join(str(temp_path), "M*.tif"))]
        # print(opened)
        
        merged, affine = merge(opened)
        out_meta = opened[0].meta.copy()
        out_meta['compression'] = 'lzw'
        out_meta['transform'] = affine
        out_meta['width'] = merged.shape[2]
        out_meta['height'] = merged.shape[1]
        
        #close rasters
        for openraster in opened:
            openraster.close() 
        
        # save merged file
        with rasterio.open(os.path.join(output_path, 'AHN.tif'), "w", **out_meta) as dest:
            dest.write(merged)
                
        #remove temporary files
        tmp_fNames = [file for file in os.listdir(temp_path)]
        for fName in tmp_fNames:
            os.remove(os.path.join(temp_path, fName))
            #print('Temporary file',fName,'removed')
        


if __name__=="__main__":
    path = r'R:\pr\4632_10\lijn_elementen\output\pertile'
    target = r'C:\4632.10'
    dirs = os.listdir(path)
    import shutil
    for num, dir in enumerate(dirs):
        if os.path.exists(os.path.join(target, 'binary_result_'+str(num)+'.tif')):
            continue
        shutil.copy(os.path.join(path,dir,'A3_v2_Results_binary.tif'), os.path.join(target, 'binary_result_'+str(num)+'.tif'))

    # ahn_path = r'W:\projects\4632_10\merge'
    # dtm = DTM(path)
    # extent_path = Path(path)/'OLO_stroomgebied_incl.maas.shp'
    # dtm.compose_ahn( extent_path=extent_path, ahn_path = Path(ahn_path), max_fill_distance=1, output_path='result_merged.tif')


    from rasterio.plot import show
    from rasterio.merge import merge
    import rasterio as rio
    from pathlib import Path
    
    
    path = Path(target)
    path.mkdir(parents=True, exist_ok=True)
    output_path = path / 'mosaic_output.tif'
    
    raster_files = list(path.iterdir())
    raster_to_mosiac = []
    
    for p in raster_files:
        raster = rio.open(p)
        raster_to_mosiac.append(raster)
    
    mosaic, output = merge(raster_to_mosiac,precision=10, method='max')
    
    crs = 'EPSG:28992'
    
    output_meta = raster.meta.copy()
    output_meta.update(
        {"driver": "GTiff",
            "compress": "lzw",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": output,
            "crs": rasterio.crs.CRS({'init' : raster.crs.data['init']})
        }
    )    
    with rio.open(output_path, 'w', **output_meta) as m:
        m.write(mosaic)
    
    #resultaat in kaart plotten
    # show(mosaic, cmap='terrain', vmin = 0.01, vmax = 1)
    
    #rasters sluiten
    raster.close()
    [x.close() for x in raster_to_mosiac]
