import os
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

from hydrolib.core.io.mdu.models import FMModel, FrictionModel

# nodig voor inladen ander script
# currentdir = os.path.dirname(os.path.abspath(__file__))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0, parentdir)
from read_dhydro import net_nc2gdf

def change_friction_shape(input_mdu,shape_path,output_path):
    fm = FMModel(input_mdu)

    df_glob, df_chan = read_frictionfiles(fm)

    gdf_frict = gpd.read_file(shape_path)
    gdf_frict_buf = gpd.GeoDataFrame(gdf_frict, geometry=gdf_frict.buffer(1))

    # Read model branches
    netfile = r"C:\Users\delanger3781\OneDrive - ARCADIS\Documents\DHydro\Zwolle-Minimodel\Zwolle-Minimodel\1D2D-DIMR\dflowfm\FlowFM_net.nc"
    branches = net_nc2gdf(netfile)["1d_branches"]
    branches.crs = "EPSG:28992"

    # Intersect shape met branch
    max_distance = float(10)
    intersect = gpd.sjoin_nearest(branches, gdf_frict, max_distance=max_distance)
    # TODO: Probleem Arjon

    # TODO: Filteren op branchtype (wordt nu nog pipe meegenomen)
    # TODO: Hoe halve watergangen verwerken
   
    for file in df_chan:
        if not df_chan[file].empty:
            for index, row in df_chan[file].iterrows():
                branchid = row["branchid"]
                
                # TODO: Als er twee waardes zijn in de intersect, wordt de laatste gepakt. Dat is nog incorrect
                if branchid in (list(intersect.id)):
                    df_chan[file].at[df_chan[file].loc[df_chan[file]["branchid"] == branchid,"frictionvalues"].index.tolist()[0],"frictionvalues"] = [intersect.loc[intersect["id"] == branchid,"fricval"].iloc[0]]*int(df_chan[file].at[df_chan[file].loc[df_chan[file]["branchid"] == branchid,"numlocations"].index.tolist()[0],"numlocations"])
            
            writefile = FrictionModel(global_=df_glob[file].to_dict("records"), branch = df_chan[file].to_dict("records"))
            writefile.save(Path(output_path) / (str("roughness-"+file + ".ini")))
        else:
            writefile = FrictionModel(global_=df_glob[file].to_dict("records"))
            writefile.save(Path(output_path) / (str("roughness-"+file + ".ini")))
            
def read_frictionfiles(fm):
    dfsglob = {}
    dfschan = {}

    # loop through friction files and store them in separate dictionaries
    for i in range(len(fm.geometry.frictfile)):
        dfglob = pd.DataFrame([f.__dict__ for f in fm.geometry.frictfile[i].global_])
        dfsglob[str(str(fm.geometry.frictfile[i].global_[0].frictionid))] = dfglob
               
        dfchan = pd.DataFrame([f.__dict__ for f in fm.geometry.frictfile[i].branch])
        dfschan[str(str(fm.geometry.frictfile[i].global_[0].frictionid))] = dfchan

    return dfsglob, dfschan


if __name__ == "__main__":
    # Read shape
    shape_path = r"C:\Users\delanger3781\OneDrive - ARCADIS\Documents\DHydro\Zwolle-Minimodel\Zwolle-Minimodel\frictfiles.shp"
    output_path = (
        r"C:\TEMP\AHT_test_output"
    )
    input_mdu = Path(
        r"C:\scripts\AHT_scriptjes\Hydrolib\Dhydro_changefrict\data\Zwolle-Minimodel\1D2D-DIMR\dflowfm\flowFM.mdu"
    )
    change_friction_shape(input_mdu,shape_path,output_path)
    

