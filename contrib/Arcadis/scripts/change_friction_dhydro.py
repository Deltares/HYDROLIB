import os
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

from hydrolib.core.io.mdu.models import FMModel, FrictionModel

# nodig voor inladen ander script
currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from read_dhydro import net_nc2gdf


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


def read_crossections(fm):

    dfcrsdef = pd.DataFrame([f.__dict__ for f in fm.geometry.crossdeffile.definition])
    dfcrsloc = pd.DataFrame([f.__dict__ for f in fm.geometry.crosslocfile.crosssection])

    return dfcrsdef, dfcrsloc


def read_shape(gispath, frictcolumn):
    areas_selection = gpd.read_file(gispath)
    friction = areas_selection[str(frictcolumn)]
    print()


if __name__ == "__main__":
    # Read shape
    shape_path = r"C:\Users\delanger3781\OneDrive - ARCADIS\Documents\DHydro\Zwolle-Minimodel\Zwolle-Minimodel\frictfiles.shp"
    output_location = (
        r"C:\scripts\AHT_scriptjes\Hydrolib\Dhydro_changefrict\roughness-Section000.ini"
    )
    input_mdu = Path(
        r"C:\scripts\AHT_scriptjes\Hydrolib\Dhydro_changefrict\data\Zwolle-Minimodel\1D2D-DIMR\dflowfm\flowFM.mdu"
    )
    fm = FMModel(input_mdu)

    dict_global, dict_chann = read_frictionfiles(fm)

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

    for file in dict_chann:
        if not dict_chann[file].empty:
            for id in dict_chann[file].branchid:
                if id in intersect["id"].values:
                    print(id)
                    # TODO: Als er twee waardes, wordt de laatste gepakt. Dat is nog incorrect
                    # dict_chann[file].loc[dict_chann[file].branchid == id, 'frictionvalues'] = list(intersect.loc[intersect["id"] == id, "fricval"])
                    dict_chann[file].at[
                        dict_chann[file]
                        .loc[dict_chann[file].branchid == id]
                        .index.to_list()[0],
                        "frictionvalues",
                    ] = list(intersect.loc[intersect["id"] == id, "fricval"])
    for file in dict_chann:
        writer = FrictionModel(branch=dict_chann[file].to_dict("records"))

    # wegschrijven van .ini file met nieuwe fricties
    # TODO: losse ini wegschrijven er vanuit gaande dat hij in "section000" staat. Nieuwe wegschrijven
    # anders: nieuwe crsdef nodig.

    global_value = 31
    frict_id = str("FloodPlain1")

    # TODO hydrolib core writer gebruiker (.save)
    with open(output_location, "w") as f:
        f.write(
            "[General]\n    fileVersion           = 2.00\n    fileType              = roughness\n"
        )
        f.write(
            "\n[Global]\n    frictionId            = "
            + frict_id
            + "\n    frictionType          = Strickler\n    frictionValue         = "
            + str(global_value)
            + "\n"
        )

        for id in intersect["id"]:
            f.write("\n[Branch]\n    branchId              = " + id + "\n")
            f.write("    frictionType          = " + "Strickler" + "\n")
            f.write("    functionType          = " + "Constant" + "\n")
            f.write("    numLocations          = " + "1" + "\n")
            f.write("    chainage              = " + "0.000000" + "\n")
            f.write(
                "    frictionValues        = "
                + str(intersect.loc[intersect["id"] == id, "fricval"].values[0])
                + "\n"
            )

    print("done")
