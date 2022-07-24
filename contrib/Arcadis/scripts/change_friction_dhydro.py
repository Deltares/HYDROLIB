# =============================================================================
#
# License: LGPL
#
# Author: Robbert de Lange Arcadis
#
# =============================================================================
import os
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from read_dhydro import net_nc2gdf

from hydrolib.core.io.mdu.models import FMModel, FrictionModel

# TODO geef optie om alle oude frictions te wipen of te houden
# (3 opties: alles wipen en nieuwe toevoegen, oude houden en nieuwe overschrijven, toevoegen waar alleen global is (is al gedefinieerd of niet?))
# new, replace & append
# TODO Er zijn verschillende files, die allemaal overschrijven?
# Overschrijven per branch en niet per cross section warning.
# TODO Halverwege watergangen verwerken?
# TODO soort friction
# Soort friction als input meegeven.


def change_friction_shape(mdu_path, shape_path, netnc_path, output_path):
    """
    Buffer used = 10

    Parameters
    ----------
    mdu_path : Path()
        DESCRIPTION.
    shape_path : TYPE
        DESCRIPTION.
    output_path : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    buffer = 10

    fm = FMModel(mdu_path)
    dict_global, dict_frictions = friction2dict(fm)
    gdf_frict = gpd.read_file(shape_path)

    # gdf_frict_buf = gpd.GeoDataFrame(gdf_frict, geometry=gdf_frict.buffer(1))

    # Read model branches
    netfile = netnc_path
    branches = net_nc2gdf(netfile)["1d_branches"]
    branches.crs = "EPSG:28992"

    # Intersect shape met branch
    # TODO: definieer id met friction value
    intersect = gpd.sjoin_nearest(branches, gdf_frict, max_distance=buffer)
    # TODO: Probleem Arjon (?)

    # TODO: Filteren op branchtype (wordt nu nog pipe meegenomen)
    # TODO: Hoe halve watergangen verwerken

    for key in dict_frictions:
        if not dict_frictions[key].empty:
            # check if intersect is in current frictionfile
            in_model = intersect.assign(
                result=intersect["id"].isin(dict_frictions[key].branchid)
            )

            # TODO: Check if the id is always id with gpd.sjoin?
            in_model.rename(columns={"id": "branchid"}, inplace=True)

            # store new values inside new frictionfile
            merge = dict_frictions[key].merge(in_model, how="left", on="branchid")
            merge.loc[merge.result == True, "frictionvalues"] = merge.loc[
                merge.result == True, "fricval"
            ]

            # Change amount of locations to 0, because the friction is defined on the branch
            merge.loc[merge.result == True, "numlocations"] = 1
            merge.loc[merge.result == True, "chainage"] = 0

            # change all values to list to make importable into hydrolib
            merge["frictionvalues"] = merge["frictionvalues"].apply(
                lambda x: [x] if isinstance(x, float) else x
            )
            merge["chainage"] = merge["chainage"].apply(
                lambda x: [x] if isinstance(x, int) else x
            )
            merge.drop(
                columns=in_model.columns.difference(dict_frictions[key].columns),
                inplace=True,
            )

            # write the file into FrictionModel and save to hydrolib
            writefile = FrictionModel(
                global_=dict_global[key].to_dict("records"),
                branch=merge.to_dict("records"),
            )

            writefile.save(Path(output_path) / (str("roughness-" + key + ".ini")))

        else:
            writefile = FrictionModel(global_=dict_global[key].to_dict("records"))
            writefile.save(Path(output_path) / (str("roughness-" + key + ".ini")))


def friction2dict(fm):
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
    netnc_path = r"C:\Users\delanger3781\OneDrive - ARCADIS\Documents\DHydro\Zwolle-Minimodel\Zwolle-Minimodel\1D2D-DIMR\dflowfm\FlowFM_net.nc"
    output_path = r"C:\TEMP\AHT_test_output"
    input_mdu = Path(
        r"C:\scripts\AHT_scriptjes\Hydrolib\Dhydro_changefrict\data\Zwolle-Minimodel\1D2D-DIMR\dflowfm\flowFM.mdu"
    )
    change_friction_shape(input_mdu, shape_path, netnc_path, output_path)
