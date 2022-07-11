import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from read_dhydro import chainage2gdf, net_nc2gdf
from shapely.geometry import LineString, Point, Polygon

from hydrolib.core.io.bc.models import ForcingBase
from hydrolib.core.io.mdu.models import FMModel
from hydrolib.core.io.net.models import Network
from hydrolib.core.io.structure.models import StructureModel


def read_strucs(input_mdu):
    fm = FMModel(input_mdu)
    print("The first structurefile is used.")
    structures = pd.DataFrame(
        [f.__dict__ for f in fm.geometry.structurefile[0].structure]
    )
    netnc_path = os.path.join(input_mdu.parent, str(fm.geometry.netfile.filepath))
    gdfs = net_nc2gdf(netnc_path)
    strucgdf = chainage2gdf(structures, gdfs["1d_branches"])

    return strucgdf


def read_bc(input_mdu):
    fm = FMModel(input_mdu)
    dfsbc = {}

    # loop through boundary files and store them in separate dictionaries
    for i in range(len(fm.external_forcing.extforcefilenew.boundary)):
        forcingdat = fm.external_forcing.extforcefilenew.boundary[i].forcing
        forceid = forcingdat.name
        data = forcingdat.datablock
        columns = [
            forcingdat.quantityunitpair[0].unit,
            str(
                forcingdat.quantityunitpair[1].quantity
                + "["
                + forcingdat.quantityunitpair[1].unit
                + "]"
            ),
        ]
        dfbc = pd.DataFrame(data=data, columns=columns)

        dfsbc[str(forceid)] = dfbc

    return dfsbc


def read_lat(input_mdu):
    dfsbc = {}
    lats = FMModel(input_mdu)
    for i in range(len(lats.external_forcing.extforcefilenew.lateral)):
        latdat = lats.external_forcing.extforcefilenew.lateral[i].forcing
        forceid = latdat.name
        data = latdat.datablock
        columns = [
            latdat.quantityunitpair[0].unit,
            str(
                latdat.quantityunitpair[1].quantity
                + "["
                + latdat.quantityunitpair[1].unit
                + "]"
            ),
        ]
        dfbc = pd.DataFrame(data=data, columns=columns)

        dfsbc[str(forceid)] = dfbc
    return dfsbc


def read_crosssections(input_mdu):
    fm = FMModel(input_mdu)
    netnc_path = os.path.join(input_mdu.parent, str(fm.geometry.netfile.filepath))
    gdfs = net_nc2gdf(netnc_path)

    dfcrsdef = pd.DataFrame([f.__dict__ for f in fm.geometry.crossdeffile.definition])

    dfcrsloc = pd.DataFrame([f.__dict__ for f in fm.geometry.crosslocfile.crosssection])
    dfcrsloc_gdf = chainage2gdf(dfcrsloc, gdfs["1d_branches"])
    return dfcrsdef, dfcrsloc_gdf


# =============================================================================
# bc = pd.DataFrame([f.__dict__ for f in fm.external_forcing.extforcefilenew.boundary])
#
# dfsbc = {}
#
# # loop through friction files and store them in separate dictionaries
# for i in range(len(fm.external_forcing.extforcefilenew.boundary)):
#     forcingdat = fm.external_forcing.extforcefilenew.boundary[i].forcing
#     forceid = forcingdat.name
#     data = forcingdat.datablock
#     columns = [forcingdat.quantityunitpair[0].unit,
#                str(forcingdat.quantityunitpair[1].quantity + "[" + forcingdat.quantityunitpair[1].unit + "]")]
#     dfbc = pd.DataFrame(data = data,columns = columns)
#
#     dfsbc[str(forceid)] = dfbc
#
#
# =============================================================================
#  #TODO: Deltares still fixing that the data of laterals is loaded.
# lats = pd.DataFrame([f.__dict__ for f in fm.external_forcing.extforcefilenew.lateral])
# for i in range(len(fm.external_forcing.extforcefilenew.lateral)):
#     latdat = fm.external_forcing.extforcefilenew.lateral[i].forcing
#     forceid = latdat.name
#     data = latdat.datablock
#     columns = [latdat.quantityunitpair[0].unit,
#                 str(latdat.quantityunitpair[1].quantity + "[" + latdat.quantityunitpair[1].unit + "]")]
#     dfbc = pd.DataFrame(data = data,columns = columns)

#     dfsbc[str(forceid)] = dfbc

if __name__ == "__main__":
    input_mdu = Path(
        r"C:\scripts\AHT_scriptjes\Hydrolib\Dhydro_changefrict\data\Zwolle-Minimodel\1D2D-DIMR\dflowfm\flowFM.mdu"
    )

    fm = FMModel(input_mdu)

    read_crosssections(fm)

    # strucs = read_strucs(input_mdu)
    # struc = StructureModel(definition=strucs.to_dict("records"))
